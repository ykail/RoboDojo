"""Keyboard input and Cartesian teleoperation for interactive RoboDojo runs.

The Omniverse adapter is deliberately thin.  The state machine and pose math
stay importable without Isaac Sim so they can be unit-tested on a normal
Python installation.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

import numpy as np

_TRANSLATION_KEYS = {
    "W": np.array([1.0, 0.0, 0.0]),
    "S": np.array([-1.0, 0.0, 0.0]),
    "A": np.array([0.0, 1.0, 0.0]),
    "D": np.array([0.0, -1.0, 0.0]),
    "Q": np.array([0.0, 0.0, 1.0]),
    "E": np.array([0.0, 0.0, -1.0]),
}
_ROTATION_KEYS = {
    "Z": np.array([1.0, 0.0, 0.0]),
    "X": np.array([-1.0, 0.0, 0.0]),
    "T": np.array([0.0, 1.0, 0.0]),
    "G": np.array([0.0, -1.0, 0.0]),
    "C": np.array([0.0, 0.0, 1.0]),
    "V": np.array([0.0, 0.0, -1.0]),
}
_MOTION_KEYS = frozenset(_TRANSLATION_KEYS) | frozenset(_ROTATION_KEYS)


def _normalise_key_name(name: str) -> str:
    name = str(name).strip().upper()
    aliases = {
        " ": "SPACE",
        "RETURN": "ENTER",
        "KEY_I": "I",
        "KEY_1": "1",
        "NUM_1": "1",
        "KEY_2": "2",
        "NUM_2": "2",
    }
    return aliases.get(name, name)


@dataclass(frozen=True)
class KeyboardSnapshot:
    """One atomic view of keyboard state at a 25 Hz policy tick."""

    deadman: bool
    active_arm: str
    delta_pose: np.ndarray
    gripper_toggles: tuple[str, ...] = ()
    takeover_pressed: bool = False
    takeover_released: bool = False
    arm_changed: bool = False
    accept_requested: bool = False
    save_retry_requested: bool = False
    abort_requested: bool = False


class KeyboardState:
    """Thread-safe, repeat-resistant toggle state machine used by Kit."""

    def __init__(
        self,
        pos_step: float = 0.005,
        rot_step: float = 0.02,
        deadman_timeout: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if pos_step <= 0 or rot_step <= 0:
            raise ValueError("Keyboard position and rotation steps must be positive.")
        if deadman_timeout <= 0:
            raise ValueError("Keyboard deadman timeout must be positive.")
        self.pos_step = float(pos_step)
        self.rot_step = float(rot_step)
        self.deadman_timeout = float(deadman_timeout)
        self._clock = clock
        self._last_input_time = self._clock()
        self._lock = threading.Lock()
        self._held: set[str] = set()
        self._active_arm = "left"
        self._takeover_active = False
        self._gripper_toggles: list[str] = []
        self._takeover_pressed = False
        self._takeover_released = False
        self._arm_changed = False
        self._accept_requested = False
        self._save_retry_requested = False
        self._abort_requested = False

    def handle_key(self, key_name: str, pressed: bool) -> None:
        """Apply a press/release event; duplicate key-repeat presses are ignored."""
        key = _normalise_key_name(key_name)
        with self._lock:
            self._last_input_time = self._clock()
            if pressed:
                if key in self._held:
                    return
                # L is an emergency-off interlock. Do not permit a fresh I
                # edge until L has physically been released.
                if key == "I" and "L" in self._held:
                    return
                self._held.add(key)
                if key == "I":
                    # Requiring a release before another press makes the
                    # toggle immune to duplicate KEY_PRESS and KEY_REPEAT.
                    self._held.difference_update(_MOTION_KEYS)
                    self._takeover_active = not self._takeover_active
                    self._takeover_pressed = self._takeover_active
                    self._takeover_released = not self._takeover_active
                elif key == "1" and self._active_arm != "left":
                    self._active_arm = "left"
                    self._arm_changed = True
                elif key == "2" and self._active_arm != "right":
                    self._active_arm = "right"
                    self._arm_changed = True
                elif key == "K" and self._takeover_active:
                    self._gripper_toggles.append(self._active_arm)
                elif key in {"N", "ENTER"}:
                    self._accept_requested = True
                elif key == "R":
                    self._save_retry_requested = True
                elif key == "BACKSPACE":
                    self._abort_requested = True
                elif key == "L":
                    was_active = self._takeover_active
                    self._takeover_active = False
                    # Clear motion/command keys, but preserve a physically
                    # held I so a duplicate KEY_PRESS cannot re-arm takeover.
                    self._held.intersection_update({"I", "L"})
                    self._takeover_pressed = False
                    self._takeover_released = self._takeover_released or was_active
            else:
                self._held.discard(key)

    def touch(self) -> None:
        """Refresh the input heartbeat for a held key-repeat event."""
        with self._lock:
            self._last_input_time = self._clock()

    def snapshot(self) -> KeyboardSnapshot:
        """Read current held keys and consume one-shot operator commands."""
        with self._lock:
            # A lost release must stop Cartesian motion, but a timeout must
            # never silently hand control back to the policy in toggle mode.
            if self._held and self._clock() - self._last_input_time > self.deadman_timeout:
                self._held.clear()
            translation = sum(
                (_TRANSLATION_KEYS[key] for key in self._held if key in _TRANSLATION_KEYS),
                start=np.zeros(3),
            )
            rotation = sum(
                (_ROTATION_KEYS[key] for key in self._held if key in _ROTATION_KEYS),
                start=np.zeros(3),
            )
            snapshot = KeyboardSnapshot(
                # The field name is kept for compatibility; it now represents
                # the latched intervention-active state.
                deadman=self._takeover_active,
                active_arm=self._active_arm,
                delta_pose=np.concatenate([translation * self.pos_step, rotation * self.rot_step]),
                gripper_toggles=tuple(self._gripper_toggles),
                takeover_pressed=self._takeover_pressed,
                takeover_released=self._takeover_released,
                arm_changed=self._arm_changed,
                accept_requested=self._accept_requested,
                save_retry_requested=self._save_retry_requested,
                abort_requested=self._abort_requested,
            )
            self._gripper_toggles.clear()
            self._takeover_pressed = False
            self._takeover_released = False
            self._arm_changed = False
            self._accept_requested = False
            self._save_retry_requested = False
            self._abort_requested = False
            return snapshot


class KitKeyboardDevice:
    """Omniverse keyboard-event adapter with press and release support."""

    def __init__(self, pos_step: float = 0.005, rot_step: float = 0.02, deadman_timeout: float = 2.0):
        import carb
        import omni.appwindow

        self._carb = carb
        self.state = KeyboardState(
            pos_step=pos_step,
            rot_step=rot_step,
            deadman_timeout=deadman_timeout,
        )
        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("No Omniverse app window is available; keyboard teleop cannot run headless.")
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        if self._keyboard is None:
            raise RuntimeError("The Omniverse window did not expose a keyboard device.")
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def _on_keyboard_event(self, event: Any, *args: Any) -> bool:
        del args
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            self.state.handle_key(event.input.name, pressed=True)
        elif event.type == self._carb.input.KeyboardEventType.KEY_RELEASE:
            self.state.handle_key(event.input.name, pressed=False)
        elif event.type == getattr(self._carb.input.KeyboardEventType, "KEY_REPEAT", None):
            self.state.touch()
        return True

    def snapshot(self) -> KeyboardSnapshot:
        return self.state.snapshot()

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def help_text() -> str:
        return (
            "I=toggle manual control on/off | 1/2=left/right arm | "
            "W/S x, A/D y, Q/E z | Z/X roll, T/G pitch, C/V yaw | "
            "K=toggle selected gripper while manual | N or Enter=save/finish | "
            "R=save/retry same layout | Backspace=reject/retry same layout | "
            "L=emergency manual-off/clear held keys"
        )


def compose_world_delta_pose(pose: np.ndarray, delta_pose: np.ndarray) -> np.ndarray:
    """Apply an xyz/RPY increment in the environment frame to a qwxyz pose."""
    import transforms3d as t3d

    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    delta_pose = np.asarray(delta_pose, dtype=np.float64).reshape(6)
    result = pose.copy()
    result[:3] += delta_pose[:3]
    current_rotation = t3d.quaternions.quat2mat(pose[3:])
    delta_rotation = t3d.euler.euler2mat(*delta_pose[3:], axes="sxyz")
    quaternion = t3d.quaternions.mat2quat(delta_rotation @ current_rotation)
    quaternion /= np.linalg.norm(quaternion)
    result[3:] = quaternion
    return result


class CartesianTeleopController:
    """Convert selected-arm Cartesian keyboard deltas to safe full joint actions."""

    def __init__(self, task_env, max_joint_delta: float = 0.35):
        self.task_env = task_env
        self.max_joint_delta = float(max_joint_delta)
        self._robots = {
            robot.arm_name.split("_")[0]: robot for robot in task_env.robot_manager.robot_list if robot.type == "target"
        }
        if set(self._robots) != {"left", "right"}:
            raise ValueError(
                "Keyboard intervention currently requires the dual-arm left/right configuration; "
                f"found arms={sorted(self._robots)}"
            )
        self._target_pose: dict[str, np.ndarray] = {}
        self._gripper_target: dict[str, float] = {}
        self._last_active_arm: str | None = None
        self._was_deadman = False

    def reset(self, obs: dict) -> None:
        self._target_pose = {arm: self._read_pose(arm) for arm in self._robots}
        self._gripper_target = {
            arm: float(np.clip(np.asarray(obs["state"][f"{arm}_ee_joint_state"]).reshape(-1)[0], 0.0, 1.0))
            for arm in self._robots
        }
        self._last_active_arm = None
        self._was_deadman = False

    def _read_pose(self, arm: str) -> np.ndarray:
        pose = self.task_env.robot_manager.get_real_endpose(self._robots[arm], env_idx_list=[0], is_relative=True)[0]
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        if not np.isfinite(pose).all():
            raise ValueError(f"Non-finite {arm} end-effector pose: {pose}")
        return pose

    @staticmethod
    def _hold_action(obs: dict) -> dict[str, np.ndarray]:
        state = obs["state"]
        return {
            "left_arm_joint_state": np.asarray(state["left_arm_joint_state"], dtype=np.float64).copy(),
            "left_ee_joint_state": np.asarray(state["left_ee_joint_state"], dtype=np.float64).copy(),
            "right_arm_joint_state": np.asarray(state["right_arm_joint_state"], dtype=np.float64).copy(),
            "right_ee_joint_state": np.asarray(state["right_ee_joint_state"], dtype=np.float64).copy(),
        }

    def build_action(self, obs: dict, snapshot: KeyboardSnapshot) -> tuple[dict, dict]:
        if not self._target_pose:
            self.reset(obs)
        arm = snapshot.active_arm
        if arm not in self._robots:
            raise ValueError(f"Unknown active arm: {arm}")

        reanchor = snapshot.takeover_pressed or snapshot.arm_changed or not self._was_deadman
        if reanchor or self._last_active_arm != arm:
            self._target_pose[arm] = self._read_pose(arm)
            self._gripper_target[arm] = float(
                np.clip(np.asarray(obs["state"][f"{arm}_ee_joint_state"]).reshape(-1)[0], 0.0, 1.0)
            )

        for toggle_arm in snapshot.gripper_toggles:
            if toggle_arm in self._gripper_target:
                self._gripper_target[toggle_arm] = 1.0 - self._gripper_target[toggle_arm]

        hold_action = self._hold_action(obs)
        candidate_pose = compose_world_delta_pose(self._target_pose[arm], snapshot.delta_pose)
        current_joint = hold_action[f"{arm}_arm_joint_state"]
        result = self.task_env.robot_manager.solve_ik(
            target_pose=candidate_pose.tolist(), env_idx=0, robot=self._robots[arm]
        )
        joint_value = np.asarray(result.get("joint_value", []), dtype=np.float64).reshape(-1)
        ik_success = (
            result.get("status") == "Success"
            and joint_value.shape == current_joint.shape
            and np.isfinite(joint_value).all()
            and np.max(np.abs(joint_value - current_joint), initial=0.0) <= self.max_joint_delta
        )

        self._last_active_arm = arm
        self._was_deadman = snapshot.deadman
        if not ik_success:
            self._target_pose[arm] = self._read_pose(arm)
            return hold_action, {
                "action_source": "safety_hold",
                "intervention_mask": 0,
                "ik_success": 0,
                "active_arm": arm,
            }

        self._target_pose[arm] = candidate_pose
        hold_action[f"{arm}_arm_joint_state"] = joint_value
        hold_action[f"{arm}_ee_joint_state"] = np.asarray([self._gripper_target[arm]], dtype=np.float64)
        return hold_action, {
            "action_source": "human",
            "intervention_mask": 1,
            "ik_success": 1,
            "active_arm": arm,
        }
