"""Relative dual-arm SE(3) retargeting from PiPER-X leaders to ARX X5 sim."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from src.eval_client.piperx_bridge_client import (
    OperatorArmSample,
    OperatorSample,
    SimArmTarget,
    SimTargets,
)

EMBODIMENT_PROFILE = "arx_x5_piperx_relative_v1"
PIPERX_GRIPPER_STROKE_M = 0.102
_ARMS = ("left", "right")


class PiperXRetargetError(RuntimeError):
    """A physical sample, relative pose or IK result is unsafe or unusable."""


def _array(value: Any, shape: tuple[int, ...], *, label: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, float, np.number)) for item in raw.flat
    ):
        raise ValueError(f"{label} must be numeric with shape {shape}, got {value}")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}, got {result}")
    return result


def _finite_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _normalized_quaternion(value: Any, *, label: str) -> np.ndarray:
    quaternion = _array(value, (4,), label=label)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ValueError(f"{label} quaternion norm is zero")
    return quaternion / norm


def quaternion_to_matrix(quaternion: Any) -> np.ndarray:
    """Convert a normalized-or-normalizable qwxyz quaternion to a matrix."""

    w, x, y, z = _normalized_quaternion(quaternion, label="quaternion")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: Any) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to canonical qwxyz."""

    rotation = _array(matrix, (3, 3), label="rotation")
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


def _rotation_angle(rotation: np.ndarray) -> float:
    return math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


@dataclass(frozen=True)
class RetargetConfig:
    """Versioned PiPER-X-to-ARX relative-motion adapter and safety gates.

    Absolute physical and simulator poses are paired automatically at takeover.
    The supported profile uses the canonical right-handed base convention from
    the two bundled robot descriptions; it is not a per-machine calibration.
    """

    embodiment_profile: str = EMBODIMENT_PROFILE
    max_translation_from_anchor_m: float = 0.25
    max_rotation_from_anchor_rad: float = 1.57
    max_joint_delta_rad: float = 0.35

    def __post_init__(self) -> None:
        if self.embodiment_profile != EMBODIMENT_PROFILE:
            raise ValueError(
                f"unsupported PiPER-X embodiment profile {self.embodiment_profile!r}; "
                f"expected {EMBODIMENT_PROFILE!r}"
            )
        max_translation = _finite_scalar(
            self.max_translation_from_anchor_m,
            label="max_translation_from_anchor_m",
        )
        max_rotation = _finite_scalar(
            self.max_rotation_from_anchor_rad,
            label="max_rotation_from_anchor_rad",
        )
        max_joint_delta = _finite_scalar(self.max_joint_delta_rad, label="max_joint_delta_rad")
        if max_translation <= 0 or max_rotation <= 0 or max_joint_delta <= 0:
            raise ValueError("retarget safety limits must be positive and finite")
        object.__setattr__(self, "max_translation_from_anchor_m", max_translation)
        object.__setattr__(self, "max_rotation_from_anchor_rad", max_rotation)
        object.__setattr__(self, "max_joint_delta_rad", max_joint_delta)


class RelativeSE3Retargeter:
    """Map one accepted PiPER-X leader sample into relative ARX SE(3)."""

    def __init__(self, config: RetargetConfig):
        self.config = config
        self._leader_anchor: np.ndarray | None = None
        self._sim_anchor: np.ndarray | None = None
        self._leader_gripper_anchor_m: float | None = None
        self._sim_gripper_anchor: float | None = None

    def anchor(
        self,
        leader_pose: Any,
        sim_pose: Any,
        *,
        leader_gripper_m: float,
        sim_gripper: float,
    ) -> None:
        self._leader_anchor = _array(leader_pose, (7,), label="leader anchor pose").copy()
        self._leader_anchor[3:] = _normalized_quaternion(self._leader_anchor[3:], label="leader anchor quaternion")
        self._sim_anchor = _array(sim_pose, (7,), label="sim anchor pose").copy()
        self._sim_anchor[3:] = _normalized_quaternion(self._sim_anchor[3:], label="sim anchor quaternion")
        self._leader_gripper_anchor_m = float(leader_gripper_m)
        self._sim_gripper_anchor = float(sim_gripper)
        if not math.isfinite(self._leader_gripper_anchor_m):
            raise ValueError("leader gripper anchor must be finite")
        if not 0.0 <= self._sim_gripper_anchor <= 1.0:
            raise ValueError("sim gripper anchor must be in [0, 1]")

    def clear(self) -> None:
        self._leader_anchor = None
        self._sim_anchor = None
        self._leader_gripper_anchor_m = None
        self._sim_gripper_anchor = None

    def map_pose(self, leader_pose: Any) -> np.ndarray:
        if self._leader_anchor is None or self._sim_anchor is None:
            raise PiperXRetargetError("retargeter has no takeover anchor")
        leader = _array(leader_pose, (7,), label="leader pose")
        leader_rotation = quaternion_to_matrix(leader[3:])
        anchor_leader_rotation = quaternion_to_matrix(self._leader_anchor[3:])
        sim_anchor_rotation = quaternion_to_matrix(self._sim_anchor[3:])

        leader_translation_delta = leader[:3] - self._leader_anchor[:3]
        sim_translation_delta = leader_translation_delta
        leader_rotation_delta = leader_rotation @ anchor_leader_rotation.T
        sim_rotation_delta = leader_rotation_delta
        translation_distance = float(np.linalg.norm(sim_translation_delta))
        rotation_distance = _rotation_angle(sim_rotation_delta)
        if translation_distance > self.config.max_translation_from_anchor_m:
            raise PiperXRetargetError(
                f"retarget translation {translation_distance:.4f} m exceeds "
                f"{self.config.max_translation_from_anchor_m:.4f} m"
            )
        if rotation_distance > self.config.max_rotation_from_anchor_rad:
            raise PiperXRetargetError(
                f"retarget rotation {rotation_distance:.4f} rad exceeds "
                f"{self.config.max_rotation_from_anchor_rad:.4f} rad"
            )
        return np.concatenate(
            [
                self._sim_anchor[:3] + sim_translation_delta,
                matrix_to_quaternion(sim_rotation_delta @ sim_anchor_rotation),
            ]
        )

    def map_gripper(self, gripper_m: float) -> float:
        if self._leader_gripper_anchor_m is None or self._sim_gripper_anchor is None:
            raise PiperXRetargetError("retargeter has no gripper takeover anchor")
        normalized_delta = (float(gripper_m) - self._leader_gripper_anchor_m) / PIPERX_GRIPPER_STROKE_M
        return float(np.clip(self._sim_gripper_anchor + normalized_delta, 0.0, 1.0))


def sim_targets_from_env(task_env: Any, obs: dict[str, Any]) -> SimTargets:
    """Read accepted dual-arm simulator poses and normalized grippers."""

    robots = {
        robot.arm_name.split("_")[0]: robot for robot in task_env.robot_manager.robot_list if robot.type == "target"
    }
    if set(robots) != set(_ARMS):
        raise PiperXRetargetError(f"PiPER-X DAgger requires left/right target arms, got {sorted(robots)}")
    state = obs.get("state")
    if not isinstance(state, dict):
        raise PiperXRetargetError("sim observation has no state mapping")
    targets: dict[str, SimArmTarget] = {}
    for arm in _ARMS:
        pose = task_env.robot_manager.get_real_endpose(robots[arm], env_idx_list=[0], is_relative=True)[0]
        gripper = _array(state[f"{arm}_ee_joint_state"], (1,), label=f"{arm} gripper state")[0]
        targets[arm] = SimArmTarget(
            pose=tuple(_array(pose, (7,), label=f"{arm} sim pose").tolist()),
            gripper=float(np.clip(gripper, 0.0, 1.0)),
        )
    return SimTargets(left=targets["left"], right=targets["right"])


class ArxPiperXRetargetController:
    """Convert a dual-leader sample to one safe full ARX X5 joint action."""

    def __init__(self, task_env: Any, config: RetargetConfig | None = None):
        self.task_env = task_env
        self.config = RetargetConfig() if config is None else config
        self._robots = {
            robot.arm_name.split("_")[0]: robot for robot in task_env.robot_manager.robot_list if robot.type == "target"
        }
        if set(self._robots) != set(_ARMS):
            raise ValueError(
                "PiPER-X DAgger requires the dual-arm left/right ARX X5 configuration; "
                f"found arms={sorted(self._robots)}"
            )
        self._retargeters = {
            "left": RelativeSE3Retargeter(self.config),
            "right": RelativeSE3Retargeter(self.config),
        }
        self._generation: int | None = None

    @staticmethod
    def hold_action(obs: dict[str, Any]) -> dict[str, np.ndarray]:
        state = obs["state"]
        return {
            f"{arm}_arm_joint_state": np.asarray(state[f"{arm}_arm_joint_state"], dtype=np.float64).copy()
            for arm in _ARMS
        } | {
            f"{arm}_ee_joint_state": np.asarray(state[f"{arm}_ee_joint_state"], dtype=np.float64).copy()
            for arm in _ARMS
        }

    def enter(self, sample: OperatorSample, sim: SimTargets) -> None:
        if sample.mode != "intervention":
            raise PiperXRetargetError("cannot anchor a non-intervention operator sample")
        for arm in _ARMS:
            leader_arm = getattr(sample, arm)
            sim_arm = getattr(sim, arm)
            self._retargeters[arm].anchor(
                leader_arm.pose,
                sim_arm.pose,
                leader_gripper_m=leader_arm.gripper_m,
                sim_gripper=sim_arm.gripper,
            )
        self._generation = sample.generation

    def exit(self) -> None:
        for retargeter in self._retargeters.values():
            retargeter.clear()
        self._generation = None

    def build_action(
        self,
        obs: dict[str, Any],
        sample: OperatorSample,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if sample.mode != "intervention" or self._generation != sample.generation:
            raise PiperXRetargetError("operator generation does not match the active intervention anchor")
        action = self.hold_action(obs)
        failures: dict[str, str] = {}
        desired_poses: dict[str, list[float]] = {}
        for arm in _ARMS:
            operator_arm: OperatorArmSample = getattr(sample, arm)
            retargeter = self._retargeters[arm]
            try:
                desired_pose = retargeter.map_pose(operator_arm.pose)
                desired_poses[arm] = desired_pose.tolist()
                result = self.task_env.robot_manager.solve_ik(
                    target_pose=desired_pose.tolist(),
                    env_idx=0,
                    robot=self._robots[arm],
                )
                candidate = np.asarray(result.get("joint_value", []), dtype=np.float64).reshape(-1)
                current = action[f"{arm}_arm_joint_state"]
                if result.get("status") != "Success":
                    raise PiperXRetargetError(f"IK status={result.get('status')!r}")
                if candidate.shape != current.shape or not np.isfinite(candidate).all():
                    raise PiperXRetargetError(
                        f"IK returned unsafe shape/value {candidate.shape}; expected {current.shape}"
                    )
                joint_delta = float(np.max(np.abs(candidate - current), initial=0.0))
                if joint_delta > self.config.max_joint_delta_rad:
                    raise PiperXRetargetError(
                        f"joint delta {joint_delta:.4f} rad exceeds {self.config.max_joint_delta_rad:.4f} rad"
                    )
                action[f"{arm}_arm_joint_state"] = candidate
                action[f"{arm}_ee_joint_state"] = np.asarray(
                    [retargeter.map_gripper(operator_arm.gripper_m)], dtype=np.float64
                )
            except (KeyError, TypeError, ValueError, PiperXRetargetError) as exc:
                failures[arm] = str(exc)

        if failures:
            return self.hold_action(obs), {
                "action_source": "safety_hold",
                "intervention_mask": 0,
                "ik_success": 0,
                "active_arm": "both",
                "operator_generation": sample.generation,
                "retarget_failures": failures,
            }
        return action, {
            "action_source": "human",
            "intervention_mask": 1,
            "ik_success": 1,
            "active_arm": "both",
            "operator_generation": sample.generation,
            "retarget_target_pose": desired_poses,
        }
