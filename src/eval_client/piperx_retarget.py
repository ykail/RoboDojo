"""Relative direct-joint retargeting from PiPER-X leaders to ARX X5 sim."""

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
from src.eval_client.policy_runtime.execution_profile import ARX_X5_SIM_ARM_LIMITS

EMBODIMENT_PROFILE = "arx_x5_piperx_relative_joint_v1"
PIPERX_GRIPPER_STROKE_M = 0.102
_ARMS = ("left", "right")
_JOINT_SIGNS = np.asarray([1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float64)
_ARX_LOWER = np.asarray(ARX_X5_SIM_ARM_LIMITS.lower, dtype=np.float64)
_ARX_UPPER = np.asarray(ARX_X5_SIM_ARM_LIMITS.upper, dtype=np.float64)


class PiperXRetargetError(RuntimeError):
    """A physical sample or direct-joint target is unusable."""


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


@dataclass(frozen=True)
class RetargetConfig:
    """Versioned PiPER-X-to-ARX direct-joint adapter."""

    embodiment_profile: str = EMBODIMENT_PROFILE
    max_joint_delta_rad: float = 0.35

    def __post_init__(self) -> None:
        if self.embodiment_profile != EMBODIMENT_PROFILE:
            raise ValueError(
                f"unsupported PiPER-X embodiment profile {self.embodiment_profile!r}; "
                f"expected {EMBODIMENT_PROFILE!r}"
            )
        max_joint_delta = _finite_scalar(self.max_joint_delta_rad, label="max_joint_delta_rad")
        if max_joint_delta <= 0:
            raise ValueError("max_joint_delta_rad must be positive and finite")
        object.__setattr__(self, "max_joint_delta_rad", max_joint_delta)


def sim_targets_from_env(task_env: Any, obs: dict[str, Any]) -> SimTargets:
    """Read measured dual-arm simulator joints and normalized grippers."""

    robots = {
        robot.arm_name.split("_")[0]: robot for robot in task_env.robot_manager.robot_list if robot.type == "target"
    }
    if set(robots) != set(_ARMS):
        raise PiperXRetargetError(f"PiPER-X DAgger requires left/right target arms, got {sorted(robots)}")
    targets: dict[str, SimArmTarget] = {}
    for arm in _ARMS:
        robot = robots[arm]
        joints = np.asarray(
            task_env.robot_manager.get_joint(robot, env_idx_list=[0])[0],
            dtype=np.float64,
        )[:6]
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise PiperXRetargetError(f"{arm} simulator joints are invalid")
        raw_gripper = float(
            task_env.robot_manager.get_end_effector_real_val(robot, env_idx_list=[0])[0][0]
        )
        low, high = (float(value) for value in robot.gripper_scale)
        if not high > low:
            raise PiperXRetargetError(f"{arm} simulator gripper scale is invalid")
        gripper = (raw_gripper - low) / (high - low)
        if robot.gripper_move["sign"] != 1:
            gripper = 1.0 - gripper
        targets[arm] = SimArmTarget(
            joints_rad=tuple(joints.tolist()),
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
        self._operator_joint_anchors: dict[str, np.ndarray] = {}
        self._sim_joint_anchors: dict[str, np.ndarray] = {}
        self._operator_gripper_anchors: dict[str, float] = {}
        self._sim_gripper_anchors: dict[str, float] = {}
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
            self._operator_joint_anchors[arm] = _array(
                leader_arm.joints_rad,
                (6,),
                label=f"{arm} PiPER-X joint anchor",
            )
            self._sim_joint_anchors[arm] = _array(
                sim_arm.joints_rad,
                (6,),
                label=f"{arm} ARX joint anchor",
            )
            self._operator_gripper_anchors[arm] = float(leader_arm.gripper_m)
            self._sim_gripper_anchors[arm] = float(sim_arm.gripper)
        self._generation = sample.generation

    def exit(self) -> None:
        self._operator_joint_anchors.clear()
        self._sim_joint_anchors.clear()
        self._operator_gripper_anchors.clear()
        self._sim_gripper_anchors.clear()
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
        desired_joints: dict[str, list[float]] = {}
        for arm in _ARMS:
            operator_arm: OperatorArmSample = getattr(sample, arm)
            try:
                operator_joints = _array(
                    operator_arm.joints_rad,
                    (6,),
                    label=f"{arm} effective PiPER-X joints",
                )
                candidate = self._sim_joint_anchors[arm] + _JOINT_SIGNS * (
                    operator_joints - self._operator_joint_anchors[arm]
                )
                current = action[f"{arm}_arm_joint_state"]
                if candidate.shape != current.shape or not np.isfinite(candidate).all():
                    raise PiperXRetargetError(
                        f"direct mapping returned shape/value {candidate.shape}; expected {current.shape}"
                    )
                if np.any(candidate < _ARX_LOWER) or np.any(candidate > _ARX_UPPER):
                    raise PiperXRetargetError(
                        "direct joint target exceeds the active ARX X5 limits"
                    )
                joint_delta = float(np.max(np.abs(candidate - current), initial=0.0))
                if joint_delta > self.config.max_joint_delta_rad:
                    raise PiperXRetargetError(
                        f"joint delta {joint_delta:.4f} rad exceeds {self.config.max_joint_delta_rad:.4f} rad"
                    )
                action[f"{arm}_arm_joint_state"] = candidate
                desired_joints[arm] = candidate.tolist()
                action[f"{arm}_ee_joint_state"] = np.asarray(
                    [
                        np.clip(
                            self._sim_gripper_anchors[arm]
                            + (
                                float(operator_arm.gripper_m)
                                - self._operator_gripper_anchors[arm]
                            )
                            / PIPERX_GRIPPER_STROKE_M,
                            0.0,
                            1.0,
                        )
                    ],
                    dtype=np.float64,
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
            "retarget_target_joints_rad": desired_joints,
        }
