"""RoboDojo-owned execution profiles for policy connections."""

from __future__ import annotations

from dataclasses import dataclass

from src.eval_client.policy_runtime.canonical import (
    ACTION_SCHEMA_ID,
    ARX_X5_SIM_OBSERVATION_SPEC,
    OBSERVATION_SCHEMA_ID,
    ROBOT_SCHEMA_ID,
    ActionValidationSpec,
    JointLimits,
    ObservationValidationSpec,
)

# Current Assets/Robots/x5/ARX.usd articulation limits. These are simulation
# bounds, not physical-robot safety limits.
ARX_X5_SIM_ARM_LIMITS = JointLimits(
    lower=(-10.0, -10.0, -10.0, -10.0, -10.0, -3.14),
    upper=(10.0, 10.0, 10.0, 10.0, 10.0, 3.14),
)

# v1 executes a returned policy chunk open-loop in index order. A local task
# terminal or operator intervention may preempt only between synchronous
# control targets; the unexecuted suffix is then permanently stale.
ACTION_CHUNK_CONSUMPTION = "sequential_until_terminal_or_preempted"
ACTION_PREEMPTION_BOUNDARY = "between_control_targets"
ACTION_NEXT_INFER_OBSERVATION = "after_last_executed_target"


@dataclass(frozen=True, slots=True)
class PolicyExecutionProfile:
    """Concrete schemas and client-side validation constraints."""

    observation_spec: ObservationValidationSpec
    action_spec: ActionValidationSpec

    def __post_init__(self) -> None:
        if not isinstance(self.observation_spec, ObservationValidationSpec):
            raise TypeError("observation_spec must be ObservationValidationSpec")
        if not isinstance(self.action_spec, ActionValidationSpec):
            raise TypeError("action_spec must be ActionValidationSpec")

    def schemas_payload(self) -> dict[str, str]:
        return {
            "observation": OBSERVATION_SCHEMA_ID,
            "action": ACTION_SCHEMA_ID,
            "robot": ROBOT_SCHEMA_ID,
        }

    def execution_payload(self) -> dict[str, object]:
        return {
            "images": {
                "head": list(self.observation_spec.head_image_shape),
                "left_wrist": list(
                    self.observation_spec.left_wrist_image_shape,
                ),
                "right_wrist": list(
                    self.observation_spec.right_wrist_image_shape,
                ),
            },
            "action": {
                "horizon": self.action_spec.expected_horizon,
                "control_dt_s": self.action_spec.expected_control_dt_s,
                "chunk_consumption": ACTION_CHUNK_CONSUMPTION,
                "preemption_boundary": ACTION_PREEMPTION_BOUNDARY,
                "next_infer_observation": ACTION_NEXT_INFER_OBSERVATION,
                "left_arm_joint_limits": _joint_limits_payload(
                    self.action_spec.left_arm_limits,
                ),
                "right_arm_joint_limits": _joint_limits_payload(
                    self.action_spec.right_arm_limits,
                ),
            },
        }


def _joint_limits_payload(limits: JointLimits) -> dict[str, list[float]]:
    return {
        "lower": list(limits.lower),
        "upper": list(limits.upper),
    }


ARX_X5_SIM_PI05_PROFILE = PolicyExecutionProfile(
    observation_spec=ARX_X5_SIM_OBSERVATION_SPEC,
    action_spec=ActionValidationSpec(
        expected_horizon=50,
        expected_control_dt_s=0.04,
        left_arm_limits=ARX_X5_SIM_ARM_LIMITS,
        right_arm_limits=ARX_X5_SIM_ARM_LIMITS,
    ),
)
