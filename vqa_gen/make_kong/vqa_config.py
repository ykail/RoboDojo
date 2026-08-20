"""Task-local rendering configuration for make_kong VQA collection."""

from collections.abc import Sequence

# Keep the left target X5's gripper closed in make_kong VQA renders.  The
# collector maps this normalized opening to the robot's physical joint range
# and mimic joint, so this setting remains local to this task.
LEFT_TARGET_GRIPPER_OPENING = 0.0


def gripper_joint_positions(
    opening: float,
    *,
    gripper_scale: Sequence[float],
    gripper_sign: int,
    gripper_mimic: Sequence[float],
) -> tuple[float, float]:
    """Convert a normalized opening to the two physical gripper joints."""

    if not 0.0 <= opening <= 1.0:
        raise ValueError(f"gripper opening must be in [0, 1], got {opening}")
    lower, upper = map(float, gripper_scale)
    if upper < lower:
        raise ValueError(f"gripper scale must be ordered, got {gripper_scale}")
    if len(gripper_mimic) != 3:
        raise ValueError(f"gripper mimic must contain three values, got {gripper_mimic}")
    primary = lower + opening * (upper - lower)
    if gripper_sign != 1:
        primary = upper - opening * (upper - lower)
    multiplier, offset = map(float, gripper_mimic[1:])
    return primary, primary * multiplier + offset
