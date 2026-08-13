"""Pure helpers for the paired real-robot state pool used by VQA collectors."""

from typing import Any

import numpy as np

STATE_DIMENSION = 16
POSE_DIMENSION = 7
POOL_COLUMNS = (
    "source_index",
    "source_episode_index",
    "source_frame_index",
    "source_timestamp",
    "left_ee_pose_wxyz",
    "left_gripper",
    "right_ee_pose_wxyz",
    "right_gripper",
)


class RobotStatePoolError(ValueError):
    """A source state row cannot be used as a paired VQA robot snapshot."""


def split_paired_state(state: Any) -> tuple[list[float], float, list[float], float]:
    """Validate and split the source [left pose/gripper, right pose/gripper] vector."""

    values = np.asarray(state, dtype=np.float32).reshape(-1)
    if values.size != STATE_DIMENSION or not np.all(np.isfinite(values)):
        raise RobotStatePoolError(f"observation.state must contain {STATE_DIMENSION} finite values")
    left_pose = values[:POSE_DIMENSION]
    right_pose = values[8 : 8 + POSE_DIMENSION]
    for side, pose in (("left", left_pose), ("right", right_pose)):
        norm = float(np.linalg.norm(pose[3:]))
        if not np.isclose(norm, 1.0, atol=0.05):
            raise RobotStatePoolError(f"{side} quaternion norm must be near 1.0, got {norm:.6f}")
    return (
        [float(value) for value in left_pose],
        float(values[7]),
        [float(value) for value in right_pose],
        float(values[15]),
    )


def state_pool_record(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one LeRobot row into an explicit, provenance-preserving pool row."""

    left_pose, left_gripper, right_pose, right_gripper = split_paired_state(row["observation.state"])
    return {
        "source_index": int(row["index"]),
        "source_episode_index": int(row["episode_index"]),
        "source_frame_index": int(row["frame_index"]),
        "source_timestamp": float(row["timestamp"]),
        "left_ee_pose_wxyz": left_pose,
        "left_gripper": left_gripper,
        "right_ee_pose_wxyz": right_pose,
        "right_gripper": right_gripper,
    }


def validate_pool_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate a materialized pool row before it is loaded into Isaac."""

    missing = [column for column in POOL_COLUMNS if column not in record]
    if missing:
        raise RobotStatePoolError(f"state-pool row is missing columns: {missing}")
    left_pose, left_gripper, right_pose, right_gripper = split_paired_state(
        [*record["left_ee_pose_wxyz"], record["left_gripper"], *record["right_ee_pose_wxyz"], record["right_gripper"]]
    )
    return {
        "source_index": int(record["source_index"]),
        "source_episode_index": int(record["source_episode_index"]),
        "source_frame_index": int(record["source_frame_index"]),
        "source_timestamp": float(record["source_timestamp"]),
        "left_ee_pose_wxyz": left_pose,
        "left_gripper": left_gripper,
        "right_ee_pose_wxyz": right_pose,
        "right_gripper": right_gripper,
    }


def load_robot_state_pool(path: str) -> list[dict[str, Any]]:
    """Read every validated paired robot snapshot from a Parquet state pool."""

    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(POOL_COLUMNS))
    if table.num_rows == 0:
        raise RobotStatePoolError("state pool is empty")
    return [validate_pool_record(row) for row in table.to_pylist()]
