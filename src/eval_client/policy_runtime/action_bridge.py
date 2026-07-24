"""Pure canonical-action adapters for the current RoboDojo executor."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.eval_client.policy_runtime.canonical import CanonicalActionChunk


def iter_arx_x5_eval_actions(
    chunk: CanonicalActionChunk,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield one legacy EvalEnv absolute-joint action per canonical target."""

    if not isinstance(chunk, CanonicalActionChunk):
        raise TypeError("chunk must be CanonicalActionChunk")
    for index in range(chunk.horizon):
        yield {
            "left_arm_joint_state": chunk.left_arm_joint_position[index],
            "left_ee_joint_state": chunk.left_gripper_open_fraction[index],
            "right_arm_joint_state": chunk.right_arm_joint_position[index],
            "right_ee_joint_state": chunk.right_gripper_open_fraction[index],
        }
