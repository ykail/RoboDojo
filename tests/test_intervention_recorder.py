import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
    import h5py

    from src.eval_client.intervention_recorder import EpisodeRecorder

    HAS_HDF5_STACK = True
except ImportError:
    HAS_HDF5_STACK = False


def _action(offset):
    return {
        "left_arm_joint_state": np.arange(6, dtype=np.float32) + offset,
        "left_ee_joint_state": np.array([1.0], dtype=np.float32),
        "right_arm_joint_state": np.arange(6, dtype=np.float32) + offset + 10,
        "right_ee_joint_state": np.array([0.0], dtype=np.float32),
    }


class InterventionRecorderTest(unittest.TestCase):
    @unittest.skipUnless(HAS_HDF5_STACK, "cv2/h5py are available in the RoboDojo runtime")
    def test_round_trip_has_pi05_joint_and_jpeg_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(
                tmp,
                {
                    "task_name": "stack_bowls",
                    "env_config": "arx_x5",
                    "layout_id": 3,
                    "base_checkpoint": "test",
                },
                frequency=25,
            )
            image = np.zeros((16, 24, 3), dtype=np.uint8)
            image[:, :, 0] = 255
            for index in range(2):
                action = _action(index)
                recorder.append(
                    obs={
                        "instruction": "stack the bowls",
                        "state": action,
                        "vision": {
                            name: {"color": image}
                            for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
                        },
                    },
                    policy_action=action,
                    human_action=action if index else None,
                    executed_action=action,
                    control={
                        "timestamp": float(index),
                        "action_source": "human" if index else "policy",
                        "intervention_mask": index,
                        "ik_success": 1,
                        "active_arm": "left",
                        "takeover_edge": index,
                        "chunk_id": 0,
                        "chunk_index": index,
                    },
                )

            saved = recorder.finalize(accepted=True, success=True, reason="test")
            self.assertIsNotNone(saved)
            self.assertIn("stack_bowls/arx_x5/data", saved)
            self.assertFalse(any(Path(tmp).rglob("*.partial")))
            with h5py.File(saved, "r") as episode:
                self.assertEqual(episode["state/left_arm_joint_states"].shape, (2, 6))
                self.assertEqual(episode["state/left_ee_joint_states"].shape, (2,))
                self.assertEqual(episode["action/right_arm_joint_states"].shape, (2, 6))
                self.assertEqual(episode["control/intervention_mask"][:].tolist(), [0, 1])
                instructions = json.loads(episode["instructions"][()].decode("utf-8"))
                self.assertEqual(instructions, ["stack the bowls"])
                encoded = episode["vision/cam_head/colors"][0].rstrip(b"\0")
                decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape, image.shape)
                self.assertTrue(episode.attrs["complete"])


if __name__ == "__main__":
    unittest.main()
