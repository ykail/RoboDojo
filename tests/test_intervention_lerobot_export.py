from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import pyarrow.parquet as pq

    from scripts.RoboDojo.export_interventions_lerobot_v30 import (
        DatasetConfig,
        convert_one,
        create_empty_dataset,
        extract_intervention_mask,
        finalize_dataset,
    )
    from src.eval_client.intervention_recorder import EpisodeRecorder

    HAS_LEROBOT_STACK = True
except ImportError:
    HAS_LEROBOT_STACK = False


def _action(offset):
    return {
        "left_arm_joint_state": np.arange(6, dtype=np.float32) + offset,
        "left_ee_joint_state": np.array([1.0], dtype=np.float32),
        "right_arm_joint_state": np.arange(6, dtype=np.float32) + offset + 10,
        "right_ee_joint_state": np.array([0.0], dtype=np.float32),
    }


@unittest.skipUnless(HAS_LEROBOT_STACK, "LeRobot, PyArrow, OpenCV and HDF5 are available")
class InterventionLeRobotExportTest(unittest.TestCase):
    def test_mask_defaults_to_autonomous_and_rejects_invalid_values(self):
        np.testing.assert_array_equal(extract_intervention_mask({}, 3), np.zeros(3, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "horizon mismatch"):
            extract_intervention_mask({"control": {"intervention_mask": [0, 1]}}, 3)
        with self.assertRaisesRegex(ValueError, "0/1"):
            extract_intervention_mask({"control": {"intervention_mask": [0, 2, 1]}}, 3)

    def test_repo_id_cannot_escape_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "must resolve below"):
                create_empty_dataset(
                    repo_id="../escape",
                    robot_type="arx_x5",
                    motors=["joint_0"],
                    fps=25,
                    mode="image",
                    dataset_config=DatasetConfig(use_videos=False, streaming_encoding=False),
                    root=tmp,
                )

    def test_hdf5_mask_is_written_as_kai0_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            output_root = Path(tmp) / "lerobot"
            recorder = EpisodeRecorder(
                str(raw_root),
                {
                    "task_name": "stack_bowls",
                    "env_config": "arx_x5",
                    "layout_id": 3,
                    "base_checkpoint": "test",
                },
                frequency=25,
            )
            image = np.zeros((480, 640, 3), dtype=np.uint8)
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
            hdf5_path = recorder.finalize(accepted=True, success=True, reason="test")

            config = DatasetConfig(
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=0,
                streaming_encoding=True,
                video_backend="pyav",
                vcodec="h264",
                encoder_threads=1,
            )
            motors = [f"joint_{index}" for index in range(14)]
            dataset = create_empty_dataset(
                repo_id="mask_test",
                robot_type="arx_x5",
                motors=motors,
                fps=25,
                mode="video",
                dataset_config=config,
                root=output_root,
            )
            try:
                convert_one(
                    hdf5_path,
                    dataset,
                    data_type="RoboDojo",
                    data_version="v1.0",
                    current_dims=[7, 7],
                    target_dims=[7, 7],
                )
            finally:
                finalize_dataset(dataset)

            info_path = output_root / "mask_test" / "meta" / "info.json"
            self.assertTrue(info_path.is_file())
            parquet_files = list((output_root / "mask_test" / "data").rglob("*.parquet"))
            self.assertEqual(len(parquet_files), 1)
            table = pq.read_table(
                parquet_files[0],
                columns=["action", "complementary_info.is_intervention"],
            )
            self.assertEqual(table["complementary_info.is_intervention"].to_pylist(), [0.0, 1.0])
            self.assertEqual(len(table["action"].to_pylist()[0]), 14)
            video_files = list((output_root / "mask_test" / "videos").rglob("*.mp4"))
            self.assertEqual(len(video_files), 3)


if __name__ == "__main__":
    unittest.main()
