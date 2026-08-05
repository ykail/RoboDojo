import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.eval_client.replay_bundle import load_replay_frame


class ReplayBundleTest(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        metadata_dir = root / "meta/robodojo/episodes"
        layout_dir = root / "meta/robodojo/replay/layouts"
        state_dir = root / "data/robodojo_replay/chunk-000"
        metadata_dir.mkdir(parents=True)
        layout_dir.mkdir(parents=True)
        state_dir.mkdir(parents=True)

        state_relative = "data/robodojo_replay/chunk-000/episode_0000002.npz"
        layout_relative = "meta/robodojo/replay/layouts/episode_0000002.json"
        np.savez_compressed(
            root / state_relative,
            format_version=np.asarray(1, dtype=np.int64),
            frame_count=np.asarray(3, dtype=np.int64),
            **{
                "frame__frame.index": np.arange(3, dtype=np.int64),
                "frame__frame.timestamp_s": np.asarray([0.0, 0.04, 0.08]),
                "frame__robot.000.joint_pos": np.asarray([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]], dtype=np.float32),
                "terminal__frame.index": np.asarray(3, dtype=np.int64),
            },
        )
        (root / layout_relative).write_text(
            json.dumps(
                {
                    "snapshot_manifest": {"format_version": 1, "fps": 25},
                    "saved_layout": {"Rigid": {"bread": []}},
                }
            ),
            encoding="utf-8",
        )
        (metadata_dir / "episode_0000002.json").write_text(
            json.dumps(
                {
                    "episode_index": 2,
                    "robodojo_task": "make_toast",
                    "robodojo_env_config": "arx_x5",
                    "robodojo_eval_seed": 1,
                    "robodojo_layout_id": 7,
                    "robodojo_replay": {
                        "complete": True,
                        "layout_path": layout_relative,
                        "state_path": state_relative,
                    },
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_selects_explicit_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = load_replay_frame(self._dataset(Path(directory)), 2, frame_index=1)
        self.assertEqual(replay.task_name, "make_toast")
        self.assertEqual(replay.eval_seed, 1)
        self.assertEqual(replay.layout_id, 7)
        self.assertEqual(replay.frame_index, 1)
        self.assertAlmostEqual(replay.timestamp_s, 0.04)
        np.testing.assert_allclose(replay.state["robot.000.joint_pos"], [1.0, 1.1])

    def test_selects_nearest_video_frame_by_time(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = load_replay_frame(self._dataset(Path(directory)), 2, time_s=0.071)
        self.assertEqual(replay.frame_index, 2)

    def test_rejects_terminal_as_video_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(IndexError, "outside episode range"):
                load_replay_frame(self._dataset(Path(directory)), 2, frame_index=3)


if __name__ == "__main__":
    unittest.main()
