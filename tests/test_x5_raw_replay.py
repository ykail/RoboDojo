from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from src.eval_client.x5_raw_bundle import RawBundle
from src.eval_client.x5_raw_replay import (
    actions_from_bundle,
    raw_bundle_id,
    run_x5_raw_replay_collection,
)


def _bundle(path: Path) -> RawBundle:
    manifest = {
        "episode_index": 0,
        "collection_identity_sha256": "sha256:" + "a" * 64,
        "sim_anchor": {
            "left": {"q_rad": [1, 2, 3, 4, 5, 6], "gripper_open_fraction": 0.2},
            "right": {"q_rad": [-1, -2, -3, -4, -5, -6], "gripper_open_fraction": 0.8},
        },
        "source_anchor": {
            "left": {"q_rad": [0, 0, 0, 0, 0, 0], "gripper_open_fraction": 0.2},
            "right": {"q_rad": [0, 0, 0, 0, 0, 0], "gripper_open_fraction": 0.8},
        },
        "snapshot": {
            "replay_saved_layout": {"layout": 7},
            "replay_manifest": {"profile": "robodojo_rigid_articulation_v1"},
        },
        "metadata": {
            "task_name": "fill_pen_holder",
            "layout_id": 7,
            "eval_seed": 3,
            "operator_accepted": True,
            "policy_provenance": {
                "checkpoint_id": "RoboDojo-sim-arx_x5-joint-0/59999",
                "checkpoint_digest": "sha256:" + "7" * 64,
                "code_revision": "e" * 40,
                "dirty": False,
            },
        },
    }
    times = np.asarray([1_010_000_000, 1_050_000_000, 1_090_000_000], dtype=np.int64)
    left_q = np.asarray([[0.1] * 6, [0.5] * 6, [0.9] * 6])
    right_q = -left_q
    source = {
        "sample_monotonic_ns": times,
        "segment_start_ns": np.asarray([1_000_000_000], dtype=np.int64),
        "segment_end_ns": np.asarray([1_100_000_000], dtype=np.int64),
        "segment_index": np.asarray([0, 0, 0], dtype=np.int32),
        "segment_anchor_timestamp_ns": np.asarray([1_010_000_000], dtype=np.int64),
        "left_q_rad": left_q,
        "right_q_rad": right_q,
        "left_gripper_open_fraction": np.asarray([0.3, 0.5, 0.7]),
        "right_gripper_open_fraction": np.asarray([0.7, 0.5, 0.3]),
        "left_segment_anchor_q_rad": np.asarray([[0.0] * 6]),
        "right_segment_anchor_q_rad": np.asarray([[0.0] * 6]),
        "left_segment_anchor_gripper_open_fraction": np.asarray([0.2]),
        "right_segment_anchor_gripper_open_fraction": np.asarray([0.8]),
    }
    return RawBundle(
        path=path,
        manifest=manifest,
        source=source,
        takeover_state={"frame.index": np.asarray(4)},
        terminal_state={"frame.index": np.asarray(9)},
    )


class RawReplayTest(unittest.TestCase):
    def test_mapping_uses_takeover_anchor_and_fixed_25hz_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = _bundle(Path(temporary))
            actions = actions_from_bundle(bundle)

        self.assertEqual(len(actions), 3)
        np.testing.assert_allclose(
            actions[0]["left_arm_joint_state"], [1, 2, 3, 4, 5, 6]
        )
        np.testing.assert_allclose(
            actions[1]["left_arm_joint_state"],
            np.asarray([1, 2, 3, 4, 5, 6]) + 0.4,
        )
        np.testing.assert_allclose(
            actions[2]["right_arm_joint_state"],
            np.asarray([-1, -2, -3, -4, -5, -6]) - 0.8,
        )
        self.assertAlmostEqual(actions[0]["left_ee_joint_state"][0], 0.2)
        self.assertAlmostEqual(actions[1]["left_ee_joint_state"][0], 0.45)

    def test_batch_commits_each_frame_after_real_step_and_resumes_exactly_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "raw/pending/episode_0000000"
            raw_path.mkdir(parents=True)
            bundle = _bundle(raw_path)
            events: list[str] = []

            class Env:
                def __init__(self):
                    self.obs_manager = SimpleNamespace(collect_freq=25)
                    self.reward_manager = SimpleNamespace(
                        get_reward=lambda *, final_check: [0.0]
                    )
                    self.control_mode = "x5_raw_replay_25hz"

                def reset(self, seed):
                    events.append(f"reset:{seed[0]}")

                def get_obs(self):
                    events.append("obs")
                    return {"state": {}, "vision": {}}

                def take_action(self, action, *, interpolate=True):
                    del action
                    events.append(f"action:{interpolate}")

            env = Env()
            dataset_root = root / "lerobot"
            dataset_id = "dataset"

            class Recorder:
                def __init__(self, task_env):
                    self.task_env = task_env
                    self.frames = 0

                def append(self, **kwargs):
                    self.frames += 1
                    events.append(f"append:{kwargs['control']['timestamp']:.2f}")

                def finalize(self, *, accepted, **kwargs):
                    del kwargs
                    events.append(f"finalize:{accepted}")
                    if accepted:
                        sidecars = dataset_root / dataset_id / "meta/robodojo/episodes"
                        sidecars.mkdir(parents=True, exist_ok=True)
                        (sidecars / "episode_0000000.json").write_text(
                            json.dumps(
                                {
                                    "robodojo_recovery_source": dict(
                                        self.task_env.restore_lineage
                                    )
                                }
                            ),
                            encoding="utf-8",
                        )
                        return str(dataset_root / dataset_id)
                    return None

            def restore(task_env, replay, **kwargs):
                del task_env, kwargs
                events.append(f"restore:{int(replay.state['frame.index'])}")

            loader = lambda _root, verify=True: [bundle]
            first = run_x5_raw_replay_collection(
                env,
                root / "raw",
                dataset_root,
                dataset_id,
                _bundle_loader=loader,
                _recorder_factory=Recorder,
                _restore_fn=restore,
            )
            self.assertEqual(first.newly_committed, 1)
            self.assertEqual(events.count("action:False"), 3)
            self.assertLess(events.index("append:0.00"), events.index("action:False"))
            self.assertTrue((raw_path / "REPLAYED.json").is_file())

            before = list(events)
            second = run_x5_raw_replay_collection(
                env,
                root / "raw",
                dataset_root,
                dataset_id,
                _bundle_loader=loader,
                _recorder_factory=Recorder,
                _restore_fn=restore,
            )
            self.assertEqual(second.already_committed, 1)
            self.assertEqual(second.newly_committed, 0)
            self.assertEqual(events, before)
            self.assertIn(raw_bundle_id(bundle), (raw_path / "REPLAYED.json").read_text())


if __name__ == "__main__":
    unittest.main()
