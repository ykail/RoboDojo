from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from src.eval_client.x5_raw_bundle import RawBundle, RawBundleSegment
from src.eval_client.x5_raw_replay import (
    X5RawReplayError,
    actions_from_bundle,
    actions_from_segment,
    raw_bundle_id,
    raw_segment_id,
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


def _multi_bundle(path: Path) -> RawBundle:
    manifest = {
        "format_version": 2,
        "episode_index": 4,
        "collection_identity_sha256": "sha256:" + "b" * 64,
        "segment_count": 2,
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
    timestamps = np.asarray(
        [
            1_010_000_000,
            1_050_000_000,
            1_090_000_000,
            2_010_000_000,
            2_050_000_000,
            2_090_000_000,
        ],
        dtype=np.int64,
    )
    left_q = np.asarray(
        [[value] * 6 for value in (0.1, 0.5, 0.9, 10.1, 10.5, 10.9)],
        dtype=np.float64,
    )
    source = {
        "sample_monotonic_ns": timestamps,
        "segment_start_ns": np.asarray(
            [1_000_000_000, 2_000_000_000], dtype=np.int64
        ),
        "segment_end_ns": np.asarray(
            [1_100_000_000, 2_100_000_000], dtype=np.int64
        ),
        "segment_index": np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        "segment_anchor_timestamp_ns": np.asarray(
            [1_010_000_000, 2_010_000_000], dtype=np.int64
        ),
        "left_q_rad": left_q,
        "right_q_rad": -left_q,
        "left_gripper_open_fraction": np.asarray([0.1, 0.5, 0.9] * 2),
        "right_gripper_open_fraction": np.asarray([0.9, 0.5, 0.1] * 2),
        "left_segment_anchor_q_rad": np.asarray([[0.1] * 6, [10.1] * 6]),
        "right_segment_anchor_q_rad": np.asarray([[-0.1] * 6, [-10.1] * 6]),
        "left_segment_anchor_gripper_open_fraction": np.asarray([0.1, 0.1]),
        "right_segment_anchor_gripper_open_fraction": np.asarray([0.9, 0.9]),
    }
    segments = []
    for index in range(2):
        start_ns = (index + 1) * 1_000_000_000
        source_value = 0.1 + index * 10.0
        sim_left = np.arange(1, 7, dtype=np.float64) + index * 10.0
        segments.append(
            RawBundleSegment(
                index=index,
                manifest={
                    "segment_index": index,
                    "source_slice": {
                        "sample_start_index": index * 3,
                        "sample_stop_index": index * 3 + 3,
                        "sample_count": 3,
                        "segment_start_ns": start_ns,
                        "segment_end_ns": start_ns + 100_000_000,
                        "segment_anchor_timestamp_ns": start_ns + 10_000_000,
                    },
                    "snapshot": {
                        "replay_saved_layout": {"layout": 7, "segment": index},
                        "replay_manifest": {
                            "profile": "robodojo_rigid_articulation_v1",
                            "segment": index,
                        },
                    },
                    "sim_anchor": {
                        "left": {
                            "q_rad": sim_left.tolist(),
                            "gripper_open_fraction": 0.1,
                        },
                        "right": {
                            "q_rad": (-sim_left).tolist(),
                            "gripper_open_fraction": 0.9,
                        },
                    },
                    "source_anchor": {
                        "left": {
                            "q_rad": [source_value] * 6,
                            "gripper_open_fraction": 0.1,
                        },
                        "right": {
                            "q_rad": [-source_value] * 6,
                            "gripper_open_fraction": 0.9,
                        },
                    },
                    "metadata": {"takeover_ordinal": index},
                },
                takeover_state={"frame.index": np.asarray(4 + index * 40)},
                terminal_state={"frame.index": np.asarray(9 + index * 40)},
            )
        )
    return RawBundle(
        path=path,
        manifest=manifest,
        source=source,
        segments=tuple(segments),
    )


class RawReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dataset_environment = {
            name: os.environ.get(name)
            for name in ("ROBODOJO_LEROBOT_ROOT", "ROBODOJO_LEROBOT_REPO_ID")
        }
        os.environ.pop("ROBODOJO_LEROBOT_ROOT", None)
        os.environ.pop("ROBODOJO_LEROBOT_REPO_ID", None)

    def tearDown(self) -> None:
        for name, value in self._dataset_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    @staticmethod
    def _fake_env(events: list[str]):
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
                return {"state": {}, "vision": {}}

            def take_action(self, action, *, interpolate=True):
                del action
                events.append(f"action:{interpolate}")

        return Env()

    @staticmethod
    def _recorder_factory(
        dataset_root: Path,
        dataset_id: str,
        events: list[str],
        *,
        fail_segment: int | None = None,
    ):
        class Recorder:
            def __init__(self, task_env):
                self.task_env = task_env
                self.finished = False

            def append(self, **kwargs):
                del kwargs

            def finalize(self, *, accepted, **kwargs):
                del kwargs
                if self.finished:
                    return None
                segment_index = self.task_env.restore_lineage["raw_segment_index"]
                if accepted and segment_index == fail_segment:
                    self.finished = True
                    raise RuntimeError("injected segment failure")
                self.finished = True
                if not accepted:
                    return None
                sidecars = dataset_root / dataset_id / "meta/robodojo/episodes"
                sidecars.mkdir(parents=True, exist_ok=True)
                index = len(list(sidecars.glob("episode_*.json")))
                path = sidecars / f"episode_{index:07d}.json"
                path.write_text(
                    json.dumps(
                        {
                            "robodojo_recovery_source": dict(
                                self.task_env.restore_lineage
                            )
                        }
                    ),
                    encoding="utf-8",
                )
                events.append(f"commit:{segment_index}")
                return str(dataset_root / dataset_id)

        return Recorder

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

    def test_sub_100ms_single_sample_segment_replays_as_a_short_hold(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = _bundle(Path(temporary))
            source = {
                key: np.asarray(value)[:1].copy()
                for key, value in base.source.items()
            }
            source["segment_start_ns"] = np.asarray([1_000_000_000], dtype=np.int64)
            source["segment_end_ns"] = np.asarray([1_080_000_000], dtype=np.int64)
            short = RawBundle(
                path=base.path,
                manifest=base.manifest,
                source=source,
                takeover_state=base.takeover_state,
                terminal_state=base.terminal_state,
            )

            actions = actions_from_bundle(short)

        self.assertEqual(len(actions), 2)
        np.testing.assert_allclose(
            actions[0]["left_arm_joint_state"],
            [1, 2, 3, 4, 5, 6],
        )

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

    def test_v2_segments_have_independent_mapping_snapshots_and_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = _multi_bundle(Path(temporary))
            with self.assertRaisesRegex(X5RawReplayError, "multiple intervention"):
                actions_from_bundle(bundle)
            first = actions_from_segment(bundle, 0)
            second = actions_from_segment(bundle, 1)

        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 3)
        np.testing.assert_allclose(
            first[0]["left_arm_joint_state"], [1, 2, 3, 4, 5, 6]
        )
        np.testing.assert_allclose(
            first[1]["left_arm_joint_state"],
            np.asarray([1, 2, 3, 4, 5, 6]) + 0.3,
        )
        np.testing.assert_allclose(
            second[0]["left_arm_joint_state"], [11, 12, 13, 14, 15, 16]
        )
        np.testing.assert_allclose(
            second[2]["right_arm_joint_state"],
            np.asarray([-11, -12, -13, -14, -15, -16]) - 0.7,
        )
        parent_id = raw_bundle_id(bundle)
        self.assertEqual(raw_segment_id(bundle, 0), f"{parent_id}:segment_0000")
        self.assertEqual(raw_segment_id(bundle, 1), f"{parent_id}:segment_0001")
        legacy = _bundle(Path("legacy"))
        self.assertEqual(raw_segment_id(legacy, 0), raw_bundle_id(legacy))

    def test_v2_batch_commits_each_segment_as_an_independent_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "raw/pending/episode_0000004"
            raw_path.mkdir(parents=True)
            bundle = _multi_bundle(raw_path)
            dataset_root = root / "lerobot"
            dataset_id = "dataset"
            events: list[str] = []
            env = self._fake_env(events)

            summary = run_x5_raw_replay_collection(
                env,
                root / "raw",
                dataset_root,
                dataset_id,
                _bundle_loader=lambda _root, verify=True: [bundle],
                _recorder_factory=self._recorder_factory(
                    dataset_root, dataset_id, events
                ),
                _restore_fn=lambda task_env, replay, **kwargs: events.append(
                    f"restore:{int(replay.state['frame.index'])}"
                ),
            )

            self.assertEqual(summary.total, 2)
            self.assertEqual(summary.newly_committed, 2)
            self.assertEqual(summary.already_committed, 0)
            self.assertEqual(events.count("reset:7"), 2)
            self.assertIn("restore:4", events)
            self.assertIn("restore:44", events)
            sidecars = sorted(
                (dataset_root / dataset_id / "meta/robodojo/episodes").glob(
                    "episode_*.json"
                )
            )
            self.assertEqual(len(sidecars), 2)
            recovery = [
                json.loads(path.read_text())["robodojo_recovery_source"]
                for path in sidecars
            ]
            self.assertEqual(
                [item["raw_bundle_id"] for item in recovery],
                [raw_segment_id(bundle, 0), raw_segment_id(bundle, 1)],
            )
            self.assertEqual(
                [item["raw_segment_index"] for item in recovery], [0, 1]
            )
            self.assertTrue((raw_path / "REPLAYED.segment_0000.json").is_file())
            self.assertTrue((raw_path / "REPLAYED.segment_0001.json").is_file())
            parent_marker = json.loads((raw_path / "REPLAYED.json").read_text())
            self.assertEqual(
                parent_marker["raw_segment_ids"],
                [raw_segment_id(bundle, 0), raw_segment_id(bundle, 1)],
            )

    def test_v2_partial_failure_resumes_without_replaying_completed_segment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "raw/pending/episode_0000004"
            raw_path.mkdir(parents=True)
            bundle = _multi_bundle(raw_path)
            dataset_root = root / "lerobot"
            dataset_id = "dataset"
            events: list[str] = []
            env = self._fake_env(events)

            with self.assertRaisesRegex(X5RawReplayError, "1 raw segment"):
                run_x5_raw_replay_collection(
                    env,
                    root / "raw",
                    dataset_root,
                    dataset_id,
                    _bundle_loader=lambda _root, verify=True: [bundle],
                    _recorder_factory=self._recorder_factory(
                        dataset_root, dataset_id, events, fail_segment=1
                    ),
                    _restore_fn=lambda task_env, replay, **kwargs: events.append(
                        f"restore:{int(replay.state['frame.index'])}"
                    ),
                )

            self.assertTrue((raw_path / "REPLAYED.segment_0000.json").is_file())
            self.assertTrue((raw_path / "REPLAY_FAILED.segment_0001.json").is_file())
            self.assertFalse((raw_path / "REPLAYED.segment_0001.json").exists())
            self.assertFalse((raw_path / "REPLAYED.json").exists())
            sidecar_directory = dataset_root / dataset_id / "meta/robodojo/episodes"
            self.assertEqual(len(list(sidecar_directory.glob("episode_*.json"))), 1)

            events.clear()
            resumed = run_x5_raw_replay_collection(
                env,
                root / "raw",
                dataset_root,
                dataset_id,
                _bundle_loader=lambda _root, verify=True: [bundle],
                _recorder_factory=self._recorder_factory(
                    dataset_root, dataset_id, events
                ),
                _restore_fn=lambda task_env, replay, **kwargs: events.append(
                    f"restore:{int(replay.state['frame.index'])}"
                ),
            )

            self.assertEqual(resumed.total, 2)
            self.assertEqual(resumed.already_committed, 1)
            self.assertEqual(resumed.newly_committed, 1)
            self.assertNotIn("restore:4", events)
            self.assertEqual(events.count("restore:44"), 1)
            self.assertFalse((raw_path / "REPLAY_FAILED.segment_0001.json").exists())
            self.assertTrue((raw_path / "REPLAYED.segment_0001.json").is_file())
            self.assertTrue((raw_path / "REPLAYED.json").is_file())
            self.assertEqual(len(list(sidecar_directory.glob("episode_*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
