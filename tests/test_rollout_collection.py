import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.eval_client.rollout_collection import (
    apply_rollout_collection_config,
    build_collection_manifest,
    committed_plan_indices,
    group_entries_by_seed,
    parse_layout_ids,
    parse_layout_plan,
    remaining_entries,
    validate_collection_manifest,
)


class RolloutCollectionPlanTest(unittest.TestCase):
    def test_collection_config_is_applied_to_runtime_mapping(self):
        entries = parse_layout_plan("0:0-4")
        manifest = build_collection_manifest(
            task="make_toast", checkpoint_id="official/59999", entries=entries
        )
        runtime_config = {}
        count = apply_rollout_collection_config(
            runtime_config,
            {
                "layout_ids": [1, 3],
                # A resumed segment keeps global plan indices and may retain
                # mappings for layouts that were already committed.
                "plan_index_by_layout": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
                "manifest": manifest,
                "collection_id": manifest["collection_id"],
                "plan_hash": manifest["plan_hash"],
            },
        )

        self.assertEqual(count, 2)
        self.assertEqual(runtime_config["eval_num"], 2)
        self.assertEqual(runtime_config["selected_layout_ids"], [1, 3])
        self.assertEqual(runtime_config["collection_plan_index_by_layout"][3], 3)
        self.assertIs(runtime_config["collection_manifest"], manifest)
        self.assertEqual(runtime_config["collection_id"], manifest["collection_id"])
        self.assertEqual(runtime_config["collection_plan_hash"], manifest["plan_hash"])

    def test_collection_config_rejects_selected_layout_without_plan_index(self):
        with self.assertRaisesRegex(ValueError, "missing selected layout ids"):
            apply_rollout_collection_config(
                {},
                {
                    "layout_ids": [0, 1],
                    "plan_index_by_layout": {0: 7},
                    "manifest": {},
                    "collection_id": "rollout-test",
                    "plan_hash": "sha256:" + "0" * 64,
                },
            )

    def test_make_toast_plan_expands_to_ordered_70_plus_30(self):
        entries = parse_layout_plan("0:0-69,1:0-29")

        self.assertEqual(len(entries), 100)
        self.assertEqual((entries[0].eval_seed, entries[0].layout_id), (0, 0))
        self.assertEqual((entries[69].eval_seed, entries[69].layout_id), (0, 69))
        self.assertEqual((entries[70].eval_seed, entries[70].layout_id), (1, 0))
        self.assertEqual(entries[-1].plan_index, 99)
        groups = group_entries_by_seed(entries)
        self.assertEqual([(seed, len(group)) for seed, group in groups], [(0, 70), (1, 30)])

        repeated_seed_runs = group_entries_by_seed(parse_layout_plan("0:1,1:2,0:3"))
        self.assertEqual(
            [
                (seed, [entry.plan_index for entry in group])
                for seed, group in repeated_seed_runs
            ],
            [(0, [0]), (1, [1]), (0, [2])],
        )

    def test_duplicate_and_malformed_plans_are_rejected(self):
        for spec in ("", "0", "0:", "0:3-1", "0:1+1"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_layout_plan(spec)
        self.assertEqual(parse_layout_ids("0-2,5+7"), [0, 1, 2, 5, 7])

    def test_manifest_hash_is_stable_and_tamper_evident(self):
        entries = parse_layout_plan("0:0-2,1:4")
        first = build_collection_manifest(
            task="make_toast", checkpoint_id="official/59999", entries=entries
        )
        second = build_collection_manifest(
            task="make_toast", checkpoint_id="official/59999", entries=entries
        )
        self.assertEqual(first, second)
        self.assertEqual(first["policy_seed"], {"mode": "eval_seed"})
        validate_collection_manifest(first)
        first["entries"][0]["layout_id"] = 99
        with self.assertRaisesRegex(ValueError, "plan hash"):
            validate_collection_manifest(first)

        fixed = build_collection_manifest(
            task="make_toast",
            checkpoint_id="official/59999",
            entries=entries,
            policy_seed=123,
        )
        self.assertEqual(fixed["policy_seed"], {"mode": "fixed", "value": 123})
        self.assertNotEqual(fixed["collection_id"], second["collection_id"])

    def test_durable_episode_metadata_is_progress_authority(self):
        entries = parse_layout_plan("0:3+7,1:2")
        manifest = build_collection_manifest(
            task="make_toast", checkpoint_id="official/59999", entries=entries
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info_path = root / "meta" / "info.json"
            info_path.parent.mkdir(parents=True)
            info_path.write_text(
                json.dumps({"total_episodes": 1, "total_frames": 17}),
                encoding="utf-8",
            )
            collection_path = (
                root
                / "meta"
                / "robodojo"
                / "collections"
                / f"{manifest['collection_id']}.json"
            )
            collection_path.parent.mkdir(parents=True)
            collection_path.write_text(
                json.dumps({"format_version": 1, "plan": manifest, "policy_provenance": {}}),
                encoding="utf-8",
            )
            episodes = root / "meta" / "robodojo" / "episodes"
            episodes.mkdir()
            replay_paths = {
                "state": root / "data/robodojo_replay/chunk-000/episode_0000000.npz",
                "layout": root / "meta/robodojo/replay/layouts/episode_0000000.json",
                "schema": root / "meta/robodojo/replay/schema.json",
            }
            replay = {"complete": True, "snapshot_count": 17}
            for prefix, path in replay_paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                if prefix == "state":
                    np.savez_compressed(
                        path,
                        format_version=np.asarray(1, dtype=np.int64),
                        frame_count=np.asarray(17, dtype=np.int64),
                        **{
                            "frame__frame.index": np.arange(17, dtype=np.int64),
                            "terminal__frame.index": np.asarray(17, dtype=np.int64),
                            "frame__frame.timestamp_s": np.arange(17) / 25,
                            "terminal__frame.timestamp_s": np.asarray(17 / 25),
                        },
                    )
                else:
                    path.write_bytes(f"{prefix}-contents".encode())
                replay[f"{prefix}_path"] = path.relative_to(root).as_posix()
                digest_key = (
                    "layout_file_sha256" if prefix == "layout" else f"{prefix}_sha256"
                )
                replay[digest_key] = "sha256:" + hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            episodes.joinpath("episode_0000000.json").write_text(
                json.dumps(
                    {
                        "episode_index": 0,
                        "robodojo_collection_id": manifest["collection_id"],
                        "robodojo_collection_plan_hash": manifest["plan_hash"],
                        "robodojo_collection_plan_index": 1,
                        "robodojo_eval_seed": 0,
                        "robodojo_policy_seed": 0,
                        "robodojo_layout_id": 7,
                        "robodojo_frame_count": 17,
                        "robodojo_replay": replay,
                    }
                ),
                encoding="utf-8",
            )

            completed = committed_plan_indices(root, manifest)

            self.assertEqual(completed, {1})
            self.assertEqual(
                [entry.plan_index for entry in remaining_entries(manifest, completed)],
                [0, 2],
            )

    def test_resume_rejects_non_atomic_lerobot_episode_prefix(self):
        entries = parse_layout_plan("0:0")
        manifest = build_collection_manifest(
            task="make_toast", checkpoint_id="official/59999", entries=entries
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info_path = root / "meta" / "info.json"
            info_path.parent.mkdir(parents=True)
            info_path.write_text(
                json.dumps({"total_episodes": 1, "total_frames": 4}),
                encoding="utf-8",
            )
            collection_path = (
                root
                / "meta"
                / "robodojo"
                / "collections"
                / f"{manifest['collection_id']}.json"
            )
            collection_path.parent.mkdir(parents=True)
            collection_path.write_text(
                json.dumps({"format_version": 1, "plan": manifest, "policy_provenance": {}}),
                encoding="utf-8",
            )
            (root / "meta" / "robodojo" / "episodes").mkdir()

            with self.assertRaisesRegex(ValueError, "not an atomic prefix"):
                committed_plan_indices(root, manifest)


if __name__ == "__main__":
    unittest.main()
