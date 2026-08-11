from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.eval_client.x5_raw_bundle import (
    RawBundleError,
    RawBundleStore,
    load_pending_bundles,
    resample_segment_25hz,
)


class _Snapshotter:
    def metadata(self):
        return {
            "record_sim_state": True,
            "replay_layout_sha256": "sha256:" + "a" * 64,
            "replay_manifest": {"fps": 25, "profile": "test"},
            "replay_saved_layout": {"layout": 3},
        }


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fragment(path: Path, *, offset: float = 0.0) -> None:
    np.savez(
        path,
        sample_monotonic_ns=np.asarray(
            [1_000_000_000, 1_050_000_000, 1_100_000_000], dtype=np.int64
        ),
        q_rad=np.asarray(
            [[offset, 0.0], [offset + 1.0, 2.0], [offset + 2.0, 4.0]],
            dtype=np.float32,
        ),
        gripper=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        sequence=np.asarray([10, 11, 12], dtype=np.int64),
    )


def _write_multi_fragment(path: Path) -> None:
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
    left = np.asarray(
        [[value] * 6 for value in (0.1, 0.5, 0.9, 10.1, 10.5, 10.9)],
        dtype=np.float64,
    )
    np.savez(
        path,
        sample_monotonic_ns=timestamps,
        segment_index=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        left_q_rad=left,
        right_q_rad=-left,
        left_gripper_open_fraction=np.asarray([0.1, 0.5, 0.9] * 2),
        right_gripper_open_fraction=np.asarray([0.9, 0.5, 0.1] * 2),
        segment_start_ns=np.asarray([1_000_000_000, 2_000_000_000], dtype=np.int64),
        segment_end_ns=np.asarray([1_100_000_000, 2_100_000_000], dtype=np.int64),
        segment_anchor_timestamp_ns=np.asarray(
            [1_010_000_000, 2_010_000_000], dtype=np.int64
        ),
        left_segment_anchor_q_rad=np.asarray([[0.1] * 6, [10.1] * 6]),
        right_segment_anchor_q_rad=np.asarray([[-0.1] * 6, [-10.1] * 6]),
        left_segment_anchor_gripper_open_fraction=np.asarray([0.1, 0.1]),
        right_segment_anchor_gripper_open_fraction=np.asarray([0.9, 0.9]),
        format_version=np.asarray(1, dtype=np.int32),
        sampling_frequency_hz=np.asarray(100.0),
    )


class _SegmentSnapshotter:
    def __init__(self, index: int):
        self.index = index

    def metadata(self):
        return {
            "record_sim_state": True,
            "replay_saved_layout": {"layout": 3},
            "replay_manifest": {"profile": "test", "segment": self.index},
        }


def _segment(index: int) -> dict:
    return {
        "snapshotter": _SegmentSnapshotter(index),
        "takeover_state": _state(10 + index * 10),
        "terminal_state": _state(19 + index * 10),
        "sim_anchor": {
            "left": {"q_rad": [float(index)] * 6, "gripper_open_fraction": 0.1},
            "right": {"q_rad": [float(-index)] * 6, "gripper_open_fraction": 0.9},
        },
        "source_anchor": {
            "left": {
                "q_rad": [0.1 + 10.0 * index] * 6,
                "gripper_open_fraction": 0.1,
            },
            "right": {
                "q_rad": [-0.1 - 10.0 * index] * 6,
                "gripper_open_fraction": 0.9,
            },
        },
        "metadata": {"takeover_ordinal": index},
    }


def _state(frame: int) -> dict[str, np.ndarray]:
    return {
        "frame.index": np.asarray(frame, dtype=np.int64),
        "robot.000.joint_pos": np.asarray([frame, frame + 1], dtype=np.float32),
    }


class RawBundleStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "raw"
        self.fragment = Path(self.temp.name) / "fragment.npz"
        _write_fragment(self.fragment)
        self.identity = {
            "task": "fill_pen_holder",
            "checkpoint_id": "RoboDojo-sim-arx_x5-joint-0/59999",
            "checkpoint_sha256": "sha256:" + "7" * 64,
            "kai0_commit": "ecc1a7451c3156b1e5f7533851dbb0222896206f",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _store(self, target: int = 2) -> RawBundleStore:
        return RawBundleStore(self.root, target, self.identity)

    def _commit(self, store: RawBundleStore, *, fragment=None) -> Path:
        return store.commit(
            fragment or {"path": self.fragment, "sha256": _sha256(self.fragment)},
            _Snapshotter(),
            _state(4),
            _state(9),
            {"left": [[0.1, 0.2], 0.5], "right": [[0.3, 0.4], 0.6]},
            {"sample_monotonic_ns": 1_000_000_000},
            {"success": True, "finish_reason": "operator_accept_next"},
        )

    def test_commit_is_loadable_verified_and_resumable(self) -> None:
        store = self._store()
        directory = self._commit(store)

        self.assertEqual(directory.name, "episode_0000000")
        self.assertEqual(store.completed_count, 1)
        self.assertEqual(store.remaining_count, 1)
        self.assertFalse(store.target_reached)
        self.assertTrue((directory / "COMMITTED.json").is_file())

        resumed = self._store()
        self.assertEqual(resumed.completed_count, 1)
        bundles = load_pending_bundles(self.root)
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].episode_index, 0)
        np.testing.assert_array_equal(
            bundles[0].source["q_rad"],
            np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], dtype=np.float32),
        )
        self.assertEqual(int(bundles[0].takeover_state["frame.index"]), 4)
        self.assertEqual(int(bundles[0].terminal_state["frame.index"]), 9)
        self.assertEqual(
            bundles[0].manifest["metadata"]["finish_reason"],
            "operator_accept_next",
        )

    def test_collection_identity_and_target_are_locked(self) -> None:
        self._store()
        with self.assertRaisesRegex(RawBundleError, "identity/target"):
            RawBundleStore(self.root, 3, self.identity)
        with self.assertRaisesRegex(RawBundleError, "identity/target"):
            RawBundleStore(self.root, 2, {**self.identity, "task": "make_toast"})

    def test_partial_directories_do_not_count_or_load(self) -> None:
        store = self._store()
        partial = self.root / "pending" / ".episode_0000000.crashed.partial"
        partial.mkdir()
        (partial / "unflushed").write_text("not committed", encoding="utf-8")

        self.assertEqual(store.completed_count, 0)
        self.assertEqual(load_pending_bundles(self.root), [])
        committed = self._commit(store)
        self.assertEqual(committed.name, "episode_0000000")

    def test_target_reached_refuses_extra_episode(self) -> None:
        store = self._store(target=1)
        self._commit(store)
        self.assertTrue(store.target_reached)
        with self.assertRaisesRegex(RawBundleError, "reached target"):
            self._commit(store)
        self.assertEqual(store.completed_count, 1)

    def test_bad_fragment_digest_never_becomes_committed(self) -> None:
        store = self._store()
        with self.assertRaisesRegex(RawBundleError, "digest mismatch"):
            self._commit(
                store,
                fragment={"path": self.fragment, "sha256": "sha256:" + "0" * 64},
            )
        self.assertEqual(store.completed_count, 0)
        self.assertEqual(load_pending_bundles(self.root), [])

    def test_payload_corruption_is_detected(self) -> None:
        store = self._store()
        directory = self._commit(store)
        source = directory / "source.npz"
        source.write_bytes(source.read_bytes() + b"corruption")
        with self.assertRaisesRegex(RawBundleError, "byte-size mismatch"):
            load_pending_bundles(self.root)

    def test_discard_cleans_partials_and_only_deletes_fragment_explicitly(self) -> None:
        store = self._store()
        partial = self.root / "pending" / ".episode_0000000.crashed.partial"
        partial.mkdir()

        store.discard(self.fragment)
        self.assertFalse(partial.exists())
        self.assertTrue(self.fragment.exists())

        store.discard(self.fragment, delete_fragment=True)
        self.assertFalse(self.fragment.exists())
        self.assertEqual(store.completed_count, 0)

    def test_commit_manifest_and_payload_digests_match(self) -> None:
        directory = self._commit(self._store())
        manifest_bytes = (directory / "manifest.json").read_bytes()
        marker = json.loads((directory / "COMMITTED.json").read_text())
        self.assertEqual(
            marker["manifest_sha256"],
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
        )
        manifest = json.loads(manifest_bytes)
        for descriptor in manifest["files"].values():
            payload = directory / descriptor["path"]
            self.assertEqual(descriptor["bytes"], payload.stat().st_size)
            self.assertEqual(descriptor["sha256"], _sha256(payload))

    def test_v2_multi_segment_bundle_is_atomic_and_slice_addressable(self) -> None:
        multi = Path(self.temp.name) / "multi.npz"
        _write_multi_fragment(multi)
        store = self._store()
        directory = store.commit_segments(
            {
                "path": multi,
                "sha256": _sha256(multi),
                "segment_count": 2,
                "sample_count": 6,
                "frequency_hz": 100.0,
            },
            [_segment(0), _segment(1)],
            {"layout_id": 3, "operator_accepted": True},
        )

        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["format_version"], 2)
        self.assertEqual(manifest["segment_count"], 2)
        self.assertEqual(
            manifest["segments"][0]["source_slice"],
            {
                "sample_start_index": 0,
                "sample_stop_index": 3,
                "sample_count": 3,
                "segment_start_ns": 1_000_000_000,
                "segment_end_ns": 1_100_000_000,
                "segment_anchor_timestamp_ns": 1_010_000_000,
            },
        )
        self.assertEqual(
            manifest["segments"][1]["source_slice"]["sample_start_index"], 3
        )
        self.assertEqual(
            set(manifest["files"]),
            {
                "source",
                "segment_000_takeover_state",
                "segment_000_terminal_state",
                "segment_001_takeover_state",
                "segment_001_terminal_state",
            },
        )

        bundle = load_pending_bundles(self.root)[0]
        self.assertEqual(bundle.segment_count, 2)
        self.assertEqual(bundle.segments[1].snapshot["replay_manifest"]["segment"], 1)
        self.assertEqual(int(bundle.segments[0].takeover_state["frame.index"]), 10)
        self.assertEqual(int(bundle.segments[1].terminal_state["frame.index"]), 29)
        with self.assertRaisesRegex(RawBundleError, "multiple intervention"):
            _ = bundle.takeover_state

        first = bundle.source_for_segment(0)
        second = bundle.source_for_segment(1)
        np.testing.assert_array_equal(first["segment_index"], [0, 0, 0])
        np.testing.assert_array_equal(second["segment_index"], [1, 1, 1])
        self.assertEqual(first["left_segment_anchor_q_rad"].shape, (1, 6))
        np.testing.assert_allclose(second["left_q_rad"][:, 0], [10.1, 10.5, 10.9])

        replay = resample_segment_25hz(bundle, segment_index=1)
        np.testing.assert_allclose(replay["timestamp_s"], [0.0, 0.04, 0.08])
        np.testing.assert_allclose(replay["left_q_rad"][:, 0], [10.1, 10.5, 10.9])

    def test_v1_and_v2_bundles_can_coexist_and_load_in_order(self) -> None:
        store = self._store(target=3)
        self._commit(store)
        multi = Path(self.temp.name) / "multi.npz"
        _write_multi_fragment(multi)
        store.commit_segments(
            {"path": multi, "sha256": _sha256(multi), "segment_count": 2},
            [_segment(0), _segment(1)],
            {"layout_id": 3},
        )

        bundles = load_pending_bundles(self.root)
        self.assertEqual([item.episode_index for item in bundles], [0, 1])
        self.assertEqual([item.segment_count for item in bundles], [1, 2])
        self.assertEqual(bundles[0].manifest["format_version"], 1)
        self.assertEqual(bundles[1].manifest["format_version"], 2)

    def test_multi_segment_descriptor_and_source_layout_must_match(self) -> None:
        multi = Path(self.temp.name) / "multi.npz"
        _write_multi_fragment(multi)
        store = self._store()
        with self.assertRaisesRegex(RawBundleError, "descriptor segment_count"):
            store.commit_segments(
                {"path": multi, "sha256": _sha256(multi), "segment_count": 1},
                [_segment(0), _segment(1)],
            )
        self.assertEqual(store.completed_count, 0)

        broken = Path(self.temp.name) / "broken_multi.npz"
        with np.load(multi, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["segment_index"] = np.asarray([0, 1, 0, 1, 1, 1], dtype=np.int32)
        np.savez(broken, **arrays)
        with self.assertRaisesRegex(RawBundleError, "not contiguous"):
            store.commit_segments(
                {"path": broken, "sha256": _sha256(broken), "segment_count": 2},
                [_segment(0), _segment(1)],
            )
        self.assertEqual(store.completed_count, 0)


class ResampleSegmentTest(unittest.TestCase):
    def test_linear_25hz_resample_from_nanoseconds(self) -> None:
        source = {
            "sample_monotonic_ns": np.asarray(
                [9_000_000_000, 9_050_000_000, 9_100_000_000], dtype=np.int64
            ),
            "q_rad": np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
            "gripper": np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
            "sequence": np.asarray([10, 11, 12], dtype=np.int64),
            "constant": np.asarray(7, dtype=np.int64),
        }
        result = resample_segment_25hz(source)

        np.testing.assert_allclose(result["timestamp_s"], [0.0, 0.04, 0.08])
        np.testing.assert_allclose(result["q_rad"][:, 0], [0.0, 0.8, 1.6])
        np.testing.assert_allclose(result["gripper"], [0.0, 0.4, 0.8])
        np.testing.assert_array_equal(result["sequence"], [10, 10, 11])
        self.assertEqual(int(result["constant"]), 7)

    def test_seconds_timestamps_and_requested_fps(self) -> None:
        result = resample_segment_25hz(
            {
                "sample_monotonic_s": np.asarray([100.0, 100.1]),
                "value": np.asarray([2.0, 4.0]),
            },
            fps=10,
        )
        np.testing.assert_allclose(result["timestamp_s"], [0.0, 0.1])
        np.testing.assert_allclose(result["value"], [2.0, 4.0])

    def test_invalid_timestamps_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            resample_segment_25hz(
                {
                    "timestamp_s": np.asarray([1.0, 1.0]),
                    "value": np.asarray([0.0, 1.0]),
                }
            )
        with self.assertRaisesRegex(ValueError, "timestamp key"):
            resample_segment_25hz({"value": np.asarray([0.0, 1.0])})


if __name__ == "__main__":
    unittest.main()
