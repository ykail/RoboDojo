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
