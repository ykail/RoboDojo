import io
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scripts.RoboDojo import lerobot_stream_writer as writer
from src.eval_client.lerobot_stream_protocol import receive_message, send_message
from src.eval_client.lerobot_stream_recorder import (
    LeRobotStreamRecorder,
    LeRobotStreamStartupError,
    StreamConfig,
    _task_metadata,
    _WriterSidecar,
    config_from_environment,
)


def _joints(offset: float = 0.0):
    return {
        "left_arm_joint_state": np.arange(6, dtype=np.float32) + offset,
        "left_ee_joint_state": np.asarray([0.25], dtype=np.float32),
        "right_arm_joint_state": np.arange(6, dtype=np.float32) + offset + 10,
        "right_ee_joint_state": np.asarray([0.75], dtype=np.float32),
    }


def _frame_message(*, source="policy", edge=0, policy=True):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "command": "frame",
        "obs": {
            "instruction": "make toast",
            "state": _joints(),
            "vision": {
                name: {"color": image}
                for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
            },
        },
        "policy_action": _joints(1) if policy else None,
        "executed_action": _joints(2),
        "control": {
            "intervention_mask": 1 if source == "human" else 0,
            "action_source": source,
            "takeover_edge": edge,
        },
    }


class _FakeMeta:
    total_episodes = 0


class _FakeEncoder:
    def __init__(self):
        self._dropped_frames = {}
        self._frame_queues = {"cam": queue.Queue(maxsize=1)}
        self._threads = {}


class _FakeDataset:
    def __init__(self, *, drop=False, total_episodes=0):
        self.meta = _FakeMeta()
        self.meta.total_episodes = total_episodes
        self._streaming_encoder = _FakeEncoder()
        self.drop = drop
        self.frames = []
        self.clears = 0
        self.saves = []
        self.finalized = 0

    def add_frame(self, frame):
        self.frames.append(frame)
        if self.drop:
            self._streaming_encoder._dropped_frames = {"cam": 1}

    def clear_episode_buffer(self, delete_images=True):
        self.clears += 1
        self.frames.clear()
        self._streaming_encoder._dropped_frames.clear()

    # Match the LeRobot 0.4.4 API pinned by Pi_05/openpi.  In particular this
    # fake must reject the newer ``extra_episode_metadata`` keyword so the
    # production incompatibility cannot slip through the test again.
    def save_episode(self, episode_data=None, parallel_encoding=True):
        self.saves.append(
            {
                "episode_data": episode_data,
                "parallel_encoding": parallel_encoding,
            }
        )
        self.meta.total_episodes += 1
        self.frames.clear()

    def stop_image_writer(self):
        pass

    def finalize(self):
        self.finalized += 1


class _MutatingFailDataset(_FakeDataset):
    def __init__(self, root, *, total_episodes, outside_target=None, events=None):
        super().__init__(total_episodes=total_episodes)
        self.root = Path(root)
        self.outside_target = outside_target
        self.events = events if events is not None else []

    def save_episode(self, episode_data=None, parallel_encoding=True):
        del episode_data, parallel_encoding
        self.events.append("save")
        (self.root / "meta" / "info.json").write_bytes(b"mutated-info")
        (self.root / "meta" / "stats.json").write_bytes(b"mutated-stats")
        (self.root / "meta" / "tasks.parquet").write_bytes(b"mutated-tasks")
        for relative, payload in (
            ("data/chunk-000/file-001.parquet", b"candidate-data"),
            ("videos/cam_high/chunk-000/file-001.mp4", b"candidate-video"),
            ("meta/episodes/chunk-000/file-001.parquet", b"candidate-episode"),
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if self.outside_target is not None:
            (self.root / "candidate-link").symlink_to(self.outside_target)
        self.meta.total_episodes += 1
        raise RuntimeError("injected save failure")

    def clear_episode_buffer(self, delete_images=True):
        self.events.append("clear")
        super().clear_episode_buffer(delete_images=delete_images)

    def finalize(self):
        self.events.append("finalize")
        super().finalize()


class _MutatingSuccessDataset(_MutatingFailDataset):
    def save_episode(self, episode_data=None, parallel_encoding=True):
        del episode_data, parallel_encoding
        self.events.append("save")
        (self.root / "meta" / "info.json").write_bytes(b"mutated-info")
        (self.root / "meta" / "stats.json").write_bytes(b"mutated-stats")
        (self.root / "meta" / "tasks.parquet").write_bytes(b"mutated-tasks")
        for relative, payload in (
            ("data/chunk-000/file-001.parquet", b"candidate-data"),
            ("videos/cam_high/chunk-000/file-001.mp4", b"candidate-video"),
            ("meta/episodes/chunk-000/file-001.parquet", b"candidate-episode"),
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        self.meta.total_episodes += 1
        self.frames.clear()


def _packet_stream(messages):
    stream = io.BytesIO()
    for message in messages:
        send_message(stream, message)
    stream.seek(0)
    return stream


def _all_packets(stream):
    stream.seek(0)
    packets = []
    while True:
        try:
            packets.append(receive_message(stream))
        except EOFError:
            return packets


class LeRobotStreamWriterTest(unittest.TestCase):
    def test_episode_metadata_preserves_strict_policy_provenance(self):
        provenance = {
            "implementation": "kai0",
            "config_name": "pi05_robodojo_arx_x5_joint",
            "checkpoint_id": "toast/5000",
            "checkpoint_digest": "sha256:" + "a" * 64,
            "code_revision": "b" * 40,
            "dirty": False,
        }
        metadata = writer._episode_metadata(
            {
                "task_name": "make_toast",
                "policy_runtime": "robodojo_policy_v1",
                "policy_provenance": provenance,
            },
            success=True,
            reason="operator_accept",
            has_intervention=True,
            frame_count=42,
        )
        self.assertEqual(
            metadata["robodojo_policy_runtime"],
            "robodojo_policy_v1",
        )
        self.assertEqual(
            metadata["robodojo_policy_provenance"],
            provenance,
        )

    def test_frame_matches_kai0_intervention_features(self):
        frame = writer.build_frame(_frame_message(source="safety_hold", policy=False))
        self.assertEqual(frame["observation.state"].shape, (14,))
        self.assertEqual(frame["action"].shape, (14,))
        np.testing.assert_array_equal(
            frame["complementary_info.policy_action"], np.zeros(14, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            frame["complementary_info.is_intervention"], np.asarray([0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            frame["complementary_info.state"], np.asarray([1], dtype=np.float32)
        )
        self.assertEqual(frame["observation.images.cam_high"].shape, (480, 640, 3))

        release = writer.build_frame(_frame_message(source="policy", edge=-1))
        self.assertEqual(release["complementary_info.state"].item(), 2.0)

    def test_nonfinite_policy_and_pixels_are_rejected(self):
        message = _frame_message()
        message["policy_action"]["left_arm_joint_state"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            writer.build_frame(message)

        message = _frame_message()
        bad_image = np.zeros((480, 640, 3), dtype=np.float32)
        bad_image[0, 0, 0] = np.inf
        message["obs"]["vision"]["cam_head"]["color"] = bad_image
        with self.assertRaisesRegex(ValueError, "non-finite pixels"):
            writer.build_frame(message)

    def test_discard_then_commit_in_one_streaming_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "test"
            (dataset_root / "meta").mkdir(parents=True)
            (dataset_root / "meta" / "info.json").write_text(
                json.dumps({"total_episodes": 0, "total_frames": 0}), encoding="utf-8"
            )
            dataset = _FakeDataset()
            config = writer.WriterConfig(
                repo_id="test", root=Path(tmp), fps=25, resume=False,
                vcodec="h264", encoder_threads=1,
            )
            messages = [
                {"command": "begin", "metadata": {"task_name": "make_toast"}},
                _frame_message(source="human"),
                {"command": "finish", "accepted": False},
                {"command": "begin", "metadata": {"task_name": "make_toast", "layout_cycle": 2}},
                _frame_message(source="policy"),
                {"command": "finish", "accepted": True, "success": True, "reason": "right"},
            ]
            output = io.BytesIO()
            status = writer.serve(
                config,
                _packet_stream(messages),
                output,
                dataset_opener=lambda _config: (dataset, True),
            )
            self.assertEqual(status, 0)
            self.assertEqual(
                [packet["status"] for packet in _all_packets(output)],
                ["ready", "begun", "frame", "discarded", "begun", "frame", "committed"],
            )
            self.assertEqual(dataset.clears, 1)
            self.assertEqual(len(dataset.saves), 1)
            self.assertIsNone(dataset.saves[0]["episode_data"])
            metadata_path = (
                dataset_root
                / writer._ROBODOJO_EPISODE_METADATA_DIR
                / "episode_0000000.json"
            )
            self.assertTrue(metadata_path.is_file())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["episode_index"], 0)
            self.assertEqual(metadata["robodojo_layout_cycle"], 2)
            self.assertTrue(metadata["robodojo_success"])
            self.assertEqual(dataset.finalized, 1)

    def test_failed_later_commit_restores_exact_tree_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = base / "test"
            (dataset_root / "meta").mkdir(parents=True)
            originals = {
                "meta/info.json": b'{"total_episodes":1,"total_frames":7}',
                "meta/stats.json": b"original-stats",
                "meta/tasks.parquet": b"original-tasks",
                "data/chunk-000/file-000.parquet": b"accepted-data",
                "videos/cam_high/chunk-000/file-000.mp4": b"accepted-video",
                "meta/episodes/chunk-000/file-000.parquet": b"accepted-episode",
            }
            for relative, payload in originals.items():
                path = dataset_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            outside = base / "outside.bin"
            outside.write_bytes(b"must-not-be-touched")

            events = []
            dataset = _MutatingFailDataset(
                dataset_root,
                total_episodes=1,
                outside_target=outside,
                events=events,
            )
            config = writer.WriterConfig(
                repo_id="test", root=base, fps=25, resume=True,
                vcodec="h264", encoder_threads=1,
            )
            original_rollback = writer._CommitSnapshot.rollback

            def tracked_rollback(snapshot):
                events.append("rollback")
                return original_rollback(snapshot)

            output = io.BytesIO()
            with mock.patch.object(writer._CommitSnapshot, "rollback", tracked_rollback):
                status = writer.serve(
                    config,
                    _packet_stream(
                        [
                            {"command": "begin", "metadata": {"task_name": "make_toast"}},
                            _frame_message(),
                            {"command": "finish", "accepted": True},
                        ]
                    ),
                    output,
                    dataset_opener=lambda _config: (dataset, False),
                )

            self.assertEqual(status, 1)
            self.assertEqual(events, ["save", "clear", "finalize", "rollback"])
            self.assertEqual(
                [packet["status"] for packet in _all_packets(output)],
                ["ready", "begun", "frame", "error"],
            )
            self.assertTrue(_all_packets(output)[-1]["fatal"])
            for relative, payload in originals.items():
                self.assertEqual((dataset_root / relative).read_bytes(), payload)
            self.assertFalse((dataset_root / "candidate-link").exists())
            self.assertEqual(outside.read_bytes(), b"must-not-be-touched")
            self.assertEqual(writer._scan_tree(dataset_root), {
                Path(relative): "file" for relative in originals
            } | {
                Path("meta"): "directory",
                Path("data"): "directory",
                Path("data/chunk-000"): "directory",
                Path("videos"): "directory",
                Path("videos/cam_high"): "directory",
                Path("videos/cam_high/chunk-000"): "directory",
                Path("meta/episodes"): "directory",
                Path("meta/episodes/chunk-000"): "directory",
            })

    def test_failed_first_commit_rolls_back_then_removes_safe_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = base / "test"
            (dataset_root / "meta").mkdir(parents=True)
            (dataset_root / "meta" / "info.json").write_text(
                json.dumps({"total_episodes": 0, "total_frames": 0}), encoding="utf-8"
            )
            (dataset_root / writer._STAGING_MARKER).write_text("{}", encoding="utf-8")
            dataset = _MutatingFailDataset(dataset_root, total_episodes=0)
            config = writer.WriterConfig(
                repo_id="test", root=base, fps=25, resume=False,
                vcodec="h264", encoder_threads=1,
            )
            output = io.BytesIO()
            status = writer.serve(
                config,
                _packet_stream(
                    [
                        {"command": "begin", "metadata": {"task_name": "make_toast"}},
                        _frame_message(),
                        {"command": "finish", "accepted": True},
                    ]
                ),
                output,
                dataset_opener=lambda _config: (dataset, True),
            )

            self.assertEqual(status, 1)
            self.assertFalse(dataset_root.exists())

    def test_sidecar_failure_after_finalize_rolls_back_and_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = base / "test"
            originals = {
                "meta/info.json": b'{"total_episodes":1,"total_frames":7}',
                "meta/stats.json": b"original-stats",
                "meta/tasks.parquet": b"original-tasks",
                "data/chunk-000/file-000.parquet": b"accepted-data",
                "videos/cam_high/chunk-000/file-000.mp4": b"accepted-video",
                "meta/episodes/chunk-000/file-000.parquet": b"accepted-episode",
            }
            for relative, payload in originals.items():
                path = dataset_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)

            events = []
            dataset = _MutatingSuccessDataset(dataset_root, total_episodes=1, events=events)
            config = writer.WriterConfig(
                repo_id="test", root=base, fps=25, resume=True,
                vcodec="h264", encoder_threads=1,
            )
            original_rollback = writer._CommitSnapshot.rollback

            def tracked_rollback(snapshot):
                events.append("rollback")
                return original_rollback(snapshot)

            output = io.BytesIO()
            with (
                mock.patch.object(writer._CommitSnapshot, "rollback", tracked_rollback),
                mock.patch.object(
                    writer,
                    "_write_robodojo_episode_metadata",
                    side_effect=OSError("injected sidecar failure"),
                ),
            ):
                status = writer.serve(
                    config,
                    _packet_stream(
                        [
                            {"command": "begin", "metadata": {"task_name": "make_toast"}},
                            _frame_message(),
                            {"command": "finish", "accepted": True},
                        ]
                    ),
                    output,
                    dataset_opener=lambda _config: (dataset, False),
                )

            self.assertEqual(status, 1)
            self.assertEqual(events, ["save", "finalize", "rollback"])
            error = _all_packets(output)[-1]
            self.assertTrue(error["fatal"])
            self.assertIn("injected sidecar failure", error["error"])
            for relative, payload in originals.items():
                self.assertEqual((dataset_root / relative).read_bytes(), payload)
            self.assertEqual(
                writer._scan_tree(dataset_root),
                {Path(relative): "file" for relative in originals}
                | {
                    Path("meta"): "directory",
                    Path("data"): "directory",
                    Path("data/chunk-000"): "directory",
                    Path("videos"): "directory",
                    Path("videos/cam_high"): "directory",
                    Path("videos/cam_high/chunk-000"): "directory",
                    Path("meta/episodes"): "directory",
                    Path("meta/episodes/chunk-000"): "directory",
                },
            )

    def test_robodojo_metadata_resume_index_and_stale_file_are_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            (root / "meta").mkdir(parents=True)
            first = writer._write_robodojo_episode_metadata(root, 0, {"value": "first"})
            second = writer._write_robodojo_episode_metadata(root, 1, {"value": "second"})
            self.assertEqual(json.loads(first.read_text(encoding="utf-8"))["episode_index"], 0)
            self.assertEqual(json.loads(second.read_text(encoding="utf-8"))["episode_index"], 1)
            with self.assertRaisesRegex(writer.EpisodeCommitError, "already exists"):
                writer._write_robodojo_episode_metadata(root, 1, {"value": "stale"})
            self.assertEqual(json.loads(second.read_text(encoding="utf-8"))["value"], "second")

    def test_robodojo_metadata_refuses_symlink_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "dataset"
            outside = base / "outside"
            (root / "meta").mkdir(parents=True)
            outside.mkdir()
            (root / "meta" / "robodojo").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(writer.EpisodeCommitError, "real directory"):
                writer._write_robodojo_episode_metadata(root, 0, {"value": "unsafe"})
            self.assertEqual(list(outside.iterdir()), [])

    def test_incomplete_rollback_is_reported_as_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = base / "test"
            (dataset_root / "meta").mkdir(parents=True)
            (dataset_root / "meta" / "info.json").write_text(
                json.dumps({"total_episodes": 1, "total_frames": 7}), encoding="utf-8"
            )
            dataset = _MutatingFailDataset(dataset_root, total_episodes=1)
            config = writer.WriterConfig(
                repo_id="test", root=base, fps=25, resume=True,
                vcodec="h264", encoder_threads=1,
            )
            output = io.BytesIO()
            with mock.patch.object(
                writer._CommitSnapshot,
                "rollback",
                side_effect=RuntimeError("injected rollback failure"),
            ):
                status = writer.serve(
                    config,
                    _packet_stream(
                        [
                            {"command": "begin", "metadata": {"task_name": "make_toast"}},
                            _frame_message(),
                            {"command": "finish", "accepted": True},
                        ]
                    ),
                    output,
                    dataset_opener=lambda _config: (dataset, False),
                )

            self.assertEqual(status, 1)
            error = _all_packets(output)[-1]
            self.assertEqual(error["status"], "error")
            self.assertTrue(error["fatal"])
            self.assertIn("injected rollback failure", error["error"])

    def test_parent_classifies_fatal_writer_error_as_non_retryable(self):
        with self.assertRaisesRegex(LeRobotStreamStartupError, "rollback incomplete"):
            _WriterSidecar._expect(
                {
                    "status": "error",
                    "fatal": True,
                    "error": "rollback incomplete",
                },
                "committed",
            )

    def test_encoder_drop_refuses_commit_and_discards_candidate(self):
        dataset = _FakeDataset(drop=True)
        config = writer.WriterConfig(
            repo_id="test", root=Path("/tmp"), fps=25, resume=False,
            vcodec="h264", encoder_threads=1,
        )
        output = io.BytesIO()
        status = writer.serve(
            config,
            _packet_stream(
                [
                    {"command": "begin", "metadata": {}},
                    _frame_message(),
                    {"command": "finish", "accepted": True},
                ]
            ),
            output,
            dataset_opener=lambda _config: (dataset, True),
        )
        self.assertEqual(status, 1)
        self.assertEqual([packet["status"] for packet in _all_packets(output)], ["ready", "begun", "error"])
        self.assertEqual(dataset.saves, [])
        self.assertEqual(dataset.clears, 1)

    def test_only_marked_zero_episode_staging_can_be_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            (root / "meta").mkdir(parents=True)
            (root / "meta" / "info.json").write_text(
                json.dumps({"total_episodes": 0, "total_frames": 0}), encoding="utf-8"
            )
            (root / writer._STAGING_MARKER).write_text("{}", encoding="utf-8")
            (root / writer._COLLECTION_SESSION_MARKER).write_text(
                json.dumps({"run_id": "same-run"}), encoding="utf-8"
            )
            temp_video = root / "tmp-encoder" / "candidate.mp4"
            temp_video.parent.mkdir()
            temp_video.write_bytes(b"temporary")
            self.assertTrue(writer.is_safe_empty_staging(root))

            final_video = root / "videos" / "cam" / "chunk-000" / "file-000.mp4"
            final_video.parent.mkdir(parents=True)
            final_video.write_bytes(b"user data")
            self.assertFalse(writer.is_safe_empty_staging(root))
            self.assertFalse(writer.remove_safe_empty_staging(root))
            self.assertTrue(final_video.exists())

            final_video.unlink()
            self.assertTrue(writer.remove_safe_empty_staging(root))
            self.assertFalse(root.exists())

    def test_orphan_sweep_only_removes_streaming_encoder_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orphan = root / "tmp-encoder"
            orphan.mkdir()
            (orphan / "observation.images.cam_high_streaming.mp4").write_bytes(b"partial")
            final = root / "videos" / "cam"
            final.mkdir(parents=True)
            (final / "file-000.mp4").write_bytes(b"accepted")
            unfamiliar = root / "notes"
            unfamiliar.mkdir()
            (unfamiliar / "clip.mp4").write_bytes(b"user")

            self.assertEqual(writer.remove_orphan_encoder_temp_dirs(root), [orphan.resolve()])
            self.assertFalse(orphan.exists())
            self.assertTrue((final / "file-000.mp4").exists())
            self.assertTrue((unfamiliar / "clip.mp4").exists())


class _FakeSidecar:
    def __init__(self):
        self.finish_calls = []

    def finish(self, **kwargs):
        self.finish_calls.append(kwargs)
        return {"status": "discarded", "frame_count": 0}


class LeRobotStreamRecorderTest(unittest.TestCase):
    def test_task_metadata_uses_connected_checkpoint_identity(self):
        provenance = {
            "checkpoint_id": "toast/5000",
            "checkpoint_digest": "sha256:" + "a" * 64,
            "code_revision": "b" * 40,
            "dirty": False,
        }
        task_env = SimpleNamespace(
            env_seeds=[7],
            layout_cycle=2,
            task_name="make_toast",
            config_name="arx_x5",
            eval_seed=3,
            policy_name="Kai0_Pi05",
            policy_runtime="robodojo_policy_v1",
            policy_provenance=provenance,
            additional_info="human-label",
        )
        with (
            mock.patch.dict(
                os.environ,
                {
                    "ROBODOJO_RUN_ID": "run-1",
                    "ROBODOJO_CHECKPOINT": "conflicting-legacy-label",
                },
                clear=True,
            ),
            mock.patch(
                "src.eval_client.lerobot_stream_recorder._git_revision",
                return_value="revision",
            ),
        ):
            metadata = _task_metadata(task_env)

        self.assertEqual(metadata["base_checkpoint"], "toast/5000")
        self.assertEqual(metadata["policy_runtime"], "robodojo_policy_v1")
        self.assertEqual(metadata["policy_provenance"], provenance)

    def test_config_preserves_virtualenv_python_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv_python = root / "openpi" / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(Path(sys.executable).resolve())
            environment = {
                "ROBODOJO_LEROBOT_PYTHON": str(venv_python),
                "ROBODOJO_LEROBOT_ROOT": str(root / "data"),
                "ROBODOJO_LEROBOT_REPO_ID": "repo",
                "ROBODOJO_LEROBOT_STREAMING_ENCODING": "1",
            }

            with mock.patch.dict(os.environ, environment, clear=False):
                config = config_from_environment(fps=25)

            self.assertEqual(config.python, venv_python.absolute())
            self.assertNotEqual(config.python, config.python.resolve())

    def test_same_run_marker_enables_resume_after_process_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "repo"
            dataset.mkdir()
            (dataset / ".robodojo_collection_session.json").write_text(
                json.dumps({"run_id": "run-123"}), encoding="utf-8"
            )
            environment = {
                "ROBODOJO_LEROBOT_PYTHON": sys.executable,
                "ROBODOJO_LEROBOT_ROOT": str(root),
                "ROBODOJO_LEROBOT_REPO_ID": "repo",
                "ROBODOJO_LEROBOT_RESUME": "0",
                "ROBODOJO_LEROBOT_STREAMING_ENCODING": "1",
                "ROBODOJO_RUN_ID": "run-123",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                self.assertTrue(config_from_environment(fps=25).resume)
            environment["ROBODOJO_RUN_ID"] = "different-run"
            with mock.patch.dict(os.environ, environment, clear=False):
                self.assertFalse(config_from_environment(fps=25).resume)

    def test_zero_frame_accept_is_discarded_cleanly(self):
        fake = _FakeSidecar()
        config = StreamConfig(
            python=Path("/python"), project_root=Path("/repo"), repo_id="dataset",
            root=Path("/data"), fps=25, resume=False, vcodec="h264",
            encoder_threads=1, encoder_queue_maxsize=8, video_crf=18,
        )
        with mock.patch(
            "src.eval_client.lerobot_stream_recorder._acquire_sidecar", return_value=fake
        ):
            recorder = LeRobotStreamRecorder(config, {"task_name": "make_toast"})
        self.assertEqual(recorder.record_dir, "/data/dataset")
        self.assertIsNone(recorder.finalize(accepted=True, success=False, reason="escape"))
        self.assertEqual(fake.finish_calls[0]["accepted"], False)


if __name__ == "__main__":
    unittest.main()
