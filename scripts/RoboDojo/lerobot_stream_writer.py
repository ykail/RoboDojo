#!/usr/bin/env python3
"""CPU-only LeRobot v3 streaming writer used by keyboard intervention.

The Isaac process deliberately does not import LeRobot.  It sends one frame at
a time to this sidecar and waits for an acknowledgement.  Rejected candidates
are cleared in-place; an accepted candidate is saved, finalized, and closes
this process so the next episode reopens a fully valid dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time
import traceback
from typing import Any, BinaryIO, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_client.lerobot_stream_protocol import receive_message, send_message


CAMERA_SOURCES = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
MOTOR_NAMES = tuple(
    [f"left_joint_{index}" for index in range(7)]
    + [f"right_joint_{index}" for index in range(7)]
)
_JOINT_PARTS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)
_STAGING_MARKER = ".robodojo_lerobot_staging.json"
_COLLECTION_SESSION_MARKER = ".robodojo_collection_session.json"
_ROBODOJO_EPISODE_METADATA_DIR = Path("meta/robodojo/episodes")
_ROLLBACK_FILE_PATHS = (
    Path("meta/info.json"),
    Path("meta/stats.json"),
    Path("meta/tasks.parquet"),
    Path(_STAGING_MARKER),
)


class EpisodeCommitError(RuntimeError):
    """An episode could not be committed and retrying the layout cannot fix it."""


class CommitRollbackError(EpisodeCommitError):
    """An accepted commit failed and its on-disk rollback was incomplete."""


@dataclass(frozen=True)
class WriterConfig:
    repo_id: str
    root: Path
    fps: int
    resume: bool
    vcodec: str
    encoder_threads: int
    encoder_queue_maxsize: int = 128
    video_crf: int = 18
    encoder_wait_s: float = 120.0

    @property
    def dataset_root(self) -> Path:
        return resolve_dataset_root(self.root, self.repo_id)


@dataclass(frozen=True)
class _FileBackup:
    contents: bytes
    mode: int


@dataclass(frozen=True)
class _CommitSnapshot:
    """On-disk state that must survive a failed episode commit.

    A sidecar writes only one accepted episode before exiting.  Consequently,
    a freshly reopened LeRobot dataset writes candidate data, videos and
    episode metadata to new files.  LeRobot does update three shared metadata
    files in place, so those are backed up byte-for-byte.  Streaming encoder
    temp directories already exist when ``save_episode`` starts; they are
    deliberately excluded from the committed baseline and removed on failure.
    """

    root: Path
    entries: tuple[tuple[Path, str], ...]
    file_backups: tuple[tuple[Path, _FileBackup | None], ...]

    @classmethod
    def capture(
        cls,
        dataset_root: str | Path,
        *,
        transient_roots: tuple[Path, ...] = (),
    ) -> "_CommitSnapshot":
        root = Path(dataset_root).expanduser().resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError(f"LeRobot dataset root is not a real directory: {root}")

        transient = tuple(_validate_relative_path(path) for path in transient_roots)
        all_entries = _scan_tree(root)
        entries = {
            relative: kind
            for relative, kind in all_entries.items()
            if not any(_is_at_or_below(relative, prefix) for prefix in transient)
        }

        backups: list[tuple[Path, _FileBackup | None]] = []
        for relative in _ROLLBACK_FILE_PATHS:
            path = root / relative
            try:
                path_stat = path.lstat()
            except FileNotFoundError:
                backups.append((relative, None))
                continue
            if not stat.S_ISREG(path_stat.st_mode):
                raise RuntimeError(f"Refusing to snapshot non-regular file: {path}")
            backups.append(
                (
                    relative,
                    _FileBackup(path.read_bytes(), stat.S_IMODE(path_stat.st_mode)),
                )
            )

        return cls(
            root=root,
            entries=tuple(sorted(entries.items(), key=lambda item: item[0].parts)),
            file_backups=tuple(backups),
        )

    def rollback(self) -> None:
        """Restore the exact pre-commit tree without following symlinks."""

        baseline = dict(self.entries)
        current = _scan_tree(self.root)

        # Remove new non-directory entries first.  ``unlink`` removes a
        # symlink itself and never follows its target.
        new_non_directories = [
            (relative, kind)
            for relative, kind in current.items()
            if relative not in baseline and kind != "directory"
        ]
        for relative, kind in sorted(
            new_non_directories,
            key=lambda item: (len(item[0].parts), item[0].parts),
            reverse=True,
        ):
            if kind not in {"file", "symlink"}:
                raise RuntimeError(
                    f"Refusing to remove unexpected {kind} created during commit: "
                    f"{self.root / relative}"
                )
            (self.root / relative).unlink(missing_ok=True)

        # Then remove only directories created by this commit, deepest first,
        # and only if they are empty.  Never recursively delete a directory.
        current = _scan_tree(self.root)
        new_directories = [
            relative
            for relative, kind in current.items()
            if relative not in baseline and kind == "directory"
        ]
        for relative in sorted(
            new_directories,
            key=lambda path: (len(path.parts), path.parts),
            reverse=True,
        ):
            (self.root / relative).rmdir()

        for relative, backup in self.file_backups:
            path = self.root / relative
            if backup is None:
                if os.path.lexists(path):
                    path_stat = path.lstat()
                    if not (stat.S_ISREG(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode)):
                        raise RuntimeError(
                            f"Refusing to remove unexpected rollback target: {path}"
                        )
                    path.unlink()
                continue
            _restore_file_atomically(path, backup)

        restored = _scan_tree(self.root)
        if restored != baseline:
            extra = sorted(set(restored) - set(baseline), key=lambda path: path.parts)
            missing = sorted(set(baseline) - set(restored), key=lambda path: path.parts)
            changed_types = sorted(
                (
                    path
                    for path in set(restored) & set(baseline)
                    if restored[path] != baseline[path]
                ),
                key=lambda path: path.parts,
            )
            raise RuntimeError(
                "LeRobot rollback could not restore the pre-commit tree "
                f"(extra={extra}, missing={missing}, changed_types={changed_types})"
            )


def resolve_dataset_root(base_root: str | Path, repo_id: str) -> Path:
    base = Path(base_root).expanduser().resolve()
    dataset_root = (base / repo_id).resolve()
    try:
        dataset_root.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"repo_id must resolve below {base}, got {repo_id!r}") from exc
    if dataset_root == base:
        raise ValueError(f"repo_id must name a child of {base}")
    return dataset_root


def _validate_relative_path(path: str | Path) -> Path:
    relative = Path(path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Expected a path relative to the dataset root, got {path!r}")
    return relative


def _is_at_or_below(path: Path, parent: Path) -> bool:
    return path == parent or path.parts[: len(parent.parts)] == parent.parts


def _scan_tree(root: Path) -> dict[Path, str]:
    """Return every entry below *root* without traversing symlink directories."""

    entries: dict[Path, str] = {}
    pending: list[tuple[Path, Path]] = [(root, Path())]
    while pending:
        directory, relative_directory = pending.pop()
        with os.scandir(directory) as iterator:
            for entry in iterator:
                relative = relative_directory / entry.name
                if entry.is_symlink():
                    entries[relative] = "symlink"
                elif entry.is_dir(follow_symlinks=False):
                    entries[relative] = "directory"
                    pending.append((Path(entry.path), relative))
                elif entry.is_file(follow_symlinks=False):
                    entries[relative] = "file"
                else:
                    entries[relative] = "special"
    return entries


def _restore_file_atomically(path: Path, backup: _FileBackup) -> None:
    parent_stat = path.parent.lstat()
    if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode):
        raise RuntimeError(f"Refusing to restore through a non-directory: {path.parent}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.robodojo-rollback-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(backup.contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, backup.mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _encoder_transient_roots(dataset: Any, dataset_root: Path) -> tuple[Path, ...]:
    """Find exact one-level temp directories owned by the active encoder."""

    encoder = getattr(dataset, "_streaming_encoder", None)
    video_paths = getattr(encoder, "_video_paths", {}) if encoder is not None else {}
    root = dataset_root.expanduser().resolve()
    transient: set[Path] = set()
    for raw_path in video_paths.values():
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve(strict=False)
        parent = resolved.parent
        if parent.parent != root or parent.name in {"data", "videos", "meta"}:
            raise RuntimeError(f"Unexpected LeRobot encoder temp path: {path}")
        transient.add(parent.relative_to(root))
    return tuple(sorted(transient, key=lambda path: path.parts))


def build_features() -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": list(MOTOR_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": list(MOTOR_NAMES),
        },
        "complementary_info.policy_action": {
            "dtype": "float32",
            "shape": (14,),
            "names": list(MOTOR_NAMES),
        },
        "complementary_info.is_intervention": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["is_intervention"],
        },
        "complementary_info.state": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["state"],
        },
    }
    for camera_name in CAMERA_SOURCES:
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": (3, IMAGE_HEIGHT, IMAGE_WIDTH),
            "names": ["channels", "height", "width"],
        }
    return features


def _joint_vector(
    action: dict[str, Any] | None,
    *,
    label: str,
    missing_value: float | None = None,
) -> np.ndarray:
    if action is None:
        if missing_value is None:
            raise ValueError(f"{label} is missing")
        return np.full(14, missing_value, dtype=np.float32)

    parts: list[np.ndarray] = []
    for key, expected_size in _JOINT_PARTS:
        value = action.get(key)
        if value is None:
            if missing_value is None:
                raise ValueError(f"{label}.{key} is missing")
            parts.append(np.full(expected_size, missing_value, dtype=np.float32))
            continue
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != expected_size:
            raise ValueError(
                f"{label}.{key} has {array.size} value(s), expected {expected_size}"
            )
        parts.append(array)
    vector = np.concatenate(parts).astype(np.float32, copy=False)
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} contains non-finite values")
    return vector


def _image_from_observation(obs: dict[str, Any], source_name: str) -> np.ndarray:
    camera = obs.get("vision", {}).get(source_name)
    image = camera.get("color") if isinstance(camera, dict) else camera
    if image is None:
        raise ValueError(f"Missing required camera vision.{source_name}.color")
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Camera {source_name} must be a 3-D image, got {array.shape}")
    if array.shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH):
        array = np.moveaxis(array, 0, -1)
    elif array.shape[-1] != 3:
        raise ValueError(f"Camera {source_name} must have 3 channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"Camera {source_name} contains non-finite pixels")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and float(np.max(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        try:
            import cv2
        except ImportError as exc:
            raise ValueError(
                f"Camera {source_name} is {array.shape[:2]}, expected "
                f"{(IMAGE_HEIGHT, IMAGE_WIDTH)}, and OpenCV is unavailable for resizing"
            ) from exc
        array = cv2.resize(array, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(array)


def build_frame(message: dict[str, Any]) -> dict[str, Any]:
    obs = message["obs"]
    control = message.get("control", {})
    mask = float(control.get("intervention_mask", 0.0))
    if mask not in (0.0, 1.0):
        raise ValueError(f"intervention_mask must be 0 or 1, got {mask!r}")
    edge = control.get("takeover_edge", 0)
    release = edge in (-1, "-1", "release", "released")
    action_source = str(control.get("action_source", "")).strip().lower()
    manually_controlled = bool(mask) or action_source in {"human", "safety_hold"}
    intervention_state = 2.0 if release else (1.0 if manually_controlled else 0.0)

    instruction = str(obs.get("instruction") or message.get("task") or "").strip()
    if not instruction:
        raise ValueError("Observation has no instruction/task text")

    frame: dict[str, Any] = {
        "observation.state": _joint_vector(obs.get("state"), label="observation.state"),
        "action": _joint_vector(message.get("executed_action"), label="action"),
        # Kai0 stores zeros if a policy action is unavailable during manual control;
        # the intervention fields disambiguate this from a genuine zero action.
        "complementary_info.policy_action": _joint_vector(
            message.get("policy_action"),
            label="complementary_info.policy_action",
            missing_value=0.0,
        ),
        "complementary_info.is_intervention": np.asarray([mask], dtype=np.float32),
        "complementary_info.state": np.asarray([intervention_state], dtype=np.float32),
        "task": instruction,
    }
    for target_name, source_name in CAMERA_SOURCES.items():
        frame[f"observation.images.{target_name}"] = _image_from_observation(obs, source_name)
    return frame


def _read_dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Existing directory is not a LeRobot dataset: {info_path} is missing")
    with info_path.open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    if not isinstance(info, dict):
        raise ValueError(f"Invalid LeRobot info.json at {info_path}")
    return info


def is_safe_empty_staging(dataset_root: str | Path) -> bool:
    """Return whether *dataset_root* is our private zero-episode staging tree.

    LeRobot 0.4.4 cannot reopen a dataset with zero episodes.  Recovery may
    therefore recreate only a directory carrying our marker, reporting zero
    committed frames/episodes, and containing no Parquet or MP4 payloads.
    """

    root = Path(dataset_root).expanduser().resolve()
    marker = root / _STAGING_MARKER
    if not root.is_dir() or not marker.is_file():
        return False
    try:
        info = _read_dataset_info(root)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if int(info.get("total_episodes", -1)) != 0 or int(info.get("total_frames", -1)) != 0:
        return False
    # StreamingVideoEncoder writes temporary MP4 files into one-level random
    # directories directly below the dataset root.  Permit only those plus our
    # marker and LeRobot's empty info.json; never delete an unfamiliar file.
    allowed_exact = {
        root / _STAGING_MARKER,
        root / _COLLECTION_SESSION_MARKER,
        root / "meta" / "info.json",
    }
    for path in root.rglob("*"):
        if not path.is_file() or path in allowed_exact:
            continue
        relative = path.relative_to(root)
        is_encoder_temp = (
            len(relative.parts) == 2
            and relative.parts[0] not in {"data", "videos", "meta"}
            and path.suffix == ".mp4"
        )
        if not is_encoder_temp:
            return False
    return True


def remove_safe_empty_staging(dataset_root: str | Path) -> bool:
    root = Path(dataset_root).expanduser().resolve()
    if not is_safe_empty_staging(root):
        return False
    shutil.rmtree(root)
    return True


def remove_orphan_encoder_temp_dirs(dataset_root: str | Path) -> list[Path]:
    """Remove only one-level temp dirs made by StreamingVideoEncoder.

    The parent holds the dataset advisory lock before this sidecar starts.
    Final videos live under ``videos/``; encoder candidates live in a random
    one-level directory and have names ending in ``_streaming.mp4``.
    """

    root = Path(dataset_root).expanduser().resolve()
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for child in root.iterdir():
        if child.name in {"data", "videos", "meta"} or child.is_symlink() or not child.is_dir():
            continue
        entries = list(child.iterdir())
        if not entries:
            continue
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            continue
        if not all(entry.name.endswith("_streaming.mp4") for entry in entries):
            continue
        shutil.rmtree(child)
        removed.append(child)
    return removed


def _validate_existing_dataset(dataset: Any, config: WriterConfig) -> None:
    actual_fps = int(getattr(dataset, "fps", getattr(dataset.meta, "fps", -1)))
    if actual_fps != config.fps:
        raise ValueError(f"Dataset FPS is {actual_fps}, requested {config.fps}")
    expected = build_features()
    actual = dataset.features
    for key, feature in expected.items():
        if key not in actual:
            raise ValueError(f"Existing dataset is missing feature {key!r}")
        if actual[key].get("dtype") != feature["dtype"]:
            raise ValueError(f"Existing dataset feature {key!r} has incompatible dtype")
        if tuple(actual[key].get("shape", ())) != tuple(feature["shape"]):
            raise ValueError(f"Existing dataset feature {key!r} has incompatible shape")


def _configure_encoder(dataset: Any, config: WriterConfig) -> None:
    encoder = getattr(dataset, "_streaming_encoder", None)
    if encoder is None:
        raise RuntimeError("LeRobot streaming video encoder was not initialized")
    encoder.crf = config.video_crf


def open_dataset(config: WriterConfig) -> tuple[Any, bool]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = config.dataset_root
    if dataset_root.exists() and is_safe_empty_staging(dataset_root):
        # This exact tree contains no committed user data and is impossible to
        # resume through LeRobot 0.4.4, so recreate it safely.
        remove_safe_empty_staging(dataset_root)

    if dataset_root.exists():
        if not config.resume:
            raise FileExistsError(
                f"LeRobot dataset already exists: {dataset_root}. Use --resume to append."
            )
        info = _read_dataset_info(dataset_root)
        if int(info.get("total_episodes", 0)) <= 0:
            raise RuntimeError(
                f"Refusing to replace unmarked zero-episode dataset: {dataset_root}"
            )
        removed_temp_dirs = remove_orphan_encoder_temp_dirs(dataset_root)
        if removed_temp_dirs:
            print(
                f"[LEROBOT] removed {len(removed_temp_dirs)} orphan encoder temp dir(s)",
                file=sys.stderr,
                flush=True,
            )
        dataset = LeRobotDataset(
            config.repo_id,
            root=dataset_root,
            batch_encoding_size=1,
            video_backend="pyav",
            vcodec=config.vcodec,
            streaming_encoding=True,
            encoder_queue_maxsize=config.encoder_queue_maxsize,
            encoder_threads=config.encoder_threads,
        )
        _validate_existing_dataset(dataset, config)
        created = False
    else:
        dataset = LeRobotDataset.create(
            repo_id=config.repo_id,
            root=dataset_root,
            fps=config.fps,
            robot_type="arx_x5",
            features=build_features(),
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=0,
            video_backend="pyav",
            batch_encoding_size=1,
            vcodec=config.vcodec,
            metadata_buffer_size=1,
            streaming_encoding=True,
            encoder_queue_maxsize=config.encoder_queue_maxsize,
            encoder_threads=config.encoder_threads,
        )
        marker = dataset_root / _STAGING_MARKER
        marker.write_text(
            json.dumps({"pid": os.getpid(), "created_at": time.time()}),
            encoding="utf-8",
        )
        created = True
    _configure_encoder(dataset, config)
    return dataset, created


def _dropped_frame_counts(dataset: Any) -> dict[str, int]:
    encoder = getattr(dataset, "_streaming_encoder", None)
    counts = getattr(encoder, "_dropped_frames", {}) if encoder is not None else {}
    return {str(key): int(value) for key, value in counts.items() if int(value) > 0}


def _wait_for_encoder_headroom(dataset: Any, timeout_s: float) -> None:
    """Wait until every streaming camera queue has room for the next frame."""

    encoder = getattr(dataset, "_streaming_encoder", None)
    queues = getattr(encoder, "_frame_queues", {}) if encoder is not None else {}
    deadline = time.monotonic() + timeout_s
    while queues and any(frame_queue.full() for frame_queue in queues.values()):
        drops = _dropped_frame_counts(dataset)
        if drops:
            raise RuntimeError(f"LeRobot video encoder dropped frame(s): {drops}")
        threads = getattr(encoder, "_threads", {})
        dead = [name for name, thread in threads.items() if not thread.is_alive()]
        if dead:
            raise RuntimeError(f"LeRobot video encoder thread stopped: {dead}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"LeRobot video encoder did not drain within {timeout_s:g}s"
            )
        time.sleep(0.005)


def _episode_metadata(
    metadata: dict[str, Any],
    *,
    success: bool,
    reason: str,
    has_intervention: bool,
    frame_count: int,
) -> dict[str, Any]:
    def integer(key: str, default: int = -1) -> int:
        try:
            return int(metadata.get(key, default))
        except (TypeError, ValueError):
            return default

    def boolean_or_unknown(key: str) -> bool | str:
        value = metadata.get(key, "unknown")
        return value if isinstance(value, bool) else str(value)

    def float_list(key: str) -> list[float]:
        values = metadata.get(key, [])
        if not isinstance(values, (list, tuple)):
            return []
        result = [float(value) for value in values]
        if not all(math.isfinite(value) and value >= 0.0 for value in result):
            raise ValueError(f"{key} must contain finite non-negative values")
        return result

    def bool_list(key: str) -> list[bool]:
        values = metadata.get(key, [])
        if not isinstance(values, (list, tuple)):
            return []
        return [bool(value) for value in values]

    policy_provenance = metadata.get("policy_provenance", {})
    if not isinstance(policy_provenance, dict):
        policy_provenance = {}
    recovery_source = metadata.get("recovery_source", {})
    if not isinstance(recovery_source, dict):
        recovery_source = {}

    return {
        "robodojo_task": str(metadata.get("task_name", "")),
        "robodojo_env_config": str(metadata.get("env_config", "")),
        "robodojo_checkpoint": str(metadata.get("base_checkpoint", "")),
        "robodojo_policy_name": str(metadata.get("policy_name", "")),
        "robodojo_policy_runtime": str(
            metadata.get("policy_runtime", "xpolicy_ws_v0"),
        ),
        "robodojo_control_mode": str(metadata.get("control_mode", "")),
        "robodojo_piperx_bridge_protocol": str(
            metadata.get("piperx_bridge_protocol", ""),
        ),
        "robodojo_piperx_embodiment_profile": str(
            metadata.get("piperx_embodiment_profile", ""),
        ),
        "robodojo_piperx_bridge_commit": str(
            metadata.get("piperx_bridge_commit", "unknown"),
        ),
        "robodojo_piperx_bridge_dirty": boolean_or_unknown("piperx_bridge_dirty"),
        "robodojo_piperx_control_topology": str(
            metadata.get("piperx_control_topology", ""),
        ),
        "robodojo_piperx_restore_direct_control": bool(
            metadata.get("piperx_restore_direct_control", False),
        ),
        "robodojo_hardware_embodiment": str(
            metadata.get("hardware_embodiment", ""),
        ),
        "robodojo_hardware_profile": str(metadata.get("hardware_profile", "")),
        "robodojo_hardware_bridge_protocol": str(
            metadata.get("hardware_bridge_protocol", ""),
        ),
        "robodojo_hardware_bridge_commit": str(
            metadata.get("hardware_bridge_commit", "unknown"),
        ),
        "robodojo_hardware_bridge_dirty": boolean_or_unknown(
            "hardware_bridge_dirty"
        ),
        "robodojo_hardware_control_topology": str(
            metadata.get("hardware_control_topology", ""),
        ),
        "robodojo_policy_provenance": dict(policy_provenance),
        "robodojo_recovery_source": dict(recovery_source),
        "robodojo_layout_id": integer("layout_id"),
        "robodojo_layout_cycle": integer("layout_cycle", 0),
        "robodojo_eval_seed": integer("eval_seed"),
        "robodojo_run_id": str(metadata.get("run_id", "")),
        "robodojo_commit": str(metadata.get("robodojo_commit", "unknown")),
        "xpolicylab_commit": str(metadata.get("xpolicylab_commit", "unknown")),
        "robodojo_success": bool(success),
        "robodojo_finish_reason": str(reason),
        "robodojo_has_intervention": bool(has_intervention),
        "robodojo_frame_count": int(frame_count),
        "robodojo_timing_resample": str(metadata.get("timing_resample", "none")),
        "robodojo_source_frame_count": integer("source_frame_count", frame_count),
        "robodojo_manual_source_frame_count": integer(
            "manual_source_frame_count", 0
        ),
        "robodojo_manual_output_frame_count": integer(
            "manual_output_frame_count", 0
        ),
        "robodojo_max_manual_source_gap_s": float(
            metadata.get("max_manual_source_gap_s", 0.0)
        ),
        # Kept in RoboDojo's sidecar rather than as a LeRobot frame feature so
        # timing-v2 episodes remain schema-compatible with rollout datasets.
        "robodojo_source_wall_elapsed_s": float_list("source_wall_elapsed_s"),
        "robodojo_source_is_manual": bool_list("source_is_manual"),
    }


def _write_robodojo_episode_metadata(
    dataset_root: str | Path,
    episode_index: int,
    metadata: dict[str, Any],
) -> Path:
    """Persist RoboDojo-only fields without relying on a LeRobot-version API.

    LeRobot 0.4.4, which is pinned by Pi_05/openpi, accepts only
    ``episode_data`` and ``parallel_encoding`` in ``save_episode``.  Newer
    versions added an ``extra_episode_metadata`` keyword.  Keep the dataset
    writer on the public 0.4.4 API and store our additional, non-training
    fields under ``meta/robodojo`` instead.  The file is written after
    ``save_episode`` but inside the same rollback boundary.
    """

    root = Path(dataset_root).expanduser().resolve(strict=True)
    metadata_dir = root
    for component in _ROBODOJO_EPISODE_METADATA_DIR.parts:
        metadata_dir = metadata_dir / component
        try:
            path_stat = metadata_dir.lstat()
        except FileNotFoundError:
            try:
                metadata_dir.mkdir(mode=0o755)
            except FileExistsError:
                # Another filesystem actor may have created the path between
                # lstat and mkdir.  Validate exactly what appeared below.
                pass
            path_stat = metadata_dir.lstat()
        if not stat.S_ISDIR(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
            raise EpisodeCommitError(
                f"RoboDojo metadata path must be a real directory, not a symlink or file: {metadata_dir}"
            )
    if metadata_dir.resolve(strict=True) != metadata_dir:
        raise EpisodeCommitError(f"RoboDojo metadata path escapes the dataset: {metadata_dir}")

    target = metadata_dir / f"episode_{int(episode_index):07d}.json"
    if os.path.lexists(target):
        raise EpisodeCommitError(f"RoboDojo episode metadata already exists: {target}")

    payload = {
        "format_version": 1,
        "episode_index": int(episode_index),
        **dict(metadata),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=metadata_dir,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard-link publish is atomic and, unlike os.replace, can never
        # overwrite a stale episode file that appeared after the check above.
        os.link(temporary, target, follow_symlinks=False)
        temporary.unlink()
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(metadata_dir, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _validate_resume_timing_contract(
    dataset_root: Path,
    *,
    total_episodes: int,
    expected: str,
) -> None:
    """Refuse to mix incompatible time semantics in one training dataset."""

    if total_episodes <= 0:
        return
    sidecar_dir = dataset_root / _ROBODOJO_EPISODE_METADATA_DIR
    sidecars = sorted(sidecar_dir.glob("episode_*.json"))
    if len(sidecars) != total_episodes:
        raise ValueError(
            "existing dataset cannot prove one timing contract per episode: "
            f"episodes={total_episodes}, sidecars={len(sidecars)}"
        )
    for path in sidecars:
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = payload.get("robodojo_timing_resample")
        if actual != expected:
            raise ValueError(
                f"refusing to mix timing contracts in {dataset_root}: "
                f"{path.name} has {actual!r}, requested {expected!r}. "
                "Use a new --dataset-id."
            )


def _clear_episode(dataset: Any) -> None:
    clear = getattr(dataset, "clear_episode_buffer", None)
    if callable(clear):
        clear(delete_images=True)


def _normalise_singleton_numeric_episode_features(dataset: Any) -> None:
    """Make LeRobot 0.4.4 singleton features commit correctly on NumPy 2."""

    episode_buffer = getattr(dataset, "episode_buffer", None)
    features = getattr(dataset, "features", None)
    if not isinstance(episode_buffer, dict) or not isinstance(features, dict):
        return
    for key, feature in features.items():
        if not isinstance(feature, dict) or tuple(feature.get("shape", ())) != (1,):
            continue
        try:
            dtype = np.dtype(feature.get("dtype"))
        except (TypeError, ValueError):
            continue
        if dtype.kind not in "biuf":
            continue
        values = episode_buffer.get(key)
        if not isinstance(values, list) or not values:
            continue
        normalised: list[Any] = []
        changed = False
        for value in values:
            if isinstance(value, np.ndarray):
                array = np.asarray(value, dtype=dtype)
                if array.shape != (1,):
                    raise ValueError(
                        f"Buffered singleton feature {key!r} has shape {array.shape}, expected (1,)"
                    )
                normalised.append(array.reshape(())[()])
                changed = True
            else:
                normalised.append(value)
        if changed:
            episode_buffer[key] = normalised


def _finalize_dataset(dataset: Any) -> None:
    stop_image_writer = getattr(dataset, "stop_image_writer", None)
    if callable(stop_image_writer):
        stop_image_writer()
    dataset.finalize()


def _quiesce_failed_commit(dataset: Any) -> list[str]:
    """Best-effort close all writers before filesystem rollback."""

    errors: list[str] = []
    for label, operation in (
        ("clear episode buffer", lambda: _clear_episode(dataset)),
        ("finalize dataset", lambda: _finalize_dataset(dataset)),
    ):
        try:
            operation()
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    return errors


def _control_wall_time(message: dict[str, Any]) -> float:
    value = message.get("control", {}).get("timestamp")
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("X5 timing resampling requires control.timestamp") from exc
    if not math.isfinite(timestamp):
        raise ValueError(f"control.timestamp must be finite, got {timestamp!r}")
    return timestamp


def _is_manual_frame(message: dict[str, Any]) -> bool:
    control = message.get("control", {})
    mask = float(control.get("intervention_mask", 0.0))
    source = str(control.get("action_source", "")).strip().lower()
    return bool(mask) or source in {"human", "safety_hold"}


class _ManualWallTimeResampler:
    """Causally expand slow human samples onto the dataset's fixed-rate grid.

    Policy actions already represent nominal 25 Hz simulator ticks and must not
    be stretched by policy inference or rendering wall time.  Human X5 motion,
    however, is generated in wall time.  Repeat the last complete manual frame
    so that a slow Isaac loop does not compress several hundred milliseconds of
    physical motion into one nominal 40 ms training step.
    """

    def __init__(self, fps: int, emit: Callable[[dict[str, Any], float], None]):
        self.fps = int(fps)
        self.period_s = 1.0 / float(self.fps)
        self.emit = emit
        self.active = False
        self.segment_start_s = 0.0
        self.segment_output_count = 0
        self.previous_message: dict[str, Any] | None = None
        self.previous_source_s = 0.0
        self.source_count = 0
        self.output_count = 0
        self.max_source_gap_s = 0.0

    def add(self, message: dict[str, Any], timestamp_s: float) -> None:
        timestamp_s = float(timestamp_s)
        self.source_count += 1
        if not self.active:
            self.active = True
            self.segment_start_s = timestamp_s
            self.segment_output_count = 0
            self.previous_message = message
            self.previous_source_s = timestamp_s
            self._emit(message, timestamp_s)
            return
        if timestamp_s < self.previous_source_s:
            raise ValueError(
                "manual control.timestamp moved backwards: "
                f"{timestamp_s:.9f} < {self.previous_source_s:.9f}"
            )
        self.max_source_gap_s = max(
            self.max_source_gap_s,
            timestamp_s - self.previous_source_s,
        )
        desired_count = max(
            self.segment_output_count + 1,
            int(round((timestamp_s - self.segment_start_s) * self.fps)) + 1,
        )
        assert self.previous_message is not None
        while self.segment_output_count + 1 < desired_count:
            grid_s = self.segment_start_s + self.segment_output_count * self.period_s
            self._emit(self.previous_message, grid_s)
        self.previous_message = message
        self.previous_source_s = timestamp_s
        self._emit(message, timestamp_s)

    def finish_before(self, timestamp_s: float) -> None:
        if not self.active:
            return
        timestamp_s = max(float(timestamp_s), self.previous_source_s)
        self.max_source_gap_s = max(
            self.max_source_gap_s,
            timestamp_s - self.previous_source_s,
        )
        # The next policy frame (or the episode boundary) owns timestamp_s.
        # Hold the latest human target only on grid points strictly before it.
        desired_count = max(
            self.segment_output_count,
            int(math.ceil((timestamp_s - self.segment_start_s) * self.fps - 1e-9)),
        )
        assert self.previous_message is not None
        while self.segment_output_count < desired_count:
            grid_s = self.segment_start_s + self.segment_output_count * self.period_s
            self._emit(self.previous_message, grid_s)
        self.active = False
        self.previous_message = None

    def _emit(self, message: dict[str, Any], timestamp_s: float) -> None:
        self.emit(message, timestamp_s)
        self.segment_output_count += 1
        self.output_count += 1


def serve(
    config: WriterConfig,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    *,
    dataset_opener: Callable[[WriterConfig], tuple[Any, bool]] = open_dataset,
) -> int:
    dataset = None
    active = False
    frame_count = 0
    source_frame_count = 0
    has_intervention = False
    metadata: dict[str, Any] = {}
    source_wall_origin_s: float | None = None
    source_wall_elapsed_s: list[float] = []
    source_is_manual: list[bool] = []
    manual_source_frame_count = 0
    previous_manual_wall_s: float | None = None
    max_manual_source_gap_s = 0.0
    manual_resampler: _ManualWallTimeResampler | None = None
    timing_contract = "manual_wall_time_zoh_25hz_v1"
    try:
        dataset, _created = dataset_opener(config)
        total_episodes = int(getattr(dataset.meta, "total_episodes", 0))
        send_message(
            output_stream,
            {
                "status": "ready",
                "dataset_root": str(config.dataset_root),
                "total_episodes": total_episodes,
            },
        )

        def emit_frame(
            source_message: dict[str, Any],
            _wall_timestamp_s: float,
        ) -> None:
            nonlocal frame_count
            # A single low-rate source sample may expand into several 25 Hz
            # rows.  Apply encoder backpressure before every row so a long gap
            # cannot overrun the video queues before the command-level ACK.
            _wait_for_encoder_headroom(dataset, config.encoder_wait_s)
            frame = build_frame(source_message)
            dataset.add_frame(frame)
            drops = _dropped_frame_counts(dataset)
            if drops:
                raise RuntimeError(f"LeRobot video encoder dropped frame(s): {drops}")
            frame_count += 1

        while True:
            try:
                message = receive_message(input_stream)
            except EOFError:
                break
            command = message.get("command") if isinstance(message, dict) else None

            if command == "begin":
                if active:
                    raise RuntimeError("Received begin while an episode is already active")
                metadata = dict(message.get("metadata", {}))
                frame_count = 0
                source_frame_count = 0
                has_intervention = False
                source_wall_origin_s = None
                source_wall_elapsed_s = []
                source_is_manual = []
                manual_source_frame_count = 0
                previous_manual_wall_s = None
                max_manual_source_gap_s = 0.0
                timing_contract = str(
                    metadata.get(
                        "timing_contract",
                        "manual_wall_time_zoh_25hz_v1",
                    )
                )
                if timing_contract not in {
                    "manual_wall_time_zoh_25hz_v1",
                    "sim_step_exact_25hz_v1",
                }:
                    raise ValueError(
                        f"unsupported LeRobot timing contract: {timing_contract!r}"
                    )
                if timing_contract == "sim_step_exact_25hz_v1":
                    _validate_resume_timing_contract(
                        config.dataset_root,
                        total_episodes=int(getattr(dataset.meta, "total_episodes", 0)),
                        expected=timing_contract,
                    )
                manual_resampler = (
                    _ManualWallTimeResampler(config.fps, emit_frame)
                    if metadata.get("control_mode") == "x5_policy_joint_intervention"
                    and timing_contract == "manual_wall_time_zoh_25hz_v1"
                    else None
                )
                active = True
                send_message(output_stream, {"status": "begun"})
            elif command == "frame":
                if not active:
                    raise RuntimeError("Received frame without begin")
                source_frame_count += 1
                is_manual = _is_manual_frame(message)
                manual_source_frame_count += int(is_manual)
                has_intervention = has_intervention or is_manual
                if manual_resampler is not None:
                    wall_timestamp_s = _control_wall_time(message)
                    if source_wall_origin_s is None:
                        source_wall_origin_s = wall_timestamp_s
                    source_wall_elapsed_s.append(
                        max(0.0, wall_timestamp_s - source_wall_origin_s)
                    )
                    source_is_manual.append(is_manual)
                    if is_manual:
                        manual_resampler.add(message, wall_timestamp_s)
                    else:
                        manual_end_timestamp = message.get("control", {}).get(
                            "manual_end_timestamp"
                        )
                        if manual_resampler.active and manual_end_timestamp is None:
                            raise ValueError(
                                "first policy frame after X5 intervention is missing "
                                "manual_end_timestamp"
                            )
                        if manual_end_timestamp is None:
                            manual_end_timestamp = wall_timestamp_s
                        try:
                            manual_end_timestamp = float(manual_end_timestamp)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                "manual_end_timestamp must be a finite timestamp"
                            ) from exc
                        if not math.isfinite(manual_end_timestamp):
                            raise ValueError(
                                "manual_end_timestamp must be a finite timestamp"
                            )
                        if manual_end_timestamp > wall_timestamp_s:
                            raise ValueError(
                                "manual_end_timestamp cannot be after the policy frame timestamp"
                            )
                        manual_resampler.finish_before(manual_end_timestamp)
                        emit_frame(message, wall_timestamp_s)
                else:
                    if timing_contract == "sim_step_exact_25hz_v1":
                        wall_timestamp_s = _control_wall_time(message)
                        if source_wall_origin_s is None:
                            source_wall_origin_s = wall_timestamp_s
                        source_wall_elapsed_s.append(
                            max(0.0, wall_timestamp_s - source_wall_origin_s)
                        )
                        source_is_manual.append(is_manual)
                        if is_manual:
                            if previous_manual_wall_s is not None:
                                max_manual_source_gap_s = max(
                                    max_manual_source_gap_s,
                                    wall_timestamp_s - previous_manual_wall_s,
                                )
                            previous_manual_wall_s = wall_timestamp_s
                        else:
                            previous_manual_wall_s = None
                    emit_frame(message, 0.0)
                _wait_for_encoder_headroom(dataset, config.encoder_wait_s)
                drops = _dropped_frame_counts(dataset)
                if drops:
                    raise RuntimeError(f"LeRobot video encoder dropped frame(s): {drops}")
                # Per-frame ACK propagates video encoder pressure through the
                # bounded OS pipe all the way to the simulation loop.
                send_message(output_stream, {"status": "frame", "frame_count": frame_count})
            elif command == "finish":
                if not active:
                    raise RuntimeError("Received finish without begin")
                accepted = bool(message.get("accepted", False))
                success = bool(message.get("success", False))
                reason = str(message.get("reason", "operator"))
                if not accepted:
                    _clear_episode(dataset)
                    active = False
                    send_message(output_stream, {"status": "discarded", "frame_count": frame_count})
                    continue
                if manual_resampler is not None and manual_resampler.active:
                    terminal_timestamp = message.get("timestamp")
                    if terminal_timestamp is None:
                        terminal_timestamp = (
                            manual_resampler.previous_source_s
                            + manual_resampler.period_s
                        )
                    manual_resampler.finish_before(float(terminal_timestamp))
                if frame_count <= 0:
                    raise RuntimeError("Refusing to commit an empty LeRobot episode")
                drops = _dropped_frame_counts(dataset)
                if drops:
                    raise RuntimeError(f"Refusing commit after video frame drop(s): {drops}")
                episode_index = int(getattr(dataset.meta, "total_episodes", 0))
                metadata["source_frame_count"] = source_frame_count
                if manual_resampler is not None:
                    metadata["timing_resample"] = "manual_wall_time_zoh_25hz_v1"
                    metadata["manual_source_frame_count"] = manual_resampler.source_count
                    metadata["manual_output_frame_count"] = manual_resampler.output_count
                    metadata["max_manual_source_gap_s"] = (
                        manual_resampler.max_source_gap_s
                    )
                    metadata["source_wall_elapsed_s"] = source_wall_elapsed_s
                    metadata["source_is_manual"] = source_is_manual
                    print(
                        "[LEROBOT timing] manual source="
                        f"{manual_resampler.source_count} -> 25Hz output="
                        f"{manual_resampler.output_count}; max source gap="
                        f"{manual_resampler.max_source_gap_s * 1000.0:.0f}ms",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    if timing_contract == "sim_step_exact_25hz_v1":
                        if frame_count != source_frame_count:
                            raise RuntimeError(
                                "sim-step timing contract requires exactly one output row "
                                f"per source transition, got source={source_frame_count}, "
                                f"output={frame_count}"
                            )
                        metadata["timing_resample"] = "sim_step_exact_25hz_v1"
                        metadata["manual_source_frame_count"] = (
                            manual_source_frame_count
                        )
                        metadata["manual_output_frame_count"] = (
                            manual_source_frame_count
                        )
                        metadata["max_manual_source_gap_s"] = (
                            max_manual_source_gap_s
                        )
                        metadata["source_wall_elapsed_s"] = source_wall_elapsed_s
                        metadata["source_is_manual"] = source_is_manual
                        print(
                            "[LEROBOT timing] sim-step exact 25Hz: source="
                            f"{source_frame_count} output={frame_count} "
                            f"manual={manual_source_frame_count} max-manual-wall-gap="
                            f"{max_manual_source_gap_s * 1000.0:.0f}ms; "
                            "no timing-fill rows",
                            file=sys.stderr,
                            flush=True,
                        )
                commit_snapshot = _CommitSnapshot.capture(
                    config.dataset_root,
                    transient_roots=_encoder_transient_roots(dataset, config.dataset_root),
                )
                dataset_finalized = False
                try:
                    # Pi_05 pins LeRobot 0.4.4, whose public save_episode API
                    # does not accept ``extra_episode_metadata``.  Training
                    # fields are already regular frame features; persist the
                    # remaining RoboDojo provenance in a version-independent
                    # sidecar immediately after the LeRobot commit.
                    _normalise_singleton_numeric_episode_features(dataset)
                    dataset.save_episode()
                    _finalize_dataset(dataset)
                    dataset_finalized = True
                    robodojo_metadata_path = _write_robodojo_episode_metadata(
                        config.dataset_root,
                        episode_index,
                        _episode_metadata(
                            metadata,
                            success=success,
                            reason=reason,
                            has_intervention=has_intervention,
                            frame_count=frame_count,
                        ),
                    )
                    marker = config.dataset_root / _STAGING_MARKER
                    marker.unlink(missing_ok=True)
                except BaseException as commit_exc:
                    # Writers must be closed before any Parquet, video, or
                    # metadata path is removed/restored.
                    active = False
                    cleanup_errors = [] if dataset_finalized else _quiesce_failed_commit(dataset)
                    try:
                        commit_snapshot.rollback()
                    except Exception as rollback_exc:
                        cleanup_errors.append(
                            f"rollback: {type(rollback_exc).__name__}: {rollback_exc}"
                        )
                    dataset = None
                    try:
                        remove_safe_empty_staging(config.dataset_root)
                    except Exception as staging_exc:
                        cleanup_errors.append(
                            f"staging cleanup: {type(staging_exc).__name__}: {staging_exc}"
                        )
                    if cleanup_errors:
                        raise CommitRollbackError(
                            f"{type(commit_exc).__name__}: {commit_exc}; "
                            "failed-commit cleanup was incomplete: "
                            + "; ".join(cleanup_errors)
                            + ". Refusing to resume this dataset automatically."
                        ) from commit_exc
                    if isinstance(commit_exc, EpisodeCommitError):
                        raise commit_exc
                    raise EpisodeCommitError(
                        f"{type(commit_exc).__name__}: {commit_exc}; "
                        "the candidate was rolled back. Refusing to retry the same layout automatically."
                    ) from commit_exc
                active = False
                send_message(
                    output_stream,
                    {
                        "status": "committed",
                        "dataset_root": str(config.dataset_root),
                        "episode_index": episode_index,
                        "frame_count": frame_count,
                        "robodojo_metadata_path": str(robodojo_metadata_path),
                    },
                )
                return 0
            elif command == "shutdown":
                if active:
                    _clear_episode(dataset)
                    active = False
                _finalize_dataset(dataset)
                remove_safe_empty_staging(config.dataset_root)
                send_message(output_stream, {"status": "closed"})
                return 0
            else:
                raise ValueError(f"Unknown LeRobot writer command: {command!r}")

        if active:
            _clear_episode(dataset)
        _finalize_dataset(dataset)
        remove_safe_empty_staging(config.dataset_root)
        return 0
    except Exception as exc:
        if dataset is not None:
            try:
                if active:
                    _clear_episode(dataset)
                _finalize_dataset(dataset)
            except Exception as cleanup_exc:
                print(f"[LEROBOT][CLEANUP ERROR] {cleanup_exc}", file=sys.stderr, flush=True)
            remove_safe_empty_staging(config.dataset_root)
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        print(f"[LEROBOT][ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        try:
            send_message(
                output_stream,
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "fatal": isinstance(exc, EpisodeCommitError),
                },
            )
        except Exception:
            pass
        return 1


def _parse_args() -> WriterConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--encoder-queue-maxsize", type=int, default=128)
    parser.add_argument("--video-crf", type=int, default=18)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.encoder_threads <= 0:
        parser.error("--encoder-threads must be positive")
    if args.encoder_queue_maxsize <= 0:
        parser.error("--encoder-queue-maxsize must be positive")
    return WriterConfig(
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        resume=args.resume,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        video_crf=args.video_crf,
    )


def main() -> int:
    # Defense in depth: the parent also sets this before process creation.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    protocol_input = sys.stdin.buffer
    # Keep the binary protocol on a duplicated descriptor, then redirect the
    # real fd 1 as well as Python's stdout.  This also catches native FFmpeg,
    # PyAV, or C-extension writes that bypass ``sys.stdout``.
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        return serve(_parse_args(), protocol_input, protocol_output)
    finally:
        protocol_output.close()


if __name__ == "__main__":
    raise SystemExit(main())
