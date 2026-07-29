"""Small dependency-light writer for the LeRobot v3.0 video layout.

RoboDojo's collection environment deliberately does not install the full
``lerobot`` recording stack (``datasets`` and PyAV).  The released data only
needs a compact, well-defined subset of that stack: Parquet rows for numeric
features, AV1 MP4 files for cameras, and the three metadata files under
``meta``.  This module writes that subset without changing the environment.

Videos are kept one episode per MP4.  This is a valid v3.0 layout; episode
metadata gives every camera video a distinct file index and a [0, duration]
time span.  It is more robust for recovery collection because an interrupted
episode cannot corrupt video frames belonging to earlier episodes.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


CAMERA_FEATURES = {
    "cam_high": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}
STATE_NAMES = [
    "l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
    "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g",
]
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def _to_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got {array.shape}.")
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


def _vector_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    count = int(values.shape[0])
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [count] * int(values.shape[1]),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


@dataclass
class _ImageMoments:
    """Streaming RGB statistics; avoids retaining video frames in RAM."""

    minimum: np.ndarray = field(default_factory=lambda: np.full(3, np.inf, dtype=np.float64))
    maximum: np.ndarray = field(default_factory=lambda: np.full(3, -np.inf, dtype=np.float64))
    total: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    square_total: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    count: int = 0

    def update(self, image: np.ndarray) -> None:
        pixels = _to_rgb(image).reshape(-1, 3).astype(np.float64) / 255.0
        self.minimum = np.minimum(self.minimum, pixels.min(axis=0))
        self.maximum = np.maximum(self.maximum, pixels.max(axis=0))
        self.total += pixels.sum(axis=0)
        self.square_total += np.square(pixels).sum(axis=0)
        self.count += int(pixels.shape[0])

    def to_stats(self) -> dict[str, list]:
        if self.count == 0:
            raise RuntimeError("No RGB pixels were recorded.")
        mean = self.total / self.count
        std = np.sqrt(np.maximum(self.square_total / self.count - np.square(mean), 0.0))
        # v3's image quantiles are informational metadata.  Computing exact
        # video-wide image quantiles would require retaining every pixel, so
        # expose conservative bounds/central estimates with the same shape.
        shaped = lambda value: [[[float(v)]] for v in value]
        return {
            "min": shaped(self.minimum),
            "max": shaped(self.maximum),
            "mean": shaped(mean),
            "std": shaped(std),
            "count": [self.count] * 3,
            "q01": shaped(self.minimum),
            "q10": shaped(self.minimum),
            "q50": shaped(mean),
            "q90": shaped(self.maximum),
            "q99": shaped(self.maximum),
        }


@dataclass
class LeRobotEpisode:
    """Temporary frame store for one episode prior to AV1 encoding."""

    writer: "LeRobotV30Writer"
    episode_index: int
    source: dict[str, Any]
    frames_dir: Path
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    task_indices: list[int] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    image_stats: dict[str, _ImageMoments] = field(default_factory=dict)

    def add_frame(
        self,
        *,
        state: Iterable[float],
        action: Iterable[float],
        images: dict[str, np.ndarray],
        task: str,
    ) -> None:
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if state_array.shape != (16,) or action_array.shape != (16,):
            raise ValueError(f"state/action must both be 16-D, got {state_array.shape}/{action_array.shape}.")
        if task not in self.writer.task_to_index:
            raise ValueError(f"Task {task!r} is not registered in the dataset writer.")
        frame_index = len(self.states)
        for camera, feature in CAMERA_FEATURES.items():
            if camera not in images:
                raise ValueError(f"Frame is missing {camera}; available={sorted(images)}")
            image = _to_rgb(images[camera])
            if self.writer.image_shape is None:
                self.writer.image_shape = image.shape
            elif image.shape != self.writer.image_shape:
                raise ValueError(f"Image shape changed from {self.writer.image_shape} to {image.shape}.")
            path = self.frames_dir / camera / f"frame-{frame_index:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(path, format="PNG", compress_level=1)
            self.image_stats.setdefault(camera, _ImageMoments()).update(image)

        self.states.append(state_array)
        self.actions.append(action_array)
        self.task_indices.append(self.writer.task_to_index[task])
        self.tasks.append(task)

    @property
    def length(self) -> int:
        return len(self.states)


class LeRobotV30Writer:
    """Write recovery episodes in the released RoboDojo LeRobot v3 layout."""

    def __init__(self, root: Path, *, fps: int, tasks: list[str], overwrite: bool = False):
        self.root = Path(root)
        self.fps = int(fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if len(tasks) != len(set(tasks)):
            raise ValueError("Task prompts must be unique.")
        self.task_to_index = {task: index for index, task in enumerate(tasks)}
        self.tasks = list(tasks)
        self.image_shape: tuple[int, int, int] | None = None
        self.rows: list[dict[str, Any]] = []
        self.episodes: list[dict[str, Any]] = []
        self.manifest: list[dict[str, Any]] = []
        self.global_states: list[np.ndarray] = []
        self.global_actions: list[np.ndarray] = []
        self.global_timestamps: list[float] = []
        self.global_frame_indices: list[int] = []
        self.global_episode_indices: list[int] = []
        self.global_task_indices: list[int] = []
        self.global_image_stats = {camera: _ImageMoments() for camera in CAMERA_FEATURES}

        if self.root.exists():
            if not overwrite:
                raise FileExistsError(f"Output directory already exists: {self.root}. Use --overwrite to replace it.")
            shutil.rmtree(self.root)
        (self.root / ".frames").mkdir(parents=True, exist_ok=False)

    def start_episode(self, episode_index: int, *, source: dict[str, Any]) -> LeRobotEpisode:
        frame_dir = self.root / ".frames" / f"episode-{episode_index:06d}"
        frame_dir.mkdir(parents=True, exist_ok=False)
        return LeRobotEpisode(self, episode_index, source, frame_dir)

    def abort_episode(self, episode: LeRobotEpisode) -> None:
        shutil.rmtree(episode.frames_dir, ignore_errors=True)

    def _encode_video(self, episode: LeRobotEpisode, camera: str) -> Path:
        output = self.root / "videos" / CAMERA_FEATURES[camera] / "chunk-000" / f"file-{episode.episode_index:03d}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(self.fps),
            "-start_number", "0",
            "-i", str(episode.frames_dir / camera / "frame-%06d.png"),
            "-an", "-c:v", "libaom-av1", "-cpu-used", "8", "-row-mt", "1",
            "-crf", "30", "-b:v", "0", "-pix_fmt", "yuv420p", str(output),
        ]
        subprocess.run(command, check=True)
        return output

    def _episode_stats(self, episode: LeRobotEpisode) -> dict[str, dict[str, list]]:
        states = np.stack(episode.states)
        actions = np.stack(episode.actions)
        timestamps = np.arange(episode.length, dtype=np.float32) / self.fps
        frame_indices = np.arange(episode.length, dtype=np.int64)
        episode_indices = np.full(episode.length, episode.episode_index, dtype=np.int64)
        task_indices = np.asarray(episode.task_indices, dtype=np.int64)
        stats = {
            "observation.state": _vector_stats(states),
            "action": _vector_stats(actions),
            "timestamp": _vector_stats(timestamps),
            "frame_index": _vector_stats(frame_indices),
            "episode_index": _vector_stats(episode_indices),
            "index": _vector_stats(frame_indices),
            "task_index": _vector_stats(task_indices),
        }
        for camera, moments in episode.image_stats.items():
            stats[CAMERA_FEATURES[camera]] = moments.to_stats()
        return stats

    def commit_episode(self, episode: LeRobotEpisode) -> None:
        if episode.length == 0:
            raise ValueError("Cannot commit an empty episode.")
        for camera in CAMERA_FEATURES:
            self._encode_video(episode, camera)

        first_index = len(self.rows)
        for frame_index, (state, action, task_index) in enumerate(
            zip(episode.states, episode.actions, episode.task_indices, strict=True)
        ):
            timestamp = frame_index / self.fps
            self.rows.append(
                {
                    "observation.state": state,
                    "action": action,
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "episode_index": episode.episode_index,
                    "index": first_index + frame_index,
                    "task_index": task_index,
                }
            )
            self.global_states.append(state)
            self.global_actions.append(action)
            self.global_timestamps.append(timestamp)
            self.global_frame_indices.append(frame_index)
            self.global_episode_indices.append(episode.episode_index)
            self.global_task_indices.append(task_index)
        for camera, moments in episode.image_stats.items():
            # Image statistics are composable.  Feed frame-level images only
            # through local moments, then merge their sufficient statistics.
            global_moments = self.global_image_stats[camera]
            global_moments.minimum = np.minimum(global_moments.minimum, moments.minimum)
            global_moments.maximum = np.maximum(global_moments.maximum, moments.maximum)
            global_moments.total += moments.total
            global_moments.square_total += moments.square_total
            global_moments.count += moments.count

        episode_stats = self._episode_stats(episode)
        episode_row: dict[str, Any] = {
            "episode_index": episode.episode_index,
            "tasks": list(dict.fromkeys(episode.tasks)),
            "length": episode.length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": first_index,
            "dataset_to_index": first_index + episode.length,
        }
        for camera, feature in CAMERA_FEATURES.items():
            prefix = f"videos/{feature}"
            episode_row.update(
                {
                    f"{prefix}/chunk_index": 0,
                    f"{prefix}/file_index": episode.episode_index,
                    f"{prefix}/from_timestamp": 0.0,
                    f"{prefix}/to_timestamp": episode.length / self.fps,
                }
            )
        for feature, stat in episode_stats.items():
            for key in STAT_KEYS:
                episode_row[f"stats/{feature}/{key}"] = stat[key]
        self.episodes.append(episode_row)
        self.manifest.append({"episode_index": episode.episode_index, "length": episode.length, **episode.source})
        shutil.rmtree(episode.frames_dir)

    def _info(self) -> dict[str, Any]:
        if self.image_shape is None:
            raise RuntimeError("Cannot finalize without at least one frame.")
        height, width, channels = self.image_shape
        if channels != 3:
            raise RuntimeError(f"Expected RGB images, got {self.image_shape}.")
        vector_feature = {"dtype": "float32", "shape": [16], "names": [STATE_NAMES]}
        video_info = {
            "video.height": height,
            "video.width": width,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": self.fps,
            "video.channels": 3,
            "has_audio": False,
        }
        features: dict[str, Any] = {
            "observation.state": vector_feature,
            "action": vector_feature,
        }
        for feature in CAMERA_FEATURES.values():
            features[feature] = {
                "dtype": "video", "shape": [3, height, width],
                "names": ["channels", "height", "width"], "info": video_info,
            }
        features.update(
            {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            }
        )
        return {
            "codebase_version": "v3.0",
            "robot_type": "unified_robot",
            "total_episodes": len(self.episodes),
            "total_frames": len(self.rows),
            "total_tasks": len(self.tasks),
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self.episodes)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
        }

    def _global_stats(self) -> dict[str, dict[str, list]]:
        if not self.rows:
            raise RuntimeError("Cannot write metadata for an empty dataset.")
        stats = {
            "observation.state": _vector_stats(np.stack(self.global_states)),
            "action": _vector_stats(np.stack(self.global_actions)),
            "timestamp": _vector_stats(np.asarray(self.global_timestamps, dtype=np.float32)),
            "frame_index": _vector_stats(np.asarray(self.global_frame_indices, dtype=np.int64)),
            "episode_index": _vector_stats(np.asarray(self.global_episode_indices, dtype=np.int64)),
            "index": _vector_stats(np.arange(len(self.rows), dtype=np.int64)),
            "task_index": _vector_stats(np.asarray(self.global_task_indices, dtype=np.int64)),
        }
        for camera, moments in self.global_image_stats.items():
            stats[CAMERA_FEATURES[camera]] = moments.to_stats()
        return stats

    def _write_data(self) -> None:
        vectors = lambda name: pa.FixedSizeListArray.from_arrays(
            pa.array(np.concatenate([row[name] for row in self.rows]).astype(np.float32)), 16
        )
        table = pa.Table.from_arrays(
            [
                vectors("observation.state"),
                vectors("action"),
                pa.array([row["timestamp"] for row in self.rows], type=pa.float32()),
                pa.array([row["frame_index"] for row in self.rows], type=pa.int64()),
                pa.array([row["episode_index"] for row in self.rows], type=pa.int64()),
                pa.array([row["index"] for row in self.rows], type=pa.int64()),
                pa.array([row["task_index"] for row in self.rows], type=pa.int64()),
            ],
            names=["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"],
        )
        path = self.root / "data" / "chunk-000" / "file-000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="snappy", row_group_size=None)

    def finalize(self, *, extra_manifest: dict[str, Any] | None = None) -> None:
        self._write_data()
        episode_path = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(self.episodes), episode_path, compression="snappy")
        pq.write_table(
            pa.Table.from_pydict({"task_index": list(range(len(self.tasks))), "__index_level_0__": self.tasks}),
            self.root / "meta" / "tasks.parquet",
            compression="snappy",
        )
        (self.root / "meta" / "info.json").write_text(json.dumps(self._info(), indent=2) + "\n", encoding="utf-8")
        (self.root / "meta" / "stats.json").write_text(
            json.dumps(self._global_stats(), indent=2) + "\n", encoding="utf-8"
        )
        manifest = {"episodes": self.manifest}
        if extra_manifest:
            manifest.update(extra_manifest)
        (self.root / "meta" / "recovery_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        shutil.rmtree(self.root / ".frames", ignore_errors=True)
