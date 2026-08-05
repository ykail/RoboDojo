"""Read one frame from a committed RoboDojo rollout replay bundle."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReplayFrame:
    dataset_root: Path
    episode_index: int
    frame_index: int
    timestamp_s: float
    frame_count: int
    task_name: str
    env_config: str
    eval_seed: int
    layout_id: int
    saved_layout: dict[str, Any]
    manifest: dict[str, Any]
    state: dict[str, np.ndarray]


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _dataset_file(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} escapes the dataset: {value!r}")
    path = root / relative
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def load_replay_frame(
    dataset_root: str | Path,
    episode_index: int,
    *,
    frame_index: int | None = None,
    time_s: float | None = None,
) -> ReplayFrame:
    """Load the video-aligned pre-action simulator snapshot for one frame."""

    if episode_index < 0:
        raise ValueError("episode index must be non-negative")
    if (frame_index is None) == (time_s is None):
        raise ValueError("select exactly one of frame_index or time_s")
    if frame_index is not None and frame_index < 0:
        raise ValueError("frame index must be non-negative")
    if time_s is not None and (not np.isfinite(time_s) or time_s < 0):
        raise ValueError("time_s must be finite and non-negative")

    root = Path(dataset_root).expanduser().resolve(strict=True)
    metadata_path = root / "meta" / "robodojo" / "episodes" / f"episode_{episode_index:07d}.json"
    metadata = _read_json(metadata_path, label="episode metadata")
    if int(metadata.get("episode_index", -1)) != episode_index:
        raise ValueError(f"episode metadata index mismatch: {metadata_path}")

    replay = metadata.get("robodojo_replay")
    if not isinstance(replay, dict) or not replay.get("complete"):
        raise ValueError(f"episode has no complete simulator replay: {metadata_path}")
    layout_path = _dataset_file(root, replay.get("layout_path"), label="replay layout")
    state_path = _dataset_file(root, replay.get("state_path"), label="replay state")
    layout = _read_json(layout_path, label="replay layout")
    manifest = layout.get("snapshot_manifest")
    saved_layout = layout.get("saved_layout")
    if not isinstance(manifest, dict) or not isinstance(saved_layout, dict):
        raise ValueError(f"replay layout is missing manifest/saved_layout: {layout_path}")
    if int(manifest.get("format_version", -1)) != 1:
        raise ValueError("unsupported simulator snapshot format")

    try:
        with np.load(state_path, allow_pickle=False) as archive:
            frame_count = int(archive["frame_count"])
            timestamps = np.asarray(archive["frame__frame.timestamp_s"], dtype=np.float64)
            if timestamps.shape != (frame_count,):
                raise ValueError("timestamp array does not match frame_count")
            if time_s is not None:
                selected = int(np.abs(timestamps - float(time_s)).argmin())
            else:
                selected = int(frame_index)
            if not 0 <= selected < frame_count:
                raise IndexError(f"frame {selected} is outside episode range 0..{frame_count - 1}")
            state = {
                key.removeprefix("frame__"): np.array(archive[key][selected], copy=True)
                for key in archive.files
                if key.startswith("frame__")
            }
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"cannot read replay state: {state_path}: {exc}") from exc

    actual_index = int(np.asarray(state["frame.index"]).item())
    if actual_index != selected:
        raise ValueError(f"replay frame identity mismatch: requested {selected}, stored {actual_index}")
    timestamp_s = float(np.asarray(state["frame.timestamp_s"]).item())
    return ReplayFrame(
        dataset_root=root,
        episode_index=episode_index,
        frame_index=selected,
        timestamp_s=timestamp_s,
        frame_count=frame_count,
        task_name=str(metadata.get("robodojo_task", "")),
        env_config=str(metadata.get("robodojo_env_config", "")),
        eval_seed=int(metadata.get("robodojo_eval_seed", -1)),
        layout_id=int(metadata.get("robodojo_layout_id", -1)),
        saved_layout=saved_layout,
        manifest=manifest,
        state=state,
    )
