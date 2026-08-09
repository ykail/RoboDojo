"""Replay a saved human-recovery action segment inside the restored simulator.

This module deliberately has no PiPER-X or LeRobot writer dependency.  The
caller restores the source rollout frame first; this validator then compares
the live pre-action robot observation with every recorded ``observation.state``
and applies the corresponding recorded action for the next comparison.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


_STATE_PARTS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)
_JOINT_INDICES = np.asarray([*range(6), *range(7, 13)], dtype=np.int64)
_GRIPPER_INDICES = np.asarray([6, 13], dtype=np.int64)
_CAMERA_SOURCES = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read recovery metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"recovery metadata must be a JSON object: {path}")
    return payload


def _flatten_state(state: Any, *, label: str) -> np.ndarray:
    if not isinstance(state, dict):
        raise RuntimeError(f"{label} must be a joint-state dictionary")
    parts: list[np.ndarray] = []
    for key, size in _STATE_PARTS:
        value = np.asarray(state.get(key), dtype=np.float64).reshape(-1)
        if value.shape != (size,):
            raise RuntimeError(f"{label}.{key} has shape {value.shape}, expected {(size,)}")
        parts.append(value)
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise RuntimeError(f"{label} contains non-finite values")
    return result


def _action_dict(vector: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    offset = 0
    for key, size in _STATE_PARTS:
        result[key] = np.asarray(vector[offset : offset + size], dtype=np.float64).copy()
        offset += size
    return result


def _load_episode_rows(root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "restore validation needs pyarrow in the RoboDojo environment"
        ) from exc

    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    parquet_paths = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"no LeRobot parquet files found under {root}")
    for path in parquet_paths:
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "observation.state", "action"],
        )
        episodes = table.column("episode_index").combine_chunks().to_pylist()
        frames = table.column("frame_index").combine_chunks().to_pylist()
        states = table.column("observation.state").combine_chunks().to_pylist()
        actions = table.column("action").combine_chunks().to_pylist()
        for stored_episode, frame, state, action in zip(
            episodes, frames, states, actions, strict=True
        ):
            if int(stored_episode) != episode_index:
                continue
            state_array = np.asarray(state, dtype=np.float64).reshape(-1)
            action_array = np.asarray(action, dtype=np.float64).reshape(-1)
            if state_array.shape != (14,) or action_array.shape != (14,):
                raise RuntimeError(
                    f"episode {episode_index} frame {frame} does not contain 14-D state/action"
                )
            if not np.isfinite(state_array).all() or not np.isfinite(action_array).all():
                raise RuntimeError(
                    f"episode {episode_index} frame {frame} contains non-finite state/action"
                )
            rows.append((int(frame), state_array, action_array))

    rows.sort(key=lambda item: item[0])
    if not rows:
        raise RuntimeError(f"episode {episode_index} has no parquet rows under {root}")
    actual_frames = [row[0] for row in rows]
    if actual_frames != list(range(len(rows))):
        raise RuntimeError(
            f"episode {episode_index} frame indices are not contiguous from zero: "
            f"first={actual_frames[0]} last={actual_frames[-1]} rows={len(rows)}"
        )
    return (
        np.stack([row[1] for row in rows]),
        np.stack([row[2] for row in rows]),
    )


def _episode_video_locations(
    root: Path,
    episode_index: int,
) -> dict[str, tuple[Path, float]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "restore validation needs pyarrow in the RoboDojo environment"
        ) from exc

    info = _read_json(root / "meta" / "info.json")
    video_template = info.get("video_path")
    fps = float(info.get("fps", 0.0))
    if not isinstance(video_template, str) or not video_template or fps <= 0.0:
        raise RuntimeError("LeRobot info.json has no valid video_path/fps")

    matches: list[dict[str, Any]] = []
    location_columns = ["episode_index"]
    for camera_name in _CAMERA_SOURCES:
        prefix = f"videos/observation.images.{camera_name}"
        location_columns.extend(
            (
                f"{prefix}/chunk_index",
                f"{prefix}/file_index",
                f"{prefix}/from_timestamp",
            )
        )
    for path in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=location_columns)
        for row in table.to_pylist():
            if int(row.get("episode_index", -1)) == episode_index:
                matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one LeRobot episode-index row for episode {episode_index}, got {len(matches)}"
        )
    row = matches[0]

    result: dict[str, tuple[Path, float]] = {}
    for camera_name in _CAMERA_SOURCES:
        video_key = f"observation.images.{camera_name}"
        prefix = f"videos/{video_key}"
        try:
            chunk_index = int(row[f"{prefix}/chunk_index"])
            file_index = int(row[f"{prefix}/file_index"])
            from_timestamp = float(row[f"{prefix}/from_timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"episode index is missing {camera_name} video location") from exc
        relative = Path(
            video_template.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
        )
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"invalid {camera_name} video path: {relative}")
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"{camera_name} video does not exist: {path}")
        result[camera_name] = (path, from_timestamp * fps)
    return result


def _load_video_frames(
    root: Path,
    episode_index: int,
    frame_count: int,
    *,
    episode_frame_start: int = 0,
) -> dict[str, list[np.ndarray]]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("restore validation needs PyAV in the RoboDojo environment") from exc

    if episode_frame_start < 0:
        raise RuntimeError("video episode_frame_start must be non-negative")
    result: dict[str, list[np.ndarray]] = {}
    for camera_name, (path, start_frame_float) in _episode_video_locations(
        root, episode_index
    ).items():
        start_frame = int(round(start_frame_float)) + episode_frame_start
        stop_frame = start_frame + frame_count
        frames: list[np.ndarray] = []
        with av.open(str(path), mode="r") as container:
            for decoded_index, frame in enumerate(container.decode(video=0)):
                if decoded_index < start_frame:
                    continue
                if decoded_index >= stop_frame:
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) != frame_count:
            raise RuntimeError(
                f"{camera_name} decoded {len(frames)} frame(s), expected {frame_count}"
            )
        result[camera_name] = frames
    return result


def _live_image(obs: dict[str, Any], source_name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    camera = obs.get("vision", {}).get(source_name)
    value = camera.get("color") if isinstance(camera, dict) else camera
    if value is None:
        raise RuntimeError(f"live observation is missing vision.{source_name}.color")
    image = np.asarray(value)
    if image.ndim != 3:
        raise RuntimeError(f"live vision.{source_name} has invalid shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise RuntimeError(f"live vision.{source_name} contains non-finite pixels")
        if image.size and float(np.max(image)) <= 1.0:
            image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape != expected_shape:
        raise RuntimeError(
            f"live vision.{source_name} has shape {image.shape}, expected {expected_shape}"
        )
    return np.ascontiguousarray(image)


def _pixel_metrics(live: np.ndarray, expected: np.ndarray) -> tuple[float, float, float]:
    error = live.astype(np.float32) - expected.astype(np.float32)
    mad = float(np.mean(np.abs(error)))
    mse = float(np.mean(np.square(error)))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10((255.0 * 255.0) / mse)
    return mad, mse, psnr


def _check_lineage(task_env: Any, metadata: dict[str, Any]) -> None:
    expected = metadata.get("robodojo_recovery_source")
    actual = getattr(task_env, "restore_lineage", None)
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise RuntimeError("recovery source lineage is unavailable")

    expected_root = Path(str(expected.get("dataset_root", ""))).expanduser().resolve()
    actual_root = Path(str(actual.get("dataset_root", ""))).expanduser().resolve()
    mismatches: list[str] = []
    if expected_root != actual_root:
        mismatches.append(f"dataset_root expected={expected_root} actual={actual_root}")
    for key in ("episode", "frame", "layout_id", "eval_seed", "frame_count"):
        if int(expected.get(key, -1)) != int(actual.get(key, -2)):
            mismatches.append(f"{key} expected={expected.get(key)!r} actual={actual.get(key)!r}")
    expected_time = float(expected.get("time_s", math.nan))
    actual_time = float(actual.get("time_s", math.nan))
    if not math.isfinite(expected_time) or not math.isfinite(actual_time) or not math.isclose(
        expected_time, actual_time, abs_tol=1.0e-6
    ):
        mismatches.append(f"time_s expected={expected_time!r} actual={actual_time!r}")
    if mismatches:
        raise RuntimeError("restored source does not match recovery lineage: " + "; ".join(mismatches))


def run_restore_recovery_validation(
    task_env: Any,
    *,
    dataset_root: str | Path,
    episode_index: int,
    max_frames: int,
    compare_start_frame: int = 0,
) -> str:
    """Validate the restored start state and a short recorded action sequence."""

    if int(getattr(task_env, "num_envs", 0)) != 1:
        raise RuntimeError("restore validation requires exactly one Isaac environment")
    if episode_index < 0 or max_frames < 1 or compare_start_frame < 0:
        raise RuntimeError("validation episode must be non-negative and max_frames must be positive")

    root = Path(dataset_root).expanduser().resolve(strict=True)
    metadata_path = (
        root / "meta" / "robodojo" / "episodes" / f"episode_{episode_index:07d}.json"
    )
    metadata = _read_json(metadata_path)
    if int(metadata.get("episode_index", -1)) != episode_index:
        raise RuntimeError(f"episode metadata index mismatch: {metadata_path}")
    _check_lineage(task_env, metadata)
    source = metadata.get("robodojo_recovery_source")
    if not isinstance(source, dict):
        raise RuntimeError("recovery metadata has no source lineage")
    source_root = Path(str(source.get("dataset_root", ""))).expanduser().resolve(strict=True)
    source_episode = int(source.get("episode", -1))
    source_frame = int(source.get("frame", -1))
    source_frame_count = int(source.get("frame_count", -1))
    if source_episode < 0 or source_frame < 0 or source_frame_count <= source_frame:
        raise RuntimeError("recovery source episode/frame is invalid")
    # Match the action interpolation mode used during collection. Episodes
    # written before direct recovery control do not carry this field and keep
    # the original 80%-ramp behavior.
    task_env.piperx_intervention_occurred = bool(
        metadata.get("robodojo_piperx_restore_direct_control", False)
    )
    expected_states, actions = _load_episode_rows(root, episode_index)
    frame_count = min(max_frames, len(expected_states))
    expected_images = _load_video_frames(root, episode_index, frame_count)
    source_window_start = max(0, source_frame - 3)
    source_window_stop = min(source_frame_count, source_frame + 4)
    source_window_count = source_window_stop - source_window_start
    # The source rollout's tiled RGB annotators publish one frame after the
    # corresponding simulator snapshot.  A restored state N therefore renders
    # the source video image N+1; this offset is measured below as well as
    # applied to the strict visual truth check.
    source_video_frame = min(source_frame + 1, source_frame_count - 1)
    source_truth_images = _load_video_frames(
        source_root,
        source_episode,
        source_window_count,
        episode_frame_start=source_window_start,
    )

    joint_tolerance_rad = float(
        os.environ.get("ROBODOJO_RECOVERY_VALIDATION_JOINT_TOL_RAD", "0.01")
    )
    gripper_tolerance = float(
        os.environ.get("ROBODOJO_RECOVERY_VALIDATION_GRIPPER_TOL", "0.01")
    )
    image_mad_tolerance = float(
        os.environ.get("ROBODOJO_RECOVERY_VALIDATION_IMAGE_MAD_TOL", "8")
    )
    source_image_mad_tolerance = float(
        os.environ.get(
            "ROBODOJO_RECOVERY_VALIDATION_SOURCE_IMAGE_MAD_TOL",
            str(image_mad_tolerance),
        )
    )
    if (
        joint_tolerance_rad <= 0.0
        or gripper_tolerance <= 0.0
        or image_mad_tolerance <= 0.0
        or source_image_mad_tolerance <= 0.0
        or compare_start_frame >= frame_count
    ):
        raise RuntimeError(
            "validation tolerances must be positive and compare_start_frame must be within the run"
        )

    print(
        "[RestoreValidation] NO HARDWARE / NO RECORDER: "
        f"dataset={root} episode={episode_index} frames={frame_count}/{len(expected_states)} "
        f"judged_frames={compare_start_frame}..{frame_count - 1} "
        f"direct_control={task_env.piperx_intervention_occurred}",
        flush=True,
    )
    max_joint_error = -1.0
    max_joint_frame = -1
    max_gripper_error = -1.0
    max_gripper_frame = -1
    squared_joint_error = 0.0
    squared_gripper_error = 0.0
    joint_value_count = 0
    gripper_value_count = 0
    start_joint_error = math.nan
    start_gripper_error = math.nan
    source_truth_passed = True
    image_totals = {
        name: {"all_abs": 0.0, "all_sq": 0.0, "eval_abs": 0.0, "eval_sq": 0.0}
        for name in _CAMERA_SOURCES
    }

    for frame in range(frame_count):
        obs = task_env.get_obs()
        live = _flatten_state(obs.get("state"), label=f"live frame {frame}")
        error = live - expected_states[frame]
        joint_abs = np.abs(error[_JOINT_INDICES])
        gripper_abs = np.abs(error[_GRIPPER_INDICES])
        frame_joint_max = float(np.max(joint_abs))
        frame_gripper_max = float(np.max(gripper_abs))
        if frame >= compare_start_frame:
            if frame_joint_max > max_joint_error:
                max_joint_error = frame_joint_max
                max_joint_frame = frame
            if frame_gripper_max > max_gripper_error:
                max_gripper_error = frame_gripper_max
                max_gripper_frame = frame
            squared_joint_error += float(np.dot(joint_abs, joint_abs))
            squared_gripper_error += float(np.dot(gripper_abs, gripper_abs))
            joint_value_count += len(joint_abs)
            gripper_value_count += len(gripper_abs)
        if frame == 0:
            start_joint_error = frame_joint_max
            start_gripper_error = frame_gripper_max

        image_items: list[str] = []
        for camera_name, source_name in _CAMERA_SOURCES.items():
            expected_image = expected_images[camera_name][frame]
            live_image = _live_image(obs, source_name, expected_image.shape)
            mad, mse, psnr = _pixel_metrics(live_image, expected_image)
            if frame == 0:
                selected_source_offset = source_video_frame - source_window_start
                source_truth = source_truth_images[camera_name][selected_source_offset]
                source_mad, _, source_psnr = _pixel_metrics(live_image, source_truth)
                nearby = [
                    _pixel_metrics(live_image, candidate)[0]
                    for candidate in source_truth_images[camera_name]
                ]
                nearby_best_offset = int(np.argmin(nearby))
                nearby_best_frame = source_window_start + nearby_best_offset
                nearby_best_mad = nearby[nearby_best_offset]
                legacy_gap_mad, _, legacy_gap_psnr = _pixel_metrics(
                    expected_image,
                    source_truth,
                )
                camera_source_passed = source_mad <= source_image_mad_tolerance
                source_truth_passed = source_truth_passed and camera_source_passed
                print(
                    "[RestoreValidation][SOURCE_TRUTH] "
                    f"camera={camera_name} source_episode={source_episode} "
                    f"source_state_frame={source_frame} "
                    f"source_video_frame={source_video_frame} live_MAD={source_mad:.4f} "
                    f"live_PSNR={source_psnr:.3f}dB limit={source_image_mad_tolerance:.4f} "
                    f"status={'PASS' if camera_source_passed else 'FAIL'} "
                    f"nearby_best_frame={nearby_best_frame} "
                    f"nearby_best_MAD={nearby_best_mad:.4f} "
                    f"old_recovery_frame0_vs_source_MAD={legacy_gap_mad:.4f} "
                    f"old_recovery_frame0_vs_source_PSNR={legacy_gap_psnr:.3f}dB",
                    flush=True,
                )
            pixel_count = float(expected_image.size)
            totals = image_totals[camera_name]
            totals["all_abs"] += mad * pixel_count
            totals["all_sq"] += mse * pixel_count
            if frame >= compare_start_frame:
                totals["eval_abs"] += mad * pixel_count
                totals["eval_sq"] += mse * pixel_count
            image_items.append(f"{camera_name}={mad:.3f}/{psnr:.2f}dB")
        print(
            f"[RestoreValidation] frame={frame:04d} "
            f"joint_max={frame_joint_max:.8f}rad/{math.degrees(frame_joint_max):.6f}deg "
            f"gripper_max={frame_gripper_max:.8f} "
            f"images_MAD/PSNR={' '.join(image_items)}",
            flush=True,
        )

        if frame + 1 < frame_count:
            action = _action_dict(actions[frame])
            roundtrip = _flatten_state(action, label=f"action frame {frame}")
            if not np.array_equal(roundtrip, actions[frame]):
                raise RuntimeError(f"action frame {frame} did not survive dictionary conversion")
            task_env.take_action(action)

    joint_rmse = math.sqrt(squared_joint_error / joint_value_count)
    gripper_rmse = math.sqrt(squared_gripper_error / gripper_value_count)
    image_passed = True
    for camera_name, frames in expected_images.items():
        pixel_count = float(frames[0].size)
        totals = image_totals[camera_name]
        all_pixels = pixel_count * frame_count
        evaluated_pixels = pixel_count * (frame_count - compare_start_frame)
        all_mad = totals["all_abs"] / all_pixels
        all_mse = totals["all_sq"] / all_pixels
        evaluated_mad = totals["eval_abs"] / evaluated_pixels
        evaluated_mse = totals["eval_sq"] / evaluated_pixels
        all_psnr = math.inf if all_mse == 0.0 else 10.0 * math.log10((255.0 * 255.0) / all_mse)
        evaluated_psnr = (
            math.inf
            if evaluated_mse == 0.0
            else 10.0 * math.log10((255.0 * 255.0) / evaluated_mse)
        )
        image_passed = image_passed and evaluated_mad <= image_mad_tolerance
        print(
            f"[RestoreValidation][IMAGE] camera={camera_name} "
            f"all_frames_MAD={all_mad:.4f} all_frames_PSNR={all_psnr:.3f}dB "
            f"evaluated_frames={compare_start_frame}..{frame_count - 1} "
            f"MAD={evaluated_mad:.4f} PSNR={evaluated_psnr:.3f}dB "
            f"MAD_limit={image_mad_tolerance:.4f}",
            flush=True,
        )
    passed = (
        source_truth_passed
        and start_joint_error <= joint_tolerance_rad
        and start_gripper_error <= gripper_tolerance
        and max_joint_error <= joint_tolerance_rad
        and max_gripper_error <= gripper_tolerance
        and image_passed
    )
    result = "PASS" if passed else "FAIL"
    print(
        f"[RestoreValidation][{result}] start_frame=0 compared_frames={frame_count} "
        f"applied_actions={frame_count - 1} "
        f"judged_frames={compare_start_frame}..{frame_count - 1} "
        f"observed_frame0_joint_max={start_joint_error:.8f}rad "
        f"observed_frame0_gripper_max={start_gripper_error:.8f} "
        f"source_visual_truth={'PASS' if source_truth_passed else 'FAIL'} "
        f"joint_max={max_joint_error:.8f}rad/{math.degrees(max_joint_error):.6f}deg "
        f"joint_max_frame={max_joint_frame} joint_rmse={joint_rmse:.8f}rad "
        f"gripper_max={max_gripper_error:.8f} gripper_max_frame={max_gripper_frame} "
        f"gripper_rmse={gripper_rmse:.8f} "
        f"limits=({joint_tolerance_rad:.8f}rad,{gripper_tolerance:.8f},"
        f"image_MAD={image_mad_tolerance:.4f})",
        flush=True,
    )
    if not passed:
        raise RuntimeError(
            "restore recovery replay exceeded tolerance; see [RestoreValidation][FAIL] above"
        )
    return result
