#!/usr/bin/env python3
"""Read-only structural and timing audit for one online X5 DAgger episode."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import av
import numpy as np
import pyarrow.parquet as pq


TIMING_CONTRACT = "sim_step_exact_25hz_v1"


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _episode_file(root: Path, template: str, episode_index: int, **extra: str) -> Path:
    chunk_size = 1000
    values = {
        "chunk_index": episode_index // chunk_size,
        "file_index": episode_index,
        **extra,
    }
    return root / template.format(**values)


def _numeric_column(table, name: str, *, width: int | None = None) -> np.ndarray:
    if name not in table.column_names:
        _fail(f"missing parquet column: {name}")
    values = np.asarray(table[name].to_pylist())
    if width is not None and values.shape != (table.num_rows, width):
        _fail(f"{name} has shape {values.shape}, expected {(table.num_rows, width)}")
    if not np.isfinite(values).all():
        _fail(f"{name} contains non-finite values")
    return values


def _delta_stats_deg(actions: np.ndarray, mask: np.ndarray) -> dict[str, tuple[float, float]]:
    if len(actions) < 2:
        return {"policy": (0.0, 0.0), "human": (0.0, 0.0)}
    arm = np.concatenate([actions[:, :6], actions[:, 7:13]], axis=1)
    delta_deg = np.rad2deg(np.abs(np.diff(arm, axis=0))).max(axis=1)
    result: dict[str, tuple[float, float]] = {}
    for label, selected in (
        ("policy", mask[1:] < 0.5),
        ("human", mask[1:] >= 0.5),
    ):
        values = delta_deg[selected]
        result[label] = (
            float(np.percentile(values, 95)) if len(values) else 0.0,
            float(values.max()) if len(values) else 0.0,
        )
    return result


def _video_frame_count(path: Path, expected_fps: float) -> int:
    if not path.is_file():
        _fail(f"missing video: {path}")
    with av.open(str(path)) as container:
        if not container.streams.video:
            _fail(f"video stream is missing: {path}")
        stream = container.streams.video[0]
        rate = stream.average_rate
        if rate is None or not math.isclose(float(rate), expected_fps, abs_tol=1e-6):
            _fail(f"video fps mismatch in {path}: {rate} != {expected_fps}")
        return sum(1 for _ in container.decode(video=0))


def audit(dataset: Path, episode_index: int | None) -> None:
    dataset = dataset.expanduser().resolve()
    info_path = dataset / "meta/info.json"
    if not info_path.is_file():
        _fail(f"not a LeRobot v3 dataset: {info_path} is missing")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = int(info.get("total_episodes", -1))
    fps = float(info.get("fps", 0.0))
    if total_episodes <= 0:
        _fail(f"dataset has no committed episodes: {dataset}")
    if fps != 25.0:
        _fail(f"dataset fps must be 25, got {fps}")
    if episode_index is None:
        episode_index = total_episodes - 1
    if not 0 <= episode_index < total_episodes:
        _fail(f"episode {episode_index} is outside [0, {total_episodes})")

    sidecar_path = dataset / "meta/robodojo/episodes" / f"episode_{episode_index:07d}.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("robodojo_timing_resample") != TIMING_CONTRACT:
        _fail(
            f"episode timing is {sidecar.get('robodojo_timing_resample')!r}, "
            f"expected {TIMING_CONTRACT!r}"
        )
    if sidecar.get("robodojo_control_mode") != "x5_policy_joint_intervention":
        _fail(f"unexpected control mode: {sidecar.get('robodojo_control_mode')!r}")
    if not bool(sidecar.get("robodojo_has_intervention")):
        _fail("episode contains no human intervention")

    data_path = _episode_file(
        dataset,
        str(info["data_path"]),
        episode_index,
    )
    table = pq.read_table(data_path)
    rows = int(table.num_rows)
    expected_rows = int(sidecar.get("robodojo_frame_count", -1))
    source_rows = int(sidecar.get("robodojo_source_frame_count", -1))
    if rows <= 0 or rows != expected_rows or rows != source_rows:
        _fail(
            f"row mismatch: parquet={rows}, sidecar={expected_rows}, source={source_rows}"
        )

    frame_index = _numeric_column(table, "frame_index").reshape(-1)
    if not np.array_equal(frame_index, np.arange(rows)):
        _fail("frame_index is not exactly 0..N-1")
    timestamps = _numeric_column(table, "timestamp").reshape(-1)
    expected_timestamps = np.arange(rows, dtype=np.float64) / fps
    if not np.allclose(timestamps, expected_timestamps, atol=2e-5, rtol=0.0):
        _fail("timestamps are not a uniform 25 Hz simulator-time grid")

    mask = _numeric_column(table, "complementary_info.is_intervention").reshape(-1)
    if not np.isin(mask, [0.0, 1.0]).all():
        _fail("intervention mask is not binary")
    human_rows = int(np.count_nonzero(mask >= 0.5))
    policy_rows = rows - human_rows
    manual_source = int(sidecar.get("robodojo_manual_source_frame_count", -1))
    manual_output = int(sidecar.get("robodojo_manual_output_frame_count", -1))
    if human_rows <= 0 or policy_rows <= 0:
        _fail(f"episode must contain policy and human rows, got {policy_rows}/{human_rows}")
    if human_rows != manual_source or human_rows != manual_output:
        _fail(
            "manual row mismatch: "
            f"parquet={human_rows}, source={manual_source}, output={manual_output}"
        )
    human_indices = np.flatnonzero(mask >= 0.5)
    starts_with_policy = bool(human_indices[0] > 0)
    resumes_policy = bool(human_indices[-1] < rows - 1)

    wall_elapsed = sidecar.get("robodojo_source_wall_elapsed_s", [])
    source_is_manual = sidecar.get("robodojo_source_is_manual", [])
    if len(wall_elapsed) != rows or len(source_is_manual) != rows:
        _fail("source wall-time audit arrays do not match the episode row count")

    action = _numeric_column(table, "action", width=14).astype(np.float64)
    delta_stats = _delta_stats_deg(action, mask)
    max_manual_gap_ms = 1000.0 * float(
        sidecar.get("robodojo_max_manual_source_gap_s", 0.0)
    )

    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if not video_keys:
        _fail("dataset declares no video features")
    for key in video_keys:
        video_path = _episode_file(
            dataset,
            str(info["video_path"]),
            episode_index,
            video_key=key,
        )
        decoded = _video_frame_count(video_path, fps)
        if decoded != rows:
            _fail(f"video frame mismatch for {key}: decoded={decoded}, rows={rows}")
        print(f"[AUDIT] {key}: {decoded} frames @ {fps:g} Hz")

    print(f"[AUDIT] dataset={dataset}")
    print(
        f"[AUDIT] episode={episode_index}/{total_episodes - 1} rows={rows} "
        f"policy={policy_rows} human={human_rows}"
    )
    print(
        f"[AUDIT] sequence starts_with_policy={starts_with_policy} "
        f"resumes_policy_after_human={resumes_policy}"
    )
    print(
        "[AUDIT] arm target max-joint delta: "
        f"policy p95/max={delta_stats['policy'][0]:.2f}/{delta_stats['policy'][1]:.2f} deg; "
        f"human p95/max={delta_stats['human'][0]:.2f}/{delta_stats['human'][1]:.2f} deg"
    )
    print(f"[AUDIT] max consecutive manual wall gap={max_manual_gap_ms:.0f} ms")
    if not starts_with_policy:
        print("[AUDIT][WARN] episode begins in human mode; no policy prefix was recorded")
    if not resumes_policy:
        print(
            "[AUDIT][WARN] episode ends in human mode; valid as a correction trajectory, "
            "but it does not validate second-i policy resume"
        )
    if max_manual_gap_ms > 120.0:
        print(
            "[AUDIT][WARN] manual source is slower than wall-time 25 Hz; "
            "judge the action-delta tail before batch collection"
        )
    policy_p95, policy_max = delta_stats["policy"]
    human_p95, human_max = delta_stats["human"]
    if human_p95 > max(5.0, 1.5 * policy_p95) or human_max > max(
        10.0, 1.5 * policy_max
    ):
        print(
            "[AUDIT][WARN] human target has a larger fast-motion tail than policy; "
            "move the X5 more slowly/smoothly or review these frames before training"
        )
    print(
        "[AUDIT] PASS: complete mixed policy/human episode, one row per simulator "
        "transition, uniform 25 Hz sim-time, and matching videos"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=None, help="default: latest")
    args = parser.parse_args()
    try:
        audit(args.dataset, args.episode)
    except Exception as exc:
        print(f"[AUDIT][FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
