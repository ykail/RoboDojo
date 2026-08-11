"""Turn atomic raw X5 takeover bundles into genuine 25 Hz LeRobot episodes.

Unlike the former wall-time filler, every output frame here owns one real
40-ms simulator transition.  The physical X5 and policy server are not used.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable
import uuid

import numpy as np

from .x5_raw_bundle import RawBundle, load_pending_bundles


SIDES = ("left", "right")


class X5RawReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class _ReplayState:
    manifest: dict[str, Any]
    state: dict[str, np.ndarray]


@dataclass(frozen=True)
class RawReplaySummary:
    total: int
    already_committed: int
    newly_committed: int
    failed: tuple[str, ...]


def raw_bundle_id(bundle: RawBundle) -> str:
    digest = str(bundle.manifest.get("collection_identity_sha256", ""))
    if not digest.startswith("sha256:"):
        raise X5RawReplayError(f"bundle has no collection identity: {bundle.path}")
    return f"{digest}:{bundle.episode_index:07d}"


def raw_segment_id(bundle: RawBundle, segment_index: int) -> str:
    """Return the stable exactly-once identity of one LeRobot output unit.

    Version-1 bundles already shipped with the parent bundle ID in episode
    sidecars, so their sole segment deliberately keeps that identity.  Version
    2 bundles append an explicit segment suffix; changing either spelling would
    make a resumed collection duplicate previously committed episodes.
    """

    if isinstance(segment_index, bool) or not isinstance(segment_index, int):
        raise TypeError("segment_index must be an integer")
    if not 0 <= segment_index < bundle.segment_count:
        raise IndexError(f"segment index out of range: {segment_index}")
    parent = raw_bundle_id(bundle)
    if int(bundle.manifest.get("format_version", 1)) == 1:
        if bundle.segment_count != 1 or segment_index != 0:
            raise X5RawReplayError("a version-1 raw bundle must contain one segment")
        return parent
    return f"{parent}:segment_{segment_index:04d}"


def _anchor(manifest: dict[str, Any], name: str, side: str) -> tuple[np.ndarray, float]:
    table = manifest.get(name)
    if not isinstance(table, dict) or not isinstance(table.get(side), dict):
        raise X5RawReplayError(f"bundle has no {name}.{side}")
    item = table[side]
    q = np.asarray(item.get("q_rad"), dtype=np.float64).reshape(-1)
    grip = float(item.get("gripper_open_fraction"))
    if q.shape != (6,) or not np.isfinite(q).all() or not math.isfinite(grip):
        raise X5RawReplayError(f"invalid {name}.{side}")
    return q, float(np.clip(grip, 0.0, 1.0))


def actions_from_segment(
    bundle: RawBundle,
    segment_index: int,
    *,
    fps: int = 25,
) -> list[dict[str, np.ndarray]]:
    """Map one source segment onto its own simulator takeover anchor at 25 Hz."""

    if fps != 25:
        raise X5RawReplayError("canonical X5 replay is fixed at 25 Hz")
    try:
        segment = bundle.segments[segment_index]
        source = bundle.source_for_segment(segment_index)
    except (IndexError, TypeError, ValueError) as exc:
        raise X5RawReplayError(f"invalid raw segment {segment_index}: {exc}") from exc
    required = {
        "sample_monotonic_ns",
        "segment_start_ns",
        "segment_end_ns",
        "segment_index",
        "left_q_rad",
        "right_q_rad",
        "left_gripper_open_fraction",
        "right_gripper_open_fraction",
        "segment_anchor_timestamp_ns",
        "left_segment_anchor_q_rad",
        "right_segment_anchor_q_rad",
        "left_segment_anchor_gripper_open_fraction",
        "right_segment_anchor_gripper_open_fraction",
    }
    missing = sorted(required - set(source))
    if missing:
        raise X5RawReplayError(f"raw source is missing {missing}")
    starts = np.asarray(source["segment_start_ns"], dtype=np.int64).reshape(-1)
    ends = np.asarray(source["segment_end_ns"], dtype=np.int64).reshape(-1)
    sample_segment_index = np.asarray(
        source["segment_index"], dtype=np.int64
    ).reshape(-1)
    timestamps = np.asarray(source["sample_monotonic_ns"], dtype=np.int64).reshape(-1)
    if (
        starts.shape != (1,)
        or ends.shape != (1,)
        or sample_segment_index.shape != timestamps.shape
        or set(sample_segment_index.tolist()) != {segment.index}
    ):
        raise X5RawReplayError("raw source slice does not match its intervention segment")
    start_ns, end_ns = int(starts[0]), int(ends[0])
    anchor_timestamps = np.asarray(
        source["segment_anchor_timestamp_ns"], dtype=np.int64
    ).reshape(-1)
    if end_ns <= start_ns or timestamps.size < 1:
        raise X5RawReplayError("raw intervention segment is empty or has invalid bounds")
    if (
        anchor_timestamps.shape != (1,)
        or int(anchor_timestamps[0]) < start_ns
        or int(anchor_timestamps[0]) > end_ns
    ):
        raise X5RawReplayError("invalid X5 takeover anchor timestamp")
    timestamp_deltas = np.diff(timestamps)
    if np.any(timestamp_deltas <= 0):
        raise X5RawReplayError("raw source timestamps are not strictly increasing")
    if timestamps.size == 1 and (end_ns - start_ns) * 1e-9 > 0.1:
        raise X5RawReplayError(
            "single-sample X5 segment exceeds 100 ms; refusing to invent a trajectory"
        )
    if timestamp_deltas.size and float(np.max(timestamp_deltas)) * 1e-9 > 0.1:
        raise X5RawReplayError(
            "raw X5 sampling gap exceeds 100 ms; refusing to invent a trajectory"
        )
    mask = (
        (timestamps >= start_ns)
        & (timestamps <= end_ns)
        & (sample_segment_index == segment.index)
    )
    sample_t = timestamps[mask]
    if sample_t.size < 1:
        raise X5RawReplayError("raw segment has no sample inside its key boundaries")

    duration_s = (end_ns - start_ns) * 1e-9
    frame_count = max(1, int(math.ceil(duration_s * fps - 1e-12)))
    grid_ns = start_ns + np.arange(frame_count, dtype=np.float64) * (1e9 / fps)
    # The terminal boundary is exclusive: it belongs to the next control mode.
    grid_ns = grid_ns[grid_ns < end_ns]
    if grid_ns.size == 0:
        grid_ns = np.asarray([float(start_ns)])

    actions = [
        {
            "left_arm_joint_state": np.empty(6, dtype=np.float64),
            "left_ee_joint_state": np.empty(1, dtype=np.float64),
            "right_arm_joint_state": np.empty(6, dtype=np.float64),
            "right_ee_joint_state": np.empty(1, dtype=np.float64),
        }
        for _ in range(grid_ns.size)
    ]
    for side in SIDES:
        sim_q, _ = _anchor(segment.manifest, "sim_anchor", side)
        source_q_anchor, source_grip_anchor = _anchor(
            segment.manifest, "source_anchor", side
        )
        recorded_anchor_q = np.asarray(
            source[f"{side}_segment_anchor_q_rad"], dtype=np.float64
        ).reshape(-1, 6)
        recorded_anchor_grip = np.asarray(
            source[f"{side}_segment_anchor_gripper_open_fraction"],
            dtype=np.float64,
        ).reshape(-1)
        if (
            recorded_anchor_q.shape != (1, 6)
            or recorded_anchor_grip.shape != (1,)
            or not np.allclose(recorded_anchor_q[0], source_q_anchor, atol=1e-6, rtol=0)
            or not math.isclose(
                float(recorded_anchor_grip[0]),
                source_grip_anchor,
                abs_tol=1e-6,
            )
        ):
            raise X5RawReplayError(
                f"manifest and source fragment disagree on the {side} takeover anchor"
            )
        q_samples = np.asarray(source[f"{side}_q_rad"], dtype=np.float64)[mask]
        grip_samples = np.asarray(
            source[f"{side}_gripper_open_fraction"], dtype=np.float64
        ).reshape(-1)[mask]
        if (
            q_samples.shape != (sample_t.size, 6)
            or grip_samples.shape != (sample_t.size,)
            or not np.isfinite(q_samples).all()
            or not np.isfinite(grip_samples).all()
        ):
            raise X5RawReplayError(f"invalid {side} raw samples")

        later = sample_t > start_ns
        interpolation_t = np.concatenate(
            (np.asarray([float(start_ns)]), sample_t[later].astype(np.float64))
        )
        interpolation_q = np.concatenate(
            (source_q_anchor.reshape(1, 6), q_samples[later]), axis=0
        )
        # Gripper mapping is absolute, unlike the relative joint mapping.
        interpolation_grip = np.concatenate(
            (np.asarray([source_grip_anchor]), grip_samples[later]), axis=0
        )
        q_grid = np.column_stack(
            [
                np.interp(grid_ns, interpolation_t, interpolation_q[:, joint])
                for joint in range(6)
            ]
        )
        grip_grid = np.interp(grid_ns, interpolation_t, interpolation_grip)
        mapped_q = sim_q.reshape(1, 6) + q_grid - source_q_anchor.reshape(1, 6)
        mapped_q[0] = sim_q
        grip_grid[0] = source_grip_anchor
        for index, action in enumerate(actions):
            action[f"{side}_arm_joint_state"][:] = mapped_q[index]
            action[f"{side}_ee_joint_state"][0] = float(
                np.clip(grip_grid[index], 0.0, 1.0)
            )
    return actions


def actions_from_bundle(bundle: RawBundle, *, fps: int = 25) -> list[dict[str, np.ndarray]]:
    """Compatibility entry point for a legacy or otherwise single segment bundle."""

    if bundle.segment_count != 1:
        raise X5RawReplayError(
            "bundle has multiple intervention segments; use actions_from_segment"
        )
    return actions_from_segment(bundle, 0, fps=fps)


def _dataset_path(dataset_root: str | os.PathLike[str], dataset_id: str) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    result = (root / dataset_id).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise X5RawReplayError(f"dataset id escapes root: {dataset_id!r}") from exc
    if result == root:
        raise X5RawReplayError("dataset id must name a child directory")
    return result


def _committed_sidecars(dataset_path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    directory = dataset_path / "meta" / "robodojo" / "episodes"
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob("episode_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise X5RawReplayError(f"cannot read LeRobot sidecar {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise X5RawReplayError(f"LeRobot sidecar is not an object: {path}")
        if "robodojo_recovery_source" not in payload:
            continue
        recovery = payload["robodojo_recovery_source"]
        if not isinstance(recovery, dict):
            raise X5RawReplayError(f"invalid recovery_source in {path}")
        if "raw_bundle_id" not in recovery:
            # Existing policy or PiPER episodes legitimately persist {}.
            continue
        identifier = recovery.get("raw_bundle_id")
        if not isinstance(identifier, str) or not identifier:
            raise X5RawReplayError(f"empty raw_bundle_id in {path}")
        if identifier in result:
            raise X5RawReplayError(
                f"raw bundle {identifier} appears in two LeRobot episodes"
            )
        result[identifier] = path
    return result


def _is_legacy_bundle(bundle: RawBundle) -> bool:
    return int(bundle.manifest.get("format_version", 1)) == 1


def _parent_marker_path(bundle: RawBundle) -> Path:
    return bundle.path / "REPLAYED.json"


def _segment_marker_path(bundle: RawBundle, segment_index: int) -> Path:
    if _is_legacy_bundle(bundle):
        if segment_index != 0 or bundle.segment_count != 1:
            raise X5RawReplayError("a version-1 bundle must contain one segment")
        return _parent_marker_path(bundle)
    return bundle.path / f"REPLAYED.segment_{segment_index:04d}.json"


def _failure_marker_path(bundle: RawBundle, segment_index: int) -> Path:
    if _is_legacy_bundle(bundle):
        return bundle.path / "REPLAY_FAILED.json"
    return bundle.path / f"REPLAY_FAILED.segment_{segment_index:04d}.json"


def _read_marker(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise X5RawReplayError(f"invalid replay marker {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise X5RawReplayError(f"invalid replay marker object: {path}")
    return payload


def _validate_marker(
    bundle: RawBundle,
    marker: dict[str, Any],
    *,
    segment_index: int,
    identifier: str,
    dataset_path: Path,
    sidecar: Path | None,
) -> None:
    if marker.get("raw_bundle_id") != identifier:
        raise X5RawReplayError(f"replay marker raw id mismatch: {bundle.path}")
    if marker.get("dataset_path") != str(dataset_path):
        raise X5RawReplayError(
            f"raw bundle was already replayed to another dataset: {bundle.path}"
        )
    if not _is_legacy_bundle(bundle) and (
        marker.get("raw_parent_bundle_id") != raw_bundle_id(bundle)
        or marker.get("raw_segment_index") != segment_index
        or marker.get("raw_segment_count") != bundle.segment_count
    ):
        raise X5RawReplayError(f"replay marker segment identity mismatch: {bundle.path}")
    if sidecar is not None and marker.get("episode_sidecar") != str(sidecar):
        raise X5RawReplayError(f"replay marker sidecar mismatch: {bundle.path}")


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_replayed_marker(
    bundle: RawBundle,
    *,
    segment_index: int,
    identifier: str,
    dataset_path: Path,
    sidecar: Path,
) -> None:
    marker = {
        "format_version": 1 if _is_legacy_bundle(bundle) else 2,
        "raw_bundle_id": identifier,
        "raw_parent_bundle_id": raw_bundle_id(bundle),
        "raw_segment_index": int(segment_index),
        "raw_segment_count": bundle.segment_count,
        "dataset_path": str(dataset_path),
        "episode_sidecar": str(sidecar),
        "committed_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    _write_json_atomically(_segment_marker_path(bundle, segment_index), marker)


def _write_failure(
    bundle: RawBundle,
    segment_index: int,
    identifier: str,
    exc: BaseException,
) -> None:
    _write_json_atomically(
        _failure_marker_path(bundle, segment_index),
        {
            "format_version": 1 if _is_legacy_bundle(bundle) else 2,
            "raw_bundle_id": identifier,
            "raw_parent_bundle_id": raw_bundle_id(bundle),
            "raw_segment_index": int(segment_index),
            "failed_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    )


def _validate_parent_marker(
    bundle: RawBundle,
    marker: dict[str, Any],
    *,
    dataset_path: Path,
    sidecars: list[Path],
) -> None:
    if _is_legacy_bundle(bundle):
        raise X5RawReplayError("legacy replay marker is not a v2 parent marker")
    expected_ids = [raw_segment_id(bundle, index) for index in range(bundle.segment_count)]
    if (
        marker.get("raw_parent_bundle_id") != raw_bundle_id(bundle)
        or marker.get("raw_segment_count") != bundle.segment_count
        or marker.get("raw_segment_ids") != expected_ids
        or marker.get("dataset_path") != str(dataset_path)
        or marker.get("episode_sidecars") != [str(path) for path in sidecars]
    ):
        raise X5RawReplayError(f"invalid parent replay marker: {bundle.path}")


def _write_parent_marker(
    bundle: RawBundle,
    *,
    dataset_path: Path,
    sidecars: list[Path],
) -> None:
    if _is_legacy_bundle(bundle):
        return
    _write_json_atomically(
        _parent_marker_path(bundle),
        {
            "format_version": 2,
            "raw_parent_bundle_id": raw_bundle_id(bundle),
            "raw_segment_count": bundle.segment_count,
            "raw_segment_ids": [
                raw_segment_id(bundle, index) for index in range(bundle.segment_count)
            ],
            "dataset_path": str(dataset_path),
            "episode_sidecars": [str(path) for path in sidecars],
            "committed_at": datetime.now(timezone.utc).isoformat(
                timespec="microseconds"
            ),
        },
    )


def _replay_one_segment(
    task_env: Any,
    bundle: RawBundle,
    segment_index: int,
    dataset_path: Path,
    *,
    fps: int,
    recorder_factory: Callable[[Any], Any],
    restore_fn: Callable[..., Any],
) -> None:
    del dataset_path  # The writer path is pinned and checked by the collection entry.
    segment = bundle.segments[segment_index]
    identifier = raw_segment_id(bundle, segment_index)
    metadata = bundle.manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise X5RawReplayError(f"bundle metadata is not an object: {bundle.path}")
    metadata = deepcopy(metadata)
    segment_metadata = segment.manifest.get("metadata", {})
    if not isinstance(segment_metadata, dict):
        raise X5RawReplayError(
            f"segment {segment_index} metadata is not an object: {bundle.path}"
        )
    metadata.update(deepcopy(segment_metadata))
    snapshot = segment.snapshot
    provenance = metadata.get("policy_provenance")
    saved_layout = snapshot.get("replay_saved_layout")
    replay_manifest = snapshot.get("replay_manifest")
    if not isinstance(provenance, dict) or not provenance:
        raise X5RawReplayError(f"bundle has no policy provenance: {bundle.path}")
    if not isinstance(saved_layout, dict) or not isinstance(replay_manifest, dict):
        raise X5RawReplayError(f"bundle has no replay layout/manifest: {bundle.path}")
    layout_id = int(metadata.get("layout_id", -1))
    if layout_id < 0:
        raise X5RawReplayError(f"bundle has no valid layout id: {bundle.path}")
    actions = actions_from_segment(bundle, segment_index, fps=fps)

    task_env.control_mode = "x5_raw_replay_25hz"
    task_env.restore_saved_layout = deepcopy(saved_layout)
    task_env.eval_seed = int(metadata.get("eval_seed", 0))
    task_env.env_seeds = [layout_id]
    task_env.reset(seed=[layout_id])
    restore_fn(
        task_env,
        _ReplayState(manifest=deepcopy(replay_manifest), state=segment.takeover_state),
        _reset_episode=False,
    )
    task_env.policy_provenance = deepcopy(provenance)
    task_env.policy_runtime = "robodojo_policy_v1"
    task_env.restore_lineage = {
        "raw_bundle_id": identifier,
        "raw_parent_bundle_id": raw_bundle_id(bundle),
        "raw_bundle_path": str(bundle.path),
        "raw_episode_index": bundle.episode_index,
        "raw_segment_index": segment_index,
        "raw_segment_count": bundle.segment_count,
        "source_checkpoint": provenance.get("checkpoint_id", ""),
        "source_policy_provenance": deepcopy(provenance),
        "operator_accepted": bool(metadata.get("operator_accepted", True)),
        "replay_fps": fps,
    }
    recorder = recorder_factory(task_env)
    finalized = False
    try:
        obs = task_env.get_obs()
        for frame_index, action in enumerate(actions):
            control = {
                "action_source": "human",
                "intervention_mask": 1,
                "active_arm": "both",
                "takeover_edge": 1 if frame_index == 0 else 0,
                "chunk_id": -1,
                "chunk_index": -1,
                "timestamp": frame_index / float(fps),
            }
            recorder.append(
                obs=obs,
                policy_action=None,
                human_action=deepcopy(action),
                executed_action=deepcopy(action),
                control=control,
            )
            task_env.take_action(action, interpolate=False)
            if frame_index + 1 < len(actions):
                obs = task_env.get_obs()
        try:
            reward = task_env.reward_manager.get_reward(final_check=True)
            replay_success = bool(reward[0] > 1 - 1e-3)
        except Exception:
            replay_success = False
        saved = recorder.finalize(
            accepted=True,
            success=replay_success,
            reason=f"x5_raw_replay_segment_{segment_index:04d}_operator_accepted",
            timestamp_s=len(actions) / float(fps),
        )
        finalized = True
        if saved is None:
            raise X5RawReplayError("LeRobot writer did not commit the replay episode")
    except BaseException:
        if not finalized:
            try:
                recorder.finalize(
                    accepted=False,
                    success=False,
                    reason="x5_raw_replay_failed",
                )
            except Exception:
                pass
        raise


def run_x5_raw_replay_collection(
    task_env: Any,
    raw_root: str | os.PathLike[str],
    dataset_root: str | os.PathLike[str],
    dataset_id: str,
    *,
    fps: int = 25,
    max_bundles: int = 0,
    _bundle_loader: Callable[..., Iterable[RawBundle]] = load_pending_bundles,
    _recorder_factory: Callable[[Any], Any] | None = None,
    _restore_fn: Callable[..., Any] | None = None,
) -> RawReplaySummary:
    """Replay every not-yet-committed bundle, preserving exact-once output."""

    if fps != 25 or max_bundles < 0:
        raise X5RawReplayError("fps must be 25 and max_bundles must be non-negative")
    bundles = list(_bundle_loader(raw_root, verify=True))
    dataset_path = _dataset_path(dataset_root, dataset_id)
    configured_root = os.environ.get("ROBODOJO_LEROBOT_ROOT", "").strip()
    configured_id = os.environ.get("ROBODOJO_LEROBOT_REPO_ID", "").strip()
    if configured_root and Path(configured_root).expanduser().resolve() != Path(
        dataset_root
    ).expanduser().resolve():
        raise X5RawReplayError(
            "dataset_root argument differs from ROBODOJO_LEROBOT_ROOT"
        )
    if configured_id and configured_id != dataset_id:
        raise X5RawReplayError(
            "dataset_id argument differs from ROBODOJO_LEROBOT_REPO_ID"
        )
    os.environ["ROBODOJO_LEROBOT_ROOT"] = str(
        Path(dataset_root).expanduser().resolve()
    )
    os.environ["ROBODOJO_LEROBOT_REPO_ID"] = dataset_id
    collect_freq = float(getattr(task_env.obs_manager, "collect_freq", 0.0))
    if collect_freq != 25.0:
        raise X5RawReplayError(
            f"raw replay requires observation.collect_freq=25, got {collect_freq}"
        )
    task_env.control_mode = "x5_raw_replay_25hz"
    committed = _committed_sidecars(dataset_path)
    if _recorder_factory is None:
        from .lerobot_stream_recorder import recorder_for_env

        _recorder_factory = recorder_for_env
    if _restore_fn is None:
        from .sim_state_restore import restore_replay_frame

        _restore_fn = restore_replay_frame

    skipped = 0
    newly_committed = 0
    failures: list[str] = []
    attempted_bundles = 0
    for bundle in bundles:
        identifiers = [
            raw_segment_id(bundle, index) for index in range(bundle.segment_count)
        ]
        parent_marker = (
            None
            if _is_legacy_bundle(bundle)
            else _read_marker(_parent_marker_path(bundle))
        )
        known_sidecars = [committed.get(identifier) for identifier in identifiers]
        if parent_marker is not None:
            if any(sidecar is None for sidecar in known_sidecars):
                raise X5RawReplayError(
                    f"parent replay marker exists but a segment sidecar is missing: {bundle.path}"
                )
            _validate_parent_marker(
                bundle,
                parent_marker,
                dataset_path=dataset_path,
                sidecars=[sidecar for sidecar in known_sidecars if sidecar is not None],
            )

        needs_replay = any(identifier not in committed for identifier in identifiers)
        if needs_replay and max_bundles and attempted_bundles >= max_bundles:
            break
        if needs_replay:
            attempted_bundles += 1
        print(
            f"[X5 raw replay] {bundle.episode_index + 1}/{len(bundles)} "
            f"parent={raw_bundle_id(bundle)} segments={bundle.segment_count}",
            flush=True,
        )
        for segment_index, identifier in enumerate(identifiers):
            marker_path = _segment_marker_path(bundle, segment_index)
            marker = _read_marker(marker_path)
            sidecar = committed.get(identifier)
            if sidecar is not None:
                if marker is None:
                    # Covers a crash after LeRobot's atomic commit and before
                    # the per-segment raw marker.
                    _write_replayed_marker(
                        bundle,
                        segment_index=segment_index,
                        identifier=identifier,
                        dataset_path=dataset_path,
                        sidecar=sidecar,
                    )
                else:
                    _validate_marker(
                        bundle,
                        marker,
                        segment_index=segment_index,
                        identifier=identifier,
                        dataset_path=dataset_path,
                        sidecar=sidecar,
                    )
                skipped += 1
                continue
            if marker is not None:
                _validate_marker(
                    bundle,
                    marker,
                    segment_index=segment_index,
                    identifier=identifier,
                    dataset_path=dataset_path,
                    sidecar=None,
                )
                raise X5RawReplayError(
                    "segment marker exists but its LeRobot sidecar is missing: "
                    f"{marker_path}"
                )
            try:
                _replay_one_segment(
                    task_env,
                    bundle,
                    segment_index,
                    dataset_path,
                    fps=fps,
                    recorder_factory=_recorder_factory,
                    restore_fn=_restore_fn,
                )
                committed = _committed_sidecars(dataset_path)
                sidecar = committed.get(identifier)
                if sidecar is None:
                    raise X5RawReplayError(
                        "LeRobot finalize returned but no provenance sidecar was committed"
                    )
                _write_replayed_marker(
                    bundle,
                    segment_index=segment_index,
                    identifier=identifier,
                    dataset_path=dataset_path,
                    sidecar=sidecar,
                )
                _failure_marker_path(bundle, segment_index).unlink(missing_ok=True)
                newly_committed += 1
                print(
                    f"[X5 raw replay] COMMITTED {identifier} -> {sidecar}",
                    flush=True,
                )
            except Exception as exc:
                _write_failure(bundle, segment_index, identifier, exc)
                failures.append(f"{identifier}: {type(exc).__name__}: {exc}")
                print(f"[X5 raw replay][ERROR] {failures[-1]}", flush=True)

        completed_sidecars = [committed.get(identifier) for identifier in identifiers]
        if all(sidecar is not None for sidecar in completed_sidecars):
            ordered_sidecars = [
                sidecar for sidecar in completed_sidecars if sidecar is not None
            ]
            if parent_marker is None:
                _write_parent_marker(
                    bundle,
                    dataset_path=dataset_path,
                    sidecars=ordered_sidecars,
                )
            elif not _is_legacy_bundle(bundle):
                _validate_parent_marker(
                    bundle,
                    parent_marker,
                    dataset_path=dataset_path,
                    sidecars=ordered_sidecars,
                )

    summary = RawReplaySummary(
        total=sum(bundle.segment_count for bundle in bundles),
        already_committed=skipped,
        newly_committed=newly_committed,
        failed=tuple(failures),
    )
    print(
        "[X5 raw replay] summary "
        f"total={summary.total} existing={summary.already_committed} "
        f"new={summary.newly_committed} failed={len(summary.failed)}",
        flush=True,
    )
    if failures:
        raise X5RawReplayError(
            f"{len(failures)} raw segment(s) failed; successful outputs remain committed"
        )
    return summary


__all__ = [
    "RawReplaySummary",
    "X5RawReplayError",
    "actions_from_bundle",
    "actions_from_segment",
    "raw_bundle_id",
    "raw_segment_id",
    "run_x5_raw_replay_collection",
]
