"""Pure helpers for deterministic, resumable policy-rollout collections.

This module deliberately imports no Isaac, Torch, LeRobot, or CUDA package so
the top-level collection launcher can validate a plan before starting either
the policy server or Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

_COLLECTION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_PLAN_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
_EPISODE_METADATA_RE = re.compile(r"episode_\d{7}\.json")


@dataclass(frozen=True, slots=True)
class RolloutPlanEntry:
    plan_index: int
    eval_seed: int
    layout_id: int


def apply_rollout_collection_config(
    eval_config: MutableMapping[str, Any],
    settings: Mapping[str, Any],
) -> int:
    """Apply one collector segment to the config actually passed to EvalEnv.

    ``OmegaConf.create`` copies the source dictionary.  Keeping this operation
    here makes callers explicitly mutate ``env_cfg.eval_cfg`` instead of a
    detached pre-OmegaConf dictionary.
    """

    if not isinstance(eval_config, MutableMapping):
        raise TypeError("eval_config must be a mutable mapping")
    if not isinstance(settings, Mapping):
        raise TypeError("collection settings must be a mapping")

    layout_ids = [int(layout_id) for layout_id in settings["layout_ids"]]
    plan_index_by_layout = {
        int(layout_id): int(plan_index)
        for layout_id, plan_index in settings["plan_index_by_layout"].items()
    }
    missing = sorted(set(layout_ids) - set(plan_index_by_layout))
    if missing:
        raise ValueError(
            f"collection plan-index map is missing selected layout ids: {missing}"
        )

    eval_config["selected_layout_ids"] = layout_ids
    eval_config["collection_plan_index_by_layout"] = plan_index_by_layout
    eval_config["collection_manifest"] = settings["manifest"]
    eval_config["collection_id"] = str(settings["collection_id"])
    eval_config["collection_plan_hash"] = str(settings["plan_hash"])
    eval_config["eval_num"] = len(layout_ids)
    return len(layout_ids)


def _non_negative_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative, got {parsed}")
    return parsed


def _expand_layout_selection(selection: str) -> list[int]:
    """Expand ``0-3+7+9-10`` while preserving the written order."""

    if not selection:
        raise ValueError("layout selection cannot be empty")
    result: list[int] = []
    for token in selection.split("+"):
        token = token.strip()
        if not token:
            raise ValueError(f"invalid empty layout token in {selection!r}")
        if "-" not in token:
            result.append(_non_negative_int(token, "layout id"))
            continue
        bounds = token.split("-")
        if len(bounds) != 2:
            raise ValueError(f"invalid layout range {token!r}")
        start = _non_negative_int(bounds[0], "layout range start")
        end = _non_negative_int(bounds[1], "layout range end")
        if end < start:
            raise ValueError(f"layout range must be ascending, got {token!r}")
        result.extend(range(start, end + 1))
    return result


def parse_layout_plan(spec: str) -> list[RolloutPlanEntry]:
    """Parse ``SEED:IDS[,SEED:IDS...]`` into an ordered immutable plan.

    Within one seed, IDs may be joined with ``+`` and each token may be a
    single non-negative integer or an inclusive ascending range.  For example
    ``0:0-69,1:0-29`` describes 100 distinct saved layouts.
    """

    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("layout plan cannot be empty")
    entries: list[RolloutPlanEntry] = []
    seen: set[tuple[int, int]] = set()
    for segment in spec.split(","):
        segment = segment.strip()
        if segment.count(":") != 1:
            raise ValueError(
                f"layout plan segment must be SEED:IDS, got {segment!r}"
            )
        seed_text, selection = (part.strip() for part in segment.split(":", 1))
        eval_seed = _non_negative_int(seed_text, "eval seed")
        for layout_id in _expand_layout_selection(selection):
            identity = (eval_seed, layout_id)
            if identity in seen:
                raise ValueError(
                    f"duplicate layout in collection plan: seed={eval_seed} "
                    f"layout={layout_id}"
                )
            seen.add(identity)
            entries.append(
                RolloutPlanEntry(
                    plan_index=len(entries),
                    eval_seed=eval_seed,
                    layout_id=layout_id,
                )
            )
    if not entries:
        raise ValueError("layout plan contains no entries")
    return entries


def parse_layout_ids(spec: str) -> list[int]:
    """Parse evaluator-local IDs, accepting commas, plus signs, and ranges."""

    if not isinstance(spec, str) or not spec.strip():
        return []
    ids: list[int] = []
    for group in spec.split(","):
        ids.extend(_expand_layout_selection(group.strip()))
    if len(ids) != len(set(ids)):
        raise ValueError(f"layout id list contains duplicates: {spec!r}")
    return ids


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_collection_manifest(
    *,
    task: str,
    checkpoint_id: str,
    entries: Iterable[RolloutPlanEntry],
    fps: int = 25,
    policy_seed: int | None = None,
) -> dict[str, Any]:
    entry_list = [asdict(entry) for entry in entries]
    if not task:
        raise ValueError("task is required")
    if not checkpoint_id:
        raise ValueError("checkpoint_id is required")
    if fps <= 0:
        raise ValueError("fps must be positive")
    policy_seed_spec: dict[str, Any]
    if policy_seed is None:
        policy_seed_spec = {"mode": "eval_seed"}
    else:
        policy_seed_spec = {
            "mode": "fixed",
            "value": _non_negative_int(policy_seed, "policy seed"),
        }
    unsigned = {
        "format_version": 2,
        "kind": "robodojo_policy_rollout_collection",
        "task": str(task),
        "checkpoint_id": str(checkpoint_id),
        "fps": int(fps),
        "policy_seed": policy_seed_spec,
        "record_sim_state": True,
        "entries": entry_list,
    }
    digest = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return {
        **unsigned,
        "plan_hash": f"sha256:{digest}",
        "collection_id": f"rollout-{digest[:20]}",
    }


def validate_collection_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("collection manifest must be an object")
    collection_id = manifest.get("collection_id")
    plan_hash = manifest.get("plan_hash")
    if not isinstance(collection_id, str) or not _COLLECTION_ID_RE.fullmatch(
        collection_id
    ):
        raise ValueError(f"invalid collection_id: {collection_id!r}")
    if not isinstance(plan_hash, str) or not _PLAN_HASH_RE.fullmatch(plan_hash):
        raise ValueError(f"invalid plan_hash: {plan_hash!r}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("collection manifest entries must be a non-empty list")
    parsed: list[RolloutPlanEntry] = []
    for expected_index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError("collection manifest entry must be an object")
        entry = RolloutPlanEntry(
            plan_index=_non_negative_int(raw.get("plan_index"), "plan_index"),
            eval_seed=_non_negative_int(raw.get("eval_seed"), "eval_seed"),
            layout_id=_non_negative_int(raw.get("layout_id"), "layout_id"),
        )
        if entry.plan_index != expected_index:
            raise ValueError(
                "collection plan indices must be contiguous and ordered: "
                f"expected {expected_index}, got {entry.plan_index}"
            )
        parsed.append(entry)
    policy_seed_spec = manifest.get("policy_seed")
    if not isinstance(policy_seed_spec, dict):
        raise ValueError("collection manifest policy_seed must be an object")
    mode = policy_seed_spec.get("mode")
    if mode == "eval_seed":
        if set(policy_seed_spec) != {"mode"}:
            raise ValueError("eval_seed policy seed mode cannot contain extra fields")
        policy_seed = None
    elif mode == "fixed":
        if set(policy_seed_spec) != {"mode", "value"}:
            raise ValueError("fixed policy seed mode requires exactly mode/value")
        policy_seed = _non_negative_int(policy_seed_spec.get("value"), "policy seed")
    else:
        raise ValueError(f"unsupported collection policy seed mode: {mode!r}")
    rebuilt = build_collection_manifest(
        task=str(manifest.get("task", "")),
        checkpoint_id=str(manifest.get("checkpoint_id", "")),
        entries=parsed,
        fps=_non_negative_int(manifest.get("fps"), "fps"),
        policy_seed=policy_seed,
    )
    if rebuilt != manifest:
        raise ValueError("collection manifest content does not match its plan hash")


def dataset_root(base_root: str | Path, repo_id: str) -> Path:
    base = Path(base_root).expanduser().resolve()
    target = (base / repo_id).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"repo id escapes dataset root {base}: {repo_id!r}") from exc
    if target == base:
        raise ValueError("repo id must name a child dataset")
    return target


def _collection_manifest_path(root: Path, collection_id: str) -> Path:
    return root / "meta" / "robodojo" / "collections" / f"{collection_id}.json"


def validate_persisted_collection(root: Path, manifest: dict[str, Any]) -> None:
    """Validate the global manifest when at least one episode committed."""

    path = _collection_manifest_path(root, manifest["collection_id"])
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        persisted = json.load(stream)
    if not isinstance(persisted, dict) or persisted.get("plan") != manifest:
        raise ValueError(
            f"existing collection manifest does not match requested plan: {path}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _committed_sidecar_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} escapes the dataset: {value!r}")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or not a regular file: {path}")
    return path


def _validate_replay_npz(
    state_path: Path,
    *,
    frame_count: int,
    fps: int,
) -> None:
    # Keep plan parsing/dry-run dependency-free. Numpy is required only when a
    # real collection already exists and its replay NPZ files are verified.
    import numpy as np

    try:
        with np.load(state_path, allow_pickle=False) as state:
            if int(state["format_version"]) != 1:
                raise ValueError("unsupported replay NPZ format_version")
            if int(state["frame_count"]) != frame_count:
                raise ValueError("replay NPZ frame_count does not match metadata")
            frame_indices = np.asarray(state["frame__frame.index"])
            terminal_index = np.asarray(state["terminal__frame.index"])
            frame_timestamps = np.asarray(state["frame__frame.timestamp_s"])
            terminal_timestamp = np.asarray(state["terminal__frame.timestamp_s"])
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid replay NPZ structure: {state_path}: {exc}") from exc
    if frame_indices.shape != (frame_count,) or not np.array_equal(
        frame_indices,
        np.arange(frame_count, dtype=frame_indices.dtype),
    ):
        raise ValueError(f"replay NPZ frame.index is not 0..N-1: {state_path}")
    if terminal_index.shape != () or int(terminal_index) != frame_count:
        raise ValueError(f"replay NPZ terminal frame.index is not N: {state_path}")
    expected_timestamps = np.arange(frame_count, dtype=np.float64) / fps
    if frame_timestamps.shape != (frame_count,) or not np.allclose(
        frame_timestamps,
        expected_timestamps,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(f"replay NPZ timestamps do not match frame.index/fps: {state_path}")
    if terminal_timestamp.shape != () or not np.isclose(
        float(terminal_timestamp),
        frame_count / fps,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(f"replay NPZ terminal timestamp does not match N/fps: {state_path}")


def _validate_replay_files(
    root: Path,
    metadata_path: Path,
    payload: dict[str, Any],
    *,
    fps: int,
) -> None:
    replay = payload.get("robodojo_replay")
    if not isinstance(replay, dict) or not replay.get("complete"):
        raise ValueError(f"episode is missing a complete replay sidecar: {metadata_path}")
    snapshot_count = _non_negative_int(
        replay.get("snapshot_count"), "replay snapshot count"
    )
    frame_count = _non_negative_int(
        payload.get("robodojo_frame_count"), "episode frame count"
    )
    if snapshot_count != frame_count:
        raise ValueError(f"episode replay/frame count mismatch: {metadata_path}")
    state_path: Path | None = None
    for prefix in ("state", "layout", "schema"):
        sidecar = _committed_sidecar_path(
            root,
            replay.get(f"{prefix}_path"),
            label=f"episode {prefix} sidecar",
        )
        expected_digest = replay.get(
            "layout_file_sha256" if prefix == "layout" else f"{prefix}_sha256"
        )
        if not isinstance(expected_digest, str) or not _PLAN_HASH_RE.fullmatch(
            expected_digest
        ):
            raise ValueError(
                f"episode {prefix} sidecar has an invalid digest: {metadata_path}"
            )
        if _sha256_file(sidecar) != expected_digest:
            raise ValueError(
                f"episode {prefix} sidecar digest mismatch: {sidecar}"
            )
        if prefix == "state":
            state_path = sidecar
    if state_path is None:
        raise ValueError(f"episode replay state path is missing: {metadata_path}")
    _validate_replay_npz(state_path, frame_count=frame_count, fps=fps)


def committed_plan_indices(root: Path, manifest: dict[str, Any]) -> set[int]:
    """Return durable plan indices from atomically committed episode metadata."""

    validate_collection_manifest(manifest)
    if not root.exists():
        return set()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"existing path is not a LeRobot v3 dataset: {root}")
    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)
    try:
        total_episodes = int(info["total_episodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid LeRobot total_episodes in {info_path}") from exc
    if total_episodes < 0:
        raise ValueError(f"invalid negative total_episodes in {info_path}")

    metadata_dir = root / "meta" / "robodojo" / "episodes"
    if total_episodes == 0 and not metadata_dir.exists():
        return set()
    validate_persisted_collection(root, manifest)
    collection_path = _collection_manifest_path(root, manifest["collection_id"])
    if total_episodes > 0 and not collection_path.is_file():
        raise ValueError(
            "LeRobot contains committed episodes but its collection manifest is "
            f"missing: {collection_path}. Refusing automatic resume."
        )
    if not metadata_dir.is_dir():
        raise ValueError(
            "LeRobot contains committed episodes but RoboDojo episode metadata "
            f"is missing: {metadata_dir}. Refusing automatic resume."
        )
    expected_entries = {entry["plan_index"]: entry for entry in manifest["entries"]}
    completed: set[int] = set()
    episode_indices: set[int] = set()
    for path in sorted(metadata_dir.iterdir()):
        if not path.is_file() or not _EPISODE_METADATA_RE.fullmatch(path.name):
            continue
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        episode_index = _non_negative_int(
            payload.get("episode_index"), "episode index"
        )
        if episode_index in episode_indices:
            raise ValueError(f"duplicate episode index {episode_index}: {path}")
        episode_indices.add(episode_index)
        episode_collection = payload.get("robodojo_collection_id")
        if episode_collection != manifest["collection_id"]:
            raise ValueError(
                f"dataset contains an episode from another collection: {path} "
                f"({episode_collection!r})"
            )
        if payload.get("robodojo_collection_plan_hash") != manifest["plan_hash"]:
            raise ValueError(f"episode has a mismatched collection plan hash: {path}")
        plan_index = _non_negative_int(
            payload.get("robodojo_collection_plan_index"),
            "episode plan index",
        )
        expected = expected_entries.get(plan_index)
        if expected is None:
            raise ValueError(f"episode references unknown plan index {plan_index}: {path}")
        actual_identity = {
            "eval_seed": _non_negative_int(
                payload.get("robodojo_eval_seed"), "episode eval seed"
            ),
            "layout_id": _non_negative_int(
                payload.get("robodojo_layout_id"), "episode layout id"
            ),
        }
        if actual_identity != {
            "eval_seed": expected["eval_seed"],
            "layout_id": expected["layout_id"],
        }:
            raise ValueError(
                f"episode identity does not match plan index {plan_index}: {path}"
            )
        expected_policy_seed = (
            expected["eval_seed"]
            if manifest["policy_seed"]["mode"] == "eval_seed"
            else manifest["policy_seed"]["value"]
        )
        actual_policy_seed = _non_negative_int(
            payload.get("robodojo_policy_seed"), "episode policy seed"
        )
        if actual_policy_seed != expected_policy_seed:
            raise ValueError(
                f"episode policy seed {actual_policy_seed} does not match signed "
                f"plan value {expected_policy_seed}: {path}"
            )
        if plan_index in completed:
            raise ValueError(f"duplicate committed plan index {plan_index}: {path}")
        _validate_replay_files(
            root,
            path,
            payload,
            fps=int(manifest["fps"]),
        )
        completed.add(plan_index)
    expected_episode_indices = set(range(total_episodes))
    if episode_indices != expected_episode_indices:
        raise ValueError(
            "LeRobot episode count and RoboDojo metadata are not an atomic "
            f"prefix (LeRobot={total_episodes}, metadata={sorted(episode_indices)}). "
            "This usually means the previous process died during commit; refusing "
            "automatic resume."
        )
    return completed


def remaining_entries(
    manifest: dict[str, Any], completed: set[int]
) -> list[RolloutPlanEntry]:
    return [
        RolloutPlanEntry(**entry)
        for entry in manifest["entries"]
        if int(entry["plan_index"]) not in completed
    ]


def group_entries_by_seed(
    entries: Iterable[RolloutPlanEntry],
) -> list[tuple[int, list[RolloutPlanEntry]]]:
    """Split into contiguous seed runs without reordering plan entries."""

    groups: list[tuple[int, list[RolloutPlanEntry]]] = []
    for entry in entries:
        if not groups or groups[-1][0] != entry.eval_seed:
            groups.append((entry.eval_seed, []))
        groups[-1][1].append(entry)
    return groups


def parse_plan_index_map(value: str) -> dict[int, int]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("collection plan-index map is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("collection plan-index map must be a JSON object")
    result: dict[int, int] = {}
    for layout_id, plan_index in payload.items():
        layout = _non_negative_int(layout_id, "layout id")
        index = _non_negative_int(plan_index, "plan index")
        if layout in result:
            raise ValueError(f"duplicate layout id in plan-index map: {layout}")
        result[layout] = index
    return result
