"""Load and resume an ordered restored-recovery collection plan."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecoveryQueueItem:
    queue_id: str
    source_id: str
    dataset_root: Path
    episode_index: int
    time_s: float


@dataclass(frozen=True)
class RecoveryQueuePlan:
    path: Path
    sha256: str
    task_name: str
    env_config: str
    output_root: Path
    output_repo_id: str
    items: tuple[RecoveryQueueItem, ...]

    @property
    def output_path(self) -> Path:
        return self.output_root / self.output_repo_id


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read recovery queue manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("recovery queue manifest must contain a JSON object")
    return raw, payload


def load_recovery_queue(path_value: str | Path) -> RecoveryQueuePlan:
    path = Path(path_value).expanduser().resolve(strict=True)
    raw, payload = _read_object(path)
    if payload.get("format") != "robodojo_restore_batch_v1":
        raise ValueError(f"unsupported recovery queue format: {payload.get('format')!r}")
    task_name = str(payload.get("task", "")).strip()
    env_config = str(payload.get("env_config", "")).strip()
    output = payload.get("output")
    sources = payload.get("sources")
    if not task_name or not env_config or not isinstance(output, dict):
        raise ValueError("recovery queue requires task, env_config, and output")
    if not isinstance(sources, list) or not sources:
        raise ValueError("recovery queue requires at least one source")
    output_root = Path(str(output.get("root", ""))).expanduser()
    output_repo_id = str(output.get("repo_id", "")).strip()
    if not output_root.is_absolute() or not output_repo_id or "/" in output_repo_id:
        raise ValueError("recovery queue output must use an absolute root and child repo_id")

    result: list[RecoveryQueueItem] = []
    source_ids: set[str] = set()
    queue_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each recovery queue source must be an object")
        source_id = str(source.get("id", "")).strip()
        dataset_root = Path(str(source.get("dataset_root", ""))).expanduser()
        items = source.get("items")
        if not source_id or source_id in source_ids:
            raise ValueError(f"invalid or duplicate recovery source id: {source_id!r}")
        if not dataset_root.is_absolute() or not isinstance(items, list) or not items:
            raise ValueError(f"source {source_id!r} requires an absolute dataset_root and items")
        source_ids.add(source_id)
        for raw_item in items:
            if not isinstance(raw_item, list) or len(raw_item) != 2:
                raise ValueError(f"source {source_id!r} items must be [episode, time_s]")
            if not isinstance(raw_item[0], int) or isinstance(raw_item[0], bool):
                raise ValueError(f"episode must be an integer in {raw_item!r}")
            episode_index = raw_item[0]
            time_s = float(raw_item[1])
            if (
                isinstance(raw_item[1], bool)
                or episode_index < 0
                or not math.isfinite(time_s)
                or time_s < 0.0
            ):
                raise ValueError(f"invalid recovery item {raw_item!r} in {source_id!r}")
            queue_id = f"{source_id}:e{episode_index:07d}:t{time_s:.3f}"
            if queue_id in queue_ids:
                raise ValueError(f"duplicate recovery queue item: {queue_id}")
            queue_ids.add(queue_id)
            result.append(
                RecoveryQueueItem(
                    queue_id=queue_id,
                    source_id=source_id,
                    dataset_root=dataset_root,
                    episode_index=episode_index,
                    time_s=time_s,
                )
            )
    return RecoveryQueuePlan(
        path=path,
        sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
        task_name=task_name,
        env_config=env_config,
        output_root=output_root,
        output_repo_id=output_repo_id,
        items=tuple(result),
    )


def completed_queue_ids(plan: RecoveryQueuePlan) -> set[str]:
    """Return only commits whose full source lineage matches this exact plan."""

    metadata_dir = plan.output_path / "meta" / "robodojo" / "episodes"
    if not metadata_dir.is_dir():
        return set()
    expected = {item.queue_id: item for item in plan.items}
    result: set[str] = set()
    for path in sorted(metadata_dir.glob("episode_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            source = payload.get("robodojo_recovery_source", {})
            queue_id = source.get("queue_id") if isinstance(source, dict) else None
            item = expected.get(queue_id)
            source_root = Path(str(source.get("dataset_root", ""))).expanduser()
            lineage_matches = (
                item is not None
                and source.get("queue_manifest_sha256") == plan.sha256
                and source.get("source_id") == item.source_id
                and source_root == item.dataset_root
                and int(source.get("episode", -1)) == item.episode_index
                and math.isclose(
                    float(source.get("requested_time_s", -1.0)),
                    item.time_s,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError):
            continue
        if isinstance(queue_id, str) and queue_id and lineage_matches:
            result.add(queue_id)
    return result
