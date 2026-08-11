"""Crash-safe raw X5 DAgger bundles and deterministic 25 Hz resampling.

The live collector deliberately stores a small *raw bundle* instead of making
LeRobot videos while the operator is waiting.  Each accepted correction is
written into a private ``.partial`` directory, flushed to durable storage, and
then atomically renamed to ``pending/episode_XXXXXXX``.  Consequently a crash
can lose at most the correction currently being committed; prior episode
directories remain independently verifiable and resumable.

This module has no Isaac Sim or LeRobot imports.  It is safe to use from unit
tests, the hardware source, and the later offline replay process.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterator, Mapping
import uuid

import numpy as np


RAW_BUNDLE_FORMAT_VERSION = 1
COLLECTION_FORMAT_VERSION = 1
_EPISODE_RE = re.compile(r"^episode_(\d{7})$")
_SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_PARTIAL_SUFFIX = ".partial"


class RawBundleError(RuntimeError):
    """Raised when a raw bundle cannot be committed or verified safely."""


@dataclass(frozen=True)
class RawBundle:
    """A verified committed bundle returned by :func:`load_pending_bundles`."""

    path: Path
    manifest: dict[str, Any]
    source: dict[str, np.ndarray]
    takeover_state: dict[str, np.ndarray]
    terminal_state: dict[str, np.ndarray]

    @property
    def episode_index(self) -> int:
        return int(self.manifest["episode_index"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json_safe(value: Any, *, label: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item(), label=label)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufUS":
            raise TypeError(f"{label} has unsupported dtype {value.dtype}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"{label} contains non-finite values")
        return value.tolist()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, label=f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if hasattr(value, "items"):
        return _json_safe(dict(value.items()), label=label)
    raise TypeError(f"{label} contains unsupported type {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _normalise_sha256(value: str, *, label: str) -> str:
    match = _SHA256_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"{label} must be a sha256 digest")
    return "sha256:" + match.group(1).lower()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_fsynced(path: Path, value: Any) -> bytes:
    payload = _canonical_json_bytes(value) + b"\n"
    _write_bytes_fsynced(path, payload)
    return payload


def _numeric_state(value: Mapping[str, Any], *, label: str) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty numeric mapping")
    result: dict[str, np.ndarray] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if not key or key in result:
            raise ValueError(f"{label} contains an invalid or duplicate key")
        if hasattr(raw_value, "detach"):
            raw_value = raw_value.detach()
        if hasattr(raw_value, "cpu"):
            raw_value = raw_value.cpu()
        if hasattr(raw_value, "numpy"):
            raw_value = raw_value.numpy()
        array = np.asarray(raw_value)
        if array.dtype.kind not in "biuf":
            raise TypeError(f"{label}.{key} must be numeric, got {array.dtype}")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"{label}.{key} contains non-finite values")
        result[key] = array.copy() if array.ndim == 0 else np.ascontiguousarray(array)
    return result


def _write_npz_fsynced(path: Path, arrays: Mapping[str, Any], *, label: str) -> None:
    numeric = _numeric_state(arrays, label=label)
    with path.open("xb") as stream:
        np.savez(stream, **numeric)
        stream.flush()
        os.fsync(stream.fileno())


def _load_npz(path: Path, *, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                key: (
                    np.asarray(archive[key]).copy()
                    if np.asarray(archive[key]).ndim == 0
                    else np.ascontiguousarray(archive[key])
                )
                for key in archive.files
            }
    except (OSError, ValueError, TypeError) as exc:
        raise RawBundleError(f"cannot load {label} NPZ {path}: {exc}") from exc
    try:
        return _numeric_state(arrays, label=label)
    except (TypeError, ValueError) as exc:
        raise RawBundleError(f"invalid {label} NPZ {path}: {exc}") from exc


def _fragment_descriptor(fragment: Any) -> tuple[Path, str | None, dict[str, Any]]:
    if isinstance(fragment, (str, os.PathLike)):
        return Path(fragment), None, {}
    if isinstance(fragment, Mapping):
        path_value = fragment.get("path")
        expected = fragment.get("sha256", fragment.get("digest"))
        extras = {
            str(key): _json_safe(value, label=f"fragment.{key}")
            for key, value in fragment.items()
            if key not in {"path", "sha256", "digest"}
        }
    else:
        path_value = getattr(fragment, "path", None)
        expected = getattr(fragment, "sha256", getattr(fragment, "digest", None))
        extras = {}
    if path_value is None:
        raise ValueError("raw fragment descriptor has no path")
    expected_digest = (
        None
        if expected is None
        else _normalise_sha256(str(expected), label="raw fragment digest")
    )
    return Path(path_value), expected_digest, extras


def _copy_file_fsynced(source: Path, destination: Path) -> tuple[str, int]:
    if not source.is_file():
        raise FileNotFoundError(f"raw fragment does not exist: {source}")
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        while True:
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            writer.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    return "sha256:" + digest.hexdigest(), byte_count


@contextmanager
def _exclusive_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".raw_bundle.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawBundleError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RawBundleError(f"{label} {path} is not a JSON object")
    return value


def _verified_manifest(directory: Path) -> dict[str, Any]:
    marker_path = directory / "COMMITTED.json"
    manifest_path = directory / "manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        raise RawBundleError(f"incomplete committed raw bundle: {directory}")
    marker = _read_json(marker_path, label="commit marker")
    expected = marker.get("manifest_sha256")
    if expected is None:
        raise RawBundleError(f"commit marker has no manifest digest: {marker_path}")
    expected = _normalise_sha256(expected, label="manifest digest")
    actual = _sha256_file(manifest_path)
    if actual != expected:
        raise RawBundleError(
            f"manifest digest mismatch for {directory}: expected {expected}, got {actual}"
        )
    manifest = _read_json(manifest_path, label="raw bundle manifest")
    if manifest.get("format_version") != RAW_BUNDLE_FORMAT_VERSION:
        raise RawBundleError(f"unsupported raw bundle version in {manifest_path}")
    match = _EPISODE_RE.fullmatch(directory.name)
    if match is None or int(match.group(1)) != manifest.get("episode_index"):
        raise RawBundleError(f"episode directory/index mismatch: {directory}")
    return manifest


class RawBundleStore:
    """Atomic per-episode store with immutable collection identity.

    ``identity`` should include the task, checkpoint provenance, source mapping,
    and any collection settings which must not change across a resumed run.
    Reopening a root with a different identity *or target* fails closed.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        target_episodes: int,
        identity: Mapping[str, Any],
    ) -> None:
        if isinstance(target_episodes, bool) or int(target_episodes) <= 0:
            raise ValueError("target_episodes must be a positive integer")
        if not isinstance(identity, Mapping) or not identity:
            raise ValueError("collection identity must be a non-empty mapping")
        self.root = Path(root).expanduser().resolve()
        self.pending = self.root / "pending"
        self.target_episodes = int(target_episodes)
        self.identity = _json_safe(deepcopy(dict(identity)), label="identity")
        self.identity_sha256 = _sha256_bytes(_canonical_json_bytes(self.identity))

        self.root.mkdir(parents=True, exist_ok=True)
        self.pending.mkdir(exist_ok=True)
        _fsync_directory(self.root)
        with _exclusive_lock(self.root):
            self._lock_collection_identity()

    def _lock_collection_identity(self) -> None:
        path = self.root / "collection.json"
        expected = {
            "format_version": COLLECTION_FORMAT_VERSION,
            "target_episodes": self.target_episodes,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
        }
        if path.exists():
            actual = _read_json(path, label="collection identity")
            if actual != expected:
                raise RawBundleError(
                    "raw collection identity/target does not match the existing store"
                )
            return
        temporary = self.root / f".collection.{uuid.uuid4().hex}{_PARTIAL_SUFFIX}"
        try:
            _write_json_fsynced(temporary, expected)
            os.rename(temporary, path)
            _fsync_directory(self.root)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _committed(self) -> list[tuple[int, Path]]:
        result: list[tuple[int, Path]] = []
        for path in sorted(self.pending.iterdir()):
            if not path.is_dir():
                continue
            match = _EPISODE_RE.fullmatch(path.name)
            if match is None:
                # A crash may leave .partial directories.  They are never data.
                continue
            manifest = _verified_manifest(path)
            _verify_payload_files(path, manifest)
            result.append((int(match.group(1)), path))
        indices = [index for index, _ in result]
        if len(indices) != len(set(indices)):
            raise RawBundleError("duplicate committed episode indices")
        return result

    @property
    def completed_count(self) -> int:
        with _exclusive_lock(self.root):
            return len(self._committed())

    @property
    def target_reached(self) -> bool:
        return self.completed_count >= self.target_episodes

    @property
    def remaining_count(self) -> int:
        return max(0, self.target_episodes - self.completed_count)

    def commit(
        self,
        fragment: Any,
        snapshotter: Any,
        takeover_state: Mapping[str, Any],
        terminal_state: Mapping[str, Any],
        sim_anchor: Any,
        source_anchor: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Durably commit one raw episode and return its final directory.

        ``fragment`` is either an NPZ path or a descriptor containing ``path``
        and an optional ``sha256``/``digest``.  The source file is copied; it is
        never deleted by a successful commit.  ``snapshotter.metadata()`` plus
        the explicitly captured takeover/terminal states make later replay
        independent of the live Isaac process.
        """

        source_path, expected_source_digest, fragment_metadata = _fragment_descriptor(
            fragment
        )
        if snapshotter is None or not callable(getattr(snapshotter, "metadata", None)):
            raise TypeError("snapshotter must provide metadata()")
        snapshot_metadata = _json_safe(
            snapshotter.metadata(), label="snapshotter metadata"
        )
        replay_metadata = _json_safe(metadata or {}, label="episode metadata")
        sim_anchor_json = _json_safe(sim_anchor, label="sim anchor")
        source_anchor_json = _json_safe(source_anchor, label="source anchor")

        with _exclusive_lock(self.root):
            committed = self._committed()
            if len(committed) >= self.target_episodes:
                raise RawBundleError(
                    f"raw collection already reached target {self.target_episodes}"
                )
            episode_index = 0 if not committed else max(item[0] for item in committed) + 1
            final_directory = self.pending / f"episode_{episode_index:07d}"
            if final_directory.exists():
                raise RawBundleError(f"episode destination already exists: {final_directory}")
            partial = self.pending / (
                f".{final_directory.name}.{os.getpid()}.{uuid.uuid4().hex}{_PARTIAL_SUFFIX}"
            )
            partial.mkdir(mode=0o750)
            try:
                source_destination = partial / "source.npz"
                source_digest, source_size = _copy_file_fsynced(
                    source_path, source_destination
                )
                if (
                    expected_source_digest is not None
                    and source_digest != expected_source_digest
                ):
                    raise RawBundleError(
                        "raw fragment digest mismatch: expected "
                        f"{expected_source_digest}, got {source_digest}"
                    )
                # Fail during collection instead of hours later in offline replay.
                source_arrays = _load_npz(source_destination, label="source")
                if not source_arrays:
                    raise RawBundleError("raw fragment NPZ contains no arrays")

                takeover_path = partial / "takeover_state.npz"
                terminal_path = partial / "terminal_state.npz"
                _write_npz_fsynced(
                    takeover_path, takeover_state, label="takeover simulator state"
                )
                _write_npz_fsynced(
                    terminal_path, terminal_state, label="terminal simulator state"
                )

                files = {}
                for name, path in (
                    ("source", source_destination),
                    ("takeover_state", takeover_path),
                    ("terminal_state", terminal_path),
                ):
                    files[name] = {
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                # Preserve descriptor facts such as source sample_count, but the
                # copied file's size/digest above remain authoritative.
                files["source"]["descriptor"] = fragment_metadata

                manifest = {
                    "format_version": RAW_BUNDLE_FORMAT_VERSION,
                    "episode_index": episode_index,
                    "created_at": _utc_now(),
                    "collection_identity_sha256": self.identity_sha256,
                    "files": files,
                    "snapshot": snapshot_metadata,
                    "sim_anchor": sim_anchor_json,
                    "source_anchor": source_anchor_json,
                    "metadata": replay_metadata,
                }
                manifest_payload = _write_json_fsynced(
                    partial / "manifest.json", manifest
                )
                marker = {
                    "format_version": RAW_BUNDLE_FORMAT_VERSION,
                    "committed_at": _utc_now(),
                    "manifest_sha256": _sha256_bytes(manifest_payload),
                }
                _write_json_fsynced(partial / "COMMITTED.json", marker)
                _fsync_directory(partial)
                os.rename(partial, final_directory)
                _fsync_directory(self.pending)
                return final_directory
            except BaseException:
                # The private directory was never visible as a committed episode.
                # Best-effort cleanup; a power loss may still leave it behind and
                # the scanner intentionally ignores such .partial directories.
                if partial.exists():
                    shutil.rmtree(partial, ignore_errors=True)
                    _fsync_directory(self.pending)
                raise

    def discard(
        self,
        fragment: Any | None = None,
        *,
        delete_fragment: bool = False,
    ) -> None:
        """Discard an unaccepted attempt without changing committed progress.

        Stale private ``.partial`` directories are removed.  An external source
        fragment is only unlinked when ``delete_fragment=True`` is explicit;
        successful commits always retain the caller-owned original.
        """

        source_path = None
        if fragment is not None:
            source_path, _, _ = _fragment_descriptor(fragment)
        with _exclusive_lock(self.root):
            for path in self.pending.iterdir():
                if path.is_dir() and path.name.endswith(_PARTIAL_SUFFIX):
                    shutil.rmtree(path, ignore_errors=True)
            _fsync_directory(self.pending)
        if delete_fragment and source_path is not None and source_path.exists():
            if not source_path.is_file():
                raise RawBundleError(f"refusing to discard non-file fragment: {source_path}")
            source_path.unlink()
            _fsync_directory(source_path.parent)


def _verify_payload_files(directory: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise RawBundleError(f"raw bundle manifest has no files table: {directory}")
    expected_names = {"source", "takeover_state", "terminal_state"}
    if set(files) != expected_names:
        raise RawBundleError(f"unexpected raw bundle files table: {directory}")
    for name in sorted(expected_names):
        descriptor = files[name]
        if not isinstance(descriptor, Mapping):
            raise RawBundleError(f"invalid {name} descriptor in {directory}")
        relative = descriptor.get("path")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise RawBundleError(f"unsafe {name} path in {directory}")
        path = directory / relative
        if not path.is_file():
            raise RawBundleError(f"missing {name} payload in {directory}")
        expected_size = descriptor.get("bytes")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or path.stat().st_size != expected_size
        ):
            raise RawBundleError(f"{name} byte-size mismatch in {directory}")
        expected_digest = _normalise_sha256(
            descriptor.get("sha256", ""), label=f"{name} digest"
        )
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise RawBundleError(
                f"{name} digest mismatch in {directory}: expected "
                f"{expected_digest}, got {actual_digest}"
            )


def load_pending_bundles(
    root: str | os.PathLike[str],
    *,
    verify: bool = True,
) -> list[RawBundle]:
    """Load committed bundles in episode order; ignore all partial attempts."""

    root_path = Path(root).expanduser().resolve()
    collection_path = root_path / "collection.json"
    pending = root_path / "pending"
    collection = _read_json(collection_path, label="collection identity")
    if collection.get("format_version") != COLLECTION_FORMAT_VERSION:
        raise RawBundleError(f"unsupported collection version in {collection_path}")
    identity = collection.get("identity")
    expected_identity_digest = _normalise_sha256(
        collection.get("identity_sha256", ""), label="collection identity digest"
    )
    actual_identity_digest = _sha256_bytes(_canonical_json_bytes(identity))
    if actual_identity_digest != expected_identity_digest:
        raise RawBundleError("collection identity digest mismatch")
    if not pending.is_dir():
        raise RawBundleError(f"raw collection has no pending directory: {pending}")

    bundles: list[RawBundle] = []
    for directory in sorted(pending.iterdir()):
        if not directory.is_dir() or _EPISODE_RE.fullmatch(directory.name) is None:
            continue
        manifest = _verified_manifest(directory)
        if manifest.get("collection_identity_sha256") != expected_identity_digest:
            raise RawBundleError(f"collection identity mismatch in {directory}")
        if verify:
            _verify_payload_files(directory, manifest)
        files = manifest["files"]
        bundles.append(
            RawBundle(
                path=directory,
                manifest=manifest,
                source=_load_npz(directory / files["source"]["path"], label="source"),
                takeover_state=_load_npz(
                    directory / files["takeover_state"]["path"],
                    label="takeover simulator state",
                ),
                terminal_state=_load_npz(
                    directory / files["terminal_state"]["path"],
                    label="terminal simulator state",
                ),
            )
        )
    bundles.sort(key=lambda item: item.episode_index)
    return bundles


_TIMESTAMP_KEYS = (
    "sample_monotonic_ns",
    "timestamps_ns",
    "timestamp_ns",
    "sample_monotonic_s",
    "timestamps_s",
    "timestamp_s",
)


def _source_arrays(bundle: RawBundle | Mapping[str, Any]) -> dict[str, np.ndarray]:
    source = bundle.source if isinstance(bundle, RawBundle) else bundle
    if not isinstance(source, Mapping) or not source:
        raise ValueError("raw bundle/source must be a non-empty mapping")
    return _numeric_state(source, label="raw source")


def resample_segment_25hz(
    bundle: RawBundle | Mapping[str, Any],
    fps: int = 25,
) -> dict[str, np.ndarray]:
    """Linearly resample a raw hardware segment on a fixed wall-time grid.

    Floating point sample arrays are interpolated along axis 0.  Integral and
    boolean sample arrays use zero-order hold because they generally encode
    sequence numbers or discrete states.  Non-sample arrays are copied.
    ``timestamp_s`` is relative to the first hardware sample and is always
    included in the result.  No extrapolation beyond the last raw sample is
    performed.
    """

    if isinstance(fps, bool) or int(fps) <= 0:
        raise ValueError("fps must be a positive integer")
    fps = int(fps)
    source = _source_arrays(bundle)
    timestamp_key = next((key for key in _TIMESTAMP_KEYS if key in source), None)
    if timestamp_key is None:
        raise ValueError(
            "raw source has no supported hardware timestamp key: "
            f"{', '.join(_TIMESTAMP_KEYS)}"
        )
    raw_timestamp = np.asarray(source[timestamp_key])
    if raw_timestamp.ndim != 1 or raw_timestamp.size < 2:
        raise ValueError("raw hardware timestamps must be a 1-D array with >=2 samples")
    if timestamp_key.endswith("_ns"):
        # Subtract while still integral so large monotonic-ns origins do not
        # lose sub-microsecond precision during an early float64 conversion.
        elapsed = (raw_timestamp - raw_timestamp[0]).astype(np.float64) * 1e-9
    else:
        elapsed = raw_timestamp.astype(np.float64) - float(raw_timestamp[0])
    if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed) <= 0.0):
        raise ValueError("raw hardware timestamps must be finite and strictly increasing")

    duration = float(elapsed[-1])
    output_count = int(math.floor(duration * fps + 1e-9)) + 1
    output_time = np.arange(output_count, dtype=np.float64) / float(fps)
    # Floating point rounding must never push the final grid point out of range.
    output_time = output_time[output_time <= duration + 1e-12]
    result: dict[str, np.ndarray] = {"timestamp_s": output_time}

    hold_indices = np.searchsorted(elapsed, output_time, side="right") - 1
    hold_indices = np.clip(hold_indices, 0, raw_timestamp.size - 1)
    for key, value in source.items():
        if key == timestamp_key:
            continue
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != raw_timestamp.size:
            result[key] = array.copy() if array.ndim == 0 else np.ascontiguousarray(array)
            continue
        if array.dtype.kind == "f":
            flat = array.reshape(raw_timestamp.size, -1).astype(np.float64)
            interpolated = np.empty((output_time.size, flat.shape[1]), dtype=np.float64)
            for column in range(flat.shape[1]):
                interpolated[:, column] = np.interp(
                    output_time, elapsed, flat[:, column]
                )
            result[key] = np.ascontiguousarray(
                interpolated.reshape((output_time.size,) + array.shape[1:])
            )
        else:
            result[key] = np.ascontiguousarray(array[hold_indices])
    return result


__all__ = [
    "RawBundle",
    "RawBundleError",
    "RawBundleStore",
    "load_pending_bundles",
    "resample_segment_25hz",
]
