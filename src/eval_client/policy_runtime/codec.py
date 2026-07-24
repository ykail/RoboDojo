"""Binary msgpack codec for policy frames and NumPy payloads.

The NumPy wire representation intentionally matches Kai0's
``openpi_client.msgpack_numpy`` format. Both repositories must keep the same
cross-end fixtures; independent round-trip tests are not sufficient.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import msgpack
import numpy as np

from src.eval_client.policy_runtime.errors import ErrorCode, ProtocolError
from src.eval_client.policy_runtime.frame import Frame

_NDARRAY_MARKER = b"__ndarray__"
_NPGENERIC_MARKER = b"__npgeneric__"
_UNSUPPORTED_DTYPE_KINDS = frozenset({"V", "O", "c"})

MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_NUMPY_RANK = 8
MAX_NUMPY_DIM = 16_384
MAX_NUMPY_ELEMENTS = MAX_FRAME_BYTES
_MAX_MAP_ITEMS = 100_000
_MAX_ARRAY_ITEMS = 1_000_000
_MAX_STRING_BYTES = 1_000_000
_MAX_CONTAINER_DEPTH = 32


def _ensure_supported_dtype(dtype: np.dtype[Any]) -> None:
    if dtype.kind in _UNSUPPORTED_DTYPE_KINDS:
        raise ValueError(f"unsupported NumPy dtype: {dtype}")


def _validate_value(value: Any, *, depth: int = 0) -> None:
    if depth > _MAX_CONTAINER_DEPTH:
        raise ValueError(f"payload nesting exceeds {_MAX_CONTAINER_DEPTH}")
    if isinstance(value, np.ndarray):
        _ensure_supported_dtype(value.dtype)
        _validate_shape(value.shape)
        return
    if isinstance(value, np.generic):
        _ensure_supported_dtype(value.dtype)
        return
    if isinstance(value, Mapping):
        if len(value) > _MAX_MAP_ITEMS:
            raise ValueError(f"map item count exceeds {_MAX_MAP_ITEMS}")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("protocol map keys must be strings")
            if len(key.encode("utf-8")) > _MAX_STRING_BYTES:
                raise ValueError(f"map key exceeds {_MAX_STRING_BYTES} encoded bytes")
            _validate_value(item, depth=depth + 1)
        return
    if isinstance(value, list):
        if len(value) > _MAX_ARRAY_ITEMS:
            raise ValueError(f"list item count exceeds {_MAX_ARRAY_ITEMS}")
        for item in value:
            _validate_value(item, depth=depth + 1)
        return
    if isinstance(value, tuple):
        raise ValueError("protocol payloads use lists, not tuples")
    if isinstance(value, str):
        if len(value.encode("utf-8")) > _MAX_STRING_BYTES:
            raise ValueError(f"string exceeds {_MAX_STRING_BYTES} encoded bytes")
        return
    if isinstance(value, bytes):
        if len(value) > MAX_FRAME_BYTES:
            raise ValueError(f"bytes value exceeds {MAX_FRAME_BYTES} byte limit")
        return
    if value is None or isinstance(value, bool | int | float):
        return
    raise TypeError(f"unsupported protocol value type: {type(value).__name__}")


def _pack_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        _ensure_supported_dtype(obj.dtype)
        shape = _validate_shape(obj.shape)
        return {
            _NDARRAY_MARKER: True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": shape,
        }
    if isinstance(obj, np.generic):
        _ensure_supported_dtype(obj.dtype)
        return {
            _NPGENERIC_MARKER: True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    raise TypeError(f"cannot msgpack value of type {type(obj).__name__}")


def _decode_dtype(value: Any) -> np.dtype[Any]:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid NumPy dtype: {value!r}") from exc
    _ensure_supported_dtype(dtype)
    return dtype


def _validate_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("NumPy shape must be a list or tuple")
    if len(value) > MAX_NUMPY_RANK:
        raise ValueError(f"NumPy rank exceeds {MAX_NUMPY_RANK}")
    shape: list[int] = []
    for dim in value:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(f"invalid NumPy shape dimension: {dim!r}")
        if dim > MAX_NUMPY_DIM:
            raise ValueError(f"NumPy shape dimension exceeds {MAX_NUMPY_DIM}")
        shape.append(dim)
    result = tuple(shape)
    if math.prod(result) > MAX_NUMPY_ELEMENTS:
        raise ValueError(f"NumPy element count exceeds {MAX_NUMPY_ELEMENTS}")
    return result


def _unpack_numpy(obj: dict[Any, Any]) -> Any:
    if obj.get(_NDARRAY_MARKER) is True:
        required = {_NDARRAY_MARKER, b"data", b"dtype", b"shape"}
        if set(obj) != required:
            raise ValueError("invalid NumPy array envelope")
        data = obj[b"data"]
        if not isinstance(data, bytes):
            raise ValueError("NumPy array data must be bytes")
        dtype = _decode_dtype(obj[b"dtype"])
        shape = _validate_shape(obj[b"shape"])
        expected_nbytes = math.prod(shape) * dtype.itemsize
        if len(data) != expected_nbytes:
            raise ValueError("NumPy array byte length does not match dtype and shape")
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

    if obj.get(_NPGENERIC_MARKER) is True:
        required = {_NPGENERIC_MARKER, b"data", b"dtype"}
        if set(obj) != required:
            raise ValueError("invalid NumPy scalar envelope")
        dtype = _decode_dtype(obj[b"dtype"])
        scalar = dtype.type(obj[b"data"])
        if not isinstance(scalar, np.generic):
            raise ValueError("NumPy scalar envelope data is not scalar")
        if scalar.dtype != dtype:
            raise ValueError("NumPy scalar data does not match declared dtype")
        return scalar

    return obj


def encode_frame(frame: Frame) -> bytes:
    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be Frame, got {type(frame).__name__}")
    try:
        wire = frame.to_wire_dict()
        _validate_value(wire)
        encoded = msgpack.packb(
            wire,
            default=_pack_numpy,
            use_bin_type=True,
        )
        if len(encoded) > MAX_FRAME_BYTES:
            raise ValueError(f"encoded frame exceeds {MAX_FRAME_BYTES} byte limit")
        return encoded
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError(
            ErrorCode.INVALID_FRAME,
            f"msgpack encode failed: {exc}",
        ) from exc


def decode_frame(data: bytes | bytearray) -> Frame:
    if not isinstance(data, bytes | bytearray):
        raise TypeError(f"encoded frame must be bytes, got {type(data).__name__}")
    if len(data) > MAX_FRAME_BYTES:
        raise ProtocolError(
            ErrorCode.INVALID_FRAME,
            f"encoded frame exceeds {MAX_FRAME_BYTES} byte limit",
        )
    try:
        wire = msgpack.unpackb(
            bytes(data),
            raw=False,
            object_hook=_unpack_numpy,
            max_str_len=_MAX_STRING_BYTES,
            max_bin_len=MAX_FRAME_BYTES,
            max_array_len=_MAX_ARRAY_ITEMS,
            max_map_len=_MAX_MAP_ITEMS,
            max_ext_len=0,
        )
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError(
            ErrorCode.INVALID_FRAME,
            f"msgpack decode failed: {exc}",
        ) from exc
    if not isinstance(wire, Mapping):
        raise ProtocolError(ErrorCode.INVALID_FRAME, "frame must be a map")
    frame = Frame.from_wire_dict(wire)
    try:
        _validate_value(frame.to_wire_dict())
    except Exception as exc:
        raise ProtocolError(
            ErrorCode.INVALID_FRAME,
            f"decoded frame value validation failed: {exc}",
        ) from exc
    return frame
