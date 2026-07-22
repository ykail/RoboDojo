"""Small, dependency-free IPC protocol used by the LeRobot stream writer.

Messages are pickled because a frame contains several NumPy arrays.  A fixed
length prefix keeps the pipe self-delimiting, and the sender always waits for
an acknowledgement before sending another frame.  Consequently a slow video
encoder applies backpressure to the Isaac process instead of growing an
unbounded in-memory queue.
"""

from __future__ import annotations

import pickle
import os
import select
import struct
import time
from typing import Any, BinaryIO, Callable


_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 512 * 1024 * 1024


class ProtocolError(RuntimeError):
    """The sidecar pipe contained an invalid or truncated message."""


def _read_exact(
    stream: BinaryIO,
    size: int,
    *,
    deadline: float | None = None,
    health_check: Callable[[], None] | None = None,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    file_descriptor: int | None = None
    if deadline is not None:
        try:
            file_descriptor = stream.fileno()
        except (AttributeError, OSError):
            # BytesIO and similar test streams are immediately readable.
            file_descriptor = None
    while remaining:
        if file_descriptor is not None:
            while True:
                if health_check is not None:
                    health_check()
                wait_s = deadline - time.monotonic()
                if wait_s <= 0:
                    raise TimeoutError("Timed out waiting for LeRobot writer response")
                readable, _, _ = select.select(
                    [file_descriptor], [], [], min(wait_s, 0.25)
                )
                if readable:
                    break
            chunk = os.read(file_descriptor, remaining)
        else:
            chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("LeRobot writer pipe closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(stream: BinaryIO, message: Any) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ProtocolError(
            f"LeRobot IPC message is too large: {len(payload)} bytes "
            f"(limit {_MAX_MESSAGE_BYTES})"
        )
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def receive_message(
    stream: BinaryIO,
    *,
    timeout_s: float | None = None,
    health_check: Callable[[], None] | None = None,
) -> Any:
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    try:
        (size,) = _HEADER.unpack(
            _read_exact(
                stream,
                _HEADER.size,
                deadline=deadline,
                health_check=health_check,
            )
        )
    except struct.error as exc:  # Defensive: _read_exact should make this impossible.
        raise ProtocolError("Invalid LeRobot IPC header") from exc
    if size > _MAX_MESSAGE_BYTES:
        raise ProtocolError(
            f"LeRobot IPC payload claims {size} bytes "
            f"(limit {_MAX_MESSAGE_BYTES})"
        )
    try:
        return pickle.loads(
            _read_exact(
                stream,
                size,
                deadline=deadline,
                health_check=health_check,
            )
        )
    except (pickle.PickleError, AttributeError, ValueError, TypeError) as exc:
        raise ProtocolError("Invalid LeRobot IPC payload") from exc
