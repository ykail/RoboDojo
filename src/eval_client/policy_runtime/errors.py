"""Errors carried by the RoboDojo policy protocol."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INVALID_FRAME = "invalid_frame"
    UNSUPPORTED_VERSION = "unsupported_version"
    UNKNOWN_MESSAGE_TYPE = "unknown_message_type"
    INVALID_STATE = "invalid_state"
    SESSION_BUSY = "session_busy"
    EPISODE_MISMATCH = "episode_mismatch"
    INFERENCE_INDEX_MISMATCH = "inference_index_mismatch"
    EPISODE_LOST = "episode_lost"
    TIMEOUT = "timeout"
    INFER_FAILED = "infer_failed"
    RESET_FAILED = "reset_failed"
    INTERNAL = "internal"


class ProtocolError(ValueError):
    """A protocol failure with a stable machine-readable error code."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})
