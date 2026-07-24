"""Strict protocol envelope independent of the transport codec."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from src.eval_client.policy_runtime.errors import ErrorCode, ProtocolError
from src.eval_client.policy_runtime.messages import PROTOCOL_VERSION, MessageType

_WIRE_FIELDS = frozenset(
    {
        "protocol_version",
        "message_type",
        "request_id",
        "session_id",
        "episode_id",
        "inference_index",
        "payload",
    }
)

_NO_EPISODE_TYPES = frozenset({MessageType.HELLO, MessageType.HELLO_ACK})
_EPISODE_NO_INFERENCE_TYPES = frozenset(
    {
        MessageType.RESET,
        MessageType.RESET_RESULT,
        MessageType.TRIAL_END,
        MessageType.TRIAL_END_ACK,
    }
)
_EPISODE_WITH_INFERENCE_TYPES = frozenset({MessageType.INFER, MessageType.INFER_RESULT})


def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(
            ErrorCode.INVALID_FRAME,
            f"{field_name} must be a non-empty string",
        )
    return value


def _validate_optional_nonempty_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, field_name)


def _validate_optional_counter(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(
            ErrorCode.INVALID_FRAME,
            f"{field_name} must be null or a non-negative integer",
        )
    return value


def _validate_message_coordinates(
    message_type: MessageType,
    episode_id: str | None,
    inference_index: int | None,
) -> None:
    if message_type in _NO_EPISODE_TYPES:
        if episode_id is not None or inference_index is not None:
            raise ProtocolError(
                ErrorCode.INVALID_FRAME,
                f"{message_type.value} requires null episode_id and inference_index",
            )
        return
    if message_type in _EPISODE_NO_INFERENCE_TYPES:
        if episode_id is None or inference_index is not None:
            raise ProtocolError(
                ErrorCode.INVALID_FRAME,
                f"{message_type.value} requires episode_id and null inference_index",
            )
        return
    if message_type in _EPISODE_WITH_INFERENCE_TYPES:
        if episode_id is None or inference_index is None:
            raise ProtocolError(
                ErrorCode.INVALID_FRAME,
                f"{message_type.value} requires episode_id and inference_index",
            )
        return
    if message_type == MessageType.ERROR and inference_index is not None and episode_id is None:
        raise ProtocolError(
            ErrorCode.INVALID_FRAME,
            "error with inference_index also requires episode_id",
        )


@dataclass(frozen=True, slots=True)
class Frame:
    """One request or response frame.

    Message-specific lifecycle rules are intentionally handled by the session
    state machine, not by this transport-level envelope.
    """

    message_type: MessageType
    request_id: str
    session_id: str
    episode_id: str | None = None
    inference_index: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError(
                ErrorCode.UNSUPPORTED_VERSION,
                f"expected protocol_version={PROTOCOL_VERSION!r}, got {self.protocol_version!r}",
            )
        if not isinstance(self.message_type, MessageType):
            raise ProtocolError(
                ErrorCode.UNKNOWN_MESSAGE_TYPE,
                f"unknown message_type: {self.message_type!r}",
            )
        _require_nonempty_string(self.request_id, "request_id")
        _require_nonempty_string(self.session_id, "session_id")
        _validate_optional_nonempty_string(self.episode_id, "episode_id")
        _validate_optional_counter(self.inference_index, "inference_index")
        _validate_message_coordinates(
            self.message_type,
            self.episode_id,
            self.inference_index,
        )
        if not isinstance(self.payload, Mapping):
            raise ProtocolError(
                ErrorCode.INVALID_FRAME,
                "payload must be a map",
            )
        object.__setattr__(self, "payload", dict(self.payload))

    def to_wire_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "message_type": self.message_type.value,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "inference_index": self.inference_index,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_wire_dict(cls, data: Mapping[str, Any]) -> Self:
        if not isinstance(data, Mapping):
            raise ProtocolError(ErrorCode.INVALID_FRAME, "frame must be a map")

        fields = set(data)
        missing = _WIRE_FIELDS - fields
        if missing:
            raise ProtocolError(
                ErrorCode.INVALID_FRAME,
                f"frame is missing required fields: {sorted(missing)}",
            )
        extra = fields - _WIRE_FIELDS
        if extra:
            rendered_extra = sorted(repr(field_name) for field_name in extra)
            raise ProtocolError(
                ErrorCode.INVALID_FRAME,
                f"frame contains unknown fields: {rendered_extra}",
            )

        protocol_version = data["protocol_version"]
        if protocol_version != PROTOCOL_VERSION:
            raise ProtocolError(
                ErrorCode.UNSUPPORTED_VERSION,
                f"expected protocol_version={PROTOCOL_VERSION!r}, got {protocol_version!r}",
            )

        message_type_value = data["message_type"]
        try:
            message_type = MessageType(message_type_value)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(
                ErrorCode.UNKNOWN_MESSAGE_TYPE,
                f"unknown message_type: {message_type_value!r}",
            ) from exc

        return cls(
            protocol_version=protocol_version,
            message_type=message_type,
            request_id=data["request_id"],
            session_id=data["session_id"],
            episode_id=data["episode_id"],
            inference_index=data["inference_index"],
            payload=data["payload"],
        )
