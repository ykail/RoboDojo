"""Message vocabulary for ``robodojo-policy-v1``."""

from __future__ import annotations

from enum import StrEnum

PROTOCOL_VERSION = "robodojo-policy-v1"


class MessageType(StrEnum):
    HELLO = "hello"
    HELLO_ACK = "hello_ack"
    RESET = "reset"
    RESET_RESULT = "reset_result"
    INFER = "infer"
    INFER_RESULT = "infer_result"
    TRIAL_END = "trial_end"
    TRIAL_END_ACK = "trial_end_ack"
    ERROR = "error"


REQUEST_RESPONSE_PAIRS: dict[MessageType, MessageType] = {
    MessageType.HELLO: MessageType.HELLO_ACK,
    MessageType.RESET: MessageType.RESET_RESULT,
    MessageType.INFER: MessageType.INFER_RESULT,
    MessageType.TRIAL_END: MessageType.TRIAL_END_ACK,
}

REQUEST_TYPES = frozenset(REQUEST_RESPONSE_PAIRS)
RESPONSE_TYPES = frozenset(REQUEST_RESPONSE_PAIRS.values()) | frozenset({MessageType.ERROR})
