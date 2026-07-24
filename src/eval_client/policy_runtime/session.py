"""Transport-independent reference state machine for policy-server sessions.

The WebSocket dispatcher and model adapter intentionally live outside this
module.  A dispatcher validates a request with :meth:`PolicySession.begin`,
runs the corresponding backend operation, and then calls ``complete`` or
``fail`` with the returned token. It attempts the indicated reply before
submitting another frame from that connection.

This two-phase API matters for synchronous model inference: disconnecting or
cancelling an outer asyncio task does not stop the worker thread.  The session
therefore remains in ``DRAINING`` until that exact operation settles, and its
result is never sent to the disconnected client.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from src.eval_client.policy_runtime.errors import ErrorCode, ProtocolError
from src.eval_client.policy_runtime.frame import Frame
from src.eval_client.policy_runtime.messages import REQUEST_TYPES, MessageType


class SessionPhase(StrEnum):
    """Stable lifecycle phases for one WebSocket connection."""

    AWAITING_HELLO = "awaiting_hello"
    READY = "ready"
    ACTIVE = "active"
    DRAINING = "draining"
    TERMINATING = "terminating"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class OperationToken:
    """Capability returned for one accepted request.

    ``generation`` is process-local fencing metadata and is never serialized
    on the wire.
    """

    generation: int
    message_type: MessageType
    request_id: str
    session_id: str
    episode_id: str | None
    inference_index: int | None


@dataclass(frozen=True, slots=True)
class OperationResult:
    """Instructions for the dispatcher after an operation settles.

    ``operation_obsolete`` means a disconnect made any backend output stale;
    it must not be encoded or sent. ``close_after_reply`` is used for a final
    correlated ERROR: the transport is still present, so the session remains
    ``TERMINATING`` until the close attempt calls ``disconnect``.
    ``reply_allowed`` requires one reply attempt; it is not permission to retry.
    These values are a settlement-time snapshot, so a later send can still
    fail and must be followed by ``disconnect``.
    """

    phase: SessionPhase
    reply_allowed: bool
    close_after_reply: bool
    operation_obsolete: bool
    episode_lost: bool


@dataclass(frozen=True, slots=True)
class DisconnectResult:
    """What cleanup code must do after the transport disconnects.

    ``episode_lost`` means server-side episode state may still require abort or
    cleanup. It does not mean the client received the final acknowledgement.
    """

    phase: SessionPhase
    waiting_for_operation: bool
    episode_lost: bool


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Read-only state intended for logging and tests."""

    phase: SessionPhase
    session_id: str | None
    active_episode_id: str | None
    next_inference_index: int | None
    pending: OperationToken | None
    episode_lost: bool
    seen_request_count: int
    seen_episode_count: int


class SessionInvariantError(RuntimeError):
    """The dispatcher attempted to settle an unknown or stale operation."""


class PolicySession:
    """Validate and advance one server-side ``robodojo-policy-v1`` session.

    The class is thread-safe, but deliberately performs no I/O and invokes no
    model code.  At most one backend operation may be pending at a time.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._phase = SessionPhase.AWAITING_HELLO
        self._session_id: str | None = None
        self._active_episode_id: str | None = None
        self._next_inference_index: int | None = None
        self._pending: OperationToken | None = None
        self._episode_lost = False
        self._generation = 0
        self._seen_request_ids: set[str] = set()
        self._seen_episode_ids: set[str] = set()

    @property
    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            return SessionSnapshot(
                phase=self._phase,
                session_id=self._session_id,
                active_episode_id=self._active_episode_id,
                next_inference_index=self._next_inference_index,
                pending=self._pending,
                episode_lost=self._episode_lost,
                seen_request_count=len(self._seen_request_ids),
                seen_episode_count=len(self._seen_episode_ids),
            )

    def begin(self, frame: Frame) -> OperationToken:
        """Validate and reserve one request before any backend side effect.

        A request ID from the bound session is burned as soon as the request is
        received, including when later lifecycle validation rejects it. When a
        RESET reaches the READY reservation branch, its candidate episode ID
        is also burned before payload validation and is never rolled back. A
        wrong-session frame is rejected before either reservation because it
        does not belong to this session.
        """

        with self._lock:
            if self._phase in {
                SessionPhase.DRAINING,
                SessionPhase.TERMINATING,
                SessionPhase.CLOSED,
            }:
                self._invalid_state(f"session is {self._phase.value}")

            if self._phase == SessionPhase.AWAITING_HELLO:
                if self._pending is None:
                    return self._begin_hello(frame)
                if frame.session_id != self._session_id:
                    raise ProtocolError(
                        ErrorCode.INVALID_STATE,
                        "request session_id does not match the pending hello",
                        details={
                            "expected_session_id": self._session_id,
                            "actual_session_id": frame.session_id,
                        },
                    )
                self._reserve_request_id(frame.request_id)
                self._invalid_state("hello request is still pending")

            if frame.session_id != self._session_id:
                raise ProtocolError(
                    ErrorCode.INVALID_STATE,
                    "request session_id does not match the bound connection",
                    details={
                        "expected_session_id": self._session_id,
                        "actual_session_id": frame.session_id,
                    },
                )

            self._reserve_request_id(frame.request_id)

            if frame.message_type not in REQUEST_TYPES:
                self._invalid_state(f"{frame.message_type.value} is not a client request")
            if self._pending is not None:
                self._invalid_state(
                    f"{self._pending.message_type.value} request is still pending",
                )

            if self._phase == SessionPhase.READY:
                if frame.message_type != MessageType.RESET:
                    self._invalid_state("ready session accepts only reset")
                if frame.episode_id in self._seen_episode_ids:
                    self._invalid_state(
                        f"episode_id {frame.episode_id!r} was already used in this session",
                    )
                self._seen_episode_ids.add(frame.episode_id)
                return self._new_token(frame)

            if self._phase != SessionPhase.ACTIVE:
                raise SessionInvariantError(f"unhandled session phase: {self._phase}")

            if frame.message_type not in {MessageType.INFER, MessageType.TRIAL_END}:
                self._invalid_state("active session accepts only infer or trial_end")
            if frame.episode_id != self._active_episode_id:
                raise ProtocolError(
                    ErrorCode.EPISODE_MISMATCH,
                    "request episode_id does not match the active episode",
                    details={
                        "expected_episode_id": self._active_episode_id,
                        "actual_episode_id": frame.episode_id,
                    },
                )
            if frame.message_type == MessageType.INFER and frame.inference_index != self._next_inference_index:
                raise ProtocolError(
                    ErrorCode.INFERENCE_INDEX_MISMATCH,
                    "inference_index does not match the next expected request",
                    details={
                        "expected_inference_index": self._next_inference_index,
                        "actual_inference_index": frame.inference_index,
                    },
                )
            return self._new_token(frame)

    def complete(self, token: OperationToken) -> OperationResult:
        """Commit a successful backend operation.

        If the client disconnected while the operation was running, the result
        is marked obsolete and the session closes without applying its normal
        lifecycle transition. The dispatcher must attempt the indicated reply
        before it submits another frame from this connection.
        """

        with self._lock:
            self._require_pending(token)
            if self._phase == SessionPhase.DRAINING:
                return self._finish_draining()

            self._pending = None
            if token.message_type == MessageType.HELLO:
                self._phase = SessionPhase.READY
            elif token.message_type == MessageType.RESET:
                self._phase = SessionPhase.ACTIVE
                self._active_episode_id = token.episode_id
                self._next_inference_index = 0
            elif token.message_type == MessageType.INFER:
                if self._next_inference_index != token.inference_index:
                    raise SessionInvariantError(
                        "pending inference index changed before completion",
                    )
                self._next_inference_index += 1
            elif token.message_type == MessageType.TRIAL_END:
                self._phase = SessionPhase.READY
                self._active_episode_id = None
                self._next_inference_index = None
            else:
                raise SessionInvariantError(f"unhandled operation: {token.message_type}")

            return OperationResult(
                phase=self._phase,
                reply_allowed=True,
                close_after_reply=False,
                operation_obsolete=False,
                episode_lost=False,
            )

    def fail(self, token: OperationToken) -> OperationResult:
        """Settle a backend failure and make the session terminal.

        While the transport is still present, ``reply_allowed`` is true so the
        dispatcher may send one correlated ``ERROR`` and then close. The
        session remains ``TERMINATING`` until transport shutdown calls
        :meth:`disconnect`.
        """

        with self._lock:
            self._require_pending(token)
            if self._phase == SessionPhase.DRAINING:
                return self._finish_draining()

            self._pending = None
            self._episode_lost = token.message_type != MessageType.HELLO
            self._phase = SessionPhase.TERMINATING
            return OperationResult(
                phase=self._phase,
                reply_allowed=True,
                close_after_reply=True,
                operation_obsolete=False,
                episode_lost=self._episode_lost,
            )

    def reject(self, token: OperationToken) -> OperationResult:
        """Reject a payload before any backend side effect.

        The request ID remains burned. A rejected RESET also keeps its
        candidate episode ID burned, so its correction needs both a fresh
        request ID and a fresh episode ID. Rejected INFER and TRIAL_END
        requests may be corrected with a fresh request ID while retaining the
        active episode ID. Lifecycle state does not advance. A rejected HELLO
        ends the unestablished connection after its correlated error reply.

        This method must never be used after invoking the model or another
        stateful backend operation; use :meth:`fail` in that case.
        """

        with self._lock:
            self._require_pending(token)
            if self._phase == SessionPhase.DRAINING:
                return self._finish_draining()

            self._pending = None
            close_after_reply = token.message_type == MessageType.HELLO
            if close_after_reply:
                self._phase = SessionPhase.TERMINATING
            return OperationResult(
                phase=self._phase,
                reply_allowed=True,
                close_after_reply=close_after_reply,
                operation_obsolete=False,
                episode_lost=False,
            )

    def disconnect(self) -> DisconnectResult:
        """Invalidate the session after transport loss.

        A pending synchronous backend operation must actually settle before
        cleanup may start. The independent global model lease is released only
        after that cleanup succeeds. Repeated calls are idempotent.
        """

        with self._lock:
            if self._phase == SessionPhase.DRAINING:
                return DisconnectResult(
                    phase=self._phase,
                    waiting_for_operation=True,
                    episode_lost=self._episode_lost,
                )
            if self._phase == SessionPhase.TERMINATING:
                self._phase = SessionPhase.CLOSED
                return DisconnectResult(
                    phase=self._phase,
                    waiting_for_operation=False,
                    episode_lost=self._episode_lost,
                )
            if self._phase == SessionPhase.CLOSED:
                return DisconnectResult(
                    phase=self._phase,
                    waiting_for_operation=False,
                    episode_lost=self._episode_lost,
                )

            if self._pending is not None:
                self._episode_lost = self._operation_has_episode(self._pending)
                self._phase = SessionPhase.DRAINING
                return DisconnectResult(
                    phase=self._phase,
                    waiting_for_operation=True,
                    episode_lost=self._episode_lost,
                )

            self._episode_lost = self._active_episode_id is not None
            self._phase = SessionPhase.CLOSED
            return DisconnectResult(
                phase=self._phase,
                waiting_for_operation=False,
                episode_lost=self._episode_lost,
            )

    def _begin_hello(self, frame: Frame) -> OperationToken:
        if frame.message_type != MessageType.HELLO:
            self._invalid_state("the first request must be hello")
        if self._pending is not None:
            raise SessionInvariantError("cannot bind a second pending hello")
        self._reserve_request_id(frame.request_id)
        self._session_id = frame.session_id
        return self._new_token(frame)

    def _new_token(self, frame: Frame) -> OperationToken:
        self._generation += 1
        token = OperationToken(
            generation=self._generation,
            message_type=frame.message_type,
            request_id=frame.request_id,
            session_id=frame.session_id,
            episode_id=frame.episode_id,
            inference_index=frame.inference_index,
        )
        self._pending = token
        return token

    def _reserve_request_id(self, request_id: str) -> None:
        if request_id in self._seen_request_ids:
            self._invalid_state(
                f"request_id {request_id!r} was already used in this session",
            )
        self._seen_request_ids.add(request_id)

    def _require_pending(self, token: OperationToken) -> None:
        if self._pending is not token:
            raise SessionInvariantError("operation token is stale or does not belong to this session")

    def _finish_draining(self) -> OperationResult:
        self._pending = None
        self._phase = SessionPhase.CLOSED
        return OperationResult(
            phase=self._phase,
            reply_allowed=False,
            close_after_reply=False,
            operation_obsolete=True,
            episode_lost=self._episode_lost,
        )

    @staticmethod
    def _operation_has_episode(token: OperationToken) -> bool:
        return token.message_type in {
            MessageType.RESET,
            MessageType.INFER,
            MessageType.TRIAL_END,
        }

    @staticmethod
    def _invalid_state(message: str) -> None:
        raise ProtocolError(ErrorCode.INVALID_STATE, message)
