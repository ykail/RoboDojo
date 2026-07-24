"""Synchronous and asynchronous clients for ``robodojo-policy-v1``.

The synchronous :class:`PolicyClient` owns exactly one event loop and delegates
to :class:`AsyncPolicyClient`.  The asynchronous core deliberately performs one
serialized ``send``/``recv`` exchange at a time.  It has no receive task,
pending-request table, reconnect logic, or request replay.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
import inspect
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from src.eval_client.policy_runtime.canonical import (
    CanonicalActionChunk,
    CanonicalObservation,
    PayloadValidationError,
    parse_infer_payload,
    parse_infer_result_payload,
)
from src.eval_client.policy_runtime.codec import (
    MAX_FRAME_BYTES,
    decode_frame,
    encode_frame,
)
from src.eval_client.policy_runtime.errors import ErrorCode, ProtocolError
from src.eval_client.policy_runtime.execution_profile import (
    ARX_X5_SIM_PI05_PROFILE,
    PolicyExecutionProfile,
)
from src.eval_client.policy_runtime.frame import Frame
from src.eval_client.policy_runtime.lifecycle_payloads import (
    PolicyProvenance,
    RemoteErrorPayload,
    ResetPayload,
    TrialEndPayload,
    build_hello_payload,
    parse_empty_success_payload,
    parse_error_payload,
    parse_hello_ack_payload,
    parse_reset_payload,
    parse_trial_end_payload,
)
from src.eval_client.policy_runtime.messages import MessageType

DEFAULT_CONNECT_TIMEOUT_S = 30.0
DEFAULT_REQUEST_TIMEOUT_S = 600.0
DEFAULT_CLOSE_TIMEOUT_S = 5.0


class PolicyTransport(Protocol):
    """Small transport surface used by the client and scripted tests."""

    async def send(self, message: bytes) -> None: ...

    async def recv(self) -> bytes | str: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TransportOptions:
    """Connection options supplied to production and injected factories."""

    url: str
    max_size: int
    connect_timeout_s: float
    close_timeout_s: float


TransportFactory = Callable[[TransportOptions], Awaitable[PolicyTransport]]
IdFactory = Callable[[], str]


class PolicyClientState(StrEnum):
    """Client-owned connection and episode lifecycle."""

    NEW = "new"
    NEGOTIATING = "negotiating"
    READY = "ready"
    ACTIVE = "active"
    LOST = "lost"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class PendingRequest:
    """Coordinates of the one request whose outcome is not yet known."""

    message_type: MessageType
    request_id: str
    episode_id: str | None
    inference_index: int | None


@dataclass(frozen=True, slots=True)
class ClientSnapshot:
    """Immutable state for logging, diagnostics, and recovery decisions."""

    state: PolicyClientState
    session_id: str | None
    active_episode_id: str | None
    next_inference_index: int | None
    provenance: PolicyProvenance | None
    pending: PendingRequest | None
    episode_lost: bool
    seen_request_count: int
    seen_episode_count: int


class PolicyClientError(RuntimeError):
    """A stable client-side failure.

    ``episode_lost`` means the request or active rollout can no longer be
    continued safely.  It never grants permission to replay a request.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        episode_lost: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.episode_lost = episode_lost


class PolicyClientStateError(PolicyClientError):
    """A local API call is not valid in the current lifecycle state."""


class PolicyConnectionError(PolicyClientError):
    """A transport couldn't be opened, so no session was established."""


class SessionLostError(PolicyClientError):
    """The established session cannot be continued safely."""


class PolicyTimeoutError(SessionLostError):
    """A request outcome became ambiguous after its deadline."""


class PolicyTransportError(SessionLostError):
    """The established transport failed during a request."""


class PolicyResponseError(SessionLostError):
    """The server returned an invalid frame, correlation, or payload."""


class InvalidActionError(PolicyResponseError):
    """Policy inference ran but returned an action RoboDojo cannot execute."""


class RemotePolicyError(PolicyClientError):
    """A correlated, fully validated ``ERROR`` response from the server."""

    def __init__(
        self,
        error: RemoteErrorPayload,
        *,
        terminal: bool,
        episode_lost: bool,
    ) -> None:
        super().__init__(
            error.code,
            error.message,
            episode_lost=episode_lost,
        )
        self.error = error
        self.terminal = terminal


async def _open_websocket_transport(
    options: TransportOptions,
) -> PolicyTransport:
    """Open the production WebSocket transport.

    Ping tasks are disabled because the synchronous facade advances its event
    loop only while a client operation is running.  Request deadlines still
    bound every complete send/receive exchange.
    """

    import websockets

    connect_kwargs: dict[str, Any] = {
        "open_timeout": options.connect_timeout_s,
        "close_timeout": options.close_timeout_s,
        "max_size": options.max_size,
        "max_queue": 1,
        "compression": None,
        "ping_interval": None,
        "ping_timeout": None,
    }
    # websockets >=15 may inherit a process proxy automatically. Policy
    # servers are explicit endpoints and must not silently route through it.
    # websockets 12 doesn't accept this keyword, so feature-detect it.
    try:
        connect_parameters = inspect.signature(websockets.connect).parameters
    except (TypeError, ValueError):
        connect_parameters = {}
    if "proxy" in connect_parameters:
        connect_kwargs["proxy"] = None
    return await websockets.connect(options.url, **connect_kwargs)


def _positive_timeout(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a positive number")
    result = float(value)
    if not 0.0 < result < float("inf"):
        raise ValueError(f"{field_name} must be finite and positive")
    return result


def _default_id() -> str:
    return str(uuid4())


class AsyncPolicyClient:
    """One non-reconnecting ``robodojo-policy-v1`` client session."""

    def __init__(
        self,
        url: str,
        *,
        profile: PolicyExecutionProfile = ARX_X5_SIM_PI05_PROFILE,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        close_timeout_s: float = DEFAULT_CLOSE_TIMEOUT_S,
        transport_factory: TransportFactory | None = None,
        session_id_factory: IdFactory = _default_id,
        request_id_factory: IdFactory = _default_id,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string")
        if not isinstance(profile, PolicyExecutionProfile):
            raise TypeError("profile must be PolicyExecutionProfile")
        if not callable(session_id_factory):
            raise TypeError("session_id_factory must be callable")
        if not callable(request_id_factory):
            raise TypeError("request_id_factory must be callable")

        self._url = url
        self._profile = profile
        self._connect_timeout_s = _positive_timeout(
            connect_timeout_s,
            "connect_timeout_s",
        )
        self._request_timeout_s = _positive_timeout(
            request_timeout_s,
            "request_timeout_s",
        )
        self._close_timeout_s = _positive_timeout(
            close_timeout_s,
            "close_timeout_s",
        )
        self._transport_options = TransportOptions(
            url=self._url,
            max_size=MAX_FRAME_BYTES,
            connect_timeout_s=self._connect_timeout_s,
            close_timeout_s=self._close_timeout_s,
        )
        self._transport_factory = transport_factory or _open_websocket_transport
        if not callable(self._transport_factory):
            raise TypeError("transport_factory must be callable")
        self._session_id_factory = session_id_factory
        self._request_id_factory = request_id_factory

        self._state = PolicyClientState.NEW
        self._transport: PolicyTransport | None = None
        self._session_id: str | None = None
        self._active_episode_id: str | None = None
        self._next_inference_index: int | None = None
        self._provenance: PolicyProvenance | None = None
        self._episode_lost = False
        self._pending: PendingRequest | None = None
        self._seen_episode_ids: set[str] = set()
        self._seen_request_ids: set[str] = set()
        self._operation_lock = asyncio.Lock()

    @property
    def state(self) -> PolicyClientState:
        return self._state

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def active_episode_id(self) -> str | None:
        return self._active_episode_id

    @property
    def next_inference_index(self) -> int | None:
        return self._next_inference_index

    @property
    def provenance(self) -> PolicyProvenance | None:
        return self._provenance

    @property
    def episode_lost(self) -> bool:
        return self._episode_lost

    @property
    def snapshot(self) -> ClientSnapshot:
        return ClientSnapshot(
            state=self._state,
            session_id=self._session_id,
            active_episode_id=self._active_episode_id,
            next_inference_index=self._next_inference_index,
            provenance=self._provenance,
            pending=self._pending,
            episode_lost=self._episode_lost,
            seen_request_count=len(self._seen_request_ids),
            seen_episode_count=len(self._seen_episode_ids),
        )

    async def connect(self) -> PolicyProvenance:
        """Open a transport and complete the mandatory HELLO atomically."""

        async with self._operation_lock:
            self._require_state(PolicyClientState.NEW, "connect")
            session_id = self._next_session_id()
            request = self._new_request(
                MessageType.HELLO,
                session_id=session_id,
                payload=build_hello_payload(self._profile),
            )
            # HELLO is pre-encoded before opening the socket. A local codec
            # failure therefore cannot leak an unbound WebSocket in
            # NEGOTIATING.
            try:
                encoded_request = encode_frame(request)
            except Exception:
                self._session_id = None
                raise
            self._state = PolicyClientState.NEGOTIATING
            try:
                self._transport = await asyncio.wait_for(
                    self._transport_factory(self._transport_options),
                    timeout=self._connect_timeout_s,
                )
            except TimeoutError as error:
                self._state = PolicyClientState.NEW
                self._session_id = None
                raise PolicyConnectionError(
                    ErrorCode.TIMEOUT,
                    "timed out while connecting to the policy server",
                    episode_lost=False,
                ) from error
            except asyncio.CancelledError:
                self._state = PolicyClientState.NEW
                self._session_id = None
                raise
            except Exception as error:
                self._state = PolicyClientState.NEW
                self._session_id = None
                raise PolicyConnectionError(
                    ErrorCode.EPISODE_LOST,
                    f"policy transport connection failed: {error}",
                    episode_lost=False,
                ) from error

            response = await self._exchange(
                request,
                expected_type=MessageType.HELLO_ACK,
                encoded_request=encoded_request,
            )
            try:
                provenance = parse_hello_ack_payload(
                    response.payload,
                    expected_profile=self._profile,
                )
            except (PayloadValidationError, TypeError, ValueError) as error:
                await self._lose(episode_lost=False)
                raise PolicyResponseError(
                    ErrorCode.INVALID_FRAME,
                    f"invalid HELLO_ACK payload: {error}",
                    episode_lost=False,
                ) from error

            self._provenance = provenance
            self._state = PolicyClientState.READY
            return provenance

    async def reset(
        self,
        episode_id: str,
        payload: ResetPayload,
    ) -> None:
        """Start one new episode after a successful server-side RESET."""

        async with self._operation_lock:
            self._require_state(PolicyClientState.READY, "reset")
            if not isinstance(payload, ResetPayload):
                raise TypeError("payload must be ResetPayload")
            # Re-parse at the outbound boundary so object.__setattr__ or a
            # subclass cannot bypass the strict wire contract.
            reset_payload = parse_reset_payload(payload.to_payload())
            if episode_id in self._seen_episode_ids:
                raise PolicyClientStateError(
                    ErrorCode.INVALID_STATE,
                    f"episode_id {episode_id!r} was already used by this session",
                    episode_lost=False,
                )
            request = self._new_request(
                MessageType.RESET,
                episode_id=episode_id,
                payload=reset_payload.to_payload(),
            )
            # A sent or server-rejected RESET burns the candidate episode ID.
            # Reserving before encode is conservative and keeps local recovery
            # from accidentally reusing an ambiguous identifier.
            self._seen_episode_ids.add(episode_id)
            response = await self._exchange(
                request,
                expected_type=MessageType.RESET_RESULT,
            )
            await self._parse_empty_response(response, "RESET_RESULT")
            self._active_episode_id = episode_id
            self._next_inference_index = 0
            self._state = PolicyClientState.ACTIVE

    async def infer(
        self,
        observation: CanonicalObservation,
    ) -> CanonicalActionChunk:
        """Request the next action chunk for the active episode."""

        async with self._operation_lock:
            self._require_state(PolicyClientState.ACTIVE, "infer")
            if not isinstance(observation, CanonicalObservation):
                raise TypeError("observation must be CanonicalObservation")
            validated_observation = parse_infer_payload(
                {"observation": observation.to_payload()},
                observation_spec=self._profile.observation_spec,
            )
            assert self._active_episode_id is not None
            assert self._next_inference_index is not None
            request = self._new_request(
                MessageType.INFER,
                episode_id=self._active_episode_id,
                inference_index=self._next_inference_index,
                payload={"observation": validated_observation.to_payload()},
            )
            response = await self._exchange(
                request,
                expected_type=MessageType.INFER_RESULT,
            )
            try:
                action = parse_infer_result_payload(
                    response.payload,
                    action_spec=self._profile.action_spec,
                )
            except (PayloadValidationError, TypeError, ValueError) as error:
                await self._lose(episode_lost=True)
                raise InvalidActionError(
                    ErrorCode.INFER_FAILED,
                    f"invalid INFER_RESULT payload: {error}",
                    episode_lost=True,
                ) from error
            self._next_inference_index += 1
            return action

    async def trial_end(self, payload: TrialEndPayload) -> None:
        """End the active episode after the server acknowledges cleanup."""

        async with self._operation_lock:
            self._require_state(PolicyClientState.ACTIVE, "trial_end")
            if not isinstance(payload, TrialEndPayload):
                raise TypeError("payload must be TrialEndPayload")
            validated_payload = parse_trial_end_payload(payload.to_payload())
            assert self._active_episode_id is not None
            request = self._new_request(
                MessageType.TRIAL_END,
                episode_id=self._active_episode_id,
                payload=validated_payload.to_payload(),
            )
            response = await self._exchange(
                request,
                expected_type=MessageType.TRIAL_END_ACK,
            )
            await self._parse_empty_response(response, "TRIAL_END_ACK")
            self._active_episode_id = None
            self._next_inference_index = None
            self._state = PolicyClientState.READY

    async def close(self) -> None:
        """Close without attempting any lifecycle replay or fabricated end."""

        async with self._operation_lock:
            if self._state != PolicyClientState.LOST:
                self._state = PolicyClientState.CLOSED
            if self._active_episode_id is not None:
                self._episode_lost = True
            await self._close_transport()

    async def _exchange(
        self,
        request: Frame,
        *,
        expected_type: MessageType,
        encoded_request: bytes | None = None,
    ) -> Frame:
        transport = self._transport
        if transport is None:
            await self._lose(episode_lost=request.episode_id is not None)
            raise PolicyTransportError(
                ErrorCode.EPISODE_LOST,
                "policy transport is not connected",
                episode_lost=request.episode_id is not None,
            )

        # Encoding is deliberately before send. A local encoding error proves
        # that no bytes were handed to the transport, so it doesn't make the
        # established server session ambiguous.
        encoded = encoded_request if encoded_request is not None else encode_frame(request)
        self._pending = PendingRequest(
            message_type=request.message_type,
            request_id=request.request_id,
            episode_id=request.episode_id,
            inference_index=request.inference_index,
        )
        try:
            async with asyncio.timeout(self._request_timeout_s):
                await transport.send(encoded)
                raw_response = await transport.recv()
        except TimeoutError as error:
            episode_lost = request.episode_id is not None
            await self._lose(episode_lost=episode_lost)
            raise PolicyTimeoutError(
                ErrorCode.TIMEOUT,
                f"timed out waiting for {expected_type.value}",
                episode_lost=episode_lost,
            ) from error
        except asyncio.CancelledError:
            await self._lose(episode_lost=request.episode_id is not None)
            raise
        except Exception as error:
            episode_lost = request.episode_id is not None
            await self._lose(episode_lost=episode_lost)
            raise PolicyTransportError(
                ErrorCode.EPISODE_LOST,
                f"policy transport failed during {request.message_type.value}: {error}",
                episode_lost=episode_lost,
            ) from error

        if not isinstance(raw_response, bytes | bytearray):
            await self._lose(episode_lost=request.episode_id is not None)
            raise PolicyResponseError(
                ErrorCode.INVALID_FRAME,
                "policy server sent a text or unsupported WebSocket message",
                episode_lost=request.episode_id is not None,
            )
        try:
            response = decode_frame(raw_response)
            self._validate_correlation(request, response)
        except (ProtocolError, TypeError, ValueError) as error:
            await self._lose(episode_lost=request.episode_id is not None)
            code = error.code if isinstance(error, ProtocolError) else ErrorCode.INVALID_FRAME
            raise PolicyResponseError(
                code,
                f"invalid policy response: {error}",
                episode_lost=request.episode_id is not None,
            ) from error

        if response.message_type == MessageType.ERROR:
            try:
                remote_error = parse_error_payload(response.payload)
            except (PayloadValidationError, TypeError, ValueError) as error:
                await self._lose(episode_lost=request.episode_id is not None)
                raise PolicyResponseError(
                    ErrorCode.INVALID_FRAME,
                    f"invalid ERROR payload: {error}",
                    episode_lost=request.episode_id is not None,
                ) from error
            nonterminal = remote_error.code == ErrorCode.INVALID_PAYLOAD and request.message_type != MessageType.HELLO
            if nonterminal:
                self._pending = None
                raise RemotePolicyError(
                    remote_error,
                    terminal=False,
                    episode_lost=False,
                )
            episode_lost = request.episode_id is not None
            self._pending = None
            await self._lose(episode_lost=episode_lost)
            raise RemotePolicyError(
                remote_error,
                terminal=True,
                episode_lost=episode_lost,
            )

        if response.message_type != expected_type:
            await self._lose(episode_lost=request.episode_id is not None)
            raise PolicyResponseError(
                ErrorCode.INVALID_FRAME,
                f"expected {expected_type.value}, got {response.message_type.value}",
                episode_lost=request.episode_id is not None,
            )
        self._pending = None
        return response

    async def _parse_empty_response(
        self,
        response: Frame,
        response_name: str,
    ) -> None:
        try:
            parse_empty_success_payload(response.payload)
        except (PayloadValidationError, TypeError, ValueError) as error:
            await self._lose(episode_lost=response.episode_id is not None)
            raise PolicyResponseError(
                ErrorCode.INVALID_FRAME,
                f"invalid {response_name} payload: {error}",
                episode_lost=response.episode_id is not None,
            ) from error

    def _new_request(
        self,
        message_type: MessageType,
        *,
        session_id: str | None = None,
        episode_id: str | None = None,
        inference_index: int | None = None,
        payload: dict[str, Any],
    ) -> Frame:
        bound_session_id = session_id or self._session_id
        if bound_session_id is None:
            raise PolicyClientStateError(
                ErrorCode.INVALID_STATE,
                "the client does not have a bound session_id",
                episode_lost=False,
            )
        return Frame(
            message_type=message_type,
            request_id=self._next_request_id(),
            session_id=bound_session_id,
            episode_id=episode_id,
            inference_index=inference_index,
            payload=payload,
        )

    def _next_session_id(self) -> str:
        session_id = self._session_id_factory()
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id_factory must return a non-empty string")
        self._session_id = session_id
        return session_id

    def _next_request_id(self) -> str:
        request_id = self._request_id_factory()
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id_factory must return a non-empty string")
        if request_id in self._seen_request_ids:
            raise ValueError(
                f"request_id_factory reused request_id {request_id!r}",
            )
        self._seen_request_ids.add(request_id)
        return request_id

    def _validate_correlation(
        self,
        request: Frame,
        response: Frame,
    ) -> None:
        expected = {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "episode_id": request.episode_id,
            "inference_index": request.inference_index,
        }
        for field_name, expected_value in expected.items():
            actual_value = getattr(response, field_name)
            if actual_value != expected_value:
                raise ProtocolError(
                    ErrorCode.INVALID_FRAME,
                    f"response {field_name} does not match the request",
                    details={
                        "field": field_name,
                        "expected": expected_value,
                        "actual": actual_value,
                    },
                )

    def _require_state(
        self,
        expected: PolicyClientState,
        operation: str,
    ) -> None:
        if self._state != expected:
            raise PolicyClientStateError(
                ErrorCode.INVALID_STATE,
                f"{operation} requires client state {expected.value}; current state is {self._state.value}",
                episode_lost=self._episode_lost,
            )

    async def _lose(self, *, episode_lost: bool) -> None:
        self._episode_lost = self._episode_lost or episode_lost
        self._state = PolicyClientState.LOST
        await self._close_transport()

    async def _close_transport(self) -> None:
        transport = self._transport
        self._transport = None
        if transport is None:
            return
        try:
            await asyncio.wait_for(
                transport.close(),
                timeout=self._close_timeout_s,
            )
        except Exception:
            # Transport shutdown is best-effort; the client has already fenced
            # itself from further sends.
            pass


class PolicyClient:
    """Blocking facade that keeps all async transport work on one event loop."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._loop = asyncio.new_event_loop()
        self._call_lock = RLock()
        try:
            self._async_client = AsyncPolicyClient(*args, **kwargs)
        except Exception:
            self._loop.close()
            raise

    @property
    def state(self) -> PolicyClientState:
        return self._async_client.state

    @property
    def session_id(self) -> str | None:
        return self._async_client.session_id

    @property
    def active_episode_id(self) -> str | None:
        return self._async_client.active_episode_id

    @property
    def next_inference_index(self) -> int | None:
        return self._async_client.next_inference_index

    @property
    def provenance(self) -> PolicyProvenance | None:
        return self._async_client.provenance

    @property
    def episode_lost(self) -> bool:
        return self._async_client.episode_lost

    @property
    def snapshot(self) -> ClientSnapshot:
        return self._async_client.snapshot

    def connect(self) -> PolicyProvenance:
        return self._run(self._async_client.connect())

    def reset(self, episode_id: str, payload: ResetPayload) -> None:
        self._run(self._async_client.reset(episode_id, payload))

    def infer(
        self,
        observation: CanonicalObservation,
    ) -> CanonicalActionChunk:
        return self._run(self._async_client.infer(observation))

    def trial_end(self, payload: TrialEndPayload) -> None:
        self._run(self._async_client.trial_end(payload))

    def close(self) -> None:
        with self._call_lock:
            if self._loop.is_closed():
                return
            self._require_blocking_context()
            try:
                self._run(self._async_client.close())
            finally:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.close()

    def __enter__(self) -> PolicyClient:
        try:
            self.connect()
        except BaseException:
            # ``__exit__`` is never called when ``__enter__`` raises. Close
            # the private loop here so a failed context-manager connection
            # cannot leak loop or transport resources.
            try:
                self.close()
            except BaseException:
                pass
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()

    def _run(self, awaitable: Awaitable[Any]) -> Any:
        with self._call_lock:
            if self._loop.is_closed():
                if hasattr(awaitable, "close"):
                    awaitable.close()
                raise PolicyClientStateError(
                    ErrorCode.INVALID_STATE,
                    "policy client event loop is closed",
                    episode_lost=self.episode_lost,
                )
            try:
                self._require_blocking_context()
            except RuntimeError:
                if hasattr(awaitable, "close"):
                    awaitable.close()
                raise
            task = self._loop.create_task(awaitable)
            try:
                return self._loop.run_until_complete(task)
            except BaseException:
                # A synchronous KeyboardInterrupt (or equivalent outer
                # cancellation) must settle the async operation. Otherwise its
                # lifecycle lock and ambiguous transport could survive after
                # control returns to the simulator.
                if not task.done():
                    task.cancel()
                    try:
                        self._loop.run_until_complete(task)
                    except BaseException:
                        pass
                raise

    @staticmethod
    def _require_blocking_context() -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            "blocking PolicyClient methods cannot run inside an active event loop; use AsyncPolicyClient instead",
        )
