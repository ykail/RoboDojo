"""Dependency-free client for the local PiPER-X operator bridge.

The hardware process owns CAN, keyboard input and follower safety.  RoboDojo
owns simulation state, policy arbitration and recording.  They communicate
only through this deliberately small loopback TCP protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import socket
import struct
import threading
import time
from typing import Any
import uuid

PROTOCOL = "robodojo_piperx_v2"
MAX_FRAME_BYTES = 1 << 20
_ENVELOPE_KEYS = frozenset(
    {
        "protocol",
        "type",
        "session_id",
        "episode_id",
        "generation",
        "seq",
        "sent_monotonic_ns",
        "deadline_monotonic_ns",
        "payload",
    }
)
_REQUEST_TYPES = frozenset({"begin_episode", "exchange", "heartbeat", "hold", "end_episode"})
_MODES = frozenset({"policy", "intervention", "fault"})
_CONTROL_TOPOLOGY = "accepted_sim_to_follower_to_leader"
_LEADER_ACTUATION_MODES = frozenset({"output_follow", "native_leader", "disabled", "fault"})
_FOLLOWER_ACTUATION_MODES = frozenset({"sim_follow", "hold", "disabled", "fault"})
_TRANSITIONS = frozenset({None, "entering_intervention", "reattaching_policy"})
_EDGES = frozenset({None, "enter", "exit"})
_TERMINAL_REQUESTS = frozenset({None, "accept_next", "discard_retry", "accept_exit", "discard_exit"})


class PiperXBridgeError(RuntimeError):
    """Base class for a bridge transport, protocol or safety failure."""


class PiperXBridgeTransportError(PiperXBridgeError):
    """The loopback connection was lost or timed out."""


class PiperXBridgeProtocolError(PiperXBridgeError):
    """The peer returned an invalid or ambiguous response."""


class PiperXBridgeSafetyError(PiperXBridgeError):
    """The bridge rejected mirroring or reported unhealthy hardware."""


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PiperXBridgeProtocolError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PiperXBridgeProtocolError(f"{label} must be finite")
    return result


def _non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PiperXBridgeProtocolError(f"{label} must be a non-negative integer")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 256:
        raise PiperXBridgeProtocolError(f"{label} must be a non-empty string of at most 256 bytes")
    return value


def _reason(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("PiPER-X hold/end reason must be a non-empty string")
    return value.strip()[:512]


def _pose(value: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 7:
        raise PiperXBridgeProtocolError(f"{label} must be [x,y,z,qw,qx,qy,qz]")
    pose = tuple(_finite_number(item, label=f"{label}[{index}]") for index, item in enumerate(value))
    norm = math.sqrt(sum(component * component for component in pose[3:]))
    if not 0.999 <= norm <= 1.001:
        raise PiperXBridgeProtocolError(f"{label} quaternion must be normalized, got norm={norm:g}")
    return pose


@dataclass(frozen=True)
class SimArmTarget:
    """One simulator end-effector target sent to the physical mirror."""

    pose: tuple[float, ...]
    gripper: float

    def __post_init__(self) -> None:
        pose = _pose(list(self.pose), label="sim.pose")
        gripper = _finite_number(self.gripper, label="sim.gripper")
        if not 0.0 <= gripper <= 1.0:
            raise PiperXBridgeProtocolError("sim.gripper must be in [0, 1]")
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "gripper", gripper)

    def to_payload(self) -> dict[str, Any]:
        return {"pose": list(self.pose), "gripper": self.gripper}


@dataclass(frozen=True)
class SimTargets:
    """Synchronized dual-arm simulator targets."""

    left: SimArmTarget
    right: SimArmTarget

    def to_payload(self) -> dict[str, Any]:
        return {"left": self.left.to_payload(), "right": self.right.to_payload()}


@dataclass(frozen=True)
class OperatorArmSample:
    """Absolute leader pose and measured gripper opening."""

    pose: tuple[float, ...]
    gripper_m: float


@dataclass(frozen=True)
class OperatorSample:
    """One coherent bridge response for both PiPER-X leaders."""

    generation: int
    seq: int
    mode: str
    control_topology: str
    leader_actuation_mode: str
    follower_actuation_mode: str
    transition: str | None
    edge: str | None
    terminal_request: str | None
    left: OperatorArmSample
    right: OperatorArmSample
    mirror_accepted: bool
    health: dict[str, Any]
    diagnostics: dict[str, Any]


def encode_frame(message: dict[str, Any], *, max_frame_bytes: int = MAX_FRAME_BYTES) -> bytes:
    """Encode one exact JSON envelope with a four-byte big-endian length."""

    try:
        payload = json.dumps(
            message,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PiperXBridgeProtocolError(f"Bridge frame is not strict JSON: {exc}") from exc
    if not payload or len(payload) > max_frame_bytes:
        raise PiperXBridgeProtocolError(f"Bridge JSON frame size must be in [1, {max_frame_bytes}], got {len(payload)}")
    return struct.pack(">I", len(payload)) + payload


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = connection.recv(remaining)
        except TimeoutError as exc:
            raise PiperXBridgeTransportError("Timed out receiving from PiPER-X bridge") from exc
        except OSError as exc:
            raise PiperXBridgeTransportError(f"Failed receiving from PiPER-X bridge: {exc}") from exc
        if not chunk:
            raise PiperXBridgeTransportError("PiPER-X bridge closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_frame(connection: socket.socket, *, max_frame_bytes: int = MAX_FRAME_BYTES) -> dict[str, Any]:
    """Receive one bounded, duplicate-key-free JSON object."""

    length = struct.unpack(">I", _recv_exact(connection, 4))[0]
    if length == 0 or length > max_frame_bytes:
        raise PiperXBridgeProtocolError(f"Bridge JSON frame size must be in [1, {max_frame_bytes}], got {length}")

    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PiperXBridgeProtocolError(f"Duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            _recv_exact(connection, length).decode("utf-8"),
            object_pairs_hook=no_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PiperXBridgeProtocolError(f"Non-finite JSON number: {value}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise PiperXBridgeProtocolError("Bridge frame is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise PiperXBridgeProtocolError(f"Bridge frame is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise PiperXBridgeProtocolError("Bridge frame must be a JSON object")
    return decoded


def _parse_arm_sample(value: Any, *, label: str) -> OperatorArmSample:
    if not isinstance(value, dict) or set(value) != {"pose", "gripper_m"}:
        raise PiperXBridgeProtocolError(f"{label} must contain exactly pose and gripper_m")
    gripper_m = _finite_number(value["gripper_m"], label=f"{label}.gripper_m")
    if not 0.0 <= gripper_m <= 0.2:
        raise PiperXBridgeProtocolError(f"{label}.gripper_m must be in [0, 0.2]")
    return OperatorArmSample(
        pose=_pose(value["pose"], label=f"{label}.pose"),
        gripper_m=gripper_m,
    )


class PiperXBridgeClient:
    """Synchronous, fail-closed client for one physical bridge session."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        connect_timeout_s: float = 5.0,
        response_timeout_s: float = 0.2,
        heartbeat_interval_s: float = 0.25,
        session_id: str | None = None,
        max_frame_bytes: int = MAX_FRAME_BYTES,
    ):
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("PiPER-X bridge host must be loopback (127.0.0.1, ::1, or localhost)")
        if not 1 <= int(port) <= 65535:
            raise ValueError("PiPER-X bridge port must be in [1, 65535]")
        if connect_timeout_s <= 0 or response_timeout_s <= 0 or heartbeat_interval_s <= 0:
            raise ValueError("PiPER-X bridge timeouts must be positive")
        if max_frame_bytes <= 0:
            raise ValueError("PiPER-X bridge frame limit must be positive")
        self.host = host
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.response_timeout_s = float(response_timeout_s)
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.session_id = _identifier(session_id or str(uuid.uuid4()), label="session_id")
        self.max_frame_bytes = int(max_frame_bytes)
        self._socket: socket.socket | None = None
        self._episode_id: str | None = None
        self._episode_active = False
        self._generation = 0
        self._seq = 0
        self._session_lost = False
        self._request_lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_error: PiperXBridgeError | None = None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def episode_id(self) -> str | None:
        return self._episode_id if self._episode_active else None

    @property
    def episode_active(self) -> bool:
        return self._episode_active

    def connect(self) -> None:
        if self._session_lost:
            raise PiperXBridgeTransportError("PiPER-X bridge session was lost and cannot be reconnected/replayed")
        if self._socket is not None:
            return
        try:
            connection = socket.create_connection(
                (self.host, self.port),
                timeout=self.connect_timeout_s,
            )
            connection.settimeout(self.response_timeout_s)
        except OSError as exc:
            raise PiperXBridgeTransportError(
                f"Could not connect to PiPER-X bridge at {self.host}:{self.port}: {exc}"
            ) from exc
        self._socket = connection

    def _request(
        self,
        request_type: str,
        payload: dict[str, Any],
        *,
        from_heartbeat: bool = False,
        ignore_heartbeat_error: bool = False,
    ) -> OperatorSample:
        if request_type not in _REQUEST_TYPES:
            raise ValueError(f"Unknown PiPER-X bridge request type: {request_type!r}")
        with self._request_lock:
            if self._episode_id is None:
                raise PiperXBridgeProtocolError("No PiPER-X bridge episode is active")
            if self._heartbeat_error is not None and not from_heartbeat and not ignore_heartbeat_error:
                raise PiperXBridgeTransportError(
                    f"PiPER-X heartbeat failed: {self._heartbeat_error}"
                ) from self._heartbeat_error
            self.connect()
            assert self._socket is not None
            sent_ns = time.monotonic_ns()
            deadline_ns = sent_ns + int(self.response_timeout_s * 1_000_000_000)
            request_generation = self._generation
            sequence = self._seq
            envelope = {
                "protocol": PROTOCOL,
                "type": request_type,
                "session_id": self.session_id,
                "episode_id": self._episode_id,
                "generation": request_generation,
                "seq": sequence,
                "sent_monotonic_ns": sent_ns,
                "deadline_monotonic_ns": deadline_ns,
                "payload": payload,
            }
            try:
                self._socket.sendall(encode_frame(envelope, max_frame_bytes=self.max_frame_bytes))
                response = receive_frame(self._socket, max_frame_bytes=self.max_frame_bytes)
                self._seq += 1
                sample = self._parse_response(
                    response,
                    request_type=request_type,
                    request_generation=request_generation,
                    sequence=sequence,
                    deadline_ns=deadline_ns,
                )
            except (OSError, PiperXBridgeTransportError, PiperXBridgeProtocolError) as exc:
                self._close_socket_only()
                self._session_lost = True
                self._episode_id = None
                self._episode_active = False
                if isinstance(exc, PiperXBridgeError):
                    raise
                raise PiperXBridgeTransportError(f"Failed sending to PiPER-X bridge: {exc}") from exc
            return sample

    def _parse_response(
        self,
        response: dict[str, Any],
        *,
        request_type: str,
        request_generation: int,
        sequence: int,
        deadline_ns: int,
    ) -> OperatorSample:
        if set(response) != _ENVELOPE_KEYS:
            missing = sorted(_ENVELOPE_KEYS - set(response))
            extra = sorted(set(response) - _ENVELOPE_KEYS)
            raise PiperXBridgeProtocolError(f"Invalid PiPER-X response envelope; missing={missing}, extra={extra}")
        if response["protocol"] != PROTOCOL or response["type"] != request_type:
            raise PiperXBridgeProtocolError("PiPER-X response protocol/type does not match the request")
        if response["session_id"] != self.session_id or response["episode_id"] != self._episode_id:
            raise PiperXBridgeProtocolError("PiPER-X response session/episode does not match the request")
        if _non_negative_int(response["seq"], label="response.seq") != sequence:
            raise PiperXBridgeProtocolError("PiPER-X response seq does not match the request")
        _non_negative_int(response["sent_monotonic_ns"], label="response.sent_monotonic_ns")
        echoed_deadline = _non_negative_int(response["deadline_monotonic_ns"], label="response.deadline_monotonic_ns")
        if echoed_deadline != deadline_ns or time.monotonic_ns() > deadline_ns:
            raise PiperXBridgeProtocolError("PiPER-X response missed or changed the request deadline")
        generation = _non_negative_int(response["generation"], label="response.generation")
        payload = response["payload"]
        expected_payload_keys = {
            "mode",
            "control_topology",
            "leader_actuation_mode",
            "follower_actuation_mode",
            "transition",
            "edge",
            "terminal_request",
            "leader",
            "mirror_accepted",
            "health",
            "diagnostics",
        }
        if not isinstance(payload, dict) or set(payload) != expected_payload_keys:
            raise PiperXBridgeProtocolError(
                "PiPER-X response payload must contain exactly mode, control_topology, "
                "leader_actuation_mode, follower_actuation_mode, transition, edge, "
                "terminal_request, leader, mirror_accepted, health, diagnostics"
            )
        mode = payload["mode"]
        control_topology = payload["control_topology"]
        leader_actuation_mode = payload["leader_actuation_mode"]
        follower_actuation_mode = payload["follower_actuation_mode"]
        transition = payload["transition"]
        edge = payload["edge"]
        terminal_request = payload["terminal_request"]
        if mode not in _MODES or edge not in _EDGES or terminal_request not in _TERMINAL_REQUESTS:
            raise PiperXBridgeProtocolError("PiPER-X response has an invalid mode, edge, or terminal request")
        if control_topology != _CONTROL_TOPOLOGY:
            raise PiperXBridgeProtocolError("PiPER-X response has an invalid control_topology")
        if leader_actuation_mode not in _LEADER_ACTUATION_MODES:
            raise PiperXBridgeProtocolError("PiPER-X response has an invalid leader_actuation_mode")
        if follower_actuation_mode not in _FOLLOWER_ACTUATION_MODES:
            raise PiperXBridgeProtocolError("PiPER-X response has an invalid follower_actuation_mode")
        if transition not in _TRANSITIONS:
            raise PiperXBridgeProtocolError("PiPER-X response has an invalid transition")
        if transition is not None:
            expected_mode = {
                "entering_intervention": "policy",
                "reattaching_policy": "intervention",
            }[transition]
            if mode != expected_mode:
                raise PiperXBridgeProtocolError(
                    f"PiPER-X transition {transition!r} is inconsistent with mode {mode!r}"
                )
            if edge is not None or terminal_request is not None or payload["mirror_accepted"] is not False:
                raise PiperXBridgeProtocolError(
                    "PiPER-X transition must freeze mirroring and cannot carry an edge or terminal request"
                )
            if follower_actuation_mode not in {"sim_follow", "hold"}:
                raise PiperXBridgeSafetyError(
                    "PiPER-X followers must report the blocked prior sim target or measured hold "
                    "during a hardware transition"
                )
        if request_type == "heartbeat" and (edge is not None or terminal_request is not None):
            raise PiperXBridgeProtocolError("heartbeat must not consume an operator edge or terminal request")
        expected_generation = (
            request_generation if request_type == "heartbeat" else request_generation + (1 if edge is not None else 0)
        )
        if generation != expected_generation:
            raise PiperXBridgeProtocolError(
                f"Ambiguous PiPER-X generation: requested={request_generation}, edge={edge!r}, returned={generation}"
            )
        if (edge == "enter" and mode != "intervention") or (edge == "exit" and mode != "policy"):
            raise PiperXBridgeProtocolError(f"PiPER-X edge {edge!r} is inconsistent with mode {mode!r}")
        leader = payload["leader"]
        if not isinstance(leader, dict) or set(leader) != {"left", "right"}:
            raise PiperXBridgeProtocolError("PiPER-X leader sample must contain exactly left and right")
        if not isinstance(payload["mirror_accepted"], bool):
            raise PiperXBridgeProtocolError("PiPER-X mirror_accepted must be boolean")
        if not isinstance(payload["health"], dict) or not isinstance(payload["diagnostics"], dict):
            raise PiperXBridgeProtocolError("PiPER-X health and diagnostics must be objects")
        if not isinstance(payload["diagnostics"].get("request_ok"), bool):
            raise PiperXBridgeProtocolError("PiPER-X diagnostics.request_ok must be present and boolean")
        sample = OperatorSample(
            generation=generation,
            seq=sequence,
            mode=mode,
            control_topology=control_topology,
            leader_actuation_mode=leader_actuation_mode,
            follower_actuation_mode=follower_actuation_mode,
            transition=transition,
            edge=edge,
            terminal_request=terminal_request,
            left=_parse_arm_sample(leader["left"], label="leader.left"),
            right=_parse_arm_sample(leader["right"], label="leader.right"),
            mirror_accepted=payload["mirror_accepted"],
            health=dict(payload["health"]),
            diagnostics=dict(payload["diagnostics"]),
        )
        # Heartbeat proves only connection liveness.  It deliberately neither
        # acknowledges nor consumes the hardware-side operator generation;
        # the next exchange remains the sole control-state boundary.
        if request_type != "heartbeat":
            self._generation = generation
        healthy = sample.health.get("ok") is True
        if mode == "fault" or not healthy:
            raise PiperXBridgeSafetyError(
                f"PiPER-X bridge is not healthy: mode={mode}, health={sample.health}, diagnostics={sample.diagnostics}"
            )
        if sample.diagnostics["request_ok"] is not True:
            raise PiperXBridgeSafetyError(f"PiPER-X bridge rejected {request_type}: {sample.diagnostics}")
        for key, expected in (
            ("control_topology", sample.control_topology),
            ("leader_actuation_mode", sample.leader_actuation_mode),
            ("follower_actuation_mode", sample.follower_actuation_mode),
        ):
            if sample.health.get(key) != expected:
                raise PiperXBridgeSafetyError(
                    f"PiPER-X response {key} disagrees with hardware health: "
                    f"payload={expected!r}, health={sample.health.get(key)!r}"
                )
        if transition is None:
            expected_leader_actuation = {
                "policy": "output_follow",
                "intervention": "native_leader",
                "fault": "fault",
            }[mode]
            if sample.leader_actuation_mode != expected_leader_actuation:
                raise PiperXBridgeSafetyError(
                    "PiPER-X leader actuation does not match the acknowledged bridge mode: "
                    f"mode={mode}, leader_actuation_mode={sample.leader_actuation_mode}"
                )
        if (
            request_type == "exchange"
            and not sample.mirror_accepted
            and sample.edge is None
            and sample.terminal_request is None
            and sample.transition is None
        ):
            raise PiperXBridgeSafetyError(f"PiPER-X bridge rejected the simulator mirror target: {sample.diagnostics}")
        return sample

    def begin_episode(
        self,
        episode_id: str,
        sim: SimTargets,
    ) -> OperatorSample:
        if self._episode_active:
            raise PiperXBridgeProtocolError("A PiPER-X bridge episode is already active")
        with self._request_lock:
            self._episode_id = _identifier(episode_id, label="episode_id")
            self._generation = 0
            self._heartbeat_error = None
            try:
                sample = self._request(
                    "begin_episode",
                    {"sim": sim.to_payload()},
                )
            except PiperXBridgeSafetyError:
                # The peer returned a well-formed response but rejected or
                # entered an unsafe state after seeing BEGIN.  Preserve the
                # episode identity long enough for fail_closed() to request a
                # measured hold before tearing down the non-replayable session.
                raise
            except Exception:
                self._episode_id = None
                self._episode_active = False
                raise
        if sample.mode != "policy" or sample.edge is not None or sample.generation != 0:
            self.fail_closed("invalid_begin_state")
            raise PiperXBridgeProtocolError("begin_episode must reset the bridge to policy mode at generation 0")
        self._episode_active = True
        if self._heartbeat_thread is None:
            self._start_heartbeat()
        return sample

    def exchange(self, sim: SimTargets) -> OperatorSample:
        if not self._episode_active:
            raise PiperXBridgeProtocolError("exchange requires an active PiPER-X episode")
        return self._request("exchange", {"sim": sim.to_payload()})

    def hold(self, reason: str) -> OperatorSample:
        if not self._episode_active:
            raise PiperXBridgeProtocolError("hold requires an active PiPER-X episode")
        try:
            return self._request("hold", {"reason": _reason(reason)})
        finally:
            # HOLD, like END, clears motion anchors. Keep session heartbeat
            # alive in idle state, but require a new begin before exchange.
            self._episode_active = False

    def end_episode(self, *, reason: str) -> OperatorSample:
        if not self._episode_active:
            raise PiperXBridgeProtocolError("end_episode requires an active PiPER-X episode")
        try:
            return self._request(
                "end_episode",
                {"reason": _reason(reason)},
            )
        finally:
            # Keep the last episode identity and heartbeat alive while the
            # CPU LeRobot writer commits video. Followers are already held by
            # END; the heartbeat refreshes session liveness only.
            self._episode_active = False

    def fail_closed(self, reason: str) -> None:
        """Best-effort follower hold followed by connection teardown."""

        self._stop_heartbeat()
        if self._episode_id is not None and self._socket is not None:
            try:
                self._request(
                    "hold",
                    {"reason": _reason(reason)},
                    ignore_heartbeat_error=True,
                )
            except Exception:
                pass
        self._episode_id = None
        self._episode_active = False
        self.close()
        # A fail-closed boundary is intentionally non-replayable even when
        # the final HOLD acknowledgement was received.  A new process/session
        # is required before physical motion can resume.
        self._session_lost = True

    def close(self) -> None:
        self._stop_heartbeat()
        self._close_socket_only()

    def _close_socket_only(self) -> None:
        connection = self._socket
        self._socket = None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None:
            raise PiperXBridgeProtocolError("PiPER-X heartbeat is already running")
        self._heartbeat_stop.clear()

        def heartbeat_loop() -> None:
            while not self._heartbeat_stop.wait(self.heartbeat_interval_s):
                try:
                    self._request("heartbeat", {}, from_heartbeat=True)
                except PiperXBridgeError as exc:
                    self._heartbeat_error = exc
                    self._heartbeat_stop.set()
                    return

        self._heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name="robodojo-piperx-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        self._heartbeat_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.response_timeout_s * 2.0))

    def __enter__(self) -> PiperXBridgeClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()


def client_from_environment() -> PiperXBridgeClient:
    """Build the loopback client without importing any hardware package."""

    return PiperXBridgeClient(
        host=os.environ.get("ROBODOJO_PIPERX_BRIDGE_HOST", "127.0.0.1"),
        port=int(os.environ.get("ROBODOJO_PIPERX_BRIDGE_PORT", "8765")),
        connect_timeout_s=float(os.environ.get("ROBODOJO_PIPERX_CONNECT_TIMEOUT_S", "5.0")),
        response_timeout_s=float(os.environ.get("ROBODOJO_PIPERX_RESPONSE_TIMEOUT_S", "0.2")),
        heartbeat_interval_s=float(os.environ.get("ROBODOJO_PIPERX_HEARTBEAT_INTERVAL_S", "0.25")),
    )


_SHARED_CLIENT_LOCK = threading.RLock()
_SHARED_CLIENT: PiperXBridgeClient | None = None
_SHARED_CLIENT_IDENTITY: tuple[Any, ...] | None = None


def _environment_identity() -> tuple[Any, ...]:
    return (
        os.environ.get("ROBODOJO_PIPERX_BRIDGE_HOST", "127.0.0.1"),
        int(os.environ.get("ROBODOJO_PIPERX_BRIDGE_PORT", "8765")),
        float(os.environ.get("ROBODOJO_PIPERX_CONNECT_TIMEOUT_S", "5.0")),
        float(os.environ.get("ROBODOJO_PIPERX_RESPONSE_TIMEOUT_S", "0.2")),
        float(os.environ.get("ROBODOJO_PIPERX_HEARTBEAT_INTERVAL_S", "0.25")),
    )


def shared_client_from_environment() -> PiperXBridgeClient:
    """Reuse one TCP session across simulator resets and dataset episodes."""

    global _SHARED_CLIENT, _SHARED_CLIENT_IDENTITY
    identity = _environment_identity()
    with _SHARED_CLIENT_LOCK:
        if _SHARED_CLIENT is None:
            _SHARED_CLIENT = client_from_environment()
            _SHARED_CLIENT_IDENTITY = identity
        elif _SHARED_CLIENT_IDENTITY != identity:
            raise PiperXBridgeProtocolError("Cannot change PiPER-X bridge configuration while a session is open")
        return _SHARED_CLIENT


def close_piperx_bridge_session() -> None:
    """Close the process-level bridge session during final/re-exec cleanup."""

    global _SHARED_CLIENT, _SHARED_CLIENT_IDENTITY
    with _SHARED_CLIENT_LOCK:
        client = _SHARED_CLIENT
        _SHARED_CLIENT = None
        _SHARED_CLIENT_IDENTITY = None
    if client is not None:
        if client.episode_active:
            client.fail_closed("robodojo_process_exit")
        else:
            client.close()
