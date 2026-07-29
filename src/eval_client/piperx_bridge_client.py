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

PROTOCOL = "robodojo_piperx_v3"
EMBODIMENT_PROFILE = "arx_x5_piperx_relative_v1"
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
_REQUEST_TYPES = frozenset(
    {
        "arm_and_begin_episode",
        "exchange",
        "transition_ack",
        "manual_sample",
        "manual_resolve",
        "heartbeat",
        "hold",
        "end_episode",
    }
)
_MODES = frozenset({"policy", "intervention", "fault"})
_CONTROL_TOPOLOGY = "policy_sim_to_follower_to_leader_manual_leader_joint_fanout"
_LEADER_ACTUATION_MODES = frozenset({"output_follow", "native_leader", "disabled", "fault"})
_FOLLOWER_ACTUATION_MODES = frozenset({"sim_follow", "leader_follow", "hold", "disabled", "fault"})
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
    """One physical leader state whose pose was computed from the same cached joints."""

    pose: tuple[float, ...]
    gripper_m: float
    sampled_monotonic_ns: int


@dataclass(frozen=True)
class ManualSample:
    """A single-use, bimanual leader sample cached by the hardware bridge."""

    sample_id: int
    left: OperatorArmSample
    right: OperatorArmSample


@dataclass(frozen=True)
class ManualResolution:
    """Hardware outcome for one exact manual sample."""

    sample_id: int
    decision: str
    follower_commanded: bool


@dataclass(frozen=True)
class OperatorSample:
    """One coherent bridge response for both PiPER-X leaders."""

    generation: int
    seq: int
    mode: str
    control_topology: str
    embodiment_profile: str
    leader_actuation_mode: str
    follower_actuation_mode: str
    transition: str | None
    edge: str | None
    terminal_request: str | None
    manual_sample: ManualSample | None
    manual_resolution: ManualResolution | None
    motion_accepted: bool
    health: dict[str, Any]
    diagnostics: dict[str, Any]

    @property
    def left(self) -> OperatorArmSample:
        if self.manual_sample is None:
            raise PiperXBridgeProtocolError("Response does not carry a manual leader sample")
        return self.manual_sample.left

    @property
    def right(self) -> OperatorArmSample:
        if self.manual_sample is None:
            raise PiperXBridgeProtocolError("Response does not carry a manual leader sample")
        return self.manual_sample.right


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
    expected = {"pose", "gripper_m", "sampled_monotonic_ns"}
    if not isinstance(value, dict) or set(value) != expected:
        raise PiperXBridgeProtocolError(
            f"{label} must contain exactly pose, gripper_m and sampled_monotonic_ns"
        )
    gripper_m = _finite_number(value["gripper_m"], label=f"{label}.gripper_m")
    if not 0.0 <= gripper_m <= 0.2:
        raise PiperXBridgeProtocolError(f"{label}.gripper_m must be in [0, 0.2]")
    return OperatorArmSample(
        pose=_pose(value["pose"], label=f"{label}.pose"),
        gripper_m=gripper_m,
        sampled_monotonic_ns=_non_negative_int(
            value["sampled_monotonic_ns"], label=f"{label}.sampled_monotonic_ns"
        ),
    )


def _parse_manual_sample(value: Any) -> ManualSample | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"sample_id", "left", "right"}:
        raise PiperXBridgeProtocolError(
            "manual_sample must be null or contain exactly sample_id, left and right"
        )
    sample_id = _non_negative_int(value["sample_id"], label="manual_sample.sample_id")
    if sample_id < 1:
        raise PiperXBridgeProtocolError("manual_sample.sample_id must be positive")
    return ManualSample(
        sample_id=sample_id,
        left=_parse_arm_sample(value["left"], label="manual_sample.left"),
        right=_parse_arm_sample(value["right"], label="manual_sample.right"),
    )


def _parse_manual_resolution(value: Any) -> ManualResolution | None:
    if value is None:
        return None
    expected = {"sample_id", "decision", "follower_commanded"}
    if not isinstance(value, dict) or set(value) != expected:
        raise PiperXBridgeProtocolError(
            "manual_resolution must be null or contain exactly sample_id, decision and follower_commanded"
        )
    sample_id = _non_negative_int(value["sample_id"], label="manual_resolution.sample_id")
    decision = value["decision"]
    follower_commanded = value["follower_commanded"]
    if sample_id < 1 or decision not in {"anchor", "commit", "reject"} or not isinstance(
        follower_commanded, bool
    ):
        raise PiperXBridgeProtocolError("manual_resolution contains an invalid outcome")
    if follower_commanded is not (decision == "commit"):
        raise PiperXBridgeProtocolError("manual_resolution decision disagrees with follower_commanded")
    return ManualResolution(
        sample_id=sample_id,
        decision=decision,
        follower_commanded=follower_commanded,
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
        arm_timeout_s: float = 60.0,
        transition_timeout_s: float = 10.0,
        heartbeat_interval_s: float = 0.25,
        session_id: str | None = None,
        max_frame_bytes: int = MAX_FRAME_BYTES,
    ):
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("PiPER-X bridge host must be loopback (127.0.0.1, ::1, or localhost)")
        if not 1 <= int(port) <= 65535:
            raise ValueError("PiPER-X bridge port must be in [1, 65535]")
        if (
            connect_timeout_s <= 0
            or response_timeout_s <= 0
            or arm_timeout_s <= 0
            or transition_timeout_s <= 0
            or heartbeat_interval_s <= 0
        ):
            raise ValueError("PiPER-X bridge timeouts must be positive")
        if max_frame_bytes <= 0:
            raise ValueError("PiPER-X bridge frame limit must be positive")
        self.host = host
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.response_timeout_s = float(response_timeout_s)
        self.arm_timeout_s = float(arm_timeout_s)
        self.transition_timeout_s = float(transition_timeout_s)
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.session_id = _identifier(session_id or str(uuid.uuid4()), label="session_id")
        self.max_frame_bytes = int(max_frame_bytes)
        self._socket: socket.socket | None = None
        self._episode_id: str | None = None
        self._episode_active = False
        self._generation = 0
        self._seq = 0
        self._pending_manual_sample_id: int | None = None
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
        timeout_s: float | None = None,
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
            request_timeout_s = self.response_timeout_s if timeout_s is None else float(timeout_s)
            if request_timeout_s <= 0:
                raise ValueError("PiPER-X request timeout must be positive")
            sent_ns = time.monotonic_ns()
            deadline_ns = sent_ns + int(request_timeout_s * 1_000_000_000)
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
                self._socket.settimeout(request_timeout_s)
                self._socket.sendall(encode_frame(envelope, max_frame_bytes=self.max_frame_bytes))
                response = receive_frame(self._socket, max_frame_bytes=self.max_frame_bytes)
                self._socket.settimeout(self.response_timeout_s)
                self._seq += 1
                sample = self._parse_response(
                    response,
                    request_type=request_type,
                    request_payload=payload,
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
        request_payload: dict[str, Any],
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
            "edge",
            "transition",
            "terminal_request",
            "manual_sample",
            "manual_resolution",
            "motion_accepted",
            "embodiment_profile",
            "control_topology",
            "leader_actuation_mode",
            "follower_actuation_mode",
            "health",
            "diagnostics",
        }
        if not isinstance(payload, dict) or set(payload) != expected_payload_keys:
            raise PiperXBridgeProtocolError(
                "PiPER-X v3 response payload keys do not match the strict schema"
            )
        mode = payload["mode"]
        edge = payload["edge"]
        transition = payload["transition"]
        terminal_request = payload["terminal_request"]
        manual_sample = _parse_manual_sample(payload["manual_sample"])
        manual_resolution = _parse_manual_resolution(payload["manual_resolution"])
        motion_accepted = payload["motion_accepted"]
        embodiment_profile = payload["embodiment_profile"]
        control_topology = payload["control_topology"]
        leader_actuation_mode = payload["leader_actuation_mode"]
        follower_actuation_mode = payload["follower_actuation_mode"]
        if mode not in _MODES or edge not in _EDGES or terminal_request not in _TERMINAL_REQUESTS:
            raise PiperXBridgeProtocolError("PiPER-X response has an invalid mode, edge, or terminal request")
        if control_topology != _CONTROL_TOPOLOGY:
            raise PiperXBridgeProtocolError("PiPER-X response has an invalid control_topology")
        if embodiment_profile != EMBODIMENT_PROFILE:
            raise PiperXBridgeProtocolError("PiPER-X response has an invalid embodiment_profile")
        if leader_actuation_mode not in _LEADER_ACTUATION_MODES:
            raise PiperXBridgeProtocolError("PiPER-X response has an invalid leader_actuation_mode")
        if follower_actuation_mode not in _FOLLOWER_ACTUATION_MODES:
            raise PiperXBridgeProtocolError("PiPER-X response has an invalid follower_actuation_mode")
        if transition not in _TRANSITIONS:
            raise PiperXBridgeProtocolError("PiPER-X response has an invalid transition")
        if not isinstance(motion_accepted, bool):
            raise PiperXBridgeProtocolError("PiPER-X motion_accepted must be boolean")
        if transition is not None:
            expected_mode = {
                "entering_intervention": "policy",
                "reattaching_policy": "intervention",
            }[transition]
            if mode != expected_mode:
                raise PiperXBridgeProtocolError(
                    f"PiPER-X transition {transition!r} is inconsistent with mode {mode!r}"
                )
            if edge is not None or motion_accepted is not False:
                raise PiperXBridgeProtocolError(
                    "PiPER-X transition must freeze motion and cannot carry an edge"
                )
            if manual_sample is not None or manual_resolution is not None:
                raise PiperXBridgeProtocolError("PiPER-X transition cannot carry a manual sample/result")
            if follower_actuation_mode != "hold":
                raise PiperXBridgeSafetyError(
                    "PiPER-X followers must report measured hold during a transition"
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
        if not isinstance(payload["health"], dict) or not isinstance(payload["diagnostics"], dict):
            raise PiperXBridgeProtocolError("PiPER-X health and diagnostics must be objects")
        if not isinstance(payload["diagnostics"].get("request_ok"), bool):
            raise PiperXBridgeProtocolError("PiPER-X diagnostics.request_ok must be present and boolean")
        sample = OperatorSample(
            generation=generation,
            seq=sequence,
            mode=mode,
            control_topology=control_topology,
            embodiment_profile=embodiment_profile,
            leader_actuation_mode=leader_actuation_mode,
            follower_actuation_mode=follower_actuation_mode,
            transition=transition,
            edge=edge,
            terminal_request=terminal_request,
            manual_sample=manual_sample,
            manual_resolution=manual_resolution,
            motion_accepted=motion_accepted,
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
            ("embodiment_profile", sample.embodiment_profile),
            ("leader_actuation_mode", sample.leader_actuation_mode),
            ("follower_actuation_mode", sample.follower_actuation_mode),
        ):
            if sample.health.get(key) != expected:
                raise PiperXBridgeSafetyError(
                    f"PiPER-X response {key} disagrees with hardware health: "
                    f"payload={expected!r}, health={sample.health.get(key)!r}"
                )
        if transition is None and mode != "fault":
            expected_leader_actuation = {
                "policy": "output_follow",
                "intervention": "native_leader",
            }[mode]
            if sample.leader_actuation_mode != expected_leader_actuation:
                raise PiperXBridgeSafetyError(
                    "PiPER-X leader actuation does not match the acknowledged bridge mode: "
                    f"mode={mode}, leader_actuation_mode={sample.leader_actuation_mode}"
                )
        if manual_sample is not None:
            if request_type != "manual_sample" or mode != "intervention" or transition is not None:
                raise PiperXBridgeProtocolError("manual_sample appeared on an invalid response")
            if motion_accepted:
                raise PiperXBridgeProtocolError("manual sampling must not move hardware")
        elif request_type == "manual_sample" and transition is None and terminal_request is None:
            raise PiperXBridgeProtocolError("steady manual_sample response omitted its sample")

        if manual_resolution is not None:
            if request_type != "manual_resolve" or mode != "intervention" or transition is not None:
                raise PiperXBridgeProtocolError("manual_resolution appeared on an invalid response")
            if (
                manual_resolution.sample_id != request_payload.get("sample_id")
                or manual_resolution.decision != request_payload.get("decision")
            ):
                raise PiperXBridgeProtocolError(
                    "manual_resolution sample_id/decision does not match the exact request"
                )
            if not motion_accepted:
                raise PiperXBridgeProtocolError("resolved manual sample must confirm its hardware outcome")
        elif request_type == "manual_resolve" and transition is None and terminal_request is None:
            raise PiperXBridgeProtocolError("steady manual_resolve response omitted its result")

        if request_type in {"arm_and_begin_episode", "exchange", "transition_ack", "hold", "end_episode"}:
            safe_noop = transition is not None or terminal_request is not None
            if not motion_accepted and not safe_noop:
                raise PiperXBridgeSafetyError(
                    f"PiPER-X bridge rejected {request_type}: {sample.diagnostics}"
                )
        return sample

    def arm_and_begin_episode(
        self,
        episode_id: str,
        sim: SimTargets,
    ) -> OperatorSample:
        if self._episode_active:
            raise PiperXBridgeProtocolError("A PiPER-X bridge episode is already active")
        with self._request_lock:
            self._episode_id = _identifier(episode_id, label="episode_id")
            self._generation = 0
            self._pending_manual_sample_id = None
            self._heartbeat_error = None
            try:
                sample = self._request(
                    "arm_and_begin_episode",
                    {"sim": sim.to_payload()},
                    timeout_s=self.arm_timeout_s,
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
            raise PiperXBridgeProtocolError(
                "arm_and_begin_episode must enter policy mode at generation 0"
            )
        self._episode_active = True
        if self._heartbeat_thread is None:
            self._start_heartbeat()
        return sample

    # Preserve the evaluator-facing name while making the physical arming
    # boundary explicit on the wire. The hardware bridge arms only on the
    # first call and reuses its held session for later episodes.
    begin_episode = arm_and_begin_episode

    def exchange(self, sim: SimTargets) -> OperatorSample:
        if not self._episode_active:
            raise PiperXBridgeProtocolError("exchange requires an active PiPER-X episode")
        return self._request("exchange", {"sim": sim.to_payload()})

    def transition_ack(self, sim: SimTargets) -> OperatorSample:
        if not self._episode_active:
            raise PiperXBridgeProtocolError("transition_ack requires an active PiPER-X episode")
        sample = self._request(
            "transition_ack",
            {"sim": sim.to_payload()},
            timeout_s=self.transition_timeout_s,
        )
        self._pending_manual_sample_id = None
        if sample.edge not in {"enter", "exit"}:
            raise PiperXBridgeProtocolError("transition_ack did not complete one control edge")
        return sample

    def manual_sample(self) -> OperatorSample:
        if not self._episode_active:
            raise PiperXBridgeProtocolError("manual_sample requires an active PiPER-X episode")
        if self._pending_manual_sample_id is not None:
            raise PiperXBridgeProtocolError(
                f"manual sample {self._pending_manual_sample_id} must be resolved before sampling again"
            )
        sample = self._request("manual_sample", {})
        if sample.manual_sample is not None:
            self._pending_manual_sample_id = sample.manual_sample.sample_id
        return sample

    def _resolve_manual_sample(self, sample_id: int, decision: str) -> OperatorSample:
        if not self._episode_active:
            raise PiperXBridgeProtocolError("manual_resolve requires an active PiPER-X episode")
        if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 1:
            raise ValueError("manual sample_id must be a positive integer")
        if decision not in {"anchor", "commit", "reject"}:
            raise ValueError("manual decision must be anchor, commit, or reject")
        if sample_id != self._pending_manual_sample_id:
            raise PiperXBridgeProtocolError(
                f"manual sample {sample_id} is not the one pending exact resolution "
                f"({self._pending_manual_sample_id})"
            )
        sample = self._request(
            "manual_resolve",
            {"sample_id": sample_id, "decision": decision},
        )
        if sample.transition is not None or sample.manual_resolution is not None:
            self._pending_manual_sample_id = None
        return sample

    def anchor_manual_sample(self, sample_id: int) -> OperatorSample:
        """Latch the post-switch leader/follower zero from one exact sample."""

        return self._resolve_manual_sample(sample_id, "anchor")

    def manual_resolve(self, sample_id: int, *, commit: bool) -> OperatorSample:
        return self._resolve_manual_sample(sample_id, "commit" if commit else "reject")

    def hold(self, reason: str) -> OperatorSample:
        if not self._episode_active:
            raise PiperXBridgeProtocolError("hold requires an active PiPER-X episode")
        try:
            return self._request("hold", {"reason": _reason(reason)})
        finally:
            # HOLD, like END, clears motion anchors. Keep session heartbeat
            # alive in idle state, but require a new begin before exchange.
            self._episode_active = False
            self._pending_manual_sample_id = None

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
            self._pending_manual_sample_id = None

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
        self._pending_manual_sample_id = None
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
        arm_timeout_s=float(os.environ.get("ROBODOJO_PIPERX_ARM_TIMEOUT_S", "60.0")),
        transition_timeout_s=float(os.environ.get("ROBODOJO_PIPERX_TRANSITION_TIMEOUT_S", "10.0")),
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
        float(os.environ.get("ROBODOJO_PIPERX_ARM_TIMEOUT_S", "60.0")),
        float(os.environ.get("ROBODOJO_PIPERX_TRANSITION_TIMEOUT_S", "10.0")),
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
