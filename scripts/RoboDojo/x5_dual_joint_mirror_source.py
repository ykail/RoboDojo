#!/usr/bin/env python3
"""Dual ARX X5 source for the RoboDojo joint-mirror client.

The wire format is ``robodojo_dual_joint_mirror_v1`` and remains compatible
with the existing :class:`DualJointMirrorClient`.  The physical and simulated
robots are both ARX X5, so follow targets use identity relative joint deltas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import select
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
import uuid
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_client.x5_hardware import (  # noqa: E402
    ArmTarget,
    DualState,
    DualTarget,
    DualX5Hardware,
    SIDES,
    X5HardwareConfig,
    X5HardwareError,
)


PROFILE = "arx_x5_identity_joint_v1"
CONTROL_MODE = "x5_policy_joint_intervention"
PROTOCOL = "robodojo_dual_joint_mirror_v1"
JOINT_DOF = 6
RAW_FRAGMENT_FORMAT = "robodojo_x5_manual_fragment_v1"

class X5SourceError(RuntimeError):
    pass


class _RawManualFragmentRecorder:
    """Accumulate manual X5 feedback and atomically publish one compact NPZ.

    Hardware access deliberately does not live here.  ``observe`` receives
    states read by the socket server's single serial loop, so the ARX SDK is
    never entered concurrently by a sampling worker and a control request.
    """

    def __init__(self, root: Path, *, frequency_hz: float) -> None:
        self.root = Path(root).expanduser().resolve()
        self.frequency_hz = float(frequency_hz)
        self._samples: list[tuple[int, int, DualState]] = []
        self._segments: list[dict[str, Any]] = []
        self._active_segment: int | None = None
        self._finalized = False

    @staticmethod
    def _sample_timestamp_ns(state: DualState) -> int:
        left = int(state.left.sample_monotonic_ns)
        right = int(state.right.sample_monotonic_ns)
        if left != right or left < 0:
            raise X5SourceError(
                "left/right X5 raw samples must share one non-negative monotonic timestamp"
            )
        return left

    def start_segment(self, boundary_monotonic_ns: int, anchor: DualState) -> int:
        if self._finalized:
            raise X5SourceError("cannot append to a finalized X5 raw fragment")
        if self._active_segment is not None:
            raise X5SourceError("an X5 manual raw segment is already active")
        boundary = int(boundary_monotonic_ns)
        if boundary < 0:
            raise X5SourceError("manual segment boundary must be non-negative")
        # Validate the anchor before mutating recorder state.  If this fails
        # after the hardware entered teach, JointMirrorSession restores an
        # active hold and the recorder remains ready to discard cleanly.
        self._sample_timestamp_ns(anchor)
        index = len(self._segments)
        self._segments.append(
            {
                "start_ns": boundary,
                "end_ns": None,
                "anchor": anchor,
            }
        )
        self._active_segment = index
        # enter_teach() returns the first feedback acquired after changing the
        # control mode.  Preserve it as the exact manual anchor as well as the
        # first trajectory sample; later deadline reads extend the segment.
        try:
            self.observe(anchor)
        except BaseException:
            self._segments.pop()
            self._active_segment = None
            raise
        return index

    def observe(self, state: DualState) -> None:
        if self._active_segment is None or self._finalized:
            return
        timestamp_ns = self._sample_timestamp_ns(state)
        segment = self._segments[self._active_segment]
        if timestamp_ns < int(segment["start_ns"]):
            return
        if self._samples and timestamp_ns <= self._samples[-1][0]:
            # SDK feedback occasionally exposes the same cached timestamp on
            # two adjacent polls.  It is not a new physical sample.
            return
        self._samples.append((timestamp_ns, self._active_segment, state))

    def end_segment(self, boundary_monotonic_ns: int) -> int | None:
        if self._active_segment is None:
            return None
        boundary = int(boundary_monotonic_ns)
        segment_index = self._active_segment
        segment = self._segments[segment_index]
        if boundary < int(segment["start_ns"]):
            raise X5SourceError("manual end boundary precedes its start boundary")
        segment["end_ns"] = boundary
        # Key events are timestamped before the next socket boundary.  A
        # deadline read can therefore land after Right/exit but before Isaac
        # acknowledges it.  Exclude those future samples exactly at commit.
        self._samples = [
            item
            for item in self._samples
            if item[1] != segment_index or item[0] <= boundary
        ]
        self._active_segment = None
        return segment_index

    @staticmethod
    def _arm_q(states: list[DualState], side: str) -> np.ndarray:
        if not states:
            return np.empty((0, JOINT_DOF), dtype=np.float64)
        result = np.asarray([state.side(side).q_rad for state in states], dtype=np.float64)
        if result.shape != (len(states), JOINT_DOF) or not np.isfinite(result).all():
            raise X5SourceError(f"invalid {side} X5 raw joint samples")
        return result

    @staticmethod
    def _arm_gripper(states: list[DualState], side: str) -> np.ndarray:
        result = np.asarray(
            [state.side(side).gripper_open_fraction for state in states],
            dtype=np.float64,
        )
        if result.shape != (len(states),) or not np.isfinite(result).all():
            raise X5SourceError(f"invalid {side} X5 raw gripper samples")
        return result

    def _arrays(self) -> dict[str, np.ndarray]:
        if self._active_segment is not None:
            raise X5SourceError("cannot finalize while an X5 manual segment is active")
        states = [item[2] for item in self._samples]
        anchors = [item["anchor"] for item in self._segments]
        ends = [item["end_ns"] for item in self._segments]
        if any(value is None for value in ends):
            raise X5SourceError("cannot finalize an X5 fragment with an open segment")
        return {
            "sample_monotonic_ns": np.asarray(
                [item[0] for item in self._samples], dtype=np.int64
            ),
            "segment_index": np.asarray([item[1] for item in self._samples], dtype=np.int32),
            "left_q_rad": self._arm_q(states, "left"),
            "right_q_rad": self._arm_q(states, "right"),
            "left_gripper_open_fraction": self._arm_gripper(states, "left"),
            "right_gripper_open_fraction": self._arm_gripper(states, "right"),
            "segment_start_ns": np.asarray(
                [item["start_ns"] for item in self._segments], dtype=np.int64
            ),
            "segment_end_ns": np.asarray(ends, dtype=np.int64),
            "segment_anchor_timestamp_ns": np.asarray(
                [self._sample_timestamp_ns(item) for item in anchors], dtype=np.int64
            ),
            "left_segment_anchor_q_rad": self._arm_q(anchors, "left"),
            "right_segment_anchor_q_rad": self._arm_q(anchors, "right"),
            "left_segment_anchor_gripper_open_fraction": self._arm_gripper(
                anchors, "left"
            ),
            "right_segment_anchor_gripper_open_fraction": self._arm_gripper(
                anchors, "right"
            ),
            "format_version": np.asarray(1, dtype=np.int32),
            "sampling_frequency_hz": np.asarray(self.frequency_hz, dtype=np.float64),
        }

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise X5SourceError("X5 raw fragment was already finalized")
        arrays = self._arrays()
        self.root.mkdir(parents=True, exist_ok=True)
        stem = f"x5_manual_{time.monotonic_ns()}_{os.getpid()}_{uuid.uuid4().hex}"
        partial = self.root / f".{stem}.npz.partial"
        final = self.root / f"{stem}.npz"
        try:
            with partial.open("xb") as stream:
                # Uncompressed NPZ is intentionally used here: Right should
                # return quickly, while the bundle store will copy it once.
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            digest = hashlib.sha256()
            with partial.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            os.replace(partial, final)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            if partial.exists():
                partial.unlink()
            raise
        self._finalized = True
        return {
            "format": RAW_FRAGMENT_FORMAT,
            "path": str(final),
            "sha256": "sha256:" + digest.hexdigest(),
            "sample_count": int(arrays["sample_monotonic_ns"].shape[0]),
            "segment_count": int(arrays["segment_start_ns"].shape[0]),
            "frequency_hz": self.frequency_hz,
        }

    def discard(self) -> None:
        self._samples.clear()
        self._segments.clear()
        self._active_segment = None
        self._finalized = True


def _finite_vector(value: Any, *, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise X5SourceError(f"{name} must be a numeric vector") from exc
    if len(result) != JOINT_DOF or not all(math.isfinite(item) for item in result):
        raise X5SourceError(f"{name} must contain six finite values")
    return result


def parse_request(
    payload: Any,
    *,
    require_zero: bool,
) -> tuple[str, int, dict[str, tuple[tuple[float, ...], float, float | None]] | None]:
    if not isinstance(payload, dict):
        raise X5SourceError("joint-mirror request must be a JSON object")
    request_type = payload.get("type")
    seq = payload.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool):
        raise X5SourceError("joint-mirror request requires an integer seq")
    if request_type in {"heartbeat", "end"}:
        if set(payload) != {"type", "seq"}:
            raise X5SourceError(f"invalid {request_type} request envelope")
        return request_type, seq, None
    if request_type != "joint_mirror" or set(payload) != {"type", "seq", "sides"}:
        raise X5SourceError("invalid joint-mirror request envelope")
    sides = payload["sides"]
    if not isinstance(sides, dict) or set(sides) != set(SIDES):
        raise X5SourceError("joint-mirror request requires exactly left and right")

    parsed: dict[str, tuple[tuple[float, ...], float, float | None]] = {}
    for side in SIDES:
        item = sides[side]
        if not isinstance(item, dict):
            raise X5SourceError(f"invalid {side} joint-mirror payload")
        q_delta = _finite_vector(item.get("delta_q_rad"), name=f"{side} delta_q_rad")
        try:
            gripper_delta = float(item["delta_gripper_open_fraction"])
            raw_absolute = item.get("sim_gripper_open_fraction")
            absolute = None if raw_absolute is None else float(raw_absolute)
        except (KeyError, TypeError, ValueError) as exc:
            raise X5SourceError(f"invalid {side} gripper payload") from exc
        if not math.isfinite(gripper_delta) or not -1.0 <= gripper_delta <= 1.0:
            raise X5SourceError(f"invalid {side} gripper delta")
        if absolute is not None and (not math.isfinite(absolute) or not 0.0 <= absolute <= 1.0):
            raise X5SourceError(f"invalid {side} absolute gripper target")
        if require_zero and (
            any(abs(value) > 1e-6 for value in q_delta) or abs(gripper_delta) > 1e-6
        ):
            raise X5SourceError("first frame after an anchor must be zero on both sides")
        parsed[side] = (q_delta, gripper_delta, absolute)
    return request_type, seq, parsed


def _target_for_side(
    anchor: ArmTarget,
    request: tuple[tuple[float, ...], float, float | None],
) -> ArmTarget:
    delta_q, delta_gripper, absolute_gripper = request
    return ArmTarget(
        tuple(anchor_value + delta for anchor_value, delta in zip(anchor.q_rad, delta_q)),
        (
            anchor.gripper_open_fraction + delta_gripper
            if absolute_gripper is None
            else absolute_gripper
        ),
    )


class JointMirrorSession:
    """Pure request/session state machine around a serial dual-X5 backend."""

    def __init__(
        self,
        hardware: DualX5Hardware,
        *,
        raw_root: Path | None = None,
        raw_frequency_hz: float = 100.0,
    ) -> None:
        self.hardware = hardware
        self.mode = "follow"
        self.require_zero = True
        self.pending_transition: str | None = None
        self.pending_transition_monotonic_ns: int | None = None
        self.pending_terminal: str | None = None
        self.pending_terminal_monotonic_ns: int | None = None
        self.terminal_delivered: str | None = None
        self.manual_anchor: DualState | None = None
        self.follow_anchor = hardware.latched_target
        self.ended = False
        self.raw_recorder = (
            None
            if raw_root is None
            else _RawManualFragmentRecorder(raw_root, frequency_hz=raw_frequency_hz)
        )

    def observe_hardware(self, measured: DualState) -> None:
        """Accept one deadline-driven read from the serial server loop."""

        if self.raw_recorder is not None:
            self.raw_recorder.observe(measured)

    def toggle_intervention(self, event_monotonic_ns: int | None = None) -> bool:
        if (
            self.ended
            or self.pending_transition is not None
            or self.pending_terminal is not None
            or self.terminal_delivered is not None
        ):
            return False
        self.pending_transition = "enter" if self.mode == "follow" else "exit"
        self.pending_transition_monotonic_ns = int(
            time.monotonic_ns() if event_monotonic_ns is None else event_monotonic_ns
        )
        return True

    def request_terminal(
        self,
        terminal: str,
        event_monotonic_ns: int | None = None,
    ) -> bool:
        if terminal not in {"save", "retry"}:
            raise X5SourceError(f"invalid terminal request {terminal!r}")
        if self.ended or self.pending_terminal is not None or self.terminal_delivered is not None:
            return False
        self.pending_terminal = terminal
        self.pending_terminal_monotonic_ns = int(
            time.monotonic_ns() if event_monotonic_ns is None else event_monotonic_ns
        )
        self.pending_transition = None
        self.pending_transition_monotonic_ns = None
        return True

    def _response(
        self,
        request_type: str,
        seq: int,
        edge: str | None,
        terminal: str | None = None,
        boundary_monotonic_ns: int | None = None,
        measured: DualState | None = None,
        raw_fragment: dict[str, Any] | None = None,
        raw_segment_index: int | None = None,
    ) -> dict[str, Any]:
        if measured is None:
            measured = self.hardware.read()
        sample_monotonic_ns = int(measured.left.sample_monotonic_ns)
        if int(measured.right.sample_monotonic_ns) != sample_monotonic_ns:
            raise X5SourceError("left/right X5 samples do not share one monotonic timestamp")
        follow_target = self.hardware.latched_target
        sides: dict[str, dict[str, Any]] = {}
        for side in SIDES:
            state = measured.side(side)
            target = follow_target.side(side)
            item: dict[str, Any] = {
                "measured_q_rad": [float(value) for value in state.q_rad],
                "measured_gripper_width_m": float(state.gripper_pos_m),
                "leader_gripper_open_fraction": float(state.gripper_open_fraction),
                "follow_target_q_rad": [float(value) for value in target.q_rad],
            }
            if self.mode == "manual":
                if self.manual_anchor is None:
                    raise X5SourceError("manual mode has no X5 anchor")
                anchor = self.manual_anchor.side(side)
                item["leader_delta_q_rad"] = [
                    float(now - initial)
                    for now, initial in zip(state.q_rad, anchor.q_rad)
                ]
                item["leader_delta_gripper_open_fraction"] = float(
                    state.gripper_open_fraction - anchor.gripper_open_fraction
                )
            sides[side] = item
        response = {
            "ok": True,
            "type": request_type,
            "seq": seq,
            "mode": self.mode,
            "edge": edge,
            "terminal": terminal,
            "sample_monotonic_ns": sample_monotonic_ns,
            "boundary_monotonic_ns": (
                None if boundary_monotonic_ns is None else int(boundary_monotonic_ns)
            ),
            "raw_segment_index": (
                None if raw_segment_index is None else int(raw_segment_index)
            ),
            "sides": sides,
        }
        if request_type == "end":
            response["raw_fragment"] = raw_fragment
        return response

    def handle(
        self,
        payload: Any,
        *,
        measured: DualState | None = None,
    ) -> dict[str, Any]:
        if self.ended:
            raise X5SourceError("joint-mirror session has ended")
        required_zero_for_request = self.require_zero
        request_type, seq, parsed = parse_request(
            payload,
            require_zero=required_zero_for_request,
        )
        edge: str | None = None
        boundary_monotonic_ns: int | None = None
        raw_segment_index: int | None = None

        if request_type == "end":
            self.hardware.enter_hold()
            self.mode = "follow"
            self.pending_transition = None
            self.pending_transition_monotonic_ns = None
            self.pending_terminal = None
            self.pending_terminal_monotonic_ns = None
            self.manual_anchor = None
            raw_fragment = None
            if self.raw_recorder is not None:
                if self.terminal_delivered == "save":
                    raw_fragment = self.raw_recorder.finalize()
                else:
                    self.raw_recorder.discard()
            response = self._response(
                request_type,
                seq,
                None,
                measured=measured,
                raw_fragment=raw_fragment,
            )
            self.ended = True
            return response

        if request_type == "joint_mirror":
            assert parsed is not None
            if self.pending_terminal is not None:
                # A terminal boundary dominates intervention edges.  Latch the
                # current physical pose with active gains before acknowledging
                # the key; run_server performs the slower smooth HOME move
                # after the client has sent END.
                self.hardware.enter_hold()
                self.mode = "follow"
                self.manual_anchor = None
                self.follow_anchor = self.hardware.latched_target
                self.require_zero = True
                terminal = self.pending_terminal
                boundary_monotonic_ns = self.pending_terminal_monotonic_ns
                if self.raw_recorder is not None and boundary_monotonic_ns is not None:
                    self.raw_recorder.end_segment(boundary_monotonic_ns)
                self.pending_terminal = None
                self.pending_terminal_monotonic_ns = None
                self.terminal_delivered = terminal
                return self._response(
                    request_type,
                    seq,
                    None,
                    terminal,
                    boundary_monotonic_ns,
                    measured=measured,
                )
            if (
                self.pending_transition == "enter"
                and required_zero_for_request
            ):
                # After startup or an intervention exit, Isaac sends one zero
                # synchronization request and requires a stable follow reply.
                # A very fast next ``i`` may already be queued at this point;
                # keep that edge pending for the following request instead of
                # turning the mandatory zero acknowledgement into manual mode.
                if self.mode != "follow":
                    raise X5SourceError(
                        "zero synchronization before manual entry requires follow mode"
                    )
                requested = {
                    side: _target_for_side(self.follow_anchor.side(side), parsed[side])
                    for side in SIDES
                }
                self.hardware.follow(DualTarget(requested["left"], requested["right"]))
                self.require_zero = False
                return self._response(
                    request_type,
                    seq,
                    None,
                    measured=measured,
                )
            if self.pending_transition == "enter":
                boundary_monotonic_ns = self.pending_transition_monotonic_ns
                try:
                    self.manual_anchor = self.hardware.enter_teach()
                    # The socket loop may have supplied its previous 100 Hz read.
                    # The transition response must instead expose the fresh state
                    # returned by enter_teach(), because that is the source anchor
                    # stored in the raw fragment and used by live relative mapping.
                    measured = self.manual_anchor
                    if self.raw_recorder is not None and boundary_monotonic_ns is not None:
                        raw_segment_index = self.raw_recorder.start_segment(
                            boundary_monotonic_ns,
                            self.manual_anchor,
                        )
                except Exception:
                    # enter_teach() changes the real controller mode before raw
                    # bookkeeping runs.  A recorder error must never escape
                    # while leaving the arms in teach/damping-like behavior.
                    self.mode = "follow"
                    self.require_zero = True
                    self.manual_anchor = None
                    self.pending_transition = None
                    self.pending_transition_monotonic_ns = None
                    try:
                        _recover_active_hold(self.hardware, attempts=None)
                    except Exception as hold_exc:
                        raise X5SourceError(
                            "manual entry failed and active hold could not be restored"
                        ) from hold_exc
                    self.follow_anchor = self.hardware.latched_target
                    raise
                self.mode = "manual"
                self.pending_transition = None
                self.pending_transition_monotonic_ns = None
                edge = "enter"
            elif self.pending_transition == "exit":
                boundary_monotonic_ns = self.pending_transition_monotonic_ns
                self.hardware.enter_hold()
                if self.raw_recorder is not None and boundary_monotonic_ns is not None:
                    raw_segment_index = self.raw_recorder.end_segment(
                        boundary_monotonic_ns
                    )
                    if raw_segment_index is None:
                        raise X5SourceError("manual exit has no active X5 raw segment")
                self.follow_anchor = self.hardware.latched_target
                self.manual_anchor = None
                self.mode = "follow"
                self.require_zero = True
                self.pending_transition = None
                self.pending_transition_monotonic_ns = None
                edge = "exit"
            elif self.mode == "follow":
                requested = {
                    side: _target_for_side(self.follow_anchor.side(side), parsed[side])
                    for side in SIDES
                }
                self.hardware.follow(DualTarget(requested["left"], requested["right"]))
                self.require_zero = False
            elif self.mode != "manual":
                raise X5SourceError(f"invalid source mode {self.mode!r}")

        return self._response(
            request_type,
            seq,
            edge,
            boundary_monotonic_ns=(
                boundary_monotonic_ns if request_type == "joint_mirror" else None
            ),
            measured=measured,
            raw_segment_index=raw_segment_index,
        )

    def disconnect(self) -> None:
        if self.hardware.is_connected:
            self.hardware.enter_hold()
        self.mode = "follow"
        self.pending_transition = None
        self.pending_transition_monotonic_ns = None
        self.pending_terminal = None
        self.pending_terminal_monotonic_ns = None
        self.manual_anchor = None
        if self.raw_recorder is not None and not self.ended:
            self.raw_recorder.discard()


class _KeyReader:
    """Read global i/arrow hotkeys through XInput with a terminal fallback."""

    _LOCK_MASK = 1 << 1
    _MOD2_MASK = 1 << 4
    _ALLOWED_MODIFIERS = {0, _LOCK_MASK, _MOD2_MASK, _LOCK_MASK | _MOD2_MASK}
    _EVENT_RE = re.compile(r"^EVENT type \d+ \((KeyPress|KeyRelease)\)$")
    _DETAIL_RE = re.compile(r"^detail:\s*(\d+)$")
    _MODIFIERS_RE = re.compile(r"^modifiers:.*effective:\s*(0x[0-9a-fA-F]+|\d+)$")

    def __init__(self, *, global_hotkeys: bool, display_name: str) -> None:
        self.global_hotkeys = bool(global_hotkeys)
        self.display_name = str(display_name)
        self.tty_enabled = False
        self.saved: list[Any] | None = None
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._global_failed = False
        self._key_by_code: dict[int, str] = {}
        self._down_codes: set[int] = set()
        self._pending: queue.Queue[tuple[str, int]] = queue.Queue(maxsize=8)
        self._lock = threading.RLock()

    @staticmethod
    def _environment(display_name: str) -> dict[str, str]:
        environment = dict(os.environ)
        environment["DISPLAY"] = display_name
        return environment

    @staticmethod
    def _keycode_map(environment: dict[str, str]) -> dict[int, str]:
        output = subprocess.check_output(
            ["/usr/bin/xmodmap", "-pke"],
            env=environment,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=3.0,
        )
        wanted = {"i": "i", "Left": "left", "Right": "right"}
        result: dict[int, str] = {}
        for line in output.splitlines():
            match = re.match(r"^keycode\s+(\d+)\s*=\s*(\S+)", line)
            if match and match.group(2) in wanted:
                result[int(match.group(1))] = wanted[match.group(2)]
        if set(result.values()) != set(wanted.values()):
            raise RuntimeError(f"incomplete X11 i/Left/Right keymap: {result}")
        return result

    def _publish(self, kind: str, keycode: int | None, modifiers: int | None, repeated: bool) -> None:
        if keycode not in self._key_by_code:
            return
        with self._lock:
            if kind == "KeyRelease":
                self._down_codes.discard(keycode)
                return
            if (
                kind != "KeyPress"
                or repeated
                or modifiers not in self._ALLOWED_MODIFIERS
                or keycode in self._down_codes
            ):
                return
            self._down_codes.add(keycode)
            try:
                self._pending.put_nowait(
                    (self._key_by_code[keycode], time.monotonic_ns())
                )
            except queue.Full:
                pass

    def _global_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._global_failed = True
            return
        kind: str | None = None
        detail: int | None = None
        modifiers: int | None = None
        repeated = False

        def flush() -> None:
            nonlocal kind, detail, modifiers, repeated
            if kind is not None:
                self._publish(kind, detail, modifiers, repeated)
            kind = None
            detail = None
            modifiers = None
            repeated = False

        try:
            for raw_line in process.stdout:
                if self._reader_stop.is_set():
                    break
                line = raw_line.strip()
                if line.startswith("EVENT type"):
                    flush()
                    match = self._EVENT_RE.match(line)
                    kind = match.group(1) if match else None
                elif kind is not None:
                    match = self._DETAIL_RE.match(line)
                    if match:
                        detail = int(match.group(1))
                    else:
                        match = self._MODIFIERS_RE.match(line)
                        if match:
                            modifiers = int(match.group(1), 0)
                        elif line.startswith("flags:") and "repeat" in line.lower():
                            repeated = True
                        elif not line:
                            flush()
            flush()
        finally:
            if not self._reader_stop.is_set():
                self._global_failed = True

    def _try_global(self) -> bool:
        if not self.global_hotkeys:
            return False
        try:
            environment = self._environment(self.display_name)
            self._key_by_code = self._keycode_map(environment)
            self._process = subprocess.Popen(
                ["/usr/bin/stdbuf", "-oL", "/usr/bin/xinput", "test-xi2", "--root"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
                text=True,
                bufsize=1,
            )
            if self._process.stdout is None:
                raise RuntimeError("xinput did not provide stdout")
            self._reader_thread = threading.Thread(target=self._global_loop, daemon=True)
            self._reader_thread.start()
            time.sleep(0.15)
            if self._process.poll() is not None:
                raise RuntimeError(f"xinput exited with status {self._process.returncode}")
            print(
                f"[Dual X5] global hotkeys active on {self.display_name}: "
                "i=toggle, Left=discard/retry, Right=save/next. "
                "Stop only with Ctrl-C in this terminal while supporting both arms.",
                flush=True,
            )
            return True
        except Exception as exc:
            self._stop_global()
            print(
                f"[Dual X5][WARN] global hotkeys unavailable: {exc}; falling back to this terminal.",
                flush=True,
            )
            return False

    def _enable_terminal(self) -> None:
        if self.tty_enabled or not sys.stdin.isatty():
            return
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        self.saved = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        self.tty_enabled = True

    def __enter__(self) -> "_KeyReader":
        if not self._try_global():
            self._enable_terminal()
        return self

    def poll(self) -> tuple[str, int] | None:
        if self._process is not None:
            if self._global_failed or self._process.poll() is not None:
                self._stop_global()
                self._enable_terminal()
            else:
                try:
                    return self._pending.get_nowait()
                except queue.Empty:
                    return None
        if self.tty_enabled and select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            if key == "\x1b" and select.select([sys.stdin], [], [], 0.02)[0]:
                suffix = sys.stdin.read(1)
                if suffix == "[" and select.select([sys.stdin], [], [], 0.02)[0]:
                    arrow = sys.stdin.read(1)
                    key_name = {"D": "left", "C": "right"}.get(arrow)
                    return None if key_name is None else (key_name, time.monotonic_ns())
                return None
            key = key.lower()
            return (key, time.monotonic_ns()) if key in {"i", "q"} else None
        return None

    def _stop_global(self) -> None:
        self._reader_stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        thread = self._reader_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        if process is not None and process.stdout is not None:
            process.stdout.close()
        self._process = None
        self._reader_thread = None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop_global()
        if self.saved is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.saved)


def _send(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode())


def _recover_active_hold(
    hardware: DualX5Hardware,
    *,
    attempts: int | None = None,
    retry_delay_s: float = 0.10,
) -> None:
    """Keep retrying a measured-pose active hold without closing the SDK.

    Closing the ARX controller switches it to damping.  During a live
    collection that turns a recoverable socket/transition error into a falling
    arm, so connection recovery must stay inside the owner process until an
    active hold has actually been confirmed.  ``attempts=None`` is used by the
    real server; finite attempts keep this helper directly testable.
    """

    if attempts is not None and attempts <= 0:
        raise ValueError("hold recovery attempts must be positive or None")
    if retry_delay_s < 0:
        raise ValueError("hold recovery delay must be non-negative")
    attempt = 0
    last_error: BaseException | None = None
    while attempts is None or attempt < attempts:
        attempt += 1
        try:
            hardware.enter_hold()
            hardware.hold()
            if attempt > 1:
                print(
                    f"[Dual X5] active hold recovered after {attempt} attempts.",
                    flush=True,
                )
            return
        except Exception as exc:
            last_error = exc
            if attempt == 1 or attempt % 10 == 0:
                print(
                    f"[Dual X5][WARN] active hold recovery attempt {attempt} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            if attempts is None or attempt < attempts:
                time.sleep(retry_delay_s)
    raise X5SourceError(
        f"active hold recovery failed after {attempt} attempts: {last_error}"
    ) from last_error


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve two real ARX X5 arms to RoboDojo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--left-can", default="can1")
    parser.add_argument("--right-can", default="can3")
    parser.add_argument("--left-model", default="X5")
    parser.add_argument("--right-model", default="X5")
    parser.add_argument("--frequency-hz", type=float, default=100.0)
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Write accepted manual X5 NPZ fragments below this directory.",
    )
    parser.add_argument("--follow-preview-s", type=float, default=0.04)
    parser.add_argument("--liveness-timeout-s", type=float, default=1.5)
    parser.add_argument("--home-rad", type=float, nargs=6, default=(0.0,) * 6)
    parser.add_argument("--home-gripper-fraction", type=float, default=1.0)
    parser.add_argument("--home-duration-s", type=float, default=5.0)
    parser.add_argument("--skip-home", action="store_true")
    parser.add_argument("--left-gripper-min-m", type=float, default=0.0)
    parser.add_argument("--left-gripper-max-m", type=float)
    parser.add_argument("--right-gripper-min-m", type=float, default=0.0)
    parser.add_argument("--right-gripper-max-m", type=float)
    parser.add_argument(
        "--active-joint-kp", type=float, nargs=6, default=(90.0, 80.0, 80.0, 35.0, 35.0, 25.0)
    )
    parser.add_argument(
        "--active-joint-kd", type=float, nargs=6, default=(2.5, 2.2, 2.2, 1.6, 1.6, 1.0)
    )
    parser.add_argument(
        "--teach-joint-kd", type=float, nargs=6, default=(0.3, 0.3, 0.3, 0.2, 0.1, 0.1)
    )
    parser.add_argument("--left-gripper-kp", type=float, default=2.0)
    parser.add_argument("--left-gripper-kd", type=float, default=0.15)
    parser.add_argument("--right-gripper-kp", type=float, default=3.0)
    parser.add_argument("--right-gripper-kd", type=float, default=0.2)
    parser.add_argument("--teach-gripper-kp", type=float, default=0.0)
    parser.add_argument("--teach-gripper-kd", type=float, default=0.0)
    parser.add_argument("--clear-on-init", action="store_true")
    parser.add_argument("--no-gravity-compensation", action="store_true")
    parser.add_argument("--no-global-hotkeys", action="store_true")
    parser.add_argument("--display", default=":1")
    return parser


def _config_from_args(args: argparse.Namespace) -> X5HardwareConfig:
    return X5HardwareConfig(
        left_model=args.left_model,
        right_model=args.right_model,
        left_can=args.left_can,
        right_can=args.right_can,
        clear_on_init=args.clear_on_init,
        gravity_compensation=not args.no_gravity_compensation,
        active_joint_kp=tuple(args.active_joint_kp),
        active_joint_kd=tuple(args.active_joint_kd),
        teach_joint_kd=tuple(args.teach_joint_kd),
        follow_preview_s=args.follow_preview_s,
        left_gripper_kp=args.left_gripper_kp,
        left_gripper_kd=args.left_gripper_kd,
        right_gripper_kp=args.right_gripper_kp,
        right_gripper_kd=args.right_gripper_kd,
        teach_gripper_kp=args.teach_gripper_kp,
        teach_gripper_kd=args.teach_gripper_kd,
        left_gripper_min_m=args.left_gripper_min_m,
        left_gripper_max_m=args.left_gripper_max_m,
        right_gripper_min_m=args.right_gripper_min_m,
        right_gripper_max_m=args.right_gripper_max_m,
    )


def _serve_connection(
    conn: socket.socket,
    hardware: DualX5Hardware,
    keys: _KeyReader,
    *,
    period_s: float,
    liveness_timeout_s: float,
    raw_root: Path | None = None,
) -> tuple[bool, bool, str | None]:
    """Return ``(quit_requested, clean_end, terminal)`` for one connection."""

    session = JointMirrorSession(
        hardware,
        raw_root=raw_root,
        raw_frequency_hz=1.0 / period_s,
    )
    buffer = b""
    last_liveness = time.monotonic()
    raw_sampling_enabled = raw_root is not None
    next_sample_deadline = time.monotonic()
    latest_measured: DualState | None = None
    while True:
        key_event = keys.poll()
        key = None if key_event is None else key_event[0]
        key_monotonic_ns = None if key_event is None else key_event[1]
        if key == "q":
            session.disconnect()
            return True, False, session.terminal_delivered
        if key == "i" and session.toggle_intervention(key_monotonic_ns):
            print(
                f"\n[Dual X5] intervention {session.pending_transition} requested; "
                "waiting for the next Isaac boundary.",
                flush=True,
            )
        elif key in {"left", "right"}:
            terminal = "retry" if key == "left" else "save"
            if session.request_terminal(terminal, key_monotonic_ns):
                label = "discard/retry" if terminal == "retry" else "save/next"
                print(
                    f"\n[Dual X5] {label} requested; holding at the next Isaac boundary.",
                    flush=True,
                )
        now = time.monotonic()
        if raw_sampling_enabled and now >= next_sample_deadline:
            latest_measured = hardware.read()
            session.observe_hardware(latest_measured)
            now = time.monotonic()
            skipped = max(1, math.floor((now - next_sample_deadline) / period_s) + 1)
            next_sample_deadline += skipped * period_s
        until_sample = (
            max(0.0005, next_sample_deadline - time.monotonic())
            if raw_sampling_enabled
            else max(0.005, period_s)
        )
        conn.settimeout(min(0.1, until_sample))
        try:
            chunk = conn.recv(65536)
        except socket.timeout:
            if time.monotonic() - last_liveness > liveness_timeout_s:
                session.disconnect()
                print(
                    "[Dual X5][WARN] Isaac liveness timed out; holding both arms.",
                    flush=True,
                )
                return False, False, session.terminal_delivered
            continue
        except OSError:
            chunk = b""
        if not chunk:
            session.disconnect()
            return False, False, session.terminal_delivered
        buffer += chunk
        if len(buffer) > 1024 * 1024:
            session.disconnect()
            raise X5SourceError("joint-mirror request buffer exceeded 1 MiB")
        while b"\n" in buffer:
            raw, buffer = buffer.split(b"\n", 1)
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                session.disconnect()
                raise X5SourceError(f"invalid JSON request: {raw!r}") from exc
            last_liveness = time.monotonic()
            response = session.handle(payload, measured=latest_measured)
            _send(conn, response)
            edge = response["edge"]
            if edge == "enter":
                print("\n[Dual X5] manual control ON.", flush=True)
            elif edge == "exit":
                print("\n[Dual X5] manual control OFF; holding the release pose.", flush=True)
            if session.ended:
                return False, True, session.terminal_delivered


def run_server(args: argparse.Namespace, *, hardware: DualX5Hardware | None = None) -> None:
    if args.frequency_hz <= 0 or args.home_duration_s < 0 or args.liveness_timeout_s <= 0:
        raise X5SourceError(
            "frequency/liveness timeout must be positive and home duration non-negative"
        )
    owns_hardware = hardware is None
    if hardware is None:
        hardware = DualX5Hardware(_config_from_args(args))
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    quit_requested = False
    try:
        hardware.connect()
        if not args.skip_home:
            print(
                f"[Dual X5] moving both arms smoothly to home over {args.home_duration_s:.1f}s; "
                "support the arms and keep the workspace clear.",
                flush=True,
            )
            hardware.move_home(
                tuple(args.home_rad),
                args.home_gripper_fraction,
                duration_s=args.home_duration_s,
                frequency_hz=args.frequency_hz,
            )
            print("[Dual X5] HOME command complete; no strict settle timeout is applied.", flush=True)
        server.bind((args.host, args.port))
        server.listen(1)
        server.settimeout(1.0 / args.frequency_hz)
        print(
            f"[Dual X5] profile={PROFILE} control_mode={CONTROL_MODE} protocol={PROTOCOL}",
            flush=True,
        )
        print(
            f"[Dual X5] listening on {args.host}:{args.port}; "
            "i toggles intervention, Left discards/retries, Right saves/advances. "
            "Stop only with Ctrl-C in this terminal while supporting both arms.",
            flush=True,
        )
        with _KeyReader(
            global_hotkeys=not args.no_global_hotkeys,
            display_name=args.display,
        ) as keys:
            while not quit_requested:
                key_event = keys.poll()
                key = None if key_event is None else key_event[0]
                if key == "q":
                    break
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    try:
                        hardware.hold()
                    except Exception as exc:
                        print(
                            f"[Dual X5][WARN] idle hold refresh failed: "
                            f"{type(exc).__name__}: {exc}; recovering active hold.",
                            flush=True,
                        )
                        _recover_active_hold(hardware, attempts=None)
                    continue
                print("[Dual X5] Isaac connected; follow anchor is the current hold pose.", flush=True)
                with conn:
                    try:
                        quit_requested, clean_end, terminal_request = _serve_connection(
                            conn,
                            hardware,
                            keys,
                            period_s=1.0 / args.frequency_hz,
                            liveness_timeout_s=args.liveness_timeout_s,
                            raw_root=args.raw_root,
                        )
                    except Exception as exc:
                        print(
                            f"[Dual X5][WARN] connection ended: {exc}; "
                            "keeping the controller alive until active hold is restored.",
                            flush=True,
                        )
                        _recover_active_hold(hardware, attempts=None)
                        clean_end = False
                        terminal_request = None
                if terminal_request is not None and not quit_requested:
                    label = "discard/retry" if terminal_request == "retry" else "save/next"
                    print(
                        f"[Dual X5] {label} acknowledged; moving both arms smoothly "
                        f"to HOME over {args.home_duration_s:.1f}s.",
                        flush=True,
                    )
                    try:
                        hardware.move_home(
                            tuple(args.home_rad),
                            args.home_gripper_fraction,
                            duration_s=args.home_duration_s,
                            frequency_hz=args.frequency_hz,
                        )
                    except Exception as exc:
                        print(
                            f"[Dual X5][WARN] HOME failed: {type(exc).__name__}: {exc}; "
                            "recovering an active hold instead of exiting to damping.",
                            flush=True,
                        )
                        _recover_active_hold(hardware, attempts=None)
                        clean_end = False
                    else:
                        print("[Dual X5] HOME reached; ready for the next episode.", flush=True)
                if not quit_requested:
                    print(
                        "[Dual X5] episode ended; arms remain held and the source is ready for the next Isaac connection."
                        if clean_end
                        else "[Dual X5] Isaac disconnected; arms remain held and the source is listening.",
                        flush=True,
                    )
    finally:
        server.close()
        if owns_hardware:
            hardware.close()
        print("[Dual X5] source stopped.", flush=True)


def main() -> None:
    run_server(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
