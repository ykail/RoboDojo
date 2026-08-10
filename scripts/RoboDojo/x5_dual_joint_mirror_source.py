#!/usr/bin/env python3
"""Dual ARX X5 source for the RoboDojo joint-mirror client.

The wire format is ``robodojo_dual_joint_mirror_v1`` and remains compatible
with the existing :class:`DualJointMirrorClient`.  The physical and simulated
robots are both ARX X5, so follow targets use identity relative joint deltas.
"""

from __future__ import annotations

import argparse
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
from pathlib import Path
from typing import Any


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

class X5SourceError(RuntimeError):
    pass


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

    def __init__(self, hardware: DualX5Hardware) -> None:
        self.hardware = hardware
        self.mode = "follow"
        self.require_zero = True
        self.pending_transition: str | None = None
        self.manual_anchor: DualState | None = None
        self.follow_anchor = hardware.latched_target
        self.ended = False

    def toggle_intervention(self) -> bool:
        if self.ended or self.pending_transition is not None:
            return False
        self.pending_transition = "enter" if self.mode == "follow" else "exit"
        return True

    def _response(self, request_type: str, seq: int, edge: str | None) -> dict[str, Any]:
        measured = self.hardware.read()
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
        return {
            "ok": True,
            "type": request_type,
            "seq": seq,
            "mode": self.mode,
            "edge": edge,
            "terminal": None,
            "sides": sides,
        }

    def handle(self, payload: Any) -> dict[str, Any]:
        if self.ended:
            raise X5SourceError("joint-mirror session has ended")
        request_type, seq, parsed = parse_request(payload, require_zero=self.require_zero)
        edge: str | None = None

        if request_type == "end":
            self.hardware.enter_hold()
            self.mode = "follow"
            self.pending_transition = None
            self.manual_anchor = None
            response = self._response(request_type, seq, None)
            self.ended = True
            return response

        if request_type == "joint_mirror":
            assert parsed is not None
            if self.pending_transition == "enter":
                self.manual_anchor = self.hardware.enter_teach()
                self.mode = "manual"
                self.pending_transition = None
                edge = "enter"
            elif self.pending_transition == "exit":
                self.hardware.enter_hold()
                self.follow_anchor = self.hardware.latched_target
                self.manual_anchor = None
                self.mode = "follow"
                self.require_zero = True
                self.pending_transition = None
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

        return self._response(request_type, seq, edge)

    def disconnect(self) -> None:
        if self.hardware.is_connected:
            self.hardware.enter_hold()
        self.mode = "follow"
        self.pending_transition = None
        self.manual_anchor = None


class _KeyReader:
    """Read global ``i`` through XInput, with an ``i``/``q`` terminal fallback."""

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
        self._pending: queue.Queue[str] = queue.Queue(maxsize=8)
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
        result: dict[int, str] = {}
        for line in output.splitlines():
            match = re.match(r"^keycode\s+(\d+)\s*=\s*(\S+)", line)
            if match and match.group(2) == "i":
                result[int(match.group(1))] = match.group(2)
        if set(result.values()) != {"i"}:
            raise RuntimeError(f"incomplete X11 i keymap: {result}")
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
                self._pending.put_nowait(self._key_by_code[keycode])
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
                f"[Dual X5] global hotkey active on {self.display_name}: i=toggle. "
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

    def poll(self) -> str | None:
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
            key = sys.stdin.read(1).lower()
            return key if key in {"i", "q"} else None
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve two real ARX X5 arms to RoboDojo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--left-can", default="can1")
    parser.add_argument("--right-can", default="can3")
    parser.add_argument("--left-model", default="X5")
    parser.add_argument("--right-model", default="X5")
    parser.add_argument("--frequency-hz", type=float, default=100.0)
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
) -> tuple[bool, bool]:
    """Return ``(quit_requested, clean_end)`` after one client connection."""

    session = JointMirrorSession(hardware)
    conn.settimeout(min(0.1, max(0.005, period_s)))
    buffer = b""
    last_liveness = time.monotonic()
    while True:
        key = keys.poll()
        if key == "q":
            session.disconnect()
            return True, False
        if key == "i" and session.toggle_intervention():
            print(
                f"\n[Dual X5] intervention {session.pending_transition} requested; "
                "waiting for the next Isaac boundary.",
                flush=True,
            )
        try:
            chunk = conn.recv(65536)
        except socket.timeout:
            if time.monotonic() - last_liveness > liveness_timeout_s:
                session.disconnect()
                print(
                    "[Dual X5][WARN] Isaac liveness timed out; holding both arms.",
                    flush=True,
                )
                return False, False
            continue
        except OSError:
            chunk = b""
        if not chunk:
            session.disconnect()
            return False, False
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
            response = session.handle(payload)
            _send(conn, response)
            edge = response["edge"]
            if edge == "enter":
                print("\n[Dual X5] manual control ON.", flush=True)
            elif edge == "exit":
                print("\n[Dual X5] manual control OFF; holding the release pose.", flush=True)
            if session.ended:
                return False, True


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
            f"[Dual X5] listening on {args.host}:{args.port}; i toggles intervention. "
            "Stop only with Ctrl-C in this terminal while supporting both arms.",
            flush=True,
        )
        with _KeyReader(
            global_hotkeys=not args.no_global_hotkeys,
            display_name=args.display,
        ) as keys:
            while not quit_requested:
                key = keys.poll()
                if key == "q":
                    break
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    hardware.hold()
                    continue
                print("[Dual X5] Isaac connected; follow anchor is the current hold pose.", flush=True)
                with conn:
                    try:
                        quit_requested, clean_end = _serve_connection(
                            conn,
                            hardware,
                            keys,
                            period_s=1.0 / args.frequency_hz,
                            liveness_timeout_s=args.liveness_timeout_s,
                        )
                    except (OSError, X5SourceError, X5HardwareError) as exc:
                        hardware.enter_hold()
                        print(f"[Dual X5][WARN] connection ended: {exc}", flush=True)
                        clean_end = False
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
