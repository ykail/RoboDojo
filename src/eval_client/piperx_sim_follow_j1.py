"""Drive a slow ARX trajectory and mirror its measured delta to one PiPER-X."""

from __future__ import annotations

import json
import math
import os
import socket
import time
from typing import Any

import numpy as np

from src.eval_client.intervention_loop import RealtimePacer
from src.eval_client.piperx_joint_j1 import (
    PiperXJointJ1Error,
    _format_degrees,
    _hold_action,
)


class _SimFollowClient:
    def __init__(self) -> None:
        self.host = os.environ.get("ROBODOJO_CP5_HOST", "127.0.0.1")
        self.port = int(os.environ.get("ROBODOJO_CP5_PORT", "8767"))
        self.timeout_s = float(os.environ.get("ROBODOJO_CP5_TIMEOUT_S", "30"))
        self.socket: socket.socket | None = None
        self.file = None
        self.seq = 0

    def connect(self) -> None:
        try:
            self.socket = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            self.socket.settimeout(self.timeout_s)
            self.file = self.socket.makefile("rwb", buffering=0)
        except OSError as exc:
            raise PiperXJointJ1Error(
                f"cannot connect to CP5 PiPER source at {self.host}:{self.port}: {exc}"
            ) from exc

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def exchange(
        self,
        sim_delta_q_rad: np.ndarray,
        sim_delta_gripper: float | None = None,
    ) -> dict[str, Any]:
        if self.file is None:
            raise PiperXJointJ1Error("CP5 PiPER source is not connected")
        self.seq += 1
        request = {
            "type": "sim_delta",
            "seq": self.seq,
            "sim_delta_q_rad": [float(value) for value in sim_delta_q_rad],
        }
        if sim_delta_gripper is not None:
            request["sim_delta_gripper_open_fraction"] = float(sim_delta_gripper)
        try:
            self.file.write(
                (json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n").encode()
            )
            raw = self.file.readline()
        except (OSError, ValueError) as exc:
            raise PiperXJointJ1Error(f"CP5 PiPER source I/O failed: {exc}") from exc
        if not raw:
            raise PiperXJointJ1Error("CP5 PiPER source disconnected")
        try:
            payload = json.loads(raw)
            measured_q = tuple(float(value) for value in payload["measured_q_rad"])
            command_q = tuple(float(value) for value in payload["command_q_rad"])
            measured_gripper_m = float(payload["measured_gripper_width_m"])
            command_gripper_m = float(payload["command_gripper_width_m"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PiperXJointJ1Error(f"invalid CP5 PiPER response: {raw!r}") from exc
        if (
            payload.get("ok") is not True
            or payload.get("seq") != self.seq
            or len(measured_q) != 6
            or len(command_q) != 6
            or not all(math.isfinite(value) for value in measured_q + command_q)
            or not math.isfinite(measured_gripper_m)
            or not math.isfinite(command_gripper_m)
        ):
            raise PiperXJointJ1Error(f"invalid CP5 PiPER response values: {payload!r}")
        return {
            "measured_q_rad": measured_q,
            "command_q_rad": command_q,
            "measured_gripper_width_m": measured_gripper_m,
            "command_gripper_width_m": command_gripper_m,
        }


def run_piperx_sim_follow_j1_episode(task_env: Any) -> None:
    """Run CP5 until the operator presses Ctrl-C in either terminal."""

    if task_env.num_envs != 1:
        raise PiperXJointJ1Error("CP5 requires exactly one Isaac environment")

    side = os.environ.get("ROBODOJO_SIM_FOLLOW_SIDE", "right").strip().lower()
    if side not in {"left", "right"}:
        raise PiperXJointJ1Error("ROBODOJO_SIM_FOLLOW_SIDE must be left or right")
    arm_key = f"{side}_arm_joint_state"
    gripper_key = f"{side}_ee_joint_state"
    trajectory = os.environ.get("ROBODOJO_SIM_FOLLOW_TRAJECTORY", "j1").strip().lower()
    sync_gripper = os.environ.get("ROBODOJO_SIM_FOLLOW_GRIPPER", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    gripper_amplitude = float(os.environ.get("ROBODOJO_SIM_FOLLOW_GRIPPER_AMPLITUDE", "0.1"))
    amplitude_deg = float(
        os.environ.get(
            "ROBODOJO_SIM_FOLLOW_AMPLITUDE_DEG",
            os.environ.get("ROBODOJO_CP5_AMPLITUDE_DEG", "3"),
        )
    )
    period_s = float(
        os.environ.get(
            "ROBODOJO_SIM_FOLLOW_PERIOD_S",
            os.environ.get("ROBODOJO_CP5_PERIOD_S", "8"),
        )
    )
    if trajectory not in {"hold", "j1", "all_joints"}:
        raise PiperXJointJ1Error(f"unknown simulator trajectory {trajectory!r}")
    if amplitude_deg <= 0.0 or period_s <= 0.0:
        raise PiperXJointJ1Error("CP5 amplitude and period must be positive")
    if sync_gripper and not 0.0 < gripper_amplitude <= 0.25:
        raise PiperXJointJ1Error("gripper amplitude must be in (0, 0.25]")
    checkpoint = (
        "CP9"
        if side == "left"
        else "CP8"
        if trajectory == "all_joints" and sync_gripper
        else "CP7"
        if sync_gripper
        else "CP6"
        if trajectory == "all_joints"
        else "CP5"
    )

    client = _SimFollowClient()
    client.connect()
    pacer = RealtimePacer(
        frequency=float(task_env.obs_manager.collect_freq),
        enabled=os.environ.get("ROBODOJO_REALTIME", "1") != "0",
    )

    try:
        obs = task_env.get_obs()
        if sync_gripper:
            center_action = _hold_action(obs)
            center_action[gripper_key][0] = 0.5
            task_env.take_action(center_action)
            obs = task_env.get_obs()
        sim_anchor_q = _hold_action(obs)[arm_key][:6].copy()
        sim_gripper_anchor = float(_hold_action(obs)[gripper_key][0])
        response = client.exchange(
            np.zeros(6, dtype=np.float64),
            0.0 if sync_gripper else None,
        )
        print(
            f"[{checkpoint} Isaac] connected to {side}; no policy and no i key. "
            f"ARX anchor_deg={_format_degrees(sim_anchor_q)}",
            flush=True,
        )
        if sync_gripper and trajectory == "all_joints":
            print(
                f"[{checkpoint} Isaac] all six {side} joints + gripper; "
                f"joint amplitude={amplitude_deg:.1f} deg, "
                f"gripper amplitude={gripper_amplitude:.3f}, period={period_s:.1f}s. Ctrl-C exits.",
                flush=True,
            )
        elif sync_gripper:
            print(
                f"[{checkpoint} Isaac] gripper center={sim_gripper_anchor:.3f}, "
                f"amplitude={gripper_amplitude:.3f}, period={period_s:.1f}s. Ctrl-C exits.",
                flush=True,
            )
        elif trajectory == "all_joints":
            print(
                f"[{checkpoint} Isaac] all six {side}-arm joints move together; "
                f"amplitude={amplitude_deg:.1f} deg, period={period_s:.1f}s. Ctrl-C exits.",
                flush=True,
            )
        else:
            print(
                f"[{checkpoint} Isaac] {side} J1 trajectory: +/-{amplitude_deg:.1f} deg, "
                f"period={period_s:.1f}s. Ctrl-C exits.",
                flush=True,
            )

        start_s = time.monotonic()
        last_report_s = 0.0
        while True:
            obs = task_env.get_obs()
            action = _hold_action(obs)
            elapsed_s = time.monotonic() - start_s
            phase = 2.0 * math.pi * elapsed_s / period_s
            symmetric_delta_rad = math.radians(amplitude_deg) * math.sin(phase)
            if trajectory == "all_joints":
                one_sided_delta_rad = 0.5 * math.radians(amplitude_deg) * (
                    1.0 - math.cos(phase)
                )
                requested_sim_delta_q = np.asarray(
                    [
                        symmetric_delta_rad,
                        one_sided_delta_rad,
                        one_sided_delta_rad,
                        symmetric_delta_rad,
                        symmetric_delta_rad,
                        symmetric_delta_rad,
                    ],
                    dtype=np.float64,
                )
                action[arm_key][:6] = sim_anchor_q + requested_sim_delta_q
            elif trajectory == "j1":
                action[arm_key][0] = sim_anchor_q[0] + symmetric_delta_rad
            if sync_gripper:
                action[gripper_key][0] = float(
                    np.clip(
                        sim_gripper_anchor + gripper_amplitude * math.sin(phase),
                        0.0,
                        1.0,
                    )
                )
            task_env.take_action(action)

            post_obs = task_env.get_obs()
            actual_sim_q = _hold_action(post_obs)[arm_key][:6]
            if trajectory == "all_joints":
                actual_sim_delta_q = actual_sim_q - sim_anchor_q
            elif trajectory == "j1":
                actual_sim_delta_q = np.zeros(6, dtype=np.float64)
                actual_sim_delta_q[0] = actual_sim_q[0] - sim_anchor_q[0]
            else:
                actual_sim_delta_q = np.zeros(6, dtype=np.float64)
            actual_gripper = float(_hold_action(post_obs)[gripper_key][0])
            actual_gripper_delta = actual_gripper - sim_gripper_anchor
            response = client.exchange(
                actual_sim_delta_q,
                actual_gripper_delta if sync_gripper else None,
            )

            now_s = time.monotonic()
            if now_s - last_report_s >= 0.25:
                print(
                    f"\r[{checkpoint} Isaac] "
                    f"ARX_delta_deg={_format_degrees(actual_sim_delta_q)} "
                    f"PiPER_deg={_format_degrees(response['measured_q_rad'])} "
                    f"gripper={response['measured_gripper_width_m'] * 1000.0:.1f}mm",
                    end="",
                    flush=True,
                )
                last_report_s = now_s
            pacer.wait()
    finally:
        client.close()
