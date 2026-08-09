"""Map all six right PiPER-X joint deltas directly to the right ARX arm."""

from __future__ import annotations

import json
import math
import os
import socket
import time
from typing import Any

import numpy as np

from src.eval_client.intervention_loop import RealtimePacer

_JOINT_SIGNS = np.asarray([1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float64)


class PiperXJointJ1Error(RuntimeError):
    pass


class PiperXJointJ1Exit(Exception):
    pass


def _format_degrees(values: Any) -> str:
    return "[" + ",".join(f"{math.degrees(float(value)):7.2f}" for value in values) + "]"


class _JointSourceClient:
    def __init__(self) -> None:
        self.host = os.environ.get("ROBODOJO_PIPERX_JOINT_HOST", "127.0.0.1")
        self.port = int(os.environ.get("ROBODOJO_PIPERX_JOINT_PORT", "8766"))
        self.timeout_s = float(os.environ.get("ROBODOJO_PIPERX_JOINT_TIMEOUT_S", "10"))
        self.socket: socket.socket | None = None
        self.file = None

    def connect(self) -> None:
        try:
            self.socket = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            self.socket.settimeout(self.timeout_s)
            self.file = self.socket.makefile("rwb", buffering=0)
        except OSError as exc:
            raise PiperXJointJ1Error(
                f"cannot connect to right-J1 source at {self.host}:{self.port}: {exc}"
            ) from exc

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def sample(self) -> dict[str, Any]:
        if self.file is None:
            raise PiperXJointJ1Error("right-J1 source is not connected")
        try:
            self.file.write(b"sample\n")
            raw = self.file.readline()
        except OSError as exc:
            raise PiperXJointJ1Error(f"right-J1 source I/O failed: {exc}") from exc
        if not raw:
            raise PiperXJointJ1Error("right-J1 source disconnected")
        try:
            payload = json.loads(raw)
            active = payload["active"]
            edge = payload["edge"]
            exit_requested = payload["exit"]
            q_rad = tuple(float(value) for value in payload["q_rad"])
            delta_q_rad = tuple(float(value) for value in payload["delta_q_rad"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PiperXJointJ1Error(f"invalid right-arm sample: {raw!r}") from exc
        if (
            not isinstance(active, bool)
            or edge not in {None, "enter", "exit"}
            or not isinstance(exit_requested, bool)
            or len(q_rad) != 6
            or len(delta_q_rad) != 6
            or not all(math.isfinite(value) for value in q_rad)
            or not all(math.isfinite(value) for value in delta_q_rad)
        ):
            raise PiperXJointJ1Error(f"invalid right-arm sample values: {payload!r}")
        return {
            "active": active,
            "edge": edge,
            "exit": exit_requested,
            "q_rad": q_rad,
            "delta_q_rad": delta_q_rad,
        }


def _hold_action(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    state = obs["state"]
    return {
        "left_arm_joint_state": np.asarray(
            state["left_arm_joint_state"], dtype=np.float64
        ).copy(),
        "left_ee_joint_state": np.asarray(
            state["left_ee_joint_state"], dtype=np.float64
        ).copy(),
        "right_arm_joint_state": np.asarray(
            state["right_arm_joint_state"], dtype=np.float64
        ).copy(),
        "right_ee_joint_state": np.asarray(
            state["right_ee_joint_state"], dtype=np.float64
        ).copy(),
    }


def run_piperx_joint_j1_episode(task_env: Any) -> None:
    """Run the single-joint, simulation-only checkpoint until Esc."""

    if task_env.num_envs != 1:
        raise PiperXJointJ1Error("right-J1 checkpoint requires exactly one Isaac environment")

    client = _JointSourceClient()
    client.connect()
    print(
        "[Right-arm checkpoint] connected. Terminal A: i anchors/toggles; Esc exits. "
        "All six right ARX arm joints are controlled; gripper is held.",
        flush=True,
    )
    pacer = RealtimePacer(
        frequency=float(task_env.obs_manager.collect_freq),
        enabled=os.environ.get("ROBODOJO_REALTIME", "1") != "0",
    )
    sim_anchor_rad: np.ndarray | None = None
    last_report_s = 0.0

    try:
        while True:
            sample = client.sample()
            if sample["exit"]:
                raise PiperXJointJ1Exit

            obs = task_env.get_obs()
            action = _hold_action(obs)
            current_sim_q = action["right_arm_joint_state"][:6].copy()
            if sample["edge"] == "enter":
                sim_anchor_rad = current_sim_q
                print(
                    f"\n[Right-arm checkpoint] ON sim_q_deg={_format_degrees(sim_anchor_rad)}",
                    flush=True,
                )
            elif sample["edge"] == "exit":
                sim_anchor_rad = None
                print("\n[Right-arm checkpoint] OFF", flush=True)

            target_q_rad = current_sim_q
            if sample["active"]:
                if sim_anchor_rad is None:
                    raise PiperXJointJ1Error("active sample arrived without an Isaac anchor")
                target_q_rad = (
                    sim_anchor_rad
                    + _JOINT_SIGNS * np.asarray(sample["delta_q_rad"], dtype=np.float64)
                )
                action["right_arm_joint_state"][:6] = target_q_rad

            task_env.take_action(action)
            now_s = time.monotonic()
            if now_s - last_report_s >= 0.25:
                print(
                    "\r[Right-arm checkpoint] "
                    f"active={sample['active']} "
                    f"PiPER_deg={_format_degrees(sample['q_rad'])} "
                    f"delta_deg={_format_degrees(sample['delta_q_rad'])} "
                    f"ARX_deg={_format_degrees(target_q_rad)}",
                    end="",
                    flush=True,
                )
                last_report_s = now_s
            pacer.wait()
    finally:
        client.close()
