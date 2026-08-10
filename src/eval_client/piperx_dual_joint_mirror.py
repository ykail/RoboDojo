"""Small direct-joint checkpoints for two PiPER-X leader arms."""

from __future__ import annotations

import json
import math
import os
import socket
import threading
import time
from typing import Any

import numpy as np

from src.eval_client.intervention_loop import RealtimePacer
from src.eval_client.piperx_joint_j1 import _format_degrees, _hold_action


SIDES = ("left", "right")
_JOINT_PROFILES = {
    "arx_x5_piperx_relative_joint_v1": np.asarray(
        [1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float64
    ),
    "arx_x5_identity_joint_v1": np.ones(6, dtype=np.float64),
}


class DualJointMirrorError(RuntimeError):
    pass


def _joint_signs_for_profile(profile: str) -> np.ndarray:
    try:
        return _JOINT_PROFILES[profile].copy()
    except KeyError as exc:
        raise DualJointMirrorError(
            "dual mirror profile must be one of "
            f"{sorted(_JOINT_PROFILES)}, got {profile!r}"
        ) from exc


def _joint_signs() -> np.ndarray:
    """Return the explicit hardware-to-simulator joint-axis mapping.

    PiPER-X needs two sign flips because it is only kinematically similar to
    the simulated ARX X5.  A physical X5 uses the simulator's native joint
    convention and therefore has an identity mapping.  Keeping this choice in
    one declared profile prevents X5 collection from silently inheriting the
    old PiPER-X retargeting signs.
    """

    profile = os.environ.get(
        "ROBODOJO_DUAL_MIRROR_PROFILE",
        "arx_x5_piperx_relative_joint_v1",
    ).strip()
    return _joint_signs_for_profile(profile)


class DualJointMirrorClient:
    def __init__(self) -> None:
        self.host = os.environ.get("ROBODOJO_DUAL_MIRROR_HOST", "127.0.0.1")
        self.port = int(os.environ.get("ROBODOJO_DUAL_MIRROR_PORT", "8770"))
        self.timeout_s = float(os.environ.get("ROBODOJO_DUAL_MIRROR_TIMEOUT_S", "5"))
        self.socket: socket.socket | None = None
        self.file = None
        self.seq = 0
        self._lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_error: BaseException | None = None
        self._ended = False

    def connect(self) -> None:
        if self.socket is not None or self.file is not None:
            raise DualJointMirrorError("dual PiPER source is already connected")
        self._heartbeat_stop.clear()
        self._heartbeat_error = None
        self._ended = False
        self.seq = 0
        try:
            self.socket = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            self.socket.settimeout(self.timeout_s)
            self.file = self.socket.makefile("rwb", buffering=0)
        except OSError as exc:
            raise DualJointMirrorError(
                f"cannot connect to dual PiPER source at {self.host}:{self.port}: {exc}"
            ) from exc
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="piperx-dual-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def close(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread is not threading.current_thread():
            self._heartbeat_thread.join(timeout=2.0)
        self._heartbeat_thread = None
        with self._lock:
            if self.file is not None:
                self.file.close()
                self.file = None
            if self.socket is not None:
                self.socket.close()
                self.socket = None

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(0.5):
            try:
                self._request("heartbeat")
            except BaseException as exc:
                self._heartbeat_error = exc
                return

    def _request(
        self,
        request_type: str,
        sides_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self.file is None:
                raise DualJointMirrorError("dual PiPER source is not connected")
            if self._heartbeat_error is not None and request_type != "heartbeat":
                raise DualJointMirrorError(
                    f"dual PiPER heartbeat failed: {self._heartbeat_error}"
                ) from self._heartbeat_error
            self.seq += 1
            request: dict[str, Any] = {"type": request_type, "seq": self.seq}
            if sides_payload is not None:
                request["sides"] = sides_payload
            try:
                self.file.write(
                    (json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n").encode()
                )
                raw = self.file.readline()
            except (OSError, ValueError) as exc:
                raise DualJointMirrorError(f"dual PiPER source I/O failed: {exc}") from exc
            if not raw:
                raise DualJointMirrorError("dual PiPER source disconnected")
            try:
                payload = json.loads(raw)
                mode = payload["mode"]
                edge = payload["edge"]
                terminal = payload.get("terminal")
                sides = payload["sides"]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise DualJointMirrorError(f"invalid dual PiPER response: {raw!r}") from exc
            if (
                payload.get("ok") is not True
                or payload.get("type") != request_type
                or payload.get("seq") != self.seq
                or mode not in {"follow", "manual"}
                or edge not in {None, "enter", "exit"}
                or terminal not in {None, "save", "retry"}
                or (request_type != "joint_mirror" and edge is not None)
                or (
                    terminal is not None
                    and (request_type != "joint_mirror" or mode != "follow" or edge is not None)
                )
                or not isinstance(sides, dict)
                or set(sides) != set(SIDES)
            ):
                raise DualJointMirrorError(f"invalid dual PiPER response envelope: {payload!r}")
            parsed: dict[str, Any] = {}
            for side in SIDES:
                try:
                    measured_q = np.asarray(sides[side]["measured_q_rad"], dtype=np.float64)
                    measured_gripper = float(sides[side]["measured_gripper_width_m"])
                    leader_gripper = float(
                        sides[side]["leader_gripper_open_fraction"]
                    )
                    leader_delta_q = np.asarray(
                        sides[side].get("leader_delta_q_rad", [0.0] * 6), dtype=np.float64
                    )
                    follow_target_q = np.asarray(
                        sides[side]["follow_target_q_rad"], dtype=np.float64
                    )
                    leader_delta_gripper = float(
                        sides[side].get("leader_delta_gripper_open_fraction", 0.0)
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise DualJointMirrorError(f"invalid {side} dual PiPER response") from exc
                if (
                    measured_q.shape != (6,)
                    or leader_delta_q.shape != (6,)
                    or follow_target_q.shape != (6,)
                    or not np.isfinite(measured_q).all()
                    or not np.isfinite(leader_delta_q).all()
                    or not np.isfinite(follow_target_q).all()
                    or not math.isfinite(measured_gripper)
                    or not math.isfinite(leader_gripper)
                    or not math.isfinite(leader_delta_gripper)
                ):
                    raise DualJointMirrorError(f"non-finite {side} dual PiPER response")
                parsed[side] = {
                    "measured_q_rad": measured_q,
                    "measured_gripper_width_m": measured_gripper,
                    "leader_gripper_open_fraction": float(
                        np.clip(leader_gripper, 0.0, 1.0)
                    ),
                    "leader_delta_q_rad": leader_delta_q,
                    "follow_target_q_rad": follow_target_q,
                    "leader_delta_gripper_open_fraction": leader_delta_gripper,
                }
            return {"mode": mode, "edge": edge, "terminal": terminal, "sides": parsed}

    def exchange(
        self,
        deltas: dict[str, tuple[np.ndarray, float, float]],
    ) -> dict[str, Any]:
        if set(deltas) != set(SIDES):
            raise DualJointMirrorError("dual exchange requires left and right")
        return self._request(
            "joint_mirror",
            {
                side: {
                    "delta_q_rad": [float(value) for value in deltas[side][0]],
                    "delta_gripper_open_fraction": float(deltas[side][1]),
                    "sim_gripper_open_fraction": float(deltas[side][2]),
                }
                for side in SIDES
            },
        )

    def end(self) -> None:
        if self._ended:
            return
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
        response = self._request("end")
        if response["mode"] != "follow" or response["edge"] is not None:
            raise DualJointMirrorError("dual PiPER source did not acknowledge final hold")
        self._ended = True


def _target_robots(task_env: Any) -> dict[str, Any]:
    robots = {
        robot.arm_name.split("_")[0]: robot
        for robot in task_env.robot_manager.robot_list
        if robot.type == "target"
    }
    if set(robots) != set(SIDES):
        raise DualJointMirrorError(f"expected left/right target robots, got {sorted(robots)}")
    return robots


def _measured_sim_state(task_env: Any, robots: dict[str, Any]) -> dict[str, tuple[np.ndarray, float]]:
    result: dict[str, tuple[np.ndarray, float]] = {}
    for side in SIDES:
        robot = robots[side]
        joints = np.asarray(
            task_env.robot_manager.get_joint(robot, env_idx_list=[0])[0],
            dtype=np.float64,
        )[:6]
        raw_gripper = float(
            task_env.robot_manager.get_end_effector_real_val(robot, env_idx_list=[0])[0][0]
        )
        low = float(robot.gripper_scale[0])
        high = float(robot.gripper_scale[1])
        if joints.shape != (6,) or not np.isfinite(joints).all() or not high > low:
            raise DualJointMirrorError(f"invalid measured {side} simulator state")
        normalized = (raw_gripper - low) / (high - low)
        if robot.gripper_move["sign"] != 1:
            normalized = 1.0 - normalized
        result[side] = (joints, float(np.clip(normalized, 0.0, 1.0)))
    return result


def replay_sim_state(
    replay: Any,
    robots: dict[str, Any],
) -> dict[str, tuple[np.ndarray, float]]:
    """Read the dual-arm measured state directly from a replay snapshot."""

    result: dict[str, tuple[np.ndarray, float]] = {}
    for descriptor in replay.manifest.get("robots", []):
        side = str(descriptor.get("name", "")).split("_")[0]
        if side not in robots:
            continue
        slot = int(descriptor["slot"])
        joint_pos = np.asarray(
            replay.state[f"robot.{slot:03d}.joint_pos"],
            dtype=np.float64,
        ).reshape(-1)
        arm_indices = [int(index) for index in descriptor.get("arm_joint_indices", [])]
        gripper_indices = [int(index) for index in descriptor.get("gripper_joint_indices", [])]
        if len(arm_indices) != 6 or not gripper_indices:
            raise DualJointMirrorError(f"invalid replay robot descriptor for {side}")
        joints = joint_pos[arm_indices]
        raw_gripper = float(np.mean(joint_pos[gripper_indices]))
        robot = robots[side]
        low = float(robot.gripper_scale[0])
        high = float(robot.gripper_scale[1])
        if joints.shape != (6,) or not np.isfinite(joints).all() or not high > low:
            raise DualJointMirrorError(f"invalid replay {side} simulator state")
        normalized = (raw_gripper - low) / (high - low)
        if robot.gripper_move["sign"] != 1:
            normalized = 1.0 - normalized
        result[side] = (joints, float(np.clip(normalized, 0.0, 1.0)))
    if set(result) != set(SIDES):
        raise DualJointMirrorError(f"replay dual-arm state is incomplete: {sorted(result)}")
    return result


def _zero_deltas(
    state: dict[str, tuple[np.ndarray, float]],
) -> dict[str, tuple[np.ndarray, float, float]]:
    return {
        side: (np.zeros(6, dtype=np.float64), 0.0, state[side][1])
        for side in SIDES
    }


def _state_deltas(
    state: dict[str, tuple[np.ndarray, float]],
    anchor: dict[str, tuple[np.ndarray, float]],
) -> dict[str, tuple[np.ndarray, float, float]]:
    return {
        side: (
            state[side][0] - anchor[side][0],
            state[side][1] - anchor[side][1],
            state[side][1],
        )
        for side in SIDES
    }


def _action_target_state(
    action: dict[str, Any],
) -> dict[str, tuple[np.ndarray, float]]:
    """Extract the exact dual-X5 target that Isaac is about to execute."""

    result: dict[str, tuple[np.ndarray, float]] = {}
    for side in SIDES:
        try:
            joints = np.asarray(
                action[f"{side}_arm_joint_state"],
                dtype=np.float64,
            ).reshape(-1)[:6]
            gripper_values = np.asarray(
                action[f"{side}_ee_joint_state"],
                dtype=np.float64,
            ).reshape(-1)
            gripper = float(gripper_values[0])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DualJointMirrorError(
                f"invalid {side} policy target for X5 mirroring"
            ) from exc
        if joints.shape != (6,) or not np.isfinite(joints).all() or not math.isfinite(gripper):
            raise DualJointMirrorError(f"non-finite {side} policy target for X5 mirroring")
        result[side] = (joints.copy(), float(np.clip(gripper, 0.0, 1.0)))
    return result


def _manual_action(
    obs: dict[str, Any],
    sim_anchor: dict[str, tuple[np.ndarray, float]],
    response: dict[str, Any],
) -> dict[str, np.ndarray]:
    action = _hold_action(obs)
    joint_signs = _joint_signs()
    for side in SIDES:
        item = response["sides"][side]
        action[f"{side}_arm_joint_state"][:6] = (
            sim_anchor[side][0] + joint_signs * item["leader_delta_q_rad"]
        )
        # Gripper mapping is absolute: fully closed PiPER-X is simulator 0,
        # fully open PiPER-X is simulator 1.  Joint motion remains relative so
        # takeover itself cannot move either arm.
        action[f"{side}_ee_joint_state"][0] = item["leader_gripper_open_fraction"]
    return action


class _ControlRateReporter:
    def __init__(self, expected_hz: float) -> None:
        self.expected_hz = float(expected_hz)
        self.window_start = time.monotonic()
        self.last_frame = self.window_start
        self.frames = 0
        self.max_gap_s = 0.0

    def frame(self, mode: str) -> None:
        now = time.monotonic()
        self.max_gap_s = max(self.max_gap_s, now - self.last_frame)
        self.last_frame = now
        self.frames += 1
        elapsed = now - self.window_start
        if elapsed < 2.0:
            return
        print(
            f"\n[X5 timing] mode={mode} control={self.frames / elapsed:.1f}Hz "
            f"target={self.expected_hz:.1f}Hz max_gap={self.max_gap_s * 1000.0:.0f}ms",
            flush=True,
        )
        self.window_start = now
        self.frames = 0
        self.max_gap_s = 0.0


def run_piperx_dual_joint_test_episode(task_env: Any) -> None:
    """CP10: move both simulated arms and mirror their measured joint deltas."""

    if task_env.num_envs != 1:
        raise DualJointMirrorError("CP10 requires exactly one Isaac environment")
    amplitude_deg = float(os.environ.get("ROBODOJO_DUAL_TEST_AMPLITUDE_DEG", "1"))
    gripper_amplitude = float(os.environ.get("ROBODOJO_DUAL_TEST_GRIPPER_AMPLITUDE", "0.05"))
    period_s = float(os.environ.get("ROBODOJO_DUAL_TEST_PERIOD_S", "12"))
    if amplitude_deg <= 0.0 or not 0.0 < gripper_amplitude <= 0.25 or period_s <= 0.0:
        raise DualJointMirrorError("invalid CP10 trajectory settings")
    client = DualJointMirrorClient()
    client.connect()
    robots = _target_robots(task_env)
    pacer = RealtimePacer(
        frequency=float(task_env.obs_manager.collect_freq),
        enabled=os.environ.get("ROBODOJO_REALTIME", "1") != "0",
    )
    try:
        obs = task_env.get_obs()
        centered = _hold_action(obs)
        for side in SIDES:
            centered[f"{side}_ee_joint_state"][0] = 0.5
        task_env.take_action(centered)
        obs = task_env.get_obs()
        anchor = _measured_sim_state(task_env, robots)
        response = client.exchange(_zero_deltas(anchor))
        if response["mode"] != "follow" or response["edge"] is not None:
            raise DualJointMirrorError("CP10 source did not begin in stable follow mode")
        print("[CP10 Isaac] dual six-joint + gripper test running. Ctrl-C exits.", flush=True)
        start = time.monotonic()
        last_report = 0.0
        while True:
            obs = task_env.get_obs()
            action = _hold_action(obs)
            phase = 2.0 * math.pi * (time.monotonic() - start) / period_s
            symmetric = math.radians(amplitude_deg) * math.sin(phase)
            one_sided = 0.5 * math.radians(amplitude_deg) * (1.0 - math.cos(phase))
            delta = np.asarray(
                [symmetric, one_sided, one_sided, symmetric, symmetric, symmetric],
                dtype=np.float64,
            )
            for side in SIDES:
                action[f"{side}_arm_joint_state"][:6] = anchor[side][0] + delta
                action[f"{side}_ee_joint_state"][0] = float(
                    np.clip(anchor[side][1] + gripper_amplitude * math.sin(phase), 0.0, 1.0)
                )
            task_env.take_action(action)
            post_obs = task_env.get_obs()
            measured = _measured_sim_state(task_env, robots)
            response = client.exchange(_state_deltas(measured, anchor))
            if response["mode"] != "follow" or response["edge"] is not None:
                raise DualJointMirrorError("unexpected intervention state during CP10")
            now = time.monotonic()
            if now - last_report >= 0.25:
                print(
                    "\r[CP10 Isaac] "
                    f"ARX_delta={_format_degrees(measured['right'][0] - anchor['right'][0])} "
                    f"PiPER_L={_format_degrees(response['sides']['left']['measured_q_rad'])} "
                    f"PiPER_R={_format_degrees(response['sides']['right']['measured_q_rad'])}",
                    end="",
                    flush=True,
                )
                last_report = now
            obs = post_obs
            pacer.wait()
    finally:
        client.close()


def run_piperx_policy_leader_mirror_episode(
    task_env: Any,
    model_client: Any,
    *,
    allow_intervention: bool = False,
) -> None:
    """CP11-CP13: policy rollout, optional intervention, and optional recording."""

    if task_env.num_envs != 1 or getattr(task_env, "eval_batch", False):
        raise DualJointMirrorError("CP11-CP13 require one non-batched Isaac environment")
    record_value = os.environ.get("ROBODOJO_DUAL_MIRROR_RECORD", "0").strip().lower()
    if record_value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise DualJointMirrorError("ROBODOJO_DUAL_MIRROR_RECORD must be a boolean")
    record_enabled = record_value in {"1", "true", "yes", "on"}
    if record_enabled and not allow_intervention:
        raise DualJointMirrorError("dual-leader recording requires intervention mode")
    profile = os.environ.get(
        "ROBODOJO_DUAL_MIRROR_PROFILE",
        "arx_x5_piperx_relative_joint_v1",
    ).strip()
    _joint_signs_for_profile(profile)  # Validate before either real arm can move.
    hardware_label = "ARX X5" if profile == "arx_x5_identity_joint_v1" else "PiPER-X"

    client = DualJointMirrorClient()
    recorder: Any | None = None
    recorder_finalized = False
    robots = _target_robots(task_env)
    pacer = RealtimePacer(
        frequency=float(task_env.obs_manager.collect_freq),
        enabled=os.environ.get("ROBODOJO_REALTIME", "1") != "0",
    )
    try:
        client.connect()
        obs = task_env.get_obs()
        follow_anchor = _measured_sim_state(task_env, robots)
        follow_command = follow_anchor
        response = client.exchange(_zero_deltas(follow_anchor))
        if response["mode"] != "follow" or response["edge"] is not None:
            raise DualJointMirrorError("policy mirror source did not begin in stable follow mode")
        if record_enabled:
            from src.eval_client.lerobot_stream_recorder import recorder_for_env

            recorder = recorder_for_env(task_env)
        checkpoint_label = "CP13" if recorder is not None else "CP12"
        print(
            (
                f"[CP13 Isaac] two leaders + i intervention + LeRobot recording -> "
                f"{recorder.record_dir}"
                if recorder is not None
                else f"[CP12 Isaac] policy + direct-joint i intervention. Press i for {hardware_label}."
            )
            if allow_intervention
            else f"[CP11 Isaac] policy rollout -> both {hardware_label} arms; no recording.",
            flush=True,
        )
        mode = "follow"
        manual_anchor: dict[str, tuple[np.ndarray, float]] | None = None
        pending_takeover_edge = 0
        pending_release_edge = 0
        chunk_id = -1
        timing = _ControlRateReporter(float(task_env.obs_manager.collect_freq))

        while not task_env.is_episode_end():
            if mode == "manual":
                current = _measured_sim_state(task_env, robots)
                response = client.exchange(_state_deltas(current, follow_anchor))
                if response["edge"] == "exit":
                    follow_anchor = current
                    follow_command = current
                    zero = client.exchange(_zero_deltas(current))
                    if zero["mode"] != "follow":
                        raise DualJointMirrorError("source did not complete intervention exit")
                    mode = "follow"
                    manual_anchor = None
                    obs = task_env.get_obs()
                    pending_takeover_edge = 0
                    pending_release_edge = -1
                    print(
                        f"\n[{checkpoint_label} Isaac] manual OFF; "
                        "stale chunk discarded; fresh inference.",
                        flush=True,
                    )
                    continue
                if response["mode"] != "manual" or manual_anchor is None:
                    raise DualJointMirrorError("manual sample arrived without a joint anchor")
                action = _manual_action(obs, manual_anchor, response)
                # Apply the freshest hardware sample before any synchronous
                # image/dataset work.  Manual X5 targets are already sampled at
                # the dataset rate, so the ordinary 8/10-step interpolation
                # only adds 32 ms of avoidable lag.
                task_env.take_action(action, interpolate=False)
                if recorder is not None:
                    recorder.append(
                        obs=obs,
                        policy_action=None,
                        human_action=action,
                        executed_action=action,
                        control={
                            "action_source": "human",
                            "intervention_mask": 1,
                            "active_arm": "both",
                            "takeover_edge": pending_takeover_edge,
                            "chunk_id": chunk_id,
                            "chunk_index": -1,
                            "timestamp": time.monotonic(),
                        },
                    )
                pending_takeover_edge = 0
                pacer.wait()
                obs = task_env.get_obs()
                timing.frame("manual")
                continue

            current = _measured_sim_state(task_env, robots)
            response = client.exchange(_state_deltas(follow_command, follow_anchor))
            if response["edge"] == "enter":
                if not allow_intervention or response["mode"] != "manual":
                    raise DualJointMirrorError("unexpected intervention entry")
                task_env.piperx_intervention_occurred = True
                manual_anchor = current
                mode = "manual"
                pending_takeover_edge = 1
                print(
                    f"\n[{checkpoint_label} Isaac] manual ON; "
                    "policy chunk discarded; simulation anchored.",
                    flush=True,
                )
                continue
            if response["mode"] != "follow":
                raise DualJointMirrorError("policy mirror source left follow mode without an edge")

            infer_start = time.monotonic()
            model_client.call(func_name="update_obs", obs=obs)
            actions = model_client.call(func_name="get_action")
            print(
                f"\n[X5 timing] policy inference={(time.monotonic() - infer_start) * 1000.0:.0f}ms",
                flush=True,
            )
            if not actions:
                raise DualJointMirrorError("policy-v1 returned an empty action chunk")
            chunk_id += 1
            chunk_stale = False
            for action_index, action in enumerate(actions):
                # Send the exact policy target before Isaac executes it.  The
                # physical X5 and simulator now start the same target together;
                # no post-step PhysX measurement (and its PD noise/one-frame
                # lag) is copied back to the real robot.
                pre_state = _measured_sim_state(task_env, robots)
                policy_target = _action_target_state(action)
                pre = client.exchange(_state_deltas(policy_target, follow_anchor))
                if pre["edge"] == "enter":
                    if not allow_intervention or pre["mode"] != "manual":
                        raise DualJointMirrorError("unexpected intervention entry")
                    task_env.piperx_intervention_occurred = True
                    manual_anchor = pre_state
                    mode = "manual"
                    pending_takeover_edge = 1
                    chunk_stale = True
                    print(
                        f"\n[{checkpoint_label} Isaac] manual ON; "
                        "policy chunk discarded; simulation anchored.",
                        flush=True,
                    )
                    break
                if pre["mode"] != "follow":
                    raise DualJointMirrorError("source left follow mode without an edge")
                follow_command = policy_target
                task_env.take_action(action)
                if recorder is not None:
                    recorder.append(
                        obs=obs,
                        policy_action=action,
                        human_action=None,
                        executed_action=action,
                        control={
                            "action_source": "policy",
                            "intervention_mask": 0,
                            "active_arm": "both",
                            "takeover_edge": pending_release_edge,
                            "chunk_id": chunk_id,
                            "chunk_index": action_index,
                            "timestamp": time.monotonic(),
                        },
                    )
                pending_release_edge = 0
                pacer.wait()
                obs = task_env.get_obs()
                timing.frame("follow")
                if task_env.is_episode_end() or action_index + 1 == len(actions):
                    break
                model_client.call(func_name="update_obs", obs=obs)
            if chunk_stale:
                continue
        if mode == "follow":
            # The source keeps pursuing its latest target between requests.
            # Let the physical arms finish the final simulator move before END
            # converts that in-flight target into a hold at the measured pose.
            time.sleep(1.0)
        client.end()
        if recorder is not None:
            success = bool(task_env.success[0])
            saved_path = recorder.finalize(
                accepted=True,
                success=success,
                reason="task_success" if success else "task_failure",
            )
            recorder_finalized = True
            if saved_path is None:
                raise DualJointMirrorError("dual-leader episode ended before any frame was recorded")
            print(f"[CP13 Isaac] saved LeRobot episode -> {saved_path}", flush=True)
    except Exception:
        if recorder is not None and not recorder_finalized:
            recorder.finalize(accepted=False, success=False, reason="exception")
            recorder_finalized = True
        raise
    finally:
        client.close()


def run_piperx_restored_recovery_episode(
    task_env: Any,
    *,
    follow_reference: dict[str, tuple[np.ndarray, float]],
    client: DualJointMirrorClient | None = None,
) -> str:
    """Record one correction attempt and return ``save`` or ``retry``."""

    if task_env.num_envs != 1:
        raise DualJointMirrorError("restored recovery requires exactly one Isaac environment")

    validation_dataset = os.environ.get("ROBODOJO_RECOVERY_VALIDATION_DATASET", "").strip()
    if validation_dataset:
        from src.eval_client.restore_recovery_validator import (
            run_restore_recovery_validation,
        )

        return run_restore_recovery_validation(
            task_env,
            dataset_root=validation_dataset,
            episode_index=int(os.environ.get("ROBODOJO_RECOVERY_VALIDATION_EPISODE", "0")),
            max_frames=int(os.environ.get("ROBODOJO_RECOVERY_VALIDATION_FRAMES", "25")),
            compare_start_frame=int(
                os.environ.get("ROBODOJO_RECOVERY_VALIDATION_COMPARE_START_FRAME", "0")
            ),
        )

    from src.eval_client.lerobot_stream_recorder import recorder_for_env

    owns_client = client is None
    if client is None:
        client = DualJointMirrorClient()
    recorder: Any | None = None
    recorder_finalized = False
    robots = _target_robots(task_env)
    pacer = RealtimePacer(
        frequency=float(task_env.obs_manager.collect_freq),
        enabled=os.environ.get("ROBODOJO_REALTIME", "1") != "0",
    )
    try:
        if owns_client:
            client.connect()
        restored_target = _measured_sim_state(task_env, robots)
        follow_delta = _state_deltas(restored_target, follow_reference)
        response = client.exchange(_zero_deltas(restored_target))
        if response["mode"] != "follow" or response["edge"] is not None:
            raise DualJointMirrorError("PiPER source did not begin in stable follow mode")
        print(
            "[RestoreRecovery] restored simulation is paused; "
            "PiPER-X is moving gradually to the corresponding restored pose.",
            flush=True,
        )

        settled_cycles = 0
        ready_announced = False
        last_alignment_report = 0.0
        while True:
            task_env.render()
            response = client.exchange(follow_delta)
            if response["terminal"] is not None:
                raise DualJointMirrorError("save was requested before manual recording started")
            if response["edge"] == "enter":
                if response["mode"] != "manual":
                    raise DualJointMirrorError("invalid restored-recovery intervention entry")
                if not ready_announced:
                    print(
                        "[RestoreRecovery][WARN] intervention started before alignment READY.",
                        flush=True,
                    )
                break
            if response["mode"] != "follow" or response["edge"] is not None:
                raise DualJointMirrorError("unexpected PiPER mode while waiting for intervention")
            max_error_deg = max(
                float(
                    np.degrees(
                        np.max(
                            np.abs(
                                response["sides"][side]["measured_q_rad"]
                                - response["sides"][side]["follow_target_q_rad"]
                            )
                        )
                    )
                )
                for side in SIDES
            )
            max_gripper_error = max(
                abs(
                    float(response["sides"][side]["leader_gripper_open_fraction"])
                    - restored_target[side][1]
                )
                for side in SIDES
            )
            settled_cycles = (
                settled_cycles + 1
                if max_error_deg <= 2.0 and max_gripper_error <= 0.02
                else 0
            )
            now = time.monotonic()
            if not ready_announced and now - last_alignment_report >= 0.5:
                print(
                    "\r[RestoreRecovery] aligning PiPER-X, "
                    f"max joint error={max_error_deg:.2f} deg, "
                    f"gripper error={max_gripper_error:.3f}",
                    end="",
                    flush=True,
                )
                last_alignment_report = now
            if settled_cycles >= 5 and not ready_announced:
                ready_announced = True
                print(
                    "\n[RestoreRecovery] READY. Press i in the PiPER terminal to start recording.",
                    flush=True,
                )
            pacer.wait()

        sim_anchor = _measured_sim_state(task_env, robots)
        task_env.piperx_intervention_occurred = True
        recorder = recorder_for_env(task_env)
        obs = task_env.get_obs()
        # Gripper observations in RoboDojo reflect the previous control target,
        # while ``sim_anchor`` is the actual restored joint opening.  Start the
        # manual segment from that measured opening so a zero-motion takeover
        # cannot create a synthetic first-frame gripper action jump.
        for side in SIDES:
            key = f"{side}_ee_joint_state"
            anchored_gripper = np.asarray([sim_anchor[side][1]], dtype=np.float64)
            obs["state"][key] = anchored_gripper.copy()
            obs["action"][key] = anchored_gripper.copy()
        chunk_id = 0
        first_action = _hold_action(obs)
        first_timestamp = time.monotonic()
        task_env.take_action(first_action)
        recorder.append(
            obs=obs,
            policy_action=None,
            human_action=first_action,
            executed_action=first_action,
            control={
                "action_source": "human",
                "intervention_mask": 1,
                "active_arm": "both",
                "takeover_edge": 1,
                "chunk_id": chunk_id,
                "chunk_index": -1,
                "timestamp": first_timestamp,
            },
        )
        pacer.wait()
        obs = task_env.get_obs()
        print(
            "[RestoreRecovery] recording human correction; "
            "press s to save/advance or r to discard/retry.",
            flush=True,
        )

        while True:
            current = _measured_sim_state(task_env, robots)
            response = client.exchange(_state_deltas(current, sim_anchor))
            if response["terminal"] == "retry":
                recorder.finalize(
                    accepted=False,
                    success=False,
                    reason="manual_recovery_retry",
                )
                recorder_finalized = True
                print(
                    "[RestoreRecovery] discarded current candidate; restoring this item.",
                    flush=True,
                )
                return "retry"
            if response["terminal"] == "save":
                saved_path = recorder.finalize(
                    accepted=True,
                    success=False,
                    reason="manual_recovery_complete",
                )
                recorder_finalized = True
                if saved_path is None:
                    raise DualJointMirrorError("manual recovery ended before any frame was recorded")
                print(f"[RestoreRecovery] saved LeRobot segment -> {saved_path}", flush=True)
                return "save"
            if response["mode"] != "manual" or response["edge"] is not None:
                raise DualJointMirrorError("manual recovery left intervention unexpectedly")
            action = _manual_action(obs, sim_anchor, response)
            action_timestamp = time.monotonic()
            # Apply the newest PiPER sample immediately.  The writer still
            # receives the same pre-action observation/action pair, but its
            # synchronous IPC/video enqueue no longer delays simulator control.
            task_env.take_action(action)
            recorder.append(
                obs=obs,
                policy_action=None,
                human_action=action,
                executed_action=action,
                control={
                    "action_source": "human",
                    "intervention_mask": 1,
                    "active_arm": "both",
                    "takeover_edge": 0,
                    "chunk_id": chunk_id,
                    "chunk_index": -1,
                    "timestamp": action_timestamp,
                },
            )
            pacer.wait()
            obs = task_env.get_obs()
    except Exception:
        if recorder is not None and not recorder_finalized:
            recorder.finalize(accepted=False, success=False, reason="exception")
        raise
    finally:
        if owns_client:
            client.close()
