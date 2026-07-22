"""Preemptible Pi0.5 rollout loop for human keyboard intervention."""

from __future__ import annotations

import os
import time

from src.eval_client.keyboard_teleop import CartesianTeleopController, KitKeyboardDevice


class InterventionRejected(Exception):
    """The operator rejected this attempt; the layout should be retried."""


class InterventionSavedForRetry(Exception):
    """The operator saved this attempt and requested the same layout again."""

    def __init__(self, saved_path: str):
        self.saved_path = saved_path
        super().__init__(f"Saved {saved_path}; retry the same layout.")


class RealtimePacer:
    """Prevent policy targets from being applied faster than their nominal rate."""

    def __init__(self, frequency: float, enabled: bool = True):
        self.period = 1.0 / float(frequency)
        self.enabled = enabled
        self._deadline: float | None = None

    def wait(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if self._deadline is None or self._deadline < now - self.period:
            self._deadline = now
        self._deadline += self.period
        delay = self._deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def _send_observation(task_env, model_client, obs: dict) -> None:
    if task_env.eval_batch:
        model_client.call(func_name="update_obs_batch", obs=[obs])
    else:
        model_client.call(func_name="update_obs", obs=obs)


def _get_action_chunk(task_env, model_client) -> list[dict]:
    if task_env.eval_batch:
        batch = model_client.call(func_name="get_action_batch", obs=[0])
        if len(batch) != 1:
            raise ValueError(f"Keyboard mode expected one action batch, got {len(batch)}")
        chunk = list(batch[0])
    else:
        chunk = list(model_client.call(func_name="get_action"))
    if not chunk:
        raise ValueError("Policy returned an empty action chunk.")
    return chunk


def _mark_operator_end(task_env, rejected: bool = False) -> None:
    if rejected:
        task_env.success[0] = False
    else:
        try:
            reward = task_env.reward_manager.get_reward(final_check=True)
            task_env.success[0] = bool(reward[0] > 1 - 1e-3)
        except Exception:
            task_env.success[0] = False
    task_env.end_flag[0] = True


def run_keyboard_intervention_episode(
    task_env,
    model_client,
    *,
    keyboard=None,
    controller=None,
    recorder=None,
    pace_realtime: bool = True,
):
    """Run one single-env episode with immediate human preemption.

    A Space press replaces the *current* policy target with a human target and
    discards the rest of that policy chunk.  Releasing Space causes the next
    policy inference to use the latest post-intervention observation.
    """
    if task_env.num_envs != 1:
        raise ValueError("Keyboard intervention supports exactly one simulation environment.")

    owned_keyboard = keyboard is None
    if keyboard is None:
        keyboard = KitKeyboardDevice(
            pos_step=float(os.environ.get("ROBODOJO_TELEOP_POS_STEP", "0.005")),
            rot_step=float(os.environ.get("ROBODOJO_TELEOP_ROT_STEP", "0.02")),
            deadman_timeout=float(os.environ.get("ROBODOJO_TELEOP_INPUT_TIMEOUT", "2.0")),
        )
    if controller is None:
        controller = CartesianTeleopController(
            task_env,
            max_joint_delta=float(os.environ.get("ROBODOJO_TELEOP_MAX_JOINT_DELTA", "0.35")),
        )
    if recorder is None:
        from src.eval_client.intervention_recorder import recorder_for_env

        recorder = recorder_for_env(
            task_env, os.environ.get("ROBODOJO_RECORD_DIR", os.path.join(task_env.save_dir, "interventions"))
        )

    pacer = RealtimePacer(
        frequency=float(task_env.obs_manager.collect_freq),
        enabled=pace_realtime and os.environ.get("ROBODOJO_REALTIME", "1") != "0",
    )
    print(f"[Intervention] {keyboard.help_text()}")
    print(f"[Intervention] Recording under {recorder.record_dir}")

    obs = task_env.get_obs()
    controller.reset(obs)
    _send_observation(task_env, model_client, obs)
    chunk_id = -1
    pending_release_edge = 0
    accepted = True
    retry_same_layout = False
    finish_reason = "environment_end"
    saved_path = None

    def read_keyboard():
        # Kit dispatches GUI input while the app updates.  Pump one render here
        # so a Space press made during a slow policy inference is observed
        # before the first target from that new chunk can execute.
        render = getattr(task_env, "render", None)
        if callable(render):
            render()
        return keyboard.snapshot()

    def execute(action, policy_action, human_action, control):
        nonlocal obs
        control = dict(control)
        control.setdefault("timestamp", time.monotonic())
        recorder.append(
            obs=obs,
            policy_action=policy_action,
            human_action=human_action,
            executed_action=action,
            control=control,
        )
        task_env.take_action(action)
        pacer.wait()
        if not task_env.is_episode_end():
            obs = task_env.get_obs()
            _send_observation(task_env, model_client, obs)

    try:
        while not task_env.is_episode_end():
            snapshot = read_keyboard()
            if snapshot.abort_requested:
                accepted = False
                finish_reason = "operator_abort"
                _mark_operator_end(task_env, rejected=True)
                break
            if snapshot.save_retry_requested:
                retry_same_layout = True
                finish_reason = "operator_save_retry"
                _mark_operator_end(task_env)
                break
            if snapshot.accept_requested:
                finish_reason = "operator_accept"
                _mark_operator_end(task_env)
                break

            if snapshot.deadman:
                action, control = controller.build_action(obs, snapshot)
                control.update(
                    {
                        "takeover_edge": 1 if snapshot.takeover_pressed else 0,
                        "chunk_id": chunk_id,
                        "chunk_index": -1,
                    }
                )
                execute(action, None, action if control["intervention_mask"] else None, control)
                continue

            if snapshot.takeover_released:
                pending_release_edge = -1

            chunk = _get_action_chunk(task_env, model_client)
            chunk_id += 1
            stale_chunk = False
            for chunk_index, policy_action in enumerate(chunk):
                snapshot = read_keyboard()
                if snapshot.abort_requested:
                    accepted = False
                    finish_reason = "operator_abort"
                    _mark_operator_end(task_env, rejected=True)
                    stale_chunk = True
                    break
                if snapshot.save_retry_requested:
                    retry_same_layout = True
                    finish_reason = "operator_save_retry"
                    _mark_operator_end(task_env)
                    stale_chunk = True
                    break
                if snapshot.accept_requested:
                    finish_reason = "operator_accept"
                    _mark_operator_end(task_env)
                    stale_chunk = True
                    break
                if snapshot.deadman:
                    manual_action, control = controller.build_action(obs, snapshot)
                    control.update(
                        {
                            "takeover_edge": 1,
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_index,
                        }
                    )
                    execute(
                        manual_action,
                        policy_action,
                        manual_action if control["intervention_mask"] else None,
                        control,
                    )
                    stale_chunk = True
                    print(
                        f"\n[Intervention] takeover at chunk={chunk_id} index={chunk_index}; "
                        f"discarded {len(chunk) - chunk_index} stale target(s)."
                    )
                    break

                execute(
                    policy_action,
                    policy_action,
                    None,
                    {
                        "action_source": "policy",
                        "intervention_mask": 0,
                        "ik_success": 1,
                        "active_arm": snapshot.active_arm,
                        "takeover_edge": pending_release_edge,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                    },
                )
                pending_release_edge = 0
                if task_env.is_episode_end():
                    break

            if stale_chunk:
                continue

        saved_path = recorder.finalize(
            accepted=accepted,
            success=bool(task_env.success[0]),
            reason=finish_reason,
        )
        if saved_path:
            print(f"\n[Intervention] Saved trajectory: {saved_path}")
        else:
            print("\n[Intervention] Episode rejected; no HDF5 file was kept.")
            raise InterventionRejected("Operator rejected the intervention episode.")
        if retry_same_layout:
            raise InterventionSavedForRetry(saved_path)
        return saved_path
    except (InterventionRejected, InterventionSavedForRetry):
        raise
    except Exception:
        recorder.finalize(accepted=False, success=False, reason="exception")
        raise
    finally:
        if owned_keyboard:
            keyboard.close()
