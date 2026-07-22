"""Preemptible Pi0.5 rollout loop for human keyboard intervention."""

from __future__ import annotations

import os
import time

from src.eval_client.keyboard_teleop import CartesianTeleopController, KitKeyboardDevice


class InterventionRejected(Exception):
    """The operator discarded this attempt; the same layout should be retried."""


class InterventionSavedForRetry(Exception):
    """The operator saved this attempt and requested the same layout again."""

    def __init__(self, saved_path: str):
        self.saved_path = saved_path
        super().__init__(f"Saved {saved_path}; retry the same layout.")


class InterventionAcceptedAndExit(Exception):
    """The operator accepted the current attempt and requested a clean exit."""

    def __init__(self, saved_path: str | None):
        self.saved_path = saved_path
        super().__init__(f"Saved {saved_path}; stop interactive collection.")


class InterventionDiscardedAndExit(Exception):
    """The operator discarded the current attempt and requested a clean exit."""


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


def _operator_request(snapshot) -> str | None:
    """Return one unambiguous terminal request for this keyboard tick.

    Backspace is intentionally checked before the legacy ``abort_requested``
    field because the real keyboard adapter sets both for compatibility.
    Multiple physically simultaneous terminal keys are resolved in the most
    conservative order: discard/exit, accept/exit, discard/retry, then accept.
    """
    if getattr(snapshot, "discard_exit_requested", False) or getattr(snapshot, "abort_requested", False):
        return "discard_exit"
    if getattr(snapshot, "accept_exit_requested", False):
        return "accept_exit"
    if getattr(snapshot, "discard_retry_requested", False):
        return "discard_retry"
    if getattr(snapshot, "save_retry_requested", False):
        return "accept_retry"
    if getattr(snapshot, "accept_next_requested", False) or getattr(snapshot, "accept_requested", False):
        return "accept_next"
    return None


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

    Toggling I on replaces the *current* policy target with a human target and
    discards the rest of that policy chunk. Toggling I off causes the next
    policy inference to use the latest post-intervention observation.
    """
    if task_env.num_envs != 1:
        raise ValueError("Keyboard intervention supports exactly one simulation environment.")

    owned_keyboard = keyboard is None
    try:
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
            from src.eval_client.lerobot_stream_recorder import recorder_for_env

            recorder = recorder_for_env(task_env)

        pacer = RealtimePacer(
            frequency=float(task_env.obs_manager.collect_freq),
            enabled=pace_realtime and os.environ.get("ROBODOJO_REALTIME", "1") != "0",
        )
        print(f"[Intervention] {keyboard.help_text()}")
        print(f"[Intervention] Recording under {recorder.record_dir}")

        obs = task_env.get_obs()
        controller.reset(obs)
        _send_observation(task_env, model_client, obs)
    except Exception:
        # Setup can fail after the writer has already staged an episode (for
        # example on the first observation or controller reset).  Discard it
        # and release an owned Kit subscription before main decides whether
        # the environment can be retried.
        if recorder is not None:
            try:
                recorder.finalize(accepted=False, success=False, reason="setup_exception")
            except Exception:
                pass
        if owned_keyboard and keyboard is not None:
            try:
                keyboard.close()
            except Exception:
                pass
        raise
    chunk_id = -1
    pending_release_edge = 0
    accepted = False
    terminal_request: str | None = None
    finish_reason = "operator_pending"
    saved_path = None
    recorder_finalized = False

    def read_keyboard():
        # Kit dispatches GUI input while the app updates.  Pump one render here
        # so an I toggle made during a slow policy inference is observed
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
        # Operator-driven collection deliberately ignores benchmark success
        # and step-limit termination.  The enclosing environment disables its
        # own step-limit guard in this mode; here we always stage the next
        # observation until the operator labels the full trajectory.
        obs = task_env.get_obs()
        _send_observation(task_env, model_client, obs)

    try:
        while True:
            snapshot = read_keyboard()
            terminal_request = _operator_request(snapshot)
            if terminal_request in {"accept_next", "accept_retry"} and recorder.frame_count == 0:
                print("\n[Intervention] No frames have been staged yet; ignoring the accept request.")
                terminal_request = None
            if terminal_request is not None:
                accepted = terminal_request in {"accept_next", "accept_retry", "accept_exit"}
                finish_reason = f"operator_{terminal_request}"
                _mark_operator_end(task_env, rejected=not accepted)
                break

            if snapshot.deadman:
                if snapshot.takeover_pressed:
                    print("\n[Intervention] manual control ON (I toggled).")
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
                print("\n[Intervention] manual control OFF; requesting a fresh policy chunk.")

            chunk = _get_action_chunk(task_env, model_client)
            chunk_id += 1
            stale_chunk = False
            for chunk_index, policy_action in enumerate(chunk):
                snapshot = read_keyboard()
                terminal_request = _operator_request(snapshot)
                if terminal_request in {"accept_next", "accept_retry"} and recorder.frame_count == 0:
                    print("\n[Intervention] No frames have been staged yet; ignoring the accept request.")
                    terminal_request = None
                if terminal_request is not None:
                    accepted = terminal_request in {"accept_next", "accept_retry", "accept_exit"}
                    finish_reason = f"operator_{terminal_request}"
                    _mark_operator_end(task_env, rejected=not accepted)
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
                        f"\n[Intervention] manual control ON at chunk={chunk_id} index={chunk_index}; "
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

            if stale_chunk:
                if terminal_request is not None:
                    break
                continue

        saved_path = recorder.finalize(
            accepted=accepted,
            success=bool(task_env.success[0]),
            reason=finish_reason,
        )
        recorder_finalized = True
        if saved_path:
            print(f"\n[Intervention] Saved trajectory: {saved_path}")
        elif accepted and terminal_request != "accept_exit":
            raise RuntimeError("The recorder did not return a saved dataset path for an accepted episode.")
        elif not accepted:
            print("\n[Intervention] Episode discarded; no candidate data were kept.")
        else:
            print("\n[Intervention] No frames were staged; exiting without creating an empty episode.")

        if terminal_request == "discard_retry":
            raise InterventionRejected("Operator rejected the intervention episode.")
        if terminal_request == "discard_exit":
            raise InterventionDiscardedAndExit("Operator discarded the final episode and requested exit.")
        if terminal_request == "accept_retry":
            raise InterventionSavedForRetry(saved_path)
        if terminal_request == "accept_exit":
            raise InterventionAcceptedAndExit(saved_path)
        return saved_path
    except (
        InterventionRejected,
        InterventionSavedForRetry,
        InterventionAcceptedAndExit,
        InterventionDiscardedAndExit,
    ):
        raise
    except Exception:
        if not recorder_finalized:
            recorder.finalize(accepted=False, success=False, reason="exception")
        raise
    finally:
        if owned_keyboard:
            keyboard.close()
