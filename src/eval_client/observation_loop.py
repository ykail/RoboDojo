"""Visible, keyboard-advanced policy rollout for quick behavior inspection."""

from __future__ import annotations

from src.eval_client.intervention_loop import RealtimePacer
from src.eval_client.keyboard_teleop import KitKeyboardDevice


class ObservationAdvance(Exception):
    """The operator finished inspecting this layout and requested the next one."""


class ObservationExit(Exception):
    """The operator requested a clean exit from the observation session."""


def _send_observation(task_env, model_client, obs: dict) -> None:
    if task_env.eval_batch:
        model_client.call(func_name="update_obs_batch", obs=[obs])
    else:
        model_client.call(func_name="update_obs", obs=obs)


def _get_action_chunk(task_env, model_client) -> list[dict]:
    if task_env.eval_batch:
        batch = model_client.call(func_name="get_action_batch", obs=[0])
        if len(batch) != 1:
            raise ValueError(f"Keyboard observation expected one action batch, got {len(batch)}")
        chunk = list(batch[0])
    else:
        chunk = list(model_client.call(func_name="get_action"))
    if not chunk:
        raise ValueError("Policy returned an empty action chunk.")
    return chunk


def run_keyboard_observation_episode(
    task_env,
    model_client,
    *,
    keyboard=None,
    pace_realtime: bool = True,
) -> None:
    """Run policy control until LEFT advances, Escape exits, or the task ends.

    This mode deliberately has no recorder and no teleoperation controller. It
    is intended for inspecting early policy behavior such as the first object
    grasp while keeping one visible simulation environment responsive.
    """

    if task_env.num_envs != 1:
        raise ValueError("Keyboard observation supports exactly one simulation environment.")

    owned_keyboard = keyboard is None
    if keyboard is None:
        keyboard = KitKeyboardDevice()
    pacer = RealtimePacer(
        frequency=float(task_env.obs_manager.collect_freq),
        enabled=pace_realtime,
    )

    def poll_operator() -> None:
        # Kit dispatches keyboard events during rendering. Poll before asking
        # for a new chunk and before every action so LEFT drops stale targets.
        render = getattr(task_env, "render", None)
        if callable(render):
            render()
        snapshot = keyboard.snapshot()
        if snapshot.accept_exit_requested or snapshot.discard_exit_requested or snapshot.abort_requested:
            raise ObservationExit("Operator requested observation exit.")
        if snapshot.discard_retry_requested:
            raise ObservationAdvance("Operator requested the next layout.")

    print(
        "[Observer] Policy-only visible mode: LEFT ARROW=next layout | "
        "ESCAPE/BACKSPACE=exit | no data or evaluation video is saved."
    )

    try:
        obs = task_env.get_obs()
        _send_observation(task_env, model_client, obs)

        while not task_env.is_episode_end():
            poll_operator()
            chunk = _get_action_chunk(task_env, model_client)
            for action in chunk:
                poll_operator()
                task_env.take_action(action)
                pacer.wait()
                if task_env.is_episode_end():
                    return
                obs = task_env.get_obs()
                _send_observation(task_env, model_client, obs)
    finally:
        if owned_keyboard:
            keyboard.close()
