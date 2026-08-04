"""Policy-agnostic single-environment rollout loop."""

from __future__ import annotations

import time


def _finish_reason(task_env) -> str:
    if bool(task_env.success[0]):
        return "task_success"
    if int(task_env.take_action_cnt[0]) >= int(task_env.step_lim):
        return "step_limit"
    return "task_failure"


def run_single_env_policy_episode(task_env, model_client, recorder=None) -> None:
    """Execute strict chunks with the current EvalEnv preemption semantics."""

    if task_env.num_envs != 1:
        raise ValueError("policy-v1 rollout supports exactly one environment")
    chunk_id = -1
    try:
        while not task_env.is_episode_end():
            observation = task_env.get_obs()
            model_client.call(func_name="update_obs", obs=observation)
            actions = model_client.call(func_name="get_action")
            if not actions:
                raise ValueError("policy-v1 returned an empty action chunk")
            chunk_id += 1
            for action_index, action in enumerate(actions):
                if recorder is not None:
                    recorder.append(
                        obs=observation,
                        policy_action=action,
                        human_action=None,
                        executed_action=action,
                        control={
                            "timestamp": time.monotonic(),
                            "intervention_mask": 0,
                            "action_source": "policy",
                            "takeover_edge": 0,
                            "chunk_id": chunk_id,
                            "chunk_index": action_index,
                        },
                    )
                task_env.take_action(action)
                if task_env.is_episode_end() or action_index + 1 == len(actions):
                    break
                # Preserve one pre-action observation/video/snapshot per
                # control target. The bridge retains this latest observation
                # for the next inference after the current chunk.
                observation = task_env.get_obs()
                model_client.call(func_name="update_obs", obs=observation)
    except BaseException:
        if recorder is not None:
            try:
                recorder.finalize(
                    accepted=False,
                    success=False,
                    reason="rollout_exception",
                )
            except Exception:
                # Preserve the causal policy/simulator exception. The writer
                # process is dropped by recorder.finalize/atexit either way.
                pass
        raise

    if recorder is not None:
        unstable = 0 in getattr(task_env, "unstable_envs", set())
        recorder.finalize(
            accepted=not unstable,
            success=bool(task_env.success[0]) and not unstable,
            reason="unstable_simulation" if unstable else _finish_reason(task_env),
        )
