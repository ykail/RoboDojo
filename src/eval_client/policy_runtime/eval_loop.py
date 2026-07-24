"""Policy-agnostic single-environment rollout loop."""

from __future__ import annotations


def run_single_env_policy_episode(task_env, model_client) -> None:
    """Execute strict chunks with the current EvalEnv preemption semantics."""

    if task_env.num_envs != 1:
        raise ValueError("policy-v1 rollout supports exactly one environment")
    while not task_env.is_episode_end():
        observation = task_env.get_obs()
        model_client.call(func_name="update_obs", obs=observation)
        actions = model_client.call(func_name="get_action")
        if not actions:
            raise ValueError("policy-v1 returned an empty action chunk")
        for action_index, action in enumerate(actions):
            task_env.take_action(action)
            if task_env.is_episode_end() or action_index + 1 == len(actions):
                break
            # Preserve the existing evaluator's one observation/video frame
            # per control target. The bridge keeps only the latest snapshot,
            # which becomes the next INFER observation after the chunk.
            observation = task_env.get_obs()
            model_client.call(func_name="update_obs", obs=observation)
