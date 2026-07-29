from types import SimpleNamespace
import unittest

import numpy as np

from src.eval_client.intervention_loop import (
    InterventionAcceptedAndExit,
    InterventionRejected,
)
from src.eval_client.piperx_bridge_client import (
    OperatorArmSample,
    OperatorSample,
    PiperXBridgeTransportError,
)
from src.eval_client.piperx_sim_dagger_loop import run_piperx_sim_dagger_episode


def _sample(seq, *, mode="policy", edge=None, terminal=None, generation=0):
    arm = OperatorArmSample((0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0), 0.04)
    return OperatorSample(
        generation=generation,
        seq=seq,
        mode=mode,
        edge=edge,
        terminal_request=terminal,
        left=arm,
        right=arm,
        mirror_accepted=True,
        health={"ok": True},
        diagnostics={},
    )


class _Robot:
    type = "target"

    def __init__(self, arm):
        self.arm_name = f"{arm}_arm"


class _RobotManager:
    def __init__(self, env):
        self.env = env
        self.robot_list = [_Robot("left"), _Robot("right")]

    def get_real_endpose(self, robot, env_idx_list, is_relative):
        del env_idx_list, is_relative
        sign = -1 if robot.arm_name.startswith("left") else 1
        return {
            0: np.asarray(
                [sign * 0.2 + 0.01 * len(self.env.actions), 0.0, 0.8, 1, 0, 0, 0],
                dtype=np.float64,
            )
        }


class _Env:
    num_envs = 1
    eval_batch = False
    task_name = "make_toast"
    run_id = "test-run"
    layout_cycle = 0
    env_seeds = [7]
    obs_manager = SimpleNamespace(collect_freq=25)
    reward_manager = SimpleNamespace(get_reward=lambda final_check: [0.0])

    def __init__(self):
        self.actions = []
        self.success = [False]
        self.end_flag = [False]
        self.robot_manager = _RobotManager(self)
        self.render_count = 0

    def get_obs(self):
        index = len(self.actions)
        return {
            "state": {
                "left_arm_joint_state": np.full(6, index * 0.01),
                "left_ee_joint_state": np.asarray([0.5]),
                "right_arm_joint_state": np.full(6, -index * 0.01),
                "right_ee_joint_state": np.asarray([0.5]),
            },
            "vision": {},
            "instruction": "make toast",
            "env_idx": 0,
        }

    def take_action(self, action):
        self.actions.append(action)

    def render(self):
        self.render_count += 1


class _Model:
    def __init__(self):
        self.chunk = 0
        self.calls = []

    def call(self, func_name, **kwargs):
        self.calls.append((func_name, kwargs))
        if func_name == "get_action":
            base = self.chunk * 100
            self.chunk += 1
            return [{"id": base + index} for index in range(3)]
        return None


class _Bridge:
    def __init__(self, samples):
        self.samples = list(samples)
        self.sim_targets = []
        self.begin_calls = []
        self.end_reasons = []
        self.fail_reasons = []

    def begin_episode(self, episode_id, sim):
        self.begin_calls.append((episode_id, sim))

    def exchange(self, sim):
        self.sim_targets.append(sim)
        sample = self.samples.pop(0)
        if isinstance(sample, Exception):
            raise sample
        return sample

    def end_episode(self, *, reason):
        self.end_reasons.append(reason)

    def fail_closed(self, reason):
        self.fail_reasons.append(reason)


class _Controller:
    def __init__(self, *, valid=True):
        self.valid = valid
        self.human_index = 0
        self.entered = []
        self.exit_count = 0

    def enter(self, sample, sim):
        self.entered.append((sample.generation, sim))

    def exit(self):
        self.exit_count += 1

    def build_action(self, obs, sample):
        del obs, sample
        self.human_index += 1
        if not self.valid:
            return {"id": "hold"}, {
                "action_source": "safety_hold",
                "intervention_mask": 0,
                "ik_success": 0,
                "active_arm": "both",
                "retarget_failures": {"left": "IK failed"},
            }
        return {"id": f"human-{self.human_index}"}, {
            "action_source": "human",
            "intervention_mask": 1,
            "ik_success": 1,
            "active_arm": "both",
        }


class _Recorder:
    record_dir = "fake-lerobot"

    def __init__(self):
        self.rows = []
        self.finalized = None

    @property
    def frame_count(self):
        return len(self.rows)

    def append(self, **kwargs):
        self.rows.append(kwargs)

    def finalize(self, **kwargs):
        self.finalized = kwargs
        if kwargs["accepted"] and self.rows:
            return "fake-lerobot"
        return None


class PiperXSimDaggerLoopTest(unittest.TestCase):
    def test_enter_preempts_chunk_exit_reinfers_and_records_executed_actions(self):
        env = _Env()
        model = _Model()
        bridge = _Bridge(
            [
                _sample(0),  # pre-inference
                _sample(1),  # policy action 0
                _sample(2, mode="intervention", edge="enter", generation=1),
                _sample(3, mode="intervention", generation=1),
                _sample(4, mode="policy", edge="exit", generation=2),
                _sample(5, mode="policy", generation=2),  # fresh inference boundary
                _sample(6, mode="policy", generation=2),  # policy action 100
                _sample(7, mode="policy", terminal="accept_next", generation=2),
            ]
        )
        controller = _Controller()
        recorder = _Recorder()

        result = run_piperx_sim_dagger_episode(
            env,
            model,
            bridge=bridge,
            controller=controller,
            recorder=recorder,
            pace_realtime=False,
        )

        self.assertEqual(result, "fake-lerobot")
        self.assertEqual(
            env.actions,
            [
                {"id": 0},
                {"id": "human-1"},
                {"id": "human-2"},
                {"id": 100},
            ],
        )
        self.assertEqual(model.chunk, 2)
        self.assertEqual(recorder.rows[1]["policy_action"], {"id": 1})
        self.assertEqual(recorder.rows[1]["control"]["takeover_edge"], 1)
        self.assertEqual(recorder.rows[-1]["control"]["takeover_edge"], -1)
        self.assertEqual(bridge.end_reasons, ["operator_accept_next"])
        self.assertEqual(bridge.fail_reasons, [])
        # Each inference, including the post-exit replan, has a fresh update.
        get_indices = [index for index, (name, _) in enumerate(model.calls) if name == "get_action"]
        for index in get_indices:
            self.assertEqual(model.calls[index - 1][0], "update_obs")
        # The exchange after a manual action receives the newly accepted sim pose.
        self.assertGreater(
            bridge.sim_targets[3].left.pose[0],
            bridge.sim_targets[2].left.pose[0],
        )

    def test_rejected_ik_freezes_without_step_or_training_frame(self):
        env = _Env()
        bridge = _Bridge(
            [
                _sample(0, mode="intervention", edge="enter", generation=1),
                _sample(1, mode="intervention", terminal="accept_exit", generation=1),
            ]
        )
        recorder = _Recorder()

        with self.assertRaises(InterventionAcceptedAndExit) as raised:
            run_piperx_sim_dagger_episode(
                env,
                _Model(),
                bridge=bridge,
                controller=_Controller(valid=False),
                recorder=recorder,
                pace_realtime=False,
            )

        self.assertIsNone(raised.exception.saved_path)
        self.assertEqual(env.actions, [])
        self.assertEqual(recorder.rows, [])
        self.assertTrue(recorder.finalized["accepted"])
        self.assertGreater(env.render_count, 0)

    def test_empty_left_request_discards_and_retries_layout(self):
        env = _Env()
        bridge = _Bridge([_sample(0, terminal="discard_retry")])
        recorder = _Recorder()

        with self.assertRaises(InterventionRejected):
            run_piperx_sim_dagger_episode(
                env,
                _Model(),
                bridge=bridge,
                controller=_Controller(),
                recorder=recorder,
                pace_realtime=False,
            )

        self.assertEqual(bridge.end_reasons, ["operator_discard_retry"])
        self.assertFalse(recorder.finalized["accepted"])

    def test_bridge_fault_discards_candidate_and_fails_closed(self):
        env = _Env()
        failure = PiperXBridgeTransportError("socket lost")
        bridge = _Bridge([_sample(0), _sample(1), failure])
        recorder = _Recorder()

        with self.assertRaises(PiperXBridgeTransportError):
            run_piperx_sim_dagger_episode(
                env,
                _Model(),
                bridge=bridge,
                controller=_Controller(),
                recorder=recorder,
                pace_realtime=False,
            )

        self.assertEqual(env.actions, [{"id": 0}])
        self.assertFalse(recorder.finalized["accepted"])
        self.assertEqual(bridge.fail_reasons, ["episode_exception"])


if __name__ == "__main__":
    unittest.main()
