from types import SimpleNamespace
import unittest

import numpy as np

from src.eval_client.intervention_loop import (
    InterventionAcceptedAndExit,
    InterventionRejected,
)
from src.eval_client.piperx_bridge_client import (
    EMBODIMENT_PROFILE,
    ManualResolution,
    ManualSample,
    OperatorArmSample,
    OperatorSample,
    PiperXBridgeTransportError,
)
from src.eval_client.piperx_sim_dagger_loop import run_piperx_sim_dagger_episode


def _sample(
    seq,
    *,
    mode="policy",
    edge=None,
    terminal=None,
    generation=0,
    transition=None,
    manual_id=None,
    resolution=None,
    motion_accepted=None,
):
    arm = OperatorArmSample(
        (0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0),
        0.04,
        1_000_000 + seq,
    )
    manual_sample = None if manual_id is None else ManualSample(sample_id=manual_id, left=arm, right=arm)
    manual_resolution = None
    if resolution is not None:
        sample_id, decision = resolution
        manual_resolution = ManualResolution(
            sample_id=sample_id,
            decision=decision,
            follower_commanded=decision == "commit",
        )
    if motion_accepted is None:
        motion_accepted = transition is None and manual_sample is None
    if transition is not None:
        follower_mode = "hold"
    elif mode == "intervention":
        follower_mode = (
            "leader_follow" if manual_resolution is not None and manual_resolution.decision == "commit" else "hold"
        )
    else:
        follower_mode = "sim_follow"
    return OperatorSample(
        generation=generation,
        seq=seq,
        mode=mode,
        control_topology="policy_sim_to_follower_to_leader_manual_leader_joint_fanout",
        embodiment_profile=EMBODIMENT_PROFILE,
        leader_actuation_mode="native_leader" if mode == "intervention" else "output_follow",
        follower_actuation_mode=follower_mode,
        transition=transition,
        edge=edge,
        terminal_request=terminal,
        manual_sample=manual_sample,
        manual_resolution=manual_resolution,
        motion_accepted=motion_accepted,
        health={
            "ok": True,
            "control_topology": "policy_sim_to_follower_to_leader_manual_leader_joint_fanout",
            "embodiment_profile": EMBODIMENT_PROFILE,
        },
        diagnostics={
            "request_ok": True,
            "operation": (
                {"manual_anchor_latched": True}
                if manual_resolution is not None and manual_resolution.decision == "anchor"
                else None
            ),
        },
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

    def __init__(self, events=None):
        self.actions = []
        self.events = events if events is not None else []
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
        self.events.append(("sim_step", action["id"]))
        self.actions.append(action)

    def render(self):
        self.render_count += 1


class _Model:
    def __init__(self, events=None):
        self.chunk = 0
        self.calls = []
        self.events = events if events is not None else []

    def call(self, func_name, **kwargs):
        self.events.append(("policy", func_name))
        self.calls.append((func_name, kwargs))
        if func_name == "get_action":
            base = self.chunk * 100
            self.chunk += 1
            return [{"id": base + index} for index in range(3)]
        return None


class _Bridge:
    def __init__(self, responses, events=None):
        self.responses = list(responses)
        self.events = events if events is not None else []
        self.sim_targets = []
        self.arm_calls = []
        self.end_reasons = []
        self.fail_reasons = []

    def _respond(self, operation):
        if not self.responses:
            raise AssertionError(f"unexpected bridge operation {operation!r}")
        expected, response = self.responses.pop(0)
        if expected != operation:
            raise AssertionError(f"expected bridge operation {expected!r}, got {operation!r}")
        self.events.append(("bridge", operation))
        if isinstance(response, Exception):
            raise response
        return response

    def arm_and_begin_episode(self, episode_id, sim):
        self.events.append(("bridge", "arm_and_begin_episode"))
        self.arm_calls.append((episode_id, sim))

    # The concrete v3 client keeps this evaluator-facing compatibility alias;
    # either spelling must represent the same one-time SAFE_IDLE arm boundary.
    begin_episode = arm_and_begin_episode

    def exchange(self, sim):
        self.sim_targets.append(sim)
        return self._respond("exchange")

    def transition_ack(self, sim):
        self.sim_targets.append(sim)
        return self._respond("transition_ack")

    def manual_sample(self):
        return self._respond("manual_sample")

    def manual_resolve(self, sample_id, *, commit):
        operation = f"manual_resolve:{'commit' if commit else 'reject'}:{sample_id}"
        return self._respond(operation)

    def anchor_manual_sample(self, sample_id):
        return self._respond(f"manual_resolve:anchor:{sample_id}")

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
    def test_hardware_transitions_freeze_sim_and_discard_policy_chunk(self):
        events = []
        env = _Env(events)
        model = _Model(events)
        bridge = _Bridge(
            [
                ("exchange", _sample(0, transition="entering_intervention")),
                (
                    "transition_ack",
                    _sample(1, mode="intervention", edge="enter", generation=1),
                ),
                (
                    "manual_sample",
                    _sample(2, mode="intervention", generation=1, manual_id=1),
                ),
                (
                    "manual_resolve:anchor:1",
                    _sample(
                        3,
                        mode="intervention",
                        generation=1,
                        resolution=(1, "anchor"),
                    ),
                ),
                (
                    "manual_sample",
                    _sample(4, mode="intervention", generation=1, manual_id=2),
                ),
                (
                    "manual_resolve:commit:2",
                    _sample(
                        5,
                        mode="intervention",
                        generation=1,
                        resolution=(2, "commit"),
                    ),
                ),
                (
                    "manual_sample",
                    _sample(
                        6,
                        mode="intervention",
                        generation=1,
                        transition="reattaching_policy",
                    ),
                ),
                (
                    "transition_ack",
                    _sample(7, mode="policy", edge="exit", generation=2),
                ),
                ("exchange", _sample(8, mode="policy", generation=2)),
                ("exchange", _sample(9, mode="policy", generation=2)),
                (
                    "exchange",
                    _sample(10, mode="policy", terminal="accept_next", generation=2),
                ),
            ],
            events,
        )
        recorder = _Recorder()

        result = run_piperx_sim_dagger_episode(
            env,
            model,
            bridge=bridge,
            controller=_Controller(),
            recorder=recorder,
            pace_realtime=False,
        )

        self.assertEqual(result, "fake-lerobot")
        self.assertEqual(
            env.actions,
            [{"id": "human-1"}, {"id": 0}],
        )
        self.assertEqual(len(recorder.rows), 2)
        self.assertEqual(recorder.rows[0]["control"]["takeover_edge"], 1)
        self.assertGreaterEqual(env.render_count, 2)
        self.assertEqual(model.chunk, 1)
        self.assertEqual(len(bridge.arm_calls), 1)
        self.assertEqual(bridge.responses, [])
        self.assertEqual(bridge.fail_reasons, [])

        # Policy observation staging is the final software preflight before
        # the one-time v3 request is allowed to leave hardware SAFE_IDLE.
        self.assertLess(
            events.index(("policy", "update_obs")),
            events.index(("bridge", "arm_and_begin_episode")),
        )
        transition_acks = [index for index, event in enumerate(events) if event == ("bridge", "transition_ack")]
        self.assertEqual(len(transition_acks), 2)
        first_manual_sample = events.index(("bridge", "manual_sample"))
        self.assertLess(transition_acks[0], first_manual_sample)
        self.assertLess(
            events.index(("bridge", "manual_resolve:commit:2")),
            events.index(("sim_step", "human-1")),
        )
        self.assertLess(
            events.index(("bridge", "manual_resolve:anchor:1")),
            events.index(("bridge", "manual_resolve:commit:2")),
        )

    def test_enter_preempts_chunk_exit_reinfers_and_records_executed_actions(self):
        events = []
        env = _Env(events)
        model = _Model(events)
        bridge = _Bridge(
            [
                ("exchange", _sample(0)),  # pre-inference
                ("exchange", _sample(1)),  # policy action 0
                ("exchange", _sample(2, transition="entering_intervention")),
                (
                    "transition_ack",
                    _sample(3, mode="intervention", edge="enter", generation=1),
                ),
                (
                    "manual_sample",
                    _sample(4, mode="intervention", generation=1, manual_id=1),
                ),
                (
                    "manual_resolve:anchor:1",
                    _sample(
                        5,
                        mode="intervention",
                        generation=1,
                        resolution=(1, "anchor"),
                    ),
                ),
                (
                    "manual_sample",
                    _sample(6, mode="intervention", generation=1, manual_id=2),
                ),
                (
                    "manual_resolve:commit:2",
                    _sample(
                        7,
                        mode="intervention",
                        generation=1,
                        resolution=(2, "commit"),
                    ),
                ),
                (
                    "manual_sample",
                    _sample(
                        8,
                        mode="intervention",
                        generation=1,
                        transition="reattaching_policy",
                    ),
                ),
                (
                    "transition_ack",
                    _sample(9, mode="policy", edge="exit", generation=2),
                ),
                ("exchange", _sample(10, mode="policy", generation=2)),
                ("exchange", _sample(11, mode="policy", generation=2)),
                (
                    "exchange",
                    _sample(12, mode="policy", terminal="accept_next", generation=2),
                ),
            ],
            events,
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
                {"id": 100},
            ],
        )
        self.assertEqual(model.chunk, 2)
        self.assertIsNone(recorder.rows[1]["policy_action"])
        self.assertEqual(recorder.rows[1]["control"]["manual_sample_id"], 2)
        self.assertEqual(recorder.rows[1]["control"]["takeover_edge"], 1)
        self.assertEqual(recorder.rows[-1]["control"]["takeover_edge"], -1)
        self.assertEqual(bridge.end_reasons, ["operator_accept_next"])
        self.assertEqual(bridge.responses, [])
        self.assertEqual(bridge.fail_reasons, [])
        # Each inference, including the post-exit replan, has a fresh update.
        get_indices = [index for index, (name, _) in enumerate(model.calls) if name == "get_action"]
        for index in get_indices:
            self.assertEqual(model.calls[index - 1][0], "update_obs")
        # The exact cached bimanual sample must be physically resolved before
        # Isaac executes and before the row can become a training action.
        self.assertLess(
            events.index(("bridge", "manual_resolve:commit:2")),
            events.index(("sim_step", "human-1")),
        )

    def test_rejected_ik_freezes_without_step_or_training_frame(self):
        events = []
        env = _Env(events)
        bridge = _Bridge(
            [
                ("exchange", _sample(0, transition="entering_intervention")),
                (
                    "transition_ack",
                    _sample(1, mode="intervention", edge="enter", generation=1),
                ),
                (
                    "manual_sample",
                    _sample(2, mode="intervention", generation=1, manual_id=1),
                ),
                (
                    "manual_resolve:anchor:1",
                    _sample(
                        3,
                        mode="intervention",
                        generation=1,
                        resolution=(1, "anchor"),
                    ),
                ),
                (
                    "manual_sample",
                    _sample(4, mode="intervention", generation=1, manual_id=2),
                ),
                (
                    "manual_resolve:reject:2",
                    _sample(
                        5,
                        mode="intervention",
                        generation=1,
                        resolution=(2, "reject"),
                    ),
                ),
                (
                    "manual_sample",
                    _sample(
                        6,
                        mode="intervention",
                        terminal="accept_exit",
                        generation=1,
                    ),
                ),
            ],
            events,
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
        self.assertEqual(bridge.responses, [])
        self.assertIn(("bridge", "manual_resolve:reject:2"), events)
        self.assertFalse(any(event[0] == "sim_step" for event in events))

    def test_empty_left_request_discards_and_retries_layout(self):
        env = _Env()
        bridge = _Bridge([("exchange", _sample(0, terminal="discard_retry"))])
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
        self.assertEqual(bridge.responses, [])
        self.assertFalse(recorder.finalized["accepted"])

    def test_empty_accept_next_is_consumed_as_discard_and_retry(self):
        env = _Env()
        bridge = _Bridge([("exchange", _sample(0, terminal="accept_next"))])
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
        self.assertEqual(bridge.responses, [])
        self.assertFalse(recorder.finalized["accepted"])

    def test_bridge_fault_discards_candidate_and_fails_closed(self):
        env = _Env()
        failure = PiperXBridgeTransportError("socket lost")
        bridge = _Bridge(
            [
                ("exchange", _sample(0)),
                ("exchange", _sample(1)),
                ("exchange", failure),
            ]
        )
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
