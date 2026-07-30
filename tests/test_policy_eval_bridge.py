from dataclasses import replace
import unittest

import numpy as np

from src.eval_client.policy_runtime import (
    ARX_X5_SIM_PI05_PROFILE,
    MAX_LIFECYCLE_TEXT_BYTES,
    ErrorCode,
    PolicyProvenance,
    PolicyTransportError,
    PolicyV1BridgeStateError,
    PolicyV1EvalBridge,
    PolicyV1ProvenanceError,
    ResetPayload,
    ResetReason,
    TrialEndPayload,
    TrialStatus,
    interrupted_trial_end,
    operator_trial_end,
    parse_action_chunk,
    run_policy_v1_lifecycle,
    run_single_env_policy_episode,
    task_trial_end,
)


def _provenance():
    return PolicyProvenance(
        implementation="kai0",
        policy_family="pi05",
        adapter_profile="kai0_pi05_aloha_arx_x5_joint_v1",
        config_name="pi05_robodojo_arx_x5_joint",
        checkpoint_id="test/5000",
        checkpoint_digest="sha256:" + "1" * 64,
        checkpoint_step=5000,
        code_revision="2" * 40,
        dirty=False,
    )


def _reset():
    return ResetPayload(
        task_name="make_toast",
        simulator_seed=17,
        policy_seed=23,
        layout_id=17,
        layout_cycle=0,
        reason=ResetReason.EPISODE_START,
    )


def _trial_end():
    return TrialEndPayload(
        status=TrialStatus.ABORTED,
        success=None,
        score=None,
        reason="test",
    )


def _raw_observation():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "instruction": "make toast",
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": image},
            "cam_right_wrist": {"color": image},
        },
        "state": {
            "left_arm_joint_state": np.zeros(6, dtype=np.float32),
            "left_ee_joint_state": np.ones(1, dtype=np.float32),
            "right_arm_joint_state": np.zeros(6, dtype=np.float32),
            "right_ee_joint_state": np.ones(1, dtype=np.float32),
        },
        "data_format_version": "v1.0",
        "env_idx": 0,
    }


def _action_chunk():
    horizon = ARX_X5_SIM_PI05_PROFILE.action_spec.expected_horizon
    return parse_action_chunk(
        {
            "control_mode": "absolute_joint_position",
            "control_dt_s": 0.04,
            "commands": {
                "left_arm_joint_position": np.ones((horizon, 6), dtype=np.float32),
                "left_gripper_open_fraction": np.full(
                    (horizon, 1),
                    0.25,
                    dtype=np.float32,
                ),
                "right_arm_joint_position": np.full(
                    (horizon, 6),
                    2.0,
                    dtype=np.float32,
                ),
                "right_gripper_open_fraction": np.full(
                    (horizon, 1),
                    0.75,
                    dtype=np.float32,
                ),
            },
        },
        spec=ARX_X5_SIM_PI05_PROFILE.action_spec,
    )


class _FakePolicyClient:
    def __init__(self, url, *, provenance=None, **kwargs):
        self.url = url
        self.kwargs = kwargs
        self.provenance = _provenance() if provenance is None else provenance
        self.resets = []
        self.observations = []
        self.outcomes = []
        self.closed = False

    def connect(self):
        return self.provenance

    def reset(self, episode_id, payload):
        self.resets.append((episode_id, payload))

    def infer(self, observation):
        self.observations.append(observation)
        return _action_chunk()

    def trial_end(self, payload):
        self.outcomes.append(payload)

    def close(self):
        self.closed = True


class _Factory:
    def __init__(self, provenance=None):
        self.client = None
        self.provenance = provenance

    def __call__(self, url, **kwargs):
        self.client = _FakePolicyClient(url, provenance=self.provenance, **kwargs)
        return self.client


class PolicyV1EvalBridgeTest(unittest.TestCase):
    def setUp(self):
        self.factory = _Factory()
        self.bridge = PolicyV1EvalBridge(
            "ws://127.0.0.1:8000",
            client_factory=self.factory,
        )
        self.client = self.factory.client

    def tearDown(self):
        self.bridge.close()

    def test_connect_lifecycle_and_exact_action_mapping(self):
        self.assertEqual(self.bridge.provenance, _provenance())
        self.assertEqual(self.client.url, "ws://127.0.0.1:8000")
        self.assertIs(
            self.client.kwargs["profile"],
            ARX_X5_SIM_PI05_PROFILE,
        )

        self.bridge.start_episode("episode-1", _reset())
        self.bridge.call("update_obs", obs=_raw_observation())
        actions = self.bridge.call("get_action")

        self.assertEqual(len(actions), 50)
        self.assertEqual(
            set(actions[0]),
            {
                "left_arm_joint_state",
                "left_ee_joint_state",
                "right_arm_joint_state",
                "right_ee_joint_state",
            },
        )
        np.testing.assert_array_equal(
            actions[0]["left_arm_joint_state"],
            np.ones(6, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            actions[0]["right_ee_joint_state"],
            np.asarray([0.75], dtype=np.float32),
        )
        self.assertFalse(self.client.observations[0].head.flags.writeable)

        self.bridge.finish_episode(_trial_end())
        self.assertEqual(self.client.resets, [("episode-1", _reset())])
        self.assertEqual(self.client.outcomes, [_trial_end()])
        self.assertFalse(self.bridge.episode_active)

    def test_hello_provenance_requirements_fail_closed_before_episode(self):
        cases = (
            ({"expected_checkpoint_id": "wrong/59999"}, "checkpoint_id"),
            ({"expected_checkpoint_digest": "sha256:" + "a" * 64}, "checkpoint_digest"),
            ({"expected_code_revision": "f" * 40}, "code_revision"),
        )
        for requirements, label in cases:
            factory = _Factory()
            with self.subTest(label=label), self.assertRaisesRegex(
                PolicyV1ProvenanceError,
                label,
            ):
                PolicyV1EvalBridge(
                    "ws://127.0.0.1:18080",
                    client_factory=factory,
                    **requirements,
                )
            self.assertTrue(factory.client.closed)
            self.assertEqual(factory.client.resets, [])

    def test_hello_accepts_exact_clean_policy_identity(self):
        factory = _Factory()
        bridge = PolicyV1EvalBridge(
            "ws://127.0.0.1:18080",
            expected_checkpoint_id="test/5000",
            expected_checkpoint_digest="sha256:" + "1" * 64,
            expected_code_revision="2" * 40,
            require_clean=True,
            client_factory=factory,
        )
        try:
            self.assertEqual(bridge.provenance, _provenance())
            self.assertEqual(factory.client.resets, [])
        finally:
            bridge.close()

    def test_hello_rejects_dirty_kai0_before_episode(self):
        factory = _Factory(replace(_provenance(), dirty=True))
        with self.assertRaisesRegex(PolicyV1ProvenanceError, "dirty=true"):
            PolicyV1EvalBridge(
                "ws://127.0.0.1:18080",
                require_clean=True,
                client_factory=factory,
            )
        self.assertTrue(factory.client.closed)
        self.assertEqual(factory.client.resets, [])

    def test_requires_one_fresh_observation_per_infer(self):
        self.bridge.start_episode("episode-1", _reset())
        with self.assertRaises(PolicyV1BridgeStateError):
            self.bridge.call("get_action")

        self.bridge.call("update_obs", obs=_raw_observation())
        self.bridge.call("get_action")
        with self.assertRaises(PolicyV1BridgeStateError):
            self.bridge.call("get_action")

    def test_single_env_batch_compatibility_is_strict(self):
        self.bridge.start_episode("episode-1", _reset())
        self.bridge.call("update_obs_batch", obs=[_raw_observation()])
        batch = self.bridge.call("get_action_batch", obs=[0])
        self.assertEqual(len(batch), 1)
        self.assertEqual(len(batch[0]), 50)

        with self.assertRaises(ValueError):
            self.bridge.call(
                "update_obs_batch",
                obs=[_raw_observation(), _raw_observation()],
            )
        with self.assertRaises(ValueError):
            self.bridge.call("get_action_batch", obs=[1])

    def test_close_is_idempotent_and_never_fabricates_outcome(self):
        self.bridge.start_episode("episode-1", _reset())

        self.bridge.close()
        self.bridge.close()

        self.assertTrue(self.client.closed)
        self.assertEqual(self.client.outcomes, [])
        with self.assertRaises(PolicyV1BridgeStateError):
            self.bridge.call("update_obs", obs=_raw_observation())

    def test_legacy_reset_is_not_an_ambiguous_compatibility_call(self):
        self.bridge.start_episode("episode-1", _reset())

        with self.assertRaises(NotImplementedError):
            self.bridge.call("reset")

    def test_two_episodes_have_one_reset_and_trial_end_each(self):
        for index in range(2):
            self.bridge.start_episode(f"episode-{index}", _reset())
            self.bridge.finish_episode(task_trial_end(index == 0))

        self.assertEqual(
            [episode_id for episode_id, _ in self.client.resets],
            ["episode-0", "episode-1"],
        )
        self.assertEqual(
            self.client.outcomes,
            [task_trial_end(True), task_trial_end(False)],
        )

    def test_completed_trial_outcome_is_not_ambiguous(self):
        self.assertEqual(
            task_trial_end(True),
            TrialEndPayload(
                status=TrialStatus.SUCCESS,
                success=True,
                score=1.0,
                reason="task_success",
            ),
        )
        self.assertEqual(
            task_trial_end(False),
            TrialEndPayload(
                status=TrialStatus.FAILURE,
                success=False,
                score=0.0,
                reason="task_failure",
            ),
        )
        with self.assertRaises(TypeError):
            task_trial_end(1)

    def test_interrupted_trial_outcome_distinguishes_operator_and_error(self):
        operator = interrupted_trial_end(
            RuntimeError("next layout"),
            operator_boundary=True,
        )
        self.assertEqual(operator.status, TrialStatus.ABORTED)
        self.assertIsNone(operator.success)
        self.assertEqual(operator.reason, "RuntimeError: next layout")

        failure = interrupted_trial_end(
            RuntimeError(),
            operator_boundary=False,
        )
        self.assertEqual(failure.status, TrialStatus.ERROR)
        self.assertEqual(failure.reason, "RuntimeError")

        self.assertEqual(
            operator_trial_end("operator_session_complete"),
            TrialEndPayload(
                status=TrialStatus.ABORTED,
                success=None,
                score=None,
                reason="operator_session_complete",
            ),
        )

    def test_interrupted_reason_is_utf8_bounded_without_losing_error_type(self):
        for message in ("x" * 5000, "错" * 5000):
            outcome = interrupted_trial_end(
                ValueError(message),
                operator_boundary=False,
            )
            self.assertTrue(outcome.reason.startswith("ValueError: "))
            self.assertTrue(outcome.reason.endswith("…"))
            self.assertLessEqual(
                len(outcome.reason.encode("utf-8")),
                MAX_LIFECYCLE_TEXT_BYTES,
            )


class PolicyV1EvalLoopTest(unittest.TestCase):
    class _Env:
        num_envs = 1

        def __init__(self, stop_after=3):
            self.stop_after = stop_after
            self.executed = []
            self.observation_count = 0

        def is_episode_end(self):
            return len(self.executed) >= self.stop_after

        def get_obs(self):
            self.observation_count += 1
            return {"index": self.observation_count}

        def take_action(self, action):
            self.executed.append(action)

    class _Client:
        def __init__(self, actions):
            self.actions = actions
            self.calls = []

        def call(self, func_name, **kwargs):
            self.calls.append((func_name, kwargs))
            if func_name == "get_action":
                return self.actions
            return None

    def test_local_terminal_preempts_chunk_and_keeps_per_step_observations(self):
        env = self._Env(stop_after=3)
        client = self._Client(list(range(10)))

        run_single_env_policy_episode(env, client)

        self.assertEqual(env.executed, [0, 1, 2])
        update_calls = [kwargs["obs"]["index"] for name, kwargs in client.calls if name == "update_obs"]
        self.assertEqual(update_calls, [1, 2, 3])
        self.assertEqual(
            sum(name == "get_action" for name, _ in client.calls),
            1,
        )

    def test_empty_chunk_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty action chunk"):
            run_single_env_policy_episode(
                self._Env(stop_after=1),
                self._Client([]),
            )


class PolicyV1LifecycleRunnerTest(unittest.TestCase):
    class _Bridge:
        def __init__(self, finish_error=None):
            self.outcomes = []
            self.finish_error = finish_error

        def finish_episode(self, payload):
            self.outcomes.append(payload)
            if self.finish_error is not None:
                raise self.finish_error

    def test_normal_rollout_reports_exactly_one_outcome(self):
        bridge = self._Bridge()
        result = run_policy_v1_lifecycle(
            bridge,
            lambda: "result",
            normal_outcome=lambda: task_trial_end(True),
            is_operator_boundary=lambda _error: False,
        )

        self.assertEqual(result, "result")
        self.assertEqual(bridge.outcomes, [task_trial_end(True)])

    def test_operator_and_local_errors_are_terminated_without_masking(self):
        for operator_boundary, expected_status in (
            (True, TrialStatus.ABORTED),
            (False, TrialStatus.ERROR),
        ):
            bridge = self._Bridge()
            original = ValueError("rollout failed")

            def fail():
                raise original

            with self.assertRaises(ValueError) as raised:
                run_policy_v1_lifecycle(
                    bridge,
                    fail,
                    normal_outcome=lambda: task_trial_end(False),
                    is_operator_boundary=lambda _error: operator_boundary,
                )

            self.assertIs(raised.exception, original)
            self.assertEqual(len(bridge.outcomes), 1)
            self.assertEqual(bridge.outcomes[0].status, expected_status)

    def test_finish_failure_is_a_note_on_the_original_error(self):
        bridge = self._Bridge(finish_error=RuntimeError("finish failed"))
        original = ValueError("rollout failed")

        def fail():
            raise original

        with self.assertRaises(ValueError) as raised:
            run_policy_v1_lifecycle(
                bridge,
                fail,
                normal_outcome=lambda: task_trial_end(False),
                is_operator_boundary=lambda _error: False,
            )

        self.assertIs(raised.exception, original)
        self.assertTrue(
            any("TRIAL_END also failed" in note for note in original.__notes__),
        )

    def test_lost_episode_never_fabricates_trial_end(self):
        bridge = self._Bridge()
        original = PolicyTransportError(
            ErrorCode.EPISODE_LOST,
            "transport lost",
            episode_lost=True,
        )

        def fail():
            raise original

        with self.assertRaises(PolicyTransportError) as raised:
            run_policy_v1_lifecycle(
                bridge,
                fail,
                normal_outcome=lambda: task_trial_end(False),
                is_operator_boundary=lambda _error: False,
            )

        self.assertIs(raised.exception, original)
        self.assertEqual(bridge.outcomes, [])


if __name__ == "__main__":
    unittest.main()
