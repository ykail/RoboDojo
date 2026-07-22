from types import SimpleNamespace
import unittest

import numpy as np

from src.eval_client.intervention_loop import (
    InterventionAcceptedAndExit,
    InterventionDiscardedAndExit,
    InterventionRejected,
    InterventionSavedForRetry,
    run_keyboard_intervention_episode,
)
from src.eval_client.keyboard_teleop import KeyboardSnapshot


def _snapshot(
    *,
    deadman=False,
    pressed=False,
    released=False,
    accept_next=False,
    discard_retry=False,
    accept_exit=False,
    discard_exit=False,
    save_retry=False,
):
    return KeyboardSnapshot(
        deadman=deadman,
        active_arm="left",
        delta_pose=np.zeros(6),
        takeover_pressed=pressed,
        takeover_released=released,
        accept_next_requested=accept_next,
        discard_retry_requested=discard_retry,
        accept_exit_requested=accept_exit,
        discard_exit_requested=discard_exit,
        save_retry_requested=save_retry,
    )


class FakeKeyboard:
    record_dir = "unused"

    def __init__(self, snapshots):
        self.snapshots = list(snapshots)

    def snapshot(self):
        return self.snapshots.pop(0) if self.snapshots else _snapshot()

    @staticmethod
    def help_text():
        return "fake keyboard"


class FakeController:
    def reset(self, obs):
        del obs

    def build_action(self, obs, snapshot):
        del obs, snapshot
        return {"id": "human"}, {
            "action_source": "human",
            "intervention_mask": 1,
            "ik_success": 1,
            "active_arm": "left",
        }


class FakeRecorder:
    record_dir = "fake-records"

    def __init__(self):
        self.rows = []
        self.finalized = None

    def append(self, **kwargs):
        self.rows.append(kwargs)

    @property
    def frame_count(self):
        return len(self.rows)

    def finalize(self, **kwargs):
        self.finalized = kwargs
        return "fake.hdf5" if kwargs["accepted"] else None


class FakeModel:
    def __init__(self):
        self.chunk_number = 0

    def call(self, func_name, **kwargs):
        del kwargs
        if func_name == "get_action_batch":
            base = self.chunk_number * 100
            self.chunk_number += 1
            return [[{"id": base + index} for index in range(4)]]
        return None


class FakeEnv:
    num_envs = 1
    eval_batch = True
    obs_manager = SimpleNamespace(collect_freq=25)
    reward_manager = SimpleNamespace(get_reward=lambda final_check: [0.0])

    def __init__(self):
        self.success = [True]
        self.end_flag = [False]
        self.actions = []

    def get_obs(self):
        return {"state": {}, "vision": {}, "instruction": "test"}

    def take_action(self, action):
        self.actions.append(action)
        if len(self.actions) == 4:
            self.end_flag[0] = True

    def is_episode_end(self):
        return self.end_flag[0]


class InterventionLoopTest(unittest.TestCase):
    def test_setup_failure_discards_started_recorder(self):
        class BrokenInitialObservationEnv(FakeEnv):
            def get_obs(self):
                raise RuntimeError("camera failed")

        recorder = FakeRecorder()
        with self.assertRaisesRegex(RuntimeError, "camera failed"):
            run_keyboard_intervention_episode(
                BrokenInitialObservationEnv(),
                FakeModel(),
                keyboard=FakeKeyboard([]),
                controller=FakeController(),
                recorder=recorder,
                pace_realtime=False,
            )

        self.assertFalse(recorder.finalized["accepted"])
        self.assertEqual(recorder.finalized["reason"], "setup_exception")

    def test_takeover_discards_remainder_and_replans_after_release(self):
        env = FakeEnv()
        model = FakeModel()
        recorder = FakeRecorder()
        keyboard = FakeKeyboard(
            [
                _snapshot(),
                _snapshot(),
                _snapshot(deadman=True, pressed=True),
                _snapshot(released=True),
                _snapshot(),
                _snapshot(),
                _snapshot(accept_next=True),
            ]
        )

        run_keyboard_intervention_episode(
            env,
            model,
            keyboard=keyboard,
            controller=FakeController(),
            recorder=recorder,
            pace_realtime=False,
        )

        self.assertEqual(env.actions, [{"id": 0}, {"id": "human"}, {"id": 100}, {"id": 101}])
        self.assertEqual(model.chunk_number, 2)
        self.assertEqual(recorder.rows[1]["control"]["takeover_edge"], 1)
        self.assertEqual(recorder.rows[2]["control"]["takeover_edge"], -1)
        self.assertTrue(recorder.finalized["accepted"])
        self.assertEqual(recorder.finalized["reason"], "operator_accept_next")

    def test_left_discards_and_retries_without_executing_an_action(self):
        env = FakeEnv()
        model = FakeModel()
        recorder = FakeRecorder()
        abort = _snapshot(discard_retry=True)
        with self.assertRaises(InterventionRejected):
            run_keyboard_intervention_episode(
                env,
                model,
                keyboard=FakeKeyboard([abort]),
                controller=FakeController(),
                recorder=recorder,
                pace_realtime=False,
            )
        self.assertEqual(env.actions, [])
        self.assertEqual(model.chunk_number, 0)
        self.assertFalse(recorder.finalized["accepted"])
        self.assertEqual(recorder.finalized["reason"], "operator_discard_retry")

    def test_escape_accepts_and_exits(self):
        env = FakeEnv()
        recorder = FakeRecorder()
        with self.assertRaises(InterventionAcceptedAndExit) as raised:
            run_keyboard_intervention_episode(
                env,
                FakeModel(),
                keyboard=FakeKeyboard([_snapshot(accept_exit=True)]),
                controller=FakeController(),
                recorder=recorder,
                pace_realtime=False,
            )
        self.assertEqual(raised.exception.saved_path, "fake.hdf5")
        self.assertTrue(recorder.finalized["accepted"])
        self.assertEqual(recorder.finalized["reason"], "operator_accept_exit")

    def test_escape_before_first_frame_exits_without_an_empty_episode(self):
        class EmptyAwareRecorder(FakeRecorder):
            def finalize(self, **kwargs):
                self.finalized = kwargs
                return None

        recorder = EmptyAwareRecorder()
        with self.assertRaises(InterventionAcceptedAndExit) as raised:
            run_keyboard_intervention_episode(
                FakeEnv(),
                FakeModel(),
                keyboard=FakeKeyboard([_snapshot(accept_exit=True)]),
                controller=FakeController(),
                recorder=recorder,
                pace_realtime=False,
            )
        self.assertIsNone(raised.exception.saved_path)
        self.assertTrue(recorder.finalized["accepted"])

    def test_backspace_discards_and_exits(self):
        env = FakeEnv()
        recorder = FakeRecorder()
        with self.assertRaises(InterventionDiscardedAndExit):
            run_keyboard_intervention_episode(
                env,
                FakeModel(),
                keyboard=FakeKeyboard([_snapshot(discard_exit=True)]),
                controller=FakeController(),
                recorder=recorder,
                pace_realtime=False,
            )
        self.assertFalse(recorder.finalized["accepted"])
        self.assertEqual(recorder.finalized["reason"], "operator_discard_exit")

    def test_runtime_exception_discards_staged_candidate(self):
        class BrokenEnv(FakeEnv):
            def take_action(self, action):
                self.actions.append(action)
                raise RuntimeError("simulator fault")

        recorder = FakeRecorder()
        with self.assertRaisesRegex(RuntimeError, "simulator fault"):
            run_keyboard_intervention_episode(
                BrokenEnv(),
                FakeModel(),
                keyboard=FakeKeyboard([_snapshot(), _snapshot()]),
                controller=FakeController(),
                recorder=recorder,
                pace_realtime=False,
            )
        self.assertEqual(len(recorder.rows), 1)
        self.assertFalse(recorder.finalized["accepted"])
        self.assertEqual(recorder.finalized["reason"], "exception")

    def test_save_retry_keeps_episode_and_requests_same_layout(self):
        env = FakeEnv()
        model = FakeModel()
        recorder = FakeRecorder()
        keyboard = FakeKeyboard([_snapshot(), _snapshot(), _snapshot(save_retry=True)])

        with self.assertRaises(InterventionSavedForRetry) as raised:
            run_keyboard_intervention_episode(
                env,
                model,
                keyboard=keyboard,
                controller=FakeController(),
                recorder=recorder,
                pace_realtime=False,
            )

        self.assertEqual(env.actions, [{"id": 0}])
        self.assertEqual(raised.exception.saved_path, "fake.hdf5")
        self.assertTrue(recorder.finalized["accepted"])
        self.assertEqual(recorder.finalized["reason"], "operator_accept_retry")


if __name__ == "__main__":
    unittest.main()
