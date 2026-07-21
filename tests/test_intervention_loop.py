from types import SimpleNamespace
import unittest

import numpy as np

from src.eval_client.intervention_loop import InterventionRejected, run_keyboard_intervention_episode
from src.eval_client.keyboard_teleop import KeyboardSnapshot


def _snapshot(*, deadman=False, pressed=False, released=False):
    return KeyboardSnapshot(
        deadman=deadman,
        active_arm="left",
        delta_pose=np.zeros(6),
        takeover_pressed=pressed,
        takeover_released=released,
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

    def test_operator_abort_rejects_without_executing_an_action(self):
        env = FakeEnv()
        model = FakeModel()
        recorder = FakeRecorder()
        abort = KeyboardSnapshot(
            deadman=False,
            active_arm="left",
            delta_pose=np.zeros(6),
            abort_requested=True,
        )
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


if __name__ == "__main__":
    unittest.main()
