from types import SimpleNamespace
import unittest

import numpy as np

from src.eval_client.keyboard_teleop import KeyboardSnapshot
from src.eval_client.observation_loop import (
    ObservationAdvance,
    ObservationExit,
    run_keyboard_observation_episode,
)


def _snapshot(*, advance=False, exit_requested=False):
    return KeyboardSnapshot(
        deadman=False,
        active_arm="left",
        delta_pose=np.zeros(6),
        discard_retry_requested=advance,
        accept_exit_requested=exit_requested,
    )


class FakeKeyboard:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)

    def snapshot(self):
        return self.snapshots.pop(0) if self.snapshots else _snapshot()


class FakeModel:
    def __init__(self):
        self.chunk_requests = 0
        self.observation_updates = 0

    def call(self, func_name, **kwargs):
        del kwargs
        if func_name == "get_action_batch":
            self.chunk_requests += 1
            return [[{"id": index} for index in range(4)]]
        if func_name == "update_obs_batch":
            self.observation_updates += 1
        return None


class FakeEnv:
    num_envs = 1
    eval_batch = True
    obs_manager = SimpleNamespace(collect_freq=25)

    def __init__(self, end_after=100):
        self.actions = []
        self.render_count = 0
        self.end_after = end_after

    def render(self):
        self.render_count += 1

    def get_obs(self):
        return {"state": {}, "vision": {}, "instruction": "test"}

    def take_action(self, action):
        self.actions.append(action)

    def is_episode_end(self):
        return len(self.actions) >= self.end_after


class ObservationLoopTest(unittest.TestCase):
    def test_left_before_chunk_advances_without_action_or_recording(self):
        env = FakeEnv()
        model = FakeModel()

        with self.assertRaises(ObservationAdvance):
            run_keyboard_observation_episode(
                env,
                model,
                keyboard=FakeKeyboard([_snapshot(advance=True)]),
                pace_realtime=False,
            )

        self.assertEqual(env.actions, [])
        self.assertEqual(model.chunk_requests, 0)
        self.assertEqual(model.observation_updates, 1)

    def test_left_drops_the_rest_of_an_action_chunk(self):
        env = FakeEnv()
        model = FakeModel()
        keyboard = FakeKeyboard([_snapshot(), _snapshot(), _snapshot(advance=True)])

        with self.assertRaises(ObservationAdvance):
            run_keyboard_observation_episode(
                env,
                model,
                keyboard=keyboard,
                pace_realtime=False,
            )

        self.assertEqual(env.actions, [{"id": 0}])
        self.assertEqual(model.chunk_requests, 1)

    def test_escape_exits_cleanly_without_action(self):
        env = FakeEnv()

        with self.assertRaises(ObservationExit):
            run_keyboard_observation_episode(
                env,
                FakeModel(),
                keyboard=FakeKeyboard([_snapshot(exit_requested=True)]),
                pace_realtime=False,
            )

        self.assertEqual(env.actions, [])

    def test_natural_episode_end_returns(self):
        env = FakeEnv(end_after=2)
        model = FakeModel()

        run_keyboard_observation_episode(
            env,
            model,
            keyboard=FakeKeyboard([]),
            pace_realtime=False,
        )

        self.assertEqual(env.actions, [{"id": 0}, {"id": 1}])
        self.assertEqual(model.chunk_requests, 1)


if __name__ == "__main__":
    unittest.main()
