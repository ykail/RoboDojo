from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from src.eval_client import piperx_dual_joint_mirror as mirror


def _obs() -> dict:
    return {
        "state": {
            "left_arm_joint_state": np.zeros(6),
            "left_ee_joint_state": np.asarray([0.5]),
            "right_arm_joint_state": np.zeros(6),
            "right_ee_joint_state": np.asarray([0.5]),
        }
    }


def _action(value: float = 0.2) -> dict:
    return {
        "left_arm_joint_state": np.full(6, value),
        "left_ee_joint_state": np.asarray([0.3]),
        "right_arm_joint_state": np.full(6, -value),
        "right_ee_joint_state": np.asarray([0.7]),
    }


class _RobotManager:
    def __init__(self) -> None:
        self.robot_list = [
            SimpleNamespace(
                arm_name=f"{side}_arm",
                type="target",
                gripper_scale=(0.0, 1.0),
                gripper_move={"sign": 1},
            )
            for side in mirror.SIDES
        ]

    def get_joint(self, robot, env_idx_list):
        del robot, env_idx_list
        return [np.zeros(6)]

    def get_end_effector_real_val(self, robot, env_idx_list):
        del robot, env_idx_list
        return [[0.5]]


class _TaskEnv:
    def __init__(self, events: list) -> None:
        self.num_envs = 1
        self.eval_batch = False
        self.obs_manager = SimpleNamespace(collect_freq=25)
        self.robot_manager = _RobotManager()
        self.actions: list[tuple[dict, bool]] = []
        self.events = events
        self.success = [True]
        self.end_flag = [False]
        self.reward_manager = SimpleNamespace(get_reward=lambda *, final_check: [0.0])
        self.piperx_intervention_occurred = False
        self.capture_updates: list[bool] = []
        self.capture_manager = SimpleNamespace(
            set_updates_enabled=self.capture_updates.append,
        )

    def get_obs(self):
        return _obs()

    def take_action(self, action, *, interpolate=True):
        self.events.append("take_action")
        self.actions.append((action, interpolate))

    def render(self):
        self.events.append("render")

    def is_episode_end(self):
        return bool(self.actions)


class _Model:
    def call(self, *, func_name, **kwargs):
        del kwargs
        if func_name == "update_obs":
            return None
        if func_name == "get_action":
            return [_action()]
        raise AssertionError(func_name)


class _Client:
    def __init__(self, responses: list[dict], events: list) -> None:
        self.responses = list(responses)
        self.events = events
        self.deltas: list[dict] = []

    def connect(self):
        return None

    def exchange(self, deltas):
        self.events.append("exchange")
        self.deltas.append(deltas)
        return self.responses.pop(0)

    def end(self):
        return None

    def close(self):
        return None


def _response(*, mode="follow", edge=None, terminal=None, delta=0.0) -> dict:
    return {
        "mode": mode,
        "edge": edge,
        "terminal": terminal,
        "sides": {
            side: {
                "leader_delta_q_rad": np.full(6, delta),
                "leader_gripper_open_fraction": 0.5,
            }
            for side in mirror.SIDES
        },
    }


class X5PolicyMirrorLoopTest(unittest.TestCase):
    def test_left_discards_and_requests_same_layout_retry(self) -> None:
        events: list[str] = []
        client = _Client([_response(), _response(terminal="retry")], events)
        task = _TaskEnv(events)
        with mock.patch.object(mirror, "DualJointMirrorClient", return_value=client), mock.patch.dict(
            mirror.os.environ,
            {
                "ROBODOJO_DUAL_MIRROR_PROFILE": "arx_x5_identity_joint_v1",
                "ROBODOJO_DUAL_MIRROR_RECORD": "0",
                "ROBODOJO_REALTIME": "0",
            },
        ):
            from src.eval_client.intervention_loop import InterventionRejected

            with self.assertRaises(InterventionRejected):
                mirror.run_piperx_policy_leader_mirror_episode(
                    task,
                    _Model(),
                    allow_intervention=True,
                )

        self.assertEqual(task.actions, [])
        self.assertEqual((task.success, task.end_flag), ([False], [True]))

    def test_right_finishes_without_waiting_for_task_step_limit(self) -> None:
        events: list[str] = []
        client = _Client([_response(), _response(terminal="save")], events)
        task = _TaskEnv(events)
        with mock.patch.object(mirror, "DualJointMirrorClient", return_value=client), mock.patch.dict(
            mirror.os.environ,
            {
                "ROBODOJO_DUAL_MIRROR_PROFILE": "arx_x5_identity_joint_v1",
                "ROBODOJO_DUAL_MIRROR_RECORD": "0",
                "ROBODOJO_REALTIME": "0",
            },
        ):
            mirror.run_piperx_policy_leader_mirror_episode(
                task,
                _Model(),
                allow_intervention=True,
            )

        self.assertEqual(task.actions, [])
        self.assertEqual((task.success, task.end_flag), ([False], [True]))

    def test_policy_target_reaches_hardware_before_same_sim_action(self) -> None:
        events: list[str] = []
        client = _Client([_response(), _response(), _response()], events)
        task = _TaskEnv(events)
        with mock.patch.object(mirror, "DualJointMirrorClient", return_value=client), mock.patch.object(
            mirror.time, "sleep"
        ), mock.patch.dict(
            mirror.os.environ,
            {
                "ROBODOJO_DUAL_MIRROR_PROFILE": "arx_x5_identity_joint_v1",
                "ROBODOJO_DUAL_MIRROR_RECORD": "0",
                "ROBODOJO_REALTIME": "0",
            },
        ):
            mirror.run_piperx_policy_leader_mirror_episode(task, _Model(), allow_intervention=True)

        self.assertLess(events.index("exchange", 2), events.index("take_action"))
        np.testing.assert_allclose(client.deltas[-1]["left"][0], np.full(6, 0.2))
        np.testing.assert_allclose(client.deltas[-1]["right"][0], np.full(6, -0.2))

    def test_manual_action_disables_second_sim_interpolation(self) -> None:
        events: list[str] = []
        client = _Client(
            [
                _response(),
                _response(mode="manual", edge="enter"),
                _response(mode="manual", delta=0.1),
            ],
            events,
        )
        task = _TaskEnv(events)
        with mock.patch.object(mirror, "DualJointMirrorClient", return_value=client), mock.patch.dict(
            mirror.os.environ,
            {
                "ROBODOJO_DUAL_MIRROR_PROFILE": "arx_x5_identity_joint_v1",
                "ROBODOJO_DUAL_MIRROR_RECORD": "0",
                "ROBODOJO_REALTIME": "0",
            },
        ):
            mirror.run_piperx_policy_leader_mirror_episode(task, _Model(), allow_intervention=True)

        self.assertFalse(task.actions[0][1])
        np.testing.assert_allclose(
            task.actions[0][0]["left_arm_joint_state"],
            np.full(6, 0.1),
        )
        self.assertEqual(task.capture_updates, [False, True])


if __name__ == "__main__":
    unittest.main()
