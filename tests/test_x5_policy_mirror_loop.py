from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace
from pathlib import Path
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
        self.get_obs_count = 0
        self.capture_manager = SimpleNamespace(
            set_updates_enabled=self.capture_updates.append,
        )

    def get_obs(self):
        self.events.append("get_obs")
        self.get_obs_count += 1
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


class _Recorder:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.frames: list[dict] = []
        self.finishes: list[dict] = []
        self.record_dir = "/tmp/x5-online-test"

    def append(self, **kwargs):
        self.events.append("record")
        self.frames.append(kwargs)

    def finalize(self, **kwargs):
        self.events.append("finalize")
        self.finishes.append(kwargs)
        return self.record_dir if kwargs.get("accepted") else None


class _SynchronousPendingRecorder:
    """Deterministic test double for the production one-frame async wrapper."""

    def __init__(self, recorder):
        self.recorder = recorder

    @property
    def record_dir(self):
        return self.recorder.record_dir

    def append(self, **kwargs):
        self.recorder.append(**kwargs)
        return 0.0

    def finalize(self, **kwargs):
        return self.recorder.finalize(**kwargs)

    def discard(self, **kwargs):
        self.recorder.finalize(**kwargs)

    def close(self):
        return None


class _BlockingPendingRecorder(_SynchronousPendingRecorder):
    waits = [0.5, 0.0]

    def __init__(self, recorder):
        super().__init__(recorder)
        self._waits = list(self.waits)

    def append(self, **kwargs):
        self.recorder.append(**kwargs)
        return self._waits.pop(0) if self._waits else 0.0


def _response(
    *,
    mode="follow",
    edge=None,
    terminal=None,
    delta=0.0,
    sample_s=None,
    boundary_s=None,
) -> dict:
    return {
        "mode": mode,
        "edge": edge,
        "terminal": terminal,
        "sample_monotonic_s": sample_s,
        "boundary_monotonic_s": boundary_s,
        "sides": {
            side: {
                "measured_q_rad": np.full(6, delta),
                "leader_delta_q_rad": np.full(6, delta),
                "leader_gripper_open_fraction": 0.5,
            }
            for side in mirror.SIDES
        },
    }


class X5PolicyMirrorLoopTest(unittest.TestCase):
    def test_raw_mode_atomically_commits_one_takeover_segment(self) -> None:
        events: list[str] = []

        class RawClient(_Client):
            def end(self):
                return {
                    "path": "/tmp/source.npz",
                    "sha256": "sha256:" + "1" * 64,
                    "sample_count": 20,
                    "segment_count": 1,
                    "frequency_hz": 100.0,
                }

        client = RawClient(
            [
                _response(),
                _response(mode="manual", edge="enter"),
                _response(mode="manual", delta=0.1),
                _response(mode="follow", edge="exit"),
                _response(),
                _response(terminal="save", boundary_s=12.0),
            ],
            events,
        )
        task = _TaskEnv(events)
        # Even if another task subsystem marks the episode ended immediately
        # after the manual action, raw mode must keep the same layout alive
        # until an explicit Right/Left terminal event arrives.
        task.is_episode_end = lambda: bool(task.actions)
        task.take_action_cnt = [7]
        task.task_name = "fill_pen_holder"
        task.eval_seed = 3
        task.env_seeds = [11]
        task.layout_cycle = 2
        task.policy_provenance = {
            "checkpoint_id": "RoboDojo-sim-arx_x5-joint-0/59999",
            "checkpoint_digest": "sha256:" + "7" * 64,
            "code_revision": "e" * 40,
            "dirty": False,
        }

        class Snapshotter:
            def __init__(self, task_env):
                self.task_env = task_env

            def capture(self, frame_index):
                return {"frame.index": np.asarray(frame_index)}

            def metadata(self):
                return {"replay_saved_layout": {"id": 11}}

        class Store:
            instance = None

            def __init__(self, root, target_episodes, identity):
                self.root = Path(root)
                self.target_episodes = target_episodes
                self.identity = identity
                self.completed_count = 0
                self.commits = []
                Store.instance = self

            @property
            def target_reached(self):
                return self.completed_count >= self.target_episodes

            def commit(self, *args):
                self.commits.append(args)
                self.completed_count += 1
                return self.root / "pending/episode_0000000"

        bundle_module = ModuleType("src.eval_client.x5_raw_bundle")
        bundle_module.RawBundleStore = Store
        snapshot_module = ModuleType("src.eval_client.sim_state_snapshot")
        snapshot_module.SimulatorStateSnapshotter = Snapshotter

        with mock.patch.object(
            mirror, "DualJointMirrorClient", return_value=client
        ), mock.patch.dict(
            sys.modules,
            {
                "src.eval_client.x5_raw_bundle": bundle_module,
                "src.eval_client.sim_state_snapshot": snapshot_module,
            },
        ), mock.patch.dict(
            mirror.os.environ,
            {
                "ROBODOJO_DUAL_MIRROR_PROFILE": "arx_x5_identity_joint_v1",
                "ROBODOJO_DUAL_MIRROR_RECORD": "0",
                "ROBODOJO_X5_RAW_CAPTURE": "1",
                "ROBODOJO_X5_RAW_ROOT": "/tmp/x5-raw-test",
                "ROBODOJO_X5_TARGET_EPISODES": "1",
                "ROBODOJO_REALTIME": "0",
            },
            clear=False,
        ):
            mirror.run_piperx_policy_leader_mirror_episode(
                task,
                _Model(),
                allow_intervention=True,
            )

        self.assertTrue(task.raw_collection_complete)
        self.assertEqual(Store.instance.completed_count, 1)
        self.assertEqual(len(Store.instance.commits), 1)
        fragment, _, takeover, terminal, sim_anchor, source_anchor, metadata = (
            Store.instance.commits[0]
        )
        self.assertEqual(fragment["segment_count"], 1)
        self.assertEqual(int(takeover["frame.index"]), 7)
        self.assertEqual(int(terminal["frame.index"]), 7)
        self.assertEqual(sim_anchor["left"]["q_rad"].shape, (6,))
        self.assertEqual(source_anchor["right"]["q_rad"].shape, (6,))
        self.assertEqual(metadata["layout_id"], 11)

    def test_recording_manual_frame_is_online_obs_t_action_t_before_next_obs(self) -> None:
        events: list[str] = []
        client = _Client(
            [
                _response(),
                _response(mode="manual", edge="enter"),
                _response(mode="manual", delta=0.1),
                _response(mode="manual", terminal="save"),
            ],
            events,
        )
        task = _TaskEnv(events)
        task.is_episode_end = lambda: False
        recorder = _Recorder(events)
        recorder_module = ModuleType("src.eval_client.lerobot_stream_recorder")
        recorder_module.recorder_for_env = lambda _task_env: recorder

        with mock.patch.object(
            mirror, "DualJointMirrorClient", return_value=client
        ), mock.patch.object(
            mirror, "_SinglePendingRecorder", _SynchronousPendingRecorder
        ), mock.patch.dict(
            sys.modules,
            {"src.eval_client.lerobot_stream_recorder": recorder_module},
        ), mock.patch.dict(
            mirror.os.environ,
            {
                "ROBODOJO_DUAL_MIRROR_PROFILE": "arx_x5_identity_joint_v1",
                "ROBODOJO_DUAL_MIRROR_RECORD": "1",
                "ROBODOJO_REALTIME": "0",
            },
        ):
            mirror.run_piperx_policy_leader_mirror_episode(
                task,
                _Model(),
                allow_intervention=True,
            )

        self.assertEqual(len(recorder.frames), 1)
        frame = recorder.frames[0]
        np.testing.assert_allclose(
            frame["obs"]["state"]["left_arm_joint_state"],
            np.zeros(6),
        )
        np.testing.assert_allclose(
            frame["executed_action"]["left_arm_joint_state"],
            np.full(6, 0.1),
        )
        self.assertEqual(
            frame["control"],
            {
                "action_source": "human",
                "intervention_mask": 1,
                "active_arm": "both",
                "takeover_edge": 1,
                "chunk_id": -1,
                "chunk_index": -1,
                "timestamp": frame["control"]["timestamp"],
            },
        )
        self.assertLess(events.index("record"), events.index("take_action"))
        self.assertGreater(events.index("get_obs", 1), events.index("take_action"))
        self.assertEqual(task.get_obs_count, 2)
        self.assertEqual(task.capture_updates, [])
        self.assertEqual(len(recorder.finishes), 1)
        finish = dict(recorder.finishes[0])
        self.assertGreater(finish.pop("timestamp_s"), 0.0)
        self.assertEqual(
            finish,
            {"accepted": True, "success": False, "reason": "operator_accept_next"},
        )

    def test_exit_boundary_wins_if_right_arrives_before_next_policy_frame(self) -> None:
        events: list[str] = []
        client = _Client(
            [
                _response(sample_s=10.0),
                _response(mode="manual", edge="enter", sample_s=10.6, boundary_s=10.5),
                _response(mode="manual", delta=0.1, sample_s=11.0),
                _response(mode="follow", edge="exit", sample_s=12.1, boundary_s=12.0),
                _response(sample_s=12.2),
                _response(terminal="save", sample_s=13.1, boundary_s=13.0),
            ],
            events,
        )
        task = _TaskEnv(events)
        task.is_episode_end = lambda: False
        recorder = _Recorder(events)
        recorder_module = ModuleType("src.eval_client.lerobot_stream_recorder")
        recorder_module.recorder_for_env = lambda _task_env: recorder

        with mock.patch.object(
            mirror, "DualJointMirrorClient", return_value=client
        ), mock.patch.object(
            mirror, "_SinglePendingRecorder", _SynchronousPendingRecorder
        ), mock.patch.dict(
            sys.modules,
            {"src.eval_client.lerobot_stream_recorder": recorder_module},
        ), mock.patch.dict(
            mirror.os.environ,
            {
                "ROBODOJO_DUAL_MIRROR_PROFILE": "arx_x5_identity_joint_v1",
                "ROBODOJO_DUAL_MIRROR_RECORD": "1",
                "ROBODOJO_REALTIME": "0",
            },
        ):
            mirror.run_piperx_policy_leader_mirror_episode(
                task,
                _Model(),
                allow_intervention=True,
            )

        self.assertEqual(recorder.finishes[0]["timestamp_s"], 12.0)

    def test_writer_ack_wait_is_removed_from_next_manual_sample_time(self) -> None:
        events: list[str] = []
        client = _Client(
            [
                _response(sample_s=99.0),
                _response(mode="manual", edge="enter", sample_s=99.9, boundary_s=99.8),
                _response(mode="manual", delta=0.1, sample_s=100.0),
                _response(mode="manual", delta=0.2, sample_s=100.75),
                _response(mode="manual", terminal="save", sample_s=101.1, boundary_s=101.0),
            ],
            events,
        )
        task = _TaskEnv(events)
        task.is_episode_end = lambda: False
        recorder = _Recorder(events)
        recorder_module = ModuleType("src.eval_client.lerobot_stream_recorder")
        recorder_module.recorder_for_env = lambda _task_env: recorder

        with mock.patch.object(
            mirror, "DualJointMirrorClient", return_value=client
        ), mock.patch.object(
            mirror, "_SinglePendingRecorder", _BlockingPendingRecorder
        ), mock.patch.dict(
            sys.modules,
            {"src.eval_client.lerobot_stream_recorder": recorder_module},
        ), mock.patch.dict(
            mirror.os.environ,
            {
                "ROBODOJO_DUAL_MIRROR_PROFILE": "arx_x5_identity_joint_v1",
                "ROBODOJO_DUAL_MIRROR_RECORD": "1",
                "ROBODOJO_REALTIME": "0",
            },
        ):
            mirror.run_piperx_policy_leader_mirror_episode(
                task,
                _Model(),
                allow_intervention=True,
            )

        self.assertEqual(
            [frame["control"]["timestamp"] for frame in recorder.frames],
            [100.0, 100.25],
        )
        self.assertEqual(recorder.finishes[0]["timestamp_s"], 100.5)

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
