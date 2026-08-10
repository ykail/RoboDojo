from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np


def _load_module():
    src = ModuleType("src")
    src.__path__ = []
    eval_client = ModuleType("src.eval_client")
    eval_client.__path__ = []
    intervention_loop = ModuleType("src.eval_client.intervention_loop")

    class RealtimePacer:
        def __init__(self, *args, **kwargs):
            pass

    intervention_loop.RealtimePacer = RealtimePacer
    joint_j1 = ModuleType("src.eval_client.piperx_joint_j1")
    joint_j1._format_degrees = lambda value: str(value)
    joint_j1._hold_action = lambda obs: dict(obs.get("action", {}))
    injected = {
        "src": src,
        "src.eval_client": eval_client,
        "src.eval_client.intervention_loop": intervention_loop,
        "src.eval_client.piperx_joint_j1": joint_j1,
    }
    name = "src.eval_client.piperx_dual_joint_mirror"
    source_path = Path(__file__).with_name("piperx_dual_joint_mirror.py")
    if not source_path.is_file():
        source_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "eval_client"
            / "piperx_dual_joint_mirror.py"
        )
    previous = {key: sys.modules.get(key) for key in (*injected, name)}
    try:
        sys.modules.update(injected)
        spec = importlib.util.spec_from_file_location(name, source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


MODULE = _load_module()


class _Recorder:
    def __init__(self, *, fail_on: int | None = None):
        self.frames = []
        self.fail_on = fail_on
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1
        return 0.012

    def append(self, **kwargs):
        if self.fail_on == len(self.frames) + 1:
            raise RuntimeError("writer failed")
        self.frames.append(kwargs)
        return 0.0


class _TaskEnv:
    def __init__(self):
        self.marker = -1
        self.get_obs_count = 0
        self.camera_pipeline = [-101, -102]
        self.reward_manager = SimpleNamespace(
            score_completed_count=[0],
            final_score_completed_count=[0],
        )
        self.manager = SimpleNamespace(
            updates_enabled=True,
            calls=[],
        )

        def set_updates_enabled(enabled):
            self.manager.updates_enabled = bool(enabled)
            self.manager.calls.append(bool(enabled))

        self.manager.set_updates_enabled = set_updates_enabled
        self.capture_manager = self.manager

    def get_obs(self):
        self.get_obs_count += 1
        vision_marker = self.camera_pipeline.pop(0)
        self.camera_pipeline.append(self.marker)
        return {
            "state": {"marker": np.asarray([self.marker], dtype=np.float32)},
            "vision": {
                "camera": {
                    "color": np.full((1, 1, 3), vision_marker, dtype=np.int16)
                }
            },
            "instruction": "make toast",
        }


class DeferredQualityTest(unittest.TestCase):
    def setUp(self):
        self.previous_restore_module = sys.modules.get(
            "src.eval_client.sim_state_restore"
        )
        restore_module = ModuleType("src.eval_client.sim_state_restore")

        def restore_replay_frame(task_env, replay, **kwargs):
            task_env.marker = int(np.asarray(replay.state["marker"]).item())
            for key, attribute in (
                ("reward.score_completed_count", "score_completed_count"),
                (
                    "reward.final_score_completed_count",
                    "final_score_completed_count",
                ),
            ):
                if key in replay.state:
                    getattr(task_env.reward_manager, attribute)[0] = int(
                        np.asarray(replay.state[key]).item()
                    )
            self.restore_calls.append((task_env.marker, dict(kwargs)))

        restore_module.restore_replay_frame = restore_replay_frame
        sys.modules["src.eval_client.sim_state_restore"] = restore_module
        self.restore_calls = []

    def tearDown(self):
        if self.previous_restore_module is None:
            sys.modules.pop("src.eval_client.sim_state_restore", None)
        else:
            sys.modules[
                "src.eval_client.sim_state_restore"
            ] = self.previous_restore_module

    @staticmethod
    def frame(marker, source):
        return MODULE._DeferredQualityFrame(
            sim_state={
                "marker": np.asarray(marker, dtype=np.int64),
                "reward.score_completed_count": np.asarray(marker, dtype=np.int64),
                "reward.final_score_completed_count": np.asarray(marker + 10, dtype=np.int64),
            },
            policy_action=None if source == "human" else {"a": marker},
            executed_action={"a": marker},
            control={"action_source": source, "intervention_mask": int(source == "human")},
        )

    def test_ordered_render_and_terminal_restore(self):
        task_env = _TaskEnv()
        recorder = _Recorder()
        snapshotter = SimpleNamespace(manifest={"profile": "test"})
        terminal = {
            "marker": np.asarray(9, dtype=np.int64),
            "reward.score_completed_count": np.asarray(9, dtype=np.int64),
            "reward.final_score_completed_count": np.asarray(19, dtype=np.int64),
        }

        obs = MODULE._render_deferred_quality_segment(
            task_env,
            recorder,
            snapshotter,
            [
                self.frame(1, "human"),
                self.frame(2, "human"),
                self.frame(3, "policy"),
                self.frame(4, "policy"),
            ],
            terminal,
            checkpoint_label="CP13",
        )

        self.assertEqual(recorder.wait_count, 1)
        self.assertEqual(
            [int(frame["obs"]["state"]["marker"][0]) for frame in recorder.frames],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            [
                int(frame["obs"]["vision"]["camera"]["color"][0, 0, 0])
                for frame in recorder.frames
            ],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            [frame["control"]["action_source"] for frame in recorder.frames],
            ["human", "human", "policy", "policy"],
        )
        self.assertEqual(task_env.marker, 9)
        self.assertEqual(int(obs["state"]["marker"][0]), 9)
        self.assertEqual(task_env.reward_manager.score_completed_count[0], 9)
        self.assertEqual(task_env.reward_manager.final_score_completed_count[0], 19)
        self.assertEqual(task_env.get_obs_count, 15)
        self.assertTrue(
            all(
                call == {
                    "_prime_camera_pipeline": False,
                    "_refresh_camera_pipeline": False,
                    "_reset_episode": False,
                }
                for _, call in self.restore_calls
            )
        )

    def test_writer_error_still_restores_terminal(self):
        task_env = _TaskEnv()
        recorder = _Recorder(fail_on=2)
        snapshotter = SimpleNamespace(manifest={"profile": "test"})
        terminal = {"marker": np.asarray(7, dtype=np.int64)}

        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            MODULE._render_deferred_quality_segment(
                task_env,
                recorder,
                snapshotter,
                [self.frame(3, "human"), self.frame(4, "policy")],
                terminal,
                checkpoint_label="CP13",
            )

        self.assertEqual(task_env.marker, 7)
        self.assertEqual(self.restore_calls[-1][0], 7)

    def test_camera_pause_resume_adapter(self):
        task_env = _TaskEnv()
        MODULE._set_data_camera_updates(task_env, False, label="CP13")
        MODULE._set_data_camera_updates(task_env, True, label="CP13")
        self.assertEqual(task_env.manager.calls, [False, True])
        self.assertTrue(task_env.manager.updates_enabled)


if __name__ == "__main__":
    unittest.main()
