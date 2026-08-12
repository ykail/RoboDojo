from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from src.eval_client.dual_joint_collection import (
    DUAL_MIRROR_PROTOCOL,
    PIPERX_CONTROL_MODE,
    X5_CONTROL_MODE,
    is_live_dual_control_mode,
    live_dual_mode_spec,
)
from src.eval_client import piperx_dual_joint_mirror as mirror
from src.eval_client.lerobot_stream_recorder import _task_metadata
from tests.test_x5_policy_mirror_loop import (
    _Client,
    _Model,
    _Recorder,
    _SynchronousPendingRecorder,
    _TaskEnv,
    _response,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/RoboDojo"


class PiperXOnlineDaggerIntegrationTest(unittest.TestCase):
    def test_mode_contract_keeps_x5_and_piper_profiles_distinct(self) -> None:
        self.assertTrue(is_live_dual_control_mode(X5_CONTROL_MODE))
        self.assertTrue(is_live_dual_control_mode(PIPERX_CONTROL_MODE))
        self.assertEqual(
            live_dual_mode_spec(PIPERX_CONTROL_MODE).profile,
            "arx_x5_piperx_relative_joint_v1",
        )
        self.assertEqual(
            live_dual_mode_spec(X5_CONTROL_MODE).profile,
            "arx_x5_identity_joint_v1",
        )

    def test_follow_only_cp11_still_uses_the_piper_profile(self) -> None:
        events: list[str] = []
        task = _TaskEnv(events)
        task.control_mode = "piperx_policy_leader_mirror"
        client = _Client([_response(), _response(), _response()], events)
        with mock.patch.object(
            mirror, "DualJointMirrorClient", return_value=client
        ), mock.patch.dict(
            mirror.os.environ,
            {
                "ROBODOJO_DUAL_MIRROR_PROFILE": "arx_x5_piperx_relative_joint_v1",
                "ROBODOJO_DUAL_MIRROR_RECORD": "0",
                "ROBODOJO_X5_RAW_CAPTURE": "0",
                "ROBODOJO_REALTIME": "0",
            },
            clear=True,
        ):
            mirror.run_piperx_policy_leader_mirror_episode(
                task, _Model(), allow_intervention=False
            )
        self.assertEqual(len(task.actions), 1)

    def test_piper_target_episodes_prefers_generic_and_rejects_conflict(self) -> None:
        with mock.patch.dict(
            mirror.os.environ,
            {"ROBODOJO_DUAL_MIRROR_TARGET_EPISODES": "34"},
            clear=True,
        ):
            self.assertEqual(mirror._online_target_episodes(PIPERX_CONTROL_MODE), 34)
        with mock.patch.dict(
            mirror.os.environ,
            {
                "ROBODOJO_DUAL_MIRROR_TARGET_EPISODES": "34",
                "ROBODOJO_PIPERX_TARGET_EPISODES": "35",
            },
            clear=True,
        ), self.assertRaisesRegex(mirror.DualJointMirrorError, "conflicting"):
            mirror._online_target_episodes(PIPERX_CONTROL_MODE)

    @staticmethod
    def _identity_task() -> SimpleNamespace:
        return SimpleNamespace(
            control_mode=PIPERX_CONTROL_MODE,
            task_name="make_toast",
            policy_provenance={
                "checkpoint_id": "RoboDojo-sim-arx_x5-joint-0/59999",
                "checkpoint_digest": "sha256:" + "a" * 64,
                "code_revision": "b" * 40,
                "dirty": False,
            },
        )

    def test_piper_resume_accepts_only_contiguous_same_mode_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "lerobot"
            dataset = root / "dataset"
            sidecars = dataset / "meta/robodojo/episodes"
            sidecars.mkdir(parents=True)
            (dataset / "meta/info.json").write_text(
                json.dumps({"total_episodes": 1}), encoding="utf-8"
            )
            task = self._identity_task()
            sidecar = sidecars / "episode_0000000.json"
            payload = {
                "episode_index": 0,
                "robodojo_timing_resample": "sim_step_exact_25hz_v1",
                "robodojo_task": task.task_name,
                "robodojo_control_mode": PIPERX_CONTROL_MODE,
                "robodojo_policy_provenance": task.policy_provenance,
                "robodojo_layout_id": 7,
                "robodojo_layout_cycle": 3,
            }
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            environment = {
                "ROBODOJO_LEROBOT_ROOT": str(root),
                "ROBODOJO_LEROBOT_REPO_ID": "dataset",
            }
            with mock.patch.dict(mirror.os.environ, environment, clear=True):
                count, last = mirror._online_collection_state(task)
            self.assertEqual(count, 1)
            self.assertEqual(last["robodojo_layout_id"], 7)

            payload["robodojo_control_mode"] = X5_CONTROL_MODE
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.dict(
                mirror.os.environ, environment, clear=True
            ), self.assertRaisesRegex(mirror.DualJointMirrorError, "control-mode mismatch"):
                mirror._online_collection_state(task)

    def test_two_i_pairs_remain_one_candidate_until_right_commits(self) -> None:
        events: list[str] = []
        client = _Client(
            [
                _response(),
                _response(mode="manual", edge="enter"),
                _response(mode="manual", delta=0.10),
                _response(mode="follow", edge="exit"),
                _response(),
                _response(mode="manual", edge="enter"),
                _response(mode="manual", delta=0.20),
                _response(mode="follow", edge="exit"),
                _response(),
                _response(),
                _response(),
                _response(terminal="save"),
            ],
            events,
        )
        task = _TaskEnv(events)
        task.control_mode = PIPERX_CONTROL_MODE
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
                "ROBODOJO_DUAL_MIRROR_PROFILE": "arx_x5_piperx_relative_joint_v1",
                "ROBODOJO_DUAL_MIRROR_RECORD": "1",
                "ROBODOJO_X5_RAW_CAPTURE": "0",
                "ROBODOJO_DUAL_MIRROR_TARGET_EPISODES": "",
                "ROBODOJO_REALTIME": "0",
            },
            clear=True,
        ):
            mirror.run_piperx_policy_leader_mirror_episode(
                task, _Model(), allow_intervention=True
            )

        self.assertEqual(
            [frame["control"]["action_source"] for frame in recorder.frames],
            ["human", "human", "policy"],
        )
        self.assertEqual(len(recorder.finishes), 1)
        self.assertEqual(recorder.finishes[0]["reason"], "operator_accept_next")
        self.assertEqual(sum(event == "take_action" for event in events), 3)

    def test_left_discards_piper_candidate_for_same_layout(self) -> None:
        events: list[str] = []
        client = _Client([_response(), _response(terminal="retry")], events)
        task = _TaskEnv(events)
        task.control_mode = PIPERX_CONTROL_MODE
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
                "ROBODOJO_DUAL_MIRROR_PROFILE": "arx_x5_piperx_relative_joint_v1",
                "ROBODOJO_DUAL_MIRROR_RECORD": "1",
                "ROBODOJO_X5_RAW_CAPTURE": "0",
                "ROBODOJO_REALTIME": "0",
            },
            clear=True,
        ):
            from src.eval_client.intervention_loop import InterventionRejected

            with self.assertRaises(InterventionRejected):
                mirror.run_piperx_policy_leader_mirror_episode(
                    task, _Model(), allow_intervention=True
                )
        self.assertEqual(
            recorder.finishes,
            [{"accepted": False, "success": False, "reason": "operator_discard_retry"}],
        )
        self.assertEqual((task.success, task.end_flag), ([False], [True]))

    def test_piper_writer_metadata_attests_common_protocol_and_bridge(self) -> None:
        task = SimpleNamespace(
            task_name="make_toast",
            config_name="arx_x5",
            env_seeds=[5],
            layout_cycle=2,
            eval_seed=0,
            policy_name="Pi_05",
            additional_info="RoboDojo-sim-arx_x5-joint-0/59999",
            policy_runtime="robodojo_policy_v1",
            policy_provenance={"checkpoint_id": "RoboDojo-sim-arx_x5-joint-0/59999"},
            control_mode=PIPERX_CONTROL_MODE,
        )
        with mock.patch.dict(
            os.environ,
            {
                "ROBODOJO_DUAL_MIRROR_RECORD": "1",
                "ROBODOJO_PIPERX_CODE_ROOT": "/tmp/piperx-code",
                "ROBODOJO_LEROBOT_TIMING_CONTRACT": "sim_step_exact_25hz_v1",
            },
            clear=False,
        ), mock.patch(
            "src.eval_client.lerobot_stream_recorder._git_revision",
            return_value="1" * 40,
        ), mock.patch(
            "src.eval_client.lerobot_stream_recorder._git_dirty",
            return_value=False,
        ):
            metadata = _task_metadata(task)

        self.assertEqual(metadata["piperx_bridge_protocol"], DUAL_MIRROR_PROTOCOL)
        self.assertEqual(metadata["hardware_bridge_protocol"], DUAL_MIRROR_PROTOCOL)
        self.assertEqual(metadata["hardware_embodiment"], "piper_x")
        self.assertEqual(
            metadata["hardware_profile"], "arx_x5_piperx_relative_joint_v1"
        )
        self.assertEqual(metadata["timing_contract"], "sim_step_exact_25hz_v1")

    def test_hoo_launchers_are_foreground_and_pin_online_contract(self) -> None:
        launchers = (
            SCRIPTS / "run_piperx_dagger_isaac.sh",
            SCRIPTS / "run_hoo_piperx_isaac.sh",
            SCRIPTS / "run_hoo_piperx_policy_tunnel.sh",
            SCRIPTS / "run_yikai_policy_59999.sh",
        )
        for launcher in launchers:
            with self.subTest(launcher=launcher.name):
                self.assertTrue(launcher.is_file())
                self.assertTrue(os.access(launcher, os.X_OK))
                result = subprocess.run(
                    ["bash", "-n", str(launcher)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("tmux", launcher.read_text(encoding="utf-8").lower())

        isaac = launchers[0].read_text(encoding="utf-8")
        for required in (
            "--control-mode piperx_policy_joint_intervention",
            "robodojo_dual_joint_mirror_v1",
            "arx_x5_piperx_relative_joint_v1",
            'ROBODOJO_LEROBOT_TIMING_CONTRACT="sim_step_exact_25hz_v1"',
            "ROBODOJO_DUAL_MIRROR_TARGET_EPISODES",
            "ROBODOJO_X5_RAW_CAPTURE=0",
            "/home/hoo/piper_x/lerobot_sealab-piperx-online-dagger-v2",
            "one Right commits one complete policy+human episode",
            "Left retries the same layout",
        ):
            with self.subTest(required=required):
                self.assertIn(required, isaac)

        tunnel = launchers[2].read_text(encoding="utf-8")
        self.assertIn("ExitOnForwardFailure=yes", tunnel)
        self.assertIn('YIKAI_TARGET="${YIKAI_SSH_TARGET:-yikai}"', tunnel)

        wrapper = launchers[1].read_text(encoding="utf-8")
        self.assertIn("--expected-code-revision", wrapper)
        self.assertIn("--expected-checkpoint-digest", wrapper)

        yikai = launchers[3].read_text(encoding="utf-8")
        self.assertIn("ecc1a7451c3156b1e5f7533851dbb0222896206f", yikai)
        self.assertIn("assets/arx_x5_sim/norm_stats.json", yikai)

    def test_main_gives_piper_the_same_soft_reset_lifecycle_as_x5(self) -> None:
        main_source = (ROOT / "src/eval_client/main.py").read_text(encoding="utf-8")
        eval_source = (ROOT / "src/eval_client/eval_env.py").read_text(encoding="utf-8")
        self.assertIn('"piperx_policy_joint_intervention",\n        "x5_policy_joint_intervention"', main_source)
        self.assertGreaterEqual(main_source.count("if not live_dual_collection:"), 3)
        self.assertIn("is_live_dual_control_mode(self.control_mode)", eval_source)
        self.assertIn('"piperx_policy_joint_intervention",', eval_source)


if __name__ == "__main__":
    unittest.main()
