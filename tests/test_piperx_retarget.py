from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from src.eval_client.piperx_bridge_client import (
    OperatorArmSample,
    OperatorSample,
    SimArmTarget,
    SimTargets,
)
from src.eval_client.piperx_retarget import (
    ArmRetargetConfig,
    ArxPiperXRetargetController,
    RelativeSE3Retargeter,
    RetargetConfig,
)


def _arm_config(**overrides):
    values = {
        "leader_to_sim_rotation_qwxyz": (1.0, 0.0, 0.0, 0.0),
        "translation_scale": (1.0, 1.0, 1.0),
        "gripper_closed_m": 0.0,
        "gripper_open_m": 0.07,
        "max_translation_from_anchor_m": 0.25,
        "max_rotation_from_anchor_rad": 1.57,
    }
    values.update(overrides)
    return ArmRetargetConfig(**values)


def _operator(*, left_pose, right_pose=None, left_gripper=0.02, right_gripper=0.03):
    return OperatorSample(
        generation=1,
        seq=4,
        mode="intervention",
        edge="enter",
        terminal_request=None,
        left=OperatorArmSample(tuple(left_pose), left_gripper),
        right=OperatorArmSample(tuple(right_pose or left_pose), right_gripper),
        mirror_accepted=True,
        health={"ok": True},
        diagnostics={},
    )


class RelativeSE3RetargeterTest(unittest.TestCase):
    def test_calibration_requires_explicit_true_safety_gate(self):
        arm = {
            "leader_to_sim_rotation_qwxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_scale": [1.0, 1.0, 1.0],
            "gripper_closed_m": 0.0,
            "gripper_open_m": 0.07,
            "max_translation_from_anchor_m": 0.25,
            "max_rotation_from_anchor_rad": 1.57,
        }
        value = {
            "schema": "robodojo_piperx_retarget_v1",
            "calibrated": False,
            "left": arm,
            "right": dict(arm),
            "max_joint_delta_rad": 0.35,
        }
        with self.assertRaisesRegex(ValueError, "calibrated=true"):
            RetargetConfig.from_dict(value)
        value["calibrated"] = True
        parsed = RetargetConfig.from_dict(value)
        self.assertEqual(parsed.max_joint_delta_rad, 0.35)

    def test_calibration_file_rejects_duplicate_keys_and_nonfinite_values(self):
        arm = (
            '{"leader_to_sim_rotation_qwxyz":[1,0,0,0],'
            '"translation_scale":[1,1,1],"gripper_closed_m":0,'
            '"gripper_open_m":0.07,"max_translation_from_anchor_m":0.25,'
            '"max_rotation_from_anchor_rad":1.57}'
        )
        documents = {
            "duplicate": (
                '{"schema":"robodojo_piperx_retarget_v1","calibrated":false,'
                f'"calibrated":true,"left":{arm},"right":{arm},'
                '"max_joint_delta_rad":0.35}'
            ),
            "nonfinite": (
                '{"schema":"robodojo_piperx_retarget_v1","calibrated":true,'
                f'"left":{arm},"right":{arm},"max_joint_delta_rad":NaN}}'
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            for label, document in documents.items():
                with self.subTest(label=label):
                    path = Path(temporary) / f"{label}.json"
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "duplicate|non-finite"):
                        RetargetConfig.from_file(path)

    def test_calibration_rejects_boolean_numeric_fields(self):
        arm = {
            "leader_to_sim_rotation_qwxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_scale": [1.0, 1.0, 1.0],
            "gripper_closed_m": 0.0,
            "gripper_open_m": 0.07,
            "max_translation_from_anchor_m": 0.25,
            "max_rotation_from_anchor_rad": 1.57,
        }
        value = {
            "schema": "robodojo_piperx_retarget_v1",
            "calibrated": True,
            "left": arm,
            "right": dict(arm),
            "max_joint_delta_rad": 0.35,
        }
        cases = (
            ("array", {**value, "left": {**arm, "translation_scale": [True, 1.0, 1.0]}}),
            ("numeric_string", {**value, "left": {**arm, "translation_scale": ["1", 1.0, 1.0]}}),
            ("scalar", {**value, "left": {**arm, "gripper_open_m": True}}),
            ("joint_limit", {**value, "max_joint_delta_rad": True}),
        )
        for label, candidate in cases:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "number|numeric"):
                RetargetConfig.from_dict(candidate)

    def test_takeover_first_sample_preserves_pose_and_gripper(self):
        retargeter = RelativeSE3Retargeter(_arm_config())
        leader = [0.4, -0.1, 0.2, 1.0, 0.0, 0.0, 0.0]
        sim = [-0.25, 0.3, 0.85, 1.0, 0.0, 0.0, 0.0]
        retargeter.anchor(
            leader,
            sim,
            leader_gripper_m=0.021,
            sim_gripper=0.37,
        )

        np.testing.assert_allclose(retargeter.map_pose(leader), sim, atol=1e-9)
        self.assertAlmostEqual(retargeter.map_gripper(0.021), 0.37)

    def test_relative_translation_and_gripper_delta_are_scaled_from_anchor(self):
        retargeter = RelativeSE3Retargeter(_arm_config(translation_scale=(2.0, 0.5, 1.0)))
        retargeter.anchor(
            [0, 0, 0, 1, 0, 0, 0],
            [0.1, 0.2, 0.3, 1, 0, 0, 0],
            leader_gripper_m=0.02,
            sim_gripper=0.4,
        )

        mapped = retargeter.map_pose([0.01, 0.02, -0.03, 1, 0, 0, 0])
        np.testing.assert_allclose(mapped[:3], [0.12, 0.21, 0.27])
        self.assertAlmostEqual(retargeter.map_gripper(0.027), 0.5)


class ArxPiperXRetargetControllerTest(unittest.TestCase):
    class _Robot:
        type = "target"

        def __init__(self, arm):
            self.arm_name = f"{arm}_arm"

    class _RobotManager:
        def __init__(self, status="Success"):
            self.robot_list = [
                ArxPiperXRetargetControllerTest._Robot("left"),
                ArxPiperXRetargetControllerTest._Robot("right"),
            ]
            self.status = status
            self.targets = []

        def solve_ik(self, target_pose, env_idx, robot):
            self.targets.append((robot.arm_name, list(target_pose), env_idx))
            value = 0.12 if robot.arm_name.startswith("left") else -0.12
            return {"status": self.status, "joint_value": np.full(6, value)}

    @staticmethod
    def _obs():
        return {
            "state": {
                "left_arm_joint_state": np.full(6, 0.1),
                "left_ee_joint_state": np.asarray([0.3]),
                "right_arm_joint_state": np.full(6, -0.1),
                "right_ee_joint_state": np.asarray([0.6]),
            }
        }

    def _controller(self, status="Success"):
        manager = self._RobotManager(status=status)
        env = SimpleNamespace(robot_manager=manager)
        config = RetargetConfig(left=_arm_config(), right=_arm_config())
        return ArxPiperXRetargetController(env, config), manager

    def test_dual_arm_sample_produces_one_full_arx_action(self):
        controller, manager = self._controller()
        sim = SimTargets(
            left=SimArmTarget((-0.2, 0.0, 0.8, 1, 0, 0, 0), 0.3),
            right=SimArmTarget((0.2, 0.0, 0.8, 1, 0, 0, 0), 0.6),
        )
        initial = _operator(
            left_pose=(0.1, 0, 0.4, 1, 0, 0, 0),
            right_pose=(-0.1, 0, 0.4, 1, 0, 0, 0),
        )
        controller.enter(initial, sim)
        moved = _operator(
            left_pose=(0.11, 0, 0.4, 1, 0, 0, 0),
            right_pose=(-0.1, 0.01, 0.4, 1, 0, 0, 0),
            left_gripper=0.027,
            right_gripper=0.023,
        )

        action, metadata = controller.build_action(self._obs(), moved)

        self.assertEqual(
            set(action),
            {
                "left_arm_joint_state",
                "left_ee_joint_state",
                "right_arm_joint_state",
                "right_ee_joint_state",
            },
        )
        np.testing.assert_allclose(action["left_arm_joint_state"], 0.12)
        np.testing.assert_allclose(action["right_arm_joint_state"], -0.12)
        self.assertAlmostEqual(action["left_ee_joint_state"][0], 0.4)
        self.assertAlmostEqual(action["right_ee_joint_state"][0], 0.5)
        self.assertEqual(metadata["intervention_mask"], 1)
        self.assertEqual(len(manager.targets), 2)

    def test_any_ik_failure_returns_exact_dual_arm_hold(self):
        controller, _ = self._controller(status="Failure")
        sim = SimTargets(
            left=SimArmTarget((-0.2, 0.0, 0.8, 1, 0, 0, 0), 0.3),
            right=SimArmTarget((0.2, 0.0, 0.8, 1, 0, 0, 0), 0.6),
        )
        sample = _operator(left_pose=(0.1, 0, 0.4, 1, 0, 0, 0))
        controller.enter(sample, sim)

        action, metadata = controller.build_action(self._obs(), sample)

        np.testing.assert_allclose(action["left_arm_joint_state"], 0.1)
        np.testing.assert_allclose(action["right_arm_joint_state"], -0.1)
        self.assertEqual(metadata["ik_success"], 0)
        self.assertEqual(metadata["intervention_mask"], 0)


if __name__ == "__main__":
    unittest.main()
