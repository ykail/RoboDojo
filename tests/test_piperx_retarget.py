from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from src.eval_client.piperx_bridge_client import (
    EMBODIMENT_PROFILE,
    ManualSample,
    OperatorArmSample,
    OperatorSample,
    SimArmTarget,
    SimTargets,
)
from src.eval_client.piperx_retarget import (
    PIPERX_GRIPPER_STROKE_M,
    ArxPiperXRetargetController,
    PiperXRetargetError,
    RelativeSE3Retargeter,
    RetargetConfig,
)

_TOPOLOGY = "policy_sim_to_follower_to_leader_manual_leader_joint_fanout"


def _operator(
    *,
    left_pose,
    right_pose=None,
    left_gripper: float = 0.02,
    right_gripper: float = 0.03,
    sample_id: int = 1,
    generation: int = 1,
) -> OperatorSample:
    return OperatorSample(
        generation=generation,
        seq=sample_id + 10,
        mode="intervention",
        control_topology=_TOPOLOGY,
        embodiment_profile=EMBODIMENT_PROFILE,
        leader_actuation_mode="native_leader",
        follower_actuation_mode="hold",
        transition=None,
        edge="enter" if sample_id == 1 else None,
        terminal_request=None,
        manual_sample=ManualSample(
            sample_id=sample_id,
            left=OperatorArmSample(tuple(left_pose), left_gripper, 1000 + sample_id),
            right=OperatorArmSample(
                tuple(right_pose if right_pose is not None else left_pose),
                right_gripper,
                2000 + sample_id,
            ),
        ),
        manual_resolution=None,
        motion_accepted=False,
        health={"ok": True},
        diagnostics={},
    )


def _sim() -> SimTargets:
    return SimTargets(
        left=SimArmTarget((-0.2, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0), 0.3),
        right=SimArmTarget((0.2, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0), 0.6),
    )


class RelativeSE3RetargeterTest(unittest.TestCase):
    def test_default_profile_needs_no_external_calibration(self):
        config = RetargetConfig()
        self.assertEqual(config.embodiment_profile, EMBODIMENT_PROFILE)
        self.assertFalse(hasattr(RetargetConfig, "from_file"))
        self.assertFalse(hasattr(RetargetConfig, "from_environment"))

    def test_profile_and_safety_fields_are_strict(self):
        with self.assertRaisesRegex(ValueError, "unsupported.*profile"):
            RetargetConfig(embodiment_profile="site-specific-calibration")
        for field in (
            "max_translation_from_anchor_m",
            "max_rotation_from_anchor_rad",
            "max_joint_delta_rad",
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "finite number"):
                RetargetConfig(**{field: True})
            with self.subTest(field=f"{field}-zero"), self.assertRaisesRegex(ValueError, "positive"):
                RetargetConfig(**{field: 0.0})

    def test_zero_delta_preserves_pose_orientation_and_gripper_exactly(self):
        retargeter = RelativeSE3Retargeter(RetargetConfig())
        leader = [0.4, -0.1, 0.2, 0.9238795325, 0.0, 0.3826834324, 0.0]
        sim = [-0.25, 0.3, 0.85, 0.9659258263, 0.2588190451, 0.0, 0.0]
        retargeter.anchor(
            leader,
            sim,
            leader_gripper_m=0.021,
            sim_gripper=0.37,
        )

        np.testing.assert_allclose(retargeter.map_pose(leader), sim, atol=1e-9)
        self.assertAlmostEqual(retargeter.map_gripper(0.021), 0.37)

    def test_relative_metric_delta_and_official_gripper_stroke_are_used(self):
        retargeter = RelativeSE3Retargeter(RetargetConfig())
        retargeter.anchor(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            leader_gripper_m=0.02,
            sim_gripper=0.4,
        )

        mapped = retargeter.map_pose([0.01, 0.02, -0.03, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(mapped[:3], [0.11, 0.22, 0.27])
        self.assertAlmostEqual(PIPERX_GRIPPER_STROKE_M, 0.102)
        self.assertAlmostEqual(retargeter.map_gripper(0.0302), 0.5)

    def test_workspace_gate_rejects_before_ik(self):
        retargeter = RelativeSE3Retargeter(
            RetargetConfig(max_translation_from_anchor_m=0.01)
        )
        retargeter.anchor(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            leader_gripper_m=0.02,
            sim_gripper=0.4,
        )
        with self.assertRaisesRegex(PiperXRetargetError, "translation"):
            retargeter.map_pose([0.02, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


class ArxPiperXRetargetControllerTest(unittest.TestCase):
    class _Robot:
        type = "target"

        def __init__(self, arm: str):
            self.arm_name = f"{arm}_arm"

    class _RobotManager:
        def __init__(self, candidates=None, failures=()):
            self.robot_list = [
                ArxPiperXRetargetControllerTest._Robot("left"),
                ArxPiperXRetargetControllerTest._Robot("right"),
            ]
            self.candidates = candidates or {
                "left_arm": np.full(6, 0.12),
                "right_arm": np.full(6, -0.12),
            }
            self.failures = set(failures)
            self.targets: list[tuple[str, list[float], int]] = []

        def solve_ik(self, target_pose, env_idx, robot):
            self.targets.append((robot.arm_name, list(target_pose), env_idx))
            return {
                "status": "Failure" if robot.arm_name in self.failures else "Success",
                "joint_value": np.asarray(self.candidates[robot.arm_name]),
            }

    @staticmethod
    def _obs(*, left: float = 0.1, right: float = -0.1):
        return {
            "state": {
                "left_arm_joint_state": np.full(6, left),
                "left_ee_joint_state": np.asarray([0.3]),
                "right_arm_joint_state": np.full(6, right),
                "right_ee_joint_state": np.asarray([0.6]),
            }
        }

    def _controller(self, *, candidates=None, failures=(), config=None):
        manager = self._RobotManager(candidates=candidates, failures=failures)
        env = SimpleNamespace(robot_manager=manager)
        return ArxPiperXRetargetController(env, config), manager

    def test_default_adapter_builds_one_complete_dual_arm_action(self):
        controller, manager = self._controller()
        anchor = _operator(
            left_pose=(0.1, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0),
            right_pose=(-0.1, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0),
        )
        controller.enter(anchor, _sim())
        moved = _operator(
            sample_id=2,
            left_pose=(0.11, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0),
            right_pose=(-0.1, 0.01, 0.4, 1.0, 0.0, 0.0, 0.0),
            left_gripper=0.0302,
            right_gripper=0.0198,
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
        self.assertEqual(
            [(name, env_idx) for name, _, env_idx in manager.targets],
            [("left_arm", 0), ("right_arm", 0)],
        )

    def test_zero_delta_targets_the_current_sim_pose(self):
        current = {
            "left_arm": np.full(6, 0.1),
            "right_arm": np.full(6, -0.1),
        }
        controller, manager = self._controller(candidates=current)
        anchor = _operator(
            left_pose=(0.1, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0),
            right_pose=(-0.1, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0),
        )
        sim = _sim()
        controller.enter(anchor, sim)

        action, metadata = controller.build_action(self._obs(), anchor)

        np.testing.assert_allclose(manager.targets[0][1], sim.left.pose)
        np.testing.assert_allclose(manager.targets[1][1], sim.right.pose)
        np.testing.assert_allclose(action["left_arm_joint_state"], 0.1)
        np.testing.assert_allclose(action["right_arm_joint_state"], -0.1)
        self.assertEqual(metadata["ik_success"], 1)

    def test_joint_step_gate_is_relative_to_current_q_not_to_zero(self):
        current_left = 1.0
        current_right = -1.0
        candidates = {
            "left_arm": np.full(6, 1.30),
            "right_arm": np.full(6, -1.30),
        }
        controller, _ = self._controller(candidates=candidates)
        sample = _operator(
            left_pose=(0.1, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0),
            right_pose=(-0.1, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0),
        )
        controller.enter(sample, _sim())

        action, metadata = controller.build_action(
            self._obs(left=current_left, right=current_right),
            sample,
        )

        # These absolute values exceed the 0.35-rad gate, but each is only a
        # 0.30-rad step from the currently executed simulator joints.
        np.testing.assert_allclose(action["left_arm_joint_state"], 1.30)
        np.testing.assert_allclose(action["right_arm_joint_state"], -1.30)
        self.assertEqual(metadata["ik_success"], 1)

    def test_any_arm_ik_or_step_failure_returns_the_exact_dual_arm_hold(self):
        cases = {
            "right-ik": ({
                "left_arm": np.full(6, 0.12),
                "right_arm": np.full(6, -0.12),
            }, {"right_arm"}),
            "left-step": ({
                "left_arm": np.full(6, 0.46),
                "right_arm": np.full(6, -0.12),
            }, set()),
        }
        for label, (candidates, failures) in cases.items():
            with self.subTest(label=label):
                controller, _ = self._controller(candidates=candidates, failures=failures)
                sample = _operator(
                    left_pose=(0.1, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0),
                    right_pose=(-0.1, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0),
                )
                controller.enter(sample, _sim())

                action, metadata = controller.build_action(self._obs(), sample)

                np.testing.assert_allclose(action["left_arm_joint_state"], 0.1)
                np.testing.assert_allclose(action["right_arm_joint_state"], -0.1)
                np.testing.assert_allclose(action["left_ee_joint_state"], 0.3)
                np.testing.assert_allclose(action["right_ee_joint_state"], 0.6)
                self.assertEqual(metadata["ik_success"], 0)
                self.assertEqual(metadata["intervention_mask"], 0)


if __name__ == "__main__":
    unittest.main()
