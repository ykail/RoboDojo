import unittest

import numpy as np

from src.eval_client.keyboard_teleop import (
    CartesianTeleopController,
    KeyboardSnapshot,
    KeyboardState,
    compose_world_delta_pose,
)

try:
    import transforms3d  # noqa: F401

    HAS_TRANSFORMS3D = True
except ImportError:
    HAS_TRANSFORMS3D = False


class KeyboardStateTest(unittest.TestCase):
    def test_deadman_axes_arm_and_one_shot_commands(self):
        state = KeyboardState(pos_step=0.005, rot_step=0.02)
        state.handle_key("SPACE", True)
        state.handle_key("W", True)
        state.handle_key("A", True)
        state.handle_key("KEY_2", True)
        state.handle_key("K", True)

        first = state.snapshot()
        self.assertTrue(first.deadman)
        self.assertTrue(first.takeover_pressed)
        self.assertEqual(first.active_arm, "right")
        self.assertEqual(first.gripper_toggles, ("right",))
        np.testing.assert_allclose(first.delta_pose, [0.005, 0.005, 0, 0, 0, 0])

        second = state.snapshot()
        self.assertTrue(second.deadman)
        self.assertFalse(second.takeover_pressed)
        self.assertEqual(second.gripper_toggles, ())

        state.handle_key("W", False)
        state.handle_key("A", False)
        state.handle_key("SPACE", False)
        released = state.snapshot()
        self.assertFalse(released.deadman)
        self.assertTrue(released.takeover_released)
        np.testing.assert_allclose(released.delta_pose, np.zeros(6))

    def test_repeat_press_does_not_toggle_twice(self):
        state = KeyboardState()
        state.handle_key("SPACE", True)
        state.handle_key("K", True)
        state.handle_key("K", True)
        self.assertEqual(state.snapshot().gripper_toggles, ("left",))
        state.handle_key("K", False)
        state.handle_key("K", True)
        self.assertEqual(state.snapshot().gripper_toggles, ("left",))

    def test_save_retry_is_one_shot_and_repeat_resistant(self):
        state = KeyboardState()
        state.handle_key("R", True)
        state.handle_key("R", True)
        self.assertTrue(state.snapshot().save_retry_requested)
        self.assertFalse(state.snapshot().save_retry_requested)
        state.handle_key("R", False)
        state.handle_key("R", True)
        self.assertTrue(state.snapshot().save_retry_requested)

    def test_gripper_toggle_requires_deadman_and_timeout_releases(self):
        now = [0.0]
        state = KeyboardState(deadman_timeout=2.0, clock=lambda: now[0])
        state.handle_key("K", True)
        self.assertEqual(state.snapshot().gripper_toggles, ())
        state.handle_key("K", False)
        state.handle_key("SPACE", True)
        state.handle_key("W", True)
        now[0] = 2.1
        snapshot = state.snapshot()
        self.assertFalse(snapshot.deadman)
        self.assertTrue(snapshot.takeover_released)
        np.testing.assert_allclose(snapshot.delta_pose, np.zeros(6))

    @unittest.skipUnless(HAS_TRANSFORMS3D, "transforms3d is available in the RoboDojo runtime")
    def test_pose_delta_is_world_frame_and_quaternion_is_normalized(self):
        pose = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
        result = compose_world_delta_pose(pose, np.array([0.01, -0.02, 0.03, 0.1, 0.0, 0.0]))
        np.testing.assert_allclose(result[:3], [0.11, 0.18, 0.33])
        self.assertAlmostEqual(float(np.linalg.norm(result[3:])), 1.0, places=7)

    @unittest.skipUnless(HAS_TRANSFORMS3D, "transforms3d is available in the RoboDojo runtime")
    def test_controller_reanchors_gripper_and_outputs_full_joint_action(self):
        class Robot:
            def __init__(self, arm_name):
                self.arm_name = arm_name
                self.type = "target"

        class RobotManager:
            def __init__(self):
                self.robot_list = [Robot("left_arm"), Robot("right_arm")]
                self.ik_call = None

            def get_real_endpose(self, robot, env_idx_list, is_relative):
                self.assert_args = (env_idx_list, is_relative)
                x = -0.2 if robot.arm_name == "left_arm" else 0.2
                return {0: np.array([x, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0])}

            def solve_ik(self, target_pose, env_idx, robot):
                self.ik_call = (target_pose, env_idx, robot.arm_name)
                return {"status": "Success", "joint_value": np.full(6, 0.11)}

        task_env = type("TaskEnv", (), {"robot_manager": RobotManager()})()
        controller = CartesianTeleopController(task_env)
        initial_obs = {
            "state": {
                "left_arm_joint_state": np.full(6, 0.1),
                "left_ee_joint_state": np.array([1.0]),
                "right_arm_joint_state": np.full(6, -0.1),
                "right_ee_joint_state": np.array([1.0]),
            }
        }
        controller.reset(initial_obs)
        current_obs = {
            "state": {
                **initial_obs["state"],
                "left_ee_joint_state": np.array([0.2]),
            }
        }
        snapshot = KeyboardSnapshot(
            deadman=True,
            active_arm="left",
            delta_pose=np.zeros(6),
            takeover_pressed=True,
        )
        action, metadata = controller.build_action(current_obs, snapshot)
        self.assertEqual(
            set(action),
            {
                "left_arm_joint_state",
                "left_ee_joint_state",
                "right_arm_joint_state",
                "right_ee_joint_state",
            },
        )
        np.testing.assert_allclose(action["left_ee_joint_state"], [0.2])
        np.testing.assert_allclose(action["right_arm_joint_state"], np.full(6, -0.1))
        self.assertEqual(task_env.robot_manager.ik_call[1:], (0, "left_arm"))
        self.assertEqual(metadata["action_source"], "human")


if __name__ == "__main__":
    unittest.main()
