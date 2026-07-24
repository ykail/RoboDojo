import unittest

import numpy as np

from src.eval_client.policy_runtime import (
    ACTION_CHUNK_CONSUMPTION,
    ACTION_NEXT_INFER_OBSERVATION,
    ACTION_PREEMPTION_BOUNDARY,
    ACTION_SCHEMA_ID,
    ARX_X5_SIM_ARM_LIMITS,
    ARX_X5_SIM_PI05_PROFILE,
    OBSERVATION_SCHEMA_ID,
    ROBOT_SCHEMA_ID,
    ActionValidationSpec,
    JointLimits,
    iter_arx_x5_eval_actions,
    parse_action_chunk,
)


def _test_action_spec():
    limits = JointLimits(
        lower=(-20.0,) * 6,
        upper=(20.0,) * 6,
    )
    return ActionValidationSpec(
        expected_horizon=2,
        expected_control_dt_s=0.04,
        left_arm_limits=limits,
        right_arm_limits=limits,
    )


def _action_payload():
    return {
        "control_mode": "absolute_joint_position",
        "control_dt_s": 0.04,
        "commands": {
            "left_arm_joint_position": np.asarray(
                [[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]],
                dtype=np.float32,
            ),
            "left_gripper_open_fraction": np.asarray(
                [[0.25], [0.5]],
                dtype=np.float32,
            ),
            "right_arm_joint_position": np.asarray(
                [[7, 8, 9, 10, 11, 12], [17, 18, 19, 20, 19, 18]],
                dtype=np.float32,
            ),
            "right_gripper_open_fraction": np.asarray(
                [[0.75], [1.0]],
                dtype=np.float32,
            ),
        },
    }


class PolicyExecutionProfileTest(unittest.TestCase):
    def test_released_pi05_profile_is_client_owned_and_exact(self):
        profile = ARX_X5_SIM_PI05_PROFILE
        self.assertEqual(
            profile.schemas_payload(),
            {
                "observation": OBSERVATION_SCHEMA_ID,
                "action": ACTION_SCHEMA_ID,
                "robot": ROBOT_SCHEMA_ID,
            },
        )
        self.assertEqual(profile.action_spec.expected_horizon, 50)
        self.assertEqual(profile.action_spec.expected_control_dt_s, 0.04)
        self.assertAlmostEqual(
            profile.action_spec.expected_horizon * profile.action_spec.expected_control_dt_s,
            2.0,
        )
        self.assertEqual(
            profile.action_spec.left_arm_limits,
            ARX_X5_SIM_ARM_LIMITS,
        )
        self.assertEqual(
            profile.action_spec.right_arm_limits,
            ARX_X5_SIM_ARM_LIMITS,
        )
        self.assertEqual(
            ARX_X5_SIM_ARM_LIMITS.lower,
            (-10.0, -10.0, -10.0, -10.0, -10.0, -3.14),
        )
        self.assertEqual(
            ARX_X5_SIM_ARM_LIMITS.upper,
            (10.0, 10.0, 10.0, 10.0, 10.0, 3.14),
        )

        execution = profile.execution_payload()
        self.assertEqual(execution["images"]["head"], [480, 640, 3])
        self.assertEqual(execution["action"]["horizon"], 50)
        self.assertEqual(
            execution["action"]["chunk_consumption"],
            ACTION_CHUNK_CONSUMPTION,
        )
        self.assertEqual(
            execution["action"]["preemption_boundary"],
            ACTION_PREEMPTION_BOUNDARY,
        )
        self.assertEqual(
            execution["action"]["next_infer_observation"],
            ACTION_NEXT_INFER_OBSERVATION,
        )
        self.assertIsInstance(
            execution["action"]["left_arm_joint_limits"]["lower"],
            list,
        )

    def test_execution_payload_is_a_fresh_mutable_wire_projection(self):
        first = ARX_X5_SIM_PI05_PROFILE.execution_payload()
        first["images"]["head"][0] = 1
        first["action"]["left_arm_joint_limits"]["lower"][0] = -1

        second = ARX_X5_SIM_PI05_PROFILE.execution_payload()
        self.assertEqual(second["images"]["head"], [480, 640, 3])
        self.assertEqual(
            second["action"]["left_arm_joint_limits"]["lower"][0],
            -10.0,
        )

    def test_profile_rejects_unvalidated_components(self):
        profile_type = type(ARX_X5_SIM_PI05_PROFILE)
        with self.assertRaises(TypeError):
            profile_type(
                observation_spec=None,
                action_spec=ARX_X5_SIM_PI05_PROFILE.action_spec,
            )
        with self.assertRaises(TypeError):
            profile_type(
                observation_spec=ARX_X5_SIM_PI05_PROFILE.observation_spec,
                action_spec=None,
            )


class ArxX5ActionBridgeTest(unittest.TestCase):
    def test_structured_chunk_maps_to_exact_legacy_eval_keys(self):
        chunk = parse_action_chunk(
            _action_payload(),
            spec=_test_action_spec(),
        )
        steps = list(iter_arx_x5_eval_actions(chunk))

        self.assertEqual(len(steps), 2)
        self.assertEqual(
            set(steps[0]),
            {
                "left_arm_joint_state",
                "left_ee_joint_state",
                "right_arm_joint_state",
                "right_ee_joint_state",
            },
        )
        np.testing.assert_array_equal(
            steps[0]["left_arm_joint_state"],
            [0, 1, 2, 3, 4, 5],
        )
        np.testing.assert_array_equal(
            steps[0]["left_ee_joint_state"],
            [0.25],
        )
        np.testing.assert_array_equal(
            steps[0]["right_arm_joint_state"],
            [7, 8, 9, 10, 11, 12],
        )
        np.testing.assert_array_equal(
            steps[0]["right_ee_joint_state"],
            [0.75],
        )
        np.testing.assert_array_equal(
            steps[1]["left_arm_joint_state"],
            [10, 11, 12, 13, 14, 15],
        )
        self.assertEqual(steps[0]["left_arm_joint_state"].shape, (6,))
        self.assertEqual(steps[0]["left_ee_joint_state"].shape, (1,))
        self.assertFalse(steps[0]["left_arm_joint_state"].flags.writeable)

    def test_bridge_requires_a_trusted_canonical_chunk(self):
        with self.assertRaises(TypeError):
            list(iter_arx_x5_eval_actions(_action_payload()))


if __name__ == "__main__":
    unittest.main()
