import unittest

from vqa_gen.make_kong.vqa_config import LEFT_TARGET_GRIPPER_OPENING, gripper_joint_positions


class MakeKongVqaConfigTests(unittest.TestCase):
    def test_left_gripper_is_configured_closed(self) -> None:
        self.assertEqual(LEFT_TARGET_GRIPPER_OPENING, 0.0)
        self.assertEqual(
            gripper_joint_positions(
                LEFT_TARGET_GRIPPER_OPENING,
                gripper_scale=(0.0, 0.044),
                gripper_sign=1,
                gripper_mimic=(0.0, 1.0, 0.0),
            ),
            (0.0, 0.0),
        )

    def test_negative_sign_uses_upper_bound_when_closed(self) -> None:
        self.assertEqual(
            gripper_joint_positions(
                0.0,
                gripper_scale=(0.0, 0.044),
                gripper_sign=-1,
                gripper_mimic=(0.0, -1.0, 0.044),
            ),
            (0.044, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
