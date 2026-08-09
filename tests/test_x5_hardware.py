from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_client.x5_hardware import (  # noqa: E402
    ArmTarget,
    DualTarget,
    DualX5Hardware,
    X5HardwareConfig,
    X5Mode,
)


class FakeJointState:
    def __init__(self, dof: int) -> None:
        self._pos = [0.0] * dof
        self.gripper_pos = 0.0

    def pos(self):
        return self._pos


class FakeGain:
    def __init__(self, dof: int) -> None:
        self._kp = [0.0] * dof
        self._kd = [0.0] * dof
        self.gripper_kp = 0.0
        self.gripper_kd = 0.0

    def kp(self):
        return self._kp

    def kd(self):
        return self._kd


class FakeController:
    def __init__(self, interface: str) -> None:
        self.interface = interface
        self.state = FakeJointState(6)
        self.commands: list[tuple[tuple[float, ...], float]] = []
        self.gains: list[tuple[tuple[float, ...], tuple[float, ...], float, float]] = []
        self.damping_count = 0
        self.log_levels: list[object] = []

    def get_joint_state(self):
        return self.state

    def set_joint_cmd(self, command) -> None:
        q = tuple(float(value) for value in command.pos())
        gripper = float(command.gripper_pos)
        self.commands.append((q, gripper))
        self.state._pos[:] = q
        self.state.gripper_pos = gripper

    def set_gain(self, gain) -> None:
        self.gains.append(
            (tuple(gain.kp()), tuple(gain.kd()), float(gain.gripper_kp), float(gain.gripper_kd))
        )

    def set_to_damping(self) -> None:
        self.damping_count += 1

    def set_log_level(self, level) -> None:
        self.log_levels.append(level)


class FakeSdk:
    def __init__(self) -> None:
        self.controllers: dict[str, FakeController] = {}
        robot_config = SimpleNamespace(
            joint_dof=6,
            joint_pos_min=[-1.0] * 6,
            joint_pos_max=[1.0] * 6,
            gripper_width=0.08,
        )
        self.RobotConfigFactory = SimpleNamespace(
            get_instance=lambda: SimpleNamespace(get_config=lambda model: robot_config)
        )

        def controller_config(kind, dof):
            self.last_controller_kind = kind
            self.last_controller_dof = dof
            return SimpleNamespace(
                background_send_recv=False,
                clear_motor_state_on_init=True,
                gravity_compensation=False,
                controller_dt=0.01,
            )

        self.ControllerConfigFactory = SimpleNamespace(
            get_instance=lambda: SimpleNamespace(get_config=controller_config)
        )
        self.JointState = FakeJointState
        self.Gain = FakeGain
        self.LogLevel = SimpleNamespace(WARNING="warning")

    def Arx5JointController(self, robot_config, controller_config, interface):
        del robot_config, controller_config
        controller = FakeController(interface)
        self.controllers[interface] = controller
        return controller


class DualX5HardwareTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sdk = FakeSdk()
        self.hardware = DualX5Hardware(
            X5HardwareConfig(left_can="left-can", right_can="right-can"),
            sdk_factory=lambda: self.sdk,
            monotonic_ns=lambda: 123456,
            sleep=lambda _: None,
        )
        self.hardware.connect()

    def tearDown(self) -> None:
        self.hardware.close()

    def test_connect_and_follow_convert_identity_joints_and_gripper(self) -> None:
        target = DualTarget(
            ArmTarget((0.1, 0.2, 0.3, 0.4, 0.5, 2.0), 0.25),
            ArmTarget((-0.1, -0.2, -0.3, -0.4, -0.5, -2.0), 0.75),
        )

        applied = self.hardware.follow(target)

        self.assertEqual(self.hardware.mode, X5Mode.FOLLOW)
        self.assertEqual(applied.left.q_rad, (0.1, 0.2, 0.3, 0.4, 0.5, 1.0))
        self.assertEqual(applied.right.q_rad, (-0.1, -0.2, -0.3, -0.4, -0.5, -1.0))
        left_command = self.sdk.controllers["left-can"].commands[-1]
        right_command = self.sdk.controllers["right-can"].commands[-1]
        self.assertEqual(left_command[0], applied.left.q_rad)
        self.assertAlmostEqual(left_command[1], 0.02)
        self.assertEqual(right_command[0], applied.right.q_rad)
        self.assertAlmostEqual(right_command[1], 0.06)

    def test_teach_then_hold_uses_current_pose_not_old_follow_target(self) -> None:
        old = DualTarget(
            ArmTarget((0.8,) * 6, 0.8),
            ArmTarget((-0.8,) * 6, 0.2),
        )
        self.hardware.follow(old)
        self.hardware.enter_teach()
        left = self.sdk.controllers["left-can"]
        right = self.sdk.controllers["right-can"]
        left.state._pos[:] = [0.21] * 6
        left.state.gripper_pos = 0.03
        right.state._pos[:] = [-0.31] * 6
        right.state.gripper_pos = 0.05

        self.hardware.enter_hold()

        self.assertEqual(self.hardware.mode, X5Mode.HOLD)
        self.assertEqual(left.commands[-2][0], (0.21,) * 6)
        self.assertEqual(left.commands[-1][0], (0.21,) * 6)
        self.assertEqual(right.commands[-2][0], (-0.31,) * 6)
        self.assertEqual(right.commands[-1][0], (-0.31,) * 6)
        self.assertEqual(left.gains[-1][0], self.hardware.config.active_joint_kp)

    def test_enter_teach_sets_damping_and_zero_position_gain(self) -> None:
        self.hardware.enter_teach()

        self.assertEqual(self.hardware.mode, X5Mode.TEACH)
        for controller in self.sdk.controllers.values():
            self.assertGreaterEqual(controller.damping_count, 1)
            self.assertEqual(controller.gains[-1][0], (0.0,) * 6)
            self.assertEqual(controller.gains[-1][1], self.hardware.config.teach_joint_kd)

    def test_move_home_finishes_at_configured_target_without_tolerance_gate(self) -> None:
        applied = self.hardware.move_home(
            (0.1, 0.0, -0.1, 0.2, 0.0, -0.2),
            1.0,
            duration_s=0.1,
            frequency_hz=20.0,
        )

        self.assertEqual(self.hardware.mode, X5Mode.HOLD)
        self.assertEqual(applied.left.q_rad, (0.1, 0.0, -0.1, 0.2, 0.0, -0.2))
        self.assertAlmostEqual(self.sdk.controllers["left-can"].commands[-1][1], 0.08)


if __name__ == "__main__":
    unittest.main()
