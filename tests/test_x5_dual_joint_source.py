from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_client.x5_hardware import ArmState, ArmTarget, DualState, DualTarget  # noqa: E402


SOURCE_PATH = ROOT / "scripts/RoboDojo/x5_dual_joint_mirror_source.py"
SPEC = importlib.util.spec_from_file_location("x5_dual_joint_mirror_source", SOURCE_PATH)
assert SPEC is not None and SPEC.loader is not None
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)


def arm_state(q, gripper, seq=1):
    return ArmState(tuple(q), gripper * 0.08, gripper, 1000, seq)


class FakeHardware:
    def __init__(self) -> None:
        self.is_connected = True
        self.state = DualState(arm_state((0.0,) * 6, 1.0), arm_state((0.0,) * 6, 1.0))
        initial = ArmTarget((0.0,) * 6, 1.0)
        self._target = DualTarget(initial, initial)
        self.calls: list[str] = []

    @property
    def latched_target(self):
        return self._target

    def read(self):
        self.calls.append("read")
        return self.state

    def follow(self, target):
        self.calls.append("follow")
        self._target = target
        return target

    def enter_teach(self):
        self.calls.append("enter_teach")
        return self.state

    def enter_hold(self):
        self.calls.append("enter_hold")
        self._target = DualX5TargetFromState(self.state)
        return self.state


def DualX5TargetFromState(state):
    return DualTarget(
        ArmTarget(state.left.q_rad, state.left.gripper_open_fraction),
        ArmTarget(state.right.q_rad, state.right.gripper_open_fraction),
    )


def request(seq, q=(0.0,) * 6, gripper_delta=0.0, absolute_gripper=1.0):
    return {
        "type": "joint_mirror",
        "seq": seq,
        "sides": {
            side: {
                "delta_q_rad": list(q),
                "delta_gripper_open_fraction": gripper_delta,
                "sim_gripper_open_fraction": absolute_gripper,
            }
            for side in ("left", "right")
        },
    }


class JointMirrorSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hardware = FakeHardware()
        self.session = source.JointMirrorSession(self.hardware)

    def test_follow_is_identity_relative_to_connection_anchor(self) -> None:
        self.session.handle(request(1))
        response = self.session.handle(request(2, q=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), absolute_gripper=0.4))

        expected = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        self.assertEqual(self.hardware.latched_target.left.q_rad, expected)
        self.assertEqual(response["sides"]["left"]["follow_target_q_rad"], list(expected))
        self.assertAlmostEqual(self.hardware.latched_target.left.gripper_open_fraction, 0.4)

    def test_manual_wire_delta_preserves_identity_joint_signs(self) -> None:
        self.session.handle(request(1))
        self.assertTrue(self.session.toggle_intervention())
        entered = self.session.handle(request(2))
        self.assertEqual((entered["mode"], entered["edge"]), ("manual", "enter"))
        physical_delta = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        self.hardware.state = DualState(
            arm_state(physical_delta, 0.25, seq=2),
            arm_state(physical_delta, 0.75, seq=2),
        )

        manual = self.session.handle(request(3))
        wire = tuple(manual["sides"]["left"]["leader_delta_q_rad"])

        self.assertEqual(wire, physical_delta)
        self.assertAlmostEqual(
            manual["sides"]["left"]["leader_delta_gripper_open_fraction"], -0.75
        )

    def test_exit_holds_release_pose_and_requires_new_zero_frame(self) -> None:
        self.session.handle(request(1))
        self.session.toggle_intervention()
        self.session.handle(request(2))
        release_q = (0.2,) * 6
        self.hardware.state = DualState(arm_state(release_q, 0.5), arm_state(release_q, 0.5))
        self.session.toggle_intervention()

        exited = self.session.handle(request(3))

        self.assertEqual((exited["mode"], exited["edge"]), ("follow", "exit"))
        self.assertEqual(self.hardware.latched_target.left.q_rad, release_q)
        with self.assertRaises(source.X5SourceError):
            self.session.handle(request(4, q=(0.01,) * 6))
        zero = self.session.handle(request(5))
        self.assertEqual((zero["mode"], zero["edge"]), ("follow", None))

    def test_heartbeat_and_end_match_existing_client_envelope(self) -> None:
        heartbeat = self.session.handle({"type": "heartbeat", "seq": 1})
        self.assertEqual(set(heartbeat["sides"]), {"left", "right"})
        self.assertEqual(heartbeat["terminal"], None)
        ended = self.session.handle({"type": "end", "seq": 2})
        self.assertEqual((ended["type"], ended["seq"], ended["mode"], ended["edge"]), ("end", 2, "follow", None))
        self.assertTrue(self.session.ended)

    def test_left_terminal_holds_and_emits_one_retry_boundary(self) -> None:
        self.session.handle(request(1))
        self.assertTrue(self.session.request_terminal("retry"))

        response = self.session.handle(request(2, q=(0.2,) * 6))

        self.assertEqual((response["mode"], response["edge"], response["terminal"]), ("follow", None, "retry"))
        self.assertIn("enter_hold", self.hardware.calls)
        self.assertFalse(self.session.toggle_intervention())
        heartbeat = self.session.handle({"type": "heartbeat", "seq": 3})
        self.assertIsNone(heartbeat["terminal"])

    def test_right_terminal_overrides_manual_mode(self) -> None:
        self.session.handle(request(1))
        self.session.toggle_intervention()
        self.session.handle(request(2))
        self.assertTrue(self.session.request_terminal("save"))

        response = self.session.handle(request(3))

        self.assertEqual((response["mode"], response["edge"], response["terminal"]), ("follow", None, "save"))
        self.assertIsNone(self.session.manual_anchor)

    def test_cli_defaults_are_stable_for_launcher(self) -> None:
        args = source.build_arg_parser().parse_args([])
        self.assertEqual((args.host, args.port), ("127.0.0.1", 8770))
        self.assertEqual((args.left_can, args.right_can), ("can1", "can3"))
        self.assertEqual((args.left_model, args.right_model), ("X5", "X5"))
        self.assertEqual(tuple(args.home_rad), (0.0,) * 6)
        self.assertEqual(args.home_gripper_fraction, 1.0)
        self.assertEqual(args.follow_preview_s, 0.04)
        self.assertEqual(source.PROTOCOL, "robodojo_dual_joint_mirror_v1")


if __name__ == "__main__":
    unittest.main()
