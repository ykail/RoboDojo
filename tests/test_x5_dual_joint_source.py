from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_client.x5_hardware import ArmState, ArmTarget, DualState, DualTarget  # noqa: E402


SOURCE_PATH = ROOT / "scripts/RoboDojo/x5_dual_joint_mirror_source.py"
SPEC = importlib.util.spec_from_file_location("x5_dual_joint_mirror_source", SOURCE_PATH)
assert SPEC is not None and SPEC.loader is not None
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)


def arm_state(q, gripper, seq=1, sample_ns=1000):
    return ArmState(tuple(q), gripper * 0.08, gripper, sample_ns, seq)


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

    def hold(self):
        self.calls.append("hold")
        return self._target


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
        self.assertTrue(self.session.toggle_intervention(1100))
        entered = self.session.handle(request(2))
        self.assertEqual((entered["mode"], entered["edge"]), ("manual", "enter"))
        self.assertIsNone(entered["raw_segment_index"])
        self.assertEqual(entered["boundary_monotonic_ns"], 1100)
        self.assertEqual(entered["sample_monotonic_ns"], 1000)
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

    def test_enter_response_uses_fresh_teach_anchor_not_stale_deadline_read(self) -> None:
        self.session.handle(request(1))
        stale = self.hardware.state
        fresh_q = (0.25,) * 6
        self.hardware.state = DualState(
            arm_state(fresh_q, 0.3, seq=2, sample_ns=2000),
            arm_state(fresh_q, 0.7, seq=2, sample_ns=2000),
        )
        self.session.toggle_intervention(1500)

        entered = self.session.handle(request(2), measured=stale)

        self.assertEqual(entered["sample_monotonic_ns"], 2000)
        self.assertEqual(entered["sides"]["left"]["measured_q_rad"], list(fresh_q))
        self.assertEqual(entered["sides"]["left"]["leader_delta_q_rad"], [0.0] * 6)

    def test_exit_holds_release_pose_and_requires_new_zero_frame(self) -> None:
        self.session.handle(request(1))
        self.session.toggle_intervention(1100)
        self.session.handle(request(2))
        release_q = (0.2,) * 6
        self.hardware.state = DualState(arm_state(release_q, 0.5), arm_state(release_q, 0.5))
        self.session.toggle_intervention(2200)

        exited = self.session.handle(request(3))

        self.assertEqual((exited["mode"], exited["edge"]), ("follow", "exit"))
        self.assertIsNone(exited["raw_segment_index"])
        self.assertIsNone(exited["terminal"])
        self.assertEqual(exited["boundary_monotonic_ns"], 2200)
        self.assertEqual(self.hardware.latched_target.left.q_rad, release_q)
        with self.assertRaises(source.X5SourceError):
            self.session.handle(request(4, q=(0.01,) * 6))
        zero = self.session.handle(request(5))
        self.assertEqual((zero["mode"], zero["edge"]), ("follow", None))
        self.assertIsNone(zero["boundary_monotonic_ns"])

    def test_fast_next_enter_waits_until_exit_zero_is_acknowledged(self) -> None:
        self.session.handle(request(1))
        self.session.toggle_intervention(1100)
        self.session.handle(request(2))
        self.session.toggle_intervention(2200)
        exited = self.session.handle(request(3))
        self.assertEqual((exited["mode"], exited["edge"]), ("follow", "exit"))

        # Model the third i arriving before Isaac can send its mandatory zero
        # synchronization request after the first intervention exit.
        self.assertTrue(self.session.toggle_intervention(2300))
        zero = self.session.handle(request(4))
        self.assertEqual((zero["mode"], zero["edge"]), ("follow", None))
        self.assertEqual(self.session.pending_transition, "enter")

        entered = self.session.handle(request(5))
        self.assertEqual((entered["mode"], entered["edge"]), ("manual", "enter"))
        self.assertIsNone(self.session.pending_transition)

    def test_active_hold_recovery_retries_without_closing_hardware(self) -> None:
        hardware = FakeHardware()
        original = hardware.enter_hold
        failures = iter((RuntimeError("first"), RuntimeError("second")))

        def flaky_hold():
            try:
                raise next(failures)
            except StopIteration:
                return original()

        hardware.enter_hold = flaky_hold

        source._recover_active_hold(hardware, attempts=3, retry_delay_s=0.0)

        self.assertEqual(hardware.calls[-2:], ["enter_hold", "hold"])

    def test_heartbeat_and_end_match_existing_client_envelope(self) -> None:
        heartbeat = self.session.handle({"type": "heartbeat", "seq": 1})
        self.assertEqual(set(heartbeat["sides"]), {"left", "right"})
        self.assertEqual(heartbeat["terminal"], None)
        self.assertIsNone(heartbeat["raw_segment_index"])
        ended = self.session.handle({"type": "end", "seq": 2})
        self.assertEqual((ended["type"], ended["seq"], ended["mode"], ended["edge"]), ("end", 2, "follow", None))
        self.assertIsNone(ended["raw_fragment"])
        self.assertTrue(self.session.ended)

    def test_left_terminal_holds_and_emits_one_retry_boundary(self) -> None:
        self.session.handle(request(1))
        self.assertTrue(self.session.request_terminal("retry", 3300))

        response = self.session.handle(request(2, q=(0.2,) * 6))

        self.assertEqual((response["mode"], response["edge"], response["terminal"]), ("follow", None, "retry"))
        self.assertEqual(response["boundary_monotonic_ns"], 3300)
        self.assertIn("enter_hold", self.hardware.calls)
        self.assertFalse(self.session.toggle_intervention())
        heartbeat = self.session.handle({"type": "heartbeat", "seq": 3})
        self.assertIsNone(heartbeat["terminal"])

    def test_right_terminal_overrides_manual_mode(self) -> None:
        self.session.handle(request(1))
        self.session.toggle_intervention(1100)
        self.session.handle(request(2))
        self.assertTrue(self.session.request_terminal("save", 4400))

        response = self.session.handle(request(3))

        self.assertEqual((response["mode"], response["edge"], response["terminal"]), ("follow", None, "save"))
        self.assertEqual(response["boundary_monotonic_ns"], 4400)
        self.assertIsNone(self.session.manual_anchor)

    def test_cli_defaults_are_stable_for_launcher(self) -> None:
        args = source.build_arg_parser().parse_args([])
        self.assertEqual((args.host, args.port), ("127.0.0.1", 8770))
        self.assertEqual((args.left_can, args.right_can), ("can1", "can3"))
        self.assertEqual((args.left_model, args.right_model), ("X5", "X5"))
        self.assertEqual(tuple(args.home_rad), (0.0,) * 6)
        self.assertEqual(args.home_gripper_fraction, 1.0)
        self.assertEqual(args.follow_preview_s, 0.04)
        self.assertIsNone(args.raw_root)
        self.assertEqual(source.PROTOCOL, "robodojo_dual_joint_mirror_v1")

    def test_save_end_publishes_boundary_truncated_manual_npz(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fragments"
            session = source.JointMirrorSession(
                self.hardware,
                raw_root=root,
                raw_frequency_hz=100.0,
            )
            session.handle(request(1))
            self.assertTrue(session.toggle_intervention(1100))
            anchor_q = (0.1,) * 6
            self.hardware.state = DualState(
                arm_state(anchor_q, 0.2, seq=2, sample_ns=1200),
                arm_state(anchor_q, 0.8, seq=2, sample_ns=1200),
            )
            first_entered = session.handle(request(2))
            self.assertEqual(first_entered["raw_segment_index"], 0)

            kept_q = (0.2,) * 6
            kept = DualState(
                arm_state(kept_q, 0.3, seq=3, sample_ns=2000),
                arm_state(kept_q, 0.7, seq=3, sample_ns=2000),
            )
            session.observe_hardware(kept)
            self.assertTrue(session.request_terminal("save", 2500))
            late_q = (0.9,) * 6
            late = DualState(
                arm_state(late_q, 0.9, seq=4, sample_ns=3000),
                arm_state(late_q, 0.1, seq=4, sample_ns=3000),
            )
            session.observe_hardware(late)
            terminal = session.handle(request(3), measured=late)
            self.assertEqual(terminal["terminal"], "save")
            self.assertIsNone(terminal["raw_segment_index"])

            ended = session.handle({"type": "end", "seq": 4}, measured=late)
            descriptor = ended["raw_fragment"]
            self.assertEqual(descriptor["format"], source.RAW_FRAGMENT_FORMAT)
            self.assertEqual(descriptor["sample_count"], 2)
            self.assertEqual(descriptor["segment_count"], 1)
            self.assertEqual(descriptor["frequency_hz"], 100.0)
            path = Path(descriptor["path"])
            self.assertTrue(path.is_file())
            self.assertEqual(
                descriptor["sha256"],
                "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(
                    archive["sample_monotonic_ns"], np.asarray([1200, 2000])
                )
                np.testing.assert_array_equal(archive["segment_index"], np.asarray([0, 0]))
                np.testing.assert_allclose(
                    archive["left_q_rad"], np.asarray([anchor_q, kept_q])
                )
                np.testing.assert_allclose(
                    archive["right_gripper_open_fraction"], np.asarray([0.8, 0.7])
                )
                np.testing.assert_array_equal(archive["segment_start_ns"], [1100])
                np.testing.assert_array_equal(archive["segment_end_ns"], [2500])
                np.testing.assert_array_equal(
                    archive["segment_anchor_timestamp_ns"], [1200]
                )
                np.testing.assert_allclose(
                    archive["left_segment_anchor_q_rad"], np.asarray([anchor_q])
                )
            self.assertFalse(list(root.glob("*.partial")))

    def test_retry_and_unsaved_end_never_publish_a_fragment(self) -> None:
        for terminal in ("retry", None):
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "fragments"
                hardware = FakeHardware()
                session = source.JointMirrorSession(
                    hardware,
                    raw_root=root,
                    raw_frequency_hz=100.0,
                )
                session.handle(request(1))
                session.toggle_intervention(1100)
                hardware.state = DualState(
                    arm_state((0.1,) * 6, 0.2, seq=2, sample_ns=1200),
                    arm_state((0.1,) * 6, 0.8, seq=2, sample_ns=1200),
                )
                session.handle(request(2))
                if terminal is not None:
                    session.request_terminal(terminal, 1500)
                    session.handle(request(3))
                ended = session.handle({"type": "end", "seq": 4})
                self.assertIsNone(ended["raw_fragment"])
                self.assertFalse(root.exists())

    def test_raw_root_cli_is_explicit(self) -> None:
        args = source.build_arg_parser().parse_args(["--raw-root", "/tmp/x5-raw-test"])
        self.assertEqual(args.raw_root, Path("/tmp/x5-raw-test"))

    def test_save_end_publishes_two_manual_segments_in_one_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fragments"
            session = source.JointMirrorSession(
                self.hardware,
                raw_root=root,
                raw_frequency_hz=100.0,
            )
            session.handle(request(1))
            session.toggle_intervention(1100)
            first_anchor_q = (0.1,) * 6
            self.hardware.state = DualState(
                arm_state(first_anchor_q, 0.2, seq=2, sample_ns=1200),
                arm_state(first_anchor_q, 0.8, seq=2, sample_ns=1200),
            )
            first_entered = session.handle(request(2))
            self.assertEqual(first_entered["raw_segment_index"], 0)
            session.toggle_intervention(1500)
            first_exited = session.handle(request(3))
            self.assertEqual(first_exited["raw_segment_index"], 0)
            # Satisfy the fresh-zero requirement after leaving manual mode.
            session.handle(request(4))
            session.toggle_intervention(2000)
            second_anchor_q = (0.3,) * 6
            self.hardware.state = DualState(
                arm_state(second_anchor_q, 0.4, seq=3, sample_ns=2100),
                arm_state(second_anchor_q, 0.6, seq=3, sample_ns=2100),
            )
            entered = session.handle(request(5))
            self.assertEqual((entered["mode"], entered["edge"]), ("manual", "enter"))
            self.assertEqual(entered["raw_segment_index"], 1)

            second_kept_q = (0.4,) * 6
            second_kept = DualState(
                arm_state(second_kept_q, 0.5, seq=4, sample_ns=2200),
                arm_state(second_kept_q, 0.5, seq=4, sample_ns=2200),
            )
            session.observe_hardware(second_kept)
            self.assertTrue(session.request_terminal("save", 2250))
            second_late = DualState(
                arm_state((0.9,) * 6, 0.9, seq=5, sample_ns=2300),
                arm_state((0.9,) * 6, 0.1, seq=5, sample_ns=2300),
            )
            session.observe_hardware(second_late)
            terminal = session.handle(request(6), measured=second_late)
            self.assertEqual(terminal["terminal"], "save")
            self.assertIsNone(terminal["raw_segment_index"])

            ended = session.handle({"type": "end", "seq": 7}, measured=second_late)
            descriptor = ended["raw_fragment"]
            self.assertEqual(descriptor["sample_count"], 3)
            self.assertEqual(descriptor["segment_count"], 2)
            path = Path(descriptor["path"])
            self.assertTrue(path.is_file())
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(
                    archive["sample_monotonic_ns"], [1200, 2100, 2200]
                )
                np.testing.assert_array_equal(archive["segment_index"], [0, 1, 1])
                np.testing.assert_array_equal(archive["segment_start_ns"], [1100, 2000])
                np.testing.assert_array_equal(archive["segment_end_ns"], [1500, 2250])
                np.testing.assert_array_equal(
                    archive["segment_anchor_timestamp_ns"], [1200, 2100]
                )
                np.testing.assert_allclose(
                    archive["left_segment_anchor_q_rad"],
                    np.asarray([first_anchor_q, second_anchor_q]),
                )
                np.testing.assert_allclose(
                    archive["right_segment_anchor_gripper_open_fraction"], [0.8, 0.6]
                )
            self.assertFalse(list(root.glob("*.partial")))

    def test_disconnect_after_second_manual_segment_discards_everything(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fragments"
            session = source.JointMirrorSession(
                self.hardware,
                raw_root=root,
                raw_frequency_hz=100.0,
            )
            session.handle(request(1))
            session.toggle_intervention(1100)
            self.hardware.state = DualState(
                arm_state((0.1,) * 6, 0.5, seq=2, sample_ns=1200),
                arm_state((0.1,) * 6, 0.5, seq=2, sample_ns=1200),
            )
            session.handle(request(2))
            session.toggle_intervention(1500)
            session.handle(request(3))
            session.handle(request(4))
            session.toggle_intervention(2000)
            self.hardware.state = DualState(
                arm_state((0.2,) * 6, 0.5, seq=3, sample_ns=2100),
                arm_state((0.2,) * 6, 0.5, seq=3, sample_ns=2100),
            )
            session.handle(request(5))

            session.disconnect()
            self.assertFalse(root.exists())

    def test_raw_start_failure_restores_active_hold_after_entering_teach(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = source.JointMirrorSession(
                self.hardware,
                raw_root=Path(temporary) / "fragments",
                raw_frequency_hz=100.0,
            )
            session.handle(request(1))
            session.toggle_intervention(1100)
            assert session.raw_recorder is not None

            def fail_start(*_args, **_kwargs):
                raise source.X5SourceError("synthetic recorder failure")

            session.raw_recorder.start_segment = fail_start
            with self.assertRaisesRegex(source.X5SourceError, "synthetic recorder failure"):
                session.handle(request(2))

            self.assertEqual(self.hardware.calls[-3:], ["enter_teach", "enter_hold", "hold"])
            self.assertEqual(session.mode, "follow")
            self.assertTrue(session.require_zero)
            self.assertIsNone(session.manual_anchor)
            self.assertIsNone(session.pending_transition)

    def test_retry_after_second_manual_segment_discards_everything(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fragments"
            session = source.JointMirrorSession(
                self.hardware,
                raw_root=root,
                raw_frequency_hz=100.0,
            )
            session.handle(request(1))
            session.toggle_intervention(1100)
            self.hardware.state = DualState(
                arm_state((0.1,) * 6, 0.5, seq=2, sample_ns=1200),
                arm_state((0.1,) * 6, 0.5, seq=2, sample_ns=1200),
            )
            session.handle(request(2))
            session.toggle_intervention(1500)
            session.handle(request(3))
            session.handle(request(4))
            session.toggle_intervention(2000)
            self.hardware.state = DualState(
                arm_state((0.2,) * 6, 0.5, seq=3, sample_ns=2100),
                arm_state((0.2,) * 6, 0.5, seq=3, sample_ns=2100),
            )
            session.handle(request(5))
            self.assertTrue(session.request_terminal("retry", 2200))
            terminal = session.handle(request(6))
            self.assertEqual(terminal["terminal"], "retry")

            ended = session.handle({"type": "end", "seq": 7})
            self.assertIsNone(ended["raw_fragment"])
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
