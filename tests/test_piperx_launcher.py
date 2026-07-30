from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "RoboDojo" / "eval_kai0_pi05.sh"
WRAPPER = ROOT / "scripts" / "RoboDojo" / "collect_pi05_piperx_sim_dagger.sh"


class PiperXLauncherTest(unittest.TestCase):
    def test_wrapper_and_eval_help_describe_separate_supervised_bridge(self):
        wrapper = subprocess.run(
            ["bash", str(WRAPPER)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(wrapper.returncode, 0, wrapper.stderr)
        self.assertIn("first episode explicitly", wrapper.stdout)
        self.assertIn("No manual", wrapper.stdout)

        evaluator = subprocess.run(
            ["bash", str(EVAL), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(evaluator.returncode, 0, evaluator.stderr)
        self.assertIn("piperx_sim_dagger", evaluator.stdout)
        self.assertIn("--piperx-arm-timeout", evaluator.stdout)
        self.assertIn("--piperx-transition-timeout", evaluator.stdout)
        self.assertNotIn("--piperx-calibration", evaluator.stdout)

    def test_headless_is_rejected_for_visual_intervention(self):
        result = subprocess.run(
            [
                "bash",
                str(EVAL),
                "--task",
                "make_toast",
                "--checkpoint-dir",
                "/unused",
                "--checkpoint-id",
                "test-checkpoint",
                "--control-mode",
                "piperx_sim_dagger",
                "--headless",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--headless cannot be combined", result.stderr)

    def test_dry_run_exports_bridge_and_recording_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            kai0 = tmp_path / "kai0"
            checkpoint = tmp_path / "checkpoint"
            (kai0 / "scripts").mkdir(parents=True)
            checkpoint.mkdir()
            (kai0 / "scripts" / "serve_robodojo_policy.py").write_text("# test\n")
            result = subprocess.run(
                [
                    "bash",
                    str(EVAL),
                    "--task",
                    "make_toast",
                    "--checkpoint-dir",
                    str(checkpoint),
                    "--checkpoint-id",
                    "test-checkpoint",
                    "--kai0-root",
                    str(kai0),
                    "--kai0-python",
                    sys.executable,
                    "--control-mode",
                    "piperx_sim_dagger",
                    "--piperx-arm-timeout",
                    "45.0",
                    "--piperx-transition-timeout",
                    "7.5",
                    "--lerobot-root",
                    str(tmp_path / "data"),
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ROBODOJO_CONTROL_MODE=piperx_sim_dagger", result.stdout)
        self.assertIn("ROBODOJO_PIPERX_BRIDGE_PORT=8765", result.stdout)
        self.assertIn("ROBODOJO_PIPERX_ARM_TIMEOUT_S=45.0", result.stdout)
        self.assertIn("ROBODOJO_PIPERX_TRANSITION_TIMEOUT_S=7.5", result.stdout)
        self.assertIn("ROBODOJO_LEROBOT_REPO_ID=robodojo_interventions_make_toast", result.stdout)
        self.assertIn("PiPER-X profile=arx_x5_piperx_relative_v1", result.stdout)
        self.assertIn("--expected_policy_checkpoint_id", result.stdout)
        self.assertIn("--require_policy_clean", result.stdout)
        self.assertNotIn("ROBODOJO_PIPERX_CALIBRATION", result.stdout)

    def test_external_policy_dry_run_never_builds_a_local_server_command(self):
        expected_commit = "ecc1a7451c3156b1e5f7533851dbb0222896206f"
        expected_digest = "sha256:" + "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    "bash",
                    str(EVAL),
                    "--task",
                    "make_toast",
                    "--checkpoint-id",
                    "RoboDojo-sim-arx_x5-joint-0/59999",
                    "--external-policy-server-url",
                    "ws://127.0.0.1:18080",
                    "--expected-kai0-commit",
                    expected_commit,
                    "--expected-checkpoint-digest",
                    expected_digest,
                    "--lerobot-python",
                    sys.executable,
                    "--control-mode",
                    "piperx_sim_dagger",
                    "--lerobot-root",
                    str(Path(tmp) / "data"),
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("policy_mode=external", result.stdout)
        self.assertIn("policy_server_url ws://127.0.0.1:18080", result.stdout)
        self.assertIn(f"--expected_policy_code_revision {expected_commit}", result.stdout)
        self.assertIn(f"--expected_policy_checkpoint_digest {expected_digest}", result.stdout)
        self.assertIn("--require_policy_clean", result.stdout)
        self.assertIn("never started or stopped", result.stdout)
        self.assertIn("[dry-run] Policy HELLO preflight:", result.stdout)
        self.assertNotIn("[dry-run] Kai0 server:", result.stdout)
        self.assertNotIn("--checkpoint-dir", result.stdout)

    def test_external_policy_requires_expected_commit_and_loopback_url(self):
        common = [
            "bash",
            str(EVAL),
            "--task",
            "make_toast",
            "--checkpoint-id",
            "RoboDojo-sim-arx_x5-joint-0/59999",
            "--external-policy-server-url",
        ]
        missing_commit = subprocess.run(
            [*common, "ws://127.0.0.1:18080", "--dry-run"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(missing_commit.returncode, 2)
        self.assertIn("--expected-kai0-commit is required", missing_commit.stderr)

        non_loopback = subprocess.run(
            [
                *common,
                "ws://10.19.125.6:18080",
                "--expected-kai0-commit",
                "ecc1a7451c3156b1e5f7533851dbb0222896206f",
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(non_loopback.returncode, 2)
        self.assertIn("must be a loopback", non_loopback.stderr)

    def test_launcher_never_probes_hardware_socket_before_real_client(self):
        source = EVAL.read_text(encoding="utf-8")
        self.assertNotIn(
            'tcp_endpoint_is_open "${piperx_bridge_host}"',
            source,
        )

    def test_removed_calibration_option_is_rejected_as_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            kai0 = tmp_path / "kai0"
            checkpoint = tmp_path / "checkpoint"
            (kai0 / "scripts").mkdir(parents=True)
            checkpoint.mkdir()
            (kai0 / "scripts" / "serve_robodojo_policy.py").write_text("# test\n")
            result = subprocess.run(
                [
                    "bash",
                    str(EVAL),
                    "--task",
                    "make_toast",
                    "--checkpoint-dir",
                    str(checkpoint),
                    "--checkpoint-id",
                    "test-checkpoint",
                    "--kai0-root",
                    str(kai0),
                    "--kai0-python",
                    sys.executable,
                    "--control-mode",
                    "piperx_sim_dagger",
                    "--piperx-calibration",
                    "/unused/legacy-calibration.json",
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown argument: --piperx-calibration", result.stderr)

    def test_arm_and_transition_timeouts_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            kai0 = tmp_path / "kai0"
            checkpoint = tmp_path / "checkpoint"
            (kai0 / "scripts").mkdir(parents=True)
            checkpoint.mkdir()
            (kai0 / "scripts" / "serve_robodojo_policy.py").write_text("# test\n")
            result = subprocess.run(
                [
                    "bash",
                    str(EVAL),
                    "--task",
                    "make_toast",
                    "--checkpoint-dir",
                    str(checkpoint),
                    "--checkpoint-id",
                    "test-checkpoint",
                    "--kai0-root",
                    str(kai0),
                    "--kai0-python",
                    sys.executable,
                    "--control-mode",
                    "piperx_sim_dagger",
                    "--piperx-arm-timeout",
                    "0",
                    "--piperx-transition-timeout",
                    "10.0",
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("timeout/heartbeat values must be positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
