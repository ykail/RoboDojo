from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "RoboDojo" / "eval_kai0_pi05.sh"
WRAPPER = ROOT / "scripts" / "RoboDojo" / "collect_pi05_piperx_sim_dagger.sh"
CALIBRATION = ROOT / "config" / "piperx_sim_dagger.example.json"


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
        self.assertIn("never starts, enables, or configures", wrapper.stdout)

        evaluator = subprocess.run(
            ["bash", str(EVAL), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(evaluator.returncode, 0, evaluator.stderr)
        self.assertIn("piperx_sim_dagger", evaluator.stdout)
        self.assertIn("--piperx-calibration", evaluator.stdout)

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
                "--piperx-calibration",
                str(CALIBRATION),
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
            calibration = tmp_path / "calibration.json"
            (kai0 / "scripts").mkdir(parents=True)
            checkpoint.mkdir()
            calibration.write_text(
                CALIBRATION.read_text(encoding="utf-8").replace('"calibrated": false', '"calibrated": true'),
                encoding="utf-8",
            )
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
                    str(calibration),
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
        self.assertIn("ROBODOJO_LEROBOT_REPO_ID=robodojo_interventions_make_toast", result.stdout)

    def test_launcher_never_probes_hardware_socket_before_real_client(self):
        source = EVAL.read_text(encoding="utf-8")
        self.assertNotIn(
            'tcp_endpoint_is_open "${piperx_bridge_host}"',
            source,
        )

    def test_uncalibrated_example_is_rejected_before_launch(self):
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
                    str(CALIBRATION),
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("calibrated=true", result.stderr)

    def test_calibrated_flag_alone_does_not_bypass_full_schema_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            kai0 = tmp_path / "kai0"
            checkpoint = tmp_path / "checkpoint"
            calibration = tmp_path / "incomplete.json"
            (kai0 / "scripts").mkdir(parents=True)
            checkpoint.mkdir()
            calibration.write_text('{"calibrated": true}', encoding="utf-8")
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
                    str(calibration),
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("full schema", result.stderr)


if __name__ == "__main__":
    unittest.main()
