from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "RoboDojo" / "observe_pi05_keyboard.sh"


class ObservePi05KeyboardWrapperTest(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_help_describes_visible_finite_non_recording_mode(self):
        result = self.run_script("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LEFT ARROW", result.stdout)
        self.assertIn("No LeRobot data", result.stdout)
        self.assertIn("default: 10", result.stdout)

    def test_invalid_layout_count_is_rejected_before_launch(self):
        result = self.run_script(
            "--task",
            "make_toast",
            "--ckpt",
            "/tmp/checkpoint/5000",
            "--layouts",
            "0",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--layouts must be a positive integer", result.stderr)

    def test_non_arx_x5_schema_is_rejected_before_launch(self):
        result = self.run_script(
            "--task",
            "make_toast",
            "--ckpt",
            "checkpoint",
            "--env-cfg",
            "piper",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("supports only --env-cfg arx_x5", result.stderr)


if __name__ == "__main__":
    unittest.main()
