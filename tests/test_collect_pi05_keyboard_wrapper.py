from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "RoboDojo" / "collect_pi05_keyboard.sh"


class CollectPi05KeyboardWrapperTest(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_help_describes_operator_decisions_and_direct_lerobot(self):
        result = self.run_script("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LeRobot v3 output", result.stdout)
        self.assertIn("RIGHT accepts and advances", result.stdout)
        self.assertIn("--resume", result.stdout)
        self.assertNotIn("HDF5 root", result.stdout)

    def test_unsafe_repo_id_is_rejected_before_launch(self):
        result = self.run_script(
            "--task",
            "make_toast",
            "--ckpt",
            "checkpoint",
            "--lerobot-repo-id",
            "../outside",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("unsafe --lerobot-repo-id", result.stderr)

    def test_non_cpu_codec_is_rejected_before_launch(self):
        result = self.run_script(
            "--task",
            "make_toast",
            "--ckpt",
            "checkpoint",
            "--lerobot-vcodec",
            "h264_nvenc",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--lerobot-vcodec must be", result.stderr)

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

    def test_resume_rejects_non_lerobot_directory_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "not_a_dataset").mkdir()
            result = self.run_script(
                "--task",
                "make_toast",
                "--ckpt",
                "checkpoint",
                "--lerobot-root",
                tmp,
                "--lerobot-repo-id",
                "not_a_dataset",
                "--resume",
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("not a LeRobot v3 dataset", result.stderr)


if __name__ == "__main__":
    unittest.main()
