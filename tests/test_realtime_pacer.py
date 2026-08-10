from __future__ import annotations

import unittest
from unittest.mock import patch

from src.eval_client.intervention_loop import RealtimePacer


class RealtimePacerTest(unittest.TestCase):
    def test_first_wait_does_not_add_a_full_period(self) -> None:
        pacer = RealtimePacer(25.0)
        with patch("src.eval_client.intervention_loop.time.monotonic", return_value=1.0), patch(
            "src.eval_client.intervention_loop.time.sleep"
        ) as sleep:
            pacer.wait()
        sleep.assert_not_called()

    def test_missed_deadline_rebases_without_sleeping(self) -> None:
        pacer = RealtimePacer(25.0)
        pacer._deadline = 1.0
        with patch("src.eval_client.intervention_loop.time.monotonic", return_value=1.10), patch(
            "src.eval_client.intervention_loop.time.sleep"
        ) as sleep:
            pacer.wait()
        sleep.assert_not_called()
        self.assertEqual(pacer._deadline, 1.10)

    def test_early_frame_sleeps_only_until_existing_deadline(self) -> None:
        pacer = RealtimePacer(25.0)
        pacer._deadline = 1.0
        with patch(
            "src.eval_client.intervention_loop.time.monotonic",
            side_effect=[1.02, 1.02],
        ), patch("src.eval_client.intervention_loop.time.sleep") as sleep:
            pacer.wait()
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.02)


if __name__ == "__main__":
    unittest.main()
