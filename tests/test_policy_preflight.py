from contextlib import redirect_stdout
from io import StringIO
import sys
import unittest
from unittest.mock import MagicMock, patch

from scripts.RoboDojo import preflight_policy_v1


class PolicyPreflightTest(unittest.TestCase):
    def test_cli_verifies_exact_hello_identity_and_closes(self):
        provenance = MagicMock()
        provenance.to_payload.return_value = {
            "checkpoint_id": "official/59999",
            "dirty": False,
        }
        bridge = MagicMock(provenance=provenance)
        argv = [
            "preflight_policy_v1.py",
            "--url",
            "ws://127.0.0.1:18080",
            "--expected-checkpoint-id",
            "official/59999",
            "--expected-checkpoint-digest",
            "sha256:" + "a" * 64,
            "--expected-code-revision",
            "b" * 40,
            "--require-clean",
        ]

        output = StringIO()
        with (
            patch.object(sys, "argv", argv),
            patch.object(preflight_policy_v1, "PolicyV1EvalBridge", return_value=bridge) as factory,
            redirect_stdout(output),
        ):
            preflight_policy_v1.main()

        factory.assert_called_once_with(
            "ws://127.0.0.1:18080",
            connect_timeout_s=30.0,
            close_timeout_s=10.0,
            expected_checkpoint_id="official/59999",
            expected_checkpoint_digest="sha256:" + "a" * 64,
            expected_code_revision="b" * 40,
            require_clean=True,
        )
        bridge.close.assert_called_once_with()
        self.assertIn('"checkpoint_id": "official/59999"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
