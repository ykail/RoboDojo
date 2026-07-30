"""Connect to a strict policy-v1 server and verify HELLO provenance only."""

from __future__ import annotations

import argparse
import json

from src.eval_client.policy_runtime import PolicyV1EvalBridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-checkpoint-id", required=True)
    parser.add_argument("--expected-checkpoint-digest", default="")
    parser.add_argument("--expected-code-revision", default="")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--connect-timeout-s", type=float, default=30.0)
    parser.add_argument("--close-timeout-s", type=float, default=10.0)
    args = parser.parse_args()

    bridge = PolicyV1EvalBridge(
        args.url,
        connect_timeout_s=args.connect_timeout_s,
        close_timeout_s=args.close_timeout_s,
        expected_checkpoint_id=args.expected_checkpoint_id,
        expected_checkpoint_digest=args.expected_checkpoint_digest or None,
        expected_code_revision=args.expected_code_revision or None,
        require_clean=args.require_clean,
    )
    try:
        print(json.dumps(bridge.provenance.to_payload(), sort_keys=True))
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
