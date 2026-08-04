#!/usr/bin/env python3
"""Orchestrate a resumable multi-seed Kai0 policy rollout collection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_client.rollout_collection import (
    build_collection_manifest,
    committed_plan_indices,
    dataset_root,
    group_entries_by_seed,
    parse_layout_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect normal Kai0 policy rollouts as LeRobot v3 plus replayable "
            "RoboDojo simulator-state sidecars."
        )
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument(
        "--layout-plan",
        required=True,
        help="Ordered saved-layout plan, for example 0:0-69,1:0-29",
    )
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--lerobot-repo-id", required=True)
    parser.add_argument(
        "--kai0-root", default=str(PROJECT_ROOT / "third_party" / "kai0")
    )
    parser.add_argument("--kai0-python", default="")
    parser.add_argument("--policy-gpu", type=int, default=0)
    parser.add_argument("--env-gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=None,
        help=(
            "Fixed policy sampling seed for every episode. By default each "
            "segment uses its eval seed, matching eval_kai0_pi05.sh."
        ),
    )
    parser.add_argument("--lerobot-vcodec", choices=("h264", "hevc", "libsvtav1"), default="h264")
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument(
        "--max-segment-attempts",
        type=int,
        default=3,
        help=(
            "Retry a segment only when its evaluator exits cleanly but some "
            "planned layouts were unstable/uncommitted (default: 3)."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--gui", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.policy_gpu < 0 or args.env_gpu < 0:
        raise ValueError("GPU ids must be non-negative")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    if args.policy_seed is not None and not 0 <= args.policy_seed <= (1 << 32) - 1:
        raise ValueError("--policy-seed must be in [0, 2^32 - 1]")
    if args.encoder_threads <= 0 or args.max_segment_attempts <= 0:
        raise ValueError("encoder threads and segment attempts must be positive")


def _segment_command(
    args: argparse.Namespace,
    *,
    eval_seed: int,
    count: int,
    append: bool,
) -> list[str]:
    command = [
        "bash",
        str(PROJECT_ROOT / "scripts" / "RoboDojo" / "eval_kai0_pi05.sh"),
        "--task",
        args.task,
        "--checkpoint-dir",
        str(Path(args.checkpoint_dir).expanduser()),
        "--checkpoint-id",
        args.checkpoint_id,
        "--kai0-root",
        str(Path(args.kai0_root).expanduser()),
        "--eval-num",
        str(count),
        "--seed",
        str(eval_seed),
        "--policy-gpu",
        str(args.policy_gpu),
        "--env-gpu",
        str(args.env_gpu),
        "--port",
        str(args.port),
        "--control-mode",
        "policy",
        "--record-rollouts",
        "--lerobot-root",
        str(Path(args.lerobot_root).expanduser()),
        "--lerobot-repo-id",
        args.lerobot_repo_id,
        "--lerobot-vcodec",
        args.lerobot_vcodec,
        "--encoder-threads",
        str(args.encoder_threads),
    ]
    if args.policy_seed is not None:
        command.extend(["--policy-seed", str(args.policy_seed)])
    if args.kai0_python:
        command.extend(["--kai0-python", str(Path(args.kai0_python).expanduser())])
    if args.headless:
        command.append("--headless")
    if append:
        command.append("--resume")
    return command


def _scan_completed(root: Path, manifest: dict) -> set[int]:
    completed = committed_plan_indices(root, manifest)
    print(
        f"[collect_kai0_rollouts] durable progress "
        f"{len(completed)}/{len(manifest['entries'])}",
        flush=True,
    )
    return completed


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    # Resolve user paths once relative to the invocation directory. The child
    # evaluator runs with PROJECT_ROOT as cwd, so passing the original relative
    # spelling would otherwise make progress scanning and recording disagree.
    args.checkpoint_dir = str(Path(args.checkpoint_dir).expanduser().resolve())
    args.kai0_root = str(Path(args.kai0_root).expanduser().resolve())
    args.lerobot_root = str(Path(args.lerobot_root).expanduser().resolve())
    if args.kai0_python:
        # Preserve the lexical venv interpreter path. Resolving its final
        # symlink would bypass pyvenv.cfg and lose Kai0/LeRobot site-packages.
        args.kai0_python = os.path.abspath(os.path.expanduser(args.kai0_python))
    entries = parse_layout_plan(args.layout_plan)
    if len(entries) != args.episodes:
        raise ValueError(
            f"--episodes={args.episodes} but --layout-plan expands to {len(entries)}"
        )
    manifest = build_collection_manifest(
        task=args.task,
        checkpoint_id=args.checkpoint_id,
        entries=entries,
        fps=25,
        policy_seed=args.policy_seed,
    )
    root = dataset_root(args.lerobot_root, args.lerobot_repo_id)
    if root.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"dataset already exists: {root}; pass --resume to continue the same signed plan"
        )
    if root.exists() and args.resume and not (root / "meta" / "info.json").is_file():
        raise ValueError(f"existing path is not a LeRobot v3 dataset: {root}")

    completed = _scan_completed(root, manifest) if root.exists() else set()
    if len(completed) == len(entries):
        print(f"[collect_kai0_rollouts] collection already complete: {root}")
        return 0

    base_environment = os.environ.copy()
    base_environment.update(
        {
            "ROBODOJO_COLLECTION_ID": manifest["collection_id"],
            "ROBODOJO_COLLECTION_PLAN_HASH": manifest["plan_hash"],
            "ROBODOJO_COLLECTION_MANIFEST_JSON": json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        }
    )
    print(
        f"[collect_kai0_rollouts] collection={manifest['collection_id']} "
        f"plan={manifest['plan_hash']} dataset={root}",
        flush=True,
    )

    dataset_will_exist = root.exists()
    for eval_seed, original_group in group_entries_by_seed(entries):
        attempt = 0
        while True:
            if root.exists():
                completed = _scan_completed(root, manifest)
            group = [
                entry for entry in original_group if entry.plan_index not in completed
            ]
            if not group:
                break
            attempt += 1
            if attempt > args.max_segment_attempts:
                missing = [entry.plan_index for entry in group]
                raise RuntimeError(
                    f"seed {eval_seed} still has missing plan indices {missing} after "
                    f"{args.max_segment_attempts} attempt(s)"
                )

            environment = base_environment.copy()
            environment.update(
                {
                    "ROBODOJO_ROLLOUT_LAYOUT_IDS": ",".join(
                        str(entry.layout_id) for entry in group
                    ),
                    "ROBODOJO_COLLECTION_PLAN_INDEX_MAP": json.dumps(
                        {
                            str(entry.layout_id): entry.plan_index
                            for entry in group
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "ROBODOJO_RUN_ID": (
                        f"{manifest['collection_id']}-seed-{eval_seed}"
                    ),
                }
            )
            append = root.exists() or dataset_will_exist
            command = _segment_command(
                args,
                eval_seed=eval_seed,
                count=len(group),
                append=append,
            )
            print(
                f"[collect_kai0_rollouts] seed={eval_seed} "
                f"attempt={attempt}/{args.max_segment_attempts} "
                f"layouts={[entry.layout_id for entry in group]}",
                flush=True,
            )
            if args.dry_run:
                print("[dry-run] " + shlex.join(command))
                completed.update(entry.plan_index for entry in group)
                dataset_will_exist = True
                break
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"rollout segment seed={eval_seed} exited with "
                    f"status {result.returncode}; rerun with --resume after fixing it"
                )
            dataset_will_exist = True

    if args.dry_run:
        print(
            f"[dry-run] plan contains {len(entries)} episode(s); no files were written."
        )
        return 0
    completed = _scan_completed(root, manifest)
    missing = sorted(set(range(len(entries))) - completed)
    if missing:
        raise RuntimeError(f"collection stopped with missing plan indices: {missing}")
    print(
        f"[collect_kai0_rollouts] COMPLETE {len(completed)} episode(s) -> {root}",
        flush=True,
    )
    return 0


def main() -> None:
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[collect_kai0_rollouts][ERROR] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
