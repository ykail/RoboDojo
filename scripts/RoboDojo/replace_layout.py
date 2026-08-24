#!/usr/bin/env python3
"""Copy environment appearance blocks from one eval layout into another task.

By default this script replaces the top-level Room, Table, Ground, and
Background blocks. These are the fields that control the room variant, table
material, ground material, and HDR background in saved eval layouts.

The original layouts are never overwritten. When --write is set, outputs are
written next to the target layouts as:

    <target-task><output-suffix>_<layout-id>.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT_ROOT = PROJECT_ROOT / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5"
DEFAULT_FIELDS = ("Room", "Table", "Ground", "Background")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def natural_seed_key(path: Path) -> tuple[int, str]:
    if path.name.isdigit():
        return (0, f"{int(path.name):08d}")
    return (1, path.name)


def get_seed_dirs(layout_root: Path, seeds: Iterable[str] | None) -> list[Path]:
    if seeds:
        seed_dirs = [layout_root / str(seed) for seed in seeds]
    else:
        seed_dirs = sorted((p for p in layout_root.iterdir() if p.is_dir()), key=natural_seed_key)

    missing = [p for p in seed_dirs if not p.is_dir()]
    if missing:
        raise FileNotFoundError("Missing seed dirs: " + ", ".join(str(p) for p in missing))
    return seed_dirs


def collect_task_layouts(seed_dir: Path, task_name: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"{re.escape(task_name)}_(\d+)\.json$")

    def layout_id(path: Path) -> int:
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected layout name for task {task_name}: {path}")
        return int(match.group(1))

    paths = sorted(
        (p for p in seed_dir.glob(f"{task_name}_*.json") if pattern.fullmatch(p.name)),
        key=layout_id,
    )
    return [(layout_id(path), path) for path in paths]


def source_layout_path(
    layout_root: Path,
    target_seed: str,
    source_seed: str | None,
    source_task: str,
    source_id: int,
) -> Path:
    seed = source_seed if source_seed is not None else target_seed
    return layout_root / str(seed) / f"{source_task}_{source_id}.json"


def copy_selected_fields(source: dict, fields: list[str], allow_missing_fields: bool) -> dict:
    missing = [field for field in fields if field not in source]
    if missing and not allow_missing_fields:
        raise KeyError("Source layout missing fields: " + ", ".join(missing))
    return {field: copy.deepcopy(source[field]) for field in fields if field in source}


def replace_fields(target: dict, replacements: dict) -> list[str]:
    changed = []
    for field, value in replacements.items():
        if target.get(field) != value:
            target[field] = copy.deepcopy(value)
            changed.append(field)
    return changed


def output_layout_path(target_path: Path, target_task: str, target_id: int, output_suffix: str) -> Path:
    if not output_suffix:
        raise ValueError("--output-suffix cannot be empty")
    if "/" in output_suffix or "\\" in output_suffix:
        raise ValueError("--output-suffix must be a filename suffix, not a path")
    return target_path.with_name(f"{target_task}_{output_suffix}_{target_id}.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace one task's layout backgrounds/environment appearance with "
            "the blocks from a specified source layout."
        )
    )
    parser.add_argument(
        "--layout-root",
        type=Path,
        default=DEFAULT_LAYOUT_ROOT,
        help=f"Root containing per-seed layout directories. Default: {DEFAULT_LAYOUT_ROOT}",
    )
    parser.add_argument(
        "--target-task",
        required=True,
        help="Target layout filename prefix, for example sweep_blocks or sweep_blocks_random.",
    )
    parser.add_argument(
        "--source-task",
        required=True,
        help="Source layout filename prefix, for example sweep_blocks_random.",
    )
    parser.add_argument(
        "--source-id",
        type=int,
        default=None,
        help="Source layout id, for example 0 for <source-task>_0.json. Required unless --match-layout-id is set.",
    )
    parser.add_argument(
        "--match-layout-id",
        action="store_true",
        help=(
            "Use each target layout's id as the source id. For example, "
            "target_task_7.json reads <source-task>_7.json."
        ),
    )
    parser.add_argument(
        "--source-seed",
        default=None,
        help=(
            "Seed folder to read the source layout from. Omit this to use the "
            "same seed folder as each target layout."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Target seed folders to process. Omit to process every directory under --layout-root.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(DEFAULT_FIELDS),
        help=(
            "Top-level fields to replace. Default: Room Table Ground Background. "
            "Use '--fields Background' to replace only the HDR background block."
        ),
    )
    parser.add_argument(
        "--allow-missing-fields",
        action="store_true",
        help="Skip fields that are missing from the source layout instead of failing.",
    )
    parser.add_argument(
        "--skip-missing-source",
        action="store_true",
        help="Skip target seed folders where the source layout does not exist.",
    )
    parser.add_argument(
        "--output-suffix",
        required=True,
        help=(
            "Suffix appended to the target task name for generated layouts. "
            "Example: --target-task sweep_blocks --output-suffix _randbg "
            "writes sweep_blocks_randbg_0.json."
        ),
    )
    parser.add_argument(
        "--skip-existing-output",
        action="store_true",
        help="Skip targets whose suffixed output file already exists.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually create suffixed output files. Without this flag the script only prints a dry-run summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source_id is None and not args.match_layout_id:
        raise SystemExit("--source-id is required unless --match-layout-id is set.")

    layout_root = args.layout_root.resolve()
    seed_dirs = get_seed_dirs(layout_root, args.seeds)
    source_cache: dict[Path, dict] = {}

    total_targets = 0
    total_changed = 0
    total_unchanged = 0
    total_skipped_targets = 0
    total_outputs = 0
    missing_source_count = 0

    mode = "WRITE" if args.write else "DRY RUN"
    print(f"[{mode}] layout_root={layout_root}")
    source_id_label = "matching target ids" if args.match_layout_id else str(args.source_id)
    print(
        f"[{mode}] target_task={args.target_task} "
        f"source_task={args.source_task} source_id={source_id_label} "
        f"output_suffix={args.output_suffix} fields={','.join(args.fields)}"
    )

    for seed_dir in seed_dirs:
        target_layouts = collect_task_layouts(seed_dir, args.target_task)
        if not target_layouts:
            print(f"[skip] {seed_dir.name}: no target layouts for task {args.target_task}")
            continue

        # When matching layout IDs, only process IDs present on both sides.
        # This lets a shorter source or target set determine the output count.
        if args.match_layout_id:
            source_seed_dir = layout_root / str(
                args.source_seed if args.source_seed is not None else seed_dir.name
            )
            source_ids = {
                layout_id for layout_id, _ in collect_task_layouts(source_seed_dir, args.source_task)
            }
            original_target_count = len(target_layouts)
            target_layouts = [
                (target_id, target_path)
                for target_id, target_path in target_layouts
                if target_id in source_ids
            ]
            if len(target_layouts) < original_target_count:
                print(
                    f"[skip] {seed_dir.name}: processing {len(target_layouts)} matching IDs "
                    f"(target={original_target_count}, source={len(source_ids)})"
                )
            if not target_layouts:
                continue

        for target_id, target_path in target_layouts:
            total_targets += 1
            source_id = target_id if args.match_layout_id else args.source_id
            if source_id is None:
                raise AssertionError("source_id was not resolved")
            source_path = source_layout_path(
                layout_root=layout_root,
                target_seed=seed_dir.name,
                source_seed=args.source_seed,
                source_task=args.source_task,
                source_id=source_id,
            )
            if not source_path.exists():
                message = f"Source layout does not exist for target {target_path}: {source_path}"
                if args.skip_missing_source:
                    print(f"[skip] {message}")
                    missing_source_count += 1
                    total_skipped_targets += 1
                    continue
                raise FileNotFoundError(message)

            if source_path not in source_cache:
                source_cache[source_path] = copy_selected_fields(
                    read_json(source_path),
                    fields=args.fields,
                    allow_missing_fields=args.allow_missing_fields,
                )
                source_label = source_path.relative_to(layout_root)
                print(f"[source] seed={seed_dir.name} using {source_label}")
            replacements = source_cache[source_path]

            output_path = output_layout_path(
                target_path=target_path,
                target_task=args.target_task,
                target_id=target_id,
                output_suffix=args.output_suffix,
            )
            output_label = output_path.relative_to(layout_root)
            output_exists = output_path.exists()
            if output_exists and args.skip_existing_output:
                print(f"[skip] output exists: {output_label}")
                total_skipped_targets += 1
                continue
            if output_exists and args.write:
                raise FileExistsError(
                    f"Output already exists and will not be overwritten: {output_path}. "
                    "Use --skip-existing-output to skip it."
                )

            target_layout = read_json(target_path)
            changed_fields = replace_fields(target_layout, replacements)
            target_label = target_path.relative_to(layout_root)
            if changed_fields:
                total_changed += 1
                change_label = ",".join(changed_fields)
            else:
                total_unchanged += 1
                change_label = "unchanged"

            exists_note = " [output exists]" if output_exists else ""
            print(f"[output] {target_label} -> {output_label}: {change_label}{exists_note}")
            if args.write:
                write_json(output_path, target_layout)
                total_outputs += 1
            elif not output_exists:
                total_outputs += 1

    print(
        f"[done] targets={total_targets} outputs={total_outputs} changed={total_changed} "
        f"unchanged={total_unchanged} skipped={total_skipped_targets} "
        f"missing_sources={missing_source_count}"
    )
    if not args.write:
        print("[dry-run] Re-run with --write to create suffixed output files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
