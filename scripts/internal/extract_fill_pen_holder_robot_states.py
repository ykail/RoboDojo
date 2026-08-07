"""Extract paired real robot states for fill_pen_holder VQA rendering.

The extractor reads only the original LeRobot table and creates a separate,
provenance-preserving Parquet pool. It never rewrites the source dataset.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.internal.vqa.robot_state_pool import POOL_COLUMNS, RobotStatePoolError, state_pool_record

DEFAULT_INPUT = REPO_ROOT / "data" / "fill_pen_holder" / "data" / "chunk-000" / "file-000.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "fill_pen_holder" / "meta" / "vqa_robot_state_pool.parquet"
LOGGER = logging.getLogger("extract_fill_pen_holder_robot_states")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source LeRobot frame Parquet file.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Destination paired-state pool Parquet file."
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the destination pool and its report files.")
    return parser.parse_args()


def _companion_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(f"{output.stem}_manifest.json"),
        output.with_name(f"{output.stem}_report.json"),
    )


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if not args.input.is_file():
        raise FileNotFoundError(f"source state table does not exist: {args.input}")
    manifest_path, report_path = _companion_paths(args.output)
    outputs = (args.output, manifest_path, report_path)
    if any(path.exists() for path in outputs) and not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(f"refusing to replace existing state-pool outputs: {existing}")

    source = pq.read_table(
        args.input, columns=["observation.state", "timestamp", "frame_index", "episode_index", "index"]
    )
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for row in source.to_pylist():
        try:
            accepted.append(state_pool_record(row))
        except RobotStatePoolError as error:
            rejected.append({"source_index": int(row["index"]), "reason": str(error)})
    if not accepted:
        raise RobotStatePoolError("all source state rows were invalid")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("source_index", pa.int64()),
            pa.field("source_episode_index", pa.int64()),
            pa.field("source_frame_index", pa.int64()),
            pa.field("source_timestamp", pa.float64()),
            pa.field("left_ee_pose_wxyz", pa.list_(pa.float32())),
            pa.field("left_gripper", pa.float32()),
            pa.field("right_ee_pose_wxyz", pa.list_(pa.float32())),
            pa.field("right_gripper", pa.float32()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(accepted, schema=schema), args.output)
    manifest_path.write_text(
        json.dumps(
            {
                "extractor": "scripts/internal/extract_fill_pen_holder_robot_states.py",
                "input": str(args.input),
                "output": str(args.output),
                "columns": list(POOL_COLUMNS),
                "source_rows": source.num_rows,
                "accepted_rows": len(accepted),
                "rejected_rows": len(rejected),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(json.dumps({"rejected": rejected}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s paired robot states to %s", len(accepted), args.output)


if __name__ == "__main__":
    main()
