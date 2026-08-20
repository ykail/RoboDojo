"""Safely combine two non-overlapping make_kong VQA sidecar collections.

The collector writes a complete sidecar directory and deliberately does not
append to an existing collection. This tool validates two such directories and
materializes their union in a third directory, preserving both inputs.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vqa_gen.vqa.sidecar import _parquet_schema, build_report, json_dumps


PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--base-dir", type=Path, required=True, help="Existing collection, for example make_kong v4.2.")
PARSER.add_argument("--extension-dir", type=Path, required=True, help="Collection rendered from newly added layouts.")
PARSER.add_argument("--output-dir", type=Path, required=True, help="New directory receiving the merged collection.")
PARSER.add_argument("--overwrite", action="store_true", help="Replace a pre-existing output directory only.")

PARQUET_FILENAMES = ("annotations.parquet", "rejected.parquet")
ARTIFACT_DIRNAMES = ("images", "audit")


def _read_manifest(collection_dir: Path) -> dict[str, Any]:
    path = collection_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Collection manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Collection manifest must be a JSON object: {path}")
    return value


def _read_table(collection_dir: Path, filename: str) -> pa.Table:
    path = collection_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Collection annotation is missing: {path}")
    table = pq.read_table(path)
    if table.schema != _parquet_schema():
        raise ValueError(f"Unexpected Parquet schema in {path}")
    return table


def _sample_ids(table: pa.Table, label: str) -> set[str]:
    values = table.column("sample_id").to_pylist()
    if any(value is None for value in values):
        raise ValueError(f"{label} contains null sample_id values")
    sample_ids = {str(value) for value in values}
    if len(sample_ids) != len(values):
        raise ValueError(f"{label} contains duplicate sample_id values")
    return sample_ids


def _validate_collections(
    base_accepted: pa.Table,
    base_rejected: pa.Table,
    extension_accepted: pa.Table,
    extension_rejected: pa.Table,
) -> None:
    groups = {
        "base annotations": _sample_ids(base_accepted, "base annotations"),
        "base rejected": _sample_ids(base_rejected, "base rejected"),
        "extension annotations": _sample_ids(extension_accepted, "extension annotations"),
        "extension rejected": _sample_ids(extension_rejected, "extension rejected"),
    }
    labels = list(groups)
    for index, first_label in enumerate(labels):
        for second_label in labels[index + 1 :]:
            overlap = groups[first_label] & groups[second_label]
            if overlap:
                raise ValueError(
                    f"sample_id overlap between {first_label} and {second_label}: {sorted(overlap)[:5]}"
                )


def _copy_artifacts(source_dir: Path, destination_dir: Path) -> None:
    for dirname in ARTIFACT_DIRNAMES:
        source = source_dir / dirname
        if not source.is_dir():
            raise FileNotFoundError(f"Collection artifact directory is missing: {source}")
        destination = destination_dir / dirname
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(source)
            target = destination / relative_path
            if target.exists():
                raise FileExistsError(f"Artifact collision while merging: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _union_int_lists(base: dict[str, Any], extension: dict[str, Any], key: str) -> list[int] | None:
    values = [manifest.get(key) for manifest in (base, extension)]
    if any(value is None for value in values):
        return None
    if not all(isinstance(value, list) and all(isinstance(item, int) for item in value) for value in values):
        raise ValueError(f"Both manifests must use integer lists for {key}")
    return sorted({item for value in values for item in value})


def _merge_coverage(base: dict[str, Any], extension: dict[str, Any]) -> dict[str, Any] | None:
    values = [manifest.get("coverage") for manifest in (base, extension)]
    if any(value is None for value in values):
        return None
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("Both manifests must use objects for coverage")
    overlap = set(values[0]) & set(values[1])
    if overlap:
        raise ValueError(f"Manifest coverage overlaps: {sorted(overlap)[:5]}")
    return {**values[0], **values[1]}


def _merge_manifest(
    base_manifest: dict[str, Any],
    extension_manifest: dict[str, Any],
    *,
    base_dir: Path,
    extension_dir: Path,
    accepted_records: int,
    rejected_records: int,
) -> dict[str, Any]:
    for key in ("task_name", "annotation_version", "source_dataset", "seed"):
        if base_manifest.get(key) != extension_manifest.get(key):
            raise ValueError(f"Manifest values disagree for {key}")

    merged = deepcopy(base_manifest)
    for key in ("layout_ids", "target_groups"):
        value = _union_int_lists(base_manifest, extension_manifest, key)
        if value is not None:
            merged[key] = value
    variants = [*base_manifest.get("variants", []), *extension_manifest.get("variants", [])]
    if variants:
        merged["variants"] = list(dict.fromkeys(variants))
    coverage = _merge_coverage(base_manifest, extension_manifest)
    if coverage is not None:
        merged["coverage"] = coverage

    base_reference = base_manifest.get("reference_discard_bbox", {})
    extension_reference = extension_manifest.get("reference_discard_bbox", {})
    base_scene_ids = set(base_reference.get("selected_scene_ids", []))
    extension_scene_ids = set(extension_reference.get("selected_scene_ids", []))
    if base_scene_ids & extension_scene_ids:
        raise ValueError("Reference-discard manifest selections overlap")
    merged["reference_discard_bbox"] = {
        "selection_mode": "merged_collections",
        "selected_scene_count": len(base_scene_ids | extension_scene_ids),
        "selected_scene_ids": sorted(base_scene_ids | extension_scene_ids),
        "source_selections": [base_reference, extension_reference],
    }
    history = list(merged.get("merge_history", []))
    history.append(
        {
            "base_dir": str(base_dir),
            "extension_dir": str(extension_dir),
            "base_accepted_records": base_manifest.get("accepted_records"),
            "base_rejected_records": base_manifest.get("rejected_records"),
            "extension_accepted_records": extension_manifest.get("accepted_records"),
            "extension_rejected_records": extension_manifest.get("rejected_records"),
        }
    )
    merged.update(
        {
            "accepted_records": accepted_records,
            "rejected_records": rejected_records,
            "physical_annotation": "annotations.parquet",
            "rejected_annotation": "rejected.parquet",
            "merge_history": history,
            "command": " ".join(sys.argv),
        }
    )
    return merged


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    if output_dir.exists():
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return output_dir.parent / f".{output_dir.name}.merge-{uuid4().hex}"


def merge_collections(base_dir: Path, extension_dir: Path, output_dir: Path, *, overwrite: bool = False) -> dict[str, int]:
    """Validate and merge two collections into a new, atomically published directory."""

    base_dir = base_dir.resolve()
    extension_dir = extension_dir.resolve()
    output_dir = output_dir.resolve()
    if base_dir == extension_dir:
        raise ValueError("base-dir and extension-dir must be different directories")
    if output_dir in {base_dir, extension_dir}:
        raise ValueError("output-dir must differ from both input directories")

    base_manifest = _read_manifest(base_dir)
    extension_manifest = _read_manifest(extension_dir)
    base_accepted, base_rejected = (_read_table(base_dir, filename) for filename in PARQUET_FILENAMES)
    extension_accepted, extension_rejected = (_read_table(extension_dir, filename) for filename in PARQUET_FILENAMES)
    _validate_collections(base_accepted, base_rejected, extension_accepted, extension_rejected)
    accepted = pa.concat_tables([base_accepted, extension_accepted])
    rejected = pa.concat_tables([base_rejected, extension_rejected])
    manifest = _merge_manifest(
        base_manifest,
        extension_manifest,
        base_dir=base_dir,
        extension_dir=extension_dir,
        accepted_records=accepted.num_rows,
        rejected_records=rejected.num_rows,
    )
    staging_dir = _prepare_output_dir(output_dir, overwrite=overwrite)
    try:
        staging_dir.mkdir()
        _copy_artifacts(base_dir, staging_dir)
        _copy_artifacts(extension_dir, staging_dir)
        pq.write_table(accepted, staging_dir / "annotations.parquet")
        pq.write_table(rejected, staging_dir / "rejected.parquet")
        (staging_dir / "manifest.json").write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        (staging_dir / "report.json").write_text(
            json_dumps(build_report(accepted.to_pylist(), rejected.to_pylist())) + "\n", encoding="utf-8"
        )
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return {"accepted_records": accepted.num_rows, "rejected_records": rejected.num_rows}


def main() -> None:
    args = PARSER.parse_args()
    summary = merge_collections(args.base_dir, args.extension_dir, args.output_dir, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
