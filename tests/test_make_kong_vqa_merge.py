"""Tests for safely merging independently rendered make_kong VQA collections."""

import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from vqa_gen.make_kong.merge_vqa_collections import merge_collections
from vqa_gen.vqa.sidecar import _parquet_schema, base_record


def _record(sample_id: str) -> dict:
    return base_record(
        sample_id=sample_id,
        task_name="make_kong",
        question_family="fallen_tile_bboxes",
        ego_image_reference=f"images/{sample_id}.png",
        image_width=640,
        image_height=480,
        prompt_text="Which tiles fell?",
        answer_type="bbox_list",
        answer_bbox_list_yxyx_norm=[],
        world_state_valid=True,
        image_answerable=True,
        visibility_status="visible",
        target_view="ego",
        bbox_definition="visible_tight",
        gt_source="test",
        scene_id=sample_id,
    )


def _write_collection(root: Path, *, layout_id: int, accepted_ids: list[str], rejected_ids: list[str]) -> None:
    (root / "images").mkdir(parents=True)
    (root / "audit").mkdir()
    accepted = [_record(sample_id) for sample_id in accepted_ids]
    rejected = [_record(sample_id) for sample_id in rejected_ids]
    for record in rejected:
        record["image_answerable"] = False
        record["rejection_reason"] = "test rejection"
    schema = _parquet_schema()
    pq.write_table(pa.Table.from_pylist(accepted, schema=schema), root / "annotations.parquet")
    pq.write_table(pa.Table.from_pylist(rejected, schema=schema), root / "rejected.parquet")
    for sample_id in [*accepted_ids, *rejected_ids]:
        (root / "images" / f"{sample_id}.png").write_bytes(b"png")
        (root / "audit" / f"{sample_id}_instance_ids.json").write_text("{}", encoding="utf-8")
    manifest = {
        "task_name": "make_kong",
        "annotation_version": "robodojo_vqa_v3",
        "source_dataset": "RoboDojo_vqa_synthetic_v3",
        "seed": 2810,
        "variants": ["a", "b"],
        "layout_ids": [layout_id],
        "target_groups": [0, 1, 2, 3],
        "coverage": {f"layout{layout_id}": {}},
        "reference_discard_bbox": {"selected_scene_ids": [f"ref_{layout_id}"]},
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class MakeKongVqaMergeTests(unittest.TestCase):
    def test_merges_parquet_artifacts_and_manifest_without_mutating_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            extension = root / "extension"
            output = root / "merged"
            _write_collection(base, layout_id=0, accepted_ids=["base_ok"], rejected_ids=["base_rejected"])
            _write_collection(extension, layout_id=100, accepted_ids=["extension_ok"], rejected_ids=["extension_rejected"])

            summary = merge_collections(base, extension, output)

            self.assertEqual(summary, {"accepted_records": 2, "rejected_records": 2})
            self.assertEqual({row["sample_id"] for row in pq.read_table(output / "annotations.parquet").to_pylist()}, {"base_ok", "extension_ok"})
            self.assertEqual(
                {row["sample_id"] for row in pq.read_table(output / "rejected.parquet").to_pylist()},
                {"base_rejected", "extension_rejected"},
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["layout_ids"], [0, 100])
            self.assertEqual(manifest["accepted_records"], 2)
            self.assertTrue((output / "images" / "base_ok.png").is_file())
            self.assertTrue((output / "audit" / "extension_ok_instance_ids.json").is_file())
            self.assertTrue((base / "manifest.json").is_file())

    def test_rejects_duplicate_sample_ids_before_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            extension = root / "extension"
            output = root / "merged"
            _write_collection(base, layout_id=0, accepted_ids=["duplicate"], rejected_ids=[])
            _write_collection(extension, layout_id=100, accepted_ids=["duplicate"], rejected_ids=[])

            with self.assertRaisesRegex(ValueError, "sample_id overlap"):
                merge_collections(base, extension, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
