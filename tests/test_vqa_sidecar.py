"""Fast structural tests for synthetic VQA sidecar utilities."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq

from scripts.internal.vqa.overlay import numbered_overlay
from scripts.internal.vqa.sidecar import (
    SidecarWriter,
    ValidationError,
    VisibilityThresholds,
    base_record,
    bbox_from_mask,
    classify_mask_visibility,
    validate_record,
)


class VqaSidecarTests(unittest.TestCase):
    def test_visible_bbox_uses_pixel_exterior_edges(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:5, 4:8] = True
        self.assertEqual(bbox_from_mask(mask), [0.2, 0.2, 0.4, 0.5])

    def test_thin_object_visibility_threshold(self) -> None:
        thresholds = VisibilityThresholds(12, 6, 0.03, 0.5)
        mask = np.zeros((20, 20), dtype=bool)
        mask[8, 4:16] = True
        status, _, _ = classify_mask_visibility(mask, thresholds)
        self.assertEqual(status, "visible")

    def test_validator_requires_matching_answer_discriminator(self) -> None:
        valid = base_record(
            sample_id="ok",
            task_name="demo",
            question_family="demo_boolean",
            ego_image_reference="images/ok.png",
            image_width=640,
            image_height=480,
            prompt_text="Is this valid? Answer yes or no.",
            answer_type="boolean",
            answer_bool=True,
            world_state_valid=True,
            image_answerable=True,
            visibility_status="visible",
            gt_source="test",
        )
        self.assertTrue(validate_record(valid)["answer_bool"])
        valid["answer_text"] = "yes"
        with self.assertRaises(ValidationError):
            validate_record(valid)

    def test_overlay_is_in_frame_and_does_not_cover_protected_point(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = numbered_overlay(image, {"1": (50, 50), "2": (20, 70)}, protected_points_xy=[(50, 20)])
        self.assertEqual(set(result.mark_boxes_xyxy), {"1", "2"})
        for x0, y0, x1, y1 in result.mark_boxes_xyxy.values():
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x1, 100)
            self.assertLessEqual(y1, 100)
            self.assertFalse(x0 <= 50 <= x1 and y0 <= 20 <= y1)

    def test_overlay_leader_line_keeps_clear_of_protected_point(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        protected = np.asarray((50.0, 65.0))
        clearance = 6
        result = numbered_overlay(
            image,
            {"1": (50, 80)},
            protected_points_xy=[protected],
            protected_point_clearance_px=clearance,
        )
        start_x, start_y, end_x, end_y = result.leader_segments_xyxy["1"]
        start = np.asarray((start_x, start_y), dtype=np.float64)
        end = np.asarray((end_x, end_y), dtype=np.float64)
        direction = end - start
        fraction = np.clip(np.dot(protected - start, direction) / np.dot(direction, direction), 0.0, 1.0)
        self.assertGreater(float(np.linalg.norm(protected - (start + fraction * direction))), clearance)

    def test_overlay_keeps_badge_and_leader_line_off_object_mask(self) -> None:
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        tile_mask = np.zeros((120, 120), dtype=bool)
        tile_mask[50:75, 45:75] = True
        result = numbered_overlay(
            image,
            {"1": (60, 62)},
            object_masks_by_mark={"1": tile_mask},
        )
        self.assertEqual(set(result.mark_boxes_xyxy), {"1"})
        self.assertFalse(np.any(result.image[tile_mask] != image[tile_mask]))

    def test_writer_keeps_invalid_rows_in_rejected_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = SidecarWriter(Path(directory) / "sidecar")
            writer.prepare_images_dir()
            self.assertTrue(writer.prepare_audit_dir().is_dir())
            writer.add(
                base_record(
                    sample_id="accepted",
                    task_name="demo",
                    question_family="demo_boolean",
                    ego_image_reference="images/accepted.png",
                    image_width=640,
                    image_height=480,
                    prompt_text="Is this valid? Answer yes or no.",
                    answer_type="boolean",
                    answer_bool=False,
                    world_state_valid=True,
                    image_answerable=True,
                    visibility_status="visible",
                    gt_source="test",
                )
            )
            writer.add(base_record(sample_id="rejected", task_name="demo", answer_type="boolean"))
            report = writer.write({"collector": "test"})
            self.assertEqual(report["accepted_records"], 1)
            self.assertEqual(report["rejected_records"], 1)
            self.assertEqual(pq.read_table(Path(directory) / "sidecar" / "annotations.parquet").num_rows, 1)
            self.assertFalse((Path(directory) / "sidecar" / "annotations.jsonl").exists())
            rejected = pq.read_table(Path(directory) / "sidecar" / "rejected.parquet").to_pylist()
            self.assertEqual(len(rejected), 1)
            self.assertIn("prompt_text", rejected[0]["rejection_reason"])
            manifest = json.loads((Path(directory) / "sidecar" / "manifest.json").read_text())
            self.assertEqual(manifest["accepted_records"], 1)


if __name__ == "__main__":
    unittest.main()
