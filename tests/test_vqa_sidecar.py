"""Fast structural tests for synthetic VQA sidecar utilities."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq

from scripts.internal.vqa.robot_state_pool import RobotStatePoolError, split_paired_state, state_pool_record
from scripts.internal.vqa.sidecar import (
    SidecarWriter,
    ValidationError,
    VisibilityThresholds,
    base_record,
    bbox_from_mask,
    classify_mask_visibility,
    validate_record,
)
from scripts.internal.vqa.task_logic import (
    adjacent_nonmatching_labels,
    fallen_labels_for_pattern,
    holder_pose_condition,
    kong_declaration_neighbor_state_reason,
    labels_for_bitmask,
    matching_tile_indices,
    ordinal,
    pen_descriptions_from_features,
)


class VqaSidecarTests(unittest.TestCase):
    def test_make_kong_tuple_and_fallen_patterns(self) -> None:
        robot_tiles = ["tile_a", "tile_b", "tile_c", "tile_d", "tile_e"]
        matching = ["tile_e", "tile_b", "tile_d"]
        self.assertEqual(matching_tile_indices(robot_tiles, matching), (2, 4, 5))
        self.assertEqual(fallen_labels_for_pattern(matching, 0b101), ["tile_e", "tile_d"])
        self.assertEqual(adjacent_nonmatching_labels(robot_tiles, ["tile_b", "tile_c", "tile_d"]), ["tile_a", "tile_e"])
        self.assertEqual(labels_for_bitmask(["tile_a", "tile_e"], 0b11), ["tile_a", "tile_e"])
        self.assertEqual(kong_declaration_neighbor_state_reason(False), "correct")
        self.assertEqual(kong_declaration_neighbor_state_reason(True), "nonmatching_fallen")
        self.assertEqual(ordinal(1), "1st")
        self.assertEqual(ordinal(12), "12th")

    def test_pen_descriptions_and_tipped_schedule(self) -> None:
        descriptions = pen_descriptions_from_features(
            {"pen_a": (10.0, 20.0), "pen_b": (30.0, 20.0)},
            {"pen_a": "black", "pen_b": "black"},
            {"left": (9.0, 20.0), "right": (31.0, 20.0)},
        )
        self.assertEqual(descriptions["pen_a"], ("the leftmost pen", "image_leftmost"))
        self.assertEqual(descriptions["pen_b"], ("the rightmost pen", "image_rightmost"))
        conditions = [holder_pose_condition(index, seed=3, layout_index=1) for index in range(16)]
        self.assertEqual(sum(tipped for tipped, _ in conditions), 8)
        self.assertEqual(sorted(yaw for tipped, yaw in conditions if tipped), [float(step * 45) for step in range(8)])

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

    def test_writer_schema_is_the_clean_image_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = SidecarWriter(Path(directory) / "sidecar")
            writer.prepare_images_dir()
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
                    answer_bool=True,
                    world_state_valid=True,
                    image_answerable=True,
                    visibility_status="visible",
                    gt_source="test",
                )
            )
            writer.write({"collector": "test"})
            columns = set(pq.read_table(Path(directory) / "sidecar" / "annotations.parquet").column_names)
            self.assertEqual(
                columns,
                {
                    "sample_id",
                    "sample_type",
                    "source_dataset",
                    "task_name",
                    "question_family",
                    "episode_index",
                    "frame_index",
                    "timestamp",
                    "ego_image_reference",
                    "image_width",
                    "image_height",
                    "prompt_text",
                    "answer_type",
                    "answer_text",
                    "answer_bool",
                    "answer_int",
                    "answer_point_xy_norm",
                    "answer_bbox_xyxy_norm",
                    "answer_aliases",
                    "world_state_valid",
                    "image_answerable",
                    "visibility_status",
                    "visible_fraction",
                    "occlusion_ratio",
                    "target_view",
                    "coordinate_space",
                    "point_definition",
                    "bbox_definition",
                    "gt_source",
                    "annotation_version",
                    "quality_score",
                    "rejection_reason",
                    "source_layout",
                    "scene_id",
                    "audit_metadata_json",
                },
            )

    def test_robot_state_pool_preserves_paired_provenance(self) -> None:
        row = {
            "observation.state": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.6, -0.1, -0.2, 0.4, 1.0, 0.0, 0.0, 0.0, 0.7],
            "index": 42,
            "episode_index": 3,
            "frame_index": 18,
            "timestamp": 0.72,
        }
        record = state_pool_record(row)
        self.assertEqual(record["source_index"], 42)
        self.assertEqual(record["source_episode_index"], 3)
        np.testing.assert_allclose(record["left_ee_pose_wxyz"], row["observation.state"][:7])
        np.testing.assert_allclose(record["right_ee_pose_wxyz"], row["observation.state"][8:15])
        with self.assertRaises(RobotStatePoolError):
            split_paired_state([0.0] * 16)

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
