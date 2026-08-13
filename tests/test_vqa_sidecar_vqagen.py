"""Structural tests for the vqa_gen sidecar extensions (int_list / bbox_list)."""

from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

from vqa_gen.vqa.sidecar import (
    SidecarWriter,
    ValidationError,
    base_record,
    validate_record,
)


def _base(**overrides):
    defaults = dict(
        sample_id="sample",
        task_name="make_kong",
        question_family="demo",
        ego_image_reference="images/sample.png",
        image_width=640,
        image_height=480,
        prompt_text="Question?",
        world_state_valid=True,
        image_answerable=True,
        visibility_status="visible",
        gt_source="test",
    )
    defaults.update(overrides)
    return base_record(**defaults)


class VqaGenSidecarTests(unittest.TestCase):
    def test_int_list_accepts_empty_and_ordered_values(self) -> None:
        record = _base(answer_type="int_list", answer_int_list=[2, 5, 8])
        self.assertEqual(validate_record(record)["answer_int_list"], [2, 5, 8])
        none_record = _base(answer_type="int_list", answer_int_list=[])
        self.assertEqual(validate_record(none_record)["answer_int_list"], [])

    def test_int_list_rejects_malformed_values(self) -> None:
        for bad in ([1.5], [True], "2,5", 3, [None]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    validate_record(_base(answer_type="int_list", answer_int_list=bad))

    def test_bbox_list_accepts_boxes_and_empty(self) -> None:
        record = _base(
            answer_type="bbox_list",
            answer_bbox_list_yxyx_norm=[[0.25, 0.5, 0.75, 0.875], [0.125, 0.25, 0.375, 0.625]],
            target_view="ego",
            bbox_definition="visible_tight",
        )
        self.assertEqual(
            validate_record(record)["answer_bbox_list_yxyx_norm"],
            [[0.25, 0.5, 0.75, 0.875], [0.125, 0.25, 0.375, 0.625]],
        )
        none_record = _base(
            answer_type="bbox_list", answer_bbox_list_yxyx_norm=[], target_view="ego", bbox_definition="visible_tight"
        )
        self.assertEqual(validate_record(none_record)["answer_bbox_list_yxyx_norm"], [])

    def test_bbox_list_rejects_malformed_boxes(self) -> None:
        for bad in ([[0.3, 0.1, 0.2, 0.4]], [[0.1, 0.2]], [[0.1, 0.2, 0.3, 1.5]], [[-1.0, 0.0, 0.5, 0.5]]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    validate_record(
                        _base(
                            answer_type="bbox_list",
                            answer_bbox_list_yxyx_norm=bad,
                            target_view="ego",
                            bbox_definition="visible_tight",
                        )
                    )

    def test_bbox_list_requires_ego_visible_tight_metadata(self) -> None:
        with self.assertRaises(ValidationError):
            validate_record(_base(answer_type="bbox_list", answer_bbox_list_yxyx_norm=[[0.1, 0.2, 0.3, 0.4]]))
        with self.assertRaises(ValidationError):
            validate_record(
                _base(
                    answer_type="bbox_list",
                    answer_bbox_list_yxyx_norm=[[0.1, 0.2, 0.3, 0.4]],
                    target_view="ego",
                    bbox_definition="amodal",
                )
            )

    def test_exactly_one_answer_field_is_enforced(self) -> None:
        with self.assertRaises(ValidationError):
            validate_record(_base(answer_type="int_list", answer_int_list=[], answer_int=0))

    def test_parquet_roundtrip_preserves_list_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = SidecarWriter(Path(directory) / "sidecar")
            writer.prepare_images_dir()
            writer.add(
                _base(
                    sample_id="indices",
                    question_family="fallen_tile_indices",
                    answer_type="int_list",
                    answer_int_list=[2, 5, 8],
                )
            )
            writer.add(
                _base(
                    sample_id="boxes",
                    question_family="missing_matching_tile_bboxes",
                    answer_type="bbox_list",
                    answer_bbox_list_yxyx_norm=[[0.25, 0.5, 0.75, 0.875]],
                    target_view="ego",
                    bbox_definition="visible_tight",
                )
            )
            writer.write({"collector": "test"})
            table = pq.read_table(Path(directory) / "sidecar" / "annotations.parquet")
            self.assertIn("answer_int_list", set(table.column_names))
            self.assertIn("answer_bbox_list_yxyx_norm", set(table.column_names))
            rows = {row["sample_id"]: row for row in table.to_pylist()}
            self.assertEqual(rows["indices"]["answer_int_list"], [2, 5, 8])
            self.assertEqual(rows["boxes"]["answer_bbox_list_yxyx_norm"], [[0.25, 0.5, 0.75, 0.875]])


if __name__ == "__main__":
    unittest.main()
