"""Tests for the read-only VQA sidecar viewer."""

from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from vqa_viewer.visualize_vqa import (
    DatasetNotLoadedError,
    DatasetStore,
    VqaDataset,
    _answer_text,
    _answer_value,
    _serialize_answer,
    _vlm_prompt,
)


class VqaViewerTests(unittest.TestCase):
    def test_answer_text_uses_boolean_words(self) -> None:
        self.assertEqual(_answer_text({"answer_bool": True}), "yes")
        self.assertEqual(_answer_text({"answer_bool": False}), "no")
        self.assertEqual(_answer_text({"answer_point_yx_norm": [0.2, 0.8]}), "[0.2, 0.8]")

    def test_answer_text_supports_list_answers(self) -> None:
        self.assertEqual(_answer_text({"answer_int_list": [2, 5, 8]}), "[2, 5, 8]")
        self.assertEqual(_answer_text({"answer_int_list": []}), "[]")
        self.assertEqual(_answer_text({"answer_bbox_list_yxyx_norm": [[0.2, 0.1, 0.4, 0.3]]}), "[[0.2, 0.1, 0.4, 0.3]]")

    def test_serialize_answer_matches_vlm_target_format(self) -> None:
        cases = [
            ({"answer_type": "boolean", "answer_bool": True}, "yes<eos>"),
            ({"answer_type": "integer", "answer_int": 5}, "5<eos>"),
            ({"answer_type": "short_text", "answer_text": "Wan"}, "Wan<eos>"),
            (
                {"answer_type": "point2d", "answer_point_yx_norm": [0.5, 0.25]},
                "<loc0512><loc0256><eos>",
            ),
            (
                {"answer_type": "bbox2d", "answer_bbox_yxyx_norm": [0.5, 0.25, 0.875, 0.75]},
                "<loc0512><loc0256><loc0895><loc0767><eos>",
            ),
            ({"answer_type": "int_list", "answer_int_list": [2, 5, 8]}, "2;5;8<eos>"),
            ({"answer_type": "int_list", "answer_int_list": []}, "none<eos>"),
            (
                {
                    "answer_type": "bbox_list",
                    "answer_bbox_list_yxyx_norm": [
                        [0.5021, 0.4406, 0.5604, 0.4797],
                        [0.5021, 0.4781, 0.5604, 0.5156],
                    ],
                },
                "<loc0514><loc0451><loc0573><loc0491>"
                ";<loc0514><loc0489><loc0573><loc0527><eos>",
            ),
            ({"answer_type": "bbox_list", "answer_bbox_list_yxyx_norm": []}, "none<eos>"),
        ]
        for record, expected in cases:
            with self.subTest(answer_type=record["answer_type"]):
                self.assertEqual(_serialize_answer(record), expected)

    def test_vlm_prompt_assembles_the_canonical_block(self) -> None:
        prompt = _vlm_prompt({"prompt_text": "How many tiles?", "answer_type": "integer"})
        self.assertEqual(prompt, "Question: How many tiles?\nAnswer:")

    def test_dataset_query_exposes_list_answer_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "images" / "sample.png").write_bytes(b"not-a-real-image")
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "sample_id": "indices_0",
                            "question_family": "fallen_tile_indices",
                            "ego_image_reference": "images/sample.png",
                            "prompt_text": "Which tiles?",
                            "answer_type": "int_list",
                            "answer_int_list": [2, 5, 8],
                            "answer_bbox_list_yxyx_norm": None,
                            "visibility_status": "visible",
                        },
                        {
                            "sample_id": "boxes_0",
                            "question_family": "missing_matching_tile_bboxes",
                            "ego_image_reference": "images/sample.png",
                            "prompt_text": "Which boxes?",
                            "answer_type": "bbox_list",
                            "answer_int_list": None,
                            "answer_bbox_list_yxyx_norm": [
                                [0.5, 0.25, 0.875, 0.75],
                                [0.25, 0.125, 0.625, 0.375],
                            ],
                            "visibility_status": "visible",
                        },
                    ]
                ),
                root / "annotations.parquet",
            )
            dataset = VqaDataset(root)
            result = dataset.query({"limit": "12"})
            self.assertEqual(result["total"], 2)
            by_id = {row["sample_id"]: row for row in result["rows"]}
            self.assertEqual(by_id["indices_0"]["answer"], [2, 5, 8])
            self.assertEqual(by_id["indices_0"]["answer_text"], "[2, 5, 8]")
            self.assertEqual(by_id["boxes_0"]["answer"], [[0.5, 0.25, 0.875, 0.75], [0.25, 0.125, 0.625, 0.375]])
            self.assertEqual(by_id["boxes_0"]["answer_text"], "[[0.5, 0.25, 0.875, 0.75], [0.25, 0.125, 0.625, 0.375]]")

    def test_dataset_query_and_image_containment_are_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            image = root / "images" / "sample.png"
            image.write_bytes(b"not-a-real-image")
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "sample_id": "sample_0",
                            "question_family": "demo",
                            "ego_image_reference": "images/sample.png",
                            "prompt_text": "Is this visible?",
                            "answer_type": "boolean",
                            "answer_bool": True,
                            "visibility_status": "visible",
                        }
                    ]
                ),
                root / "annotations.parquet",
            )
            dataset = VqaDataset(root)
            result = dataset.query({"family": "demo", "limit": "12"})
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["rows"][0]["answer_text"], "yes")
            self.assertEqual(dataset.image_path("images/sample.png"), image)
            with self.assertRaises(FileNotFoundError):
                dataset.image_path("../sample.png")

    def test_dataset_store_starts_empty_then_loads_a_selected_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(pa.Table.from_pylist([{"sample_id": "sample_0"}]), root / "annotations.parquet")
            store = DatasetStore()
            self.assertFalse(store.summary()["loaded"])
            with self.assertRaises(DatasetNotLoadedError):
                store.require_dataset()
            summary = store.load(str(root))
            self.assertTrue(summary["loaded"])
            self.assertEqual(summary["accepted_count"], 1)


if __name__ == "__main__":
    unittest.main()
