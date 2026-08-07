"""Tests for the read-only VQA sidecar viewer."""

from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from vqa_viewer.visualize_vqa import DatasetNotLoadedError, DatasetStore, VqaDataset, _answer_text


class VqaViewerTests(unittest.TestCase):
    def test_answer_text_uses_boolean_words(self) -> None:
        self.assertEqual(_answer_text({"answer_bool": True}), "yes")
        self.assertEqual(_answer_text({"answer_bool": False}), "no")
        self.assertEqual(_answer_text({"answer_point_xy_norm": [0.2, 0.8]}), "[0.2, 0.8]")

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
