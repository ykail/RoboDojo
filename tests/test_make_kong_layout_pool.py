"""Tests for loading generated make_kong layout pools without Isaac Sim."""

import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from data_gen.make_kong.layout_pool import LayoutPoolError, load_layout_pool
from data_gen.make_kong.lerobot_writer import LeRobotWriter


class MakeKongLayoutPoolTests(unittest.TestCase):
    def test_loads_continuous_json_layouts_in_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for layout_id in (10, 2, 1, 0, 9, 8, 7, 6, 5, 4, 3):
                (root / f"make_kong_{layout_id}.json").write_text(
                    json.dumps({"layout_id": layout_id}), encoding="utf-8"
                )
            (root / "manifest.json").write_text("{}", encoding="utf-8")

            layouts = load_layout_pool(root)

            self.assertEqual(list(layouts), list(range(11)))
            self.assertEqual(layouts[2].path.name, "make_kong_2.json")
            self.assertEqual(layouts[2].scene_layout, {"layout_id": 2})

    def test_rejects_noncontinuous_or_invalid_json_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "make_kong_0.json").write_text("{}", encoding="utf-8")
            (root / "make_kong_2.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(LayoutPoolError):
                load_layout_pool(root)
            (root / "make_kong_1.json").write_text("not-json", encoding="utf-8")
            with self.assertRaises(LayoutPoolError):
                load_layout_pool(root)

    def test_writer_scopes_resume_jobs_by_layout_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            manifest = output_root / "meta" / "generation_manifest.parquet"
            manifest.parent.mkdir()
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "layout": 0,
                            "target_group": 0,
                            "status": "success",
                            "episode_index": 0,
                            "failure_reason": None,
                        }
                    ]
                ),
                manifest,
            )

            writer = LeRobotWriter(output_root)

            self.assertEqual(writer.terminal_jobs("eval_layout"), {(0, 0)})
            self.assertEqual(writer.terminal_jobs("/generated/layouts"), set())
            writer.record_failure(0, "/generated/layouts", 0, "test failure")
            reloaded = LeRobotWriter(output_root)
            self.assertEqual(reloaded.terminal_jobs("eval_layout"), {(0, 0)})
            self.assertEqual(reloaded.failed_jobs("/generated/layouts"), {(0, 0)})

if __name__ == "__main__":
    unittest.main()
