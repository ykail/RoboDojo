"""Structural tests for the make_kong JSON-only layout generator."""

import json
from pathlib import Path
import tempfile
import unittest

from data_gen.make_kong.generate_layouts import SEMANTIC_TILE_GROUPS, generate_layouts


def _source_layout(offset: int) -> dict:
    records = []
    for group_index, (_, labels) in enumerate(SEMANTIC_TILE_GROUPS):
        for label in labels:
            category_idx = offset + group_index
            records.append(
                {
                    "category": "mahjong",
                    "category_idx": category_idx,
                    "label": label,
                    "default_pos": [float(len(records)), -0.15, 0.77],
                    "default_ori": [1.0, 0.0, 0.0, 0.0],
                    "relative_plane": "Table",
                    "place_tag": "straight_front",
                    "physics": {"size": [float(category_idx), 0.04, 0.02]},
                    "visual": {"asset_id": category_idx},
                    "scale": [1.0, 1.0, 1.0],
                }
            )
    return {
        "Rigid": {"mahjong": records},
        "Room": {"default": "fixed_room"},
        "Table": {"default": "fixed_table"},
        "Ground": {"default": "fixed_ground"},
        "Background": {"category_name": "fixed_background"},
    }


def _write_source_layouts(root: Path) -> dict:
    root.mkdir(parents=True)
    template = _source_layout(0)
    (root / "make_kong_0.json").write_text(json.dumps(template), encoding="utf-8")
    (root / "make_kong_1.json").write_text(json.dumps(_source_layout(8)), encoding="utf-8")
    return template


def _records_by_label(layout: dict) -> dict[str, dict]:
    return {record["label"]: record for record in layout["Rigid"]["mahjong"]}


class MakeKongLayoutGeneratorTests(unittest.TestCase):
    def test_generation_preserves_template_and_tile_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            output_root = root / "layouts"
            template = _write_source_layouts(source_root)

            manifest = generate_layouts(count=3, rng_seed=17, output_dir=output_root, source_layout_root=source_root)

            self.assertEqual(manifest["total_layouts"], 3)
            self.assertEqual([entry["file"] for entry in manifest["layouts"]], [f"make_kong_{idx}.json" for idx in range(3)])
            template_records = _records_by_label(template)
            for layout_id in range(3):
                layout = json.loads((output_root / f"make_kong_{layout_id}.json").read_text(encoding="utf-8"))
                self.assertEqual(layout["Room"], template["Room"])
                self.assertEqual(layout["Table"], template["Table"])
                records = _records_by_label(layout)
                signature = []
                for _, labels in SEMANTIC_TILE_GROUPS:
                    indices = {records[label]["category_idx"] for label in labels}
                    self.assertEqual(len(indices), 1)
                    signature.append(next(iter(indices)))
                    for label in labels:
                        self.assertEqual(records[label]["label"], template_records[label]["label"])
                        self.assertEqual(records[label]["default_pos"], template_records[label]["default_pos"])
                        self.assertEqual(records[label]["default_ori"], template_records[label]["default_ori"])
                        self.assertEqual(records[label]["physics"]["size"][0], float(records[label]["category_idx"]))
                self.assertEqual(len(set(signature)), len(signature))

    def test_seeded_generation_is_reproducible_and_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            _write_source_layouts(source_root)
            first = root / "first"
            second = root / "second"

            generate_layouts(count=2, rng_seed=9, output_dir=first, source_layout_root=source_root)
            generate_layouts(count=2, rng_seed=9, output_dir=second, source_layout_root=source_root)
            first_layouts = [(first / f"make_kong_{idx}.json").read_text(encoding="utf-8") for idx in range(2)]
            second_layouts = [(second / f"make_kong_{idx}.json").read_text(encoding="utf-8") for idx in range(2)]
            self.assertEqual(first_layouts, second_layouts)

            manifest = generate_layouts(count=3, rng_seed=10, output_dir=first, source_layout_root=source_root)
            self.assertEqual(manifest["total_layouts"], 5)
            self.assertEqual(len(manifest["generation_runs"]), 2)
            self.assertEqual(sorted(path.name for path in first.glob("make_kong_*.json")), [f"make_kong_{idx}.json" for idx in range(5)])
            signatures = [tuple(entry["tile_signature"].values()) for entry in manifest["layouts"]]
            self.assertEqual(len(signatures), len(set(signatures)))
            self.assertEqual(json.loads((first / "manifest.json").read_text(encoding="utf-8")), manifest)

    def test_existing_layout_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            _write_source_layouts(source_root)
            output_root = root / "layouts"
            generate_layouts(count=1, rng_seed=2, output_dir=output_root, source_layout_root=source_root)
            original = (output_root / "make_kong_0.json").read_text(encoding="utf-8")

            generate_layouts(count=1, rng_seed=3, output_dir=output_root, source_layout_root=source_root)

            self.assertEqual((output_root / "make_kong_0.json").read_text(encoding="utf-8"), original)
            self.assertTrue((output_root / "make_kong_1.json").is_file())


if __name__ == "__main__":
    unittest.main()
