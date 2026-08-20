"""Structural tests for the make_kong VQA A/B layout generator."""

import json
from pathlib import Path
import tempfile
import unittest

from vqa_gen.make_kong.generate_layouts import (
    KONG_GROUPS,
    LayoutGenerationError,
    generate_vqa_layouts,
    load_vqa_layout_pool,
)
from vqa_gen.make_kong.tile_faces import face_of_category

ALL_LABELS = (
    *tuple(label for group in KONG_GROUPS for label in group),
    "mahjong4_0",
    "mahjong4_1",
    "mahjong9_0",
    "mahjong5_0",
    "mahjong6_0",
    "mahjong7_0",
    "mahjong8_0",
    "other0",
    "other1",
    "other2",
)


def _source_layout(category_start: int) -> dict:
    records = []
    for index, label in enumerate(ALL_LABELS):
        category_idx = category_start + index
        records.append(
            {
                "category": "mahjong",
                "category_idx": category_idx,
                "label": label,
                "default_pos": [float(index), -0.15, 0.77],
                "default_ori": [1.0, 0.0, 0.0, 0.0],
                "relative_plane": "Table",
                "place_tag": "straight_front",
                "physics": {"size": [float(category_idx), 0.04, 0.02]},
                "visual": {"asset_id": category_idx},
            }
        )
    return {"Rigid": {"mahjong": records}}


def _write_source_layouts(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "make_kong_0.json").write_text(json.dumps(_source_layout(0)), encoding="utf-8")
    (root / "make_kong_1.json").write_text(json.dumps(_source_layout(22)), encoding="utf-8")


def _records_by_label(layout: dict) -> dict[str, dict]:
    return {record["label"]: record for record in layout["Rigid"]["mahjong"]}


class MakeKongVqaLayoutGeneratorTests(unittest.TestCase):
    def _generate(self, root: Path, count: int, rng_seed: int):
        source_root = root / "source"
        output_root = root / "layouts"
        _write_source_layouts(source_root)
        return generate_vqa_layouts(
            count=count, rng_seed=rng_seed, output_dir=output_root, source_layout_root=source_root
        )

    def test_generation_writes_ab_pairs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._generate(root, 2, 17)
            output_root = root / "layouts"
            self.assertEqual(manifest["total_layout_pairs"], 2)
            self.assertEqual(
                sorted(path.name for path in output_root.glob("make_kong_*.json")),
                ["make_kong_a_0.json", "make_kong_a_1.json", "make_kong_b_0.json", "make_kong_b_1.json"],
            )
            for variant in ("a", "b"):
                for layout_id in range(2):
                    _records_by_label(json.loads((output_root / f"make_kong_{variant}_{layout_id}.json").read_text()))

    def test_a_layouts_use_a_permutation_of_matching_group_faces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._generate(root, 3, 5)
            output_root = root / "layouts"
            discard_labels = [f"mahjong{5 + index}_0" for index in range(4)]
            for layout_id in range(3):
                records = _records_by_label(json.loads((output_root / f"make_kong_a_{layout_id}.json").read_text()))
                discard_categories = {discard: records[discard]["category_idx"] for discard in discard_labels}
                group_categories = {records[f"mahjong{index}_0"]["category_idx"] for index in range(4)}
                self.assertEqual(set(discard_categories.values()), group_categories)

    def test_b_layouts_reuse_a_pairing_and_scatter_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._generate(root, 3, 5)
            output_root = root / "layouts"
            kong_labels = tuple(label for group in KONG_GROUPS for label in group)
            discard_labels = [f"mahjong{5 + index}_0" for index in range(4)]
            for layout_id in range(3):
                a_records = _records_by_label(json.loads((output_root / f"make_kong_a_{layout_id}.json").read_text()))
                b_records = _records_by_label(json.loads((output_root / f"make_kong_b_{layout_id}.json").read_text()))
                for discard in discard_labels:
                    self.assertEqual(a_records[discard]["category_idx"], b_records[discard]["category_idx"])
                kong_categories = [b_records[label]["category_idx"] for label in kong_labels]
                group_categories = [a_records[group[0]]["category_idx"] for group in KONG_GROUPS]
                for category in group_categories:
                    self.assertEqual(kong_categories.count(category), 3)
                self.assertNotEqual(kong_categories, [a_records[group[0]]["category_idx"] for group in KONG_GROUPS])

    def test_face_distinctness_and_support_face(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._generate(root, 5, 8)
            output_root = root / "layouts"
            kong_labels = tuple(label for group in KONG_GROUPS for label in group)
            for variant in ("a", "b"):
                for layout_id in range(5):
                    records = _records_by_label(
                        json.loads((output_root / f"make_kong_{variant}_{layout_id}.json").read_text())
                    )
                    row_faces = {face_of_category(records[label]["category_idx"]) for label in kong_labels}
                    self.assertEqual(len(row_faces), 4)
                    support_face = face_of_category(records["mahjong4_0"]["category_idx"])
                    self.assertNotIn(support_face, row_faces)

    def test_seeded_generation_is_reproducible_and_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            source_root = root / "source"
            _write_source_layouts(source_root)
            generate_vqa_layouts(count=2, rng_seed=9, output_dir=first, source_layout_root=source_root)
            generate_vqa_layouts(count=2, rng_seed=9, output_dir=second, source_layout_root=source_root)
            for variant in ("a", "b"):
                self.assertEqual(
                    (first / f"make_kong_{variant}_0.json").read_text(encoding="utf-8"),
                    (second / f"make_kong_{variant}_0.json").read_text(encoding="utf-8"),
                )
            manifest = generate_vqa_layouts(count=1, rng_seed=10, output_dir=first, source_layout_root=source_root)
            self.assertEqual(manifest["total_layout_pairs"], 3)
            self.assertEqual(len(manifest["generation_runs"]), 2)

    def test_loader_rejects_non_continuous_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "layouts"
            _write_source_layouts(root / "source")
            generate_vqa_layouts(count=2, rng_seed=3, output_dir=output_root, source_layout_root=root / "source")
            (output_root / "make_kong_a_0.json").unlink()
            with self.assertRaises(LayoutGenerationError):
                load_vqa_layout_pool(output_root)


if __name__ == "__main__":
    unittest.main()
