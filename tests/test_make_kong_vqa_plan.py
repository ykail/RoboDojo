"""Structural tests for the make_kong VQA scene planner and face vocabulary."""

import random
import unittest

from vqa_gen.make_kong.scene_plan import (
    VARIANT_A,
    VARIANT_B,
    plan_cases,
    random_discard_assignment,
    scatter_kong_group_assignment,
    select_fallen_count_stratified_scene_ids,
)
from vqa_gen.make_kong.tile_faces import FACE_NAMES, categories_for_face, face_of_category

ROW = [f"t{index}" for index in range(1, 15)]


class ScenePlanTests(unittest.TestCase):
    def test_variant_a_middle_group_coverage(self) -> None:
        cases, skipped = plan_cases(ROW, ["t4", "t5", "t6"], VARIANT_A, random.Random(0))
        self.assertEqual(skipped, {})
        self.assertEqual(
            [case.case_id for case in cases],
            [
                "f0",
                "c1_p0",
                "c1_p1",
                "c1_p2",
                "w1_s0",
                "w1_s1",
                "w1_s2",
                "c2_p01",
                "c2_p02",
                "c2_p12",
                "w2_s0",
                "w2_s1",
                "w2_s2",
                "c3",
                "w3_s0",
                "w3_s1",
                "w3_s2",
                "c3w1_L",
                "c3w1_R",
                "w4_s0",
                "w4_s1",
                "w4_s2",
                "c3w2_LL",
                "c3w2_RR",
                "c3w2_LR",
                "w5_s0",
                "w5_s1",
                "w5_s2",
            ],
        )
        counts = [len(case.fallen_labels) for case in cases]
        self.assertEqual(counts, sorted(counts))

    def test_variant_a_edge_groups_skip_infeasible_variants(self) -> None:
        left_cases, left_skipped = plan_cases(ROW, ["t1", "t2", "t3"], VARIANT_A, random.Random(0))
        self.assertEqual(set(left_skipped), {"c3w1_L", "c3w2_LL", "c3w2_LR"})
        self.assertNotIn("c3w1_L", [case.case_id for case in left_cases])

        right_cases, right_skipped = plan_cases(ROW, ["t12", "t13", "t14"], VARIANT_A, random.Random(0))
        self.assertEqual(set(right_skipped), {"c3w1_R", "c3w2_RR", "c3w2_LR"})
        self.assertNotIn("c3w1_R", [case.case_id for case in right_cases])

    def test_variant_a_wrong_sets_are_contiguous_or_neighbors(self) -> None:
        cases, _ = plan_cases(ROW, ["t4", "t5", "t6"], VARIANT_A, random.Random(0))
        for case in cases:
            if case.case_id.startswith("c3w2"):
                wrong = sorted(ROW.index(label) for label in case.wrong_fallen)
                allowed = {ROW.index("t4") - 2, ROW.index("t4") - 1, ROW.index("t6") + 1, ROW.index("t6") + 2}
                self.assertTrue(set(wrong) <= allowed, f"{case.case_id} wrong tiles must flank the group: {wrong}")
            elif case.case_id.startswith("w") and case.case_id != "w1_s":
                positions = sorted(ROW.index(label) for label in case.wrong_fallen)
                self.assertEqual(positions, list(range(positions[0], positions[-1] + 1)), case.case_id)

    def test_variant_a_wrong_sets_never_overlap_target(self) -> None:
        for target in (["t1", "t2", "t3"], ["t4", "t5", "t6"], ["t12", "t13", "t14"]):
            cases, _ = plan_cases(ROW, target, VARIANT_A, random.Random(1))
            for case in cases:
                self.assertTrue(set(case.wrong_fallen).isdisjoint(target), case.case_id)

    def test_variant_b_coverage_and_random_wrong_sets(self) -> None:
        target = ["t2", "t5", "t9"]
        cases, skipped = plan_cases(ROW, target, VARIANT_B, random.Random(0))
        self.assertEqual(skipped, {})
        case_ids = [case.case_id for case in cases]
        for suffix in ("s0", "s1", "s2"):
            for prefix in ("c3w1", "c3w2", "w1", "w2", "w3", "w4", "w5"):
                self.assertTrue(any(case_id == f"{prefix}_{suffix}" for case_id in case_ids), prefix)
        for case in cases:
            self.assertTrue(set(case.wrong_fallen).isdisjoint(target), case.case_id)

    def test_planning_is_deterministic_per_seed(self) -> None:
        first = plan_cases(ROW, ["t4", "t5", "t6"], VARIANT_B, random.Random(42))
        second = plan_cases(ROW, ["t4", "t5", "t6"], VARIANT_B, random.Random(42))
        self.assertEqual(
            [case.fallen_labels for case in first[0]], [case.fallen_labels for case in second[0]]
        )

    def test_discard_assignment_uniformly_allows_every_slot_group_pair(self) -> None:
        pairs = set()
        for seed in range(200):
            assignment = random_discard_assignment(random.Random(seed))
            self.assertEqual(sorted(assignment), [0, 1, 2, 3])
            pairs.update(enumerate(assignment))
        self.assertEqual(pairs, {(slot, group) for slot in range(4) for group in range(4)})

    def test_scatter_keeps_three_copies_of_every_group(self) -> None:
        for seed in range(50):
            scatter = scatter_kong_group_assignment(random.Random(seed))
            self.assertEqual(sorted(scatter), [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    def test_reference_bbox_selection_is_balanced_and_stable(self) -> None:
        scene_fallen_counts = {
            f"scene_{fallen_count}_{index}": fallen_count
            for fallen_count in range(6)
            for index in range(10)
        }
        first = select_fallen_count_stratified_scene_ids(scene_fallen_counts, seed=2810, samples_per_stratum=4)
        second = select_fallen_count_stratified_scene_ids(
            dict(reversed(list(scene_fallen_counts.items()))), seed=2810, samples_per_stratum=4
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(
            [sum(scene_fallen_counts[scene_id] == fallen_count for scene_id in first) for fallen_count in range(6)],
            [4, 4, 4, 4, 4, 4],
        )

    def test_reference_bbox_selection_rejects_an_underfilled_stratum(self) -> None:
        with self.assertRaisesRegex(ValueError, "fallen-count stratum 1"):
            select_fallen_count_stratified_scene_ids(
                {"zero_0": 0, "zero_1": 0, "one_0": 1}, seed=0, samples_per_stratum=2
            )


class TileFaceTests(unittest.TestCase):
    def test_face_bands_cover_every_category_exactly_once(self) -> None:
        all_categories = []
        for face_name in FACE_NAMES:
            all_categories.extend(categories_for_face(face_name))
        self.assertEqual(sorted(all_categories), list(range(42)))

    def test_face_of_category_matches_expected_bands(self) -> None:
        self.assertEqual(face_of_category(0), "Wan")
        self.assertEqual(face_of_category(8), "Wan")
        self.assertEqual(face_of_category(9), "Suo")
        self.assertEqual(face_of_category(18), "Tong")
        self.assertEqual(face_of_category(27), "Honor")
        self.assertEqual(face_of_category(34), "Bonus")
        self.assertEqual(face_of_category(41), "Bonus")


if __name__ == "__main__":
    unittest.main()
