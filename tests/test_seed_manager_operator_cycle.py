from pathlib import Path
import tempfile
import unittest
from unittest import mock

from env.seed_manager.seed_manager import SeedManager


class SeedManagerOperatorCycleTest(unittest.TestCase):
    @staticmethod
    def make_manager(seed_list):
        manager = SeedManager.__new__(SeedManager)
        manager.num_envs = 1
        manager.seed_list = list(seed_list)
        manager.idx = 0
        manager.ed_idx = len(seed_list)
        manager.cycle_index = 0
        manager._current_batch_seeds = None
        return manager

    def test_cyclic_seeds_rewind_and_track_cycle(self):
        manager = self.make_manager([3, 7])

        self.assertEqual(manager.get_cyclic_seeds(max_count=1), [3])
        self.assertEqual(manager.cycle_index, 0)
        self.assertEqual(manager.get_cyclic_seeds(max_count=1), [7])
        self.assertEqual(manager.get_cyclic_seeds(max_count=1), [3])
        self.assertEqual(manager.cycle_index, 1)

    def test_empty_layout_list_still_terminates(self):
        manager = self.make_manager([])

        self.assertIsNone(manager.get_cyclic_seeds(max_count=1))
        self.assertEqual(manager.cycle_index, 0)

    def test_finite_benchmark_api_does_not_rewind(self):
        manager = self.make_manager([2])

        self.assertEqual(manager.get_seeds(max_count=1), [2])
        self.assertIsNone(manager.get_seeds(max_count=1))
        self.assertEqual(manager.cycle_index, 0)

    def test_explicit_collection_selection_uses_filename_suffix_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout_dir = Path(tmp) / "Eval_Layout" / "RoboDojo" / "arx_x5" / "2"
            layout_dir.mkdir(parents=True)
            for suffix in (3, 9, 20):
                (layout_dir / f"make_toast_{suffix}.json").write_text(
                    "{}", encoding="utf-8"
                )
            manager = SeedManager(
                {
                    "num_envs": 1,
                    "task_name": "make_toast",
                    "config_name": "arx_x5",
                    "seed": 2,
                }
            )
            with (
                mock.patch("env.seed_manager.seed_manager.ASSETS_PATH", tmp),
                mock.patch("env.seed_manager.seed_manager.BENCHMARK", "RoboDojo"),
            ):
                manager.init_eval(selected_layout_ids=[20, 3])

            self.assertEqual(manager.seed_list, [20, 3])
            self.assertTrue(manager.seed_info[20]["scene_layout"].endswith("make_toast_20.json"))
            self.assertEqual(manager.get_seeds(max_count=1), [20])
            self.assertEqual(manager.get_seeds(max_count=1), [3])

    def test_explicit_collection_selection_rejects_missing_or_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout_dir = Path(tmp) / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
            layout_dir.mkdir(parents=True)
            (layout_dir / "make_toast_7.json").write_text("{}", encoding="utf-8")
            manager = SeedManager(
                {
                    "num_envs": 1,
                    "task_name": "make_toast",
                    "config_name": "arx_x5",
                    "seed": 0,
                }
            )
            with (
                mock.patch("env.seed_manager.seed_manager.ASSETS_PATH", tmp),
                mock.patch("env.seed_manager.seed_manager.BENCHMARK", "RoboDojo"),
            ):
                with self.assertRaisesRegex(ValueError, "duplicates"):
                    manager.init_eval(selected_layout_ids=[7, 7])
                with self.assertRaisesRegex(ValueError, "do not exist"):
                    manager.init_eval(selected_layout_ids=[8])


if __name__ == "__main__":
    unittest.main()
