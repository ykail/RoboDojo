import unittest

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


if __name__ == "__main__":
    unittest.main()
