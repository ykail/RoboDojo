import unittest

import numpy as np

from src.eval_client.sim_state_snapshot import _as_numeric_array, _first_env


class SimulatorStateSnapshotCopyTest(unittest.TestCase):
    def test_numeric_snapshot_owns_storage(self) -> None:
        live = np.arange(6, dtype=np.float32)
        snapshot = _as_numeric_array(live, label="live")
        live[:] = -1
        np.testing.assert_array_equal(snapshot, np.arange(6, dtype=np.float32))

    def test_first_environment_snapshot_owns_storage(self) -> None:
        live = np.arange(12, dtype=np.float32).reshape(2, 6)
        snapshot = _first_env(live, label="live")
        live[0] = -1
        np.testing.assert_array_equal(snapshot, np.arange(6, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
