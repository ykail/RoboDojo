from __future__ import annotations

from types import SimpleNamespace
import unittest

from src.eval_client.sim_runtime_profile import apply_x5_cuda_pipeline


class X5SimRuntimeProfileTest(unittest.TestCase):
    def test_enabled_x5_profile_selects_cuda_and_fabric(self) -> None:
        cfg = SimpleNamespace(sim={"device": "cpu", "use_fabric": False})

        selected = apply_x5_cuda_pipeline(
            cfg,
            control_mode="x5_policy_joint_intervention",
            device_id=2,
            environ={"ROBODOJO_X5_CUDA_PIPELINE": "1"},
        )

        self.assertEqual(selected, "cuda:0")
        self.assertEqual(cfg.sim["device"], "cuda:0")
        self.assertIs(cfg.sim["use_fabric"], True)

    def test_other_modes_are_never_changed(self) -> None:
        cfg = SimpleNamespace(sim={"device": "cpu", "use_fabric": False})

        selected = apply_x5_cuda_pipeline(
            cfg,
            control_mode="policy",
            device_id=0,
            environ={"ROBODOJO_X5_CUDA_PIPELINE": "1"},
        )

        self.assertIsNone(selected)
        self.assertEqual(cfg.sim, {"device": "cpu", "use_fabric": False})

    def test_disabled_profile_preserves_stock_pipeline(self) -> None:
        cfg = SimpleNamespace(sim={"device": "cpu", "use_fabric": False})

        selected = apply_x5_cuda_pipeline(
            cfg,
            control_mode="x5_policy_joint_intervention",
            device_id=0,
            environ={"ROBODOJO_X5_CUDA_PIPELINE": "0"},
        )

        self.assertIsNone(selected)
        self.assertEqual(cfg.sim, {"device": "cpu", "use_fabric": False})


if __name__ == "__main__":
    unittest.main()
