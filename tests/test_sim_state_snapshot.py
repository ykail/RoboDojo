from types import SimpleNamespace
import unittest

import numpy as np

from src.eval_client.sim_state_snapshot import SimulatorStateSnapshotter


class _SceneObject:
    def __init__(self, prim_path, pose, *, joints=None):
        self._prim_path = prim_path
        self.pose = np.asarray(pose, dtype=np.float32)
        self.linear = np.zeros(3, dtype=np.float32)
        self.angular = np.zeros(3, dtype=np.float32)
        self.joints = None if joints is None else np.asarray(joints, dtype=np.float32)
        self.dof_names = [] if joints is None else ["lever_joint"]

    def get_local_pose(self):
        return self.pose[:3], self.pose[3:]

    def get_linear_velocity(self):
        return self.linear

    def get_angular_velocity(self):
        return self.angular

    def get_joint_positions(self):
        return self.joints

    def get_joint_velocities(self):
        return np.zeros_like(self.joints)


class _LayoutManager:
    def __init__(self, objects):
        self.objects = objects
        self.instance_type_by_env = [
            {"bread-inst": "rigid", "toaster-inst": "articulation", "shelf": "geometry"}
        ]
        self.saved_layouts = [{"Rigid": {"bread": [{"inst_name": "bread-inst"}]}}]
        records = SimpleNamespace(
            layout_records_by_env=[
                [
                    {"inst_name": "bread-inst", "label": "bread_0"},
                    {"inst_name": "toaster-inst", "label": "toaster"},
                ]
            ]
        )
        self.object_records_by_type = {"Rigid": records}

    def get_scene_object(self, env_idx, inst_name):
        self.assert_env = env_idx
        return self.objects[inst_name]


def _fake_env():
    robot_data = SimpleNamespace(
        root_pose_w=np.asarray([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float32),
        root_vel_w=np.zeros((1, 6), dtype=np.float32),
        joint_pos=np.arange(8, dtype=np.float32)[None, :],
        joint_vel=np.zeros((1, 8), dtype=np.float32),
        joint_pos_target=np.ones((1, 8), dtype=np.float32),
        joint_vel_target=np.zeros((1, 8), dtype=np.float32),
        joint_effort_target=None,
    )
    robot_articulation = SimpleNamespace(
        data=robot_data,
        cfg=SimpleNamespace(prim_path="/World/envs/env_.*/robot0"),
        joint_names=[f"joint{i}" for i in range(8)],
    )
    robot = SimpleNamespace(
        arm_name="left_arm",
        robot_name="x5",
        arm_joint_indices=list(range(6)),
        gripper_joint_indices=[6, 7],
    )
    objects = {
        "bread-inst": _SceneObject(
            "/World/envs/env_0/Rigid/bread-inst",
            [0.1, 0.2, 0.3, 1, 0, 0, 0],
        ),
        "toaster-inst": _SceneObject(
            "/World/envs/env_0/Articulation/toaster-inst",
            [0.4, 0.5, 0.6, 1, 0, 0, 0],
            joints=[0.25],
        ),
    }
    layout_manager = _LayoutManager(objects)
    func_parser = SimpleNamespace(
        pre_state=[{"bread-inst": {"pose": np.zeros(7, dtype=np.float32)}}],
        robot_origin_endpose=[{"left_arm": np.zeros(7, dtype=np.float32)}],
        joint_ratio_transition_state=[{}],
    )
    reward_manager = SimpleNamespace(
        func_parser=func_parser,
        score_completed_count=[1],
        final_score_completed_count=[0],
    )
    control_manager = SimpleNamespace(
        prev_control=[
            {
                "left_arm_joint_state": {
                    "position": np.arange(6, dtype=np.float32),
                    "velocity": np.zeros(6, dtype=np.float32),
                }
            }
        ]
    )
    return SimpleNamespace(
        num_envs=1,
        obs_manager=SimpleNamespace(collect_freq=25),
        robot_manager=SimpleNamespace(
            robot_list=[robot],
            robot_key=[robot_articulation],
            control_manager=control_manager,
        ),
        scene_manager=SimpleNamespace(layout_manager=layout_manager),
        seed_manager=SimpleNamespace(),
        env_seeds=[7],
        env_origins=np.zeros((1, 3), dtype=np.float32),
        sim=SimpleNamespace(
            unwrapped=SimpleNamespace(
                physics_dt=0.004, _sim_step_counter=123, common_step_counter=45
            )
        ),
        dt=0.004,
        take_action_cnt=[3],
        success=[True],
        end_flag=[False],
        reward_manager=reward_manager,
    )


class SimulatorStateSnapshotterTest(unittest.TestCase):
    def test_make_toast_style_inventory_and_numeric_snapshot(self):
        env = _fake_env()
        snapshotter = SimulatorStateSnapshotter(env)

        state = snapshotter.capture(3)

        self.assertEqual(snapshotter.manifest["fps"], 25)
        self.assertEqual(len(snapshotter.manifest["robots"]), 1)
        self.assertEqual(len(snapshotter.manifest["rigid_objects"]), 1)
        self.assertEqual(len(snapshotter.manifest["articulations"]), 1)
        self.assertEqual(state["frame.index"].item(), 3)
        self.assertAlmostEqual(state["frame.timestamp_s"].item(), 0.12)
        np.testing.assert_allclose(
            state["rigid.000.local_pose"],
            [0.1, 0.2, 0.3, 1, 0, 0, 0],
        )
        np.testing.assert_allclose(state["articulation.000.joint_pos"], [0.25])
        self.assertEqual(state["robot.000.joint_pos"].shape, (8,))
        np.testing.assert_array_equal(
            state["control.000.position"], np.arange(6, dtype=np.float32)
        )
        self.assertEqual(
            snapshotter.manifest["control_targets"][0]["name"],
            "left_arm_joint_state",
        )
        self.assertTrue(snapshotter.layout_sha256.startswith("sha256:"))

    def test_nonfinite_physical_state_is_rejected(self):
        env = _fake_env()
        snapshotter = SimulatorStateSnapshotter(env)
        env.scene_manager.layout_manager.objects["bread-inst"].pose[0] = np.nan

        with self.assertRaisesRegex(ValueError, "non-finite"):
            snapshotter.capture(0)

    def test_unsupported_moving_object_types_fail_closed(self):
        env = _fake_env()
        env.scene_manager.layout_manager.instance_type_by_env[0]["bread-inst"] = "dynamic"

        with self.assertRaisesRegex(NotImplementedError, "do not yet support"):
            SimulatorStateSnapshotter(env)


if __name__ == "__main__":
    unittest.main()
