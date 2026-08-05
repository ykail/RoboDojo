from types import SimpleNamespace
import unittest

import numpy as np

from src.eval_client.replay_bundle import ReplayFrame
from src.eval_client.sim_state_restore import restore_replay_frame


class _RobotArticulation:
    def __init__(self):
        self.data = SimpleNamespace(
            root_pose_w=np.zeros((1, 7), dtype=np.float32),
            root_vel_w=np.zeros((1, 6), dtype=np.float32),
            joint_pos=np.zeros((1, 2), dtype=np.float32),
            joint_vel=np.zeros((1, 2), dtype=np.float32),
            joint_pos_target=np.zeros((1, 2), dtype=np.float32),
            joint_vel_target=np.zeros((1, 2), dtype=np.float32),
            joint_effort_target=None,
        )

    def write_root_pose_to_sim(self, value):
        self.data.root_pose_w = np.asarray(value)

    def write_root_velocity_to_sim(self, value):
        self.data.root_vel_w = np.asarray(value)

    def write_joint_state_to_sim(self, position, velocity):
        self.data.joint_pos = np.asarray(position)
        self.data.joint_vel = np.asarray(velocity)

    def set_joint_position_target(self, value):
        self.data.joint_pos_target = np.asarray(value)

    def set_joint_velocity_target(self, value):
        self.data.joint_vel_target = np.asarray(value)


class _SceneObject:
    def __init__(self, *, joints=False):
        self.pose = np.zeros(7, dtype=np.float32)
        self.linear = np.zeros(3, dtype=np.float32)
        self.angular = np.zeros(3, dtype=np.float32)
        self.joint_pos = np.zeros(1, dtype=np.float32) if joints else None
        self.joint_vel = np.zeros(1, dtype=np.float32) if joints else None

    def get_local_pose(self):
        return self.pose[:3], self.pose[3:]

    def set_local_pose(self, position, orientation):
        self.pose = np.concatenate((position, orientation))

    def get_linear_velocity(self):
        return self.linear

    def set_linear_velocity(self, value):
        self.linear = np.asarray(value)

    def get_angular_velocity(self):
        return self.angular

    def set_angular_velocity(self, value):
        self.angular = np.asarray(value)

    def get_joint_positions(self):
        return self.joint_pos

    def set_joint_positions(self, value):
        self.joint_pos = np.asarray(value)

    def get_joint_velocities(self):
        return self.joint_vel

    def set_joint_velocities(self, value):
        self.joint_vel = np.asarray(value)


class SimStateRestoreTest(unittest.TestCase):
    def test_writes_robot_objects_control_and_task_state(self):
        robot = _RobotArticulation()
        bread = _SceneObject()
        toaster = _SceneObject(joints=True)
        objects = {"bread-inst": bread, "toaster-inst": toaster}
        names_by_label = {"bread_0": "bread-inst", "toaster": "toaster-inst"}
        layout_manager = SimpleNamespace(
            get_scene_object=lambda env_idx, inst_name: objects.get(inst_name),
            get_instance_name=lambda env_idx, label: names_by_label.get(label),
            object_records_by_type={},
        )
        scene = SimpleNamespace(writes=0)
        scene.write_data_to_sim = lambda: setattr(scene, "writes", scene.writes + 1)
        env = SimpleNamespace(
            num_envs=1,
            robot_manager=SimpleNamespace(
                robot_list=[SimpleNamespace(arm_name="left_arm")],
                robot_key=[robot],
                control_manager=SimpleNamespace(
                    prev_control=[
                        {
                            "left_arm_joint_state": {
                                "position": [0.0, 0.0],
                                "velocity": [1.0, 1.0],
                            }
                        }
                    ]
                ),
            ),
            scene_manager=SimpleNamespace(layout_manager=layout_manager),
            sim=SimpleNamespace(scene=scene),
            take_action_cnt=[0],
            success=[True],
            end_flag=[False],
            reward_manager=SimpleNamespace(score_completed_count=[0], final_score_completed_count=[0]),
            renders=0,
        )
        env.render = lambda: setattr(env, "renders", env.renders + 1)
        state = {
            "robot.000.root_pose_w": np.asarray([0, 0, 0, 1, 0, 0, 0]),
            "robot.000.root_vel_w": np.arange(6),
            "robot.000.joint_pos": np.asarray([0.4, 0.5]),
            "robot.000.joint_vel": np.asarray([0.1, 0.2]),
            "robot.000.joint_pos_target": np.asarray([0.6, 0.7]),
            "robot.000.joint_vel_target": np.asarray([0.0, 0.0]),
            "rigid.000.local_pose": np.asarray([1, 2, 3, 1, 0, 0, 0]),
            "rigid.000.linear_velocity_w": np.asarray([1, 0, 0]),
            "rigid.000.angular_velocity_w": np.asarray([0, 1, 0]),
            "articulation.000.local_pose": np.asarray([4, 5, 6, 1, 0, 0, 0]),
            "articulation.000.linear_velocity_w": np.zeros(3),
            "articulation.000.angular_velocity_w": np.zeros(3),
            "articulation.000.joint_pos": np.asarray([0.8]),
            "articulation.000.joint_vel": np.asarray([0.3]),
            "control.000.position": np.asarray([0.6, 0.7]),
            "task.take_action_count": np.asarray(12),
            "task.success": np.asarray(False),
            "task.end_flag": np.asarray(False),
            "reward.score_completed_count": np.asarray(2),
            "reward.final_score_completed_count": np.asarray(1),
        }
        replay = ReplayFrame(
            dataset_root=None,
            episode_index=2,
            frame_index=12,
            timestamp_s=0.48,
            frame_count=100,
            task_name="make_toast",
            env_config="arx_x5",
            eval_seed=1,
            layout_id=7,
            saved_layout={},
            manifest={
                "profile": "robodojo_rigid_articulation_v1",
                "robots": [{"slot": 0, "name": "left_arm"}],
                "rigid_objects": [{"slot": 0, "instance_name": "bread-inst", "labels": ["bread_0"]}],
                "articulations": [{"slot": 0, "instance_name": "toaster-inst", "labels": ["toaster"]}],
                "control_targets": [{"slot": 0, "name": "left_arm_joint_state"}],
            },
            state=state,
        )

        summary = restore_replay_frame(env, replay)

        self.assertEqual(summary.robots, 1)
        self.assertEqual(summary.rigid_objects, 1)
        self.assertEqual(summary.articulations, 1)
        np.testing.assert_allclose(robot.data.joint_pos, [[0.4, 0.5]])
        np.testing.assert_allclose(bread.pose, [1, 2, 3, 1, 0, 0, 0])
        np.testing.assert_allclose(toaster.joint_pos, [0.8])
        self.assertEqual(
            env.robot_manager.control_manager.prev_control[0]["left_arm_joint_state"]["position"],
            [0.6, 0.7],
        )
        self.assertEqual(
            env.robot_manager.control_manager.prev_control[0]["left_arm_joint_state"]["velocity"],
            [0.0, 0.0],
        )
        self.assertEqual(env.take_action_cnt, [0])
        self.assertEqual(env.success, [True])
        self.assertEqual(env.end_flag, [False])
        self.assertEqual(scene.writes, 1)
        self.assertEqual(env.renders, 1)


if __name__ == "__main__":
    unittest.main()
