"""Write an explicit RoboDojo replay snapshot back into a live environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .replay_bundle import ReplayFrame


@dataclass(frozen=True)
class RestoreSummary:
    robots: int
    rigid_objects: int
    articulations: int


def _row_like(value: Any, reference: Any):
    array = np.asarray(value)
    if hasattr(reference, "new_tensor"):
        return reference.new_tensor(array).reshape(1, -1)
    return np.asarray(array).reshape(1, -1)


def _vector_like(value: Any, reference: Any):
    array = np.asarray(value)
    if hasattr(reference, "new_tensor"):
        return reference.new_tensor(array).reshape(-1)
    return np.asarray(array).reshape(-1)


def _stored_like(value: np.ndarray, previous: Any):
    if hasattr(previous, "new_tensor"):
        return previous.new_tensor(value)
    if isinstance(previous, np.ndarray):
        return np.asarray(value, dtype=previous.dtype).copy()
    if isinstance(previous, tuple):
        return tuple(np.asarray(value).tolist())
    return np.asarray(value).tolist()


def _scene_object_by_descriptor(layout_manager: Any, descriptor: dict[str, Any]):
    wanted_labels = {str(label) for label in descriptor.get("labels", [])}
    candidates: list[str] = []
    get_instance_name = getattr(layout_manager, "get_instance_name", None)
    if callable(get_instance_name):
        for label in sorted(wanted_labels):
            instance_name = get_instance_name(env_idx=0, label=label)
            if instance_name:
                candidates.append(str(instance_name))
    for type_records in getattr(layout_manager, "object_records_by_type", {}).values():
        records_by_env = getattr(type_records, "layout_records_by_env", [])
        if not records_by_env:
            continue
        for record in records_by_env[0]:
            labels = record.get("label", [])
            labels = labels if isinstance(labels, list) else [labels]
            if wanted_labels.intersection(str(label) for label in labels):
                candidates.append(str(record.get("inst_name", "")))
    candidates = sorted({name for name in candidates if name})
    if len(candidates) == 1:
        obj = layout_manager.get_scene_object(env_idx=0, inst_name=candidates[0])
        if obj is not None:
            return obj
    instance_name = str(descriptor.get("instance_name", ""))
    if not wanted_labels and instance_name:
        obj = layout_manager.get_scene_object(env_idx=0, inst_name=instance_name)
        if obj is not None:
            return obj
    raise ValueError(
        f"cannot resolve restored scene object instance={instance_name!r} labels={sorted(wanted_labels)!r}"
    )


def _restore_robot(articulation: Any, state: dict[str, np.ndarray], prefix: str) -> None:
    data = articulation.data
    articulation.write_root_pose_to_sim(_row_like(state[f"{prefix}.root_pose_w"], data.root_pose_w))
    articulation.write_root_velocity_to_sim(_row_like(state[f"{prefix}.root_vel_w"], data.root_vel_w))
    articulation.write_joint_state_to_sim(
        _row_like(state[f"{prefix}.joint_pos"], data.joint_pos),
        _row_like(state[f"{prefix}.joint_vel"], data.joint_vel),
    )
    for field, method_name in (
        ("joint_pos_target", "set_joint_position_target"),
        ("joint_vel_target", "set_joint_velocity_target"),
        ("joint_effort_target", "set_joint_effort_target"),
    ):
        key = f"{prefix}.{field}"
        method = getattr(articulation, method_name, None)
        reference = getattr(data, field, None)
        if key in state and callable(method) and reference is not None:
            method(_row_like(state[key], reference))


def _restore_scene_object(
    obj: Any,
    state: dict[str, np.ndarray],
    prefix: str,
    *,
    articulation: bool,
) -> None:
    pose = np.asarray(state[f"{prefix}.local_pose"]).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"{prefix}.local_pose must have seven values")
    current_position, current_orientation = obj.get_local_pose()
    obj.set_local_pose(
        _vector_like(pose[:3], current_position),
        _vector_like(pose[3:], current_orientation),
    )
    linear = obj.get_linear_velocity()
    angular = obj.get_angular_velocity()
    obj.set_linear_velocity(_vector_like(state[f"{prefix}.linear_velocity_w"], linear))
    obj.set_angular_velocity(_vector_like(state[f"{prefix}.angular_velocity_w"], angular))
    if articulation:
        current_joint_pos = obj.get_joint_positions()
        current_joint_vel = obj.get_joint_velocities()
        obj.set_joint_positions(_vector_like(state[f"{prefix}.joint_pos"], current_joint_pos))
        obj.set_joint_velocities(_vector_like(state[f"{prefix}.joint_vel"], current_joint_vel))


def restore_replay_frame(
    task_env: Any,
    replay: ReplayFrame,
    *,
    _prime_camera_pipeline: bool = True,
    _refresh_camera_pipeline: bool = True,
    _reset_episode: bool = True,
) -> RestoreSummary:
    """Restore physical state for one already-reset, single-environment task."""

    if int(getattr(task_env, "num_envs", 0)) != 1:
        raise ValueError("simulator state restore supports exactly one environment")
    manifest = replay.manifest
    if manifest.get("profile") != "robodojo_rigid_articulation_v1":
        raise ValueError(f"unsupported replay profile: {manifest.get('profile')!r}")
    state = replay.state

    robot_manager = task_env.robot_manager
    current_robots = {
        str(getattr(robot, "arm_name", f"robot_{index}")): articulation
        for index, (robot, articulation) in enumerate(
            zip(robot_manager.robot_list, robot_manager.robot_key, strict=True)
        )
    }
    robot_descriptors = sorted(manifest.get("robots", []), key=lambda item: int(item["slot"]))
    for descriptor in robot_descriptors:
        slot = int(descriptor["slot"])
        name = str(descriptor.get("name", ""))
        try:
            articulation = current_robots[name]
        except KeyError as exc:
            raise ValueError(f"restored robot is missing from the scene: {name!r}") from exc
        _restore_robot(articulation, state, f"robot.{slot:03d}")

    layout_manager = task_env.scene_manager.layout_manager
    rigid_descriptors = sorted(manifest.get("rigid_objects", []), key=lambda item: int(item["slot"]))
    for descriptor in rigid_descriptors:
        slot = int(descriptor["slot"])
        obj = _scene_object_by_descriptor(layout_manager, descriptor)
        _restore_scene_object(obj, state, f"rigid.{slot:03d}", articulation=False)

    articulation_descriptors = sorted(
        manifest.get("articulations", []), key=lambda item: int(item["slot"])
    )
    for descriptor in articulation_descriptors:
        slot = int(descriptor["slot"])
        obj = _scene_object_by_descriptor(layout_manager, descriptor)
        _restore_scene_object(obj, state, f"articulation.{slot:03d}", articulation=True)

    controls = robot_manager.control_manager.prev_control[0]
    for descriptor in manifest.get("control_targets", []):
        slot = int(descriptor["slot"])
        name = str(descriptor["name"])
        if name not in controls or "position" not in controls[name]:
            raise ValueError(f"restored control target is missing: {name!r}")
        key = f"control.{slot:03d}.position"
        controls[name]["position"] = _stored_like(np.asarray(state[key]), controls[name]["position"])
        if "velocity" in controls[name]:
            controls[name]["velocity"] = _stored_like(
                np.zeros_like(np.asarray(state[key])), controls[name]["velocity"]
            )

    if _reset_episode:
        task_env.take_action_cnt[0] = 0
        task_env.success[0] = True
        task_env.end_flag[0] = False
    else:
        task_env.take_action_cnt[0] = int(
            np.asarray(state.get("task.take_action_count", task_env.take_action_cnt[0])).item()
        )
        task_env.success[0] = bool(
            np.asarray(state.get("task.success", task_env.success[0])).item()
        )
        task_env.end_flag[0] = bool(
            np.asarray(state.get("task.end_flag", task_env.end_flag[0])).item()
        )
        reward_manager = getattr(task_env, "reward_manager", None)
        if reward_manager is not None:
            for state_name, attribute_name in (
                ("reward.score_completed_count", "score_completed_count"),
                ("reward.final_score_completed_count", "final_score_completed_count"),
            ):
                values = getattr(reward_manager, attribute_name, None)
                if state_name in state and values is not None:
                    values[0] = int(np.asarray(state[state_name]).item())

    sim = getattr(task_env, "sim", None)
    scene = getattr(sim, "scene", None)
    if scene is None:
        scene = getattr(task_env, "scene", None)
    write_data = getattr(scene, "write_data_to_sim", None)
    if callable(write_data):
        write_data()

    # Direct state writes update PhysX, but do not by themselves refresh the
    # articulation kinematics/Fabric data consumed by the viewport and RTX
    # cameras.  A normal action step eventually performs that synchronization,
    # which used to make the restored pose appear only after intervention had
    # started.  ``forward`` and a zero-dt scene update synchronize rendering
    # without advancing physics away from the selected replay frame.
    simulation_context = getattr(sim, "sim", None)
    if simulation_context is None:
        raise RuntimeError("restored environment has no SimulationContext")

    # Updating articulation DOF tensors is not enough to recompute every
    # dynamic link transform consumed by Fabric/RTX.  Finish one PhysX
    # simulate/fetch cycle at dt=0.  Isaac Sim explicitly supports this path;
    # it publishes the restored transforms without integrating gravity or
    # advancing the selected replay timestamp.
    from omni.physics.core import get_physics_simulation_interface

    physics_sim_view = getattr(simulation_context, "physics_sim_view", None)
    update_articulations = getattr(
        physics_sim_view,
        "update_articulations_kinematic",
        None,
    )
    forward = getattr(simulation_context, "forward", None)
    update = getattr(scene, "update", None)
    get_obs = getattr(task_env, "get_obs", None)
    current_time = float(getattr(simulation_context, "current_time", 0.0))
    physics_simulation = get_physics_simulation_interface()

    # Publish the exact state once without advancing simulation time.  The two
    # normal hold cycles below are what drain the tiled-camera pipeline.
    physics_simulation.simulate(0.0, current_time)
    physics_simulation.fetch_results()
    if callable(update_articulations):
        update_articulations()
    if callable(update):
        update(dt=0.0)
    if callable(forward):
        forward()
    if _refresh_camera_pipeline:
        task_env.render()
        if callable(get_obs):
            get_obs()

    if _prime_camera_pipeline and _refresh_camera_pipeline:
        # Replicator's tiled RGB annotators publish two frames behind physics.
        # Prime them through the normal RoboDojo control path before hardware
        # alignment or recording, then restore the selected snapshot a second
        # time.  The discarded hold actions are never recorded, and the second
        # restore resets state and episode counters exactly while the camera
        # pipeline contains the restored scene instead of reset-at-zero images.
        state_keys = (
            "left_arm_joint_state",
            "left_ee_joint_state",
            "right_arm_joint_state",
            "right_ee_joint_state",
        )
        for _ in range(2):
            if not callable(get_obs):
                raise RuntimeError("restored environment cannot produce a hold observation")
            prime_obs = get_obs()
            prime_action = {
                key: np.asarray(prime_obs["state"][key], dtype=np.float64).copy()
                for key in state_keys
            }
            task_env.take_action(prime_action)
        return restore_replay_frame(
            task_env,
            replay,
            _prime_camera_pipeline=False,
            _refresh_camera_pipeline=True,
            _reset_episode=_reset_episode,
        )
    return RestoreSummary(
        robots=len(robot_descriptors),
        rigid_objects=len(rigid_descriptors),
        articulations=len(articulation_descriptors),
    )
