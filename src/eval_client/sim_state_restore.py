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
    # Older replay manifests may not have labels. Instance names are safe only
    # as that compatibility fallback; RoboDojo increments their suffixes as
    # layouts are reloaded in one process.
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


def restore_replay_frame(task_env: Any, replay: ReplayFrame) -> RestoreSummary:
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

    articulation_descriptors = sorted(manifest.get("articulations", []), key=lambda item: int(item["slot"]))
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
            controls[name]["velocity"] = _stored_like(np.zeros_like(np.asarray(state[key])), controls[name]["velocity"])

    # The restored frame is a new correction boundary, not a continuation of
    # benchmark bookkeeping. Keep the physical snapshot exact while leaving
    # the environment active for the manual-control layer that will be added
    # separately.
    task_env.take_action_cnt[0] = 0
    task_env.success[0] = True
    task_env.end_flag[0] = False

    sim = getattr(task_env, "sim", None)
    scene = getattr(sim, "scene", None)
    if scene is None:
        # Keep the lightweight fake/test boundary usable without importing
        # IsaacLab; live RoboDojo environments use task_env.sim.scene.
        scene = getattr(task_env, "scene", None)
    write_data = getattr(scene, "write_data_to_sim", None)
    if callable(write_data):
        write_data()
    task_env.render()
    return RestoreSummary(
        robots=len(robot_descriptors),
        rigid_objects=len(rigid_descriptors),
        articulations=len(articulation_descriptors),
    )
