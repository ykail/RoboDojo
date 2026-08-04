"""Capture replay-oriented simulator state without importing LeRobot.

Snapshots are deliberately numeric, flat dictionaries.  This lets the CPU
writer validate a fixed schema and store safe ``allow_pickle=False`` NPZ files
while the accompanying manifest maps stable numeric slots back to scene prims.
The first supported profile covers RoboDojo rigid bodies, passive scene
articulations, and IsaacLab robot articulations (including ``make_toast``).
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import time
from typing import Any

import numpy as np

from .rollout_collection import canonical_json_bytes


SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_PROFILE = "robodojo_rigid_articulation_v1"


def _as_numeric_array(value: Any, *, label: str, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=dtype)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{label} must be numeric, got dtype {array.dtype}")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(array)


def _float32(value: Any, *, label: str) -> np.ndarray:
    return _as_numeric_array(value, label=label, dtype=np.float32)


def _first_env(value: Any, *, label: str) -> np.ndarray:
    array = _float32(value, label=label)
    if array.ndim == 0:
        return array
    if array.shape[0] < 1:
        raise ValueError(f"{label} has no environment row")
    return np.ascontiguousarray(array[0])


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("replay metadata contains a non-finite float")
        return value
    if hasattr(value, "detach") or hasattr(value, "numpy"):
        return _as_numeric_array(value, label="replay metadata").tolist()
    if isinstance(value, np.ndarray):
        return _as_numeric_array(value, label="replay metadata").tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    # OmegaConf containers expose ``items`` but are not always ``dict``.
    if hasattr(value, "items"):
        return {str(key): _json_safe(item) for key, item in value.items()}
    raise TypeError(f"replay metadata contains unsupported type {type(value).__name__}")


def _object_prim_path(obj: Any) -> str:
    for attribute in ("_prim_path", "prim_path", "usd_prim_path"):
        value = getattr(obj, attribute, None)
        if value:
            return str(value)
    raise ValueError(f"scene object {obj!r} has no stable prim path")


def _local_pose(obj: Any, *, label: str) -> np.ndarray:
    position, orientation = obj.get_local_pose()
    position_array = _float32(position, label=f"{label}.position").reshape(-1)
    orientation_array = _float32(
        orientation, label=f"{label}.orientation"
    ).reshape(-1)
    if position_array.size != 3 or orientation_array.size != 4:
        raise ValueError(
            f"{label} local pose must be position(3)+quaternion(4), got "
            f"{position_array.size}+{orientation_array.size}"
        )
    return np.concatenate((position_array, orientation_array)).astype(
        np.float32, copy=False
    )


def _labels_by_instance(layout_manager: Any, env_idx: int) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    records = getattr(layout_manager, "object_records_by_type", {})
    for type_records in records.values():
        by_env = getattr(type_records, "layout_records_by_env", [])
        if env_idx >= len(by_env):
            continue
        for record in by_env[env_idx]:
            instance_name = record.get("inst_name")
            label = record.get("label")
            if instance_name and label is not None:
                labels = label if isinstance(label, list) else [label]
                result.setdefault(str(instance_name), []).extend(
                    str(item) for item in labels
                )
    return result


class SimulatorStateSnapshotter:
    """Stable inventory plus one numeric snapshot per policy control frame."""

    def __init__(self, task_env: Any):
        if int(getattr(task_env, "num_envs", 0)) != 1:
            raise ValueError("simulator replay snapshots support exactly one environment")
        self.task_env = task_env
        self.fps = int(task_env.obs_manager.collect_freq)
        if self.fps <= 0:
            raise ValueError("snapshot FPS must be positive")
        self._started_ns = time.monotonic_ns()
        self._robots: list[tuple[Any, dict[str, Any]]] = []
        self._rigid: list[tuple[Any, dict[str, Any]]] = []
        self._articulations: list[tuple[Any, dict[str, Any]]] = []
        self._control_names: list[str] = []
        self._build_inventory()

        layout_manager = task_env.scene_manager.layout_manager
        saved_layouts = getattr(layout_manager, "saved_layouts", None)
        saved_layout = saved_layouts[0] if saved_layouts else None
        if saved_layout is None:
            env_seeds = getattr(task_env, "env_seeds", None)
            if not env_seeds:
                raise ValueError("cannot capture replay metadata without a layout id")
            saved_layout = task_env.seed_manager.get_seed_scene_info(int(env_seeds[0]))
        self.saved_layout = _json_safe(deepcopy(saved_layout))
        self.layout_sha256 = "sha256:" + hashlib.sha256(
            canonical_json_bytes(self.saved_layout)
        ).hexdigest()

        sim = getattr(task_env, "sim", None)
        unwrapped = getattr(sim, "unwrapped", sim)
        self.manifest = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "profile": SNAPSHOT_PROFILE,
            "frame_semantics": "pre_action_observation_t_and_action_t",
            "fps": self.fps,
            "physics_dt": float(
                getattr(unwrapped, "physics_dt", getattr(task_env, "dt", 0.0))
            ),
            "env_origin": _json_safe(task_env.env_origins[0]),
            "robots": [descriptor for _, descriptor in self._robots],
            "rigid_objects": [descriptor for _, descriptor in self._rigid],
            "articulations": [
                descriptor for _, descriptor in self._articulations
            ],
            "control_targets": [
                {"slot": index, "name": name}
                for index, name in enumerate(self._control_names)
            ],
            "initial_task_state": self._initial_task_state(),
        }

    def _build_inventory(self) -> None:
        robot_manager = self.task_env.robot_manager
        seen_robot_prims: set[str] = set()
        for index, (robot, articulation) in enumerate(
            zip(robot_manager.robot_list, robot_manager.robot_key, strict=True)
        ):
            prim_path = str(
                getattr(articulation, "cfg", None).prim_path
                if getattr(getattr(articulation, "cfg", None), "prim_path", None)
                else getattr(articulation, "prim_path", f"robot-{index}")
            )
            if prim_path in seen_robot_prims:
                raise ValueError(f"duplicate robot prim in replay inventory: {prim_path}")
            seen_robot_prims.add(prim_path)
            data = articulation.data
            available_targets = [
                field
                for field in (
                    "joint_pos_target",
                    "joint_vel_target",
                    "joint_effort_target",
                )
                if getattr(data, field, None) is not None
            ]
            descriptor = {
                "slot": index,
                "name": str(getattr(robot, "arm_name", f"robot_{index}")),
                "robot_name": str(getattr(robot, "robot_name", "")),
                "prim_path": prim_path,
                "joint_names": [
                    str(name) for name in getattr(articulation, "joint_names", [])
                ],
                "arm_joint_indices": [
                    int(item) for item in getattr(robot, "arm_joint_indices", [])
                ],
                "gripper_joint_indices": [
                    int(item) for item in getattr(robot, "gripper_joint_indices", [])
                ],
                "captured_targets": available_targets,
            }
            self._robots.append((articulation, descriptor))

        control_manager = getattr(robot_manager, "control_manager", None)
        previous_controls = getattr(control_manager, "prev_control", None)
        if not isinstance(previous_controls, list) or not previous_controls:
            raise ValueError("replay snapshot requires initialized control_manager.prev_control")
        current_controls = previous_controls[0]
        if not isinstance(current_controls, dict) or not current_controls:
            raise ValueError("replay snapshot requires non-empty previous control targets")
        self._control_names = sorted(str(name) for name in current_controls)
        for name in self._control_names:
            value = current_controls[name]
            if not isinstance(value, dict) or "position" not in value:
                raise ValueError(f"previous control target {name!r} has no position")
            _float32(value["position"], label=f"control target {name}.position")

        layout_manager = self.task_env.scene_manager.layout_manager
        labels = _labels_by_instance(layout_manager, 0)
        active_types = getattr(layout_manager, "instance_type_by_env", [{}])[0]
        seen_scene_prims: set[str] = set()
        for instance_name, object_type in sorted(active_types.items()):
            normalized_type = str(object_type).lower()
            if normalized_type in {"dynamic", "garment", "fluid"}:
                raise NotImplementedError(
                    "replay snapshots do not yet support moving scene object "
                    f"type {normalized_type!r} (instance={instance_name!r})"
                )
            if normalized_type not in {"rigid", "articulation"}:
                continue
            obj = layout_manager.get_scene_object(
                env_idx=0, inst_name=str(instance_name)
            )
            if obj is None:
                raise ValueError(
                    f"active replay object cannot be resolved: {instance_name}"
                )
            prim_path = _object_prim_path(obj)
            if prim_path in seen_scene_prims:
                raise ValueError(f"duplicate scene prim in replay inventory: {prim_path}")
            seen_scene_prims.add(prim_path)
            descriptor = {
                "instance_name": str(instance_name),
                "labels": labels.get(str(instance_name), []),
                "object_type": normalized_type,
                "prim_path": prim_path,
            }
            if normalized_type == "articulation":
                descriptor["slot"] = len(self._articulations)
                descriptor["dof_names"] = [
                    str(name) for name in getattr(obj, "dof_names", [])
                ]
                self._articulations.append((obj, descriptor))
            else:
                descriptor["slot"] = len(self._rigid)
                self._rigid.append((obj, descriptor))

    def _initial_task_state(self) -> dict[str, Any]:
        reward_manager = getattr(self.task_env, "reward_manager", None)
        func_parser = getattr(reward_manager, "func_parser", None)
        control_manager = getattr(self.task_env.robot_manager, "control_manager", None)
        return _json_safe(
            {
                "func_parser_pre_state": getattr(func_parser, "pre_state", None),
                "func_parser_robot_origin_endpose": getattr(
                    func_parser, "robot_origin_endpose", None
                ),
                "func_parser_joint_ratio_transition_state": getattr(
                    func_parser, "joint_ratio_transition_state", None
                ),
                "control_prev": getattr(control_manager, "prev_control", None),
            }
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "record_sim_state": True,
            "replay_manifest": deepcopy(self.manifest),
            "replay_saved_layout": deepcopy(self.saved_layout),
            "replay_layout_sha256": self.layout_sha256,
        }

    def capture(self, frame_index: int) -> dict[str, np.ndarray]:
        if frame_index < 0:
            raise ValueError("frame index must be non-negative")
        env = self.task_env
        sim = getattr(env, "sim", None)
        unwrapped = getattr(sim, "unwrapped", sim)
        state: dict[str, np.ndarray] = {
            "frame.index": np.asarray(frame_index, dtype=np.int64),
            "frame.timestamp_s": np.asarray(frame_index / self.fps, dtype=np.float64),
            "frame.wall_elapsed_s": np.asarray(
                (time.monotonic_ns() - self._started_ns) / 1e9,
                dtype=np.float64,
            ),
            "simulation.physics_step": np.asarray(
                int(getattr(unwrapped, "_sim_step_counter", -1)), dtype=np.int64
            ),
            "simulation.common_step": np.asarray(
                int(getattr(unwrapped, "common_step_counter", -1)), dtype=np.int64
            ),
            "task.take_action_count": np.asarray(
                int(env.take_action_cnt[0]), dtype=np.int64
            ),
            "task.success": np.asarray(bool(env.success[0]), dtype=np.bool_),
            "task.end_flag": np.asarray(bool(env.end_flag[0]), dtype=np.bool_),
        }

        reward_manager = getattr(env, "reward_manager", None)
        if reward_manager is not None:
            state["reward.score_completed_count"] = np.asarray(
                int(reward_manager.score_completed_count[0]), dtype=np.int64
            )
            state["reward.final_score_completed_count"] = np.asarray(
                int(reward_manager.final_score_completed_count[0]), dtype=np.int64
            )

        for index, (articulation, descriptor) in enumerate(self._robots):
            del descriptor
            data = articulation.data
            prefix = f"robot.{index:03d}"
            state[f"{prefix}.root_pose_w"] = _first_env(
                data.root_pose_w, label=f"{prefix}.root_pose_w"
            )
            state[f"{prefix}.root_vel_w"] = _first_env(
                data.root_vel_w, label=f"{prefix}.root_vel_w"
            )
            state[f"{prefix}.joint_pos"] = _first_env(
                data.joint_pos, label=f"{prefix}.joint_pos"
            )
            state[f"{prefix}.joint_vel"] = _first_env(
                data.joint_vel, label=f"{prefix}.joint_vel"
            )
            for target_name in (
                "joint_pos_target",
                "joint_vel_target",
                "joint_effort_target",
            ):
                target = getattr(data, target_name, None)
                if target is not None:
                    state[f"{prefix}.{target_name}"] = _first_env(
                        target, label=f"{prefix}.{target_name}"
                    )

        current_controls = env.robot_manager.control_manager.prev_control[0]
        if set(current_controls) != set(self._control_names):
            raise ValueError("previous control target names changed within an episode")
        for index, name in enumerate(self._control_names):
            state[f"control.{index:03d}.position"] = _float32(
                current_controls[name]["position"],
                label=f"control.{index:03d}.position",
            ).reshape(-1)

        for index, (obj, descriptor) in enumerate(self._rigid):
            del descriptor
            prefix = f"rigid.{index:03d}"
            state[f"{prefix}.local_pose"] = _local_pose(obj, label=prefix)
            state[f"{prefix}.linear_velocity_w"] = _float32(
                obj.get_linear_velocity(), label=f"{prefix}.linear_velocity_w"
            ).reshape(-1)
            state[f"{prefix}.angular_velocity_w"] = _float32(
                obj.get_angular_velocity(), label=f"{prefix}.angular_velocity_w"
            ).reshape(-1)

        for index, (obj, descriptor) in enumerate(self._articulations):
            del descriptor
            prefix = f"articulation.{index:03d}"
            state[f"{prefix}.local_pose"] = _local_pose(obj, label=prefix)
            state[f"{prefix}.linear_velocity_w"] = _float32(
                obj.get_linear_velocity(), label=f"{prefix}.linear_velocity_w"
            ).reshape(-1)
            state[f"{prefix}.angular_velocity_w"] = _float32(
                obj.get_angular_velocity(), label=f"{prefix}.angular_velocity_w"
            ).reshape(-1)
            state[f"{prefix}.joint_pos"] = _float32(
                obj.get_joint_positions(), label=f"{prefix}.joint_pos"
            ).reshape(-1)
            state[f"{prefix}.joint_vel"] = _float32(
                obj.get_joint_velocities(), label=f"{prefix}.joint_vel"
            ).reshape(-1)

        return state
