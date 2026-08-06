"""Generate typed, ego-only VQA sidecar samples for ``fill_pen_holder``.

Source layouts are never modified.  Each selected layout is rendered in an
independent Isaac environment and contributes Parquet physical VQA
annotations plus clean and overlay ``cam_head`` images.  No policy rollout or
LeRobot action-data mutation is required.

Batch example (all ``fill_pen_holder`` layouts for seed 0, 30 deterministic
scene snapshots per layout):

    python scripts/internal/generate_fill_pen_holder_vqa.py \
        --headless --enable_cameras --seed 0 --max-layouts 5 \
        --scene-count 30 --scenario layout gripper_content visible_counts \
        --gripper-position-jitter 0.10 --gripper-height-jitter 0.05 \
        --output-dir ./output/RoboDojo_vqa_v1/fill_pen_holder_seed0

The output directory must be new. To deliberately replace an existing batch,
add ``--overwrite`` after reviewing the directory target.
"""

import argparse
from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import random
import sys
from typing import Any

from isaaclab.app import AppLauncher
import numpy as np
import torch
import transforms3d.quaternions as t3q

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.internal.vqa.overlay import OverlayError, numbered_overlay
from scripts.internal.vqa.sidecar import (
    SidecarWriter,
    VisibilityThresholds,
    base_record,
    bbox_from_mask,
    classify_mask_visibility,
    json_dumps,
)

DEFAULT_LAYOUT_ROOT = PROJECT_ROOT / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "RoboDojo_vqa_v1" / "fill_pen_holder"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
PEN_LABELS = ("target0", "target1", "target2", "target3")
SCENARIOS = ("layout", "gripper_content", "visible_counts")
# X5 geometry (metres). The inner finger gap is approximately 2*q + 0.0008
# for the two prismatic joints. The TCP offset is read from
# robot.gripper_bias (0.145 for X5), rather than duplicated here.
X5_FINGER_ZERO_GAP = 0.0008
GRIPPER_CLEARANCE = 0.003
HOLDER_GRASP_HEIGHT = 0.085
DEFAULT_GRIPPER_POSITION_JITTER_M = 0.10
DEFAULT_GRIPPER_HEIGHT_JITTER_M = 0.05
# DLAA is temporal. The tiled camera's first buffer after a pose write can
# still contain a prior render-product frame, so retain enough clean frames
# after the reset for both that buffer and DLAA history to settle.
RENDER_STABILIZATION_FRAMES = 12
HOLDER_VISIBILITY = VisibilityThresholds(64, 1, 0.05, 0.5)
PEN_VISIBILITY = VisibilityThresholds(12, 6, 0.03, 0.5)
# Count questions require an instance to be readily distinguishable, not just
# represented by a small visible fragment. Keep this stricter than the nib and
# gripper-object thresholds because the answer is an integer a human must count.
COUNT_PEN_VISIBILITY = VisibilityThresholds(48, 12, 0.03, 0.5)
NIB_PATCH_RADIUS_PX = 3
NIB_MIN_MATCHING_PIXELS = 3
NIB_DEPTH_TOLERANCE_M = 0.01
LOGGER = logging.getLogger("fill_pen_holder_vqa")


class AnnotationMappingError(RuntimeError):
    """The renderer could not prove a one-to-one object/mask association."""


def _configure_logging() -> None:
    """Emit collector progress without enabling verbose dependency loggers."""

    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for dependency in ("curobo", "matplotlib", "PIL"):
        logging.getLogger(dependency).setLevel(logging.WARNING)


def _reset_renderer_accumulation() -> None:
    """Discard DLAA history after direct robot/object teleports.

    Direct pose writes do not advance animation time, so DLAA cannot always
    infer that its temporal history is invalid.  Resetting the renderer makes
    the following stabilization frames independent of a previous layout or
    sample while preserving DLAA image quality.
    """

    import omni.usd

    omni.usd.get_context().reset_renderer_accumulation()


def _reset_ego_render_product(env) -> None:
    """Recreate only cam_head's tiled render product after a snapshot teleport."""

    camera_names = env.camera_manager.camera_names[0]
    if "cam_head" not in camera_names:
        raise AnnotationMappingError(f"cam_head is unavailable; cameras={camera_names}")
    env.capture_manager.recreate_render_products([camera_names.index("cam_head")])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layout-index",
        default="all",
        help='One fill_pen_holder layout index, or "all" (the default).',
    )
    parser.add_argument("--max-layouts", type=int, default=None, help="Optional debug cap after choosing layouts.")
    parser.add_argument(
        "--scene-count",
        "--count",
        dest="scene_count",
        type=int,
        default=10,
        help="Number of scene snapshots to render; each snapshot may yield multiple VQA records.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="First deterministic scene-snapshot index.")
    parser.add_argument(
        "--scenario",
        "--case",
        dest="scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
        help=(
            "Scenarios to cycle through: layout creates holder/nib/bbox questions; "
            "gripper_content creates left/right holding questions; visible_counts creates table/holder counts."
        ),
    )
    parser.add_argument("--layout-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--camera-jitter", type=float, default=0.0, help="Uniform head-camera position jitter in metres; default 0."
    )
    parser.add_argument(
        "--gripper-position-jitter",
        type=float,
        default=DEFAULT_GRIPPER_POSITION_JITTER_M,
        help="Maximum independent XY TCP perturbation for gripper_content snapshots, in metres.",
    )
    parser.add_argument(
        "--gripper-height-jitter",
        type=float,
        default=DEFAULT_GRIPPER_HEIGHT_JITTER_M,
        help="Maximum independent Z TCP perturbation for gripper_content snapshots, in metres.",
    )
    parser.add_argument("--sim-gpu-id", type=int, default=0, help="GPU index used in the RoboDojo sim config.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing VQA sidecar output directory.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _quat_matrix(quaternion: np.ndarray | list[float]) -> np.ndarray:
    """Return a 3x3 rotation matrix for a normalized wxyz quaternion."""
    return np.asarray(t3q.quat2mat(np.asarray(quaternion, dtype=np.float64)), dtype=np.float64)


def _fallen_quat(yaw_deg: float) -> list[float]:
    # Same convention as generate_stand_up_pen_holder_layouts.py.
    q = t3q.qmult(
        t3q.axangle2quat((0.0, 0.0, 1.0), math.radians(yaw_deg)),
        t3q.axangle2quat((1.0, 0.0, 0.0), math.radians(90.0)),
    )
    return (q / np.linalg.norm(q)).tolist()


def _numeric_suffix(path: Path) -> tuple[int, str]:
    suffix = path.stem.rsplit("_", 1)[-1]
    return (int(suffix), path.name) if suffix.isdigit() else (-1, path.name)


def _load_source_layouts(layout_root: Path, seed: int) -> list[tuple[Path, dict[str, Any]]]:
    source_dir = layout_root / str(seed)
    paths = sorted(source_dir.glob("fill_pen_holder_*.json"), key=_numeric_suffix)
    if not paths:
        raise FileNotFoundError(f"no fill_pen_holder layouts found in {source_dir}")
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def _set_holder_fallen(layout: dict[str, Any], fallen: bool, yaw_deg: float) -> None:
    holders = layout.get("Rigid", {}).get("pen_holder", [])
    if len(holders) != 1:
        raise ValueError("expected exactly one Rigid.pen_holder instance")
    holder = holders[0]
    holder["default_ori"] = _fallen_quat(yaw_deg) if fallen else [1.0, 0.0, 0.0, 0.0]
    holder["qpos"] = list(holder["default_ori"])
    if fallen:
        pos = list(holder["default_pos"])
        pos[2] = 0.805
        holder["default_pos"] = pos


def _set_record_pose(record: dict[str, Any], position: np.ndarray, quaternion: np.ndarray) -> None:
    record["default_pos"] = [float(value) for value in position]
    record["default_ori"] = [float(value) for value in quaternion]
    record["qpos"] = list(record["default_ori"])


def _make_case_layout(source: dict[str, Any], scenario: str, index: int, rng: random.Random) -> dict[str, Any]:
    layout = deepcopy(source)
    if scenario == "layout":
        _set_holder_fallen(layout, fallen=(index % 2 == 1), yaw_deg=90.0 + 45.0 * (index % 6))
        return layout

    # For cases 2 and 3, all object placement relative to the grippers is done
    # after reset, once the actual robot link pose is available.
    if scenario not in ("gripper_content", "visible_counts"):
        raise ValueError(f"unknown scenario: {scenario}")
    # Keep the source labels and category association intact.  RoboDojo uses
    # both fields to derive the runtime instance name; changing labels here
    # would make a subsequent reset unable to resolve the original rigid
    # objects.  Visual variation can still come from the source layouts and
    # optional camera jitter without invalidating the annotation mapping.
    rigid = layout.get("Rigid", {})
    pen_records = [record for category in ("pen", "oil_pen") for record in rigid.get(category, [])]
    if len(pen_records) != 4:
        raise ValueError(f"expected four pen records, found {len(pen_records)}")
    return layout


def _load_config(args: argparse.Namespace):
    from omegaconf import OmegaConf

    from env.global_configs import ENV_CONFIG_PATH, ROOT_DIR
    from task.RoboDojo.task_registry import task_config_path
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config

    sim = load_yaml(Path(ENV_CONFIG_PATH) / "sim" / "sim_config.yml")
    sim["scene"]["num_envs"] = 1
    sim["seed"] = [args.seed]
    sim["device"] = f"cuda:{args.sim_gpu_id}"
    # Keep DLAA for high-quality training images. Fresh renders after every
    # direct pose teleport let its temporal history converge before capture.
    sim["render"] = {
        "rendering_mode": "quality",
        "antialiasing_mode": "DLAA",
        "samples_per_pixel": 1024,
    }
    scene = load_yaml(Path(ENV_CONFIG_PATH) / "scene" / "default.yml")
    robot = load_yaml(Path(ENV_CONFIG_PATH) / "robot" / "dual_x5.yml")
    camera_source = load_yaml(Path(ENV_CONFIG_PATH) / "camera" / "camera_config.yml")
    # CameraManager treats every top-level key except its documented metadata
    # fields as a camera entry.  Do not add boolean feature flags here: an
    # ``include_robot_cameras=False`` key is iterated as a camera and causes
    # ``AttributeError: 'bool' object has no attribute 'camera'`` at startup.
    camera = {
        "cam_head": deepcopy(camera_source["cam_head"]),
        "default_frequency": 25,
        "annotator": {
            # RobotManager adds wrist cameras to the camera map for dual-arm
            # robots.  The explicit disabled fallback keeps this collector
            # ego-only while allowing TiledCaptureManager to configure those
            # cameras without trying to dereference a missing capture config.
            "common": {"enabled": False},
            "cam_head": {
                "enabled": True,
                "rgb_capture": {"type": "rgb", "device": "cpu"},
                "depth_capture": {"type": "distance_to_image_plane", "device": "cpu"},
                "instance_capture": {"type": "instance_segmentation_fast", "device": "cpu"},
            },
        },
    }
    task_cfg = load_yaml(task_config_path(Path(ROOT_DIR) / "task" / "RoboDojo" / "config", "fill_pen_holder"))
    config = OmegaConf.create({"sim": sim, "scene": scene, "robot": robot, "camera": camera, "task_env": task_cfg})
    config, _ = process_config(config, task_name="fill_pen_holder")
    # Keep the generation camera contract even if a task later gains a custom
    # camera config in `_task.yml`.
    config.camera = OmegaConf.create(camera)
    config.sim.scene.num_envs = 1
    config.sim.seed = [args.seed]
    config.sim.device = f"cuda:{args.sim_gpu_id}"
    return config


def _set_robot_pose(env, robot, target_pose: np.ndarray, gripper_joint: float) -> None:
    result = env.robot_manager.solve_ik(target_pose.tolist(), env_idx=0, robot=robot)
    if result.get("status") != "Success":
        raise RuntimeError(f"IK failed for {robot.arm_name}: {result}")
    key = env.robot_manager.robot_key[env.robot_manager.robot_list.index(robot)]
    joint_pos = key.data.joint_pos.clone()
    # This is a snapshot teleport, not a trajectory command.  Retaining a
    # prior sample's joint velocity produces a real transition frame and
    # invalid motion vectors for DLAA.
    joint_vel = torch.zeros_like(key.data.joint_vel)
    arm_joints = np.asarray(result["joint_value"], dtype=np.float32).reshape(-1)
    joint_pos[0, robot.arm_joint_indices] = torch.as_tensor(arm_joints, device=joint_pos.device)
    gripper_value = float(np.clip(gripper_joint, robot.gripper_scale[0], robot.gripper_scale[1]))
    gripper = [gripper_value, gripper_value * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2]]
    joint_pos[0, robot.gripper_joint_indices] = torch.as_tensor(gripper, device=joint_pos.device)
    key.write_joint_state_to_sim(joint_pos, joint_vel)
    key.set_joint_position_target(joint_pos)


def _set_local_pose(obj, position: np.ndarray, quaternion: np.ndarray | list[float]) -> None:
    obj.set_local_pose(
        translation=np.asarray(position, dtype=np.float32), orientation=np.asarray(quaternion, dtype=np.float32)
    )
    if hasattr(obj, "set_velocities"):
        # RigidObject inherits Isaac Sim's SingleRigidPrim API, which expects
        # one [linear_xyz, angular_xyz] row.  A pose snapshot must not carry
        # velocity from the preceding synthetic scene.
        obj.set_velocities(np.zeros((1, 6), dtype=np.float32))


def _x5_grasp_center(end_link_pose: np.ndarray, gripper_bias: float) -> np.ndarray:
    """Convert the link6 origin returned by get_real_endpose to the X5 TCP."""
    rotation = _quat_matrix(end_link_pose[3:])
    return end_link_pose[:3] + rotation @ np.array([float(gripper_bias), 0.0, 0.0])


def _horizontal_side_grasp_pose(robot, tcp_xy: tuple[float, float], tcp_z: float) -> np.ndarray:
    """Return an X5 link6 pose for a horizontal side grasp at the requested TCP.

    The X5 fingers extend along local +X and close along local Y.  Keeping
    local Z world-up makes an upright holder sit between the fingers instead
    of intersecting the wrist housing.  The link6 origin is derived by
    subtracting the configured TCP bias along local X.
    """
    base_position = np.asarray(robot.entity_origin_pose[:3], dtype=np.float64)
    tcp = np.array([tcp_xy[0], tcp_xy[1], tcp_z], dtype=np.float64)
    forward = tcp[:2] - base_position[:2]
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-6:
        raise ValueError(f"cannot construct side grasp at robot base for {robot.arm_name}")
    forward /= forward_norm
    yaw = math.atan2(forward[1], forward[0])
    quaternion = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)
    rotation = _quat_matrix(quaternion)
    link6_position = tcp - rotation @ np.array([float(robot.gripper_bias), 0.0, 0.0])
    return np.concatenate((link6_position, quaternion))


def _upright_orientation_for_closing_axis(closing_axis: np.ndarray) -> np.ndarray:
    """Orient an upright holder so its narrow local Y dimension meets the fingers."""
    axis = np.asarray(closing_axis[:2], dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        raise ValueError("X5 closing axis must have a horizontal component")
    axis /= axis_norm
    # Rz(yaw) maps local +Y to [-sin(yaw), cos(yaw)].
    yaw = math.atan2(-axis[0], axis[1])
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def _metadata_bbox_vertices(category: str, model_index: int) -> np.ndarray:
    metadata_path = (
        PROJECT_ROOT / "Assets" / "Object" / "RoboDojo" / "Rigid" / category / f"{model_index:05d}" / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return np.asarray(metadata["geometry"]["aligned_bbox"]["vertices"], dtype=np.float64)


def _x5_gripper_joint_for_width(robot, object_width: float) -> float:
    """Open the parallel jaws just wider than the selected object dimension."""
    target_gap = float(object_width) + GRIPPER_CLEARANCE
    joint_value = (target_gap - X5_FINGER_ZERO_GAP) * 0.5
    return float(np.clip(joint_value, robot.gripper_scale[0], robot.gripper_scale[1]))


def _pen_record(layout: dict[str, Any], label: str) -> tuple[str, dict[str, Any]]:
    for category in ("pen", "oil_pen"):
        for record in layout.get("Rigid", {}).get(category, []):
            if record.get("label") == label:
                return category, record
    raise KeyError(f"pen label {label!r} not found in layout")


def _upright_pen_geometry(layout: dict[str, Any], label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return upright wxyz orientation and rotated local bbox vertices."""
    category, record = _pen_record(layout, label)
    model_index = int(record["category_idx"])
    vertices = _metadata_bbox_vertices(category, model_index)
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    long_axis = int(np.argmax(extents))
    if long_axis == 0:
        quaternion = np.array([math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0])
    elif long_axis == 1:
        quaternion = np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
    else:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    rotated_vertices = (_quat_matrix(quaternion) @ vertices.T).T
    return quaternion, rotated_vertices


def _place_pen_centered(obj, layout: dict[str, Any], label: str, center: np.ndarray) -> None:
    quaternion, vertices = _upright_pen_geometry(layout, label)
    bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    _set_local_pose(obj, np.asarray(center) - bbox_center, quaternion)


def _place_pen_in_holder(
    obj,
    layout: dict[str, Any],
    label: str,
    xy_center: np.ndarray,
    bottom_z: float,
) -> None:
    quaternion, vertices = _upright_pen_geometry(layout, label)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    position = np.array(
        [xy_center[0] - (mins[0] + maxs[0]) * 0.5, xy_center[1] - (mins[1] + maxs[1]) * 0.5, bottom_z - mins[2]]
    )
    _set_local_pose(obj, position, quaternion)


def _holder_gripper_joint(layout: dict[str, Any], robot) -> float:
    holder_record = layout["Rigid"]["pen_holder"][0]
    vertices = _metadata_bbox_vertices("pen_holder", int(holder_record["category_idx"]))
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    return _x5_gripper_joint_for_width(robot, float(np.min(extents[:2])))


def _pen_gripper_joint(layout: dict[str, Any], label: str, robot) -> float:
    _, vertices = _upright_pen_geometry(layout, label)
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    return _x5_gripper_joint_for_width(robot, float(np.max(extents[:2])))


def _sample_gripper_tcp(
    side: str,
    rng: random.Random,
    position_jitter: float,
    height_jitter: float,
) -> tuple[tuple[float, float], float]:
    """Sample a deterministic, camera-visible TCP target for one X5 arm."""

    base_xy = (-0.10, -0.12) if side == "left" else (0.10, -0.12)
    return (
        (
            base_xy[0] + rng.uniform(-position_jitter, position_jitter),
            base_xy[1] + rng.uniform(-position_jitter, position_jitter),
        ),
        0.95 + rng.uniform(-height_jitter, height_jitter),
    )


def _stage_robot_objects(
    env,
    layout: dict[str, Any],
    scenario: str,
    index: int,
    rng: random.Random,
    gripper_position_jitter: float,
    gripper_height_jitter: float,
) -> dict[str, Any]:
    robots = {robot.arm_name.split("_")[0]: robot for robot in env.robot_manager.robot_list if robot.type == "target"}
    if set(robots) != {"left", "right"}:
        raise RuntimeError(f"expected left/right target robots, got {sorted(robots)}")
    layout_manager = env.scene_manager.layout_manager
    holder_name = layout_manager.get_instance_name(env_idx=0, label="pen_holder")
    holder = layout_manager.get_scene_object(env_idx=0, inst_name=holder_name)
    pens = {
        label: layout_manager.get_scene_object(env_idx=0, inst_name=layout_manager.get_instance_name(0, label))
        for label in PEN_LABELS
    }

    if scenario == "gripper_content":
        patterns = (
            ("pen_holder", "nothing"),
            ("nothing", "pen_holder"),
            ("pen", "nothing"),
            ("nothing", "pen"),
            ("pen_holder", "pen"),
            ("pen", "pen_holder"),
        )
        left_value, right_value = patterns[index % len(patterns)]
        hand_values = {"left": left_value, "right": right_value}
    elif scenario == "visible_counts":
        holder_hand = "left" if index % 2 == 0 else "right"
        hand_values = {
            "left": "pen_holder" if holder_hand == "left" else "nothing",
            "right": "pen_holder" if holder_hand == "right" else "nothing",
        }
    else:
        return {}

    # The target TCPs remain inside the fixed head-camera crop. For
    # gripper_content, both arms receive independent deterministic perturbations
    # (including an empty gripper), so VQA does not learn a fixed arm pose.
    sampled_tcp: dict[str, tuple[tuple[float, float], float]] = {}
    for side, value in hand_values.items():
        if value == "nothing" and scenario != "gripper_content":
            continue
        robot = robots[side]
        position_jitter = gripper_position_jitter if scenario == "gripper_content" else 0.0
        height_jitter = gripper_height_jitter if scenario == "gripper_content" else 0.0
        tcp_xy, tcp_z = _sample_gripper_tcp(side, rng, position_jitter, height_jitter)
        sampled_tcp[side] = (tcp_xy, tcp_z)
        target_pose = _horizontal_side_grasp_pose(robot, tcp_xy, tcp_z=tcp_z)
        gripper_joint = {
            "nothing": float(robot.gripper_scale[1]),
            "pen": _pen_gripper_joint(layout, "target0", robot),
            "pen_holder": _holder_gripper_joint(layout, robot),
        }[value]
        _set_robot_pose(env, robot, target_pose, gripper_joint=gripper_joint)

    # Forward kinematics/body-link buffers are refreshed by a physics step.
    # Held objects are placed only after this step and are captured without a
    # subsequent step, so gravity cannot make them drift out of the grippers.
    env.sim_step(render=False)
    ee_poses = {
        side: np.asarray(env.robot_manager.get_real_endpose(robots[side], env_idx_list=[0])[0], dtype=np.float64)
        for side, value in hand_values.items()
        if value != "nothing"
    }

    if scenario == "gripper_content":
        for side, value in hand_values.items():
            if value == "pen_holder":
                grasp_center = _x5_grasp_center(ee_poses[side], robots[side].gripper_bias)
                pos = grasp_center - np.array([0.0, 0.0, HOLDER_GRASP_HEIGHT])
                closing_axis = _quat_matrix(ee_poses[side][3:])[:, 1]
                _set_local_pose(holder, pos, _upright_orientation_for_closing_axis(closing_axis))
            elif value == "pen":
                grasp_center = _x5_grasp_center(ee_poses[side], robots[side].gripper_bias)
                _place_pen_centered(pens["target0"], layout, "target0", grasp_center)
        return {
            "left_hand": hand_values["left"],
            "right_hand": hand_values["right"],
            "gripper_tcp_positions": {
                side: [float(tcp_xy[0]), float(tcp_xy[1]), float(tcp_z)]
                for side, (tcp_xy, tcp_z) in sampled_tcp.items()
            },
        }

    holder_hand = "left" if hand_values["left"] == "pen_holder" else "right"
    holder_grasp_center = _x5_grasp_center(ee_poses[holder_hand], robots[holder_hand].gripper_bias)
    holder_pos = holder_grasp_center - np.array([0.0, 0.0, HOLDER_GRASP_HEIGHT])
    holder_closing_axis = _quat_matrix(ee_poses[holder_hand][3:])[:, 1]
    _set_local_pose(holder, holder_pos, _upright_orientation_for_closing_axis(holder_closing_axis))
    in_holder = 1 + index % 3
    available = list(PEN_LABELS)
    rng.shuffle(available)
    for slot, label in enumerate(available[:in_holder]):
        pen_xy = holder_pos[:2] + np.array([(slot - (in_holder - 1) / 2.0) * 0.018, 0.0])
        _place_pen_in_holder(pens[label], layout, label, pen_xy, bottom_z=holder_pos[2] + 0.025)
    # Keep every table pen visible for the counting question.  Use the
    # opposite half of the unchanged EvalResult table and retain each model's
    # original lying orientation and table-height z value.
    table_slots = (
        ((0.10, -0.18), (0.24, -0.08), (0.14, 0.06))
        if holder_hand == "left"
        else ((-0.28, -0.18), (-0.15, -0.06), (-0.28, 0.07))
    )
    for slot_xy, label in zip(table_slots, available[in_holder:], strict=False):
        _, record = _pen_record(layout, label)
        table_position = np.asarray(record["default_pos"], dtype=np.float64).copy()
        table_position[:2] = slot_xy
        _set_local_pose(pens[label], table_position, record["default_ori"])
    return {
        "holder_hand": holder_hand,
        "pens_in_holder": in_holder,
        "pens_on_table": len(PEN_LABELS) - in_holder,
        "in_holder_labels": available[:in_holder],
        "on_table_labels": available[in_holder:],
    }


def _capture_ego_annotations(env) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Capture the only model-visible view plus required geometry annotations."""
    captured = env.capture_manager.step(env_ids=[0])
    camera_names = env.camera_manager.camera_names[0]
    if "cam_head" not in camera_names:
        raise AnnotationMappingError(f"cam_head is unavailable; cameras={camera_names}")
    data = captured[camera_names.index("cam_head")]
    required = {"rgb", "distance_to_image_plane", "instance_segmentation_fast"}
    missing = required.difference(data)
    if missing:
        raise AnnotationMappingError(f"missing annotation buffers: {sorted(missing)}")
    rgb = np.asarray(data["rgb"][0]["data"], dtype=np.uint8)[..., :3]
    depth = np.asarray(data["distance_to_image_plane"][0]["data"], dtype=np.float32).squeeze(-1)
    instance = np.asarray(data["instance_segmentation_fast"][0]["data"]).squeeze(-1)
    info = data["instance_segmentation_fast"][0].get("info", {})
    if rgb.shape[:2] != depth.shape or rgb.shape[:2] != instance.shape:
        raise AnnotationMappingError("RGB, depth, and instance segmentation shapes differ")
    return rgb, depth, instance, info


def _label_matches(value: Any, semantic_label: str) -> bool:
    if isinstance(value, dict):
        return any(_label_matches(item, semantic_label) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_label_matches(item, semantic_label) for item in value)
    text = str(value)
    return text == semantic_label or (not text.startswith("/") and semantic_label in text)


def _prim_path_matches(value: Any, prim_path: str) -> bool:
    """Match an instance identity path exactly, never by a shared prefix."""

    if isinstance(value, dict):
        return any(_prim_path_matches(item, prim_path) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_prim_path_matches(item, prim_path) for item in value)
    return str(value) == prim_path


def _renderer_identity_matches(value: Any, semantic_label: str, prim_path: str) -> bool:
    """Match either a renderer semantic label or its unique object prim path."""

    return _label_matches(value, semantic_label) or _prim_path_matches(value, prim_path)


def _mask_for_semantic(instance: np.ndarray, info: dict[str, Any], semantic_label: str, prim_path: str) -> np.ndarray:
    id_to_labels = info.get("idToLabels") or info.get("id_to_labels") or info.get("labels")
    if not isinstance(id_to_labels, dict):
        raise AnnotationMappingError("instance segmentation did not return idToLabels metadata")
    ids = [
        int(identifier)
        for identifier, labels in id_to_labels.items()
        if _renderer_identity_matches(labels, semantic_label, prim_path)
    ]
    if len(ids) != 1:
        raise AnnotationMappingError(
            f"renderer identity for {semantic_label!r} at {prim_path!r} resolves to {len(ids)} instance IDs"
        )
    return instance == ids[0]


def _semantic_id_map(
    info: dict[str, Any], semantic_labels: dict[str, str], semantic_prim_paths: dict[str, str]
) -> dict[str, list[int]]:
    """Return every renderer ID associated with each semantic label and object path."""

    id_to_labels = info.get("idToLabels") or info.get("id_to_labels") or info.get("labels")
    if not isinstance(id_to_labels, dict):
        raise AnnotationMappingError("instance segmentation did not return idToLabels metadata")
    return {
        label: [
            int(identifier)
            for identifier, labels in id_to_labels.items()
            if _renderer_identity_matches(labels, semantic_label, semantic_prim_paths[label])
        ]
        for label, semantic_label in semantic_labels.items()
    }


def _write_segmentation_audit(
    audit_dir: Path,
    scene_id: str,
    instance: np.ndarray,
    info: dict[str, Any],
    semantic_labels: dict[str, str],
    semantic_prim_paths: dict[str, str],
) -> dict[str, list[int]]:
    """Persist a colorized ID image and ID-to-semantic mapping for review."""

    from PIL import Image

    instance = np.asarray(instance)
    if instance.ndim != 2:
        raise AnnotationMappingError(f"instance segmentation must be 2-D, got {instance.shape}")
    semantic_ids = _semantic_id_map(info, semantic_labels, semantic_prim_paths)
    visualization = np.zeros((*instance.shape, 3), dtype=np.uint8)
    for identifier in np.unique(instance):
        identifier = int(identifier)
        if identifier == 0:
            continue
        visualization[instance == identifier] = (
            (identifier * 37) % 251 + 4,
            (identifier * 73) % 251 + 4,
            (identifier * 109) % 251 + 4,
        )
    Image.fromarray(visualization).save(audit_dir / f"{scene_id}_instance_ids.png")
    id_to_labels = info.get("idToLabels") or info.get("id_to_labels") or info.get("labels")
    (audit_dir / f"{scene_id}_instance_ids.json").write_text(
        json_dumps(
            {
                "semantic_labels": semantic_labels,
                "semantic_prim_paths": semantic_prim_paths,
                "matched_instance_ids": semantic_ids,
                "id_to_labels": id_to_labels,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return semantic_ids


def _set_semantic_label(entity: Any, semantic_label: str) -> str:
    from isaacsim.core.utils.semantics import add_labels, remove_labels
    import omni.usd

    prim = getattr(entity, "prim", None)
    if prim is None:
        prim_path = getattr(entity, "prim_path", None)
        if prim_path is None:
            raise AnnotationMappingError(f"cannot locate prim for semantic label {semantic_label!r}")
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(str(prim_path))
    if not prim or not prim.IsValid():
        raise AnnotationMappingError(f"invalid prim for semantic label {semantic_label!r}")
    # Replace inherited asset labels within this object subtree. The renderer
    # then observes one unambiguous object-level label rather than an asset
    # category shared by multiple pens.
    remove_labels(prim, include_descendants=True)
    add_labels(prim, [semantic_label])
    return str(prim.GetPath())


def _label_scene_instances(env) -> tuple[dict[str, str], dict[str, str]]:
    manager = env.scene_manager.layout_manager
    labels = {label: f"vqa_fill_pen_holder_{label}" for label in (*PEN_LABELS, "pen_holder")}
    prim_paths = {}
    for label, semantic_label in labels.items():
        entity = manager.get_scene_object(0, manager.get_instance_name(0, label))
        if entity is None:
            raise AnnotationMappingError(f"missing scene object for {label}")
        prim_paths[label] = _set_semantic_label(entity, semantic_label)
    return labels, prim_paths


def _mask_anchor(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise AnnotationMappingError("cannot anchor an empty instance mask")
    center_x, center_y = float(xs.mean()), float(ys.mean())
    nearest = int(np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
    return int(xs[nearest]), int(ys[nearest])


def _project_world_point(env, point_world: np.ndarray, cam_id: int = 0) -> tuple[float, float, float] | None:
    intrinsic = np.asarray(env.camera_manager.get_camera_intrinsics(cam_id, 0), dtype=np.float64)
    extrinsic = np.asarray(env.camera_manager.get_camera_extrinsics(cam_id, 0), dtype=np.float64)
    point_camera = extrinsic[:3, :3].T @ (np.asarray(point_world, dtype=np.float64) - extrinsic[:3, 3])
    if point_camera[2] >= -1e-6:
        return None
    depth = -float(point_camera[2])
    return (
        float(intrinsic[0, 0] * point_camera[0] / depth + intrinsic[0, 2]),
        float(intrinsic[1, 2] - intrinsic[1, 1] * point_camera[1] / depth),
        depth,
    )


def _nib_world_point(env, label: str) -> np.ndarray:
    manager = env.scene_manager.layout_manager
    instance_name = manager.get_instance_name(0, label)
    metadata = manager.get_instance_metadata(env_idx=0, label=label)
    points = manager.get_functional_points(
        tag="check", type="passive", config=metadata, obj_name=instance_name, env_idx=0
    )
    if len(points) != 1:
        raise AnnotationMappingError(f"{label} must have exactly one passive functional check point")
    return np.asarray(points[0][:3], dtype=np.float64)


def _visible_nib_point(env, mask: np.ndarray, depth: np.ndarray, label: str) -> tuple[list[float] | None, str]:
    projected = _project_world_point(env, _nib_world_point(env, label))
    if projected is None:
        return None, "out_of_frame"
    u, v, projected_depth = projected
    height, width = mask.shape
    if u < 0 or v < 0 or u >= width or v >= height:
        return None, "out_of_frame"
    x0, x1 = max(0, int(round(u)) - NIB_PATCH_RADIUS_PX), min(width, int(round(u)) + NIB_PATCH_RADIUS_PX + 1)
    y0, y1 = max(0, int(round(v)) - NIB_PATCH_RADIUS_PX), min(height, int(round(v)) + NIB_PATCH_RADIUS_PX + 1)
    local_mask = mask[y0:y1, x0:x1]
    local_depth = depth[y0:y1, x0:x1]
    matching = local_mask & np.isfinite(local_depth) & (np.abs(local_depth - projected_depth) <= NIB_DEPTH_TOLERANCE_M)
    if int(matching.sum()) >= NIB_MIN_MATCHING_PIXELS:
        return [float(u / width), float(v / height)], "visible"
    if not np.any(local_mask):
        return None, "occluded"
    return None, "ambiguous"


def _apply_camera_jitter(
    env,
    jitter: float,
    rng: random.Random,
    base_pose: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    xform = env.camera_manager.cameras_xform[0][0]
    position, orientation = base_pose if base_pose is not None else xform.get_local_pose()
    if jitter <= 0:
        xform.set_local_pose(position, orientation)
        return
    delta = np.array(
        [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)], dtype=np.float32
    )
    xform.set_local_pose(position + torch.as_tensor(delta, device=position.device), orientation)


def _restore_layout_holder_pose(env, layout: dict[str, Any]) -> None:
    holder_record = layout["Rigid"]["pen_holder"][0]
    layout_manager = env.scene_manager.layout_manager
    holder = layout_manager.get_scene_object(0, layout_manager.get_instance_name(0, "pen_holder"))
    _set_local_pose(holder, np.asarray(holder_record["default_pos"]), holder_record["default_ori"])


def _restore_layout_object_poses(env, layout: dict[str, Any]) -> None:
    """Restore poses in-place without reloading USD assets.

    TaskEnv.reset() rebuilds the scene registry and advances per-category
    instance counters, so a second reset can no longer resolve the original
    object names. All scenarios use one fixed layout; restoring these transforms
    is sufficient between frames and keeps the registry stable.
    """
    manager = env.scene_manager.layout_manager
    for category in ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid"):
        for records in layout.get(category, {}).values():
            for record in records:
                label = record.get("label")
                if label is None:
                    continue
                inst_name = manager.get_instance_name(0, label)
                obj = manager.get_scene_object(0, inst_name)
                if obj is not None and "default_pos" in record and "default_ori" in record:
                    _set_local_pose(obj, np.asarray(record["default_pos"]), record["default_ori"])


def _restore_robot_targets(env) -> None:
    """Teleport both arms to default state instead of only changing targets."""

    env.robot_manager.reset()
    for robot, key in zip(env.robot_manager.robot_list, env.robot_manager.robot_key, strict=True):
        if robot.type != "target":
            continue
        default_joint_pos = key.data.default_joint_pos.clone()
        zero_joint_vel = torch.zeros_like(key.data.default_joint_vel)
        key.write_joint_state_to_sim(default_joint_pos, zero_joint_vel)
        key.set_joint_position_target(default_joint_pos)
    env.sim_step(render=False)


def _select_layouts(
    layouts: list[tuple[Path, dict[str, Any]]], selection: str, max_layouts: int | None
) -> list[tuple[int, Path, dict[str, Any]]]:
    if selection.strip().lower() == "all":
        selected = [(index, path, layout) for index, (path, layout) in enumerate(layouts)]
    else:
        try:
            index = int(selection)
        except ValueError as error:
            raise ValueError('--layout-index must be a non-negative integer or "all"') from error
        if index < 0 or index >= len(layouts):
            raise ValueError(f"--layout-index must be in [0, {len(layouts) - 1}]")
        path, layout = layouts[index]
        selected = [(index, path, layout)]
    if max_layouts is not None:
        if max_layouts <= 0:
            raise ValueError("--max-layouts must be positive")
        selected = selected[:max_layouts]
    return selected


def _record_base(
    *, sample_id: str, image_reference: str, scenario: str, source_layout: Path, rgb: np.ndarray
) -> dict[str, Any]:
    return base_record(
        sample_id=sample_id,
        task_name="fill_pen_holder",
        ego_image_reference=image_reference,
        image_width=int(rgb.shape[1]),
        image_height=int(rgb.shape[0]),
        source_layout=str(source_layout),
        scene_id=sample_id.rsplit("_", 1)[0],
        audit_metadata_json=json_dumps(
            {"scenario": scenario, "image_masks": {"ego": True, "left_wrist": False, "right_wrist": False}}
        ),
    )


def _holder_tilt_degrees(quaternion: np.ndarray | list[float]) -> float:
    up_axis = _quat_matrix(quaternion)[:, 2]
    cosine = float(np.clip(np.dot(up_axis, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _add_case_records(
    writer: SidecarWriter,
    env,
    *,
    scenario: str,
    index: int,
    seed: int,
    layout_index: int,
    source_layout: Path,
    layout: dict[str, Any],
    scene_state: dict[str, Any],
    semantic_labels: dict[str, str],
    semantic_prim_paths: dict[str, str],
    rgb: np.ndarray,
    depth: np.ndarray,
    instance: np.ndarray,
    info: dict[str, Any],
    image_dir: Path,
    audit_dir: Path,
    rng: random.Random,
) -> None:
    """Persist clean/overlay images and emit only contract-compliant VQA rows."""

    from PIL import Image

    scene_id = f"fill_pen_holder_seed{seed}_layout{layout_index:03d}_{scenario}_{index:06d}"
    clean_name = f"{scene_id}.png"
    Image.fromarray(rgb).save(image_dir / clean_name)
    clean_ref = str(Path("images") / clean_name)
    semantic_ids = _write_segmentation_audit(audit_dir, scene_id, instance, info, semantic_labels, semantic_prim_paths)
    if scenario == "gripper_content":
        required_labels = {
            "pen_holder" if held == "pen_holder" else "target0"
            for held in (scene_state["left_hand"], scene_state["right_hand"])
            if held != "nothing"
        }
    else:
        # The layout and count families need every object mask: they either
        # mark every pen or derive a per-instance visible count.
        required_labels = set(semantic_labels)
    invalid_mappings = {label: semantic_ids[label] for label in required_labels if len(semantic_ids[label]) != 1}
    if invalid_mappings:
        details = ", ".join(f"{label}={ids}" for label, ids in sorted(invalid_mappings.items()))
        raise AnnotationMappingError(f"semantic-to-instance mapping is not one-to-one: {details}")
    masks = {
        label: _mask_for_semantic(instance, info, semantic, semantic_prim_paths[label])
        for label, semantic in semantic_labels.items()
        if label in required_labels
    }

    def candidate(family: str, suffix: str, image_ref: str = clean_ref) -> dict[str, Any]:
        return _record_base(
            sample_id=f"{scene_id}_{suffix}",
            image_reference=image_ref,
            scenario=scenario,
            source_layout=source_layout,
            rgb=rgb,
        ) | {"question_family": family}

    if scenario == "layout":
        holder_status, holder_fraction, holder_occlusion = classify_mask_visibility(
            masks["pen_holder"], HOLDER_VISIBILITY
        )
        tilt = _holder_tilt_degrees(layout["Rigid"]["pen_holder"][0]["default_ori"])
        holder = candidate("holder_fallen", "holder_fallen") | {
            "prompt_text": "Is the pen holder lying on its side? Answer yes or no.",
            "answer_type": "boolean",
            "answer_bool": tilt >= 60.0,
            "world_state_valid": tilt <= 30.0 or tilt >= 60.0,
            "image_answerable": holder_status in {"visible", "partially_visible"},
            "visibility_status": holder_status,
            "visible_fraction": holder_fraction,
            "occlusion_ratio": holder_occlusion,
            "gt_source": "simulated_holder_orientation",
            "quality_score": float(masks["pen_holder"].sum()),
            "audit_metadata_json": json_dumps({"holder_tilt_degrees": tilt}),
        }
        if 30.0 < tilt < 60.0:
            writer.reject(holder, "holder_orientation_dead_zone")
        else:
            writer.add(holder)
        box = bbox_from_mask(masks["pen_holder"])
        holder_box = candidate("holder_bbox", "holder_bbox") | {
            "prompt_text": "Locate the visible pen holder in the ego-view image. Return one bounding box.",
            "answer_type": "bbox2d",
            "answer_bbox_xyxy_norm": box,
            "world_state_valid": True,
            "image_answerable": box is not None and holder_status in {"visible", "partially_visible"},
            "visibility_status": holder_status,
            "visible_fraction": holder_fraction,
            "occlusion_ratio": holder_occlusion,
            "target_view": "ego",
            "coordinate_space": "original_image_normalized_xyxy",
            "bbox_definition": "visible_tight",
            "gt_source": "ego_instance_segmentation",
            "quality_score": float(masks["pen_holder"].sum()),
        }
        writer.add(holder_box)

        ordered_labels = list(PEN_LABELS)
        rng.shuffle(ordered_labels)
        mark_to_label = {str(mark): label for mark, label in enumerate(ordered_labels, start=1)}
        nib_results = {label: _visible_nib_point(env, masks[label], depth, label) for label in PEN_LABELS}
        try:
            overlay = numbered_overlay(
                rgb,
                {mark: _mask_anchor(masks[label]) for mark, label in mark_to_label.items()},
                protected_points_xy=[
                    (point[0] * rgb.shape[1], point[1] * rgb.shape[0])
                    for point, status in nib_results.values()
                    if point is not None and status == "visible"
                ],
                protected_point_clearance_px=NIB_PATCH_RADIUS_PX + 3,
            )
        except (AnnotationMappingError, OverlayError) as error:
            for mark, label in mark_to_label.items():
                writer.reject(candidate("pen_nib_grounding", f"nib_{mark}"), f"overlay_unusable: {error}")
            return
        overlay_name = f"{scene_id}_pen_marks.png"
        Image.fromarray(overlay.image).save(image_dir / overlay_name)
        overlay_ref = str(Path("images") / overlay_name)
        mapping_json = json_dumps(
            {
                mark: env.scene_manager.layout_manager.get_instance_name(0, label)
                for mark, label in mark_to_label.items()
            }
        )
        for mark, label in mark_to_label.items():
            point, status = nib_results[label]
            record = candidate("pen_nib_grounding", f"nib_{mark}", overlay_ref) | {
                "prompt_text": f"Locate the nib of the pen marked {mark} in the ego-view image. Return one point.",
                "answer_type": "point2d",
                "answer_point_xy_norm": point,
                "world_state_valid": True,
                "image_answerable": point is not None and status == "visible",
                "visibility_status": status,
                "target_view": "ego",
                "coordinate_space": "original_image_normalized_xy",
                "point_definition": "functional_nib_tip",
                "gt_source": "simulated_3d_functional_point_projection",
                "overlay_type": "numbered_object_marks",
                "overlay_version": "fill_pen_holder_v1",
                "overlay_mark_to_instance_json": mapping_json,
                "audit_metadata_json": json_dumps({"mark_index": int(mark), "target_label": label}),
            }
            writer.add(record)
        return

    if scenario == "gripper_content":
        answer_text = {"pen": "pen", "pen_holder": "pen holder", "nothing": "nothing"}
        for side in ("left", "right"):
            held = scene_state[f"{side}_hand"]
            if held == "nothing":
                status, fraction, occlusion, visible = "not_applicable", None, None, True
            else:
                held_label = "pen_holder" if held == "pen_holder" else "target0"
                status, fraction, occlusion = classify_mask_visibility(
                    masks[held_label], PEN_VISIBILITY if held == "pen" else HOLDER_VISIBILITY
                )
                visible = status in {"visible", "partially_visible"}
            record = candidate(f"{side}_gripper_content", f"{side}_gripper") | {
                "prompt_text": (
                    f"What is the {side} gripper directly holding? "
                    "Answer with exactly one of: nothing, pen, pen holder."
                ),
                "answer_type": "short_text",
                "answer_text": answer_text[held],
                "world_state_valid": True,
                "image_answerable": visible,
                "visibility_status": status,
                "visible_fraction": fraction,
                "occlusion_ratio": occlusion,
                "gt_source": "dedicated_stable_grasp_generator",
                "audit_metadata_json": json_dumps(
                    {
                        "gripper_side": side,
                        "grasp_class": answer_text[held],
                        "render_stabilization_frames": RENDER_STABILIZATION_FRAMES,
                        "tcp_position": scene_state["gripper_tcp_positions"][side],
                    }
                ),
            }
            writer.add(record)
        return

    for relation, family, answer_key, prompt in (
        (
            "ON_TABLE",
            "visible_pens_on_table",
            "pens_on_table",
            "How many pens are visible on the table? Answer with one integer.",
        ),
        (
            "IN_HOLDER",
            "visible_pens_in_holder",
            "pens_in_holder",
            "How many pens are visible inside the pen holder? Answer with one integer.",
        ),
    ):
        qualifying = list(
            scene_state["in_holder_labels"] if relation == "IN_HOLDER" else scene_state["on_table_labels"]
        )
        counted = []
        for label in qualifying:
            status, _, _ = classify_mask_visibility(masks[label], COUNT_PEN_VISIBILITY)
            if status == "visible":
                counted.append(label)
        record = candidate(family, "table_count" if relation == "ON_TABLE" else "holder_count") | {
            "prompt_text": prompt,
            "answer_type": "integer",
            "answer_int": len(counted),
            "world_state_valid": True,
            "image_answerable": True,
            "visibility_status": "not_applicable",
            "gt_source": "simulated_relation_plus_ego_instance_visibility",
            "audit_metadata_json": json_dumps(
                {
                    "qualifying_pen_instance_ids": qualifying,
                    "counted_pen_instance_ids": counted,
                    "semantic_state": relation,
                }
            ),
        }
        writer.add(record)


def _manifest(
    args: argparse.Namespace,
    selected_layouts: list[tuple[int, Path, dict[str, Any]]],
    failure: BaseException | None,
) -> dict[str, Any]:
    """Build manifest data even when Isaac exits before the first capture."""

    manifest: dict[str, Any] = {
        "collector": "scripts/internal/generate_fill_pen_holder_vqa.py",
        "task_name": "fill_pen_holder",
        "source_action_dataset": "RoboDojo_ee_lerobot_v30_video",
        "seed": args.seed,
        "layout_index": args.layout_index,
        "max_layouts": args.max_layouts,
        "source_layouts": [str(path) for _, path, _ in selected_layouts],
        "scene_count": args.scene_count,
        "start_index": args.start_index,
        "scenarios": args.scenarios,
        "camera": "cam_head",
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "visibility_thresholds": {
            "holder": HOLDER_VISIBILITY.__dict__,
            "pen": PEN_VISIBILITY.__dict__,
            "count_pen": COUNT_PEN_VISIBILITY.__dict__,
            "nib_patch_radius_px": NIB_PATCH_RADIUS_PX,
            "nib_depth_tolerance_m": NIB_DEPTH_TOLERANCE_M,
        },
        "gripper_position_jitter_m": args.gripper_position_jitter,
        "gripper_height_jitter_m": args.gripper_height_jitter,
        "render": {
            "antialiasing_mode": "DLAA",
            "stabilization_frames": RENDER_STABILIZATION_FRAMES,
            "ego_render_product_recreated_per_snapshot": True,
        },
        "command": " ".join(sys.argv),
        "run_status": "failed" if failure is not None else "completed",
    }
    if failure is not None:
        manifest["failure_type"] = type(failure).__name__
        manifest["failure_message"] = str(failure)
    return manifest


def main() -> None:
    args = _parse_args()
    if args.scene_count <= 0 or args.start_index < 0:
        raise ValueError("--scene-count must be positive and --start-index must be non-negative")
    # if not 0.0 <= args.gripper_position_jitter <= 0.08:
    #     raise ValueError("--gripper-position-jitter must be within [0.0, 0.08] metres")
    # if not 0.0 <= args.gripper_height_jitter <= 0.05:
    #     raise ValueError("--gripper-height-jitter must be within [0.0, 0.05] metres")
    writer = SidecarWriter(args.output_dir, overwrite=args.overwrite)
    image_dir = writer.prepare_images_dir()
    audit_dir = writer.prepare_audit_dir()
    _configure_logging()
    simulation_app = None
    selected_layouts: list[tuple[int, Path, dict[str, Any]]] = []
    failure: BaseException | None = None
    try:
        LOGGER.info("starting output_dir=%s", args.output_dir)
        LOGGER.info("launching Isaac Sim (layouts=%s, scene_count=%s)", args.max_layouts or "all", args.scene_count)
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

        from env.environment.task_env import TaskEnv

        source_layouts = _load_source_layouts(args.layout_root, args.seed)
        selected_layouts = _select_layouts(source_layouts, args.layout_index, args.max_layouts)
        for layout_index, source_path, source_layout in selected_layouts:
            env = None
            try:
                LOGGER.info("creating layout=%s source=%s", layout_index, source_path)
                env = TaskEnv(_load_config(args), simulation_app)
                # A new environment per layout is required because Isaac's
                # object registry cannot safely replace category instances.
                initial_layout = _make_case_layout(source_layout, "layout", 0, random.Random(args.seed))
                env.scene_manager.layout_manager.set_saved_layout(0, initial_layout)
                LOGGER.info("resetting layout=%s", layout_index)
                env.reset(seed=[args.seed])
                env.scene_manager.apply_saved_poses(env_idx_list=[0])
                LOGGER.info("warming up layout=%s", layout_index)
                for warmup_step in range(200):
                    env.sim_step(render=False)
                    if warmup_step % 5 == 0:
                        env.render()
                semantic_labels, _ = _label_scene_instances(env)
                base_camera_pose = env.camera_manager.cameras_xform[0][0].get_local_pose()
                for index in range(args.start_index, args.start_index + args.scene_count):
                    scenario = args.scenarios[index % len(args.scenarios)]
                    scenario_index = index // len(args.scenarios)
                    rng = random.Random(args.seed * 1000003 + layout_index * 65537 + index * 9176)
                    layout = _make_case_layout(source_layout, scenario, scenario_index, rng)
                    _restore_layout_object_poses(env, initial_layout)
                    _restore_robot_targets(env)
                    _apply_camera_jitter(env, args.camera_jitter, rng, base_pose=base_camera_pose)
                    if scenario == "layout":
                        _restore_layout_holder_pose(env, layout)
                    try:
                        scene_state = _stage_robot_objects(
                            env,
                            layout,
                            scenario,
                            scenario_index,
                            rng,
                            args.gripper_position_jitter,
                            args.gripper_height_jitter,
                        )
                    except RuntimeError as error:
                        writer.reject(
                            base_record(
                                sample_id=(
                                    f"fill_pen_holder_seed{args.seed}_layout{layout_index:03d}_{scenario}_{index:06d}"
                                ),
                                task_name="fill_pen_holder",
                                question_family="scene_staging",
                                source_layout=str(source_path),
                            ),
                            f"scene_staging: {error}",
                        )
                        LOGGER.warning("rejected layout=%s sample=%s during staging: %s", layout_index, index, error)
                        continue
                    # Apply labels after every object/robot placement so a
                    # scene reload or asset-level semantic relationship cannot
                    # silently replace the VQA object identities.
                    semantic_labels, semantic_prim_paths = _label_scene_instances(env)
                    # DLAA history is owned by the tiled render product. A
                    # stage-level reset alone may leave a previous snapshot in
                    # cam_head's GPU history, so recreate only that product
                    # before its fixed stabilization renders.
                    _reset_renderer_accumulation()
                    _reset_ego_render_product(env)
                    for _ in range(RENDER_STABILIZATION_FRAMES):
                        env.render()
                    try:
                        LOGGER.info("capturing layout=%s sample=%s scenario=%s", layout_index, index, scenario)
                        rgb, depth, instance, info = _capture_ego_annotations(env)
                        _add_case_records(
                            writer,
                            env,
                            scenario=scenario,
                            index=index,
                            seed=args.seed,
                            layout_index=layout_index,
                            source_layout=source_path,
                            layout=layout,
                            scene_state=scene_state,
                            semantic_labels=semantic_labels,
                            semantic_prim_paths=semantic_prim_paths,
                            rgb=rgb,
                            depth=depth,
                            instance=instance,
                            info=info,
                            image_dir=image_dir,
                            audit_dir=audit_dir,
                            rng=rng,
                        )
                    except AnnotationMappingError as error:
                        writer.reject(
                            base_record(
                                sample_id=(
                                    f"fill_pen_holder_seed{args.seed}_layout{layout_index:03d}_{scenario}_{index:06d}"
                                ),
                                task_name="fill_pen_holder",
                                question_family="scene_validation",
                                source_layout=str(source_path),
                            ),
                            str(error),
                        )
            finally:
                if env is not None:
                    env.close()
            LOGGER.info("finished layout=%s", layout_index)
    except BaseException as error:
        failure = error
        writer.reject(
            base_record(
                sample_id=f"fill_pen_holder_seed{args.seed}_run_failure",
                task_name="fill_pen_holder",
                question_family="run_failure",
                source_layout=str(args.layout_root / str(args.seed)),
            ),
            f"{type(error).__name__}: {error}",
        )
        LOGGER.exception("generation failed; writing diagnostic sidecar")
        raise
    finally:
        try:
            report = writer.write(_manifest(args, selected_layouts, failure))
            LOGGER.info(
                "wrote accepted=%s rejected=%s output_dir=%s",
                report["accepted_records"],
                report["rejected_records"],
                args.output_dir,
            )
        finally:
            if simulation_app is not None:
                simulation_app.close()


if __name__ == "__main__":
    main()
