"""Render typed, ego-only VQA sidecar samples for ``make_kong``.

For each selected layout and target group, the collector holds robots at home,
renders clean ``cam_head`` images, and emits two physical VQA families: the
three left-to-right positions of the matching tiles and per-tile decisions for
every matching-tile fallen bitmask.

Example:
    python scripts/internal/generate_make_kong_vqa.py \
        --headless --enable_cameras --device-id 0 --seed 0 \
        --max-layouts 1 \
        --output-dir ./output/RoboDojo_vqa_v2/make_kong_seed0 --overwrite
"""

import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# RoboDojo must precede XPolicyLab: both have a top-level ``utils`` package.
for package_root in (REPO_ROOT / "XPolicyLab", REPO_ROOT):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from isaaclab.app import AppLauncher

from scripts.internal.vqa.sidecar import (
    SidecarWriter,
    VisibilityThresholds,
    base_record,
    classify_mask_visibility,
    json_dumps,
)
from scripts.internal.vqa.task_logic import (
    adjacent_nonmatching_labels,
    fallen_labels_for_pattern,
    kong_declaration_neighbor_state_reason,
    labels_for_bitmask,
    matching_tile_indices,
    ordinal,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device-id", type=int, default=0)
parser.add_argument("--seed", type=int, default=0, help="Fixed evaluation-layout seed, e.g. 0, 1, or 2.")
parser.add_argument(
    "--layout-ids",
    default="all",
    help='Comma-separated layout IDs, e.g. 0,1,2, or "all" for every eval layout under --seed.',
)
parser.add_argument("--max-layouts", type=int, default=None, help="Optional debug cap after expanding --layout-ids.")
parser.add_argument("--fps", type=int, default=25, help="Camera capture frequency.")
parser.add_argument(
    "--target-groups",
    default="0,1,2,3",
    help="Comma-separated make_kong matching groups to render. Valid values: 0,1,2,3.",
)
parser.add_argument(
    "--state-patterns",
    default="all",
    help='Comma-separated matching-tile fallen bitmasks (0..7), or "all" for every combination.',
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("data/RoboDojo_vqa_v2/make_kong"),
    help="Output directory for the typed VQA sidecar.",
)
parser.add_argument("--overwrite", action="store_true", help="Replace --output-dir if it already exists.")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()

app_launcher = AppLauncher(ARGS)
simulation_app = app_launcher.app

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
from env.observation_manager.obs_manager import ObsManager
from env.seed_manager.seed_manager import SeedManager
from task.RoboDojo import task_registry
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config, process_randomization

# Robot-side tiles are pushed away from the robot and rotate onto the table.
# The opponent discard moves toward the robot and turns face-up first, which is
# the sole visual reference for deciding the matching robot-side tile group.
ROBOT_SIDE_FALLEN_QUATERNION = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
REFERENCE_TILE_FALLEN_QUATERNION = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
ROBOT_SIDE_PUSH_DIRECTION = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
REFERENCE_TILE_PUSH_DIRECTION = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
FALLEN_FORWARD_OFFSET_M = 0.045
FALLEN_Z_OFFSET_M = -0.016
KONG_GROUPS = (
    ("mahjong0_0", "mahjong0_1", "mahjong0_2"),
    ("mahjong1_0", "mahjong1_1", "mahjong1_2"),
    ("mahjong2_0", "mahjong2_1", "mahjong2_2"),
    ("mahjong3_0", "mahjong3_1", "mahjong3_2"),
)
DISTRACTOR_ROBOT_SIDE_GROUP = ("mahjong4_0", "mahjong4_1")
DISCARD_LABELS = ("mahjong5_0", "mahjong6_0", "mahjong7_0", "mahjong8_0")
TILE_VISIBILITY = VisibilityThresholds(32, 4, 0.05, 0.5)
RENDER_STABILIZATION_FRAMES = 12
LOGGER = logging.getLogger("make_kong_vqa")


def _configure_logging() -> None:
    """Keep collector progress while suppressing verbose dependencies."""

    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for dependency in ("curobo", "matplotlib", "PIL"):
        logging.getLogger(dependency).setLevel(logging.WARNING)


def _parse_int_csv(value: str, *, valid: set[int] | None = None, flag: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError(f"{flag} must contain at least one integer.")
    if valid is not None:
        invalid = [item for item in result if item not in valid]
        if invalid:
            raise ValueError(f"{flag} contains invalid values {invalid}; expected values in {sorted(valid)}.")
    return result


def _parse_layout_ids(value: str, seed_manager: SeedManager) -> list[int]:
    if value.strip().lower() == "all":
        layout_ids = sorted(int(layout_id) for layout_id in seed_manager.seed_info)
    else:
        layout_ids = _parse_int_csv(value, flag="--layout-ids")
        unknown = [layout_id for layout_id in layout_ids if layout_id not in seed_manager.seed_info]
        if unknown:
            raise ValueError(f"--layout-ids contains unknown eval layout IDs {unknown}.")
    if ARGS.max_layouts is not None:
        if ARGS.max_layouts <= 0:
            raise ValueError("--max-layouts must be positive.")
        layout_ids = layout_ids[: ARGS.max_layouts]
    if not layout_ids:
        raise ValueError("No layout IDs selected.")
    return layout_ids


def _as_numpy(value: Any, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _build_env_config(device_id: int, seed: int):
    task_name = "make_kong"
    eval_config = load_yaml(Path(ENV_CONFIG_PATH) / "arx_x5.yml")
    eval_config.update(
        {
            "task_name": task_name,
            "num_envs": 1,
            "device_id": device_id,
            "eval_batch": False,
            "policy_name": "vqa_renderer",
            "additional_info": "static_fallen_tile_vqa",
            "seed": seed,
        }
    )
    task_config_path = task_registry.task_config_path(Path(ROOT_DIR) / "task" / BENCHMARK / "config", task_name)
    config = OmegaConf.create(
        {
            "sim": load_yaml(Path(ENV_CONFIG_PATH) / "sim" / f"{eval_config['config']['sim']}.yml"),
            "scene": load_yaml(Path(ENV_CONFIG_PATH) / "scene" / f"{eval_config['config']['scene']}.yml"),
            "camera": load_yaml(Path(ENV_CONFIG_PATH) / "camera" / f"{eval_config['config']['camera']}.yml"),
            "robot": load_yaml(Path(ENV_CONFIG_PATH) / "robot" / f"{eval_config['config']['robot']}.yml"),
            "task_env": load_yaml(task_config_path),
            "eval_cfg": eval_config,
            "deploy_cfg": {},
        }
    )
    config.sim.scene.num_envs = 1
    config.eval_cfg.num_envs = 1
    config.sim.device = f"cuda:{device_id}"
    config.sim.seed = [0]
    config.sim.render = {
        "rendering_mode": "quality",
        "antialiasing_mode": "DLAA",
        "enable_dl_denoiser": True,
        "samples_per_pixel": 1024,
    }
    config = process_randomization(config)
    config, _ = process_config(config, task_name=task_name)
    for robot_cfg in config.robot.robots:
        robot_cfg["need_planner"] = False
    config.camera.default_frequency = ARGS.fps
    config.camera.annotator = {
        "common": {"enabled": False},
        "cam_head": {
            "enabled": True,
            "rgb_capture": {"type": "rgb", "device": "cpu"},
            "depth_capture": {"type": "distance_to_image_plane", "device": "cpu"},
            "instance_capture": {"type": "instance_segmentation_fast", "device": "cpu"},
        },
    }
    return config


def _create_env(config):
    _, task_class = task_registry.load_task_class("make_kong")
    env = task_class(config, simulation_app)
    env.eval_cfg = config.eval_cfg
    env.seed_manager = SeedManager(config.eval_cfg)
    env.seed_manager.init_eval()
    env.scene_manager.layout_manager.replay = True

    obs_manager = ObsManager(
        obs_config=deepcopy(config.eval_cfg.get("observation", {})),
        num_envs=env.num_envs,
        dt=env.dt,
        task_name="make_kong",
        description_cfg=config.eval_cfg.get("description", {}),
        seeds_per_env=env.env_seed_list,
    )
    original_post_setup_scene = env._post_setup_scene

    def post_setup_scene(sim) -> None:
        original_post_setup_scene(sim)
        obs_manager.initialize(env)

    env._post_setup_scene = post_setup_scene
    env.obs_manager = obs_manager
    env.interact = False
    env.step_lim = 100000
    return env


def _reset_layout(env, layout_id: int) -> None:
    env.scene_manager.layout_manager.set_saved_layout(0, env.seed_manager.get_seed_scene_info(layout_id))
    env.reset(seed=[layout_id])
    _ensure_layout_scene_objects(env)
    env.scene_manager.apply_saved_poses(env_idx_list=[0])
    for _ in range(10):
        env.render()
    for _ in range(80):
        env.sim_step(render=False)
    env.obs_manager.reset()
    _restore_robot_home(env)


def _scene_object_exists(env, label: str) -> bool:
    layout = env.scene_manager.layout_manager
    instance_name = layout.get_instance_name(env_idx=0, label=label)
    if instance_name is None:
        return False
    instance_type = layout.instance_type_by_env[0].get(instance_name)
    if instance_type is None:
        return False
    key = f"env0_{instance_type}_{instance_name}"
    objects = env.scene_manager.get_objects(env_ids=[0], object_name=instance_name, object_type=instance_type)
    return objects.get(key) is not None


def _ensure_layout_scene_objects(env) -> None:
    missing_labels = [label for label in _tile_labels() if not _scene_object_exists(env, label)]
    if missing_labels:
        raise RuntimeError(
            "Scene objects are missing after creating a fresh layout environment: "
            f"{missing_labels}. The selected eval layout likely lacks required make_kong labels."
        )


def _restore_robot_home(env) -> None:
    """Teleport target robots home with zero velocity for a static snapshot."""

    env.robot_manager.reset()
    for robot, key in zip(env.robot_manager.robot_list, env.robot_manager.robot_key, strict=True):
        if robot.type != "target":
            continue
        default_joint_pos = key.data.default_joint_pos.clone()
        zero_joint_vel = torch.zeros_like(key.data.default_joint_vel)
        key.write_joint_state_to_sim(default_joint_pos, zero_joint_vel)
        key.set_joint_position_target(default_joint_pos)
    env.sim_step(render=False)
    env.robot_manager.set_robot_init_pose()
    env.robot_manager.set_robot_init_state()


def _label_object(env, label: str):
    layout = env.scene_manager.layout_manager
    instance_name = layout.get_instance_name(env_idx=0, label=label)
    if instance_name is None:
        raise RuntimeError(f"No scene instance for label {label!r}")
    obj = layout.get_scene_object(env_idx=0, inst_name=instance_name)
    if obj is None:
        raise RuntimeError(f"No scene object for label {label!r} (instance {instance_name!r})")
    position, orientation = layout.get_instance_pose(env_idx=0, inst_name=instance_name)
    return obj, _as_numpy(position, dtype=np.float32), _as_numpy(orientation, dtype=np.float32)


def _set_label_pose(env, label: str, position: np.ndarray, orientation: np.ndarray) -> None:
    obj, _, _ = _label_object(env, label)
    obj.set_local_pose(
        translation=np.asarray(position, dtype=np.float32), orientation=np.asarray(orientation, dtype=np.float32)
    )
    if hasattr(obj, "set_velocities"):
        obj.set_velocities(np.zeros((1, 6), dtype=np.float32))


def _robot_side_labels() -> tuple[str, ...]:
    """Return the 12 robot-side tiles."""

    return tuple(label for group in KONG_GROUPS for label in group)


def _tile_labels() -> tuple[str, ...]:
    return tuple(list(_robot_side_labels()) + list(DISTRACTOR_ROBOT_SIDE_GROUP) + list(DISCARD_LABELS) + ["mahjong9_0"])


def _snapshot_tile_poses(env) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        label: (position.copy(), orientation.copy())
        for label in _tile_labels()
        for _, position, orientation in [_label_object(env, label)]
    }


def _restore_tile_poses(env, poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    for label, (position, orientation) in poses.items():
        _set_label_pose(env, label, position, orientation)


def _settle(env, *, sim_steps: int = 20) -> None:
    for _ in range(sim_steps):
        env.sim_step(render=False)
    env.obs_manager.reset()


class AnnotationMappingError(RuntimeError):
    """The renderer could not establish an unambiguous instance mask."""


def _capture_ego_annotations(env) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
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
    return rgb, depth, instance, data["instance_segmentation_fast"][0].get("info", {})


def _reset_ego_render_product(env) -> None:
    """Discard tiled-camera DLAA history after a static-scene teleport."""

    import omni.usd

    camera_names = env.camera_manager.camera_names[0]
    if "cam_head" not in camera_names:
        raise AnnotationMappingError(f"cam_head is unavailable; cameras={camera_names}")
    omni.usd.get_context().reset_renderer_accumulation()
    env.capture_manager.recreate_render_products([camera_names.index("cam_head")])
    for _ in range(RENDER_STABILIZATION_FRAMES):
        env.render()


def _set_semantic_label(entity: Any, semantic_label: str) -> str:
    from isaacsim.core.utils.semantics import add_labels, remove_labels
    import omni.usd

    prim = getattr(entity, "prim", None)
    if prim is None:
        prim_path = getattr(entity, "prim_path", None)
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(str(prim_path)) if prim_path else None
    if not prim or not prim.IsValid():
        raise AnnotationMappingError(f"invalid prim for {semantic_label!r}")
    remove_labels(prim, include_descendants=True)
    add_labels(prim, [semantic_label])
    return str(prim.GetPath())


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
    return _label_matches(value, semantic_label) or _prim_path_matches(value, prim_path)


def _semantic_id_map(
    info: dict[str, Any], semantic_labels: dict[str, str], semantic_prim_paths: dict[str, str]
) -> dict[str, list[int]]:
    labels = info.get("idToLabels") or info.get("id_to_labels") or info.get("labels")
    if not isinstance(labels, dict):
        raise AnnotationMappingError("instance segmentation did not provide idToLabels")
    return {
        label: [
            int(key)
            for key, value in labels.items()
            if _renderer_identity_matches(value, semantic_label, semantic_prim_paths[label])
        ]
        for label, semantic_label in semantic_labels.items()
    }


def _mask_for_semantic(
    instance: np.ndarray,
    info: dict[str, Any],
    semantic_label: str,
    prim_path: str,
) -> np.ndarray:
    labels = info.get("idToLabels") or info.get("id_to_labels") or info.get("labels")
    if not isinstance(labels, dict):
        raise AnnotationMappingError("instance segmentation did not provide idToLabels")
    ids = [int(key) for key, value in labels.items() if _renderer_identity_matches(value, semantic_label, prim_path)]
    if len(ids) != 1:
        raise AnnotationMappingError(
            f"renderer identity for {semantic_label!r} at {prim_path!r} resolves to {len(ids)} instance IDs"
        )
    return instance == ids[0]


def _write_segmentation_audit(
    audit_dir: Path,
    scene_id: str,
    instance: np.ndarray,
    info: dict[str, Any],
    semantic_labels: dict[str, str],
    semantic_prim_paths: dict[str, str],
) -> dict[str, list[int]]:
    """Persist renderer identity evidence before accepting any tile label."""

    semantic_ids = _semantic_id_map(info, semantic_labels, semantic_prim_paths)
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"{scene_id}_instance_ids.json").write_text(
        json.dumps(
            {
                "semantic_labels": semantic_labels,
                "semantic_prim_paths": semantic_prim_paths,
                "matched_instance_ids": semantic_ids,
                "id_to_labels": info.get("idToLabels") or info.get("id_to_labels") or info.get("labels"),
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    color = np.zeros((*instance.shape, 3), dtype=np.uint8)
    for identifier in np.unique(instance):
        identifier = int(identifier)
        if identifier == 0:
            continue
        color[instance == identifier] = (
            (identifier * 37) % 251 + 4,
            (identifier * 73) % 251 + 4,
            (identifier * 109) % 251 + 4,
        )
    Image.fromarray(color).save(audit_dir / f"{scene_id}_instance_ids.png")
    return semantic_ids


def _label_tile_instances(env) -> tuple[dict[str, str], dict[str, str]]:
    layout = env.scene_manager.layout_manager
    labels = (*_robot_side_labels(), *DISCARD_LABELS)
    semantic = {label: f"vqa_make_kong_{label}" for label in labels}
    prim_paths = {}
    for label, semantic_label in semantic.items():
        entity = layout.get_scene_object(0, layout.get_instance_name(0, label))
        if entity is None:
            raise AnnotationMappingError(f"missing tile {label}")
        prim_paths[label] = _set_semantic_label(entity, semantic_label)
    return semantic, prim_paths


def _target_labels_left_to_right(env, target_group: int) -> list[str]:
    labels_with_x = []
    for label in KONG_GROUPS[target_group]:
        _, position, _ = _label_object(env, label)
        labels_with_x.append((label, float(position[0])))
    return [label for label, _ in sorted(labels_with_x, key=lambda item: item[1])]


def _robot_side_labels_left_to_right(env) -> list[str]:
    labels_with_x = []
    for label in _robot_side_labels():
        _, position, _ = _label_object(env, label)
        labels_with_x.append((label, float(position[0])))
    return [label for label, _ in sorted(labels_with_x, key=lambda item: item[1])]


def _fallen_tile_position(position: np.ndarray, direction: np.ndarray) -> np.ndarray:
    fallen_position = np.asarray(position, dtype=np.float32).copy()
    fallen_position += np.asarray(direction, dtype=np.float32) * FALLEN_FORWARD_OFFSET_M
    fallen_position[2] += FALLEN_Z_OFFSET_M
    return fallen_position


def _robot_side_fallen_quaternion() -> np.ndarray:
    return ROBOT_SIDE_FALLEN_QUATERNION.copy()


def _reference_tile_fallen_quaternion() -> np.ndarray:
    return REFERENCE_TILE_FALLEN_QUATERNION.copy()


def _parse_state_patterns(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(8))
    return _parse_int_csv(value, valid=set(range(8)), flag="--state-patterns")


def _default_neighbor_state_patterns(neighbor_count: int) -> list[int]:
    """Return every control/error state for one or two adjacent tiles."""

    if neighbor_count not in {1, 2}:
        raise ValueError(f"expected one or two adjacent nonmatching tiles, got {neighbor_count}")
    return list(range(1 << neighbor_count))


def _render_state(
    env,
    *,
    writer: SidecarWriter,
    image_dir: Path,
    audit_dir: Path,
    layout_id: int,
    target_group: int,
    state_pattern: int,
) -> None:
    target_labels = _target_labels_left_to_right(env, target_group)
    robot_side_labels = _robot_side_labels_left_to_right(env)
    fallen_labels = fallen_labels_for_pattern(target_labels, state_pattern)
    discard_label = DISCARD_LABELS[target_group]
    robot_side_fallen_quaternion = _robot_side_fallen_quaternion()
    reference_tile_fallen_quaternion = _reference_tile_fallen_quaternion()

    # The opponent declares the type first by knocking its discard down with
    # the face visible from cam_head. The robot-side matching relationship is
    # then determined from this rendered reference, not from a hidden prompt.
    _, reference_position, _ = _label_object(env, discard_label)
    reference_fallen_position = _fallen_tile_position(reference_position, REFERENCE_TILE_PUSH_DIRECTION)
    _set_label_pose(env, discard_label, reference_fallen_position, reference_tile_fallen_quaternion)

    for label in fallen_labels:
        _, position, _ = _label_object(env, label)
        fallen_position = _fallen_tile_position(position, ROBOT_SIDE_PUSH_DIRECTION)
        _set_label_pose(env, label, fallen_position, robot_side_fallen_quaternion)
    _restore_robot_home(env)
    _settle(env)

    scene_id = f"make_kong_seed{ARGS.seed}_layout{layout_id:03d}_group{target_group}_pattern{state_pattern:03b}"
    try:
        semantic_labels, semantic_prim_paths = _label_tile_instances(env)
        _reset_ego_render_product(env)
        rgb, _, instance, info = _capture_ego_annotations(env)
        semantic_ids = _write_segmentation_audit(
            audit_dir, scene_id, instance, info, semantic_labels, semantic_prim_paths
        )
        required_labels = set(robot_side_labels) | {discard_label}
        invalid_mappings = {label: semantic_ids[label] for label in required_labels if len(semantic_ids[label]) != 1}
        if invalid_mappings:
            details = ", ".join(f"{label}={ids}" for label, ids in sorted(invalid_mappings.items()))
            raise AnnotationMappingError(f"semantic-to-instance mapping is not one-to-one: {details}")
        masks = {
            label: _mask_for_semantic(instance, info, semantic_labels[label], semantic_prim_paths[label])
            for label in required_labels
        }
    except AnnotationMappingError as error:
        writer.reject(
            base_record(
                sample_id=scene_id, task_name="make_kong", question_family="scene_validation", scene_id=scene_id
            ),
            str(error),
        )
        return
    image_name = f"{scene_id}.png"
    Image.fromarray(rgb).save(image_dir / image_name)
    image_reference = str(Path("images") / image_name)
    reference_status, reference_fraction, _ = classify_mask_visibility(masks[discard_label], TILE_VISIBILITY)

    def record(query_label: str, answer: bool, prompt: str) -> dict[str, Any]:
        status, fraction, occlusion = classify_mask_visibility(masks[query_label], TILE_VISIBILITY)
        query_index = robot_side_labels.index(query_label) + 1
        return base_record(
            sample_id=f"{scene_id}_tile_{query_index}",
            task_name="make_kong",
            question_family="matching_tile_still_needs_action",
            ego_image_reference=image_reference,
            image_width=int(rgb.shape[1]),
            image_height=int(rgb.shape[0]),
            prompt_text=prompt,
            answer_type="boolean",
            answer_bool=answer,
            world_state_valid=True,
            image_answerable=(status == "visible" and reference_status == "visible"),
            visibility_status=status,
            visible_fraction=fraction,
            occlusion_ratio=occlusion,
            target_view="ego",
            gt_source="simulated_tile_identity_and_pose",
            source_layout=f"eval_seed:{ARGS.seed}/layout:{layout_id}",
            scene_id=scene_id,
            audit_metadata_json=json_dumps(
                {
                    "query_label": query_label,
                    "query_left_to_right_index": query_index,
                    "reference_label": discard_label,
                    "reference_visible_fraction": reference_fraction,
                    "reference_visibility_status": reference_status,
                    "semantic_instance_ids": {label: semantic_ids[label][0] for label in sorted(required_labels)},
                    "target_group": target_group,
                    "state_pattern": state_pattern,
                    "fallen_labels": fallen_labels,
                }
            ),
        )

    query_labels = target_labels
    for query_label in query_labels:
        query_index = robot_side_labels.index(query_label) + 1
        writer.add(
            record(
                query_label,
                query_label not in fallen_labels,
                (
                    "Based on the face-up reference tile and the current board state, does the "
                    f"{ordinal(query_index)} tile from the left still need to be knocked down? Answer yes or no."
                ),
            )
        )


def _render_neighbor_state(
    env,
    *,
    writer: SidecarWriter,
    image_dir: Path,
    audit_dir: Path,
    layout_id: int,
    target_group: int,
    neighbor_state_pattern: int,
) -> None:
    """Render a state where adjacent nonmatching tiles may be incorrectly fallen."""

    target_labels = _target_labels_left_to_right(env, target_group)
    robot_side_labels = _robot_side_labels_left_to_right(env)
    neighbor_labels = adjacent_nonmatching_labels(robot_side_labels, target_labels)
    fallen_neighbor_labels = labels_for_bitmask(neighbor_labels, neighbor_state_pattern)
    discard_label = DISCARD_LABELS[target_group]
    _, reference_position, _ = _label_object(env, discard_label)
    _set_label_pose(
        env,
        discard_label,
        _fallen_tile_position(reference_position, REFERENCE_TILE_PUSH_DIRECTION),
        _reference_tile_fallen_quaternion(),
    )
    for label in target_labels:
        _, position, _ = _label_object(env, label)
        _set_label_pose(
            env,
            label,
            _fallen_tile_position(position, ROBOT_SIDE_PUSH_DIRECTION),
            _robot_side_fallen_quaternion(),
        )
    for label in fallen_neighbor_labels:
        _, position, _ = _label_object(env, label)
        _set_label_pose(
            env,
            label,
            _fallen_tile_position(position, ROBOT_SIDE_PUSH_DIRECTION),
            _robot_side_fallen_quaternion(),
        )
    _restore_robot_home(env)
    _settle(env)

    scene_id = (
        f"make_kong_seed{ARGS.seed}_layout{layout_id:03d}_group{target_group}_"
        f"kong_declaration_neighbor_pattern{neighbor_state_pattern:02b}"
    )
    try:
        semantic_labels, semantic_prim_paths = _label_tile_instances(env)
        _reset_ego_render_product(env)
        rgb, _, instance, info = _capture_ego_annotations(env)
        semantic_ids = _write_segmentation_audit(
            audit_dir, scene_id, instance, info, semantic_labels, semantic_prim_paths
        )
        required_labels = set(neighbor_labels) | {discard_label}
        invalid_mappings = {label: semantic_ids[label] for label in required_labels if len(semantic_ids[label]) != 1}
        if invalid_mappings:
            details = ", ".join(f"{label}={ids}" for label, ids in sorted(invalid_mappings.items()))
            raise AnnotationMappingError(f"semantic-to-instance mapping is not one-to-one: {details}")
        masks = {
            label: _mask_for_semantic(instance, info, semantic_labels[label], semantic_prim_paths[label])
            for label in required_labels
        }
    except AnnotationMappingError as error:
        writer.reject(
            base_record(
                sample_id=scene_id,
                task_name="make_kong",
                question_family="scene_validation",
                scene_id=scene_id,
            ),
            str(error),
        )
        return

    image_name = f"{scene_id}.png"
    Image.fromarray(rgb).save(image_dir / image_name)
    image_reference = str(Path("images") / image_name)
    reference_status, reference_fraction, _ = classify_mask_visibility(masks[discard_label], TILE_VISIBILITY)
    for query_label in neighbor_labels:
        status, fraction, occlusion = classify_mask_visibility(masks[query_label], TILE_VISIBILITY)
        query_index = robot_side_labels.index(query_label) + 1
        state_reason = kong_declaration_neighbor_state_reason(query_label in fallen_neighbor_labels)
        writer.add(
            base_record(
                sample_id=f"{scene_id}_tile_{query_index}",
                task_name="make_kong",
                question_family="kong_declaration_neighbor_state_reason",
                ego_image_reference=image_reference,
                image_width=int(rgb.shape[1]),
                image_height=int(rgb.shape[0]),
                prompt_text=(
                    "During kong declaration, only tiles matching the face-up tile should be down. Before the "
                    f"left-stack draw, classify the {ordinal(query_index)} tile: correct or nonmatching_fallen."
                ),
                answer_type="short_text",
                answer_text=state_reason,
                world_state_valid=True,
                image_answerable=(status == "visible" and reference_status == "visible"),
                visibility_status=status,
                visible_fraction=fraction,
                occlusion_ratio=occlusion,
                target_view="ego",
                gt_source="simulated_kong_declaration_state_and_expected_matching_state",
                source_layout=f"eval_seed:{ARGS.seed}/layout:{layout_id}",
                scene_id=scene_id,
                audit_metadata_json=json_dumps(
                    {
                        "query_label": query_label,
                        "query_left_to_right_index": query_index,
                        "expected_state": "upright_nonmatching_tile",
                        "state_reason": state_reason,
                        "kong_declaration_matching_labels_fallen": target_labels,
                        "target_group": target_group,
                        "matching_labels": target_labels,
                        "adjacent_nonmatching_labels": neighbor_labels,
                        "neighbor_state_pattern": neighbor_state_pattern,
                        "incorrectly_fallen_labels": fallen_neighbor_labels,
                        "reference_label": discard_label,
                        "reference_visible_fraction": reference_fraction,
                        "reference_visibility_status": reference_status,
                        "semantic_instance_ids": {label: semantic_ids[label][0] for label in sorted(required_labels)},
                    }
                ),
            )
        )


def _render_matching_indices(
    env,
    *,
    writer: SidecarWriter,
    image_dir: Path,
    audit_dir: Path,
    layout_id: int,
    target_group: int,
) -> None:
    """Render the pre-action scene and identify the three matching tile positions."""

    target_labels = _target_labels_left_to_right(env, target_group)
    robot_side_labels = _robot_side_labels_left_to_right(env)
    discard_label = DISCARD_LABELS[target_group]
    _, reference_position, _ = _label_object(env, discard_label)
    _set_label_pose(
        env,
        discard_label,
        _fallen_tile_position(reference_position, REFERENCE_TILE_PUSH_DIRECTION),
        _reference_tile_fallen_quaternion(),
    )
    _restore_robot_home(env)
    _settle(env)
    scene_id = f"make_kong_seed{ARGS.seed}_layout{layout_id:03d}_group{target_group}_initial"
    try:
        semantic_labels, semantic_prim_paths = _label_tile_instances(env)
        _reset_ego_render_product(env)
        rgb, _, instance, info = _capture_ego_annotations(env)
        semantic_ids = _write_segmentation_audit(
            audit_dir, scene_id, instance, info, semantic_labels, semantic_prim_paths
        )
        required_labels = set(target_labels) | {discard_label}
        invalid_mappings = {label: semantic_ids[label] for label in required_labels if len(semantic_ids[label]) != 1}
        if invalid_mappings:
            details = ", ".join(f"{label}={ids}" for label, ids in sorted(invalid_mappings.items()))
            raise AnnotationMappingError(f"semantic-to-instance mapping is not one-to-one: {details}")
        masks = {
            label: _mask_for_semantic(instance, info, semantic_labels[label], semantic_prim_paths[label])
            for label in required_labels
        }
    except AnnotationMappingError as error:
        writer.reject(
            base_record(
                sample_id=scene_id, task_name="make_kong", question_family="scene_validation", scene_id=scene_id
            ),
            str(error),
        )
        return
    image_name = f"{scene_id}.png"
    Image.fromarray(rgb).save(image_dir / image_name)
    matching_indices = matching_tile_indices(robot_side_labels, target_labels)
    statuses = [classify_mask_visibility(masks[label], TILE_VISIBILITY)[0] for label in required_labels]
    writer.add(
        base_record(
            sample_id=f"{scene_id}_matching_indices",
            task_name="make_kong",
            question_family="matching_tile_indices_to_push",
            ego_image_reference=str(Path("images") / image_name),
            image_width=int(rgb.shape[1]),
            image_height=int(rgb.shape[0]),
            prompt_text=(
                "After the face-up opponent tile is knocked down, which three tiles on our side should be "
                "knocked down? Return their 1-based left-to-right indices as a tuple."
            ),
            answer_type="short_text",
            answer_text=f"({matching_indices[0]}, {matching_indices[1]}, {matching_indices[2]})",
            world_state_valid=True,
            image_answerable=all(status == "visible" for status in statuses),
            visibility_status="visible" if all(status == "visible" for status in statuses) else "partially_visible",
            target_view="ego",
            gt_source="simulated_tile_identity_and_pose",
            source_layout=f"eval_seed:{ARGS.seed}/layout:{layout_id}",
            scene_id=scene_id,
            audit_metadata_json=json_dumps(
                {
                    "matching_left_to_right_indices": matching_indices,
                    "reference_label": discard_label,
                    "target_group": target_group,
                    "semantic_instance_ids": {label: semantic_ids[label][0] for label in sorted(required_labels)},
                }
            ),
        )
    )


def main() -> None:
    _configure_logging()
    target_groups = _parse_int_csv(ARGS.target_groups, valid={0, 1, 2, 3}, flag="--target-groups")
    state_patterns = _parse_state_patterns(ARGS.state_patterns)
    neighbor_state_patterns_by_group: dict[int, list[int]] = {}
    if ARGS.fps <= 0:
        raise ValueError("--fps must be positive.")
    writer = SidecarWriter(ARGS.output_dir, overwrite=ARGS.overwrite)
    image_dir = writer.prepare_images_dir()
    audit_dir = writer.prepare_audit_dir()

    env = None
    try:
        config = _build_env_config(ARGS.device_id, ARGS.seed)
        planner_flags = [(cfg.get("robot_name"), cfg.get("need_planner")) for cfg in config.robot.robots]
        LOGGER.info("planner flags=%s", planner_flags)
        seed_manager = SeedManager(config.eval_cfg)
        seed_manager.init_eval()
        layout_ids = _parse_layout_ids(ARGS.layout_ids, seed_manager)
        LOGGER.info(
            "seed=%s layouts=%s groups=%s state_patterns=%s adjacent-state-coverage=all output=%s",
            ARGS.seed,
            layout_ids,
            target_groups,
            state_patterns,
            ARGS.output_dir,
        )
        for layout_id in layout_ids:
            env = _create_env(deepcopy(config))
            try:
                LOGGER.info("creating and resetting layout=%s", layout_id)
                _reset_layout(env, layout_id)
                base_poses = _snapshot_tile_poses(env)
                for target_group in target_groups:
                    target_labels = _target_labels_left_to_right(env, target_group)
                    robot_side_labels = _robot_side_labels_left_to_right(env)
                    neighbor_labels = adjacent_nonmatching_labels(robot_side_labels, target_labels)
                    neighbor_state_patterns = _default_neighbor_state_patterns(len(neighbor_labels))
                    existing_patterns = neighbor_state_patterns_by_group.setdefault(
                        target_group, neighbor_state_patterns
                    )
                    if existing_patterns != neighbor_state_patterns:
                        raise RuntimeError(
                            f"adjacent tile order changed across layouts for target group {target_group}"
                        )
                    _restore_tile_poses(env, base_poses)
                    _render_matching_indices(
                        env,
                        writer=writer,
                        image_dir=image_dir,
                        audit_dir=audit_dir,
                        layout_id=layout_id,
                        target_group=target_group,
                    )
                    for state_pattern in state_patterns:
                        _restore_tile_poses(env, base_poses)
                        LOGGER.info("capturing layout=%s group=%s pattern=%s", layout_id, target_group, state_pattern)
                        _render_state(
                            env,
                            writer=writer,
                            image_dir=image_dir,
                            audit_dir=audit_dir,
                            layout_id=layout_id,
                            target_group=target_group,
                            state_pattern=state_pattern,
                        )
                    for neighbor_state_pattern in neighbor_state_patterns:
                        _restore_tile_poses(env, base_poses)
                        LOGGER.info(
                            "capturing layout=%s group=%s adjacent-pattern=%s",
                            layout_id,
                            target_group,
                            neighbor_state_pattern,
                        )
                        _render_neighbor_state(
                            env,
                            writer=writer,
                            image_dir=image_dir,
                            audit_dir=audit_dir,
                            layout_id=layout_id,
                            target_group=target_group,
                            neighbor_state_pattern=neighbor_state_pattern,
                        )
            finally:
                env.close()
                env = None

        report = writer.write(
            {
                "collector": "scripts/internal/generate_make_kong_vqa.py",
                "task_name": "make_kong",
                "source_action_dataset": "RoboDojo_ee_lerobot_v30_video",
                "seed": ARGS.seed,
                "layout_ids": layout_ids,
                "max_layouts": ARGS.max_layouts,
                "target_groups": target_groups,
                "state_patterns": state_patterns,
                "neighbor_state_patterns": {
                    str(group): patterns for group, patterns in sorted(neighbor_state_patterns_by_group.items())
                },
                "neighbor_state_pattern_semantics": {
                    "bit_0": "immediate left adjacent nonmatching tile",
                    "bit_1": "immediate right adjacent nonmatching tile",
                    "zero": "both adjacent nonmatching tiles remain upright and correct",
                },
                "camera": "cam_head",
                "fps": ARGS.fps,
                "robot_side_tiles": len(_robot_side_labels()),
                "fallen_forward_offset_m": FALLEN_FORWARD_OFFSET_M,
                "fallen_z_offset_m": FALLEN_Z_OFFSET_M,
                "render": {
                    "antialiasing_mode": "DLAA",
                    "stabilization_frames": RENDER_STABILIZATION_FRAMES,
                    "ego_render_product_recreated_per_snapshot": True,
                },
                "command": " ".join(sys.argv),
            }
        )
        LOGGER.info("wrote %s accepted records to %s", report["accepted_records"], ARGS.output_dir)
    except Exception as error:
        LOGGER.exception("generation failed: %s", error)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
