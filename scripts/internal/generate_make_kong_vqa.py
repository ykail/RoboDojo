"""Render typed, ego-only VQA sidecar samples for ``make_kong``.

For each selected layout and target group, the collector holds robots at home,
enumerates the eight matching-tile fallen bitmasks, and writes only numbered
``cam_head`` images with physical Parquet annotations.  The two outputs
are fixed-answer boolean VQA families; no custom variable-length answer string
or wrist image is generated.

Example:
    python scripts/internal/generate_make_kong_vqa.py \
        --headless --enable_cameras --device-id 0 --seed 0 \
        --max-layouts 1 --target-groups 0 --state-patterns all \
        --output-dir /tmp/make_kong_vqa_check --overwrite
"""

import argparse
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import sys
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# RoboDojo must precede XPolicyLab: both have a top-level ``utils`` package.
for package_root in (REPO_ROOT / "XPolicyLab", REPO_ROOT):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from isaaclab.app import AppLauncher

from scripts.internal.vqa.overlay import OverlayError, numbered_overlay
from scripts.internal.vqa.sidecar import (
    SidecarWriter,
    VisibilityThresholds,
    base_record,
    classify_mask_visibility,
    json_dumps,
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
    default=Path("data/RoboDojo_vqa_v1/make_kong"),
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

# Robot-side tiles are pushed away from the robot. The opponent tile is pushed
# from the opposite side, so it moves toward the robot and needs a 180-degree
# yaw to keep the printed face orientation consistent.
ROBOT_SIDE_FALLEN_QUATERNION = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
OPPONENT_SIDE_FALLEN_QUATERNION = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
ROBOT_SIDE_PUSH_DIRECTION = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
OPPONENT_SIDE_PUSH_DIRECTION = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
FALLEN_FORWARD_OFFSET_M = 0.045
FALLEN_Z_OFFSET_M = -0.016
CAMERA_ALIASES = {
    "cam_high": "cam_head",
    "cam_head": "cam_head",
    "head": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
    "right_wrist": "cam_right_wrist",
}
KONG_GROUPS = (
    ("mahjong0_0", "mahjong0_1", "mahjong0_2"),
    ("mahjong1_0", "mahjong1_1", "mahjong1_2"),
    ("mahjong2_0", "mahjong2_1", "mahjong2_2"),
    ("mahjong3_0", "mahjong3_1", "mahjong3_2"),
)
DISTRACTOR_ROBOT_SIDE_GROUP = ("mahjong4_0", "mahjong4_1")
DISCARD_LABELS = ("mahjong5_0", "mahjong6_0", "mahjong7_0", "mahjong8_0")
TILE_VISIBILITY = VisibilityThresholds(32, 4, 0.05, 0.5)


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


def _parse_cameras(value: str) -> list[str]:
    cameras = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        if item not in CAMERA_ALIASES:
            raise ValueError(f"Unknown camera {item!r}; valid values are {sorted(CAMERA_ALIASES)}.")
        camera = CAMERA_ALIASES[item]
        if camera not in cameras:
            cameras.append(camera)
    if not cameras:
        raise ValueError("--cameras must contain at least one camera.")
    return cameras


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
    env.robot_manager.set_robot_init_pose()
    for _ in range(40):
        env.sim_step(render=False)
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


def _robot_side_labels() -> tuple[str, ...]:
    return tuple(label for group in (*KONG_GROUPS, DISTRACTOR_ROBOT_SIDE_GROUP) for label in group)


def _tile_labels() -> tuple[str, ...]:
    return tuple(list(_robot_side_labels()) + list(DISCARD_LABELS) + ["mahjong9_0"])


def _snapshot_tile_poses(env) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        label: (position.copy(), orientation.copy())
        for label in _tile_labels()
        for _, position, orientation in [_label_object(env, label)]
    }


def _restore_tile_poses(env, poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    for label, (position, orientation) in poses.items():
        _set_label_pose(env, label, position, orientation)


def _settle(env, *, sim_steps: int = 20, render_frames: int = 12) -> None:
    for _ in range(sim_steps):
        env.sim_step(render=False)
    for _ in range(render_frames):
        env.render()
    env.obs_manager.reset()


def _capture_images(env, cameras: list[str]) -> dict[str, np.ndarray]:
    env.render()
    observation = env.obs_manager.get_obs(env_idx_list=[0])[0]
    vision = observation.get("vision", {})
    images = {}
    for camera in cameras:
        color = vision.get(camera, {}).get("color")
        if color is None:
            raise RuntimeError(f"Expected {camera} image, found cameras {sorted(vision)}")
        images[camera] = _as_numpy(color, dtype=np.uint8).copy()
    return images


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


def _set_semantic_label(entity: Any, semantic_label: str) -> None:
    from isaacsim.core.utils.semantics import add_labels
    import omni.usd

    prim = getattr(entity, "prim", None)
    if prim is None:
        prim_path = getattr(entity, "prim_path", None)
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(str(prim_path)) if prim_path else None
    if not prim or not prim.IsValid():
        raise AnnotationMappingError(f"invalid prim for {semantic_label!r}")
    add_labels(prim, [semantic_label])


def _label_matches(value: Any, semantic_label: str) -> bool:
    if isinstance(value, dict):
        return any(_label_matches(item, semantic_label) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_label_matches(item, semantic_label) for item in value)
    return semantic_label in str(value)


def _mask_for_semantic(instance: np.ndarray, info: dict[str, Any], semantic_label: str) -> np.ndarray:
    labels = info.get("idToLabels") or info.get("id_to_labels") or info.get("labels")
    if not isinstance(labels, dict):
        raise AnnotationMappingError("instance segmentation did not provide idToLabels")
    ids = [int(key) for key, value in labels.items() if _label_matches(value, semantic_label)]
    if len(ids) != 1:
        raise AnnotationMappingError(f"semantic label {semantic_label!r} resolves to {len(ids)} IDs")
    return instance == ids[0]


def _label_tile_instances(env) -> dict[str, str]:
    layout = env.scene_manager.layout_manager
    labels = (*_robot_side_labels(), *DISCARD_LABELS)
    semantic = {label: f"vqa_make_kong_{label}" for label in labels}
    for label, semantic_label in semantic.items():
        entity = layout.get_scene_object(0, layout.get_instance_name(0, label))
        if entity is None:
            raise AnnotationMappingError(f"missing tile {label}")
        _set_semantic_label(entity, semantic_label)
    return semantic


def _mask_anchor(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise AnnotationMappingError("cannot anchor empty tile mask")
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def _category_index(env, label: str) -> int | None:
    saved_layout = env.scene_manager.layout_manager.saved_layouts[0]
    if saved_layout is not None:
        for object_group in saved_layout.values():
            if not isinstance(object_group, dict):
                continue
            for instances in object_group.values():
                if not isinstance(instances, list):
                    continue
                for instance in instances:
                    if instance.get("label") == label and instance.get("category_idx") is not None:
                        return int(instance["category_idx"])
    metadata = env.scene_manager.layout_manager.get_instance_metadata(env_idx=0, label=label)
    if metadata is None:
        return None
    value = metadata.get("category_idx")
    return None if value is None else int(value)


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


def _tile_indices(robot_side_labels: list[str], labels: list[str]) -> list[int]:
    label_to_index = {label: idx + 1 for idx, label in enumerate(robot_side_labels)}
    return sorted(label_to_index[label] for label in labels)


def _balanced_fallen_labels(
    target_labels: list[str],
    *,
    layout_id: int,
    target_group: int,
    fallen_count: int,
) -> list[str]:
    if fallen_count < 0 or fallen_count > len(target_labels):
        raise ValueError(f"fallen_count must be between 0 and {len(target_labels)}, got {fallen_count}.")
    if fallen_count in {0, len(target_labels)}:
        return target_labels[:fallen_count]

    index_combinations = list(combinations(range(len(target_labels)), fallen_count))
    combo_index = (layout_id + target_group + ARGS.seed * len(KONG_GROUPS)) % len(index_combinations)
    selected_indices = index_combinations[combo_index]
    return [target_labels[index] for index in selected_indices]


def _fallen_tile_position(position: np.ndarray, direction: np.ndarray) -> np.ndarray:
    fallen_position = np.asarray(position, dtype=np.float32).copy()
    fallen_position += np.asarray(direction, dtype=np.float32) * FALLEN_FORWARD_OFFSET_M
    fallen_position[2] += FALLEN_Z_OFFSET_M
    return fallen_position


def _robot_side_fallen_quaternion() -> np.ndarray:
    return ROBOT_SIDE_FALLEN_QUATERNION.copy()


def _opponent_side_fallen_quaternion() -> np.ndarray:
    return OPPONENT_SIDE_FALLEN_QUATERNION.copy()


def _parse_state_patterns(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(8))
    return _parse_int_csv(value, valid=set(range(8)), flag="--state-patterns")


def _fallen_labels_for_pattern(target_labels: list[str], pattern: int) -> list[str]:
    return [label for index, label in enumerate(target_labels) if pattern & (1 << index)]


def _render_state(
    env,
    *,
    writer: SidecarWriter,
    image_dir: Path,
    semantic_labels: dict[str, str],
    layout_id: int,
    target_group: int,
    state_pattern: int,
) -> None:
    target_labels = _target_labels_left_to_right(env, target_group)
    robot_side_labels = _robot_side_labels_left_to_right(env)
    fallen_labels = _fallen_labels_for_pattern(target_labels, state_pattern)
    fallen_label_set = set(fallen_labels)
    discard_label = DISCARD_LABELS[target_group]
    robot_side_fallen_quaternion = _robot_side_fallen_quaternion()
    opponent_side_fallen_quaternion = _opponent_side_fallen_quaternion()

    _, discard_position, _ = _label_object(env, discard_label)
    discard_fallen_position = _fallen_tile_position(discard_position, OPPONENT_SIDE_PUSH_DIRECTION)
    _set_label_pose(env, discard_label, discard_fallen_position, opponent_side_fallen_quaternion)

    for label in fallen_labels:
        _, position, _ = _label_object(env, label)
        fallen_position = _fallen_tile_position(position, ROBOT_SIDE_PUSH_DIRECTION)
        _set_label_pose(env, label, fallen_position, robot_side_fallen_quaternion)
    _restore_robot_home(env)
    _settle(env)

    scene_id = f"make_kong_seed{ARGS.seed}_layout{layout_id:03d}_group{target_group}_pattern{state_pattern:03b}"
    try:
        rgb, _, instance, info = _capture_ego_annotations(env)
        masks = {label: _mask_for_semantic(instance, info, semantic) for label, semantic in semantic_labels.items()}
        mark_to_label = {str(index): label for index, label in enumerate(robot_side_labels, start=1)}
        overlay = numbered_overlay(rgb, {mark: _mask_anchor(masks[label]) for mark, label in mark_to_label.items()})
    except (AnnotationMappingError, OverlayError) as error:
        writer.reject(
            base_record(
                sample_id=scene_id, task_name="make_kong", question_family="scene_validation", scene_id=scene_id
            ),
            str(error),
        )
        return
    image_name = f"{scene_id}_tile_marks.png"
    Image.fromarray(overlay.image).save(image_dir / image_name)
    image_reference = str(Path("images") / image_name)
    reference_status, reference_fraction, _ = classify_mask_visibility(masks[discard_label], TILE_VISIBILITY)
    label_to_mark = {label: mark for mark, label in mark_to_label.items()}
    overlay_mapping = json_dumps(
        {mark: env.scene_manager.layout_manager.get_instance_name(0, label) for mark, label in mark_to_label.items()}
    )

    def record(family: str, suffix: str, query_label: str, answer: bool, prompt: str) -> dict[str, Any]:
        status, fraction, occlusion = classify_mask_visibility(masks[query_label], TILE_VISIBILITY)
        return base_record(
            sample_id=f"{scene_id}_{suffix}",
            task_name="make_kong",
            question_family=family,
            ego_image_reference=image_reference,
            image_width=int(rgb.shape[1]),
            image_height=int(rgb.shape[0]),
            prompt_text=prompt,
            answer_type="boolean",
            answer_bool=answer,
            world_state_valid=True,
            image_answerable=(
                status in {"visible", "partially_visible"} and reference_status in {"visible", "partially_visible"}
            ),
            visibility_status=status,
            visible_fraction=fraction,
            occlusion_ratio=occlusion,
            gt_source="simulated_tile_identity_and_pose",
            overlay_type="numbered_object_marks",
            overlay_version="make_kong_v1",
            overlay_mark_to_instance_json=overlay_mapping,
            source_layout=f"eval_seed:{ARGS.seed}/layout:{layout_id}",
            scene_id=scene_id,
            audit_metadata_json=json_dumps(
                {
                    "query_label": query_label,
                    "query_mark": int(label_to_mark[query_label]),
                    "reference_label": discard_label,
                    "reference_visible_fraction": reference_fraction,
                    "target_group": target_group,
                    "state_pattern": state_pattern,
                    "fallen_labels": fallen_labels,
                }
            ),
        )

    non_matching = [
        label for label in robot_side_labels if label not in fallen_label_set and label not in target_labels
    ]
    # Every state has three positive and three deterministically rotated negative identity questions.
    selected_negative = [non_matching[(layout_id + state_pattern + offset) % len(non_matching)] for offset in range(3)]
    for query_label in [*target_labels, *selected_negative]:
        mark = label_to_mark[query_label]
        writer.add(
            record(
                "tile_matches_reference",
                f"matches_{mark}",
                query_label,
                query_label in target_labels,
                f"Does the robot-side tile marked {mark} match the face-up reference tile? Answer yes or no.",
            )
        )
    for query_label in target_labels:
        mark = label_to_mark[query_label]
        writer.add(
            record(
                "matching_tile_already_pushed",
                f"pushed_{mark}",
                query_label,
                query_label in fallen_label_set,
                (
                    f"The tile marked {mark} matches the face-up reference tile. "
                    "Is it already lying down? Answer yes or no."
                ),
            )
        )


def main() -> None:
    target_groups = _parse_int_csv(ARGS.target_groups, valid={0, 1, 2, 3}, flag="--target-groups")
    state_patterns = _parse_state_patterns(ARGS.state_patterns)
    if ARGS.fps <= 0:
        raise ValueError("--fps must be positive.")
    writer = SidecarWriter(ARGS.output_dir, overwrite=ARGS.overwrite)
    image_dir = writer.prepare_images_dir()

    env = None
    try:
        config = _build_env_config(ARGS.device_id, ARGS.seed)
        planner_flags = [(cfg.get("robot_name"), cfg.get("need_planner")) for cfg in config.robot.robots]
        print(f"[make_kong_vqa] planner flags={planner_flags}", flush=True)
        seed_manager = SeedManager(config.eval_cfg)
        seed_manager.init_eval()
        layout_ids = _parse_layout_ids(ARGS.layout_ids, seed_manager)
        print(
            f"[make_kong_vqa] seed={ARGS.seed} layouts={layout_ids} groups={target_groups} "
            f"state_patterns={state_patterns} output={ARGS.output_dir}",
            flush=True,
        )
        for layout_id in layout_ids:
            env = _create_env(deepcopy(config))
            try:
                _reset_layout(env, layout_id)
                semantic_labels = _label_tile_instances(env)
                base_poses = _snapshot_tile_poses(env)
                for target_group in target_groups:
                    for state_pattern in state_patterns:
                        _restore_tile_poses(env, base_poses)
                        _render_state(
                            env,
                            writer=writer,
                            image_dir=image_dir,
                            semantic_labels=semantic_labels,
                            layout_id=layout_id,
                            target_group=target_group,
                            state_pattern=state_pattern,
                        )
                        print(
                            f"[make_kong_vqa] rendered layout={layout_id} group={target_group} pattern={state_pattern}",
                            flush=True,
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
                "camera": "cam_head",
                "fps": ARGS.fps,
                "fallen_forward_offset_m": FALLEN_FORWARD_OFFSET_M,
                "fallen_z_offset_m": FALLEN_Z_OFFSET_M,
                "command": " ".join(sys.argv),
            }
        )
        print(f"[make_kong_vqa] wrote {report['accepted_records']} accepted records to {ARGS.output_dir}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
