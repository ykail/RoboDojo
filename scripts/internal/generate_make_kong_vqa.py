"""Render static make_kong VQA samples for target-tile recognition.

For each selected layout and target group, this collector keeps all robots at
their initial/home pose, synthetically sets 0/1/2/3 matching robot-side tiles as
already pushed down, and saves head/wrist RGB images plus VQA metadata. The VQA
target distinguishes tiles that are already pushed down from matching tiles that
still need to be pushed down to declare a kong.

An example command to generate a small set of samples is:
    python scripts/internal/generate_make_kong_vqa.py \
        --headless --enable_cameras --device-id 0 --seed 0 \
        --layout-ids all --max-layouts 3 --target-groups 0 --fallen-counts 0 \
        --cameras cam_head --output-dir /tmp/make_kong_vqa_all_check --overwrite

"""

import argparse
from itertools import combinations
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# RoboDojo must precede XPolicyLab: both have a top-level ``utils`` package.
for package_root in (REPO_ROOT / "XPolicyLab", REPO_ROOT):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from isaaclab.app import AppLauncher


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
    "--fallen-counts",
    default="0,1,2",
    help="Comma-separated counts of matching robot-side tiles to set already pushed down. Valid values: 0,1,2,3.",
)
parser.add_argument(
    "--cameras",
    default="cam_head",
    help="Comma-separated cameras to save. Full choices: cam_head,cam_left_wrist,cam_right_wrist",
)
parser.add_argument(
    "--question",
    default=(
        "Look at the face-up target tile above the robot-side row."
        "Number the 12 robot-side tiles from left to right as 1 through 12."
        "Which matching tiles are already lying down, and which matching tiles remain upright and should be pushed down to make a kong? "
        "Return exactly: <already_pushed>...</already_pushed><need_push>...</need_push>. "
        "Use none for an empty set."
    ),
    help="VQA question text stored with every sample.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("output/make_kong_vqa"),
    help="Output directory for PNG images and JSONL metadata.",
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
            f"{missing_labels}. This usually means the selected eval layout does not contain the required make_kong labels."
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
    obj.set_local_pose(translation=np.asarray(position, dtype=np.float32), orientation=np.asarray(orientation, dtype=np.float32))


def _robot_side_labels() -> tuple[str, ...]:
    return tuple(label for group in (*KONG_GROUPS, DISTRACTOR_ROBOT_SIDE_GROUP) for label in group)


def _tile_labels() -> tuple[str, ...]:
    return tuple(list(_robot_side_labels()) + list(DISCARD_LABELS) + ["mahjong9_0"])


def _snapshot_tile_poses(env) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {label: (position.copy(), orientation.copy()) for label in _tile_labels() for _, position, orientation in [_label_object(env, label)]}


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


def _format_tile_ids(tile_ids: list[int]) -> str:
    if not tile_ids:
        return "none"
    return ",".join(str(tile_id) for tile_id in tile_ids)


def _answer(already_pushed_tile_ids: list[int], need_push_tile_ids: list[int]) -> str:
    return (
        f"<already_pushed>{_format_tile_ids(already_pushed_tile_ids)}</already_pushed>"
        f"<need_push>{_format_tile_ids(need_push_tile_ids)}</need_push>"
    )


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _render_state(
    env,
    *,
    layout_id: int,
    target_group: int,
    fallen_count: int,
    cameras: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    target_labels = _target_labels_left_to_right(env, target_group)
    robot_side_labels = _robot_side_labels_left_to_right(env)
    target_tile_ids = _tile_indices(robot_side_labels, target_labels)
    fallen_labels = _balanced_fallen_labels(
        target_labels,
        layout_id=layout_id,
        target_group=target_group,
        fallen_count=fallen_count,
    )
    fallen_label_set = set(fallen_labels)
    not_fallen_labels = [label for label in target_labels if label not in fallen_label_set]
    already_pushed_tile_ids = _tile_indices(robot_side_labels, fallen_labels)
    need_push_tile_ids = _tile_indices(robot_side_labels, not_fallen_labels)
    discard_label = DISCARD_LABELS[target_group]
    fallen_pose_by_label = {}
    robot_side_fallen_quaternion = _robot_side_fallen_quaternion()
    opponent_side_fallen_quaternion = _opponent_side_fallen_quaternion()

    _, discard_position, _ = _label_object(env, discard_label)
    discard_fallen_position = _fallen_tile_position(discard_position, OPPONENT_SIDE_PUSH_DIRECTION)
    _set_label_pose(env, discard_label, discard_fallen_position, opponent_side_fallen_quaternion)
    fallen_pose_by_label[discard_label] = {
        "side": "opponent",
        "push_direction_xyz": OPPONENT_SIDE_PUSH_DIRECTION.tolist(),
        "upright_position": discard_position.tolist(),
        "fallen_position": discard_fallen_position.tolist(),
        "fallen_quaternion_wxyz": opponent_side_fallen_quaternion.tolist(),
    }

    for label in fallen_labels:
        _, position, _ = _label_object(env, label)
        fallen_position = _fallen_tile_position(position, ROBOT_SIDE_PUSH_DIRECTION)
        _set_label_pose(env, label, fallen_position, robot_side_fallen_quaternion)
        fallen_pose_by_label[label] = {
            "side": "robot",
            "push_direction_xyz": ROBOT_SIDE_PUSH_DIRECTION.tolist(),
            "upright_position": position.tolist(),
            "fallen_position": fallen_position.tolist(),
            "fallen_quaternion_wxyz": robot_side_fallen_quaternion.tolist(),
        }
    _restore_robot_home(env)
    _settle(env)

    rel_dir = Path(f"layout_{layout_id:03d}") / f"target_group_{target_group}" / f"fallen_{fallen_count}"
    image_dir = output_dir / "images" / rel_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths: dict[str, str] = {}
    for camera, image in _capture_images(env, cameras).items():
        path = image_dir / f"{camera}.png"
        Image.fromarray(np.ascontiguousarray(image)).save(path)
        image_paths[camera] = str(path.relative_to(output_dir))

    row = {
        "sample_id": f"make_kong_seed{ARGS.seed}_layout{layout_id:03d}_group{target_group}_fallen{fallen_count}",
        "task": "make_kong",
        "seed": ARGS.seed,
        "layout_id": layout_id,
        "target_group": target_group,
        "discard_label": discard_label,
        "robot_side_tiles": [
            {
                "index": idx + 1,
                "label": label,
                "category_index": _category_index(env, label),
            }
            for idx, label in enumerate(robot_side_labels)
        ],
        "target_labels": target_labels,
        "target_tile_ids": target_tile_ids,
        "fallen_count": fallen_count,
        "fallen_selection_rule": "balanced_deterministic_cycle" if 0 < fallen_count < len(target_labels) else "boundary_count",
        "fallen_labels": fallen_labels,
        "fallen_tile_ids": already_pushed_tile_ids,
        "already_pushed_tile_ids": already_pushed_tile_ids,
        "not_fallen_labels": not_fallen_labels,
        "not_fallen_tile_ids": need_push_tile_ids,
        "need_push_tile_ids": need_push_tile_ids,
        "answer_labels": target_labels,
        "target_category_indices": {label: _category_index(env, label) for label in target_labels},
        "not_fallen_category_indices": {label: _category_index(env, label) for label in not_fallen_labels},
        "fallen_pose_by_label": fallen_pose_by_label,
        "fallen_pose_rule": {
            "forward_offset_m": FALLEN_FORWARD_OFFSET_M,
            "z_offset_m": FALLEN_Z_OFFSET_M,
            "robot_side": {
                "direction": "toward_opponent_positive_y",
                "direction_xyz": ROBOT_SIDE_PUSH_DIRECTION.tolist(),
                "quaternion_wxyz": robot_side_fallen_quaternion.tolist(),
            },
            "opponent_side": {
                "direction": "toward_robot_negative_y",
                "direction_xyz": OPPONENT_SIDE_PUSH_DIRECTION.tolist(),
                "quaternion_wxyz": opponent_side_fallen_quaternion.tolist(),
            },
        },
        "question": ARGS.question,
        "answer": _answer(already_pushed_tile_ids, need_push_tile_ids),
        "images": image_paths,
    }
    return row


def main() -> None:
    target_groups = _parse_int_csv(ARGS.target_groups, valid={0, 1, 2, 3}, flag="--target-groups")
    fallen_counts = _parse_int_csv(ARGS.fallen_counts, valid={0, 1, 2, 3}, flag="--fallen-counts")
    cameras = _parse_cameras(ARGS.cameras)
    robot_side_fallen_quaternion = _robot_side_fallen_quaternion()
    opponent_side_fallen_quaternion = _opponent_side_fallen_quaternion()
    if ARGS.fps <= 0:
        raise ValueError("--fps must be positive.")
    _prepare_output_dir(ARGS.output_dir, ARGS.overwrite)

    env = None
    rows: list[dict[str, Any]] = []
    try:
        config = _build_env_config(ARGS.device_id, ARGS.seed)
        print(f"[make_kong_vqa] planner flags={[(cfg.get('robot_name'), cfg.get('need_planner')) for cfg in config.robot.robots]}", flush=True)
        seed_manager = SeedManager(config.eval_cfg)
        seed_manager.init_eval()
        layout_ids = _parse_layout_ids(ARGS.layout_ids, seed_manager)
        print(
            f"[make_kong_vqa] seed={ARGS.seed} layouts={layout_ids} groups={target_groups} "
            f"fallen_counts={fallen_counts} cameras={cameras} output={ARGS.output_dir}",
            flush=True,
        )
        for layout_id in layout_ids:
            env = _create_env(deepcopy(config))
            try:
                _reset_layout(env, layout_id)
                base_poses = _snapshot_tile_poses(env)
                for target_group in target_groups:
                    for fallen_count in fallen_counts:
                        _restore_tile_poses(env, base_poses)
                        row = _render_state(
                            env,
                            layout_id=layout_id,
                            target_group=target_group,
                            fallen_count=fallen_count,
                            cameras=cameras,
                            output_dir=ARGS.output_dir,
                        )
                        rows.append(row)
                        print(f"[make_kong_vqa] wrote {row['sample_id']}", flush=True)
            finally:
                env.close()
                env = None

        per_image_rows = []
        vqa_rows = []
        for row in rows:
            for camera, image_path in row["images"].items():
                per_image = dict(row)
                per_image.pop("images")
                per_image["camera"] = camera
                per_image["image"] = image_path
                per_image["sample_id"] = f"{row['sample_id']}_{camera}"
                per_image_rows.append(per_image)
                vqa_rows.append(
                    {
                        "image": image_path,
                        "question": row["question"],
                        "answer": row["answer"],
                        "target_tile_ids": row["target_tile_ids"],
                        "already_pushed_tile_ids": row["already_pushed_tile_ids"],
                        "need_push_tile_ids": row["need_push_tile_ids"],
                    }
                )

        _save_jsonl(ARGS.output_dir / "samples.jsonl", rows)
        _save_jsonl(ARGS.output_dir / "per_image_samples.jsonl", per_image_rows)
        _save_jsonl(ARGS.output_dir / "vqa_samples.jsonl", vqa_rows)
        with (ARGS.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "collector": "scripts/internal/generate_make_kong_vqa.py",
                    "seed": ARGS.seed,
                    "layout_ids": layout_ids,
                    "max_layouts": ARGS.max_layouts,
                    "target_groups": target_groups,
                    "fallen_counts": fallen_counts,
                    "cameras": cameras,
                    "fps": ARGS.fps,
                    "fallen_forward_offset_m": FALLEN_FORWARD_OFFSET_M,
                    "fallen_z_offset_m": FALLEN_Z_OFFSET_M,
                    "robot_side_push_direction_xyz": ROBOT_SIDE_PUSH_DIRECTION.tolist(),
                    "robot_side_fallen_quaternion_wxyz": robot_side_fallen_quaternion.tolist(),
                    "opponent_side_push_direction_xyz": OPPONENT_SIDE_PUSH_DIRECTION.tolist(),
                    "opponent_side_fallen_quaternion_wxyz": opponent_side_fallen_quaternion.tolist(),
                    "question": ARGS.question,
                    "state_rows": len(rows),
                    "per_image_rows": len(per_image_rows),
                    "vqa_rows": len(vqa_rows),
                    "notes": "Robots are held at init/home pose. The opponent discard tile is pushed toward the robot with a 180-degree yaw. The selected matching robot-side tiles are pushed toward the opponent, lowered to table height, and kept face-up. VQA answers distinguish matching tiles already pushed down from matching tiles that still need to be pushed down to declare a kong.",
                },
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        print(f"[make_kong_vqa] wrote {len(rows)} states and {len(per_image_rows)} image VQA rows to {ARGS.output_dir}", flush=True)
    except Exception:
        if not rows:
            shutil.rmtree(ARGS.output_dir, ignore_errors=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
