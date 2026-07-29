"""Collect executed make_kong recovery episodes in LeRobot v3.0 video format.

Unlike ``generate_make_kong_recovery.py``, this collector does not write
single rendered states.  Every sample is a 25 Hz episode with three AV1 MP4
streams, 16-D end-effector state/action records, and standard v3 metadata.

The source failure state is still deterministic pose injection: it represents
the already-made policy error.  Recovery is then executed in the simulator.
cuRobo plans every free-space transfer (home/approach/retreat); deliberate
tile-contact segments use IK targets, because collision avoidance would make
the required push/grasp impossible.  Tile pose changes at contact are made
explicitly and recorded in ``meta/recovery_manifest.json`` so that this does
not misrepresent synthetic recovery supervision as unassisted grasp physics.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
# RoboDojo must precede XPolicyLab: both have a top-level ``utils`` package.
for package_root in (REPO_ROOT / "XPolicyLab", REPO_ROOT):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device-id", type=int, default=0)
parser.add_argument("--seed", type=int, default=0, help="Fixed evaluation-layout seed (0, 1, or 2).")
parser.add_argument("--layout-ids", default="0", help="Comma-separated layout IDs, e.g. 0,1,2.")
parser.add_argument(
    "--output-dir", type=Path, default=Path("recovery_data/make_kong_lerobot_v30_video"),
    help="Root of a new LeRobot v3.0 dataset.",
)
parser.add_argument("--fps", type=int, default=25, help="Video/control recording frequency; must divide the 250 Hz sim.")
parser.add_argument("--max-scenarios", type=int, default=None, help="Debug cap per layout.")
parser.add_argument("--max-plan-frames", type=int, default=80, help="Upper bound after resampling one cuRobo route.")
parser.add_argument("--overwrite", action="store_true", help="Replace --output-dir if it already exists.")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()

app_launcher = AppLauncher(ARGS)
simulation_app = app_launcher.app

import numpy as np
from omegaconf import OmegaConf
import torch

from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
from env.observation_manager.obs_manager import ObsManager
from env.robot_manager.control_manager import MetaControl
from env.seed_manager.seed_manager import SeedManager
from task.RoboDojo import task_registry
from utils.lerobot_v30_writer import CAMERA_FEATURES, LeRobotEpisode, LeRobotV30Writer
from utils.load_file import load_yaml
from utils.make_kong_recovery_data import (
    CANONICAL_PROMPTS,
    RecoveryScenario,
    canonical_prompt,
    enumerate_recovery_scenarios,
)
from utils.pipeline_utils import process_config, process_randomization


# RewardManager identifies a knocked-down tile by this orientation.  It is the
# canonical synthetic post-contact state used by the earlier single-frame
# collector too.
PUSHED_TILE_QUATERNION = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
SOP_TASK = canonical_prompt("sop.default")
RECOVERY_TASKS = [
    canonical_prompt("recovery.push_remaining_matching_tiles"),
    canonical_prompt("recovery.upright_wrong_tile"),
    SOP_TASK,
]


class PlanningError(RuntimeError):
    """A recovery route cannot be collision-free planned from this state."""


@dataclass(frozen=True)
class MotionPhase:
    name: str
    mode: Literal["plan", "ik", "hold"]
    arm: str | None
    target_pose: np.ndarray | None
    gripper: float | None
    frames: int
    task: str
    on_complete: Callable[[], None] | None = None


def _parse_layout_ids(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise ValueError("--layout-ids must contain one or more non-negative integers.")
    return result


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
            "policy_name": "recovery_collector",
            "additional_info": "lerobot_v30_video",
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
    config = process_randomization(config)
    config, _ = process_config(config, task_name=task_name)
    # make_kong's two X5 target arms share one cuRobo planner.  The opponent
    # Franka only replays its released discard trajectory and must not pay a
    # planner warmup cost.
    for robot_cfg in config.robot.robots:
        robot_cfg["need_planner"] = robot_cfg.get("type", "target") == "target"
    config.camera.default_frequency = ARGS.fps
    return config


def _create_collector_env(config):
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


def _reset_collector_layout(env, layout_id: int) -> None:
    env.scene_manager.layout_manager.set_saved_layout(0, env.seed_manager.get_seed_scene_info(layout_id))
    env.reset(seed=[layout_id])
    env.scene_manager.apply_saved_poses(env_idx_list=[0])
    for _ in range(10):
        env.render()
    for _ in range(80):
        env.sim_step(render=False)
    env.obs_manager.reset()
    env.robot_manager.set_origin_endpose()
    env.robot_manager.set_robot_init_state()


def _target_robots(env):
    targets = {robot.arm_name.split("_")[0]: robot for robot in env.robot_manager.robot_list if robot.type == "target"}
    if set(targets) != {"left", "right"}:
        raise RuntimeError(f"Expected left/right X5 target arms, got {sorted(targets)}")
    return targets["left"], targets["right"]


def _normalized_gripper(env, robot) -> float:
    value = float(env.robot_manager.get_end_effector_real_val(robot, env_idx_list=[0])[0][0])
    lower, upper = robot.gripper_scale
    if robot.gripper_move["sign"] == 1:
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
    return float(np.clip((upper - value) / (upper - lower), 0.0, 1.0))


def _state_vector(env) -> np.ndarray:
    left, right = _target_robots(env)
    left_pose = _as_numpy(env.robot_manager.get_real_endpose(left, env_idx_list=[0])[0], dtype=np.float32)
    right_pose = _as_numpy(env.robot_manager.get_real_endpose(right, env_idx_list=[0])[0], dtype=np.float32)
    return np.concatenate([left_pose, [_normalized_gripper(env, left)], right_pose, [_normalized_gripper(env, right)]]).astype(
        np.float32
    )


def _capture_images(env) -> dict[str, np.ndarray]:
    env.render()
    observation = env.obs_manager.get_obs(env_idx_list=[0])[0]
    vision = observation.get("vision", {})
    images = {}
    # The simulator configuration calls the static ego camera ``cam_head``;
    # released RoboDojo LeRobot data exposes the same stream as ``cam_high``.
    # Keep the published key at the dataset boundary.
    source_cameras = {"cam_high": "cam_head", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}
    for camera in CAMERA_FEATURES:
        source_camera = source_cameras[camera]
        color = vision.get(source_camera, {}).get("color")
        if color is None:
            raise RuntimeError(f"Expected {source_camera} image for {camera}, found cameras {sorted(vision)}")
        images[camera] = _as_numpy(color, dtype=np.uint8).copy()
    return images


def _label_object(env, label: str):
    layout = env.scene_manager.layout_manager
    instance_name = layout.get_instance_name(env_idx=0, label=label)
    if instance_name is None:
        raise RuntimeError(f"No scene instance for label {label!r}")
    obj = layout.get_scene_object(env_idx=0, inst_name=instance_name)
    if obj is None:
        raise RuntimeError(f"No scene object for label {label!r}")
    position, orientation = layout.get_instance_pose(env_idx=0, inst_name=instance_name)
    return obj, _as_numpy(position, dtype=np.float32), _as_numpy(orientation, dtype=np.float32)


def _set_label_pose(env, label: str, position: np.ndarray, orientation: np.ndarray) -> None:
    obj, _, _ = _label_object(env, label)
    obj.set_local_pose(translation=np.asarray(position, dtype=np.float32), orientation=np.asarray(orientation, dtype=np.float32))


def _tile_labels() -> tuple[str, ...]:
    return tuple([f"mahjong{group}_{index}" for group in range(4) for index in range(3)] + [f"mahjong{group}_0" for group in range(5, 9)])


def _snapshot_tile_poses(env) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {label: (position.copy(), orientation.copy()) for label in _tile_labels() for _, position, orientation in [_label_object(env, label)]}


def _restore_tile_poses(env, poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    for label, (position, orientation) in poses.items():
        _set_label_pose(env, label, position, orientation)


def _apply_failure_state(env, scenario: RecoveryScenario) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    labels = [f"mahjong{scenario.target_group}_{index}" for index in range(3)] if scenario.failure_kind == "partial_correct_stop" else list(scenario.pushed_correct_labels)
    if scenario.wrong_label is not None:
        labels.append(scenario.wrong_label)
    labels.append(scenario.discard_label)
    original: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    positions: dict[str, np.ndarray] = {}
    for label in dict.fromkeys(labels):
        _, position, orientation = _label_object(env, label)
        original[label] = (position.copy(), orientation.copy())
        positions[label] = position.copy()
    discard_position, _ = original[scenario.discard_label]
    _set_label_pose(env, scenario.discard_label, discard_position, PUSHED_TILE_QUATERNION)
    for label in scenario.pushed_correct_labels:
        position, _ = original[label]
        _set_label_pose(env, label, position, PUSHED_TILE_QUATERNION)
    if scenario.wrong_label is not None:
        position, _ = original[scenario.wrong_label]
        _set_label_pose(env, scenario.wrong_label, position, PUSHED_TILE_QUATERNION)
    for _ in range(8):
        env.sim_step(render=False)
    return original, positions


def _restore_robot_home(env) -> None:
    env.robot_manager.set_robot_init_pose()
    for _ in range(40):
        env.sim_step(render=False)
    env.robot_manager.set_robot_init_state()


def _active_arm(position: np.ndarray) -> str:
    return "left" if float(position[0]) < 0.07 else "right"


def _pose(position: np.ndarray, offset: tuple[float, float, float], orientation: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(position, dtype=np.float32) + np.asarray(offset, dtype=np.float32), orientation]).astype(
        np.float32
    )


def _gripper_target(robot, normalized: float) -> list[float]:
    normalized = float(np.clip(normalized, 0.0, 1.0))
    lower, upper = robot.gripper_scale
    value = lower + normalized * (upper - lower) if robot.gripper_move["sign"] == 1 else upper - normalized * (upper - lower)
    mimic = robot.gripper_move["mimic"]
    return [value, value * mimic[1] + mimic[2]]


def _sim_steps_per_frame(env) -> int:
    steps = 1.0 / (ARGS.fps * env.dt)
    if not float(steps).is_integer():
        raise ValueError(f"--fps={ARGS.fps} is incompatible with sim dt={env.dt}; expected an integer sim-step ratio.")
    return int(steps)


def _apply_joint_target(env, joint_targets: dict[str, np.ndarray], grippers: dict[str, float]) -> None:
    """Apply one complete robot command, holding every uninvolved robot."""

    control: dict[str, dict[str, list[float]]] = {}
    for robot in env.robot_manager.robot_list:
        current_joint = _as_numpy(env.robot_manager.get_joint(robot, env_idx_list=[0])[0], dtype=np.float32)
        target_joint = _as_numpy(joint_targets.get(robot.arm_name, current_joint), dtype=np.float32)
        target_gripper = grippers.get(robot.arm_name, _normalized_gripper(env, robot))
        control[env.robot_manager.process_name(robot.arm_name)] = {
            "position": target_joint.tolist(), "velocity": [0.0] * int(target_joint.size)
        }
        control[env.robot_manager.process_name(robot.gripper_name)] = {
            "position": _gripper_target(robot, target_gripper), "velocity": [0.0, 0.0]
        }
    env.step([MetaControl(control)])
    for _ in range(_sim_steps_per_frame(env)):
        env.sim_step(render=False)


def _record_command(
    env,
    episode: LeRobotEpisode,
    *,
    joint_targets: dict[str, np.ndarray],
    grippers: dict[str, float],
    task: str,
) -> None:
    # The saved label is the next measured end-effector state after one 25 Hz
    # command interval, which is the same state_t/action_t convention used by
    # the released EE LeRobot data.
    state = _state_vector(env)
    images = _capture_images(env)
    _apply_joint_target(env, joint_targets, grippers)
    action = _state_vector(env)
    episode.add_frame(state=state, action=action, images=images, task=task)


def _sample_plan(env, position: np.ndarray) -> list[np.ndarray]:
    position = np.asarray(position, dtype=np.float32)
    if position.ndim != 2 or position.shape[0] == 0:
        raise PlanningError(f"cuRobo returned an invalid joint plan with shape {position.shape}")
    stride = _sim_steps_per_frame(env)
    indices = list(range(0, position.shape[0], stride))
    if indices[-1] != position.shape[0] - 1:
        indices.append(position.shape[0] - 1)
    if len(indices) > ARGS.max_plan_frames:
        indices = np.linspace(0, position.shape[0] - 1, ARGS.max_plan_frames, dtype=int).tolist()
    return [position[index] for index in indices]


def _execute_plan(env, episode: LeRobotEpisode, phase: MotionPhase) -> None:
    if phase.arm is None or phase.target_pose is None:
        raise ValueError(f"Plan phase {phase.name} needs an arm and target pose.")
    robot = _target_robots(env)[0] if phase.arm == "left" else _target_robots(env)[1]
    planner = env.robot_manager.planner.get(robot.robot_name)
    if planner is None:
        raise PlanningError(f"No cuRobo planner is configured for {robot.robot_name}.")
    current_joint = env.robot_manager.get_joint(robot, env_idx_list=[0])[0]
    result = planner.plan_path(
        curr_joint_pos=current_joint,
        target_ee_pose=phase.target_pose.tolist(),
        real_robot_pose=robot.entity_origin_pose,
    )
    if result.get("status") != "Success" or result.get("position") is None:
        raise PlanningError(f"cuRobo failed {phase.name} for {phase.arm}: {result.get('status')}")
    for target_joint in _sample_plan(env, result["position"]):
        _record_command(
            env, episode, joint_targets={robot.arm_name: target_joint},
            grippers={robot.arm_name: 1.0 if phase.gripper is None else phase.gripper}, task=phase.task,
        )


def _execute_ik(env, episode: LeRobotEpisode, phase: MotionPhase) -> None:
    if phase.arm is None or phase.target_pose is None:
        raise ValueError(f"IK phase {phase.name} needs an arm and target pose.")
    robot = _target_robots(env)[0] if phase.arm == "left" else _target_robots(env)[1]
    result = env.robot_manager.solve_ik(target_pose=phase.target_pose.tolist(), env_idx=0, robot=robot)
    if result.get("status") != "Success":
        raise PlanningError(f"IK failed {phase.name} for {phase.arm}: {result.get('status')}")
    for _ in range(phase.frames):
        _record_command(
            env, episode, joint_targets={robot.arm_name: _as_numpy(result["joint_value"], dtype=np.float32)},
            grippers={robot.arm_name: 1.0 if phase.gripper is None else phase.gripper}, task=phase.task,
        )


def _execute_hold(env, episode: LeRobotEpisode, phase: MotionPhase) -> None:
    grippers: dict[str, float] = {}
    if phase.arm is not None and phase.gripper is not None:
        robot = _target_robots(env)[0] if phase.arm == "left" else _target_robots(env)[1]
        grippers[robot.arm_name] = phase.gripper
    for _ in range(phase.frames):
        _record_command(env, episode, joint_targets={}, grippers=grippers, task=phase.task)


def _execute_route(env, episode: LeRobotEpisode, phases: list[MotionPhase]) -> list[str]:
    executed = []
    for phase in phases:
        if phase.mode == "plan":
            _execute_plan(env, episode, phase)
        elif phase.mode == "ik":
            _execute_ik(env, episode, phase)
        elif phase.mode == "hold":
            _execute_hold(env, episode, phase)
        else:
            raise ValueError(f"Unknown phase mode {phase.mode}")
        if phase.on_complete is not None:
            phase.on_complete()
            # Let the recorded following frame show the manipulated tile in a
            # settled state rather than a USD transform mid-update.
            for _ in range(3):
                env.sim_step(render=False)
        executed.append(phase.name)
    return executed


def _push_route(env, scenario: RecoveryScenario, positions: dict[str, np.ndarray], home: np.ndarray) -> list[MotionPhase]:
    task = scenario.prompt
    phases: list[MotionPhase] = []
    for label in [f"mahjong{scenario.target_group}_{index}" for index in range(scenario.pushed_correct_count, 3)]:
        position = positions[label]
        arm = _active_arm(position)
        orientation = home[3:7] if arm == "left" else home[11:15]
        approach = _pose(position, (0.0, -0.085, 0.135), orientation)
        contact = _pose(position, (0.0, -0.032, 0.042), orientation)
        sweep = _pose(position, (0.0, 0.065, 0.042), orientation)
        retreat = _pose(position, (0.0, -0.035, 0.145), orientation)
        phases.extend(
            [
                MotionPhase(f"plan_approach_{label}", "plan", arm, approach, 1.0, 0, task),
                MotionPhase(f"ik_contact_{label}", "ik", arm, contact, 1.0, 8, task),
                MotionPhase(
                    f"ik_push_{label}", "ik", arm, sweep, 1.0, 10, task,
                    on_complete=lambda label=label, position=position: _set_label_pose(env, label, position, PUSHED_TILE_QUATERNION),
                ),
                MotionPhase(f"plan_retreat_{label}", "plan", arm, retreat, 1.0, 0, task),
            ]
        )
    left_home, right_home = home[:7], home[8:15]
    phases.extend(
        [
            MotionPhase("plan_left_home", "plan", "left", left_home, 1.0, 0, SOP_TASK),
            MotionPhase("plan_right_home", "plan", "right", right_home, 1.0, 0, SOP_TASK),
            MotionPhase("sop_hold", "hold", None, None, None, 5, SOP_TASK),
        ]
    )
    return phases


def _upright_route(
    env, scenario: RecoveryScenario, positions: dict[str, np.ndarray], original: dict[str, tuple[np.ndarray, np.ndarray]], home: np.ndarray
) -> list[MotionPhase]:
    if scenario.wrong_label is None:
        raise ValueError("wrong-tile route needs scenario.wrong_label")
    task = scenario.prompt
    label = scenario.wrong_label
    position = positions[label]
    arm = _active_arm(position)
    orientation = home[3:7] if arm == "left" else home[11:15]
    approach = _pose(position, (0.0, -0.080, 0.145), orientation)
    grasp = _pose(position, (0.0, -0.020, 0.052), orientation)
    lift = _pose(position, (0.0, -0.020, 0.165), orientation)
    retreat = _pose(position, (0.0, -0.085, 0.160), orientation)
    original_position, original_orientation = original[label]
    left_home, right_home = home[:7], home[8:15]
    return [
        MotionPhase(f"plan_approach_{label}", "plan", arm, approach, 1.0, 0, task),
        MotionPhase(f"ik_grasp_{label}", "ik", arm, grasp, 1.0, 10, task),
        MotionPhase(
            f"close_and_upright_{label}", "hold", arm, None, 0.0, 8, task,
            on_complete=lambda: _set_label_pose(env, label, original_position, original_orientation),
        ),
        MotionPhase(f"ik_lift_{label}", "ik", arm, lift, 0.0, 10, task),
        MotionPhase(f"plan_retreat_{label}", "plan", arm, retreat, 0.0, 0, task),
        MotionPhase(f"release_{label}", "hold", arm, None, 1.0, 6, task),
        MotionPhase("plan_left_home", "plan", "left", left_home, 1.0, 0, SOP_TASK),
        MotionPhase("plan_right_home", "plan", "right", right_home, 1.0, 0, SOP_TASK),
        MotionPhase("sop_hold", "hold", None, None, None, 5, SOP_TASK),
    ]


def _collect_one_episode(
    env, writer: LeRobotV30Writer, *, episode_index: int, scenario: RecoveryScenario, seed: int, layout_id: int
) -> tuple[int, list[str]]:
    original, positions = _apply_failure_state(env, scenario)
    home = _state_vector(env)
    source = {
        "seed": seed,
        "layout_id": layout_id,
        "scenario": scenario.to_dict(),
        "recovery_execution": "curobo_planned_free_space + IK_contact + explicit_tile_pose_at_contact",
    }
    episode = writer.start_episode(episode_index, source=source)
    try:
        route = _push_route(env, scenario, positions, home) if scenario.failure_kind == "partial_correct_stop" else _upright_route(env, scenario, positions, original, home)
        phases = _execute_route(env, episode, route)
        writer.commit_episode(episode)
        return episode.length, phases
    except Exception:
        writer.abort_episode(episode)
        raise


def main() -> None:
    if ARGS.fps <= 0 or ARGS.max_plan_frames <= 0:
        raise ValueError("--fps and --max-plan-frames must be positive.")
    layout_ids = _parse_layout_ids(ARGS.layout_ids)
    scenarios = enumerate_recovery_scenarios()
    if ARGS.max_scenarios is not None:
        if ARGS.max_scenarios <= 0:
            raise ValueError("--max-scenarios must be positive.")
        scenarios = scenarios[: ARGS.max_scenarios]
    print(
        f"[make_kong_lerobot] seed={ARGS.seed} layouts={layout_ids} scenarios_per_layout={len(scenarios)} output={ARGS.output_dir}",
        flush=True,
    )
    writer = LeRobotV30Writer(ARGS.output_dir, fps=ARGS.fps, tasks=RECOVERY_TASKS, overwrite=ARGS.overwrite)
    env = None
    skipped: list[dict[str, Any]] = []
    try:
        config = _build_env_config(ARGS.device_id, ARGS.seed)
        print(f"[make_kong_lerobot] planner flags={[(cfg.get('robot_name'), cfg.get('need_planner')) for cfg in config.robot.robots]}", flush=True)
        env = _create_collector_env(config)
        _sim_steps_per_frame(env)  # validates exact 25 Hz capture cadence
        episode_index = 0
        for layout_id in layout_ids:
            _reset_collector_layout(env, layout_id)
            base_poses = _snapshot_tile_poses(env)
            for scenario in scenarios:
                _restore_tile_poses(env, base_poses)
                _restore_robot_home(env)
                try:
                    length, phases = _collect_one_episode(
                        env, writer, episode_index=episode_index, scenario=scenario, seed=ARGS.seed, layout_id=layout_id
                    )
                except PlanningError as exc:
                    skipped.append({"seed": ARGS.seed, "layout_id": layout_id, "scenario": scenario.to_dict(), "reason": str(exc)})
                    print(f"[make_kong_lerobot] SKIP layout={layout_id} scenario={scenario.scenario_id}: {exc}", flush=True)
                    continue
                print(
                    f"[make_kong_lerobot] episode={episode_index} layout={layout_id} scenario={scenario.scenario_id} frames={length} phases={len(phases)}",
                    flush=True,
                )
                episode_index += 1
        if episode_index == 0:
            raise RuntimeError("No episode completed; refusing to write an empty LeRobot dataset.")
        writer.finalize(
            extra_manifest={
                "collector": "generate_make_kong_recovery_lerobot.py",
                "seed": ARGS.seed,
                "layout_ids": layout_ids,
                "fps": ARGS.fps,
                "skipped": skipped,
            }
        )
        print(f"[make_kong_lerobot] wrote {episode_index} video episodes / {sum(row['length'] for row in writer.episodes)} frames to {ARGS.output_dir}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
