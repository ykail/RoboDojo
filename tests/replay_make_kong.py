"""Replay generated make_kong demonstrations through the evaluation environment."""

import argparse
from pathlib import Path
import subprocess
import sys
import traceback

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "XPolicyLab")]

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "output" / "make_kong_expert_v2")
PARSER.add_argument("--layout-root", type=Path, default=REPO_ROOT / "data_gen" / "make_kong" / "layouts")
PARSER.add_argument("--episode-indices", type=str, default="0", help="Comma-separated dataset episode indices.")
PARSER.add_argument("--modes", choices=("both", "controller", "state"), default="both")
PARSER.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tmp" / "make_kong_replay_smoke")
PARSER.add_argument("--device-id", type=int, default=0)
PARSER.add_argument("--eval-seed", type=int, choices=(0, 1, 2), default=0)
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
ARGS.enable_cameras = True

APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

from data_gen.make_kong.env_builder import FIXED_SEED, build_config, build_task_class  # noqa: E402
from data_gen.make_kong.layout_pool import SavedLayout, load_layout_pool  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from src.eval_client.eval_env import create_eval_env  # noqa: E402
import torch  # noqa: E402
from utils.save_file import save_json  # noqa: E402


def _parse_episode_indices(value: str) -> list[int]:
    indices = [int(part) for part in value.split(",") if part]
    if not indices:
        raise ValueError("--episode-indices must select at least one episode.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"--episode-indices contains duplicates: {indices}.")
    if min(indices) < 0:
        raise ValueError(f"--episode-indices must be non-negative, got {indices}.")
    return indices


def _load_dataset(dataset_root: Path):
    data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet trajectory files found under {dataset_root / 'data'}.")
    data = pq.read_table(data_files)
    manifest = pq.read_table(dataset_root / "meta" / "generation_manifest.parquet").to_pydict()
    episodes = pq.read_table(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pydict()
    manifest_by_episode = {
        int(episode_index): {
            "layout_id": int(manifest["layout"][index]),
            "target_group": int(manifest["target_group"][index]),
        }
        for index, episode_index in enumerate(manifest["episode_index"])
        if manifest["status"][index] == "success" and episode_index is not None
    }
    episode_rows = {int(value): index for index, value in enumerate(episodes["episode_index"])}
    return data, manifest_by_episode, episodes, episode_rows


def _episode_actions(data, episode_index: int) -> np.ndarray:
    frame_indices = np.asarray(data["episode_index"].to_numpy()) == episode_index
    actions = np.asarray(data["action"].filter(pa.array(frame_indices)).to_pylist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) == 0:
        raise ValueError(f"Episode {episode_index} has invalid action shape {actions.shape}.")
    return actions


def _action_dict(action: np.ndarray) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (14,):
        raise ValueError(f"Expected one 14-D recorded action, got {action.shape}.")
    return {
        "left_arm_joint_state": action[:6],
        "left_ee_joint_state": action[6:7],
        "right_arm_joint_state": action[7:13],
        "right_ee_joint_state": action[13:14],
    }


def _state_vector(env) -> np.ndarray:
    values = []
    for arm_name in ("left_arm", "right_arm"):
        robot = env.robot_manager.get_robot_by_arm_name(arm_name)
        joints = env.robot_manager.get_joint(robot, env_idx_list=[0])[0]
        gripper = env.robot_manager.get_end_effector_real_val(robot, env_idx_list=[0])[0]
        opening = float(np.mean(gripper))
        lower, upper = robot.gripper_scale
        if robot.gripper_move["sign"] == 1:
            opening = (opening - lower) / (upper - lower)
        else:
            opening = (upper - opening) / (upper - lower)
        values.extend(np.asarray(joints, dtype=np.float32).tolist())
        values.append(float(np.clip(opening, 0.0, 1.0)))
    return np.asarray(values, dtype=np.float32)


def _write_target_state(env, action: np.ndarray) -> None:
    for arm_name, joint_values, gripper_opening in (
        ("left_arm", action[:6], float(action[6])),
        ("right_arm", action[7:13], float(action[13])),
    ):
        robot = env.robot_manager.get_robot_by_arm_name(arm_name)
        articulation = env.robot_manager.robot_key[env.robot_manager.robot_list.index(robot)]
        env_ids = torch.tensor([0], dtype=torch.int32, device=articulation.device)
        arm_values = torch.tensor(joint_values, dtype=torch.float32, device=articulation.device).reshape(1, -1)
        arm_velocity = torch.zeros_like(arm_values)
        lower, upper = robot.gripper_scale
        if robot.gripper_move["sign"] == 1:
            opening = gripper_opening * (upper - lower) + lower
        else:
            opening = (1 - gripper_opening) * (upper - lower) + lower
        mimic_multiplier, mimic_offset = robot.gripper_move["mimic"][1:]
        gripper_values = torch.tensor(
            [[opening, opening * mimic_multiplier + mimic_offset]], dtype=torch.float32, device=articulation.device
        )
        articulation.write_joint_state_to_sim(arm_values, arm_velocity, robot.arm_joint_indices, env_ids)
        articulation.write_joint_state_to_sim(
            gripper_values, torch.zeros_like(gripper_values), robot.gripper_joint_indices, env_ids
        )
        articulation.set_joint_position_target(arm_values, robot.arm_joint_indices, env_ids)
        articulation.set_joint_position_target(gripper_values, robot.gripper_joint_indices, env_ids)


def _reference_videos(dataset_root: Path, episodes: dict, row: int) -> dict[str, str]:
    videos = {}
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        feature = f"observation.images.{camera}"
        chunk = int(episodes[f"videos/{feature}/chunk_index"][row])
        file_index = int(episodes[f"videos/{feature}/file_index"][row])
        videos[camera] = str(dataset_root / "videos" / feature / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4")
    return videos


def _write_comparisons(reference_videos: dict[str, str], replay_dir: Path, episode_index: int, tag: str) -> dict[str, str]:
    comparisons = {}
    for reference_camera, reference_path in reference_videos.items():
        replay_camera = "cam_head" if reference_camera == "cam_high" else reference_camera
        replay_path = replay_dir / f"episode_{episode_index:07d}_{replay_camera}_{tag}.mp4"
        if not Path(reference_path).is_file() or not replay_path.is_file():
            continue
        comparison_path = replay_dir / f"episode_{episode_index:07d}_{reference_camera}_comparison.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                reference_path,
                "-i",
                str(replay_path),
                "-filter_complex",
                "[0:v][1:v]hstack=inputs=2",
                "-shortest",
                "-pix_fmt",
                "yuv420p",
                str(comparison_path),
            ],
            check=True,
        )
        comparisons[reference_camera] = str(comparison_path)
    return comparisons


def _run_episode(
    *,
    mode: str,
    output_dir: Path,
    saved_layout: SavedLayout,
    target_group: int,
    episode_index: int,
    actions: np.ndarray,
    reference_videos: dict[str, str],
) -> dict:
    config = build_config(num_envs=1, device_id=ARGS.device_id)
    config.eval_cfg.update(
        {
            "offline_replay": True,
            "policy_name": "make_kong_replay",
            "config_name": "arx_x5",
            "additional_info": mode,
            "replay_output_dir": str(output_dir),
            "seed": ARGS.eval_seed,
        }
    )
    config.deploy_cfg = {}
    task_class = build_task_class()
    env = create_eval_env(config, SIMULATION_APP, task_class_override=task_class)
    env.forced_target_groups = [target_group]
    env.step_lim = len(actions)
    try:
        env.reset_from_saved_layouts([saved_layout.scene_layout], seed=[ARGS.eval_seed])
        env.run_reward()
        env.query_support_arm_traj(env_idx=0)
        env.get_obs()
        errors = []
        original_is_episode_end = env.is_episode_end
        env.is_episode_end = lambda: False
        try:
            for action_index, action in enumerate(actions):
                if mode == "state":
                    _write_target_state(env, action)
                env.take_action(_action_dict(action))
                if action_index + 1 < len(actions):
                    env.get_obs()
                difference = _state_vector(env) - action
                errors.append(
                    {
                        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                        "max_abs": float(np.max(np.abs(difference))),
                    }
                )
        finally:
            env.is_episode_end = original_is_episode_end

        final_reward = float(env.reward_manager.get_reward(final_check=True)[0])
        success = final_reward > 1 - 1e-3
        env.success[0] = success
        env.end_flag[0] = True
        tag = "success" if success else "failure"
        env.save_video(0, str(output_dir / f"episode_{episode_index:07d}.mp4"), tag)
        comparisons = _write_comparisons(reference_videos, output_dir, episode_index, tag)
        summary = {
            "episode_index": episode_index,
            "layout_id": saved_layout.layout_id,
            "target_group": target_group,
            "mode": mode,
            "recorded_action_count": len(actions),
            "final_reward": final_reward,
            "success": success,
            "termination_reason": "recorded_actions_exhausted",
            "state_error": {
                "mean_rmse": float(np.mean([item["rmse"] for item in errors])),
                "max_abs": float(np.max([item["max_abs"] for item in errors])),
            },
            "reference_videos": reference_videos,
            "comparison_videos": comparisons,
        }
        return summary
    finally:
        env.close()


def _result(summaries: list[dict]) -> dict:
    success_count = sum(summary["success"] for summary in summaries)
    total = len(summaries)
    return {
        "success_rate": success_count / total if total else 0.0,
        "eval_time": total,
        "score": success_count / total * 100 if total else 0.0,
        "details": {
            index: {
                "layout_id": summary["layout_id"],
                "success": summary["success"],
                "score": float(summary["success"]),
                "final_reward": summary["final_reward"],
                "termination_reason": summary["termination_reason"],
            }
            for index, summary in enumerate(summaries)
        },
    }


def main() -> int:
    dataset_root = ARGS.dataset_root.resolve()
    layout_pool = load_layout_pool(ARGS.layout_root.resolve())
    data, manifest_by_episode, episodes, episode_rows = _load_dataset(dataset_root)
    episode_indices = _parse_episode_indices(ARGS.episode_indices)
    missing = sorted(set(episode_indices) - set(manifest_by_episode))
    if missing:
        raise ValueError(f"Episodes are not successful dataset entries: {missing}.")
    modes = ("controller", "state") if ARGS.modes == "both" else (ARGS.modes,)
    try:
        for mode in modes:
            mode_dir = ARGS.output_dir.resolve() / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            summaries = []
            for episode_index in episode_indices:
                metadata = manifest_by_episode[episode_index]
                layout_id = metadata["layout_id"]
                if layout_id not in layout_pool:
                    raise ValueError(f"Episode {episode_index} references unavailable layout {layout_id}.")
                if episode_index not in episode_rows:
                    raise ValueError(f"Episode metadata is missing index {episode_index}.")
                try:
                    summary = _run_episode(
                        mode=mode,
                        output_dir=mode_dir,
                        saved_layout=layout_pool[layout_id],
                        target_group=metadata["target_group"],
                        episode_index=episode_index,
                        actions=_episode_actions(data, episode_index),
                        reference_videos=_reference_videos(dataset_root, episodes, episode_rows[episode_index]),
                    )
                except Exception as error:
                    save_json(
                        {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()},
                        mode_dir / "replay_error.json",
                    )
                    raise
                summaries.append(summary)
            save_json(_result(summaries), mode_dir / "_result.json")
            save_json({"episodes": summaries}, mode_dir / "replay_diagnostics.json")
    finally:
        SIMULATION_APP.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
