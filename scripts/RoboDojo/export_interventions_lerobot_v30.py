"""Export corrective RoboDojo HDF5 episodes as a Kai0-compatible LeRobot v3 dataset.

This intentionally runs outside the Isaac Sim process.  The interactive
collector keeps atomic HDF5 as its source of truth; this exporter rebuilds a
training dataset from those accepted files after collection completes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from XPolicyLab.scripts import transform_lerobot_v30_format as base


@dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 8
    image_writer_threads: int = 2
    streaming_encoding: bool = True
    video_crf: int | None = 18
    video_backend: str | None = "pyav"
    vcodec: str = "h264"
    encoder_threads: int | None = 2


DEFAULT_DATASET_CONFIG = DatasetConfig()


def extract_intervention_mask(data: dict, horizon: int) -> np.ndarray:
    """Load and validate the per-frame human intervention flag."""
    value = base._get_nested(data, "control", "intervention_mask")
    if value is None:
        return np.zeros(horizon, dtype=np.float32)

    mask = np.asarray(value, dtype=np.float32).reshape(-1)
    if mask.shape[0] != horizon:
        raise ValueError(
            "control.intervention_mask horizon mismatch: "
            f"expected {horizon}, got {mask.shape[0]}"
        )
    if not np.isfinite(mask).all() or not np.isin(mask, (0.0, 1.0)).all():
        raise ValueError("control.intervention_mask must contain only finite 0/1 values")
    return mask


def create_empty_dataset(
    *,
    repo_id: str,
    robot_type: str,
    motors: list[str],
    fps: int,
    root: str | Path,
    mode: str = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    overwrite: bool = False,
) -> Any:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "complementary_info.is_intervention": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["is_intervention"],
        },
    }
    for camera_name in base.CAMERA_CANDIDATES:
        features[f"observation.images.{camera_name}"] = {
            "dtype": mode,
            "shape": (3, base.TARGET_IMAGE_HEIGHT, base.TARGET_IMAGE_WIDTH),
            "names": ["channels", "height", "width"],
        }

    dataset_base = Path(root).expanduser().resolve()
    dataset_root = (dataset_base / repo_id).resolve()
    if dataset_root == dataset_base or not dataset_root.is_relative_to(dataset_base):
        raise ValueError(f"repo_id must resolve below {dataset_base}, got {repo_id!r}")
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"LeRobot dataset already exists: {dataset_root}. "
                "Pass --overwrite to rebuild it from the source HDF5 files."
            )
        shutil.rmtree(dataset_root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        streaming_encoding=dataset_config.streaming_encoding,
        video_backend=dataset_config.video_backend,
        vcodec=dataset_config.vcodec,
        encoder_threads=dataset_config.encoder_threads,
    )


def configure_video_encoding(dataset: Any, dataset_config: DatasetConfig) -> None:
    if not dataset_config.use_videos:
        return
    encoder = getattr(dataset, "_streaming_encoder", None)
    if encoder is None:
        raise RuntimeError("LeRobot streaming encoder was not initialized")
    encoder.crf = dataset_config.video_crf


def finalize_dataset(dataset: Any) -> None:
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()
    dataset.finalize()


def convert_one(
    input_path: str | Path,
    dataset: Any,
    *,
    data_type: str,
    data_version: str,
    current_dims: list[int],
    target_dims: list[int],
) -> None:
    data = base.load(str(input_path), data_type=data_type, data_version=data_version)
    state = base._extract_qpos(data)
    action = base._extract_action(data)
    if state.shape != action.shape:
        raise ValueError(f"state/action mismatch: {state.shape} vs {action.shape}")

    intervention_mask = extract_intervention_mask(data, state.shape[0])
    state = base._pad_state_to_target_dims(state, current_dims, target_dims, "state")
    action = base._pad_state_to_target_dims(action, current_dims, target_dims, "action")
    instruction = base._choose_instruction(data)
    if not instruction:
        raise ValueError("No instruction found")

    images = {}
    for camera_name in base.CAMERA_CANDIDATES:
        image_array = base._find_camera_array(data, camera_name)
        if image_array is None:
            raise ValueError(f"Missing required camera: {camera_name}")
        if len(image_array) != state.shape[0]:
            raise ValueError(
                f"Camera {camera_name} horizon mismatch: "
                f"expected {state.shape[0]}, got {len(image_array)}"
            )
        images[camera_name] = image_array

    for index in range(state.shape[0]):
        frame = {
            "observation.state": state[index],
            "action": action[index],
            "complementary_info.is_intervention": np.asarray(
                [intervention_mask[index]], dtype=np.float32
            ),
            "task": instruction,
        }
        for camera_name, image_array in images.items():
            frame[f"observation.images.{camera_name}"] = image_array[index]
        dataset.add_frame(frame)
    dataset.save_episode()


def _clear_failed_episode(dataset: Any) -> None:
    clear = getattr(dataset, "clear_episode_buffer", None)
    if callable(clear):
        clear(delete_images=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RoboDojo intervention HDF5 files to Kai0-compatible LeRobot v3.0."
    )
    parser.add_argument("patterns", nargs="+", help='Patterns like "RoboDojo_interventions.stack_bowls.arx_x5"')
    parser.add_argument("--repo-id", "--repo_id", dest="repo_id", required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/piper/data/lerobot"))
    parser.add_argument("--data-type", "--data_type", dest="data_type", default=base.DEFAULT_DATASET_NAME)
    parser.add_argument("--data-version", "--data_version", dest="data_version", default="v1.0")
    parser.add_argument("--max-episodes", "--max_episode", dest="max_episodes", type=int, default=1_000_000)
    parser.add_argument(
        "--vcodec",
        choices=("h264", "hevc", "libsvtav1", "h264_nvenc", "hevc_nvenc", "auto"),
        default=DEFAULT_DATASET_CONFIG.vcodec,
    )
    parser.add_argument("--encoder-threads", type=int, default=DEFAULT_DATASET_CONFIG.encoder_threads)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    if args.encoder_threads is not None and args.encoder_threads <= 0:
        raise ValueError("--encoder-threads must be positive")

    targets = base._discover_conversion_targets(args.patterns)
    if not targets:
        raise FileNotFoundError(f"No matching targets found: {args.patterns}")
    metadata_by_target, target_dims, max_fps = base._plan_target_metadata(targets)
    target_inputs = base._collect_target_input_files(targets)
    base._print_matched_targets(target_inputs)
    if not any(input_files for *_, input_files in target_inputs):
        raise FileNotFoundError("No HDF5 episodes matched the requested targets")

    config = DatasetConfig(vcodec=args.vcodec, encoder_threads=args.encoder_threads)
    dataset = create_empty_dataset(
        repo_id=args.repo_id,
        robot_type="unified_robot",
        motors=base._build_motor_names_from_dims(target_dims),
        fps=max_fps or 50,
        root=args.root,
        dataset_config=config,
        overwrite=args.overwrite,
    )
    configure_video_encoding(dataset, config)

    failures = []
    converted = 0
    try:
        for bench_name, task_name, env_cfg_type, _input_dir, input_files in target_inputs:
            current_dims = metadata_by_target[(bench_name, task_name, env_cfg_type)]["per_arm_dims"]
            for input_path in tqdm(
                input_files[: args.max_episodes],
                desc=f"Exporting {bench_name}/{task_name}/{env_cfg_type}",
            ):
                try:
                    convert_one(
                        input_path,
                        dataset,
                        data_type=args.data_type,
                        data_version=args.data_version,
                        current_dims=current_dims,
                        target_dims=target_dims,
                    )
                    converted += 1
                except Exception as exc:
                    _clear_failed_episode(dataset)
                    failures.append((str(input_path), str(exc)))
    finally:
        finalize_dataset(dataset)

    print(f"Exported {converted} episode(s) to {dataset.root}")
    if failures:
        details = "\n".join(f"  - {path}: {reason}" for path, reason in failures)
        raise RuntimeError(f"Failed to export {len(failures)} episode(s):\n{details}")


if __name__ == "__main__":
    main()
