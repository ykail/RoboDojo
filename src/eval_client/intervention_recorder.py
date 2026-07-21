"""HDF5 trajectory writer for policy/human intervention episodes."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import cv2
import h5py
import numpy as np


def _safe_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "unknown"


def _git_revision(path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _trajectory_key(observation_key: str) -> str:
    if observation_key.endswith("_joint_state"):
        return observation_key + "s"
    if observation_key.endswith("_pose"):
        return observation_key + "s"
    return observation_key


def _copy_action(action: dict | None) -> dict[str, np.ndarray] | None:
    if action is None:
        return None
    return {key: np.asarray(value, dtype=np.float32).copy() for key, value in action.items()}


class EpisodeRecorder:
    """Buffer one episode as JPEG/action records and atomically write HDF5."""

    def __init__(
        self,
        record_dir: str,
        metadata: dict[str, Any],
        frequency: int = 25,
        jpeg_quality: int = 90,
    ):
        self.record_dir = Path(record_dir).expanduser().resolve()
        self.metadata = dict(metadata)
        self.frequency = int(frequency)
        self.jpeg_quality = int(jpeg_quality)
        self._frames: list[dict[str, Any]] = []
        self._instruction: str | None = None
        self._started_at = datetime.now()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def append(
        self,
        obs: dict,
        policy_action: dict | None,
        human_action: dict | None,
        executed_action: dict,
        control: dict[str, Any],
    ) -> None:
        if self._instruction is None:
            self._instruction = str(obs.get("instruction", ""))
        encoded_vision = {}
        for camera_name, camera_data in obs.get("vision", {}).items():
            color = camera_data.get("color") if isinstance(camera_data, dict) else None
            if color is None:
                continue
            image = np.asarray(color)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected RGB HxWx3 for {camera_name}, got {image.shape}")
            image = np.ascontiguousarray(image.astype(np.uint8, copy=False))
            ok, encoded = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                raise ValueError(f"JPEG encoding failed for camera {camera_name}")
            encoded_vision[camera_name] = encoded.tobytes()

        self._frames.append(
            {
                "state": {key: np.asarray(value, dtype=np.float32).copy() for key, value in obs["state"].items()},
                "vision": encoded_vision,
                "policy_action": _copy_action(policy_action),
                "human_action": _copy_action(human_action),
                "executed_action": _copy_action(executed_action),
                "control": dict(control),
            }
        )

    def _write_numeric_dict(self, parent: h5py.Group, name: str, records: list[dict | None]) -> None:
        group = parent.create_group(name)
        keys = sorted({key for record in records if record is not None for key in record})
        for key in keys:
            exemplar = next(np.asarray(record[key]) for record in records if record is not None and key in record)
            values = []
            for record in records:
                if record is None or key not in record:
                    values.append(np.full(exemplar.shape, np.nan, dtype=np.float32))
                else:
                    values.append(np.asarray(record[key], dtype=np.float32))
            array = np.stack(values)
            if key.endswith("_ee_joint_state") and array.ndim == 2 and array.shape[1] == 1:
                array = array[:, 0]
            group.create_dataset(_trajectory_key(key), data=array, compression="lzf")

    def _write_images(self, root: h5py.File) -> None:
        vision_group = root.create_group("vision")
        camera_names = sorted({name for frame in self._frames for name in frame["vision"]})
        for camera_name in camera_names:
            payloads = []
            for frame in self._frames:
                if camera_name not in frame["vision"]:
                    raise ValueError(f"Camera {camera_name} is missing from one or more trajectory frames")
                payloads.append(frame["vision"][camera_name])
            max_length = max(len(payload) for payload in payloads)
            padded = np.asarray([payload.ljust(max_length, b"\0") for payload in payloads], dtype=f"S{max_length}")
            camera_group = vision_group.create_group(camera_name)
            camera_group.create_dataset("colors", data=padded, compression="lzf")

    def finalize(self, accepted: bool, success: bool, reason: str) -> str | None:
        if not accepted:
            self._frames.clear()
            return None
        if not self._frames:
            return None

        task_name = _safe_component(self.metadata.get("task_name", "task"))
        env_config = _safe_component(self.metadata.get("env_config", "unknown_env"))
        layout_id = _safe_component(self.metadata.get("layout_id", "unknown"))
        timestamp = self._started_at.strftime("%Y%m%d_%H%M%S_%f")
        output_dir = self.record_dir / task_name / env_config / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"episode_{timestamp}_layout_{layout_id}.hdf5"
        partial_path = Path(str(final_path) + ".partial")

        try:
            with h5py.File(partial_path, "w") as root:
                root.create_dataset("data_format_version", data=np.bytes_("v1.0"))
                root.create_dataset("instructions", data=np.bytes_(json.dumps([self._instruction or ""])))
                additional_info = root.create_group("additional_info")
                additional_info.create_dataset("frequency", data=self.frequency)

                states = [frame["state"] for frame in self._frames]
                policy_actions = [frame["policy_action"] for frame in self._frames]
                human_actions = [frame["human_action"] for frame in self._frames]
                executed_actions = [frame["executed_action"] for frame in self._frames]
                self._write_numeric_dict(root, "state", states)
                self._write_numeric_dict(root, "action", executed_actions)
                self._write_numeric_dict(root, "executed_action", executed_actions)
                self._write_numeric_dict(root, "policy_action", policy_actions)
                self._write_numeric_dict(root, "human_action", human_actions)
                self._write_images(root)

                control_group = root.create_group("control")
                controls = [frame["control"] for frame in self._frames]
                for key in ("intervention_mask", "ik_success", "takeover_edge", "chunk_id", "chunk_index"):
                    control_group.create_dataset(key, data=np.asarray([item.get(key, 0) for item in controls]))
                for key in ("action_source", "active_arm"):
                    values = [str(item.get(key, "")) for item in controls]
                    width = max(1, max(len(value.encode("utf-8")) for value in values))
                    control_group.create_dataset(key, data=np.asarray(values, dtype=f"S{width}"))
                root.create_dataset(
                    "timestamps",
                    data=np.asarray([item.get("timestamp", np.nan) for item in controls], dtype=np.float64),
                )

                metadata = dict(self.metadata)
                metadata.update(
                    {
                        "accepted": bool(accepted),
                        "success": bool(success),
                        "finish_reason": reason,
                        "frame_count": self.frame_count,
                        "has_intervention": any(item.get("intervention_mask", 0) for item in controls),
                        "complete": True,
                    }
                )
                for key, value in metadata.items():
                    if isinstance(value, (str, bytes, int, float, bool, np.number)):
                        root.attrs[key] = value
                    else:
                        root.attrs[key] = json.dumps(value, default=str)
                root.flush()

            os.replace(partial_path, final_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise
        self._frames.clear()
        return str(final_path)


def recorder_for_env(task_env, record_dir: str) -> EpisodeRecorder:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    metadata = {
        "task_name": task_env.task_name,
        "env_config": task_env.config_name,
        "layout_id": int(task_env.env_seeds[0]),
        "eval_seed": int(task_env.eval_seed),
        "policy_name": task_env.policy_name,
        "base_checkpoint": task_env.additional_info,
        "robodojo_commit": _git_revision(repo_root),
        "xpolicylab_commit": _git_revision(os.path.join(repo_root, "XPolicyLab")),
        "control_mode": "keyboard_intervention",
    }
    return EpisodeRecorder(
        record_dir=record_dir,
        metadata=metadata,
        frequency=int(task_env.obs_manager.collect_freq),
    )
