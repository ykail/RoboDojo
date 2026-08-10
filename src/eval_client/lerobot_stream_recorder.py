"""Isaac-side proxy for direct LeRobot v3 intervention recording.

This module intentionally imports neither LeRobot, OpenCV, Torch nor HDF5.  A
CPU-only subprocess in the Pi_05 environment owns all dataset/video state.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from .lerobot_stream_protocol import receive_message, send_message


class LeRobotStreamError(RuntimeError):
    """Direct LeRobot streaming failed and the current episode was not saved."""


class LeRobotStreamStartupError(LeRobotStreamError):
    """Non-retryable dataset/configuration or integrity failure."""


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be 0/1 or true/false, got {value!r}")


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty(path: Path) -> bool | str:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output)
    except Exception:
        return "unknown"


def _resolve_dataset_root(base_root: Path, repo_id: str) -> Path:
    base = base_root.expanduser().resolve()
    dataset_root = (base / repo_id).resolve()
    try:
        dataset_root.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"ROBODOJO_LEROBOT_REPO_ID escapes {base}: {repo_id!r}") from exc
    if dataset_root == base:
        raise ValueError("ROBODOJO_LEROBOT_REPO_ID must name a child dataset")
    return dataset_root


@dataclass(frozen=True)
class StreamConfig:
    python: Path
    project_root: Path
    repo_id: str
    root: Path
    fps: int
    resume: bool
    vcodec: str
    encoder_threads: int
    encoder_queue_maxsize: int
    video_crf: int
    run_id: str = ""
    startup_timeout_s: float = 300.0
    response_timeout_s: float = 180.0

    @property
    def dataset_root(self) -> Path:
        return _resolve_dataset_root(self.root, self.repo_id)

    @property
    def identity(self) -> tuple[Any, ...]:
        return (
            self.python.resolve(),
            self.project_root.resolve(),
            self.repo_id,
            self.root.resolve(),
            self.fps,
            self.vcodec,
            self.encoder_threads,
            self.encoder_queue_maxsize,
            self.video_crf,
            self.run_id,
            self.startup_timeout_s,
            self.response_timeout_s,
        )


_COLLECTION_SESSION_MARKER = ".robodojo_collection_session.json"


def _collection_marker_run_id(dataset_root: Path) -> str | None:
    marker = dataset_root / _COLLECTION_SESSION_MARKER
    try:
        with marker.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    return str(run_id) if run_id else None


def _write_collection_marker(config: StreamConfig) -> None:
    if not config.run_id:
        return
    marker = config.dataset_root / _COLLECTION_SESSION_MARKER
    temporary = marker.with_name(f"{marker.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps({"run_id": config.run_id, "pid": os.getpid()}),
        encoding="utf-8",
    )
    os.replace(temporary, marker)


def _remove_collection_marker(dataset_root: Path, run_id: str) -> None:
    marker = dataset_root / _COLLECTION_SESSION_MARKER
    if run_id and _collection_marker_run_id(dataset_root) == run_id:
        marker.unlink(missing_ok=True)


def config_from_environment(*, fps: int) -> StreamConfig:
    project_root = Path(__file__).resolve().parents[2]
    default_python = project_root / "XPolicyLab" / "policy" / "Pi_05" / "openpi" / ".venv" / "bin" / "python"
    python = Path(os.environ.get("ROBODOJO_LEROBOT_PYTHON", str(default_python))).expanduser()
    if not python.is_absolute():
        python = Path.cwd() / python
    # Do not call Path.resolve() here.  uv/venv commonly makes ``bin/python``
    # a symlink to the system interpreter; executing the lexical venv path is
    # what makes CPython discover pyvenv.cfg and the venv's site-packages.
    python = Path(os.path.abspath(python))
    root_value = os.environ.get("ROBODOJO_LEROBOT_ROOT")
    repo_id = os.environ.get("ROBODOJO_LEROBOT_REPO_ID", "").strip()
    if not root_value:
        raise ValueError("ROBODOJO_LEROBOT_ROOT is required for direct LeRobot collection")
    if not repo_id:
        raise ValueError("ROBODOJO_LEROBOT_REPO_ID is required for direct LeRobot collection")
    if not _env_bool("ROBODOJO_LEROBOT_STREAMING_ENCODING", True):
        raise ValueError("Direct collection requires ROBODOJO_LEROBOT_STREAMING_ENCODING=1")
    if not python.is_file():
        raise FileNotFoundError(
            f"LeRobot Python was not found: {python}. Run the Pi_05 install script first."
        )
    config = StreamConfig(
        python=python,
        project_root=project_root,
        repo_id=repo_id,
        root=Path(root_value).expanduser().resolve(),
        fps=int(fps),
        resume=_env_bool("ROBODOJO_LEROBOT_RESUME", False),
        vcodec=os.environ.get("ROBODOJO_LEROBOT_VCODEC", "h264"),
        encoder_threads=_positive_int("ROBODOJO_LEROBOT_ENCODER_THREADS", 2),
        encoder_queue_maxsize=_positive_int("ROBODOJO_LEROBOT_ENCODER_QUEUE_MAXSIZE", 128),
        video_crf=int(os.environ.get("ROBODOJO_LEROBOT_VIDEO_CRF", 18)),
        run_id=os.environ.get("ROBODOJO_RUN_ID", "").strip(),
        startup_timeout_s=_positive_float("ROBODOJO_LEROBOT_STARTUP_TIMEOUT_S", 300.0),
        response_timeout_s=_positive_float("ROBODOJO_LEROBOT_RESPONSE_TIMEOUT_S", 180.0),
    )
    if config.fps <= 0:
        raise ValueError(f"LeRobot FPS must be positive, got {config.fps}")
    # Resolve now so a path traversal fails before a subprocess is started.
    config.dataset_root
    if (
        not config.resume
        and config.run_id
        and _collection_marker_run_id(config.dataset_root) == config.run_id
    ):
        # Isaac may re-exec after a PhysX failure.  The run id proves this is
        # the same collection command, so append without requiring --resume.
        config = replace(config, resume=True)
    return config


class _WriterSidecar:
    def __init__(self, config: StreamConfig):
        self.config = config
        script = config.project_root / "scripts" / "RoboDojo" / "lerobot_stream_writer.py"
        if not script.is_file():
            raise FileNotFoundError(f"LeRobot writer script is missing: {script}")
        command = [
            str(config.python),
            str(script),
            "--repo-id",
            config.repo_id,
            "--root",
            str(config.root),
            "--fps",
            str(config.fps),
            "--vcodec",
            config.vcodec,
            "--encoder-threads",
            str(config.encoder_threads),
            "--encoder-queue-maxsize",
            str(config.encoder_queue_maxsize),
            "--video-crf",
            str(config.video_crf),
        ]
        if config.resume:
            command.append("--resume")
        child_env = os.environ.copy()
        # Never hide the GPU from Isaac itself; only the writer subprocess is
        # forced onto CPU.
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        child_env["PYTHONPATH"] = str(config.project_root)
        child_env["PYTHONNOUSERSITE"] = "1"
        for inherited_env in (
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
        ):
            child_env.pop(inherited_env, None)
        self.process = subprocess.Popen(
            command,
            cwd=config.project_root,
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        if self.process.stdin is None or self.process.stdout is None:
            self.process.kill()
            raise LeRobotStreamError("Failed to create LeRobot writer pipes")
        try:
            ready = receive_message(
                self.process.stdout,
                timeout_s=config.startup_timeout_s,
                health_check=self._check_process,
            )
            self._expect(ready, "ready")
            _write_collection_marker(config)
        except Exception as exc:
            self._terminate()
            detail = str(exc) if isinstance(exc, LeRobotStreamError) else repr(exc)
            raise LeRobotStreamStartupError(
                f"LeRobot writer failed during startup: {detail} "
                f"(status {self.process.returncode})"
            ) from exc
        self.active = False

    @staticmethod
    def _expect(message: Any, expected: str) -> dict[str, Any]:
        if not isinstance(message, dict):
            raise LeRobotStreamError(f"Invalid response from LeRobot writer: {message!r}")
        if message.get("status") == "error":
            error = str(message.get("error", "LeRobot writer failed"))
            if message.get("fatal"):
                raise LeRobotStreamStartupError(error)
            raise LeRobotStreamError(error)
        if message.get("status") != expected:
            raise LeRobotStreamError(
                f"Expected LeRobot writer status {expected!r}, got {message.get('status')!r}"
            )
        return message

    def _exchange(self, message: dict[str, Any], expected: str) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise LeRobotStreamError(
                f"LeRobot writer is not running (exit status {self.process.returncode})"
            )
        try:
            send_message(self.process.stdin, message)
            response = receive_message(
                self.process.stdout,
                timeout_s=self.config.response_timeout_s,
                health_check=self._check_process,
            )
            return self._expect(response, expected)
        except Exception as exc:
            self._terminate()
            if isinstance(exc, LeRobotStreamError):
                raise
            raise LeRobotStreamError("Lost connection to LeRobot writer") from exc

    def _check_process(self) -> None:
        return_code = self.process.poll()
        if return_code is not None:
            raise EOFError(f"LeRobot writer exited with status {return_code}")

    def begin(self, metadata: dict[str, Any]) -> None:
        if self.active:
            raise LeRobotStreamError("A LeRobot candidate episode is already active")
        self._exchange({"command": "begin", "metadata": metadata}, "begun")
        self.active = True

    def append(self, message: dict[str, Any]) -> int:
        if not self.active:
            raise LeRobotStreamError("No active LeRobot candidate episode")
        response = self._exchange({"command": "frame", **message}, "frame")
        return int(response["frame_count"])

    def finish(
        self,
        *,
        accepted: bool,
        success: bool,
        reason: str,
        timestamp_s: float | None = None,
    ) -> dict[str, Any]:
        if not self.active:
            raise LeRobotStreamError("No active LeRobot candidate episode")
        expected = "committed" if accepted else "discarded"
        response = self._exchange(
            {
                "command": "finish",
                "accepted": bool(accepted),
                "success": bool(success),
                "reason": str(reason),
                "timestamp": (
                    time.monotonic() if timestamp_s is None else float(timestamp_s)
                ),
            },
            expected,
        )
        self.active = False
        if accepted:
            try:
                return_code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired as exc:
                self._terminate()
                raise LeRobotStreamError("LeRobot writer did not exit after finalizing") from exc
            if return_code != 0:
                raise LeRobotStreamError(
                    f"LeRobot writer exited with status {return_code} after commit"
                )
        return response

    def shutdown(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            response = self._exchange({"command": "shutdown"}, "closed")
            del response
            self.process.wait(timeout=30)
        except Exception:
            self._terminate()
        finally:
            self.active = False

    def _terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


_SESSION_LOCK = threading.RLock()
_SIDECAR: _WriterSidecar | None = None
_SESSION_DATASET_ROOTS: set[Path] = set()
_SESSION_FILE_LOCKS: dict[Path, Any] = {}
_SESSION_MARKERS: dict[Path, str] = {}


def _acquire_dataset_file_lock(config: StreamConfig) -> bool:
    """Hold a process-wide advisory lock across per-episode sidecar restarts."""

    dataset_root = config.dataset_root
    if dataset_root in _SESSION_FILE_LOCKS:
        return False
    lock_dir = config.root / ".robodojo_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(dataset_root).encode("utf-8")).hexdigest()[:24]
    handle = (lock_dir / f"{digest}.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise LeRobotStreamStartupError(
            f"Another collector is already writing {dataset_root}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\ndataset={dataset_root}\n")
    handle.flush()
    _SESSION_FILE_LOCKS[dataset_root] = handle
    return True


def _release_dataset_file_locks() -> None:
    handles = list(_SESSION_FILE_LOCKS.values())
    _SESSION_FILE_LOCKS.clear()
    for handle in handles:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _acquire_sidecar(config: StreamConfig, metadata: dict[str, Any]) -> _WriterSidecar:
    global _SIDECAR
    with _SESSION_LOCK:
        if _SIDECAR is not None and _SIDECAR.process.poll() is not None:
            _SIDECAR = None
        if _SIDECAR is None:
            effective_resume = config.resume or config.dataset_root in _SESSION_DATASET_ROOTS
            new_lock = False
            try:
                new_lock = _acquire_dataset_file_lock(config)
                _SIDECAR = _WriterSidecar(replace(config, resume=effective_resume))
            except Exception as exc:
                if new_lock:
                    handle = _SESSION_FILE_LOCKS.pop(config.dataset_root)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
                if isinstance(exc, LeRobotStreamStartupError):
                    raise
                raise LeRobotStreamStartupError(
                    f"Could not start direct LeRobot collection: {exc}"
                ) from exc
            _SESSION_DATASET_ROOTS.add(config.dataset_root)
            if config.run_id:
                _SESSION_MARKERS[config.dataset_root] = config.run_id
        elif _SIDECAR.config.identity != config.identity:
            raise LeRobotStreamStartupError(
                "Cannot switch LeRobot dataset/configuration while a writer session is open"
            )
        _SIDECAR.begin(metadata)
        return _SIDECAR


def _drop_sidecar(sidecar: _WriterSidecar) -> None:
    global _SIDECAR
    with _SESSION_LOCK:
        if _SIDECAR is sidecar:
            _SIDECAR = None


class LeRobotStreamRecorder:
    """One candidate episode with the same append/finalize API as EpisodeRecorder."""

    def __init__(self, config: StreamConfig, metadata: dict[str, Any]):
        self.config = config
        self.metadata = dict(metadata)
        self._sidecar = _acquire_sidecar(config, self.metadata)
        self._frame_count = 0
        self._finished = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def record_dir(self) -> str:
        """Compatibility label used by the intervention loop's status line."""

        return str(self.config.dataset_root)

    def append(
        self,
        obs: dict,
        policy_action: dict | None,
        human_action: dict | None,
        executed_action: dict,
        control: dict[str, Any],
    ) -> None:
        del human_action  # The executed action plus mask/state are the training source of truth.
        if self._finished:
            raise LeRobotStreamError("Cannot append to a finished LeRobot episode")
        try:
            self._frame_count = self._sidecar.append(
                {
                    "obs": obs,
                    "policy_action": policy_action,
                    "executed_action": executed_action,
                    "control": control,
                    "task": self.metadata.get("task_name", ""),
                }
            )
        except Exception:
            _drop_sidecar(self._sidecar)
            self._finished = True
            raise

    def finalize(
        self,
        accepted: bool,
        success: bool,
        reason: str,
        timestamp_s: float | None = None,
    ) -> str | None:
        if self._finished:
            return None
        self._finished = True
        # An immediate ESC before the first simulation step is an empty
        # candidate, not a valid training episode.  Clear it cleanly; the main
        # loop still interprets ESC as an exit request.
        commit = bool(accepted) and self._frame_count > 0
        try:
            result = self._sidecar.finish(
                accepted=commit,
                success=bool(success),
                reason=str(reason),
                timestamp_s=timestamp_s,
            )
        except Exception:
            _drop_sidecar(self._sidecar)
            raise
        if not commit:
            return None
        _drop_sidecar(self._sidecar)
        episode_index = int(result["episode_index"])
        print(
            f"[LEROBOT] committed episode {episode_index} "
            f"({result['frame_count']} frames) -> {result['dataset_root']}",
            flush=True,
        )
        return str(result["dataset_root"])


def _task_metadata(task_env: Any) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    piperx_code_root = Path(
        os.environ.get(
            "ROBODOJO_PIPERX_CODE_ROOT",
            "/home/hoo/piper_x/lerobot_sealab-robodojo-v2",
        )
    )
    x5_code_root = Path(os.environ.get("ROBODOJO_X5_CODE_ROOT", str(project_root)))
    env_seeds = getattr(task_env, "env_seeds", None)
    layout_id = env_seeds[0] if env_seeds is not None and len(env_seeds) else -1
    seed_manager = getattr(task_env, "seed_manager", None)
    layout_cycle = getattr(task_env, "layout_cycle", getattr(seed_manager, "cycle_index", 0))
    policy_provenance = getattr(task_env, "policy_provenance", None)
    if not isinstance(policy_provenance, dict):
        policy_provenance = {}
    checkpoint_id = policy_provenance.get("checkpoint_id")
    control_mode = getattr(task_env, "control_mode", "keyboard_intervention")
    metadata = {
        "task_name": getattr(task_env, "task_name", os.environ.get("ROBODOJO_TASK_NAME", "")),
        "env_config": getattr(task_env, "config_name", os.environ.get("ROBODOJO_ENV_CFG", "")),
        "layout_id": int(layout_id),
        "layout_cycle": int(layout_cycle),
        "eval_seed": int(getattr(task_env, "eval_seed", -1)),
        "policy_name": getattr(task_env, "policy_name", "Pi_05"),
        "base_checkpoint": str(
            checkpoint_id
            or os.environ.get(
                "ROBODOJO_CHECKPOINT",
                getattr(task_env, "additional_info", ""),
            ),
        ),
        "policy_runtime": getattr(task_env, "policy_runtime", "xpolicy_ws_v0"),
        "policy_provenance": dict(policy_provenance),
        "robodojo_commit": _git_revision(project_root),
        "xpolicylab_commit": _git_revision(project_root / "XPolicyLab"),
        "run_id": os.environ.get("ROBODOJO_RUN_ID", ""),
        "control_mode": control_mode,
    }
    restore_lineage = getattr(task_env, "restore_lineage", None)
    if isinstance(restore_lineage, dict):
        metadata["recovery_source"] = dict(restore_lineage)
        if control_mode == "piperx_restore_recovery":
            source_checkpoint = restore_lineage.get("source_checkpoint")
            source_policy_provenance = restore_lineage.get(
                "source_policy_provenance"
            )
            if source_checkpoint:
                metadata["base_checkpoint"] = str(source_checkpoint)
            if isinstance(source_policy_provenance, dict):
                metadata["policy_provenance"] = dict(source_policy_provenance)
    if control_mode == "x5_policy_joint_intervention":
        metadata["hardware_embodiment"] = "arx_x5"
        metadata["hardware_profile"] = "arx_x5_identity_joint_v1"
        metadata["hardware_bridge_protocol"] = "robodojo_dual_joint_mirror_v1"
        metadata["hardware_bridge_commit"] = _git_revision(x5_code_root)
        metadata["hardware_bridge_dirty"] = _git_dirty(x5_code_root)
        metadata["hardware_control_topology"] = (
            "policy_sim_to_two_x5_manual_two_x5_to_sim"
        )
    elif control_mode == "piperx_sim_dagger":
        metadata["piperx_embodiment_profile"] = "arx_x5_piperx_relative_joint_v1"
        metadata["piperx_bridge_protocol"] = "robodojo_piperx_v4"
        metadata["piperx_bridge_commit"] = _git_revision(piperx_code_root)
        metadata["piperx_bridge_dirty"] = _git_dirty(piperx_code_root)
        metadata["piperx_control_topology"] = (
            "direct_joint_policy_sim_to_follower_to_leader_manual_joint_fanout"
        )
    elif (
        control_mode == "piperx_restore_recovery"
        or (
            control_mode == "piperx_policy_joint_intervention"
            and _env_bool("ROBODOJO_DUAL_MIRROR_RECORD", False)
        )
    ):
        metadata["piperx_embodiment_profile"] = "arx_x5_piperx_relative_joint_v1"
        metadata["piperx_bridge_protocol"] = "robodojo_piperx_dual_joint_mirror_v1"
        metadata["piperx_bridge_commit"] = _git_revision(piperx_code_root)
        metadata["piperx_bridge_dirty"] = _git_dirty(piperx_code_root)
        metadata["piperx_control_topology"] = (
            "restored_sim_to_two_leader_relative_joint_recovery"
            if control_mode == "piperx_restore_recovery"
            else "policy_sim_to_two_leaders_manual_two_leaders_to_sim"
        )
        metadata["piperx_restore_direct_control"] = (
            control_mode == "piperx_restore_recovery"
        )
    return metadata


def recorder_for_env(task_env: Any, record_dir: str | None = None) -> LeRobotStreamRecorder:
    """Create a direct LeRobot recorder; ``record_dir`` is a legacy no-op."""

    del record_dir
    frequency = int(task_env.obs_manager.collect_freq)
    try:
        config = config_from_environment(fps=frequency)
    except (OSError, TypeError, ValueError) as exc:
        raise LeRobotStreamStartupError(
            f"Invalid direct LeRobot collection configuration: {exc}"
        ) from exc
    return LeRobotStreamRecorder(
        config=config,
        metadata=_task_metadata(task_env),
    )


def close_lerobot_stream_session(*, clean_exit: bool = True) -> None:
    """Discard the candidate and close the writer.

    A clean operator exit removes the run marker.  Interpreter teardown keeps
    it so an external/PhysX restart with the same ``ROBODOJO_RUN_ID`` resumes
    automatically.
    """

    global _SIDECAR
    with _SESSION_LOCK:
        sidecar = _SIDECAR
        _SIDECAR = None
    if sidecar is not None:
        sidecar.shutdown()
    with _SESSION_LOCK:
        if clean_exit:
            markers = list(_SESSION_MARKERS.items())
            _SESSION_MARKERS.clear()
            for dataset_root, run_id in markers:
                _remove_collection_marker(dataset_root, run_id)
        _release_dataset_file_locks()


def _close_at_interpreter_exit() -> None:
    close_lerobot_stream_session(clean_exit=False)


atexit.register(_close_at_interpreter_exit)
