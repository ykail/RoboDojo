from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any


X5_CONTROL_MODE = "x5_policy_joint_intervention"
X5_CUDA_PIPELINE_ENV = "ROBODOJO_X5_CUDA_PIPELINE"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(environ: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE | _FALSE)}, got {raw!r}"
    )


def apply_x5_cuda_pipeline(
    env_cfg: Any,
    *,
    control_mode: str,
    device_id: int,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Select CUDA tensors/Fabric for the live dual-X5 path only.

    RoboDojo already calls ``overwrite_gpu_setting(1)``, which forces GPU
    PhysX.  Its stock sim config nevertheless leaves IsaacLab's tensor
    pipeline on CPU.  That combination reads scene state back to CPU and
    uploads joint targets again on every physics tick.  The X5 launcher opts
    into a coherent CUDA pipeline so ten 4-ms ticks do not pay those transfers.

    The feature remains explicitly reversible because a few RoboDojo tasks use
    CPU-only deformables.  This function never changes another control mode.
    """

    active_environ = os.environ if environ is None else environ
    if control_mode != X5_CONTROL_MODE or not _env_bool(
        active_environ,
        X5_CUDA_PIPELINE_ENV,
        default=False,
    ):
        return None
    if device_id < 0:
        raise ValueError("device_id must be non-negative for the X5 CUDA pipeline")

    # eval_policy.sh isolates the requested physical GPU with
    # CUDA_VISIBLE_DEVICES=<device_id>.  Inside that process it is always the
    # first (and normally only) logical CUDA device.
    device = "cuda:0"
    env_cfg.sim["device"] = device
    env_cfg.sim["use_fabric"] = True
    return device
