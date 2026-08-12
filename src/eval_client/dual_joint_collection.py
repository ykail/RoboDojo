"""Shared identity for live dual-joint online DAgger collectors.

The ARX X5 and PiPER-X hardware owners intentionally share one wire protocol
and one Isaac-side collection loop.  Their joint profiles and provenance are
different, but episode/reset/camera semantics must remain identical.  Keeping
that distinction here prevents a new embodiment from silently missing one of
the live-collector lifecycle branches in ``main`` or ``eval_env``.
"""

from __future__ import annotations

from dataclasses import dataclass


X5_CONTROL_MODE = "x5_policy_joint_intervention"
PIPERX_CONTROL_MODE = "piperx_policy_joint_intervention"
LIVE_DUAL_CONTROL_MODES = frozenset({X5_CONTROL_MODE, PIPERX_CONTROL_MODE})
DUAL_MIRROR_PROTOCOL = "robodojo_dual_joint_mirror_v1"


@dataclass(frozen=True)
class LiveDualModeSpec:
    control_mode: str
    hardware_label: str
    hardware_embodiment: str
    profile: str
    legacy_target_episodes_env: str


_MODE_SPECS = {
    X5_CONTROL_MODE: LiveDualModeSpec(
        control_mode=X5_CONTROL_MODE,
        hardware_label="ARX X5",
        hardware_embodiment="arx_x5",
        profile="arx_x5_identity_joint_v1",
        legacy_target_episodes_env="ROBODOJO_X5_TARGET_EPISODES",
    ),
    PIPERX_CONTROL_MODE: LiveDualModeSpec(
        control_mode=PIPERX_CONTROL_MODE,
        hardware_label="PiPER-X",
        hardware_embodiment="piper_x",
        profile="arx_x5_piperx_relative_joint_v1",
        legacy_target_episodes_env="ROBODOJO_PIPERX_TARGET_EPISODES",
    ),
}


def is_live_dual_control_mode(control_mode: str) -> bool:
    return str(control_mode) in LIVE_DUAL_CONTROL_MODES


def live_dual_mode_spec(control_mode: str) -> LiveDualModeSpec:
    try:
        return _MODE_SPECS[str(control_mode)]
    except KeyError as exc:
        raise ValueError(f"not a live dual-joint control mode: {control_mode!r}") from exc


def target_episode_environment_names(control_mode: str) -> tuple[str, str]:
    """Return the generic setting followed by the embodiment legacy alias."""

    spec = live_dual_mode_spec(control_mode)
    return (
        "ROBODOJO_DUAL_MIRROR_TARGET_EPISODES",
        spec.legacy_target_episodes_env,
    )
