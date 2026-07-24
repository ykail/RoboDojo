"""RoboDojo raw-observation adapters for canonical policy payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from src.eval_client.policy_runtime.canonical import (
    ROBOT_SCHEMA_ID,
    CanonicalObservation,
    ObservationValidationSpec,
    PayloadErrorKind,
    PayloadValidationError,
    parse_observation,
)

_CAMERA_MAP = {
    "head": "cam_head",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}
_STATE_MAP = {
    "left_arm_joint_position": ("left_arm_joint_state", (6,)),
    "left_gripper_open_fraction_commanded": ("left_ee_joint_state", (1,)),
    "right_arm_joint_position": ("right_arm_joint_state", (6,)),
    "right_gripper_open_fraction_commanded": ("right_ee_joint_state", (1,)),
}
_CANONICAL_TO_RAW_PATH = {
    ("instruction",): ("instruction",),
    ("images", "head"): ("vision", "cam_head", "color"),
    ("images", "left_wrist"): ("vision", "cam_left_wrist", "color"),
    ("images", "right_wrist"): ("vision", "cam_right_wrist", "color"),
    ("proprio", "left_arm_joint_position"): (
        "state",
        "left_arm_joint_state",
    ),
    ("proprio", "left_gripper_open_fraction_commanded"): (
        "state",
        "left_ee_joint_state",
    ),
    ("proprio", "right_arm_joint_position"): (
        "state",
        "right_arm_joint_state",
    ),
    ("proprio", "right_gripper_open_fraction_commanded"): (
        "state",
        "right_ee_joint_state",
    ),
}


class RawObservationBuildError(ValueError):
    """A path-aware failure while mapping RoboDojo raw observation fields."""

    def __init__(
        self,
        path: tuple[str, ...],
        reason: str,
        *,
        canonical_error_kind: PayloadErrorKind | None = None,
    ) -> None:
        self.path = path
        self.reason = reason
        self.canonical_error_kind = canonical_error_kind
        rendered_path = "$" + "".join(f".{part}" for part in path)
        super().__init__(f"{rendered_path}: {reason}")


@dataclass(frozen=True, slots=True)
class ArxX5ObservationBuilder:
    """Map one current RoboDojo ARX X5 observation to the v1 wire schema.

    The builder performs only source-format conversion. The canonical parser
    remains the final strict boundary and owns immutable snapshots.
    """

    spec: ObservationValidationSpec
    expected_env_idx: int

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ObservationValidationSpec):
            raise TypeError("spec must be ObservationValidationSpec")
        if isinstance(self.expected_env_idx, bool) or not isinstance(
            self.expected_env_idx,
            Integral,
        ):
            raise TypeError("expected_env_idx must be an integer")
        if self.expected_env_idx < 0:
            raise ValueError("expected_env_idx must be non-negative")
        object.__setattr__(self, "expected_env_idx", int(self.expected_env_idx))

    def build(self, raw_observation: Mapping[str, Any]) -> CanonicalObservation:
        raw = _require_mapping(raw_observation, ())
        _require_raw_version(raw)
        _require_env_idx(raw, self.expected_env_idx)
        vision = _require_mapping(
            _require_field(raw, "vision", ()),
            ("vision",),
        )
        state = _require_mapping(
            _require_field(raw, "state", ()),
            ("state",),
        )

        images: dict[str, Any] = {}
        for canonical_name, raw_name in _CAMERA_MAP.items():
            camera_path = ("vision", raw_name)
            camera = _require_mapping(
                _require_field(vision, raw_name, ("vision",)),
                camera_path,
            )
            images[canonical_name] = _require_field(
                camera,
                "color",
                camera_path,
            )

        proprio: dict[str, Any] = {"robot_schema": ROBOT_SCHEMA_ID}
        for canonical_name, (raw_name, expected_shape) in _STATE_MAP.items():
            source = _require_field(state, raw_name, ("state",))
            if raw_name.endswith("_arm_joint_state"):
                proprio[canonical_name] = _coerce_arm_joint_vector(
                    source,
                    expected_shape=expected_shape,
                    path=("state", raw_name),
                )
            else:
                proprio[canonical_name] = _coerce_gripper_vector(
                    source,
                    expected_shape=expected_shape,
                    path=("state", raw_name),
                )

        payload = {
            "instruction": _coerce_instruction(
                _require_field(raw, "instruction", ()),
            ),
            "images": images,
            "proprio": proprio,
        }
        try:
            return parse_observation(payload, spec=self.spec)
        except PayloadValidationError as error:
            raw_path = _CANONICAL_TO_RAW_PATH.get(error.path, error.path)
            raise RawObservationBuildError(
                raw_path,
                error.reason,
                canonical_error_kind=error.kind,
            ) from error


def _require_mapping(
    value: Any,
    path: tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RawObservationBuildError(path, "expected a map")
    return value


def _require_field(
    value: Mapping[str, Any],
    field: str,
    path: tuple[str, ...],
) -> Any:
    if field not in value:
        raise RawObservationBuildError(
            (*path, field),
            f"missing required raw field {field!r}",
        )
    return value[field]


def _coerce_arm_joint_vector(
    value: Any,
    *,
    expected_shape: tuple[int, ...],
    path: tuple[str, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise RawObservationBuildError(
            path,
            "expected a NumPy ndarray for measured arm joints",
        )
    source = np.asarray(value)
    if source.dtype.kind != "f":
        raise RawObservationBuildError(
            path,
            f"expected floating-point arm joints, got dtype {source.dtype}",
        )
    return _coerce_float32_array(
        source,
        expected_shape=expected_shape,
        path=path,
    )


def _coerce_gripper_vector(
    value: Any,
    *,
    expected_shape: tuple[int, ...],
    path: tuple[str, ...],
) -> np.ndarray:
    if not isinstance(value, list | tuple | np.ndarray):
        raise RawObservationBuildError(
            path,
            "expected a list, tuple, or NumPy ndarray for gripper command",
        )
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise RawObservationBuildError(
            path,
            f"cannot convert value to an array: {error}",
        ) from error
    if source.dtype.kind != "f":
        raise RawObservationBuildError(
            path,
            f"expected floating-point gripper command, got dtype {source.dtype}",
        )
    return _coerce_float32_array(
        source,
        expected_shape=expected_shape,
        path=path,
    )


def _coerce_float32_array(
    source: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    path: tuple[str, ...],
) -> np.ndarray:
    if source.shape != expected_shape:
        raise RawObservationBuildError(
            path,
            f"expected shape {list(expected_shape)}, got {list(source.shape)}",
        )
    if not np.isfinite(source).all():
        raise RawObservationBuildError(path, "all values must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        result = source.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise RawObservationBuildError(
            path,
            "values cannot be represented as finite float32",
        )
    return result


def _coerce_instruction(value: Any) -> str:
    if isinstance(value, np.str_):
        return str(value)
    if not isinstance(value, str):
        raise RawObservationBuildError(
            ("instruction",),
            "expected a string",
        )
    return value


def _require_raw_version(raw: Mapping[str, Any]) -> None:
    version = _require_field(raw, "data_format_version", ())
    if not isinstance(version, str) or version != "v1.0":
        raise RawObservationBuildError(
            ("data_format_version",),
            f"expected raw observation version 'v1.0', got {version!r}",
        )


def _require_env_idx(
    raw: Mapping[str, Any],
    expected_env_idx: int,
) -> None:
    env_idx = _require_field(raw, "env_idx", ())
    if isinstance(env_idx, bool) or not isinstance(env_idx, Integral):
        raise RawObservationBuildError(
            ("env_idx",),
            "expected an integer",
        )
    if int(env_idx) != expected_env_idx:
        raise RawObservationBuildError(
            ("env_idx",),
            f"expected environment {expected_env_idx}, got {int(env_idx)}",
        )
