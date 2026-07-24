"""Canonical ARX X5 observation and action schemas.

These parsers are strict protocol boundaries. Format conversion belongs in the
RoboDojo observation builder or a policy-side adapter, not here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from src.eval_client.policy_runtime.codec import MAX_FRAME_BYTES

OBSERVATION_SCHEMA_ID = "robodojo-arx-x5-dual-rgb-joint-v1"
ACTION_SCHEMA_ID = "robodojo-arx-x5-dual-absolute-joint-position-v1"
ROBOT_SCHEMA_ID = "arx_x5_dual_v1"
ACTION_CONTROL_MODE = "absolute_joint_position"

ARX_X5_SIM_RGB_SHAPE = (480, 640, 3)
MAX_IMAGE_DIMENSION = 4096
MAX_INSTRUCTION_BYTES = 4096
MAX_ACTION_HORIZON = 1024
MAX_CANONICAL_IMAGE_BYTES = MAX_FRAME_BYTES - 1024 * 1024

_OBSERVATION_FIELDS = frozenset({"instruction", "images", "proprio"})
_IMAGE_FIELDS = frozenset({"head", "left_wrist", "right_wrist"})
_PROPRIO_FIELDS = frozenset(
    {
        "robot_schema",
        "left_arm_joint_position",
        "left_gripper_open_fraction_commanded",
        "right_arm_joint_position",
        "right_gripper_open_fraction_commanded",
    }
)
_ACTION_FIELDS = frozenset({"control_mode", "control_dt_s", "commands"})
_INFER_FIELDS = frozenset({"observation"})
_INFER_RESULT_FIELDS = frozenset({"action"})
_COMMAND_FIELDS = frozenset(
    {
        "left_arm_joint_position",
        "left_gripper_open_fraction",
        "right_arm_joint_position",
        "right_gripper_open_fraction",
    }
)


class PayloadErrorKind(StrEnum):
    MISSING_FIELD = "missing_field"
    UNKNOWN_FIELD = "unknown_field"
    INVALID_TYPE = "invalid_type"
    INVALID_DTYPE = "invalid_dtype"
    INVALID_SHAPE = "invalid_shape"
    NON_FINITE = "non_finite"
    OUT_OF_RANGE = "out_of_range"
    CONSTRAINT_MISMATCH = "constraint_mismatch"


class PayloadValidationError(ValueError):
    """A direction-independent, path-aware canonical payload error.

    The dispatcher maps an invalid client request to ``invalid_payload`` and
    ``reject(token)``. An invalid backend result instead maps to
    ``infer_failed`` and ``fail(token)`` because policy state may have advanced.
    """

    def __init__(
        self,
        kind: PayloadErrorKind,
        path: tuple[str, ...],
        message: str,
    ) -> None:
        self.kind = kind
        self.path = path
        self.reason = message
        rendered_path = "$" + "".join(f".{part}" for part in path)
        self.message = f"{rendered_path}: {message}"
        self.details = {
            "kind": kind.value,
            "path": list(path),
        }
        super().__init__(self.message)


@dataclass(frozen=True, slots=True, init=False)
class CanonicalObservation:
    """Immutable snapshot of one canonical observation."""

    instruction: str
    head: np.ndarray
    left_wrist: np.ndarray
    right_wrist: np.ndarray
    left_arm_joint_position: np.ndarray
    left_gripper_open_fraction_commanded: np.ndarray
    right_arm_joint_position: np.ndarray
    right_gripper_open_fraction_commanded: np.ndarray

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("use parse_observation() to create CanonicalObservation")

    @classmethod
    def _from_validated(
        cls,
        *,
        instruction: str,
        head: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
        left_arm_joint_position: np.ndarray,
        left_gripper_open_fraction_commanded: np.ndarray,
        right_arm_joint_position: np.ndarray,
        right_gripper_open_fraction_commanded: np.ndarray,
    ) -> CanonicalObservation:
        instance = object.__new__(cls)
        object.__setattr__(instance, "instruction", instruction)
        object.__setattr__(instance, "head", head)
        object.__setattr__(instance, "left_wrist", left_wrist)
        object.__setattr__(instance, "right_wrist", right_wrist)
        object.__setattr__(
            instance,
            "left_arm_joint_position",
            left_arm_joint_position,
        )
        object.__setattr__(
            instance,
            "left_gripper_open_fraction_commanded",
            left_gripper_open_fraction_commanded,
        )
        object.__setattr__(
            instance,
            "right_arm_joint_position",
            right_arm_joint_position,
        )
        object.__setattr__(
            instance,
            "right_gripper_open_fraction_commanded",
            right_gripper_open_fraction_commanded,
        )
        return instance

    def to_payload(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "images": {
                "head": self.head,
                "left_wrist": self.left_wrist,
                "right_wrist": self.right_wrist,
            },
            "proprio": {
                "robot_schema": ROBOT_SCHEMA_ID,
                "left_arm_joint_position": self.left_arm_joint_position,
                "left_gripper_open_fraction_commanded": self.left_gripper_open_fraction_commanded,
                "right_arm_joint_position": self.right_arm_joint_position,
                "right_gripper_open_fraction_commanded": self.right_gripper_open_fraction_commanded,
            },
        }


@dataclass(frozen=True, slots=True)
class ObservationValidationSpec:
    """Client-owned image-shape constraints for one connection."""

    head_image_shape: tuple[int, int, int]
    left_wrist_image_shape: tuple[int, int, int]
    right_wrist_image_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        head_shape = _normalize_image_shape(
            self.head_image_shape,
            "head_image_shape",
        )
        left_wrist_shape = _normalize_image_shape(
            self.left_wrist_image_shape,
            "left_wrist_image_shape",
        )
        right_wrist_shape = _normalize_image_shape(
            self.right_wrist_image_shape,
            "right_wrist_image_shape",
        )
        total_image_bytes = sum(math.prod(shape) for shape in (head_shape, left_wrist_shape, right_wrist_shape))
        if total_image_bytes > MAX_CANONICAL_IMAGE_BYTES:
            raise ValueError(
                f"three RGB images exceed the canonical wire-byte budget of {MAX_CANONICAL_IMAGE_BYTES}",
            )
        object.__setattr__(
            self,
            "head_image_shape",
            head_shape,
        )
        object.__setattr__(
            self,
            "left_wrist_image_shape",
            left_wrist_shape,
        )
        object.__setattr__(
            self,
            "right_wrist_image_shape",
            right_wrist_shape,
        )


@dataclass(frozen=True, slots=True)
class JointLimits:
    """Six absolute joint-position limits for one arm, in radians."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        lower = _normalize_limit_tuple(self.lower, "lower")
        upper = _normalize_limit_tuple(self.upper, "upper")
        if any(low > high for low, high in zip(lower, upper, strict=True)):
            raise ValueError("joint lower limits must not exceed upper limits")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class ActionValidationSpec:
    """RoboDojo client-owned execution constraints for one connection."""

    expected_horizon: int
    expected_control_dt_s: float
    left_arm_limits: JointLimits
    right_arm_limits: JointLimits

    def __post_init__(self) -> None:
        if (
            isinstance(self.expected_horizon, bool)
            or not isinstance(self.expected_horizon, int)
            or not 1 <= self.expected_horizon <= MAX_ACTION_HORIZON
        ):
            raise ValueError(
                f"expected_horizon must be in [1, {MAX_ACTION_HORIZON}]",
            )
        if (
            isinstance(self.expected_control_dt_s, bool)
            or not isinstance(self.expected_control_dt_s, Real)
            or not math.isfinite(float(self.expected_control_dt_s))
            or self.expected_control_dt_s <= 0
        ):
            raise ValueError("expected_control_dt_s must be finite and positive")
        object.__setattr__(
            self,
            "expected_control_dt_s",
            float(self.expected_control_dt_s),
        )
        if not isinstance(self.left_arm_limits, JointLimits):
            raise TypeError("left_arm_limits must be JointLimits")
        if not isinstance(self.right_arm_limits, JointLimits):
            raise TypeError("right_arm_limits must be JointLimits")


@dataclass(frozen=True, slots=True, init=False)
class CanonicalActionChunk:
    """Immutable snapshot of one absolute joint-position action chunk."""

    control_dt_s: float
    left_arm_joint_position: np.ndarray
    left_gripper_open_fraction: np.ndarray
    right_arm_joint_position: np.ndarray
    right_gripper_open_fraction: np.ndarray

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("use parse_action_chunk() to create CanonicalActionChunk")

    @classmethod
    def _from_validated(
        cls,
        *,
        control_dt_s: float,
        left_arm_joint_position: np.ndarray,
        left_gripper_open_fraction: np.ndarray,
        right_arm_joint_position: np.ndarray,
        right_gripper_open_fraction: np.ndarray,
    ) -> CanonicalActionChunk:
        instance = object.__new__(cls)
        object.__setattr__(instance, "control_dt_s", control_dt_s)
        object.__setattr__(
            instance,
            "left_arm_joint_position",
            left_arm_joint_position,
        )
        object.__setattr__(
            instance,
            "left_gripper_open_fraction",
            left_gripper_open_fraction,
        )
        object.__setattr__(
            instance,
            "right_arm_joint_position",
            right_arm_joint_position,
        )
        object.__setattr__(
            instance,
            "right_gripper_open_fraction",
            right_gripper_open_fraction,
        )
        return instance

    @property
    def horizon(self) -> int:
        return self.left_arm_joint_position.shape[0]

    def to_payload(self) -> dict[str, Any]:
        return {
            "control_mode": ACTION_CONTROL_MODE,
            "control_dt_s": self.control_dt_s,
            "commands": {
                "left_arm_joint_position": self.left_arm_joint_position,
                "left_gripper_open_fraction": self.left_gripper_open_fraction,
                "right_arm_joint_position": self.right_arm_joint_position,
                "right_gripper_open_fraction": self.right_gripper_open_fraction,
            },
        }


def parse_infer_payload(
    payload: Mapping[str, Any],
    *,
    observation_spec: ObservationValidationSpec,
) -> CanonicalObservation:
    """Validate the exact ``INFER.payload`` wrapper and its observation."""

    infer = _require_exact_map(payload, _INFER_FIELDS, ())
    try:
        return parse_observation(
            infer["observation"],
            spec=observation_spec,
        )
    except PayloadValidationError as error:
        raise _with_path_prefix(error, "observation") from error


def parse_infer_result_payload(
    payload: Mapping[str, Any],
    *,
    action_spec: ActionValidationSpec,
) -> CanonicalActionChunk:
    """Validate the exact ``INFER_RESULT.payload`` wrapper and its action."""

    result = _require_exact_map(payload, _INFER_RESULT_FIELDS, ())
    try:
        return parse_action_chunk(result["action"], spec=action_spec)
    except PayloadValidationError as error:
        raise _with_path_prefix(error, "action") from error


def parse_observation(
    payload: Mapping[str, Any],
    *,
    spec: ObservationValidationSpec,
) -> CanonicalObservation:
    """Validate and snapshot one canonical observation payload."""

    if not isinstance(spec, ObservationValidationSpec):
        raise TypeError("spec must be ObservationValidationSpec")
    observation = _require_exact_map(payload, _OBSERVATION_FIELDS, ())
    instruction = _require_instruction(observation["instruction"])
    images = _require_exact_map(observation["images"], _IMAGE_FIELDS, ("images",))
    proprio = _require_exact_map(
        observation["proprio"],
        _PROPRIO_FIELDS,
        ("proprio",),
    )
    _require_literal_string(
        proprio["robot_schema"],
        ROBOT_SCHEMA_ID,
        ("proprio", "robot_schema"),
    )

    # Check every shape before copying any ndarray. This prevents a malformed
    # local strided view from forcing a large allocation before it is rejected.
    head_source = _require_array_metadata(
        images["head"],
        np.dtype(np.uint8),
        spec.head_image_shape,
        ("images", "head"),
    )
    left_wrist_source = _require_array_metadata(
        images["left_wrist"],
        np.dtype(np.uint8),
        spec.left_wrist_image_shape,
        ("images", "left_wrist"),
    )
    right_wrist_source = _require_array_metadata(
        images["right_wrist"],
        np.dtype(np.uint8),
        spec.right_wrist_image_shape,
        ("images", "right_wrist"),
    )
    left_arm_source = _require_array_metadata(
        proprio["left_arm_joint_position"],
        np.dtype(np.float32),
        (6,),
        ("proprio", "left_arm_joint_position"),
    )
    left_gripper_source = _require_array_metadata(
        proprio["left_gripper_open_fraction_commanded"],
        np.dtype(np.float32),
        (1,),
        ("proprio", "left_gripper_open_fraction_commanded"),
    )
    right_arm_source = _require_array_metadata(
        proprio["right_arm_joint_position"],
        np.dtype(np.float32),
        (6,),
        ("proprio", "right_arm_joint_position"),
    )
    right_gripper_source = _require_array_metadata(
        proprio["right_gripper_open_fraction_commanded"],
        np.dtype(np.float32),
        (1,),
        ("proprio", "right_gripper_open_fraction_commanded"),
    )

    head = _freeze_array(head_source)
    left_wrist = _freeze_array(left_wrist_source)
    right_wrist = _freeze_array(right_wrist_source)
    left_arm = _freeze_array(left_arm_source)
    left_gripper = _freeze_array(left_gripper_source)
    right_arm = _freeze_array(right_arm_source)
    right_gripper = _freeze_array(right_gripper_source)

    _require_finite(
        left_arm,
        ("proprio", "left_arm_joint_position"),
    )
    _require_finite(
        left_gripper,
        ("proprio", "left_gripper_open_fraction_commanded"),
    )
    _require_finite(
        right_arm,
        ("proprio", "right_arm_joint_position"),
    )
    _require_finite(
        right_gripper,
        ("proprio", "right_gripper_open_fraction_commanded"),
    )
    _require_unit_interval(
        left_gripper,
        ("proprio", "left_gripper_open_fraction_commanded"),
    )
    _require_unit_interval(
        right_gripper,
        ("proprio", "right_gripper_open_fraction_commanded"),
    )

    return CanonicalObservation._from_validated(
        instruction=instruction,
        head=head,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        left_arm_joint_position=left_arm,
        left_gripper_open_fraction_commanded=left_gripper,
        right_arm_joint_position=right_arm,
        right_gripper_open_fraction_commanded=right_gripper,
    )


def parse_action_chunk(
    payload: Mapping[str, Any],
    *,
    spec: ActionValidationSpec,
) -> CanonicalActionChunk:
    """Validate and snapshot one canonical action payload."""

    if not isinstance(spec, ActionValidationSpec):
        raise TypeError("spec must be ActionValidationSpec")
    action = _require_exact_map(payload, _ACTION_FIELDS, ())
    _require_literal_string(
        action["control_mode"],
        ACTION_CONTROL_MODE,
        ("control_mode",),
    )
    control_dt_s = _require_control_dt(
        action["control_dt_s"],
        spec.expected_control_dt_s,
    )
    commands = _require_exact_map(
        action["commands"],
        _COMMAND_FIELDS,
        ("commands",),
    )

    left_arm_source = _require_array_metadata(
        commands["left_arm_joint_position"],
        np.dtype(np.float32),
        (None, 6),
        ("commands", "left_arm_joint_position"),
    )
    left_gripper_source = _require_array_metadata(
        commands["left_gripper_open_fraction"],
        np.dtype(np.float32),
        (None, 1),
        ("commands", "left_gripper_open_fraction"),
    )
    right_arm_source = _require_array_metadata(
        commands["right_arm_joint_position"],
        np.dtype(np.float32),
        (None, 6),
        ("commands", "right_arm_joint_position"),
    )
    right_gripper_source = _require_array_metadata(
        commands["right_gripper_open_fraction"],
        np.dtype(np.float32),
        (None, 1),
        ("commands", "right_gripper_open_fraction"),
    )

    horizons = {
        left_arm_source.shape[0],
        left_gripper_source.shape[0],
        right_arm_source.shape[0],
        right_gripper_source.shape[0],
    }
    if len(horizons) != 1:
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            ("commands",),
            "all command arrays must share the same horizon",
        )
    horizon = horizons.pop()
    if not 1 <= horizon <= MAX_ACTION_HORIZON:
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            ("commands",),
            f"action horizon must be in [1, {MAX_ACTION_HORIZON}]",
        )
    if horizon != spec.expected_horizon:
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            ("commands",),
            f"expected client profile horizon {spec.expected_horizon}, got {horizon}",
        )

    left_arm = _freeze_array(left_arm_source)
    left_gripper = _freeze_array(left_gripper_source)
    right_arm = _freeze_array(right_arm_source)
    right_gripper = _freeze_array(right_gripper_source)

    _require_finite(
        left_arm,
        ("commands", "left_arm_joint_position"),
    )
    _require_finite(
        left_gripper,
        ("commands", "left_gripper_open_fraction"),
    )
    _require_finite(
        right_arm,
        ("commands", "right_arm_joint_position"),
    )
    _require_finite(
        right_gripper,
        ("commands", "right_gripper_open_fraction"),
    )
    _require_unit_interval(
        left_gripper,
        ("commands", "left_gripper_open_fraction"),
    )
    _require_unit_interval(
        right_gripper,
        ("commands", "right_gripper_open_fraction"),
    )
    _require_joint_limits(
        left_arm,
        spec.left_arm_limits,
        ("commands", "left_arm_joint_position"),
    )
    _require_joint_limits(
        right_arm,
        spec.right_arm_limits,
        ("commands", "right_arm_joint_position"),
    )

    return CanonicalActionChunk._from_validated(
        control_dt_s=control_dt_s,
        left_arm_joint_position=left_arm,
        left_gripper_open_fraction=left_gripper,
        right_arm_joint_position=right_arm,
        right_gripper_open_fraction=right_gripper,
    )


def _require_exact_map(
    value: Any,
    expected_fields: frozenset[str],
    path: tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(PayloadErrorKind.INVALID_TYPE, path, "expected a map")
    fields = set(value)
    non_string = [field for field in fields if not isinstance(field, str)]
    if non_string:
        _fail(
            PayloadErrorKind.INVALID_TYPE,
            path,
            "map keys must be strings",
        )
    missing = sorted(expected_fields - fields)
    if missing:
        _fail(
            PayloadErrorKind.MISSING_FIELD,
            path + (missing[0],),
            f"missing required field {missing[0]!r}",
        )
    extra = sorted(fields - expected_fields)
    if extra:
        _fail(
            PayloadErrorKind.UNKNOWN_FIELD,
            path + (extra[0],),
            f"unknown field {extra[0]!r}",
        )
    return value


def _require_instruction(value: Any) -> str:
    if not isinstance(value, str):
        _fail(
            PayloadErrorKind.INVALID_TYPE,
            ("instruction",),
            "expected a string",
        )
    if not value.strip():
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            ("instruction",),
            "instruction must not be empty",
        )
    encoded_length = len(value.encode("utf-8"))
    if encoded_length > MAX_INSTRUCTION_BYTES:
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            ("instruction",),
            f"UTF-8 length exceeds {MAX_INSTRUCTION_BYTES} bytes",
        )
    return str(value)


def _require_literal_string(
    value: Any,
    expected: str,
    path: tuple[str, ...],
) -> None:
    if not isinstance(value, str):
        _fail(PayloadErrorKind.INVALID_TYPE, path, "expected a string")
    if value != expected:
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            path,
            f"expected {expected!r}",
        )


def _require_array_metadata(
    value: Any,
    expected_dtype: np.dtype[Any],
    expected_shape: tuple[int | None, ...],
    path: tuple[str, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        _fail(
            PayloadErrorKind.INVALID_TYPE,
            path,
            "expected a NumPy ndarray",
        )
    # Strip ndarray subclasses without copying before trusting metadata:
    # subclasses may override shape, dtype, or ndim properties.
    array = np.asarray(value)
    if array.dtype != expected_dtype:
        _fail(
            PayloadErrorKind.INVALID_DTYPE,
            path,
            f"expected dtype {expected_dtype}, got {array.dtype}",
        )
    if array.ndim != len(expected_shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, expected_shape, strict=True)
    ):
        rendered_shape = [dimension if dimension is not None else "*" for dimension in expected_shape]
        _fail(
            PayloadErrorKind.INVALID_SHAPE,
            path,
            f"expected shape {rendered_shape}, got {list(array.shape)}",
        )
    return array


def _freeze_array(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value)
    return np.frombuffer(
        source.tobytes(order="C"),
        dtype=source.dtype,
    ).reshape(source.shape)


def _require_finite(value: np.ndarray, path: tuple[str, ...]) -> None:
    if not np.isfinite(value).all():
        _fail(PayloadErrorKind.NON_FINITE, path, "all values must be finite")


def _require_unit_interval(value: np.ndarray, path: tuple[str, ...]) -> None:
    if np.any(value < 0.0) or np.any(value > 1.0):
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            path,
            "values must be in the closed interval [0, 1]",
        )


def _require_control_dt(value: Any, expected: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail(
            PayloadErrorKind.INVALID_TYPE,
            ("control_dt_s",),
            "expected a real number",
        )
    result = float(value)
    if not math.isfinite(result):
        _fail(
            PayloadErrorKind.NON_FINITE,
            ("control_dt_s",),
            "control period must be finite",
        )
    if result <= 0:
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            ("control_dt_s",),
            "control period must be positive",
        )
    if not math.isclose(result, expected, rel_tol=1e-6, abs_tol=1e-9):
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            ("control_dt_s",),
            f"expected client profile control period {expected}, got {result}",
        )
    return result


def _require_joint_limits(
    value: np.ndarray,
    limits: JointLimits,
    path: tuple[str, ...],
) -> None:
    lower = np.asarray(limits.lower, dtype=np.float64)
    upper = np.asarray(limits.upper, dtype=np.float64)
    if np.any(value < lower) or np.any(value > upper):
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            path,
            "joint position exceeds the active RoboDojo limits",
        )


def _normalize_image_shape(
    value: Sequence[Integral],
    field_name: str,
) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{field_name} must be a three-element sequence")
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain height, width, and channels")
    normalized: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise TypeError(f"{field_name} dimensions must be integers")
        normalized.append(int(item))
    height, width, channels = normalized
    if not 1 <= height <= MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"{field_name} height must be in [1, {MAX_IMAGE_DIMENSION}]",
        )
    if not 1 <= width <= MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"{field_name} width must be in [1, {MAX_IMAGE_DIMENSION}]",
        )
    if channels != 3:
        raise ValueError(f"{field_name} must have exactly three RGB channels")
    return height, width, channels


def _normalize_limit_tuple(
    value: Sequence[Real],
    field_name: str,
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{field_name} joint limits must be a sequence")
    if len(value) != 6:
        raise ValueError(f"{field_name} joint limits must contain six values")
    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{field_name} joint limits must be real numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} joint limits must be finite")
        normalized.append(number)
    return tuple(normalized)


def _fail(
    kind: PayloadErrorKind,
    path: tuple[str, ...],
    message: str,
) -> None:
    raise PayloadValidationError(kind, path, message)


def _with_path_prefix(
    error: PayloadValidationError,
    prefix: str,
) -> PayloadValidationError:
    return PayloadValidationError(
        error.kind,
        (prefix, *error.path),
        error.reason,
    )


ARX_X5_SIM_OBSERVATION_SPEC = ObservationValidationSpec(
    head_image_shape=ARX_X5_SIM_RGB_SHAPE,
    left_wrist_image_shape=ARX_X5_SIM_RGB_SHAPE,
    right_wrist_image_shape=ARX_X5_SIM_RGB_SHAPE,
)
