"""Strict lifecycle payloads for ``robodojo-policy-v1``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

from src.eval_client.policy_runtime.canonical import (
    ACTION_SCHEMA_ID,
    MAX_ACTION_HORIZON,
    MAX_IMAGE_DIMENSION,
    OBSERVATION_SCHEMA_ID,
    ROBOT_SCHEMA_ID,
    ActionValidationSpec,
    JointLimits,
    ObservationValidationSpec,
    PayloadErrorKind,
    PayloadValidationError,
)
from src.eval_client.policy_runtime.errors import ErrorCode
from src.eval_client.policy_runtime.execution_profile import (
    ACTION_CHUNK_CONSUMPTION,
    ACTION_NEXT_INFER_OBSERVATION,
    ACTION_PREEMPTION_BOUNDARY,
    PolicyExecutionProfile,
)

MAX_LIFECYCLE_TEXT_BYTES = 4096
MAX_ERROR_DETAILS_DEPTH = 8
MAX_ERROR_DETAILS_ITEMS = 1000
MAX_MSGPACK_UNSIGNED_INT = (1 << 64) - 1

_HELLO_FIELDS = frozenset({"schemas", "execution_profile"})
_HELLO_ACK_FIELDS = frozenset({"schemas", "execution_profile", "policy"})
_SCHEMA_FIELDS = frozenset({"observation", "action", "robot"})
_EXECUTION_FIELDS = frozenset({"images", "action"})
_IMAGE_PROFILE_FIELDS = frozenset({"head", "left_wrist", "right_wrist"})
_ACTION_PROFILE_FIELDS = frozenset(
    {
        "horizon",
        "control_dt_s",
        "chunk_consumption",
        "preemption_boundary",
        "next_infer_observation",
        "left_arm_joint_limits",
        "right_arm_joint_limits",
    }
)
_JOINT_LIMIT_FIELDS = frozenset({"lower", "upper"})
_POLICY_FIELDS = frozenset(
    {
        "implementation",
        "policy_family",
        "adapter_profile",
        "config_name",
        "checkpoint_id",
        "checkpoint_digest",
        "checkpoint_step",
        "code_revision",
        "dirty",
    }
)
_RESET_FIELDS = frozenset(
    {
        "task_name",
        "simulator_seed",
        "policy_seed",
        "layout_id",
        "layout_cycle",
        "reason",
    }
)
_TRIAL_END_FIELDS = frozenset({"status", "success", "score", "reason"})
_ERROR_FIELDS = frozenset({"code", "message", "details", "retryable"})

_CORRELATED_SERVER_ERROR_CODES = frozenset(
    {
        ErrorCode.INVALID_PAYLOAD,
        ErrorCode.INVALID_STATE,
        ErrorCode.SESSION_BUSY,
        ErrorCode.EPISODE_MISMATCH,
        ErrorCode.INFERENCE_INDEX_MISMATCH,
        ErrorCode.INFER_FAILED,
        ErrorCode.RESET_FAILED,
        ErrorCode.INTERNAL,
    }
)


class ResetReason(StrEnum):
    EPISODE_START = "episode_start"
    OPERATOR_RETRY = "operator_retry"
    SIMULATOR_RECOVERY = "simulator_recovery"
    TRANSPORT_RECOVERY = "transport_recovery"


class TrialStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    ABORTED = "aborted"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PolicyProvenance:
    implementation: str
    policy_family: str
    adapter_profile: str
    config_name: str
    checkpoint_id: str
    checkpoint_digest: str
    checkpoint_step: int | None
    code_revision: str
    dirty: bool

    def __post_init__(self) -> None:
        for field_name in (
            "implementation",
            "policy_family",
            "adapter_profile",
            "config_name",
        ):
            _validate_local_text(getattr(self, field_name), field_name)
        _validate_local_checkpoint_id(self.checkpoint_id)
        _validate_local_git_revision(self.code_revision, "code_revision")
        _validate_local_sha256_digest(
            self.checkpoint_digest,
            "checkpoint_digest",
        )
        _validate_local_optional_counter(
            self.checkpoint_step,
            "checkpoint_step",
        )
        if not isinstance(self.dirty, bool):
            raise TypeError("dirty must be a bool")

    def to_payload(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "policy_family": self.policy_family,
            "adapter_profile": self.adapter_profile,
            "config_name": self.config_name,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_digest": self.checkpoint_digest,
            "checkpoint_step": self.checkpoint_step,
            "code_revision": self.code_revision,
            "dirty": self.dirty,
        }


@dataclass(frozen=True, slots=True)
class ResetPayload:
    task_name: str
    simulator_seed: int
    policy_seed: int
    layout_id: int
    layout_cycle: int
    reason: ResetReason

    def __post_init__(self) -> None:
        _validate_local_text(self.task_name, "task_name")
        for field_name in (
            "simulator_seed",
            "layout_id",
            "layout_cycle",
        ):
            _validate_local_counter(getattr(self, field_name), field_name)
        _validate_local_uint32(self.policy_seed, "policy_seed")
        if not isinstance(self.reason, ResetReason):
            raise TypeError("reason must be ResetReason")

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "simulator_seed": self.simulator_seed,
            "policy_seed": self.policy_seed,
            "layout_id": self.layout_id,
            "layout_cycle": self.layout_cycle,
            "reason": self.reason.value,
        }


@dataclass(frozen=True, slots=True)
class TrialEndPayload:
    status: TrialStatus
    success: bool | None
    score: float | None
    reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, TrialStatus):
            raise TypeError("status must be TrialStatus")
        _validate_trial_outcome(self.status, self.success)
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, Real) or not math.isfinite(float(self.score)):
                raise ValueError("score must be null or a finite real number")
            object.__setattr__(self, "score", float(self.score))
        if self.reason is not None:
            _validate_local_text(self.reason, "reason")

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "success": self.success,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RemoteErrorPayload:
    code: ErrorCode
    message: str
    details: Mapping[str, Any]
    retryable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.code, ErrorCode):
            raise TypeError("code must be ErrorCode")
        if self.code not in _CORRELATED_SERVER_ERROR_CODES:
            raise ValueError("code must be a correlated server ErrorCode")
        _validate_local_text(self.message, "message")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a map")
        object.__setattr__(
            self,
            "details",
            _transform_error_detail(
                self.details,
                ("details",),
                freeze=True,
                payload_errors=False,
            ),
        )
        if self.retryable is not False:
            raise ValueError("robodojo-policy-v1 errors require retryable=false")

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": _thaw_error_detail(self.details),
            "retryable": self.retryable,
        }


def build_hello_payload(
    profile: PolicyExecutionProfile,
) -> dict[str, Any]:
    _require_profile_instance(profile)
    return {
        "schemas": profile.schemas_payload(),
        "execution_profile": profile.execution_payload(),
    }


def parse_hello_payload(
    payload: Mapping[str, Any],
    *,
    supported_profile: PolicyExecutionProfile,
) -> PolicyExecutionProfile:
    _require_profile_instance(supported_profile)
    hello = _require_exact_map(payload, _HELLO_FIELDS, ())
    _parse_schemas(hello["schemas"], ("schemas",))
    requested_profile = _parse_execution_profile(
        hello["execution_profile"],
        ("execution_profile",),
    )
    if requested_profile != supported_profile:
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            ("execution_profile",),
            "client requested an execution profile this server does not support",
        )
    return requested_profile


def build_hello_ack_payload(
    profile: PolicyExecutionProfile,
    provenance: PolicyProvenance,
) -> dict[str, Any]:
    _require_profile_instance(profile)
    if not isinstance(provenance, PolicyProvenance):
        raise TypeError("provenance must be PolicyProvenance")
    return {
        "schemas": profile.schemas_payload(),
        "execution_profile": profile.execution_payload(),
        "policy": provenance.to_payload(),
    }


def parse_hello_ack_payload(
    payload: Mapping[str, Any],
    *,
    expected_profile: PolicyExecutionProfile,
) -> PolicyProvenance:
    _require_profile_instance(expected_profile)
    ack = _require_exact_map(payload, _HELLO_ACK_FIELDS, ())
    _parse_schemas(ack["schemas"], ("schemas",))
    actual_profile = _parse_execution_profile(
        ack["execution_profile"],
        ("execution_profile",),
    )
    if actual_profile != expected_profile:
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            ("execution_profile",),
            "server did not confirm the exact client-owned execution profile",
        )
    return _parse_policy_provenance(ack["policy"], ("policy",))


def parse_reset_payload(payload: Mapping[str, Any]) -> ResetPayload:
    reset = _require_exact_map(payload, _RESET_FIELDS, ())
    return ResetPayload(
        task_name=_require_text(reset["task_name"], ("task_name",)),
        simulator_seed=_require_counter(
            reset["simulator_seed"],
            ("simulator_seed",),
        ),
        policy_seed=_require_uint32(
            reset["policy_seed"],
            ("policy_seed",),
        ),
        layout_id=_require_counter(reset["layout_id"], ("layout_id",)),
        layout_cycle=_require_counter(
            reset["layout_cycle"],
            ("layout_cycle",),
        ),
        reason=_require_enum(
            reset["reason"],
            ResetReason,
            ("reason",),
        ),
    )


def parse_trial_end_payload(
    payload: Mapping[str, Any],
) -> TrialEndPayload:
    trial_end = _require_exact_map(payload, _TRIAL_END_FIELDS, ())
    status = _require_enum(
        trial_end["status"],
        TrialStatus,
        ("status",),
    )
    success = trial_end["success"]
    if success is not None and not isinstance(success, bool):
        _fail(
            PayloadErrorKind.INVALID_TYPE,
            ("success",),
            "expected null or a bool",
        )
    try:
        _validate_trial_outcome(status, success)
    except ValueError as error:
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            ("success",),
            str(error),
        )
    score = _require_optional_finite_real(trial_end["score"], ("score",))
    reason = _require_optional_text(trial_end["reason"], ("reason",))
    return TrialEndPayload(
        status=status,
        success=success,
        score=score,
        reason=reason,
    )


def parse_error_payload(payload: Mapping[str, Any]) -> RemoteErrorPayload:
    remote_error = _require_exact_map(payload, _ERROR_FIELDS, ())
    code = _require_enum(
        remote_error["code"],
        ErrorCode,
        ("code",),
    )
    if code not in _CORRELATED_SERVER_ERROR_CODES:
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            ("code",),
            "ERROR payload requires a correlated server error code",
        )
    details_value = remote_error["details"]
    if not isinstance(details_value, Mapping):
        _fail(
            PayloadErrorKind.INVALID_TYPE,
            ("details",),
            "expected a map",
        )
    details = _transform_error_detail(
        details_value,
        ("details",),
        freeze=False,
        payload_errors=True,
    )
    retryable = remote_error["retryable"]
    if not isinstance(retryable, bool):
        _fail(
            PayloadErrorKind.INVALID_TYPE,
            ("retryable",),
            "expected a bool",
        )
    if retryable:
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            ("retryable",),
            "robodojo-policy-v1 errors require retryable=false",
        )
    return RemoteErrorPayload(
        code=code,
        message=_require_text(remote_error["message"], ("message",)),
        details=details,
        retryable=retryable,
    )


def parse_empty_success_payload(payload: Mapping[str, Any]) -> None:
    _require_exact_map(payload, frozenset(), ())


def _parse_schemas(value: Any, path: tuple[str, ...]) -> None:
    schemas = _require_exact_map(value, _SCHEMA_FIELDS, path)
    for field, expected in (
        ("observation", OBSERVATION_SCHEMA_ID),
        ("action", ACTION_SCHEMA_ID),
        ("robot", ROBOT_SCHEMA_ID),
    ):
        _require_literal_string(
            schemas[field],
            expected,
            (*path, field),
        )


def _parse_execution_profile(
    value: Any,
    path: tuple[str, ...],
) -> PolicyExecutionProfile:
    execution = _require_exact_map(value, _EXECUTION_FIELDS, path)
    images_path = (*path, "images")
    images = _require_exact_map(
        execution["images"],
        _IMAGE_PROFILE_FIELDS,
        images_path,
    )
    head_shape = _require_image_shape(
        images["head"],
        (*images_path, "head"),
    )
    left_wrist_shape = _require_image_shape(
        images["left_wrist"],
        (*images_path, "left_wrist"),
    )
    right_wrist_shape = _require_image_shape(
        images["right_wrist"],
        (*images_path, "right_wrist"),
    )

    action_path = (*path, "action")
    action = _require_exact_map(
        execution["action"],
        _ACTION_PROFILE_FIELDS,
        action_path,
    )
    horizon = _require_counter(
        action["horizon"],
        (*action_path, "horizon"),
    )
    if not 1 <= horizon <= MAX_ACTION_HORIZON:
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            (*action_path, "horizon"),
            f"horizon must be in [1, {MAX_ACTION_HORIZON}]",
        )
    control_dt_s = _require_positive_finite_real(
        action["control_dt_s"],
        (*action_path, "control_dt_s"),
    )
    _require_literal_string(
        action["chunk_consumption"],
        ACTION_CHUNK_CONSUMPTION,
        (*action_path, "chunk_consumption"),
    )
    _require_literal_string(
        action["preemption_boundary"],
        ACTION_PREEMPTION_BOUNDARY,
        (*action_path, "preemption_boundary"),
    )
    _require_literal_string(
        action["next_infer_observation"],
        ACTION_NEXT_INFER_OBSERVATION,
        (*action_path, "next_infer_observation"),
    )
    left_limits = _parse_joint_limits(
        action["left_arm_joint_limits"],
        (*action_path, "left_arm_joint_limits"),
    )
    right_limits = _parse_joint_limits(
        action["right_arm_joint_limits"],
        (*action_path, "right_arm_joint_limits"),
    )

    try:
        return PolicyExecutionProfile(
            observation_spec=ObservationValidationSpec(
                head_image_shape=head_shape,
                left_wrist_image_shape=left_wrist_shape,
                right_wrist_image_shape=right_wrist_shape,
            ),
            action_spec=ActionValidationSpec(
                expected_horizon=horizon,
                expected_control_dt_s=control_dt_s,
                left_arm_limits=left_limits,
                right_arm_limits=right_limits,
            ),
        )
    except (TypeError, ValueError) as error:
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            path,
            str(error),
        )


def _parse_joint_limits(
    value: Any,
    path: tuple[str, ...],
) -> JointLimits:
    limits = _require_exact_map(value, _JOINT_LIMIT_FIELDS, path)
    lower = _require_real_list(limits["lower"], (*path, "lower"), length=6)
    upper = _require_real_list(limits["upper"], (*path, "upper"), length=6)
    try:
        return JointLimits(lower=tuple(lower), upper=tuple(upper))
    except (TypeError, ValueError) as error:
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            path,
            str(error),
        )


def _parse_policy_provenance(
    value: Any,
    path: tuple[str, ...],
) -> PolicyProvenance:
    policy = _require_exact_map(value, _POLICY_FIELDS, path)
    checkpoint_step = policy["checkpoint_step"]
    if checkpoint_step is not None:
        checkpoint_step = _require_counter(
            checkpoint_step,
            (*path, "checkpoint_step"),
        )
    dirty = policy["dirty"]
    if not isinstance(dirty, bool):
        _fail(
            PayloadErrorKind.INVALID_TYPE,
            (*path, "dirty"),
            "expected a bool",
        )
    return PolicyProvenance(
        implementation=_require_text(
            policy["implementation"],
            (*path, "implementation"),
        ),
        policy_family=_require_text(
            policy["policy_family"],
            (*path, "policy_family"),
        ),
        adapter_profile=_require_text(
            policy["adapter_profile"],
            (*path, "adapter_profile"),
        ),
        config_name=_require_text(
            policy["config_name"],
            (*path, "config_name"),
        ),
        checkpoint_id=_require_checkpoint_id(
            policy["checkpoint_id"],
            (*path, "checkpoint_id"),
        ),
        checkpoint_digest=_require_sha256_digest(
            policy["checkpoint_digest"],
            (*path, "checkpoint_digest"),
        ),
        checkpoint_step=checkpoint_step,
        code_revision=_require_git_revision(
            policy["code_revision"],
            (*path, "code_revision"),
        ),
        dirty=dirty,
    )


def _require_exact_map(
    value: Any,
    expected_fields: frozenset[str],
    path: tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(PayloadErrorKind.INVALID_TYPE, path, "expected a map")
    fields = set(value)
    if any(not isinstance(field, str) for field in fields):
        _fail(
            PayloadErrorKind.INVALID_TYPE,
            path,
            "map keys must be strings",
        )
    missing = sorted(expected_fields - fields)
    if missing:
        _fail(
            PayloadErrorKind.MISSING_FIELD,
            (*path, missing[0]),
            f"missing required field {missing[0]!r}",
        )
    extra = sorted(fields - expected_fields)
    if extra:
        _fail(
            PayloadErrorKind.UNKNOWN_FIELD,
            (*path, extra[0]),
            f"unknown field {extra[0]!r}",
        )
    return value


def _require_text(value: Any, path: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        _fail(PayloadErrorKind.INVALID_TYPE, path, "expected a string")
    if not value.strip():
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            path,
            "string must not be empty",
        )
    if len(value.encode("utf-8")) > MAX_LIFECYCLE_TEXT_BYTES:
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            path,
            f"UTF-8 length exceeds {MAX_LIFECYCLE_TEXT_BYTES} bytes",
        )
    return value


def _require_optional_text(
    value: Any,
    path: tuple[str, ...],
) -> str | None:
    if value is None:
        return None
    return _require_text(value, path)


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


def _require_counter(value: Any, path: tuple[str, ...]) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _fail(
            PayloadErrorKind.INVALID_TYPE,
            path,
            "expected a non-negative integer",
        )
    result = int(value)
    if result < 0:
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            path,
            "expected a non-negative integer",
        )
    if result > MAX_MSGPACK_UNSIGNED_INT:
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            path,
            "integer is outside the msgpack unsigned 64-bit range",
        )
    return result


def _require_uint32(value: Any, path: tuple[str, ...]) -> int:
    result = _require_counter(value, path)
    if result > (1 << 32) - 1:
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            path,
            "expected an integer in [0, 2^32 - 1]",
        )
    return result


def _require_sha256_digest(
    value: Any,
    path: tuple[str, ...],
) -> str:
    digest = _require_text(value, path)
    prefix = "sha256:"
    hexadecimal = digest[len(prefix) :] if digest.startswith(prefix) else ""
    if len(hexadecimal) != 64 or any(character not in "0123456789abcdef" for character in hexadecimal):
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            path,
            "expected 'sha256:' followed by 64 lowercase hexadecimal characters",
        )
    return digest


def _require_checkpoint_id(
    value: Any,
    path: tuple[str, ...],
) -> str:
    checkpoint_id = _require_text(value, path)
    if _looks_like_local_path(checkpoint_id):
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            path,
            "checkpoint_id must be a portable ID or non-file URI, not a local path",
        )
    return checkpoint_id


def _require_git_revision(
    value: Any,
    path: tuple[str, ...],
) -> str:
    revision = _require_text(value, path)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            path,
            "expected a full 40-character lowercase Git commit SHA",
        )
    return revision


def _require_optional_finite_real(
    value: Any,
    path: tuple[str, ...],
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail(
            PayloadErrorKind.INVALID_TYPE,
            path,
            "expected null or a real number",
        )
    result = float(value)
    if not math.isfinite(result):
        _fail(
            PayloadErrorKind.NON_FINITE,
            path,
            "number must be finite",
        )
    return result


def _require_positive_finite_real(
    value: Any,
    path: tuple[str, ...],
) -> float:
    result = _require_optional_finite_real(value, path)
    if result is None:
        _fail(PayloadErrorKind.INVALID_TYPE, path, "expected a real number")
    if result <= 0:
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            path,
            "number must be positive",
        )
    return result


def _require_image_shape(
    value: Any,
    path: tuple[str, ...],
) -> tuple[int, int, int]:
    if not isinstance(value, list):
        _fail(
            PayloadErrorKind.INVALID_TYPE,
            path,
            "expected a three-element list",
        )
    if len(value) != 3:
        _fail(
            PayloadErrorKind.INVALID_SHAPE,
            path,
            "expected [height, width, 3]",
        )
    dimensions: list[int] = []
    for index, dimension in enumerate(value):
        if isinstance(dimension, bool) or not isinstance(dimension, Integral):
            _fail(
                PayloadErrorKind.INVALID_TYPE,
                (*path, str(index)),
                "image dimensions must be integers",
            )
        dimensions.append(int(dimension))
    height, width, channels = dimensions
    if not 1 <= height <= MAX_IMAGE_DIMENSION:
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            (*path, "0"),
            f"height must be in [1, {MAX_IMAGE_DIMENSION}]",
        )
    if not 1 <= width <= MAX_IMAGE_DIMENSION:
        _fail(
            PayloadErrorKind.OUT_OF_RANGE,
            (*path, "1"),
            f"width must be in [1, {MAX_IMAGE_DIMENSION}]",
        )
    if channels != 3:
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            (*path, "2"),
            "RGB images require exactly three channels",
        )
    return height, width, channels


def _require_real_list(
    value: Any,
    path: tuple[str, ...],
    *,
    length: int,
) -> list[float]:
    if not isinstance(value, list):
        _fail(PayloadErrorKind.INVALID_TYPE, path, "expected a list")
    if len(value) != length:
        _fail(
            PayloadErrorKind.INVALID_SHAPE,
            path,
            f"expected {length} values",
        )
    result: list[float] = []
    for index, item in enumerate(value):
        item_path = (*path, str(index))
        if isinstance(item, bool) or not isinstance(item, Real):
            _fail(
                PayloadErrorKind.INVALID_TYPE,
                item_path,
                "expected a real number",
            )
        number = float(item)
        if not math.isfinite(number):
            _fail(
                PayloadErrorKind.NON_FINITE,
                item_path,
                "number must be finite",
            )
        result.append(number)
    return result


def _require_enum(
    value: Any,
    enum_type: type[StrEnum],
    path: tuple[str, ...],
) -> Any:
    if not isinstance(value, str):
        _fail(PayloadErrorKind.INVALID_TYPE, path, "expected a string")
    try:
        return enum_type(value)
    except ValueError:
        _fail(
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            path,
            f"unsupported value {value!r}",
        )


def _validate_trial_outcome(
    status: TrialStatus,
    success: bool | None,
) -> None:
    expected = {
        TrialStatus.SUCCESS: True,
        TrialStatus.FAILURE: False,
        TrialStatus.ABORTED: None,
        TrialStatus.ERROR: None,
    }[status]
    if success is not expected:
        raise ValueError(
            f"status {status.value!r} requires success={expected!r}",
        )


def _validate_local_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if len(value.encode("utf-8")) > MAX_LIFECYCLE_TEXT_BYTES:
        raise ValueError(
            f"{field_name} exceeds {MAX_LIFECYCLE_TEXT_BYTES} UTF-8 bytes",
        )


def _validate_local_counter(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    if value > MAX_MSGPACK_UNSIGNED_INT:
        raise ValueError(
            f"{field_name} is outside the msgpack unsigned 64-bit range",
        )


def _validate_local_uint32(value: Any, field_name: str) -> None:
    _validate_local_counter(value, field_name)
    if value > (1 << 32) - 1:
        raise ValueError(f"{field_name} must be at most 2^32 - 1")


def _validate_local_sha256_digest(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    prefix = "sha256:"
    hexadecimal = value[len(prefix) :] if value.startswith(prefix) else ""
    if len(hexadecimal) != 64 or any(character not in "0123456789abcdef" for character in hexadecimal):
        raise ValueError(
            f"{field_name} must be 'sha256:' followed by 64 lowercase hexadecimal characters",
        )


def _validate_local_checkpoint_id(value: Any) -> None:
    _validate_local_text(value, "checkpoint_id")
    if _looks_like_local_path(value):
        raise ValueError(
            "checkpoint_id must be a portable ID or non-file URI, not a local path",
        )


def _looks_like_local_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    lowered = normalized.lower()
    if (
        normalized.startswith(("/", "~/"))
        or lowered.startswith("file:")
        or (len(normalized) >= 2 and normalized[0].isalpha() and normalized[1] == ":")
    ):
        return True
    return any(part in {".", ".."} for part in normalized.split("/"))


def _validate_local_git_revision(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(
            f"{field_name} must be a full 40-character lowercase Git commit SHA",
        )


def _validate_local_optional_counter(
    value: Any,
    field_name: str,
) -> None:
    if value is not None:
        _validate_local_counter(value, field_name)


def _transform_error_detail(
    value: Any,
    path: tuple[str, ...],
    *,
    freeze: bool,
    payload_errors: bool,
    depth: int = 0,
) -> Any:
    def reject(kind: PayloadErrorKind, reason: str) -> None:
        if payload_errors:
            _fail(kind, path, reason)
        if kind == PayloadErrorKind.INVALID_TYPE:
            raise TypeError(f"{_render_path(path)}: {reason}")
        raise ValueError(f"{_render_path(path)}: {reason}")

    if depth > MAX_ERROR_DETAILS_DEPTH:
        reject(
            PayloadErrorKind.OUT_OF_RANGE,
            f"details nesting exceeds {MAX_ERROR_DETAILS_DEPTH}",
        )
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not -(1 << 63) <= value <= (1 << 64) - 1:
            reject(
                PayloadErrorKind.OUT_OF_RANGE,
                "integer is outside the msgpack 64-bit range",
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            reject(PayloadErrorKind.NON_FINITE, "float must be finite")
        return value
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_LIFECYCLE_TEXT_BYTES:
            reject(
                PayloadErrorKind.OUT_OF_RANGE,
                f"string exceeds {MAX_LIFECYCLE_TEXT_BYTES} UTF-8 bytes",
            )
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_ERROR_DETAILS_ITEMS:
            reject(
                PayloadErrorKind.OUT_OF_RANGE,
                f"map exceeds {MAX_ERROR_DETAILS_ITEMS} items",
            )
        transformed: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                reject(
                    PayloadErrorKind.INVALID_TYPE,
                    "map keys must be strings",
                )
            transformed[key] = _transform_error_detail(
                item,
                (*path, key),
                freeze=freeze,
                payload_errors=payload_errors,
                depth=depth + 1,
            )
        if freeze:
            return MappingProxyType(transformed)
        return transformed
    if isinstance(value, list):
        if len(value) > MAX_ERROR_DETAILS_ITEMS:
            reject(
                PayloadErrorKind.OUT_OF_RANGE,
                f"list exceeds {MAX_ERROR_DETAILS_ITEMS} items",
            )
        transformed_items = [
            _transform_error_detail(
                item,
                (*path, str(index)),
                freeze=freeze,
                payload_errors=payload_errors,
                depth=depth + 1,
            )
            for index, item in enumerate(value)
        ]
        if freeze:
            return tuple(transformed_items)
        return transformed_items
    reject(
        PayloadErrorKind.INVALID_TYPE,
        f"unsupported details value type {type(value).__name__}",
    )


def _thaw_error_detail(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_error_detail(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_error_detail(item) for item in value]
    return value


def _render_path(path: tuple[str, ...]) -> str:
    return "$" + "".join(f".{part}" for part in path)


def _require_profile_instance(profile: Any) -> None:
    if not isinstance(profile, PolicyExecutionProfile):
        raise TypeError("profile must be PolicyExecutionProfile")


def _fail(
    kind: PayloadErrorKind,
    path: tuple[str, ...],
    reason: str,
) -> None:
    raise PayloadValidationError(kind, path, reason)
