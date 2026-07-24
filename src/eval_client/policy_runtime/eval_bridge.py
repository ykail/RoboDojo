"""RoboDojo EvalEnv bridge for the strict policy-v1 client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeVar

from src.eval_client.policy_runtime.action_bridge import iter_arx_x5_eval_actions
from src.eval_client.policy_runtime.canonical import CanonicalActionChunk
from src.eval_client.policy_runtime.client import PolicyClient, PolicyClientError
from src.eval_client.policy_runtime.execution_profile import ARX_X5_SIM_PI05_PROFILE
from src.eval_client.policy_runtime.lifecycle_payloads import (
    MAX_LIFECYCLE_TEXT_BYTES,
    PolicyProvenance,
    ResetPayload,
    TrialEndPayload,
    TrialStatus,
)
from src.eval_client.policy_runtime.observation_builder import ArxX5ObservationBuilder


class PolicyV1BridgeStateError(RuntimeError):
    """EvalEnv called the policy bridge out of lifecycle order."""


class _PolicyClientLike(Protocol):
    def connect(self) -> PolicyProvenance: ...

    def reset(self, episode_id: str, payload: ResetPayload) -> None: ...

    def infer(self, observation: Any) -> CanonicalActionChunk: ...

    def trial_end(self, payload: TrialEndPayload) -> None: ...

    def close(self) -> None: ...


class _PolicyEvalBridgeLike(Protocol):
    def finish_episode(self, payload: TrialEndPayload) -> None: ...


PolicyClientFactory = Callable[..., _PolicyClientLike]
_ResultT = TypeVar("_ResultT")


def task_trial_end(success: bool) -> TrialEndPayload:
    """Build the one authoritative outcome for a completed task rollout."""

    if not isinstance(success, bool):
        raise TypeError("success must be a bool")
    return TrialEndPayload(
        status=TrialStatus.SUCCESS if success else TrialStatus.FAILURE,
        success=success,
        score=1.0 if success else 0.0,
        reason="task_success" if success else "task_failure",
    )


def interrupted_trial_end(
    error: BaseException,
    *,
    operator_boundary: bool,
) -> TrialEndPayload:
    """Map an interrupted rollout without pretending it was task success/failure."""

    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    message = str(error).strip()
    reason = type(error).__name__
    if message:
        reason = f"{reason}: {message}"
    encoded_reason = reason.encode("utf-8")
    if len(encoded_reason) > MAX_LIFECYCLE_TEXT_BYTES:
        suffix = "…".encode()
        reason = (
            encoded_reason[: MAX_LIFECYCLE_TEXT_BYTES - len(suffix)].decode("utf-8", errors="ignore") + suffix.decode()
        )
    return TrialEndPayload(
        status=TrialStatus.ABORTED if operator_boundary else TrialStatus.ERROR,
        success=None,
        score=None,
        reason=reason,
    )


def operator_trial_end(reason: str) -> TrialEndPayload:
    """Build an aborted outcome for a normal operator-controlled boundary."""

    return TrialEndPayload(
        status=TrialStatus.ABORTED,
        success=None,
        score=None,
        reason=reason,
    )


def run_policy_v1_lifecycle(
    bridge: _PolicyEvalBridgeLike,
    rollout: Callable[[], _ResultT],
    *,
    normal_outcome: Callable[[], TrialEndPayload],
    is_operator_boundary: Callable[[BaseException], bool],
) -> _ResultT:
    """Run one active episode and terminate it exactly once when possible."""

    for value, name in (
        (rollout, "rollout"),
        (normal_outcome, "normal_outcome"),
        (is_operator_boundary, "is_operator_boundary"),
    ):
        if not callable(value):
            raise TypeError(f"{name} must be callable")

    try:
        result = rollout()
    except BaseException as error:
        # An ambiguous/lost request has already fenced the transport. Sending
        # a fabricated TRIAL_END would be a replay against unknown state.
        if isinstance(error, PolicyClientError) and error.episode_lost:
            raise
        payload = interrupted_trial_end(
            error,
            operator_boundary=bool(is_operator_boundary(error)),
        )
        try:
            bridge.finish_episode(payload)
        except BaseException as finish_error:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(
                    f"policy-v1 TRIAL_END also failed: {type(finish_error).__name__}: {finish_error}",
                )
        raise

    bridge.finish_episode(normal_outcome())
    return result


class PolicyV1EvalBridge:
    """Adapt current EvalEnv calls to one strict, single-env policy session.

    ``call()`` is intentionally a narrow observation/inference compatibility
    surface for the existing keyboard loops. RESET and TRIAL_END stay explicit
    so no legacy call can accidentally create a second episode boundary.
    """

    def __init__(
        self,
        url: str,
        *,
        expected_env_idx: int = 0,
        connect_timeout_s: float = 30.0,
        request_timeout_s: float = 600.0,
        close_timeout_s: float = 10.0,
        client_factory: PolicyClientFactory = PolicyClient,
    ) -> None:
        if not callable(client_factory):
            raise TypeError("client_factory must be callable")
        self._builder = ArxX5ObservationBuilder(
            spec=ARX_X5_SIM_PI05_PROFILE.observation_spec,
            expected_env_idx=expected_env_idx,
        )
        self._expected_env_idx = expected_env_idx
        self._client = client_factory(
            url,
            profile=ARX_X5_SIM_PI05_PROFILE,
            connect_timeout_s=connect_timeout_s,
            request_timeout_s=request_timeout_s,
            close_timeout_s=close_timeout_s,
        )
        self._episode_active = False
        self._pending_observation = None
        self._closed = False
        try:
            self._provenance = self._client.connect()
        except BaseException:
            self._closed = True
            try:
                self._client.close()
            except BaseException:
                pass
            raise

    @property
    def provenance(self) -> PolicyProvenance:
        return self._provenance

    @property
    def episode_active(self) -> bool:
        return self._episode_active

    def start_episode(
        self,
        episode_id: str,
        payload: ResetPayload,
    ) -> None:
        self._require_open()
        if self._episode_active:
            raise PolicyV1BridgeStateError("cannot start a second active episode")
        self._client.reset(episode_id, payload)
        self._pending_observation = None
        self._episode_active = True

    def finish_episode(self, payload: TrialEndPayload) -> None:
        self._require_open()
        if not self._episode_active:
            raise PolicyV1BridgeStateError("cannot finish without an active episode")
        try:
            self._client.trial_end(payload)
        finally:
            self._pending_observation = None
        self._episode_active = False

    def call(self, func_name: str, **kwargs: Any) -> Any:
        """Serve only the legacy calls needed by current EvalEnv loops."""

        self._require_open()
        if func_name == "update_obs":
            self._require_exact_kwargs(kwargs, {"obs"}, func_name)
            self._stage_observation(kwargs["obs"])
            return None
        if func_name == "get_action":
            self._require_exact_kwargs(kwargs, set(), func_name)
            return self._infer_action_chunk()
        if func_name == "update_obs_batch":
            self._require_exact_kwargs(kwargs, {"obs"}, func_name)
            observations = kwargs["obs"]
            if not isinstance(observations, list | tuple) or len(observations) != 1:
                raise ValueError("policy-v1 supports exactly one observation per batch call")
            self._stage_observation(observations[0])
            return None
        if func_name == "get_action_batch":
            self._require_exact_kwargs(kwargs, {"obs"}, func_name)
            env_indices = kwargs["obs"]
            if list(env_indices) != [self._expected_env_idx]:
                raise ValueError(
                    "policy-v1 batch compatibility requires exactly the configured environment",
                )
            return [self._infer_action_chunk()]
        raise NotImplementedError(f"unsupported policy-v1 EvalEnv call: {func_name}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending_observation = None
        self._client.close()

    def _stage_observation(self, raw_observation: Mapping[str, Any]) -> None:
        if not self._episode_active:
            raise PolicyV1BridgeStateError("observation update requires an active episode")
        self._pending_observation = self._builder.build(raw_observation)

    def _infer_action_chunk(self) -> list[dict[str, Any]]:
        if not self._episode_active:
            raise PolicyV1BridgeStateError("inference requires an active episode")
        observation = self._pending_observation
        if observation is None:
            raise PolicyV1BridgeStateError(
                "inference requires one fresh observation update",
            )
        self._pending_observation = None
        chunk = self._client.infer(observation)
        return list(iter_arx_x5_eval_actions(chunk))

    def _require_open(self) -> None:
        if self._closed:
            raise PolicyV1BridgeStateError("policy-v1 bridge is closed")

    @staticmethod
    def _require_exact_kwargs(
        kwargs: Mapping[str, Any],
        expected: set[str],
        operation: str,
    ) -> None:
        if set(kwargs) != expected:
            raise TypeError(
                f"{operation} requires keyword arguments {sorted(expected)}, got {sorted(kwargs)}",
            )
