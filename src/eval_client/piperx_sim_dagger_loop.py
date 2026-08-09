"""Pi0.5 rollout with physical PiPER-X mirroring and leader intervention."""

from __future__ import annotations

import os
import time
from typing import Any
import uuid

from src.eval_client.intervention_loop import (
    InterventionAcceptedAndExit,
    InterventionDiscardedAndExit,
    InterventionRejected,
    InterventionSavedForRetry,
    RealtimePacer,
    _get_action_chunk,
    _mark_operator_end,
    _send_observation,
)
from src.eval_client.piperx_bridge_client import (
    OperatorSample,
    PiperXBridgeClient,
    PiperXBridgeError,
    PiperXBridgeSafetyError,
    shared_client_from_environment,
)
from src.eval_client.piperx_retarget import (
    ArxPiperXRetargetController,
    RetargetConfig,
    sim_targets_from_env,
)


def _terminal_request(sample: OperatorSample, frame_count: int) -> str | None:
    request = sample.terminal_request
    if request in {"accept_next", "discard_retry", "accept_exit", "discard_exit", None}:
        if request == "accept_next" and frame_count == 0:
            # The bridge has already delivered and consumed this terminal key;
            # silently ignoring it would make the next request illegal. End the
            # empty candidate cleanly and retry the layout instead.
            print("\n[PiPER-X DAgger] Empty accept-next candidate; discarding and retrying the layout.")
            return "discard_retry"
        return request
    raise ValueError(f"Unknown PiPER-X terminal request: {request!r}")


def _episode_id(task_env: Any) -> str:
    seeds = getattr(task_env, "env_seeds", [])
    layout_id = seeds[0] if seeds else -1
    return (
        f"{getattr(task_env, 'run_id', 'robodojo')}-"
        f"layout-{layout_id}-cycle-{getattr(task_env, 'layout_cycle', 0)}-"
        f"{uuid.uuid4().hex[:12]}"
    )


def run_piperx_sim_dagger_episode(
    task_env: Any,
    model_client: Any | None,
    *,
    bridge: PiperXBridgeClient | None = None,
    controller: ArxPiperXRetargetController | None = None,
    recorder: Any | None = None,
    pace_realtime: bool = True,
    manual_only: bool = False,
) -> str | None:
    """Run one PiPER-X episode, with an optional policy-free manual mode.

    Policy actions drive the ARX X5 simulation. Before every action, the
    latest accepted simulator joint state is exchanged with the local hardware
    bridge until the two PiPER-X followers reach that relative joint target;
    the leaders then follow fresh
    follower deltas around runtime-relative anchors. A bridge-side ``I`` edge
    immediately invalidates the current policy chunk. During the physical mode
    transition the simulator is frozen and no frame is recorded. While
    intervention is active,
    each synchronized leader sample is first checked against the ARX joint target, then explicitly
    committed as a calibration-free relative joint delta to the matching
    PiPER-X follower. The simulator executes only after that exact physical
    sample has been acknowledged. In ``manual_only`` mode no policy client or
    recorder is created: policy mode is a stationary bridge hold until ``I``
    enters intervention, and the second ``I`` returns to that hold.
    """

    if task_env.num_envs != 1:
        raise ValueError("PiPER-X simulator DAgger supports exactly one environment")
    if getattr(task_env, "eval_batch", False):
        raise ValueError("PiPER-X simulator DAgger does not support batched policy evaluation")
    if manual_only and model_client is not None:
        raise ValueError("PiPER-X manual-only mode must not receive a policy client")
    if manual_only and recorder is not None:
        raise ValueError("PiPER-X manual-only mode does not record a LeRobot episode")
    record_enabled = (
        recorder is not None
        or os.environ.get("ROBODOJO_PIPERX_RECORD", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )

    bridge = bridge or shared_client_from_environment()
    try:
        max_manual_failures = int(os.environ.get("ROBODOJO_PIPERX_MAX_CONSECUTIVE_IK_FAILURES", "25"))
    except ValueError as exc:
        raise PiperXBridgeSafetyError("ROBODOJO_PIPERX_MAX_CONSECUTIVE_IK_FAILURES must be a positive integer") from exc
    if max_manual_failures <= 0:
        raise PiperXBridgeSafetyError("ROBODOJO_PIPERX_MAX_CONSECUTIVE_IK_FAILURES must be positive")
    recorder_finalized = False
    bridge_ended = False
    accepted = False
    terminal_request: str | None = None
    finish_reason = "operator_pending"
    saved_path: str | None = None

    try:
        if controller is None:
            controller = ArxPiperXRetargetController(task_env, RetargetConfig())
        if recorder is None and not manual_only and record_enabled:
            from src.eval_client.lerobot_stream_recorder import recorder_for_env

            recorder = recorder_for_env(task_env)
        pacer = RealtimePacer(
            frequency=float(task_env.obs_manager.collect_freq),
            enabled=pace_realtime and os.environ.get("ROBODOJO_REALTIME", "1") != "0",
        )
        obs = task_env.get_obs()
        initial_sim = sim_targets_from_env(task_env, obs)
        # Finish all policy-side staging before authorizing the separate
        # hardware owner to leave SAFE_IDLE and arm the four physical devices.
        if not manual_only:
            _send_observation(task_env, model_client, obs)
        bridge.begin_episode(
            _episode_id(task_env),
            initial_sim,
        )
        if manual_only:
            print(
                "[PiPER-X Manual] stationary hold; I toggles leader intervention; "
                "Esc/Backspace exits. No policy client or dataset recorder is active."
            )
        else:
            print(
                "[PiPER-X DAgger] policy -> simulation -> followers -> leaders; "
                "I toggles leader intervention; Right/Left/Esc/Backspace label the episode."
            )
            if recorder is not None:
                print(f"[PiPER-X DAgger] Recording under {recorder.record_dir}")
            else:
                print("[PiPER-X DAgger] Checkpoint mode: LeRobot recording disabled.")
    except Exception as exc:
        if recorder is not None:
            try:
                recorder.finalize(accepted=False, success=False, reason="setup_exception")
            except Exception:
                pass
        bridge.fail_closed("setup_exception")
        if isinstance(exc, PiperXBridgeError):
            raise
        raise PiperXBridgeSafetyError(
            f"PiPER-X DAgger setup failed; the physical session cannot be retried: {exc}"
        ) from exc

    chunk_id = -1
    pending_takeover_edge = 0
    pending_release_edge = 0
    consecutive_manual_failures = 0
    reported_transition: str | None = None
    manual_frame_count = 0

    def policy_exchange() -> tuple[OperatorSample, Any]:
        sim = sim_targets_from_env(task_env, obs)
        if manual_only:
            # A policy-free wait must not call exchange: exchange is a motion
            # command even when the simulator state appears unchanged.
            return bridge.poll(), sim
        while True:
            sample = bridge.exchange(sim)
            if sample.transition is not None or sample.terminal_request is not None:
                return sample, sim
            operation = sample.diagnostics.get("operation")
            if not isinstance(operation, dict) or "mirror_complete" not in operation:
                raise PiperXBridgeSafetyError(
                    "direct-joint bridge exchange omitted mirror_complete"
                )
            if operation["mirror_complete"] is True:
                return sample, sim
            if operation["mirror_complete"] is not False:
                raise PiperXBridgeSafetyError("mirror_complete must be boolean")
            pacer.wait()

    def execute(
        action: dict[str, Any],
        *,
        policy_action: dict[str, Any] | None,
        human_action: dict[str, Any] | None,
        control: dict[str, Any],
        sample: OperatorSample,
    ) -> None:
        nonlocal obs, manual_frame_count
        metadata = dict(control)
        metadata.setdefault("timestamp", time.monotonic())
        metadata.update(
            {
                "bridge_generation": sample.generation,
                "bridge_seq": sample.seq,
                "bridge_mode": sample.mode,
                "leader_actuation_mode": sample.leader_actuation_mode,
                "follower_actuation_mode": sample.follower_actuation_mode,
            }
        )
        if recorder is not None:
            recorder.append(
                obs=obs,
                policy_action=policy_action,
                human_action=human_action,
                executed_action=action,
                control=metadata,
            )
        task_env.take_action(action)
        manual_frame_count += 1
        pacer.wait()
        obs = task_env.get_obs()
        if not manual_only:
            _send_observation(task_env, model_client, obs)

    def process_terminal(sample: OperatorSample) -> bool:
        nonlocal accepted, finish_reason, terminal_request
        frame_count = recorder.frame_count if recorder is not None else manual_frame_count
        request = _terminal_request(sample, frame_count)
        if request is None:
            return False
        terminal_request = request
        accepted = request in {"accept_next", "accept_exit"}
        finish_reason = f"operator_{request}"
        _mark_operator_end(task_env, rejected=not accepted)
        return True

    def reject_unsafe_manual(control: dict[str, Any]) -> None:
        """Freeze both sides without creating a non-expert training label."""

        nonlocal consecutive_manual_failures
        consecutive_manual_failures += 1
        if consecutive_manual_failures == 1:
            print(
                "\n[PiPER-X DAgger] direct-joint target rejected; simulation remains frozen. "
                f"detail={control.get('retarget_failures', {})}"
            )
        if consecutive_manual_failures >= max_manual_failures:
            raise PiperXBridgeSafetyError(
                f"direct-joint mapping failed {consecutive_manual_failures} consecutive samples; "
                "holding followers and aborting the collection session"
            )
        pacer.wait()
        render = getattr(task_env, "render", None)
        if callable(render):
            render()

    def freeze_transition(sample: OperatorSample) -> None:
        """Keep Isaac and the recorder idle while the hardware changes authority."""

        nonlocal reported_transition
        if sample.transition != reported_transition:
            print(
                "\n[PiPER-X DAgger] hardware transition in progress: "
                f"{sample.transition}; simulation frozen and no frame recorded."
            )
            reported_transition = sample.transition
        pacer.wait()
        render = getattr(task_env, "render", None)
        if callable(render):
            render()

    intervention_active = False

    def complete_transition(sample: OperatorSample, sim: Any) -> None:
        """Acknowledge a hardware hold before either control authority changes."""

        nonlocal intervention_active, pending_takeover_edge, pending_release_edge, reported_transition
        nonlocal consecutive_manual_failures
        freeze_transition(sample)
        completed = bridge.transition_ack(sim)
        reported_transition = None
        if completed.edge == "enter":
            # The leaders are native now, but the operator is still instructed
            # not to move. Capture one exact post-switch sample and use it as
            # both the ARX Cartesian zero and the PiPER-X qL/qF zero. Only then
            # announce manual authority; this avoids any drift between the
            # physical role switch and the later RoboDojo anchor.
            anchor = bridge.manual_sample()
            if anchor.transition is not None:
                complete_transition(anchor, sim)
                return
            if process_terminal(anchor):
                return
            if anchor.manual_sample is None:
                raise PiperXBridgeSafetyError("entry anchor omitted the exact manual sample")
            controller.enter(anchor, sim)
            resolution = bridge.anchor_manual_sample(anchor.manual_sample.sample_id)
            if resolution.transition is not None:
                controller.exit()
                complete_transition(resolution, sim)
                return
            result = resolution.manual_resolution
            operation = resolution.diagnostics.get("operation")
            if (
                result is None
                or result.sample_id != anchor.manual_sample.sample_id
                or result.decision != "anchor"
                or result.follower_commanded
                or not isinstance(operation, dict)
                or operation.get("manual_anchor_latched") is not True
            ):
                raise PiperXBridgeSafetyError(
                    "entry sample did not latch the same physical and simulator manual anchor"
                )
            intervention_active = True
            pending_takeover_edge = 1
            consecutive_manual_failures = 0
            print(
                "\n[PiPER-X Manual] manual control ON; leader sample fan-out active."
                if manual_only
                else "\n[PiPER-X DAgger] manual control ON; leader sample fan-out active; policy chunk preempted."
            )
            return
        if completed.edge == "exit":
            controller.exit()
            intervention_active = False
            pending_release_edge = -1
            consecutive_manual_failures = 0
            print(
                "\n[PiPER-X Manual] manual control OFF; returned to stationary hold."
                if manual_only
                else "\n[PiPER-X DAgger] manual control OFF; policy mapping re-anchored; requesting a fresh chunk."
            )
            process_terminal(completed)
            return
        raise PiperXBridgeSafetyError("transition acknowledgement returned no enter/exit edge")

    try:
        while terminal_request is None:
            if intervention_active:
                sim = sim_targets_from_env(task_env, obs)
                sample = bridge.manual_sample()
                if sample.transition is not None:
                    complete_transition(sample, sim)
                    continue
                if process_terminal(sample):
                    break
                reported_transition = None
                if sample.mode != "intervention" or sample.manual_sample is None:
                    raise PiperXBridgeSafetyError("manual_sample returned outside intervention mode")
                action, control = controller.build_action(obs, sample)
                control.update(
                    {
                        "takeover_edge": pending_takeover_edge,
                        "chunk_id": chunk_id,
                        "chunk_index": -1,
                        "manual_sample_id": sample.manual_sample.sample_id,
                    }
                )
                if not control["ik_success"]:
                    resolution = bridge.manual_resolve(sample.manual_sample.sample_id, commit=False)
                    if resolution.transition is not None:
                        complete_transition(resolution, sim)
                        continue
                    result = resolution.manual_resolution
                    if (
                        result is None
                        or result.sample_id != sample.manual_sample.sample_id
                        or result.decision != "reject"
                        or result.follower_commanded
                    ):
                        raise PiperXBridgeSafetyError(
                            "rejected ARX joint target did not produce a confirmed follower hold"
                        )
                    reject_unsafe_manual(control)
                    continue
                resolution = bridge.manual_resolve(sample.manual_sample.sample_id, commit=True)
                if resolution.transition is not None:
                    complete_transition(resolution, sim)
                    continue
                result = resolution.manual_resolution
                if (
                    result is None
                    or result.sample_id != sample.manual_sample.sample_id
                    or result.decision != "commit"
                    or not result.follower_commanded
                ):
                    raise PiperXBridgeSafetyError("manual sample was not committed to both followers")
                consecutive_manual_failures = 0
                execute(
                    action,
                    policy_action=None,
                    human_action=action if control["intervention_mask"] else None,
                    control=control,
                    sample=resolution,
                )
                pending_takeover_edge = 0
                continue

            # Check bridge state immediately before inference.  The heartbeat
            # continues while the synchronous model call is blocked; any edge
            # it observes is surfaced by the post-inference exchange below.
            sample, sim = policy_exchange()
            if sample.transition is not None:
                complete_transition(sample, sim)
                continue
            if process_terminal(sample):
                break
            reported_transition = None
            if sample.mode != "policy":
                raise PiperXBridgeSafetyError("policy exchange returned outside policy mode")

            if manual_only:
                pacer.wait()
                render = getattr(task_env, "render", None)
                if callable(render):
                    render()
                continue

            # Policy-v1 consumes its staged observation when inference starts.
            # Re-stage unconditionally: an I enter+exit can invalidate a slow
            # inference without executing a simulator step, so no execute()
            # path would otherwise provide the required fresh observation.
            _send_observation(task_env, model_client, obs)
            chunk = _get_action_chunk(task_env, model_client)
            chunk_id += 1
            stale_chunk = False
            for chunk_index, policy_action in enumerate(chunk):
                sample, sim = policy_exchange()
                if sample.transition is not None:
                    complete_transition(sample, sim)
                    stale_chunk = True
                    print(
                        f"[PiPER-X DAgger] discarded {len(chunk) - chunk_index} "
                        "policy target(s) at the hardware transition boundary."
                    )
                    break
                if process_terminal(sample):
                    stale_chunk = True
                    break
                reported_transition = None
                if sample.mode != "policy":
                    raise PiperXBridgeSafetyError("policy exchange returned outside policy mode")
                execute(
                    policy_action,
                    policy_action=policy_action,
                    human_action=None,
                    control={
                        "action_source": "policy",
                        "intervention_mask": 0,
                        "ik_success": 1,
                        "active_arm": "both",
                        "takeover_edge": pending_release_edge,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                    },
                    sample=sample,
                )
                pending_release_edge = 0
            if stale_chunk:
                continue

        assert terminal_request is not None
        # end_episode is the hardware-side hold boundary.  Perform it before
        # potentially slow video finalization; failure discards the candidate.
        bridge.end_episode(
            reason=finish_reason,
        )
        bridge_ended = True
        if recorder is not None:
            saved_path = recorder.finalize(
                accepted=accepted,
                success=bool(task_env.success[0]),
                reason=finish_reason,
            )
            recorder_finalized = True
            if saved_path:
                print(f"\n[PiPER-X DAgger] Saved trajectory: {saved_path}")
            elif accepted and terminal_request != "accept_exit":
                raise RuntimeError("Recorder returned no path for an accepted PiPER-X episode")
            elif not accepted:
                print("\n[PiPER-X DAgger] Episode discarded; candidate data were not kept.")
        elif manual_only:
            recorder_finalized = True
            print("\n[PiPER-X Manual] Manual-only session ended; no dataset was written.")
        else:
            recorder_finalized = True
            print("\n[PiPER-X DAgger] Checkpoint ended; recording was disabled.")

        if terminal_request == "discard_retry":
            raise InterventionRejected("Operator rejected the PiPER-X DAgger episode")
        if terminal_request == "discard_exit":
            raise InterventionDiscardedAndExit("Operator discarded the final PiPER-X episode")
        if terminal_request == "accept_exit":
            raise InterventionAcceptedAndExit(saved_path)
        return saved_path
    except (
        InterventionRejected,
        InterventionSavedForRetry,
        InterventionAcceptedAndExit,
        InterventionDiscardedAndExit,
    ):
        raise
    except Exception as exc:
        if not bridge_ended:
            bridge.fail_closed("episode_exception")
        if not recorder_finalized and recorder is not None:
            try:
                recorder.finalize(accepted=False, success=False, reason="exception")
            except Exception as finalize_error:
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(
                        f"LeRobot candidate discard also failed: {type(finalize_error).__name__}: {finalize_error}"
                    )
        if isinstance(exc, PiperXBridgeError):
            raise
        raise PiperXBridgeSafetyError(
            f"PiPER-X DAgger episode failed; the physical session cannot be retried: {exc}"
        ) from exc
    finally:
        if controller is not None:
            controller.exit()
