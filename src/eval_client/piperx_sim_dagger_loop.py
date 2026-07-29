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
            print(f"\n[PiPER-X DAgger] Ignoring {request}: no frames have been staged yet.")
            return None
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
    model_client: Any,
    *,
    bridge: PiperXBridgeClient | None = None,
    controller: ArxPiperXRetargetController | None = None,
    recorder: Any | None = None,
    pace_realtime: bool = True,
) -> str | None:
    """Run one operator-labelled DAgger episode.

    Policy actions drive the ARX X5 simulation.  Before every action, the
    latest accepted simulator pose is exchanged with the local hardware
    bridge so the two PiPER-X followers mirror it.  A bridge-side ``I`` edge
    immediately invalidates the current policy chunk.  While intervention is
    active, synchronized leader samples are retargeted into the simulator;
    the followers receive only the simulator's accepted result on the next
    exchange.
    """

    if task_env.num_envs != 1:
        raise ValueError("PiPER-X simulator DAgger supports exactly one environment")
    if getattr(task_env, "eval_batch", False):
        raise ValueError("PiPER-X simulator DAgger does not support batched policy evaluation")

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
            controller = ArxPiperXRetargetController(
                task_env,
                RetargetConfig.from_environment(),
            )
        if recorder is None:
            from src.eval_client.lerobot_stream_recorder import recorder_for_env

            recorder = recorder_for_env(task_env)
        pacer = RealtimePacer(
            frequency=float(task_env.obs_manager.collect_freq),
            enabled=pace_realtime and os.environ.get("ROBODOJO_REALTIME", "1") != "0",
        )
        obs = task_env.get_obs()
        initial_sim = sim_targets_from_env(task_env, obs)
        bridge.begin_episode(
            _episode_id(task_env),
            initial_sim,
        )
        _send_observation(task_env, model_client, obs)
        print(
            "[PiPER-X DAgger] policy -> simulation -> followers; "
            "I toggles leader intervention; Right/Left/Esc/Backspace label the episode."
        )
        print(f"[PiPER-X DAgger] Recording under {recorder.record_dir}")
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
    pending_release_edge = 0
    consecutive_manual_failures = 0

    def exchange() -> tuple[OperatorSample, Any]:
        sim = sim_targets_from_env(task_env, obs)
        return bridge.exchange(sim), sim

    def execute(
        action: dict[str, Any],
        *,
        policy_action: dict[str, Any] | None,
        human_action: dict[str, Any] | None,
        control: dict[str, Any],
        sample: OperatorSample,
    ) -> None:
        nonlocal obs
        metadata = dict(control)
        metadata.setdefault("timestamp", time.monotonic())
        metadata.update(
            {
                "bridge_generation": sample.generation,
                "bridge_seq": sample.seq,
                "bridge_mode": sample.mode,
            }
        )
        recorder.append(
            obs=obs,
            policy_action=policy_action,
            human_action=human_action,
            executed_action=action,
            control=metadata,
        )
        task_env.take_action(action)
        pacer.wait()
        obs = task_env.get_obs()
        _send_observation(task_env, model_client, obs)

    def process_terminal(sample: OperatorSample) -> bool:
        nonlocal accepted, finish_reason, terminal_request
        request = _terminal_request(sample, recorder.frame_count)
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
                "\n[PiPER-X DAgger] retarget/IK rejected; simulation remains frozen. "
                f"detail={control.get('retarget_failures', {})}"
            )
        if consecutive_manual_failures >= max_manual_failures:
            raise PiperXBridgeSafetyError(
                f"retarget/IK failed {consecutive_manual_failures} consecutive samples; "
                "holding followers and aborting the collection session"
            )
        pacer.wait()
        render = getattr(task_env, "render", None)
        if callable(render):
            render()

    try:
        intervention_active = False
        while terminal_request is None:
            if intervention_active:
                sample, sim = exchange()
                if process_terminal(sample):
                    break
                if sample.mode == "policy":
                    controller.exit()
                    intervention_active = False
                    pending_release_edge = -1
                    print(
                        "\n[PiPER-X DAgger] manual control OFF; "
                        "discarding stale policy state and requesting a fresh chunk."
                    )
                    continue
                action, control = controller.build_action(obs, sample)
                control.update(
                    {
                        "takeover_edge": 1 if sample.edge == "enter" else 0,
                        "chunk_id": chunk_id,
                        "chunk_index": -1,
                    }
                )
                if not control["ik_success"]:
                    reject_unsafe_manual(control)
                    continue
                consecutive_manual_failures = 0
                execute(
                    action,
                    policy_action=None,
                    human_action=action if control["intervention_mask"] else None,
                    control=control,
                    sample=sample,
                )
                continue

            # Check bridge state immediately before inference.  The heartbeat
            # continues while the synchronous model call is blocked; any edge
            # it observes is surfaced by the post-inference exchange below.
            sample, sim = exchange()
            if process_terminal(sample):
                break
            if sample.mode == "intervention":
                controller.enter(sample, sim)
                intervention_active = True
                action, control = controller.build_action(obs, sample)
                control.update(
                    {
                        "takeover_edge": 1,
                        "chunk_id": chunk_id,
                        "chunk_index": -1,
                    }
                )
                if not control["ik_success"]:
                    reject_unsafe_manual(control)
                    continue
                consecutive_manual_failures = 0
                execute(
                    action,
                    policy_action=None,
                    human_action=action if control["intervention_mask"] else None,
                    control=control,
                    sample=sample,
                )
                print("\n[PiPER-X DAgger] manual control ON; policy chunk preempted.")
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
                sample, sim = exchange()
                if process_terminal(sample):
                    stale_chunk = True
                    break
                if sample.mode == "intervention":
                    controller.enter(sample, sim)
                    intervention_active = True
                    manual_action, control = controller.build_action(obs, sample)
                    control.update(
                        {
                            "takeover_edge": 1,
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_index,
                        }
                    )
                    if not control["ik_success"]:
                        reject_unsafe_manual(control)
                        stale_chunk = True
                        break
                    consecutive_manual_failures = 0
                    execute(
                        manual_action,
                        policy_action=policy_action,
                        human_action=(manual_action if control["intervention_mask"] else None),
                        control=control,
                        sample=sample,
                    )
                    stale_chunk = True
                    print(
                        f"\n[PiPER-X DAgger] manual control ON at chunk={chunk_id} "
                        f"index={chunk_index}; discarded "
                        f"{len(chunk) - chunk_index} stale target(s)."
                    )
                    break
                if sample.edge == "exit":
                    # An enter+exit may have happened during slow inference.
                    # Even with no manual simulator step, that generation
                    # change invalidates the just-returned chunk.
                    controller.exit()
                    pending_release_edge = -1
                    stale_chunk = True
                    print(
                        "\n[PiPER-X DAgger] operator generation changed during inference; "
                        "discarding the returned policy chunk."
                    )
                    break
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
        if not recorder_finalized:
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
        controller.exit()
