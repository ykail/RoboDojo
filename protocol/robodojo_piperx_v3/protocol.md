# RoboDojo PiPER-X bridge protocol v3

Status: experimental, safety-critical, loopback-only integration contract.

Protocol identifier: `robodojo_piperx_v3`.

V3 replaces v2 because manual control is now an exact, two-phase fan-out of
one cached leader sample. A v2 peer routes manual motion through RoboDojo and
then independently solves PiPER-X IK. A v3 peer sends the same PiPER-X leader
joint delta directly to its matching PiPER-X follower and sends the leader FK
pose to RoboDojo for ARX X5 IK. V2 and v3 peers must reject each other.

## Ownership and topology

RoboDojo owns Isaac Sim, Kai0 policy arbitration, ARX X5 IK, and LeRobot-v3
recording. The LeRobot bridge exclusively owns the four PiPER-X CAN channels,
the operator keyboard, leader/follower mode changes, watchdogs, and four-arm
fail-closed behavior. Kai0 never communicates with PiPER-X directly.

The fixed topology token is:

```text
policy_sim_to_follower_to_leader_manual_leader_joint_fanout
```

The two directions are deliberately asymmetric:

```text
policy:
  observation -> Kai0 -> ARX action -> Isaac Sim
  accepted ARX end poses -> relative SE(3) -> PiPER-X IK -> followers
  fresh measured follower joints/grippers -> matching motor-driven leaders

manual:
  one cached bimanual leader joint sample
    +-> relative leader FK pose -> RoboDojo ARX IK -> Isaac Sim
    `-> relative PiPER-X joint/gripper delta -> matching followers
```

PiPER-X leader and follower arms use the same joint convention, so the manual
physical path does not need a second IK solve. ARX X5 has different kinematics,
so its path remains Cartesian and uses RoboDojo IK. Absolute base placement is
removed by takeover anchors. The only accepted embodiment profile is
`arx_x5_piperx_relative_v1`; runtime frame matrices, axis maps, hand-tuned
scales, and per-site calibration JSON are not part of v3.

## Safe startup

Starting Terminal A creates the loopback listener and keyboard reader only.
It must not connect, enable, or command any arm. The first
`arm_and_begin_episode` request, sent after RoboDojo has created Isaac Sim and
successfully staged the first policy observation, is the sole authorization
to connect and arm hardware. It has a separate long deadline because CAN
bring-up and safe leader attachment are slower than steady control requests.

## Transport and envelope

- TCP loopback only; both processes use the same Linux monotonic clock.
- One persistent connection is reused across episode resets.
- Each frame is a four-byte big-endian length followed by strict finite UTF-8
  JSON; maximum size is 1 MiB by default.
- Exactly one request is outstanding. `seq` increases across the connection.
- A malformed, late, mismatched, disconnected, or wrong-version response
  permanently loses the physical session. A request is never replayed.

Every request and response contains exactly:

| Field | Type | Meaning |
|---|---|---|
| `protocol` | string | Exactly `robodojo_piperx_v3` |
| `type` | string | Request type; response echoes it |
| `session_id` | string | Fixed for one connection |
| `episode_id` | string | Fixed between arm/begin and end |
| `generation` | integer | Increases exactly once per completed enter/exit edge |
| `seq` | integer | Connection-wide sequence; response echoes it |
| `sent_monotonic_ns` | integer | Sender timestamp |
| `deadline_monotonic_ns` | integer | Request deadline; response echoes it |
| `payload` | object | Exact body described below |

Request payloads are:

```text
arm_and_begin_episode  {"sim": SIM}
exchange               {"sim": SIM}
transition_ack         {"sim": SIM}
manual_sample          {}
manual_resolve         {"sample_id": POSITIVE_INT, "decision": "anchor"|"commit"|"reject"}
heartbeat              {}
hold                   {"reason": NON_EMPTY_STRING}
end_episode            {"reason": NON_EMPTY_STRING}
```

`SIM` is the last state accepted by Isaac, never a raw policy proposal:

```json
{
  "left":  {"pose": [0, 0, 0, 1, 0, 0, 0], "gripper": 1.0},
  "right": {"pose": [0, 0, 0, 1, 0, 0, 0], "gripper": 1.0}
}
```

Pose order is `[x,y,z,qw,qx,qy,qz]`; simulator gripper values are normalized
from zero closed to one open.

Every response payload contains exactly:

```json
{
  "mode": "policy",
  "edge": null,
  "transition": null,
  "terminal_request": null,
  "manual_sample": null,
  "manual_resolution": null,
  "motion_accepted": true,
  "embodiment_profile": "arx_x5_piperx_relative_v1",
  "control_topology": "policy_sim_to_follower_to_leader_manual_leader_joint_fanout",
  "leader_actuation_mode": "output_follow",
  "follower_actuation_mode": "sim_follow",
  "health": {"ok": true},
  "diagnostics": {"request_ok": true}
}
```

Allowed state values are:

- `mode`: `policy`, `intervention`, or sticky `fault`;
- `transition`: null, `entering_intervention`, or `reattaching_policy`;
- `edge`: null, `enter`, or `exit`;
- `terminal_request`: null, `accept_next`, `discard_retry`, `accept_exit`, or
  `discard_exit`;
- leader actuation: `output_follow`, `native_leader`, `disabled`, or `fault`;
- follower actuation: `sim_follow`, `leader_follow`, `hold`, `disabled`, or
  `fault`.

The profile, topology, and both actuation modes in `health` must exactly match
their top-level values. `health.ok` and `diagnostics.request_ok` must both be
true before RoboDojo accepts a response. Heartbeats refresh liveness but never
consume an edge, terminal request, manual sample, or manual resolution.

## Control transitions

```text
SAFE_LISTEN_ONLY
  -> ARMING_POLICY
  -> POLICY_FOLLOW
  -> ENTERING_INTERVENTION
  -> INTERVENTION_NATIVE
  -> REATTACHING_POLICY
  -> POLICY_FOLLOW

any unrecoverable failure -> sticky FAULT
```

In policy mode each follower tracks the last accepted simulator pose. Each
leader tracks fresh measured state from its matching follower. Pulling a
motor-driven leader in this mode is forbidden.

On the first local `i`, the bridge blocks new policy motion, places followers
in measured-position hold, and holds the leaders in output mode. It returns
`entering_intervention` with no edge and no accepted motion. RoboDojo freezes
Isaac and acknowledges that exact simulator state with `transition_ack`.
Only then does the bridge switch both leaders to native/backdrivable mode,
increment generation, and return `edge=enter`. RoboDojo takes one exact
post-switch sample and resolves it as `anchor`. That operation holds fresh
follower state and atomically uses the same cached leader sample as both the
physical joint zero and the simulator Cartesian zero. Manual control is not
announced before this acknowledgement.

The local key is asynchronous to the simulator process. A policy exchange
whose success response was already delivered is an authorized boundary; its
corresponding Isaac action may complete before the next request observes `i`.
This is bounded to one control tick. No subsequent policy action may execute,
and the explicit takeover anchor pairs the resulting simulator state with the
held physical state. `i` is therefore not a substitute for the hard E-stop.

On the second `i`, the bridge invalidates any unresolved sample and holds both
followers. RoboDojo again freezes Isaac and sends `transition_ack` with the
last executed manual state. The bridge anchors the policy mapping at this
state, safely reattaches both motor-driven leaders to fresh follower feedback,
increments generation, and returns `edge=exit`. RoboDojo discards the old
policy chunk and requests new inference from the post-intervention observation.

No transition frame steps Isaac or enters the training dataset.

## Exact manual transaction

Manual control uses one pending sample at a time:

1. The first `manual_sample` after `edge=enter` atomically caches fresh
   joint/gripper states from both leaders. RoboDojo resolves it with
   `manual_resolve(..., anchor)`. The bridge takes a fresh follower hold and
   latches the cached leader joints and held follower joints as the common
   physical zero. This sample commands no displacement and is not recorded.
2. Each subsequent `manual_sample` atomically caches fresh states from both
   leaders and returns one positive `sample_id`. The two FK poses in the
   response are computed from those exact cached joints. Sampling commands no
   motion.
3. RoboDojo computes relative ARX targets and solves both ARX IK problems
   without stepping Isaac.
4. If either target is unsafe, RoboDojo sends `manual_resolve(..., reject)`.
   The bridge keeps both followers at measured hold and invalidates the sample.
5. If both targets are safe, RoboDojo sends `manual_resolve(..., commit)`.
   The bridge computes, preflights, and sends both follower targets from the
   cached values:

   ```text
   q_follower_target = q_follower_anchor
                     + (q_leader_sample - q_leader_anchor)
   ```

   The same relative rule applies to the gripper, with the built-in 0.102 m
   stroke limit. Only after both follower commands are acknowledged does the
   response report `follower_commanded=true`.
6. RoboDojo executes and records the already-validated ARX action only after
   receiving that commit response.

A sample ID can be resolved exactly once. Sampling again before resolution,
resolving the wrong ID, or crossing a transition invalidates the transaction.
There is no moving rollback after a partial dual-arm command: the bridge
disables all four arms and enters sticky fault.

This ordering prevents the IK ambiguity raised for an end-pose round trip:
the follower never receives a PiPER-X IK result derived from the simulator.
It receives the original leader joint delta; RoboDojo independently receives
the matching leader FK displacement for its different ARX X5 embodiment.

## Recording and failure behavior

An accepted episode contains every executed simulator step, including policy
and manual portions. `action` is the ARX action actually executed by Isaac;
`complementary_info.policy_action` preserves the available policy proposal;
`complementary_info.is_intervention` marks accepted human steps. Transition,
rejected IK, anchor, and physical-hold samples are not recorded as expert
actions.

Deadlines, stale feedback, watchdog expiry, TTY loss, CAN error, socket loss,
invalid response, transition failure, or either-arm command failure cause
four-arm fail-closed behavior and discard the staged candidate. Software stop
cannot guarantee disable after process or bus failure, so the first HIL run
still requires an onsite operator supporting all arms with the hard E-stop.
