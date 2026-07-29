# RoboDojo PiPER-X bridge protocol v2

Status: experimental, safety-critical local integration contract.

Protocol identifier: `robodojo_piperx_v2`.

V2 replaces v1 because leader actuation has incompatible semantics. A v1
leader is always native/backdrivable. A v2 leader is motor-driven from fresh
follower feedback in policy mode and becomes native/backdrivable only after a
completed operator takeover. Peers must reject the other protocol version.

## Ownership and topology

RoboDojo owns simulator observations, Pi0.5 inference, ARX X5 action
arbitration/IK, and LeRobot-v3 recording. The LeRobot bridge exclusively owns
all four PiPER-X CAN interfaces, the physical keyboard, mode transitions,
feedback freshness, watchdogs, and four-arm fail-closed behavior. Kai0 never
talks to PiPER-X.

```text
policy:
  RoboDojo observation -> Kai0 -> ARX action -> Isaac Sim
  accepted sim pose -> PiPER-X follower IK/servo
  fresh measured follower joints/gripper -> matching PiPER-X leader servo

intervention:
  native/backdrivable leaders -> relative SE(3) -> RoboDojo ARX IK -> Isaac Sim
  accepted sim pose -> PiPER-X follower IK/servo
```

The fixed topology token is:

```text
accepted_sim_to_follower_to_leader
```

PiPER-X follower feedback may drive its matching PiPER-X leader because they
share the same native joint convention. Leader motion must never drive a
follower directly during intervention: ARX X5 and PiPER-X have different
kinematics, so the leader first goes through relative Cartesian retargeting,
RoboDojo IK, and an accepted Isaac step.

## Transport and envelope

- TCP loopback only; both processes run on the same Linux host.
- One persistent connection is reused across episode resets.
- Each frame is `uint32_be byte_length` followed by strict finite UTF-8 JSON.
- Maximum frame size is 1 MiB.
- Exactly one request is outstanding. Requests use a connection-wide,
  monotonically increasing `seq`.
- A malformed, late, mismatched, disconnected, or v1 response permanently
  loses the physical session. RoboDojo never reconnects/replays an episode.

Every envelope contains exactly:

| Field | Type | Meaning |
|---|---|---|
| `protocol` | string | Exactly `robodojo_piperx_v2` |
| `type` | string | Request type; response echoes it |
| `session_id` | string | Fixed for one TCP connection |
| `episode_id` | string | Fixed between begin/end |
| `generation` | integer | Last non-heartbeat control edge acknowledged by RoboDojo |
| `seq` | integer | Connection-wide sequence; response echoes it |
| `sent_monotonic_ns` | integer | Sender timestamp |
| `deadline_monotonic_ns` | integer | Request deadline; response echoes it |
| `payload` | object | Exact body below |

Request payloads are unchanged from v1:

```text
begin_episode  {"sim": SIM}
exchange       {"sim": SIM}
heartbeat      {}
hold           {"reason": NON_EMPTY_STRING}
end_episode    {"reason": NON_EMPTY_STRING}
```

`SIM` is the last state accepted by Isaac, not a raw policy proposal:

```json
{
  "left":  {"pose": [0, 0, 0, 1, 0, 0, 0], "gripper": 1.0},
  "right": {"pose": [0, 0, 0, 1, 0, 0, 0], "gripper": 1.0}
}
```

Pose order is `[x,y,z,qw,qx,qy,qz]`; gripper is normalized with zero closed
and one open.

Every response payload contains exactly:

```json
{
  "mode": "policy",
  "control_topology": "accepted_sim_to_follower_to_leader",
  "leader_actuation_mode": "output_follow",
  "follower_actuation_mode": "sim_follow",
  "transition": null,
  "edge": null,
  "terminal_request": null,
  "leader": {
    "left":  {"pose": [0, 0, 0, 1, 0, 0, 0], "gripper_m": 0.07},
    "right": {"pose": [0, 0, 0, 1, 0, 0, 0], "gripper_m": 0.07}
  },
  "mirror_accepted": true,
  "health": {
    "ok": true,
    "control_topology": "accepted_sim_to_follower_to_leader",
    "leader_actuation_mode": "output_follow",
    "follower_actuation_mode": "sim_follow"
  },
  "diagnostics": {"request_ok": true}
}
```

Allowed values:

- `mode`: `policy`, `intervention`, or sticky `fault`.
- `leader_actuation_mode`: `output_follow`, `native_leader`, `disabled`, or
  `fault`.
- `follower_actuation_mode`: `sim_follow`, `hold`, `disabled`, or `fault`.
- `transition`: null, `entering_intervention`, or `reattaching_policy`.
- `edge`: null, `enter`, or `exit`.
- `terminal_request`: null, `accept_next`, `discard_retry`, `accept_exit`, or
  `discard_exit`.

Outside a transition, policy requires `leader_actuation_mode=output_follow`
and intervention requires `leader_actuation_mode=native_leader`. A transition
requires `mirror_accepted=false` and null edge and terminal request. While the
initial measured hold is being acknowledged, the follower may accurately
report the blocked prior `sim_follow`; it reports `hold` once that boundary is
confirmed. No new simulator target is authorized in either case.
`entering_intervention` retains acknowledged `mode=policy`;
`reattaching_policy` retains `mode=intervention`.

The topology and both actuation-mode values in `health` must exactly match the
corresponding top-level payload fields. `health.ok` must be the JSON boolean
true and `diagnostics.request_ok` must be true for a usable response.

`transition` remains non-null after physical switching completes until a
non-heartbeat exchange consumes its edge. This prevents a heartbeat from
reporting a physical state inconsistent with its deliberately unadvanced
generation. Heartbeats never consume an edge or terminal request.

## Physical state machine

```text
DISABLED -> ARMING_POLICY -> POLICY_FOLLOW
POLICY_FOLLOW -> ENTERING_INTERVENTION -> INTERVENTION_NATIVE
INTERVENTION_NATIVE -> REATTACHING_POLICY -> POLICY_FOLLOW
any unrecoverable failure -> sticky FAULT
```

Policy bring-up holds each leader at its own measured pose before output is
enabled. The bridge verifies the leader/follower synchronization gap and uses
bounded commands to converge before declaring policy follow. It then drives
leaders only from fresh measured follower feedback, never from an unaccepted
simulator target.

At the next serialized control boundary after the first local `i`, the bridge
blocks simulator mirror motion, holds followers, stops leader following,
changes both leaders to native mode, and verifies both sides before publishing
`enter`. An already-authorized request and its response are wholly before this
boundary; no exchange can straddle it. RoboDojo freezes Isaac, discards the
policy chunk, and records no frame throughout the transition.

On the second `i`, the bridge first checks the leader/follower gap while the
leaders remain native. An excessive gap sends no catch-up target and enters the
same sticky, four-arm fail-closed state used for any transition failure. A valid
reattachment holds each leader at its measured pose, enables output, converges
with bounded commands to fresh follower feedback, and only then publishes
`exit`. RoboDojo resumes with a new observation and fresh Pi0.5 inference.

## Motion, watchdogs, and failure

- Both sides are fully preflighted before any dual-arm command is committed.
- CAN cannot provide a true four-device transaction. If a partial transition
  or commit occurs, the bridge does not attempt a moving rollback; it disables
  both leaders and both followers and enters sticky fault.
- Joint and gripper targets have freshness, finite/range, step, synchronization
  gap, and following-error gates.
- A motion freshness timeout holds the controlled devices; it never repeats a
  stale target.
- Session timeout, socket loss, TTY loss, CAN/feedback failure, transition
  failure, or either-arm command failure disables all four arms. Missing
  disable acknowledgement requires the hard E-stop.
- A stopped/killed process cannot guarantee software disable. All initial HIL
  tests require an onsite operator supporting all arms with the hard E-stop.

## Episodes and recording

After simulator reset, `begin_episode` establishes a new sim-to-follower
relative anchor without turning reset displacement into physical motion. It
also requires policy-follow topology before the episode is accepted.

Every valid executed simulation frame is stored in one full LeRobot-v3
episode:

- `observation.*`: simulation observation before the action;
- `action`: ARX X5 action actually accepted by Isaac;
- `complementary_info.policy_action`: original Pi0.5 proposal, or zeros when
  unavailable;
- `complementary_info.is_intervention`: one only for a valid human-retargeted
  action.

Transition, rejected IK, and physical hold samples do not step Isaac and are
not written as training actions. Right/Left/Esc/Backspace keep their
accept-next, discard-retry, accept-exit, and discard-exit meanings.
