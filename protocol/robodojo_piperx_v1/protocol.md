# RoboDojo PiPER-X bridge protocol v1

Status: experimental, safety-critical local integration contract.

Protocol identifier: `robodojo_piperx_v1`.

## Ownership and control direction

RoboDojo is authoritative for simulator observations, Pi0.5 inference, action
arbitration, ARX X5 IK and LeRobot recording. The separate LeRobot process is
authoritative for both PiPER-X leaders/followers, CAN, the physical keyboard,
watchdogs and fail-closed hardware behavior. Kai0 never talks to PiPER-X.

```text
policy mode:
  RoboDojo observation -> Kai0 -> ARX action -> Isaac Sim
  accepted simulated EE poses -> bridge -> PiPER-X follower IK/servo

intervention mode:
  PiPER-X leaders -> bridge -> RoboDojo relative SE(3) retarget -> ARX IK -> Isaac Sim
  accepted simulated EE poses on the next exchange -> bridge -> followers
```

The bridge must never copy leader joints to ARX X5 or bypass the simulator by
driving leaders directly into followers. ARX X5 and PiPER-X have different
kinematics. Both directions use Cartesian poses and robot-specific IK.

`config/piperx_sim_dagger.example.json` is intentionally shipped with
`"calibrated": false`. It is not runnable calibration. Copy it, measure the
leader-to-simulator frame rotation, scales and gripper ranges for both arms,
verify the mapping without follower motion, and only then set the copy to
`true`. RoboDojo rejects anything except the JSON boolean `true` before
opening a bridge episode.

## Transport

- TCP loopback only. RoboDojo rejects non-loopback hosts.
- One persistent connection is reused across episode resets.
- Each frame is `uint32_be byte_length` followed by exactly that many UTF-8
  bytes containing strict JSON. NaN/Infinity and duplicate keys are invalid.
- The maximum frame is 1 MiB.
- Exactly one request is outstanding. RoboDojo serializes normal requests and
  background heartbeat requests with one mutex and one monotonically
  increasing connection-wide `seq`.
- A malformed, late, mismatched or disconnected response makes the session
  permanently lost. RoboDojo does not reconnect or replay that episode.

## Exact envelope

Every request and response has exactly these fields:

| Field | Type | Meaning |
|---|---|---|
| `protocol` | string | Exactly `robodojo_piperx_v1` |
| `type` | string | Request type; response echoes it |
| `session_id` | string | Fixed for the TCP session |
| `episode_id` | string | Fixed between begin/end; changes after reset |
| `generation` | integer | Last control generation acknowledged by RoboDojo |
| `seq` | integer | Connection-wide request sequence; response echoes it |
| `sent_monotonic_ns` | integer | Sender timestamp |
| `deadline_monotonic_ns` | integer | Request deadline; response echoes it |
| `payload` | object | Exact body below |

Request payloads:

```text
begin_episode  {"sim": SIM}
exchange       {"sim": SIM}
heartbeat      {}
hold           {"reason": NON_EMPTY_STRING}
end_episode    {"reason": NON_EMPTY_STRING}
```

`SIM` is the last simulator state accepted by RoboDojo, not the raw policy
target:

```json
{
  "left":  {"pose": [0, 0, 0, 1, 0, 0, 0], "gripper": 1.0},
  "right": {"pose": [0, 0, 0, 1, 0, 0, 0], "gripper": 1.0}
}
```

Pose order is `[x,y,z,qw,qx,qy,qz]`; quaternion norm is one. `gripper` is the
normalized open fraction (`0=closed`, `1=open`).

Every response payload contains exactly:

```json
{
  "mode": "policy",
  "edge": null,
  "terminal_request": null,
  "leader": {
    "left":  {"pose": [0, 0, 0, 1, 0, 0, 0], "gripper_m": 0.07},
    "right": {"pose": [0, 0, 0, 1, 0, 0, 0], "gripper_m": 0.07}
  },
  "mirror_accepted": true,
  "health": {"ok": true},
  "diagnostics": {"request_ok": true}
}
```

- `mode`: `policy`, `intervention`, or sticky `fault`.
- `edge`: `enter`, `exit`, or null. A bridge-side `I` press toggles mode and
  increments `generation` once. The response carrying that edge returns the
  new generation; the next request uses it.
- `terminal_request`: `accept_next`, `discard_retry`, `accept_exit`,
  `discard_exit`, or null. These preserve Right/Left/Esc/Backspace semantics.
- `leader`: one coherent dual-arm sample. `gripper_m` is measured opening in
  meters.
- `mirror_accepted`: whether an ordinary `exchange` simulator target passed
  follower IK/safety and was accepted. It may be false when the same response
  carries an `edge` or `terminal_request`: that UI event safely preempts motion
  and the bridge holds instead. It is also false for heartbeat/hold/end because
  those messages never command motion. Only a plain exchange with no UI event
  and `mirror_accepted=false` is a fatal mirror rejection.
- `health.ok` must be exactly true for a usable response.
- `diagnostics.request_ok` is required and boolean. False means the request was
  rejected and is fatal even when hardware health itself remains true. A local
  UI preemption is not a rejected request: it returns `request_ok=true`,
  `mirror_accepted=false`, and `no_motion_reason=local_ui_preemption`.

A successful `hold` has the same hardware lifecycle boundary as
`end_episode`: measured follower hold, cleared anchors, idle heartbeat with the
last episode identity, and no further `exchange` until a new `begin_episode`.

## Heartbeat and command freshness

Synchronous Kai0 inference may take seconds, so RoboDojo sends background
`heartbeat` requests. A heartbeat proves only TCP/session liveness. It must
not refresh follower motion-command freshness and must not dequeue or
acknowledge `I` edges or terminal requests. Its response therefore echoes the
request generation with `edge=null` and `terminal_request=null`. The next
`exchange` remains the only control-state boundary.

The bridge independently enforces:

- motion freshness timeout: hold measured follower positions; do not invent a
  new target;
- session heartbeat timeout: disable both followers and enter sticky fault.

## Episode and preemption semantics

1. After each simulator reset RoboDojo sends `begin_episode` with the new
   accepted simulator poses. Both sides reset pose/gripper anchors and mode to
   policy generation zero. Reset displacement must never become a follower
   motion delta.
2. RoboDojo sends `exchange` before every policy action. The response can
   preempt that very action.
3. `enter` discards the current and remaining Pi0.5 chunk. Leader pose and
   gripper deltas are anchored to the current simulator state, guaranteeing no
   jump on the takeover sample.
4. A valid dual-arm retarget/ARX IK result is executed and recorded. Rejected
   IK freezes the simulation and followers and is not recorded as an expert
   action.
5. `exit` discards all stale policy targets. The next inference uses the latest
   post-intervention simulator observation.
6. `end_episode` holds followers before potentially slow LeRobot video commit.
   Heartbeats continue with the last ended `episode_id` and generation during
   that commit; motion freshness is inactive because followers are already
   held. The next reset uses a new `episode_id`, generation zero and new
   anchors on the same TCP session/global `seq` stream.

## Recorded training source of truth

The existing LeRobot v3 stream recorder remains authoritative:

- `observation.*`: RoboDojo simulation observation before the action;
- `action`: ARX X5 action actually accepted by the simulation;
- `complementary_info.policy_action`: original Pi0.5 proposal or zeros if no
  proposal existed on a manual frame;
- `complementary_info.is_intervention`: one only for a valid human-retargeted
  ARX action;
- RoboDojo episode sidecar includes `robodojo_control_mode=piperx_sim_dagger`.

PiPER-X joints are diagnostics, not training actions.

The supervised two-terminal startup sequence and operator key workflow are in
`docs/PIPERX_SIM_DAGGER.md`.
