# PiPER-X-assisted RoboDojo DAgger

This mode connects a Kai0 Pi0.5 rollout in RoboDojo to two PiPER-X
leader/follower pairs. Humans move the **leaders only**, and only after the
first `i` takeover has completed. Before takeover, followers track the latest
pose accepted by RoboDojo and the motor-driven leaders track fresh measured
follower feedback.

```text
policy mode:       Kai0 -> RoboDojo ARX X5 -> accepted sim pose -> followers
                              fresh measured follower joints -> leaders
intervention mode: leaders -> relative SE(3) -> RoboDojo IK/ARX X5
                                                -> accepted sim pose -> followers
recording:         RoboDojo observation + actual ARX action + policy proposal
                   + is_intervention -> one full LeRobot v3 episode
```

## Current deployment constraint

Protocol v2 is loopback-only. The LeRobot hardware bridge and RoboDojo must run
on the **same Linux host**, and the bridge listens only on `127.0.0.1:8765`.
Do not expose the hardware bridge on a LAN or forward this v2 protocol between
machines. Cross-host operation needs a separately designed authenticated
transport and cannot reuse the monotonic-deadline assumptions in v2.

The bridge refuses a v1 peer because v1 leaves leaders permanently
backdrivable. No real PiPER-X or Isaac Sim hardware-in-the-loop acceptance test
has been run for v2 yet. The first run must therefore be supervised, with all
four arms supported and the hard E-stop in hand.

## Two-terminal startup

The example configurations in both repositories are intentionally disabled.
Do not merely change their booleans. First measure and verify every frame map,
axis sign/scale, gripper range and CAN assignment locally.
Also review the leader/follower synchronization-gap and per-command step
limits. Align each matching PiPER-X pair before startup; an excessive initial
gap is rejected rather than turned into a catch-up movement.

Terminal A owns PiPER-X hardware and keyboard events. Start it first:

```bash
cd /home/hoo/piper_x/lerobot_sealab

cp configs/piper_x_robodojo_bridge.json \
  /home/hoo/piper_x/robodojo_bridge.reviewed.json

# Physically calibrate/review axis_map, position_scale, both gripper ranges,
# and all four CAN assignments. Only after verification, set the reviewed
# file's retarget_calibrated field to the JSON boolean true.

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /home/hoo/piper_x/.conda

PIPERX_ROBODOJO_BRIDGE_CONFIG=/home/hoo/piper_x/robodojo_bridge.reviewed.json \
PIPERX_MOTION_ACK=I_HAVE_ESTOP_AND_SUPPORT \
bash cmds/piper_x_robodojo_bridge.sh
```

Do not continue until this terminal reports:

```text
PiPER-X RoboDojo bridge ready on 127.0.0.1:8765; i toggles intervention.
```

Keep Terminal A focused for operator keys. Its standard input must remain a
TTY.

Terminal B starts Kai0, RoboDojo and Isaac Sim:

```bash
cd /path/to/RoboDojo

cp config/piperx_sim_dagger.example.json \
  /absolute/path/piperx_sim_dagger.reviewed.json

# Calibrate leader_to_sim_rotation_qwxyz, translation_scale, both gripper
# ranges and safety limits. After an offline/no-follower verification, set
# calibrated to the JSON boolean true in this reviewed copy.

bash scripts/RoboDojo/collect_pi05_piperx_sim_dagger.sh \
  --task make_toast \
  --checkpoint-dir /absolute/path/to/kai0_checkpoint/5000 \
  --checkpoint-id make_toast-left-dagger/5000 \
  --piperx-calibration /absolute/path/piperx_sim_dagger.reviewed.json \
  --lerobot-root /absolute/path/to/data \
  --lerobot-repo-id robodojo_piperx_make_toast
```

Add `--resume` only when appending to an existing compatible LeRobot v3
dataset. The RoboDojo launcher never starts, enables or configures PiPER-X; it
only connects to the already-ready local bridge.

## Operator workflow

- In policy mode, Kai0 controls RoboDojo and the followers mirror the accepted
  simulated motion. Each leader is motor-driven from its matching follower's
  fresh measured joints and gripper. **Do not pull a leader in this mode.**
- Press `i` in **Terminal A** once to enter intervention. The bridge freezes
  followers, stops leader following, and switches both leaders to native
  backdrivable mode at the next serialized control boundary. Terminal A first
  prints `TRANSITION ... DO NOT MOVE`; do not move either leader until it
  prints `MANUAL READY`. That line confirms only the physical role. Begin the
  recorded correction after **Terminal B** prints `[PiPER-X DAgger] manual
  control ON`, which is emitted after RoboDojo has consumed the takeover edge
  and anchored the leader poses. Then move both leaders;
  RoboDojo anchors their current poses to the current simulated end-effector
  poses, applies relative Cartesian deltas, solves ARX X5 IK, and executes the
  safe result. The followers then mirror that accepted simulated result.
- Press `i` again to leave intervention. Both leaders first hold their current
  pose and safely reattach to fresh follower feedback. If their synchronization
  gap is too large, no catch-up target is sent; the bridge fails closed and
  disables all four arms, so the session must be inspected and restarted.
  Wait for Terminal A to print `POLICY FOLLOW READY`, then for Terminal B to
  print `manual control OFF`. After the completed `exit`, RoboDojo
  discards the stale Pi0.5 action chunk and runs fresh inference from the
  post-intervention observation.
- `Right Arrow`: accept and commit the complete episode, then load the next
  layout.
- `Left Arrow`: discard the candidate and retry the same layout.
- `Esc`: accept and commit the complete episode, then exit cleanly.
- `Backspace`: discard the candidate, then exit cleanly.

During either hardware transition Isaac is frozen and no dataset frame is
written. An accepted episode stores every executed simulation step, not only the
intervention segment. `action` is the action actually executed by the ARX X5
simulation; `complementary_info.policy_action` preserves the available policy
proposal; and `complementary_info.is_intervention` identifies valid human
steps. Unsafe/failed IK samples freeze both arms and are not written as expert
training actions.

If the bridge disconnects, misses a deadline, rejects a mirror target or
reports unhealthy hardware, the bridge disables both leaders and both
followers; RoboDojo discards the staged candidate and ends the session without
reconnecting or replaying it.

The exact wire contract and lifecycle invariants are documented in
`protocol/robodojo_piperx_v2/protocol.md`.
