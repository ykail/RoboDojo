# PiPER-X-assisted RoboDojo DAgger

This mode connects a Kai0 Pi0.5 rollout in RoboDojo to two PiPER-X
leader/follower pairs. Humans move the **leaders only**. The followers always
track the latest pose accepted by the RoboDojo simulation; they never follow a
leader through a direct joint-copy path.

```text
policy mode:       Kai0 -> RoboDojo ARX X5 -> accepted sim pose -> followers
intervention mode: leaders -> relative SE(3) -> RoboDojo IK/ARX X5
                                                -> accepted sim pose -> followers
recording:         RoboDojo observation + actual ARX action + policy proposal
                   + is_intervention -> one full LeRobot v3 episode
```

## Current deployment constraint

Protocol v1 is loopback-only. The LeRobot hardware bridge and RoboDojo must run
on the **same Linux host**, and the bridge listens only on `127.0.0.1:8765`.
Do not expose the hardware bridge on a LAN or forward this v1 protocol between
machines. Cross-host operation needs a separately designed authenticated
transport and cannot reuse the monotonic-deadline assumptions in v1.

No real PiPER-X or Isaac Sim hardware-in-the-loop acceptance test has been run
for this implementation yet. The first run must therefore be supervised, with
four-arm support and an emergency stop in hand.

## Two-terminal startup

The example configurations in both repositories are intentionally disabled.
Do not merely change their booleans. First measure and verify every frame map,
axis sign/scale, gripper range and CAN assignment locally.

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
  simulated motion. The leaders remain available for the human to hold/move.
- Press `i` in **Terminal A** once to enter intervention. Move both leaders;
  RoboDojo anchors their current poses to the current simulated end-effector
  poses, applies relative Cartesian deltas, solves ARX X5 IK, and executes the
  safe result. The followers then mirror that accepted simulated result.
- Press `i` again to leave intervention. RoboDojo discards the stale Pi0.5
  action chunk and runs fresh inference from the post-intervention observation.
- `Right Arrow`: accept and commit the complete episode, then load the next
  layout.
- `Left Arrow`: discard the candidate and retry the same layout.
- `Esc`: accept and commit the complete episode, then exit cleanly.
- `Backspace`: discard the candidate, then exit cleanly.

An accepted episode stores every executed simulation step, not only the
intervention segment. `action` is the action actually executed by the ARX X5
simulation; `complementary_info.policy_action` preserves the available policy
proposal; and `complementary_info.is_intervention` identifies valid human
steps. Unsafe/failed IK samples freeze both arms and are not written as expert
training actions.

If the bridge disconnects, misses a deadline, rejects a mirror target or
reports unhealthy hardware, RoboDojo discards the staged candidate and ends
the bridge session without reconnecting or replaying it.

The exact wire contract and lifecycle invariants are documented in
`protocol/robodojo_piperx_v1/protocol.md`.
