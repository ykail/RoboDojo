# PiPER-X-assisted RoboDojo DAgger

This mode runs a Kai0 Pi0.5 policy in RoboDojo, mirrors the accepted simulated
motion on two PiPER-X follower arms, and makes the two motor-driven leaders
display the measured follower motion. Pressing `i` transfers authority to the
leaders; the same cached leader sample then drives both the followers and the
ARX X5 simulation.

```text
policy:
  Pi0.5 -> ARX X5 simulation -> PiPER-X followers -> PiPER-X leaders

manual after i:
  PiPER-X leaders
    +-> direct relative joint delta -> PiPER-X followers
    `-> relative FK pose -> ARX X5 IK -> simulation

recording:
  full simulation observation + executed ARX action + policy proposal
  + intervention flag -> one LeRobot v3 episode
```

The manual fan-out avoids a PiPER-X end-pose/IK round trip. The followers use
the original leader joint delta, while RoboDojo independently solves IK only
for its different ARX X5 embodiment.

## Safety and deployment constraint

Protocol v3 is loopback-only: the LeRobot hardware bridge and RoboDojo must run
on the same Linux host. Do not expose or forward port 8765. The protocol uses
host-monotonic deadlines and is not an authenticated network transport.

No real four-arm/Isaac acceptance test is implied by the software tests. The
first run must have an onsite operator supporting all four arms, clear follower
workspaces, and a hard E-stop in hand. Stop immediately for vibration,
knocking, unexpected self-motion, or heating.

V3 has no per-machine retarget calibration file. It accepts only the built-in
`arx_x5_piperx_relative_v1` profile and takes relative anchors at episode and
takeover boundaries. You must still verify all four CAN names, physical arm
identity, workspace clearance, joint zero behavior, and the safety thresholds
in `configs/piper_x_robodojo_bridge.json`.

## Why Terminal A starts first

Terminal A is the exclusive hardware owner and local keyboard endpoint. At
startup it is **listen-only**: it opens the local socket but does not connect,
enable, or command any arm. Terminal B first starts Kai0, creates Isaac Sim,
and stages the initial policy observation. Only its subsequent
`arm_and_begin_episode` request authorizes Terminal A to perform bounded
four-arm bring-up. Starting A first therefore guarantees a ready local endpoint
without moving hardware before the policy and simulation exist.

## Terminal A: local PiPER-X owner

Use the LeRobot checkout that contains the v3 bridge:

```bash
cd /path/to/lerobot_sealab

cp --no-clobber configs/piper_x_robodojo_bridge.json \
  /path/to/piper_x_robodojo_bridge.reviewed.json

# Review the four CAN names and safety limits in the copied file.

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /path/to/piper_x_conda_env

PIPERX_ROBODOJO_BRIDGE_CONFIG=/path/to/piper_x_robodojo_bridge.reviewed.json \
PIPERX_MOTION_ACK=I_HAVE_ESTOP_AND_SUPPORT \
bash cmds/piper_x_robodojo_bridge.sh
```

Keep this terminal focused: `i` and the episode-label keys are read from its
local TTY, not from the Isaac window. The initial ready message must say v3 and
`SAFE LISTEN-ONLY`; at that moment the arms are still untouched.

## Terminal B: Kai0, RoboDojo, and Isaac Sim

On a fresh RoboDojo checkout, first initialize the assets and Kai0 environment
as described in the repository setup guide. Then run:

```bash
cd /path/to/RoboDojo

OMNI_KIT_ACCEPT_EULA=YES \
bash scripts/RoboDojo/collect_pi05_piperx_sim_dagger.sh \
  --task make_toast \
  --checkpoint-dir /path/to/checkpoint/59999 \
  --checkpoint-id RoboDojo-sim-arx_x5-joint-0/59999 \
  --kai0-root /path/to/RoboDojo/third_party/kai0 \
  --kai0-python /path/to/RoboDojo/third_party/kai0/.venv/bin/python \
  --piperx-response-timeout 1.0 \
  --piperx-arm-timeout 60 \
  --piperx-transition-timeout 10 \
  --lerobot-root /path/to/data/lerobot \
  --lerobot-repo-id robodojo_piperx_make_toast_official_59999 \
  --eval-num 10 \
  --policy-gpu 0 \
  --env-gpu 0
```

The 1.0-second steady-response deadline is conservative for the first HIL run;
the launcher default is 0.2 seconds. It is independent of hardware watchdogs.
The default is GUI mode. Add `--headless` only when a window is intentionally
unnecessary. Add `--resume` only when appending to an existing compatible
LeRobot v3 dataset.

The command launches the Kai0 server itself, waits for it to become ready, and
then starts the RoboDojo evaluator. A third policy-server command is neither
needed nor allowed on the same port.

### External Kai0 on Coffee

Start the strict server on Coffee with the clean Kai0 checkout and official
checkpoint:

```bash
cd /home/ykail/vibe_code/RoboDojo/third_party/kai0

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src \
.venv/bin/python scripts/serve_robodojo_policy.py \
  --checkpoint-dir /home/ykail/data/RoboDojo_hf/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999 \
  --checkpoint-id RoboDojo-sim-arx_x5-joint-0/59999 \
  --host 127.0.0.1 \
  --port 18080
```

On Hoo, create a fail-fast loopback tunnel in a separately supervised
terminal:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=5 \
  -o ServerAliveCountMax=2 \
  -L 127.0.0.1:18080:127.0.0.1:18080 \
  yikai
```

Then Terminal B on Hoo uses external mode:

```bash
cd /home/hoo/RoboDojo-piperx-dagger-v2

OMNI_KIT_ACCEPT_EULA=YES \
bash scripts/RoboDojo/collect_pi05_piperx_sim_dagger.sh \
  --task make_toast \
  --checkpoint-id RoboDojo-sim-arx_x5-joint-0/59999 \
  --external-policy-server-url ws://127.0.0.1:18080 \
  --expected-kai0-commit ecc1a7451c3156b1e5f7533851dbb0222896206f \
  --expected-checkpoint-digest sha256:70bb68139ba717553d9a9d9c3055bb322b85046d729377ee46eaaf997c1eaac4 \
  --lerobot-python /home/hoo/RoboDojo/third_party/kai0/.venv/bin/python \
  --piperx-response-timeout 1.0 \
  --piperx-arm-timeout 60 \
  --piperx-transition-timeout 10 \
  --lerobot-root /home/hoo/data/lerobot \
  --lerobot-repo-id robodojo_piperx_make_toast_official_59999 \
  --eval-num 10 \
  --env-gpu 0
```

External mode never validates or loads a local checkpoint/JAX, and it never
starts, signals, or kills a local Kai0 process. The launcher first requires the
loopback tunnel to be reachable, then performs a no-Isaac/no-hardware HELLO
preflight that checks checkpoint ID, digest, full Kai0 commit, and
`dirty=false`. The evaluator verifies the same provenance again for its formal
session before starting an episode or authorizing the hardware bridge. The
client never reconnects a lost session; a broken tunnel therefore aborts and
follows the existing four-arm fail-closed path.

## Operator workflow

1. In policy mode, Pi0.5 moves the ARX X5 simulation. Followers mirror the last
   accepted simulated end poses, then leaders mirror fresh measured follower
   joint deltas around runtime leader/follower anchors. Calibrated absolute
   zero offsets are preserved. Do not pull a motor-driven leader.
2. Press `i` once in Terminal A. Both followers take measured hold and both
   leaders remain held while RoboDojo freezes Isaac. Do not move until Terminal
   B prints `[PiPER-X DAgger] manual control ON`. Before printing it, RoboDojo
   resolves one exact post-switch sample as `anchor`, so the physical joint
   zero and simulated Cartesian zero come from the same leader state.
3. Move the two leaders. For every bimanual sample RoboDojo first checks both
   ARX IK results. On success, that exact sample is committed to both followers
   by relative PiPER-X joint deltas; only after the physical acknowledgement
   does Isaac execute and record the ARX action. If either IK/safety check
   fails, both followers and Isaac hold and no expert frame is recorded.
4. Press `i` again. Both sides hold, the bridge anchors policy mapping at the
   final manual state, latches fresh relative leader/follower anchors, and
   leaders safely reattach without an absolute-coordinate jump. RoboDojo
   discards the stale Pi0.5 action chunk and infers again from the
   post-correction observation.
5. When the complete rollout can be judged, label it:

   | Key in Terminal A | Result |
   |---|---|
   | Right Arrow | save complete episode, then load next layout |
   | Left Arrow | discard candidate, retry same layout |
   | Esc | save complete episode, then exit |
   | Backspace | discard candidate, then exit |

If a terminal key is pressed during intervention, the bridge first performs
the same safe exit/reattachment and only then delivers the label.

`i` is a control-boundary request, not a hard E-stop. If it is pressed after a
policy exchange has already been acknowledged but just before Isaac applies
that action, the one already-authorized simulator step may finish before the
transition is observed (bounded to one control tick). No later policy action is
executed; takeover then pairs the resulting simulator state with the held
physical state through the explicit relative anchor. Use the hardware E-stop,
not `i`, for an immediate safety stop.

## What is stored

Only accepted episodes are committed. Each contains all executed policy and
manual steps, not merely the intervention interval:

- `action`: action actually executed by the ARX X5 simulation;
- `complementary_info.policy_action`: original policy proposal when available;
- `complementary_info.is_intervention`: one for accepted human corrections;
- episode metadata: task, checkpoint provenance, v3 protocol, fixed embodiment
  profile, layout/seed, success, finish reason, and intervention presence.

Transition frames, the takeover anchor, rejected IK, and physical holds do not
step Isaac and are not stored as expert actions. A disconnect, missed deadline,
stale feedback, TTY loss, CAN failure, partial command, or invalid protocol
response disables all four arms, discards the candidate, and makes that session
non-replayable.

The complete contract is in
`protocol/robodojo_piperx_v3/protocol.md`.
