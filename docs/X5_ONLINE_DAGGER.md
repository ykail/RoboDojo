# ARX X5 online DAgger

This is the minimal two-arm replacement for the PiPER-X bridge. The same two
physical ARX X5 arms follow the simulated ARX X5 during policy rollout. The
global `i` key switches those arms to manual/gravity-compensated control; their
relative joint motion then drives Isaac with an identity joint mapping. A
second `i` returns to policy control, discards the old action chunk, and starts
fresh inference from the intervention result.

No tmux is started by these scripts.

## Machines and ports

| Machine | Responsibility | Local ports |
| --- | --- | --- |
| Hoo | Strict Kai0 `robodojo-policy-v1` server and official 59999 checkpoint | `127.0.0.1:18080` |
| X5 PC | Isaac Sim, two physical ARX X5 arms, and LeRobot recording | source `127.0.0.1:8770`; forwarded policy `127.0.0.1:18080` |

The implementation is staged on Piper at
`/home/piper/vibe_code/RoboDojo-x5-dagger`. Piper is only the preparation
host; it does not need to be one of the runtime machines.

## Copy to the actual X5 PC

A complete portable Git bundle is staged at
`/home/piper/vibe_code/RoboDojo-x5-dagger.bundle`. Copy that file to the X5 PC
with `scp` or removable storage, then run:

```bash
git clone -b feat/arx-x5-online-dagger \
  /path/to/RoboDojo-x5-dagger.bundle RoboDojo-x5-dagger
cd RoboDojo-x5-dagger
git submodule update --init --recursive
```

The prepared branch is `feat/arx-x5-online-dagger`. Keep this RoboDojo
worktree clean so every recorded episode has useful hardware-code provenance.

The hardware protocol is `robodojo_dual_joint_mirror_v1`. The required
embodiment profile is `arx_x5_identity_joint_v1`, whose six joint signs are
`[+1,+1,+1,+1,+1,+1]`. Do not reuse the PiPER-X sign flips.

## Required state

On Hoo:

- Kai0 must be exactly commit
  `ecc1a7451c3156b1e5f7533851dbb0222896206f`, with a clean worktree.
- Checkpoint `RoboDojo-sim-arx_x5-joint-0/59999` must contain `params/` and
  `assets/arx_x5_sim/norm_stats.json`.
- The strict expected checkpoint digest is
  `sha256:70bb68139ba717553d9a9d9c3055bb322b85046d729377ee46eaaf997c1eaac4`.

On the X5 PC:

- The RoboDojo checkout must contain
  `scripts/RoboDojo/x5_dual_joint_mirror_source.py` and the X5 control mode.
- `X5_PYTHON` must import `arx5_interface`. The launcher first tries
  `$ROBOT_LAB_ROOT/.venv/bin/python`, with `ROBOT_LAB_ROOT` defaulting to
  `$HOME/Robot_Lab`. If that environment cannot import the SDK, point
  `X5_PYTHON` at the actual vendor/robot environment; do not bypass the check.
- CAN defaults are left `can1` and right `can3`; both must already exist and be
  `UP`. Override them with `X5_LEFT_CAN` and `X5_RIGHT_CAN` only when the
  physical wiring has been verified.
- Activate the normal RoboDojo/Isaac environment before starting the Isaac
  launcher. `X5_LEROBOT_PYTHON` must point to a Python that can import the
  LeRobot v3 dataset writer.
- The operator must be beside the arms, with the workspace clear and the
  emergency stop reachable. By default the source moves both arms to the six
  joint zero pose over five seconds before accepting Isaac.

## Startup order

Start the Hoo policy server first in its own Hoo terminal:

```bash
ssh hoo
cd /home/hoo/RoboDojo-piperx-dagger-v2
./scripts/RoboDojo/run_hoo_policy_59999.sh
```

Wait until checkpoint loading completes and the server reports that it is
listening. The launcher itself verifies the Kai0 commit, clean state, required
checkpoint files, checkpoint ID, and step. The Isaac HELLO preflight performs
the final strict commit/checkpoint/digest match before Isaac starts.

Then use three separate terminals on the X5 PC.

Terminal 1 — foreground SSH tunnel:

```bash
cd /path/to/RoboDojo
./scripts/RoboDojo/run_hoo_policy_tunnel_from_x5.sh
```

Terminal 2 — sole owner of both physical X5 arms:

```bash
cd /path/to/RoboDojo
X5_PYTHON=/path/to/python-with-arx5-interface \
  ./scripts/RoboDojo/run_x5_dagger_hardware.sh
```

Keep clear while the source performs its gradual home move. Wait until it is
listening on port 8770. This process is the only process allowed to own `can1`
and `can3`.

Terminal 3 — Isaac, policy client, and recorder:

```bash
cd /path/to/RoboDojo
conda activate RoboDojo
X5_LEROBOT_PYTHON=/path/to/lerobot-writer-python \
  ./scripts/RoboDojo/run_x5_dagger_isaac.sh
```

The default output is:

```text
$HOME/data/lerobot/robodojo_make_toast_x5_online_dagger_v1
```

An existing compatible LeRobot v3 dataset is resumed automatically. To collect
several episodes in one invocation, set `ROBODOJO_EVAL_NUM`, for example:

```bash
ROBODOJO_EVAL_NUM=20 ./scripts/RoboDojo/run_x5_dagger_isaac.sh
```

## Operator controls and saving

There is only one operator key:

- Global `i`: toggle manual intervention ON/OFF. The hardware terminal does not
  need focus.

There is no `s` or `r` key in this online rollout mode. Recording starts with
the episode. Policy frames and human frames are stored in the same episode with
their action-source/intervention metadata. When the task succeeds, fails, or
reaches its episode limit, the recorder commits the episode automatically.

If an exception or Ctrl-C interrupts the current episode, the incomplete
candidate is rolled back rather than committed. Previously committed episodes
remain in the dataset.

Stop in this order: Isaac first, then the hardware source after the arms are in
a supported/held state, then the SSH tunnel, and finally the Hoo policy server.

## Minimal physical acceptance

Use a clear scene and one short episode:

1. Start all processes and confirm the Isaac launcher accepts the strict HELLO
   before the simulator appears.
2. During policy rollout, verify both physical X5 arms follow the corresponding
   simulated arms in the same direction. Stop immediately if any joint uses the
   opposite sign.
3. With both systems stationary, press global `i`. Entry must not produce a
   joint jump. Move one arm by only a few degrees and verify the corresponding
   simulated joint moves by approximately the same amount and direction.
4. Check both grippers at fully open and fully closed.
5. Press global `i` again. The source must hold the intervention result while
   RoboDojo discards the stale policy chunk and performs fresh inference.
6. Let the episode terminate and require a console line containing
   `committed episode` at the expected dataset path.

Only after these six checks pass should a longer DAgger collection be run.

## Useful overrides

Hoo policy paths:

```bash
KAI0_ROOT=/path/to/clean/kai0 \
KAI0_PYTHON=/path/to/kai0-python \
ROBODOJO_CHECKPOINT_DIR=/path/to/59999 \
ROBODOJO_POLICY_RUN_DIR=/path/to/policy-runs \
  ./scripts/RoboDojo/run_hoo_policy_59999.sh
```

Tunnel identity or host:

```bash
HOO_SSH_TARGET=hoo@10.19.127.58 \
HOO_SSH_IDENTITY_FILE=/path/to/id_ed25519 \
  ./scripts/RoboDojo/run_hoo_policy_tunnel_from_x5.sh
```

Hardware layout and home pose:

```bash
ROBOT_LAB_ROOT=/path/to/Robot_Lab \
X5_PYTHON=/path/to/python-with-arx5-interface \
X5_LEFT_CAN=can1 X5_RIGHT_CAN=can3 \
X5_HOME_RAD="0 0 0 0 0 0" \
X5_HOME_DURATION_S=5 \
  ./scripts/RoboDojo/run_x5_dagger_hardware.sh
```

Dataset paths:

```bash
ROBODOJO_LEROBOT_ROOT="$HOME/data/lerobot" \
ROBODOJO_LEROBOT_REPO_ID=robodojo_make_toast_x5_online_dagger_v1 \
X5_LEROBOT_PYTHON=/path/to/lerobot-writer-python \
  ./scripts/RoboDojo/run_x5_dagger_isaac.sh
```

Ports may be changed consistently with `ROBODOJO_POLICY_PORT`,
`HOO_POLICY_PORT`, and `ROBODOJO_X5_SOURCE_PORT`, but policy and source ports
must remain different.

## Fault recovery

- `address already in use` on 18080: another tunnel or local process owns the
  X5-side policy port. Inspect the owner, stop that specific process/terminal,
  then restart the foreground tunnel.
- `address already in use` on 8770: an earlier hardware source still owns the
  bridge port or CAN. Stop it before starting another source.
- `arx5_interface` import failure: select the verified SDK Python through
  `X5_PYTHON`. The source intentionally does not start without the SDK.
- CAN missing/not `UP`: restore the site-approved CAN configuration, then rerun
  the hardware launcher. Do not skip the preflight.
- Policy connection, invalid HTTP response, or HELLO provenance failure: keep
  Isaac stopped; verify the Hoo strict server is fully ready and restart the
  foreground tunnel. A commit, dirty-state, checkpoint-ID, or digest mismatch
  must be fixed on Hoo rather than overridden.
- Hardware-source disconnect: Isaac fails closed and the current episode is not
  committed. Support the arms, stop Isaac, restore the hardware source, and
  restart a new episode.
- Unexpected physical motion or a sign mismatch: use the emergency stop and
  terminate the hardware owner. Do not compensate by selecting the PiPER-X
  profile; investigate the X5 model/CAN identity first.
- Existing output path is not a valid LeRobot dataset: choose a new
  `ROBODOJO_LEROBOT_REPO_ID` or repair the intended dataset; the launcher will
  not overwrite it.
