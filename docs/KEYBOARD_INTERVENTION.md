# Pi0.5 keyboard intervention

This mode runs one visible RoboDojo simulation, lets Pi0.5 act normally, and
allows an operator to preempt it at any 25 Hz action tick.  It is designed for
collecting corrective HDF5 trajectories, not benchmark scoring.

## Start

The Pi0.5 server and checkpoint must already be installed.  From the RoboDojo
root, run:

```bash
bash scripts/RoboDojo/collect_pi05_keyboard.sh \
  --task stack_bowls \
  --ckpt RoboDojo-sim-arx_x5-joint-0 \
  --record-dir /home/piper/data/RoboDojo_interventions
```

The wrapper always uses `action_type=joint`, a visible Isaac Sim window and a
single environment.  Keep the Isaac Sim window focused while operating.
It explicitly sets `ROBODOJO_HEADLESS=0`, `HEADLESS=0`, and `LIVESTREAM=0`;
launch it from a graphical Piper session with a valid display. `--episodes`
is a requested count and remains capped by the task's configured number of
evaluation layouts.

## Controls

| Key | Effect |
| --- | --- |
| Hold `Space` | Take control. Releasing it returns to Pi0.5. |
| `1` / `2` | Select left / right arm. |
| `W` / `S` | End-effector +x / -x in the environment frame. |
| `A` / `D` | End-effector +y / -y. |
| `Q` / `E` | End-effector +z / -z. |
| `Z` / `X` | Roll + / -. |
| `T` / `G` | Pitch + / -. |
| `C` / `V` | Yaw + / -. |
| `Space` + `K` | Toggle the selected gripper while in control. |
| `N` or `Enter` | Finish and save the current episode. |
| `Backspace` | Reject the attempt; keep no HDF5 and retry the same layout. |
| `L` | Clear held-key state if window focus was lost. |

`Space` is a deadman/clutch key; it does not enable mouse dragging.  On the
press edge, the selected Cartesian target is anchored to the arm's measured
end-effector pose.  The inactive arm explicitly holds its measured joints.
The deadman state also expires after two seconds without any keyboard event;
normal OS key-repeat refreshes this heartbeat. This limits sustained motion if
the window loses focus and a release event is missed.

## Chunk and safety behavior

Pi0.5 normally returns 50 joint targets (about two seconds at 25 Hz).  If the
operator presses `Space` at target `k`, target `k` and every later target in
that chunk are discarded.  After release, a fresh chunk is inferred from the
latest post-correction observation; the old chunk is never resumed.

Each Cartesian keyboard delta is solved to a complete 14-dimensional dual-arm
joint target.  A non-finite, failed, or excessively discontinuous IK result
becomes a full hold action and is recorded as `action_source=safety_hold`, not
as an expert intervention.

## Output

Accepted episodes are written atomically under
`<record-dir>/<task>/<env-cfg>/data/`. The collection wrapper also creates a
safe `data/<dataset-name>` link, so XPolicyLab's generic converter can discover
the files. The
standard `state`, `action` and JPEG `vision/*/colors` groups are accompanied by:

- `policy_action`, `human_action`, and `executed_action`;
- `control/intervention_mask`, `action_source`, `ik_success`, and active arm;
- action chunk id/index and takeover edges;
- layout, checkpoint, repository revisions, frequency, success and acceptance
  metadata.

The recorder stores exactly one HDF5 sample per 25 Hz action sent to the
environment.  It never records the observation's lagged `action` field as the
executed target.

## Convert for Pi0.5 fine-tuning

Use the generic LeRobot v3 converter, which reads the explicit executed
`action/*` datasets and obtains 25 Hz from `env_cfg/arx_x5.yml`:

```bash
python XPolicyLab/scripts/transform_lerobot_v30_format.py \
  "RoboDojo_interventions.stack_bowls.arx_x5" \
  --repo_id robodojo_interventions_stack_bowls
```

Do **not** feed these HDF5 files to the current
`XPolicyLab/policy/Pi_05/openpi/scripts/process_data.py`: that policy-specific
script ignores explicit actions, constructs labels from the next state, and
currently hard-codes 50 Hz. It would lose the intended human correction labels
and use the wrong timing.
