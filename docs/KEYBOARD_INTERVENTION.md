# Pi0.5 keyboard intervention

This mode runs one visible RoboDojo simulation, lets Pi0.5 act normally, and
allows an operator to preempt it at any 25 Hz action tick.  It is designed for
collecting corrective HDF5 trajectories, not benchmark scoring.

For installation, checkpoint and asset migration, and machine-specific
preflight checks, see [the deployment guide](KEYBOARD_INTERVENTION_DEPLOYMENT.md).

## Start

The Pi0.5 server and checkpoint must already be installed.  From the RoboDojo
root, run:

```bash
bash scripts/RoboDojo/collect_pi05_keyboard.sh \
  --task stack_bowls \
  --ckpt RoboDojo-sim-arx_x5-joint-0 \
  --record-dir /home/piper/data/RoboDojo_interventions \
  --episodes 20 \
  --rendering-mode quality
```

The wrapper always uses `action_type=joint`, a visible Isaac Sim window and a
single environment.  Keep the Isaac Sim window focused while operating.
It explicitly sets `ROBODOJO_HEADLESS=0`, `HEADLESS=0`, and `LIVESTREAM=0`;
launch it from a graphical Piper session with a valid display. `--episodes`
is a requested count and remains capped by the task's configured number of
evaluation layouts. Reaching that count closes Isaac Sim normally; natural
success, the task step limit, and `N`/`Enter` each complete one rollout. For
example, `stack_bowls` supports up to 25 layouts, while `R` and `Backspace`
retry the same layout without consuming that count.

`--rendering-mode quality` preserves the benchmark camera preset and is the
recommended default for policy inference and training data. Use `balanced` or
`performance` only when interactive responsiveness is more important than
matching the training image distribution.

## Controls

| Key | Effect |
| --- | --- |
| `I` | Toggle manual control: first press takes over; second press returns to Pi0.5. |
| `1` / `2` | Select left / right arm. |
| `W` / `S` | End-effector +x / -x in the environment frame. |
| `A` / `D` | End-effector +y / -y. |
| `Q` / `E` | End-effector +z / -z. |
| `Z` / `X` | Roll + / -. |
| `T` / `G` | Pitch + / -. |
| `C` / `V` | Yaw + / -. |
| `K` | Toggle the selected gripper while manual control is active. |
| `N` or `Enter` | Save and finish this rollout; advance only if another requested layout remains. |
| `R` | Save, then restore the same layout for another correction. |
| `Backspace` | Reject the attempt; keep no HDF5 and retry the same layout. |
| `L` | Emergency manual-control exit and clear held keys. |

Use the plain `I` key, without Ctrl or Shift. It is a latched mode switch, not a
mouse-drag command: releasing `I` does not end manual control. On the first
press, the selected Cartesian target is anchored to the arm's measured
end-effector pose. The inactive arm explicitly holds its measured joints. A
second press returns control to Pi0.5. `Space` is deliberately unused because
Isaac Sim binds it to Play/Pause.

If window focus is lost, two seconds without keyboard events clears potentially
stuck motion keys but keeps manual control active and the robot holding. Press
`I` to return to Pi0.5, or `L` for an emergency manual-control exit.

## Chunk and safety behavior

Pi0.5 normally returns 50 joint targets (about two seconds at 25 Hz). If the
operator toggles `I` on at target `k`, target `k` and every later target in
that chunk are discarded. After `I` is toggled off, a fresh chunk is inferred
from the latest post-correction observation; the old chunk is never resumed.

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

`N`/`Enter`, `R`, and `Backspace` all reset the simulation through the normal
episode scheduler.  Reset reloads the saved object layout, restores the robot
initial state, resets rewards, and calls the policy server's `reset`; the
checkpoint server itself stays running.  `R` saves a trajectory but deliberately
does not consume the current layout or the requested `--episodes` count.  This
allows repeated collection of one corner case; press `N`/`Enter` when ready to
advance.

The recorder stores exactly one HDF5 sample per 25 Hz action sent to the
environment.  It never records the observation's lagged `action` field as the
executed target.

## LeRobot v3.0 output

The control process always writes atomic HDF5 first.  This is the recoverable
source of truth: LeRobot writes Parquet, MP4, and metadata files and cannot be
rolled back atomically when an episode is rejected or the simulator crashes.

To collect HDF5 and automatically rebuild a training-ready LeRobot v3.0 dataset
when the collection session finishes, add:

```bash
bash scripts/RoboDojo/collect_pi05_keyboard.sh \
  --task stack_bowls \
  --ckpt RoboDojo-sim-arx_x5-joint-0 \
  --record-dir /home/piper/data/RoboDojo_interventions \
  --episodes 20 \
  --lerobot-repo-id robodojo_interventions_stack_bowls \
  --lerobot-root /home/piper/data/lerobot
```

The exporter uses the CPU `h264` codec and explicitly hides CUDA devices.  It
does not compete for GPU memory.  The resulting dataset is stored at
`/home/piper/data/lerobot/robodojo_interventions_stack_bowls`.  Existing output
with that repo id is rebuilt from all accepted source HDF5 files, so rerunning
the command is idempotent rather than appending duplicate episodes.

Every LeRobot frame contains the standard Pi0.5 state, actual executed action,
three cameras and task, plus the Kai0-compatible feature:

```text
complementary_info.is_intervention  float32[1]
```

It is copied from `control/intervention_mask`: `1` is a successful human IK
action and `0` is policy or safety-hold control.  Kai0 can select intervention
anchors with:

```python
column_filter_column="complementary_info.is_intervention"
column_filter_keep_values=(1.0,)
```

## Manual LeRobot conversion

Use the generic LeRobot v3 converter, which reads the explicit executed
`action/*` datasets, preserves the intervention mask, and obtains 25 Hz from
`env_cfg/arx_x5.yml`:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/RoboDojo/export_interventions_lerobot_v30.py \
  "RoboDojo_interventions.stack_bowls.arx_x5" \
  --repo-id robodojo_interventions_stack_bowls \
  --root /home/piper/data/lerobot \
  --vcodec h264 \
  --overwrite
```

Do **not** feed these HDF5 files to the current
`XPolicyLab/policy/Pi_05/openpi/scripts/process_data.py`: that policy-specific
script ignores explicit actions, constructs labels from the next state, and
currently hard-codes 50 Hz. It would lose the intended human correction labels
and use the wrong timing.
