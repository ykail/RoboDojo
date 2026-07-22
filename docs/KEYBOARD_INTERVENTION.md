# Pi0.5 keyboard intervention collection

This mode runs one visible RoboDojo simulation. Pi0.5 controls the dual-arm
ARX-X5 robot until the operator presses `I`, makes a Cartesian keyboard
correction, and presses `I` again to return control to the policy. The complete
candidate trajectory is staged from the start of each reset; after watching the
result, the operator explicitly accepts or discards it.

The output is direct LeRobot v3.0. There is no intermediate HDF5 dataset and no
post-session conversion. For installation, checkpoints, assets, and graphical
session setup, see [the deployment guide](KEYBOARD_INTERVENTION_DEPLOYMENT.md).

## Start

Run this single command from the RoboDojo root:

```bash
bash scripts/RoboDojo/collect_pi05_keyboard.sh \
  --task make_toast \
  --ckpt RoboDojo-sim-arx_x5-joint-0 \
  --lerobot-repo-id robodojo_interventions_make_toast \
  --lerobot-root /home/piper/data/lerobot \
  --rendering-mode quality
```

The wrapper starts both the Pi0.5 policy server (using the named checkpoint)
and Isaac Sim. Do not start another policy server. It forces one visible Isaac
environment, joint-action inference, and real-time pacing. Keep the Isaac Sim
window focused while operating.

On Coffee, replace `/home/piper` with `/home/ykail`. The default output root is
`$HOME/data/lerobot`, and the default repo id is
`robodojo_interventions_<task>`, so both LeRobot arguments may be omitted.

If the dataset path already exists, the wrapper refuses to touch it unless
`--resume` is passed:

```bash
bash scripts/RoboDojo/collect_pi05_keyboard.sh \
  --task make_toast \
  --ckpt RoboDojo-sim-arx_x5-joint-0 \
  --lerobot-repo-id robodojo_interventions_make_toast \
  --lerobot-root /home/piper/data/lerobot \
  --resume
```

`--episodes` is retained only so old commands fail safely: its value is
ignored. The session runs until the operator chooses one of the two exit keys.

## Episode decision loop

`RIGHT` and `LEFT` below mean the keyboard arrow keys, not mouse buttons.

| Key | Decision |
| --- | --- |
| `Right Arrow` | Accept the entire current candidate, commit it as one LeRobot episode, then load the next saved layout. |
| `Left Arrow` | Discard the entire current candidate and reload the same saved layout. |
| `Escape` | Accept and commit the current candidate, finalize the dataset, then close Isaac Sim and the policy server. |
| `Backspace` | Discard the current candidate, finalize all earlier accepted episodes, then close Isaac Sim and the policy server. |

An attempt does **not** end merely because the reward reports success, and it
does not have a task step timeout. This is deliberate: the operator can first
observe the complete outcome and only then decide whether it is useful training
data. A reset occurs only after one of the four decisions above, an invalid
simulation attempt, or an unrecoverable process failure.

After every accepted `Right Arrow` episode, the scheduler advances to the next
saved evaluation layout. Once the finite layout list is exhausted, it cycles
back to the first layout instead of exiting. The LeRobot episode metadata stores
both `layout_id` and `layout_cycle`, so repeated passes are distinguishable.
`Left Arrow` reloads the same saved layout id and cycle. Object poses and other
properties encoded in that saved layout are restored; renderer-level noise is
not promised to be bit-identical.

`layout_cycle` counts uninterrupted passes through the scheduler. After a
process-level PhysX recovery it may restart at zero; the monotonically appended
LeRobot `episode_index` and `robodojo_run_id` metadata still distinguish every
committed episode.

Legacy aliases remain available: `N`/`Enter` behave like `Right Arrow`, and `R`
accepts then retries the same layout. New collection sessions should use the
arrow/exit controls because their intent is clearer.

## Manual correction controls

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
| `L` | Emergency manual-control exit and clear held motion keys. |

Use plain `I`, without Ctrl or Shift. It is a latched switch; it does not need
to be held. The first press anchors the selected Cartesian target to the arm's
measured end-effector pose. The inactive arm holds its measured joints. The
second press requests a fresh Pi0.5 action chunk from the post-correction state.
`Space` is deliberately unused because Isaac Sim binds it to Play/Pause.

Pi0.5 normally returns a chunk of 50 joint targets (about two seconds at 25
Hz). If takeover starts at target `k`, target `k` and the remaining old chunk
are discarded. A failed, non-finite, or discontinuous IK result becomes a hold
action and is not marked as a valid human intervention.

## LeRobot v3 output

For the first command above, the dataset is:

```text
/home/piper/data/lerobot/robodojo_interventions_make_toast
```

Every committed episode contains the normal Pi0.5 training fields:

- dual-arm observation state and the actual 14-dimensional executed action;
- `cam_high`, `cam_left_wrist`, and `cam_right_wrist` video observations;
- the task instruction;
- `complementary_info.is_intervention`, with `1` only for a successful human
  IK action and `0` for policy or safety-hold control;
- the policy proposal and control-state metadata needed to distinguish policy,
  takeover, and hold frames;
- episode metadata including task, checkpoint, layout id/cycle, collection run
  id, success-at-decision, and repository revisions.

Kai0 can select only the corrective frames with the same column filter used by
its HIL datasets:

```python
column_filter_column = "complementary_info.is_intervention"
column_filter_keep_values = (1.0,)
```

Keep the full episodes on disk even when training uses this filter: the policy
context before and after each correction remains available for alternative
sampling strategies.

The simulator sends candidate frames to an independent child process running
the Pi0.5 `openpi/.venv`. That child hides CUDA and performs LeRobot video
encoding on CPU, so it does not allocate an additional GPU context or import
LeRobot's Torch stack into Isaac Sim. `--encoder-threads` controls CPU encoder
parallelism; `--lerobot-vcodec h264` is the default.

LeRobot streaming encoding writes each camera to temporary MP4 storage while
the attempt is running. State/action rows remain part of the current episode
buffer until the decision. Therefore `Right Arrow`/`Escape` mean **commit the
already collected trajectory**, not “start recording now.” `Left Arrow` and
`Backspace` clear the episode buffer and its temporary videos.

Each accepted episode is finalized before the next layout starts, and a new
writer resumes from the durable dataset metadata. This keeps all previously
accepted episodes usable if Isaac Sim, the policy connection, or the machine
later fails. A hard crash can lose only the in-progress, not-yet-accepted
candidate. Encoder frame drops, NaNs, communication errors, and ordinary
exceptions never turn a candidate into an accepted episode; the candidate is
discarded and the same layout is retried when the simulator is still healthy.

There is no rollout-duration limit. Separately, the recorder has a health
watchdog for an unresponsive CPU writer (300 seconds during startup and 180
seconds per frame/commit response by default); this detects a crashed or hung
sidecar rather than ending a valid episode based on task duration.

## Rendering and responsiveness

Keep `--rendering-mode quality` for normal policy inference and training data;
it best matches the benchmark camera distribution. `balanced` and
`performance` can improve keyboard responsiveness but change rendered images.
Tune translation/rotation increments with `--pos-step` and `--rot-step`.

If keyboard input stops after changing windows, click the Isaac Sim viewport.
Two seconds without keyboard events clears potentially stuck motion keys while
keeping the current takeover state latched.
