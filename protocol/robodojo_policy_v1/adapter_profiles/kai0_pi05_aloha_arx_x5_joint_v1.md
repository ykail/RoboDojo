# Kai0 Pi0.5 Aloha/ARX X5 joint adapter profile v1

Profile identifier:

```text
kai0_pi05_aloha_arx_x5_joint_v1
```

This profile fixes the conversion between RoboDojo's canonical ARX X5 wire
payloads and the Kai0 Pi0.5 Aloha transform stack used by the released
RoboDojo simulation checkpoint. The conversion belongs to Kai0. RoboDojo does
not import Kai0 model or transform code.

## Startup contract

The adapter refuses to serve unless the loaded model and effective TrainConfig
have all of these properties:

- Pi0.5 action horizon is 50 and output action dimension is at least 14.
- The data transform is the RoboDojo Aloha joint transform:
  `adapt_to_pi=False` and `use_delta_joint_actions=True`.
- The output transform returns absolute joint targets after undoing the delta
  action transform.
- Effective sampling is explicitly `num_steps=10`; callers cannot override it
  per INFER.
- Effective normalization assets are part of the checkpoint artifact manifest
  reported by `checkpoint_digest`.
- The server supports
  `ARX_X5_SIM_PI05_PROFILE` exactly.

Changing any mapping, transform flag, sampling parameter, or required model
shape requires a new adapter profile ID.

## Canonical observation to Kai0 input

The canonical observation is already validated and immutable. The adapter
creates a new Kai0 input with no simulator-only fields:

```python
{
    "images": {
        "cam_high": CHW_uint8(canonical.images.head),
        "cam_left_wrist": CHW_uint8(canonical.images.left_wrist),
        "cam_right_wrist": CHW_uint8(canonical.images.right_wrist),
    },
    "state": float32[
        left_arm[0:6],
        left_gripper_commanded,
        right_arm[0:6],
        right_gripper_commanded,
    ],
    "prompt": canonical.instruction,
}
```

Each canonical image is HWC `uint8[480,640,3]`. `CHW_uint8` is an explicit
`(2,0,1)` transpose followed by a C-contiguous immutable snapshot; it performs
no resize, normalization, color conversion, or cast. This transpose is
required because Kai0 `AlohaInputs` expects CHW and performs its own CHW-to-HWC
conversion internally. Passing canonical HWC directly would silently swap the
axes.

The 14-dimensional state is a C-contiguous `float32` snapshot in the exact
order shown. Gripper entries are normalized commanded open fractions in
`[0,1]`; they are not measured finger joints. `prompt` comes from every
canonical INFER observation and may change between inference calls.

## Kai0 output to canonical action

Kai0 inference runs in the Kai0 process. After its configured output transforms
have converted delta predictions back to absolute Aloha-space targets, the
adapter requires `actions` to have shape `[50,D]` with `D >= 14`, selects the
first 14 columns, and explicitly converts them to a C-contiguous `float32`
snapshot:

```text
0:6   left arm joint1..joint6, rad
6     left gripper open fraction
7:13  right arm joint1..joint6, rad
13    right gripper open fraction
```

Every value must be finite. Finite gripper predictions are compatibility
clipped to `[0,1]` before constructing the canonical payload; the server logs
the number of clipped values and maximum overshoot. Arm targets are never
clipped. A target outside the client-owned simulation limits is an invalid
policy result, maps to `infer_failed`, and terminates the session.

The adapter then constructs
`robodojo-arx-x5-dual-absolute-joint-position-v1` with
`control_dt_s=0.04`. RoboDojo validates the result again before executing it.

## Episode lifecycle and RNG

`reset_episode(context)` is synchronous and completes before RESET_RESULT. It
clears every episode-local wrapper state and initializes the Pi0.5 sampler from
the RESET `policy_seed`. JAX uses an episode-local JAX key; another backend
must provide the same episode-level reset guarantee and report its scheme in
server logs. Model weights, normalization assets, and compilation caches
remain loaded.

`finish_episode(outcome)` clears episode-local state for success, failure,
abort, and error without online learning. Disconnect cleanup has the same state
clearing guarantee after an in-flight worker drains. A wrapper for subtask,
memory, history, RTC, or an action buffer must explicitly propagate these
lifecycle calls to all children; an inherited no-op `reset()` is not enough.

The server receives one observation per INFER and must not assume the prior
50-target chunk was fully executed. v1 provides no per-target feedback. The
client follows the chunk consumption and preemption semantics in the parent
protocol.

## Conformance sentinels

At minimum, both repositories test:

- a non-square `uint8[480,640,3]` image with axis-dependent sentinel values,
  proving HWC-to-CHW conversion and preventing a silent double transpose;
- a 14-value state sentinel proving
  `[left6,left_gripper,right6,right_gripper]`;
- a `[50,14]` action whose columns are `0..13`, proving the inverse split;
- gripper finite clipping versus non-finite rejection;
- arm limit rejection without clipping;
- identical first INFER output for identical observation, checkpoint, and
  `policy_seed` after independent RESETs.
