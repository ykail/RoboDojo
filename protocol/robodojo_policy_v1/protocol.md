# RoboDojo Policy Protocol v1

Status: initial contract. Transport client/server integration is implemented in
later milestones.

## Ownership

RoboDojo owns this wire contract and canonical observation/action schemas.
Policy repositories, including Kai0, implement adapters behind the contract.
RoboDojo must not import policy-specific model code.

Protocol identifier:

```text
robodojo-policy-v1
```

## Transport

- One binary msgpack map per WebSocket message.
- Encoded frames are limited to 64 MiB. The WebSocket transport and decoder
  must enforce the same limit.
- WebSocket ping/pong provides transport keepalive; there is no application
  heartbeat message.
- NumPy arrays use the same marker representation as
  `openpi_client.msgpack_numpy`.
- Object, structured/void, and complex NumPy dtypes are rejected.
- Application map keys are strings. Lists are used for protocol sequences;
  tuples are not accepted because msgpack would decode them as lists.
- Encoder and decoder enforce the same string, collection, nesting, NumPy rank,
  dimension, and element-count limits.
- Text frames are not protocol frames.

## Frame envelope

Every frame contains exactly these fields:

| Field | Type | Meaning |
|---|---|---|
| `protocol_version` | non-empty string | Must equal `robodojo-policy-v1` |
| `message_type` | enum string | Message name below |
| `request_id` | non-empty string | Correlation ID; responses echo it |
| `session_id` | non-empty string | New client-generated ID for each connection |
| `episode_id` | string or null | Unique rollout ID; presence is fixed below |
| `inference_index` | non-negative integer or null | Index of the inference request within an episode |
| `payload` | map | Message-specific body |

Unknown or missing top-level fields are rejected. Responses echo
`request_id`, `session_id`, `episode_id`, and `inference_index`.

Presence matrix:

| Messages | `episode_id` | `inference_index` |
|---|---|---|
| `HELLO`, `HELLO_ACK` | null | null |
| `RESET`, `RESET_RESULT` | required | null |
| `INFER`, `INFER_RESULT` | required | required |
| `TRIAL_END`, `TRIAL_END_ACK` | required | null |
| `ERROR` | echoes the valid triggering request | echoes the valid triggering request |

`inference_index` counts policy inference requests, not simulator control steps.
One inference can return an action chunk containing multiple control steps.
Canonical observation/action payloads must not duplicate envelope identity
fields; this envelope is authoritative.

## Message pairs

```text
HELLO      -> HELLO_ACK
RESET      -> RESET_RESULT
INFER      -> INFER_RESULT
TRIAL_END  -> TRIAL_END_ACK
valid request state/business error -> ERROR
```

`PREPARE_CASE` is not part of v1; task, simulator seed, and policy seed
metadata belongs in `RESET`.
Policy-specific `update_obs`/`get_action` method names are not part of the
wire API. Connection shutdown uses the WebSocket close control frame, not a
second application-level `CLOSE` message.

## Lifecycle

```text
connected
  -> HELLO
ready
  -> RESET(unique episode_id)
active(inference_index = 0)
  -> INFER(inference_index = expected)
active(inference_index += 1)
  -> ...
  -> TRIAL_END
ready
```

Rules:

- `INFER` before a successful `RESET` is rejected.
- `episode_id` must match the active episode.
- `inference_index` starts at zero and is strictly monotonic.
- The first server supports one active client/session.
- The first frame on a connection is exactly one `HELLO`. Its `session_id`
  becomes bound to that connection; every later frame must use the same ID.
- `request_id` is unique for the lifetime of a session.
- A RESET candidate `episode_id` is reserved for the lifetime of the session
  as soon as `begin()` accepts the RESET envelope in READY. Payload rejection
  never rolls that reservation back. Correcting a rejected RESET therefore
  requires both a fresh `request_id` and a fresh `episode_id`.
- The server atomically reserves its single global model lease when accepting a
  valid `HELLO`, before invoking any adapter operation. A second connection
  receives `session_busy`. The lease becomes owned after `HELLO_ACK` and is not
  released until transport shutdown, worker settlement, and adapter cleanup
  all finish.
- `RESET`, `INFER`, and `TRIAL_END` are serialized by one lifecycle lock.
- A synchronous inference worker cannot be made safe merely by cancelling its
  outer asyncio task. A connection permits only one outstanding request. To
  abandon an in-flight inference, the client closes that session; the server
  waits for the real worker to finish and discards its result. A new session's
  `RESET` cannot acquire the global lease until that drain and cleanup finish.
- After `HELLO` succeeds, an ambiguous send, timeout, or disconnect during
  `RESET`, `INFER`, or `TRIAL_END` invalidates the session. None of those
  operations may be replayed: a lost RESET acknowledgement can hide an already
  executed reset, just as a lost INFER response can hide an advanced RNG.
- Recovery creates a new connection/session and a new episode, resets the
  simulator, then sends a new explicit `RESET`.
- An active episode cannot be reset in place. A normal operator retry is
  `TRIAL_END(status="aborted")`, acknowledgement, simulator reset, then RESET
  with a new episode ID. A lost session follows the recovery rule above and
  does not fabricate a `TRIAL_END` result.

The client may retry only before the WebSocket/HELLO session is established
while a cold policy server is starting. It may not silently reconnect and
continue an established session or trajectory.

### Reference session state machine

RoboDojo keeps a transport- and model-independent executable reference in
`src/eval_client/policy_runtime/session.py`. Kai0 keeps its own implementation
and is checked against the same fixtures and black-box conformance tests; it
must not import its parent RoboDojo checkout at runtime.

```text
AWAITING_HELLO -> READY -> ACTIVE -> READY
       |            |        |
       +------------+--------+-- pending + disconnect --> DRAINING --> CLOSED
backend failure ---------------------------------------> TERMINATING --> CLOSED
```

Request processing is two-phase: `begin(frame)` atomically validates and
reserves IDs, then `complete(token)` or `fail(token)` settles that exact
operation. A disconnect with an in-flight synchronous worker enters
`DRAINING`; the worker result is discarded, and cleanup may release the global
model lease only after the worker has actually settled. The internal operation
generation is not a wire field.

The reference state machine does not implement the server-global model lease.
Kai0's dispatcher needs a separate lease owner token. It must reserve that token
before any HELLO-side adapter work, keep the real synchronous worker future
alive under outer-task cancellation, wait for it to settle, run adapter
cleanup, and only then release the lease. `waiting_for_operation: false` means
cleanup may start; it never means the lease is already safe to release.

Dispatcher ordering for one connection is:

1. `begin(frame)` before any backend side effect.
2. Parse the request payload. If it is invalid, call `reject(token)`: the
   request ID stays burned, but RESET/INFER/TRIAL_END state does not advance.
   INFER/TRIAL_END corrections use a fresh request ID and retain the active
   episode ID. A RESET correction must also use a fresh candidate episode ID,
   because `begin()` reserved the rejected candidate before payload parsing.
3. Start and retain the real backend future; shield it from outer-task
   cancellation.
4. Validate backend output, then call `complete(token)` only for valid output.
   Backend failure or invalid output calls `fail(token)` because policy RNG or
   state may already have advanced.
5. If `reply_allowed`, attempt the one correlated encode-and-send before
   submitting the next frame from that connection. Do not spawn detached
   per-frame dispatch tasks and do not retry the same reply.
6. Send failure, send cancellation, or handler cancellation must call
   `disconnect()` in `finally`. If `close_after_reply`, close the WebSocket and
   call `disconnect()` whether the reply succeeded or failed.
7. On earlier transport loss, call `disconnect()` immediately, continue
   draining the retained worker, settle its token, run adapter cleanup, and
   finally release the global lease.

`episode_lost` describes server-side episode state that may still need abort or
cleanup; it does not assert that the client received an acknowledgement. For
example, after `TRIAL_END` completes, a lost ACK closes the session but does not
make the already-ended server episode active again.

Only a fully parsed v1 request can receive an application `ERROR` response.
Malformed envelopes, unknown protocol versions, missing correlation fields, and
text frames are closed at the WebSocket protocol boundary because the server
cannot safely construct a correlated v1 response.

Error-code categories:

- Protocol-close errors: `invalid_frame`, `unsupported_version`,
  `unknown_message_type`.
- Correlated server errors: `invalid_payload`, `invalid_state`, `session_busy`,
  `episode_mismatch`, `inference_index_mismatch`, `infer_failed`,
  `reset_failed`, `internal`.
- Client-local failures: `episode_lost`, `timeout`.

An unsupported but otherwise well-formed HELLO profile is a correlated
`invalid_payload` error with constraint-mismatch details. Because a rejected
HELLO cannot establish a session, the server sends that one ERROR if possible
and then closes the connection.

`PayloadValidationError` is deliberately direction-neutral rather than a
`ProtocolError`. The dispatcher maps an invalid inbound observation to
`invalid_payload` plus `reject(token)`, but maps an invalid policy action to
`infer_failed` plus `fail(token)`. A RoboDojo client receiving an invalid
`INFER_RESULT` closes the session and marks the episode lost. This distinction
prevents an already-advanced policy from being treated as if no backend work
occurred.

## Payload direction

- `HELLO`: one fixed observation/action/execution profile assertion.
- `HELLO_ACK`: an exact semantic confirmation plus policy/config/checkpoint
  provenance.
- `RESET`: task, simulator seed, policy seed, reset reason, and episode
  metadata.
- `INFER`: exactly `{"observation": <canonical observation>}`.
- `INFER_RESULT`: exactly `{"action": <canonical action chunk>}`.
- `TRIAL_END`: success/failure/aborted/error status, `score`, and `reason`.
- `ERROR`: stable error code, message, details, and `retryable: false` by
  default.

The `INFER` and `INFER_RESULT` wrappers are exact maps: v1 has no sibling
diagnostics or latency field. A later protocol version may add a typed
diagnostics object without making v1 implementations guess which fields to
ignore.

Adding an exact-map field, execution feedback, or new lifecycle meaning
requires a new protocol/schema version. Replacing a Kai0 implementation branch
or checkpoint while preserving this complete contract does not; HELLO_ACK
provenance identifies the actual code and artifacts.

### Lifecycle payloads

All lifecycle payloads are exact maps. `HELLO.payload` is:

```python
{
    "schemas": {
        "observation": "robodojo-arx-x5-dual-rgb-joint-v1",
        "action": "robodojo-arx-x5-dual-absolute-joint-position-v1",
        "robot": "arx_x5_dual_v1",
    },
    "execution_profile": {
        "images": {
            "head": [480, 640, 3],
            "left_wrist": [480, 640, 3],
            "right_wrist": [480, 640, 3],
        },
        "action": {
            "horizon": 50,
            "control_dt_s": 0.04,
            "chunk_consumption":
                "sequential_until_terminal_or_preempted",
            "preemption_boundary": "between_control_targets",
            "next_infer_observation": "after_last_executed_target",
            "left_arm_joint_limits": {
                "lower": [-10, -10, -10, -10, -10, -3.14],
                "upper": [10, 10, 10, 10, 10, 3.14],
            },
            "right_arm_joint_limits": {
                "lower": [-10, -10, -10, -10, -10, -3.14],
                "upper": [10, 10, 10, 10, 10, 3.14],
            },
        },
    },
}
```

The shape and limit sequences are wire lists, not tuples. `HELLO_ACK.payload`
must echo `schemas` and `execution_profile` exactly, then add:

```python
{
    "policy": {
        "implementation": "kai0",
        "policy_family": "pi05",
        "adapter_profile": "kai0_pi05_aloha_arx_x5_joint_v1",
        "config_name": str,
        "checkpoint_id": str,
        "checkpoint_digest": "sha256:<64 lowercase hex characters>",
        "checkpoint_step": int | None,
        "code_revision": "<40 lowercase Git hex characters>",
        "dirty": bool,
    },
}
```

The displayed arm limits are the current released ARX X5 simulation asset
limits. They are intentionally named and implemented as
`ARX_X5_SIM_ARM_LIMITS`; they are not a claim about safe physical X5 hardware
limits.

`checkpoint_id` is a portable human-readable name or URI, not a host-local
absolute path. `checkpoint_digest` is the SHA-256 of the deterministic
effective artifact manifest and is the machine identity of the loaded
checkpoint. The manifest covers every regular file under the effective
`params/` and `assets/` trees, including normalization statistics, plus a
top-level `model.safetensors` when present. Any auxiliary artifact that can
change inference is included under a stable named role. Symbolic links inside
artifact roots are rejected.

The manifest wire bytes use `robodojo-artifact-manifest-v1`:

1. The first UTF-8 JSONL line is exactly
   `{"format":"robodojo-artifact-manifest-v1"}` followed by LF.
2. Each file entry has exactly `path`, `role`, `sha256`, and `size`. `path` is
   its slash-separated UTF-8 path relative to that role root, without empty,
   `.` or `..` components. The `params/` tree uses role `params`; `assets/`
   uses `assets`; top-level `model.safetensors` uses role `model` and path
   `model.safetensors`. Auxiliary roots use `aux:<portable-role-id>`.
   `(role,path)` pairs are unique. `sha256` is 64 lowercase hexadecimal
   characters for the file bytes; `size` is the byte length.
3. Entries sort component-wise by
   `(role.encode("utf-8"), path.encode("utf-8"))`.
4. Each entry is encoded as UTF-8 JSON with keys sorted lexicographically,
   `ensure_ascii=false`, no insignificant whitespace, then one LF. In Python
   this is `json.dumps(entry, ensure_ascii=False, sort_keys=True,
   separators=(",", ":")) + "\n"`.
5. `checkpoint_digest` is `"sha256:" + sha256(all JSONL bytes).hexdigest()`.

The adapter persists those exact JSONL bytes beside evaluation logs so another
implementation can recompute the identity. `checkpoint_step` is display
metadata and never substitutes for the digest. `code_revision` is the full
Kai0 commit SHA. Formally comparable runs require `dirty=false`; branch and
worktree names are not identity.

`RESET.payload` is:

```python
{
    "task_name": str,
    "simulator_seed": int,     # non-negative simulator/layout RNG seed
    "policy_seed": int,        # [0, 2^32 - 1], episode sampling RNG seed
    "layout_id": int,          # non-negative
    "layout_cycle": int,       # non-negative
    "reason": "episode_start"
              | "operator_retry"
              | "simulator_recovery"
              | "transport_recovery",
}
```

RESET has stateful adapter semantics. Before `RESET_RESULT`, Kai0 must clear
all episode-local observation/history/memory/subtask/action-buffer state and
initialize the sampler from `policy_seed`; it must not reload weights, assets,
or compilation caches. `simulator_seed` records the independently chosen
RoboDojo scene seed. `task_name`, seeds, layout fields, and reason are
lifecycle context and logging metadata; they do not replace the canonical
observation instruction. Each INFER observation's `instruction` is the
authoritative prompt for that inference.

`TRIAL_END.payload` is:

```python
{
    "status": "success" | "failure" | "aborted" | "error",
    "success": bool | None,
    "score": finite_float | None,
    "reason": nonempty_str | None,
}
```

The outcome is unambiguous: `success` status requires `success=True`,
`failure` requires `False`, and `aborted`/`error` require `None`.
`RESET_RESULT.payload` and `TRIAL_END_ACK.payload` are exact empty maps.

TRIAL_END records the outcome and clears all episode-local adapter state
before acknowledgement. It cannot update weights, learn online, or retain
episode state for a later RESET in v1. Transport loss during an active episode
performs the same abort cleanup after any retained worker settles, but records
no fabricated trial outcome. `success` means task success; `failure` means a
normal terminal condition without success; `aborted` means an operator/client
ended the trial; and `error` means simulator/client execution failed.

One correlated `ERROR.payload` is:

```python
{
    "code": correlated_server_error_code,
    "message": nonempty_str,
    "details": {str: any, ...},
    "retryable": False,
}
```

`retryable=False` means the same request is never replayed. An
`invalid_payload` response can leave the session usable for a newly constructed
request with a fresh `request_id`; it does not authorize replay.
`details` is a recursively bounded JSON-like map: string keys and values made
from null, bool, msgpack-range integers, finite floats, strings, maps, and
lists. Tuples, bytes, NumPy values, and arbitrary Python objects are not error
details. Reference objects recursively snapshot it before an ERROR is queued.

`src/eval_client/policy_runtime/lifecycle_payloads.py` is the executable
reference for these maps. A server calls `parse_hello_payload()` with its
concrete `supported_profile`; merely parsing and echoing an arbitrary
well-formed client profile is not a valid capability check.

## Canonical ARX X5 joint schemas

The fixed v1 schema pair is:

```text
observation: robodojo-arx-x5-dual-rgb-joint-v1
action:      robodojo-arx-x5-dual-absolute-joint-position-v1
robot:       arx_x5_dual_v1
```

One logical observation is:

```python
{
    "instruction": str,
    "images": {
        "head": uint8[H, W, 3],         # RGB, HWC
        "left_wrist": uint8[H, W, 3],
        "right_wrist": uint8[H, W, 3],
    },
    "proprio": {
        "robot_schema": "arx_x5_dual_v1",
        "left_arm_joint_position": float32[6],   # joint1..joint6, rad
        "left_gripper_open_fraction_commanded": float32[1],
        "right_arm_joint_position": float32[6],
        "right_gripper_open_fraction_commanded": float32[1],
    },
}
```

The current released ARX X5 simulation profile requires all three images to be
exactly `uint8[480, 640, 3]`. The general schema annotation names symbolic
`H/W` so future profiles can select another exact size, but the
RoboDojo-owned `ObservationValidationSpec` always supplies and enforces the
three concrete shapes for a connection. Kai0 confirms the asserted profile in
`HELLO_ACK`; it does not choose the simulator camera shape. Exact confirmation
means equality of normalized field values, not byte-for-byte equality of
independently encoded msgpack frames.
The three uncompressed images together must fit a 63 MiB canonical image
budget, leaving at least 1 MiB inside the 64 MiB frame limit for the envelope,
instruction, proprioception, and msgpack metadata.

The arm positions are measured. RoboDojo's current raw gripper observation is
the previous normalized, post-rate-limited target sent by its control manager,
not a measured gripper joint or necessarily the raw policy prediction, so v1
names it explicitly. Gripper fraction is dimensionless: zero is closed and one
is open.

RoboDojo maps its raw observation to canonical fields as follows:

| RoboDojo raw field | Canonical field |
|---|---|
| `vision.cam_head.color` | `images.head` |
| `vision.cam_left_wrist.color` | `images.left_wrist` |
| `vision.cam_right_wrist.color` | `images.right_wrist` |
| `state.left_arm_joint_state` | `proprio.left_arm_joint_position` |
| `state.left_ee_joint_state` | `proprio.left_gripper_open_fraction_commanded` |
| `state.right_arm_joint_state` | `proprio.right_arm_joint_position` |
| `state.right_ee_joint_state` | `proprio.right_gripper_open_fraction_commanded` |

Raw `action`, EE pose, depth, `env_idx`, camera `shape`,
`data_format_version`, `additional_info.frequency`, and simulator object truth
are deliberately not part of this visual-policy observation.

The executable projection is
`src/eval_client/policy_runtime/observation_builder.py`. One
`ArxX5ObservationBuilder` instance binds a concrete image profile and expected
`env_idx` for a connection. It also requires raw
`data_format_version == "v1.0"` before dropping both local metadata fields.
Images are passed through without cast, resize, transpose, or channel
conversion. Measured floating arm arrays and normalized floating gripper
commands are converted explicitly to `float32`; the canonical parser then
performs the single immutable snapshot. Builder failures always surface as
`RawObservationBuildError` with a raw-source path; when the final canonical
validator found the problem, its `PayloadErrorKind` is retained as
`canonical_error_kind` and the original exception remains chained.

One logical action chunk is:

```python
{
    "control_mode": "absolute_joint_position",
    "control_dt_s": 0.04,
    "commands": {
        "left_arm_joint_position": float32[T, 6],  # rad
        "left_gripper_open_fraction": float32[T, 1],
        "right_arm_joint_position": float32[T, 6],
        "right_gripper_open_fraction": float32[T, 1],
    },
}
```

All four command arrays share horizon `T`; every floating value is finite.
Gripper commands are strictly in `[0, 1]`. Active RoboDojo robot limits, not
policy metadata, constrain arm commands. The RoboDojo execution profile owns
the concrete horizon, control period, camera shapes, and arm limits. The server
may only confirm an exact match during HELLO; it cannot weaken these execution
constraints. For the released Pi0.5 profile, `T=50` and
`control_dt_s=0.04` (25 Hz simulated targets), so one full chunk represents
2.0 seconds of simulated targets.

The released v1 consumer is open-loop, not receding-horizon or temporal
ensemble execution. After validating one INFER_RESULT, RoboDojo owns the
chunk locally and normally executes target indices `0..49` once, in order. A
task terminal, operator takeover, executor error, or episode-ending intent may
preempt only between synchronous control targets. The currently entered target
finishes; the unexecuted suffix is permanently discarded and is never resumed,
replayed, or carried into another episode/session. `control_dt_s` is simulated
time between executed targets, not a wall-clock response deadline.

The next INFER, if any, uses a fresh observation obtained after the final
actually executed target. The server must not assume that the previous chunk
was fully consumed. INFER counts policy calls, not executed targets. A policy
that requires executed-step counts, previous-chunk feedback, per-target
observations, RTC, or overlapping chunks requires a later protocol/profile;
those fields cannot be added to v1's exact INFER wrapper. Before TRIAL_END or
recovery RESET, the client stops its local executor and clears every stale
suffix. Once TRIAL_END is sent, no target from that episode may execute.

The existing Pi0.5/Aloha output adapter packs one target as
`left_arm[0:6], left_gripper[6], right_arm[7:13], right_gripper[13]`.
That model-side packing is not exposed on the wire. The Kai0 adapter unpacks it
into the structured fields above. The legacy execution path clips every finite
model gripper prediction inside RoboDojo's executor. In the new architecture,
Kai0 deliberately performs an equivalent compatibility clip to `[0, 1]`
before constructing the strict canonical action, and records the clip count
and maximum overshoot in server logs. Non-finite predictions are rejected. The
canonical parser itself never clips.

The executable Kai0 mapping is the versioned
[`kai0_pi05_aloha_arx_x5_joint_v1` adapter profile](adapter_profiles/kai0_pi05_aloha_arx_x5_joint_v1.md).
An implementation using the same wire schemas but a different image/state
mapping or sampling setup must publish a new adapter profile ID.

The JSON schema files document exact maps and NumPy annotations. Standard JSON
Schema cannot enforce ndarray dtype, layout, symbolic dimensions, or asserted
client-profile constraints, so
`src/eval_client/policy_runtime/canonical.py` is the executable runtime
boundary. Its parsers make immutable, C-contiguous snapshots; they do not
transpose, cast, resize, normalize, fill missing cameras, or clip values.

## Cross-end codec fixture

`fixtures/numpy_frame_v1.hex` was generated with Kai0
`openpi_client.msgpack_numpy` at Kai0 commit
`5c2645f5a2a5a005d43efafa42f338a14dc7442c`. It contains a nested uint8 image,
float32 state, and NumPy scalar. Both repositories must decode this fixture and
must reproduce the same bytes when encoding its logical frame.
