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

`PREPARE_CASE` is not part of v1; task/seed metadata belongs in `RESET`.
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
- A session owns the model from successful `HELLO` until its WebSocket is
  released. A disconnected session with a still-running synchronous inference
  worker continues to own it; another `HELLO` receives `session_busy` until the
  worker has actually settled and cleanup finishes.
- `RESET`, `INFER`, and `TRIAL_END` are serialized by one lifecycle lock.
- A synchronous inference worker cannot be made safe merely by cancelling its
  outer asyncio task. Reset waits for the worker to actually finish and then
  discards obsolete results before policy state is reset.
- After `HELLO` succeeds, an ambiguous send, timeout, or disconnect during
  `RESET`, `INFER`, or `TRIAL_END` invalidates the session. None of those
  operations may be replayed: a lost RESET acknowledgement can hide an already
  executed reset, just as a lost INFER response can hide an advanced RNG.
- Recovery creates a new connection/session and a new episode, resets the
  simulator, then sends a new explicit `RESET`.

The client may retry only before the WebSocket/HELLO session is established
while a cold policy server is starting. It may not silently reconnect and
continue an established session or trajectory.

Only a fully parsed v1 request can receive an application `ERROR` response.
Malformed envelopes, unknown protocol versions, missing correlation fields, and
text frames are closed at the WebSocket protocol boundary because the server
cannot safely construct a correlated v1 response.

Error-code categories:

- Protocol-close errors: `invalid_frame`, `unsupported_version`,
  `unknown_message_type`.
- Correlated server errors: `invalid_state`, `session_busy`,
  `episode_mismatch`, `inference_index_mismatch`, `infer_failed`,
  `reset_failed`, `internal`.
- Client-local failures: `episode_lost`, `timeout`.

## Payload direction

- `HELLO`: requested observation/action schema IDs and client capabilities.
- `HELLO_ACK`: policy/config/checkpoint provenance and supported schemas.
- `RESET`: task, seed, reset reason, and episode metadata.
- `INFER`: one canonical observation.
- `INFER_RESULT`: one canonical action chunk and optional diagnostics/latency.
- `TRIAL_END`: success/failure/aborted/error status and optional metrics.
- `ERROR`: stable error code, message, details, and `retryable: false` by
  default.

Canonical observation/action schemas and the stateful session implementation
are separate milestones.

## Cross-end codec fixture

`fixtures/numpy_frame_v1.hex` was generated with Kai0
`openpi_client.msgpack_numpy` at Kai0 commit
`5c2645f5a2a5a005d43efafa42f338a14dc7442c`. It contains a nested uint8 image,
float32 state, and NumPy scalar. Both repositories must decode this fixture and
must reproduce the same bytes when encoding its logical frame.
