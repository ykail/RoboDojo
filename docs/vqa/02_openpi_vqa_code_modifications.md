# VQA Support in openpi π0.5 (Implementation Reference)

This document describes the VQA support as **actually implemented** in this
repository (branch `vqa-xjx`). It is the companion to
`01_vqa_data_format.md` (the data contract) and `04_make_kong_vqa_examples.md`
(the make_kong task spec).

## 0. Scope

The implementation adds ego-view VQA/text-generation training to π0.5 while
preserving the existing continuous flow-matching action path.

- Two task types: `action` and `vqa`.
- Seven answer types: `point2d`, `bbox2d`, `boolean`, `short_text`,
  `integer`, `int_list`, `bbox_list`. `int_list`/`bbox_list` are
  variable-size lists; the empty list is the `none` answer.
- Only two losses: `L_action` (existing flow-matching loss) and `L_text`
  (autoregressive token cross-entropy). No coordinate-regression loss.
- Task-specific semantics never enter core model code.

## 1. Data contract summary

Read `01_vqa_data_format.md` for the full contract. Key points:

- Bounding-box storage is **yxyx** (`y_min, x_min, y_max, x_max`), identical
  to the model token order — no reordering at serialization. Point storage is
  **yx** (`[y, x]`), also identical to the model token order — no reordering
  at serialization.
- Location tokens use the pretrained PaliGemma vocabulary, single tokens
  `<loc0000>`..`<loc1023>` (zero-padded, **no underscore**):
  `q = round(clamp(value, 0, 1) * 1023)`.
- The serialized answer is plain text over the pretrained vocabulary: no
  custom markers (`<mode_vqa>`, `<answer_*>`, `<sep>`, `<none>`) exist.
  List answers join elements with `;`; the empty list serializes to the
  plain word `none`.
- List element order is **preserved from the canonical target**: the question
  prompt dictates ordering (e.g. make_kong "left-to-right"), so the data is
  pre-sorted by the collector. The serializer never re-sorts.
- A record is trainable only when `world_state_valid` and `image_answerable`
  are both true.

## 2. Module layout (actual)

```text
src/openpi/
├── vqa/                          # task-independent VQA core
│   ├── answer_types.py           # SampleType, AnswerType, CanonicalTarget, CanonicalSample
│   ├── serializer.py             # AnswerSerializer (+ VQAFormatConfig, location tokens)
│   ├── parser.py                 # AnswerParser (strict inverse), VQAResult
│   ├── metrics.py                # evaluate_answer per answer type
│   └── validation.py             # CanonicalSampleValidator
├── training/
│   ├── vqa_dataset.py            # RoboDojoVQADataset (Parquet -> CanonicalSample -> model dict)
│   ├── data_loader.py            # create_vqa_data_loader, PairedDataLoader, task-mode routing
│   └── config.py                 # VQATaskConfig, TaskMode, RoboDojoActionVQADataConfig, _CONFIGS
├── models/
│   ├── tokenizer.py              # PaligemmaTokenizer.tokenize_vqa (prompt protocol + masks)
│   └── pi0_rtc.py                # _vqa_forward, _compute_vqa_loss, compute_vqa_metrics,
│                                 #   generate_vqa(_batch), loss routing, image masking
├── policies/policy.py            # Policy.infer_vqa(_batch)
└── serving/websocket_policy_server.py  # request_type routing: vqa / vqa_batch
```

## 3. VQA prompt protocol

`tokenize_vqa` builds:

```text
Question: {question}
Answer: {serialized_answer}<eos>
```

- The protocol is plain text only: no mode/answer-type markers and no robot
  state. `answer_type` is passed separately by callers and never appears in
  the token stream.
- During teacher-forced training the serialized answer and `<eos>` are
  appended; only those positions are supervised.

## 4. Answer serialization

One serializer (`serialize_answer(target, config) -> SerializedAnswer`):

| type | storage field | serialized |
| --- | --- | --- |
| `point2d` | `point_yx_norm` `[y, x]` | `<loc{y}><loc{x}>` (no reorder) |
| `bbox2d` | `bbox_yxyx_norm` `[ymin, xmin, ymax, xmax]` | `<loc..><loc..><loc..><loc..>` (no reorder) |
| `boolean` | `boolean` | `yes` / `no` |
| `short_text` | `text` | normalized canonical text (lowercased) |
| `integer` | `integer` | base-10 digits |
| `int_list` | `int_list` | `2;5;8`; empty list -> `none` |
| `bbox_list` | `bbox_list_yxyx_norm` `(N, 4)` | `<loc..><loc..><loc..><loc..>;...`; empty -> `none` |

- Quantization: `q = min(bins-1, max(0, round(value * (bins-1))))`, bins=1024.
- Coordinates are normalized to the original ego image.
- The parser is the strict inverse and rejects: missing/wrong number of
  location tokens, tokens outside the vocabulary, unexpected text or
  answer-type markers in spatial answers, reversed/degenerate boxes, malformed
  integers, unknown booleans, empty short text, dangling `;` (trailing
  separator), and unterminated list elements. It never repairs malformed
  output.

## 5. Masking

### 5.1 Prompt

The VQA prompt is fully valid: every prompt token participates in attention
and none is supervised. There is no reserved state span and no interior
invalid region.

### 5.2 Images

Fixed visual slots `base_0_rgb` (ego), `left_wrist_0_rgb`,
`right_wrist_0_rgb` shared by both tasks. For VQA only the ego slot is valid:
`image_masks = {"base_0_rgb": True, "left_wrist_0_rgb": False,
"right_wrist_0_rgb": False}` with zero-filled wrist tensors. Masked image
positions are excluded from attention, so wrist content cannot influence VQA
logits.

### 5.3 Labels

Only answer tokens and `<eos>` are supervised. Prompt and padding positions
are masked out of the loss.

## 6. Loss routing, augmentation and metrics

- `compute_loss` routes per batch: with a `vqa_answer` output configured, an
  action batch computes `action/loss` only; a VQA batch computes
  `vqa/text_loss` only (teacher-forced next-token CE over the answer span,
  denominator = number of supervised positions). Action flow-matching math is
  unchanged from the action-only implementation.
- Geometry-aware augmentation (random crop 0.95, rotation ±5°, horizontal
  flip, color jitter) is applied in the dataset adapter
  (`RoboDojoVQADataset.__getitem__`), where image and coordinates are
  transformed by the identical mapping; border boxes that would collapse fall
  back to the un-augmented sample. The model forward bypasses augmentation
  for VQA batches (no double augmentation).
- Metrics: `vqa/text_loss`, `vqa/token_acc`, `vqa/exact`, `vqa/answer_len`
  during training; `scripts/eval_vqa.py` reports per-answer-type and
  per-question-family: exact match, absolute error, normalized L2, IoU,
  and for lists sequence/count exact match, per-position accuracy, edit
  distance, mean IoU, IoU@0.5.

## 7. Batching

- Batches are homogeneous (one task type per batch).
- `task_mode`: `action_only`, `vqa_only`, `action_vqa`.
- `action_vqa` uses `PairedDataLoader`: every global step draws one action
  batch and one VQA batch and sums their gradients into a single optimizer
  step (fixed 1:1; no per-step task scheduler and no ratio configuration).
  `vqa_only` and `action_only` use the single-task loader.
- Resume positions loaders via `set_global_step`: the seeded shuffle
  (`_EpochShuffleIndexSampler`) draws one deterministic permutation per epoch
  from a dedicated generator, so a resumed run continues the exact data order
  of the interrupted run. Loaders built with an explicit sampler (e.g.
  frame-level weighting) restart from the beginning.

## 8. Generation and serving

- `Pi0RTC.generate_vqa(_batch)`: greedy decoding, `<eos>` termination,
  configurable `max_answer_tokens` (default 64), then strict typed parsing.
- Spatial results are mapped back to **original-image normalized coordinates**
  by the inverse letterbox (`_inverse_letterbox_result`), so
  `Policy.infer_vqa(_batch)` returns the same coordinate frame as the data
  contract (`01_vqa_data_format.md`). Non-spatial answers are unaffected.
- `Policy.infer_vqa(_batch)` validates uint8 ego images and returns
  `dataclasses.asdict(VQAResult)`; malformed outputs return `valid=false`
  with an explicit `error`.
- The websocket server dispatches on `request_type`: `vqa` and `vqa_batch`
  messages are routed to the VQA path; legacy action messages are unchanged.
- Serving a VQA policy without VQA: `--disable-vqa` strips the `vqa_answer`
  output (the head reuses the shared PaliGemma embedding, so checkpoint
  params are unchanged).

## 9. Configuration

Training configs (in `_CONFIGS`):

```text
pi05_rtc_robodojo_action_only   task_mode=action_only   data/make_kong/demo
pi05_rtc_robodojo_vqa_only      task_mode=vqa_only      + data/RoboDojo_vqa_v3/make_kong_seed2810
pi05_rtc_robodojo_action_vqa    task_mode=action_vqa    + data/RoboDojo_vqa_v3/make_kong_seed2810
```

Model: Pi0RTC π0.5, `max_token_len=200`, action_dim 32, action_horizon 50,
`vqa_answer` AR spec with `max_tokens=64`. Action data uses delta joint
actions with `meta/stats.json` normalization
(`RoboDojoActionVQADataConfig`). `VQATaskConfig` exposes the VQA root and
`num_location_bins` (default 1024); answer-type vocabulary lives in
`openpi.vqa.answer_types`, never in config.

## 10. Known limitations (not bugs, by design)

- `PairedDataLoader` is a fixed 1:1 action/VQA joint training mode with no
  configurable weighting or cross-rank synchronization (single-process
  multi-device).
- Legacy v1/v2 VQA datasets (xyxy storage, five answer types) are not
  supported; only the v3 contract is read.
