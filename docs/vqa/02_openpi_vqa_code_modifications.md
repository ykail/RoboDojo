# Universal openpi π0.5 Code Modification Instruction for VQA

## 0. Role and objective

Modify the open-source `openpi` codebase to add ego-view VQA/text-generation training to π0.5 while preserving the existing continuous flow-matching action path.

Read together with:

```text
01_vqa_data_format.md
```

Task-specific VQA semantics must not be hard-coded into core model code.

The implementation supports two training task types:

```text
action
vqa
```

VQA answer types:

```text
point2d
bbox2d
boolean
short_text
integer
```

---

## 1. Non-negotiable architecture

Preserve:

```text
existing π0.5 continuous flow-matching action expert
existing action normalization
existing action horizon
existing action dimensionality
existing action-loss semantics
```

Add:

```text
shared visual-language backbone
├── autoregressive LM output path for VQA
└── existing flow-matching action path
```

Do not tokenize actions.

Do not replace the flow head with an autoregressive action decoder.

The only losses are:

```text
L_action = existing flow-matching loss
L_text   = autoregressive token cross-entropy
```

Do not add coordinate regression loss in the initial implementation.

---

## 2. Loss routing

For an action batch:

```text
total_loss = action_loss
text_loss_mask = all false
```

For a VQA batch:

```text
total_loss = text_loss
action_mask = all false
```

Inactive losses must not contribute gradients.

Prefer explicit task branching over computing an unnecessary inactive graph and multiplying the loss by zero.

Required task-level metrics:

```text
action/loss
vqa/text_loss
```

---

## 3. Canonical interfaces

Implement typed structures based on the data contract.

Recommended components:

```text
CanonicalSample
CanonicalTarget
UnifiedBatch
AnswerSerializer
AnswerParser
CanonicalSampleValidator
ActionDatasetAdapter
VQADatasetAdapter
UnifiedTaskCollator
SynchronizedTaskBatchScheduler
```

Adapt names to repository conventions.

Do not pass physical Parquet rows or LeRobot-native records directly into model code.

---

## 4. Dataset adapters

### 4.1 Action adapter

Responsibilities:

- retain existing LeRobot action sampling;
- retain action chunk and state semantics;
- map source camera keys to fixed slots;
- populate canonical action and state masks;
- emit `sample_type = action`;
- emit `target.answer_type = none`;
- emit no supervised text labels.

### 4.2 VQA adapter

Responsibilities:

- load only the ego RGB as a valid VQA image;
- load the question and typed answer;
- run universal and task-specific validation;
- generate masked wrist placeholders for static shapes;
- generate a zero-filled, fully masked state placeholder;
- generate a zero-filled, fully masked action placeholder;
- emit `sample_type = vqa`.

Required VQA values:

```python
image_masks = {
    "ego": True,
    "left_wrist": False,
    "right_wrist": False,
}

state_present = False
state = zeros([state_dim])
state_value_mask = false([state_dim])

actions = zeros([action_horizon, action_dim])
action_mask = false([action_horizon, action_dim])
```

---

## 5. Fixed visual slots

Use fixed visual slots shared by both tasks.

Recommended logical order:

```text
ego
left_wrist
right_wrist
```

Centralize source-key mapping in configuration.

For VQA:

```text
ego is valid
all wrist slots are invalid
```

Masked wrist tensors may exist only for static shape compatibility.

Their content must not influence VQA logits.

Required invariance test:

> Replace masked wrist-image tensors with random images. VQA logits must remain unchanged within tolerance.

---

## 6. VQA state masking

VQA does not semantically use robot state.

The required implementation masks state at all of the following levels:

1. state numeric-value mask;
2. state placeholder positions in the token sequence;
3. state positions as attention keys;
4. state positions as attention queries;
5. state positions in text loss;
6. state embeddings, where practical.

### 6.1 Do not serialize zero state as real state

Incorrect:

```text
State: 0 0 0 0 ...
```

when those values are normal state tokens.

Correct conceptual behavior:

```text
State: {reserved invalid state span}
```

The span preserves static shape or common code paths but carries no information.

### 6.2 Required masks

Produce at least:

```python
state_value_mask: bool[B, D]
state_token_mask: bool[B, L]
token_valid_mask: bool[B, L]
text_loss_mask: bool[B, L]
```

For VQA:

```text
state_value_mask = all false
state_token_mask[state_span] = false
token_valid_mask[state_span] = false
text_loss_mask[state_span] = false
```

### 6.3 Attention construction

For a two-dimensional attention mask:

```python
attention_allowed[q, k] &= token_valid_mask[q]
attention_allowed[q, k] &= token_valid_mask[k]
```

Then apply the repository's causal or prefix-LM pattern.

Later answer tokens must not attend to state-placeholder keys.

State-placeholder queries must not influence valid tokens.

Do not assume `labels = -100` is sufficient.

### 6.4 Mandatory state invariance test

For an identical VQA image and question:

1. create two different random state tensors;
2. apply the VQA state-mask path;
3. run the model in evaluation mode;
4. compare answer logits.

Expected:

```text
all valid-token logits are numerically identical within tolerance
```

This test is mandatory.

---

## 7. Prompt construction

Introduce a deterministic task-mode and answer-type protocol.

Recommended conceptual markers:

```text
<mode_action>
<mode_vqa>
<answer_point2d>
<answer_bbox2d>
<answer_boolean>
<answer_short_text>
<answer_integer>
```

Use tokenizer special tokens or stable textual markers, but do so consistently.

### 7.1 Action prompt

Conceptual form:

```text
<mode_action>
Task: {task_instruction}
State: {valid_discretized_state_tokens}
Action:
```

There is no supervised text target for action samples.

### 7.2 VQA prompt

Conceptual form:

```text
<mode_vqa>
Question: {question}
Expected answer type: <answer_type>
State: {reserved masked state span}
Answer:
```

During teacher-forced training, append:

```text
{serialized_answer}<eos>
```

---

## 8. Answer serializer

Implement one centralized serializer.

Required API:

```python
serialize_answer(target, config) -> SerializedAnswer
```

It must handle:

```text
point2d
bbox2d
boolean
short_text
integer
```

### 8.1 Spatial answers

Read normalized source-of-truth values from the canonical target.

Quantize with configurable location bins.

Convert:

```text
point storage xy -> model yx
bbox storage xyxy -> model yxyx
```

Never duplicate this ordering logic in task-specific code.

### 8.2 Text answers

Canonical forms:

```text
boolean: yes | no
integer: base-10 digits
short_text: normalized canonical text
```

### 8.3 Labels

Only answer tokens and EOS are supervised.

Set all prompt, image-prefix, answer-prefix, state-placeholder, and padding labels to:

```python
-100
```

---

## 9. Tokenized batch fields

The collator or tokenizer must produce repository-native equivalents of:

```python
input_ids: int64[B, L]
token_valid_mask: bool[B, L]
state_token_mask: bool[B, L]
text_labels: int64[B, L]
text_loss_mask: bool[B, L]
attention_mask: bool[B, L, L] | native equivalent
```

For action batches:

```text
text_labels = all -100
text_loss_mask = all false
```

For VQA batches:

```text
only answer and EOS positions are supervised
state positions are invalid
padding positions are invalid
```

---

## 10. Unified batch

Recommended logical batch:

```python
class UnifiedBatch(TypedDict):
    sample_type: str | int

    images: Array                  # [B, V, C, H, W]
    image_masks: Array             # [B, V]

    state: Array                   # [B, D]
    state_value_mask: Array        # [B, D]
    state_present: Array           # [B]

    input_ids: Array               # [B, L]
    token_valid_mask: Array        # [B, L]
    state_token_mask: Array        # [B, L]
    text_labels: Array             # [B, L]
    text_loss_mask: Array          # [B, L]

    actions: Array                 # [B, H, A]
    action_mask: Array             # [B, H, A]

    sample_ids: list[str]
```

Use JAX or PyTorch arrays according to the repository.

Keep static shapes when needed for compilation.

---

## 11. Homogeneous-task batching

Every batch must contain exactly one task type.

Do not mix action and VQA samples inside one batch.

For distributed training, all ranks must execute the same task type on the same global step.

Recommended approach:

1. create separate action and VQA loaders;
2. use one synchronized task scheduler;
3. choose a task at every global step;
4. broadcast or deterministically derive the task choice across ranks;
5. fetch the corresponding local shard on every rank;
6. execute one task-specific forward and loss path.

Example configuration:

```yaml
task_sampling:
  action_weight: 4
  vqa_weight: 1
  synchronized_across_ranks: true
```

The ratio must be configurable.

The scheduler must support:

```text
deterministic seeded sampling
distributed synchronization
checkpoint save and restore
per-task step counters
```

Checkpoint resume must preserve the task sequence.

---

## 12. Text loss

Use next-token cross-entropy over supervised answer positions.

Conceptually:

```python
per_token_loss = cross_entropy(logits, text_labels)
denominator = max(sum(text_loss_mask), 1)
text_loss = sum(per_token_loss * text_loss_mask) / denominator
```

Use the repository's numerically stable implementation.

Do not include prompt, state-placeholder, or padding positions in the numerator or denominator.

Track:

```text
vqa/text_loss
vqa/token_accuracy
vqa/sequence_exact_match
```

Also track metrics by answer type and question family.

---

## 13. Action loss

Keep the existing flow-matching action loss.

Use `action_mask` only to exclude invalid padded action positions or dimensions.

Required regression:

> With VQA disabled and the same seed/configuration, action-only behavior must match the original implementation within tolerance.

Do not change action-loss scaling as a side effect of adding VQA.

---

## 14. Autoregressive VQA generation

Add a VQA inference API.

Inputs:

```text
ego image
question
expected answer type
```

State and wrist placeholders must use the same mask path as training.

Initial generation requirements:

```text
greedy decoding
EOS termination
configurable maximum answer length
deterministic answer parsing
typed validation
```

Recommended result:

```python
class VQAResult(TypedDict):
    answer_type: str
    raw_text: str
    valid: bool

    text: Optional[str]
    boolean: Optional[bool]
    integer: Optional[int]
    point_xy_norm: Optional[list[float]]
    bbox_xyxy_norm: Optional[list[float]]

    error: Optional[str]
```

Do not silently repair malformed output into a plausible answer.

Return:

```text
valid = false
error = explicit parse failure
```

---

## 15. Answer parser

Implement one centralized parser inverse to the serializer.

Required failures:

```text
missing location token
wrong number of location tokens
location token outside configured vocabulary
malformed integer
unknown boolean answer
empty short text
reversed bbox corners
degenerate bbox
unexpected answer-type marker
```

Parsing and metrics must use the same coordinate-order convention.

---

## 16. Evaluation

### 16.1 Point

Report:

```text
mean normalized L2 error
median normalized L2 error
accuracy within configured normalized-radius thresholds
```

### 16.2 Bounding box

Report:

```text
mean IoU
median IoU
accuracy at IoU >= 0.5
```

### 16.3 Boolean

Report canonical exact accuracy.

### 16.4 Short text

Report:

```text
normalized exact match
alias match
```

Do not use an LLM judge as the only metric.

### 16.5 Integer

Report:

```text
exact accuracy
mean absolute error
```

Report all metrics by question family where possible.

---

## 17. Geometry-aware preprocessing

The VQA adapter must pass spatial annotations through a geometry-aware image transform.

Safe initial behavior:

```text
resize: allowed
letterbox padding: allowed
photometric augmentation: allowed
random crop: disabled
rotation: disabled
perspective: disabled
horizontal flip: disabled unless fully supported
```

Transform together:

```python
image
point target
bbox target
overlay anchors
```

Do not resize or pad coordinates independently from the image.

---

## 18. Configuration

Expose behavior in repository-native configuration.

Recommended options:

```yaml
tasks:
  enable_action: true
  enable_vqa: true

vqa:
  answer_types:
    - point2d
    - bbox2d
    - boolean
    - short_text
    - integer
  target_view: ego
  bbox_definition: visible_tight
  num_location_bins: 1024
  max_answer_tokens: 32
  boolean_true_text: "yes"
  boolean_false_text: "no"

state_masking:
  mask_vqa_state_values: true
  mask_vqa_state_prompt_positions: true
  mask_vqa_state_attention_queries: true
  mask_vqa_state_attention_keys: true
  zero_vqa_state_embeddings: true

batching:
  homogeneous_task_batches: true
  synchronize_task_across_ranks: true
  action_weight: 4
  vqa_weight: 1

images:
  fixed_slots:
    - ego
    - left_wrist
    - right_wrist
  vqa_valid_slots:
    - ego
```

Do not hard-code task-specific question vocabularies into this universal configuration.

---

## 19. Suggested module boundaries

Adapt to repository layout while preserving separation of concerns.

```text
src/openpi/
├── data/
│   ├── canonical_sample.py
│   ├── action_adapter.py
│   ├── vqa_adapter.py
│   ├── unified_collator.py
│   └── task_scheduler.py
├── vqa/
│   ├── answer_types.py
│   ├── serializer.py
│   ├── parser.py
│   ├── metrics.py
│   └── validation.py
├── models/
│   ├── existing_pi05_files
│   └── text_generation_extensions.py
└── training/
    └── multitask_loss_routing.py
```

Do not combine dataset parsing, serialization, attention masking, generation parsing, and evaluation into one monolithic module.

---

## 20. Required tests

### 20.1 Serializer and parser

- point float-to-token-to-float round trip;
- bbox float-to-token-to-float round trip;
- correct `xy -> yx`;
- correct `xyxy -> yxyx`;
- boundary coordinates at 0 and 1;
- malformed spatial output rejection;
- canonical boolean parsing;
- valid and invalid integer parsing;
- short-text normalization.

### 20.2 State masking

- random VQA state values do not change VQA logits;
- VQA answer tokens cannot attend to state-placeholder keys;
- state-placeholder queries cannot affect valid tokens;
- state-placeholder positions receive no text loss;
- action samples retain normal state visibility.

### 20.3 Image masking

- random masked wrist images do not change VQA logits;
- ego image remains visible;
- action camera masks preserve existing behavior.

### 20.4 Loss isolation

- action batch has zero text-loss contribution;
- VQA batch has zero action-loss contribution;
- inactive path does not accidentally receive gradients;
- action-only regression matches the original implementation.

### 20.5 Distributed batching

- all samples in a batch have one task type;
- all ranks use the same task type per global step;
- checkpoint resume preserves scheduler state;
- static shapes are valid for both tasks.

### 20.6 Geometry

- resize transforms points and boxes correctly;
- letterbox transforms points and boxes correctly;
- photometric augmentation leaves geometry unchanged;
- unsupported geometric transforms are disabled or rejected.

### 20.7 Learning smoke tests

Verify overfitting on tiny datasets:

```text
one boolean example
one integer example
one short-text example
one point example
one bbox example
one small action batch
```

---

## 21. Implementation sequence

Implement in this order:

1. typed canonical sample and enums;
2. validation framework;
3. answer serializer and parser;
4. VQA dataset adapter;
5. action adapter output compatibility;
6. unified collator;
7. prompt-state masking and attention masking;
8. autoregressive LM loss path;
9. task-homogeneous loss routing;
10. synchronized distributed task scheduler;
11. VQA generation API;
12. per-answer-type metrics;
13. action-only regression tests;
14. VQA overfit tests;
15. distributed mixed-task smoke test.

Prefer small, reviewable changes.

Do not begin with a deep refactor of the existing action model.

---

## 22. Explicit non-goals

Do not implement:

- multi-view VQA;
- cross-view grounding;
- 3D reconstruction;
- depth prediction;
- segmentation output;
- traces;
- polygons;
- amodal boxes;
- subtask generation;
- action hints;
- joint text-and-action targets in one sample;
- action tokenization;
- coordinate-regression auxiliary loss;
- mixed task types inside one batch.

---

## 23. Definition of done

The code modification is complete only when:

- action-only training remains compatible and regression-tested;
- VQA uses only ego as a valid visual input;
- VQA state is masked in value, prompt, attention-key, attention-query, and loss paths;
- the five answer types train, generate, parse, and evaluate;
- action and VQA use one canonical sample and batch interface;
- batches are task-homogeneous;
- distributed task selection is synchronized;
- VQA batches compute text loss only;
- action batches compute flow action loss only;
- masked state and wrist tensors are proven non-influential;
- spatial serialization is deterministic and tested;
- task-specific semantics remain outside core model code.
