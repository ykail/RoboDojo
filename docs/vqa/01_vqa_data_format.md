# Universal VQA Data Format Contract for openpi π0.5

## 0. Purpose

This document defines the universal data contract for adding ego-view VQA samples to an openpi π0.5 training pipeline.

It specifies:

- the physical annotation schema;
- the canonical in-memory sample format;
- supported answer types;
- spatial-coordinate conventions;
- prompt and answer serialization;
- visibility and answerability rules;
- image-transform requirements;
- validation requirements.

This document is task-independent. Task-specific VQA definitions must be placed in separate files.

The implementation supports two dataset sample types:

```text
action
vqa
```

The supported VQA answer types are fixed to:

```text
point2d
bbox2d
boolean
short_text
integer
```

VQA uses only the ego-view image as a valid visual input.

---

## 1. Universal design principles

### 1.1 Separate physical storage from model serialization

The physical annotation must store typed, semantic ground truth.

Do not store an already-tokenized answer string as the only source of truth.

Correct:

```json
{
  "answer_type": "point2d",
  "point_xy_norm": [0.421, 0.673]
}
```

Incorrect as the sole annotation:

```text
<answer_point2d><loc_689><loc_431>
```

The tokenizer or answer serializer must dynamically convert typed ground truth into model tokens.

### 1.2 Use one canonical sample interface

Action and VQA data may come from different physical sources, but dataset adapters must convert all records into one canonical sample structure before collation.

### 1.3 VQA must be image-answerable

Simulator state may be used to compute ground truth, but a VQA sample must be retained for training only when the answer is inferable from the valid ego-view image and question.

Maintain a distinction between:

```text
world_state_valid
image_answerable
```

A world-state label is not automatically a valid VQA label.

### 1.4 Spatial answers refer only to ego-view

For all VQA samples:

```text
target_view = ego
```

`point2d` and `bbox2d` must refer to the ego image used as model input.

No wrist-view or cross-view coordinates are supported in this version.

---

## 2. Canonical sample schema

Use a typed representation equivalent to the following.

```python
from typing import Literal, Optional, TypedDict
import numpy as np

SampleType = Literal["action", "vqa"]

AnswerType = Literal[
    "none",
    "point2d",
    "bbox2d",
    "boolean",
    "short_text",
    "integer",
]

VisibilityStatus = Literal[
    "not_applicable",
    "visible",
    "partially_visible",
    "occluded",
    "out_of_frame",
    "ambiguous",
]

class CanonicalTarget(TypedDict):
    answer_type: AnswerType

    # Exactly one field below is populated for a VQA sample.
    text: Optional[str]
    boolean: Optional[bool]
    integer: Optional[int]
    point_xy_norm: Optional[np.ndarray]       # float32, shape [2]
    bbox_xyxy_norm: Optional[np.ndarray]      # float32, shape [4]

    # Evaluation-only acceptable text alternatives.
    aliases: Optional[list[str]]

class CanonicalSample(TypedDict):
    sample_id: str
    sample_type: SampleType
    source_dataset: str
    task_name: str
    question_family: Optional[str]

    # Fixed camera slots shared with the action pipeline.
    images: dict[str, np.ndarray]
    image_masks: dict[str, bool]

    # Fixed-size state and action placeholders.
    state: np.ndarray
    state_value_mask: np.ndarray
    state_present: bool
    actions: np.ndarray
    action_mask: np.ndarray

    # Natural-language model input.
    prompt_text: str

    # Typed target.
    target: CanonicalTarget

    # VQA answerability metadata.
    world_state_valid: bool
    image_answerable: bool
    visibility_status: VisibilityStatus
    visible_fraction: Optional[float]
    occlusion_ratio: Optional[float]

    # Spatial metadata.
    target_view: Optional[str]
    coordinate_space: Optional[str]
    point_definition: Optional[str]
    bbox_definition: Optional[str]

    # Source indexing.
    episode_index: Optional[int]
    frame_index: Optional[int]
    timestamp: Optional[float]

    # Annotation provenance.
    gt_source: str
    annotation_version: str
    quality_score: Optional[float]
```

Use dataclasses, `TypedDict`, Pydantic, or an equivalent typed representation.

Avoid untyped nested dictionaries in core model and training code.

---

## 3. Fixed camera slots

Use fixed camera slots compatible with the existing action policy.

Recommended logical slots:

```text
ego
left_wrist
right_wrist
```

These may map to existing openpi keys such as:

```text
base_0_rgb
left_wrist_0_rgb
right_wrist_0_rgb
```

The mapping must be centralized in configuration.

### 3.1 Action samples

Action samples may use multiple valid views:

```python
image_masks = {
    "ego": True,
    "left_wrist": True,
    "right_wrist": True,
}
```

The exact valid set is policy-dependent.

### 3.2 VQA samples

VQA samples must expose only the ego image:

```python
image_masks = {
    "ego": True,
    "left_wrist": False,
    "right_wrist": False,
}
```

For static shapes, missing wrist tensors may be zero-filled placeholders.

Masked images must not influence the model.

---

## 4. Physical VQA annotation schema

A Parquet-friendly annotation table should use nullable typed columns.

Recommended columns:

```text
sample_id: string
sample_type: string                    # always "vqa" in a VQA table
source_dataset: string
task_name: string
question_family: string

episode_index: int64 | null
frame_index: int64 | null
timestamp: float64 | null

ego_image_reference: string
image_width: int32
image_height: int32

prompt_text: string
answer_type: string

answer_text: string | null
answer_bool: bool | null
answer_int: int64 | null
answer_point_xy_norm: fixed_size_list<float32>[2] | null
answer_bbox_xyxy_norm: fixed_size_list<float32>[4] | null
answer_aliases: list<string> | null

world_state_valid: bool
image_answerable: bool
visibility_status: string
visible_fraction: float32 | null
occlusion_ratio: float32 | null

target_view: string | null
coordinate_space: string | null
point_definition: string | null
bbox_definition: string | null

gt_source: string
annotation_version: string
quality_score: float32 | null
rejection_reason: string | null
```

Recommended values:

```text
target_view = "ego"
coordinate_space = "original_image_normalized_xy"
coordinate_space = "original_image_normalized_xyxy"
bbox_definition = "visible_tight"
```

Keep rejected or ambiguous records in a separate manifest when useful for auditing, but do not train on them.

---

## 5. Answer-type invariants

`answer_type` is a discriminator.

For every VQA record, exactly one matching answer field must be populated.

All other answer fields must be null.

---

## 6. `point2d`

Required source-of-truth field:

```python
point_xy_norm: float32[2]
```

Storage order:

```text
[x, y]
```

Range:

```text
0.0 <= x <= 1.0
0.0 <= y <= 1.0
```

The point refers to the original ego image.

A task-specific file must define the point semantics, such as:

```text
visible_region_center
functional_tip
grasp_point
placement_point
```

Do not mix different point semantics under one question family.

A point sample is trainable only when:

```text
world_state_valid = true
image_answerable = true
visibility_status in {"visible", "partially_visible"}
```

The target point itself must be visibly supported.

Do not encode invisibility using sentinel coordinates such as:

```text
[0, 0]
[-1, -1]
image center
```

If visibility is a desired target, create a separate boolean question.

---

## 7. `bbox2d`

Required source-of-truth field:

```python
bbox_xyxy_norm: float32[4]
```

Storage order:

```text
[x_min, y_min, x_max, y_max]
```

Validation:

```text
0.0 <= x_min < x_max <= 1.0
0.0 <= y_min < y_max <= 1.0
```

Use the visible tight 2D bounding box of the target instance in the ego image.

Do not use:

- an amodal box;
- the projection of a 3D bounding box;
- a box that includes fully occluded object regions;
- a sentinel box for an invisible target.

A bbox sample is trainable only when the visible target mask passes configured size and visibility thresholds.

Recommended metadata:

```text
bbox_definition = "visible_tight"
target_view = "ego"
coordinate_space = "original_image_normalized_xyxy"
```

---

## 8. `boolean`

Required field:

```python
boolean: bool
```

Canonical serialized answers:

```text
yes
no
```

Do not alternate between:

```text
yes/no
true/false
1/0
Y/N
```

Questions may be multilingual, but target serialization must use one configured canonical vocabulary.

For visually grounded state questions, simulator state may define the class while visibility filters determine whether the sample is trainable.

Use a dead zone around ambiguous physical thresholds when applicable.

---

## 9. `short_text`

Required field:

```python
text: str
```

Requirements:

- keep the answer concise;
- use stable canonical wording;
- normalize Unicode;
- strip surrounding whitespace;
- avoid explanatory sentences;
- store aliases only for evaluation.

Examples:

```text
nothing
pen
pen holder
red pen
left side
```

For closed-set classification represented as `short_text`, define the exact allowed answer vocabulary in the task-specific file.

Do not use opaque one-character class labels unless there is a strong compatibility reason.

---

## 10. `integer`

Required field:

```python
integer: int
```

Canonical serialization:

```text
0
1
2
3
...
```

Requirements:

- serialize in base-10 digits;
- do not use number words;
- do not use commas;
- do not use leading zeros except for zero;
- define the valid range per question family.

For questions using the word `visible`, count only visually visible qualifying instances according to the task-specific visibility rule.

Do not substitute hidden simulator-state counts for visible counts.

---

## 11. Spatial-token serialization

### 11.1 Normalized floating-point source of truth

Store:

```text
point: [x, y]
bbox:  [x_min, y_min, x_max, y_max]
```

Coordinates refer to the original ego image before model preprocessing.

### 11.2 Quantization

Use a configurable number of bins:

```python
num_location_bins = 1024
```

Quantization:

```python
q = round(clamp(value, 0.0, 1.0) * (num_location_bins - 1))
```

Dequantization:

```python
value = q / (num_location_bins - 1)
```

Use existing PaliGemma-compatible location tokens when available.

### 11.3 Model ordering

Storage order and model serialization order differ.

Point:

```text
storage: [x, y]
model:   [y, x]
```

Serialized target:

```text
<answer_point2d><loc_y><loc_x><eos>
```

Bounding box:

```text
storage: [x_min, y_min, x_max, y_max]
model:   [y_min, x_min, y_max, x_max]
```

Serialized target:

```text
<answer_bbox2d><loc_ymin><loc_xmin><loc_ymax><loc_xmax><eos>
```

All conversions must be centralized in one serializer.

Annotators and task generators must use only `xy` or `xyxy` storage order.

### 11.4 Required round-trip behavior

Provide tested functions equivalent to:

```python
serialize_point(point_xy_norm)
parse_point(tokens)

serialize_bbox(bbox_xyxy_norm)
parse_bbox(tokens)
```

Round-trip error must not exceed one quantization bin plus floating-point tolerance.

---

## 12. Prompt protocol

Every prompt must explicitly state the expected answer type.

Recommended conceptual format:

```text
<mode_vqa>
Question: {question}
Expected answer type: <answer_type>
State: {reserved masked state span}
Answer:
```

The target answer is appended during teacher-forced training.

Examples follow.

### 12.1 Point

```text
Question: Locate the requested target in the ego-view image.
Expected answer type: point2d.
Answer:
```

Target:

```text
<answer_point2d><loc_y><loc_x><eos>
```

### 12.2 Bounding box

```text
Question: Locate the visible target in the ego-view image.
Expected answer type: bbox2d.
Answer:
```

Target:

```text
<answer_bbox2d><loc_ymin><loc_xmin><loc_ymax><loc_xmax><eos>
```

### 12.3 Boolean

Target:

```text
<answer_boolean>yes<eos>
```

or:

```text
<answer_boolean>no<eos>
```

### 12.4 Short text

Target:

```text
<answer_short_text>{canonical_short_answer}<eos>
```

### 12.5 Integer

Target:

```text
<answer_integer>{base_10_integer}<eos>
```

Whitespace and punctuation must be deterministic.

---

## 13. VQA state placeholder contract

VQA does not use robot state as semantic input.

The canonical VQA sample must contain:

```python
state_present = False
state = np.zeros([state_dim], dtype=np.float32)
state_value_mask = np.zeros([state_dim], dtype=bool)
```

The tokenizer may reserve a fixed state span for compatibility, but:

- zero state must not be serialized into meaningful state tokens;
- every state-placeholder token must be invalid;
- every state-placeholder token must be excluded from attention;
- every state-placeholder token must be excluded from LM loss.

Detailed implementation requirements belong in the code-modification instruction.

---

## 14. Visibility and answerability

### 14.1 Required states

Use at least:

```text
visible
partially_visible
occluded
out_of_frame
ambiguous
not_applicable
```

### 14.2 Universal training rule

A VQA record may be used for training only when:

```python
world_state_valid is True
image_answerable is True
```

Spatial questions additionally require that the target point or visible mask is supported by the ego image.

### 14.3 Fast simulator-based visibility pipeline

For simulator-generated data, prefer a single-pass annotation pipeline using:

- ego RGB;
- instance or semantic-instance segmentation;
- distance-to-image-plane or compatible depth;
- object poses and camera matrices;
- optional tight 2D boxes;
- optional object-level occlusion statistics.

Use object-level occlusion only as a coarse filter.

For keypoints, use local projected-point visibility checks.

For boxes and visible counts, use visible instance masks.

### 14.4 Keypoint visibility

Recommended procedure:

1. project the 3D functional point into the ego image;
2. reject out-of-frame points;
3. inspect a small pixel disk around the projected location;
4. require target-instance pixels in that disk;
5. require depth agreement between the projected point and visible target surface;
6. classify borderline cases as `ambiguous`.

A small surface patch or keypoint cluster is more robust than a mathematical zero-area point.

### 14.5 Visible-instance test

For object visibility and counting, compute:

```text
visible_pixel_count
visible_fraction
mask_connected_components
image-boundary truncation
```

A task-specific file must define thresholds.

An object counts as visible only if it passes all required thresholds.

---

## 15. Image preprocessing

Spatial ground truth must remain synchronized with image geometry.

Safe initial policy:

- allow resize;
- allow aspect-ratio-preserving padding;
- allow photometric augmentation;
- disable random crop;
- disable rotation;
- disable perspective transform;
- disable horizontal flip unless both coordinates and semantics are updated.

Use one geometry-aware transform contract:

```python
result = transform(
    image=ego_image,
    point_xy=optional_point,
    bbox_xyxy=optional_bbox,
)
```

After transforms, validate coordinates and box geometry.

---

## 16. Validation rules

A central validator must reject malformed samples before training.

### 17.1 Common VQA checks

```text
sample_id is non-empty
sample_type == vqa
ego image exists
ego image mask is true
all wrist image masks are false
prompt_text is non-empty
answer_type is supported
world_state_valid is true
image_answerable is true
all action-mask entries are false
state_present is false
all state-value-mask entries are false
exactly one typed answer field is populated
```

### 17.2 Point checks

```text
point is finite
point has shape [2]
point lies within [0, 1]
target_view == ego
point_definition is non-empty
visibility_status is visible or partially_visible
```

### 17.3 Box checks

```text
box is finite
box has shape [4]
box lies within [0, 1]
x_min < x_max
y_min < y_max
target_view == ego
bbox_definition == visible_tight
visible mask passes task thresholds
```

### 17.4 Boolean checks

```text
exactly one boolean target exists
physical-state ambiguity filter passes
visual answerability filter passes
```

### 17.5 Short-text checks

```text
answer is in the question-family vocabulary when closed-set
answer is normalized and non-empty
```

### 17.6 Integer checks

```text
value lies in question-family range
count definition is explicit
all counted instances satisfy the visibility rule
```

Fail fast with sample ID, source location, and rejection reason.

---

## 18. Dataset statistics

Report at least:

```text
samples by question_family
samples by answer_type
answer distribution
visibility-status distribution
rejection-reason distribution
point and box spatial heatmaps
box-size distribution
integer class balance
short-text class balance
```

For simulator-generated datasets, also report:

```text
world_state_valid but image_answerable false
out_of_frame targets
occluded targets
ambiguous targets
```

These statistics are required to detect hidden-state supervision and class imbalance.

---

## 19. Non-goals

This universal data contract does not support:

- multi-view VQA;
- wrist-view grounding;
- cross-view projection;
- 3D position output;
- depth output;
- traces;
- polygons;
- segmentation answers;
- amodal boxes;
- multiple typed answers in one record;
- joint text-and-action targets;
- free-form JSON generation.

---

## 20. Definition of done

### 20.1 RoboDojo synthetic sidecar mapping

The RoboDojo generator implements this physical record as an independent
sidecar. Each generated task directory contains `images/`, `audit/`,
`annotations.parquet`, `rejected.parquet`, `manifest.json`, and `report.json`.
`annotations.parquet` contains only
accepted records; rejected candidates are retained in `rejected.parquet` with
a stable `rejection_reason` for audit.

The sidecar uses `source_dataset=RoboDojo_vqa_synthetic_v1`.  Its manifest
records the command, seed, source layout, simulator commit when available, and
the action-data reference `RoboDojo_ee_lerobot_v30_video`.  It neither copies
nor rewrites LeRobot action data.  A future canonical adapter is responsible
for adding canonical VQA placeholders (`state_present=false`, no action mask,
ego-image mask only).

The typed-answer discriminator remains unchanged: exactly one of
`answer_point_xy_norm`, `answer_bbox_xyxy_norm`, `answer_bool`,
`answer_text`, or `answer_int` is populated.  The current PyArrow writer uses
nullable `list<float32>` columns for point and bbox values rather than nullable
fixed-size Arrow lists, because PyArrow 25 does not reliably round-trip null
fixed-size lists.  This is only a storage compatibility detail: the validator
enforces lengths two and four respectively, and preserves the normalized
coordinate contract above.

`source_layout`, `scene_id`, and `audit_metadata_json` are retained for
provenance and visual audit.  They are not alternate answer channels or model
targets.

`audit/` may contain segmentation-ID visualizations and ID-to-label metadata.
It is for validation only and is never a model input or annotation target.

The data format is complete when:

- all VQA tasks use one canonical typed schema;
- all spatial targets use normalized ego-view coordinates;
- all answer types have strict invariants;
- visibility and answerability are stored separately;
- invisible spatial targets are rejected instead of assigned sentinel values;
- model tokens are generated from typed source-of-truth targets;
- transforms update coordinates correctly;
- validation and dataset statistics are implemented.
