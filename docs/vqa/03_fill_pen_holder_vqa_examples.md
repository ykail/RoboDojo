# Task-Specific VQA Specification: RoboDojo `fill-pen-holder`

## 0. Scope

This document defines VQA data generation for the RoboDojo-style task:

```text
fill-pen-holder
```

It is task-specific and must be read together with:

```text
01_vqa_data_format.md
02_openpi_vqa_code_modifications.md
```

The scene contains:

- one ego-view image used for VQA;
- four pens;
- one pen holder;
- a left robot gripper;
- a right robot gripper.

This task defines the following VQA families:

```text
pen_nib_grounding
holder_fallen
holder_bbox
left_gripper_content
right_gripper_content
visible_pens_on_table
visible_pens_in_holder
```

No task-specific logic from this file should be hard-coded into the universal model architecture.

---

## 1. Global task-generation principles

### 1.1 Use simulator state for GT and rendered evidence for answerability

For every candidate record compute:

```text
world_state_valid
image_answerable
```

Use simulator geometry, poses, contact state, and containment state to compute ground truth.

Use ego-view segmentation, depth, and visibility tests to determine whether the current image supports that answer.

Only train on:

```text
world_state_valid = true
image_answerable = true
```

### 1.2 Use only ego-view

All spatial coordinates and all visibility judgments refer to the VQA ego image.

```text
target_view = ego
```

### 1.3 Generate dedicated VQA states

For gripper-content questions, generate dedicated, stable grasp snapshots rather than sampling arbitrary transition frames.

For zero-position questions, generate dedicated randomized object-layout snapshots.

This makes class balance and answerability controllable.

---

## 2. Required simulator annotations

For every rendered candidate frame, obtain or compute:

```text
ego RGB
ego image width and height
camera view and projection matrices
distance-to-image-plane depth or compatible depth
semantic-instance or instance segmentation
pen instance IDs
pen-holder instance ID
left-gripper semantic mask
right-gripper semantic mask
3D pen nib functional points
pen-holder pose
stable grasp state for both grippers
pen containment state
object support/contact state
```

Optional but useful:

```text
tight visible 2D boxes
object-level occlusion ratio
visible-pixel count
visible fraction relative to unoccluded render
```

Object-level occlusion is only a coarse filter.

Keypoint and visible-count questions require local or instance-level visibility.

---

## 3. Fast visibility and occlusion pipeline

Render the following once per candidate frame:

```text
RGB
instance segmentation
depth
```

Use the same outputs to process all pens, the holder, and both grippers.

Do not re-render once per question unless an unoccluded reference pass is explicitly required.

### 3.1 Object visible-mask statistics

For each instance:

```python
mask = instance_map == instance_id
visible_pixel_count = mask.sum()
visible_bbox = tight_bbox(mask)
```

Recommended configurable thresholds:

```yaml
visibility:
  min_visible_pixels_object: 64
  min_visible_pixels_thin_object: 16
  min_visible_fraction: 0.05
  max_boundary_truncation_fraction: 0.5
```

Exact thresholds must be calibrated for final image resolution.

### 3.2 Optional unoccluded reference pass

To estimate visible fraction more robustly:

1. render the normal scene;
2. obtain the target visible mask;
3. optionally render an annotation-only reference where known occluders are hidden or the target is isolated;
4. compute:

```text
visible_fraction = visible_pixels_normal / projected_pixels_reference
occlusion_ratio = 1 - visible_fraction
```

Use this only during offline data generation.

Do not include unoccluded reference images as model input.

### 3.3 Keypoint visibility

For each pen nib:

1. transform the nib functional point to world coordinates;
2. project it into the ego image;
3. reject if behind the camera or outside the image;
4. inspect a small disk around the projected pixel;
5. require pixels belonging to that pen instance;
6. require depth agreement;
7. classify borderline cases as ambiguous.

Conceptual implementation:

```python
def keypoint_visibility(
    point_world,
    target_instance_id,
    camera,
    instance_map,
    depth_image,
    radius_px,
    depth_tolerance,
    min_matching_pixels,
):
    u, v, point_depth = project_world_point(point_world, camera)

    if not in_image(u, v):
        return "out_of_frame", None

    patch = disk_pixels(u, v, radius_px)

    instance_ok = instance_map[patch] == target_instance_id
    depth_ok = abs(depth_image[patch] - point_depth) <= depth_tolerance

    matched = instance_ok & depth_ok

    if matched.sum() >= min_matching_pixels:
        return "visible", [u, v]

    if instance_ok.sum() == 0:
        return "occluded", None

    return "ambiguous", None
```

Prefer a small nib-surface patch or cluster of functional points over a single mathematical point.

### 3.4 Visible object counting

For questions containing the word `visible`, count only instances whose visible mask passes the configured thresholds.

Do not use the hidden simulator-state count as the answer.

---

## 4. Scenario A: zero-position randomized layout

Scene generation:

- robot arms are in the configured zero position;
- four pens are randomly placed on the table;
- the pen holder is randomly placed on the table;
- the pen holder may be upright or lying down;
- placements must pass collision and scene-validity checks.

Generate three question families:

```text
pen_nib_grounding
holder_fallen
holder_bbox
```

---

## 5. Numbered pen overlay

Use overlay scheme A: numbered marks rendered on the final ego image.

Assign each pen a unique mark:

```text
1
2
3
4
```

The mapping may be random per scene.

Store:

```json
{
  "overlay_type": "numbered_object_marks",
  "overlay_version": "fill_pen_holder_v1",
  "mark_to_instance": {
    "1": "pen_instance_a",
    "2": "pen_instance_b",
    "3": "pen_instance_c",
    "4": "pen_instance_d"
  }
}
```

### 5.1 Overlay placement

For each pen:

1. project a stable pen-body anchor point;
2. place the numeric badge near the projected pen body;
3. use a leader line to establish association;
4. do not place the badge on the nib;
5. keep both the badge and the complete leader line outside the nib protection
   radius;
6. do not cover the pen body excessively;
7. resolve badge overlap;
8. keep the badge and leader line within image bounds.

Overlay rendering order:

```text
scene render
geometric preprocessing
numeric overlay
```

When overlays are rendered before preprocessing, transform anchors and overlay graphics exactly with the image.

### 5.2 Avoid answer leakage

Do not:

- put the mark at the nib position;
- make the leader line terminate at the nib;
- use a target-specific mark color that encodes the answer;
- highlight the queried pen differently from other pens.

All four marks should use the same visual style.

---

## 6. Question family: `pen_nib_grounding`

### 6.1 Prompt

Canonical template:

```text
Locate the nib of the pen marked {mark} in the ego-view image.
Return one point.
```

Alternative prompt variants may be added, but every variant must preserve the exact meaning.

### 6.2 Answer type

```text
point2d
```

### 6.3 Ground truth

Use the designated 3D `nib_tip` functional point for the pen mapped to `{mark}`.

Do not use:

- the pen center;
- the visible-mask centroid of the whole pen;
- a generic asset origin;
- a task-success point unless it has been verified to represent the nib.

Store:

```text
point_definition = functional_nib_tip
coordinate_space = original_image_normalized_xy
```

### 6.4 Trainability

Generate a trainable sample only when:

```text
queried nib point is in frame
queried nib is locally visible
local target-instance pixels pass threshold
local depth agreement passes threshold
visibility status is not ambiguous
overlay-to-instance mapping is unambiguous
```

If the pen body is visible but the nib is occluded, reject the point question.

Do not use a projected hidden nib as a point target.

### 6.5 Example record

```json
{
  "sample_id": "fill_pen_holder_nib_000042",
  "sample_type": "vqa",
  "task_name": "fill-pen-holder",
  "question_family": "pen_nib_grounding",
  "prompt_text": "Locate the nib of the pen marked 3 in the ego-view image. Return one point.",
  "answer_type": "point2d",
  "answer_point_xy_norm": [0.592, 0.518],
  "target_view": "ego",
  "coordinate_space": "original_image_normalized_xy",
  "point_definition": "functional_nib_tip",
  "world_state_valid": true,
  "image_answerable": true,
  "visibility_status": "visible",
  "gt_source": "simulated_3d_functional_point_projection",
  "annotation_version": "fill_pen_holder_v1"
}
```

---

## 7. Question family: `holder_fallen`

### 7.1 Prompt

Canonical template:

```text
Is the pen holder lying on its side?
Answer yes or no.
```

### 7.2 Answer type

```text
boolean
```

### 7.3 Ground truth

Compute the holder's local up axis in world coordinates.

Let:

```text
theta = angle(holder_up_world, world_up)
```

Use configurable thresholds with a dead zone.

Recommended initial values:

```yaml
holder_orientation:
  upright_max_degrees: 30
  fallen_min_degrees: 60
```

Targets:

```text
theta <= 30 degrees       -> no
theta >= 60 degrees       -> yes
30 < theta < 60 degrees   -> reject as ambiguous
```

### 7.4 Visual answerability

Retain only if the holder:

```text
has sufficient visible pixels
has sufficient visible fraction
is not excessively truncated
has enough visible geometry to distinguish upright from fallen
```

A known simulator orientation alone is not enough.

### 7.5 Example record

```json
{
  "sample_id": "fill_pen_holder_fallen_000042",
  "sample_type": "vqa",
  "task_name": "fill-pen-holder",
  "question_family": "holder_fallen",
  "prompt_text": "Is the pen holder lying on its side? Answer yes or no.",
  "answer_type": "boolean",
  "answer_bool": false,
  "world_state_valid": true,
  "image_answerable": true,
  "visibility_status": "visible",
  "gt_source": "simulated_holder_orientation",
  "annotation_version": "fill_pen_holder_v1"
}
```

---

## 8. Question family: `holder_bbox`

### 8.1 Prompt

Canonical template:

```text
Locate the visible pen holder in the ego-view image.
Return one bounding box.
```

### 8.2 Answer type

```text
bbox2d
```

### 8.3 Ground truth

Use the tight rectangle around visible pen-holder instance pixels.

```python
mask = instance_map == holder_instance_id
ys, xs = where(mask)

x_min = xs.min()
y_min = ys.min()
x_max = xs.max()
y_max = ys.max()
```

Normalize by image width and height.

Store:

```text
bbox_definition = visible_tight
coordinate_space = original_image_normalized_xyxy
```

### 8.4 Trainability

Retain only if:

```text
visible pixel count passes threshold
visible fraction passes threshold
box width and height are non-degenerate
target is not excessively truncated
segmentation instance mapping is valid
```

Partial occlusion is allowed because the target is a visible tight box.

Do not use a projected 3D box or amodal extent.

### 8.5 Example record

```json
{
  "sample_id": "fill_pen_holder_bbox_000042",
  "sample_type": "vqa",
  "task_name": "fill-pen-holder",
  "question_family": "holder_bbox",
  "prompt_text": "Locate the visible pen holder in the ego-view image. Return one bounding box.",
  "answer_type": "bbox2d",
  "answer_bbox_xyxy_norm": [0.402, 0.492, 0.472, 0.647],
  "target_view": "ego",
  "coordinate_space": "original_image_normalized_xyxy",
  "bbox_definition": "visible_tight",
  "world_state_valid": true,
  "image_answerable": true,
  "visibility_status": "partially_visible",
  "gt_source": "ego_instance_segmentation",
  "annotation_version": "fill_pen_holder_v1"
}
```

---

## 9. Scenario B: stable gripper-content snapshots

Generate dedicated snapshots where one or both arms are freely positioned in the workspace.

The generator must explicitly choose and stabilize the grasp class.

Allowed answer vocabulary:

```text
nothing
pen
pen holder
```

Use full canonical text rather than `N`, `P`, and `H`.

These are closed-set `short_text` questions.

Generate balanced classes and balanced left/right-arm examples.

---

## 10. Stable grasp-state definition

The data generator must create or verify a stable grasp state.

A grasp is valid when the task generator or simulator confirms that the gripper directly holds the object.

Recommended conditions:

```text
grasp mode explicitly selected by generator
object is inside the intended gripper grasp region
gripper is closed to the configured grasp width
contact or attachment state is valid
relative gripper-object pose is stable for K frames
object is not in a release or slip transition
```

For `nothing`:

```text
no object is directly grasped
gripper interior is visually inspectable
nearby objects are not mistaken for held objects
```

Do not sample arbitrary contact or approach frames.

---

## 11. Question families: gripper content

### 11.1 Left gripper

Prompt:

```text
What is the left gripper directly holding?
Answer with exactly one of: nothing, pen, pen holder.
```

Question family:

```text
left_gripper_content
```

### 11.2 Right gripper

Prompt:

```text
What is the right gripper directly holding?
Answer with exactly one of: nothing, pen, pen holder.
```

Question family:

```text
right_gripper_content
```

### 11.3 Answer type

```text
short_text
```

### 11.4 Direct-object semantics

If a gripper holds the pen holder and the holder contains pens, the answer is:

```text
pen holder
```

The contained pens are not directly held by the gripper.

### 11.5 Visual answerability

Retain a sample only if the relevant gripper and the evidence distinguishing its class are sufficiently visible.

For `pen`:

```text
the gripper is visible
a visible portion of the held pen passes threshold
the pen-gripper relation is visually plausible
```

For `pen holder`:

```text
the gripper is visible
a visible portion of the held holder passes threshold
the holder-gripper relation is visually plausible
```

For `nothing`:

```text
the gripper interior region is sufficiently visible
no held-object mask is present in the grasp region
```

Dedicated generation should position the arm to satisfy these constraints.

### 11.6 Class balance

Recommended target distribution per side:

```text
nothing : pen : pen holder = 1 : 1 : 1
```

Also balance:

```text
left arm
right arm
object poses
arm poses
background layouts
```

Do not allow one arm to be strongly correlated with one object class.

### 11.7 Example record

```json
{
  "sample_id": "fill_pen_holder_left_grasp_000105",
  "sample_type": "vqa",
  "task_name": "fill-pen-holder",
  "question_family": "left_gripper_content",
  "prompt_text": "What is the left gripper directly holding? Answer with exactly one of: nothing, pen, pen holder.",
  "answer_type": "short_text",
  "answer_text": "pen holder",
  "world_state_valid": true,
  "image_answerable": true,
  "visibility_status": "partially_visible",
  "gt_source": "dedicated_stable_grasp_generator",
  "annotation_version": "fill_pen_holder_v1"
}
```

---

## 12. Scenario C: visible pen counting

Split counting into two visually grounded questions:

```text
How many pens are visible on the table?
How many pens are visible inside the pen holder?
```

These are visible-count questions, not hidden world-state questions.

Do not use conservation reasoning or infer hidden pens by subtraction in this version.

Answer type:

```text
integer
```

Valid range:

```text
0 through 4
```

---

## 13. Per-pen semantic state

For auditing, assign each pen one simulator state:

```text
IN_HOLDER
HELD_LEFT
HELD_RIGHT
ON_TABLE
TRANSITION
UNKNOWN
```

This state is useful for candidate generation and consistency checks.

However, the final visible-count target is computed from visible masks and the question's qualifying relation.

Do not count a hidden object merely because its simulator state is known.

Reject count samples containing unresolved transition or unknown states when those states could affect the visual category definition.

---

## 14. Visible-instance predicate for thin pens

Define a reusable predicate:

```python
is_visibly_countable(pen, relation, frame) -> bool
```

Recommended factors:

```text
instance visible-pixel count
visible connected-component size
visible mask length or extent
visible fraction
image-boundary truncation
relation-specific geometry
```

Because pens are thin objects, a raw pixel-count threshold alone may be unstable.

Recommended configurable criteria:

```yaml
pen_visibility:
  min_visible_pixels: 12
  min_visible_major_axis_pixels: 6
  min_visible_fraction: 0.03
max_boundary_truncation_fraction: 0.5
```

Calibrate at final training resolution.

The current RoboDojo collector uses a stricter count-only default of 48
visible pixels and a 12-pixel major axis. A mask classified as
`partially_visible` does not pass this predicate and is not counted.

A pen should count only once even if its visible mask has multiple connected fragments due to occlusion.

Use instance IDs, not connected-component count, to count pens.

---

## 15. Question family: `visible_pens_on_table`

### 15.1 Prompt

```text
How many pens are visible on the table?
Answer with one integer.
```

### 15.2 Answer type

```text
integer
```

### 15.3 Qualifying relation

A pen qualifies as `on the table` when the simulator confirms it is supported by or resting on the table and it is not:

```text
inside the pen holder
directly held by a gripper
in a transition state
```

### 15.4 Visibility rule

Count a qualifying pen only if its ego-view instance mask passes the pen-visible predicate.

Conceptually:

```python
count = 0
for pen in pens:
    if pen.semantic_state == ON_TABLE and is_visibly_countable(pen, "on_table", frame):
        count += 1
```

If a pen is on the table but fully hidden behind a robot arm, it does not contribute to the answer because the question asks for visible pens.

### 15.5 Answerability

The question is answerable when the visible-count annotation is reliable.

Reject when:

```text
segmentation IDs are invalid
the pen mask is dominated by rendering artifacts
multiple pen instances are merged
a qualifying pen is exactly at a visibility threshold
the scene is in an unresolved transition
```

### 15.6 Example record

```json
{
  "sample_id": "fill_pen_holder_visible_table_000230",
  "sample_type": "vqa",
  "task_name": "fill-pen-holder",
  "question_family": "visible_pens_on_table",
  "prompt_text": "How many pens are visible on the table? Answer with one integer.",
  "answer_type": "integer",
  "answer_int": 1,
  "world_state_valid": true,
  "image_answerable": true,
  "visibility_status": "not_applicable",
  "gt_source": "simulated_relation_plus_ego_instance_visibility",
  "annotation_version": "fill_pen_holder_v1"
}
```

---

## 16. Question family: `visible_pens_in_holder`

### 16.1 Prompt

```text
How many pens are visible inside the pen holder?
Answer with one integer.
```

### 16.2 Answer type

```text
integer
```

### 16.3 Qualifying relation

A pen qualifies as `inside the pen holder` when the simulator's containment test confirms that the designated pen geometry lies within the holder's containment region according to the task definition.

This visible-count question does not require the pen to satisfy task-success orientation unless that is explicitly added as a separate question family.

### 16.4 Visibility rule

Count a qualifying pen only if a visible portion of that pen passes the pen-visible predicate in the ego image.

Conceptually:

```python
count = 0
for pen in pens:
    if pen.semantic_state == IN_HOLDER and is_visibly_countable(pen, "in_holder", frame):
        count += 1
```

A pen that is physically inside the holder but fully hidden by the gripper, holder wall, or another object does not count.

### 16.5 Occlusion handling

The gripper may partially block the holder opening.

Use instance segmentation to identify visible pen pixels independently of the holder and robot masks.

Count by pen instance ID.

Do not estimate hidden pens by:

```text
4 - visible_table_pens - held_pens
```

That is a different reasoning task and is explicitly excluded.

### 16.6 Example record

```json
{
  "sample_id": "fill_pen_holder_visible_inside_000230",
  "sample_type": "vqa",
  "task_name": "fill-pen-holder",
  "question_family": "visible_pens_in_holder",
  "prompt_text": "How many pens are visible inside the pen holder? Answer with one integer.",
  "answer_type": "integer",
  "answer_int": 2,
  "world_state_valid": true,
  "image_answerable": true,
  "visibility_status": "not_applicable",
  "gt_source": "simulated_containment_plus_ego_instance_visibility",
  "annotation_version": "fill_pen_holder_v1"
}
```

---

## 17. Counting edge cases

Apply these rules consistently.

### 17.1 Partially visible pen

Count it if it passes the configured visible-instance predicate.

### 17.2 Fully occluded pen

Do not count it.

### 17.3 Pen visible through holder opening

Count it if its instance pixels are visible and containment is true.

### 17.4 Pen visible outside holder while mostly inside

For `visible_pens_in_holder`, count it when containment is true and visible pixels pass threshold.

For `visible_pens_on_table`, do not count it.

### 17.5 Pen held while being inserted

Treat as transition unless the task generator defines a deterministic precedence.

Recommended initial behavior:

```text
reject transition frame for visible-count generation
```

### 17.6 Pen mask split into fragments

Count once by instance ID.

### 17.7 Two pens visually overlap

Count each visible instance whose own mask passes threshold.

Do not infer a hidden instance from simulator state.

---

## 18. Recommended task-specific annotation fields

Add fields such as:

```text
mark_index: int | null
target_instance_id: string | null

holder_tilt_degrees: float | null

gripper_side: string | null
grasp_class: string | null
stable_grasp_frames: int | null

visible_pen_instance_ids: list<string> | null
qualifying_pen_instance_ids: list<string> | null
counted_pen_instance_ids: list<string> | null

overlay_type: string | null
overlay_version: string | null
```

These fields are generator and audit metadata.

They need not be passed to the model.

---

## 19. Dataset balancing

Balance at least:

### 19.1 Pen marks

```text
queried mark 1/2/3/4
physical pen instance
pen type
pen orientation
nib image region
```

Randomize mark-to-instance mapping to prevent a fixed semantic association.

### 19.2 Holder orientation

Balance:

```text
upright
fallen
```

Exclude the dead-zone angles.

### 19.3 Gripper content

Balance per side:

```text
nothing
pen
pen holder
```

### 19.4 Integer answers

For both visible-count families, monitor counts:

```text
0
1
2
3
4
```

Use controlled generation or weighted sampling to prevent extreme imbalance.

---

## 20. Task-specific quality reports

Produce:

```text
nib visibility rejection rate
nib point spatial heatmap
overlay collision and clipping rate
holder upright/fallen class balance
holder bbox size distribution
left/right gripper class confusion matrix
visible-table count distribution
visible-inside count distribution
per-pen visible-pixel distribution
count samples near visibility thresholds
```

Perform visual audits by rendering examples with:

```text
question text
typed answer
instance masks
projected nib point
visible bbox
counted instance IDs
rejection reason
```

---

## 21. Required task-specific tests

### 21.1 Overlay tests

- all four marks are present;
- mark-to-instance mapping is bijective;
- marks are inside image;
- marks do not cover nib targets;
- queried pen is not styled differently;
- leader lines associate with the correct pen.

### 21.2 Nib visibility tests

- visible nib is accepted;
- arm-occluded nib is rejected;
- out-of-frame nib is rejected;
- body-visible but nib-hidden case is rejected;
- depth mismatch is rejected or marked ambiguous.

### 21.3 Holder-state tests

- clear upright pose yields `no`;
- clear fallen pose yields `yes`;
- dead-zone orientation is rejected;
- heavily occluded holder is rejected.

### 21.4 Holder-box tests

- box equals tight visible instance mask;
- partial occlusion produces a visible box;
- invisible holder is rejected;
- projected 3D extent is not used.

### 21.5 Gripper tests

- stable pen grasp yields `pen`;
- stable holder grasp yields `pen holder`;
- empty stable gripper yields `nothing`;
- approach, release, and slip frames are rejected;
- holding a holder containing pens still yields `pen holder`.

### 21.6 Visible-count tests

- visible on-table pen is counted;
- occluded on-table pen is not counted;
- visible in-holder pen is counted;
- hidden in-holder pen is not counted;
- one fragmented instance is counted once;
- overlapping visible instances are counted separately;
- transition frames are rejected.

---

## 22. Non-goals for this task version

Do not add:

- conservation-based hidden-pen counting;
- total physical pen count in the holder when some are invisible;
- reasoning by subtraction;
- multi-frame memory;
- wrist-view evidence;
- amodal pen or holder boxes;
- hidden nib projection targets;
- arbitrary grasp transition frames;
- one-letter grasp answers `N/P/H` as training targets.

---

## 23. Definition of done

The `fill-pen-holder` VQA generator is complete when:

- zero-position randomized layouts generate valid overlay nib, holder-state, and holder-box samples;
- every queried nib is visibly supported;
- numbered overlays are unambiguous and non-leaking;
- stable left/right grasp samples are deliberately generated and class-balanced;
- gripper answers use `nothing`, `pen`, and `pen holder`;
- visible table count uses visible on-table instance masks;
- visible holder count uses visible contained instance masks;
- fully occluded pens do not contribute to visible counts;
- all questions use only ego-view evidence;
- fast visibility checks reuse one segmentation/depth render per frame;
- ambiguous and transition cases are rejected;
- all records follow the universal VQA data contract.

## 24. RoboDojo sidecar implementation and Isaac validation gate

The RoboDojo implementation writes an ego-only sidecar following
`01_vqa_data_format.md`, with head RGB, distance-to-image-plane, and instance
segmentation evidence.  This implementation note does **not** relax any
normative requirement in Sections 1--23.

In particular, the `gripper_content` scenario (what each gripper holds) is pre-validation code until
an Isaac run demonstrates the required gripper semantic-ID/interior-mask and
stable relative-pose checks for all three answers: `nothing`, `pen`, and `pen
holder`.  A held-object mask or a fixed number of rendered frames is not a
substitute for these checks. No `gripper_content` record is suitable for training until
this gate has passed.

Before admitting any generated `fill_pen_holder` record to training, retain a
small Isaac visual audit that proves all of the following:

- the four pen IDs and holder ID form a one-to-one semantic mapping;
- nib points satisfy local-mask and depth-alignment checks for visible,
  occluded, and out-of-frame cases;
- holder orientation, mask visibility, tight bbox, and the fallen dead zone
  meet this specification's thresholds;
- both grippers pass the `nothing` / `pen` / `pen holder` checks above; and
- table/holder counts use visible instance masks and reject transition frames.

Until that audit is explicitly run and reviewed, the sidecar is an
implementation artifact for schema and pipeline testing only, not an approved
training dataset.

## 25. CLI scene scheduler

The generator schedules **scene snapshots**, not individual VQA records.
`--scene-count` (with `--count` retained as an alias) is the number of
snapshots to render per selected layout. `--scenario` selects an ordered cycle
from `layout`, `gripper_content`, and `visible_counts`; `--case` is retained as
an alias for the same option. Snapshot `i` uses
`scenarios[i % len(scenarios)]`, while its deterministic within-scenario index
is `i // len(scenarios)`.

With the default scenario cycle, `--scene-count 3` renders three base ego
images: one `layout`, one `gripper_content`, and one `visible_counts` snapshot.
It does **not** mean three annotation records. The expected accepted-record
fan-out per answerable snapshot is:

| Scenario | Question families | Maximum records |
| --- | --- | ---: |
| `layout` | `holder_fallen`, `holder_bbox`, four `pen_nib_grounding` records | 6 |
| `gripper_content` | left and right gripper-content records | 2 |
| `visible_counts` | table and holder visible-pen counts | 2 |

The `layout` scenario may additionally save one numbered-overlay image. It is
saved only if all required instance masks and overlay anchors pass validation.
Use a single scenario during focused validation, for example
`--scenario layout --scene-count 1`.

For `gripper_content`, both arms receive independent, deterministic TCP
perturbations derived from the layout/scene seed, including an arm whose
answer is `nothing`. The default bounds are 4 cm in XY and 2.5 cm in Z; they
can be changed with `--gripper-position-jitter` and
`--gripper-height-jitter`. The commanded TCP position is retained in each
gripper record's audit metadata. IK or visibility failures are rejected rather
than converted into a fixed-pose sample.
