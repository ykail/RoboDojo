# Universal VQA Data Format Contract for openpi π0.5

This contract defines ego-view VQA samples for the openpi π0.5 training
pipeline. It is task-independent; task-specific definitions live in separate
files (e.g. `04_make_kong_vqa_examples.md`). VQA uses only the ego-view image
as valid visual input. Two dataset sample types are supported: `action` and
`vqa`.

## 1. Basic VQA answer types

`answer_type` is a discriminator: exactly one answer field is populated per
record; all other answer fields are null.

| answer_type | storage field | content |
| --- | --- | --- |
| `point2d` | `answer_point_xy_norm` | `[x, y]`, both in `[0, 1]` |
| `bbox2d` | `answer_bbox_yxyx_norm` | `[y_min, x_min, y_max, x_max]` in `[0, 1]`, positive size, visible tight |
| `boolean` | `answer_bool` | `yes` / `no` |
| `short_text` | `answer_text` | canonical closed-set text |
| `integer` | `answer_int` | base-10 integer |
| `int_list` | `answer_int_list` | ordered integers; empty list == `none` |
| `bbox_list` | `answer_bbox_list_yxyx_norm` | ordered yxyx boxes; empty list == `none` |

Bounding-box storage order is **yxyx** (`y_min, x_min, y_max, x_max`), which is
also the model token order — no reordering happens at serialization. Point
storage stays `xy`; points swap to `yx` during serialization (`[loc_y][loc_x]`).

Rules:

* Coordinates are normalized to `[0, 1]` and refer to the original ego image.
* `bbox_list` boxes default to top-left priority sorting (ascending `y_min`, then ascending `x_min`); `int_list` elements are ascending. **If `question_family` or `prompt_text` explicitly dictates a spatial constraint (e.g., "left-to-right"), the explicit constraint overrides the default top-left priority sorting.**
* `none` is always the empty list (represented by explicit `<none>` token), never a sentinel box or coordinate.
* Invisible spatial targets are rejected, never assigned sentinel values.
* A record is trainable only when `world_state_valid` and `image_answerable` are both true.

## 2. Storage format

Physical annotation table (Parquet-friendly, nullable columns):

```text
sample_id, sample_type, source_dataset, task_name, question_family
episode_index, frame_index, timestamp
ego_image_reference, image_width, image_height
prompt_text, answer_type
answer_text, answer_bool, answer_int
answer_point_xy_norm, answer_bbox_yxyx_norm
answer_int_list, answer_bbox_list_yxyx_norm, answer_aliases
world_state_valid, image_answerable, visibility_status, visible_fraction, occlusion_ratio
target_view, coordinate_space, point_definition, bbox_definition
gt_source, annotation_version, quality_score, rejection_reason

```

Recommended values:

```text
target_view = "ego"
coordinate_space = "original_image_normalized_xy" | "original_image_normalized_yxyx"
bbox_definition = "visible_tight"

```

Visibility statuses: `visible`, `partially_visible`, `occluded`,
`out_of_frame`, `ambiguous`, `not_applicable`.

### Spatial serialization

* **Quantization Standard (1024 bins, range `[0, 1023]`):**
Unified with the PaliGemma / π0 pretrained base position token vocabulary.
```python
q = min(1023, max(0, round(clamp(value, 0.0, 1.0) * 1023)))
dequantized_value = q / 1023.0

```


* **Point:** Storage `[x, y]` -> model tokens `[loc_y][loc_x]` ($y$-first).
* *Example:* Storage `[0.30, 0.80]` ($x=0.30, y=0.80$) -> Tokens `<loc_818><loc_307>` ($0.80 \times 1023 = 818, 0.30 \times 1023 = 307$).


* **Bbox:** Storage `[y_min, x_min, y_max, x_max]` -> model tokens `[loc_ymin][loc_xmin][loc_ymax][loc_xmax]` (identical order; no reordering).
* One serializer/parser pair per answer type; round-trip error must not exceed one quantization bin.

## 3. Example data cases

### Answer serialization formats

| Category | Recommended format | Description |
| --- | --- | --- |
| BBox List | `<answer_bbox_list><loc_514><loc_451><loc_573><loc_491><sep><loc_514>...<eos>` | Coordinates uniformly quantized to `[0, 1023]`; ordering follows the prompt/rule-specified constraint; token order y-first: `ymin, xmin, ymax, xmax` |
| Empty BBox | `<answer_bbox_list><none><eos>` | Explicit empty-set special token |
| Int List | `<answer_int_list>2<sep>5<sep>8<eos>` | Keep ascending order |

Other types serialize as `<answer_point2d>...`, `<answer_bbox2d>...`,
`<answer_boolean>yes|no`, `<answer_short_text>{text}`,
`<answer_integer>{int}`, each terminated by `<eos>`.

### Prompt protocol

```text
<mode_vqa>
Question: {question}
Expected answer type: {answer_type}
State: {reserved masked state span}
Answer:

```

The serialized answer is appended during teacher-forced training.

### Worked record example (make_kong `missing_matching_tile_bboxes`)

Physical record:

```json
{
  "question_family": "missing_matching_tile_bboxes",
  "prompt_text": "Which of the three tiles matching the suit of the tile knocked down by the opponent are still standing? Output one bounding box per tile in left-to-right order, or 'none' if all three are down.",
  "answer_type": "bbox_list",
  "answer_bbox_list_yxyx_norm": [
    [0.5021, 0.4406, 0.5604, 0.4797],
    [0.5021, 0.4781, 0.5604, 0.5156],
    [0.5021, 0.5156, 0.5604, 0.5547]
  ],
  "target_view": "ego",
  "bbox_definition": "visible_tight",
  "world_state_valid": true,
  "image_answerable": true
}

```

VLM input (quantized with 1024 bins):

```text
<mode_vqa>
Question: Which of the three tiles matching the suit of the tile knocked down by the opponent are still standing? Output one bounding box per tile in left-to-right order, or 'none' if all three are down.
Expected answer type: bbox_list
State: {reserved masked state span}
Answer:
<answer_bbox_list><loc_514><loc_451><loc_573><loc_491><sep><loc_514><loc_489><loc_573><loc_527><sep><loc_514><loc_527><loc_573><loc_567><eos>

```