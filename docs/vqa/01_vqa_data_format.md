# Universal VQA Data Format Contract for openpi π0.5

This contract defines ego-view VQA samples for the openpi π0.5 training
pipeline. It is task-independent; task-specific definitions live in separate
files. VQA uses only the ego-view image
as valid visual input. Two dataset sample types are supported: `action` and
`vqa`.

## 1. Basic VQA answer types

`answer_type` is a discriminator: exactly one answer field is populated per
record; all other answer fields are null.

| answer_type | storage field | content |
| --- | --- | --- |
| `point2d` | `answer_point_yx_norm` | `[y, x]`, both in `[0, 1]` |
| `bbox2d` | `answer_bbox_yxyx_norm` | `[y_min, x_min, y_max, x_max]` in `[0, 1]`, positive size, visible tight |
| `boolean` | `answer_bool` | `yes` / `no` |
| `short_text` | `answer_text` | canonical closed-set text |
| `integer` | `answer_int` | base-10 integer |
| `int_list` | `answer_int_list` | ordered integers; empty list == `none` |
| `bbox_list` | `answer_bbox_list_yxyx_norm` | ordered yxyx boxes; empty list == `none` |

Bounding-box storage order is **yxyx** (`y_min, x_min, y_max, x_max`), which is
also the model token order — no reordering happens at serialization. Point
storage is **yx** (`[y, x]`), also the model token order — no reordering
happens at serialization (`[loc_y][loc_x]`).

Rules:

* Coordinates are normalized to `[0, 1]` and refer to the original ego image.
* `bbox_list` boxes and `int_list` elements follow the ordering dictated by
  the question (`prompt_text`); make_kong dictates "left-to-right order" /
  ascending indices. The collector pre-sorts the stored answer accordingly,
  and the serializer **preserves the stored order** (it never re-sorts).
* `none` is always the empty list (serialized as the plain word `none`,
  never a sentinel box or coordinate).
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
answer_point_yx_norm, answer_bbox_yxyx_norm
answer_int_list, answer_bbox_list_yxyx_norm, answer_aliases
world_state_valid, image_answerable, visibility_status, visible_fraction, occlusion_ratio
target_view, coordinate_space, point_definition, bbox_definition
gt_source, annotation_version, quality_score, rejection_reason

```

Recommended values:

```text
target_view = "ego"
coordinate_space = "original_image_normalized_yx" | "original_image_normalized_yxyx"
bbox_definition = "visible_tight"

```

Visibility statuses: `visible`, `partially_visible`, `occluded`,
`out_of_frame`, `ambiguous`, `not_applicable`.

### Spatial serialization

* **Quantization Standard (1024 bins, range `[0, 1023]`):**
Unified with the PaliGemma / π0 pretrained base position token vocabulary,
which is the single token `<locXXXX>` (zero-padded to four digits, no
separator):
```python
q = min(1023, max(0, round(clamp(value, 0.0, 1.0) * 1023)))
dequantized_value = q / 1023.0

```


* **Point:** Storage `[y, x]` -> model tokens `[loc_y][loc_x]` (identical
  order; no reordering).
* *Example:* Storage `[0.80, 0.30]` ($y=0.80, x=0.30$) -> Tokens `<loc0818><loc0307>` ($0.80 \times 1023 = 818, 0.30 \times 1023 = 307$).


* **Bbox:** Storage `[y_min, x_min, y_max, x_max]` -> model tokens `[loc_ymin][loc_xmin][loc_ymax][loc_xmax]` (identical order; no reordering).
* One serializer/parser pair per answer type; round-trip error must not exceed one quantization bin.

## 3. Example data cases

### Answer serialization formats

The serialized answer is plain text over the pretrained PaliGemma vocabulary:
no custom markers (`<mode_vqa>`, `<answer_*>`, `<sep>`, `<none>`) are used.
List elements are joined with `;`; the empty list serializes to the plain
word `none`. The serialized answer is appended bare after `Answer:` during
teacher-forced training and terminated by `<eos>`.

| Category | Recommended format | Description |
| --- | --- | --- |
| BBox List | `<loc0514><loc0451><loc0573><loc0491>;<loc0514><loc0489><loc0573><loc0527><eos>` | Coordinates uniformly quantized to `[0, 1023]`; `;` separates boxes; ordering follows the prompt-specified constraint; token order y-first: `ymin, xmin, ymax, xmax` |
| Empty BBox | `none<eos>` | Plain-word empty-set answer |
| Int List | `2;5;8<eos>` | `;` separates elements; keep the prompt-dictated order (make_kong: ascending) |

Other types serialize as `{loc tokens}` for `point2d`/`bbox2d`, `yes|no` for
`boolean`, `{text}` for `short_text`, `{int}` for `integer`, each terminated
by `<eos>`.

### Prompt protocol

```text
Question: {question}
Answer:
```

The serialized answer is appended during teacher-forced training. The VQA
prompt contains no mode/answer-type markers and no robot state.

### Worked record example (make_kong `missing_matching_tile_bboxes`)

Physical record:

```json
{
  "question_family": "missing_matching_tile_bboxes",
  "prompt_text": "Which of the three tiles in the 14-tile row on our side that match the suit of the face-up reference discard on the opponent's side are still standing? Output one bounding box per tile in left-to-right order, or 'none' if all three are down.",
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
Question: Which of the three tiles in the 14-tile row on our side that match the suit of the face-up reference discard on the opponent's side are still standing? Output one bounding box per tile in left-to-right order, or 'none' if all three are down.
Answer:
<loc0514><loc0451><loc0573><loc0491>;<loc0514><loc0489><loc0573><loc0527>;<loc0514><loc0527><loc0573><loc0567><eos>
```

## 4. Evaluation output format

Run evaluation through `cmds/eval_vqa.sh`. It writes an appendable
`predictions.jsonl` and a summarized `metrics.json` in the chosen output
directory. Saved predictions can be scored again offline without contacting a
policy server; the offline scorer rebuilds both files from each record's saved
`target` and `vqa_result`.

### Per-sample records (`predictions.jsonl`)

Every JSONL record contains the sample identity and prompt, the typed GT in
`target`, the parsed policy response in `vqa_result`, and the resulting
`metrics`. `metrics.valid` is a format-validity flag only: `1.0` means the
response parsed according to its declared `answer_type`; it does not mean the
answer is semantically correct.

Common type-specific fields are:

| answer_type | Per-sample metric fields |
| --- | --- |
| `boolean`, `short_text`, `integer` | `exact_match`; `integer` also has `absolute_error` |
| `point2d` | `normalized_l2` |
| `bbox2d` | `iou`, `iou_at_0.5`, `iou_at_0.75` |
| `int_list` | `num_match`, `exact_match`, `edit_distance` |
| `bbox_list` | `num_match`, `empty_gt`, `empty_list_correct`, `iou_box_count`, and, when at least one box is evaluated, `iou_values`, `mean_iou`, `median_iou`, `iou_at_0.5`, `iou_at_0.75` |

`int_list.exact_match` requires identical length, order, and integer values.
For `bbox_list`, corresponding boxes are compared positionally in the
prompt-defined order. If the list lengths differ, each unpaired predicted or
GT box contributes an IoU of zero. Therefore a list count mismatch is visible
both in `num_match=0` and in the spatial scores.

When both the GT and prediction are empty (`none`), `num_match=1` and
`empty_list_correct=1`. No IoU is emitted for that sample because there is no
box to compare. If only one side is empty, the list does not match and the
unpaired boxes contribute zero IoU.

### Aggregate report (`metrics.json`)

`metrics.json` contains `overall`, `by_answer_type`, and
`by_question_family`, plus run metadata such as `dataset_root` and
`server_url`. `overall` intentionally contains only the universal fields
`sample_count`, `valid_count`, and `valid_rate`; it does not combine
incompatible metrics such as text exact match and spatial IoU.

Each answer-type or homogeneous question-family summary contains those same
universal fields and its relevant aggregate metrics:

| answer_type | Aggregate metric fields |
| --- | --- |
| `boolean`, `short_text` | `exact_match` |
| `integer` | `exact_match`, `mean_absolute_error`, `median_absolute_error` |
| `point2d` | `mean_normalized_l2`, `median_normalized_l2` |
| `bbox2d` | `iou_valid_count`, `mean_iou`, `median_iou`, `iou_at_0.5_count/rate`, `iou_at_0.75_count/rate` |
| `int_list` | `num_match_count/rate`, `exact_match`, `edit_distance_valid_count`, `mean_edit_distance`, `median_edit_distance` |
| `bbox_list` | `num_match_count/rate`, `empty_gt_count`, `empty_list_correct_count/rate`, `iou_box_count`, `mean_iou`, `median_iou`, `iou_at_0.5`, `iou_at_0.75` |

`bbox_list.iou_box_count` is the number of individual box comparisons in the
spatial aggregate, including zero-IoU unmatched boxes but excluding samples
where both lists are empty. `iou_at_0.5` and `iou_at_0.75` are proportions over
this box count, while `mean_iou` and `median_iou` are computed over the same
individual IoU values.
