# Task-Specific VQA Specification: RoboDojo `fill_pen_holder`

This collector writes only the original ego `cam_head` RGB image. Instance
segmentation and depth are used only to validate physical annotations.

## Real robot-state pool

Run `scripts/internal/extract_fill_pen_holder_robot_states.py` before VQA
generation. It extracts every valid paired left/right `xyz + wxyz + gripper`
state from `data/fill_pen_holder` and retains source index, episode, frame, and
timestamp. Every generated scene samples one complete paired state; it is
loaded through IK for both robots together. A state that cannot be realized or
does not produce image-supported evidence is rejected rather than replaced.

## Question families

### `pen_nib_grounding`

```text
Locate the nib of {unique pen description} in the ego-view image. Return one point.
```

The description is grounded in a visible pen feature: an asset color when that
color is unique in the scene, otherwise one of `leftmost`, `rightmost`,
`closest to the left robot arm`, or `closest to the right robot arm`. The
collector keeps a point record only when its projection, local instance pixels,
and depth agree.

### `pen_holder_is_tipped_over`

```text
Has the pen holder fallen over? Answer yes or no.
```

The default scene schedule allocates half of snapshots to holder-state scenes.
Upright and tipped states are exactly balanced; tipped holder yaw spans the
eight 45-degree directions uniformly in each shuffled cycle.

### Remaining families

- `holder_bbox`: visible tight ego-image bounding box for the pen holder.
- `left_gripper_content` and `right_gripper_content`: `nothing`, `pen`, or
  `pen holder`, subject to visible grasp evidence.
- `visible_pens_on_table` and `visible_pens_in_holder`: integer counts based on
  simulator relation plus visible instance masks.

## Output invariants

- Every record references the same clean RGB file used as model input.
- Point and box coordinates are normalized in the original 640×480 ego image.
- `audit_metadata_json` records robot-state provenance and the chosen
  human-facing pen description, but never internal pen labels.
- Physical source layouts and the original LeRobot data are read-only.
