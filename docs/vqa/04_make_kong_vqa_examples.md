# Task-Specific VQA Specification: RoboDojo `make_kong`

This document is the single source of truth for make_kong VQA.  It
consolidates the former `docs/vqa/05_make_kong_vqa_layouts.md` (layout
variants A and B) and `docs/vqa/adr/0001/0002` (decision records) into this
task spec; those files no longer exist.

The collector (`vqa_gen/make_kong/generate_make_kong_vqa.py`) renders A/B
variant layouts from `vqa_gen/make_kong/layouts/` and writes the original ego
`cam_head` RGB image per scene.  Instance segmentation and depth are used only
to validate annotations and to derive visible-tight boxes.

## Board model

- The robot-side row contains **14 tiles**, indexed 1-14 left-to-right: the
  four matching groups (tiles 1-12) plus the support pair `mahjong4_0/4_1`
  (tiles 13-14).
- The opponent robot arm knocks one discard face-up on the far side; it is the
  visual reference (`reference tile`). The three robot-side tiles sharing its
  face are the `matching group` (Target Kong Tiles) and are required to be
  knocked down regardless of their current state.

## Face vocabulary

The five canonical suits are fixed; every A/B layout uses all five (four
matching groups plus the support pair), so the closed vocabulary never changes
between layouts:

```text
Wan
Tong
Suo
Honor
Bonus
```

## Layout variants (A and B)

`vqa_gen/make_kong/generate_layouts.py` writes `make_kong_a_<id>.json` /
`make_kong_b_<id>.json` pairs into `vqa_gen/make_kong/layouts/`.

### Why variants are needed

The evaluation layouts (and the original `data_gen/make_kong` pool) hard-code
two shortcuts that would make the tile face irrelevant for VQA:

1. discard `mahjong5_0` always matches group `mahjong0_*`, and so on — the
   pushed-down tile's face is predictable from its slot;
2. every matching group is a contiguous block at a fixed position — the target
   group is predictable from the discard slot.

Variant A removes shortcut 1, variant B removes shortcut 2.

### Shared invariants (both variants)

- Scene geometry, labels, and poses come from the evaluation template; only
  `category_idx` changes per Mahjong record.
- The four matching groups use four pairwise-distinct faces, and the support
  pair (`mahjong4_0/4_1`, `mahjong9_0`) uses the fifth face.  Every layout
  therefore uses all five faces exactly once each as a semantic slot, which
  keeps every discard-to-group matching unambiguous.
- `other0/1/2` keep pairwise-distinct categories (faces may repeat).
- All eight semantic-slot categories (4 groups + support + 3 others) are
  pairwise distinct.
- The face band map (`MJ01..MJ05` -> `Wan/Tong/Suo/Honor/Bonus`) is defined in
  `vqa_gen/make_kong/tile_faces.py` and may be verified against the USD
  textures with `--verify-faces`.

### Variant A: random discard pairing

- The 12 kong slots remain contiguous group blocks (nominal labels).
- The four discard tiles get a uniform random permutation of the four group
  faces, including nominal slot-group pairs. Across layouts, every discard
  slot can therefore match every robot-side group with equal probability.

### Variant B: scattered group faces

- Reuses variant A's discard pairing (identical discard categories).
- The twelve kong slots receive a uniform random permutation of the multiset
  `{group0 x3, group1 x3, group2 x3, group3 x3}` — a group's three tiles may
  land anywhere in the row, contiguous or not.

### Layout manifest

`manifest.json` records per layout pair: `group_categories` /
`group_face_names` / `support_*` / `other_categories`; `discard_assignment`
(discard label -> category); `kong_scatter` (per-slot group index for variant
B); `generation_runs` (seeds and counts), `face_vocabulary`,
`total_layout_pairs`.  `load_vqa_layout_pool` re-validates every layout on
load and requires continuous per-variant IDs from 0.

### Consumer contract

`vqa_gen/make_kong/generate_make_kong_vqa.py` resolves each target group
entirely from the layout JSON: for a discard label, the matching group is the
three kong labels sharing its `category_idx`.  No pairing metadata is needed at
runtime; the manifest remains an audit artifact.

## Question families

For every target group the collector plans the coverage matrix (per variant):
zero fallen, one/two/three fallen matching tiles, 3+1, 3+2, and pure wrong
sets of 1-5 tiles. Every scene emits the four row questions below. A
deterministic, fallen-count-balanced subset additionally emits the
reference-discard localization question.

### `reference_discard_bbox`

The collector selects `variants × layout_ids × target_groups` scenes from
each 0-5 robot-side fallen-count stratum. This is the one-per-group count of
zero-fallen reference candidates in the selected layout scope. In selected
scenes, only the face-up reference discard receives a deterministic `x/y`
table-plane offset and yaw; the HDR lighting profile remains fixed.

```text
Locate the face-up reference discard on the opponent's side, knocked down by the opponent robot arm. Output its bounding box.
```

Answer type `bbox2d`; the answer is the reference discard's visible-tight ego
mask box. A perturbation is accepted only after settling when its mask is
visible, has at least 50% of the unperturbed mask pixel count, and has an
unambiguous segmentation identity. Each row records the derived seed, offset,
yaw, and attempt in audit metadata.

### `fallen_tile_bboxes`

Every scene:

```text
Which tiles in the 14-tile row on our side are currently knocked down? Output one bounding box per tile in left-to-right order, or 'none' if no tiles are down.
```

Answer type `bbox_list`; boxes are the visible tight masks of every fallen
tile in the full 14-tile row. The empty list is the `none` answer.

### `missing_matching_tile_bboxes`

Every scene:

```text
Which of the three tiles in the 14-tile row on our side that match the suit of the face-up reference discard on the opponent's side are still standing? Output one bounding box per tile in left-to-right order, or 'none' if all three are down.
```

Answer type `bbox_list`; boxes are the visible tight masks of the matching
tiles that should still be knocked down; the empty list is the `none` answer.

### `target_kong_tile_bboxes`

Every scene:

```text
Which three tiles in the 14-tile row on our side that match the suit of the face-up reference discard on the opponent's side? Output their bounding boxes in left-to-right order.
```

Answer type `bbox_list`; boxes are the visible-tight masks of all three Target
Kong Tiles, whether they are standing or fallen. Accepted records always
contain exactly three boxes in left-to-right order; if any target tile lacks
visible evidence, the record is rejected rather than returning a partial list
or `none`.

### `wrong_fallen_tile_bboxes`

Every scene:

```text
Which tiles in the 14-tile row on our side do not match the suit of the face-up reference discard on the opponent's side and are knocked down? Output one bounding box per tile in left-to-right order, or 'none' if none.
```

Answer type `bbox_list`; boxes are the visible tight masks of fallen
non-matching row tiles (including the support pair when fallen).

## Scene coverage matrix

Per layout variant and target group, with deterministic seeded random
selection (three samples per wrong-tile case):

| fallen | case ids | variant A rule | variant B rule |
| --- | --- | --- | --- |
| 0 | `f0` | - | - |
| 1 correct | `c1_p0..p2` | each matching tile | same |
| 1 wrong | `w1_s0..s2` | single non-matching tile | same |
| 2 correct | `c2_p01..p12` | each pair | same |
| 2 wrong | `w2_s0..s2` | contiguous pair | random pair |
| 3 correct | `c3` | full group | same |
| 3 wrong | `w3_s0..s2` | contiguous triple | random triple |
| 3+1 | `c3w1_L/R` (A), `c3w1_s0..s2` (B) | immediate left/right neighbour | random tile |
| 4 wrong | `w4_s0..s2` | contiguous block | random 4-set |
| 3+2 | `c3w2_LL/RR/LR` (A), `c3w2_s0..s2` (B) | two left / two right / one each | random pair |
| 5 wrong | `w5_s0..s2` | contiguous block | random 5-set |

Variant A edge groups degrade gracefully: infeasible cases are skipped and
recorded in the manifest's `coverage` section (e.g. no left neighbour for the
leftmost group).

## Output invariants

- Every accepted row requires visible reference-tile evidence and, per family,
  visible evidence of every answered tile.
- `reference_discard_bbox` uses one axis-aligned visible-tight box. Boxes for
  the other list-valued families may overlap even when their instance masks do
  not: each box is independently derived from its projected mask's min/max
  extent.
- `bbox_list` answers are sorted left-to-right (ascending `x_min`) because the
  prompt explicitly dictates "in left-to-right order", which overrides the
  contract's default top-left priority; the empty list is the canonical
  `none` answer. `target_kong_tile_bboxes` is the exception: it always has
  exactly three boxes and never uses `none`.
- Every record references the same clean RGB file used as model input.
- Point and box coordinates are normalized in the original 640x480 ego image.
- `audit_metadata_json` records variant, layout id, group index, discard label,
  case id, fallen/missing/wrong/target labels, and the discard suit.
- Source layouts and the action dataset are read-only.

## Decision records (merged ADRs)

### Layout variants A/B (former ADR 0001)

The evaluation and data-gen make_kong layouts hard-code the discard-to-group
pairing and the contiguous group blocks, which lets a VQA model answer
matching questions from slot positions alone.  We generate variant-A layouts
(deranged discard faces) and variant-B layouts (scattered group faces, reusing
A's pairing) so the tile face is always the only valid cue, and we require the
four group faces plus the support face to be pairwise distinct (all five
available faces) to keep every discard-to-group matching unambiguous.  This
replaces the original `data_gen/make_kong/generate_layouts.py` for VQA
purposes, which cannot express either variant.

### `bbox_list` answer type (former ADR 0002)

The make_kong VQA families answer "which tiles fell", "which tiles are
missing", and "which tiles fell incorrectly" as variable-length bounding-box
lists, possibly empty. The universal contract supports `bbox_list` as an
ordered list of normalized yxyx boxes with the empty list as the canonical
`none` answer, serialized through the plain word `none` and `;` separators.
We chose typed bbox lists over short-text serialization to keep spatial
tokenization and per-box validation; the extension lives in the
`vqa_gen/vqa/sidecar.py` copy, leaving the legacy sidecar in
`scripts/internal/vqa` for the pre-migration fill_pen_holder output.
