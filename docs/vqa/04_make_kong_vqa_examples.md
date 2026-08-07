# Task-Specific VQA Specification: RoboDojo `make_kong`

The collector writes the original ego `cam_head` RGB image. The face-up
opponent discard is the visual reference for the matching group among the 12
robot-side tiles.

## Question families

### `matching_tile_indices_to_push`

After the opponent tile is knocked down and before any matching robot-side tile
is knocked down, identify the matching group:

```text
After the face-up opponent tile is knocked down, which three tiles on our side should be knocked down? Return their 1-based left-to-right indices as a tuple.
```

The answer is an ordered short-text triple, for example `(2, 6, 11)`, where
indices refer to the complete robot-side row from left to right.

### `matching_tile_still_needs_action`

For each fallen-state pattern `000` through `111`, the collector asks all three
matching tiles:

```text
Based on the face-up reference tile and the current board state, does the {N}th tile from the left still need to be knocked down? Answer yes or no.
```

`yes` means this matching tile is still upright and needs an action. `no`
means it has already fallen. The complete bitmask set includes cases such as
`111`, `011`, and `101`.

### `kong_declaration_neighbor_state_reason`

The collector also produces control and error scenes for the immediate
nonmatching tile(s) beside the matching triplet during kong declaration. The
matching tiles are fallen, while zero, one, or two adjacent nonmatching tiles
are fallen according to the available left/right neighbours. This phase is
before drawing the replacement tile from the left stack and placing it at the
right end, so it is not a complete make_kong task state.

```text
During kong declaration, only tiles matching the face-up tile should be down. Before the left-stack draw, classify the {N}th tile: correct or nonmatching_fallen.
```

The short-text answer is `correct` when that nonmatching tile remains upright,
or `nonmatching_fallen` when it was incorrectly knocked down.
The default scheduling includes the all-upright control (`00`) as well as every
available one- and two-tile error pattern.

## Output invariants

- Every accepted row has visible reference and queried tile evidence in the
  clean ego image.
- Tuple answers use `short_text`; all per-tile decisions use `boolean`.
- Segmentation identity files in `audit/` validate annotations only and are not
  model inputs.
