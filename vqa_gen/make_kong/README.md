# make_kong VQA generation (A/B variants)

This directory generates the VQA sidecar dataset for `make_kong` from
variant-A and variant-B tile layouts:

- `generate_layouts.py` — JSON-only A/B layout pool generator (no Isaac Sim);
- `generate_make_kong_vqa.py` — Isaac Sim collector that renders ego
  `cam_head` scenes and writes typed VQA annotations;
- `layouts/` — generated `make_kong_a_<id>.json` / `make_kong_b_<id>.json`
  pool (default output of the generator);
- `scene_plan.py`, `tile_faces.py` — pure logic used by both scripts.

Layout A keeps the matching groups contiguous but deranges the discard faces;
layout B reuses A's discard pairing and scatters the group faces across the
twelve kong slots.  See `docs/vqa/04_make_kong_vqa_examples.md`.

## Step 1 — generate the A/B layout pool (CPU only)

```bash
conda activate RoboDojo
python vqa_gen/make_kong/generate_layouts.py \
  --count 50 --rng-seed 2810 --verify-faces \
  --output-dir vqa_gen/make_kong/layouts/
```

Writes `make_kong_a_<id>.json` / `make_kong_b_<id>.json` pairs plus
`manifest.json` into `vqa_gen/make_kong/layouts/` (the collector's default
`--layout-root`).  `--verify-faces`
cross-checks the canonical face band map (MJ01..MJ05 ->
Wan/Tong/Suo/Honor/Bonus) against the mahjong USD textures.  Re-running
appends new pairs; IDs stay continuous per variant.

## Step 2 — collect the VQA sidecar (GPU, Isaac Sim)

```bash
conda activate RoboDojo
python vqa_gen/make_kong/generate_make_kong_vqa.py \
  --headless --enable_cameras --seed 2810 \
  --variant a,b --layout-ids all \
  --target-groups 0,1,2,3 \
  --output-dir output/RoboDojo_vqa_v3/make_kong_seed2810 --overwrite
```

- `--variant a,b` selects which layout variants to render; `--layout-ids`
  defaults to `all` (every pair in the pool); `--layout-root` overrides the
  pool directory (default: `vqa_gen/make_kong/layouts/`).
- Each scene writes one clean image plus four records (`fallen_tile_count`,
  `fallen_tile_indices`, `missing_matching_tile_bboxes`,
  `wrong_fallen_tile_bboxes`); the zero-fallen scene additionally emits
  `discard_suit`.
- Output contains `images/`, `audit/`, `annotations.parquet`,
  `rejected.parquet`, `manifest.json`, and `report.json`.  Rejected rows
  keep a stable `rejection_reason`.

## Small smoke test

```bash
conda activate RoboDojo
python vqa_gen/make_kong/generate_layouts.py --count 2 --rng-seed 0 --output-dir tmp/vqa_layouts
python vqa_gen/make_kong/generate_make_kong_vqa.py --headless --enable_cameras --device-id 0 --seed 0 \
  --variant a --layout-ids 0 --max-layouts 1 --target-groups 0,1 --fps 25 \
  --layout-root tmp/vqa_layouts --output-dir tmp/make_kong_vqa_smoke --overwrite
```

## Tests

```bash
conda activate RoboDojo
python -m unittest tests.test_make_kong_vqa_plan tests.test_make_kong_vqa_layouts tests.test_vqa_sidecar_vqagen
```
