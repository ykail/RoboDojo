# VQA Sidecar Viewer

Use the local read-only viewer to inspect VQA PNG images together with their
Parquet records without manually opening one file at a time.

```bash
conda activate RoboDojo
python vqa_viewer/visualize_vqa.py
```

Open `http://127.0.0.1:8765` in a browser. Use `--port` to select a different
local port. The viewer binds to `127.0.0.1` by default. Paste or type a local
VQA sidecar directory into the **Select dataset** panel, then click **Load
read-only dataset**. The server does not receive a dataset path at startup.
It only reads the selected dataset's:

- `annotations.parquet` and `rejected.parquet`;
- `manifest.json` and `report.json` when present;
- RGB files under `images/` referenced by the annotation rows.

It supports accepted/rejected-record switching, question-family/answer-type/
visibility filtering, text search, pagination, and detailed audit metadata.
For point and bounding-box answers, it draws a temporary browser-only guide
over the displayed image. It never alters the stored PNG or writes labels,
Parquet files, caches, or review results.
