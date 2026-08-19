# ruff: noqa: E501
"""Launch a read-only local browser for physical VQA sidecar datasets.

Example:
    conda activate RoboDojo
    python vqa_viewer/visualize_vqa.py

Choose a sidecar directory in the browser after the server starts. The viewer
only reads ``annotations.parquet``, ``rejected.parquet``, manifest/report JSON
files, and referenced RGB images. It never modifies the dataset or writes
review labels. Each record also shows the canonical VLM input: the assembled
plain-text VLM prompt and serialized answer tokens (1024 location bins,
y-first points and boxes, ``;`` separators, and the plain word ``none``).
Bounding-box (including
``bbox_list``) and point answers can be drawn temporarily in the browser for
inspection; those guides are not saved.
"""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import pyarrow.parquet as pq

LOGGER = logging.getLogger("visualize_vqa")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
NUM_LOCATION_BINS = 1024
NONE_TOKEN = "none"
SEP_TOKEN = ";"
ANSWER_COLUMNS = (
    "answer_text",
    "answer_bool",
    "answer_int",
    "answer_point_yx_norm",
    "answer_bbox_yxyx_norm",
    "answer_int_list",
    "answer_bbox_list_yxyx_norm",
)
VIEW_COLUMNS = (
    "sample_id",
    "task_name",
    "question_family",
    "ego_image_reference",
    "image_width",
    "image_height",
    "prompt_text",
    "answer_type",
    *ANSWER_COLUMNS,
    "world_state_valid",
    "image_answerable",
    "visibility_status",
    "visible_fraction",
    "occlusion_ratio",
    "source_layout",
    "scene_id",
    "audit_metadata_json",
    "rejection_reason",
)


def _evaluation_answer(result: dict[str, Any]) -> Any:
    """Return the typed prediction value emitted by a VQA evaluation record."""

    answer_type = result.get("answer_type")
    value_columns = {
        "boolean": "boolean",
        "integer": "integer",
        "short_text": "text",
        "point2d": "point_yx_norm",
        "bbox2d": "bbox_yxyx_norm",
        "int_list": "int_list",
        "bbox_list": "bbox_list_yxyx_norm",
    }
    column = value_columns.get(answer_type)
    return result.get(column) if column else None


def _evaluation_outcome(prediction: dict[str, Any] | None) -> str:
    """Classify an evaluation row for viewer filtering.

    Discrete answers require exact match. Bounding boxes require IoU@0.75.
    A ``bbox_list`` whose GT and prediction are both empty is correct without
    an IoU value, as prescribed by the VQA data-format contract. Point answers
    intentionally remain ``evaluated`` because they have no binary threshold.
    """

    if prediction is None:
        return "not_evaluated"
    metrics = prediction.get("metrics") or {}
    result = prediction.get("vqa_result") or {}
    if metrics.get("valid") != 1.0 or result.get("valid") is not True:
        return "invalid"
    for key in ("exact_match", "sequence_exact_match"):
        if key in metrics:
            return "correct" if metrics[key] == 1.0 else "incorrect"

    answer_type = result.get("answer_type")
    is_bbox_list = (
        answer_type == "bbox_list" or "num_match" in metrics or "count_exact_match" in metrics
    )
    if is_bbox_list:
        if metrics.get("empty_list_correct") == 1.0:
            return (
                "correct"
                if metrics.get("num_match") == 1.0 and metrics.get("empty_list_correct") == 1.0
                else "incorrect"
            )
        if "num_match" in metrics and "iou_at_0.75" in metrics:
            return (
                "correct"
                if metrics.get("num_match") == 1.0 and metrics.get("iou_at_0.75") == 1.0
                else "incorrect"
            )
        return "evaluated"

    if answer_type == "point2d" or "normalized_l2" in metrics:
        return "evaluated"
    if answer_type == "bbox2d" or "iou" in metrics or "iou_at_0.75" in metrics:
        if "iou_at_0.75" in metrics:
            return "correct" if metrics["iou_at_0.75"] == 1.0 else "incorrect"
        if "iou" in metrics:
            return "correct" if metrics["iou"] >= 0.75 else "incorrect"
        return "evaluated"

    return "evaluated"


class EvaluationRun:
    """Read a VQA evaluator result directory without changing it."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"VQA evaluation directory does not exist: {self.root}")
        self.metrics = _read_json(self.root / "metrics.json")
        self.predictions = self._read_predictions()

    def _read_predictions(self) -> dict[str, dict[str, Any]]:
        path = self.root / "predictions.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing required VQA prediction file: {path}")
        predictions = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    prediction = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"malformed prediction JSON at {path}:{line_number}") from error
                sample_id = prediction.get("sample_id") if isinstance(prediction, dict) else None
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"prediction at {path}:{line_number} has no sample_id")
                predictions[sample_id] = prediction
        return predictions

    def summary(self) -> dict[str, Any]:
        outcomes: dict[str, int] = {}
        for prediction in self.predictions.values():
            outcome = _evaluation_outcome(prediction)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        return {
            "evaluation_dir": str(self.root),
            "prediction_count": len(self.predictions),
            "outcomes": outcomes,
            "metrics": self.metrics,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Local interface to bind; default: 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port; default: 8765.")
    return parser.parse_args()


def _answer_value(record: dict[str, Any]) -> Any:
    for column in ANSWER_COLUMNS:
        value = record.get(column)
        if value is not None:
            return value
    return None


def _answer_text(record: dict[str, Any]) -> str:
    value = _answer_value(record)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def _loc_token(value: float) -> str:
    quantized = min(NUM_LOCATION_BINS - 1, max(0, round(float(value) * (NUM_LOCATION_BINS - 1))))
    return f"<loc{quantized:04d}>"


def _serialize_answer(record: dict[str, Any]) -> str:
    """Serialize the typed answer exactly as the VLM answer target is emitted.

    Follows the canonical formats in docs/vqa/01 (1024 location bins, y-first
    point and box order, prompt-defined list order, ``;`` separators, and
    ``none`` for empty lists).
    """

    answer_type = record.get("answer_type")
    value = _answer_value(record)
    if answer_type == "boolean":
        return f"{'yes' if value else 'no'}<eos>"
    if answer_type == "integer":
        return f"{int(value)}<eos>"
    if answer_type == "short_text":
        return f"{value}<eos>"
    if answer_type == "point2d" and value is not None:
        y, x = value
        return f"{_loc_token(y)}{_loc_token(x)}<eos>"
    if answer_type == "bbox2d" and value is not None:
        y_min, x_min, y_max, x_max = value
        return (
            f"{_loc_token(y_min)}{_loc_token(x_min)}"
            f"{_loc_token(y_max)}{_loc_token(x_max)}<eos>"
        )
    if answer_type == "int_list":
        if not value:
            return f"{NONE_TOKEN}<eos>"
        return f"{SEP_TOKEN.join(str(int(item)) for item in value)}<eos>"
    if answer_type == "bbox_list":
        if not value:
            return f"{NONE_TOKEN}<eos>"
        boxes = []
        for box in value:
            y_min, x_min, y_max, x_max = box
            boxes.append(f"{_loc_token(y_min)}{_loc_token(x_min)}{_loc_token(y_max)}{_loc_token(x_max)}")
        return f"{SEP_TOKEN.join(boxes)}<eos>"
    return ""


def _vlm_prompt(record: dict[str, Any]) -> str:
    """Assemble the canonical VLM prompt block from the physical question."""

    return f"Question: {record.get('prompt_text')}\nAnswer:"


def _as_json_record(
    record: dict[str, Any],
    *,
    prediction: dict[str, Any] | None = None,
    evaluation_outcome: str = "not_evaluated",
) -> dict[str, Any]:
    value = _answer_value(record)
    result = prediction.get("vqa_result") if prediction else None
    predicted_answer = _evaluation_answer(result) if isinstance(result, dict) else None
    return {
        "sample_id": record.get("sample_id"),
        "task_name": record.get("task_name"),
        "question_family": record.get("question_family"),
        "image_reference": record.get("ego_image_reference"),
        "image_url": f"/image/{quote(str(record.get('ego_image_reference') or ''), safe='/')}",
        "image_width": record.get("image_width"),
        "image_height": record.get("image_height"),
        "prompt_text": record.get("prompt_text"),
        "answer_type": record.get("answer_type"),
        "answer": value,
        "answer_text": _answer_text(record),
        "vlm_prompt": _vlm_prompt(record),
        "vlm_answer": _serialize_answer(record),
        "world_state_valid": record.get("world_state_valid"),
        "image_answerable": record.get("image_answerable"),
        "visibility_status": record.get("visibility_status"),
        "visible_fraction": record.get("visible_fraction"),
        "occlusion_ratio": record.get("occlusion_ratio"),
        "source_layout": record.get("source_layout"),
        "scene_id": record.get("scene_id"),
        "audit_metadata": _parse_audit_metadata(record.get("audit_metadata_json")),
        "rejection_reason": record.get("rejection_reason"),
        "evaluation": None
        if prediction is None
        else {
            "outcome": evaluation_outcome,
            "metrics": prediction.get("metrics") or {},
            "raw_text": result.get("raw_text") if isinstance(result, dict) else None,
            "answer": predicted_answer,
            "answer_text": _answer_text(
                {
                    "answer_type": result.get("answer_type"),
                    "answer_bool": result.get("boolean"),
                    "answer_int": result.get("integer"),
                    "answer_point_yx_norm": result.get("point_yx_norm"),
                    "answer_bbox_yxyx_norm": result.get("bbox_yxyx_norm"),
                    "answer_int_list": result.get("int_list"),
                    "answer_bbox_list_yxyx_norm": result.get("bbox_list_yxyx_norm"),
                }
            )
            if isinstance(result, dict)
            else "",
            "valid": result.get("valid") if isinstance(result, dict) else None,
            "error": result.get("error") if isinstance(result, dict) else None,
        },
    }


def _parse_audit_metadata(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        LOGGER.warning("ignoring malformed JSON file %s: %s", path, error)
        return None
    return value if isinstance(value, dict) else None


class VqaDataset:
    """Read and filter a VQA sidecar without changing it."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"VQA sidecar directory does not exist: {self.root}")
        self.rows = {
            "accepted": self._read_rows("annotations.parquet", required=True),
            "rejected": self._read_rows("rejected.parquet", required=False),
        }
        self.manifest = _read_json(self.root / "manifest.json")
        self.report = _read_json(self.root / "report.json")

    def _read_rows(self, filename: str, *, required: bool) -> list[dict[str, Any]]:
        path = self.root / filename
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"missing required VQA annotation file: {path}")
            return []
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [column for column in VIEW_COLUMNS if column in schema_names]
        return pq.read_table(path, columns=columns).to_pylist()

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.root),
            "accepted_count": len(self.rows["accepted"]),
            "rejected_count": len(self.rows["rejected"]),
            "question_families": sorted(
                {
                    str(row.get("question_family"))
                    for source in self.rows.values()
                    for row in source
                    if row.get("question_family")
                }
            ),
            "answer_types": sorted(
                {
                    str(row.get("answer_type"))
                    for source in self.rows.values()
                    for row in source
                    if row.get("answer_type")
                }
            ),
            "visibility_statuses": sorted(
                {
                    str(row.get("visibility_status"))
                    for source in self.rows.values()
                    for row in source
                    if row.get("visibility_status")
                }
            ),
            "manifest": self.manifest,
            "report": self.report,
        }

    def query(self, params: dict[str, str], evaluation: EvaluationRun | None = None) -> dict[str, Any]:
        source = params.get("source", "accepted")
        if source not in self.rows:
            raise ValueError("source must be accepted or rejected")
        family = params.get("family", "")
        answer_type = params.get("answer_type", "")
        visibility = params.get("visibility", "")
        outcome = params.get("outcome", "")
        search = params.get("search", "").strip().lower()
        rows = self.rows[source]
        filtered = []
        for row in rows:
            if family and row.get("question_family") != family:
                continue
            if answer_type and row.get("answer_type") != answer_type:
                continue
            if visibility and row.get("visibility_status") != visibility:
                continue
            prediction = evaluation.predictions.get(row.get("sample_id")) if evaluation else None
            row_outcome = _evaluation_outcome(prediction)
            if outcome and row_outcome != outcome:
                continue
            if search:
                haystack = " ".join(
                    str(row.get(key) or "") for key in ("sample_id", "prompt_text", "question_family", "scene_id")
                ).lower()
                if search not in haystack:
                    continue
            filtered.append((row, prediction, row_outcome))
        offset = max(0, int(params.get("offset", "0")))
        limit = min(100, max(1, int(params.get("limit", "12"))))
        return {
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "rows": [
                _as_json_record(row, prediction=prediction, evaluation_outcome=row_outcome)
                for row, prediction, row_outcome in filtered[offset : offset + limit]
            ],
        }

    def image_path(self, reference: str) -> Path:
        candidate = (self.root / reference).resolve()
        image_root = (self.root / "images").resolve()
        if candidate.suffix.lower() not in IMAGE_SUFFIXES or not candidate.is_relative_to(image_root):
            raise FileNotFoundError("image path is outside this sidecar's images directory")
        if not candidate.is_file():
            raise FileNotFoundError(f"image does not exist: {reference}")
        return candidate


class DatasetNotLoadedError(RuntimeError):
    """The browser has not selected a VQA sidecar directory yet."""


class DatasetStore:
    """Keep the browser-selected dataset only in server memory."""

    def __init__(self) -> None:
        self._dataset: VqaDataset | None = None
        self._evaluation: EvaluationRun | None = None
        self._lock = threading.RLock()

    def load(self, dataset_dir: str) -> dict[str, Any]:
        if not dataset_dir.strip():
            raise ValueError("dataset_dir must be a non-empty local path")
        dataset = VqaDataset(Path(dataset_dir).expanduser())
        with self._lock:
            self._dataset = dataset
        return self.summary()

    def load_evaluation(self, evaluation_dir: str) -> dict[str, Any]:
        if not evaluation_dir.strip():
            raise ValueError("evaluation_dir must be a non-empty local path")
        evaluation = EvaluationRun(Path(evaluation_dir).expanduser())
        with self._lock:
            self._evaluation = evaluation
        return self.summary()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            if self._dataset is None:
                return {"loaded": False}
            summary = {"loaded": True, **self._dataset.summary()}
            if self._evaluation is not None:
                evaluation_summary = self._evaluation.summary()
                dataset_sample_ids = {
                    str(row.get("sample_id"))
                    for source in self._dataset.rows.values()
                    for row in source
                    if row.get("sample_id")
                }
                evaluation_summary["matched_dataset_count"] = len(
                    dataset_sample_ids.intersection(self._evaluation.predictions)
                )
                summary["evaluation"] = evaluation_summary
            return summary

    def require_dataset(self) -> VqaDataset:
        with self._lock:
            if self._dataset is None:
                raise DatasetNotLoadedError("select a VQA sidecar directory in the browser first")
            return self._dataset

    def query(self, params: dict[str, str]) -> dict[str, Any]:
        with self._lock:
            if self._dataset is None:
                raise DatasetNotLoadedError("select a VQA sidecar directory in the browser first")
            return self._dataset.query(params, self._evaluation)


class ViewerHandler(BaseHTTPRequestHandler):
    dataset_store: DatasetStore

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", HTML.encode("utf-8"))
                return
            if parsed.path == "/api/summary":
                self._send_json(self.dataset_store.summary())
                return
            if parsed.path == "/api/samples":
                query = {key: values[-1] for key, values in parse_qs(parsed.query).items() if values}
                self._send_json(self.dataset_store.query(query))
                return
            if parsed.path.startswith("/image/"):
                path = self.dataset_store.require_dataset().image_path(unquote(parsed.path.removeprefix("/image/")))
                self._send_file(path)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
        except DatasetNotLoadedError as error:
            self.send_error(HTTPStatus.CONFLICT, str(error))
        except (FileNotFoundError, ValueError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))
        except BrokenPipeError:
            return

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path not in {"/api/dataset", "/api/evaluation"}:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 16_384:
                raise ValueError("request body must contain a dataset path no longer than 16 KiB")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request JSON must contain an object")
            if parsed.path == "/api/dataset":
                if not isinstance(payload.get("dataset_dir"), str):
                    raise ValueError("request JSON must contain a string dataset_dir")
                self._send_json(self.dataset_store.load(payload["dataset_dir"]))
            else:
                if not isinstance(payload.get("evaluation_dir"), str):
                    raise ValueError("request JSON must contain a string evaluation_dir")
                self._send_json(self.dataset_store.load_evaluation(payload["evaluation_dir"]))
        except (json.JSONDecodeError, FileNotFoundError, ValueError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))
        except BrokenPipeError:
            return

    def log_message(self, fmt: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, value: Any) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._send_bytes(HTTPStatus.OK, "application/json; charset=utf-8", body)

    def _send_file(self, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_bytes(HTTPStatus.OK, content_type, path.read_bytes())

    def _send_bytes(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoboDojo VQA Viewer</title>
  <style>
    :root { color-scheme: dark; --bg:#10141b; --panel:#171d27; --line:#2b3545; --text:#edf2f7; --muted:#9aa7b9; --accent:#48d1b0; --warn:#ffb454; --bad:#ff7185; --good:#48d1b0; }
    * { box-sizing:border-box; } body { margin:0; color:var(--text); background:var(--bg); font:14px/1.45 ui-sans-serif,system-ui,sans-serif; }
    header { padding:20px 28px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:20px; align-items:center; }
    h1,h2,p { margin:0; } h1 { font-size:21px; } h2 { font-size:15px; } .muted { color:var(--muted); } .pill { padding:4px 9px; border-radius:999px; background:#203242; color:var(--accent); font-weight:600; }
    main { display:grid; grid-template-columns:280px minmax(0,1fr); min-height:calc(100vh - 73px); } aside { border-right:1px solid var(--line); padding:18px; } .panel { margin-bottom:16px; padding:14px; border:1px solid var(--line); border-radius:10px; background:var(--panel); }
    label { display:block; margin-top:10px; color:var(--muted); font-size:12px; } input,select,button { font:inherit; color:var(--text); border:1px solid var(--line); border-radius:7px; background:#10151d; padding:8px; width:100%; } button { cursor:pointer; } button:hover { border-color:var(--accent); } section { padding:18px 24px; min-width:0; }
    .toolbar { display:grid; grid-template-columns:minmax(160px,1fr) 160px 160px 160px; gap:10px; margin-bottom:14px; } .pager { display:flex; justify-content:space-between; align-items:center; margin:12px 0; gap:12px; } .pager button { width:auto; min-width:96px; }
    #grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; } .card { overflow:hidden; text-align:left; padding:0; background:var(--panel); } .image-wrap { position:relative; background:#0a0d12; aspect-ratio:4/3; } .image-wrap img,.image-wrap canvas { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; } .card-body { padding:12px; display:grid; gap:8px; } .family { color:var(--accent); font:600 12px ui-monospace,SFMono-Regular,monospace; } .answer { color:#111; background:var(--warn); padding:4px 7px; border-radius:5px; width:max-content; max-width:100%; overflow-wrap:anywhere; font-weight:700; } .result { padding:4px 7px; border-radius:5px; width:max-content; font-weight:700; text-transform:uppercase; font-size:11px; } .result.correct { background:var(--good); color:#111; } .result.incorrect,.result.invalid { background:var(--bad); color:#111; } .result.evaluated,.result.not_evaluated { background:#334156; color:var(--text); } .legend { color:var(--muted); font-size:12px; }
    dialog { width:min(1100px,96vw); border:1px solid var(--line); border-radius:12px; background:var(--panel); color:var(--text); padding:0; } dialog::backdrop { background:rgba(0,0,0,.7); } .detail { display:grid; grid-template-columns:minmax(0,2fr) minmax(280px,1fr); } .detail .image-wrap { margin:16px; } .detail-info { padding:20px 20px 20px 0; overflow-wrap:anywhere; } pre { overflow:auto; max-height:280px; background:#10151d; padding:10px; border-radius:7px; color:#c6d2df; white-space:pre-wrap; } .close { float:right; width:auto; } .empty { padding:45px; text-align:center; color:var(--muted); border:1px dashed var(--line); border-radius:10px; }
    @media(max-width:780px) { main { grid-template-columns:1fr; } aside { border-right:0; border-bottom:1px solid var(--line); } .toolbar,.detail { grid-template-columns:1fr; } .detail-info { padding:0 16px 20px; } }
  </style>
</head>
<body>
  <header><div><h1>RoboDojo VQA Viewer</h1><p id="dataset" class="muted">No dataset selected</p></div><span class="pill">read-only</span></header>
  <main>
    <aside><div class="panel"><h2>Select dataset</h2><label>VQA sidecar directory<input id="datasetPath" placeholder="/path/to/vqa_sidecar"></label><button id="loadDataset">Load read-only dataset</button><p id="loadStatus" class="muted"></p></div><div class="panel"><h2>Load evaluation</h2><label>Evaluation result directory<input id="evaluationPath" placeholder="/path/to/vqa_results/run"></label><button id="loadEvaluation">Load read-only evaluation</button><p id="evaluationStatus" class="muted"></p></div><div class="panel"><h2>Dataset</h2><p id="counts" class="muted"></p></div><div class="panel"><h2>Viewer controls</h2><label>Records<select id="source"><option value="accepted">Accepted</option><option value="rejected">Rejected</option></select></label><label>Question family<select id="family"></select></label><label>Answer type<select id="answerType"></select></label><label>Visibility<select id="visibility"></select></label><label>Evaluation result<select id="outcome"><option value="">All</option><option value="incorrect">Incorrect</option><option value="invalid">Invalid</option><option value="correct">Correct</option><option value="not_evaluated">Not evaluated</option></select></label><label>Search<input id="search" placeholder="sample ID, prompt, scene"></label></div><div class="panel"><h2>Run metadata</h2><pre id="manifest">No dataset selected.</pre></div><div class="panel"><h2>Evaluation metrics</h2><pre id="evaluationMetrics">No evaluation selected.</pre></div></aside>
    <section><div class="toolbar"><input id="quickSearch" placeholder="Search (Enter)"><select id="pageSize"><option>12</option><option>24</option><option>48</option></select><button id="clear">Clear filters</button><button id="refresh">Refresh</button></div><div class="pager"><button id="previous">← Previous</button><span id="pageInfo" class="muted"></span><button id="next">Next →</button></div><div id="grid"></div></section>
  </main>
  <dialog id="dialog"><button id="close" class="close">Close</button><div id="detail"></div></dialog>
  <script>
    const state = { offset:0, summary:null, rows:[] };
    const $ = id => document.getElementById(id);
    function option(select, value, text) { const node=document.createElement('option'); node.value=value; node.textContent=text; select.append(node); }
    function values() { return { source:$('source').value, family:$('family').value, answer_type:$('answerType').value, visibility:$('visibility').value, outcome:$('outcome').value, search:($('search').value || $('quickSearch').value).trim(), offset:state.offset, limit:$('pageSize').value }; }
    function guide(container, row) { const canvas=document.createElement('canvas'); canvas.width=row.image_width||640; canvas.height=row.image_height||480; const ctx=canvas.getContext('2d'); const drawPoint=(point,color)=>{if(!Array.isArray(point))return;ctx.fillStyle=color;ctx.strokeStyle='#111';ctx.lineWidth=3;ctx.beginPath();ctx.arc(point[1]*canvas.width,point[0]*canvas.height,8,0,Math.PI*2);ctx.fill();ctx.stroke();}; const drawBox=(box,color)=>{if(!Array.isArray(box)||box.length!==4)return;ctx.strokeStyle=color;ctx.lineWidth=4;ctx.strokeRect(box[1]*canvas.width,box[0]*canvas.height,(box[3]-box[1])*canvas.width,(box[2]-box[0])*canvas.height);}; const drawAnswer=(answer,color)=>{if(row.answer_type==='point2d')drawPoint(answer,color); if(row.answer_type==='bbox2d')drawBox(answer,color); if(row.answer_type==='bbox_list'&&Array.isArray(answer))answer.forEach(box=>drawBox(box,color));}; drawAnswer(row.answer,'#48d1b0'); if(row.evaluation)drawAnswer(row.evaluation.answer,'#ff7185'); container.append(canvas); if(row.evaluation&&['point2d','bbox2d','bbox_list'].includes(row.answer_type)){const legend=document.createElement('div');legend.className='legend';legend.textContent='green: GT · pink: prediction';container.append(legend);} }
    function image(row) { const wrap=document.createElement('div'); wrap.className='image-wrap'; const img=document.createElement('img'); img.src=row.image_url; img.alt=row.sample_id; wrap.append(img); if(['point2d','bbox2d','bbox_list'].includes(row.answer_type)) guide(wrap,row); return wrap; }
    function card(row) { const card=document.createElement('button'); card.className='card'; card.append(image(row)); const body=document.createElement('div'); body.className='card-body'; const evaluation=row.evaluation; body.innerHTML=`<div class="family">${escape(row.question_family||'unknown')}</div><div>${escape(row.prompt_text||'')}</div><div class="answer">GT: ${escape(row.vlm_answer||'—')}</div>${evaluation?`<div class="result ${escape(evaluation.outcome)}">${escape(evaluation.outcome)}</div><div>Pred: ${escape(evaluation.raw_text||evaluation.answer_text||'—')}</div>`:''}<div class="muted">${escape(row.sample_id||'')}</div>`; card.append(body); card.onclick=()=>show(row); return card; }
    function escape(text) { const node=document.createElement('span'); node.textContent=String(text); return node.innerHTML; }
    function show(row) { $('detail').replaceChildren(); const shell=document.createElement('div'); shell.className='detail'; const visual=image(row); const info=document.createElement('div'); info.className='detail-info'; const evaluation=row.evaluation; const evaluationDetail=evaluation?`<h2>Evaluation result</h2><p class="result ${escape(evaluation.outcome)}">${escape(evaluation.outcome)}</p><p><b>Raw prediction:</b></p><pre>${escape(evaluation.raw_text||'—')}</pre><p><b>Parsed prediction:</b></p><pre>${escape(evaluation.answer_text||'—')}</pre><p><b>Per-sample metrics:</b></p><pre>${escape(JSON.stringify(evaluation.metrics,null,2))}</pre>${evaluation.error?`<p><b>Parser error:</b> ${escape(evaluation.error)}</p>`:''}`:'<h2>Evaluation result</h2><p class="muted">No loaded evaluation contains this sample.</p>'; info.innerHTML=`<h2>${escape(row.question_family||'')}</h2><p class="muted">${escape(row.sample_id||'')}</p><h2>VLM input (question)</h2><pre>${escape(row.vlm_prompt||'')}</pre><p><b>Ground-truth VLM answer:</b></p><pre>${escape(row.vlm_answer||'—')}</pre><p><b>Answerability:</b> ${String(row.image_answerable)} · <b>Visibility:</b> ${escape(row.visibility_status||'')}</p>${row.rejection_reason?`<p><b>Rejection:</b> ${escape(row.rejection_reason)}</p>`:''}<h2>Physical ground truth</h2><pre>${escape(row.answer_text||'—')}</pre>${evaluationDetail}<h2>Audit metadata</h2><pre>${escape(JSON.stringify(row.audit_metadata,null,2))}</pre>`; shell.append(visual,info); $('detail').append(shell); $('dialog').showModal(); }
    async function requestJson(url, options) { const response=await fetch(url,options); const data=await response.json().catch(()=>({})); if(!response.ok) throw new Error(data.message||data.detail||response.statusText); return data; }
    function emptyGrid(message) { $('grid').innerHTML=`<div class="empty">${escape(message)}</div>`; $('pageInfo').textContent=''; $('previous').disabled=true; $('next').disabled=true; }
    async function loadSummary() { const data=await requestJson('/api/summary'); state.summary=data; if(!data.loaded){ $('dataset').textContent='No dataset selected'; $('counts').textContent='Choose a local VQA sidecar directory above.'; $('manifest').textContent='No dataset selected.'; $('evaluationMetrics').textContent='No evaluation selected.'; emptyGrid('Choose a VQA sidecar directory to begin review.'); return false; } $('dataset').textContent=data.dataset_dir; $('counts').textContent=`${data.accepted_count} accepted · ${data.rejected_count} rejected`; $('manifest').textContent=JSON.stringify(data.manifest||data.report||{},null,2); const evaluation=data.evaluation; $('evaluationMetrics').textContent=evaluation?JSON.stringify({matched_dataset_count:evaluation.matched_dataset_count,prediction_count:evaluation.prediction_count,outcomes:evaluation.outcomes,overall:evaluation.metrics?.overall,by_question_family:evaluation.metrics?.by_question_family},null,2):'No evaluation selected.'; [['family',data.question_families],['answerType',data.answer_types],['visibility',data.visibility_statuses]].forEach(([id,items])=>{const selected=$(id).value; const select=$(id); select.replaceChildren(); option(select,'','All'); items.forEach(value=>option(select,value,value)); select.value=selected;}); return true; }
    async function loadRows() { if(!state.summary?.loaded){ emptyGrid('Choose a VQA sidecar directory to begin review.'); return; } const query=new URLSearchParams(values()); const data=await requestJson('/api/samples?'+query); state.rows=data.rows; const grid=$('grid'); grid.replaceChildren(); if(!data.rows.length){grid.innerHTML='<div class="empty">No records match these filters.</div>';} else data.rows.forEach(row=>grid.append(card(row))); $('pageInfo').textContent=`${data.total ? data.offset+1 : 0}–${Math.min(data.offset+data.rows.length,data.total)} of ${data.total}`; $('previous').disabled=data.offset===0; $('next').disabled=data.offset+data.limit>=data.total; }
    function reset() { state.offset=0; loadRows(); } async function selectDataset() { const path=$('datasetPath').value.trim(); $('loadStatus').textContent='Loading…'; try { await requestJson('/api/dataset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_dir:path})}); const loaded=await loadSummary(); if(loaded) { state.offset=0; await loadRows(); $('loadStatus').textContent='Loaded read-only.'; } } catch(error) { $('loadStatus').textContent=`Unable to load: ${error.message}`; } } async function selectEvaluation() { const path=$('evaluationPath').value.trim(); $('evaluationStatus').textContent='Loading…'; try { await requestJson('/api/evaluation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({evaluation_dir:path})}); const loaded=await loadSummary(); if(loaded) { state.offset=0; await loadRows(); $('evaluationStatus').textContent='Loaded read-only.'; } } catch(error) { $('evaluationStatus').textContent=`Unable to load: ${error.message}`; } } ['source','family','answerType','visibility','outcome','search','pageSize'].forEach(id=>$(id).addEventListener('change',reset)); $('quickSearch').addEventListener('keydown',event=>{if(event.key==='Enter'){ $('search').value=$('quickSearch').value; reset(); }}); $('datasetPath').addEventListener('keydown',event=>{if(event.key==='Enter') selectDataset();}); $('evaluationPath').addEventListener('keydown',event=>{if(event.key==='Enter') selectEvaluation();}); $('loadDataset').onclick=selectDataset; $('loadEvaluation').onclick=selectEvaluation; $('clear').onclick=()=>{['family','answerType','visibility','outcome','search','quickSearch'].forEach(id=>$(id).value='');reset();}; $('refresh').onclick=()=>{loadSummary().then(loaded=>{if(loaded) reset();});}; $('previous').onclick=()=>{state.offset=Math.max(0,state.offset-Number($('pageSize').value));loadRows();}; $('next').onclick=()=>{state.offset+=Number($('pageSize').value);loadRows();}; $('close').onclick=()=>$('dialog').close(); document.addEventListener('keydown',event=>{if(event.key==='ArrowLeft'&&!$('dialog').open)$('previous').click(); if(event.key==='ArrowRight'&&!$('dialog').open)$('next').click();});
    loadSummary().catch(error=>{ emptyGrid(error); });
  </script>
</body></html>"""


def main() -> None:
    args = _parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    ViewerHandler.dataset_store = DatasetStore()
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    LOGGER.info("read-only VQA viewer: http://%s:%s", args.host, args.port)
    LOGGER.info("select a VQA sidecar directory in the browser to begin review")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("viewer stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
