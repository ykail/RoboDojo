# ruff: noqa: E501
"""Launch a read-only local browser for physical VQA sidecar datasets.

Example:
    conda activate RoboDojo
    python vqa_viewer/visualize_vqa.py

Choose a sidecar directory in the browser after the server starts. The viewer
only reads ``annotations.parquet``, ``rejected.parquet``, manifest/report JSON
files, and referenced RGB images. It never modifies the dataset or writes
review labels. Each record also shows the canonical VLM input: the assembled
``<mode_vqa>`` prompt and the serialized answer tokens (1024 location bins,
y-first boxes, ``<sep>``/``<none>`` markers). Bounding-box (including
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
NONE_TOKEN = "<none>"
SEP_TOKEN = "<sep>"
ANSWER_COLUMNS = (
    "answer_text",
    "answer_bool",
    "answer_int",
    "answer_point_xy_norm",
    "answer_bbox_xyxy_norm",
    "answer_bbox_yxyx_norm",
    "answer_int_list",
    "answer_bbox_list_xyxy_norm",
    "answer_bbox_list_yxyx_norm",
)
# Legacy collectors (scripts/internal/vqa, fill_pen_holder) store xyxy boxes;
# the vqa_gen contract stores yxyx.  The viewer normalizes everything to yxyx.
BBOX_CONVENTIONS = {
    "answer_bbox_xyxy_norm": "xyxy",
    "answer_bbox_yxyx_norm": "yxyx",
    "answer_bbox_list_xyxy_norm": "xyxy",
    "answer_bbox_list_yxyx_norm": "yxyx",
}
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Local interface to bind; default: 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port; default: 8765.")
    return parser.parse_args()


def _to_yxyx(value: Any, convention: str) -> Any:
    """Convert one box or a list of boxes from xyxy to yxyx storage order."""

    if convention == "yxyx":
        return value
    if value and isinstance(value[0], (list, tuple)):
        return [[box[1], box[0], box[3], box[2]] for box in value]
    return [value[1], value[0], value[3], value[2]]


def _answer_value(record: dict[str, Any]) -> Any:
    for column in ANSWER_COLUMNS:
        value = record.get(column)
        if value is not None:
            convention = BBOX_CONVENTIONS.get(column)
            return _to_yxyx(value, convention) if convention else value
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
    return f"<loc_{quantized:03d}>"


def _serialize_answer(record: dict[str, Any]) -> str:
    """Serialize the typed answer exactly as the VLM answer target is emitted.

    Follows the canonical formats in docs/vqa/01 (1024 location bins, y-first
    box order, top-left priority, ``<sep>`` separators, ``<none>`` for empty
    lists).
    """

    answer_type = record.get("answer_type")
    value = _answer_value(record)
    if answer_type == "boolean":
        return f"<answer_boolean>{'yes' if value else 'no'}<eos>"
    if answer_type == "integer":
        return f"<answer_integer>{int(value)}<eos>"
    if answer_type == "short_text":
        return f"<answer_short_text>{value}<eos>"
    if answer_type == "point2d" and value is not None:
        x, y = value
        return f"<answer_point2d>{_loc_token(y)}{_loc_token(x)}<eos>"
    if answer_type == "bbox2d" and value is not None:
        y_min, x_min, y_max, x_max = value
        return (
            f"<answer_bbox2d>{_loc_token(y_min)}{_loc_token(x_min)}"
            f"{_loc_token(y_max)}{_loc_token(x_max)}<eos>"
        )
    if answer_type == "int_list":
        if not value:
            return f"<answer_int_list>{NONE_TOKEN}<eos>"
        return f"<answer_int_list>{SEP_TOKEN.join(str(int(item)) for item in value)}<eos>"
    if answer_type == "bbox_list":
        if not value:
            return f"<answer_bbox_list>{NONE_TOKEN}<eos>"
        boxes = []
        for box in value:
            y_min, x_min, y_max, x_max = box
            boxes.append(f"{_loc_token(y_min)}{_loc_token(x_min)}{_loc_token(y_max)}{_loc_token(x_max)}")
        return f"<answer_bbox_list>{SEP_TOKEN.join(boxes)}<eos>"
    return ""


def _vlm_prompt(record: dict[str, Any]) -> str:
    """Assemble the canonical VLM prompt block from the physical question."""

    return (
        "<mode_vqa>\n"
        f"Question: {record.get('prompt_text')}\n"
        f"Expected answer type: {record.get('answer_type')}\n"
        "State: {reserved masked state span}\n"
        "Answer:"
    )


def _as_json_record(record: dict[str, Any]) -> dict[str, Any]:
    value = _answer_value(record)
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

    def query(self, params: dict[str, str]) -> dict[str, Any]:
        source = params.get("source", "accepted")
        if source not in self.rows:
            raise ValueError("source must be accepted or rejected")
        family = params.get("family", "")
        answer_type = params.get("answer_type", "")
        visibility = params.get("visibility", "")
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
            if search:
                haystack = " ".join(
                    str(row.get(key) or "") for key in ("sample_id", "prompt_text", "question_family", "scene_id")
                ).lower()
                if search not in haystack:
                    continue
            filtered.append(row)
        offset = max(0, int(params.get("offset", "0")))
        limit = min(100, max(1, int(params.get("limit", "12"))))
        return {
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "rows": [_as_json_record(row) for row in filtered[offset : offset + limit]],
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
        self._lock = threading.RLock()

    def load(self, dataset_dir: str) -> dict[str, Any]:
        if not dataset_dir.strip():
            raise ValueError("dataset_dir must be a non-empty local path")
        dataset = VqaDataset(Path(dataset_dir).expanduser())
        with self._lock:
            self._dataset = dataset
        return self.summary()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            if self._dataset is None:
                return {"loaded": False}
            return {"loaded": True, **self._dataset.summary()}

    def require_dataset(self) -> VqaDataset:
        with self._lock:
            if self._dataset is None:
                raise DatasetNotLoadedError("select a VQA sidecar directory in the browser first")
            return self._dataset


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
                self._send_json(self.dataset_store.require_dataset().query(query))
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
            if parsed.path != "/api/dataset":
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 16_384:
                raise ValueError("request body must contain a dataset path no longer than 16 KiB")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("dataset_dir"), str):
                raise ValueError("request JSON must contain a string dataset_dir")
            self._send_json(self.dataset_store.load(payload["dataset_dir"]))
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
    :root { color-scheme: dark; --bg:#10141b; --panel:#171d27; --line:#2b3545; --text:#edf2f7; --muted:#9aa7b9; --accent:#48d1b0; --warn:#ffb454; }
    * { box-sizing:border-box; } body { margin:0; color:var(--text); background:var(--bg); font:14px/1.45 ui-sans-serif,system-ui,sans-serif; }
    header { padding:20px 28px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:20px; align-items:center; }
    h1,h2,p { margin:0; } h1 { font-size:21px; } h2 { font-size:15px; } .muted { color:var(--muted); } .pill { padding:4px 9px; border-radius:999px; background:#203242; color:var(--accent); font-weight:600; }
    main { display:grid; grid-template-columns:280px minmax(0,1fr); min-height:calc(100vh - 73px); } aside { border-right:1px solid var(--line); padding:18px; } .panel { margin-bottom:16px; padding:14px; border:1px solid var(--line); border-radius:10px; background:var(--panel); }
    label { display:block; margin-top:10px; color:var(--muted); font-size:12px; } input,select,button { font:inherit; color:var(--text); border:1px solid var(--line); border-radius:7px; background:#10151d; padding:8px; width:100%; } button { cursor:pointer; } button:hover { border-color:var(--accent); } section { padding:18px 24px; min-width:0; }
    .toolbar { display:grid; grid-template-columns:minmax(160px,1fr) 160px 160px 160px; gap:10px; margin-bottom:14px; } .pager { display:flex; justify-content:space-between; align-items:center; margin:12px 0; gap:12px; } .pager button { width:auto; min-width:96px; }
    #grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; } .card { overflow:hidden; text-align:left; padding:0; background:var(--panel); } .image-wrap { position:relative; background:#0a0d12; aspect-ratio:4/3; } .image-wrap img,.image-wrap canvas { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; } .card-body { padding:12px; display:grid; gap:8px; } .family { color:var(--accent); font:600 12px ui-monospace,SFMono-Regular,monospace; } .answer { color:#111; background:var(--warn); padding:4px 7px; border-radius:5px; width:max-content; max-width:100%; overflow-wrap:anywhere; font-weight:700; }
    dialog { width:min(1100px,96vw); border:1px solid var(--line); border-radius:12px; background:var(--panel); color:var(--text); padding:0; } dialog::backdrop { background:rgba(0,0,0,.7); } .detail { display:grid; grid-template-columns:minmax(0,2fr) minmax(280px,1fr); } .detail .image-wrap { margin:16px; } .detail-info { padding:20px 20px 20px 0; overflow-wrap:anywhere; } pre { overflow:auto; max-height:280px; background:#10151d; padding:10px; border-radius:7px; color:#c6d2df; white-space:pre-wrap; } .close { float:right; width:auto; } .empty { padding:45px; text-align:center; color:var(--muted); border:1px dashed var(--line); border-radius:10px; }
    @media(max-width:780px) { main { grid-template-columns:1fr; } aside { border-right:0; border-bottom:1px solid var(--line); } .toolbar,.detail { grid-template-columns:1fr; } .detail-info { padding:0 16px 20px; } }
  </style>
</head>
<body>
  <header><div><h1>RoboDojo VQA Viewer</h1><p id="dataset" class="muted">No dataset selected</p></div><span class="pill">read-only</span></header>
  <main>
    <aside><div class="panel"><h2>Select dataset</h2><label>VQA sidecar directory<input id="datasetPath" placeholder="/path/to/vqa_sidecar"></label><button id="loadDataset">Load read-only dataset</button><p id="loadStatus" class="muted"></p></div><div class="panel"><h2>Dataset</h2><p id="counts" class="muted"></p></div><div class="panel"><h2>Viewer controls</h2><label>Records<select id="source"><option value="accepted">Accepted</option><option value="rejected">Rejected</option></select></label><label>Question family<select id="family"></select></label><label>Answer type<select id="answerType"></select></label><label>Visibility<select id="visibility"></select></label><label>Search<input id="search" placeholder="sample ID, prompt, scene"></label></div><div class="panel"><h2>Run metadata</h2><pre id="manifest">No dataset selected.</pre></div></aside>
    <section><div class="toolbar"><input id="quickSearch" placeholder="Search (Enter)"><select id="pageSize"><option>12</option><option>24</option><option>48</option></select><button id="clear">Clear filters</button><button id="refresh">Refresh</button></div><div class="pager"><button id="previous">← Previous</button><span id="pageInfo" class="muted"></span><button id="next">Next →</button></div><div id="grid"></div></section>
  </main>
  <dialog id="dialog"><button id="close" class="close">Close</button><div id="detail"></div></dialog>
  <script>
    const state = { offset:0, summary:null, rows:[] };
    const $ = id => document.getElementById(id);
    function option(select, value, text) { const node=document.createElement('option'); node.value=value; node.textContent=text; select.append(node); }
    function values() { return { source:$('source').value, family:$('family').value, answer_type:$('answerType').value, visibility:$('visibility').value, search:($('search').value || $('quickSearch').value).trim(), offset:state.offset, limit:$('pageSize').value }; }
    function guide(container, row) { const canvas=document.createElement('canvas'); canvas.width=row.image_width||640; canvas.height=row.image_height||480; const ctx=canvas.getContext('2d'); const answer=row.answer; if(row.answer_type==='point2d' && Array.isArray(answer)){ctx.fillStyle='#ffb454';ctx.strokeStyle='#111';ctx.lineWidth=3;ctx.beginPath();ctx.arc(answer[0]*canvas.width,answer[1]*canvas.height,8,0,Math.PI*2);ctx.fill();ctx.stroke();} const drawBox=box=>{if(Array.isArray(box)&&box.length===4)ctx.strokeRect(box[1]*canvas.width,box[0]*canvas.height,(box[3]-box[1])*canvas.width,(box[2]-box[0])*canvas.height);}; if(row.answer_type==='bbox2d' && Array.isArray(answer)){ctx.strokeStyle='#48d1b0';ctx.lineWidth=4;drawBox(answer);} if(row.answer_type==='bbox_list' && Array.isArray(answer)){ctx.strokeStyle='#48d1b0';ctx.lineWidth=4;answer.forEach(drawBox);} container.append(canvas); }
    function image(row) { const wrap=document.createElement('div'); wrap.className='image-wrap'; const img=document.createElement('img'); img.src=row.image_url; img.alt=row.sample_id; wrap.append(img); if(['point2d','bbox2d','bbox_list'].includes(row.answer_type)) guide(wrap,row); return wrap; }
    function card(row) { const card=document.createElement('button'); card.className='card'; card.append(image(row)); const body=document.createElement('div'); body.className='card-body'; body.innerHTML=`<div class="family">${escape(row.question_family||'unknown')}</div><div>${escape(row.prompt_text||'')}</div><div class="answer">${escape(row.vlm_answer||'—')}</div><div class="muted">${escape(row.sample_id||'')}</div>`; card.append(body); card.onclick=()=>show(row); return card; }
    function escape(text) { const node=document.createElement('span'); node.textContent=String(text); return node.innerHTML; }
    function show(row) { $('detail').replaceChildren(); const shell=document.createElement('div'); shell.className='detail'; const visual=image(row); const info=document.createElement('div'); info.className='detail-info'; info.innerHTML=`<h2>${escape(row.question_family||'')}</h2><p class="muted">${escape(row.sample_id||'')}</p><h2>VLM input (question)</h2><pre>${escape(row.vlm_prompt||'')}</pre><p><b>VLM answer:</b></p><pre>${escape(row.vlm_answer||'—')}</pre><p><b>Answerability:</b> ${String(row.image_answerable)} · <b>Visibility:</b> ${escape(row.visibility_status||'')}</p>${row.rejection_reason?`<p><b>Rejection:</b> ${escape(row.rejection_reason)}</p>`:''}<h2>Physical answer</h2><pre>${escape(row.answer_text||'—')}</pre><h2>Audit metadata</h2><pre>${escape(JSON.stringify(row.audit_metadata,null,2))}</pre>`; shell.append(visual,info); $('detail').append(shell); $('dialog').showModal(); }
    async function requestJson(url, options) { const response=await fetch(url,options); const data=await response.json().catch(()=>({})); if(!response.ok) throw new Error(data.message||data.detail||response.statusText); return data; }
    function emptyGrid(message) { $('grid').innerHTML=`<div class="empty">${escape(message)}</div>`; $('pageInfo').textContent=''; $('previous').disabled=true; $('next').disabled=true; }
    async function loadSummary() { const data=await requestJson('/api/summary'); state.summary=data; if(!data.loaded){ $('dataset').textContent='No dataset selected'; $('counts').textContent='Choose a local VQA sidecar directory above.'; $('manifest').textContent='No dataset selected.'; emptyGrid('Choose a VQA sidecar directory to begin review.'); return false; } $('dataset').textContent=data.dataset_dir; $('counts').textContent=`${data.accepted_count} accepted · ${data.rejected_count} rejected`; $('manifest').textContent=JSON.stringify(data.manifest||data.report||{},null,2); [['family',data.question_families],['answerType',data.answer_types],['visibility',data.visibility_statuses]].forEach(([id,items])=>{const select=$(id); select.replaceChildren(); option(select,'','All'); items.forEach(value=>option(select,value,value));}); return true; }
    async function loadRows() { if(!state.summary?.loaded){ emptyGrid('Choose a VQA sidecar directory to begin review.'); return; } const query=new URLSearchParams(values()); const data=await requestJson('/api/samples?'+query); state.rows=data.rows; const grid=$('grid'); grid.replaceChildren(); if(!data.rows.length){grid.innerHTML='<div class="empty">No records match these filters.</div>';} else data.rows.forEach(row=>grid.append(card(row))); $('pageInfo').textContent=`${data.total ? data.offset+1 : 0}–${Math.min(data.offset+data.rows.length,data.total)} of ${data.total}`; $('previous').disabled=data.offset===0; $('next').disabled=data.offset+data.limit>=data.total; }
    function reset() { state.offset=0; loadRows(); } async function selectDataset() { const path=$('datasetPath').value.trim(); $('loadStatus').textContent='Loading…'; try { await requestJson('/api/dataset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_dir:path})}); const loaded=await loadSummary(); if(loaded) { state.offset=0; await loadRows(); $('loadStatus').textContent='Loaded read-only.'; } } catch(error) { $('loadStatus').textContent=`Unable to load: ${error.message}`; } } ['source','family','answerType','visibility','search','pageSize'].forEach(id=>$(id).addEventListener('change',reset)); $('quickSearch').addEventListener('keydown',event=>{if(event.key==='Enter'){ $('search').value=$('quickSearch').value; reset(); }}); $('datasetPath').addEventListener('keydown',event=>{if(event.key==='Enter') selectDataset();}); $('loadDataset').onclick=selectDataset; $('clear').onclick=()=>{['family','answerType','visibility','search','quickSearch'].forEach(id=>$(id).value='');reset();}; $('refresh').onclick=()=>{loadSummary().then(loaded=>{if(loaded) reset();});}; $('previous').onclick=()=>{state.offset=Math.max(0,state.offset-Number($('pageSize').value));loadRows();}; $('next').onclick=()=>{state.offset+=Number($('pageSize').value);loadRows();}; $('close').onclick=()=>$('dialog').close(); document.addEventListener('keydown',event=>{if(event.key==='ArrowLeft'&&!$('dialog').open)$('previous').click(); if(event.key==='ArrowRight'&&!$('dialog').open)$('next').click();});
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
