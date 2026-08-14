"""Typed physical VQA annotation storage and validation utilities.

The functions in this module deliberately operate on physical annotation
records, not model tokens.  They are usable without Isaac Sim and are shared
by the vqa_gen collectors.

It evolved from the legacy ``scripts.internal.vqa.sidecar``: it adds the
variable-size ``int_list`` and ``bbox_list`` answer types whose empty value
encodes the canonical ``none`` answer, and stores points and boxes in model-token
order (``yx`` and ``yxyx``).
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

ANNOTATION_VERSION = "robodojo_vqa_v3"
SOURCE_DATASET = "RoboDojo_vqa_synthetic_v3"
ANSWER_FIELDS = (
    "answer_text",
    "answer_bool",
    "answer_int",
    "answer_point_yx_norm",
    "answer_bbox_yxyx_norm",
    "answer_int_list",
    "answer_bbox_list_yxyx_norm",
)
ANSWER_TYPES = {"point2d", "bbox2d", "boolean", "short_text", "integer", "int_list", "bbox_list"}
VISIBILITY_STATUSES = {
    "not_applicable",
    "visible",
    "partially_visible",
    "occluded",
    "out_of_frame",
    "ambiguous",
}


class ValidationError(ValueError):
    """A physical annotation violates the VQA data contract."""


@dataclass(frozen=True)
class VisibilityThresholds:
    """Final-image visibility thresholds used by the synthetic collectors."""

    min_pixels: int
    min_major_axis_pixels: int
    min_visible_fraction: float
    max_boundary_truncation_fraction: float


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    """Return the visible tight box in normalized ``[y0, x0, y1, x1]`` order.

    Pixel boxes are represented by their exterior edges, therefore a one-pixel
    object has non-zero width and a mask touching the final column normalizes
    to exactly 1.0.
    """

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return None
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    return [
        float(ys.min() / height),
        float(xs.min() / width),
        float((ys.max() + 1) / height),
        float((xs.max() + 1) / width),
    ]


def mask_major_axis_pixels(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return 0
    ys, xs = np.nonzero(mask)
    return int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def boundary_truncation_fraction(mask: np.ndarray) -> float:
    """Conservative boundary-contact score in ``[0, 1]``.

    A mask touching each of the four image edges receives 1.0.  This avoids
    treating an object at a single image edge as a fully trustworthy spatial
    target while remaining stable for thin objects.
    """

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return 1.0
    edges = (mask[0, :].any(), mask[-1, :].any(), mask[:, 0].any(), mask[:, -1].any())
    return float(sum(edges) / len(edges))


def classify_mask_visibility(
    mask: np.ndarray,
    thresholds: VisibilityThresholds,
    *,
    reference_pixel_count: int | None = None,
) -> tuple[str, float | None, float | None]:
    """Classify a visible instance mask without inventing hidden evidence."""

    mask = np.asarray(mask, dtype=bool)
    visible_pixels = int(mask.sum())
    if visible_pixels == 0:
        return "occluded", 0.0 if reference_pixel_count else None, 1.0 if reference_pixel_count else None
    visible_fraction = None
    occlusion_ratio = None
    if reference_pixel_count is not None and reference_pixel_count > 0:
        visible_fraction = float(min(1.0, visible_pixels / reference_pixel_count))
        occlusion_ratio = float(1.0 - visible_fraction)
    enough_pixels = visible_pixels >= thresholds.min_pixels
    enough_extent = mask_major_axis_pixels(mask) >= thresholds.min_major_axis_pixels
    enough_fraction = visible_fraction is None or visible_fraction >= thresholds.min_visible_fraction
    not_truncated = boundary_truncation_fraction(mask) <= thresholds.max_boundary_truncation_fraction
    if enough_pixels and enough_extent and enough_fraction and not_truncated:
        return "visible", visible_fraction, occlusion_ratio
    if visible_pixels > 0:
        return "partially_visible", visible_fraction, occlusion_ratio
    return "occluded", visible_fraction, occlusion_ratio


def base_record(**overrides: Any) -> dict[str, Any]:
    """Create a nullable physical VQA record with all contract columns."""

    record: dict[str, Any] = {
        "sample_id": None,
        "sample_type": "vqa",
        "source_dataset": SOURCE_DATASET,
        "task_name": None,
        "question_family": None,
        "episode_index": None,
        "frame_index": None,
        "timestamp": None,
        "ego_image_reference": None,
        "image_width": None,
        "image_height": None,
        "prompt_text": None,
        "answer_type": None,
        "answer_text": None,
        "answer_bool": None,
        "answer_int": None,
        "answer_point_yx_norm": None,
        "answer_bbox_yxyx_norm": None,
        "answer_int_list": None,
        "answer_bbox_list_yxyx_norm": None,
        "answer_aliases": None,
        "world_state_valid": False,
        "image_answerable": False,
        "visibility_status": "not_applicable",
        "visible_fraction": None,
        "occlusion_ratio": None,
        "target_view": None,
        "coordinate_space": None,
        "point_definition": None,
        "bbox_definition": None,
        "gt_source": None,
        "annotation_version": ANNOTATION_VERSION,
        "quality_score": None,
        "rejection_reason": None,
        "source_layout": None,
        "scene_id": None,
        "audit_metadata_json": None,
    }
    record.update(overrides)
    return record


def _exact_answer_field(record: dict[str, Any]) -> str:
    fields = [field for field in ANSWER_FIELDS if record.get(field) is not None]
    if len(fields) != 1:
        raise ValidationError(f"expected exactly one answer field, found {fields}")
    expected = {
        "point2d": "answer_point_yx_norm",
        "bbox2d": "answer_bbox_yxyx_norm",
        "boolean": "answer_bool",
        "short_text": "answer_text",
        "integer": "answer_int",
        "int_list": "answer_int_list",
        "bbox_list": "answer_bbox_list_yxyx_norm",
    }[record["answer_type"]]
    if fields[0] != expected:
        raise ValidationError(f"answer_type {record['answer_type']!r} requires {expected}, found {fields[0]}")
    return expected


def _validate_normalized(values: Any, expected_size: int, name: str) -> list[float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != expected_size or not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain {expected_size} finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValidationError(f"{name} must be within [0, 1]")
    return [float(value) for value in array]


def _validate_int_list(values: Any, name: str) -> list[int]:
    if not isinstance(values, (list, tuple)):
        raise ValidationError(f"{name} must be a list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValidationError(f"{name} must contain only integers")
    return [int(value) for value in values]


def _validate_bbox_list(values: Any, name: str) -> list[list[float]]:
    if not isinstance(values, (list, tuple)):
        raise ValidationError(f"{name} must be a list of boxes")
    boxes = []
    for item in values:
        box = _validate_normalized(item, 4, name)
        if not box[0] < box[2] or not box[1] < box[3]:
            raise ValidationError("each box must have positive width and height")
        boxes.append(box)
    return boxes


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one accepted physical VQA record."""

    record = base_record(**record)
    if not isinstance(record["sample_id"], str) or not record["sample_id"]:
        raise ValidationError("sample_id must be non-empty")
    if record["sample_type"] != "vqa":
        raise ValidationError("sample_type must be vqa")
    if record["answer_type"] not in ANSWER_TYPES:
        raise ValidationError(f"unsupported answer_type {record['answer_type']!r}")
    if not isinstance(record["prompt_text"], str) or not record["prompt_text"].strip():
        raise ValidationError("prompt_text must be non-empty")
    if not isinstance(record["ego_image_reference"], str) or not record["ego_image_reference"]:
        raise ValidationError("ego_image_reference must be non-empty")
    if record["target_view"] not in {None, "ego"}:
        raise ValidationError("VQA targets may only use ego view")
    if record["visibility_status"] not in VISIBILITY_STATUSES:
        raise ValidationError(f"unsupported visibility_status {record['visibility_status']!r}")
    if not record["world_state_valid"] or not record["image_answerable"]:
        raise ValidationError("accepted records require valid world state and image answerability")
    answer_field = _exact_answer_field(record)
    if record["answer_type"] == "boolean" and not isinstance(record[answer_field], bool):
        raise ValidationError("answer_bool must be bool")
    if record["answer_type"] == "short_text":
        answer = record[answer_field]
        if not isinstance(answer, str) or answer != answer.strip() or not answer:
            raise ValidationError("answer_text must be non-empty normalized text")
    if record["answer_type"] == "integer" and (
        not isinstance(record[answer_field], int) or isinstance(record[answer_field], bool)
    ):
        raise ValidationError("answer_int must be an integer")
    if record["answer_type"] == "point2d":
        record[answer_field] = _validate_normalized(record[answer_field], 2, answer_field)
        if record["target_view"] != "ego" or not record["point_definition"]:
            raise ValidationError("point records require ego target_view and point_definition")
        if record["visibility_status"] not in {"visible", "partially_visible"}:
            raise ValidationError("point target must be visibly supported")
    if record["answer_type"] == "bbox2d":
        box = _validate_normalized(record[answer_field], 4, answer_field)
        if not box[0] < box[2] or not box[1] < box[3]:
            raise ValidationError("bbox must have positive width and height")
        record[answer_field] = box
        if record["target_view"] != "ego" or record["bbox_definition"] != "visible_tight":
            raise ValidationError("bbox records require ego visible_tight metadata")
    if record["answer_type"] == "int_list":
        record[answer_field] = _validate_int_list(record[answer_field], answer_field)
    if record["answer_type"] == "bbox_list":
        record[answer_field] = _validate_bbox_list(record[answer_field], answer_field)
        if record["target_view"] != "ego" or record["bbox_definition"] != "visible_tight":
            raise ValidationError("bbox_list records require ego visible_tight metadata")
    return record


def _parquet_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("sample_id", pa.string()),
            pa.field("sample_type", pa.string()),
            pa.field("source_dataset", pa.string()),
            pa.field("task_name", pa.string()),
            pa.field("question_family", pa.string()),
            pa.field("episode_index", pa.int64()),
            pa.field("frame_index", pa.int64()),
            pa.field("timestamp", pa.float64()),
            pa.field("ego_image_reference", pa.string()),
            pa.field("image_width", pa.int32()),
            pa.field("image_height", pa.int32()),
            pa.field("prompt_text", pa.string()),
            pa.field("answer_type", pa.string()),
            pa.field("answer_text", pa.string()),
            pa.field("answer_bool", pa.bool_()),
            pa.field("answer_int", pa.int64()),
            # Variable Arrow lists avoid a PyArrow 25 null fixed-size-list
            # round-trip bug. ``validate_record`` still enforces [2]/[4].
            pa.field("answer_point_yx_norm", pa.list_(pa.float32())),
            pa.field("answer_bbox_yxyx_norm", pa.list_(pa.float32())),
            pa.field("answer_int_list", pa.list_(pa.int64())),
            pa.field("answer_bbox_list_yxyx_norm", pa.list_(pa.list_(pa.float32()))),
            pa.field("answer_aliases", pa.list_(pa.string())),
            pa.field("world_state_valid", pa.bool_()),
            pa.field("image_answerable", pa.bool_()),
            pa.field("visibility_status", pa.string()),
            pa.field("visible_fraction", pa.float32()),
            pa.field("occlusion_ratio", pa.float32()),
            pa.field("target_view", pa.string()),
            pa.field("coordinate_space", pa.string()),
            pa.field("point_definition", pa.string()),
            pa.field("bbox_definition", pa.string()),
            pa.field("gt_source", pa.string()),
            pa.field("annotation_version", pa.string()),
            pa.field("quality_score", pa.float32()),
            pa.field("rejection_reason", pa.string()),
            pa.field("source_layout", pa.string()),
            pa.field("scene_id", pa.string()),
            pa.field("audit_metadata_json", pa.string()),
        ]
    )


def build_report(accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    """Create compact, deterministic dataset diagnostics."""

    by_family = Counter(record.get("question_family") for record in accepted)
    by_type = Counter(record.get("answer_type") for record in accepted)
    by_visibility = Counter(record.get("visibility_status") for record in accepted)
    answers: dict[str, Counter] = defaultdict(Counter)
    for record in accepted:
        value = next((record[field] for field in ANSWER_FIELDS if record.get(field) is not None), None)
        answers[str(record.get("question_family"))][json_dumps(value)] += 1
    return {
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "samples_by_question_family": dict(sorted(by_family.items())),
        "samples_by_answer_type": dict(sorted(by_type.items())),
        "visibility_status_distribution": dict(sorted(by_visibility.items())),
        "answer_distribution": {key: dict(sorted(value.items())) for key, value in sorted(answers.items())},
        "rejection_reason_distribution": dict(
            sorted(Counter(record.get("rejection_reason", "unknown") for record in rejected).items())
        ),
    }


class SidecarWriter:
    """Collect, validate, and atomically materialize a VQA sidecar dataset."""

    def __init__(self, output_dir: Path, *, overwrite: bool = False):
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.accepted: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(f"output directory already exists: {self.output_dir}")
            shutil.rmtree(self.output_dir)

    def add(self, record: dict[str, Any]) -> bool:
        try:
            self.accepted.append(validate_record(record))
            return True
        except ValidationError as error:
            rejected = base_record(**record)
            rejected["rejection_reason"] = str(error)
            self.rejected.append(rejected)
            return False

    def reject(self, record: dict[str, Any], reason: str) -> None:
        rejected = base_record(**record)
        rejected["rejection_reason"] = reason
        self.rejected.append(rejected)

    def prepare_images_dir(self) -> Path:
        path = self.output_dir / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prepare_audit_dir(self) -> Path:
        """Create the non-training visual and metadata audit directory."""

        path = self.output_dir / "audit"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write(self, manifest: dict[str, Any]) -> dict[str, Any]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.output_dir.mkdir(parents=True, exist_ok=True)
        schema = _parquet_schema()
        accepted = [base_record(**record) for record in self.accepted]
        rejected = [base_record(**record) for record in self.rejected]
        pq.write_table(pa.Table.from_pylist(accepted, schema=schema), self.output_dir / "annotations.parquet")
        pq.write_table(pa.Table.from_pylist(rejected, schema=schema), self.output_dir / "rejected.parquet")
        report = build_report(accepted, rejected)
        manifest = dict(manifest)
        manifest.update(
            {
                "annotation_version": ANNOTATION_VERSION,
                "source_dataset": SOURCE_DATASET,
                "accepted_records": len(accepted),
                "rejected_records": len(rejected),
                "physical_annotation": "annotations.parquet",
                "rejected_annotation": "rejected.parquet",
            }
        )
        (self.output_dir / "manifest.json").write_text(json_dumps(manifest) + "\n", encoding="utf-8")
        (self.output_dir / "report.json").write_text(json_dumps(report) + "\n", encoding="utf-8")
        return report
