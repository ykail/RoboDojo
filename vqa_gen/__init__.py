"""VQA data generation toolkit (vqa_gen workspace).

``vqa_gen.vqa`` is the canonical typed-VQA sidecar module.  It evolved from the
legacy ``scripts/internal/vqa`` package to serve the make_kong and
fill_pen_holder collectors after their migration out of ``scripts/``.
"""

from .vqa import (  # noqa: F401
    ANNOTATION_VERSION,
    SOURCE_DATASET,
    RobotStatePoolError,
    SidecarWriter,
    ValidationError,
    bbox_from_mask,
    classify_mask_visibility,
    load_robot_state_pool,
    split_paired_state,
    validate_record,
)
