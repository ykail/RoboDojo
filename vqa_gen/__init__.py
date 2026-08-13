"""VQA data generation toolkit (make_kong-focused vqa_gen workspace).

``vqa_gen.vqa`` mirrors ``scripts.internal.vqa`` but is free to evolve for the
make_kong VQA pipeline; ``scripts.internal.vqa`` stays untouched for the
fill_pen_holder collector.
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
