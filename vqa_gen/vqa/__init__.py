"""Typed physical VQA annotation storage and validation utilities.

This package is the canonical VQA sidecar module used by the make_kong and
fill_pen_holder collectors.  It grew out of the legacy ``scripts/internal/vqa``
package: the ``int_list`` and ``bbox_list`` answer types and the unified yxyx
bbox storage convention were added here.
"""

from .robot_state_pool import RobotStatePoolError, load_robot_state_pool, split_paired_state
from .sidecar import (
    ANNOTATION_VERSION,
    SOURCE_DATASET,
    SidecarWriter,
    ValidationError,
    bbox_from_mask,
    classify_mask_visibility,
    validate_record,
)

__all__ = [
    "ANNOTATION_VERSION",
    "SOURCE_DATASET",
    "SidecarWriter",
    "ValidationError",
    "bbox_from_mask",
    "classify_mask_visibility",
    "validate_record",
    "RobotStatePoolError",
    "load_robot_state_pool",
    "split_paired_state",
]
