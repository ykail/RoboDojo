"""Typed physical VQA annotation storage and validation utilities.

This package is the vqa_gen copy of ``scripts.internal.vqa``.  It is free to
evolve for the make_kong pipeline: the ``int_list`` and ``bbox_list`` answer
types were added here and do not exist in the scripts/ copy.
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
