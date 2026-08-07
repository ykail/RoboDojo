"""Shared helpers for generating RoboDojo VQA sidecar datasets."""

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
