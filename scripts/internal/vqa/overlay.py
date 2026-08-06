"""Non-leaking numbered object overlays for VQA renderings."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class OverlayError(ValueError):
    """A safe placement could not be found for one or more object marks."""


@dataclass(frozen=True)
class OverlayResult:
    image: np.ndarray
    mark_boxes_xyxy: dict[str, tuple[int, int, int, int]]
    anchors_xy: dict[str, tuple[int, int]]
    leader_segments_xyxy: dict[str, tuple[int, int, int, int]]


def _intersects(first: tuple[int, int, int, int], second: tuple[int, int, int, int], padding: int = 2) -> bool:
    return not (
        first[2] + padding <= second[0]
        or second[2] + padding <= first[0]
        or first[3] + padding <= second[1]
        or second[3] + padding <= first[1]
    )


def _squared_distance_to_segment(point: tuple[float, float], start: tuple[int, int], end: tuple[int, int]) -> float:
    """Return the squared distance from ``point`` to a finite 2D segment."""

    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    point_array = np.asarray(point, dtype=np.float64)
    direction = end_array - start_array
    length_squared = float(np.dot(direction, direction))
    if length_squared == 0.0:
        return float(np.dot(point_array - start_array, point_array - start_array))
    fraction = float(np.clip(np.dot(point_array - start_array, direction) / length_squared, 0.0, 1.0))
    closest = start_array + fraction * direction
    return float(np.dot(point_array - closest, point_array - closest))


def _point_overlaps_box(point: tuple[float, float], box: tuple[int, int, int, int], clearance: int) -> bool:
    """Whether a protected point lies in a badge box expanded by ``clearance``."""

    return box[0] - clearance <= point[0] <= box[2] + clearance and box[1] - clearance <= point[1] <= box[3] + clearance


def _box_overlaps_mask(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Return whether an inclusive pixel box covers protected image content."""

    x0, y0, x1, y1 = box
    return bool(np.any(mask[y0 : y1 + 1, x0 : x1 + 1]))


def _segment_overlaps_mask(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
    """Conservatively test a two-pixel leader line against protected content."""

    distance = int(np.ceil(np.hypot(end[0] - start[0], end[1] - start[1])))
    steps = max(2, distance * 3)
    xs = np.rint(np.linspace(start[0], end[0], steps)).astype(np.intp)
    ys = np.rint(np.linspace(start[1], end[1], steps)).astype(np.intp)
    height, width = mask.shape
    for x, y in zip(xs, ys, strict=True):
        x0, x1 = max(0, x - 1), min(width, x + 2)
        y0, y1 = max(0, y - 1), min(height, y + 2)
        if np.any(mask[y0:y1, x0:x1]):
            return True
    return False


def _leader_start_outside_mask(
    anchor: tuple[int, int],
    badge_center: tuple[int, int],
    object_mask: np.ndarray | None,
) -> tuple[int, int]:
    """Move a leader-line endpoint from a tile centroid to beyond its mask."""

    if object_mask is None:
        return anchor
    height, width = object_mask.shape
    if not object_mask[anchor[1], anchor[0]]:
        raise OverlayError("object-mask anchor is outside its own instance mask")
    distance = int(np.ceil(np.hypot(badge_center[0] - anchor[0], badge_center[1] - anchor[1])))
    if distance == 0:
        return anchor
    xs = np.rint(np.linspace(anchor[0], badge_center[0], max(2, distance * 3))).astype(np.intp)
    ys = np.rint(np.linspace(anchor[1], badge_center[1], max(2, distance * 3))).astype(np.intp)
    inside = np.flatnonzero(object_mask[ys, xs])
    if inside.size == 0:
        raise OverlayError("could not find tile boundary for leader line")
    boundary_index = int(inside[-1])
    boundary = np.asarray((xs[boundary_index], ys[boundary_index]), dtype=np.float64)
    direction = np.asarray(badge_center, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    start = np.rint(boundary + 3.0 * direction).astype(np.intp)
    start[0] = np.clip(start[0], 0, width - 1)
    start[1] = np.clip(start[1], 0, height - 1)
    return int(start[0]), int(start[1])


def numbered_overlay(
    image: np.ndarray,
    anchors_xy: Mapping[str, Sequence[float]],
    *,
    protected_points_xy: Sequence[Sequence[float]] = (),
    object_masks_by_mark: Mapping[str, np.ndarray] | None = None,
    badge_size: int = 20,
    protected_point_clearance_px: int = 6,
) -> OverlayResult:
    """Draw uniform numbered badges and leader lines after image geometry.

    The caller supplies a body anchor for every mark.  Candidate badge
    locations are deterministic and reject overlap with another badge or a
    protected target point such as a pen nib. When ``object_masks_by_mark``
    is supplied, badges and leader lines also stay outside all supplied object
    masks, ending just beyond the marked object's visible boundary. This keeps
    a number association readable without covering tile-face texture.
    """

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise OverlayError("image must be HxWx3 uint8")
    if badge_size <= 0 or protected_point_clearance_px < 0:
        raise OverlayError("badge_size must be positive and protected-point clearance must be non-negative")
    canvas = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    height, width = rgb.shape[:2]
    object_masks_by_mark = object_masks_by_mark or {}
    unexpected_masks = set(object_masks_by_mark).difference(str(mark) for mark in anchors_xy)
    if unexpected_masks:
        raise OverlayError(f"object masks contain unknown marks: {sorted(unexpected_masks)}")
    protected_mask = np.zeros((height, width), dtype=bool)
    normalized_object_masks: dict[str, np.ndarray] = {}
    for mark, mask in object_masks_by_mark.items():
        normalized_mask = np.asarray(mask, dtype=bool)
        if normalized_mask.shape != (height, width):
            raise OverlayError(
                f"object mask for mark {mark!r} has shape {normalized_mask.shape}, expected {(height, width)}"
            )
        normalized_object_masks[str(mark)] = normalized_mask
        protected_mask |= normalized_mask
    half = badge_size // 2
    offsets = (
        (0, -34),
        (30, -30),
        (-30, -30),
        (38, 0),
        (-38, 0),
        (0, 34),
        (30, 30),
        (-30, 30),
        (0, -54),
        (48, -48),
        (-48, -48),
        (54, 0),
        (-54, 0),
        (0, 54),
        (48, 48),
        (-48, 48),
    )
    placed: dict[str, tuple[int, int, int, int]] = {}
    normalized_anchors: dict[str, tuple[int, int]] = {}
    leader_starts: dict[str, tuple[int, int]] = {}
    for mark, value in anchors_xy.items():
        anchor = (int(round(float(value[0]))), int(round(float(value[1]))))
        normalized_anchors[str(mark)] = anchor
        selected = None
        for dx, dy in offsets:
            center = (anchor[0] + dx, anchor[1] + dy)
            box = (center[0] - half, center[1] - half, center[0] + half, center[1] + half)
            if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                continue
            if any(_intersects(box, other) for other in placed.values()):
                continue
            if _box_overlaps_mask(protected_mask, box):
                continue
            if any(
                _point_overlaps_box((float(point[0]), float(point[1])), box, protected_point_clearance_px)
                for point in protected_points_xy
            ):
                continue
            center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            leader_start = _leader_start_outside_mask(anchor, center, normalized_object_masks.get(str(mark)))
            if _segment_overlaps_mask(protected_mask, leader_start, center):
                continue
            if any(
                _squared_distance_to_segment((float(point[0]), float(point[1])), leader_start, center)
                <= protected_point_clearance_px**2
                for point in protected_points_xy
            ):
                continue
            selected = (box, leader_start)
            break
        if selected is None:
            raise OverlayError(f"no non-overlapping in-frame badge position for mark {mark}")
        selected_box, leader_start = selected
        center = ((selected_box[0] + selected_box[2]) // 2, (selected_box[1] + selected_box[3]) // 2)
        draw.line((leader_start, center), fill=(255, 255, 255), width=2)
        draw.rounded_rectangle(selected_box, radius=4, fill=(20, 20, 20), outline=(255, 255, 255), width=1)
        text_box = draw.textbbox((0, 0), str(mark), font=font)
        text_x = center[0] - (text_box[2] - text_box[0]) // 2
        text_y = center[1] - (text_box[3] - text_box[1]) // 2
        draw.text((text_x, text_y), str(mark), fill=(255, 255, 255), font=font)
        placed[str(mark)] = selected_box
        leader_starts[str(mark)] = leader_start
    leader_segments = {
        mark: (*leader_starts[mark], (box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        for mark, box in placed.items()
    }
    return OverlayResult(np.asarray(canvas), placed, normalized_anchors, leader_segments)
