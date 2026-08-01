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


def _intersects(first: tuple[int, int, int, int], second: tuple[int, int, int, int], padding: int = 2) -> bool:
    return not (
        first[2] + padding <= second[0]
        or second[2] + padding <= first[0]
        or first[3] + padding <= second[1]
        or second[3] + padding <= first[1]
    )


def numbered_overlay(
    image: np.ndarray,
    anchors_xy: Mapping[str, Sequence[float]],
    *,
    protected_points_xy: Sequence[Sequence[float]] = (),
    badge_size: int = 20,
) -> OverlayResult:
    """Draw uniform numbered badges and leader lines after image geometry.

    The caller supplies a body anchor for every mark.  Candidate badge
    locations are deterministic and reject overlap with another badge or a
    protected target point such as a pen nib.
    """

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise OverlayError("image must be HxWx3 uint8")
    canvas = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    height, width = rgb.shape[:2]
    half = badge_size // 2
    offsets = ((0, -30), (25, -25), (-25, -25), (30, 0), (-30, 0), (0, 30), (25, 25), (-25, 25))
    placed: dict[str, tuple[int, int, int, int]] = {}
    normalized_anchors: dict[str, tuple[int, int]] = {}
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
            if any(box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3] for point in protected_points_xy):
                continue
            selected = box
            break
        if selected is None:
            raise OverlayError(f"no non-overlapping in-frame badge position for mark {mark}")
        center = ((selected[0] + selected[2]) // 2, (selected[1] + selected[3]) // 2)
        draw.line((anchor, center), fill=(255, 255, 255), width=2)
        draw.rounded_rectangle(selected, radius=4, fill=(20, 20, 20), outline=(255, 255, 255), width=1)
        text_box = draw.textbbox((0, 0), str(mark), font=font)
        text_x = center[0] - (text_box[2] - text_box[0]) // 2
        text_y = center[1] - (text_box[3] - text_box[1]) // 2
        draw.text((text_x, text_y), str(mark), fill=(255, 255, 255), font=font)
        placed[str(mark)] = selected
    return OverlayResult(np.asarray(canvas), placed, normalized_anchors)
