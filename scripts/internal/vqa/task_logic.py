"""Pure task-specific scheduling and reference helpers for VQA collectors."""

import random
from typing import Mapping


def ordinal(index: int) -> str:
    """Return the English ordinal for a positive 1-based index."""

    if index <= 0:
        raise ValueError("ordinal index must be positive")
    remainder = index % 100
    if 10 < remainder < 14:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(index % 10, "th")
    return f"{index}{suffix}"


def matching_tile_indices(robot_side_labels: list[str], target_labels: list[str]) -> tuple[int, int, int]:
    """Return the ordered 1-based robot-side positions of exactly three targets."""

    if len(target_labels) != 3 or len(set(target_labels)) != 3:
        raise ValueError("a matching group must contain exactly three distinct tiles")
    positions = [robot_side_labels.index(label) + 1 for label in target_labels]
    return tuple(sorted(positions))


def fallen_labels_for_pattern(target_labels: list[str], pattern: int) -> list[str]:
    """Return matching tiles encoded as fallen by a three-bit pattern."""

    if len(target_labels) != 3 or pattern not in range(8):
        raise ValueError("expected three target labels and a fallen pattern in [0, 7]")
    return labels_for_bitmask(target_labels, pattern)


def labels_for_bitmask(labels: list[str], pattern: int) -> list[str]:
    """Return labels selected by a least-significant-bit-first bitmask."""

    if not labels or pattern not in range(1 << len(labels)):
        raise ValueError("bitmask does not fit the provided labels")
    return [label for index, label in enumerate(labels) if pattern & (1 << index)]


def adjacent_nonmatching_labels(robot_side_labels: list[str], target_labels: list[str]) -> list[str]:
    """Return immediate left/right nonmatching neighbours of a matching triplet."""

    if len(target_labels) != 3 or len(set(target_labels)) != 3:
        raise ValueError("a matching group must contain exactly three distinct tiles")
    positions = sorted(robot_side_labels.index(label) for label in target_labels)
    target_positions = set(positions)
    neighbours: list[str] = []
    for position in (positions[0] - 1, positions[-1] + 1):
        if 0 <= position < len(robot_side_labels) and position not in target_positions:
            neighbours.append(robot_side_labels[position])
    return neighbours


def kong_declaration_neighbor_state_reason(is_fallen: bool) -> str:
    """Return the fixed reason for a neighbouring tile during kong declaration."""

    return "nonmatching_fallen" if is_fallen else "correct"


def holder_pose_condition(index: int, seed: int, layout_index: int) -> tuple[bool, float]:
    """Return paired upright/tipped states with seed-shuffled tipped headings.

    Each consecutive pair contains one upright and one tipped holder. Across
    16 layout scenes, the eight tipped headings occur exactly once each.
    """

    if index < 0:
        raise ValueError("holder pose index must be non-negative")
    pair_index, position_in_pair = divmod(index, 2)
    heading_cycle, heading_offset = divmod(pair_index, 8)
    headings = list(range(8))
    rng = random.Random(seed * 1000003 + layout_index * 65537 + heading_cycle)
    rng.shuffle(headings)
    tipped_position = rng.randrange(2)
    return position_in_pair == tipped_position, float(headings[heading_offset] * 45)


def pen_descriptions_from_features(
    centers: Mapping[str, tuple[float, float]],
    colors: Mapping[str, str | None],
    robot_centers: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[str, str]]:
    """Choose unique natural-language pen references from visible features."""

    if not centers:
        return {}
    candidates: dict[str, list[tuple[str, str]]] = {label: [] for label in centers}
    for label, color in colors.items():
        if label in centers and color and sum(other == color for other in colors.values()) == 1:
            candidates[label].append((f"the {color} pen", "asset_color"))
    ordered = sorted(centers, key=lambda label: centers[label][0])
    candidates[ordered[0]].append(("the leftmost pen", "image_leftmost"))
    candidates[ordered[-1]].append(("the rightmost pen", "image_rightmost"))
    for side, robot_center in robot_centers.items():
        distances = {
            label: (center[0] - robot_center[0]) ** 2 + (center[1] - robot_center[1]) ** 2
            for label, center in centers.items()
        }
        nearest_distance = min(distances.values())
        nearest = [label for label, distance in distances.items() if abs(distance - nearest_distance) < 1e-9]
        if len(nearest) == 1:
            candidates[nearest[0]].append((f"the pen closest to the {side} robot arm", f"nearest_{side}_arm"))
    descriptions: dict[str, tuple[str, str]] = {}
    for label, choices in candidates.items():
        for text, kind in choices:
            owners = [owner for owner, owner_choices in candidates.items() if (text, kind) in owner_choices]
            if len(owners) == 1:
                descriptions[label] = (text, kind)
                break
    return descriptions
