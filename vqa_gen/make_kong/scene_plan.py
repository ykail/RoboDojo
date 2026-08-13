"""Pure fall-scene planning for make_kong VQA collectors.

This module is Isaac-free.  It plans which row tiles fall for every question
scene of the make_kong VQA coverage matrix (variant A: contiguous layouts,
variant B: scattered layouts), assigns the deranged discard pairing, and
scatters the kong faces for variant-B layouts.

All random choice is driven by an explicit ``random.Random`` so collectors and
layout generators stay deterministic per seed.
"""

from dataclasses import dataclass
import random
from typing import Sequence

CONSECUTIVE_WRONG_SAMPLES = 3
RANDOM_WRONG_SAMPLES = 3
VARIANT_A = "a"
VARIANT_B = "b"


@dataclass(frozen=True)
class FallenCase:
    """One VQA scene: which matching tiles fell and which wrong tiles fell."""

    case_id: str
    correct_fallen: tuple[str, ...]
    wrong_fallen: tuple[str, ...]

    @property
    def fallen_labels(self) -> tuple[str, ...]:
        return self.correct_fallen + self.wrong_fallen


def deranged_discard_assignment(rng: random.Random) -> tuple[int, int, int, int]:
    """Return a derangement of the four discard slots.

    ``result[i]`` is the matching-group index for discard slot ``i``; no
    discard keeps its nominal group (``result[i] != i``), which removes the
    fixed slot-to-group shortcut from the VQA scenes.
    """

    for _ in range(1000):
        permutation = list(range(4))
        rng.shuffle(permutation)
        if all(permutation[index] != index for index in range(4)):
            return tuple(permutation)
    return (1, 2, 3, 0)


def scatter_kong_group_assignment(rng: random.Random) -> tuple[int, ...]:
    """Return a uniformly random assignment of 12 kong slots to four groups.

    Each group index appears exactly three times; the three matching tiles of
    a group may end up anywhere in the row (no contiguity guarantee).
    """

    multiset = [index for index in range(4) for _ in range(3)]
    rng.shuffle(multiset)
    return tuple(multiset)


def _windows_of_length(positions: set[int], length: int, max_position: int) -> list[tuple[int, ...]]:
    return [
        tuple(range(start, start + length))
        for start in range(0, max_position - length + 1)
        if not positions.intersection(range(start, start + length))
    ]


def _sample_distinct(rng: random.Random, candidates: list[tuple[int, ...]], count: int) -> list[tuple[int, ...]]:
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def _sample_subsets(
    rng: random.Random, pool: Sequence[str], size: int, count: int
) -> list[tuple[str, ...]]:
    if size > len(pool):
        return []
    seen: set[frozenset[str]] = set()
    sampled: list[tuple[str, ...]] = []
    attempts = 0
    while len(sampled) < count and attempts < 1000:
        attempts += 1
        subset = tuple(sorted(rng.sample(pool, size)))
        key = frozenset(subset)
        if key in seen:
            continue
        seen.add(key)
        sampled.append(subset)
    return sampled


def _wrong_label_subsets(
    rng: random.Random,
    *,
    variant: str,
    row_labels: Sequence[str],
    target_positions: set[int],
    size: int,
) -> list[tuple[str, ...]]:
    """Sample wrong-tile sets; variant A requires contiguous row blocks."""

    if variant == VARIANT_B:
        non_target = [label for index, label in enumerate(row_labels) if index not in target_positions]
        return _sample_subsets(rng, non_target, size, RANDOM_WRONG_SAMPLES)
    windows = _windows_of_length(target_positions, size, len(row_labels))
    return [
        tuple(row_labels[position] for position in window)
        for window in _sample_distinct(rng, windows, CONSECUTIVE_WRONG_SAMPLES)
    ]


def _neighbor_labels(row_labels: Sequence[str], target_positions: set[int]) -> tuple[str | None, str | None]:
    first = min(target_positions)
    last = max(target_positions)
    left = row_labels[first - 1] if first - 1 >= 0 else None
    right = row_labels[last + 1] if last + 1 < len(row_labels) else None
    return left, right


def plan_cases(
    row_labels: Sequence[str],
    target_labels: Sequence[str],
    variant: str,
    rng: random.Random,
) -> tuple[list[FallenCase], dict[str, str]]:
    """Plan the full VQA coverage matrix for one target group.

    Returns ``(cases, skipped)`` where ``skipped`` maps infeasible case ids to
    the reason (variant-A boundary conditions).  ``target_labels`` must be in
    left-to-right row order and contain exactly three labels.
    """

    if variant not in (VARIANT_A, VARIANT_B):
        raise ValueError(f"variant must be 'a' or 'b', got {variant!r}")
    if len(set(target_labels)) != 3 or len(target_labels) != 3:
        raise ValueError("a target group must contain exactly three distinct tiles")
    row_labels = list(row_labels)
    target_positions = {row_labels.index(label) for label in target_labels}
    non_target = [label for index, label in enumerate(row_labels) if index not in target_positions]
    cases: list[FallenCase] = []
    skipped: dict[str, str] = {}

    cases.append(FallenCase("f0", (), ()))

    for index in range(3):
        cases.append(FallenCase(f"c1_p{index}", (target_labels[index],), ()))
    for index, wrong in enumerate(_sample_subsets(rng, non_target, 1, RANDOM_WRONG_SAMPLES)):
        cases.append(FallenCase(f"w1_s{index}", (), wrong))

    for combo_index, combo in enumerate(((0, 1), (0, 2), (1, 2))):
        cases.append(FallenCase(f"c2_p{''.join(map(str, combo))}", tuple(target_labels[i] for i in combo), ()))
    for index, wrong in enumerate(
        _wrong_label_subsets(rng, variant=variant, row_labels=row_labels, target_positions=target_positions, size=2)
    ):
        cases.append(FallenCase(f"w2_s{index}", (), wrong))

    cases.append(FallenCase("c3", tuple(target_labels), ()))
    for index, wrong in enumerate(
        _wrong_label_subsets(rng, variant=variant, row_labels=row_labels, target_positions=target_positions, size=3)
    ):
        cases.append(FallenCase(f"w3_s{index}", (), wrong))

    left_neighbor, right_neighbor = _neighbor_labels(row_labels, target_positions)
    if variant == VARIANT_A:
        if left_neighbor is not None:
            cases.append(FallenCase("c3w1_L", tuple(target_labels), (left_neighbor,)))
        else:
            skipped["c3w1_L"] = "no_left_neighbor"
        if right_neighbor is not None:
            cases.append(FallenCase("c3w1_R", tuple(target_labels), (right_neighbor,)))
        else:
            skipped["c3w1_R"] = "no_right_neighbor"
    else:
        for index, wrong in enumerate(_sample_subsets(rng, non_target, 1, RANDOM_WRONG_SAMPLES)):
            cases.append(FallenCase(f"c3w1_s{index}", tuple(target_labels), wrong))

    for index, wrong in enumerate(
        _wrong_label_subsets(rng, variant=variant, row_labels=row_labels, target_positions=target_positions, size=4)
    ):
        cases.append(FallenCase(f"w4_s{index}", (), wrong))

    if variant == VARIANT_A:
        left_two = (
            None
            if left_neighbor is None or min(target_positions) - 2 < 0
            else (row_labels[min(target_positions) - 2], left_neighbor)
        )
        right_two = (
            None
            if right_neighbor is None or max(target_positions) + 2 >= len(row_labels)
            else (right_neighbor, row_labels[max(target_positions) + 2])
        )
        left_right = None if left_neighbor is None or right_neighbor is None else (left_neighbor, right_neighbor)
        for case_id, wrong in (
            ("c3w2_LL", left_two),
            ("c3w2_RR", right_two),
            ("c3w2_LR", left_right),
        ):
            if wrong is None:
                skipped[case_id] = "insufficient_adjacent_slots"
                continue
            cases.append(FallenCase(case_id, tuple(target_labels), wrong))
    else:
        for index, wrong in enumerate(_sample_subsets(rng, non_target, 2, RANDOM_WRONG_SAMPLES)):
            cases.append(FallenCase(f"c3w2_s{index}", tuple(target_labels), wrong))

    for index, wrong in enumerate(
        _wrong_label_subsets(rng, variant=variant, row_labels=row_labels, target_positions=target_positions, size=5)
    ):
        cases.append(FallenCase(f"w5_s{index}", (), wrong))

    return cases, skipped
