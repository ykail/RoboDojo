"""Generate variant-A and variant-B tile-type-only layouts for make_kong VQA.

Layout A keeps the four matching groups as contiguous robot-side blocks but
assigns the four discard faces by derangement, so the opponent's pushed-down
tile no longer matches its nominal slot group.  Layout B reuses A's discard
pairing but scatters the four group faces across the twelve robot-side slots,
so a matching group is not a contiguous block.  Both variants preserve the
scene geometry of the evaluation template; only Mahjong tile types change.

Faces are canonicalized through ``vqa_gen.make_kong.tile_faces``: the four
matching groups plus the support pair use five pairwise-distinct faces (all
five available faces), which keeps every discard-to-group matching
unambiguous.  This script does not launch Isaac Sim.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vqa_gen.make_kong.scene_plan import deranged_discard_assignment, scatter_kong_group_assignment
from vqa_gen.make_kong.tile_faces import (
    FACE_NAMES,
    MAX_CATEGORY,
    categories_for_face,
    face_of_category,
    verify_face_map,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_LAYOUT_ROOT = REPO_ROOT / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
DEFAULT_MAHJONG_ASSET_ROOT = REPO_ROOT / "Assets" / "Object" / "RoboDojo" / "Rigid" / "mahjong"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("layouts")
SOURCE_LAYOUT_PATTERN = re.compile(r"make_kong_(\d+)\.json")
VARIANT_LAYOUT_PATTERN = re.compile(r"make_kong_([ab])_(\d+)\.json")
MANIFEST_NAME = "manifest.json"
VARIANTS = ("a", "b")

KONG_GROUPS = (
    ("mahjong0_0", "mahjong0_1", "mahjong0_2"),
    ("mahjong1_0", "mahjong1_1", "mahjong1_2"),
    ("mahjong2_0", "mahjong2_1", "mahjong2_2"),
    ("mahjong3_0", "mahjong3_1", "mahjong3_2"),
)
KONG_LABELS = tuple(label for group in KONG_GROUPS for label in group)
SUPPORT_GROUP = ("mahjong4_0", "mahjong4_1", "mahjong9_0")
DISCARD_LABELS = ("mahjong5_0", "mahjong6_0", "mahjong7_0", "mahjong8_0")
OTHER_LABELS = ("other0", "other1", "other2")
SIGNATURE_LABELS = (*KONG_LABELS, *SUPPORT_GROUP, *OTHER_LABELS)

PLACEMENT_FIELDS = (
    "label",
    "group",
    "xlim",
    "ylim",
    "zlim",
    "qpos",
    "rotate_deg",
    "rotate_rand",
    "relative_plane",
    "place_tag",
    "margin",
    "check_mode",
    "need_check_stable",
    "default_pos",
    "default_ori",
)


class LayoutGenerationError(ValueError):
    """Raised when source or output layouts do not meet the generator contract."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Number of A/B layout pairs to append (default: 100).")
    parser.add_argument("--rng-seed", type=int, default=2810, help="Seed for deterministic face sampling.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory receiving A/B layouts.")
    parser.add_argument(
        "--source-layout-root",
        type=Path,
        default=DEFAULT_SOURCE_LAYOUT_ROOT,
        help="Existing make_kong layout directory used as the template and tile-type library.",
    )
    parser.add_argument(
        "--verify-faces",
        action="store_true",
        help="Cross-check the hard-coded face band map against the mahjong USD textures before sampling.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LayoutGenerationError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise LayoutGenerationError(f"Layout must contain a JSON object: {path}")
    return value


def _mahjong_records(layout: dict[str, Any], source_name: str) -> list[dict[str, Any]]:
    records = layout.get("Rigid", {}).get("mahjong")
    if not isinstance(records, list):
        raise LayoutGenerationError(f"{source_name} is missing Rigid.mahjong records")
    if not all(isinstance(record, dict) for record in records):
        raise LayoutGenerationError(f"{source_name} contains a non-object Mahjong record")
    return records


def _records_by_label(layout: dict[str, Any], source_name: str) -> dict[str, dict[str, Any]]:
    records = _mahjong_records(layout, source_name)
    by_label = {record.get("label"): record for record in records}
    if len(by_label) != len(records) or None in by_label:
        raise LayoutGenerationError(f"{source_name} must contain one uniquely labelled record per Mahjong tile")
    expected = set(KONG_LABELS) | set(SUPPORT_GROUP) | set(DISCARD_LABELS) | set(OTHER_LABELS)
    missing = sorted(expected - set(by_label))
    if missing:
        raise LayoutGenerationError(f"{source_name} is missing expected Mahjong labels: {missing}")
    return by_label


def _source_layout_paths(layout_root: Path) -> list[Path]:
    if not layout_root.is_dir():
        raise LayoutGenerationError(f"Source layout directory does not exist: {layout_root}")
    paths = []
    for path in layout_root.iterdir():
        match = SOURCE_LAYOUT_PATTERN.fullmatch(path.name)
        if match and path.is_file():
            paths.append((int(match.group(1)), path))
    if not paths:
        raise LayoutGenerationError(f"No make_kong_<id>.json files found in {layout_root}")
    return [path for _, path in sorted(paths)]


def _build_donor_library(source_layouts: list[dict[str, Any]], source_paths: list[Path]) -> dict[int, dict[str, Any]]:
    donors: dict[int, dict[str, Any]] = {}
    for layout, path in zip(source_layouts, source_paths):
        for record in _mahjong_records(layout, str(path)):
            category_idx = record.get("category_idx")
            if not isinstance(category_idx, int):
                raise LayoutGenerationError(f"{path} has Mahjong record without an integer category_idx")
            donors.setdefault(category_idx, deepcopy(record))
    missing = [category for category in range(MAX_CATEGORY + 1) if category not in donors]
    if missing:
        raise LayoutGenerationError(f"Source library is missing mahjong categories: {missing[:5]} ...")
    return donors


def _replace_tile_record(template_record: dict[str, Any], donor_record: dict[str, Any]) -> dict[str, Any]:
    replacement = deepcopy(donor_record)
    for field in PLACEMENT_FIELDS:
        if field in template_record:
            replacement[field] = deepcopy(template_record[field])
        else:
            replacement.pop(field, None)
    return replacement


def _sample_layout_spec(rng: random.Random) -> dict[str, Any]:
    """Sample one signature plus its derangement and scatter from ``rng``."""

    faces = list(FACE_NAMES)
    rng.shuffle(faces)
    group_faces = faces[:4]
    support_face = faces[4]
    group_categories = [rng.choice(categories_for_face(face)) for face in group_faces]
    support_category = rng.choice(categories_for_face(support_face))
    chosen = set(group_categories) | {support_category}
    remaining = [category for category in range(MAX_CATEGORY + 1) if category not in chosen]
    other_categories = rng.sample(remaining, len(OTHER_LABELS))
    derangement = deranged_discard_assignment(rng)
    scatter = scatter_kong_group_assignment(rng)
    categories: dict[str, int] = {}
    for group_index, labels in enumerate(KONG_GROUPS):
        for label in labels:
            categories[label] = group_categories[group_index]
    for label in SUPPORT_GROUP:
        categories[label] = support_category
    for index, label in enumerate(DISCARD_LABELS):
        categories[label] = group_categories[derangement[index]]
    for label, category in zip(OTHER_LABELS, other_categories):
        categories[label] = category
    return {
        "group_categories": group_categories,
        "group_face_names": group_faces,
        "support_category": support_category,
        "support_face_name": support_face,
        "other_categories": other_categories,
        "derangement": derangement,
        "scatter": scatter,
        "categories": categories,
    }


def _build_variant(
    template: dict[str, Any],
    donors: dict[int, dict[str, Any]],
    spec: dict[str, Any],
    *,
    variant: str,
) -> dict[str, Any]:
    categories = spec["categories"]
    scatter = spec["scatter"]
    template_by_label = _records_by_label(template, "template layout")
    replacements: dict[str, dict[str, Any]] = {}
    for slot, label in enumerate(KONG_LABELS):
        if variant == "a":
            group_index = next(index for index, labels in enumerate(KONG_GROUPS) if label in labels)
        else:
            group_index = scatter[slot]
        category_idx = spec["group_categories"][group_index]
        replacements[label] = _replace_tile_record(template_by_label[label], donors[category_idx])
    for label in (*SUPPORT_GROUP, *DISCARD_LABELS, *OTHER_LABELS):
        replacements[label] = _replace_tile_record(template_by_label[label], donors[categories[label]])
    layout = deepcopy(template)
    records = _mahjong_records(layout, "template layout")
    layout["Rigid"]["mahjong"] = [replacements.get(record["label"], record) for record in records]
    return layout


def _slot_categories(spec: dict[str, Any] | dict[str, dict[str, Any]]) -> list[int]:
    """Return one category per semantic slot (4 groups, support, 4 discards, 3 others).

    Discard slots duplicate the group categories by design, so the distinct
    category count of the twelve slots must be eight.
    """

    if "group_categories" in spec:
        derangement = spec["derangement"]
        return [
            *spec["group_categories"],
            spec["support_category"],
            *[spec["group_categories"][derangement[index]] for index in range(4)],
            *spec["other_categories"],
        ]
    by_label = spec
    return [
        by_label[labels[0]]["category_idx"] for labels in KONG_GROUPS
    ] + [
        by_label[SUPPORT_GROUP[0]]["category_idx"],
        *[by_label[label]["category_idx"] for label in DISCARD_LABELS],
        *[by_label[label]["category_idx"] for label in OTHER_LABELS],
    ]


def _validate_spec(spec: dict[str, Any], source_name: str) -> None:
    categories = spec["categories"]
    if any(not isinstance(value, int) for value in categories.values()):
        raise LayoutGenerationError(f"{source_name} must contain integer categories")
    if len(set(_slot_categories(spec))) != 8:
        raise LayoutGenerationError(f"{source_name} reuses a category across semantic slots")
    if len(set(spec["group_face_names"])) != 4:
        raise LayoutGenerationError(f"{source_name} matching groups must use four distinct faces")
    if spec["support_face_name"] in spec["group_face_names"]:
        raise LayoutGenerationError(f"{source_name} support face must differ from every matching group face")
    if any(index == target for index, target in enumerate(spec["derangement"])):
        raise LayoutGenerationError(f"{source_name} discard assignment is not a derangement")
    if sorted(spec["scatter"]) != [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]:
        raise LayoutGenerationError(f"{source_name} kong scatter must contain three copies of every group")


def _validate_variant(layout: dict[str, Any], source_name: str) -> None:
    by_label = _records_by_label(layout, source_name)
    for label in (*KONG_LABELS, *SUPPORT_GROUP, *DISCARD_LABELS, *OTHER_LABELS):
        face_of_category(by_label[label]["category_idx"])
    row_faces = {face_of_category(by_label[label]["category_idx"]) for label in KONG_LABELS}
    if len(row_faces) != 4:
        raise LayoutGenerationError(f"{source_name} the twelve kong slots must show exactly four distinct faces")
    support_face = face_of_category(by_label[SUPPORT_GROUP[0]]["category_idx"])
    if support_face in row_faces:
        raise LayoutGenerationError(f"{source_name} support face must differ from every matching group face")
    if len(set(_slot_categories(by_label))) != 8:
        raise LayoutGenerationError(f"{source_name} reuses a category across semantic slots")


def _write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _existing_layouts(output_dir: Path) -> list[tuple[str, int, Path]]:
    if not output_dir.exists():
        return []
    if not output_dir.is_dir():
        raise LayoutGenerationError(f"Output path is not a directory: {output_dir}")
    matches = []
    for path in output_dir.iterdir():
        match = VARIANT_LAYOUT_PATTERN.fullmatch(path.name)
        if match and path.is_file():
            matches.append((match.group(1), int(match.group(2)), path))
    for variant in VARIANTS:
        ids = sorted(layout_id for current_variant, layout_id, _ in matches if current_variant == variant)
        if ids and ids != list(range(ids[-1] + 1)):
            raise LayoutGenerationError(f"Existing {variant} layout IDs must be continuous from 0 in {output_dir}")
    return matches


def _layout_signature(layout: dict[str, Any], source_name: str) -> tuple[int, ...]:
    by_label = _records_by_label(layout, source_name)
    return tuple(by_label[label]["category_idx"] for label in SIGNATURE_LABELS)


def generate_vqa_layouts(
    *,
    count: int = 100,
    rng_seed: int = 0,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    source_layout_root: Path = DEFAULT_SOURCE_LAYOUT_ROOT,
    verify_faces: bool = False,
    mahjong_asset_root: Path = DEFAULT_MAHJONG_ASSET_ROOT,
) -> dict[str, Any]:
    """Append ``count`` A/B layout pairs and return the updated manifest."""

    if count < 1:
        raise LayoutGenerationError("count must be positive")
    source_layout_root = Path(source_layout_root)
    output_dir = Path(output_dir)
    source_paths = _source_layout_paths(source_layout_root)
    source_layouts = [_load_json(path) for path in source_paths]
    if verify_faces:
        verify_face_map(mahjong_asset_root)
    donors = _build_donor_library(source_layouts, source_paths)
    template = source_layouts[0]

    existing = _existing_layouts(output_dir)
    existing_signatures: set[tuple[tuple[int, ...], str]] = set()
    for variant, layout_id, path in existing:
        existing_signatures.add((_layout_signature(_load_json(path), str(path)), variant))
    existing_pairs = len(existing) // 2

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(rng_seed)
    generated_entries = []
    next_id = max((layout_id for _, layout_id, _ in existing), default=-1) + 1
    attempts = 0
    max_attempts = max(count * 100, 1000)
    while len(generated_entries) < count:
        attempts += 1
        if attempts > max_attempts:
            raise LayoutGenerationError("Unable to sample enough unique A/B layout signatures")
        spec = _sample_layout_spec(rng)
        _validate_spec(spec, "sampled signature")
        signature = tuple(spec["categories"][label] for label in SIGNATURE_LABELS)
        if any((signature, variant) in existing_signatures for variant in VARIANTS):
            continue
        files = {}
        for variant in VARIANTS:
            layout = _build_variant(template, donors, spec, variant=variant)
            _validate_variant(layout, f"{variant} layout {next_id}")
            path = output_dir / f"make_kong_{variant}_{next_id}.json"
            _write_json(path, layout)
            files[variant] = path.name
            existing_signatures.add((signature, variant))
        generated_entries.append(
            {
                "id": next_id,
                "files": files,
                "group_categories": spec["group_categories"],
                "group_face_names": spec["group_face_names"],
                "support_category": spec["support_category"],
                "support_face_name": spec["support_face_name"],
                "other_categories": spec["other_categories"],
                "discard_derangement": {
                    label: spec["categories"][label] for label in DISCARD_LABELS
                },
                "kong_scatter": list(spec["scatter"]),
            }
        )
        next_id += 1

    manifest_path = output_dir / MANIFEST_NAME
    previous_manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    previous_runs = previous_manifest.get("generation_runs", [])
    previous_layouts = previous_manifest.get("layouts", [])
    manifest = {
        "schema_version": 2,
        "task_name": "make_kong_vqa",
        "source_layout_root": str(source_layout_root),
        "face_vocabulary": list(FACE_NAMES),
        "total_layout_pairs": existing_pairs + len(generated_entries),
        "generation_runs": [
            *previous_runs,
            {
                "rng_seed": rng_seed,
                "requested_count": count,
                "generated_count": len(generated_entries),
            },
        ],
        "layouts": [*previous_layouts, *generated_entries],
    }
    _write_json(manifest_path, manifest)
    return manifest


def load_vqa_layout_pool(layout_root: Path, variants=VARIANTS) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    """Load every ``make_kong_<variant>_<id>.json`` layout in ``layout_root``."""

    layout_root = Path(layout_root)
    if not layout_root.is_dir():
        raise LayoutGenerationError(f"Generated layout directory does not exist: {layout_root}")
    pool: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for path in layout_root.iterdir():
        match = VARIANT_LAYOUT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        variant, layout_id = match.group(1), int(match.group(2))
        if variant not in variants:
            continue
        layout = _load_json(path)
        _validate_variant(layout, str(path))
        pool[(variant, layout_id)] = (path, layout)
    for variant in variants:
        ids = sorted(layout_id for current_variant, layout_id in pool if current_variant == variant)
        if ids and ids != list(range(ids[-1] + 1)):
            raise LayoutGenerationError(
                f"Variant {variant} layout IDs must be continuous from 0 in {layout_root}; found {ids}"
            )
    if not pool:
        raise LayoutGenerationError(f"No make_kong_<variant>_<id>.json layouts found in {layout_root}")
    return pool


def main() -> int:
    args = _parse_args()
    manifest = generate_vqa_layouts(
        count=args.count,
        rng_seed=args.rng_seed,
        output_dir=args.output_dir,
        source_layout_root=args.source_layout_root,
        verify_faces=args.verify_faces,
    )
    print(
        f"Generated {manifest['generation_runs'][-1]['generated_count']} A/B layout pairs; "
        f"total={manifest['total_layout_pairs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
