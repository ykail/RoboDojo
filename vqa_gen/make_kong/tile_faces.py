"""Canonical mahjong tile-face vocabulary for make_kong VQA.

The 42 category indices of the mahjong asset library map to only five distinct
tile faces (USD textures MJ01..MJ05).  The VQA pipeline refers to these faces
by the canonical English suit names below; ``scripts/internal`` and
``data_gen/make_kong`` remain unaware of this vocabulary.
"""

FACE_NAMES = ("Wan", "Tong", "Suo", "Honor", "Bonus")

# (first_category, last_category_inclusive, face_name) in category order.
# Verified against the USD texture names in Assets/Object/RoboDojo/Rigid/mahjong/.
_FACE_BANDS = (
    (0, 8, "Wan"),
    (9, 17, "Suo"),
    (18, 26, "Tong"),
    (27, 33, "Honor"),
    (34, 41, "Bonus"),
)

MAX_CATEGORY = 41
_CATEGORY_TO_FACE: dict[int, str] = {
    category: face_name for first, last, face_name in _FACE_BANDS for category in range(first, last + 1)
}


class TileFaceError(ValueError):
    """A category index or face name is not part of the mahjong library."""


def face_of_category(category_idx: int) -> str:
    """Return the canonical face name for a mahjong category index."""

    face_name = _CATEGORY_TO_FACE.get(int(category_idx))
    if face_name is None:
        raise TileFaceError(f"category index {category_idx} is outside the mahjong library")
    return face_name


def categories_for_face(face_name: str) -> tuple[int, ...]:
    """Return every category index that renders with the given face."""

    for first, last, band_name in _FACE_BANDS:
        if band_name == face_name:
            return tuple(range(first, last + 1))
    raise TileFaceError(f"unknown face name {face_name!r}; expected one of {FACE_NAMES}")


def verify_face_map(mahjong_asset_dir, *, logger=None) -> dict[int, str]:
    """Cross-check the hard-coded band map against USD texture names.

    Scans every ``<category>/object.usdz`` under ``mahjong_asset_dir`` and
    returns the observed ``{category: face_name}`` map, raising
    ``TileFaceError`` on any mismatch with the hard-coded bands.
    """

    from pathlib import Path
    import zipfile

    observed: dict[int, str] = {}
    root = Path(mahjong_asset_dir)
    if not root.is_dir():
        raise TileFaceError(f"mahjong asset directory does not exist: {root}")
    texture_to_face = {"MJ01": "Wan", "MJ02": "Tong", "MJ03": "Suo", "MJ04": "Honor", "MJ05": "Bonus"}
    for category in sorted(_CATEGORY_TO_FACE):
        archive = root / f"{category:05d}" / "object.usdz"
        if not archive.is_file():
            raise TileFaceError(f"missing mahjong asset archive: {archive}")
        face_name = None
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                base = Path(name).name.upper()
                for texture_key, mapped_face in texture_to_face.items():
                    if base.startswith(texture_key + "."):
                        face_name = mapped_face
                        break
                if face_name is not None:
                    break
        if face_name is None:
            raise TileFaceError(f"no MJ texture found in {archive}")
        expected = _CATEGORY_TO_FACE[category]
        if face_name != expected:
            raise TileFaceError(
                f"category {category:05d} shows texture {face_name} but the band map expects {expected}"
            )
        observed[category] = face_name
    if logger is not None:
        logger.info("verified %s mahjong category textures against the face band map", len(observed))
    return observed
