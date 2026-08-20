"""make_kong VQA data generation (A/B variant layouts, scene planning, collector)."""

from .scene_plan import VARIANT_A, VARIANT_B, FallenCase, plan_cases, random_discard_assignment  # noqa: F401
from .tile_faces import FACE_NAMES, face_of_category  # noqa: F401
