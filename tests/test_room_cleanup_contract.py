from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_delete_fixture_prims():
    path = ROOT / "env/scene_manager/scene_manager.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    scene_manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SceneManager"
    )
    method = next(
        node
        for node in scene_manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "delete_fixture_prims"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace: dict[str, object] = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["delete_fixture_prims"]


class RoomCleanupContractTest(unittest.TestCase):
    def test_room_cleanup_targets_plural_stage_root(self) -> None:
        checked: list[str] = []
        deleted: list[str] = []
        method = _load_delete_fixture_prims()
        method.__globals__.update(
            is_prim_path_valid=lambda path: checked.append(path) or True,
            delete_prim=deleted.append,
        )
        manager = type("Manager", (), {"env_roots": ["/World/envs/env_0"]})()

        method(manager, 0, "Room")

        self.assertEqual(checked, ["/World/envs/env_0/Rooms"])
        self.assertEqual(deleted, ["/World/envs/env_0/Rooms"])

    def test_other_fixture_roots_are_unchanged(self) -> None:
        checked: list[str] = []
        method = _load_delete_fixture_prims()
        method.__globals__.update(
            is_prim_path_valid=lambda path: checked.append(path) or False,
            delete_prim=lambda _path: None,
        )
        manager = type("Manager", (), {"env_roots": ["/World/envs/env_0"]})()

        for fixture_type in ("Table", "Light", "Ground"):
            method(manager, 0, fixture_type)
