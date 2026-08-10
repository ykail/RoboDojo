from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CameraCleanupContractTest(unittest.TestCase):
    def test_tiled_camera_explicitly_deregisters_xform_callbacks(self) -> None:
        path = ROOT / "env/camera_manager/capture/camera_view.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        camera_view = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CameraView"
        )
        destroy = next(
            node
            for node in camera_view.body
            if isinstance(node, ast.FunctionDef) and node.name == "destroy"
        )
        calls = [node for node in ast.walk(destroy) if isinstance(node, ast.Call)]
        self.assertTrue(
            any(
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "XFormPrim"
                and call.func.attr == "destroy"
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "self"
                for call in calls
            ),
            "CameraView.destroy must explicitly deregister XFormPrim callbacks",
        )


if __name__ == "__main__":
    unittest.main()
