from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src" / "eval_client"
SCRIPT_DIR = ROOT / "scripts" / "RoboDojo"

X5_CONTROL_MODE = "x5_policy_joint_intervention"
X5_PROFILE = "arx_x5_identity_joint_v1"
PIPERX_PROFILE = "arx_x5_piperx_relative_joint_v1"


def _tree(relative_path: str) -> ast.Module:
    path = ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _strings(node: ast.AST) -> set[str]:
    return {
        item.value
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        if isinstance(item.func, ast.Name):
            names.add(item.func.id)
        elif isinstance(item.func, ast.Attribute):
            names.add(item.func.attr)
    return names


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} was not found")


class _Vector(list[float]):
    """Tiny element-wise vector used to execute mapping code without NumPy."""

    def copy(self) -> "_Vector":
        return _Vector(self)

    def _binary(self, other, operation) -> "_Vector":
        if isinstance(other, (int, float)):
            return _Vector(operation(value, other) for value in self)
        if len(self) != len(other):
            raise ValueError("vector lengths differ")
        return _Vector(operation(left, right) for left, right in zip(self, other))

    def __mul__(self, other) -> "_Vector":
        return self._binary(other, lambda left, right: left * right)

    def __rmul__(self, other) -> "_Vector":
        return self.__mul__(other)

    def __add__(self, other) -> "_Vector":
        return self._binary(other, lambda left, right: left + right)


class _NumpyStub:
    float64 = float
    ndarray = _Vector

    @staticmethod
    def asarray(values, dtype=None) -> _Vector:
        del dtype
        return _Vector(float(value) for value in values)

    @staticmethod
    def ones(size: int, dtype=None) -> _Vector:
        del dtype
        return _Vector([1.0] * size)


def _load_mapping_contract() -> dict[str, object]:
    """Execute only the profile and manual-mapping definitions under test."""

    tree = _tree("src/eval_client/piperx_dual_joint_mirror.py")
    selected: list[ast.stmt] = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        )
        or (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "_JOINT_PROFILES"
                for target in (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
            )
        )
        or (
            isinstance(node, ast.ClassDef)
            and node.name == "DualJointMirrorError"
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name
            in {"_joint_signs_for_profile", "_joint_signs", "_manual_action"}
        )
    ]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))

    def hold_action(_obs):
        return {
            "left_arm_joint_state": _Vector([0.0] * 6),
            "left_ee_joint_state": _Vector([0.0]),
            "right_arm_joint_state": _Vector([0.0] * 6),
            "right_ee_joint_state": _Vector([0.0]),
        }

    namespace: dict[str, object] = {
        "np": _NumpyStub,
        "os": os,
        "SIDES": ("left", "right"),
        "_hold_action": hold_action,
    }
    exec(compile(module, "<x5-mapping-contract>", "exec"), namespace)
    return namespace


def _load_task_metadata_contract() -> tuple[object, list[Path]]:
    """Execute `_task_metadata` alone, replacing git subprocesses with fakes."""

    tree = _tree("src/eval_client/lerobot_stream_recorder.py")
    function = _function(tree, "_task_metadata")
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                function,
            ],
            type_ignores=[],
        )
    )
    revision_paths: list[Path] = []

    def git_revision(path: Path) -> str:
        revision_paths.append(Path(path))
        return "0123456789abcdef0123456789abcdef01234567"

    namespace: dict[str, object] = {
        "Any": object,
        "Path": Path,
        "os": os,
        "__file__": str(SRC_DIR / "lerobot_stream_recorder.py"),
        "_git_revision": git_revision,
        "_git_dirty": lambda _path: False,
        "_env_bool": lambda _name, default=False: default,
    }
    exec(compile(module, "<x5-provenance-contract>", "exec"), namespace)
    return namespace["_task_metadata"], revision_paths


def _load_episode_metadata_contract():
    tree = _tree("scripts/RoboDojo/lerobot_stream_writer.py")
    function = _function(tree, "_episode_metadata")
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                function,
            ],
            type_ignores=[],
        )
    )
    namespace: dict[str, object] = {"Any": object}
    exec(compile(module, "<x5-writer-provenance-contract>", "exec"), namespace)
    return namespace["_episode_metadata"]


class X5DaggerIntegrationTest(unittest.TestCase):
    def test_x5_control_mode_is_accepted_and_dispatched_to_intervention_loop(self):
        main_tree = _tree("src/eval_client/main.py")
        mode_sets = []
        for node in ast.walk(main_tree):
            if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
                continue
            if not isinstance(node.left, ast.Name) or node.left.id != "control_mode":
                continue
            comparator = node.comparators[0]
            if isinstance(comparator, (ast.Set, ast.Tuple, ast.List)):
                mode_sets.append(_strings(comparator))

        self.assertTrue(
            any(
                {"policy", "piperx_policy_joint_intervention", X5_CONTROL_MODE}
                <= values
                for values in mode_sets
            ),
            "main.py must accept the X5 control mode in its top-level whitelist",
        )
        self.assertTrue(
            any(
                {"piperx_policy_joint_intervention", X5_CONTROL_MODE} <= values
                for values in mode_sets
            ),
            "main.py must apply the policy-v1/live-window single-env checks to X5",
        )
        self.assertTrue(
            any(
                {"keyboard_intervention", X5_CONTROL_MODE} <= values
                for values in mode_sets
            ),
            "X5 collection must be operator-driven so task step limits cannot end it",
        )

        eval_tree = _tree("src/eval_client/eval_env.py")
        route = None
        for node in ast.walk(eval_tree):
            if not isinstance(node, ast.If) or X5_CONTROL_MODE not in _strings(node.test):
                continue
            if "run_piperx_policy_leader_mirror_episode" in _called_names(node):
                route = node
                break
        self.assertIsNotNone(
            route,
            "EvalEnv.eval_one_episode must route X5 through the proven dual-joint intervention loop",
        )

        calls = [
            node
            for node in ast.walk(route)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "run_piperx_policy_leader_mirror_episode"
            )
        ]
        self.assertEqual(len(calls), 1)
        allow_intervention = next(
            (keyword.value for keyword in calls[0].keywords if keyword.arg == "allow_intervention"),
            None,
        )
        self.assertIsNotNone(allow_intervention)
        self.assertIn(
            X5_CONTROL_MODE,
            _strings(allow_intervention),
            "X5 route must enable i-key intervention, not policy-follow-only mode",
        )

        launcher_source = _source("scripts/RoboDojo/eval_kai0_pi05.sh")
        self.assertIn(X5_CONTROL_MODE, launcher_source)
        self.assertIn("ROBODOJO_DUAL_MIRROR_PROFILE", launcher_source)

    def test_x5_profile_selects_identity_mapping_for_both_manual_arms(self):
        namespace = _load_mapping_contract()
        resolve = namespace["_joint_signs_for_profile"]
        resolve_from_env = namespace["_joint_signs"]
        manual_action = namespace["_manual_action"]

        self.assertEqual(list(resolve(X5_PROFILE)), [1.0] * 6)
        self.assertEqual(
            list(resolve(PIPERX_PROFILE)),
            [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            "the new X5 profile must not change the already-validated PiPER-X profile",
        )
        with self.assertRaises(namespace["DualJointMirrorError"]):
            resolve("unknown-profile")

        sim_anchor = {
            "left": (_Vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0.2),
            "right": (_Vector([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]), 0.8),
        }
        response = {
            "sides": {
                "left": {
                    "leader_delta_q_rad": _Vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                    "leader_gripper_open_fraction": 0.25,
                },
                "right": {
                    "leader_delta_q_rad": _Vector([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]),
                    "leader_gripper_open_fraction": 0.75,
                },
            }
        }
        with patch.dict(
            os.environ,
            {"ROBODOJO_DUAL_MIRROR_PROFILE": X5_PROFILE},
            clear=False,
        ):
            self.assertEqual(list(resolve_from_env()), [1.0] * 6)
            action = manual_action({}, sim_anchor, response)

        self.assertEqual(
            list(action["left_arm_joint_state"]),
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        )
        self.assertEqual(
            list(action["right_arm_joint_state"]),
            [-1.1, -2.2, -3.3, -4.4, -5.5, -6.6],
        )
        self.assertEqual(action["left_ee_joint_state"][0], 0.25)
        self.assertEqual(action["right_ee_joint_state"][0], 0.75)

    def test_lerobot_metadata_records_x5_hardware_provenance(self):
        task_metadata, revision_paths = _load_task_metadata_contract()
        x5_root = Path("/tmp/robot_lab_x5_test_root")
        task_env = SimpleNamespace(
            task_name="make_toast",
            config_name="arx_x5",
            env_seeds=[17],
            layout_cycle=3,
            eval_seed=9,
            policy_name="Pi_05",
            additional_info="RoboDojo-sim-arx_x5-joint-0/59999",
            policy_runtime="robodojo_policy_v1",
            policy_provenance={
                "checkpoint_id": "RoboDojo-sim-arx_x5-joint-0/59999",
            },
            control_mode=X5_CONTROL_MODE,
        )
        with patch.dict(
            os.environ,
            {
                "ROBODOJO_X5_CODE_ROOT": str(x5_root),
                "ROBODOJO_RUN_ID": "x5-test-run",
            },
            clear=False,
        ):
            metadata = task_metadata(task_env)

        expected = {
            "hardware_embodiment": "arx_x5",
            "hardware_profile": X5_PROFILE,
            "hardware_bridge_protocol": "robodojo_dual_joint_mirror_v1",
            "hardware_bridge_commit": "0123456789abcdef0123456789abcdef01234567",
            "hardware_bridge_dirty": False,
            "hardware_control_topology": "policy_sim_to_two_x5_manual_two_x5_to_sim",
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(metadata.get(key), value)
        self.assertIn(
            x5_root,
            revision_paths,
            "X5 commit provenance must be read from ROBODOJO_X5_CODE_ROOT",
        )

        persisted = _load_episode_metadata_contract()(
            metadata,
            success=True,
            reason="task_success",
            has_intervention=True,
            frame_count=25,
        )
        for key, value in expected.items():
            with self.subTest(persisted_key=key):
                self.assertEqual(persisted.get(f"robodojo_{key}"), value)

    def test_x5_launchers_are_foreground_and_pin_the_integration_contract(self):
        hardware = SCRIPT_DIR / "run_x5_dagger_hardware.sh"
        isaac = SCRIPT_DIR / "run_x5_dagger_isaac.sh"
        pen_holder = SCRIPT_DIR / "run_acone_x5_pen_holder_isaac.sh"
        missing = [str(script) for script in (hardware, isaac, pen_holder) if not script.is_file()]
        self.assertFalse(missing, f"missing X5 launcher(s): {missing}")
        for script in (hardware, isaac, pen_holder):
            with self.subTest(script=script.name):
                self.assertTrue(
                    os.access(script, os.X_OK),
                    f"launcher must be executable with ./scripts/RoboDojo/{script.name}",
                )
                syntax = subprocess.run(
                    ["bash", "-n", str(script)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(syntax.returncode, 0, syntax.stderr)
                self.assertNotIn("tmux", script.read_text(encoding="utf-8").lower())

        hardware_source = hardware.read_text(encoding="utf-8")
        for required in (
            "x5_dual_joint_mirror_source.py",
            'SOURCE_HOST="${ROBODOJO_X5_SOURCE_HOST:-127.0.0.1}"',
            'SOURCE_PORT="${ROBODOJO_X5_SOURCE_PORT:-8770}"',
            'LEFT_CAN="${X5_LEFT_CAN:-can1}"',
            'RIGHT_CAN="${X5_RIGHT_CAN:-can3}"',
            'LEFT_MODEL="${X5_LEFT_MODEL:-X5}"',
            'RIGHT_MODEL="${X5_RIGHT_MODEL:-X5}"',
            'FREQUENCY_HZ="${X5_FREQUENCY_HZ:-100}"',
            "--left-can",
            "--right-can",
            "--frequency-hz",
        ):
            with self.subTest(hardware_required=required):
                self.assertIn(required, hardware_source)
        self.assertNotIn("eval_kai0_pi05.sh", hardware_source)

        isaac_source = isaac.read_text(encoding="utf-8")
        for required in (
            f'MIRROR_PROFILE="{X5_PROFILE}"',
            'export ROBODOJO_DUAL_MIRROR_PROFILE="${MIRROR_PROFILE}"',
            "ROBODOJO_DUAL_MIRROR_RECORD=1",
            "ROBODOJO_X5_CODE_ROOT",
            "ROBODOJO_RENDERING_MODE=quality",
            'ROBODOJO_X5_CUDA_PIPELINE="${ROBODOJO_X5_CUDA_PIPELINE:-0}"',
            "--control-mode x5_policy_joint_intervention",
            'TASK="${ROBODOJO_TASK:-make_toast}"',
            'CHECKPOINT_ID="${ROBODOJO_CHECKPOINT_ID:-RoboDojo-sim-arx_x5-joint-0/59999}"',
            "--checkpoint-id",
            '--external-policy-server-url "ws://127.0.0.1:${POLICY_PORT}"',
            'EXPECTED_KAI0_COMMIT="${ROBODOJO_EXPECTED_KAI0_COMMIT-ecc1a7451c3156b1e5f7533851dbb0222896206f}"',
            "--expected-kai0-commit",
            'EXPECTED_CHECKPOINT_DIGEST="${ROBODOJO_EXPECTED_CHECKPOINT_DIGEST-sha256:70bb68139ba717553d9a9d9c3055bb322b85046d729377ee46eaaf997c1eaac4}"',
            "--expected-checkpoint-digest",
            "ROBODOJO_DUAL_MIRROR_TIMEOUT_S",
            "eval_kai0_pi05.sh",
        ):
            with self.subTest(required=required):
                self.assertIn(required, isaac_source)
        self.assertNotIn("run_x5_dagger_hardware.sh", isaac_source)

        pen_source = pen_holder.read_text(encoding="utf-8")
        for required in (
            'ROBODOJO_TASK="fill_pen_holder"',
            'ROBODOJO_CHECKPOINT_ID="fill_pen_holder/9999_my"',
            "robodojo_fill_pen_holder_x5_online_dagger_9999_my_v1",
            "run_acone_x5_isaac.sh",
        ):
            with self.subTest(pen_holder_required=required):
                self.assertIn(required, pen_source)

        mirror_source = _source("src/eval_client/piperx_dual_joint_mirror.py")
        self.assertIn("set_updates_enabled", mirror_source)
        self.assertIn("SimulatorStateSnapshotter", mirror_source)
        self.assertIn("restore_replay_frame", mirror_source)


if __name__ == "__main__":
    unittest.main()
