import unittest

import numpy as np

from src.eval_client.policy_runtime import (
    ARX_X5_SIM_OBSERVATION_SPEC,
    ArxX5ObservationBuilder,
    ObservationValidationSpec,
    PayloadErrorKind,
    RawObservationBuildError,
)


def _image(height, width, rgb):
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    color = rgba[..., :3]
    if width > 1:
        assert not color.flags.c_contiguous
    return color


def _spec():
    return ObservationValidationSpec(
        head_image_shape=(2, 3, 3),
        left_wrist_image_shape=(3, 2, 3),
        right_wrist_image_shape=(1, 4, 3),
    )


def _raw_observation():
    return {
        "instruction": np.str_("make toast"),
        "vision": {
            "cam_head": {
                "color": _image(2, 3, (1, 2, 3)),
                "shape": (2, 3, 3),
                "depth": np.zeros((2, 3), dtype=np.float32),
            },
            "cam_left_wrist": {
                "color": _image(3, 2, (11, 12, 13)),
                "shape": (3, 2, 3),
            },
            "cam_right_wrist": {
                "color": _image(1, 4, (21, 22, 23)),
                "shape": (1, 4, 3),
            },
        },
        "state": {
            "left_arm_joint_state": np.arange(6, dtype=np.float64),
            "left_ee_joint_state": [0.25],
            "right_arm_joint_state": np.arange(6, dtype=np.float32) + 10,
            "right_ee_joint_state": np.asarray([0.75], dtype=np.float32),
            "left_ee_pose": np.zeros(7, dtype=np.float32),
            "right_ee_pose": np.zeros(7, dtype=np.float32),
        },
        "action": {"must": "be dropped"},
        "data_format_version": "v1.0",
        "additional_info": {"frequency": 25},
        "env_idx": 0,
    }


class _ArrayLikeGripper:
    def __array__(self):
        return np.asarray([0.5], dtype=np.float32)


class ArxX5ObservationBuilderTest(unittest.TestCase):
    def test_maps_real_raw_shape_and_drops_non_policy_fields(self):
        raw = _raw_observation()
        result = ArxX5ObservationBuilder(
            spec=_spec(),
            expected_env_idx=0,
        ).build(raw)

        np.testing.assert_array_equal(result.head[0, 0], [1, 2, 3])
        np.testing.assert_array_equal(result.left_wrist[0, 0], [11, 12, 13])
        np.testing.assert_array_equal(result.right_wrist[0, 0], [21, 22, 23])
        np.testing.assert_array_equal(
            result.left_arm_joint_position,
            np.arange(6, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result.right_arm_joint_position,
            np.arange(6, dtype=np.float32) + 10,
        )
        self.assertEqual(
            result.left_gripper_open_fraction_commanded.dtype,
            np.dtype(np.float32),
        )
        self.assertIs(type(result.instruction), str)
        self.assertFalse(result.head.flags.writeable)

        payload = result.to_payload()
        self.assertEqual(set(payload), {"instruction", "images", "proprio"})
        self.assertNotIn("action", payload)
        self.assertNotIn("left_ee_pose", payload["proprio"])
        self.assertNotIn("env_idx", payload)
        self.assertNotIn("depth", payload["images"]["head"])

    def test_result_is_independent_from_mutable_raw_arrays(self):
        raw = _raw_observation()
        source_image = raw["vision"]["cam_head"]["color"]
        source_state = raw["state"]["left_arm_joint_state"]
        result = ArxX5ObservationBuilder(
            spec=_spec(),
            expected_env_idx=0,
        ).build(raw)

        source_image.fill(99)
        source_state.fill(99)
        np.testing.assert_array_equal(result.head[0, 0], [1, 2, 3])
        np.testing.assert_array_equal(
            result.left_arm_joint_position,
            np.arange(6, dtype=np.float32),
        )

    def test_missing_raw_fields_have_source_paths(self):
        cases = (
            (("vision",), lambda raw: raw.pop("vision")),
            (
                ("vision", "cam_left_wrist"),
                lambda raw: raw["vision"].pop("cam_left_wrist"),
            ),
            (
                ("vision", "cam_head", "color"),
                lambda raw: raw["vision"]["cam_head"].pop("color"),
            ),
            (
                ("state", "right_ee_joint_state"),
                lambda raw: raw["state"].pop("right_ee_joint_state"),
            ),
            (("instruction",), lambda raw: raw.pop("instruction")),
            (
                ("data_format_version",),
                lambda raw: raw.pop("data_format_version"),
            ),
            (("env_idx",), lambda raw: raw.pop("env_idx")),
        )
        for path, mutate in cases:
            with self.subTest(path=path):
                raw = _raw_observation()
                mutate(raw)
                with self.assertRaises(RawObservationBuildError) as raised:
                    ArxX5ObservationBuilder(
                        spec=_spec(),
                        expected_env_idx=0,
                    ).build(raw)
                self.assertEqual(raised.exception.path, path)

    def test_state_vectors_are_converted_explicitly_but_strictly(self):
        cases = (
            (
                "left_arm_joint_state",
                np.zeros((1, 6), dtype=np.float32),
                "expected shape",
            ),
            (
                "left_arm_joint_state",
                np.asarray(["bad"] * 6),
                "floating-point arm",
            ),
            (
                "left_arm_joint_state",
                [0.0] * 6,
                "NumPy ndarray",
            ),
            (
                "right_arm_joint_state",
                np.arange(6, dtype=np.int32),
                "floating-point arm",
            ),
            (
                "right_arm_joint_state",
                np.asarray([0, 0, 0, np.nan, 0, 0]),
                "finite",
            ),
            (
                "left_ee_joint_state",
                np.asarray([1e300], dtype=np.float64),
                "finite float32",
            ),
            (
                "right_ee_joint_state",
                [1],
                "floating-point gripper",
            ),
            (
                "right_ee_joint_state",
                np.float32(0.5),
                "list, tuple",
            ),
            (
                "right_ee_joint_state",
                _ArrayLikeGripper(),
                "list, tuple",
            ),
            (
                "right_ee_joint_state",
                np.asarray([[0.5]], dtype=np.float32),
                "expected shape",
            ),
            (
                "right_ee_joint_state",
                [np.nan],
                "finite",
            ),
            (
                "right_ee_joint_state",
                [1.01],
                "closed interval",
            ),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                raw = _raw_observation()
                raw["state"][field] = value
                with self.assertRaisesRegex(RawObservationBuildError, message):
                    ArxX5ObservationBuilder(
                        spec=_spec(),
                        expected_env_idx=0,
                    ).build(raw)

    def test_images_are_not_cast_resized_or_reordered(self):
        cases = (
            (
                "cam_head",
                np.zeros((2, 3, 3), dtype=np.float32),
                PayloadErrorKind.INVALID_DTYPE,
            ),
            (
                "cam_head",
                np.zeros((2, 3, 4), dtype=np.uint8),
                PayloadErrorKind.INVALID_SHAPE,
            ),
            (
                "cam_head",
                np.zeros((3, 2, 3), dtype=np.uint8),
                PayloadErrorKind.INVALID_SHAPE,
            ),
            (
                "cam_left_wrist",
                np.zeros((3, 3, 2), dtype=np.uint8),
                PayloadErrorKind.INVALID_SHAPE,
            ),
        )
        for camera, image, kind in cases:
            with self.subTest(camera=camera, shape=image.shape, kind=kind):
                raw = _raw_observation()
                raw["vision"][camera]["color"] = image
                with self.assertRaises(RawObservationBuildError) as raised:
                    ArxX5ObservationBuilder(
                        spec=_spec(),
                        expected_env_idx=0,
                    ).build(raw)
                self.assertEqual(raised.exception.canonical_error_kind, kind)
                self.assertEqual(
                    raised.exception.path,
                    ("vision", camera, "color"),
                )

    def test_instruction_and_raw_containers_are_strict(self):
        for value, message in (
            (b"make toast", "string"),
            ("   ", "must not be empty"),
            (None, "string"),
        ):
            with self.subTest(value=value):
                raw = _raw_observation()
                raw["instruction"] = value
                with self.assertRaisesRegex(RawObservationBuildError, message):
                    ArxX5ObservationBuilder(
                        spec=_spec(),
                        expected_env_idx=0,
                    ).build(raw)

        for path, mutate in (
            ((), lambda raw: []),
            (("vision",), lambda raw: {**raw, "vision": []}),
            (
                ("vision", "cam_head"),
                lambda raw: {
                    **raw,
                    "vision": {**raw["vision"], "cam_head": []},
                },
            ),
            (("state",), lambda raw: {**raw, "state": []}),
        ):
            with self.subTest(path=path):
                raw = mutate(_raw_observation())
                with self.assertRaises(RawObservationBuildError) as raised:
                    ArxX5ObservationBuilder(
                        spec=_spec(),
                        expected_env_idx=0,
                    ).build(raw)
                self.assertEqual(raised.exception.path, path)

    def test_camera_alias_does_not_replace_the_raw_contract(self):
        raw = _raw_observation()
        raw["vision"]["cam_high"] = raw["vision"].pop("cam_head")
        with self.assertRaises(RawObservationBuildError) as raised:
            ArxX5ObservationBuilder(
                spec=_spec(),
                expected_env_idx=0,
            ).build(raw)
        self.assertEqual(
            raised.exception.path,
            ("vision", "cam_head"),
        )

    def test_builder_requires_a_concrete_profile(self):
        with self.assertRaises(TypeError):
            ArxX5ObservationBuilder(spec=None, expected_env_idx=0)
        with self.assertRaises(TypeError):
            ArxX5ObservationBuilder(spec=_spec(), expected_env_idx=True)
        with self.assertRaises(ValueError):
            ArxX5ObservationBuilder(spec=_spec(), expected_env_idx=-1)

    def test_raw_version_and_environment_are_connection_invariants(self):
        cases = (
            ("data_format_version", "v2.0", "version"),
            ("data_format_version", 1, "version"),
            ("env_idx", 1, "environment 0"),
            ("env_idx", True, "integer"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                raw = _raw_observation()
                raw[field] = value
                with self.assertRaisesRegex(RawObservationBuildError, message):
                    ArxX5ObservationBuilder(
                        spec=_spec(),
                        expected_env_idx=0,
                    ).build(raw)

    def test_released_profile_accepts_full_size_raw_rgb_views(self):
        raw = _raw_observation()
        raw["vision"]["cam_head"]["color"] = _image(
            480,
            640,
            (31, 32, 33),
        )
        raw["vision"]["cam_left_wrist"]["color"] = _image(
            480,
            640,
            (41, 42, 43),
        )
        raw["vision"]["cam_right_wrist"]["color"] = _image(
            480,
            640,
            (51, 52, 53),
        )
        result = ArxX5ObservationBuilder(
            spec=ARX_X5_SIM_OBSERVATION_SPEC,
            expected_env_idx=0,
        ).build(raw)
        np.testing.assert_array_equal(result.head[0, 0], [31, 32, 33])
        np.testing.assert_array_equal(result.left_wrist[0, 0], [41, 42, 43])
        np.testing.assert_array_equal(result.right_wrist[0, 0], [51, 52, 53])
        self.assertTrue(result.head.flags.c_contiguous)
        self.assertFalse(result.head.flags.writeable)


if __name__ == "__main__":
    unittest.main()
