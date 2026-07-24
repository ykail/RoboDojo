import json
from pathlib import Path
import unittest

import numpy as np

from src.eval_client.policy_runtime import (
    ACTION_CONTROL_MODE,
    ACTION_SCHEMA_ID,
    ARX_X5_SIM_OBSERVATION_SPEC,
    ARX_X5_SIM_RGB_SHAPE,
    MAX_ACTION_HORIZON,
    MAX_CANONICAL_IMAGE_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_INSTRUCTION_BYTES,
    OBSERVATION_SCHEMA_ID,
    ROBOT_SCHEMA_ID,
    ActionValidationSpec,
    CanonicalActionChunk,
    CanonicalObservation,
    ErrorCode,
    Frame,
    JointLimits,
    MessageType,
    ObservationValidationSpec,
    PayloadErrorKind,
    PayloadValidationError,
    PolicySession,
    ProtocolError,
    SessionPhase,
    decode_frame,
    encode_frame,
    parse_action_chunk,
    parse_infer_payload,
    parse_infer_result_payload,
    parse_observation,
)


def _rgb(height, width, offset=0):
    values = np.arange(height * width * 3, dtype=np.uint8)
    return (values + offset).reshape(height, width, 3)


class _SpoofedShapeArray(np.ndarray):
    spoofed_shape = (1,)

    @property
    def shape(self):
        return self.spoofed_shape


def _observation_payload():
    return {
        "instruction": "make toast",
        "images": {
            "head": _rgb(2, 3),
            "left_wrist": _rgb(3, 2, 17),
            "right_wrist": _rgb(1, 4, 31),
        },
        "proprio": {
            "robot_schema": ROBOT_SCHEMA_ID,
            "left_arm_joint_position": np.linspace(-0.5, 0.5, 6, dtype=np.float32),
            "left_gripper_open_fraction_commanded": np.array([0.0], dtype=np.float32),
            "right_arm_joint_position": np.linspace(0.5, -0.5, 6, dtype=np.float32),
            "right_gripper_open_fraction_commanded": np.array([1.0], dtype=np.float32),
        },
    }


def _observation_spec():
    return ObservationValidationSpec(
        head_image_shape=(2, 3, 3),
        left_wrist_image_shape=(3, 2, 3),
        right_wrist_image_shape=(1, 4, 3),
    )


def _parse_observation(payload):
    return parse_observation(payload, spec=_observation_spec())


def _action_spec(horizon=2):
    return ActionValidationSpec(
        expected_horizon=horizon,
        expected_control_dt_s=0.04,
        left_arm_limits=JointLimits(lower=(-1,) * 6, upper=(1,) * 6),
        right_arm_limits=JointLimits(lower=(-2,) * 6, upper=(2,) * 6),
    )


def _action_payload(horizon=2):
    return {
        "control_mode": ACTION_CONTROL_MODE,
        "control_dt_s": 0.04,
        "commands": {
            "left_arm_joint_position": np.full((horizon, 6), 0.5, dtype=np.float32),
            "left_gripper_open_fraction": np.linspace(0, 1, horizon, dtype=np.float32).reshape(-1, 1),
            "right_arm_joint_position": np.full((horizon, 6), 1.5, dtype=np.float32),
            "right_gripper_open_fraction": np.linspace(1, 0, horizon, dtype=np.float32).reshape(-1, 1),
        },
    }


def _hello(request_id="hello-1"):
    return Frame(
        message_type=MessageType.HELLO,
        request_id=request_id,
        session_id="session-1",
        payload={},
    )


def _reset(request_id="reset-1", episode_id="episode-1"):
    return Frame(
        message_type=MessageType.RESET,
        request_id=request_id,
        session_id="session-1",
        episode_id=episode_id,
        payload={},
    )


def _infer(request_id, payload, inference_index=0):
    return Frame(
        message_type=MessageType.INFER,
        request_id=request_id,
        session_id="session-1",
        episode_id="episode-1",
        inference_index=inference_index,
        payload=payload,
    )


class CanonicalSchemaDocumentTest(unittest.TestCase):
    def test_schema_documents_match_runtime_constants_and_annotations(self):
        schema_dir = Path(__file__).resolve().parents[1] / "protocol" / "robodojo_policy_v1"
        observation = json.loads((schema_dir / "observation.schema.json").read_text(encoding="utf-8"))
        action = json.loads((schema_dir / "action.schema.json").read_text(encoding="utf-8"))

        self.assertEqual(observation["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(action["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(observation["$id"], OBSERVATION_SCHEMA_ID)
        self.assertEqual(action["$id"], ACTION_SCHEMA_ID)
        self.assertEqual(
            observation["properties"]["proprio"]["properties"]["robot_schema"]["const"],
            ROBOT_SCHEMA_ID,
        )
        self.assertEqual(
            observation["$defs"]["rgbImage"]["x-robodojo-numpy"],
            {
                "dtype": "uint8",
                "shape": ["H", "W", 3],
                "layout": "HWC",
                "channel_order": "RGB",
                "dimension_range": {
                    "H": [1, MAX_IMAGE_DIMENSION],
                    "W": [1, MAX_IMAGE_DIMENSION],
                },
                "shape_owned_by": "RoboDojo ObservationValidationSpec",
            },
        )
        self.assertEqual(observation["properties"]["instruction"]["pattern"], r".*\S.*")
        self.assertFalse(observation["additionalProperties"])
        self.assertFalse(observation["properties"]["images"]["additionalProperties"])
        self.assertEqual(
            observation["properties"]["images"]["x-totalImageBytesMaximum"],
            MAX_CANONICAL_IMAGE_BYTES,
        )
        self.assertFalse(observation["properties"]["proprio"]["additionalProperties"])
        self.assertEqual(
            observation["$defs"]["armJointPosition"]["x-robodojo-numpy"],
            {
                "dtype": "float32",
                "shape": [6],
                "finite": True,
                "unit": "rad",
            },
        )
        self.assertEqual(
            observation["$defs"]["gripperCommand"]["x-robodojo-numpy"]["range"],
            [0.0, 1.0],
        )
        self.assertEqual(
            action["properties"]["control_mode"]["const"],
            ACTION_CONTROL_MODE,
        )
        self.assertFalse(action["additionalProperties"])
        self.assertFalse(action["properties"]["commands"]["additionalProperties"])
        self.assertEqual(
            action["properties"]["commands"]["x-shared-dimension"]["maximum"],
            MAX_ACTION_HORIZON,
        )
        self.assertEqual(
            action["$defs"]["armJointPositionChunk"]["x-robodojo-numpy"]["shape"],
            ["T", 6],
        )
        self.assertTrue(
            action["$defs"]["armJointPositionChunk"]["x-robodojo-numpy"]["finite"],
        )
        self.assertEqual(
            action["$defs"]["gripperChunk"]["x-robodojo-numpy"]["range"],
            [0.0, 1.0],
        )


class CanonicalObservationTest(unittest.TestCase):
    def assert_payload_error(self, payload, kind, path):
        with self.assertRaises(PayloadValidationError) as raised:
            _parse_observation(payload)
        self.assertEqual(raised.exception.kind, kind)
        self.assertEqual(raised.exception.path, path)
        self.assertEqual(raised.exception.details["path"], list(path))

    def test_valid_observation_is_an_immutable_independent_snapshot(self):
        payload = _observation_payload()
        source_head = _rgb(4, 6)[::2, ::2, :]
        source_left_arm = np.linspace(-0.5, 0.5, 6, dtype=np.float32)[::-1]
        self.assertFalse(source_head.flags.c_contiguous)
        self.assertFalse(source_left_arm.flags.c_contiguous)
        payload["images"]["head"] = source_head
        payload["proprio"]["left_arm_joint_position"] = source_left_arm

        parsed = _parse_observation(payload)
        expected_head = source_head.copy()
        expected_left_arm = source_left_arm.copy()
        self.assertTrue(parsed.head.flags.c_contiguous)
        self.assertTrue(parsed.left_arm_joint_position.flags.c_contiguous)
        self.assertFalse(parsed.head.flags.writeable)
        self.assertFalse(parsed.left_arm_joint_position.flags.writeable)
        self.assertFalse(np.shares_memory(parsed.head, source_head))
        self.assertFalse(np.shares_memory(parsed.left_arm_joint_position, source_left_arm))

        source_head.fill(255)
        source_left_arm.fill(0)
        np.testing.assert_array_equal(parsed.head, expected_head)
        np.testing.assert_array_equal(
            parsed.left_arm_joint_position,
            expected_left_arm,
        )
        with self.assertRaises(ValueError):
            parsed.head[0, 0, 0] = 0

        logical = parsed.to_payload()
        self.assertEqual(set(logical), {"instruction", "images", "proprio"})
        self.assertEqual(set(logical["images"]), {"head", "left_wrist", "right_wrist"})
        self.assertNotIn("session_id", logical)

    def test_exact_fields_are_enforced_at_every_level(self):
        cases = []

        missing_top = _observation_payload()
        missing_top.pop("instruction")
        cases.append((missing_top, PayloadErrorKind.MISSING_FIELD, ("instruction",)))

        extra_top = _observation_payload()
        extra_top["session_id"] = "must stay in Frame"
        cases.append((extra_top, PayloadErrorKind.UNKNOWN_FIELD, ("session_id",)))

        missing_image = _observation_payload()
        missing_image["images"].pop("head")
        cases.append(
            (
                missing_image,
                PayloadErrorKind.MISSING_FIELD,
                ("images", "head"),
            )
        )

        extra_image = _observation_payload()
        extra_image["images"]["depth"] = np.zeros((2, 3), dtype=np.float32)
        cases.append(
            (
                extra_image,
                PayloadErrorKind.UNKNOWN_FIELD,
                ("images", "depth"),
            )
        )

        missing_proprio = _observation_payload()
        missing_proprio["proprio"].pop("right_arm_joint_position")
        cases.append(
            (
                missing_proprio,
                PayloadErrorKind.MISSING_FIELD,
                ("proprio", "right_arm_joint_position"),
            )
        )

        extra_proprio = _observation_payload()
        extra_proprio["proprio"]["left_ee_pose"] = np.zeros(7, dtype=np.float32)
        cases.append(
            (
                extra_proprio,
                PayloadErrorKind.UNKNOWN_FIELD,
                ("proprio", "left_ee_pose"),
            )
        )

        for payload, kind, path in cases:
            with self.subTest(path=path):
                self.assert_payload_error(payload, kind, path)

    def test_instruction_is_nonempty_unicode_with_a_utf8_byte_limit(self):
        unicode_payload = _observation_payload()
        unicode_payload["instruction"] = "把面包放进烤面包机"
        self.assertEqual(
            _parse_observation(unicode_payload).instruction,
            "把面包放进烤面包机",
        )

        for value, kind in (
            ("   ", PayloadErrorKind.OUT_OF_RANGE),
            (b"make toast", PayloadErrorKind.INVALID_TYPE),
            ("中" * (MAX_INSTRUCTION_BYTES // 3 + 1), PayloadErrorKind.OUT_OF_RANGE),
        ):
            with self.subTest(value_type=type(value).__name__):
                payload = _observation_payload()
                payload["instruction"] = value
                self.assert_payload_error(payload, kind, ("instruction",))

    def test_image_dtype_layout_shape_and_size_are_strict(self):
        oversized = np.lib.stride_tricks.as_strided(
            np.zeros(1, dtype=np.uint8),
            shape=(MAX_IMAGE_DIMENSION + 1, 1, 3),
            strides=(0, 0, 0),
        )
        cases = (
            ([0, 1, 2], PayloadErrorKind.INVALID_TYPE),
            (np.zeros((2, 3, 3), dtype=np.float32), PayloadErrorKind.INVALID_DTYPE),
            (np.zeros((3, 2, 2), dtype=np.uint8), PayloadErrorKind.INVALID_SHAPE),
            (np.zeros((2, 2, 4), dtype=np.uint8), PayloadErrorKind.INVALID_SHAPE),
            (np.zeros((0, 2, 3), dtype=np.uint8), PayloadErrorKind.INVALID_SHAPE),
            (oversized, PayloadErrorKind.INVALID_SHAPE),
        )
        for value, kind in cases:
            with self.subTest(kind=kind, shape=getattr(value, "shape", None)):
                payload = _observation_payload()
                payload["images"]["head"] = value
                self.assert_payload_error(payload, kind, ("images", "head"))

    def test_proprio_dtype_shape_finite_range_and_embodiment_are_strict(self):
        cases = (
            (
                "robot_schema",
                "piper_dual_v1",
                PayloadErrorKind.CONSTRAINT_MISMATCH,
            ),
            ("robot_schema", np.array(["arx"]), PayloadErrorKind.INVALID_TYPE),
            (
                "left_arm_joint_position",
                np.zeros(6, dtype=np.float64),
                PayloadErrorKind.INVALID_DTYPE,
            ),
            (
                "left_arm_joint_position",
                np.zeros((1, 6), dtype=np.float32),
                PayloadErrorKind.INVALID_SHAPE,
            ),
            (
                "right_arm_joint_position",
                np.array([0, 0, 0, np.nan, 0, 0], dtype=np.float32),
                PayloadErrorKind.NON_FINITE,
            ),
            (
                "left_gripper_open_fraction_commanded",
                np.array([-0.01], dtype=np.float32),
                PayloadErrorKind.OUT_OF_RANGE,
            ),
            (
                "right_gripper_open_fraction_commanded",
                np.array([np.inf], dtype=np.float32),
                PayloadErrorKind.NON_FINITE,
            ),
        )
        for field, value, kind in cases:
            with self.subTest(field=field, kind=kind):
                payload = _observation_payload()
                payload["proprio"][field] = value
                self.assert_payload_error(payload, kind, ("proprio", field))

    def test_image_shapes_are_owned_by_the_connection_spec(self):
        self.assertEqual(
            ARX_X5_SIM_OBSERVATION_SPEC.head_image_shape,
            ARX_X5_SIM_RGB_SHAPE,
        )
        self.assertEqual(
            ARX_X5_SIM_OBSERVATION_SPEC.left_wrist_image_shape,
            ARX_X5_SIM_RGB_SHAPE,
        )
        self.assertEqual(
            ARX_X5_SIM_OBSERVATION_SPEC.right_wrist_image_shape,
            ARX_X5_SIM_RGB_SHAPE,
        )

        released_payload = _observation_payload()
        for camera in ("head", "left_wrist", "right_wrist"):
            released_payload["images"][camera] = np.zeros(
                ARX_X5_SIM_RGB_SHAPE,
                dtype=np.uint8,
            )
        released = parse_observation(
            released_payload,
            spec=ARX_X5_SIM_OBSERVATION_SPEC,
        )
        self.assertEqual(released.head.shape, ARX_X5_SIM_RGB_SHAPE)
        self.assertEqual(released.left_wrist.shape, ARX_X5_SIM_RGB_SHAPE)
        self.assertEqual(released.right_wrist.shape, ARX_X5_SIM_RGB_SHAPE)

        payload = _observation_payload()
        payload["images"]["head"] = np.zeros((3, 2, 3), dtype=np.uint8)
        self.assert_payload_error(
            payload,
            PayloadErrorKind.INVALID_SHAPE,
            ("images", "head"),
        )
        for camera in ("head", "left_wrist", "right_wrist"):
            with self.subTest(camera=camera):
                swapped = _observation_payload()
                for field in ("head", "left_wrist", "right_wrist"):
                    swapped["images"][field] = np.zeros(
                        ARX_X5_SIM_RGB_SHAPE,
                        dtype=np.uint8,
                    )
                swapped["images"][camera] = np.zeros(
                    (640, 480, 3),
                    dtype=np.uint8,
                )
                with self.assertRaises(PayloadValidationError) as raised:
                    parse_observation(
                        swapped,
                        spec=ARX_X5_SIM_OBSERVATION_SPEC,
                    )
                self.assertEqual(
                    raised.exception.path,
                    ("images", camera),
                )

        for shape, expected_exception in (
            ((0, 640, 3), ValueError),
            ((480, MAX_IMAGE_DIMENSION + 1, 3), ValueError),
            ((480, 640, 4), ValueError),
            ((480, 640), ValueError),
            ((True, 640, 3), TypeError),
            ((480.0, 640, 3), TypeError),
        ):
            with self.subTest(shape=shape):
                with self.assertRaises(expected_exception):
                    ObservationValidationSpec(
                        head_image_shape=shape,
                        left_wrist_image_shape=(480, 640, 3),
                        right_wrist_image_shape=(480, 640, 3),
                    )
        with self.assertRaisesRegex(ValueError, "wire-byte budget"):
            ObservationValidationSpec(
                head_image_shape=(4096, 4096, 3),
                left_wrist_image_shape=(4096, 4096, 3),
                right_wrist_image_shape=(4096, 4096, 3),
            )

    def test_ndarray_subclass_cannot_spoof_image_metadata(self):
        spoofed = np.zeros((10, 10, 3), dtype=np.uint8).view(
            _SpoofedShapeArray,
        )
        spoofed.spoofed_shape = (2, 3, 3)
        payload = _observation_payload()
        payload["images"]["head"] = spoofed
        self.assert_payload_error(
            payload,
            PayloadErrorKind.INVALID_SHAPE,
            ("images", "head"),
        )

    def test_direct_constructor_cannot_bypass_validation(self):
        with self.assertRaisesRegex(TypeError, "parse_observation"):
            CanonicalObservation(
                instruction="make toast",
                head=np.zeros((2, 3, 3), dtype=np.uint8),
                left_wrist=np.zeros((3, 2, 3), dtype=np.uint8),
                right_wrist=np.zeros((1, 4, 3), dtype=np.uint8),
                left_arm_joint_position=np.zeros(6, dtype=np.float32),
                left_gripper_open_fraction_commanded=np.zeros(1, dtype=np.float32),
                right_arm_joint_position=np.zeros(6, dtype=np.float32),
                right_gripper_open_fraction_commanded=np.zeros(1, dtype=np.float32),
            )


class CanonicalActionTest(unittest.TestCase):
    def assert_payload_error(self, payload, kind, path, spec=None):
        with self.assertRaises(PayloadValidationError) as raised:
            parse_action_chunk(payload, spec=spec or _action_spec())
        self.assertEqual(raised.exception.kind, kind)
        self.assertEqual(raised.exception.path, path)

    def test_valid_action_is_an_immutable_independent_snapshot(self):
        payload = _action_payload()
        source = np.arange(24, dtype=np.float32).reshape(2, 12)[:, ::2]
        source /= 24
        self.assertEqual(source.shape, (2, 6))
        self.assertFalse(source.flags.c_contiguous)
        payload["commands"]["left_arm_joint_position"] = source

        parsed = parse_action_chunk(payload, spec=_action_spec())
        expected = source.copy()
        self.assertEqual(parsed.horizon, 2)
        self.assertTrue(parsed.left_arm_joint_position.flags.c_contiguous)
        self.assertFalse(parsed.left_arm_joint_position.flags.writeable)
        self.assertFalse(
            np.shares_memory(parsed.left_arm_joint_position, source),
        )
        source.fill(0)
        np.testing.assert_array_equal(parsed.left_arm_joint_position, expected)
        with self.assertRaises(ValueError):
            parsed.left_arm_joint_position[0, 0] = 0

    def test_exact_fields_control_mode_and_control_period_are_strict(self):
        cases = []

        extra_top = _action_payload()
        extra_top["episode_id"] = "must stay in Frame"
        cases.append(
            (
                extra_top,
                PayloadErrorKind.UNKNOWN_FIELD,
                ("episode_id",),
            )
        )

        missing_command = _action_payload()
        missing_command["commands"].pop("right_gripper_open_fraction")
        cases.append(
            (
                missing_command,
                PayloadErrorKind.MISSING_FIELD,
                ("commands", "right_gripper_open_fraction"),
            )
        )

        wrong_mode = _action_payload()
        wrong_mode["control_mode"] = "delta_joint_position"
        cases.append(
            (
                wrong_mode,
                PayloadErrorKind.CONSTRAINT_MISMATCH,
                ("control_mode",),
            )
        )

        wrong_mode_type = _action_payload()
        wrong_mode_type["control_mode"] = np.array(["absolute_joint_position"])
        cases.append(
            (
                wrong_mode_type,
                PayloadErrorKind.INVALID_TYPE,
                ("control_mode",),
            )
        )

        for value, kind in (
            (True, PayloadErrorKind.INVALID_TYPE),
            (0.0, PayloadErrorKind.OUT_OF_RANGE),
            (np.nan, PayloadErrorKind.NON_FINITE),
            (np.inf, PayloadErrorKind.NON_FINITE),
            (0.05, PayloadErrorKind.CONSTRAINT_MISMATCH),
        ):
            payload = _action_payload()
            payload["control_dt_s"] = value
            cases.append((payload, kind, ("control_dt_s",)))

        for payload, kind, path in cases:
            with self.subTest(kind=kind, path=path):
                self.assert_payload_error(payload, kind, path)

    def test_action_array_dtype_shape_finite_and_gripper_range_are_strict(self):
        cases = (
            (
                "left_arm_joint_position",
                [[0.0] * 6] * 2,
                PayloadErrorKind.INVALID_TYPE,
            ),
            (
                "left_arm_joint_position",
                np.zeros((2, 6), dtype=np.float64),
                PayloadErrorKind.INVALID_DTYPE,
            ),
            (
                "right_arm_joint_position",
                np.zeros((2, 7), dtype=np.float32),
                PayloadErrorKind.INVALID_SHAPE,
            ),
            (
                "right_arm_joint_position",
                np.full((2, 6), np.nan, dtype=np.float32),
                PayloadErrorKind.NON_FINITE,
            ),
            (
                "left_gripper_open_fraction",
                np.array([[0], [1.01]], dtype=np.float32),
                PayloadErrorKind.OUT_OF_RANGE,
            ),
            (
                "right_gripper_open_fraction",
                np.array([[np.inf], [0]], dtype=np.float32),
                PayloadErrorKind.NON_FINITE,
            ),
        )
        for field, value, kind in cases:
            with self.subTest(field=field, kind=kind):
                payload = _action_payload()
                payload["commands"][field] = value
                self.assert_payload_error(payload, kind, ("commands", field))

    def test_ndarray_subclass_cannot_spoof_action_metadata(self):
        spoofed = np.zeros((2, 7), dtype=np.float32).view(
            _SpoofedShapeArray,
        )
        spoofed.spoofed_shape = (2, 6)
        payload = _action_payload()
        payload["commands"]["left_arm_joint_position"] = spoofed
        self.assert_payload_error(
            payload,
            PayloadErrorKind.INVALID_SHAPE,
            ("commands", "left_arm_joint_position"),
        )

    def test_horizon_must_be_shared_bounded_and_negotiated(self):
        mismatched = _action_payload()
        mismatched["commands"]["right_arm_joint_position"] = np.zeros(
            (1, 6),
            dtype=np.float32,
        )
        self.assert_payload_error(
            mismatched,
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            ("commands",),
        )

        empty = _action_payload(horizon=0)
        self.assert_payload_error(
            empty,
            PayloadErrorKind.OUT_OF_RANGE,
            ("commands",),
            spec=_action_spec(horizon=1),
        )

        payload = _action_payload(horizon=2)
        self.assert_payload_error(
            payload,
            PayloadErrorKind.CONSTRAINT_MISMATCH,
            ("commands",),
            spec=_action_spec(horizon=3),
        )

    def test_joint_limits_are_client_owned_and_side_specific(self):
        valid = parse_action_chunk(_action_payload(), spec=_action_spec())
        self.assertEqual(valid.horizon, 2)

        left_out_of_range = _action_payload()
        left_out_of_range["commands"]["left_arm_joint_position"][0, 0] = 1.5
        self.assert_payload_error(
            left_out_of_range,
            PayloadErrorKind.OUT_OF_RANGE,
            ("commands", "left_arm_joint_position"),
        )

        right_still_valid = _action_payload()
        right_still_valid["commands"]["right_arm_joint_position"][0, 0] = 1.9
        parse_action_chunk(right_still_valid, spec=_action_spec())

    def test_validation_spec_rejects_invalid_local_configuration(self):
        limits = JointLimits(lower=(-1,) * 6, upper=(1,) * 6)
        for kwargs in (
            {
                "expected_horizon": True,
                "expected_control_dt_s": 0.04,
                "left_arm_limits": limits,
                "right_arm_limits": limits,
            },
            {
                "expected_horizon": 0,
                "expected_control_dt_s": 0.04,
                "left_arm_limits": limits,
                "right_arm_limits": limits,
            },
            {
                "expected_horizon": MAX_ACTION_HORIZON + 1,
                "expected_control_dt_s": 0.04,
                "left_arm_limits": limits,
                "right_arm_limits": limits,
            },
            {
                "expected_horizon": 2,
                "expected_control_dt_s": True,
                "left_arm_limits": limits,
                "right_arm_limits": limits,
            },
            {
                "expected_horizon": 2,
                "expected_control_dt_s": np.nan,
                "left_arm_limits": limits,
                "right_arm_limits": limits,
            },
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ActionValidationSpec(**kwargs)

        with self.assertRaises(ValueError):
            JointLimits(lower=(0,) * 5, upper=(1,) * 6)
        with self.assertRaises(ValueError):
            JointLimits(lower=(1,) * 6, upper=(0,) * 6)
        with self.assertRaises(TypeError):
            ActionValidationSpec(
                expected_horizon=2,
                expected_control_dt_s=0.04,
            )
        with self.assertRaises(TypeError):
            ActionValidationSpec(
                expected_horizon=2,
                expected_control_dt_s=0.04,
                left_arm_limits=limits,
                right_arm_limits=None,
            )

    def test_released_pi05_profile_is_a_two_second_action_chunk(self):
        action = parse_action_chunk(
            _action_payload(horizon=50),
            spec=_action_spec(horizon=50),
        )
        self.assertEqual(action.horizon, 50)
        self.assertAlmostEqual(action.horizon * action.control_dt_s, 2.0)

    def test_direct_constructor_cannot_bypass_validation(self):
        with self.assertRaisesRegex(TypeError, "parse_action_chunk"):
            CanonicalActionChunk(
                control_dt_s=np.nan,
                left_arm_joint_position=np.zeros((2, 6), dtype=np.float32),
                left_gripper_open_fraction=np.zeros((2, 1), dtype=np.float32),
                right_arm_joint_position=np.zeros((2, 6), dtype=np.float32),
                right_gripper_open_fraction=np.zeros((2, 1), dtype=np.float32),
            )


class CanonicalContractIntegrationTest(unittest.TestCase):
    def test_canonical_payloads_survive_msgpack_round_trip(self):
        observation = _parse_observation(_observation_payload())
        infer = _infer(
            "infer-0",
            {"observation": observation.to_payload()},
        )
        decoded_infer = decode_frame(encode_frame(infer))
        decoded_observation = parse_infer_payload(
            decoded_infer.payload,
            observation_spec=_observation_spec(),
        )
        np.testing.assert_array_equal(decoded_observation.head, observation.head)

        action = parse_action_chunk(_action_payload(), spec=_action_spec())
        result = Frame(
            message_type=MessageType.INFER_RESULT,
            request_id="infer-0",
            session_id="session-1",
            episode_id="episode-1",
            inference_index=0,
            payload={"action": action.to_payload()},
        )
        decoded_result = decode_frame(encode_frame(result))
        decoded_action = parse_infer_result_payload(
            decoded_result.payload,
            action_spec=_action_spec(),
        )
        np.testing.assert_array_equal(
            decoded_action.right_arm_joint_position,
            action.right_arm_joint_position,
        )

    def test_infer_wrappers_are_exact_maps(self):
        invalid_infer = {
            "observation": _observation_payload(),
            "diagnostics": {},
        }
        with self.assertRaises(PayloadValidationError) as infer_error:
            parse_infer_payload(
                invalid_infer,
                observation_spec=_observation_spec(),
            )
        self.assertEqual(infer_error.exception.kind, PayloadErrorKind.UNKNOWN_FIELD)
        self.assertEqual(infer_error.exception.path, ("diagnostics",))

        invalid_result = {
            "action": _action_payload(),
            "latency_ms": 1.0,
        }
        with self.assertRaises(PayloadValidationError) as result_error:
            parse_infer_result_payload(
                invalid_result,
                action_spec=_action_spec(),
            )
        self.assertEqual(result_error.exception.kind, PayloadErrorKind.UNKNOWN_FIELD)
        self.assertEqual(result_error.exception.path, ("latency_ms",))

        invalid_observation = _observation_payload()
        invalid_observation["images"]["head"] = np.zeros(
            (2, 3, 4),
            dtype=np.uint8,
        )
        with self.assertRaises(PayloadValidationError) as nested_infer_error:
            parse_infer_payload(
                {"observation": invalid_observation},
                observation_spec=_observation_spec(),
            )
        self.assertEqual(
            nested_infer_error.exception.path,
            ("observation", "images", "head"),
        )

        invalid_action = _action_payload()
        invalid_action["commands"]["right_gripper_open_fraction"][0, 0] = 2.0
        with self.assertRaises(PayloadValidationError) as nested_result_error:
            parse_infer_result_payload(
                {"action": invalid_action},
                action_spec=_action_spec(),
            )
        self.assertEqual(
            nested_result_error.exception.path,
            ("action", "commands", "right_gripper_open_fraction"),
        )

    def test_invalid_observation_is_recoverable_before_backend_runs(self):
        session = PolicySession()
        session.complete(session.begin(_hello()))
        session.complete(session.begin(_reset()))

        invalid = _observation_payload()
        invalid["images"]["head"] = np.zeros((3, 2, 2), dtype=np.uint8)
        token = session.begin(
            _infer("invalid-observation", {"observation": invalid}),
        )
        with self.assertRaises(PayloadValidationError):
            _parse_observation(invalid)
        rejected = session.reject(token)
        self.assertEqual(rejected.phase, SessionPhase.ACTIVE)
        self.assertTrue(rejected.reply_allowed)
        self.assertFalse(rejected.close_after_reply)
        self.assertEqual(session.snapshot.next_inference_index, 0)

        with self.assertRaises(ProtocolError) as duplicate:
            session.begin(
                _infer(
                    "invalid-observation",
                    {"observation": _observation_payload()},
                )
            )
        self.assertEqual(duplicate.exception.code, ErrorCode.INVALID_STATE)

        corrected_frame = _infer(
            "corrected-observation",
            {"observation": _observation_payload()},
        )
        corrected = session.begin(corrected_frame)
        parse_infer_payload(
            corrected_frame.payload,
            observation_spec=_observation_spec(),
        )
        session.complete(corrected)
        self.assertEqual(session.snapshot.next_inference_index, 1)

    def test_invalid_backend_action_is_terminal_after_backend_runs(self):
        session = PolicySession()
        session.complete(session.begin(_hello()))
        session.complete(session.begin(_reset()))
        token = session.begin(
            _infer("infer-0", {"observation": _observation_payload()}),
        )

        invalid_action = _action_payload()
        invalid_action["commands"]["left_arm_joint_position"][0, 0] = np.nan
        with self.assertRaises(PayloadValidationError):
            parse_action_chunk(invalid_action, spec=_action_spec())
        failed = session.fail(token)
        self.assertEqual(failed.phase, SessionPhase.TERMINATING)
        self.assertTrue(failed.episode_lost)


if __name__ == "__main__":
    unittest.main()
