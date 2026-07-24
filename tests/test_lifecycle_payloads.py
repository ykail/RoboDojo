import unittest

import numpy as np

from src.eval_client.policy_runtime import (
    ARX_X5_SIM_PI05_PROFILE,
    ErrorCode,
    Frame,
    MessageType,
    PayloadErrorKind,
    PayloadValidationError,
    PolicyProvenance,
    RemoteErrorPayload,
    ResetPayload,
    ResetReason,
    TrialEndPayload,
    TrialStatus,
    build_hello_ack_payload,
    build_hello_payload,
    decode_frame,
    encode_frame,
    parse_empty_success_payload,
    parse_error_payload,
    parse_hello_ack_payload,
    parse_hello_payload,
    parse_reset_payload,
    parse_trial_end_payload,
)


def _provenance():
    return PolicyProvenance(
        implementation="kai0",
        policy_family="pi05",
        adapter_profile="kai0_pi05_aloha_arx_x5_joint_v1",
        config_name="pi05_robodojo",
        checkpoint_id="make_toast_left_dagger/15000",
        checkpoint_digest="sha256:" + "a" * 64,
        checkpoint_step=15000,
        code_revision="5c2645f5a2a5a005d43efafa42f338a14dc7442c",
        dirty=False,
    )


def _reset():
    return ResetPayload(
        task_name="make_toast",
        simulator_seed=17,
        policy_seed=23,
        layout_id=17,
        layout_cycle=0,
        reason=ResetReason.EPISODE_START,
    )


class HelloPayloadTest(unittest.TestCase):
    def test_hello_round_trip_preserves_the_client_owned_profile(self):
        payload = build_hello_payload(ARX_X5_SIM_PI05_PROFILE)
        parsed = parse_hello_payload(
            payload,
            supported_profile=ARX_X5_SIM_PI05_PROFILE,
        )

        self.assertEqual(parsed, ARX_X5_SIM_PI05_PROFILE)
        self.assertEqual(
            payload["schemas"],
            ARX_X5_SIM_PI05_PROFILE.schemas_payload(),
        )
        self.assertEqual(payload["execution_profile"]["images"]["head"], [480, 640, 3])
        self.assertEqual(payload["execution_profile"]["action"]["horizon"], 50)
        self.assertEqual(
            payload["execution_profile"]["action"]["control_dt_s"],
            0.04,
        )

    def test_hello_ack_confirms_profile_and_returns_provenance(self):
        payload = build_hello_ack_payload(
            ARX_X5_SIM_PI05_PROFILE,
            _provenance(),
        )
        parsed = parse_hello_ack_payload(
            payload,
            expected_profile=ARX_X5_SIM_PI05_PROFILE,
        )

        self.assertEqual(parsed, _provenance())
        self.assertEqual(parsed.checkpoint_step, 15000)
        self.assertFalse(parsed.dirty)

    def test_hello_payload_survives_real_frame_codec(self):
        frame = Frame(
            message_type=MessageType.HELLO,
            request_id="hello-1",
            session_id="session-1",
            payload=build_hello_payload(ARX_X5_SIM_PI05_PROFILE),
        )
        decoded = decode_frame(encode_frame(frame))
        self.assertEqual(
            parse_hello_payload(
                decoded.payload,
                supported_profile=ARX_X5_SIM_PI05_PROFILE,
            ),
            ARX_X5_SIM_PI05_PROFILE,
        )

    def test_hello_exact_maps_schema_literals_and_wire_lists_are_strict(self):
        cases = []

        extra = build_hello_payload(ARX_X5_SIM_PI05_PROFILE)
        extra["capabilities"] = []
        cases.append(
            (
                extra,
                PayloadErrorKind.UNKNOWN_FIELD,
                ("capabilities",),
            )
        )

        schema = build_hello_payload(ARX_X5_SIM_PI05_PROFILE)
        schema["schemas"]["observation"] = "unknown"
        cases.append(
            (
                schema,
                PayloadErrorKind.CONSTRAINT_MISMATCH,
                ("schemas", "observation"),
            )
        )

        tuple_shape = build_hello_payload(ARX_X5_SIM_PI05_PROFILE)
        tuple_shape["execution_profile"]["images"]["head"] = (480, 640, 3)
        cases.append(
            (
                tuple_shape,
                PayloadErrorKind.INVALID_TYPE,
                ("execution_profile", "images", "head"),
            )
        )

        wrong_channels = build_hello_payload(ARX_X5_SIM_PI05_PROFILE)
        wrong_channels["execution_profile"]["images"]["right_wrist"] = [
            480,
            640,
            4,
        ]
        cases.append(
            (
                wrong_channels,
                PayloadErrorKind.CONSTRAINT_MISMATCH,
                ("execution_profile", "images", "right_wrist", "2"),
            )
        )

        bad_horizon = build_hello_payload(ARX_X5_SIM_PI05_PROFILE)
        bad_horizon["execution_profile"]["action"]["horizon"] = 0
        cases.append(
            (
                bad_horizon,
                PayloadErrorKind.OUT_OF_RANGE,
                ("execution_profile", "action", "horizon"),
            )
        )

        bad_limits = build_hello_payload(ARX_X5_SIM_PI05_PROFILE)
        bad_limits["execution_profile"]["action"]["left_arm_joint_limits"]["lower"][0] = 11.0
        cases.append(
            (
                bad_limits,
                PayloadErrorKind.CONSTRAINT_MISMATCH,
                ("execution_profile", "action", "left_arm_joint_limits"),
            )
        )

        bad_chunk_consumption = build_hello_payload(
            ARX_X5_SIM_PI05_PROFILE,
        )
        bad_chunk_consumption["execution_profile"]["action"]["chunk_consumption"] = "receding_horizon"
        cases.append(
            (
                bad_chunk_consumption,
                PayloadErrorKind.CONSTRAINT_MISMATCH,
                (
                    "execution_profile",
                    "action",
                    "chunk_consumption",
                ),
            )
        )

        for payload, kind, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(PayloadValidationError) as raised:
                    parse_hello_payload(
                        payload,
                        supported_profile=ARX_X5_SIM_PI05_PROFILE,
                    )
                self.assertEqual(raised.exception.kind, kind)
                self.assertEqual(raised.exception.path, path)

    def test_server_rejects_a_well_formed_but_unsupported_profile(self):
        hello = build_hello_payload(ARX_X5_SIM_PI05_PROFILE)
        hello["execution_profile"]["action"]["control_dt_s"] = 0.05
        with self.assertRaises(PayloadValidationError) as raised:
            parse_hello_payload(
                hello,
                supported_profile=ARX_X5_SIM_PI05_PROFILE,
            )
        self.assertEqual(
            raised.exception.kind,
            PayloadErrorKind.CONSTRAINT_MISMATCH,
        )
        self.assertEqual(raised.exception.path, ("execution_profile",))

    def test_ack_must_echo_the_exact_client_profile(self):
        ack = build_hello_ack_payload(
            ARX_X5_SIM_PI05_PROFILE,
            _provenance(),
        )
        ack["execution_profile"]["action"]["control_dt_s"] = 0.05
        with self.assertRaises(PayloadValidationError) as raised:
            parse_hello_ack_payload(
                ack,
                expected_profile=ARX_X5_SIM_PI05_PROFILE,
            )
        self.assertEqual(
            raised.exception.kind,
            PayloadErrorKind.CONSTRAINT_MISMATCH,
        )
        self.assertEqual(raised.exception.path, ("execution_profile",))

    def test_policy_provenance_is_exact_and_validated(self):
        payload = build_hello_ack_payload(
            ARX_X5_SIM_PI05_PROFILE,
            _provenance(),
        )
        payload["policy"]["branch"] = "main"
        with self.assertRaises(PayloadValidationError) as raised:
            parse_hello_ack_payload(
                payload,
                expected_profile=ARX_X5_SIM_PI05_PROFILE,
            )
        self.assertEqual(raised.exception.path, ("policy", "branch"))

        for kwargs, exception in (
            ({"implementation": " "}, ValueError),
            ({"checkpoint_id": "/home/user/checkpoints/5000"}, ValueError),
            ({"checkpoint_id": r"C:\checkpoints\5000"}, ValueError),
            ({"checkpoint_id": "file:///data/checkpoints/5000"}, ValueError),
            ({"checkpoint_id": "../checkpoints/5000"}, ValueError),
            ({"checkpoint_digest": "sha256:ABC"}, ValueError),
            ({"checkpoint_step": True}, TypeError),
            ({"checkpoint_step": -1}, ValueError),
            ({"code_revision": "5c2645f"}, ValueError),
            ({"dirty": 0}, TypeError),
        ):
            with self.subTest(kwargs=kwargs):
                values = _provenance().to_payload()
                values.update(kwargs)
                with self.assertRaises(exception):
                    PolicyProvenance(**values)

        invalid_digest = build_hello_ack_payload(
            ARX_X5_SIM_PI05_PROFILE,
            _provenance(),
        )
        invalid_digest["policy"]["checkpoint_digest"] = "sha256:" + "A" * 64
        with self.assertRaises(PayloadValidationError) as raised:
            parse_hello_ack_payload(
                invalid_digest,
                expected_profile=ARX_X5_SIM_PI05_PROFILE,
            )
        self.assertEqual(
            raised.exception.path,
            ("policy", "checkpoint_digest"),
        )

        local_checkpoint = build_hello_ack_payload(
            ARX_X5_SIM_PI05_PROFILE,
            _provenance(),
        )
        local_checkpoint["policy"]["checkpoint_id"] = "/data/checkpoints/5000"
        with self.assertRaises(PayloadValidationError) as raised:
            parse_hello_ack_payload(
                local_checkpoint,
                expected_profile=ARX_X5_SIM_PI05_PROFILE,
            )
        self.assertEqual(
            raised.exception.kind,
            PayloadErrorKind.CONSTRAINT_MISMATCH,
        )
        self.assertEqual(
            raised.exception.path,
            ("policy", "checkpoint_id"),
        )


class ResetPayloadTest(unittest.TestCase):
    def test_reset_payload_round_trip(self):
        reset = _reset()
        self.assertEqual(parse_reset_payload(reset.to_payload()), reset)
        self.assertEqual(
            reset.to_payload(),
            {
                "task_name": "make_toast",
                "simulator_seed": 17,
                "policy_seed": 23,
                "layout_id": 17,
                "layout_cycle": 0,
                "reason": "episode_start",
            },
        )

    def test_reset_exact_fields_types_and_reason_are_strict(self):
        cases = []

        missing = _reset().to_payload()
        missing.pop("simulator_seed")
        cases.append(
            (
                missing,
                PayloadErrorKind.MISSING_FIELD,
                ("simulator_seed",),
            )
        )

        extra = _reset().to_payload()
        extra["episode_id"] = "envelope-only"
        cases.append((extra, PayloadErrorKind.UNKNOWN_FIELD, ("episode_id",)))

        bool_seed = _reset().to_payload()
        bool_seed["simulator_seed"] = True
        cases.append(
            (
                bool_seed,
                PayloadErrorKind.INVALID_TYPE,
                ("simulator_seed",),
            )
        )

        oversized_policy_seed = _reset().to_payload()
        oversized_policy_seed["policy_seed"] = 1 << 32
        cases.append(
            (
                oversized_policy_seed,
                PayloadErrorKind.OUT_OF_RANGE,
                ("policy_seed",),
            )
        )

        negative_cycle = _reset().to_payload()
        negative_cycle["layout_cycle"] = -1
        cases.append(
            (
                negative_cycle,
                PayloadErrorKind.OUT_OF_RANGE,
                ("layout_cycle",),
            )
        )

        bad_reason = _reset().to_payload()
        bad_reason["reason"] = "retry"
        cases.append(
            (
                bad_reason,
                PayloadErrorKind.CONSTRAINT_MISMATCH,
                ("reason",),
            )
        )

        for payload, kind, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(PayloadValidationError) as raised:
                    parse_reset_payload(payload)
                self.assertEqual(raised.exception.kind, kind)
                self.assertEqual(raised.exception.path, path)

    def test_local_reset_object_cannot_encode_invalid_values(self):
        values = _reset().to_payload()
        values["reason"] = ResetReason.EPISODE_START
        for key, value, exception in (
            ("task_name", "", ValueError),
            ("simulator_seed", True, TypeError),
            ("simulator_seed", 1 << 64, ValueError),
            ("policy_seed", 1 << 32, ValueError),
            ("layout_id", -1, ValueError),
            ("reason", "episode_start", TypeError),
        ):
            with self.subTest(key=key):
                invalid = dict(values)
                invalid[key] = value
                with self.assertRaises(exception):
                    ResetPayload(**invalid)


class TrialEndPayloadTest(unittest.TestCase):
    def test_all_unambiguous_outcomes_round_trip(self):
        values = (
            TrialEndPayload(TrialStatus.SUCCESS, True, 1.0, None),
            TrialEndPayload(TrialStatus.FAILURE, False, 0.25, "missed"),
            TrialEndPayload(TrialStatus.ABORTED, None, None, "operator"),
            TrialEndPayload(TrialStatus.ERROR, None, None, "simulator"),
        )
        for value in values:
            with self.subTest(status=value.status):
                self.assertEqual(
                    parse_trial_end_payload(value.to_payload()),
                    value,
                )

    def test_outcome_cross_fields_and_values_are_strict(self):
        cases = (
            (
                {
                    "status": "success",
                    "success": False,
                    "score": 1.0,
                    "reason": None,
                },
                PayloadErrorKind.CONSTRAINT_MISMATCH,
                ("success",),
            ),
            (
                {
                    "status": "aborted",
                    "success": False,
                    "score": None,
                    "reason": "operator",
                },
                PayloadErrorKind.CONSTRAINT_MISMATCH,
                ("success",),
            ),
            (
                {
                    "status": "failure",
                    "success": False,
                    "score": np.nan,
                    "reason": None,
                },
                PayloadErrorKind.NON_FINITE,
                ("score",),
            ),
            (
                {
                    "status": "failure",
                    "success": False,
                    "score": 0.0,
                    "reason": "",
                },
                PayloadErrorKind.OUT_OF_RANGE,
                ("reason",),
            ),
        )
        for payload, kind, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(PayloadValidationError) as raised:
                    parse_trial_end_payload(payload)
                self.assertEqual(raised.exception.kind, kind)
                self.assertEqual(raised.exception.path, path)


class ErrorAndEmptyPayloadTest(unittest.TestCase):
    def test_correlated_error_round_trip_and_details_snapshot(self):
        details = {
            "kind": "invalid_shape",
            "path": ["images", "head"],
            "context": {"index": 1},
        }
        error = RemoteErrorPayload(
            code=ErrorCode.INVALID_PAYLOAD,
            message="bad observation",
            details=details,
            retryable=False,
        )
        details["kind"] = "mutated"
        details["path"].append("mutated")
        details["context"]["index"] = 2

        parsed = parse_error_payload(error.to_payload())
        self.assertEqual(parsed.code, ErrorCode.INVALID_PAYLOAD)
        self.assertEqual(parsed.details["kind"], "invalid_shape")
        self.assertEqual(parsed.details["path"], ("images", "head"))
        self.assertEqual(parsed.details["context"]["index"], 1)
        self.assertFalse(parsed.retryable)
        with self.assertRaises(TypeError):
            parsed.details["new"] = "not mutable"

        wire_projection = parsed.to_payload()
        wire_projection["path"] = ["wrong top-level"]
        wire_projection["details"]["path"].append("mutated")
        self.assertEqual(parsed.details["path"], ("images", "head"))

        frame = Frame(
            message_type=MessageType.ERROR,
            request_id="infer-1",
            session_id="session-1",
            episode_id="episode-1",
            inference_index=0,
            payload=parsed.to_payload(),
        )
        decoded = decode_frame(encode_frame(frame))
        self.assertEqual(
            parse_error_payload(decoded.payload),
            parsed,
        )

    def test_error_code_retryability_and_details_are_strict(self):
        base = RemoteErrorPayload(
            code=ErrorCode.INFER_FAILED,
            message="inference failed",
            details={},
            retryable=False,
        ).to_payload()
        cases = (
            (
                {**base, "code": ErrorCode.INVALID_FRAME.value},
                PayloadErrorKind.CONSTRAINT_MISMATCH,
                ("code",),
            ),
            (
                {**base, "code": "unknown"},
                PayloadErrorKind.CONSTRAINT_MISMATCH,
                ("code",),
            ),
            (
                {**base, "retryable": True},
                PayloadErrorKind.CONSTRAINT_MISMATCH,
                ("retryable",),
            ),
            (
                {**base, "details": {1: "bad key"}},
                PayloadErrorKind.INVALID_TYPE,
                ("details",),
            ),
            (
                {**base, "details": {"nested": {1: "bad key"}}},
                PayloadErrorKind.INVALID_TYPE,
                ("details", "nested"),
            ),
            (
                {**base, "details": {"tuple": ("not", "a", "wire list")}},
                PayloadErrorKind.INVALID_TYPE,
                ("details", "tuple"),
            ),
            (
                {**base, "details": {"value": np.inf}},
                PayloadErrorKind.NON_FINITE,
                ("details", "value"),
            ),
        )
        for payload, kind, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(PayloadValidationError) as raised:
                    parse_error_payload(payload)
                self.assertEqual(raised.exception.kind, kind)
                self.assertEqual(raised.exception.path, path)

    def test_local_error_details_cannot_defer_an_encode_failure(self):
        for details, exception in (
            ({"object": object()}, TypeError),
            ({"tuple": ("not", "allowed")}, TypeError),
            ({"integer": 1 << 65}, ValueError),
            ({"float": np.nan}, ValueError),
            ([], TypeError),
        ):
            with self.subTest(details=details):
                with self.assertRaises(exception):
                    RemoteErrorPayload(
                        code=ErrorCode.INTERNAL,
                        message="internal",
                        details=details,
                        retryable=False,
                    )

    def test_success_ack_payloads_are_exact_empty_maps(self):
        self.assertIsNone(parse_empty_success_payload({}))
        for payload, kind, path in (
            ({"result": None}, PayloadErrorKind.UNKNOWN_FIELD, ("result",)),
            ([], PayloadErrorKind.INVALID_TYPE, ()),
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(PayloadValidationError) as raised:
                    parse_empty_success_payload(payload)
                self.assertEqual(raised.exception.kind, kind)
                self.assertEqual(raised.exception.path, path)


if __name__ == "__main__":
    unittest.main()
