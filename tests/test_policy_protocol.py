from pathlib import Path
import unittest

import msgpack
import numpy as np

from src.eval_client.policy_runtime import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    REQUEST_RESPONSE_PAIRS,
    ErrorCode,
    Frame,
    MessageType,
    ProtocolError,
    decode_frame,
    encode_frame,
)
from src.eval_client.policy_runtime.messages import REQUEST_TYPES, RESPONSE_TYPES


def _frame(**overrides):
    values = {
        "message_type": MessageType.INFER,
        "request_id": "request-1",
        "session_id": "session-1",
        "episode_id": "make_toast-0001",
        "inference_index": 0,
        "payload": {"observation": {"instruction": "make toast"}},
    }
    values.update(overrides)
    return Frame(**values)


def _wire_dict(**overrides):
    values = _frame().to_wire_dict()
    values.update(overrides)
    return values


class PolicyMessageTest(unittest.TestCase):
    def test_v1_message_vocabulary_is_policy_agnostic(self):
        self.assertEqual(PROTOCOL_VERSION, "robodojo-policy-v1")
        self.assertEqual(
            {item.value for item in MessageType},
            {
                "hello",
                "hello_ack",
                "reset",
                "reset_result",
                "infer",
                "infer_result",
                "trial_end",
                "trial_end_ack",
                "error",
            },
        )
        self.assertNotIn("prepare_case", {item.value for item in MessageType})
        self.assertNotIn("get_action", {item.value for item in MessageType})

    def test_request_response_pairs_are_unambiguous(self):
        self.assertEqual(
            REQUEST_RESPONSE_PAIRS,
            {
                MessageType.HELLO: MessageType.HELLO_ACK,
                MessageType.RESET: MessageType.RESET_RESULT,
                MessageType.INFER: MessageType.INFER_RESULT,
                MessageType.TRIAL_END: MessageType.TRIAL_END_ACK,
            },
        )
        self.assertFalse(REQUEST_TYPES & RESPONSE_TYPES)

    def test_v1_error_codes_use_inference_vocabulary(self):
        self.assertEqual(
            {item.value for item in ErrorCode},
            {
                "invalid_frame",
                "unsupported_version",
                "unknown_message_type",
                "invalid_state",
                "session_busy",
                "episode_mismatch",
                "inference_index_mismatch",
                "episode_lost",
                "timeout",
                "infer_failed",
                "reset_failed",
                "internal",
            },
        )


class PolicyFrameTest(unittest.TestCase):
    def test_frame_wire_round_trip(self):
        original = _frame()
        decoded = Frame.from_wire_dict(original.to_wire_dict())
        self.assertEqual(decoded, original)

    def test_missing_and_extra_envelope_fields_are_rejected(self):
        missing = _wire_dict()
        missing.pop("session_id")
        with self.assertRaises(ProtocolError) as raised:
            Frame.from_wire_dict(missing)
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)

        with self.assertRaises(ProtocolError) as raised:
            Frame.from_wire_dict({**_wire_dict(), "policy_name": "Pi_05"})
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)

        with self.assertRaises(ProtocolError) as raised:
            Frame.from_wire_dict({**_wire_dict(), b"legacy": True})
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)

    def test_version_and_message_type_are_strict(self):
        with self.assertRaises(ProtocolError) as raised:
            Frame.from_wire_dict(_wire_dict(protocol_version="robodojo-policy-v2"))
        self.assertEqual(raised.exception.code, ErrorCode.UNSUPPORTED_VERSION)

        with self.assertRaises(ProtocolError) as raised:
            Frame.from_wire_dict(_wire_dict(message_type="get_action"))
        self.assertEqual(raised.exception.code, ErrorCode.UNKNOWN_MESSAGE_TYPE)

    def test_identifiers_and_inference_index_are_validated(self):
        for key, value in (
            ("request_id", ""),
            ("session_id", "  "),
            ("episode_id", ""),
            ("inference_index", -1),
            ("inference_index", True),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ProtocolError) as raised:
                    Frame.from_wire_dict(_wire_dict(**{key: value}))
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)

    def test_payload_must_be_a_map(self):
        with self.assertRaises(ProtocolError) as raised:
            Frame.from_wire_dict(_wire_dict(payload=["not", "a", "map"]))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)

    def test_message_coordinates_follow_presence_matrix(self):
        valid = (
            _frame(
                message_type=MessageType.HELLO,
                episode_id=None,
                inference_index=None,
            ),
            _frame(
                message_type=MessageType.RESET,
                inference_index=None,
            ),
            _frame(),
            _frame(
                message_type=MessageType.TRIAL_END,
                inference_index=None,
            ),
        )
        self.assertEqual(len(valid), 4)

        invalid = (
            {
                "message_type": MessageType.HELLO,
                "episode_id": "unexpected",
                "inference_index": None,
            },
            {
                "message_type": MessageType.RESET,
                "episode_id": None,
                "inference_index": None,
            },
            {
                "message_type": MessageType.RESET,
                "inference_index": 0,
            },
            {
                "message_type": MessageType.INFER,
                "inference_index": None,
            },
            {
                "message_type": MessageType.TRIAL_END,
                "inference_index": 0,
            },
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ProtocolError) as raised:
                    _frame(**overrides)
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)


class PolicyCodecTest(unittest.TestCase):
    def test_numpy_payload_round_trip_preserves_shape_and_dtype(self):
        head = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        state = np.linspace(-1.0, 1.0, 14, dtype=np.float32)
        original = _frame(
            payload={
                "observation": {
                    "images": {"head": head},
                    "state": state,
                    "camera_count": np.int64(3),
                }
            }
        )

        decoded = decode_frame(encode_frame(original))
        decoded_obs = decoded.payload["observation"]
        np.testing.assert_array_equal(decoded_obs["images"]["head"], head)
        np.testing.assert_array_equal(decoded_obs["state"], state)
        self.assertEqual(decoded_obs["images"]["head"].dtype, np.uint8)
        self.assertEqual(decoded_obs["state"].dtype, np.float32)
        self.assertIsInstance(decoded_obs["camera_count"], np.int64)

    def test_numpy_wire_markers_match_kai0_codec(self):
        encoded = encode_frame(_frame(payload={"state": np.zeros(14, dtype=np.float32)}))
        unpacked = msgpack.unpackb(encoded, raw=False)
        encoded_state = unpacked["payload"]["state"]
        self.assertEqual(
            set(encoded_state),
            {b"__ndarray__", b"data", b"dtype", b"shape"},
        )
        self.assertIs(encoded_state[b"__ndarray__"], True)
        self.assertEqual(encoded_state[b"dtype"], "<f4")
        self.assertEqual(encoded_state[b"shape"], [14])

    def test_numpy_bytes_match_kai0_golden_fixture(self):
        head = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        state = np.linspace(-1.0, 1.0, 14, dtype=np.float32)
        frame = Frame(
            message_type=MessageType.INFER,
            request_id="golden-request",
            session_id="golden-session",
            episode_id="make_toast-golden",
            inference_index=0,
            payload={
                "observation": {
                    "images": {"head": head},
                    "state": state,
                    "camera_count": np.int64(3),
                }
            },
        )
        fixture_path = (
            Path(__file__).resolve().parents[1] / "protocol" / "robodojo_policy_v1" / "fixtures" / "numpy_frame_v1.hex"
        )
        kai0_bytes = bytes.fromhex(fixture_path.read_text(encoding="utf-8").strip())

        self.assertEqual(encode_frame(frame), kai0_bytes)
        decoded = decode_frame(kai0_bytes)
        np.testing.assert_array_equal(
            decoded.payload["observation"]["images"]["head"],
            head,
        )
        np.testing.assert_array_equal(
            decoded.payload["observation"]["state"],
            state,
        )
        self.assertEqual(
            decoded.payload["observation"]["camera_count"],
            np.int64(3),
        )

    def test_unsafe_numpy_dtypes_are_rejected(self):
        for value in (
            np.array([object()], dtype=object),
            np.array([1 + 2j], dtype=np.complex64),
            np.array([(1,)], dtype=[("value", "i4")]),
        ):
            with self.subTest(dtype=value.dtype):
                with self.assertRaises(ProtocolError) as raised:
                    encode_frame(_frame(payload={"value": value}))
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)

    def test_malformed_numpy_envelope_is_rejected(self):
        wire = _wire_dict(
            payload={
                "state": {
                    b"__ndarray__": True,
                    b"data": b"\x00",
                    b"dtype": "<f4",
                    b"shape": [14],
                }
            }
        )
        with self.assertRaises(ProtocolError) as raised:
            decode_frame(msgpack.packb(wire, use_bin_type=True))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)

    def test_numpy_shape_limits_reject_zero_size_bypass(self):
        wire = _wire_dict(
            payload={
                "state": {
                    b"__ndarray__": True,
                    b"data": b"",
                    b"dtype": "<f4",
                    b"shape": [10**12, 0],
                }
            }
        )
        with self.assertRaises(ProtocolError) as raised:
            decode_frame(msgpack.packb(wire, use_bin_type=True))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)

    def test_numpy_shape_limits_are_symmetric_on_encode(self):
        value = np.empty((10**12, 0), dtype=np.float32)
        with self.assertRaises(ProtocolError) as raised:
            encode_frame(_frame(payload={"value": value}))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)

    def test_numpy_scalar_envelope_rejects_collection_data(self):
        wire = _wire_dict(
            payload={
                "value": {
                    b"__npgeneric__": True,
                    b"data": [1, 2, 3],
                    b"dtype": "<i8",
                }
            }
        )
        with self.assertRaises(ProtocolError) as raised:
            decode_frame(msgpack.packb(wire, use_bin_type=True))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)

    def test_encoded_frame_size_limit_is_enforced_before_unpack(self):
        from unittest import mock

        with mock.patch(
            "src.eval_client.policy_runtime.codec.MAX_FRAME_BYTES",
            16,
        ):
            with self.assertRaises(ProtocolError) as raised:
                decode_frame(b"x" * 17)
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)
        self.assertEqual(MAX_FRAME_BYTES, 64 * 1024 * 1024)

    def test_encode_rejects_values_that_decoder_cannot_accept(self):
        from unittest import mock

        with mock.patch(
            "src.eval_client.policy_runtime.codec._MAX_STRING_BYTES",
            4,
        ):
            with self.assertRaises(ProtocolError) as raised:
                encode_frame(_frame(payload={"value": "12345"}))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)

        for value in ({1: "integer key"}, ("tuple", "changes", "type")):
            with self.subTest(value=value):
                with self.assertRaises(ProtocolError) as raised:
                    encode_frame(_frame(payload={"value": value}))
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)

    def test_non_map_frame_is_rejected(self):
        with self.assertRaises(ProtocolError) as raised:
            decode_frame(msgpack.packb(["not", "a", "frame"]))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_FRAME)


if __name__ == "__main__":
    unittest.main()
