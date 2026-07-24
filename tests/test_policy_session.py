from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import unittest

from src.eval_client.policy_runtime import (
    ErrorCode,
    Frame,
    MessageType,
    OperationToken,
    PolicySession,
    ProtocolError,
    SessionInvariantError,
    SessionPhase,
)


def _request(message_type, request_id, **overrides):
    coordinates = {
        MessageType.HELLO: (None, None),
        MessageType.RESET: ("episode-1", None),
        MessageType.INFER: ("episode-1", 0),
        MessageType.TRIAL_END: ("episode-1", None),
    }
    episode_id, inference_index = coordinates[message_type]
    values = {
        "message_type": message_type,
        "request_id": request_id,
        "session_id": "session-1",
        "episode_id": episode_id,
        "inference_index": inference_index,
        "payload": {},
    }
    values.update(overrides)
    return Frame(**values)


def _start_active_session():
    session = PolicySession()
    session.complete(session.begin(_request(MessageType.HELLO, "hello-1")))
    session.complete(session.begin(_request(MessageType.RESET, "reset-1")))
    return session


class PolicySessionLifecycleTest(unittest.TestCase):
    def test_happy_path_advances_only_after_operation_completion(self):
        session = PolicySession()
        self.assertEqual(session.snapshot.phase, SessionPhase.AWAITING_HELLO)

        hello = session.begin(_request(MessageType.HELLO, "hello-1"))
        self.assertEqual(session.snapshot.phase, SessionPhase.AWAITING_HELLO)
        self.assertEqual(session.complete(hello).phase, SessionPhase.READY)

        reset = session.begin(_request(MessageType.RESET, "reset-1"))
        self.assertEqual(session.snapshot.phase, SessionPhase.READY)
        self.assertIsNone(session.snapshot.active_episode_id)
        self.assertEqual(session.complete(reset).phase, SessionPhase.ACTIVE)
        self.assertEqual(session.snapshot.active_episode_id, "episode-1")
        self.assertEqual(session.snapshot.next_inference_index, 0)

        infer_0 = session.begin(_request(MessageType.INFER, "infer-0"))
        self.assertEqual(session.snapshot.next_inference_index, 0)
        self.assertEqual(session.complete(infer_0).phase, SessionPhase.ACTIVE)
        self.assertEqual(session.snapshot.next_inference_index, 1)

        infer_1 = session.begin(
            _request(MessageType.INFER, "infer-1", inference_index=1),
        )
        session.complete(infer_1)
        self.assertEqual(session.snapshot.next_inference_index, 2)

        trial_end = session.begin(_request(MessageType.TRIAL_END, "end-1"))
        self.assertEqual(session.complete(trial_end).phase, SessionPhase.READY)
        self.assertIsNone(session.snapshot.active_episode_id)
        self.assertIsNone(session.snapshot.next_inference_index)

    def test_first_non_hello_does_not_bind_or_burn_identifiers(self):
        session = PolicySession()
        with self.assertRaises(ProtocolError) as raised:
            session.begin(_request(MessageType.RESET, "shared-id"))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)
        self.assertIsNone(session.snapshot.session_id)
        self.assertEqual(session.snapshot.seen_request_count, 0)

        hello = session.begin(_request(MessageType.HELLO, "shared-id"))
        session.complete(hello)
        self.assertEqual(session.snapshot.session_id, "session-1")

    def test_bound_request_during_pending_hello_is_burned(self):
        session = PolicySession()
        hello = session.begin(_request(MessageType.HELLO, "hello-1"))

        with self.assertRaises(ProtocolError) as raised:
            session.begin(_request(MessageType.RESET, "early-reset"))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)

        session.complete(hello)
        with self.assertRaises(ProtocolError) as raised:
            session.begin(_request(MessageType.RESET, "early-reset"))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)

        reset = session.begin(_request(MessageType.RESET, "reset-1"))
        session.complete(reset)

    def test_wrong_session_during_pending_hello_does_not_burn_id(self):
        session = PolicySession()
        hello = session.begin(_request(MessageType.HELLO, "hello-1"))

        with self.assertRaises(ProtocolError) as raised:
            session.begin(
                _request(
                    MessageType.RESET,
                    "reset-1",
                    session_id="some-other-session",
                )
            )
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)

        session.complete(hello)
        reset = session.begin(_request(MessageType.RESET, "reset-1"))
        session.complete(reset)

    def test_duplicate_hello_and_response_frames_are_rejected(self):
        session = PolicySession()
        session.complete(session.begin(_request(MessageType.HELLO, "hello-1")))

        with self.assertRaises(ProtocolError) as raised:
            session.begin(_request(MessageType.HELLO, "hello-2"))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)

        response = Frame(
            message_type=MessageType.RESET_RESULT,
            request_id="response-as-request",
            session_id="session-1",
            episode_id="episode-1",
            payload={},
        )
        with self.assertRaises(ProtocolError) as raised:
            session.begin(response)
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)

    def test_wrong_session_does_not_consume_request_id(self):
        session = PolicySession()
        session.complete(session.begin(_request(MessageType.HELLO, "hello-1")))

        with self.assertRaises(ProtocolError) as raised:
            session.begin(
                _request(
                    MessageType.RESET,
                    "reset-1",
                    session_id="some-other-session",
                )
            )
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)

        reset = session.begin(_request(MessageType.RESET, "reset-1"))
        session.complete(reset)
        self.assertEqual(session.snapshot.phase, SessionPhase.ACTIVE)

    def test_bound_invalid_request_id_is_burned(self):
        session = _start_active_session()

        with self.assertRaises(ProtocolError) as raised:
            session.begin(
                _request(
                    MessageType.INFER,
                    "bad-index",
                    inference_index=3,
                )
            )
        self.assertEqual(raised.exception.code, ErrorCode.INFERENCE_INDEX_MISMATCH)
        self.assertEqual(session.snapshot.next_inference_index, 0)

        with self.assertRaises(ProtocolError) as raised:
            session.begin(_request(MessageType.INFER, "bad-index"))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)

        infer = session.begin(_request(MessageType.INFER, "infer-0"))
        session.complete(infer)
        self.assertEqual(session.snapshot.next_inference_index, 1)

    def test_episode_mismatch_has_priority_over_inference_index(self):
        session = _start_active_session()
        with self.assertRaises(ProtocolError) as raised:
            session.begin(
                _request(
                    MessageType.INFER,
                    "wrong-episode",
                    episode_id="episode-other",
                    inference_index=99,
                )
            )
        self.assertEqual(raised.exception.code, ErrorCode.EPISODE_MISMATCH)

    def test_state_error_has_priority_when_no_episode_is_active(self):
        session = PolicySession()
        session.complete(session.begin(_request(MessageType.HELLO, "hello-1")))
        with self.assertRaises(ProtocolError) as raised:
            session.begin(_request(MessageType.INFER, "infer-too-early"))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)

    def test_episode_id_cannot_be_reused_within_session(self):
        session = _start_active_session()
        session.complete(session.begin(_request(MessageType.TRIAL_END, "end-1")))

        with self.assertRaises(ProtocolError) as raised:
            session.begin(_request(MessageType.RESET, "reset-2"))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)

        reset = session.begin(
            _request(
                MessageType.RESET,
                "reset-3",
                episode_id="episode-2",
            )
        )
        session.complete(reset)
        self.assertEqual(session.snapshot.active_episode_id, "episode-2")

    def test_only_one_concurrent_request_is_accepted(self):
        session = _start_active_session()
        barrier = Barrier(3)

        def begin_infer(request_id):
            barrier.wait()
            try:
                return session.begin(_request(MessageType.INFER, request_id))
            except ProtocolError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(begin_infer, "infer-a"),
                executor.submit(begin_infer, "infer-b"),
            ]
            barrier.wait()
            results = [future.result() for future in futures]

        tokens = [result for result in results if isinstance(result, OperationToken)]
        errors = [result for result in results if isinstance(result, ProtocolError)]
        self.assertEqual(len(tokens), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].code, ErrorCode.INVALID_STATE)
        session.complete(tokens[0])
        self.assertEqual(session.snapshot.next_inference_index, 1)

    def test_pending_infer_rejects_lifecycle_requests_without_replacing_it(self):
        for message_type, request_id in (
            (MessageType.RESET, "reset-too-early"),
            (MessageType.TRIAL_END, "end-too-early"),
        ):
            with self.subTest(message_type=message_type):
                session = _start_active_session()
                infer = session.begin(_request(MessageType.INFER, "infer-0"))

                with self.assertRaises(ProtocolError) as raised:
                    session.begin(_request(message_type, request_id))
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)
                self.assertIs(session.snapshot.pending, infer)
                self.assertEqual(session.snapshot.next_inference_index, 0)

                session.complete(infer)
                self.assertEqual(session.snapshot.next_inference_index, 1)
                with self.assertRaises(ProtocolError) as raised:
                    session.begin(
                        _request(
                            MessageType.INFER,
                            request_id,
                            inference_index=1,
                        )
                    )
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)


class PolicySessionDisconnectTest(unittest.TestCase):
    def test_disconnect_without_pending_operation_closes_immediately(self):
        new_session = PolicySession()
        disconnected = new_session.disconnect()
        self.assertEqual(disconnected.phase, SessionPhase.CLOSED)
        self.assertFalse(disconnected.waiting_for_operation)
        self.assertFalse(disconnected.episode_lost)

        ready_session = PolicySession()
        ready_session.complete(
            ready_session.begin(_request(MessageType.HELLO, "hello-1")),
        )
        disconnected = ready_session.disconnect()
        self.assertEqual(disconnected.phase, SessionPhase.CLOSED)
        self.assertFalse(disconnected.episode_lost)

        active_session = _start_active_session()
        disconnected = active_session.disconnect()
        self.assertEqual(disconnected.phase, SessionPhase.CLOSED)
        self.assertTrue(disconnected.episode_lost)

    def test_disconnect_during_infer_drains_and_discards_worker_result(self):
        session = _start_active_session()
        infer = session.begin(_request(MessageType.INFER, "infer-0"))

        disconnected = session.disconnect()
        self.assertEqual(disconnected.phase, SessionPhase.DRAINING)
        self.assertTrue(disconnected.waiting_for_operation)
        self.assertTrue(disconnected.episode_lost)
        self.assertEqual(session.disconnect(), disconnected)

        settled = session.complete(infer)
        self.assertEqual(settled.phase, SessionPhase.CLOSED)
        self.assertTrue(settled.operation_obsolete)
        self.assertFalse(settled.reply_allowed)
        self.assertTrue(settled.episode_lost)
        self.assertEqual(session.snapshot.next_inference_index, 0)

    def test_disconnect_during_reset_marks_candidate_episode_lost(self):
        session = PolicySession()
        session.complete(session.begin(_request(MessageType.HELLO, "hello-1")))
        reset = session.begin(_request(MessageType.RESET, "reset-1"))

        disconnected = session.disconnect()
        self.assertEqual(disconnected.phase, SessionPhase.DRAINING)
        self.assertTrue(disconnected.episode_lost)

        settled = session.fail(reset)
        self.assertTrue(settled.operation_obsolete)
        self.assertFalse(settled.reply_allowed)
        self.assertEqual(settled.phase, SessionPhase.CLOSED)

    def test_disconnect_during_hello_has_no_lost_episode(self):
        session = PolicySession()
        hello = session.begin(_request(MessageType.HELLO, "hello-1"))
        disconnected = session.disconnect()
        self.assertTrue(disconnected.waiting_for_operation)
        self.assertFalse(disconnected.episode_lost)

        settled = session.complete(hello)
        self.assertTrue(settled.operation_obsolete)
        self.assertFalse(settled.reply_allowed)
        self.assertFalse(settled.episode_lost)

    def test_backend_failure_is_correlatable_but_terminal(self):
        session = _start_active_session()
        infer = session.begin(_request(MessageType.INFER, "infer-0"))

        failed = session.fail(infer)
        self.assertEqual(failed.phase, SessionPhase.TERMINATING)
        self.assertTrue(failed.reply_allowed)
        self.assertTrue(failed.close_after_reply)
        self.assertFalse(failed.operation_obsolete)
        self.assertTrue(failed.episode_lost)

        with self.assertRaises(ProtocolError) as raised:
            session.begin(_request(MessageType.INFER, "infer-1"))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)
        self.assertEqual(session.disconnect().phase, SessionPhase.CLOSED)

    def test_disconnect_during_trial_end_discards_ack_and_never_returns_ready(self):
        session = _start_active_session()
        trial_end = session.begin(_request(MessageType.TRIAL_END, "end-1"))

        disconnected = session.disconnect()
        self.assertEqual(disconnected.phase, SessionPhase.DRAINING)
        self.assertTrue(disconnected.episode_lost)

        with self.assertRaises(ProtocolError) as raised:
            session.begin(_request(MessageType.INFER, "infer-too-late"))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)
        self.assertIs(session.snapshot.pending, trial_end)

        settled = session.complete(trial_end)
        self.assertEqual(settled.phase, SessionPhase.CLOSED)
        self.assertTrue(settled.operation_obsolete)
        self.assertNotEqual(settled.phase, SessionPhase.READY)

    def test_backend_failure_semantics_for_all_lifecycle_operations(self):
        cases = []

        hello_session = PolicySession()
        cases.append(
            (
                MessageType.HELLO,
                hello_session,
                hello_session.begin(_request(MessageType.HELLO, "hello-1")),
                False,
            )
        )

        reset_session = PolicySession()
        reset_session.complete(
            reset_session.begin(_request(MessageType.HELLO, "hello-1")),
        )
        cases.append(
            (
                MessageType.RESET,
                reset_session,
                reset_session.begin(_request(MessageType.RESET, "reset-1")),
                True,
            )
        )

        infer_session = _start_active_session()
        cases.append(
            (
                MessageType.INFER,
                infer_session,
                infer_session.begin(_request(MessageType.INFER, "infer-0")),
                True,
            )
        )

        end_session = _start_active_session()
        cases.append(
            (
                MessageType.TRIAL_END,
                end_session,
                end_session.begin(_request(MessageType.TRIAL_END, "end-1")),
                True,
            )
        )

        for message_type, session, token, episode_lost in cases:
            with self.subTest(message_type=message_type):
                failed = session.fail(token)
                self.assertEqual(failed.phase, SessionPhase.TERMINATING)
                self.assertTrue(failed.reply_allowed)
                self.assertTrue(failed.close_after_reply)
                self.assertFalse(failed.operation_obsolete)
                self.assertEqual(failed.episode_lost, episode_lost)
                self.assertEqual(session.disconnect().phase, SessionPhase.CLOSED)

    def test_stale_or_foreign_operation_token_is_a_local_invariant_error(self):
        session = PolicySession()
        hello = session.begin(_request(MessageType.HELLO, "hello-1"))
        foreign_session = PolicySession()
        foreign = foreign_session.begin(_request(MessageType.HELLO, "hello-1"))
        self.assertEqual(foreign, hello)

        with self.assertRaises(SessionInvariantError):
            session.complete(foreign)
        session.complete(hello)
        foreign_session.complete(foreign)
        with self.assertRaises(SessionInvariantError):
            session.complete(hello)


class PolicySessionRejectTest(unittest.TestCase):
    def test_rejected_hello_replies_once_then_closes(self):
        session = PolicySession()
        hello = session.begin(_request(MessageType.HELLO, "hello-1"))

        rejected = session.reject(hello)
        self.assertEqual(rejected.phase, SessionPhase.TERMINATING)
        self.assertTrue(rejected.reply_allowed)
        self.assertTrue(rejected.close_after_reply)
        self.assertFalse(rejected.episode_lost)
        self.assertEqual(session.disconnect().phase, SessionPhase.CLOSED)

    def test_rejected_reset_preserves_ready_but_burns_episode_id(self):
        session = PolicySession()
        session.complete(session.begin(_request(MessageType.HELLO, "hello-1")))
        reset = session.begin(_request(MessageType.RESET, "reset-invalid"))
        self.assertEqual(session.snapshot.seen_request_count, 2)
        self.assertEqual(session.snapshot.seen_episode_count, 1)
        self.assertIsNone(session.snapshot.active_episode_id)

        rejected = session.reject(reset)
        self.assertEqual(rejected.phase, SessionPhase.READY)
        self.assertFalse(rejected.close_after_reply)
        self.assertEqual(session.snapshot.seen_request_count, 2)
        self.assertEqual(session.snapshot.seen_episode_count, 1)

        with self.assertRaises(ProtocolError) as raised:
            session.begin(_request(MessageType.RESET, "reset-reuse"))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)

        with self.assertRaises(ProtocolError) as raised:
            session.begin(
                _request(
                    MessageType.RESET,
                    "reset-invalid",
                    episode_id="episode-2",
                )
            )
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_STATE)
        self.assertEqual(session.snapshot.seen_episode_count, 1)

        corrected = session.begin(
            _request(
                MessageType.RESET,
                "reset-corrected",
                episode_id="episode-2",
            )
        )
        session.complete(corrected)
        self.assertEqual(session.snapshot.active_episode_id, "episode-2")

    def test_rejected_infer_and_trial_end_do_not_advance_lifecycle(self):
        session = _start_active_session()

        infer = session.begin(_request(MessageType.INFER, "infer-invalid"))
        rejected = session.reject(infer)
        self.assertEqual(rejected.phase, SessionPhase.ACTIVE)
        self.assertEqual(session.snapshot.next_inference_index, 0)

        trial_end = session.begin(_request(MessageType.TRIAL_END, "end-invalid"))
        rejected = session.reject(trial_end)
        self.assertEqual(rejected.phase, SessionPhase.ACTIVE)
        self.assertEqual(session.snapshot.active_episode_id, "episode-1")

        corrected = session.begin(_request(MessageType.INFER, "infer-corrected"))
        session.complete(corrected)
        self.assertEqual(session.snapshot.next_inference_index, 1)


if __name__ == "__main__":
    unittest.main()
