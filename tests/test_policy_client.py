import asyncio
import importlib.util
from threading import Event, Thread
import unittest

import numpy as np

from src.eval_client.policy_runtime import (
    ActionValidationSpec,
    CanonicalActionChunk,
    ErrorCode,
    Frame,
    JointLimits,
    MessageType,
    ObservationValidationSpec,
    PolicyClient,
    PolicyClientState,
    PolicyClientStateError,
    PolicyConnectionError,
    PolicyExecutionProfile,
    PolicyProvenance,
    PolicyResponseError,
    PolicyTimeoutError,
    PolicyTransportError,
    ProtocolError,
    RemoteErrorPayload,
    RemotePolicyError,
    ResetPayload,
    ResetReason,
    TrialEndPayload,
    TrialStatus,
    build_hello_ack_payload,
    decode_frame,
    encode_frame,
    parse_observation,
)


def _profile():
    limits = JointLimits(
        lower=(-2.0,) * 6,
        upper=(2.0,) * 6,
    )
    return PolicyExecutionProfile(
        observation_spec=ObservationValidationSpec(
            head_image_shape=(2, 3, 3),
            left_wrist_image_shape=(2, 3, 3),
            right_wrist_image_shape=(2, 3, 3),
        ),
        action_spec=ActionValidationSpec(
            expected_horizon=2,
            expected_control_dt_s=0.04,
            left_arm_limits=limits,
            right_arm_limits=limits,
        ),
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


def _reset(seed=17):
    return ResetPayload(
        task_name="make_toast",
        simulator_seed=seed,
        policy_seed=seed + 1,
        layout_id=seed,
        layout_cycle=0,
        reason=ResetReason.EPISODE_START,
    )


def _trial_end():
    return TrialEndPayload(
        status=TrialStatus.SUCCESS,
        success=True,
        score=1.0,
        reason=None,
    )


def _observation(profile):
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    zeros = np.zeros(6, dtype=np.float32)
    gripper = np.ones(1, dtype=np.float32)
    return parse_observation(
        {
            "instruction": "make toast",
            "images": {
                "head": image,
                "left_wrist": image,
                "right_wrist": image,
            },
            "proprio": {
                "robot_schema": "arx_x5_dual_v1",
                "left_arm_joint_position": zeros,
                "left_gripper_open_fraction_commanded": gripper,
                "right_arm_joint_position": zeros,
                "right_gripper_open_fraction_commanded": gripper,
            },
        },
        spec=profile.observation_spec,
    )


def _action_payload(profile, *, horizon=None):
    horizon = profile.action_spec.expected_horizon if horizon is None else horizon
    arms = np.zeros((horizon, 6), dtype=np.float32)
    grippers = np.ones((horizon, 1), dtype=np.float32)
    return {
        "action": {
            "control_mode": "absolute_joint_position",
            "control_dt_s": profile.action_spec.expected_control_dt_s,
            "commands": {
                "left_arm_joint_position": arms,
                "left_gripper_open_fraction": grippers,
                "right_arm_joint_position": arms,
                "right_gripper_open_fraction": grippers,
            },
        },
    }


def _response(
    request,
    message_type,
    payload,
    **coordinate_overrides,
):
    coordinates = {
        "request_id": request.request_id,
        "session_id": request.session_id,
        "episode_id": request.episode_id,
        "inference_index": request.inference_index,
    }
    coordinates.update(coordinate_overrides)
    return encode_frame(
        Frame(
            message_type=message_type,
            payload=payload,
            **coordinates,
        )
    )


def _hello_handler(profile):
    return lambda request: _response(
        request,
        MessageType.HELLO_ACK,
        build_hello_ack_payload(profile, _provenance()),
    )


class _NoResponse:
    pass


class _RecvFailure:
    def __init__(self, error):
        self.error = error


class _SendFailure:
    def __init__(self, error):
        self.error = error


class ScriptedTransport:
    def __init__(self, handlers):
        self.handlers = list(handlers)
        self.sent = []
        self.closed = False
        self.loops = []
        self._pending = None

    async def send(self, message):
        self.loops.append(asyncio.get_running_loop())
        request = decode_frame(message)
        self.sent.append(request)
        if not self.handlers:
            raise AssertionError("unexpected client request")
        handler = self.handlers.pop(0)
        if isinstance(handler, _SendFailure):
            raise handler.error
        self._pending = handler(request)

    async def recv(self):
        self.loops.append(asyncio.get_running_loop())
        pending = self._pending
        self._pending = None
        if isinstance(pending, _RecvFailure):
            raise pending.error
        if isinstance(pending, _NoResponse):
            await asyncio.Future()
        return pending

    async def close(self):
        self.loops.append(asyncio.get_running_loop())
        self.closed = True


class _Factory:
    def __init__(self, transport):
        self.transport = transport
        self.calls = 0
        self.loops = []

    async def __call__(self, options):
        self.calls += 1
        self.loops.append(asyncio.get_running_loop())
        self.options = options
        return self.transport


class _FailOnceFactory(_Factory):
    async def __call__(self, options):
        self.calls += 1
        self.loops.append(asyncio.get_running_loop())
        self.options = options
        if self.calls == 1:
            raise ConnectionError("server is still starting")
        return self.transport


def _client(handlers, *, timeout=0.5):
    profile = _profile()
    transport = ScriptedTransport(handlers)
    factory = _Factory(transport)
    client = PolicyClient(
        "ws://policy.invalid:8000",
        profile=profile,
        transport_factory=factory,
        request_timeout_s=timeout,
        connect_timeout_s=timeout,
        close_timeout_s=timeout,
    )
    return client, transport, factory, profile


class PolicyClientHappyPathTest(unittest.TestCase):
    def test_complete_episode_uses_one_loop_and_strict_coordinates(self):
        profile = _profile()
        handlers = [
            _hello_handler(profile),
            lambda request: _response(
                request,
                MessageType.RESET_RESULT,
                {},
            ),
            lambda request: _response(
                request,
                MessageType.INFER_RESULT,
                _action_payload(profile),
            ),
            lambda request: _response(
                request,
                MessageType.TRIAL_END_ACK,
                {},
            ),
        ]
        client, transport, factory, profile = _client(handlers)
        try:
            provenance = client.connect()
            self.assertEqual(provenance, _provenance())
            self.assertEqual(client.state, PolicyClientState.READY)

            client.reset("episode-1", _reset())
            self.assertEqual(client.state, PolicyClientState.ACTIVE)
            self.assertEqual(client.active_episode_id, "episode-1")
            self.assertEqual(client.next_inference_index, 0)

            action = client.infer(_observation(profile))
            self.assertIsInstance(action, CanonicalActionChunk)
            self.assertEqual(action.horizon, 2)
            self.assertEqual(client.next_inference_index, 1)

            client.trial_end(_trial_end())
            self.assertEqual(client.state, PolicyClientState.READY)
            self.assertIsNone(client.active_episode_id)

            self.assertEqual(
                [frame.message_type for frame in transport.sent],
                [
                    MessageType.HELLO,
                    MessageType.RESET,
                    MessageType.INFER,
                    MessageType.TRIAL_END,
                ],
            )
            self.assertEqual(transport.sent[2].inference_index, 0)
            self.assertEqual(transport.sent[2].episode_id, "episode-1")
            request_ids = [frame.request_id for frame in transport.sent]
            self.assertEqual(len(request_ids), len(set(request_ids)))
            self.assertEqual(factory.calls, 1)
            self.assertEqual(factory.options.close_timeout_s, 0.5)
            self.assertEqual(len(set(factory.loops + transport.loops)), 1)
        finally:
            client.close()
        self.assertTrue(transport.closed)
        self.assertEqual(client.state, PolicyClientState.CLOSED)


class PolicyClientErrorHandlingTest(unittest.TestCase):
    def test_context_manager_connect_failure_closes_private_loop(self):
        profile = _profile()
        transport = ScriptedTransport([])
        client = PolicyClient(
            "ws://policy.invalid:8000",
            profile=profile,
            transport_factory=_FailOnceFactory(transport),
        )

        with self.assertRaises(PolicyConnectionError):
            with client:
                self.fail("a failed __enter__ must not yield the client")

        self.assertTrue(client._loop.is_closed())
        self.assertEqual(client.state, PolicyClientState.CLOSED)
        self.assertFalse(transport.closed)

    def test_invalid_hello_encoding_never_opens_a_transport(self):
        profile = _profile()
        transport = ScriptedTransport([])
        factory = _Factory(transport)
        client = PolicyClient(
            "ws://policy.invalid:8000",
            profile=profile,
            transport_factory=factory,
            request_id_factory=lambda: "x" * 1_000_001,
        )
        try:
            with self.assertRaises(ProtocolError):
                client.connect()
            self.assertEqual(client.state, PolicyClientState.NEW)
            self.assertIsNone(client.session_id)
            self.assertEqual(factory.calls, 0)
            self.assertFalse(transport.closed)
        finally:
            client.close()

    def test_blocking_close_inside_running_loop_does_not_corrupt_private_loop(self):
        client, _, _, _ = _client([])

        async def misuse_blocking_client():
            with self.assertRaisesRegex(
                RuntimeError,
                "use AsyncPolicyClient",
            ):
                client.close()

        asyncio.run(misuse_blocking_client())
        self.assertEqual(client.state, PolicyClientState.NEW)
        client.close()
        self.assertEqual(client.state, PolicyClientState.CLOSED)

    def test_close_is_idempotent_after_connection(self):
        profile = _profile()
        client, transport, _, _ = _client([_hello_handler(profile)])
        client.connect()

        client.close()
        client.close()

        self.assertTrue(transport.closed)
        self.assertEqual(client.state, PolicyClientState.CLOSED)

    def test_pre_transport_connect_failure_is_explicitly_retryable(self):
        profile = _profile()
        transport = ScriptedTransport([_hello_handler(profile)])
        factory = _FailOnceFactory(transport)
        client = PolicyClient(
            "ws://policy.invalid:8000",
            profile=profile,
            transport_factory=factory,
            request_timeout_s=0.5,
            connect_timeout_s=0.5,
            close_timeout_s=0.5,
        )
        try:
            with self.assertRaises(PolicyConnectionError) as raised:
                client.connect()
            self.assertFalse(raised.exception.episode_lost)
            self.assertEqual(client.state, PolicyClientState.NEW)
            self.assertIsNone(client.session_id)
            self.assertIsNone(client.snapshot.pending)
            self.assertEqual(factory.calls, 1)
            self.assertEqual(transport.sent, [])

            self.assertEqual(client.connect(), _provenance())
            self.assertEqual(client.state, PolicyClientState.READY)
            self.assertEqual(factory.calls, 2)
            self.assertEqual(len(transport.sent), 1)
            self.assertEqual(client.snapshot.seen_request_count, 2)
        finally:
            client.close()

    def test_invalid_payload_is_nonterminal_and_requires_fresh_ids(self):
        profile = _profile()
        invalid_payload = RemoteErrorPayload(
            code=ErrorCode.INVALID_PAYLOAD,
            message="bad payload",
            details={"kind": "invalid_shape"},
            retryable=False,
        )
        handlers = [
            _hello_handler(profile),
            lambda request: _response(
                request,
                MessageType.ERROR,
                invalid_payload.to_payload(),
            ),
            lambda request: _response(
                request,
                MessageType.RESET_RESULT,
                {},
            ),
            lambda request: _response(
                request,
                MessageType.ERROR,
                invalid_payload.to_payload(),
            ),
            lambda request: _response(
                request,
                MessageType.INFER_RESULT,
                _action_payload(profile),
            ),
            lambda request: _response(
                request,
                MessageType.TRIAL_END_ACK,
                {},
            ),
        ]
        client, transport, _, profile = _client(handlers)
        try:
            client.connect()
            with self.assertRaises(RemotePolicyError) as raised:
                client.reset("episode-rejected", _reset())
            self.assertFalse(raised.exception.terminal)
            self.assertFalse(raised.exception.episode_lost)
            self.assertEqual(client.state, PolicyClientState.READY)
            self.assertFalse(transport.closed)

            with self.assertRaises(PolicyClientStateError):
                client.reset("episode-rejected", _reset())
            self.assertEqual(len(transport.sent), 2)

            client.reset("episode-accepted", _reset(18))
            with self.assertRaises(RemotePolicyError) as raised:
                client.infer(_observation(profile))
            self.assertFalse(raised.exception.terminal)
            self.assertEqual(client.state, PolicyClientState.ACTIVE)
            self.assertEqual(client.next_inference_index, 0)

            action = client.infer(_observation(profile))
            self.assertEqual(action.horizon, 2)
            infer_requests = [frame for frame in transport.sent if frame.message_type == MessageType.INFER]
            self.assertEqual(
                [frame.inference_index for frame in infer_requests],
                [0, 0],
            )
            self.assertNotEqual(
                infer_requests[0].request_id,
                infer_requests[1].request_id,
            )
            client.trial_end(_trial_end())
        finally:
            client.close()

    def test_trial_end_invalid_payload_keeps_episode_active_for_fresh_request(self):
        profile = _profile()
        invalid_payload = RemoteErrorPayload(
            code=ErrorCode.INVALID_PAYLOAD,
            message="bad outcome",
            details={"field": "status"},
            retryable=False,
        )
        handlers = [
            _hello_handler(profile),
            lambda request: _response(
                request,
                MessageType.RESET_RESULT,
                {},
            ),
            lambda request: _response(
                request,
                MessageType.ERROR,
                invalid_payload.to_payload(),
            ),
            lambda request: _response(
                request,
                MessageType.TRIAL_END_ACK,
                {},
            ),
        ]
        client, transport, _, _ = _client(handlers)
        try:
            client.connect()
            client.reset("episode-1", _reset())

            with self.assertRaises(RemotePolicyError) as raised:
                client.trial_end(_trial_end())
            self.assertFalse(raised.exception.terminal)
            self.assertFalse(raised.exception.episode_lost)
            self.assertEqual(client.state, PolicyClientState.ACTIVE)
            self.assertEqual(client.active_episode_id, "episode-1")

            client.trial_end(_trial_end())
            self.assertEqual(client.state, PolicyClientState.READY)
            requests = [frame for frame in transport.sent if frame.message_type == MessageType.TRIAL_END]
            self.assertEqual(len(requests), 2)
            self.assertNotEqual(requests[0].request_id, requests[1].request_id)
        finally:
            client.close()

    def test_terminal_remote_error_closes_and_fences_the_session(self):
        profile = _profile()
        terminal_error = RemoteErrorPayload(
            code=ErrorCode.INFER_FAILED,
            message="model failed",
            details={},
            retryable=False,
        )
        handlers = [
            _hello_handler(profile),
            lambda request: _response(
                request,
                MessageType.RESET_RESULT,
                {},
            ),
            lambda request: _response(
                request,
                MessageType.ERROR,
                terminal_error.to_payload(),
            ),
        ]
        client, transport, factory, profile = _client(handlers)
        try:
            client.connect()
            client.reset("episode-1", _reset())
            with self.assertRaises(RemotePolicyError) as raised:
                client.infer(_observation(profile))
            self.assertTrue(raised.exception.terminal)
            self.assertTrue(raised.exception.episode_lost)
            self.assertEqual(raised.exception.code, ErrorCode.INFER_FAILED)
            self.assertEqual(client.state, PolicyClientState.LOST)
            self.assertTrue(client.episode_lost)
            self.assertTrue(transport.closed)
            self.assertIsNone(client.snapshot.pending)

            sent_count = len(transport.sent)
            with self.assertRaises(PolicyClientStateError):
                client.infer(_observation(profile))
            self.assertEqual(len(transport.sent), sent_count)
            self.assertEqual(factory.calls, 1)
        finally:
            client.close()
        self.assertEqual(client.state, PolicyClientState.LOST)

    def test_invalid_payload_during_hello_is_terminal(self):
        error = RemoteErrorPayload(
            code=ErrorCode.INVALID_PAYLOAD,
            message="unsupported profile",
            details={},
            retryable=False,
        )
        handlers = [
            lambda request: _response(
                request,
                MessageType.ERROR,
                error.to_payload(),
            )
        ]
        client, transport, _, _ = _client(handlers)
        try:
            with self.assertRaises(RemotePolicyError) as raised:
                client.connect()
            self.assertTrue(raised.exception.terminal)
            self.assertFalse(raised.exception.episode_lost)
            self.assertEqual(client.state, PolicyClientState.LOST)
            self.assertTrue(transport.closed)
        finally:
            client.close()

    def test_invalid_action_is_infer_failed_and_loses_episode(self):
        profile = _profile()
        handlers = [
            _hello_handler(profile),
            lambda request: _response(
                request,
                MessageType.RESET_RESULT,
                {},
            ),
            lambda request: _response(
                request,
                MessageType.INFER_RESULT,
                _action_payload(profile, horizon=1),
            ),
        ]
        client, transport, _, profile = _client(handlers)
        try:
            client.connect()
            client.reset("episode-1", _reset())
            with self.assertRaises(PolicyResponseError) as raised:
                client.infer(_observation(profile))
            self.assertEqual(raised.exception.code, ErrorCode.INFER_FAILED)
            self.assertTrue(raised.exception.episode_lost)
            self.assertEqual(client.state, PolicyClientState.LOST)
            self.assertTrue(transport.closed)
        finally:
            client.close()

    def test_timeout_send_recv_failures_and_text_frames_are_terminal(self):
        profile = _profile()
        cases = {
            "timeout": lambda: _NoResponse(),
            "send": lambda: _SendFailure(ConnectionError("send failed")),
            "recv": lambda: _RecvFailure(EOFError("closed")),
            "text": lambda: "not binary",
        }
        for name, failure_factory in cases.items():
            with self.subTest(name=name):
                failure = failure_factory()
                handlers = [
                    _hello_handler(profile),
                    failure if isinstance(failure, _SendFailure) else lambda request, failure=failure: failure,
                ]
                client, transport, factory, _ = _client(
                    handlers,
                    timeout=0.01,
                )
                try:
                    client.connect()
                    expected_exception = {
                        "timeout": PolicyTimeoutError,
                        "send": PolicyTransportError,
                        "recv": PolicyTransportError,
                        "text": PolicyResponseError,
                    }[name]
                    with self.assertRaises(expected_exception) as raised:
                        client.reset("episode-1", _reset())
                    if name == "timeout":
                        self.assertEqual(
                            raised.exception.code,
                            ErrorCode.TIMEOUT,
                        )
                    elif name == "text":
                        self.assertEqual(
                            raised.exception.code,
                            ErrorCode.INVALID_FRAME,
                        )
                    else:
                        self.assertEqual(
                            raised.exception.code,
                            ErrorCode.EPISODE_LOST,
                        )
                    self.assertTrue(raised.exception.episode_lost)
                    self.assertEqual(client.state, PolicyClientState.LOST)
                    self.assertTrue(transport.closed)
                    self.assertEqual(
                        client.snapshot.pending.message_type,
                        MessageType.RESET,
                    )
                    self.assertEqual(factory.calls, 1)
                    reset_requests = [frame for frame in transport.sent if frame.message_type == MessageType.RESET]
                    self.assertEqual(len(reset_requests), 1)
                finally:
                    client.close()

    def test_all_response_coordinates_are_checked(self):
        profile = _profile()
        hello_cases = {
            "request_id": {"request_id": "foreign-request"},
            "session_id": {"session_id": "foreign-session"},
        }
        for name, overrides in hello_cases.items():
            with self.subTest(name=name):
                handlers = [
                    lambda request, overrides=overrides: _response(
                        request,
                        MessageType.HELLO_ACK,
                        build_hello_ack_payload(profile, _provenance()),
                        **overrides,
                    )
                ]
                client, transport, _, _ = _client(handlers)
                try:
                    with self.assertRaises(PolicyResponseError) as raised:
                        client.connect()
                    self.assertEqual(
                        raised.exception.code,
                        ErrorCode.INVALID_FRAME,
                    )
                    self.assertEqual(client.state, PolicyClientState.LOST)
                    self.assertTrue(transport.closed)
                finally:
                    client.close()

        episode_handlers = [
            _hello_handler(profile),
            lambda request: _response(
                request,
                MessageType.RESET_RESULT,
                {},
                episode_id="foreign-episode",
            ),
        ]
        client, transport, _, _ = _client(episode_handlers)
        try:
            client.connect()
            with self.assertRaises(PolicyResponseError):
                client.reset("episode-1", _reset())
            self.assertEqual(client.state, PolicyClientState.LOST)
            self.assertTrue(transport.closed)
        finally:
            client.close()

        inference_handlers = [
            _hello_handler(profile),
            lambda request: _response(
                request,
                MessageType.RESET_RESULT,
                {},
            ),
            lambda request: _response(
                request,
                MessageType.INFER_RESULT,
                _action_payload(profile),
                inference_index=1,
            ),
        ]
        client, transport, _, profile = _client(inference_handlers)
        try:
            client.connect()
            client.reset("episode-1", _reset())
            with self.assertRaises(PolicyResponseError):
                client.infer(_observation(profile))
            self.assertEqual(client.state, PolicyClientState.LOST)
            self.assertTrue(transport.closed)
        finally:
            client.close()

    def test_unexpected_success_type_is_terminal(self):
        handlers = [
            lambda request: _response(
                request,
                MessageType.HELLO,
                {},
            )
        ]
        client, transport, _, _ = _client(handlers)
        try:
            with self.assertRaises(PolicyResponseError):
                client.connect()
            self.assertEqual(client.state, PolicyClientState.LOST)
            self.assertTrue(transport.closed)
        finally:
            client.close()


class _LoopbackPolicyServer:
    def __init__(self, profile):
        self.profile = profile
        self.ready = Event()
        self.thread = Thread(target=self._thread_main, daemon=True)
        self.loop = None
        self.stop_future = None
        self.port = None
        self.error = None

    def start(self):
        self.thread.start()
        if not self.ready.wait(timeout=5):
            raise TimeoutError("loopback WebSocket server did not start")
        if self.error is not None:
            raise self.error

    def close(self):
        if self.loop is not None and self.stop_future is not None:
            self.loop.call_soon_threadsafe(self._set_stop_result)
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise TimeoutError("loopback WebSocket server did not stop")
        if self.error is not None:
            raise self.error

    def _set_stop_result(self):
        if not self.stop_future.done():
            self.stop_future.set_result(None)

    def _thread_main(self):
        try:
            asyncio.run(self._run())
        except BaseException as error:
            self.error = error
            self.ready.set()

    async def _run(self):
        import websockets

        self.loop = asyncio.get_running_loop()
        self.stop_future = self.loop.create_future()

        async def handler(websocket, *unused):
            request = decode_frame(await websocket.recv())
            await websocket.send(
                _response(
                    request,
                    MessageType.HELLO_ACK,
                    build_hello_ack_payload(self.profile, _provenance()),
                )
            )
            await websocket.wait_closed()

        server = await websockets.serve(
            handler,
            "127.0.0.1",
            0,
            compression=None,
        )
        self.port = server.sockets[0].getsockname()[1]
        self.ready.set()
        try:
            await self.stop_future
        finally:
            server.close()
            await server.wait_closed()


@unittest.skipUnless(
    importlib.util.find_spec("websockets"),
    "websockets isn't installed",
)
class ProductionWebSocketTransportTest(unittest.TestCase):
    def test_loopback_hello_uses_supported_production_kwargs(self):
        profile = _profile()
        server = _LoopbackPolicyServer(profile)
        server.start()
        client = PolicyClient(
            f"ws://127.0.0.1:{server.port}",
            profile=profile,
            connect_timeout_s=2,
            request_timeout_s=2,
            close_timeout_s=2,
        )
        try:
            self.assertEqual(client.connect(), _provenance())
            self.assertEqual(client.state, PolicyClientState.READY)
        finally:
            client.close()
            server.close()


if __name__ == "__main__":
    unittest.main()
