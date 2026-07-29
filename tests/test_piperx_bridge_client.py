from __future__ import annotations

import socket
import threading
import time
import unittest

from src.eval_client.piperx_bridge_client import (
    EMBODIMENT_PROFILE,
    PROTOCOL,
    PiperXBridgeClient,
    PiperXBridgeProtocolError,
    PiperXBridgeTransportError,
    SimArmTarget,
    SimTargets,
    encode_frame,
    receive_frame,
)

_TOPOLOGY = "policy_sim_to_follower_to_leader_manual_leader_joint_fanout"


def _sim(x: float = 0.0) -> SimTargets:
    return SimTargets(
        left=SimArmTarget((x, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0), 0.7),
        right=SimArmTarget((-x, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0), 0.8),
    )


def _arm_sample(x: float, *, sampled_ns: int) -> dict[str, object]:
    return {
        "pose": [x, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        "gripper_m": 0.04 if x >= 0 else 0.05,
        "sampled_monotonic_ns": sampled_ns,
    }


def _manual_sample(sample_id: int = 1) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "left": _arm_sample(0.1, sampled_ns=1001),
        "right": _arm_sample(-0.1, sampled_ns=1002),
    }


def _manual_resolution(sample_id: int, decision: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "decision": decision,
        "follower_commanded": decision == "commit",
    }


def _response(
    request: dict[str, object],
    *,
    generation: int | None = None,
    mode: str = "policy",
    edge: str | None = None,
    terminal: str | None = None,
    seq: int | None = None,
    transition: str | None = None,
    manual_sample: dict[str, object] | None = None,
    manual_resolution: dict[str, object] | None = None,
    motion_accepted: bool | None = None,
    leader_actuation_mode: str | None = None,
    follower_actuation_mode: str | None = None,
) -> dict[str, object]:
    request_type = request["type"]
    if leader_actuation_mode is None:
        leader_actuation_mode = "native_leader" if mode == "intervention" else "output_follow"
    if follower_actuation_mode is None:
        if transition is not None or request_type == "manual_sample":
            follower_actuation_mode = "hold"
        elif manual_resolution is not None and manual_resolution["decision"] == "commit":
            follower_actuation_mode = "leader_follow"
        else:
            follower_actuation_mode = "sim_follow"
    if motion_accepted is None:
        motion_accepted = request_type not in {"heartbeat", "manual_sample"} and transition is None

    health = {
        "ok": True,
        "control_topology": _TOPOLOGY,
        "embodiment_profile": EMBODIMENT_PROFILE,
        "leader_actuation_mode": leader_actuation_mode,
        "follower_actuation_mode": follower_actuation_mode,
    }
    return {
        "protocol": PROTOCOL,
        "type": request_type,
        "session_id": request["session_id"],
        "episode_id": request["episode_id"],
        "generation": request["generation"] if generation is None else generation,
        "seq": request["seq"] if seq is None else seq,
        "sent_monotonic_ns": time.monotonic_ns(),
        "deadline_monotonic_ns": request["deadline_monotonic_ns"],
        "payload": {
            "mode": mode,
            "edge": edge,
            "transition": transition,
            "terminal_request": terminal,
            "manual_sample": manual_sample,
            "manual_resolution": manual_resolution,
            "motion_accepted": motion_accepted,
            "embodiment_profile": EMBODIMENT_PROFILE,
            "control_topology": _TOPOLOGY,
            "leader_actuation_mode": leader_actuation_mode,
            "follower_actuation_mode": follower_actuation_mode,
            "health": health,
            "diagnostics": {"request_ok": True},
        },
    }


class _Server:
    def __init__(self, handler):
        self.handler = handler
        self.requests: list[dict[str, object]] = []
        self.error: BaseException | None = None
        self.ready = threading.Event()
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self):
        try:
            self.listener.listen(1)
            self.ready.set()
            connection, _ = self.listener.accept()
            with connection:
                self.handler(self, connection)
        except BaseException as exc:  # pragma: no cover - surfaced by join()
            self.error = exc
        finally:
            self.listener.close()

    def start(self):
        self.thread.start()
        if not self.ready.wait(1):
            raise AssertionError("fake bridge server did not start")
        return self

    def join(self):
        self.thread.join(2)
        if self.thread.is_alive():
            self.listener.close()
            raise AssertionError("fake bridge server did not stop")
        if self.error is not None:
            raise self.error


def _send(connection: socket.socket, response: dict[str, object]) -> None:
    connection.sendall(encode_frame(response))


def _receive(server: _Server, connection: socket.socket) -> dict[str, object]:
    request = receive_frame(connection)
    server.requests.append(request)
    return request


def _begin_response(server: _Server, connection: socket.socket) -> dict[str, object]:
    request = _receive(server, connection)
    if request["type"] != "arm_and_begin_episode":
        raise AssertionError(f"expected arm_and_begin_episode, got {request['type']!r}")
    _send(connection, _response(request))
    return request


def _enter_intervention(server: _Server, connection: socket.socket) -> None:
    exchange = _receive(server, connection)
    _send(
        connection,
        _response(
            exchange,
            transition="entering_intervention",
            follower_actuation_mode="hold",
            motion_accepted=False,
        ),
    )
    ack = _receive(server, connection)
    _send(
        connection,
        _response(
            ack,
            generation=int(ack["generation"]) + 1,
            mode="intervention",
            edge="enter",
            follower_actuation_mode="hold",
            motion_accepted=True,
        ),
    )


class PiperXBridgeClientTest(unittest.TestCase):
    def _client(self, server: _Server, **overrides) -> PiperXBridgeClient:
        values = {
            "port": server.port,
            "response_timeout_s": 1.0,
            "arm_timeout_s": 1.0,
            "transition_timeout_s": 1.0,
            "heartbeat_interval_s": 10.0,
        }
        values.update(overrides)
        return PiperXBridgeClient(**values)

    def test_v3_schema_parses_exact_manual_sample(self):
        def handler(server, connection):
            _begin_response(server, connection)
            _enter_intervention(server, connection)
            request = _receive(server, connection)
            self.assertEqual(request["type"], "manual_sample")
            self.assertEqual(request["payload"], {})
            _send(
                connection,
                _response(
                    request,
                    mode="intervention",
                    manual_sample=_manual_sample(7),
                    motion_accepted=False,
                    follower_actuation_mode="hold",
                ),
            )

        server = _Server(handler).start()
        client = self._client(server)
        self.assertEqual(PROTOCOL, "robodojo_piperx_v3")
        client.begin_episode("strict-v3", _sim())
        transition = client.exchange(_sim())
        self.assertEqual(transition.transition, "entering_intervention")
        client.transition_ack(_sim())
        sample = client.manual_sample()

        self.assertEqual(sample.manual_sample.sample_id, 7)
        self.assertEqual(sample.left.sampled_monotonic_ns, 1001)
        self.assertEqual(sample.right.pose[0], -0.1)
        self.assertFalse(sample.motion_accepted)
        client.close()
        server.join()

    def test_response_payload_and_manual_arm_schemas_are_exact(self):
        cases = ("payload-extra", "arm-missing-timestamp")
        for case in cases:
            with self.subTest(case=case):
                def handler(server, connection, *, case=case):
                    request = _receive(server, connection)
                    response = _response(request)
                    if case == "payload-extra":
                        response["payload"]["unexpected"] = True
                    else:
                        response = _response(
                            request,
                            mode="intervention",
                            manual_sample=_manual_sample(),
                            motion_accepted=False,
                            follower_actuation_mode="hold",
                        )
                        response["type"] = "manual_sample"
                        del response["payload"]["manual_sample"]["left"]["sampled_monotonic_ns"]
                    _send(connection, response)

                server = _Server(handler).start()
                client = self._client(server)
                if case == "payload-extra":
                    with self.assertRaisesRegex(PiperXBridgeProtocolError, "strict schema"):
                        client.begin_episode(case, _sim())
                else:
                    # Parse against the request type that carries the nested arm schema.
                    client._episode_id = case
                    with self.assertRaisesRegex(PiperXBridgeProtocolError, "sampled_monotonic_ns"):
                        client._request("manual_sample", {})
                client.close()
                server.join()

    def test_v2_peer_is_rejected_before_any_hardware_state_is_assumed(self):
        def handler(server, connection):
            request = _receive(server, connection)
            response = _response(request)
            response["protocol"] = "robodojo_piperx_v2"
            _send(connection, response)

        server = _Server(handler).start()
        client = self._client(server)
        with self.assertRaisesRegex(PiperXBridgeProtocolError, "protocol/type"):
            client.begin_episode("old-peer", _sim())
        self.assertFalse(client.episode_active)
        with self.assertRaisesRegex(PiperXBridgeTransportError, "cannot be reconnected"):
            client.connect()
        server.join()

    def test_arming_and_transition_use_independent_long_deadlines(self):
        def handler(server, connection):
            begin = _receive(server, connection)
            time.sleep(0.06)
            _send(connection, _response(begin))

            exchange = _receive(server, connection)
            _send(
                connection,
                _response(
                    exchange,
                    transition="entering_intervention",
                    follower_actuation_mode="hold",
                    motion_accepted=False,
                ),
            )

            ack = _receive(server, connection)
            time.sleep(0.06)
            _send(
                connection,
                _response(
                    ack,
                    generation=1,
                    mode="intervention",
                    edge="enter",
                    follower_actuation_mode="hold",
                ),
            )

        server = _Server(handler).start()
        client = self._client(
            server,
            response_timeout_s=0.02,
            arm_timeout_s=0.25,
            transition_timeout_s=0.20,
        )
        client.begin_episode("long-boundaries", _sim())
        self.assertEqual(client.exchange(_sim()).transition, "entering_intervention")
        self.assertEqual(client.transition_ack(_sim()).edge, "enter")

        begin, exchange, ack = server.requests
        begin_budget = int(begin["deadline_monotonic_ns"]) - int(begin["sent_monotonic_ns"])
        exchange_budget = int(exchange["deadline_monotonic_ns"]) - int(exchange["sent_monotonic_ns"])
        ack_budget = int(ack["deadline_monotonic_ns"]) - int(ack["sent_monotonic_ns"])
        self.assertEqual(begin_budget, 250_000_000)
        self.assertEqual(exchange_budget, 20_000_000)
        self.assertEqual(ack_budget, 200_000_000)
        self.assertAlmostEqual(client._socket.gettimeout(), 0.02, places=3)
        client.close()
        server.join()

    def test_manual_sample_and_resolution_are_exactly_once_client_transactions(self):
        def handler(server, connection):
            _begin_response(server, connection)
            _enter_intervention(server, connection)

            sample_request = _receive(server, connection)
            _send(
                connection,
                _response(
                    sample_request,
                    mode="intervention",
                    manual_sample=_manual_sample(11),
                    motion_accepted=False,
                    follower_actuation_mode="hold",
                ),
            )

            resolve_request = _receive(server, connection)
            self.assertEqual(
                resolve_request["payload"],
                {"sample_id": 11, "decision": "commit"},
            )
            _send(
                connection,
                _response(
                    resolve_request,
                    mode="intervention",
                    manual_resolution=_manual_resolution(11, "commit"),
                    motion_accepted=True,
                    follower_actuation_mode="leader_follow",
                ),
            )

        server = _Server(handler).start()
        client = self._client(server)
        client.begin_episode("exact-once", _sim())
        client.exchange(_sim())
        client.transition_ack(_sim())
        sample = client.manual_sample()

        with self.assertRaisesRegex(PiperXBridgeProtocolError, "resolved before sampling again"):
            client.manual_sample()
        with self.assertRaisesRegex(PiperXBridgeProtocolError, "not the one pending"):
            client.manual_resolve(12, commit=True)

        result = client.manual_resolve(11, commit=True)
        self.assertEqual(result.manual_resolution.sample_id, sample.manual_sample.sample_id)
        self.assertTrue(result.manual_resolution.follower_commanded)
        with self.assertRaisesRegex(PiperXBridgeProtocolError, "not the one pending"):
            client.manual_resolve(11, commit=True)

        # The three rejected local operations above must not create wire requests.
        self.assertEqual(
            [request["type"] for request in server.requests],
            [
                "arm_and_begin_episode",
                "exchange",
                "transition_ack",
                "manual_sample",
                "manual_resolve",
            ],
        )
        client.close()
        server.join()

    def test_post_switch_anchor_is_an_exact_non_motion_resolution(self):
        def handler(server, connection):
            _begin_response(server, connection)
            _enter_intervention(server, connection)
            sample_request = _receive(server, connection)
            _send(
                connection,
                _response(
                    sample_request,
                    mode="intervention",
                    manual_sample=_manual_sample(13),
                    motion_accepted=False,
                    follower_actuation_mode="hold",
                ),
            )
            anchor_request = _receive(server, connection)
            self.assertEqual(
                anchor_request["payload"],
                {"sample_id": 13, "decision": "anchor"},
            )
            response = _response(
                anchor_request,
                mode="intervention",
                manual_resolution=_manual_resolution(13, "anchor"),
                motion_accepted=True,
                follower_actuation_mode="hold",
            )
            response["payload"]["diagnostics"]["operation"] = {
                "manual_anchor_latched": True,
            }
            _send(connection, response)

        server = _Server(handler).start()
        client = self._client(server)
        client.begin_episode("post-switch-anchor", _sim())
        client.exchange(_sim())
        client.transition_ack(_sim())
        sampled = client.manual_sample()
        result = client.anchor_manual_sample(sampled.manual_sample.sample_id)

        self.assertEqual(result.manual_resolution.decision, "anchor")
        self.assertFalse(result.manual_resolution.follower_commanded)
        self.assertTrue(result.diagnostics["operation"]["manual_anchor_latched"])
        client.close()
        server.join()

    def test_manual_resolve_response_must_match_the_pending_sample_id(self):
        def handler(server, connection):
            _begin_response(server, connection)
            _enter_intervention(server, connection)
            sample_request = _receive(server, connection)
            _send(
                connection,
                _response(
                    sample_request,
                    mode="intervention",
                    manual_sample=_manual_sample(21),
                    motion_accepted=False,
                    follower_actuation_mode="hold",
                ),
            )
            resolve_request = _receive(server, connection)
            _send(
                connection,
                _response(
                    resolve_request,
                    mode="intervention",
                    manual_resolution=_manual_resolution(22, "reject"),
                    motion_accepted=True,
                    follower_actuation_mode="hold",
                ),
            )

        server = _Server(handler).start()
        client = self._client(server)
        client.begin_episode("mismatched-resolution", _sim())
        client.exchange(_sim())
        client.transition_ack(_sim())
        client.manual_sample()
        with self.assertRaisesRegex(PiperXBridgeProtocolError, "sample.*match|pending"):
            client.manual_resolve(21, commit=False)
        self.assertFalse(client.episode_active)
        server.join()

    def test_transition_ack_clears_an_unresolved_manual_sample(self):
        def handler(server, connection):
            _begin_response(server, connection)
            _enter_intervention(server, connection)
            sample_request = _receive(server, connection)
            _send(
                connection,
                _response(
                    sample_request,
                    mode="intervention",
                    manual_sample=_manual_sample(31),
                    motion_accepted=False,
                    follower_actuation_mode="hold",
                ),
            )
            ack = _receive(server, connection)
            _send(
                connection,
                _response(
                    ack,
                    generation=2,
                    mode="policy",
                    edge="exit",
                    follower_actuation_mode="hold",
                    motion_accepted=True,
                ),
            )

        server = _Server(handler).start()
        client = self._client(server)
        client.begin_episode("transition-invalidates-sample", _sim())
        client.exchange(_sim())
        client.transition_ack(_sim())
        client.manual_sample()
        client.transition_ack(_sim())
        with self.assertRaisesRegex(PiperXBridgeProtocolError, "not the one pending"):
            client.manual_resolve(31, commit=True)
        client.close()
        server.join()

    def test_malformed_response_loses_and_closes_the_session(self):
        def handler(server, connection):
            request = _receive(server, connection)
            _send(connection, _response(request, seq=int(request["seq"]) + 1))
            connection.settimeout(1)
            self.assertEqual(connection.recv(1), b"")

        server = _Server(handler).start()
        client = self._client(server)
        with self.assertRaisesRegex(PiperXBridgeProtocolError, "seq"):
            client.begin_episode("bad-response", _sim())
        self.assertIsNone(client._socket)
        with self.assertRaisesRegex(PiperXBridgeTransportError, "cannot be reconnected"):
            client.connect()
        server.join()

    def test_non_loopback_host_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            PiperXBridgeClient(host="192.0.2.10")


if __name__ == "__main__":
    unittest.main()
