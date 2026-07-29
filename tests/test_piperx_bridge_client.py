import socket
import threading
import time
import unittest

from src.eval_client.piperx_bridge_client import (
    PROTOCOL,
    PiperXBridgeClient,
    PiperXBridgeProtocolError,
    PiperXBridgeSafetyError,
    PiperXBridgeTransportError,
    SimArmTarget,
    SimTargets,
    encode_frame,
    receive_frame,
)


def _sim(x=0.0):
    return SimTargets(
        left=SimArmTarget((x, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0), 0.7),
        right=SimArmTarget((-x, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0), 0.8),
    )


def _response(
    request,
    *,
    generation=None,
    mode="policy",
    edge=None,
    terminal=None,
    seq=None,
    mirror_accepted=None,
):
    return {
        "protocol": PROTOCOL,
        "type": request["type"],
        "session_id": request["session_id"],
        "episode_id": request["episode_id"],
        "generation": request["generation"] if generation is None else generation,
        "seq": request["seq"] if seq is None else seq,
        "sent_monotonic_ns": time.monotonic_ns(),
        "deadline_monotonic_ns": request["deadline_monotonic_ns"],
        "payload": {
            "mode": mode,
            "edge": edge,
            "terminal_request": terminal,
            "leader": {
                "left": {
                    "pose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                    "gripper_m": 0.04,
                },
                "right": {
                    "pose": [-0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                    "gripper_m": 0.05,
                },
            },
            "mirror_accepted": (request["type"] == "exchange" if mirror_accepted is None else mirror_accepted),
            "health": {"ok": True},
            "diagnostics": {"request_ok": True},
        },
    }


class _Server:
    def __init__(self, handler):
        self.handler = handler
        self.requests = []
        self.error = None
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
        except Exception as exc:  # pragma: no cover - surfaced in join()
            self.error = exc
        finally:
            self.listener.close()

    def start(self):
        self.thread.start()
        self.ready.wait(1)
        return self

    def join(self):
        self.thread.join(2)
        if self.thread.is_alive():
            self.listener.close()
            self.fail("fake bridge server did not stop")
        if self.error is not None:
            raise self.error

    @staticmethod
    def fail(message):
        raise AssertionError(message)


class PiperXBridgeClientTest(unittest.TestCase):
    def test_invalid_begin_state_requests_hold_and_permanently_loses_session(self):
        def handler(server, connection):
            begin = receive_frame(connection)
            server.requests.append(begin)
            connection.sendall(
                encode_frame(
                    _response(
                        begin,
                        mode="intervention",
                    )
                )
            )
            hold = receive_frame(connection)
            server.requests.append(hold)
            connection.sendall(encode_frame(_response(hold)))

        server = _Server(handler).start()
        client = PiperXBridgeClient(
            port=server.port,
            response_timeout_s=1.0,
            heartbeat_interval_s=10.0,
        )
        with self.assertRaisesRegex(PiperXBridgeProtocolError, "policy mode"):
            client.begin_episode("invalid-begin", _sim())
        with self.assertRaisesRegex(PiperXBridgeTransportError, "cannot be reconnected"):
            client.connect()
        server.join()

        self.assertEqual(
            [request["type"] for request in server.requests],
            ["begin_episode", "hold"],
        )
        self.assertEqual(server.requests[1]["payload"], {"reason": "invalid_begin_state"})

    def test_rejected_hold_response_is_fatal(self):
        def handler(server, connection):
            begin = receive_frame(connection)
            server.requests.append(begin)
            connection.sendall(encode_frame(_response(begin)))
            hold = receive_frame(connection)
            server.requests.append(hold)
            response = _response(hold)
            response["payload"]["diagnostics"] = {
                "request_ok": False,
                "reason": "hold_not_confirmed",
            }
            connection.sendall(encode_frame(response))

        server = _Server(handler).start()
        client = PiperXBridgeClient(
            port=server.port,
            response_timeout_s=1.0,
            heartbeat_interval_s=10.0,
        )
        client.begin_episode("hold-rejected", _sim())
        with self.assertRaisesRegex(PiperXBridgeSafetyError, "rejected hold"):
            client.hold("test hold")
        self.assertFalse(client.episode_active)
        client.close()
        server.join()

    def test_rejected_begin_response_is_fatal(self):
        def handler(server, connection):
            request = receive_frame(connection)
            server.requests.append(request)
            response = _response(request)
            response["payload"]["diagnostics"] = {
                "request_ok": False,
                "reason": "followers_not_held",
            }
            connection.sendall(encode_frame(response))

        server = _Server(handler).start()
        client = PiperXBridgeClient(
            port=server.port,
            response_timeout_s=1.0,
            heartbeat_interval_s=10.0,
        )
        with self.assertRaisesRegex(PiperXBridgeSafetyError, "rejected begin_episode"):
            client.begin_episode("rejected", _sim())
        client.close()
        server.join()

    def test_plain_exchange_mirror_rejection_is_fatal(self):
        def handler(server, connection):
            begin = receive_frame(connection)
            server.requests.append(begin)
            connection.sendall(encode_frame(_response(begin)))
            exchange = receive_frame(connection)
            server.requests.append(exchange)
            connection.sendall(encode_frame(_response(exchange, mirror_accepted=False)))

        server = _Server(handler).start()
        client = PiperXBridgeClient(
            port=server.port,
            response_timeout_s=1.0,
            heartbeat_interval_s=10.0,
        )
        client.begin_episode("mirror-rejected", _sim())
        with self.assertRaisesRegex(PiperXBridgeSafetyError, "rejected"):
            client.exchange(_sim())
        client.close()
        server.join()

    def test_idle_heartbeat_keeps_session_alive_during_slow_episode_commit(self):
        idle_heartbeat = threading.Event()

        def handler(server, connection):
            ended_once = False
            while True:
                request = receive_frame(connection)
                server.requests.append(request)
                connection.sendall(encode_frame(_response(request)))
                if request["type"] == "end_episode":
                    if ended_once:
                        return
                    ended_once = True
                elif request["type"] == "heartbeat" and ended_once:
                    idle_heartbeat.set()

        server = _Server(handler).start()
        client = PiperXBridgeClient(
            port=server.port,
            response_timeout_s=1.0,
            heartbeat_interval_s=0.01,
        )
        client.begin_episode("episode-before-commit", _sim())
        client.end_episode(reason="accept_and_commit")
        # Stand in for CPU video encoding/metadata commit. The persistent
        # heartbeat must remain active even though no episode is active.
        self.assertTrue(idle_heartbeat.wait(1))
        self.assertFalse(client.episode_active)
        client.begin_episode("episode-after-commit", _sim(0.2))
        client.end_episode(reason="done")
        client.close()
        server.join()

        types = [request["type"] for request in server.requests]
        first_end = types.index("end_episode")
        second_begin = types.index("begin_episode", 1)
        self.assertIn("heartbeat", types[first_end + 1 : second_begin])
        self.assertEqual(
            [request["seq"] for request in server.requests],
            list(range(len(server.requests))),
        )

    def test_two_episodes_reuse_one_connection_and_global_seq(self):
        def handler(server, connection):
            for _ in range(6):
                request = receive_frame(connection)
                server.requests.append(request)
                connection.sendall(encode_frame(_response(request)))

        server = _Server(handler).start()
        client = PiperXBridgeClient(
            port=server.port,
            response_timeout_s=1.0,
            heartbeat_interval_s=10.0,
        )
        client.begin_episode("episode-1", _sim())
        client.exchange(_sim(0.01))
        client.end_episode(reason="operator_accept_next")
        client.begin_episode("episode-2", _sim(0.2))
        client.exchange(_sim(0.21))
        client.end_episode(reason="operator_discard_exit")
        client.close()
        server.join()

        self.assertEqual(
            [request["type"] for request in server.requests],
            [
                "begin_episode",
                "exchange",
                "end_episode",
                "begin_episode",
                "exchange",
                "end_episode",
            ],
        )
        self.assertEqual([request["seq"] for request in server.requests], list(range(6)))
        self.assertEqual(
            server.requests[0]["payload"],
            {"sim": _sim().to_payload()},
        )
        self.assertEqual(server.requests[2]["payload"], {"reason": "operator_accept_next"})
        self.assertEqual(
            {request["session_id"] for request in server.requests},
            {client.session_id},
        )

    def test_heartbeat_does_not_consume_generation_or_operator_edge(self):
        heartbeat_seen = threading.Event()

        def handler(server, connection):
            while True:
                request = receive_frame(connection)
                server.requests.append(request)
                if request["type"] == "heartbeat":
                    heartbeat_seen.set()
                    response = _response(request)
                elif request["type"] == "exchange":
                    response = _response(
                        request,
                        generation=1,
                        mode="intervention",
                        edge="enter",
                        mirror_accepted=False,
                    )
                else:
                    response = _response(
                        request,
                        generation=request["generation"],
                        mode="intervention" if request["generation"] else "policy",
                    )
                connection.sendall(encode_frame(response))
                if request["type"] == "end_episode":
                    return

        server = _Server(handler).start()
        client = PiperXBridgeClient(
            port=server.port,
            response_timeout_s=1.0,
            heartbeat_interval_s=0.01,
        )
        client.begin_episode("episode-heartbeat", _sim())
        self.assertTrue(heartbeat_seen.wait(1))
        self.assertEqual(client.generation, 0)
        sample = client.exchange(_sim())
        self.assertEqual((sample.generation, sample.edge, sample.mode), (1, "enter", "intervention"))
        client.end_episode(reason="done")
        client.close()
        server.join()
        self.assertIn("heartbeat", [request["type"] for request in server.requests])

    def test_malformed_response_loses_and_closes_session(self):
        def handler(server, connection):
            request = receive_frame(connection)
            server.requests.append(request)
            connection.sendall(encode_frame(_response(request, seq=request["seq"] + 1)))
            connection.settimeout(1)
            self.assertEqual(connection.recv(1), b"")

        server = _Server(handler).start()
        client = PiperXBridgeClient(
            port=server.port,
            response_timeout_s=1.0,
            heartbeat_interval_s=10.0,
        )
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
