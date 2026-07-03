"""
Tests for orchestrator/server.py, over real TCP sockets on loopback.

Run: PYTHONPATH=. python3 -m pytest tests/test_orchestrator.py -v
"""

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ringsync.orchestrator.server import OrchestratorServer
from ringsync.network.protocol import send_json, recv_json, MsgType
from ringsync.network.socket_utils import set_common_socket_options

TEST_PORT = 17500


def _connect(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    set_common_socket_options(sock)
    return sock


def _register(port, rank, worker_host, worker_port, shard_range):
    sock = _connect(port)
    send_json(sock, MsgType.REGISTER, {
        "rank": rank, "host": worker_host, "port": worker_port, "shard_range": list(shard_range),
    })
    msg_type, resp = recv_json(sock)
    assert msg_type == MsgType.REGISTER_ACK
    assert resp["ok"] is True
    return sock  # keep open so heartbeats can reuse it


def _heartbeat(sock, rank):
    send_json(sock, MsgType.HEARTBEAT, {"rank": rank})
    msg_type, resp = recv_json(sock)
    assert msg_type == MsgType.HEARTBEAT_ACK
    return resp["ok"]


def _keep_alive(sock, rank, duration_s, interval_s=0.25):
    """Send heartbeats repeatedly for duration_s. Runs synchronously (call in a thread if needed)."""
    end = time.time() + duration_s
    while time.time() < end:
        _heartbeat(sock, rank)
        time.sleep(interval_s)


def _get_roster(port):
    sock = _connect(port)
    send_json(sock, MsgType.JSON, {"request": "GET_ROSTER"})
    msg_type, resp = recv_json(sock)
    sock.close()
    return resp


@pytest.fixture
def running_server():
    global TEST_PORT
    port = TEST_PORT
    TEST_PORT += 1  # avoid port collisions between tests

    server = OrchestratorServer(control_port=port, heartbeat_timeout_s=1.0, check_interval_s=0.2)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)  # let the listener bind
    yield server, port
    server.stop()


def test_registration_and_roster(running_server):
    server, port = running_server

    _register(port, rank=0, worker_host="127.0.0.1", worker_port=20000, shard_range=(0, 50))
    _register(port, rank=1, worker_host="127.0.0.1", worker_port=20001, shard_range=(50, 100))

    roster = _get_roster(port)
    assert roster["world_size"] == 2
    assert "0" in roster["addresses"] and "1" in roster["addresses"]


def test_heartbeat_keeps_worker_alive(running_server):
    server, port = running_server
    sock = _register(port, rank=0, worker_host="127.0.0.1", worker_port=20000, shard_range=(0, 100))

    for _ in range(3):
        time.sleep(0.3)
        assert _heartbeat(sock, rank=0) is True

    roster = server.get_active_roster()
    assert 0 in roster
    sock.close()


def test_missed_heartbeats_trigger_eviction_and_redistribution(running_server):
    server, port = running_server

    sock0 = _register(port, rank=0, worker_host="127.0.0.1", worker_port=20000, shard_range=(0, 50))
    sock1 = _register(port, rank=1, worker_host="127.0.0.1", worker_port=20001, shard_range=(50, 100))

    # rank 0 keeps heartbeating throughout; rank 1 heartbeats once then goes silent
    keepalive_thread = threading.Thread(target=_keep_alive, args=(sock0, 0, 1.5))
    keepalive_thread.start()

    time.sleep(1.5)  # exceed heartbeat_timeout_s=1.0 for the now-silent rank 1
    keepalive_thread.join()
    # Eviction may have already happened via the background monitor thread
    # (it also runs check_timeouts_once() every check_interval_s) -- either
    # way, by now rank 1 must be gone from the active roster.
    server.check_timeouts_once()
    roster = server.get_active_roster()
    assert len(roster) == 1
    # the surviving worker's shard should now cover the full original range [0, 100)
    survivor = list(roster.values())[0]
    assert survivor.shard_range == (0, 100)

    sock0.close()
    sock1.close()


def test_multiple_workers_one_dies_others_survive(running_server):
    server, port = running_server

    socks = []
    for rank, (start, end) in enumerate([(0, 25), (25, 50), (50, 75), (75, 100)]):
        s = _register(port, rank=rank, worker_host="127.0.0.1", worker_port=20100 + rank, shard_range=(start, end))
        socks.append(s)

    # ranks 0, 2, 3 heartbeat continuously; rank 1 does not
    threads = [
        threading.Thread(target=_keep_alive, args=(socks[0], 0, 1.5)),
        threading.Thread(target=_keep_alive, args=(socks[2], 2, 1.5)),
        threading.Thread(target=_keep_alive, args=(socks[3], 3, 1.5)),
    ]
    for t in threads:
        t.start()

    time.sleep(1.5)
    for t in threads:
        t.join()
    server.check_timeouts_once()
    roster = server.get_active_roster()
    assert len(roster) == 3

    # data conservation check: total shard coverage should still be [0, 100)
    ranges = sorted(w.shard_range for w in roster.values())
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 100
    for i in range(len(ranges) - 1):
        assert ranges[i][1] == ranges[i + 1][0]

    for s in socks:
        s.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
