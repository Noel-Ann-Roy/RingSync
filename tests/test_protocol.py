"""
Tests for network/protocol.py and network/socket_utils.py.

Run: PYTHONPATH=. python3 -m pytest tests/test_protocol.py -v
"""

import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ringsync.network.protocol import (
    MsgType, send_message, recv_message, send_json, recv_json,
    tensor_to_bytes, bytes_to_tensor,
)
from ringsync.network.socket_utils import send_exact, recv_exact, ConnectionClosedError


@pytest.fixture
def socket_pair():
    """A connected pair of TCP sockets over loopback, for realistic I/O."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    accepted, _ = server.accept()

    yield client, accepted

    client.close()
    accepted.close()
    server.close()


def test_send_recv_exact_roundtrip(socket_pair):
    client, accepted = socket_pair
    payload = b"x" * 100_000  # larger than a single recv() will typically return
    send_exact(client, payload)
    received = recv_exact(accepted, len(payload))
    assert received == payload


def test_recv_exact_detects_closed_connection(socket_pair):
    client, accepted = socket_pair
    client.close()
    with pytest.raises(ConnectionClosedError):
        recv_exact(accepted, 10)


def test_json_message_roundtrip(socket_pair):
    client, accepted = socket_pair
    obj = {"rank": 2, "world_size": 4, "step": 17}
    send_json(client, MsgType.HEARTBEAT, obj)
    msg_type, received = recv_json(accepted)
    assert msg_type == MsgType.HEARTBEAT
    assert received == obj


def test_tensor_chunk_message_roundtrip(socket_pair):
    client, accepted = socket_pair
    arr = np.random.randn(64, 128).astype(np.float32)

    send_message(client, MsgType.TENSOR_CHUNK, tensor_to_bytes(arr))
    msg_type, payload = recv_message(accepted)

    assert msg_type == MsgType.TENSOR_CHUNK
    recovered = bytes_to_tensor(payload, arr.shape)
    assert np.allclose(arr, recovered)


def test_multiple_messages_do_not_interleave(socket_pair):
    """
    The bug this guards against: if framing is wrong, a fast sequence
    of sends can get mis-parsed on the receiving end (e.g. reading the
    second message's header as part of the first message's payload).
    """
    client, accepted = socket_pair

    def sender():
        for i in range(20):
            send_json(client, MsgType.HEARTBEAT, {"seq": i})

    t = threading.Thread(target=sender)
    t.start()

    for i in range(20):
        msg_type, obj = recv_json(accepted)
        assert msg_type == MsgType.HEARTBEAT
        assert obj["seq"] == i

    t.join()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
