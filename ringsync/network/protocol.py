"""
RingSync wire protocol.

Frame format (every message on every socket looks like this):

    +----------------+----------------------+------------------+
    | 1 byte         | 8 bytes              | N bytes          |
    | msg_type (u8)  | payload_length (u64) | payload          |
    +----------------+----------------------+------------------+

Design choices, and why:

  - Fixed-size header (9 bytes) sent first, always. The receiver reads
    exactly 9 bytes, learns the payload length, then reads exactly that
    many more. This is the standard length-prefixing pattern for framing
    messages over a byte stream -- no delimiters to escape, no scanning.

  - msg_type is 1 byte (not a string) so routing on the receiving end is
    a simple int switch, and the header is a fixed, tiny, predictable
    size regardless of message type.

  - Two payload kinds ride inside this same frame format:
      * JSON control messages (heartbeats, registration, barrier
        signals) -- small, human-inspectable, easy to extend.
      * Raw tensor bytes (gradient chunks during ring all-reduce) --
        no JSON/pickle overhead on the hot path, since this is the data
        that actually needs to move fast and often.

  - Tensors are serialized as raw float32 bytes (via numpy), not
    pickle. Pickle is slower, has arbitrary-code-execution risk on
    untrusted input, and encodes far more than we need. Since sender
    and receiver already agree on shape/dtype out of band (both know
    the model architecture and the chunk boundaries), we don't need to
    embed a shape header in every tensor message either -- that
    information is reconstructed from context, not the wire.
"""

import json
import socket
import struct
from enum import IntEnum

import numpy as np

from ringsync.network.socket_utils import send_exact, recv_exact

HEADER_FORMAT = "!BQ"  # network byte order: 1 byte type, 8 byte length (u64)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class MsgType(IntEnum):
    # --- control plane (orchestrator <-> worker) ---
    REGISTER = 1
    REGISTER_ACK = 2
    HEARTBEAT = 3
    HEARTBEAT_ACK = 4
    START_EPOCH = 5
    WORKER_EVICTED = 6
    SHUTDOWN = 7

    # --- ring plane (worker <-> worker) ---
    RING_HELLO = 10
    TENSOR_CHUNK = 11
    BARRIER_ENTER = 12
    BARRIER_RELEASE = 13

    # --- generic ---
    JSON = 20


def send_message(sock: socket.socket, msg_type: MsgType, payload: bytes) -> None:
    header = struct.pack(HEADER_FORMAT, int(msg_type), len(payload))
    send_exact(sock, header + payload)


def recv_message(sock: socket.socket) -> "tuple[MsgType, bytes]":
    header = recv_exact(sock, HEADER_SIZE)
    msg_type_int, payload_len = struct.unpack(HEADER_FORMAT, header)
    payload = recv_exact(sock, payload_len) if payload_len > 0 else b""
    return MsgType(msg_type_int), payload


def send_json(sock: socket.socket, msg_type: MsgType, obj: dict) -> None:
    payload = json.dumps(obj).encode("utf-8")
    send_message(sock, msg_type, payload)


def recv_json(sock: socket.socket) -> "tuple[MsgType, dict]":
    msg_type, payload = recv_message(sock)
    obj = json.loads(payload.decode("utf-8")) if payload else {}
    return msg_type, obj


def tensor_to_bytes(arr: np.ndarray) -> bytes:
    """
    Serialize a numpy array to raw bytes, forcing float32 and C-contiguous
    layout. Forcing the dtype means the receiver never has to guess or
    negotiate it -- both sides of the ring were built to agree on float32
    gradients, so this stays a non-issue rather than a hidden bug source.
    """
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def bytes_to_tensor(data: bytes, shape: "tuple[int, ...]") -> np.ndarray:
    """Deserialize raw float32 bytes back into a numpy array of `shape`."""
    arr = np.frombuffer(data, dtype=np.float32)
    return arr.reshape(shape)


if __name__ == "__main__":
    # Round-trip sanity check without needing a live socket: pack/unpack
    # a header, and pack/unpack a tensor.
    header = struct.pack(HEADER_FORMAT, int(MsgType.TENSOR_CHUNK), 1234)
    msg_type_int, length = struct.unpack(HEADER_FORMAT, header)
    assert MsgType(msg_type_int) == MsgType.TENSOR_CHUNK
    assert length == 1234
    print(f"Header round-trip OK. HEADER_SIZE={HEADER_SIZE} bytes")

    arr = np.random.randn(4, 4).astype(np.float32)
    raw = tensor_to_bytes(arr)
    recovered = bytes_to_tensor(raw, arr.shape)
    assert np.allclose(arr, recovered)
    print("Tensor serialization round-trip OK.")
