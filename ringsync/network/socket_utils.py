"""
Low-level TCP helpers.

The one rule this module exists to enforce: TCP is a byte STREAM, not a
message protocol. A single call to socket.recv(n) is only guaranteed to
return "up to n bytes" -- it can return fewer, even if more are on the
way. Code that assumes one recv() call == one message is the single
most common bug in from-scratch socket code, and it tends to work fine
on localhost (where packets rarely fragment) and then break the moment
you run across real machines or containers. Every read/write here loops
until the exact number of bytes requested has been transferred.
"""

import socket


class ConnectionClosedError(Exception):
    """Raised when the peer closes the connection mid-read/write."""
    pass


def send_exact(sock: socket.socket, data: bytes) -> None:
    """Send exactly len(data) bytes, looping over partial sends."""
    total_sent = 0
    view = memoryview(data)
    while total_sent < len(data):
        sent = sock.send(view[total_sent:])
        if sent == 0:
            raise ConnectionClosedError("Socket connection broken during send")
        total_sent += sent


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Receive exactly num_bytes, looping over partial reads."""
    chunks = []
    bytes_received = 0
    while bytes_received < num_bytes:
        chunk = sock.recv(min(num_bytes - bytes_received, 65536))
        if chunk == b"":
            raise ConnectionClosedError("Socket connection broken during recv")
        chunks.append(chunk)
        bytes_received += len(chunk)
    return b"".join(chunks)


def set_common_socket_options(sock: socket.socket) -> None:
    """
    TCP_NODELAY disables Nagle's algorithm, which by default batches
    small writes for up to ~40ms to reduce packet count. For a training
    loop doing many small, latency-sensitive control messages (heartbeats,
    barrier signals), that delay is a real, measurable cost per step,
    so we trade a few extra packets for lower latency.
    """
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
