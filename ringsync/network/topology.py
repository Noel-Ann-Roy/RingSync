"""
Ring topology setup.

Turns "I am rank r out of N, here's everyone's address" into two live,
connected TCP sockets:
  - inbound_sock:  accepted connection FROM the left neighbor (rank-1)
  - outbound_sock: a connection this worker initiated TO the right
                    neighbor (rank+1)

Every worker is simultaneously a server (listening for its left
neighbor to connect in) and a client (connecting out to its right
neighbor). Both must happen concurrently: if every worker tried to
connect out before accepting inbound connections, and dialing an address
that isn't listening yet fails, workers can get stuck retrying in
lockstep. The listener starts first and accept() runs in a background
thread while the main thread dials out, so a worker that starts up in
any order relative to its neighbors will still converge.
"""

import socket
import threading
import time

from ringsync.network.socket_utils import set_common_socket_options
from ringsync.network.protocol import send_json, recv_json, MsgType


def _start_listener(port: int, host: str = "0.0.0.0") -> socket.socket:
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind((host, port))
    listen_sock.listen(1)
    return listen_sock


def _connect_outbound(
    host: str, port: int, retry_interval: float = 0.3, max_retries: int = 100
) -> socket.socket:
    """
    Retries connecting until the right neighbor's listener is up.
    In a freshly-started ring, there's no guarantee about start order,
    so a few connection-refused errors early on are expected and not
    an error condition -- only exhausting max_retries is.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            return sock
        except (ConnectionRefusedError, OSError) as e:
            last_error = e
            time.sleep(retry_interval)
    raise ConnectionError(
        f"Could not connect to {host}:{port} after {max_retries} attempts"
    ) from last_error


def setup_ring(
    rank: int,
    world_size: int,
    worker_addresses: "list[tuple[str, int]]",
) -> "tuple[socket.socket, socket.socket]":
    """
    worker_addresses[i] = (host, port) that rank i listens on.
    Returns (inbound_sock, outbound_sock) for the calling worker.
    """
    my_host, my_port = worker_addresses[rank]
    right_rank = (rank + 1) % world_size
    right_host, right_port = worker_addresses[right_rank]

    listen_sock = _start_listener(my_port, host="0.0.0.0")

    accepted_holder = {}

    def _accept():
        conn, _addr = listen_sock.accept()
        accepted_holder["sock"] = conn

    accept_thread = threading.Thread(target=_accept)
    accept_thread.start()

    outbound_sock = _connect_outbound(right_host, right_port)

    accept_thread.join(timeout=30)
    listen_sock.close()

    if "sock" not in accepted_holder:
        raise ConnectionError(
            f"rank {rank}: timed out waiting for left neighbor to connect"
        )
    inbound_sock = accepted_holder["sock"]

    set_common_socket_options(inbound_sock)
    set_common_socket_options(outbound_sock)

    # handshake: confirm both ends agree on who's who -- catches
    # misconfigured address lists (e.g. off-by-one rank assignment)
    # immediately at startup instead of producing silently-wrong
    # gradients later.
    send_json(outbound_sock, MsgType.RING_HELLO, {"from_rank": rank})
    _msg_type, hello = recv_json(inbound_sock)
    expected_left = (rank - 1) % world_size
    if hello.get("from_rank") != expected_left:
        raise ConnectionError(
            f"rank {rank}: expected handshake from rank {expected_left}, "
            f"got {hello.get('from_rank')}"
        )

    return inbound_sock, outbound_sock
