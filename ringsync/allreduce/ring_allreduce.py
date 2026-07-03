"""
Ring all-reduce: reduce-scatter + all-gather.

This is the algorithmic core of RingSync. It is deliberately written
against abstract `send_fn` / `recv_fn` callables rather than raw
sockets, so the *math* can be unit-tested with an in-memory simulated
ring (see tests/test_allreduce.py) completely independent of the
network layer. If there's a bug here, you want to find it with a fast
in-process test, not by staring at tcpdump output.

--- The algorithm ---

Given N workers, each holding a local array of the same shape (their
locally-computed gradient), ring all-reduce computes the elementwise
sum (or average) across all N arrays, and ensures every worker ends up
with the identical, fully-reduced result -- without any central
coordinator doing the summing. It does this in two phases:

1. Reduce-scatter (N-1 steps):
   The array is split into N contiguous chunks. At each step, every
   worker forwards one chunk to its right neighbor and accumulates an
   incoming chunk from its left neighbor into its own copy. After N-1
   steps, chunk `(rank+1) % N` at worker `rank` holds the fully-reduced
   sum for that chunk -- each worker ends up owning exactly one
   completely-reduced chunk, and the chunks are scattered across
   workers (hence the name).

2. All-gather (N-1 steps):
   Now every worker needs every chunk, not just the one it reduced.
   Each worker forwards the chunk it currently holds (no more adding,
   just passing along) to its right neighbor, and overwrites the
   corresponding slot with what it receives from its left neighbor.
   After N-1 steps, every worker holds all N fully-reduced chunks.

Total data moved per worker: 2*(N-1)/N * array_size -- independent of
N for large N, and never funnelled through a single bottleneck node.
This is why ring all-reduce scales better than a parameter-server
architecture as worker count grows.
"""

import threading
from typing import Callable, List, Tuple

import numpy as np

SendFn = Callable[[np.ndarray], None]
RecvFn = Callable[[], np.ndarray]


def compute_chunk_boundaries(total_len: int, world_size: int) -> List[Tuple[int, int]]:
    """Split [0, total_len) into world_size contiguous, near-equal ranges."""
    base = total_len // world_size
    remainder = total_len % world_size
    bounds = []
    start = 0
    for i in range(world_size):
        size = base + (1 if i < remainder else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def _exchange(send_fn: SendFn, recv_fn: RecvFn, to_send: np.ndarray) -> np.ndarray:
    """
    Send `to_send` to the right neighbor and receive from the left
    neighbor, concurrently. This MUST happen concurrently, not
    sequentially -- if every worker in the ring calls send() and blocks
    waiting for the peer to be ready to receive before anyone calls
    recv(), the entire ring deadlocks (everyone's blocked in send,
    nobody's in recv). Running send on a background thread while recv
    happens on the calling thread avoids this.
    """
    received: List[np.ndarray] = [None]  # type: ignore

    def _do_recv():
        received[0] = recv_fn()

    recv_thread = threading.Thread(target=_do_recv)
    recv_thread.start()
    send_fn(to_send)
    recv_thread.join()
    return received[0]


def ring_reduce_scatter(
    local_array: np.ndarray,
    rank: int,
    world_size: int,
    send_fn: SendFn,
    recv_fn: RecvFn,
) -> Tuple[np.ndarray, int]:
    """
    Phase 1. Returns (reduced_chunk, chunk_index) -- the single chunk
    this worker now fully owns the reduced value of.
    """
    bounds = compute_chunk_boundaries(len(local_array), world_size)
    chunks = [local_array[s:e].copy() for s, e in bounds]

    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        incoming = _exchange(send_fn, recv_fn, chunks[send_idx])
        chunks[recv_idx] = chunks[recv_idx] + incoming

    owned_idx = (rank + 1) % world_size
    return chunks[owned_idx], owned_idx


def ring_all_gather(
    owned_chunk: np.ndarray,
    owned_idx: int,
    rank: int,
    world_size: int,
    send_fn: SendFn,
    recv_fn: RecvFn,
) -> List[np.ndarray]:
    """
    Phase 2. Every worker starts holding one fully-reduced chunk
    (`owned_chunk` at `owned_idx`) and ends up holding all world_size
    fully-reduced chunks, in order.
    """
    chunks: List[np.ndarray] = [None] * world_size  # type: ignore
    chunks[owned_idx] = owned_chunk

    for step in range(world_size - 1):
        send_idx = (owned_idx - step) % world_size
        recv_idx = (owned_idx - step - 1) % world_size

        incoming = _exchange(send_fn, recv_fn, chunks[send_idx])
        chunks[recv_idx] = incoming

    return chunks


def ring_all_reduce(
    local_array: np.ndarray,
    rank: int,
    world_size: int,
    send_fn: SendFn,
    recv_fn: RecvFn,
    average: bool = True,
) -> np.ndarray:
    """
    Full ring all-reduce: every worker passes in its own local array
    (same shape on every worker) and gets back the elementwise
    sum-across-all-workers (or average, if average=True), reassembled
    in original order.
    """
    original_shape = local_array.shape
    flat = local_array.flatten().astype(np.float32)

    reduced_chunk, owned_idx = ring_reduce_scatter(flat, rank, world_size, send_fn, recv_fn)
    all_chunks = ring_all_gather(reduced_chunk, owned_idx, rank, world_size, send_fn, recv_fn)

    result = np.concatenate(all_chunks)
    if average:
        result = result / world_size
    return result.reshape(original_shape)


if __name__ == "__main__":
    bounds = compute_chunk_boundaries(10, 3)
    print(f"Chunk boundaries for len=10, world_size=3: {bounds}")
    assert bounds == [(0, 4), (4, 7), (7, 10)]
    print("compute_chunk_boundaries OK")
