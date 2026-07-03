"""
Tests for allreduce/ring_allreduce.py.

The core claim being tested: after ring_all_reduce runs across a
simulated ring of N workers, every worker ends up with the identical
result, and that result equals the plain elementwise average computed
directly with numpy. This is the ground-truth correctness check for
the algorithm itself, independent of any networking code.

Run: PYTHONPATH=. python3 -m pytest tests/test_allreduce.py -v
"""

import queue
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ringsync.allreduce.ring_allreduce import ring_all_reduce, compute_chunk_boundaries


def run_simulated_ring(local_arrays, average=True):
    """
    Runs ring_all_reduce across len(local_arrays) simulated workers in
    parallel threads, connected in a ring via queue.Queue as the
    "network" for each directed edge. Returns the list of each worker's
    final result (should all be identical).
    """
    world_size = len(local_arrays)

    # edge_queues[i] = channel carrying messages FROM worker i TO worker (i+1) % world_size
    edge_queues = [queue.Queue() for _ in range(world_size)]

    results = [None] * world_size
    errors = [None] * world_size

    def worker_thread(rank):
        try:
            def send_fn(arr):
                edge_queues[rank].put(arr)

            def recv_fn():
                left = (rank - 1) % world_size
                return edge_queues[left].get()

            results[rank] = ring_all_reduce(
                local_arrays[rank], rank, world_size, send_fn, recv_fn, average=average
            )
        except Exception as e:  # surface thread exceptions to the test
            errors[rank] = e

    threads = [threading.Thread(target=worker_thread, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    for e in errors:
        if e is not None:
            raise e

    return results


@pytest.mark.parametrize("world_size", [2, 3, 4, 5])
def test_ring_all_reduce_matches_manual_average(world_size):
    np.random.seed(0)
    shape = (37,)  # deliberately not divisible by any small world_size, exercises remainder handling
    local_arrays = [np.random.randn(*shape).astype(np.float32) for _ in range(world_size)]

    expected = np.mean(np.stack(local_arrays), axis=0)

    results = run_simulated_ring(local_arrays, average=True)

    for rank, result in enumerate(results):
        assert result.shape == shape
        assert np.allclose(result, expected, atol=1e-4), f"worker {rank} diverged from expected average"

    # every worker's result should also be identical to every other worker's
    for rank in range(1, world_size):
        assert np.allclose(results[0], results[rank], atol=1e-6)


def test_ring_all_reduce_sum_mode():
    np.random.seed(1)
    world_size = 4
    shape = (16,)
    local_arrays = [np.random.randn(*shape).astype(np.float32) for _ in range(world_size)]

    expected_sum = np.sum(np.stack(local_arrays), axis=0)
    results = run_simulated_ring(local_arrays, average=False)

    for result in results:
        assert np.allclose(result, expected_sum, atol=1e-4)


def test_ring_all_reduce_multidim_shape():
    """Gradients are rarely 1D in practice (conv kernels, weight matrices) -- confirm reshape survives."""
    np.random.seed(2)
    world_size = 3
    shape = (8, 4, 3)
    local_arrays = [np.random.randn(*shape).astype(np.float32) for _ in range(world_size)]

    expected = np.mean(np.stack(local_arrays), axis=0)
    results = run_simulated_ring(local_arrays, average=True)

    for result in results:
        assert result.shape == shape
        assert np.allclose(result, expected, atol=1e-4)


def test_chunk_boundaries_cover_full_range_no_overlap():
    for total_len, world_size in [(10, 3), (100, 7), (5, 5), (1, 1)]:
        bounds = compute_chunk_boundaries(total_len, world_size)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == total_len
        for i in range(len(bounds) - 1):
            assert bounds[i][1] == bounds[i + 1][0]  # contiguous, no gaps/overlaps


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
