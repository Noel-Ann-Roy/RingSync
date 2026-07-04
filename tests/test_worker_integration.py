"""
End-to-end integration test: spawns real worker PROCESSES (not threads),
each with its own real TCP listener/connector, forming a real ring over
127.0.0.1. This is the strongest correctness check in the project: if
synchronous ring all-reduce is working correctly, every worker applies
the identical averaged gradient at every step to an identically-
initialized model, so after training, every worker's final model
weights should be numerically identical -- despite each worker having
trained on a *different* slice of data.


Run: PYTHONPATH=. python3 tests/test_worker_integration.py
"""

import json
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ringsync.worker.worker_node import run_worker

BASE_PORT = 15000


def _worker_entrypoint(rank, world_size, addresses, epochs, batch_size, results_dir):
    run_worker(
        rank=rank,
        world_size=world_size,
        worker_addresses=addresses,
        epochs=epochs,
        batch_size=batch_size,
        seed=42,
        use_synthetic=True,
        shuffle=False,
        results_dir=results_dir,
    )


def run_distributed_job(world_size: int, epochs: int, batch_size: int, results_dir: str):
    addresses = [("127.0.0.1", BASE_PORT + i) for i in range(world_size)]

    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=_worker_entrypoint,
            args=(rank, world_size, addresses, epochs, batch_size, results_dir),
        )
        procs.append(p)

    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)

    for rank, p in enumerate(procs):
        if p.is_alive():
            p.terminate()
            raise TimeoutError(f"worker {rank} did not finish in time")
        if p.exitcode != 0:
            raise RuntimeError(f"worker {rank} exited with code {p.exitcode}")


def check_final_weights_match(world_size: int, results_dir: str, atol: float = 1e-4):
    histories = []
    for rank in range(world_size):
        path = Path(results_dir) / f"worker_{rank}_history.json"
        with open(path) as f:
            histories.append(json.load(f))

    ref = histories[0]["final_state_dict"]
    max_diffs = {}
    for rank in range(1, world_size):
        other = histories[rank]["final_state_dict"]
        for key in ref:
            a = np.array(ref[key])
            b = np.array(other[key])
            diff = np.max(np.abs(a - b))
            max_diffs[(rank, key)] = diff
            assert diff < atol, (
                f"rank {rank} diverged from rank 0 on parameter '{key}': "
                f"max abs diff = {diff}"
            )
    return max_diffs, histories


def main():
    global BASE_PORT
    for world_size in [2, 4]:
        print(f"\n=== running distributed job with world_size={world_size} ===")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_distributed_job(world_size=world_size, epochs=2, batch_size=8, results_dir=tmpdir)
            max_diffs, histories = check_final_weights_match(world_size, tmpdir)

            largest = max(max_diffs.values())
            print(f"world_size={world_size}: PASSED. largest cross-worker weight diff = {largest:.2e}")

            for h in histories:
                total_steps = len(h["loss_history"])
                print(
                    f"  rank {h['rank']}: {total_steps} steps, "
                    f"compute={h['compute_seconds']:.2f}s comm={h['comm_seconds']:.2f}s"
                )
        BASE_PORT += 100  # avoid port collision between iterations


if __name__ == "__main__":
    main()
