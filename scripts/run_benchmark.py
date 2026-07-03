# # """
# # Benchmarking harness.

# # Runs the single-process baseline once, then the real distributed
# # training job (actual separate OS processes, actual TCP ring) at
# # several world sizes, and measures:

# #   1. Speedup vs. baseline wall-clock time, compared against ideal
# #      linear speedup (speedup = N). The GAP between actual and ideal is
# #      the interesting result -- it's caused by communication overhead,
# #      and this script quantifies exactly how much.
# #   2. Compute time vs. communication time per step, at each world size.
# #      As world_size grows, each worker's local compute shrinks (smaller
# #      shard... no, wait: same total dataset, so per-worker steps shrink,
# #      but comm time grows because more ring hops are needed (2*(N-1)
# #      messages per step) -- this is exactly why speedup saturates rather
# #      than scaling linearly forever.

# # All configs process the SAME total dataset per epoch (2000 synthetic
# # samples), just split across a different number of workers -- so wall-
# # clock comparisons are apples-to-apples: they all measure "how long to
# # get through one epoch of the same data."

# # Run: PYTHONPATH=. python3 scripts/run_benchmark.py
# # """

# # import json
# # import multiprocessing as mp
# # import sys
# # import tempfile
# # import time
# # from pathlib import Path

# # import numpy as np

# # sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# # from ringsync.worker.worker_node import run_worker
# # from baseline.train_single_process import train as run_baseline_train

# # RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
# # RESULTS_DIR.mkdir(exist_ok=True)

# # EPOCHS = 3
# # BATCH_SIZE = 128     # bumped up from 16: at batch=16 this model's forward/backward
# #                       # is so cheap that PyTorch's built-in multi-threading inside
# #                       # a single process already saturates available compute --
# #                       # there's no compute-bound bottleneck left for distribution
# #                       # to solve. A larger batch makes each step's compute cost
# #                       # substantial enough that splitting work across workers
# #                       # actually pays for its communication overhead -- which is
# #                       # the realistic regime real distributed training operates in.
# # NUM_SAMPLES = 8000    # scaled up to match: keeps a reasonable number of steps
# #                       # per epoch even with the larger batch size and up to 8-way sharding
# # SEED = 42
# # WORLD_SIZES = [2, 4, 8]
# # BASE_PORT_START = 18000


# # def _worker_entrypoint(rank, world_size, addresses, epochs, batch_size, results_dir):
# #     run_worker(
# #         rank=rank, world_size=world_size, worker_addresses=addresses,
# #         epochs=epochs, batch_size=batch_size, seed=SEED,
# #         use_synthetic=True, shuffle=False, results_dir=results_dir,
# #         num_samples=NUM_SAMPLES,
# #     )


# # def _single_threaded_baseline_entrypoint(result_queue):
# #     """
# #     Runs in its own subprocess (not the main script process). This
# #     isolation matters: torch.set_num_threads() is generally meant to be
# #     called once, early, before any parallel work has happened in that
# #     process -- calling it after the multi-threaded baseline has already
# #     run in the same process risks inconsistent behavior. Running this
# #     in a fresh subprocess sidesteps that entirely and mirrors exactly
# #     how the real distributed workers are launched (also fresh
# #     subprocesses, also pinned to 1 thread), which is the point: this
# #     baseline should differ from a distributed worker ONLY in "1 process
# #     doing everything" vs "N processes splitting the work" -- not in any
# #     other incidental way.
# #     """
# #     _, elapsed = run_baseline_train(
# #         epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED, use_synthetic=True,
# #         num_samples=NUM_SAMPLES, num_threads=1,
# #         output_filename="baseline_singlethread_loss_history.json",
# #     )
# #     result_queue.put(elapsed)


# # def run_single_threaded_baseline() -> float:
# #     result_queue = mp.Queue()
# #     p = mp.Process(target=_single_threaded_baseline_entrypoint, args=(result_queue,))
# #     p.start()
# #     p.join(timeout=180)
# #     if p.is_alive():
# #         p.terminate()
# #         raise TimeoutError("single-threaded baseline did not finish in time")
# #     if p.exitcode != 0:
# #         raise RuntimeError(f"single-threaded baseline exited with code {p.exitcode}")
# #     return result_queue.get()


# # def run_distributed_config(world_size: int, base_port: int, results_dir: str) -> dict:
# #     addresses = [("127.0.0.1", base_port + i) for i in range(world_size)]

# #     procs = []
# #     for rank in range(world_size):
# #         p = mp.Process(
# #             target=_worker_entrypoint,
# #             args=(rank, world_size, addresses, EPOCHS, BATCH_SIZE, results_dir),
# #         )
# #         procs.append(p)

# #     # This wraps process spawn -> join, which on some platforms (notably
# #     # Windows, where multiprocessing defaults to "spawn" instead of
# #     # "fork") includes each process re-importing torch from scratch --
# #     # several seconds of fixed overhead per process that has nothing to
# #     # do with training speed. We still enforce a generous timeout here,
# #     # but we do NOT use this figure as the reported wall-clock time.
# #     outer_wall_start = time.time()
# #     for p in procs:
# #         p.start()
# #     for p in procs:
# #         p.join(timeout=180)
# #     outer_wall_elapsed = time.time() - outer_wall_start

# #     for rank, p in enumerate(procs):
# #         if p.is_alive():
# #             p.terminate()
# #             raise TimeoutError(f"world_size={world_size}: worker {rank} did not finish in time")
# #         if p.exitcode != 0:
# #             raise RuntimeError(f"world_size={world_size}: worker {rank} exited with code {p.exitcode}")

# #     per_worker_compute = []
# #     per_worker_comm = []
# #     per_worker_steps = []
# #     per_worker_wall_clock = []
# #     for rank in range(world_size):
# #         with open(Path(results_dir) / f"worker_{rank}_history.json") as f:
# #             h = json.load(f)
# #         n_steps = len(h["loss_history"])
# #         per_worker_compute.append(h["compute_seconds"] / n_steps)
# #         per_worker_comm.append(h["comm_seconds"] / n_steps)
# #         per_worker_steps.append(n_steps)
# #         per_worker_wall_clock.append(h["wall_clock_seconds"])

# #     return {
# #         "world_size": world_size,
# #         # Each worker times its OWN training loop internally, starting
# #         # after imports and ring setup are already done -- this is the
# #         # number that actually reflects training speed. Synchronous
# #         # training is bounded by the slowest worker, so we take the max
# #         # across workers, not the average.
# #         "wall_clock_seconds": max(per_worker_wall_clock),
# #         "outer_process_wall_clock_seconds": outer_wall_elapsed,  # kept for reference/debugging only
# #         "avg_compute_per_step_s": float(np.mean(per_worker_compute)),
# #         "avg_comm_per_step_s": float(np.mean(per_worker_comm)),
# #         "steps_per_worker": per_worker_steps[0],
# #     }


# # def run_full_benchmark():
# #     print("=== Running single-process baseline (multi-threaded, PyTorch default) ===")
# #     _, baseline_elapsed = run_baseline_train(
# #         epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED, use_synthetic=True,
# #         num_samples=NUM_SAMPLES,
# #     )
# #     print(f"Baseline (multi-threaded) wall-clock: {baseline_elapsed:.2f}s\n")

# #     print("=== Running single-process baseline (single-threaded, apples-to-apples with a worker) ===")
# #     baseline_singlethread_elapsed = run_single_threaded_baseline()
# #     print(f"Baseline (single-threaded) wall-clock: {baseline_singlethread_elapsed:.2f}s\n")

# #     results = [{
# #         "world_size": 1,
# #         "wall_clock_seconds": baseline_elapsed,
# #         "avg_compute_per_step_s": None,
# #         "avg_comm_per_step_s": 0.0,
# #         "steps_per_worker": None,
# #     }]

# #     base_port = BASE_PORT_START
# #     for world_size in WORLD_SIZES:
# #         print(f"=== Running distributed config: world_size={world_size} ===")
# #         with tempfile.TemporaryDirectory() as tmpdir:
# #             r = run_distributed_config(world_size, base_port, tmpdir)
# #         print(
# #             f"world_size={world_size}: training_wall_clock={r['wall_clock_seconds']:.2f}s  "
# #             f"(process spawn+import overhead included: {r['outer_process_wall_clock_seconds']:.2f}s)  "
# #             f"avg_compute/step={r['avg_compute_per_step_s']:.4f}s  "
# #             f"avg_comm/step={r['avg_comm_per_step_s']:.4f}s\n"
# #         )
# #         results.append(r)
# #         base_port += 200  # avoid port collisions between successive runs

# #     for r in results:
# #         # Speedup against the standard (multi-threaded) baseline -- what
# #         # a typical single-process run looks like by default.
# #         r["speedup_vs_baseline"] = baseline_elapsed / r["wall_clock_seconds"]
# #         # Speedup against the single-threaded baseline -- isolates the
# #         # pure data-parallelism effect, with threading held equal on
# #         # both sides. This number should be noticeably higher and
# #         # closer to the ideal line, since it's not fighting an
# #         # artificially crippled distributed side against an artificially
# #         # advantaged (multi-threaded) baseline.
# #         r["speedup_vs_singlethread_baseline"] = baseline_singlethread_elapsed / r["wall_clock_seconds"]
# #         r["ideal_speedup"] = float(r["world_size"])
# #         r["scaling_efficiency"] = r["speedup_vs_baseline"] / r["world_size"]
# #         r["scaling_efficiency_singlethread"] = r["speedup_vs_singlethread_baseline"] / r["world_size"]

# #     out_path = RESULTS_DIR / "benchmark_results.json"
# #     with open(out_path, "w") as f:
# #         json.dump({
# #             "epochs": EPOCHS, "batch_size": BATCH_SIZE,
# #             "baseline_multithread_wall_clock_seconds": baseline_elapsed,
# #             "baseline_singlethread_wall_clock_seconds": baseline_singlethread_elapsed,
# #             "results": results,
# #         }, f, indent=2)
# #     print(f"Saved benchmark results -> {out_path}")

# #     return results


# # if __name__ == "__main__":
# #     run_full_benchmark()


# """
# CLI entrypoint for the RingSync benchmark suite.

# This is now a thin wrapper around scripts/benchmark_core.py -- all the
# actual benchmarking logic (timing methodology, speedup/efficiency
# calculations, subprocess handling) lives there and is unchanged from
# before. This file just parses CLI args and calls it, and also backs
# the interactive Streamlit Benchmark Studio (dashboard/benchmark_studio.py).

# Run: PYTHONPATH=. python3 scripts/run_benchmark.py
#      PYTHONPATH=. python3 scripts/run_benchmark.py --world-sizes "2,4,8,10,12,13,14,15"
#      PYTHONPATH=. python3 scripts/run_benchmark.py --epochs 5 --batch-size 64
# """

# import argparse
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# from scripts.benchmark_core import (
#     run_benchmark_suite, DEFAULT_WORLD_SIZES, DEFAULT_EPOCHS,
#     DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_NUM_SAMPLES,
# )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run the RingSync benchmark suite.")
#     parser.add_argument(
#         "--world-sizes", type=str, default=None,
#         help='Comma-separated list of world sizes to test, e.g. "2,4,8,10,12,13,14,15". '
#              f"Defaults to {DEFAULT_WORLD_SIZES} if not given.",
#     )
#     parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
#     parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
#     parser.add_argument("--lr", type=float, default=DEFAULT_LR)
#     parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
#     parser.add_argument(
#         "--baseline", type=str, default="both", choices=["multithread", "singlethread", "both"],
#         help="Which baseline(s) to run and compare against (default: both).",
#     )
#     args = parser.parse_args()

#     world_sizes = (
#         [int(x.strip()) for x in args.world_sizes.split(",")]
#         if args.world_sizes is not None else DEFAULT_WORLD_SIZES
#     )

#     run_benchmark_suite(
#         world_sizes=world_sizes,
#         baseline_mode=args.baseline,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         lr=args.lr,
#         num_samples=args.num_samples,
#     )


"""
CLI entrypoint for the RingSync benchmark suite.

This is now a thin wrapper around scripts/benchmark_core.py -- all the
actual benchmarking logic (timing methodology, speedup/efficiency
calculations, subprocess handling) lives there and is unchanged from
before. This file just parses CLI args and calls it, and also backs
the interactive Streamlit Benchmark Studio (dashboard/benchmark_studio.py).

Run: PYTHONPATH=. python3 scripts/run_benchmark.py
     PYTHONPATH=. python3 scripts/run_benchmark.py --world-sizes "2,4,8,10,12,13,14,15"
     PYTHONPATH=. python3 scripts/run_benchmark.py --epochs 5 --batch-size 64
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_core import (
    run_benchmark_suite, DEFAULT_WORLD_SIZES, DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_NUM_SAMPLES,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RingSync benchmark suite.")
    parser.add_argument(
        "--world-sizes", type=str, default=None,
        help='Comma-separated list of world sizes to test, e.g. "2,4,8,10,12,13,14,15". '
             f"Defaults to {DEFAULT_WORLD_SIZES} if not given.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--baseline", type=str, default="both", choices=["multithread", "singlethread", "both"],
        help="Which baseline(s) to run and compare against (default: both).",
    )
    parser.add_argument(
        "--experiment-id", type=str, default=None,
        help="Reuse a specific experiment ID (e.g. RS-20260703-001) instead of auto-generating one.",
    )
    args = parser.parse_args()

    world_sizes = (
        [int(x.strip()) for x in args.world_sizes.split(",")]
        if args.world_sizes is not None else DEFAULT_WORLD_SIZES
    )

    run_benchmark_suite(
        world_sizes=world_sizes,
        baseline_mode=args.baseline,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
        experiment_id=args.experiment_id,
    )