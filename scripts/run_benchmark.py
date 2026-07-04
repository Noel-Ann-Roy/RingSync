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