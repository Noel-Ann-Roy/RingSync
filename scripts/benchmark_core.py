# """
# Benchmark core -- parameterized, importable benchmarking logic.

# This is a refactor, not a rewrite: every calculation here (speedup,
# scaling efficiency, communication overhead, the timing methodology
# that excludes process-spawn overhead) is identical to what was in the
# original scripts/run_benchmark.py. The only change is that values
# which used to be hardcoded module-level constants (WORLD_SIZES,
# EPOCHS, BATCH_SIZE, ...) are now function arguments, so this can be
# called both from a CLI script and from the Streamlit Benchmark Studio
# with different configurations, without editing code either way.

# Used by:
#   - scripts/run_benchmark.py       (CLI entrypoint, unchanged behavior)
#   - dashboard/benchmark_studio.py  (interactive Streamlit app)
# """

# import json
# import multiprocessing as mp
# import sys
# import tempfile
# import time
# from pathlib import Path
# from typing import Callable, Optional

# import numpy as np

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# from ringsync.worker.worker_node import run_worker
# from baseline.train_single_process import train as run_baseline_train

# RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
# RESULTS_DIR.mkdir(exist_ok=True)

# DEFAULT_EPOCHS = 3
# DEFAULT_BATCH_SIZE = 128
# DEFAULT_LR = 0.01
# DEFAULT_NUM_SAMPLES = 8000
# DEFAULT_SEED = 42
# DEFAULT_WORLD_SIZES = [2, 4, 8]
# DEFAULT_BASE_PORT_START = 18000

# ProgressFn = Optional[Callable[[str], None]]


# def _emit(progress_fn: ProgressFn, message: str) -> None:
#     print(message)
#     if progress_fn is not None:
#         progress_fn(message)


# def _worker_entrypoint(rank, world_size, addresses, epochs, batch_size, lr, seed, num_samples, results_dir):
#     run_worker(
#         rank=rank, world_size=world_size, worker_addresses=addresses,
#         epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
#         use_synthetic=True, shuffle=False, results_dir=results_dir,
#         num_samples=num_samples,
#     )


# def _single_threaded_baseline_entrypoint(result_queue, epochs, batch_size, lr, seed, num_samples):
#     """
#     Runs in its own subprocess -- see the original run_benchmark.py's
#     docstring for why this isolation matters (torch.set_num_threads
#     should be set once, early, in a clean process; this also mirrors
#     how real distributed workers are launched, keeping the comparison
#     to a real worker apples-to-apples).
#     """
#     _, elapsed = run_baseline_train(
#         epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, use_synthetic=True,
#         num_samples=num_samples, num_threads=1,
#         output_filename="baseline_singlethread_loss_history.json",
#     )
#     result_queue.put(elapsed)


# def run_single_threaded_baseline(epochs, batch_size, lr, seed, num_samples) -> float:
#     result_queue = mp.Queue()
#     p = mp.Process(
#         target=_single_threaded_baseline_entrypoint,
#         args=(result_queue, epochs, batch_size, lr, seed, num_samples),
#     )
#     p.start()
#     p.join(timeout=180)
#     if p.is_alive():
#         p.terminate()
#         raise TimeoutError("single-threaded baseline did not finish in time")
#     if p.exitcode != 0:
#         raise RuntimeError(f"single-threaded baseline exited with code {p.exitcode}")
#     return result_queue.get()


# def run_distributed_config(
#     world_size: int, base_port: int, results_dir: str,
#     epochs: int, batch_size: int, lr: float, seed: int, num_samples: int,
# ) -> dict:
#     addresses = [("127.0.0.1", base_port + i) for i in range(world_size)]

#     procs = []
#     for rank in range(world_size):
#         p = mp.Process(
#             target=_worker_entrypoint,
#             args=(rank, world_size, addresses, epochs, batch_size, lr, seed, num_samples, results_dir),
#         )
#         procs.append(p)

#     # Excludes process-spawn/import overhead from the reported timing --
#     # see per-worker internal wall_clock_seconds below, which is the
#     # figure actually used. This outer measurement is kept only for
#     # reference/debugging.
#     outer_wall_start = time.time()
#     for p in procs:
#         p.start()
#     for p in procs:
#         p.join(timeout=300)
#     outer_wall_elapsed = time.time() - outer_wall_start

#     for rank, p in enumerate(procs):
#         if p.is_alive():
#             p.terminate()
#             raise TimeoutError(f"world_size={world_size}: worker {rank} did not finish in time")
#         if p.exitcode != 0:
#             raise RuntimeError(f"world_size={world_size}: worker {rank} exited with code {p.exitcode}")

#     per_worker_compute = []
#     per_worker_comm = []
#     per_worker_steps = []
#     per_worker_wall_clock = []
#     per_worker_epoch_losses = []  # list of {epoch: avg_loss} dicts, one per worker
#     for rank in range(world_size):
#         with open(Path(results_dir) / f"worker_{rank}_history.json") as f:
#             h = json.load(f)
#         n_steps = len(h["loss_history"])
#         per_worker_compute.append(h["compute_seconds"] / n_steps)
#         per_worker_comm.append(h["comm_seconds"] / n_steps)
#         per_worker_steps.append(n_steps)
#         per_worker_wall_clock.append(h["wall_clock_seconds"])

#         epoch_losses: dict = {}
#         for entry in h["loss_history"]:
#             epoch_losses.setdefault(entry["epoch"], []).append(entry["loss"])
#         per_worker_epoch_losses.append({ep: float(np.mean(vals)) for ep, vals in epoch_losses.items()})

#     # Average each epoch's loss across all workers -- this is the
#     # single "training loss vs epoch" curve for this world_size config,
#     # comparable directly against the baseline's own per-epoch curve.
#     all_epochs = sorted(per_worker_epoch_losses[0].keys())
#     epoch_avg_loss = [
#         float(np.mean([w[ep] for w in per_worker_epoch_losses])) for ep in all_epochs
#     ]

#     return {
#         "world_size": world_size,
#         "wall_clock_seconds": max(per_worker_wall_clock),
#         "outer_process_wall_clock_seconds": outer_wall_elapsed,
#         "avg_compute_per_step_s": float(np.mean(per_worker_compute)),
#         "avg_comm_per_step_s": float(np.mean(per_worker_comm)),
#         "steps_per_worker": per_worker_steps[0],
#         "epoch_avg_loss": epoch_avg_loss,
#     }


# def run_benchmark_suite(
#     world_sizes: "list[int]",
#     baseline_mode: str = "both",           # "multithread" | "singlethread" | "both"
#     epochs: int = DEFAULT_EPOCHS,
#     batch_size: int = DEFAULT_BATCH_SIZE,
#     lr: float = DEFAULT_LR,
#     num_samples: int = DEFAULT_NUM_SAMPLES,
#     seed: int = DEFAULT_SEED,
#     base_port_start: int = DEFAULT_BASE_PORT_START,
#     progress_fn: ProgressFn = None,
# ) -> dict:
#     """
#     Runs the full benchmark suite and returns the same result structure
#     as before (and writes it to results/benchmark_results.json), but
#     every parameter that used to be a hardcoded constant is now an
#     argument, and which baseline(s) to run is now a real choice rather
#     than always running both.
#     """
#     if baseline_mode not in ("multithread", "singlethread", "both"):
#         raise ValueError(f"baseline_mode must be 'multithread', 'singlethread', or 'both', got {baseline_mode!r}")

#     baseline_elapsed = None
#     baseline_singlethread_elapsed = None
#     baseline_epoch_avg_loss = None
#     baseline_singlethread_epoch_avg_loss = None

#     def _epoch_avg_loss(loss_history):
#         by_epoch: dict = {}
#         for entry in loss_history:
#             by_epoch.setdefault(entry["epoch"], []).append(entry["loss"])
#         return [float(np.mean(by_epoch[ep])) for ep in sorted(by_epoch.keys())]

#     if baseline_mode in ("multithread", "both"):
#         _emit(progress_fn, "Running single-process baseline (multi-threaded, PyTorch default)...")
#         baseline_loss_history, baseline_elapsed = run_baseline_train(
#             epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, use_synthetic=True,
#             num_samples=num_samples,
#         )
#         baseline_epoch_avg_loss = _epoch_avg_loss(baseline_loss_history)
#         _emit(progress_fn, f"Baseline (multi-threaded) wall-clock: {baseline_elapsed:.2f}s")

#     if baseline_mode in ("singlethread", "both"):
#         _emit(progress_fn, "Running single-process baseline (single-threaded, apples-to-apples with a worker)...")
#         baseline_singlethread_elapsed = run_single_threaded_baseline(epochs, batch_size, lr, seed, num_samples)
#         # train() runs in a subprocess for the single-threaded case, so its
#         # loss_history return value isn't directly available here -- but it
#         # persists the same data to disk, which we read back.
#         with open(RESULTS_DIR / "baseline_singlethread_loss_history.json") as f:
#             baseline_singlethread_epoch_avg_loss = _epoch_avg_loss(json.load(f)["loss_history"])
#         _emit(progress_fn, f"Baseline (single-threaded) wall-clock: {baseline_singlethread_elapsed:.2f}s")

#     results = [{
#         "world_size": 1,
#         "wall_clock_seconds": baseline_elapsed if baseline_elapsed is not None else baseline_singlethread_elapsed,
#         "avg_compute_per_step_s": None,
#         "avg_comm_per_step_s": 0.0,
#         "steps_per_worker": None,
#     }]

#     base_port = base_port_start
#     for world_size in world_sizes:
#         _emit(progress_fn, f"Running distributed config: world_size={world_size} ...")
#         with tempfile.TemporaryDirectory() as tmpdir:
#             r = run_distributed_config(world_size, base_port, tmpdir, epochs, batch_size, lr, seed, num_samples)
#         _emit(
#             progress_fn,
#             f"world_size={world_size}: training_wall_clock={r['wall_clock_seconds']:.2f}s  "
#             f"avg_compute/step={r['avg_compute_per_step_s']:.4f}s  "
#             f"avg_comm/step={r['avg_comm_per_step_s']:.4f}s",
#         )
#         results.append(r)
#         base_port += 200

#     for r in results:
#         r["ideal_speedup"] = float(r["world_size"])
#         if baseline_elapsed is not None:
#             r["speedup_vs_baseline"] = baseline_elapsed / r["wall_clock_seconds"]
#             r["scaling_efficiency"] = r["speedup_vs_baseline"] / r["world_size"]
#         if baseline_singlethread_elapsed is not None:
#             r["speedup_vs_singlethread_baseline"] = baseline_singlethread_elapsed / r["wall_clock_seconds"]
#             r["scaling_efficiency_singlethread"] = r["speedup_vs_singlethread_baseline"] / r["world_size"]

#     out_data = {
#         "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
#         "world_sizes": world_sizes, "baseline_mode": baseline_mode,
#         "baseline_multithread_wall_clock_seconds": baseline_elapsed,
#         "baseline_singlethread_wall_clock_seconds": baseline_singlethread_elapsed,
#         "baseline_multithread_epoch_avg_loss": baseline_epoch_avg_loss,
#         "baseline_singlethread_epoch_avg_loss": baseline_singlethread_epoch_avg_loss,
#         "results": results,
#     }

#     out_path = RESULTS_DIR / "benchmark_results.json"
#     with open(out_path, "w") as f:
#         json.dump(out_data, f, indent=2)
#     _emit(progress_fn, f"Saved benchmark results -> {out_path}")

#     return out_data


"""
Benchmark core -- parameterized, importable benchmarking logic.

This is a refactor, not a rewrite: every calculation here (speedup,
scaling efficiency, communication overhead, the timing methodology
that excludes process-spawn overhead) is identical to what was in the
original scripts/run_benchmark.py. The only change is that values
which used to be hardcoded module-level constants (WORLD_SIZES,
EPOCHS, BATCH_SIZE, ...) are now function arguments, so this can be
called both from a CLI script and from the Streamlit Benchmark Studio
with different configurations, without editing code either way.

Used by:
  - scripts/run_benchmark.py       (CLI entrypoint, unchanged behavior)
  - dashboard/benchmark_studio.py  (interactive Streamlit app)
"""

import json
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ringsync.worker.worker_node import run_worker
from baseline.train_single_process import train as run_baseline_train
from scripts.experiment_utils import generate_experiment_id, get_experiment_dir

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 0.01
DEFAULT_NUM_SAMPLES = 8000
DEFAULT_SEED = 42
DEFAULT_WORLD_SIZES = [2, 4, 8]
DEFAULT_BASE_PORT_START = 18000

ProgressFn = Optional[Callable[[str], None]]


def _emit(progress_fn: ProgressFn, message: str) -> None:
    print(message)
    if progress_fn is not None:
        progress_fn(message)


def _worker_entrypoint(rank, world_size, addresses, epochs, batch_size, lr, seed, num_samples, results_dir):
    run_worker(
        rank=rank, world_size=world_size, worker_addresses=addresses,
        epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
        use_synthetic=True, shuffle=False, results_dir=results_dir,
        num_samples=num_samples,
    )


def _single_threaded_baseline_entrypoint(result_queue, epochs, batch_size, lr, seed, num_samples, output_dir):
    """
    Runs in its own subprocess -- see the original run_benchmark.py's
    docstring for why this isolation matters (torch.set_num_threads
    should be set once, early, in a clean process; this also mirrors
    how real distributed workers are launched, keeping the comparison
    to a real worker apples-to-apples).
    """
    _, elapsed = run_baseline_train(
        epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, use_synthetic=True,
        num_samples=num_samples, num_threads=1,
        output_filename="baseline_singlethread_loss_history.json",
        output_dir=output_dir, progress_config="baseline_singlethread",
    )
    result_queue.put(elapsed)


def run_single_threaded_baseline(epochs, batch_size, lr, seed, num_samples, output_dir) -> float:
    result_queue = mp.Queue()
    p = mp.Process(
        target=_single_threaded_baseline_entrypoint,
        args=(result_queue, epochs, batch_size, lr, seed, num_samples, output_dir),
    )
    p.start()
    p.join(timeout=180)
    if p.is_alive():
        p.terminate()
        raise TimeoutError("single-threaded baseline did not finish in time")
    if p.exitcode != 0:
        raise RuntimeError(f"single-threaded baseline exited with code {p.exitcode}")
    return result_queue.get()


def run_distributed_config(
    world_size: int, base_port: int, results_dir: str,
    epochs: int, batch_size: int, lr: float, seed: int, num_samples: int,
) -> dict:
    addresses = [("127.0.0.1", base_port + i) for i in range(world_size)]

    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=_worker_entrypoint,
            args=(rank, world_size, addresses, epochs, batch_size, lr, seed, num_samples, results_dir),
        )
        procs.append(p)

    # Excludes process-spawn/import overhead from the reported timing --
    # see per-worker internal wall_clock_seconds below, which is the
    # figure actually used. This outer measurement is kept only for
    # reference/debugging.
    outer_wall_start = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
    outer_wall_elapsed = time.time() - outer_wall_start

    for rank, p in enumerate(procs):
        if p.is_alive():
            p.terminate()
            raise TimeoutError(f"world_size={world_size}: worker {rank} did not finish in time")
        if p.exitcode != 0:
            raise RuntimeError(f"world_size={world_size}: worker {rank} exited with code {p.exitcode}")

    per_worker_compute = []
    per_worker_comm = []
    per_worker_steps = []
    per_worker_wall_clock = []
    per_worker_epoch_losses = []  # list of {epoch: avg_loss} dicts, one per worker
    for rank in range(world_size):
        with open(Path(results_dir) / f"worker_{rank}_history.json") as f:
            h = json.load(f)
        n_steps = len(h["loss_history"])
        per_worker_compute.append(h["compute_seconds"] / n_steps)
        per_worker_comm.append(h["comm_seconds"] / n_steps)
        per_worker_steps.append(n_steps)
        per_worker_wall_clock.append(h["wall_clock_seconds"])

        epoch_losses: dict = {}
        for entry in h["loss_history"]:
            epoch_losses.setdefault(entry["epoch"], []).append(entry["loss"])
        per_worker_epoch_losses.append({ep: float(np.mean(vals)) for ep, vals in epoch_losses.items()})

    # Average each epoch's loss across all workers -- this is the
    # single "training loss vs epoch" curve for this world_size config,
    # comparable directly against the baseline's own per-epoch curve.
    all_epochs = sorted(per_worker_epoch_losses[0].keys())
    epoch_avg_loss = [
        float(np.mean([w[ep] for w in per_worker_epoch_losses])) for ep in all_epochs
    ]

    return {
        "world_size": world_size,
        "wall_clock_seconds": max(per_worker_wall_clock),
        "outer_process_wall_clock_seconds": outer_wall_elapsed,
        "avg_compute_per_step_s": float(np.mean(per_worker_compute)),
        "avg_comm_per_step_s": float(np.mean(per_worker_comm)),
        "steps_per_worker": per_worker_steps[0],
        "epoch_avg_loss": epoch_avg_loss,
    }


def run_benchmark_suite(
    world_sizes: "list[int]",
    baseline_mode: str = "both",           # "multithread" | "singlethread" | "both"
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
    base_port_start: int = DEFAULT_BASE_PORT_START,
    progress_fn: ProgressFn = None,
    experiment_id: "str | None" = None,
) -> dict:
    """
    Runs the full benchmark suite and returns the same result structure
    as before, but every parameter that used to be a hardcoded constant
    is now an argument, and which baseline(s) to run is now a real
    choice rather than always running both.

    Each run gets an experiment_id (auto-generated as RS-YYYYMMDD-NNN
    if not provided) and all of its output -- benchmark_results.json,
    baseline loss histories -- is saved under
    results/<experiment_id>/ rather than the flat results/ folder, so
    runs never overwrite each other and can be listed/compared later
    (see scripts/experiment_utils.py and the Benchmark History panel).
    """
    if baseline_mode not in ("multithread", "singlethread", "both"):
        raise ValueError(f"baseline_mode must be 'multithread', 'singlethread', or 'both', got {baseline_mode!r}")

    if experiment_id is None:
        experiment_id = generate_experiment_id()
    results_dir = get_experiment_dir(experiment_id)
    _emit(progress_fn, f"Experiment ID: {experiment_id}")
    print(f'RS_PROGRESS|{{"type":"experiment_id","experiment_id":"{experiment_id}"}}')

    # Total config count (for overall progress), in the order they'll run.
    total_configs = len(world_sizes) + (1 if baseline_mode in ("multithread", "both") else 0) \
        + (1 if baseline_mode in ("singlethread", "both") else 0)
    print(f'RS_PROGRESS|{{"type":"plan","total_configs":{total_configs},'
          f'"world_sizes":{json.dumps(world_sizes)},"baseline_mode":"{baseline_mode}"}}')

    baseline_elapsed = None
    baseline_singlethread_elapsed = None
    baseline_epoch_avg_loss = None
    baseline_singlethread_epoch_avg_loss = None

    def _epoch_avg_loss(loss_history):
        by_epoch: dict = {}
        for entry in loss_history:
            by_epoch.setdefault(entry["epoch"], []).append(entry["loss"])
        return [float(np.mean(by_epoch[ep])) for ep in sorted(by_epoch.keys())]

    if baseline_mode in ("multithread", "both"):
        print('RS_PROGRESS|{"type":"config_start","config":"baseline"}')
        _emit(progress_fn, "Running single-process baseline (multi-threaded, PyTorch default)...")
        baseline_loss_history, baseline_elapsed = run_baseline_train(
            epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, use_synthetic=True,
            num_samples=num_samples, output_dir=results_dir, progress_config="baseline",
        )
        baseline_epoch_avg_loss = _epoch_avg_loss(baseline_loss_history)
        _emit(progress_fn, f"Baseline (multi-threaded) wall-clock: {baseline_elapsed:.2f}s")
        print('RS_PROGRESS|{"type":"config_done","config":"baseline"}')

    if baseline_mode in ("singlethread", "both"):
        print('RS_PROGRESS|{"type":"config_start","config":"baseline_singlethread"}')
        _emit(progress_fn, "Running single-process baseline (single-threaded, apples-to-apples with a worker)...")
        baseline_singlethread_elapsed = run_single_threaded_baseline(
            epochs, batch_size, lr, seed, num_samples, results_dir,
        )
        # train() runs in a subprocess for the single-threaded case, so its
        # loss_history return value isn't directly available here -- but it
        # persists the same data to disk, which we read back.
        with open(results_dir / "baseline_singlethread_loss_history.json") as f:
            baseline_singlethread_epoch_avg_loss = _epoch_avg_loss(json.load(f)["loss_history"])
        _emit(progress_fn, f"Baseline (single-threaded) wall-clock: {baseline_singlethread_elapsed:.2f}s")
        print('RS_PROGRESS|{"type":"config_done","config":"baseline_singlethread"}')

    results = [{
        "world_size": 1,
        "wall_clock_seconds": baseline_elapsed if baseline_elapsed is not None else baseline_singlethread_elapsed,
        "avg_compute_per_step_s": None,
        "avg_comm_per_step_s": 0.0,
        "steps_per_worker": None,
    }]

    base_port = base_port_start
    for world_size in world_sizes:
        config_tag = f"world_size_{world_size}"
        print(f'RS_PROGRESS|{{"type":"config_start","config":"{config_tag}"}}')
        _emit(progress_fn, f"Running distributed config: world_size={world_size} ...")
        with tempfile.TemporaryDirectory() as tmpdir:
            r = run_distributed_config(world_size, base_port, tmpdir, epochs, batch_size, lr, seed, num_samples)
        _emit(
            progress_fn,
            f"world_size={world_size}: training_wall_clock={r['wall_clock_seconds']:.2f}s  "
            f"avg_compute/step={r['avg_compute_per_step_s']:.4f}s  "
            f"avg_comm/step={r['avg_comm_per_step_s']:.4f}s",
        )
        print(f'RS_PROGRESS|{{"type":"config_done","config":"{config_tag}"}}')
        results.append(r)
        base_port += 200

    for r in results:
        r["ideal_speedup"] = float(r["world_size"])
        if baseline_elapsed is not None:
            r["speedup_vs_baseline"] = baseline_elapsed / r["wall_clock_seconds"]
            r["scaling_efficiency"] = r["speedup_vs_baseline"] / r["world_size"]
        if baseline_singlethread_elapsed is not None:
            r["speedup_vs_singlethread_baseline"] = baseline_singlethread_elapsed / r["wall_clock_seconds"]
            r["scaling_efficiency_singlethread"] = r["speedup_vs_singlethread_baseline"] / r["world_size"]

    out_data = {
        "experiment_id": experiment_id,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
        "world_sizes": world_sizes, "baseline_mode": baseline_mode,
        "baseline_multithread_wall_clock_seconds": baseline_elapsed,
        "baseline_singlethread_wall_clock_seconds": baseline_singlethread_elapsed,
        "baseline_multithread_epoch_avg_loss": baseline_epoch_avg_loss,
        "baseline_singlethread_epoch_avg_loss": baseline_singlethread_epoch_avg_loss,
        "results": results,
    }

    out_path = results_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    _emit(progress_fn, f"Saved benchmark results -> {out_path}")
    print(f'RS_PROGRESS|{{"type":"all_done","experiment_id":"{experiment_id}"}}')

    return out_data