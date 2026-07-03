# """
# Generates the benchmark charts from results/benchmark_results.json:

#   1. speedup_vs_multithread_baseline.png -- actual speedup vs. the
#      default (multi-threaded) single-process baseline, vs. ideal
#   2. speedup_vs_singlethread_baseline.png -- actual speedup vs. the
#      single-threaded baseline (apples-to-apples threading), vs. ideal
#   3. compute_vs_comm_chart.png -- stacked bar of avg compute vs comm
#      time per step, across world sizes
#   4. communication_overhead_pct.png -- communication time as a % of
#      total step time, vs. number of workers -- this is the single
#      clearest number for explaining WHY the speedup curve saturates:
#      as this percentage climbs, more and more of each step is being
#      spent on synchronization rather than useful compute, which is
#      exactly why speedup falls further below ideal as world_size grows.

# Run AFTER scripts/run_benchmark.py:
#     PYTHONPATH=. python3 scripts/plot_benchmark.py

# IMPORTANT: if this machine has only 1 CPU core (check with `nproc`),
# the resulting speedup curves will show sub-linear or even negative
# speedup -- not because the implementation is wrong, but because
# spawning N worker processes on 1 core gives zero real parallelism.
# Re-run scripts/run_benchmark.py on a multi-core machine to get
# representative curves before using these charts in a portfolio/interview
# context.
# """

# import json
# import sys
# from pathlib import Path

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# def load_results():
#     with open(RESULTS_DIR / "benchmark_results.json") as f:
#         return json.load(f)


# STANDARD_FIGSIZE = (7, 5)   # for charts displayed side-by-side, 2 per row
#                               # (Speedup x2, Compute vs. Comm + Overhead)

# WIDE_FIGSIZE = (13, 5)      # for charts displayed full-width, alone
#                               # (Scaling Efficiency, Training Loss).
#                               # Same height as STANDARD_FIGSIZE, roughly
#                               # double the width -- since Streamlit scales
#                               # each image to fill its container while
#                               # preserving aspect ratio, a full-width slot
#                               # is roughly 2x wider than a half-width one,
#                               # so matching the rendered HEIGHT across both
#                               # layouts requires widening (not just scaling)
#                               # the figure, not simply reusing the same
#                               # width/height ratio.


# def _plot_single_speedup_curve(world_sizes, actual, ideal, label, color, title, out_filename, cpu_count_note=""):
#     fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
#     ax.plot(world_sizes, ideal, "--", color="gray", label="Ideal linear speedup", linewidth=1.5)
#     ax.plot(world_sizes, actual, "o-", color=color, label=label, linewidth=2, markersize=7)

#     ax.set_xlabel("Number of workers (world size)")
#     ax.set_ylabel("Speedup vs. single-process baseline")
#     full_title = title
#     if cpu_count_note:
#         full_title += f"\n({cpu_count_note})"
#     ax.set_title(full_title, fontsize=10)
#     ax.set_xticks(world_sizes)
#     ax.legend(fontsize=8)
#     ax.grid(True, alpha=0.3)

#     caption = (
#         "Only RingSync performance is plotted. Speedup = Baseline Runtime \u00f7 RingSync Runtime.\n"
#         "Baseline is a fixed single-process measurement, used only as the reference, not plotted."
#     )
#     fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7, color="#444444", wrap=True)

#     out_path = RESULTS_DIR / out_filename
#     fig.tight_layout(rect=[0, 0.10, 1, 1])  # reserve space at bottom for the caption
#     fig.savefig(out_path, dpi=150)
#     print(f"Saved -> {out_path}")
#     return fig


# def plot_speedup_vs_multithread(data, cpu_count_note: str = ""):
#     if data.get("baseline_multithread_wall_clock_seconds") is None:
#         print("Skipping speedup_vs_multithread_baseline.png -- multi-threaded baseline was not run.")
#         return None

#     results = [r for r in data["results"] if r["world_size"] >= 2]
#     world_sizes = [r["world_size"] for r in results]
#     actual = [r["speedup_vs_baseline"] for r in results]
#     ideal = [r["ideal_speedup"] for r in results]

#     fig = _plot_single_speedup_curve(
#         world_sizes, actual, ideal,
#         label="RingSync Speedup (relative to Default PyTorch)",
#         color="#2563eb",
#         title="RingSync Speedup vs. Default Multi-threaded PyTorch Baseline",
#         out_filename="speedup_vs_multithread_baseline.png",
#         cpu_count_note=cpu_count_note,
#     )
#     return fig


# def plot_speedup_vs_singlethread(data, cpu_count_note: str = ""):
#     if data.get("baseline_singlethread_wall_clock_seconds") is None:
#         print("Skipping speedup_vs_singlethread_baseline.png -- single-threaded baseline was not run.")
#         return None

#     results = [r for r in data["results"] if r["world_size"] >= 2]
#     world_sizes = [r["world_size"] for r in results]
#     actual = [r["speedup_vs_singlethread_baseline"] for r in results]
#     ideal = [r["ideal_speedup"] for r in results]

#     fig = _plot_single_speedup_curve(
#         world_sizes, actual, ideal,
#         label="RingSync Speedup (relative to Single-threaded Baseline)",
#         color="#16a34a",
#         title="RingSync Speedup vs. Single-threaded Baseline",
#         out_filename="speedup_vs_singlethread_baseline.png",
#         cpu_count_note=cpu_count_note,
#     )
#     return fig


# def plot_compute_vs_comm(data):
#     results = [r for r in data["results"] if r["world_size"] > 1]  # baseline has no comm phase
#     world_sizes = [r["world_size"] for r in results]
#     compute = [r["avg_compute_per_step_s"] for r in results]
#     comm = [r["avg_comm_per_step_s"] for r in results]

#     fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
#     x = range(len(world_sizes))
#     ax.bar(x, compute, label="Compute time / step", color="#2563eb")
#     ax.bar(x, comm, bottom=compute, label="Communication time / step", color="#f59e0b")

#     ax.set_xticks(list(x))
#     ax.set_xticklabels([str(ws) for ws in world_sizes])
#     ax.set_xlabel("Number of workers (world size)")
#     ax.set_ylabel("Avg. time per training step (s)")
#     ax.set_title("RingSync: Compute vs. Communication Time per Step", fontsize=10)
#     ax.legend(fontsize=8)
#     ax.grid(True, alpha=0.3, axis="y")

#     out_path = RESULTS_DIR / "compute_vs_comm_chart.png"
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=150)
#     print(f"Saved -> {out_path}")
#     return fig


# def plot_communication_overhead_pct(data):
#     """
#     Communication overhead (%) = comm_time / (compute_time + comm_time) * 100,
#     per step, at each world size. This single number is the cleanest way
#     to explain WHY the speedup curve bends away from ideal: as world_size
#     grows, the ring has more hops (2*(N-1) messages per step), so a
#     growing share of every step is spent synchronizing rather than
#     computing -- directly visualizing the mechanism behind the
#     saturating speedup curves in the other two charts.
#     """
#     results = [r for r in data["results"] if r["world_size"] > 1]
#     world_sizes = [r["world_size"] for r in results]
#     overhead_pct = [
#         100.0 * r["avg_comm_per_step_s"] / (r["avg_compute_per_step_s"] + r["avg_comm_per_step_s"])
#         for r in results
#     ]

#     fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
#     ax.plot(world_sizes, overhead_pct, "o-", color="#dc2626", linewidth=2, markersize=8)

#     for ws, pct in zip(world_sizes, overhead_pct):
#         ax.annotate(f"{pct:.1f}%", (ws, pct), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

#     ax.set_xlabel("Number of workers (world size)")
#     ax.set_ylabel("Communication overhead (% of step time)")
#     ax.set_title("Communication Overhead (% of Training Step Time)\nvs. Number of Workers", fontsize=10)
#     ax.set_xticks(world_sizes)
#     ax.set_ylim(bottom=0)
#     ax.grid(True, alpha=0.3)

#     caption = (
#         "Overhead % = (Communication Time \u00f7 Total Step Time) \u00d7 100, "
#         "where Total Step Time = Communication + Compute (per step, averaged across workers)."
#     )
#     fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7, color="#444444", wrap=True)

#     out_path = RESULTS_DIR / "communication_overhead_pct.png"
#     fig.tight_layout(rect=[0, 0.08, 1, 1])  # reserve space at bottom for the caption
#     fig.savefig(out_path, dpi=150)
#     print(f"Saved -> {out_path}")
#     return fig


# def plot_scaling_efficiency(data):
#     """
#     Scaling efficiency (%) = (Speedup / Number of Workers) * 100, at each
#     tested world_size. This is the metric that makes the "more workers
#     isn't automatically better" story visible as a trend rather than a
#     single number: efficiency starts near 100% (or wherever it lands at
#     the smallest tested world_size) and decays as communication/
#     synchronization overhead eats a larger share of each step -- this is
#     the expected, textbook shape for a real ring all-reduce system, not
#     a sign of a bug. A single KPI number for "best efficiency" is
#     misleading precisely because this metric is close to monotonically
#     decreasing by construction -- the trend (this chart) is the
#     meaningful artifact, not any single point on it.
#     """
#     if data.get("baseline_multithread_wall_clock_seconds") is not None:
#         eff_key, baseline_label = "scaling_efficiency", "default multi-threaded baseline"
#     elif data.get("baseline_singlethread_wall_clock_seconds") is not None:
#         eff_key, baseline_label = "scaling_efficiency_singlethread", "single-threaded baseline"
#     else:
#         print("Skipping scaling_efficiency_pct.png -- no baseline data available.")
#         return None

#     results = [r for r in data["results"] if r["world_size"] >= 2 and eff_key in r]
#     if not results:
#         print("Skipping scaling_efficiency_pct.png -- no scaling efficiency data available.")
#         return None

#     world_sizes = [r["world_size"] for r in results]
#     efficiency_pct = [r[eff_key] * 100.0 for r in results]

#     fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
#     ax.axhline(100, linestyle="--", color="gray", linewidth=1.3, label="Ideal (100% efficiency)")
#     ax.plot(world_sizes, efficiency_pct, "o-", color="#7c3aed", linewidth=2, markersize=8, label="RingSync")

#     for ws, pct in zip(world_sizes, efficiency_pct):
#         ax.annotate(f"{pct:.1f}%", (ws, pct), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

#     ax.set_xlabel("Number of workers (world size)")
#     ax.set_ylabel("Scaling efficiency (%)")
#     ax.set_title("Scaling Efficiency vs. Number of Workers", fontsize=10)
#     ax.set_xticks(world_sizes)
#     ax.set_ylim(bottom=0, top=max(110, max(efficiency_pct) * 1.15))
#     ax.legend(fontsize=8)
#     ax.grid(True, alpha=0.3)

#     caption = f"Scaling Efficiency (%) = (Speedup \u00f7 Number of Workers) \u00d7 100, relative to the {baseline_label}."
#     fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7, color="#444444", wrap=True)

#     out_path = RESULTS_DIR / "scaling_efficiency_pct.png"
#     fig.tight_layout(rect=[0, 0.08, 1, 1])
#     fig.savefig(out_path, dpi=150)
#     print(f"Saved -> {out_path}")
#     return fig


# def plot_training_loss_vs_epoch(data):
#     """
#     One line per configuration (each selected world_size, plus whichever
#     baseline(s) were run), showing average training loss per epoch.
#     This is the loss-curve counterpart to the speedup charts: it answers
#     "did distribution change what the model actually learned," which
#     matters because a speedup number is meaningless if the distributed
#     runs were secretly training worse models. Lines overlapping closely
#     is the expected, correct result -- ring all-reduce should produce
#     numerically equivalent training to the baseline, just faster (or not).
#     """
#     results = [r for r in data["results"] if r["world_size"] >= 2 and r.get("epoch_avg_loss")]
#     if not results:
#         print("Skipping training_loss_vs_epoch.png -- no per-epoch loss data available.")
#         return None

#     fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)

#     if data.get("baseline_multithread_epoch_avg_loss"):
#         losses = data["baseline_multithread_epoch_avg_loss"]
#         ax.plot(range(1, len(losses) + 1), losses, "--", color="black",
#                 label="Baseline (multi-threaded)", linewidth=1.5)

#     if data.get("baseline_singlethread_epoch_avg_loss"):
#         losses = data["baseline_singlethread_epoch_avg_loss"]
#         ax.plot(range(1, len(losses) + 1), losses, ":", color="gray",
#                 label="Baseline (single-threaded)", linewidth=1.5)

#     cmap = plt.get_cmap("viridis")
#     for i, r in enumerate(results):
#         losses = r["epoch_avg_loss"]
#         color = cmap(i / max(len(results) - 1, 1))
#         ax.plot(range(1, len(losses) + 1), losses, "o-", color=color,
#                 label=f"RingSync ({r['world_size']} workers)", linewidth=1.5, markersize=5)

#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Average training loss")
#     ax.set_title("Training Loss vs. Epoch", fontsize=10)
#     ax.legend(fontsize=7)
#     ax.grid(True, alpha=0.3)

#     out_path = RESULTS_DIR / "training_loss_vs_epoch.png"
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=150)
#     print(f"Saved -> {out_path}")
#     return fig


# if __name__ == "__main__":
#     import os
#     cpu_count = os.cpu_count()
#     note = f"{cpu_count} CPU core(s) available on this machine"
#     if cpu_count == 1:
#         note += " -- NOT representative of real multi-core scaling"

#     data = load_results()
#     for fig in [
#         plot_speedup_vs_multithread(data, cpu_count_note=note),
#         plot_speedup_vs_singlethread(data, cpu_count_note=note),
#         plot_compute_vs_comm(data),
#         plot_communication_overhead_pct(data),
#         plot_scaling_efficiency(data),
#         plot_training_loss_vs_epoch(data),
#     ]:
#         if fig is not None:
#             plt.close(fig)


"""
Generates the benchmark charts from results/benchmark_results.json:

  1. speedup_vs_multithread_baseline.png -- actual speedup vs. the
     default (multi-threaded) single-process baseline, vs. ideal
  2. speedup_vs_singlethread_baseline.png -- actual speedup vs. the
     single-threaded baseline (apples-to-apples threading), vs. ideal
  3. compute_vs_comm_chart.png -- stacked bar of avg compute vs comm
     time per step, across world sizes
  4. communication_overhead_pct.png -- communication time as a % of
     total step time, vs. number of workers -- this is the single
     clearest number for explaining WHY the speedup curve saturates:
     as this percentage climbs, more and more of each step is being
     spent on synchronization rather than useful compute, which is
     exactly why speedup falls further below ideal as world_size grows.

Run AFTER scripts/run_benchmark.py:
    PYTHONPATH=. python3 scripts/plot_benchmark.py

IMPORTANT: if this machine has only 1 CPU core (check with `nproc`),
the resulting speedup curves will show sub-linear or even negative
speedup -- not because the implementation is wrong, but because
spawning N worker processes on 1 core gives zero real parallelism.
Re-run scripts/run_benchmark.py on a multi-core machine to get
representative curves before using these charts in a portfolio/interview
context.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def set_results_dir(path) -> None:
    """
    Points every plot_* function (and load_results) at a specific
    experiment's folder (e.g. results/RS-20260703-001/) instead of the
    legacy flat results/ folder. Call once before generating charts for
    a given experiment. Global, not per-call, because each Streamlit
    script rerun is a fresh top-to-bottom execution -- there's no
    concurrent overlap within a single run to worry about.
    """
    global RESULTS_DIR
    RESULTS_DIR = Path(path)


def load_results():
    with open(RESULTS_DIR / "benchmark_results.json") as f:
        return json.load(f)


STANDARD_FIGSIZE = (7, 5)   # for charts displayed side-by-side, 2 per row
                              # (Speedup x2, Compute vs. Comm + Overhead)

WIDE_FIGSIZE = (13, 5)      # for charts displayed full-width, alone
                              # (Scaling Efficiency, Training Loss).
                              # Same height as STANDARD_FIGSIZE, roughly
                              # double the width -- since Streamlit scales
                              # each image to fill its container while
                              # preserving aspect ratio, a full-width slot
                              # is roughly 2x wider than a half-width one,
                              # so matching the rendered HEIGHT across both
                              # layouts requires widening (not just scaling)
                              # the figure, not simply reusing the same
                              # width/height ratio.


def _plot_single_speedup_curve(world_sizes, actual, ideal, label, color, title, out_filename, cpu_count_note=""):
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
    ax.plot(world_sizes, ideal, "--", color="gray", label="Ideal linear speedup", linewidth=1.5)
    ax.plot(world_sizes, actual, "o-", color=color, label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Number of workers (world size)")
    ax.set_ylabel("Speedup vs. single-process baseline")
    full_title = title
    if cpu_count_note:
        full_title += f"\n({cpu_count_note})"
    ax.set_title(full_title, fontsize=10)
    ax.set_xticks(world_sizes)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    caption = (
        "Only RingSync performance is plotted. Speedup = Baseline Runtime \u00f7 RingSync Runtime.\n"
        "Baseline is a fixed single-process measurement, used only as the reference, not plotted."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7, color="#444444", wrap=True)

    out_path = RESULTS_DIR / out_filename
    fig.tight_layout(rect=[0, 0.10, 1, 1])  # reserve space at bottom for the caption
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    return fig


def plot_speedup_vs_multithread(data, cpu_count_note: str = ""):
    if data.get("baseline_multithread_wall_clock_seconds") is None:
        print("Skipping speedup_vs_multithread_baseline.png -- multi-threaded baseline was not run.")
        return None

    results = [r for r in data["results"] if r["world_size"] >= 2]
    world_sizes = [r["world_size"] for r in results]
    actual = [r["speedup_vs_baseline"] for r in results]
    ideal = [r["ideal_speedup"] for r in results]

    fig = _plot_single_speedup_curve(
        world_sizes, actual, ideal,
        label="RingSync Speedup (relative to Default PyTorch)",
        color="#2563eb",
        title="RingSync Speedup vs. Default Multi-threaded PyTorch Baseline",
        out_filename="speedup_vs_multithread_baseline.png",
        cpu_count_note=cpu_count_note,
    )
    return fig


def plot_speedup_vs_singlethread(data, cpu_count_note: str = ""):
    if data.get("baseline_singlethread_wall_clock_seconds") is None:
        print("Skipping speedup_vs_singlethread_baseline.png -- single-threaded baseline was not run.")
        return None

    results = [r for r in data["results"] if r["world_size"] >= 2]
    world_sizes = [r["world_size"] for r in results]
    actual = [r["speedup_vs_singlethread_baseline"] for r in results]
    ideal = [r["ideal_speedup"] for r in results]

    fig = _plot_single_speedup_curve(
        world_sizes, actual, ideal,
        label="RingSync Speedup (relative to Single-threaded Baseline)",
        color="#16a34a",
        title="RingSync Speedup vs. Single-threaded Baseline",
        out_filename="speedup_vs_singlethread_baseline.png",
        cpu_count_note=cpu_count_note,
    )
    return fig


def plot_compute_vs_comm(data):
    results = [r for r in data["results"] if r["world_size"] > 1]  # baseline has no comm phase
    world_sizes = [r["world_size"] for r in results]
    compute = [r["avg_compute_per_step_s"] for r in results]
    comm = [r["avg_comm_per_step_s"] for r in results]

    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
    x = range(len(world_sizes))
    ax.bar(x, compute, label="Compute time / step", color="#2563eb")
    ax.bar(x, comm, bottom=compute, label="Communication time / step", color="#f59e0b")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(ws) for ws in world_sizes])
    ax.set_xlabel("Number of workers (world size)")
    ax.set_ylabel("Avg. time per training step (s)")
    ax.set_title("RingSync: Compute vs. Communication Time per Step", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    out_path = RESULTS_DIR / "compute_vs_comm_chart.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    return fig


def plot_communication_overhead_pct(data):
    """
    Communication overhead (%) = comm_time / (compute_time + comm_time) * 100,
    per step, at each world size. This single number is the cleanest way
    to explain WHY the speedup curve bends away from ideal: as world_size
    grows, the ring has more hops (2*(N-1) messages per step), so a
    growing share of every step is spent synchronizing rather than
    computing -- directly visualizing the mechanism behind the
    saturating speedup curves in the other two charts.
    """
    results = [r for r in data["results"] if r["world_size"] > 1]
    world_sizes = [r["world_size"] for r in results]
    overhead_pct = [
        100.0 * r["avg_comm_per_step_s"] / (r["avg_compute_per_step_s"] + r["avg_comm_per_step_s"])
        for r in results
    ]

    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
    ax.plot(world_sizes, overhead_pct, "o-", color="#dc2626", linewidth=2, markersize=8)

    for ws, pct in zip(world_sizes, overhead_pct):
        ax.annotate(f"{pct:.1f}%", (ws, pct), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel("Number of workers (world size)")
    ax.set_ylabel("Communication overhead (% of step time)")
    ax.set_title("Communication Overhead (% of Training Step Time)\nvs. Number of Workers", fontsize=10)
    ax.set_xticks(world_sizes)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    caption = (
        "Overhead % = (Communication Time \u00f7 Total Step Time) \u00d7 100, "
        "where Total Step Time = Communication + Compute (per step, averaged across workers)."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7, color="#444444", wrap=True)

    out_path = RESULTS_DIR / "communication_overhead_pct.png"
    fig.tight_layout(rect=[0, 0.08, 1, 1])  # reserve space at bottom for the caption
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    return fig


def plot_scaling_efficiency(data):
    """
    Scaling efficiency (%) = (Speedup / Number of Workers) * 100, at each
    tested world_size. This is the metric that makes the "more workers
    isn't automatically better" story visible as a trend rather than a
    single number: efficiency starts near 100% (or wherever it lands at
    the smallest tested world_size) and decays as communication/
    synchronization overhead eats a larger share of each step -- this is
    the expected, textbook shape for a real ring all-reduce system, not
    a sign of a bug. A single KPI number for "best efficiency" is
    misleading precisely because this metric is close to monotonically
    decreasing by construction -- the trend (this chart) is the
    meaningful artifact, not any single point on it.
    """
    if data.get("baseline_multithread_wall_clock_seconds") is not None:
        eff_key, baseline_label = "scaling_efficiency", "default multi-threaded baseline"
    elif data.get("baseline_singlethread_wall_clock_seconds") is not None:
        eff_key, baseline_label = "scaling_efficiency_singlethread", "single-threaded baseline"
    else:
        print("Skipping scaling_efficiency_pct.png -- no baseline data available.")
        return None

    results = [r for r in data["results"] if r["world_size"] >= 2 and eff_key in r]
    if not results:
        print("Skipping scaling_efficiency_pct.png -- no scaling efficiency data available.")
        return None

    world_sizes = [r["world_size"] for r in results]
    efficiency_pct = [r[eff_key] * 100.0 for r in results]

    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    ax.axhline(100, linestyle="--", color="gray", linewidth=1.3, label="Ideal (100% efficiency)")
    ax.plot(world_sizes, efficiency_pct, "o-", color="#7c3aed", linewidth=2, markersize=8, label="RingSync")

    for ws, pct in zip(world_sizes, efficiency_pct):
        ax.annotate(f"{pct:.1f}%", (ws, pct), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel("Number of workers (world size)")
    ax.set_ylabel("Scaling efficiency (%)")
    ax.set_title("Scaling Efficiency vs. Number of Workers", fontsize=10)
    ax.set_xticks(world_sizes)
    ax.set_ylim(bottom=0, top=max(110, max(efficiency_pct) * 1.15))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    caption = f"Scaling Efficiency (%) = (Speedup \u00f7 Number of Workers) \u00d7 100, relative to the {baseline_label}."
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7, color="#444444", wrap=True)

    out_path = RESULTS_DIR / "scaling_efficiency_pct.png"
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    return fig


def plot_training_loss_vs_epoch(data):
    """
    One line per configuration (each selected world_size, plus whichever
    baseline(s) were run), showing average training loss per epoch.
    This is the loss-curve counterpart to the speedup charts: it answers
    "did distribution change what the model actually learned," which
    matters because a speedup number is meaningless if the distributed
    runs were secretly training worse models. Lines overlapping closely
    is the expected, correct result -- ring all-reduce should produce
    numerically equivalent training to the baseline, just faster (or not).
    """
    results = [r for r in data["results"] if r["world_size"] >= 2 and r.get("epoch_avg_loss")]
    if not results:
        print("Skipping training_loss_vs_epoch.png -- no per-epoch loss data available.")
        return None

    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)

    if data.get("baseline_multithread_epoch_avg_loss"):
        losses = data["baseline_multithread_epoch_avg_loss"]
        ax.plot(range(1, len(losses) + 1), losses, "--", color="black",
                label="Baseline (multi-threaded)", linewidth=1.5)

    if data.get("baseline_singlethread_epoch_avg_loss"):
        losses = data["baseline_singlethread_epoch_avg_loss"]
        ax.plot(range(1, len(losses) + 1), losses, ":", color="gray",
                label="Baseline (single-threaded)", linewidth=1.5)

    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(results):
        losses = r["epoch_avg_loss"]
        color = cmap(i / max(len(results) - 1, 1))
        ax.plot(range(1, len(losses) + 1), losses, "o-", color=color,
                label=f"RingSync ({r['world_size']} workers)", linewidth=1.5, markersize=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average training loss")
    ax.set_title("Training Loss vs. Epoch", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    out_path = RESULTS_DIR / "training_loss_vs_epoch.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    return fig


def plot_compare_runs(experiments: "list[dict]") -> "plt.Figure | None":
    """
    Overlays the speedup curve of multiple past experiments on one
    chart, for Benchmark History's "Compare Runs" feature. Each
    experiment dict is expected to have 'experiment_id' and 'data'
    keys (exactly what scripts/experiment_utils.list_experiments()
    returns). Prefers each experiment's multi-threaded-baseline speedup
    if available, falling back to single-threaded, so experiments run
    with different --baseline choices can still be compared on a
    common "speedup" axis (not identical baselines, but the same metric).
    """
    if not experiments:
        return None

    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    cmap = plt.get_cmap("tab10")

    plotted_any = False
    for i, exp in enumerate(experiments):
        data = exp["data"]
        results = [r for r in data["results"] if r["world_size"] >= 2]
        if not results:
            continue

        if "speedup_vs_baseline" in results[0]:
            key = "speedup_vs_baseline"
        elif "speedup_vs_singlethread_baseline" in results[0]:
            key = "speedup_vs_singlethread_baseline"
        else:
            continue

        world_sizes = [r["world_size"] for r in results]
        speedups = [r[key] for r in results]
        color = cmap(i % 10)
        ax.plot(world_sizes, speedups, "o-", color=color, linewidth=2, markersize=6,
                label=exp.get("experiment_id", f"Run {i+1}"))
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None

    ax.set_xlabel("Number of workers (world size)")
    ax.set_ylabel("Speedup (vs. that run's own baseline)")
    ax.set_title("Compare Runs: Speedup Across Experiments", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    caption = (
        "Each line uses that experiment's own available baseline (multi-threaded preferred, "
        "single-threaded as fallback) -- compare trend shapes, not necessarily identical baselines."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7, color="#444444", wrap=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


if __name__ == "__main__":
    import os
    cpu_count = os.cpu_count()
    note = f"{cpu_count} CPU core(s) available on this machine"
    if cpu_count == 1:
        note += " -- NOT representative of real multi-core scaling"

    data = load_results()
    for fig in [
        plot_speedup_vs_multithread(data, cpu_count_note=note),
        plot_speedup_vs_singlethread(data, cpu_count_note=note),
        plot_compute_vs_comm(data),
        plot_communication_overhead_pct(data),
        plot_scaling_efficiency(data),
        plot_training_loss_vs_epoch(data),
    ]:
        if fig is not None:
            plt.close(fig)