
def compute_summary_stats(data: dict) -> dict:
    """
    Pulls every number the Benchmark Summary and Insights panel need
    into one place. Pure read of already-computed fields from
    benchmark_results.json -- no new calculations, just selection and
    formatting of existing values.
    """
    dist_results = [r for r in data["results"] if r["world_size"] >= 2]

    speedup_key, eff_key, baseline_label = None, None, None
    if data.get("baseline_multithread_wall_clock_seconds") is not None:
        speedup_key, eff_key = "speedup_vs_baseline", "scaling_efficiency"
        baseline_label = "default multi-threaded PyTorch baseline"
    elif data.get("baseline_singlethread_wall_clock_seconds") is not None:
        speedup_key, eff_key = "speedup_vs_singlethread_baseline", "scaling_efficiency_singlethread"
        baseline_label = "single-threaded baseline"

    best_speedup = best_speedup_ws = max_overhead = None
    min_overhead = None
    fastest = None
    crossover_ws = None  # first world_size where RingSync beats the (multithread) baseline
    max_workers_result = None
    speedup_at_max_workers = efficiency_at_max_workers = None

    if dist_results:
        fastest = min(dist_results, key=lambda r: r["wall_clock_seconds"])
        max_workers_result = max(dist_results, key=lambda r: r["world_size"])

        if speedup_key:
            best = max(dist_results, key=lambda r: r[speedup_key])
            best_speedup = best[speedup_key]
            best_speedup_ws = best["world_size"]

          
            speedup_at_max_workers = max_workers_result.get(speedup_key)
            efficiency_at_max_workers = max_workers_result.get(eff_key)

        overheads = [
            100.0 * r["avg_comm_per_step_s"] / (r["avg_compute_per_step_s"] + r["avg_comm_per_step_s"])
            for r in dist_results
        ]
        max_overhead = max(overheads)
        min_overhead = min(overheads)

        if "speedup_vs_baseline" in dist_results[0]:
            for r in sorted(dist_results, key=lambda r: r["world_size"]):
                if r["speedup_vs_baseline"] > 1.0:
                    crossover_ws = r["world_size"]
                    break

    return {
        "dist_results": dist_results,
        "speedup_key": speedup_key,
        "eff_key": eff_key,
        "baseline_label": baseline_label,
        "best_speedup": best_speedup,
        "best_speedup_ws": best_speedup_ws,
        "max_overhead": max_overhead,
        "min_overhead": min_overhead,
        "fastest": fastest,
        "crossover_ws": crossover_ws,
        "max_workers_result": max_workers_result,
        "speedup_at_max_workers": speedup_at_max_workers,
        "efficiency_at_max_workers": efficiency_at_max_workers,
    }


def generate_insights(data: dict, stats: dict) -> "list[str]":
    """Auto-generated, data-driven bullet points -- every claim here
    is computed directly from data/stats, not hardcoded."""
    insights = []

    if stats["best_speedup"] is not None:
        insights.append(
            f"**Best speedup achieved:** {stats['best_speedup']:.2f}\u00d7 using "
            f"**{stats['best_speedup_ws']} workers** (relative to the {stats['baseline_label']})."
        )

    if stats["dist_results"] and stats["min_overhead"] is not None:
        lowest_ws = min(stats["dist_results"], key=lambda r: r["world_size"])
        highest_ws = max(stats["dist_results"], key=lambda r: r["world_size"])
        lo_overhead = 100.0 * lowest_ws["avg_comm_per_step_s"] / (
            lowest_ws["avg_compute_per_step_s"] + lowest_ws["avg_comm_per_step_s"]
        )
        hi_overhead = 100.0 * highest_ws["avg_comm_per_step_s"] / (
            highest_ws["avg_compute_per_step_s"] + highest_ws["avg_comm_per_step_s"]
        )
        insights.append(
            f"**Communication overhead** increased from {lo_overhead:.1f}% "
            f"({lowest_ws['world_size']} workers) to {hi_overhead:.1f}% ({highest_ws['world_size']} workers), "
            "illustrating the expected cost of synchronization as the number of workers grows."
        )

    if data.get("baseline_multithread_wall_clock_seconds") is not None:
        if stats["crossover_ws"] is not None:
            insights.append(
                f"Compared to the default multi-threaded PyTorch baseline, RingSync becomes advantageous "
                f"starting at **{stats['crossover_ws']} workers**."
            )
        else:
            insights.append(
                "Within the tested worker range, RingSync did not exceed the default multi-threaded "
                "PyTorch baseline -- try higher worker counts or a heavier compute workload (larger "
                "batch size) to shift the compute-to-communication ratio further in RingSync's favor."
            )

    epoch_results = [r for r in stats["dist_results"] if r.get("epoch_avg_loss")]
    if epoch_results:
        final_losses = [r["epoch_avg_loss"][-1] for r in epoch_results]
        if data.get("baseline_multithread_epoch_avg_loss"):
            final_losses.append(data["baseline_multithread_epoch_avg_loss"][-1])
        spread_pct = 100.0 * (max(final_losses) - min(final_losses)) / abs(sum(final_losses) / len(final_losses))
        if spread_pct < 2.0:
            insights.append(
                f"**Training loss remained consistent** across all worker configurations "
                f"(final-epoch spread: {spread_pct:.2f}%), indicating distributed synchronization "
                "preserved model convergence."
            )
        else:
            insights.append(
                f"Final-epoch training loss varied by {spread_pct:.2f}% across configurations -- "
                "worth investigating whether this reflects normal run-to-run noise or a synchronization issue."
            )

    return insights
