"""
Experiment IDs and per-experiment result directories.

Every benchmark run gets an ID like RS-20260703-001 (date + a daily
sequence number), and all of its output -- benchmark_results.json,
every chart PNG, the PDF report -- is saved under
results/<experiment_id>/ instead of a single shared results/ folder.
This is what makes runs reproducible and comparable: nothing from one
run overwrites another, and Benchmark History can list/compare past
runs because they're each a self-contained folder.
"""

import re
from datetime import datetime
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
RESULTS_ROOT.mkdir(exist_ok=True)

_ID_PATTERN = re.compile(r"^RS-(\d{8})-(\d{3})$")


def generate_experiment_id(now: "datetime | None" = None) -> str:
    """
    RS-YYYYMMDD-NNN, where NNN is a zero-padded sequence number that
    resets each day. Scans existing experiment folders for today's
    date to find the next free number, so this is safe to call even if
    previous runs exist.
    """
    now = now or datetime.now()
    date_str = now.strftime("%Y%m%d")

    existing_numbers = []
    if RESULTS_ROOT.exists():
        for entry in RESULTS_ROOT.iterdir():
            if not entry.is_dir():
                continue
            match = _ID_PATTERN.match(entry.name)
            if match and match.group(1) == date_str:
                existing_numbers.append(int(match.group(2)))

    next_number = max(existing_numbers, default=0) + 1
    return f"RS-{date_str}-{next_number:03d}"


def get_experiment_dir(experiment_id: str) -> Path:
    """Returns (and creates) the results/<experiment_id>/ directory."""
    path = RESULTS_ROOT / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_experiment_date(experiment_id: str) -> "datetime | None":
    """Extracts the date portion of an experiment ID for display/sorting."""
    match = _ID_PATTERN.match(experiment_id)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d")


def list_experiments() -> "list[dict]":
    """
    Scans results/ for all experiment folders containing a valid
    benchmark_results.json, and returns a summary dict per experiment
    (id, date, workers tested, best speedup, etc.) for the Benchmark
    History table. Folders that don't match the RS-YYYYMMDD-NNN pattern
    or don't have a valid results file are silently skipped, not errors --
    older runs from before this feature existed simply won't appear.
    """
    import json

    from scripts.report_content import compute_summary_stats

    experiments = []
    if not RESULTS_ROOT.exists():
        return experiments

    for entry in sorted(RESULTS_ROOT.iterdir(), reverse=True):
        if not entry.is_dir() or not _ID_PATTERN.match(entry.name):
            continue
        results_json = entry / "benchmark_results.json"
        if not results_json.exists():
            continue

        try:
            with open(results_json) as f:
                data = json.load(f)
            stats = compute_summary_stats(data)
            date = parse_experiment_date(entry.name)
            experiments.append({
                "experiment_id": entry.name,
                "date": date.strftime("%Y-%m-%d") if date else "Unknown",
                "workers_tested": ", ".join(str(r["world_size"]) for r in stats["dist_results"]),
                "best_speedup": stats["best_speedup"],
                "best_speedup_ws": stats["best_speedup_ws"],
                "baseline_mode": data.get("baseline_mode", "N/A"),
                "epochs": data.get("epochs", "N/A"),
                "batch_size": data.get("batch_size", "N/A"),
                "data": data,  # full data, for the Compare Runs feature
            })
        except (json.JSONDecodeError, KeyError, OSError):
            continue  # a corrupted or partial run shouldn't break the whole history list

    return experiments


if __name__ == "__main__":
    eid = generate_experiment_id()
    print(f"Next experiment ID: {eid}")
    print(f"Directory: {get_experiment_dir(eid)}")

    print("\nExisting experiments:")
    for exp in list_experiments():
        print(f"  {exp['experiment_id']}  ({exp['date']})  workers={exp['workers_tested']}  "
              f"best_speedup={exp['best_speedup']}")
