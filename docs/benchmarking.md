# RingSync Benchmarking Methodology

This document describes how RingSync measures its own performance: what is compared against what, why, and how each reported number is computed. It is written to be read the way a methodology section of a systems paper is read — precisely enough that the results could be reproduced or challenged, not just admired.

Source: `scripts/benchmark_core.py`, `scripts/plot_benchmark.py`, `scripts/report_content.py`, `scripts/system_info.py`, `scripts/experiment_utils.py`.

---

## Benchmark Philosophy

Benchmarking a distributed system honestly is harder than benchmarking a single function, for a specific reason: a distributed run has more moving parts whose costs are easy to misattribute to each other. Process startup time, Python interpreter and library import time, actual computation, and actual network communication are four genuinely different costs, and a benchmark that conflates them will report a number that is real but means something different from what it appears to mean. Early in this project's own development, this exact mistake produced a benchmark showing distributed training as *slower* at every worker count tested — not because the distributed algorithm was slow, but because the timing methodology included multi-second Python/PyTorch process-spawn overhead inside what was labeled "training time." Fixing the methodology, not the algorithm, resolved it. That experience is the direct reason this document exists: to make the measurement boundaries explicit rather than assumed.

A second, related difficulty: a single baseline cannot answer more than one question. If RingSync is compared only against a default single-process PyTorch run, a "no speedup" result is ambiguous — it could mean the distributed algorithm has no value, or it could mean the single process was already using every available CPU core efficiently via its own internal multi-threading, leaving nothing for process-level parallelism to add. These are different findings requiring different baselines to distinguish, which is why RingSync's benchmark engine defines three configurations, not one, and reports metrics against two independent reference points rather than a single number.

---

## Benchmark Configurations

| Configuration | Description |
|---|---|
| **Default multi-threaded baseline** | A single Python process training with PyTorch's default behavior: automatic multi-threading of tensor operations across every available CPU core. This is what a user gets by default, with zero configuration. |
| **Single-threaded baseline** | The identical single-process training run, with `torch.set_num_threads(1)` — every RingSync worker is also pinned to one thread, so this baseline is directly comparable to one worker. |
| **RingSync distributed benchmark** | The real ring all-reduce training run, at a selected world size, across genuinely separate OS processes communicating over TCP. |

All three exist because each answers a distinct question that the others cannot: the multi-threaded baseline answers *"is distribution worth it compared to what I already get for free?"*; the single-threaded baseline answers *"is the distribution mechanism itself working, independent of how many CPU threads a single process happens to use?"*; the distributed benchmark is the thing actually being evaluated. Reporting results against only one of the two baselines would leave one of these two questions permanently unanswered.

---

## Hardware

RingSync's benchmark results are meaningless without the hardware context they were produced on, and are actively misleading if that context is omitted or assumed. A reported speedup of `1.25×` at 8 workers describes something entirely different on a 4-core laptop than on a 32-core workstation — on the former, 8 workers already exceeds available physical parallelism (oversubscription); on the latter, it may be far from saturating it.

Specifically, the following hardware and software facts change what a given result means, and RingSync's dashboard auto-detects and reports all of them alongside every result:

- **CPU model** — different microarchitectures have different single-core throughput, changing the absolute (not just relative) numbers.
- **Logical vs. physical core count** — a machine with hyperthreading reports more logical cores than it has physical execution units; worker counts approaching or exceeding *physical* core count are in a fundamentally different regime than counts safely below it.
- **RAM** — insufficient memory can force swapping under many concurrent worker processes, which manifests as a performance collapse unrelated to the distributed algorithm itself.
- **Threading configuration** — whether the baseline is allowed default multi-threading or pinned to one thread is not a hardware fact but is inseparable from interpreting any result against it (see Why Two Baselines, below).
- **Operating system** — process-spawn cost, in particular, differs meaningfully between platforms (Windows' default `multiprocessing` start method, `spawn`, re-imports the entire Python process per worker; Linux's default, `fork`, does not), which is why spawn/import time is measured and reported separately rather than folded into "training time" (see Metrics, below).

---

## Metrics

Let `N` be world size (number of workers), `B` be a baseline's measured wall-clock time, and `T` be the distributed run's measured wall-clock time at world size `N`.

**Speedup**
```
Speedup(N) = B / T(N)
```
How many times faster the distributed run is than the baseline. A value above 1 means the distributed run finished faster; below 1 means the baseline was faster.

**Scaling Efficiency**
```
Efficiency(N) = Speedup(N) / N   (often reported ×100 as a percentage)
```
What fraction of *ideal* linear speedup was actually achieved. Ideal, embarrassingly-parallel scaling would give `Speedup(N) = N`, i.e. 100% efficiency; any communication or synchronization cost pulls this below 100%, and it is expected to decrease as `N` grows (see Interpreting Results).

**Communication Overhead**
```
Overhead(N) = 100 × CommTime(N) / (ComputeTime(N) + CommTime(N))
```
Where `ComputeTime(N)` and `CommTime(N)` are the average per-step compute and communication durations, measured directly inside each worker process and averaged across all workers at that world size. This is the fraction of a single training step spent synchronizing rather than computing — the single clearest metric for explaining *why* speedup falls short of ideal.

**Compute Time** and **Communication Time** (per step) are measured with wall-clock timers placed directly around the forward/backward pass and around the ring all-reduce call, respectively, inside `ringsync/worker/worker_node.py`'s training loop — not inferred or estimated, but timed at the exact code boundary each phase occupies.

**Wall Clock** (`wall_clock_seconds`) is the maximum, across all workers at a given world size, of each worker's own internally-measured total training time — bounded, deliberately, by internal timers that start *after* process startup and library imports have already completed. Synchronous training is only as fast as its slowest worker, which is why the maximum (not the mean) across workers is used.

**Spawn Time / Import Time.** Separately, `outer_process_wall_clock_seconds` measures wall-clock time from the moment worker processes are launched to the moment they all exit — a strictly larger number than `wall_clock_seconds`, since it also includes OS process creation and every worker's Python interpreter startup and library imports (PyTorch import alone is not free). The *difference* between these two figures is, by construction, the process-spawn-and-import overhead — tracked explicitly and separately so it is never mistaken for training cost, which is precisely the error described in Benchmark Philosophy above.

---

## Why Two Baselines?

**Default PyTorch baseline** measures the realistic, common case: what happens if a user simply runs their training script normally, with no distribution and no manual threading configuration. PyTorch will automatically parallelize tensor operations across every available core. Comparing RingSync against this baseline answers whether adopting distributed training is worth it *relative to doing nothing*.

**Single-threaded baseline** isolates one specific confound: PyTorch's automatic multi-threading is itself a form of parallelism, entirely unrelated to distributed training, and can substantially close the gap that distribution would otherwise show — particularly for small models where a single, well-threaded process already saturates available compute. Pinning this baseline to one thread, exactly as every RingSync worker is pinned, produces a comparison where the *only* variable is "one process doing all the work" versus "`N` processes splitting the work" — the pure data-parallelism effect, with threading held constant on both sides.

The practical consequence of running both: RingSync frequently shows a lower speedup against the default multi-threaded baseline than against the single-threaded one, at the same world size, from the same run. This is not a contradiction — it is two different, both-correct answers to two different questions, and reporting both is more honest than reporting either alone.

---

## Benchmark Procedure

Each benchmark run, whether launched from the CLI (`scripts/run_benchmark.py`) or the dashboard, follows the same sequence:

1. **Generate an experiment ID** (`RS-YYYYMMDD-NNN`) and create its dedicated results directory.
2. **Run the selected baseline(s).** Multi-threaded and/or single-threaded, each a genuine training run (not simulated), saving its own loss history.
3. **For each selected world size:**
   a. **Initialize workers** — spawn `N` separate OS processes, each establishing its position in the TCP ring (see [protocol.md](protocol.md)).
   b. **Load dataset shard** — each worker loads only its own disjoint slice of the data, deterministically sharded by rank.
   c. **Train** — standard forward/backward pass, timed.
   d. **Synchronize** — ring all-reduce over the flattened gradient, timed separately from compute (see [ring-allreduce.md](ring-allreduce.md)).
   e. **Collect metrics** — each worker writes its own per-step compute/communication timings and loss history; the orchestrating process reads all of them back after every worker exits.
4. **Compute derived metrics** — speedup, scaling efficiency, and communication overhead, from the raw timings collected above.
5. **Generate charts** — all five chart types (below), saved into the experiment's results directory.
6. **Generate insights** — a short, data-driven narrative summary (best speedup, overhead trend, baseline crossover point, convergence consistency), computed from the same results, not authored separately.
7. **Save the experiment** — the complete result set (raw JSON, every chart, optionally a PDF report) is written under `results/<experiment_id>/`, self-contained and independent of any other run.

---

## Graphs

**Speedup vs. Default Multi-threaded Baseline** and **Speedup vs. Single-threaded Baseline.** Actual speedup at each tested world size, plotted against the ideal linear-speedup line (`Speedup = N`). The gap between the actual curve and the ideal line is the visual answer to "how much is overhead costing," for each of the two baselines independently.

**Compute vs. Communication Time.** A stacked bar per world size, showing the average per-step time spent computing versus synchronizing. This is the most direct visual explanation of where a training step's time actually goes, and how that split shifts as workers are added.

**Communication Overhead (%).** The single percentage described under Metrics, plotted against world size. This is usually the clearest chart in the whole set for explaining *why* the speedup and efficiency curves bend away from ideal — rising overhead is the mechanism, not a separate phenomenon.

**Scaling Efficiency.** Efficiency (as a percentage) against world size, with a 100%-efficiency reference line. Because efficiency is speedup divided by worker count, and speedup grows sub-linearly, this curve is expected to trend downward — the chart exists specifically so that trend is visible as a trend, rather than reduced to a single potentially-misleading summary number (see the next section).

**Training Loss vs. Epoch.** Per-epoch average loss, one line per world size tested plus the baseline(s), overlaid. This graph is not about speed at all — it exists to verify that distributed training produces the *same learning outcome* as the baseline. Lines that converge to the same final loss are the visual counterpart to the bit-for-bit weight comparison described in [ring-allreduce.md](ring-allreduce.md#correctness): distribution changed how fast training ran, not what it learned.

---

## Interpreting Results

**Why speedup eventually plateaus (or declines).** Every additional worker adds a fixed amount of communication (more ring hops: `2(N-1)` rounds) while the useful compute per worker shrinks (each worker's data shard gets smaller). At some world size, the marginal communication cost of one more worker exceeds the marginal compute time saved by splitting the work further — beyond that point, adding workers stops helping and can start hurting.

**Why communication overhead increases with world size.** Ring all-reduce requires `2(N-1)` communication rounds — a strictly increasing function of `N`. More workers mechanically means more round-trip exchanges per training step, which is why this metric's upward trend is expected and is not, by itself, evidence of a problem.

**Why scaling efficiency decreases.** Efficiency is speedup divided by worker count. Because speedup grows sub-linearly (per the two points above) while worker count grows linearly by definition, their ratio necessarily declines. This is the textbook shape for a real synchronous distributed system, not a defect specific to RingSync.

**Why perfect (100%) scaling is impossible.** Perfect scaling requires zero communication cost and zero synchronization cost — i.e., truly embarrassingly parallel work with no dependency between workers. Synchronous data-parallel training is not embarrassingly parallel by construction: every worker must exchange gradients with every other worker, every single step, which is exactly the cost that keeps efficiency below 100% even in a well-implemented system running on ideal hardware.

---

## Limitations

- **CPU-only.** No GPU-to-GPU communication path exists; all measurements reflect CPU-bound training and CPU-mediated networking.
- **Localhost only.** All benchmark runs to date communicate over `127.0.0.1`. Real network latency and bandwidth constraints between physically separate machines are not represented in any current result.
- **Small model.** The benchmarked model is intentionally small (a compact CNN), chosen to keep experiments fast to run and easy to reason about — results should not be assumed to hold, unchanged, for models with substantially different compute-to-parameter-count ratios.
- **Small dataset / synthetic data.** Default benchmarks use a synthetic, CIFAR-shaped dataset rather than a full real-world dataset, prioritizing reproducibility and fast iteration over realism.
- **Educational implementation.** The benchmark suite is built to make a from-scratch distributed system's behavior legible, not to serve as a general-purpose ML systems benchmarking tool comparable in scope to, e.g., MLPerf.

---

## Reproducibility

Every benchmark run is assigned a unique **experiment ID** (`RS-YYYYMMDD-NNN`) and writes its complete output — raw results JSON, every generated chart, and an optional PDF report — into its own directory under `results/`, entirely independent of any other run. Nothing is overwritten by a subsequent run.

Each experiment's stored JSON includes the full configuration used to produce it (world sizes, baseline mode, batch size, epochs, learning rate) alongside auto-detected **system information** (CPU, core counts, RAM, Python and PyTorch versions, OS) — sufficient to know both *what* was run and *on what hardware*, without relying on memory or external notes.

The **PDF report** packages all of the above into a single shareable document — summary, configuration, system information, insights, and every chart — so a result can be interpreted correctly by someone who was not present when it was generated.

**Experiment history** lists every past experiment (date, workers tested, baseline mode, speedup), and **Compare Runs** overlays the speedup curves of multiple selected experiments on one chart, making it possible to see directly whether a change (to code, to configuration, or to hardware) actually moved the numbers, rather than relying on memory of a previous run's results.

---

## Best Practices

**Avoid oversubscribing CPU cores.** Running more worker processes than physical (or even logical) cores available causes processes to compete for the same execution units, which manifests as *increasing* per-step compute time as world size grows — a measurement artifact, not a property of the distributed algorithm. RingSync's dashboard caps selectable worker counts below the detected core count specifically to reduce this risk, but it does not prevent choosing a count close to that ceiling; leaving headroom is a user responsibility the tool cannot fully enforce.

**Compare identical workloads.** Batch size, epoch count, and dataset must be held constant across the configurations being compared — RingSync's benchmark engine already enforces this within a single run (baseline and every distributed configuration share the same parameters), but comparing *across* separately-run experiments (via Compare Runs) requires the same discipline manually, since two experiments run with different batch sizes are not a fair comparison of world size's effect alone.

**Interpret results correctly.** A speedup below 1.0 is not necessarily a failure — depending on which baseline it's measured against, it may simply indicate that the workload is too small, or the world size too low, for distribution's benefits to outweigh its fixed costs yet (see Interpreting Results). Conversely, a high speedup at a small world size does not guarantee the trend continues; scaling efficiency and communication overhead, tracked across the full tested range rather than at a single point, are what reveal whether a result generalizes.
