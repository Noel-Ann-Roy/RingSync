
# """
# RingSync Benchmark Studio.

# An interactive local web app (Streamlit) for configuring, running, and
# visualizing RingSync benchmarks -- replaces manually editing constants
# in scripts/run_benchmark.py and running it from the command line.

# This file is UI/presentation only. It does not implement or alter any
# benchmarking logic, distributed training code, or metric calculation --
# all of that lives in scripts/benchmark_core.py and scripts/plot_benchmark.py,
# unchanged, and this file only arranges how their outputs are configured,
# triggered, and displayed.

# Design note on WHY the benchmark runs as a subprocess rather than
# calling scripts/benchmark_core.py's functions directly in-process:
# RingSync's distributed workers are spawned via Python's multiprocessing
# module, which on Windows uses the "spawn" start method -- this re-
# launches a fresh Python interpreter for each worker. Streamlit apps
# run inside their own already-unusual process/thread model, and mixing
# that with multiprocessing.Process spawned directly from within a
# running Streamlit app is a known source of platform-specific breakage
# (worker processes can end up trying to re-import and re-run the
# Streamlit app itself). Invoking the existing, already-tested CLI script
# (scripts/run_benchmark.py) as a clean subprocess sidesteps this
# entirely. Plotting (scripts/plot_benchmark.py) has no multiprocessing
# in it at all, so that IS called directly, in-process.

# Run:
#     streamlit run dashboard/benchmark_studio.py
# """

# import json
# import os
# import subprocess
# import sys
# import time
# from pathlib import Path

# import matplotlib.pyplot as plt
# import streamlit as st
# from PIL import Image

# ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(ROOT))

# from scripts.plot_benchmark import (
#     plot_speedup_vs_multithread,
#     plot_speedup_vs_singlethread,
#     plot_compute_vs_comm,
#     plot_communication_overhead_pct,
#     plot_scaling_efficiency,
#     plot_training_loss_vs_epoch,
# )
# from scripts.report_content import compute_summary_stats, generate_insights
# from scripts.report_export import generate_pdf_report
# from scripts.system_info import get_system_info

# # Logo, used for both the browser tab icon and the page header. Prefers
# # the transparent variant (see scripts/make_logo_transparent.py) since a
# # non-transparent logo shows a visible background square as a favicon;
# # falls back to the original file, then to the emoji, so a missing or
# # not-yet-processed asset never crashes the app.
# LOGO_PATH = ROOT / "assets" / "logo_transparent.png"
# if not LOGO_PATH.exists():
#     LOGO_PATH = ROOT / "assets" / "logo.png"
# _logo_image = Image.open(LOGO_PATH) if LOGO_PATH.exists() else None

# RESULTS_DIR = ROOT / "results"
# RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"

# st.set_page_config(
#     page_title="RingSync Benchmark Studio",
#     page_icon=_logo_image if _logo_image is not None else "📊",
#     layout="wide",
# )

# # A single config.toml with [theme.light] and [theme.dark] sections
# # (verified working: Streamlit correctly resolves both from one file)
# # means Streamlit's OWN native theme picker (top-right "⋮" -> Settings
# # -> Theme) now switches between RingSync's exact light and dark
# # palettes -- no custom toggle or live CSS override needed. This CSS
# # only adjusts SHAPE (size, padding, radius, shadow), deliberately
# # leaving color untouched so the button correctly follows whichever
# # theme -- and therefore whichever primaryColor -- is actually active,
# # rather than being locked to one hardcoded hex value.
# st.markdown(
#     """
#     <style>
#     div.stButton > button[kind="primary"] {
#         font-size: 1.2rem;
#         font-weight: 700;
#         padding: 0.9rem 2.5rem;
#         border-radius: 10px;
#         box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
#     }
#     div.stButton > button[kind="primary"]:hover {
#         box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
#     }
#     div[data-testid="stMetricValue"] {
#         font-size: 1.6rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# # --------------------------------------------------------------------
# # Helpers
# # --------------------------------------------------------------------

# def candidate_worker_counts(cpu_count: int) -> "list[int]":
#     """
#     A curated, not-overwhelming set of selectable worker counts, capped
#     below the machine's core count. Deliberately leaves at least 1 core
#     unclaimed (cpu_count - 1 ceiling) -- pushing worker count all the
#     way to cpu_count leaves no headroom for the OS, Streamlit itself,
#     or the baseline's own multi-threaded run, and risks the same kind
#     of oversubscription contention the project already had to fix once.
#     """
#     max_workers = max(cpu_count - 1, 2)
#     all_candidates = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
#     return [w for w in all_candidates if w <= max_workers] or [2]


# def load_results():
#     if not RESULTS_JSON.exists():
#         return None
#     with open(RESULTS_JSON) as f:
#         return json.load(f)


# def kpi_card(column, label: str, value: str, delta: str = None):
#     """A single bordered KPI card -- the building block of the
#     Benchmark Summary and Grafana/W&B-style dashboard feel."""
#     with column:
#         with st.container(border=True):
#             st.metric(label, value, delta)


# # --------------------------------------------------------------------
# # Purple section-header icons
# # --------------------------------------------------------------------
# # Emoji (⚙️ 📊 💡 📈) can't be recolored with CSS -- they're pre-rendered,
# # multi-color glyphs supplied by the OS/browser (Twemoji, Segoe UI Emoji,
# # Apple Color Emoji, etc.), not single-color text that responds to
# # `color`. The only way to get a genuinely on-brand purple icon is to
# # replace them with simple, single-color SVGs using stroke="currentColor",
# # which DOES respond to CSS color. These are generic, functional UI
# # glyphs (a gear, bars, a bulb, an arrow) in the widely-used
# # Feather/Lucide open-source icon style -- not creative or branded art.
# ICON_COLOR = "#7C3AED"

# ICONS = {
#     "settings": (
#         '<circle cx="12" cy="12" r="3"></circle>'
#         '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 '
#         '1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 '
#         '0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 '
#         '0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 '
#         '1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 '
#         '2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 '
#         '0 0-1.51 1z"></path>'
#     ),
#     "bar-chart": (
#         '<line x1="12" y1="20" x2="12" y2="10"></line>'
#         '<line x1="18" y1="20" x2="18" y2="4"></line>'
#         '<line x1="6" y1="20" x2="6" y2="16"></line>'
#     ),
#     "bulb": (
#         '<path d="M9 18h6"></path>'
#         '<path d="M10 22h4"></path>'
#         '<path d="M12 2a7 7 0 0 0-7 7c0 2.5 1.5 4 2.5 5.5S9 18 9 18h6s.5-2 1.5-3.5S19 11.5 19 9a7 7 0 0 0-7-7z"></path>'
#     ),
#     "trending-up": (
#         '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>'
#         '<polyline points="17 6 23 6 23 12"></polyline>'
#     ),
#     "cpu": (
#         '<rect x="4" y="4" width="16" height="16" rx="2"></rect>'
#         '<rect x="9" y="9" width="6" height="6"></rect>'
#         '<line x1="9" y1="1" x2="9" y2="4"></line>'
#         '<line x1="15" y1="1" x2="15" y2="4"></line>'
#         '<line x1="9" y1="20" x2="9" y2="23"></line>'
#         '<line x1="15" y1="20" x2="15" y2="23"></line>'
#         '<line x1="20" y1="9" x2="23" y2="9"></line>'
#         '<line x1="20" y1="14" x2="23" y2="14"></line>'
#         '<line x1="1" y1="9" x2="4" y2="9"></line>'
#         '<line x1="1" y1="14" x2="4" y2="14"></line>'
#     ),
#     "download": (
#         '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>'
#         '<polyline points="7 10 12 15 17 10"></polyline>'
#         '<line x1="12" y1="15" x2="12" y2="3"></line>'
#     ),
# }

# # Matches Streamlit's own default heading sizes/weights exactly (these
# # are the documented defaults when headingFontSizes/Weights aren't
# # overridden in config.toml: h2=2.25rem/600, h3=1.75rem/600) so swapping
# # a real st.header/st.subheader call for this raw-HTML version doesn't
# # change any font size, weight, or spacing -- only the icon.
# _HEADING_STYLES = {
#     "h2": "font-size:2.25rem; font-weight:600; margin:0.5rem 0 1rem 0;",
#     "h3": "font-size:1.75rem; font-weight:600; margin:0.5rem 0 0.75rem 0;",
# }


# def icon_header(icon_name: str, text: str, level: str = "h3", icon_size: str = "0.85em"):
#     """Renders a section header with a purple SVG icon in place of an
#     emoji, at the same size/weight/spacing as the st.header (h2) or
#     st.subheader (h3) it's replacing."""
#     style = _HEADING_STYLES[level]
#     st.markdown(
#         f"""
#         <{level} style="display:flex; align-items:center; gap:0.5rem; {style}">
#             <svg width="{icon_size}" height="{icon_size}" viewBox="0 0 24 24" fill="none"
#                  stroke="{ICON_COLOR}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
#                  style="flex-shrink:0;">
#                 {ICONS[icon_name]}
#             </svg>
#             <span>{text}</span>
#         </{level}>
#         """,
#         unsafe_allow_html=True,
#     )


# # compute_summary_stats() and generate_insights() now live in
# # scripts/report_content.py, shared with the PDF exporter -- see that
# # module for their (unchanged) implementation.


# # --------------------------------------------------------------------
# # Header
# # --------------------------------------------------------------------

# def _load_logo_base64(path: Path) -> str:
#     import base64
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode()


# if _logo_image is not None:
#     logo_b64 = _load_logo_base64(LOGO_PATH)
#     st.markdown(
#         f"""
#         <div style="display:flex; align-items:center; gap:16px; margin-bottom:0.3rem;">
#             <img src="data:image/png;base64,{logo_b64}"
#                  style="width:56px; height:56px; object-fit:contain; flex-shrink:0;" />
#             <h1 style="margin:0; padding:0; font-size:2.5rem; font-weight:700; line-height:1.2;">
#                 RingSync Benchmark Studio
#             </h1>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
# else:
#     st.title("📊 RingSync Benchmark Studio")

# st.caption(
#     "Configure an experiment, run RingSync's real distributed training benchmark "
#     "(actual OS processes, actual TCP ring all-reduce), and explore the results."
# )

# cpu_count = os.cpu_count() or 1
# st.caption(f"Detected **{cpu_count} CPU core(s)** on this machine.")

# st.divider()

# # --------------------------------------------------------------------
# # Experiment Configuration
# # --------------------------------------------------------------------

# with st.container(border=True):
#     icon_header("settings", "Experiment Configuration", level="h3")

#     st.markdown("**Worker Configuration**")
#     worker_options = candidate_worker_counts(cpu_count)
#     default_workers = [w for w in (2, 4, 8) if w in worker_options] or [worker_options[0]]

#     # A multiselect renders each choice as a small removable chip/tag --
#     # this is the "chip-style selection" requested, using a native,
#     # well-supported Streamlit widget rather than a custom component.
#     selected_workers = st.multiselect(
#         "Select worker counts to benchmark",
#         options=worker_options,
#         default=default_workers,
#         label_visibility="collapsed",
#     )
#     if not selected_workers:
#         st.warning("Select at least one worker count to run the benchmark.")
#     st.caption(
#         f"Options above {max(cpu_count - 1, 2)} are hidden on this {cpu_count}-core machine "
#         "to avoid CPU oversubscription."
#     )

#     st.markdown("**Comparison Baseline**")
#     baseline_choice = st.radio(
#         "Compare RingSync against",
#         options=["Default PyTorch", "Single-threaded PyTorch", "Both"],
#         index=2,
#         horizontal=True,
#         label_visibility="collapsed",
#         help=(
#             "Default PyTorch: a normal single-process run, using PyTorch's automatic "
#             "multi-threading across all cores. Single-threaded PyTorch: the same run, "
#             "pinned to 1 thread -- an apples-to-apples comparison against a single "
#             "RingSync worker, which is also pinned to 1 thread."
#         ),
#     )
#     baseline_mode = {
#         "Default PyTorch": "multithread",
#         "Single-threaded PyTorch": "singlethread",
#         "Both": "both",
#     }[baseline_choice]

#     st.markdown("**Training Configuration**")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         batch_size = st.number_input("Batch size", min_value=1, value=128, step=1)
#     with c2:
#         epochs = st.number_input("Epochs", min_value=1, value=3, step=1)
#     with c3:
#         lr = st.number_input("Learning rate", min_value=0.0001, value=0.01, step=0.001, format="%.4f")

#     c4, c5 = st.columns(2)
#     with c4:
#         st.selectbox(
#             "Dataset", options=["Synthetic CIFAR-like (fixed)"], disabled=True,
#             help="Real-dataset selection (e.g. CIFAR-10, MNIST) is a planned future extension.",
#         )
#     with c5:
#         st.selectbox(
#             "Model", options=["Small CNN (fixed)"], disabled=True,
#             help="Model selection is a planned future extension.",
#         )

# st.write("")

# run_clicked = st.button(
#     "▶  Run Benchmark", type="primary",
#     disabled=(len(selected_workers) == 0), width="stretch",
# )

# # ======================================================================
# # Run Log -- unchanged from the previous version, exactly as requested.
# # ======================================================================

# if run_clicked:
#     world_sizes_arg = ",".join(str(w) for w in sorted(selected_workers))
#     cmd = [
#         sys.executable, str(ROOT / "scripts" / "run_benchmark.py"),
#         "--world-sizes", world_sizes_arg,
#         "--epochs", str(int(epochs)),
#         "--batch-size", str(int(batch_size)),
#         "--lr", str(lr),
#         "--baseline", baseline_mode,
#     ]

#     st.subheader("Run Log")
#     log_placeholder = st.empty()
#     log_lines = []

#     _run_start_time = time.time()  # silent timing capture for the Experiment Information card below

#     with st.spinner("Running benchmark -- this can take several minutes depending on worker counts and epochs..."):
#         process = subprocess.Popen(
#             cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
#             text=True, bufsize=1,
#         )
#         for line in process.stdout:
#             log_lines.append(line.rstrip())
#             log_placeholder.code("\n".join(log_lines[-40:]), language="text")
#         process.wait()

#     st.session_state["last_run_duration_s"] = time.time() - _run_start_time

#     if process.returncode != 0:
#         st.error(f"Benchmark process exited with code {process.returncode}. See log above for details.")
#     else:
#         st.success("Benchmark complete.")

# # ======================================================================
# # End of unchanged Run Log section.
# # ======================================================================

# st.divider()

# # --------------------------------------------------------------------
# # Results Dashboard
# # --------------------------------------------------------------------

# data = load_results()

# if data is None:
#     st.info("Configure your experiment above and click **Run Benchmark** to get started.")
# else:
#     cpu_note = f"{cpu_count} CPU core(s) available on this machine"
#     if cpu_count == 1:
#         cpu_note += " -- NOT representative of real multi-core scaling"

#     stats = compute_summary_stats(data)

#     icon_header("bar-chart", "Results Dashboard", level="h2")

#     # --- 1. Benchmark Summary (KPI cards) ---
#     st.subheader("Benchmark Summary")
#     row1 = st.columns(4)
#     kpi_card(row1[0], "Model", "Small CNN")
#     kpi_card(row1[1], "Dataset", "Synthetic CIFAR-like")
#     kpi_card(
#         row1[2], "Workers Tested",
#         ", ".join(str(r["world_size"]) for r in stats["dist_results"]) if stats["dist_results"] else "N/A",
#     )
#     kpi_card(
#         row1[3], "Best Speedup",
#         f"{stats['best_speedup']:.2f}\u00d7" if stats["best_speedup"] else "N/A",
#         f"at {stats['best_speedup_ws']} workers" if stats["best_speedup_ws"] else None,
#     )

#     row2 = st.columns(4)
#     max_ws = stats["max_workers_result"]["world_size"] if stats["max_workers_result"] else None
#     kpi_card(
#         row2[0], "Speedup at Max Workers",
#         f"{stats['speedup_at_max_workers']:.2f}\u00d7" if stats["speedup_at_max_workers"] else "N/A",
#         f"{max_ws} workers" if max_ws else None,
#     )
#     kpi_card(
#         row2[1], "Scaling Efficiency at Max Workers",
#         f"{stats['efficiency_at_max_workers'] * 100:.1f}%" if stats["efficiency_at_max_workers"] else "N/A",
#         f"{max_ws} workers" if max_ws else None,
#     )
#     kpi_card(
#         row2[2], "Max Communication Overhead",
#         f"{stats['max_overhead']:.1f}%" if stats["max_overhead"] is not None else "N/A",
#     )
#     kpi_card(
#         row2[3], "Fastest Configuration",
#         f"{stats['fastest']['world_size']} workers" if stats["fastest"] else "N/A",
#         f"{stats['fastest']['wall_clock_seconds']:.2f}s" if stats["fastest"] else None,
#     )

#     st.caption(
#         "\u2139\ufe0f Speedup and efficiency above are both reported at the largest worker count tested, "
#         "so they describe one real configuration rather than two independently cherry-picked ones. "
#         "Scaling efficiency naturally decreases as workers increase (communication overhead grows) -- "
#         "see the Scaling Efficiency chart below for the full trend."
#     )

#     st.write("")

#     # --- 2. Experiment Information ---
#     st.subheader("Experiment Information")
#     with st.container(border=True):
#         info_cols = st.columns(4)
#         info_cols[0].markdown(
#             f"**Selected workers**\n\n{', '.join(str(ws) for ws in data.get('world_sizes', []))}"
#         )
#         info_cols[1].markdown(f"**Selected baseline**\n\n{data.get('baseline_mode', 'N/A')}")
#         info_cols[2].markdown(f"**Batch size**\n\n{data.get('batch_size', 'N/A')}")
#         info_cols[3].markdown(f"**Epochs**\n\n{data.get('epochs', 'N/A')}")

#         info_cols2 = st.columns(4)
#         info_cols2[0].markdown(f"**Learning rate**\n\n{data.get('learning_rate', 'N/A')}")
#         info_cols2[1].markdown("**Dataset**\n\nSynthetic CIFAR-like")
#         info_cols2[2].markdown("**Model**\n\nSmall CNN")
#         last_duration = st.session_state.get("last_run_duration_s")
#         info_cols2[3].markdown(
#             "**Total benchmark runtime**\n\n"
#             + (f"{last_duration:.1f}s" if last_duration else "N/A (re-run in this session to measure)")
#         )

#     st.write("")

#     # --- System Information ---
#     icon_header("cpu", "System Information", level="h3")
#     sysinfo = get_system_info()
#     with st.container(border=True):
#         sys_cols = st.columns(4)
#         sys_cols[0].markdown(f"**CPU**\n\n{sysinfo['cpu']}")
#         sys_cols[1].markdown(f"**Logical Cores**\n\n{sysinfo['logical_cores']}")
#         sys_cols[2].markdown(f"**Physical Cores**\n\n{sysinfo['physical_cores']}")
#         sys_cols[3].markdown(f"**RAM**\n\n{sysinfo['ram_gb']}")

#         sys_cols2 = st.columns(4)
#         sys_cols2[0].markdown(f"**Python**\n\n{sysinfo['python_version']}")
#         sys_cols2[1].markdown(f"**PyTorch**\n\n{sysinfo['pytorch_version']}")
#         sys_cols2[2].markdown(f"**Operating System**\n\n{sysinfo['os']}")

#     st.caption(
#         "Benchmark results should always be interpreted alongside the hardware configuration. "
#         "The same speedup can represent very different performance characteristics on different systems."
#     )

    
#     st.write("")

#     # --- 3. Benchmark Insights (auto-generated narrative) ---
#     insights = generate_insights(data, stats)
#     if insights:
#         icon_header("bulb", "Benchmark Insights", level="h3")
#         with st.container(border=True):
#             for point in insights:
#                 st.markdown(f"- {point}")
#         st.write("")

#     # --- 4. Performance Graphs ---
#     st.subheader("Performance Graphs")

#     speedup_multi_fig = plot_speedup_vs_multithread(data, cpu_count_note=cpu_note)
#     speedup_single_fig = plot_speedup_vs_singlethread(data, cpu_count_note=cpu_note)
#     compute_comm_fig = plot_compute_vs_comm(data)
#     overhead_fig = plot_communication_overhead_pct(data)
#     efficiency_fig = plot_scaling_efficiency(data)

#     if speedup_multi_fig or speedup_single_fig:
#         cols = st.columns(2)
#         if speedup_multi_fig:
#             with cols[0]:
#                 st.pyplot(speedup_multi_fig, width="stretch")
#                 plt.close(speedup_multi_fig)
#         if speedup_single_fig:
#             with cols[1]:
#                 st.pyplot(speedup_single_fig, width="stretch")
#                 plt.close(speedup_single_fig)
#     else:
#         st.info("No speedup chart available for this run.")

#     cols2 = st.columns(2)
#     with cols2[0]:
#         if compute_comm_fig:
#             st.pyplot(compute_comm_fig, width="stretch")
#             plt.close(compute_comm_fig)
#         else:
#             st.info("No compute/communication data available.")
#     with cols2[1]:
#         if overhead_fig:
#             st.pyplot(overhead_fig, width="stretch")
#             plt.close(overhead_fig)
#         else:
#             st.info("No communication overhead data available.")

#     # Scaling efficiency gets its own full-width row -- it's the metric
#     # that replaced a single (misleading) KPI number, so it earns visual
#     # prominence as the trend that actually explains the KPI above.
#     if efficiency_fig:
#         st.pyplot(efficiency_fig, width="stretch")
#         plt.close(efficiency_fig)
#     else:
#         st.info("No scaling efficiency data available.")

#     st.write("")

#     # --- 5. Training Behaviour ---
#     icon_header("trending-up", "Training Behaviour", level="h3")
#     loss_fig = plot_training_loss_vs_epoch(data)
#     if loss_fig:
#         st.pyplot(loss_fig, width="stretch")
#         plt.close(loss_fig)
#     else:
#         st.info("No per-epoch loss data available.")

#     st.write("")
#     st.divider()

#     # --- Export Benchmark Report ---
#     icon_header("download", "Export Benchmark Report", level="h3")
#     st.caption(
#         "Generates a self-contained PDF with the logo, benchmark summary, experiment "
#         "configuration, system information, key insights, timestamp, and every chart above -- "
#         "everything needed to interpret these results without the live app open."
#     )
#     try:
#         pdf_output_path = RESULTS_DIR / "ringsync_benchmark_report.pdf"
#         generate_pdf_report(data, pdf_output_path, logo_path=LOGO_PATH)
#         with open(pdf_output_path, "rb") as f:
#             pdf_bytes = f.read()
#         st.download_button(
#             "⬇ Download PDF Report",
#             data=pdf_bytes,
#             file_name="ringsync_benchmark_report.pdf",
#             mime="application/pdf",
#             type="primary",
#             width="stretch",
#         )
#     except Exception as e:
#         st.error(f"Could not generate the PDF report: {e}")










"""
RingSync Benchmark Studio.

An interactive local web app (Streamlit) for configuring, running, and
visualizing RingSync benchmarks -- replaces manually editing constants
in scripts/run_benchmark.py and running it from the command line.

This file is UI/presentation only. It does not implement or alter any
benchmarking logic, distributed training code, or metric calculation --
all of that lives in scripts/benchmark_core.py and scripts/plot_benchmark.py,
unchanged, and this file only arranges how their outputs are configured,
triggered, and displayed.

Design note on WHY the benchmark runs as a subprocess rather than
calling scripts/benchmark_core.py's functions directly in-process:
RingSync's distributed workers are spawned via Python's multiprocessing
module, which on Windows uses the "spawn" start method -- this re-
launches a fresh Python interpreter for each worker. Streamlit apps
run inside their own already-unusual process/thread model, and mixing
that with multiprocessing.Process spawned directly from within a
running Streamlit app is a known source of platform-specific breakage
(worker processes can end up trying to re-import and re-run the
Streamlit app itself). Invoking the existing, already-tested CLI script
(scripts/run_benchmark.py) as a clean subprocess sidesteps this
entirely. Plotting (scripts/plot_benchmark.py) has no multiprocessing
in it at all, so that IS called directly, in-process.

Run:
    streamlit run dashboard/benchmark_studio.py
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import plot_benchmark, report_export
from scripts.plot_benchmark import (
    plot_speedup_vs_multithread,
    plot_speedup_vs_singlethread,
    plot_compute_vs_comm,
    plot_communication_overhead_pct,
    plot_scaling_efficiency,
    plot_training_loss_vs_epoch,
    plot_compare_runs,
)
from scripts.report_content import compute_summary_stats, generate_insights
from scripts.report_export import generate_pdf_report
from scripts.system_info import get_system_info
from scripts.experiment_utils import generate_experiment_id, get_experiment_dir, list_experiments, RESULTS_ROOT
from scripts.resource_monitor import ProcessTreeMonitor, sample_network_io

# Logo, used for both the browser tab icon and the page header. Prefers
# the transparent variant (see scripts/make_logo_transparent.py) since a
# non-transparent logo shows a visible background square as a favicon;
# falls back to the original file, then to the emoji, so a missing or
# not-yet-processed asset never crashes the app.
LOGO_PATH = ROOT / "assets" / "logo_transparent.png"
if not LOGO_PATH.exists():
    LOGO_PATH = ROOT / "assets" / "logo.png"
_logo_image = Image.open(LOGO_PATH) if LOGO_PATH.exists() else None

RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"  # legacy flat-folder location, kept for backward compatibility


def parse_progress_line(line: str) -> "dict | None":
    """
    Parses a single line of subprocess output for the RS_PROGRESS|<json>
    sentinel format emitted by benchmark_core.py, worker_node.py, and
    train_single_process.py. Returns the parsed event dict, or None if
    the line isn't a progress sentinel (i.e. it's an ordinary log line
    meant for the Run Log display instead).

    Kept as a pure function (no Streamlit calls) so it can be unit
    tested directly without needing a live app or subprocess.
    """
    prefix = "RS_PROGRESS|"
    if not line.startswith(prefix):
        return None
    try:
        return json.loads(line[len(prefix):])
    except json.JSONDecodeError:
        return None


def resolve_active_experiment():
    """
    Decides which experiment's results the dashboard should currently
    display: prefers the experiment just run in this session, falls
    back to the most recent experiment on disk, then to the legacy
    flat results/benchmark_results.json for backward compatibility
    with results generated before per-experiment folders existed.
    Returns (experiment_id_or_None, results_dir_Path, data_dict_or_None).
    """
    current_id = st.session_state.get("current_experiment_id")
    if current_id:
        exp_dir = RESULTS_ROOT / current_id
        exp_json = exp_dir / "benchmark_results.json"
        if exp_json.exists():
            with open(exp_json) as f:
                return current_id, exp_dir, json.load(f)

    experiments = list_experiments()
    if experiments:
        latest = experiments[0]  # list_experiments() already sorts newest-first
        return latest["experiment_id"], RESULTS_ROOT / latest["experiment_id"], latest["data"]

    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return None, RESULTS_DIR, json.load(f)

    return None, None, None

st.set_page_config(
    page_title="RingSync Benchmark Studio",
    page_icon=_logo_image if _logo_image is not None else "📊",
    layout="wide",
)

# A single config.toml with [theme.light] and [theme.dark] sections
# (verified working: Streamlit correctly resolves both from one file)
# means Streamlit's OWN native theme picker (top-right "⋮" -> Settings
# -> Theme) now switches between RingSync's exact light and dark
# palettes -- no custom toggle or live CSS override needed. This CSS
# only adjusts SHAPE (size, padding, radius, shadow), deliberately
# leaving color untouched so the button correctly follows whichever
# theme -- and therefore whichever primaryColor -- is actually active,
# rather than being locked to one hardcoded hex value.
st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.9rem 2.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def candidate_worker_counts(cpu_count: int) -> "list[int]":
    """
    A curated, not-overwhelming set of selectable worker counts, capped
    below the machine's core count. Deliberately leaves at least 1 core
    unclaimed (cpu_count - 1 ceiling) -- pushing worker count all the
    way to cpu_count leaves no headroom for the OS, Streamlit itself,
    or the baseline's own multi-threaded run, and risks the same kind
    of oversubscription contention the project already had to fix once.
    """
    max_workers = max(cpu_count - 1, 2)
    all_candidates = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
    return [w for w in all_candidates if w <= max_workers] or [2]


def kpi_card(column, label: str, value: str, delta: str = None):
    """A single bordered KPI card -- the building block of the
    Benchmark Summary and Grafana/W&B-style dashboard feel."""
    with column:
        with st.container(border=True):
            st.metric(label, value, delta)


# --------------------------------------------------------------------
# Purple section-header icons
# --------------------------------------------------------------------
# Emoji (⚙️ 📊 💡 📈) can't be recolored with CSS -- they're pre-rendered,
# multi-color glyphs supplied by the OS/browser (Twemoji, Segoe UI Emoji,
# Apple Color Emoji, etc.), not single-color text that responds to
# `color`. The only way to get a genuinely on-brand purple icon is to
# replace them with simple, single-color SVGs using stroke="currentColor",
# which DOES respond to CSS color. These are generic, functional UI
# glyphs (a gear, bars, a bulb, an arrow) in the widely-used
# Feather/Lucide open-source icon style -- not creative or branded art.
ICON_COLOR = "#7C3AED"

ICONS = {
    "settings": (
        '<circle cx="12" cy="12" r="3"></circle>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 '
        '1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 '
        '0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 '
        '0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 '
        '1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 '
        '2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 '
        '0 0-1.51 1z"></path>'
    ),
    "bar-chart": (
        '<line x1="12" y1="20" x2="12" y2="10"></line>'
        '<line x1="18" y1="20" x2="18" y2="4"></line>'
        '<line x1="6" y1="20" x2="6" y2="16"></line>'
    ),
    "bulb": (
        '<path d="M9 18h6"></path>'
        '<path d="M10 22h4"></path>'
        '<path d="M12 2a7 7 0 0 0-7 7c0 2.5 1.5 4 2.5 5.5S9 18 9 18h6s.5-2 1.5-3.5S19 11.5 19 9a7 7 0 0 0-7-7z"></path>'
    ),
    "trending-up": (
        '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>'
        '<polyline points="17 6 23 6 23 12"></polyline>'
    ),
    "cpu": (
        '<rect x="4" y="4" width="16" height="16" rx="2"></rect>'
        '<rect x="9" y="9" width="6" height="6"></rect>'
        '<line x1="9" y1="1" x2="9" y2="4"></line>'
        '<line x1="15" y1="1" x2="15" y2="4"></line>'
        '<line x1="9" y1="20" x2="9" y2="23"></line>'
        '<line x1="15" y1="20" x2="15" y2="23"></line>'
        '<line x1="20" y1="9" x2="23" y2="9"></line>'
        '<line x1="20" y1="14" x2="23" y2="14"></line>'
        '<line x1="1" y1="9" x2="4" y2="9"></line>'
        '<line x1="1" y1="14" x2="4" y2="14"></line>'
    ),
    "download": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>'
        '<polyline points="7 10 12 15 17 10"></polyline>'
        '<line x1="12" y1="15" x2="12" y2="3"></line>'
    ),
    "clock": (
        '<circle cx="12" cy="12" r="10"></circle>'
        '<polyline points="12 6 12 12 16 14"></polyline>'
    ),
}

# Matches Streamlit's own default heading sizes/weights exactly (these
# are the documented defaults when headingFontSizes/Weights aren't
# overridden in config.toml: h2=2.25rem/600, h3=1.75rem/600) so swapping
# a real st.header/st.subheader call for this raw-HTML version doesn't
# change any font size, weight, or spacing -- only the icon.
_HEADING_STYLES = {
    "h2": "font-size:2.25rem; font-weight:600; margin:0.5rem 0 1rem 0;",
    "h3": "font-size:1.75rem; font-weight:600; margin:0.5rem 0 0.75rem 0;",
}


def icon_header(icon_name: str, text: str, level: str = "h3", icon_size: str = "0.85em"):
    """Renders a section header with a purple SVG icon in place of an
    emoji, at the same size/weight/spacing as the st.header (h2) or
    st.subheader (h3) it's replacing."""
    style = _HEADING_STYLES[level]
    st.markdown(
        f"""
        <{level} style="display:flex; align-items:center; gap:0.5rem; {style}">
            <svg width="{icon_size}" height="{icon_size}" viewBox="0 0 24 24" fill="none"
                 stroke="{ICON_COLOR}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                 style="flex-shrink:0;">
                {ICONS[icon_name]}
            </svg>
            <span>{text}</span>
        </{level}>
        """,
        unsafe_allow_html=True,
    )


# compute_summary_stats() and generate_insights() now live in
# scripts/report_content.py, shared with the PDF exporter -- see that
# module for their (unchanged) implementation.


# --------------------------------------------------------------------
# Header
# --------------------------------------------------------------------

def _load_logo_base64(path: Path) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


if _logo_image is not None:
    logo_b64 = _load_logo_base64(LOGO_PATH)
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:0.3rem;">
            <img src="data:image/png;base64,{logo_b64}"
                 style="width:56px; height:56px; object-fit:contain; flex-shrink:0;" />
            <h1 style="margin:0; padding:0; font-size:2.5rem; font-weight:700; line-height:1.2;">
                RingSync Benchmark Studio
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("📊 RingSync Benchmark Studio")

st.caption(
    "Configure an experiment, run RingSync's real distributed training benchmark "
    "(actual OS processes, actual TCP ring all-reduce), and explore the results."
)

cpu_count = os.cpu_count() or 1
st.caption(f"Detected **{cpu_count} CPU core(s)** on this machine.")

st.divider()

# --------------------------------------------------------------------
# Experiment Configuration
# --------------------------------------------------------------------

with st.container(border=True):
    icon_header("settings", "Experiment Configuration", level="h3")

    st.markdown("**Worker Configuration**")
    worker_options = candidate_worker_counts(cpu_count)
    default_workers = [w for w in (2, 4, 8) if w in worker_options] or [worker_options[0]]

    # A multiselect renders each choice as a small removable chip/tag --
    # this is the "chip-style selection" requested, using a native,
    # well-supported Streamlit widget rather than a custom component.
    selected_workers = st.multiselect(
        "Select worker counts to benchmark",
        options=worker_options,
        default=default_workers,
        label_visibility="collapsed",
    )
    if not selected_workers:
        st.warning("Select at least one worker count to run the benchmark.")
    st.caption(
        f"Options above {max(cpu_count - 1, 2)} are hidden on this {cpu_count}-core machine "
        "to avoid CPU oversubscription."
    )

    st.markdown("**Comparison Baseline**")
    baseline_choice = st.radio(
        "Compare RingSync against",
        options=["Default PyTorch", "Single-threaded PyTorch", "Both"],
        index=2,
        horizontal=True,
        label_visibility="collapsed",
        help=(
            "Default PyTorch: a normal single-process run, using PyTorch's automatic "
            "multi-threading across all cores. Single-threaded PyTorch: the same run, "
            "pinned to 1 thread -- an apples-to-apples comparison against a single "
            "RingSync worker, which is also pinned to 1 thread."
        ),
    )
    baseline_mode = {
        "Default PyTorch": "multithread",
        "Single-threaded PyTorch": "singlethread",
        "Both": "both",
    }[baseline_choice]

    st.markdown("**Training Configuration**")
    c1, c2, c3 = st.columns(3)
    with c1:
        batch_size = st.number_input("Batch size", min_value=1, value=128, step=1)
    with c2:
        epochs = st.number_input("Epochs", min_value=1, value=3, step=1)
    with c3:
        lr = st.number_input("Learning rate", min_value=0.0001, value=0.01, step=0.001, format="%.4f")

    c4, c5 = st.columns(2)
    with c4:
        st.selectbox(
            "Dataset", options=["Synthetic CIFAR-like (fixed)"], disabled=True,
            help="Real-dataset selection (e.g. CIFAR-10, MNIST) is a planned future extension.",
        )
    with c5:
        st.selectbox(
            "Model", options=["Small CNN (fixed)"], disabled=True,
            help="Model selection is a planned future extension.",
        )

st.write("")

run_clicked = st.button(
    "▶  Run Benchmark", type="primary",
    disabled=(len(selected_workers) == 0), width="stretch",
)

if run_clicked:
    experiment_id = generate_experiment_id()
    world_sizes_arg = ",".join(str(w) for w in sorted(selected_workers))
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_benchmark.py"),
        "--world-sizes", world_sizes_arg,
        "--epochs", str(int(epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(lr),
        "--baseline", baseline_mode,
        "--experiment-id", experiment_id,
    ]

    st.info(f"Experiment ID: **{experiment_id}**")

    # Ordered list of configs this run will execute, for the progress bars.
    planned_configs = []
    if baseline_mode in ("multithread", "both"):
        planned_configs.append(("baseline", "Baseline (multi-threaded)"))
    if baseline_mode in ("singlethread", "both"):
        planned_configs.append(("baseline_singlethread", "Baseline (single-threaded)"))
    for w in sorted(selected_workers):
        planned_configs.append((f"world_size_{w}", f"{w} workers"))

    progress_state = {
        tag: {"status": "pending", "epoch": 0, "total_epochs": int(epochs)}
        for tag, _label in planned_configs
    }

    # --- 3. Live Progress ---
    icon_header("bar-chart", "Live Progress", level="h3")
    progress_bars = {}
    for tag, label in planned_configs:
        progress_bars[tag] = st.progress(0, text=f"{label}: waiting...")

    # --- 6. Resource Monitoring ---
    icon_header("cpu", "Resource Monitoring", level="h3")
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        cpu_bar = st.progress(0, text="CPU: 0%")
    with res_col2:
        ram_bar = st.progress(0, text="RAM: 0%")
    with res_col3:
        proc_count_display = st.empty()
    net_display = st.caption("Network I/O since run started: 0.0 KB sent, 0.0 KB received (system-wide)")

    st.subheader("Run Log")
    log_placeholder = st.empty()
    log_lines = []

    _run_start_time = time.time()

    process = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    # The subprocess's stdout is read on a background thread into a
    # queue, rather than the main thread blocking on `for line in
    # process.stdout`, specifically so the main thread can also poll
    # resource usage on a fixed interval even when no new log line has
    # arrived yet -- a blocking read loop can't interleave periodic
    # sampling with itself.
    line_queue: "queue.Queue" = queue.Queue()

    def _reader(pipe, q):
        for line in pipe:
            q.put(line)
        q.put(None)  # sentinel: stream closed

    reader_thread = threading.Thread(target=_reader, args=(process.stdout, line_queue), daemon=True)
    reader_thread.start()

    monitor = ProcessTreeMonitor()
    net_start = sample_network_io()
    logical_cores = os.cpu_count() or 1
    stream_closed = False

    with st.spinner("Running benchmark -- this can take several minutes depending on worker counts and epochs..."):
        while True:
            drained_any = False
            while True:
                try:
                    line = line_queue.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    stream_closed = True
                    break
                drained_any = True
                stripped = line.rstrip()
                event = parse_progress_line(stripped)
                if event is not None:
                    etype = event.get("type")
                    cfg = event.get("config")
                    if etype == "config_start" and cfg in progress_state:
                        progress_state[cfg]["status"] = "running"
                    elif etype == "config_done" and cfg in progress_state:
                        progress_state[cfg]["status"] = "done"
                        progress_state[cfg]["epoch"] = progress_state[cfg]["total_epochs"]
                    elif etype == "epoch" and cfg in progress_state:
                        progress_state[cfg]["status"] = "running"
                        progress_state[cfg]["epoch"] = event["epoch"]
                        progress_state[cfg]["total_epochs"] = event["total_epochs"]
                else:
                    log_lines.append(stripped)

            if drained_any:
                log_placeholder.code("\n".join(log_lines[-40:]), language="text")
                for tag, label in planned_configs:
                    s = progress_state[tag]
                    pct = int(100 * s["epoch"] / max(s["total_epochs"], 1))
                    if s["status"] == "pending":
                        progress_bars[tag].progress(0, text=f"{label}: waiting...")
                    elif s["status"] == "done":
                        progress_bars[tag].progress(100, text=f"{label}: complete \u2705")
                    else:
                        progress_bars[tag].progress(
                            pct, text=f"{label}: {pct}% (epoch {s['epoch']}/{s['total_epochs']})",
                        )

            res_stats = monitor.sample(process.pid)
            net_now = sample_network_io()
            sent_kb = max(net_now["bytes_sent"] - net_start["bytes_sent"], 0) / 1024
            recv_kb = max(net_now["bytes_recv"] - net_start["bytes_recv"], 0) / 1024
            cpu_pct_normalized = min(res_stats["cpu_percent"] / logical_cores, 100.0)

            cpu_bar.progress(int(cpu_pct_normalized), text=f"CPU: {cpu_pct_normalized:.0f}%")
            ram_bar.progress(int(min(res_stats["ram_percent"], 100.0)), text=f"RAM: {res_stats['ram_percent']:.0f}%")
            proc_count_display.metric("Processes", res_stats["process_count"])
            net_display.caption(
                f"Network I/O since run started: {sent_kb:.1f} KB sent, {recv_kb:.1f} KB received (system-wide)"
            )

            if stream_closed and process.poll() is not None:
                break
            time.sleep(0.4)

    process.wait()
    reader_thread.join(timeout=5)

    # Final pass: mark every config still "running" as done, in case the
    # last config_done sentinel arrived in the same batch as stream close.
    for tag, label in planned_configs:
        s = progress_state[tag]
        if s["status"] != "done":
            s["status"] = "done"
            s["epoch"] = s["total_epochs"]
        pct = 100
        progress_bars[tag].progress(pct, text=f"{label}: complete \u2705")

    st.session_state["last_run_duration_s"] = time.time() - _run_start_time
    st.session_state["current_experiment_id"] = experiment_id

    if process.returncode != 0:
        st.error(f"Benchmark process exited with code {process.returncode}. See log above for details.")
    else:
        st.success(f"Benchmark complete. Experiment ID: {experiment_id}")

st.divider()

# --------------------------------------------------------------------
# Results Dashboard
# --------------------------------------------------------------------

active_experiment_id, active_results_dir, data = resolve_active_experiment()

if data is None:
    st.info("Configure your experiment above and click **Run Benchmark** to get started.")
else:
    # Point the plotting and PDF-export modules at this specific
    # experiment's folder (rather than the legacy flat results/) before
    # generating anything -- see scripts/plot_benchmark.py's
    # set_results_dir for why this is a module-level pointer rather
    # than a parameter threaded through every function.
    plot_benchmark.set_results_dir(active_results_dir)
    report_export.set_results_dir(active_results_dir)

    cpu_note = f"{cpu_count} CPU core(s) available on this machine"
    if cpu_count == 1:
        cpu_note += " -- NOT representative of real multi-core scaling"

    stats = compute_summary_stats(data)

    icon_header("bar-chart", "Results Dashboard", level="h2")
    if active_experiment_id:
        st.caption(f"Showing experiment **{active_experiment_id}**")

    # --- 1. Benchmark Summary (KPI cards) ---
    st.subheader("Benchmark Summary")
    row1 = st.columns(4)
    kpi_card(row1[0], "Model", "Small CNN")
    kpi_card(row1[1], "Dataset", "Synthetic CIFAR-like")
    kpi_card(
        row1[2], "Workers Tested",
        ", ".join(str(r["world_size"]) for r in stats["dist_results"]) if stats["dist_results"] else "N/A",
    )
    kpi_card(
        row1[3], "Best Speedup",
        f"{stats['best_speedup']:.2f}\u00d7" if stats["best_speedup"] else "N/A",
        f"at {stats['best_speedup_ws']} workers" if stats["best_speedup_ws"] else None,
    )

    row2 = st.columns(4)
    max_ws = stats["max_workers_result"]["world_size"] if stats["max_workers_result"] else None
    kpi_card(
        row2[0], "Speedup at Largest Configuration",
        f"{stats['speedup_at_max_workers']:.2f}\u00d7" if stats["speedup_at_max_workers"] else "N/A",
        f"{max_ws} workers" if max_ws else None,
    )
    kpi_card(
        row2[1], "Scaling Efficiency at Max Workers",
        f"{stats['efficiency_at_max_workers'] * 100:.1f}%" if stats["efficiency_at_max_workers"] else "N/A",
        f"{max_ws} workers" if max_ws else None,
    )
    kpi_card(
        row2[2], "Max Communication Overhead",
        f"{stats['max_overhead']:.1f}%" if stats["max_overhead"] is not None else "N/A",
    )
    kpi_card(
        row2[3], "Fastest Configuration",
        f"{stats['fastest']['world_size']} workers" if stats["fastest"] else "N/A",
        f"{stats['fastest']['wall_clock_seconds']:.2f}s" if stats["fastest"] else None,
    )

    st.caption(
        "\u2139\ufe0f Speedup and efficiency above are both reported at the largest worker count tested, "
        "so they describe one real configuration rather than two independently cherry-picked ones. "
        "Scaling efficiency naturally decreases as workers increase (communication overhead grows) -- "
        "see the Scaling Efficiency chart below for the full trend."
    )

    st.write("")

    # --- 2. Experiment Information ---
    st.subheader("Experiment Information")
    with st.container(border=True):
        info_cols = st.columns(4)
        info_cols[0].markdown(f"**Experiment**\n\n{data.get('experiment_id', active_experiment_id or 'N/A')}")
        info_cols[1].markdown(
            f"**Selected workers**\n\n{', '.join(str(ws) for ws in data.get('world_sizes', []))}"
        )
        info_cols[2].markdown(f"**Selected baseline**\n\n{data.get('baseline_mode', 'N/A')}")
        info_cols[3].markdown(f"**Batch size**\n\n{data.get('batch_size', 'N/A')}")

        info_cols2 = st.columns(4)
        info_cols2[0].markdown(f"**Epochs**\n\n{data.get('epochs', 'N/A')}")
        info_cols2[1].markdown(f"**Learning rate**\n\n{data.get('learning_rate', 'N/A')}")
        info_cols2[2].markdown("**Dataset**\n\nSynthetic CIFAR-like")
        info_cols2[3].markdown("**Model**\n\nSmall CNN")

        last_duration = st.session_state.get("last_run_duration_s")
        st.markdown(
            "**Total benchmark runtime:** "
            + (f"{last_duration:.1f}s" if last_duration else "N/A (re-run in this session to measure)")
        )

    st.write("")

    # --- System Information ---
    icon_header("cpu", "System Information", level="h3")
    sysinfo = get_system_info()
    with st.container(border=True):
        sys_cols = st.columns(4)
        sys_cols[0].markdown(f"**CPU**\n\n{sysinfo['cpu']}")
        sys_cols[1].markdown(f"**Logical Cores**\n\n{sysinfo['logical_cores']}")
        sys_cols[2].markdown(f"**Physical Cores**\n\n{sysinfo['physical_cores']}")
        sys_cols[3].markdown(f"**RAM**\n\n{sysinfo['ram_gb']}")

        sys_cols2 = st.columns(4)
        sys_cols2[0].markdown(f"**Python**\n\n{sysinfo['python_version']}")
        sys_cols2[1].markdown(f"**PyTorch**\n\n{sysinfo['pytorch_version']}")
        sys_cols2[2].markdown(f"**Operating System**\n\n{sysinfo['os']}")

    st.caption(
        "Hardware context matters: the same speedup number means something different on a "
        "4-core laptop than a 32-core workstation."
    )
    st.write("")

    # --- 3. Benchmark Insights (auto-generated narrative) ---
    insights = generate_insights(data, stats)
    if insights:
        icon_header("bulb", "Benchmark Insights", level="h3")
        with st.container(border=True):
            for point in insights:
                st.markdown(f"- {point}")
        st.write("")

    # --- 4. Performance Graphs ---
    st.subheader("Performance Graphs")

    speedup_multi_fig = plot_speedup_vs_multithread(data, cpu_count_note=cpu_note)
    speedup_single_fig = plot_speedup_vs_singlethread(data, cpu_count_note=cpu_note)
    compute_comm_fig = plot_compute_vs_comm(data)
    overhead_fig = plot_communication_overhead_pct(data)
    efficiency_fig = plot_scaling_efficiency(data)

    if speedup_multi_fig or speedup_single_fig:
        cols = st.columns(2)
        if speedup_multi_fig:
            with cols[0]:
                st.pyplot(speedup_multi_fig, width="stretch")
                plt.close(speedup_multi_fig)
        if speedup_single_fig:
            with cols[1]:
                st.pyplot(speedup_single_fig, width="stretch")
                plt.close(speedup_single_fig)
    else:
        st.info("No speedup chart available for this run.")

    cols2 = st.columns(2)
    with cols2[0]:
        if compute_comm_fig:
            st.pyplot(compute_comm_fig, width="stretch")
            plt.close(compute_comm_fig)
        else:
            st.info("No compute/communication data available.")
    with cols2[1]:
        if overhead_fig:
            st.pyplot(overhead_fig, width="stretch")
            plt.close(overhead_fig)
        else:
            st.info("No communication overhead data available.")

    # Scaling efficiency gets its own full-width row -- it's the metric
    # that replaced a single (misleading) KPI number, so it earns visual
    # prominence as the trend that actually explains the KPI above.
    if efficiency_fig:
        st.pyplot(efficiency_fig, width="stretch")
        plt.close(efficiency_fig)
    else:
        st.info("No scaling efficiency data available.")

    st.write("")

    # --- 5. Training Behaviour ---
    icon_header("trending-up", "Training Behaviour", level="h3")
    loss_fig = plot_training_loss_vs_epoch(data)
    if loss_fig:
        st.pyplot(loss_fig, width="stretch")
        plt.close(loss_fig)
    else:
        st.info("No per-epoch loss data available.")

    st.write("")
    st.divider()

    # --- Export Benchmark Report ---
    icon_header("download", "Export Benchmark Report", level="h3")
    st.caption(
        "Generates a self-contained PDF with the logo, benchmark summary, experiment "
        "configuration, system information, key insights, timestamp, and every chart above -- "
        "everything needed to interpret these results without the live app open."
    )
    try:
        pdf_output_path = active_results_dir / "ringsync_benchmark_report.pdf"
        generate_pdf_report(data, pdf_output_path, logo_path=LOGO_PATH)
        with open(pdf_output_path, "rb") as f:
            pdf_bytes = f.read()
        download_filename = (
            f"ringsync_benchmark_report_{active_experiment_id}.pdf"
            if active_experiment_id else "ringsync_benchmark_report.pdf"
        )
        st.download_button(
            "⬇ Download PDF Report",
            data=pdf_bytes,
            file_name=download_filename,
            mime="application/pdf",
            type="primary",
            width="stretch",
        )
    except Exception as e:
        st.error(f"Could not generate the PDF report: {e}")

st.divider()

# --------------------------------------------------------------------
# Benchmark History
# --------------------------------------------------------------------
# Shown regardless of whether there's a currently-active experiment
# above, since it lists everything on disk -- a fresh install with no
# runs yet just sees an empty state here.

icon_header("clock", "Benchmark History", level="h2")

past_experiments = list_experiments()

if not past_experiments:
    st.info("No past experiments yet -- run a benchmark above to start building history.")
else:
    history_rows = []
    for exp in past_experiments:
        history_rows.append({
            "Experiment": exp["experiment_id"],
            "Date": exp["date"],
            "Workers": exp["workers_tested"],
            "Baseline": exp["baseline_mode"],
            "Speedup": f"{exp['best_speedup']:.2f}\u00d7" if exp["best_speedup"] else "N/A",
        })
    st.dataframe(history_rows, width="stretch", hide_index=True)

    st.markdown("**Compare Runs**")
    experiment_labels = {exp["experiment_id"]: exp for exp in past_experiments}
    selected_ids = st.multiselect(
        "Select two or more experiments to overlay their speedup curves",
        options=list(experiment_labels.keys()),
        default=list(experiment_labels.keys())[:min(2, len(experiment_labels))],
    )

    if len(selected_ids) >= 2:
        selected_experiments = [experiment_labels[eid] for eid in selected_ids]
        compare_fig = plot_compare_runs(selected_experiments)
        if compare_fig is not None:
            st.pyplot(compare_fig, width="stretch")
            plt.close(compare_fig)
        else:
            st.warning("Selected experiments don't have comparable speedup data to plot.")
    elif len(selected_ids) == 1:
        st.caption("Select at least one more experiment to compare.")

st.divider()
st.markdown(
    """
    <div style="text-align:center; padding: 1.5rem 0 1rem 0; opacity:0.7; font-size:0.85rem;">
        RingSync Benchmark Studio &middot; Built by <b>Noel Ann Roy</b> &middot; 2026
    </div>
    """,
    unsafe_allow_html=True,
)