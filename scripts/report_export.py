# """
# PDF benchmark report exporter.

# Builds a self-contained PDF containing everything needed to interpret
# the benchmark results without also having the live dashboard open:
# logo, timestamp, benchmark summary, experiment configuration, system
# information (hardware context -- a speedup number means something
# different on a 4-core laptop than a 32-core workstation), auto-
# generated insights, and every chart.

# Uses reportlab (pure Python, no external system binaries like
# wkhtmltopdf needed -- important for a clean Windows setup) rather than
# an HTML-to-PDF converter.

# Summary stats and insights are computed via scripts/report_content.py,
# the SAME module the Streamlit dashboard uses -- the PDF and the live
# app will never disagree about what the "best speedup" was.
# """

# from datetime import datetime
# from pathlib import Path

# from PIL import Image as PILImage
# from reportlab.lib import colors
# from reportlab.lib.pagesizes import letter
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.units import inch
# from reportlab.platypus import (
#     SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
#     PageBreak, KeepTogether,
# )

# from scripts.report_content import compute_summary_stats, generate_insights
# from scripts.system_info import get_system_info

# ROOT = Path(__file__).resolve().parent.parent
# RESULTS_DIR = ROOT / "results"
# ACCENT_COLOR = colors.HexColor("#7C3AED")
# BORDER_COLOR = colors.HexColor("#e5e7eb")

# # (display title, filename in results/) -- filenames match exactly what
# # plot_benchmark.py already saves to disk, so no chart re-rendering is
# # needed here, just reading the existing PNGs.
# CHART_FILES = [
#     ("Speedup vs. Default Multi-threaded Baseline", "speedup_vs_multithread_baseline.png"),
#     ("Speedup vs. Single-threaded Baseline", "speedup_vs_singlethread_baseline.png"),
#     ("Compute vs. Communication Time per Step", "compute_vs_comm_chart.png"),
#     ("Communication Overhead vs. Number of Workers", "communication_overhead_pct.png"),
#     ("Scaling Efficiency vs. Number of Workers", "scaling_efficiency_pct.png"),
#     ("Training Loss vs. Epoch", "training_loss_vs_epoch.png"),
# ]


# def _md_bold_to_html(text: str) -> str:
#     """Converts markdown **bold** (used in generate_insights' output,
#     written for Streamlit's st.markdown) into reportlab-compatible
#     <b>bold</b> tags."""
#     parts = text.split("**")
#     html = ""
#     for i, part in enumerate(parts):
#         html += f"<b>{part}</b>" if i % 2 == 1 else part
#     return html


# def _kv_table(rows: "list[list[str]]") -> Table:
#     table = Table(rows, colWidths=[2.3 * inch, 3.7 * inch])
#     table.setStyle(TableStyle([
#         ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
#         ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
#         ("FONTSIZE", (0, 0), (-1, -1), 10),
#         ("TEXTCOLOR", (0, 0), (0, -1), ACCENT_COLOR),
#         ("TOPPADDING", (0, 0), (-1, -1), 6),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
#         ("LINEBELOW", (0, 0), (-1, -2), 0.5, BORDER_COLOR),
#         ("VALIGN", (0, 0), (-1, -1), "TOP"),
#     ]))
#     return table


# def _fitted_image(path: Path, max_width: float) -> RLImage:
#     """Scales an image to max_width while preserving its real aspect
#     ratio (read from the file itself, not assumed) -- avoids stretching
#     charts that use different figsize aspect ratios (see
#     plot_benchmark.py's STANDARD_FIGSIZE vs WIDE_FIGSIZE)."""
#     with PILImage.open(path) as img:
#         w, h = img.size
#     height = max_width * (h / w)
#     return RLImage(str(path), width=max_width, height=height)


# def generate_pdf_report(data: dict, output_path: Path, logo_path: "Path | None" = None) -> Path:
#     stats = compute_summary_stats(data)
#     insights = generate_insights(data, stats)
#     sysinfo = get_system_info()

#     styles = getSampleStyleSheet()
#     title_style = ParagraphStyle(
#         "TitleAccent", parent=styles["Title"], textColor=ACCENT_COLOR, spaceAfter=4,
#     )
#     heading_style = ParagraphStyle(
#         "HeadingAccent", parent=styles["Heading2"], textColor=ACCENT_COLOR,
#         spaceBefore=16, spaceAfter=8,
#     )
#     subheading_style = ParagraphStyle(
#         "SubheadingAccent", parent=styles["Heading3"], textColor=colors.HexColor("#374151"),
#         spaceBefore=10, spaceAfter=4,
#     )
#     body_style = styles["BodyText"]
#     bullet_style = ParagraphStyle("Bullet", parent=body_style, leftIndent=12, spaceAfter=6)
#     meta_style = ParagraphStyle("Meta", parent=body_style, textColor=colors.HexColor("#6b7280"))

#     story = []

#     if logo_path is not None and Path(logo_path).exists():
#         story.append(_fitted_image(Path(logo_path), max_width=0.9 * inch))
#         story.append(Spacer(1, 8))

#     story.append(Paragraph("RingSync Benchmark Report", title_style))
#     story.append(Paragraph(
#         f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_style,
#     ))
#     story.append(Spacer(1, 10))

#     # --- Benchmark Summary ---
#     story.append(Paragraph("Benchmark Summary", heading_style))
#     summary_rows = [
#         ["Model", "Small CNN"],
#         ["Dataset", "Synthetic CIFAR-like"],
#         [
#             "Workers Tested",
#             ", ".join(str(r["world_size"]) for r in stats["dist_results"]) if stats["dist_results"] else "N/A",
#         ],
#         [
#             "Best Speedup",
#             f"{stats['best_speedup']:.2f}\u00d7 at {stats['best_speedup_ws']} workers"
#             if stats["best_speedup"] else "N/A",
#         ],
#         [
#             "Scaling Efficiency at Max Workers",
#             f"{stats['efficiency_at_max_workers'] * 100:.1f}%" if stats["efficiency_at_max_workers"] else "N/A",
#         ],
#         [
#             "Max Communication Overhead",
#             f"{stats['max_overhead']:.1f}%" if stats["max_overhead"] is not None else "N/A",
#         ],
#         [
#             "Fastest Configuration",
#             f"{stats['fastest']['world_size']} workers ({stats['fastest']['wall_clock_seconds']:.2f}s)"
#             if stats["fastest"] else "N/A",
#         ],
#     ]
#     story.append(_kv_table(summary_rows))
#     story.append(Spacer(1, 10))

#     # --- Experiment Information ---
#     story.append(Paragraph("Experiment Information", heading_style))
#     exp_rows = [
#         ["Selected Workers", ", ".join(str(w) for w in data.get("world_sizes", [])) or "N/A"],
#         ["Baseline Mode", str(data.get("baseline_mode", "N/A"))],
#         ["Batch Size", str(data.get("batch_size", "N/A"))],
#         ["Epochs", str(data.get("epochs", "N/A"))],
#         ["Learning Rate", str(data.get("learning_rate", "N/A"))],
#     ]
#     story.append(_kv_table(exp_rows))
#     story.append(Spacer(1, 10))

#     # --- System Information ---
#     story.append(Paragraph("System Information", heading_style))
#     sys_rows = [
#         ["CPU", sysinfo["cpu"]],
#         ["Logical Cores", sysinfo["logical_cores"]],
#         ["Physical Cores", sysinfo["physical_cores"]],
#         ["RAM", sysinfo["ram_gb"]],
#         ["Python", sysinfo["python_version"]],
#         ["PyTorch", sysinfo["pytorch_version"]],
#         ["Operating System", sysinfo["os"]],
#     ]
#     story.append(_kv_table(sys_rows))
#     story.append(Spacer(1, 10))

#     # --- Key Insights ---
#     if insights:
#         story.append(Paragraph("Key Insights", heading_style))
#         for point in insights:
#             story.append(Paragraph(f"\u2022 {_md_bold_to_html(point)}", bullet_style))
#         story.append(Spacer(1, 10))

#     story.append(PageBreak())

#     # --- Graphs ---
#     story.append(Paragraph("Benchmark Graphs", heading_style))
#     for title, filename in CHART_FILES:
#         chart_path = RESULTS_DIR / filename
#         if chart_path.exists():
#             story.append(KeepTogether([
#                 Paragraph(title, subheading_style),
#                 _fitted_image(chart_path, max_width=6.3 * inch),
#                 Spacer(1, 14),
#             ]))

#     doc = SimpleDocTemplate(
#         str(output_path), pagesize=letter,
#         topMargin=0.75 * inch, bottomMargin=0.75 * inch,
#         leftMargin=0.75 * inch, rightMargin=0.75 * inch,
#     )
#     doc.build(story)
#     return output_path


# if __name__ == "__main__":
#     import json

#     results_json = RESULTS_DIR / "benchmark_results.json"
#     if not results_json.exists():
#         print(f"No benchmark results found at {results_json} -- run scripts/run_benchmark.py first.")
#     else:
#         with open(results_json) as f:
#             data = json.load(f)

#         logo_candidates = [
#             ROOT / "assets" / "logo_transparent.png",
#             ROOT / "assets" / "logo.png",
#         ]
#         logo = next((p for p in logo_candidates if p.exists()), None)

#         out_path = RESULTS_DIR / "ringsync_benchmark_report.pdf"
#         generate_pdf_report(data, out_path, logo_path=logo)
#         print(f"Saved -> {out_path}")


"""
PDF benchmark report exporter.

Builds a self-contained PDF containing everything needed to interpret
the benchmark results without also having the live dashboard open:
logo, timestamp, benchmark summary, experiment configuration, system
information (hardware context -- a speedup number means something
different on a 4-core laptop than a 32-core workstation), auto-
generated insights, and every chart.

Uses reportlab (pure Python, no external system binaries like
wkhtmltopdf needed -- important for a clean Windows setup) rather than
an HTML-to-PDF converter.

Summary stats and insights are computed via scripts/report_content.py,
the SAME module the Streamlit dashboard uses -- the PDF and the live
app will never disagree about what the "best speedup" was.
"""

from datetime import datetime
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
    PageBreak, KeepTogether,
)

from scripts.report_content import compute_summary_stats, generate_insights
from scripts.system_info import get_system_info

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def set_results_dir(path) -> None:
    """Same pattern as plot_benchmark.py's set_results_dir -- points
    chart lookups and the PDF's default save location at a specific
    experiment's folder."""
    global RESULTS_DIR
    RESULTS_DIR = Path(path)
ACCENT_COLOR = colors.HexColor("#7C3AED")
BORDER_COLOR = colors.HexColor("#e5e7eb")

# (display title, filename in results/) -- filenames match exactly what
# plot_benchmark.py already saves to disk, so no chart re-rendering is
# needed here, just reading the existing PNGs.
CHART_FILES = [
    ("Speedup vs. Default Multi-threaded Baseline", "speedup_vs_multithread_baseline.png"),
    ("Speedup vs. Single-threaded Baseline", "speedup_vs_singlethread_baseline.png"),
    ("Compute vs. Communication Time per Step", "compute_vs_comm_chart.png"),
    ("Communication Overhead vs. Number of Workers", "communication_overhead_pct.png"),
    ("Scaling Efficiency vs. Number of Workers", "scaling_efficiency_pct.png"),
    ("Training Loss vs. Epoch", "training_loss_vs_epoch.png"),
]


def _md_bold_to_html(text: str) -> str:
    """Converts markdown **bold** (used in generate_insights' output,
    written for Streamlit's st.markdown) into reportlab-compatible
    <b>bold</b> tags."""
    parts = text.split("**")
    html = ""
    for i, part in enumerate(parts):
        html += f"<b>{part}</b>" if i % 2 == 1 else part
    return html


def _kv_table(rows: "list[list[str]]") -> Table:
    table = Table(rows, colWidths=[2.3 * inch, 3.7 * inch])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), ACCENT_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def _fitted_image(path: Path, max_width: float) -> RLImage:
    """Scales an image to max_width while preserving its real aspect
    ratio (read from the file itself, not assumed) -- avoids stretching
    charts that use different figsize aspect ratios (see
    plot_benchmark.py's STANDARD_FIGSIZE vs WIDE_FIGSIZE)."""
    with PILImage.open(path) as img:
        w, h = img.size
    height = max_width * (h / w)
    return RLImage(str(path), width=max_width, height=height)


def generate_pdf_report(data: dict, output_path: Path, logo_path: "Path | None" = None) -> Path:
    stats = compute_summary_stats(data)
    insights = generate_insights(data, stats)
    sysinfo = get_system_info()

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleAccent", parent=styles["Title"], textColor=ACCENT_COLOR, spaceAfter=4,
    )
    heading_style = ParagraphStyle(
        "HeadingAccent", parent=styles["Heading2"], textColor=ACCENT_COLOR,
        spaceBefore=16, spaceAfter=8,
    )
    subheading_style = ParagraphStyle(
        "SubheadingAccent", parent=styles["Heading3"], textColor=colors.HexColor("#374151"),
        spaceBefore=10, spaceAfter=4,
    )
    body_style = styles["BodyText"]
    bullet_style = ParagraphStyle("Bullet", parent=body_style, leftIndent=12, spaceAfter=6)
    meta_style = ParagraphStyle("Meta", parent=body_style, textColor=colors.HexColor("#6b7280"))

    story = []

    if logo_path is not None and Path(logo_path).exists():
        story.append(_fitted_image(Path(logo_path), max_width=0.9 * inch))
        story.append(Spacer(1, 8))

    story.append(Paragraph("RingSync Benchmark Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_style,
    ))
    story.append(Spacer(1, 10))

    # --- Benchmark Summary ---
    story.append(Paragraph("Benchmark Summary", heading_style))
    summary_rows = [
        ["Model", "Small CNN"],
        ["Dataset", "Synthetic CIFAR-like"],
        [
            "Workers Tested",
            ", ".join(str(r["world_size"]) for r in stats["dist_results"]) if stats["dist_results"] else "N/A",
        ],
        [
            "Best Speedup",
            f"{stats['best_speedup']:.2f}\u00d7 at {stats['best_speedup_ws']} workers"
            if stats["best_speedup"] else "N/A",
        ],
        [
            "Scaling Efficiency at Max Workers",
            f"{stats['efficiency_at_max_workers'] * 100:.1f}%" if stats["efficiency_at_max_workers"] else "N/A",
        ],
        [
            "Max Communication Overhead",
            f"{stats['max_overhead']:.1f}%" if stats["max_overhead"] is not None else "N/A",
        ],
        [
            "Fastest Configuration",
            f"{stats['fastest']['world_size']} workers ({stats['fastest']['wall_clock_seconds']:.2f}s)"
            if stats["fastest"] else "N/A",
        ],
    ]
    story.append(_kv_table(summary_rows))
    story.append(Spacer(1, 10))

    # --- Experiment Information ---
    story.append(Paragraph("Experiment Information", heading_style))
    exp_rows = [
        ["Selected Workers", ", ".join(str(w) for w in data.get("world_sizes", [])) or "N/A"],
        ["Baseline Mode", str(data.get("baseline_mode", "N/A"))],
        ["Batch Size", str(data.get("batch_size", "N/A"))],
        ["Epochs", str(data.get("epochs", "N/A"))],
        ["Learning Rate", str(data.get("learning_rate", "N/A"))],
    ]
    story.append(_kv_table(exp_rows))
    story.append(Spacer(1, 10))

    # --- System Information ---
    story.append(Paragraph("System Information", heading_style))
    sys_rows = [
        ["CPU", sysinfo["cpu"]],
        ["Logical Cores", sysinfo["logical_cores"]],
        ["Physical Cores", sysinfo["physical_cores"]],
        ["RAM", sysinfo["ram_gb"]],
        ["Python", sysinfo["python_version"]],
        ["PyTorch", sysinfo["pytorch_version"]],
        ["Operating System", sysinfo["os"]],
    ]
    story.append(_kv_table(sys_rows))
    story.append(Spacer(1, 10))

    # --- Key Insights ---
    if insights:
        story.append(Paragraph("Key Insights", heading_style))
        for point in insights:
            story.append(Paragraph(f"\u2022 {_md_bold_to_html(point)}", bullet_style))
        story.append(Spacer(1, 10))

    story.append(PageBreak())

    # --- Graphs ---
    story.append(Paragraph("Benchmark Graphs", heading_style))
    for title, filename in CHART_FILES:
        chart_path = RESULTS_DIR / filename
        if chart_path.exists():
            story.append(KeepTogether([
                Paragraph(title, subheading_style),
                _fitted_image(chart_path, max_width=6.3 * inch),
                Spacer(1, 14),
            ]))

    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    doc.build(story)
    return output_path


if __name__ == "__main__":
    import json

    results_json = RESULTS_DIR / "benchmark_results.json"
    if not results_json.exists():
        print(f"No benchmark results found at {results_json} -- run scripts/run_benchmark.py first.")
    else:
        with open(results_json) as f:
            data = json.load(f)

        logo_candidates = [
            ROOT / "assets" / "logo_transparent.png",
            ROOT / "assets" / "logo.png",
        ]
        logo = next((p for p in logo_candidates if p.exists()), None)

        out_path = RESULTS_DIR / "ringsync_benchmark_report.pdf"
        generate_pdf_report(data, out_path, logo_path=logo)
        print(f"Saved -> {out_path}")