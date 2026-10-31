"""Manuscript figures for the current BVS specification.

Draws the five figures of the paper from pipeline outputs, so that each
figure describes the configuration that produced the reported results:
observed variables from national data alone (Martins et al., 2024) and
external evidence only for the outcome the national survey does not measure.

Inputs (paths relative to --base):
    dag_edges_3391.csv          learned structure (step1.py)
    target_evidence.csv         retained evidence per target outcome (evidence_table.py)
    transfer_output/profiles_centered.csv   transfer-model projections (transfer_model.py)
    metadata/3391_num.csv       CIS microdata, used for the example query

Outputs (in --out), each as vector PDF and 600 dpi PNG at printed size
(6.5 in wide; include with width=\textwidth):
    fig_pipeline_steps, fig_dag, fig_step2_pooling, step4, fig_gradio

Usage:
    python make_figures.py --base . --out figures
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FONT = ["Inter", "DejaVu Sans"]  # Inter if installed, otherwise DejaVu Sans

# Two semantic colours: national data and external evidence. Everything else is neutral.
NATIONAL = "#2F5D8A"
NATIONAL_FILL = "#E6EEF6"
EXTERNAL = "#A8641E"
EXTERNAL_FILL = "#F6ECE0"
NEUTRAL = "#4A4A4A"
NEUTRAL_FILL = "#F2F2F2"
MUTED = "#8C8C8C"

plt.rcParams.update({"font.family": FONT, "font.size": 7.5, "mathtext.default": "regular"})

BLOCK_LABELS = {
    "demographics": "Demographics",
    "employment": "Employment",
    "origin_family": "Family origin",
    "satisfaction": "Life satisfaction",
    "env_concern_a": "Env. concern A",
    "env_concern_b": "Env. concern B",
    "wtp_acceptance": "Willingness to pay",
    "behavior": "Behaviour",
    "ecological_footprint": "Ecological footprint",
    "danger_pollution": "Perceived danger",
    "science_tech": "Science and tech.",
    "trust_institutions": "Institutional trust",
    "political_issues": "Political issues",
    "political_values": "Political values",
    "political_ideology": "Political ideology",
}

TRANSFER_PREDICTORS = {
    "SEX": "female",
    "BIRTH": "under 30",
    "ESTUDIOS": "higher education",
    "V52": "recycles",
    "V15": "environmental worry",
}

TARGET_LABELS = {
    "acceptance_general": "General acceptance",
    "acceptance_food_packaging": "Food-packaging acceptance",
    "perception_safety": "Perceived safety",
    "perception_quality": "Perceived quality",
    "price_willingness_to_pay_premium": "WTP premium",
}

STUDY_LABELS = {
    "cao_lu_zhu_2022_eol_recycled_preferences": "Cao et al. 2022",
    "widiastuti_2025": "Widiastuti et al. 2025",
    "ruokamo_2022_recycled_plastics": "Ruokamo et al. 2022",
    "leal_filho_2025": "Leal Filho et al. 2021",
}


# ---------------------------------------------------------------------------
# Drawing primitives
#
# Every figure is drawn at its printed size (TEXT_WIDTH inches wide, one data
# unit = one inch), so font sizes below are the sizes on the page.
# ---------------------------------------------------------------------------

TEXT_WIDTH = 6.5
TITLE_PT = 8.5
BODY_PT = 7.5
SMALL_PT = 7.0
LINE = BODY_PT * 1.3 / 72  # height of one body line, in inches


def box(ax, x, y, w, h, title, body="", edge=NEUTRAL, fill=NEUTRAL_FILL, title_pt=TITLE_PT, body_pt=BODY_PT):
    """Rounded box anchored at its lower-left corner; title and body centred as one block."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0.9, edgecolor=edge, facecolor=fill))
    if not body:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_pt, fontweight="bold", color=edge)
        return
    n_lines = body.count("\n") + 1
    block = title_pt * 1.35 / 72 + n_lines * body_pt * 1.3 / 72
    top = y + h / 2 + block / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=title_pt, fontweight="bold", color=edge)
    ax.text(x + w / 2, top - title_pt * 1.45 / 72, body, ha="center", va="top",
            fontsize=body_pt, color=NEUTRAL, linespacing=1.3)


def arrow(ax, start, end, color=NEUTRAL, dashed=False, label="", width=0.9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=width,
                                 color=color, linestyle=(0, (3, 2)) if dashed else "solid",
                                 shrinkA=1.5, shrinkB=1.5))
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.04, label, ha="center", va="bottom", fontsize=SMALL_PT, color=color, style="italic")


def canvas(height, width=TEXT_WIDTH):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def save(fig, path: Path):
    """Save at printed size as vector PDF and as 600 dpi PNG."""
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=600, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Icons (drawn with patches so the figures need no external images)
# ---------------------------------------------------------------------------

from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle  # noqa: E402


def icon_database(ax, cx, cy, size, color):
    w, h = size * 0.75, size * 0.85
    for k in range(3):
        y = cy - h / 2 + k * h / 3
        ax.add_patch(Rectangle((cx - w / 2, y), w, h / 3 - 0.012, facecolor=color, edgecolor="none"))
        ax.add_patch(Ellipse((cx, y + h / 3 - 0.012), w, h * 0.18, facecolor=color, edgecolor="white", linewidth=0.8))
    ax.add_patch(Ellipse((cx, cy - h / 2), w, h * 0.18, facecolor=color, edgecolor="none"))


def icon_globe(ax, cx, cy, size, color):
    r = size * 0.42
    kw = dict(facecolor="none", edgecolor=color, linewidth=1.1)
    ax.add_patch(Circle((cx, cy), r, **kw))
    ax.add_patch(Ellipse((cx, cy), r, 2 * r, **kw))
    for dy in (-0.45 * r, 0, 0.45 * r):
        half = np.sqrt(r**2 - dy**2)
        ax.plot([cx - half, cx + half], [cy + dy, cy + dy], color=color, linewidth=1.0)


def icon_table(ax, cx, cy, size, color, rows=4, cols=4):
    w, h = size * 0.85, size * 0.62
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white", edgecolor=color, linewidth=1.1))
    ax.add_patch(Rectangle((x0, y0 + h * (rows - 1) / rows), w, h / rows, facecolor=color, alpha=0.35, edgecolor="none"))
    for i in range(1, rows):
        ax.plot([x0, x0 + w], [y0 + i * h / rows] * 2, color=color, linewidth=0.7)
    for j in range(1, cols):
        ax.plot([x0 + j * w / cols] * 2, [y0, y0 + h], color=color, linewidth=0.7)


def icon_graph(ax, cx, cy, size, color):
    pts = np.array([(-0.42, 0.05), (-0.2, 0.32), (0.05, 0.12), (0.3, 0.3), (0.4, -0.05),
                    (0.1, -0.3), (-0.22, -0.25)]) * size + (cx, cy)
    for a, b in ((0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 0), (2, 5)):
        ax.add_patch(FancyArrowPatch(pts[a], pts[b], arrowstyle="-|>", mutation_scale=5, color=NEUTRAL,
                                     linewidth=0.6, shrinkA=3, shrinkB=3))
    for x, y in pts:
        ax.add_patch(Circle((x, y), size * 0.055, facecolor=color, edgecolor=NEUTRAL, linewidth=0.5, alpha=0.85))


def icon_documents(ax, cx, cy, size, color):
    w, h = size * 0.5, size * 0.62
    for k, dx in enumerate((0.1, 0.0)):
        x0, y0 = cx - w / 2 + dx * size, cy - h / 2 + (0.08 - dx) * size
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white", edgecolor=color, linewidth=1.0, zorder=3 + k))
        if k == 1:
            for i in range(4):
                ax.plot([x0 + 0.15 * w, x0 + 0.85 * w], [y0 + h * (0.75 - 0.17 * i)] * 2,
                        color=color, linewidth=0.8, zorder=5)


def icon_bars(ax, cx, cy, size, color):
    for i, hgt in enumerate((0.35, 0.22, 0.5)):
        ax.add_patch(Rectangle((cx - 0.3 * size + i * 0.22 * size, cy - 0.25 * size), 0.15 * size, hgt * size,
                               facecolor=color, edgecolor="none"))
    ax.plot([cx - 0.36 * size, cx + 0.38 * size], [cy - 0.25 * size] * 2, color=color, linewidth=0.8)


def icon_gear(ax, cx, cy, r, color, teeth=8):
    angles = np.linspace(0, 2 * np.pi, teeth * 4, endpoint=False)
    radii = np.tile([r, r, r * 0.78, r * 0.78], teeth)
    ax.add_patch(Polygon(np.c_[cx + radii * np.cos(angles), cy + radii * np.sin(angles)],
                         facecolor=color, edgecolor="none"))
    ax.add_patch(Circle((cx, cy), r * 0.35, facecolor="white", edgecolor="none"))


def icon_people(ax, cx, cy, size, color):
    for dx, dy, s in ((-0.32, 0.12, 0.8), (0.32, 0.12, 0.8), (-0.12, 0.2, 0.85), (0.14, 0.2, 0.85),
                      (-0.2, -0.18, 0.95), (0.2, -0.18, 0.95)):
        x, y = cx + dx * size, cy + dy * size
        ax.add_patch(Circle((x, y + 0.11 * size * s), 0.075 * size * s, facecolor=color, edgecolor="white", linewidth=0.6))
        ax.add_patch(FancyBboxPatch((x - 0.11 * size * s, y - 0.1 * size * s), 0.22 * size * s, 0.14 * size * s,
                                    boxstyle="round,pad=0,rounding_size=0.03", facecolor=color,
                                    edgecolor="white", linewidth=0.6))


def icon_monitor(ax, cx, cy, size, color):
    w, h = size * 1.05, size * 0.62
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2 + 0.08 * size), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor="white", edgecolor=color, linewidth=1.3))
    ax.add_patch(Rectangle((cx - 0.04 * size, cy - h / 2 - 0.05 * size), 0.08 * size, 0.13 * size, facecolor=color))
    ax.add_patch(Rectangle((cx - 0.2 * size, cy - h / 2 - 0.08 * size), 0.4 * size, 0.04 * size, facecolor=color))
    ax.add_patch(FancyBboxPatch((cx - 0.42 * size, cy + 0.02 * size), 0.34 * size, 0.2 * size,
                                boxstyle="round,pad=0,rounding_size=0.03", facecolor="none", edgecolor=color, linewidth=1.0))
    for i in range(3):
        ax.add_patch(Circle((cx - 0.33 * size + i * 0.08 * size, cy + 0.12 * size), 0.017 * size, facecolor=color))
    icon_bars(ax, cx + 0.22 * size, cy + 0.1 * size, 0.45 * size, color)


def icon_document(ax, cx, cy, size, color):
    w, h = size * 0.48, size * 0.62
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h * 0.78), (x0 + w * 0.72, y0 + h),
                          (x0, y0 + h)], facecolor="white", edgecolor=color, linewidth=1.0))
    for i in range(4):
        ax.plot([x0 + 0.18 * w, x0 + 0.82 * w], [y0 + h * (0.62 - 0.14 * i)] * 2, color=color, linewidth=0.8)


# ---------------------------------------------------------------------------
# Figure: pipeline (one row per step: input -> step panel -> output)
# ---------------------------------------------------------------------------

STEP_COLOURS = (
    ("#1F4E79", "#EEF3F9"),
    ("#2E7D7A", "#ECF5F4"),
    ("#4F7F3A", "#F1F6EC"),
    ("#D9772B", "#FDF2E8"),
    ("#5E4B8B", "#F2EFF8"),
)


def side_box(ax, x, y, w, h, color, fill, icon, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=fill, edgecolor=color, linewidth=0.8))
    icon(ax, x + w / 2, y + h - 0.27, 0.44, color)
    ax.text(x + w / 2, y + h - 0.5, lines, ha="center", va="top", fontsize=SMALL_PT - 0.5,
            color="#222222", linespacing=1.3)


def figure_pipeline(path: Path, n_graphs: int, n_per_graph: int):
    steps = [
        ("Structure learning", "(posterior over dependency graphs)",
         ["MC$^3$ sampling (parallel tempering)", "15 thematic blocks", "four-layer cross-block screening",
          "forbidden edges (variable ordering)"],
         icon_graph, "",
         (icon_database, "National data\n(CIS 3391)\n$n$ = 2,254, $d$ = 90"),
         (icon_graph, "Posterior ensemble\nof DAGs\n$p(G \\mid X)$")),
        ("External evidence", "(only for the unmeasured outcome)",
         ["11 studies, 80 entries extracted", "language model + human review",
          "review rules (15 corrections)", "transfer model for food packaging"],
         lambda ax, x, y, s, c: (icon_documents(ax, x - 0.12, y + 0.05, s, c), icon_bars(ax, x + 0.22, y - 0.05, s * 0.7, c)),
         "Reviewed evidence",
         (icon_globe, "External studies\n11 studies\n~3,200 respondents"),
         (icon_table, "Transfer model\n$P(Y \\mid x)$ +\nevidence tables")),
        ("Parameter estimation", "(national data only)",
         ["BDeu prior (ESS 10)", "PESOFIN-weighted counts", "no external evidence in", "observed variables"],
         lambda ax, x, y, s, c: (icon_gear(ax, x - 0.12, y + 0.1, 0.13, c), icon_gear(ax, x + 0.08, y + 0.17, 0.08, c),
                                 icon_table(ax, x + 0.02, y - 0.17, s * 0.75, c)),
         "CPTs",
         (icon_table, "Counts from\nnational data\n+ DAGs (Step 1)"),
         (icon_table, "Probability tables\n(CPTs)\n$p(\\Theta \\mid X)$")),
        ("Posterior predictive generation", "(synthetic population)",
         [f"ancestral sampling over {n_graphs} graphs", f"{n_per_graph} individuals per graph",
          "acceptance attached by the", "transfer model (Step 2)"],
         icon_people, f"$n$ = {n_graphs * n_per_graph:,}",
         (icon_table, "CPTs (Step 3)\n+ DAGs (Step 1)\n+ transfer model"),
         (icon_database, "Synthetic dataset\n90 observed variables\n+ 1 transferred outcome")),
        ("Validation and decision support", "(evaluation and use)",
         ["marginal and dependency fidelity", "external comparison (Ipsos)", "transport error (ISSP 2020)",
          "conversational interface"],
         icon_monitor, "",
         (icon_database, "Synthetic dataset\n+ CIS microdata\n+ Ipsos, ISSP"),
         (icon_document, "Projections with\nintervals +\nmeasurement plan")),
    ]
    row_h, gap, legend_h = 1.42, 0.2, 0.78
    height = len(steps) * row_h + (len(steps) - 1) * gap + 0.2 + legend_h
    fig, ax = canvas(height)
    in_x, in_w, main_x, main_w, out_x, out_w = 0.02, 1.08, 1.32, 3.75, 5.28, 1.2

    for k, (title, subtitle, bullets, icon, icon_label, (in_icon, in_text), (out_icon, out_text)) in enumerate(steps):
        color, fill = STEP_COLOURS[k]
        y = height - 0.03 - (k + 1) * row_h - k * gap
        ax.add_patch(FancyBboxPatch((main_x, y), main_w, row_h, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=fill, edgecolor=color, linewidth=1.0))
        ax.add_patch(Circle((main_x + 0.27, y + row_h - 0.25), 0.16, facecolor=color, edgecolor="none"))
        ax.text(main_x + 0.27, y + row_h - 0.255, str(k + 1), ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", family="serif")
        ax.text(main_x + 0.52, y + row_h - 0.17, title, ha="left", va="center", fontsize=10,
                fontweight="bold", color="#222222")
        ax.text(main_x + 0.52, y + row_h - 0.35, subtitle, ha="left", va="center", fontsize=SMALL_PT,
                color="#333333")
        for i, b in enumerate(bullets):
            ax.text(main_x + 0.3, y + row_h - 0.62 - i * 0.18, "•  " + b, ha="left", va="center",
                    fontsize=BODY_PT, color="#222222")
        icon(ax, main_x + main_w - 0.62, y + row_h / 2 + 0.05, 0.62 if icon is not icon_people else 0.85, color)
        if icon_label:
            ax.text(main_x + main_w - 0.62, y + 0.17, icon_label, ha="center", va="center",
                    fontsize=SMALL_PT, color="#222222")

        side_box(ax, in_x, y + 0.1, in_w, row_h - 0.2, color, fill, in_icon, in_text)
        side_box(ax, out_x, y + 0.1, out_w, row_h - 0.2, color, "white", out_icon, out_text)
        arrow(ax, (in_x + in_w, y + row_h / 2), (main_x, y + row_h / 2), "#222222", dashed=True, width=0.8)
        arrow(ax, (main_x + main_w, y + row_h / 2), (out_x, y + row_h / 2), "#222222", width=0.8)
        if k < len(steps) - 1:
            arrow(ax, (main_x + main_w / 2, y), (main_x + main_w / 2, y - gap), "#222222", width=0.8)

    ax.add_patch(FancyBboxPatch((0.02, 0.02), 6.46, legend_h - 0.08, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor="white", edgecolor="#BBBBBB", linewidth=0.6))
    ax.text(0.15, legend_h - 0.17, "Notation", fontsize=SMALL_PT, fontweight="bold", va="center")
    col1 = ("CIS 3391: ISSP Environment IV, Spain 2023", "DAG: directed acyclic graph",
            "CPT: conditional probability table")
    col2 = ("MC$^3$: Markov chain Monte Carlo over models", "$n$: sample size; $d$: number of variables",
            "$p(\\cdot)$: probability / posterior distribution")
    for i, t in enumerate(col1):
        ax.text(0.15, legend_h - 0.35 - i * 0.15, t, fontsize=SMALL_PT - 0.5, va="center")
    for i, t in enumerate(col2):
        ax.text(2.85, legend_h - 0.35 - i * 0.15, t, fontsize=SMALL_PT - 0.5, va="center")
    arrow(ax, (5.3, legend_h - 0.32), (5.65, legend_h - 0.32), "#222222", dashed=True, width=0.8)
    ax.text(5.72, legend_h - 0.32, "Input", fontsize=SMALL_PT - 0.5, va="center")
    arrow(ax, (5.3, legend_h - 0.52), (5.65, legend_h - 0.52), "#222222", width=0.8)
    ax.text(5.72, legend_h - 0.52, "Output", fontsize=SMALL_PT - 0.5, va="center")
    save(fig, path)


# ---------------------------------------------------------------------------
# Figure: core dependencies and transfer-model links
# ---------------------------------------------------------------------------

DAG_COLOURS = {"demographics": "#333D5A", "attitudes": "#59695D", "behaviour": "#5A4486", "target": "#A6703F"}
LEARNED, TRANSFER = "#24395C", "#A6703F"


def figure_dag(path: Path, dependencies: pd.DataFrame, coefficients: pd.DataFrame):
    v = {r.pair.replace(" ", ""): r.V_CIS for r in dependencies.itertuples()}
    beta = coefficients.set_index("predictor")["coefficient"]
    fig, ax = canvas(4.6)
    nodes = {
        "demo": (3.25, 3.95, "Demographics", "sex, age, education\nSEX  BIRTH  ESTUDIOS", "demographics"),
        "env": (1.15, 2.75, "Env. concern", "V15", "attitudes"),
        "wtp": (5.35, 2.75, "WTP prices", "V26", "attitudes"),
        "rec": (1.6, 1.3, "Recycling", "V52", "behaviour"),
        "food": (4.75, 1.3, "Food-packaging acceptance", "not in CIS: transfer model", "target"),
    }
    w, h = 1.75, 0.62

    def edge(a, b, label, colour, dashed=False, offset=(0.0, 0.0)):
        (xa, ya), (xb, yb) = nodes[a][:2], nodes[b][:2]
        ax.plot([xa, xb], [ya, yb], color=colour, linewidth=1.4, linestyle=(0, (4, 3)) if dashed else "solid",
                zorder=1)
        mx, my = (xa + xb) / 2 + offset[0], (ya + yb) / 2 + offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=SMALL_PT, fontweight="bold", color=colour,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"), zorder=2)

    edge("demo", "env", f"V = {v['education->env_concern']:.2f}", LEARNED)
    edge("demo", "wtp", f"V = {v['education->WTP']:.2f}", LEARNED)
    edge("env", "wtp", f"V = {v['env_concern->WTP']:.2f}", LEARNED, offset=(-0.75, 0))
    edge("env", "rec", f"V = {v['env_concern->recycling']:.2f}", LEARNED)
    edge("demo", "food", f"β = {beta['female']:.2f} / {beta['under_30']:.2f} / {beta['high_education']:.2f}",
         TRANSFER, dashed=True, offset=(0.75, -0.25))
    edge("env", "food", f"β = {beta['env_worry']:.2f}", TRANSFER, dashed=True, offset=(0.35, 0.15))
    edge("rec", "food", f"β = {beta['recycles']:.2f}", TRANSFER, dashed=True)

    for key, (x, y, title, sub, kind) in nodes.items():
        width = 2.3 if key == "food" else w
        ax.add_patch(FancyBboxPatch((x - width / 2, y - h / 2), width, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=DAG_COLOURS[kind], edgecolor="none", zorder=3))
        ax.text(x, y + 0.11, title, ha="center", va="center", fontsize=TITLE_PT + 0.5, fontweight="bold",
                color="white", zorder=4)
        ax.text(x, y - 0.13, sub, ha="center", va="center", fontsize=SMALL_PT - 0.5, color="#F0F0F0",
                linespacing=1.2, zorder=4)

    lx, ly = 0.05, 0.05
    ax.add_patch(FancyBboxPatch((lx, ly), 2.45, 0.58, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor="white", edgecolor="#BBBBBB", linewidth=0.6))
    for i, (kind, label) in enumerate((("demographics", "Demographics"), ("attitudes", "Attitudes"),
                                       ("behaviour", "Behaviour"), ("target", "Target (transferred)"))):
        cx, cy = lx + 0.1 + (i % 2) * 1.15, ly + 0.43 - (i // 2) * 0.17
        ax.add_patch(Rectangle((cx, cy - 0.04), 0.14, 0.08, facecolor=DAG_COLOURS[kind]))
        ax.text(cx + 0.19, cy, label, fontsize=SMALL_PT - 0.5, va="center")
    ax.plot([lx + 0.1, lx + 0.3], [ly + 0.09] * 2, color=LEARNED, linewidth=1.4)
    ax.text(lx + 0.35, ly + 0.09, "CIS (Cramér's V)", fontsize=SMALL_PT - 0.5, va="center")
    ax.plot([lx + 1.25, lx + 1.45], [ly + 0.09] * 2, color=TRANSFER, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.text(lx + 1.5, ly + 0.09, "transfer (β)", fontsize=SMALL_PT - 0.5, va="center")
    save(fig, path)


# ---------------------------------------------------------------------------
# Figure: external evidence
# ---------------------------------------------------------------------------

def figure_evidence(path: Path, evidence: pd.DataFrame, studies: pd.DataFrame | None = None):
    teal, red, purple, grey = "#1F665F", "#D45A50", "#5F3D8C", "#6E6E6E"
    sources = [("Cao et al. 2022", "China, n = 926"), ("Widiastuti et al. 2025", "Indonesia, n = 116"),
               ("Ruokamo et al. 2022", "Finland, n = 301"), ("Leal Filho et al. 2021", "Europe, n = 127")]
    fig, ax = canvas(3.1)
    ys = np.linspace(2.55, 0.75, len(sources))
    for (name, detail), y in zip(sources, ys):
        ax.add_patch(FancyBboxPatch((0.05, y - 0.22), 1.75, 0.44, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=teal, edgecolor="none"))
        ax.text(0.925, y + 0.07, name, ha="center", va="center", fontsize=BODY_PT + 0.5, color="white")
        ax.text(0.925, y - 0.1, detail, ha="center", va="center", fontsize=SMALL_PT - 0.5, color="#D8EAE7")
        ax.plot([1.8, 2.05], [y, y], color="#8A8A8A", linewidth=0.9)
    ax.plot([2.05, 2.05], [ys[-1], ys[0]], color="#8A8A8A", linewidth=0.9)
    arrow(ax, (2.05, 1.65), (2.35, 1.65), "#8A8A8A")

    ax.add_patch(FancyBboxPatch((2.35, 1.0), 1.6, 1.3, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=red, edgecolor="none"))
    ax.text(3.15, 1.92, "Review rules +\nn-weighted pooling", ha="center", va="center", fontsize=TITLE_PT + 1,
            fontweight="bold", color="white", linespacing=1.15)
    ax.text(3.15, 1.33, "15 corrections\n$I^2$ reported ($\\lambda$ = 0)", ha="center", va="center",
            fontsize=SMALL_PT, color="#FBE3E0", linespacing=1.3)

    ax.add_patch(FancyBboxPatch((4.3, 1.75), 2.15, 0.95, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=purple, edgecolor="none"))
    ax.text(5.375, 2.4, "Transfer model", ha="center", va="center", fontsize=TITLE_PT + 1.5,
            fontweight="bold", color="white")
    ax.text(5.375, 2.02, "food-packaging acceptance\n$p_0$ = 0.38, 5 coefficients", ha="center", va="center",
            fontsize=SMALL_PT, color="#E4DAF2", linespacing=1.3)
    ax.add_patch(FancyBboxPatch((4.3, 0.6), 2.15, 0.95, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor="#ECE7F4", edgecolor=purple, linewidth=0.9))
    ax.text(5.375, 1.25, "Evidence tables", ha="center", va="center", fontsize=TITLE_PT + 1,
            fontweight="bold", color=purple)
    ax.text(5.375, 0.88, "general acceptance, safety,\nWTP premium: not projected", ha="center", va="center",
            fontsize=SMALL_PT, color="#333333", linespacing=1.3)
    arrow(ax, (3.95, 1.9), (4.3, 2.2), "#8A8A8A")
    arrow(ax, (3.95, 1.4), (4.3, 1.08), "#8A8A8A")
    ax.text(0.05, 0.22, "Retained evidence for the target outcomes. The other 7 of the 11 extracted studies supply "
            "no retained entry; Ipsos (2022)\nis used only for external comparison. Under $\\lambda$ = 0 the "
            "effective sample size equals the pooled n.", fontsize=SMALL_PT - 0.5, color=grey, va="center",
            linespacing=1.3)
    save(fig, path)


# ---------------------------------------------------------------------------
# Figure: generation of the synthetic population
# ---------------------------------------------------------------------------

def mini_graph(ax, x0, y0, w, h, pts):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor="white", edgecolor="#C8CDD3", linewidth=0.8))
    pts = [(x0 + px * w, y0 + py * h) for px, py in pts]
    for a, b in zip(pts, pts[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#5E9A8C", linewidth=1.2)
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.04, facecolor="#5E9A8C", edgecolor="none"))


def figure_generation(path: Path, n_graphs: int, n_per_graph: int):
    dark, green, brown = "#2B3644", "#4E8C7D", "#A6703F"
    fig, ax = canvas(2.15)
    ax.add_patch(FancyBboxPatch((0.02, 0.25), 1.05, 1.85, boxstyle="round,pad=0,rounding_size=0.1",
                                facecolor="#F2F3F5", edgecolor="none"))
    ax.text(0.545, 1.88, f"{n_graphs} networks\nfrom the posterior", ha="center", va="center",
            fontsize=SMALL_PT, color="#444444", linespacing=1.2)
    mini_graph(ax, 0.12, 1.12, 0.85, 0.45, [(0.18, 0.7), (0.38, 0.3), (0.6, 0.7), (0.82, 0.3)])
    ax.text(0.545, 0.98, "…", ha="center", va="center", fontsize=9, color="#888888")
    mini_graph(ax, 0.12, 0.38, 0.85, 0.45, [(0.18, 0.7), (0.38, 0.3), (0.6, 0.7), (0.82, 0.7)])
    arrow(ax, (1.12, 1.17), (1.32, 1.17), "#9AA3AD")

    labels = ("Sex · Birth year", "Education · Income", "Env. concern · Trust", "Recycling · WTP")
    for i, lab in enumerate(labels):
        y = 1.72 - i * 0.44
        ax.add_patch(FancyBboxPatch((1.35, y), 1.45, 0.3, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=dark, edgecolor="none"))
        ax.text(1.43, y + 0.15, lab, ha="left", va="center", fontsize=BODY_PT + 0.5, fontweight="bold", color="white")
        if i < 3:
            arrow(ax, (2.075, y), (2.075, y - 0.14), green, width=0.8)
    ax.text(2.075, 0.2, f"× {n_per_graph} → {n_graphs * n_per_graph:,} people", ha="center", va="center",
            fontsize=BODY_PT, fontweight="bold", color=green)
    ax.text(3.0, 1.22, "then", ha="center", va="bottom", fontsize=SMALL_PT - 0.5, color="#888888")
    arrow(ax, (2.85, 1.17), (3.15, 1.17), "#9AA3AD")

    x0 = 3.2
    ax.add_patch(FancyBboxPatch((x0, 0.25), 3.27, 1.85, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor="#F7F8F9", edgecolor="none"))
    cols = [("SEX", 0.3), ("EDU", 0.45), ("V15", 0.35), ("V52", 0.35), ("…", 0.3), ("Food pkg.", 0.75)]
    rows = [("F", "Univ", "5", "1", "…", "accepts"), ("M", "None", "1", "4", "…", "rejects"),
            ("F", "Sec", "3", "2", "…", "rejects")]
    ax.add_patch(FancyBboxPatch((x0 + 0.08, 1.7), 3.1, 0.3, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor="white", edgecolor="#D5D9DE", linewidth=0.6))
    cx = x0 + 0.15
    positions = []
    for name, width in cols:
        positions.append(cx + width / 2)
        target = name.startswith("Food")
        ax.text(cx + width / 2, 1.85, name, ha="center", va="center", fontsize=BODY_PT, fontweight="bold",
                color=brown if target else green)
        cx += width + 0.05
    for r, row in enumerate(rows):
        y = 1.48 - r * 0.28
        if r != 1:
            ax.add_patch(Rectangle((x0 + 0.08, y - 0.12), 3.1, 0.24, facecolor="#EEF6F1", edgecolor="none"))
        for (name, _), xc, val in zip(cols, positions, row):
            target = name.startswith("Food")
            ax.text(xc, y, val, ha="center", va="center", fontsize=BODY_PT, fontweight="bold" if target else "normal",
                    color=brown if target else "#333333")
    ax.text(x0 + 1.64, 0.68, f"···  {n_graphs * n_per_graph:,} rows  ···", ha="center",
            va="center", fontsize=BODY_PT, color="#888888")
    for (label, colour, fill, xl, wl) in (("90 CIS variables", green, "#E3F0EA", x0 + 0.1, 1.5),
                                          ("1 transferred outcome", brown, "#F6E9DD", x0 + 1.68, 1.5)):
        ax.add_patch(FancyBboxPatch((xl, 0.33), wl, 0.22, boxstyle="round,pad=0,rounding_size=0.1",
                                    facecolor=fill, edgecolor="none"))
        ax.text(xl + wl / 2, 0.44, label, ha="center", va="center", fontsize=BODY_PT, fontweight="bold", color=colour)
    save(fig, path)


# ---------------------------------------------------------------------------
# Figure: interface result screen, with the current model's output
# ---------------------------------------------------------------------------

def figure_interface(path: Path, profiles: pd.DataFrame):
    row = profiles.set_index("quantity").loc["most_receptive"]
    est, low, high = row.estimate, row.ci_low, row.ci_high
    green, blue, orange = "#6DB96D", "#4A7FD6", "#E9A649"
    fig, ax = canvas(3.2)
    ax.add_patch(Rectangle((0.0, 2.72), 6.5, 0.45, facecolor="#7DBF72", edgecolor="none"))
    ax.text(0.2, 2.945, "ASAP – Recycled Plastic Virtual Survey", ha="left", va="center", fontsize=11,
            fontweight="bold", color="#3E2C80")
    ax.add_patch(Rectangle((0.02, 0.95), 6.46, 1.7, facecolor="white", edgecolor=orange, linewidth=1.0))
    narrative = ("A woman under 30 with a university degree, who\nalways recycles and is extremely worried about\n"
                 f"the environment, has a projected {est:.0%} probability of\njudging recycled plastic suitable "
                 f"for food packaging\n(95% interval: {low:.0%}–{high:.0%}). The projection rests on\n"
                 "transferred Finnish evidence and has not been\nvalidated in Spain.")
    ax.add_patch(Rectangle((0.12, 1.05), 2.95, 1.5, facecolor="#F7F8FA", edgecolor="none"))
    ax.text(0.2, 2.47, narrative, ha="left", va="top", fontsize=BODY_PT, color="#333333", linespacing=1.35)

    ax.text(3.35, 2.52, "RESULT", fontsize=SMALL_PT - 1.5, color="#999999", va="center")
    ax.add_patch(FancyBboxPatch((3.35, 1.58), 3.0, 0.82, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor="#F2F9F1", edgecolor="#CFE6CB", linewidth=0.8))
    ax.text(3.45, 2.28, "PROBABILITY OF ACCEPTANCE", fontsize=SMALL_PT - 1.5, color="#888888", va="center")
    ax.add_patch(FancyBboxPatch((4.95, 2.21), 1.0, 0.14, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor="#E3E9FA", edgecolor="none"))
    ax.text(5.45, 2.28, "transferred evidence", fontsize=SMALL_PT - 1.5, color="#3E57A8", ha="center", va="center")
    ax.text(3.45, 1.97, f"{est:.0%}", fontsize=22, color="#222222", va="center")
    ax.text(3.45, 1.74, f"95% interval: {low:.1%} – {high:.1%}", fontsize=SMALL_PT - 1, color="#777777",
            va="center")
    ax.add_patch(Rectangle((3.45, 1.64), 2.8, 0.04, facecolor="#E5E7EA", edgecolor="none"))
    ax.add_patch(Rectangle((3.45, 1.64), 2.8 * est, 0.04, facecolor=green, edgecolor="none"))
    for xb, head, text in ((3.35, "PROFILE", "Woman, under 30, university,\nalways recycles, very worried"),
                           (4.9, "OUTCOME", "Food packaging\nnot validated in Spain")):
        ax.add_patch(FancyBboxPatch((xb, 1.05), 1.45, 0.45, boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor="white", edgecolor="#E1E3E6", linewidth=0.8))
        ax.text(xb + 0.07, 1.4, head, fontsize=SMALL_PT - 2, color="#999999", va="center")
        ax.text(xb + 0.07, 1.2, text, fontsize=SMALL_PT - 1.5, color="#333333", va="center", linespacing=1.2)

    ax.add_patch(Rectangle((0.02, 0.08), 4.1, 0.68, facecolor="white", edgecolor=blue, linewidth=1.0))
    ax.add_patch(FancyBboxPatch((0.12, 0.18), 2.85, 0.48, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor="#F3F4F6", edgecolor="#D9DCE0", linewidth=0.6))
    ax.text(0.2, 0.42, "Would a woman under 30 with a university degree, who\nalways recycles, accept recycled "
            "plastic in food packaging?", fontsize=SMALL_PT - 0.5, color="#333333", va="center", linespacing=1.25)
    ax.add_patch(FancyBboxPatch((3.07, 0.24), 0.95, 0.36, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor="#CFE3F6", edgecolor="none"))
    ax.text(3.545, 0.42, "Query →", ha="center", va="center", fontsize=BODY_PT, fontweight="bold", color="#4F86C6")
    arrow(ax, (4.2, 0.42), (4.6, 0.42), blue, width=1.4)
    ax.text(4.7, 0.42, "Query", fontsize=11, color=blue, va="center", family="serif")
    arrow(ax, (5.85, 0.92), (5.85, 0.55), orange, width=1.6)
    ax.text(5.85, 0.4, "Results", fontsize=11, color=orange, ha="center", va="center", family="serif")
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=Path("."))
    parser.add_argument("--evidence", type=Path, default=None, help="target_evidence.csv (default: BASE)")
    parser.add_argument("--transfer", type=Path, default=None, help="transfer_output dir (default: BASE)")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--graphs", type=int, default=50)
    parser.add_argument("--per-graph", type=int, default=200)
    args = parser.parse_args()

    base = args.base
    evidence_path = args.evidence or base / "target_evidence.csv"
    transfer_dir = args.transfer or base / "transfer_output"
    args.out.mkdir(parents=True, exist_ok=True)

    dependencies = pd.read_csv(base / "step6_output" / "chisq_dependencies_fixed.csv")
    coefficients = pd.read_csv(base / "transfer_coefficients.csv")
    evidence = pd.read_csv(evidence_path)
    profiles = pd.read_csv(transfer_dir / "profiles_centered.csv")

    figure_pipeline(args.out / "fig_pipeline_steps.png", args.graphs, args.per_graph)
    figure_dag(args.out / "fig_dag.png", dependencies, coefficients)
    figure_evidence(args.out / "fig_step2_pooling.png", evidence)
    figure_generation(args.out / "step4.png", args.graphs, args.per_graph)
    figure_interface(args.out / "fig_gradio.png", profiles)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
