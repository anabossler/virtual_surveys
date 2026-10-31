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

FONT = "DejaVu Sans"

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
# Figure: pipeline
# ---------------------------------------------------------------------------

def figure_pipeline(path: Path, n_graphs: int, n_per_graph: int):
    fig, ax = canvas(4.35)
    left, right, w, h = 0.05, 3.45, 3.0, 0.72
    ax.text(left + w / 2, 4.2, "National data (observed variables)", ha="center", va="center",
            fontsize=9, fontweight="bold", color=NATIONAL)
    ax.text(right + w / 2, 4.2, "External evidence (unobserved outcome)", ha="center", va="center",
            fontsize=9, fontweight="bold", color=EXTERNAL)
    ax.plot([3.25, 3.25], [1.3, 4.3], color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)))

    rows = (3.2, 2.25, 1.3)
    box(ax, left, rows[0], w, h, "CIS 3391 microdata",
        "Spain 2023, $n$ = 2,254, 90 variables\nfinal weight PESOFIN", NATIONAL, NATIONAL_FILL)
    box(ax, left, rows[1], w, h, "Step 1  Structure learning",
        "block-wise parallel-tempered MC$^3$\nposterior over graphs, forbidden edges", NATIONAL, NATIONAL_FILL)
    box(ax, left, rows[2], w, h, "Step 3  Conditional tables",
        "BDeu prior (ESS 10), PESOFIN-weighted counts\nno external evidence", NATIONAL, NATIONAL_FILL)
    box(ax, right, rows[0], w, h, "Extraction corpus",
        "11 studies, 80 entries\nlanguage-model extraction, human review", EXTERNAL, EXTERNAL_FILL)
    box(ax, right, rows[1], w, h, "Step 2  Review rules and evidence",
        "evidence_rules.csv; retained entries\nfor the five target outcomes", EXTERNAL, EXTERNAL_FILL)
    box(ax, right, rows[2], w, h, "Transfer model",
        "probit; coefficients and base rate from\nRuokamo et al. (2022); $a_0$ = 0.80; Beta update",
        EXTERNAL, EXTERNAL_FILL)
    for x, color in ((left + w / 2, NATIONAL), (right + w / 2, EXTERNAL)):
        for upper, lower in ((rows[0], rows[1]), (rows[1], rows[2])):
            arrow(ax, (x, upper), (x, lower + h), color)

    box(ax, 0.05, 0.05, 5.1, 0.85, "Step 4  Synthetic population",
        f"{n_graphs} graphs $\\times$ {n_per_graph} individuals = {n_graphs * n_per_graph:,}; "
        "observed variables by ancestral sampling;\n"
        "food-packaging acceptance attached from each individual's own characteristics",
        NEUTRAL, NEUTRAL_FILL)
    arrow(ax, (left + w / 2, rows[2]), (1.55, 0.9), NATIONAL)
    arrow(ax, (right + w / 2, rows[2]), (3.65, 0.9), EXTERNAL)
    box(ax, 5.45, 0.05, 1.0, 0.85, "Step 5", "validation\nand queries", NEUTRAL, "white")
    arrow(ax, (5.15, 0.475), (5.45, 0.475), NEUTRAL)
    save(fig, path)


# ---------------------------------------------------------------------------
# Figure: learned structure at block level
# ---------------------------------------------------------------------------

def figure_dag(path: Path, edges: pd.DataFrame):
    block_of = {}
    for row in edges.itertuples():
        block_of[row.source] = row.block_source
        block_of[row.target] = row.block_target
    sizes = Counter(block_of.values())
    intra = Counter(edges.loc[edges.edge_type == "intra", "block_source"])
    inter = Counter(
        tuple(sorted((r.block_source, r.block_target)))
        for r in edges.itertuples() if r.edge_type == "inter"
    )
    order = [
        "demographics", "employment", "origin_family", "satisfaction", "political_ideology",
        "political_values", "political_issues", "trust_institutions", "science_tech",
        "danger_pollution", "ecological_footprint", "behavior", "wtp_acceptance",
        "env_concern_b", "env_concern_a",
    ]
    # Blocks on the perimeter of a rectangle, clockwise from the top left, so that
    # thematically adjacent blocks are neighbours and no labels overlap.
    xs = (0.65, 1.95, 3.25, 4.55, 5.85)
    top, bottom = 5.0, 1.15
    perimeter = (
        [(x, top) for x in xs[1:]]                     # demographics .. satisfaction (4)
        + [(xs[-1], y) for y in (4.05, 3.08, 2.11)]    # right side (3)
        + [(x, bottom) for x in reversed(xs)]          # bottom row (5)
        + [(xs[0], y) for y in (2.43, 3.72)]           # left side (2)
        + [(xs[0], top)]                               # top left (1)
    )
    pos = {b: np.array(p) for b, p in zip(order, perimeter)}

    fig, ax = canvas(5.45)
    max_count = max(inter.values())
    for (a, b), count in sorted(inter.items(), key=lambda item: item[1]):
        ax.plot(*zip(pos[a], pos[b]), color=NATIONAL, linewidth=0.3 + 2.4 * count / max_count,
                alpha=0.25 + 0.55 * count / max_count, solid_capstyle="round", zorder=1)

    predictor_blocks = {block_of[v] for v in TRANSFER_PREDICTORS}
    bw, bh = 1.22, 0.42
    for block in order:
        x, y = pos[block]
        marked = block in predictor_blocks
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                    linewidth=1.4 if marked else 0.8, edgecolor=EXTERNAL if marked else NATIONAL,
                                    facecolor="white", zorder=2))
        ax.text(x, y + 0.07, BLOCK_LABELS[block], ha="center", va="center", fontsize=BODY_PT,
                fontweight="bold", color=NEUTRAL, zorder=3)
        ax.text(x, y - 0.1, f"{sizes[block]} var., {intra.get(block, 0)} within", ha="center", va="center",
                fontsize=SMALL_PT - 0.5, color=MUTED, zorder=3)

    by_block = {}
    for var in TRANSFER_PREDICTORS:
        by_block.setdefault(BLOCK_LABELS[block_of[var]], []).append(var)
    legend = "; ".join(f"{block}: {', '.join(v)}" for block, v in by_block.items())
    n_intra = int((edges.edge_type == "intra").sum())
    n_inter = int((edges.edge_type == "inter").sum())
    ax.text(0.05, 0.62, "Orange outline: blocks containing the predictors of the transfer model\n"
            f"({legend}).", fontsize=SMALL_PT, color=EXTERNAL, va="top", linespacing=1.3)
    ax.text(0.05, 0.27, f"{len(block_of)} variables, {len(edges)} edges ({n_intra} within blocks, "
            f"{n_inter} across). Line width: number of edges between two blocks.",
            fontsize=SMALL_PT, color=MUTED, va="top")
    save(fig, path)


# ---------------------------------------------------------------------------
# Figure: external evidence
# ---------------------------------------------------------------------------

def figure_evidence(path: Path, evidence: pd.DataFrame):
    fig, ax = canvas(3.75)
    retained_ids = sorted({s for ids in evidence.study_ids.dropna() for s in ids.split(";") if s})
    n_other = 11 - len(retained_ids)

    box(ax, 0.05, 2.75, 1.5, 0.75, "Extraction corpus", "11 studies\n80 entries", EXTERNAL, EXTERNAL_FILL)
    box(ax, 0.05, 1.7, 1.5, 0.75, "Review rules", "evidence_rules.csv\n15 corrections", EXTERNAL, EXTERNAL_FILL)
    arrow(ax, (0.8, 2.75), (0.8, 2.45), EXTERNAL)
    ax.text(0.8, 1.5, f"{n_other} studies with no\nretained entry for a\ntarget outcome; entries\n"
            "for CIS-observed\nvariables are not used", ha="center", va="top", fontsize=SMALL_PT,
            color=MUTED, linespacing=1.3)

    mx, mw, rh = 1.95, 2.45, 0.6
    y_rows = np.linspace(3.35, 0.4, len(TARGET_LABELS))
    for (construct, label), y in zip(TARGET_LABELS.items(), y_rows):
        row = evidence.loc[evidence.construct == construct].iloc[0]
        studies = [STUDY_LABELS[s] for s in str(row.study_ids).split(";") if s in STUDY_LABELS]
        n_entries = int(row.entries)
        detail = ("no usable entry" if n_entries == 0 else
                  f"{', '.join(studies)}\n{n_entries} entr{'y' if n_entries == 1 else 'ies'}, "
                  f"independent $n$ = {int(row.independent_n):,}")
        is_food = construct == "acceptance_food_packaging"
        box(ax, mx, y - rh / 2, mw, rh, label, detail, EXTERNAL if is_food else NEUTRAL,
            EXTERNAL_FILL if is_food else "white", title_pt=BODY_PT + 0.5, body_pt=SMALL_PT - 0.5)
        arrow(ax, (1.55, 2.07), (mx, y), MUTED, width=0.6)

    rx, rw = 4.8, 1.65
    food_y = y_rows[list(TARGET_LABELS).index("acceptance_food_packaging")]
    box(ax, rx, food_y - 0.4, rw, 0.8, "Transfer model",
        "base rate $p_0$ = 0.38\n5 coefficients, $a_0$ = 0.80", EXTERNAL, EXTERNAL_FILL)
    arrow(ax, (mx + mw, food_y), (rx, food_y), EXTERNAL, label="projected")
    box(ax, rx, 0.45, rw, 0.95, "Reported only",
        "pooled shares and $I^2$\nin the evidence tables;\nnot projected", NEUTRAL, NEUTRAL_FILL)
    for construct, y in zip(TARGET_LABELS, y_rows):
        if construct not in ("acceptance_food_packaging", "perception_quality"):
            arrow(ax, (mx + mw, y), (rx, 0.925), MUTED, width=0.6)
    save(fig, path)


# ---------------------------------------------------------------------------
# Figure: generation of the synthetic population
# ---------------------------------------------------------------------------

def figure_generation(path: Path, n_graphs: int, n_per_graph: int):
    fig, ax = canvas(1.95)
    box(ax, 0.05, 0.3, 1.3, 1.3, f"{n_graphs} graphs",
        "drawn from the\nposterior of Step 1;\nconditional tables\nfrom Step 3", NATIONAL, NATIONAL_FILL)
    box(ax, 1.55, 0.3, 1.5, 1.3, f"{n_per_graph} per graph",
        "ancestral sampling of\nthe 90 observed variables;\none draw of the\ntransfer inputs per graph",
        NEUTRAL, NEUTRAL_FILL)
    arrow(ax, (1.35, 0.95), (1.55, 0.95), NATIONAL)
    arrow(ax, (3.05, 0.95), (3.28, 0.95), NEUTRAL)

    x0, y0, cw, rh = 3.3, 0.3, 0.31, 0.22
    observed = ["SEX", "BIRTH", "EDUC.", "V15", "V52", "...", "V58"]
    for j, name in enumerate(observed + ["Food pkg."]):
        target = j == len(observed)
        x = x0 + j * cw + (0.08 if target else 0)
        width = 0.58 if target else cw - 0.03
        ax.add_patch(FancyBboxPatch((x, y0 + 4 * rh), width, rh, boxstyle="square,pad=0",
                                    facecolor=EXTERNAL_FILL if target else NATIONAL_FILL, edgecolor="none"))
        ax.text(x + width / 2, y0 + 4.5 * rh, name, ha="center", va="center", fontsize=SMALL_PT - 1,
                fontweight="bold", color=EXTERNAL if target else NATIONAL)
        for i in range(4):
            ax.add_patch(FancyBboxPatch((x, y0 + i * rh), width, rh - 0.03, boxstyle="square,pad=0",
                                        facecolor="white", edgecolor="#D0D0D0", linewidth=0.5))
    ax.text(x0 + 3.5 * cw, y0 + 5 * rh + 0.08, "observed: national data only", ha="center", va="bottom",
            fontsize=SMALL_PT, color=NATIONAL)
    ax.text(x0 + 7 * cw + 0.08 + 0.29, y0 + 5 * rh + 0.08, "transfer model", ha="center", va="bottom",
            fontsize=SMALL_PT, color=EXTERNAL)
    ax.text(x0 + 4.3 * cw, 0.08, f"{n_graphs * n_per_graph:,} synthetic individuals", ha="center",
            va="bottom", fontsize=SMALL_PT, color=NEUTRAL)
    save(fig, path)


# ---------------------------------------------------------------------------
# Figure: query flow of the interface, with the current model's output
# ---------------------------------------------------------------------------

def figure_interface(path: Path, profiles: pd.DataFrame):
    row = profiles.set_index("quantity").loc["most_receptive"]
    estimate, low, high = row.estimate, row.ci_low, row.ci_high
    fig, ax = canvas(1.95)
    w, gap, y, h = 1.93, 0.3, 0.3, 1.6
    box(ax, 0.05, y, w, h, "1  Free-text query",
        "“Would a woman under 30 with a\nuniversity degree, who always\nrecycles and is extremely\n"
        "worried about the environment,\naccept recycled plastic in food\npackaging?”", NEUTRAL, NEUTRAL_FILL)
    box(ax, 0.05 + w + gap, y, w, h, "2  Parsed profile",
        "language model, structured output\n\nfemale = 1, under 30 = 1,\nhigher education = 1,\n"
        "recycles = 1, worry = 5;\noutcome: food packaging", NEUTRAL, NEUTRAL_FILL)
    box(ax, 0.05 + 2 * (w + gap), y, w, h, "3  Inference engine",
        f"transfer model, Monte Carlo\n\n{estimate:.0%} (95% interval {low:.0%}–{high:.0%})\n\n"
        "projected from transferred\nevidence; not validated in Spain", EXTERNAL, EXTERNAL_FILL)
    for k in (1, 2):
        x = 0.05 + k * w + (k - 1) * gap
        arrow(ax, (x, y + h / 2), (x + gap, y + h / 2), NEUTRAL)
    ax.text(TEXT_WIDTH / 2, 0.08, "The language model parses the query and writes the narrative; "
            "it generates no probability.", ha="center", va="bottom", fontsize=SMALL_PT, color=MUTED,
            style="italic")
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

    edges = pd.read_csv(base / "dag_edges_3391.csv")
    evidence = pd.read_csv(evidence_path)
    profiles = pd.read_csv(transfer_dir / "profiles_centered.csv")

    figure_pipeline(args.out / "fig_pipeline_steps.png", args.graphs, args.per_graph)
    figure_dag(args.out / "fig_dag.png", edges)
    figure_evidence(args.out / "fig_step2_pooling.png", evidence)
    figure_generation(args.out / "step4.png", args.graphs, args.per_graph)
    figure_interface(args.out / "fig_gradio.png", profiles)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
