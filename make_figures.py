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

Outputs (in --out):
    fig_pipeline_steps.png, fig_dag.png, fig_step2_pooling.png,
    step4.png, fig_gradio.png

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

DPI = 300
FONT = "DejaVu Sans"

# Two semantic colours: national data and external evidence. Everything else is neutral.
NATIONAL = "#2F5D8A"
NATIONAL_FILL = "#E6EEF6"
EXTERNAL = "#A8641E"
EXTERNAL_FILL = "#F6ECE0"
NEUTRAL = "#4A4A4A"
NEUTRAL_FILL = "#F2F2F2"
MUTED = "#8C8C8C"

plt.rcParams.update({"font.family": FONT, "font.size": 9, "savefig.dpi": DPI})

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
# ---------------------------------------------------------------------------

def box(ax, x, y, w, h, title, body="", edge=NEUTRAL, fill=NEUTRAL_FILL,
        title_size=9.5, body_size=8, dashed=False):
    """Rounded box anchored at its lower-left corner, with a bold title and body text."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12",
        linewidth=1.1, edgecolor=edge, facecolor=fill,
        linestyle=(0, (4, 3)) if dashed else "solid",
    ))
    if body:
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=edge)
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top",
                fontsize=body_size, color=NEUTRAL, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=edge)


def arrow(ax, start, end, color=NEUTRAL, dashed=False, label="", label_offset=(0, 0.12), width=1.1):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11, linewidth=width,
        color=color, linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=2, shrinkB=2,
    ))
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label, ha="center", va="bottom",
                fontsize=7.5, color=color, style="italic")


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: pipeline
# ---------------------------------------------------------------------------

def figure_pipeline(path: Path, n_graphs: int, n_per_graph: int):
    fig, ax = canvas(10.0, 6.4)

    ax.text(2.55, 6.2, "National data (observed variables)", ha="center",
            fontsize=10, fontweight="bold", color=NATIONAL)
    ax.text(7.55, 6.2, "External evidence (unobserved outcome)", ha="center",
            fontsize=10, fontweight="bold", color=EXTERNAL)
    ax.plot([5.05, 5.05], [1.75, 6.3], color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))

    w, h = 4.1, 1.05
    box(ax, 0.5, 4.85, w, h, "CIS 3391 microdata",
        "Spain 2023, n = 2,254, 90 variables\nfinal weight PESOFIN", NATIONAL, NATIONAL_FILL)
    box(ax, 0.5, 3.45, w, h, "Step 1  Structure learning",
        "Block-wise parallel-tempered MC3\nposterior over graphs, forbidden edges", NATIONAL, NATIONAL_FILL)
    box(ax, 0.5, 2.05, w, h, "Step 3  Conditional tables",
        "BDeu prior (ESS 10), PESOFIN-weighted counts\nno external evidence", NATIONAL, NATIONAL_FILL)

    box(ax, 5.5, 4.85, w, h, "Extraction corpus",
        "11 studies, 80 entries\nlanguage-model extraction, human review", EXTERNAL, EXTERNAL_FILL)
    box(ax, 5.5, 3.45, w, h, "Step 2  Review rules and evidence",
        "evidence_rules.csv; retained entries\nfor the five target outcomes", EXTERNAL, EXTERNAL_FILL)
    box(ax, 5.5, 2.05, w, h, "Transfer model",
        "Probit, coefficients and base rate\nfrom Ruokamo et al. (2022), a0 = 0.80", EXTERNAL, EXTERNAL_FILL)

    for x in (2.55, 7.55):
        arrow(ax, (x, 4.85), (x, 4.5), NATIONAL if x < 5 else EXTERNAL)
        arrow(ax, (x, 3.45), (x, 3.1), NATIONAL if x < 5 else EXTERNAL)

    box(ax, 1.6, 0.35, 6.8, 1.15, "Step 4  Synthetic population",
        f"{n_graphs} graphs x {n_per_graph} individuals = {n_graphs * n_per_graph:,}; "
        "observed variables by ancestral sampling,\n"
        "food-packaging acceptance attached to each individual from its own characteristics",
        NEUTRAL, NEUTRAL_FILL)
    arrow(ax, (2.55, 2.05), (3.3, 1.5), NATIONAL)
    arrow(ax, (7.55, 2.05), (6.7, 1.5), EXTERNAL)

    ax.text(9.75, 0.92, "Step 5\nvalidation\nand queries", ha="center", va="center",
            fontsize=8, color=NEUTRAL)
    arrow(ax, (8.4, 0.92), (9.15, 0.92), NEUTRAL)

    ax.annotate("Spanish data, when\navailable: Beta update", xy=(9.6, 2.3), xytext=(9.9, 1.75),
                fontsize=7.5, color=EXTERNAL, ha="center", style="italic",
                arrowprops=dict(arrowstyle="-|>", color=EXTERNAL, linewidth=0.9, linestyle=(0, (4, 3))))
    save(fig, path)


# ---------------------------------------------------------------------------
# Figure 2: learned structure at block level
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
    angles = np.pi / 2 - np.linspace(0, 2 * np.pi, len(order), endpoint=False)
    centre, rx, ry = np.array([5.5, 4.6]), 4.6, 3.5
    pos = {b: centre + np.array([rx * np.cos(a), ry * np.sin(a)]) for b, a in zip(order, angles)}

    fig, ax = canvas(11.0, 9.4)
    max_count = max(inter.values())
    for (a, b), count in sorted(inter.items(), key=lambda item: item[1]):
        width = 0.4 + 3.2 * count / max_count
        ax.plot(*zip(pos[a], pos[b]), color=NATIONAL, linewidth=width,
                alpha=0.25 + 0.5 * count / max_count, solid_capstyle="round", zorder=1)

    predictor_blocks = Counter(block_of[v] for v in TRANSFER_PREDICTORS)
    for block in order:
        x, y = pos[block]
        n_pred = predictor_blocks.get(block, 0)
        ax.add_patch(FancyBboxPatch(
            (x - 0.78, y - 0.27), 1.56, 0.54, boxstyle="round,pad=0.0,rounding_size=0.1",
            linewidth=1.6 if n_pred else 1.0, edgecolor=EXTERNAL if n_pred else NATIONAL,
            facecolor="white", zorder=2,
        ))
        ax.text(x, y + 0.08, BLOCK_LABELS[block], ha="center", va="center",
                fontsize=7.2, fontweight="bold", color=NEUTRAL, zorder=3)
        ax.text(x, y - 0.14, f"{sizes[block]} var., {intra.get(block, 0)} within",
                ha="center", va="center", fontsize=6.6, color=MUTED, zorder=3)

    lines = [f"{v}: {label} ({BLOCK_LABELS[block_of[v]]})" for v, label in TRANSFER_PREDICTORS.items()]
    ax.text(0.05, 0.0, "Predictors of the transfer model (orange outline)\n" + "\n".join(lines),
            fontsize=7, color=EXTERNAL, va="bottom", linespacing=1.4)
    ax.text(10.95, 0.0,
            f"{len(block_of)} variables, {len(edges)} edges\n"
            f"{int((edges.edge_type == 'intra').sum())} within blocks, "
            f"{int((edges.edge_type == 'inter').sum())} across blocks\n"
            "line width: number of edges between two blocks",
            fontsize=7, color=MUTED, ha="right", va="bottom", linespacing=1.4)
    save(fig, path)


# ---------------------------------------------------------------------------
# Figure 3: external evidence
# ---------------------------------------------------------------------------

def figure_evidence(path: Path, evidence: pd.DataFrame):
    fig, ax = canvas(10.0, 4.9)

    retained_ids = sorted({s for ids in evidence.study_ids.dropna() for s in ids.split(";") if s})
    n_other = 11 - len(retained_ids)

    box(ax, 0.2, 3.55, 2.5, 1.0, "Extraction corpus", "11 studies\n80 entries", EXTERNAL, EXTERNAL_FILL)
    box(ax, 0.2, 2.1, 2.5, 1.0, "Review rules", "evidence_rules.csv\n15 corrections", EXTERNAL, EXTERNAL_FILL)
    arrow(ax, (1.45, 3.55), (1.45, 3.1), EXTERNAL)
    ax.text(1.45, 1.75, f"{n_other} studies: no retained\nentry for a target outcome\n"
            "(entries for CIS-observed\nvariables are not used)",
            ha="center", va="top", fontsize=7.5, color=MUTED, linespacing=1.35)

    y_rows = np.linspace(4.2, 0.6, len(TARGET_LABELS))
    for (construct, label), y in zip(TARGET_LABELS.items(), y_rows):
        row = evidence.loc[evidence.construct == construct].iloc[0]
        studies = [STUDY_LABELS[s] for s in str(row.study_ids).split(";") if s in STUDY_LABELS]
        n_entries = int(row.entries)
        if n_entries == 0:
            detail = "no usable entry"
        else:
            detail = (f"{', '.join(studies)}\n{n_entries} entr{'y' if n_entries == 1 else 'ies'}, "
                      f"independent n = {int(row.independent_n):,}")
        is_food = construct == "acceptance_food_packaging"
        ax.add_patch(FancyBboxPatch(
            (3.4, y - 0.36), 3.4, 0.72, boxstyle="round,pad=0.0,rounding_size=0.1", linewidth=1.1,
            edgecolor=EXTERNAL if is_food else NEUTRAL, facecolor=EXTERNAL_FILL if is_food else "white"))
        ax.text(5.1, y + 0.2, label, ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=EXTERNAL if is_food else NEUTRAL)
        ax.text(5.1, y - 0.1, detail, ha="center", va="center", fontsize=6.8, color=MUTED, linespacing=1.25)
        arrow(ax, (2.7, 2.6), (3.4, y), MUTED, width=0.8)

    food_y = y_rows[list(TARGET_LABELS).index("acceptance_food_packaging")]
    box(ax, 7.5, food_y - 0.55, 2.4, 1.1, "Transfer model",
        "base rate p0 = 0.38\n5 coefficients, a0 = 0.80", EXTERNAL, EXTERNAL_FILL)
    arrow(ax, (6.8, food_y), (7.5, food_y), EXTERNAL, label="projected")
    box(ax, 7.5, 0.5, 2.4, 1.3, "Reported as evidence",
        "pooled shares and I2\nin the evidence tables\nnot projected", NEUTRAL, NEUTRAL_FILL)
    for construct, y in zip(TARGET_LABELS, y_rows):
        if construct not in ("acceptance_food_packaging", "perception_quality"):
            arrow(ax, (6.8, y), (7.5, 1.15), MUTED, width=0.7)
    save(fig, path)


# ---------------------------------------------------------------------------
# Figure 4: generation of the synthetic population
# ---------------------------------------------------------------------------

def figure_generation(path: Path, n_graphs: int, n_per_graph: int):
    fig, ax = canvas(11.2, 3.6)

    box(ax, 0.2, 1.0, 2.2, 1.9, f"{n_graphs} graphs",
        "drawn from the\nposterior of Step 1\n\nconditional tables\nfrom Step 3", NATIONAL, NATIONAL_FILL)
    box(ax, 3.0, 1.0, 2.4, 1.9, f"{n_per_graph} per graph",
        "ancestral sampling of the\n90 observed variables\n\none draw of the transfer\ninputs per graph", NEUTRAL, NEUTRAL_FILL)
    arrow(ax, (2.4, 1.95), (3.0, 1.95), NATIONAL)

    x0, y0, cw, rh = 6.05, 0.75, 0.6, 0.38
    observed = ["SEX", "BIRTH", "EDUC.", "V15", "V52", "...", "V58"]
    headers = observed + ["Food pkg."]
    for j, name in enumerate(headers):
        is_target = j == len(headers) - 1
        x = x0 + j * cw + (0.12 if is_target else 0)
        ax.add_patch(FancyBboxPatch((x, y0 + 4 * rh), cw - 0.04, rh, boxstyle="square,pad=0",
                                    facecolor=EXTERNAL_FILL if is_target else NATIONAL_FILL,
                                    edgecolor="none"))
        ax.text(x + cw / 2 - 0.02, y0 + 4.5 * rh, name, ha="center", va="center", fontsize=6.8,
                fontweight="bold", color=EXTERNAL if is_target else NATIONAL)
        for i in range(4):
            ax.add_patch(FancyBboxPatch((x, y0 + i * rh), cw - 0.04, rh - 0.04, boxstyle="square,pad=0",
                                        facecolor="white", edgecolor="#D9D9D9", linewidth=0.6))
    ax.text(x0 + 3.5 * cw, y0 - 0.3, f"{n_graphs * n_per_graph:,} synthetic individuals",
            ha="center", fontsize=8, color=NEUTRAL)
    ax.text(x0 + 3.5 * cw, y0 + 5.35 * rh, "observed variables: national data only",
            ha="center", fontsize=7.5, color=NATIONAL)
    ax.text(x0 + 7.5 * cw + 0.12, y0 + 5.35 * rh, "Eq. (transfer)", ha="center", fontsize=7.5, color=EXTERNAL)
    arrow(ax, (5.4, 1.95), (6.0, 1.95), NEUTRAL)
    save(fig, path)


# ---------------------------------------------------------------------------
# Figure 5: query flow of the interface, with the current model's output
# ---------------------------------------------------------------------------

def figure_interface(path: Path, profiles: pd.DataFrame):
    row = profiles.set_index("quantity").loc["most_receptive"]
    estimate, low, high = row.estimate, row.ci_low, row.ci_high

    fig, ax = canvas(10.0, 3.3)
    box(ax, 0.2, 0.75, 2.75, 2.1, "1  Free-text query",
        "\"Would a woman under 30 with\na university degree, who always\nrecycles and is extremely worried\n"
        "about the environment, accept\nrecycled plastic in food packaging?\"",
        NEUTRAL, NEUTRAL_FILL, body_size=7.4)
    box(ax, 3.55, 0.75, 2.75, 2.1, "2  Parsed profile",
        "language model, structured output\n\nfemale = 1, under 30 = 1\nhigher education = 1\n"
        "recycles = 1, worry = 5\noutcome = food packaging", NEUTRAL, NEUTRAL_FILL, body_size=7.4)
    box(ax, 6.9, 0.75, 2.9, 2.1, "3  Inference engine",
        f"transfer model, Monte Carlo\n\n{estimate:.0%} (95% interval {low:.0%}\u2013{high:.0%})\n\n"
        "projected from transferred\nevidence; not validated in Spain", EXTERNAL, EXTERNAL_FILL, body_size=7.4)
    arrow(ax, (2.95, 1.8), (3.55, 1.8), NEUTRAL)
    arrow(ax, (6.3, 1.8), (6.9, 1.8), NEUTRAL)
    ax.text(5.0, 0.3, "The language model parses the query and writes the narrative; "
            "it generates no probability.", ha="center", fontsize=7.5, color=MUTED, style="italic")
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
