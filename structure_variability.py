"""Compare structure learning and fidelity across runs with different seeds.

Each run is given as SEED=STRUCTURE_DIR:DIAGNOSTICS_DIR, where STRUCTURE_DIR
holds dag_edges_3391.csv and step4_output/target_summary.json, and
DIAGNOSTICS_DIR holds the step6_output of that run.

Usage
-----
    python structure_variability.py --out structure_variability.json \
        42=runs/s42:runs/s42 7=runs/s7:runs/s7/chk 2024=runs/s2024:runs/s2024/chk
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd


def edges(path: Path) -> set[tuple[str, str]]:
    frame = pd.read_csv(path / "dag_edges_3391.csv")
    return set(zip(frame["source"], frame["target"]))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def summarise(seed: str, structure: Path, diagnostics: Path) -> dict:
    report = json.loads((diagnostics / "step6_output" / "full_report_fixed.json").read_text())
    chi = pd.read_csv(diagnostics / "step6_output" / "chisq_dependencies_fixed.csv")
    target = json.loads((structure / "step4_output" / "target_summary.json").read_text())
    return {
        "seed": seed,
        "n_edges": len(edges(structure)),
        "tvd_mean": report["tvd_mean"],
        "tvd_median": report["tvd_median"],
        "dependencies_kept": int(chi["preserved"].sum()),
        "v_ratio": dict(zip(chi["pair"], chi["V_ratio"])),
        "share_accepting": target["share_accepting"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("structure_variability.json"))
    args = parser.parse_args()

    runs, graphs = [], {}
    for spec in args.runs:
        seed, dirs = spec.split("=", 1)
        structure, diagnostics = (Path(p) for p in dirs.split(":", 1))
        runs.append(summarise(seed, structure, diagnostics))
        graphs[seed] = edges(structure)

    pairwise = {f"{a}-{b}": jaccard(graphs[a], graphs[b]) for a, b in combinations(graphs, 2)}
    pairs = runs[0]["v_ratio"].keys()
    v_range = {p: (min(r["v_ratio"][p] for r in runs), max(r["v_ratio"][p] for r in runs)) for p in pairs}
    tvd = [r["tvd_mean"] for r in runs]
    kept = [r["dependencies_kept"] for r in runs]
    share = [r["share_accepting"] for r in runs]
    rec = next(p for p in pairs if "recycling" in p)
    # Pairs whose two variables share a thematic block in step1.py.
    within_block = {rec, "env_concern -> WTP"}
    cross = [v_range[p] for p in pairs if p not in within_block]

    text = (
        f"Across {len(runs)} structure-learning seeds, the learned graphs share "
        f"{100 * min(pairwise.values()):.0f}--{100 * max(pairwise.values()):.0f}\\% of their edges (Jaccard), "
        f"mean TVD ranges from ${min(tvd):.3f}$ to ${max(tvd):.3f}$, and {min(kept)}--{max(kept)} of the ten "
        f"dependencies remain detectable. The within-block concern--recycling ratio stays between "
        f"{v_range[rec][0]:.2f} and {v_range[rec][1]:.2f}, while the cross-block ratios vary between "
        f"{min(lo for lo, _ in cross):.2f} and {max(hi for _, hi in cross):.2f}. "
        f"The share accepting recycled food packaging in the synthetic population ranges "
        f"from {100 * min(share):.1f}\\% to {100 * max(share):.1f}\\%."
    )
    out = {"runs": runs, "edge_jaccard": pairwise, "v_ratio_range": v_range, "text": text}
    args.out.write_text(json.dumps(out, indent=2))
    print(text)


if __name__ == "__main__":
    main()
