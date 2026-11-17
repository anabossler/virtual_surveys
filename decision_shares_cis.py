"""Population shares for the decision-robustness analysis, from CIS 3391.

Recomputes the acceptance classes of the 80 profile types (sex x under-30 x
higher education x recycling x five worry levels) under the sweeps of a0, p0
and the worry coding, and reports the PESOFIN-weighted share of CIS complete
cases in the types that keep or change class. It reuses the complete-case
filter, the worry codings and the intercept calibration of transfer_model.py,
so the shares refer to exactly the 2,133 respondents used in the paper.

Usage: python decision_shares_cis.py metadata/3391_num.csv
Output: transfer_output/decision_shares_cis.csv
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from transfer_model import PREDICTORS, calibrate_intercept, load_design, worry_score

A0_CENTRAL, P0_CENTRAL = 0.80, 0.38
A0_GRID = (0.50, 0.70, 0.90, 1.00)
P0_GRID = (0.25, 0.50, 0.60)
CODINGS = ("contrast", "per_step")
TYPES = list(itertools.product([0, 1], [0, 1], [0, 1], [0, 1], [1, 2, 3, 4, 5]))


def load_beta(path: Path) -> np.ndarray:
    coef = pd.read_csv(path)
    coef = coef[coef.iloc[:, 0] == "acceptance_food_packaging"].set_index("predictor")
    return coef.loc[list(PREDICTORS)].iloc[:, 3].astype(float).to_numpy()


def classes(design: pd.DataFrame, beta: np.ndarray, coding: str, a0: float, p0: float) -> np.ndarray:
    x = design[list(PREDICTORS[:-1])].to_numpy()
    x = np.column_stack([x, worry_score(design["V15"].to_numpy(), coding)])
    c = calibrate_intercept(a0 * (x @ beta), p0, design["weight"].to_numpy())
    t = np.array([[f, u, h, r, worry_score(np.array([v]), coding)[0]] for f, u, h, r, v in TYPES])
    p = norm.cdf(c + a0 * (t @ beta))
    return np.where(p < 0.25, 0, np.where(p <= 0.50, 1, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cis", type=Path)
    ap.add_argument("--coefficients", type=Path, default=Path("transfer_coefficients.csv"))
    ap.add_argument("--out", type=Path, default=Path("transfer_output/decision_shares_cis.csv"))
    args = ap.parse_args()

    design = load_design(args.cis, "centered")
    beta = load_beta(args.coefficients)
    print(f"complete cases: {len(design)} (paper: 2,133); beta = {np.round(beta, 4)}")

    keys = list(zip(*(design[c].astype(int) for c in PREDICTORS[:-1]), design["V15"]))
    w = pd.Series(design["weight"].to_numpy(), index=pd.MultiIndex.from_tuples(keys))
    w = w.groupby(level=list(range(5))).sum() / w.sum()
    share = np.array([w.get(k, 0.0) for k in TYPES])
    count = pd.Series(1, index=pd.MultiIndex.from_tuples(keys)).groupby(level=list(range(5))).sum()
    n_types_observed = int(sum(count.get(k, 0) > 0 for k in TYPES))

    base = classes(design, beta, "centered", A0_CENTRAL, P0_CENTRAL)
    print("central classes (<25 / 25-50 / >50):", [int((base == k).sum()) for k in range(3)], "(paper: 40/27/13)")
    print(f"profile types observed in the CIS: {n_types_observed}/80")

    rows, kept = [], {"a0": np.ones(80, bool), "p0": np.ones(80, bool)}
    scenarios = (
        [("a0", v, "centered", v, P0_CENTRAL) for v in A0_GRID]
        + [("p0", v, "centered", A0_CENTRAL, v) for v in P0_GRID]
        + [("worry", c, c, A0_CENTRAL, P0_CENTRAL) for c in CODINGS]
    )
    for dim, value, coding, a0, p0 in scenarios:
        cl = classes(design, beta, coding, a0, p0)
        changed = cl != base
        if dim in kept:
            kept[dim] &= ~changed
        rows.append(
            {
                "assumption": dim, "value": value,
                "below_25": int((cl == 0).sum()), "25_50": int((cl == 1).sum()), "above_50": int((cl == 2).sum()),
                "types_changed": int(changed.sum()), "share_changed": float(share[changed].sum()),
            }
        )
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for dim, k in kept.items():
        print(f"types keeping their class across the {dim} sweep: {int(k.sum())}/80; CIS share {share[k].sum():.1%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("->", args.out)


if __name__ == "__main__":
    main()
