"""Segment-level sensitivity and measurement plan for the transferred outcome.

Two outputs support the use of the transfer model in pilot and survey design:

1. segments_a0.csv
   Projected food-packaging acceptance for the main CIS segments under each
   transfer factor a0 (the population mean stays at p0 by calibration).

2. measurement_plan.csv
   For each segment at the central specification: the Monte Carlo mean and
   95% interval of the projection, the concentration m of the Beta prior
   matched to the draws (m = mean (1 - mean) / var - 1), and the number of
   Spanish respondents N needed for a 95% posterior interval of half-width h
   under a power-prior discount delta (Ibrahim and Chen, 2000):

       N >= z^2 p (1 - p) / h^2 - delta m - 1

   delta = 1 uses the transferred evidence at face value; delta < 1 discounts
   it for transfer and extraction uncertainty that the intervals omit.

Usage
-----
    python measurement_plan.py --cis metadata/3391_num.csv --out transfer_output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from transfer_model import (
    A0, N_DRAWS, P0, P0_N, PREDICTORS, SEED, calibrate_intercept, load_design,
)

CODING = "centered"
A0_GRID = (0.50, 0.65, 0.80, 1.00)
DELTAS = (1.0, 0.5, 0.25)
HALF_WIDTH = 0.05
Z = norm.ppf(0.975)


def segments(design: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks of the reported segments."""
    return {
        "All residents": np.ones(len(design), dtype=bool),
        "Women": design["female"].to_numpy() == 1,
        "Men": design["female"].to_numpy() == 0,
        "Higher education": design["high_education"].to_numpy() == 1,
        "Below higher education": design["high_education"].to_numpy() == 0,
        "Recycles always or often": design["recycles"].to_numpy() == 1,
        "Recycles sometimes or never": design["recycles"].to_numpy() == 0,
        "Extremely worried (V15 = 5)": design["V15"].to_numpy() == 5,
        "Not at all worried (V15 = 1)": design["V15"].to_numpy() == 1,
    }


def segment_means(x, weights, masks, beta, p0, a0) -> dict[str, float]:
    linear = a0 * (x @ beta)
    probabilities = norm.cdf(calibrate_intercept(linear, p0, weights) + linear)
    return {name: float(np.average(probabilities[m], weights=weights[m])) for name, m in masks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cis", type=Path, default=Path("metadata/3391_num.csv"))
    parser.add_argument("--coefficients", type=Path, default=Path("transfer_coefficients.csv"))
    parser.add_argument("--out", type=Path, default=Path("transfer_output"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.coefficients).set_index("predictor").loc[list(PREDICTORS)]
    beta, se = table["coefficient"].to_numpy(), table["std_error"].to_numpy()
    design = load_design(args.cis, CODING)
    x = design[list(PREDICTORS)].to_numpy()
    weights = design["weight"].to_numpy()
    masks = segments(design)

    rows = []
    for a0 in A0_GRID:
        for name, value in segment_means(x, weights, masks, beta, P0, a0).items():
            rows.append({"a0": a0, "segment": name, "n": int(masks[name].sum()), "projected_acceptance": value})
    pd.DataFrame(rows).to_csv(args.out / "segments_a0.csv", index=False)

    rng = np.random.default_rng(SEED)
    betas = rng.normal(beta, se, size=(N_DRAWS, len(beta)))
    p0s = rng.beta(P0 * P0_N + 0.5, (1 - P0) * P0_N + 0.5, size=N_DRAWS)
    draws = pd.DataFrame([segment_means(x, weights, masks, b, p, A0) for b, p in zip(betas, p0s)])

    rows = []
    for name in masks:
        values = draws[name].to_numpy()
        mean, var = float(values.mean()), float(values.var(ddof=1))
        m = mean * (1.0 - mean) / var - 1.0
        row = {
            "segment": name,
            "n_cis": int(masks[name].sum()),
            "mean": mean,
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
            "m": m,
        }
        for delta in DELTAS:
            needed = Z**2 * mean * (1.0 - mean) / HALF_WIDTH**2 - delta * m - 1.0
            row[f"N_delta_{delta:g}"] = max(0, int(np.ceil(needed)))
        rows.append(row)
    plan = pd.DataFrame(rows)
    plan.to_csv(args.out / "measurement_plan.csv", index=False)
    print(plan.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
