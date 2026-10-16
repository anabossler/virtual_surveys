"""Transfer-model projections with empirical transport error.

Extends the Monte Carlo of transfer_model.py with three sources of
cross-national error measured on ISSP 2020 (transport_issp.py):

    a0      drawn from N(a0_fi_es, sd), the Finland-to-Spain slope of
            coefficients on proxy outcomes, instead of fixed at 0.80
    eta_j   per-coefficient transport noise N(0, transport_sd^2), the
            Spain-Finland discrepancy left after a0 and sampling error
    delta   level shift N(0, level_rms^2) on the probit scale, added to the
            calibrated intercept, so the Spanish mean is no longer fixed at p0

    P(Y = 1 | x) = Phi(c + delta + sum_j (a0 beta_j + eta_j) x_j)

Outputs (in --out):
    transport_profiles.csv   two contrasting profiles and their spread
    transport_segments.csv   segments of Table 6 and the population mean
Each row also gives the Beta concentration m matched to the draws and the
Spanish sample size N for a 95% posterior interval of +/-5 points.

Usage
-----
    python transfer_transport.py --cis metadata/3391_num.csv \
        --transport transport_output/issp_transport.json --out transfer_output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from measurement_plan import segments
from transfer_model import (
    N_DRAWS, P0, P0_N, PREDICTORS, PROFILES, SEED, calibrate_intercept, load_design, profile_vector,
)

CODING = "centered"
HALF_WIDTH = 0.05


def interval(values: np.ndarray) -> tuple[float, float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(np.median(values)), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cis", type=Path, default=Path("metadata/3391_num.csv"))
    parser.add_argument("--coefficients", type=Path, default=Path("transfer_coefficients.csv"))
    parser.add_argument("--transport", type=Path, default=Path("transport_output/issp_transport.json"))
    parser.add_argument("--out", type=Path, default=Path("transfer_output"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    transport = json.loads(args.transport.read_text(encoding="utf-8"))
    a0_mean, a0_sd = transport["a0_fi_es"]["estimate"], transport["a0_fi_es"]["sd"]
    eta_sd = transport["coefficient_residual_fi_es"]["transport_sd"]
    level_sd = transport["level_rms_fi_es"]

    table = pd.read_csv(args.coefficients).set_index("predictor").loc[list(PREDICTORS)]
    beta, se = table["coefficient"].to_numpy(), table["std_error"].to_numpy()
    design = load_design(args.cis, CODING)
    x = design[list(PREDICTORS)].to_numpy()
    weights = design["weight"].to_numpy()
    masks = segments(design)
    profiles = {name: profile_vector(p, CODING) for name, p in PROFILES.items()}

    rng = np.random.default_rng(SEED)
    rows = {"profiles": [], "segments": []}
    for _ in range(N_DRAWS):
        a0 = rng.normal(a0_mean, a0_sd)
        b = a0 * rng.normal(beta, se) + rng.normal(0.0, eta_sd, len(beta))
        p0 = rng.beta(P0 * P0_N + 0.5, (1 - P0) * P0_N + 0.5)
        linear = x @ b
        shift = calibrate_intercept(linear, p0, weights) + rng.normal(0.0, level_sd)
        probabilities = norm.cdf(shift + linear)
        prof = {name: float(norm.cdf(shift + v @ b)) for name, v in profiles.items()}
        prof["spread"] = prof["most_receptive"] - prof["least_receptive"]
        rows["profiles"].append(prof)
        rows["segments"].append({n: float(np.average(probabilities[m], weights=weights[m])) for n, m in masks.items()})

    for key, name in (("profiles", "transport_profiles.csv"), ("segments", "transport_segments.csv")):
        draws = pd.DataFrame(rows[key])
        records = []
        for c in draws.columns:
            values = draws[c].to_numpy()
            mean, var = float(values.mean()), float(values.var(ddof=1))
            m = mean * (1.0 - mean) / var - 1.0
            needed = norm.ppf(0.975) ** 2 * mean * (1.0 - mean) / HALF_WIDTH**2 - m - 1.0
            records.append(dict(zip(("quantity", "median", "ci_low", "ci_high"), (c, *interval(values))))
                           | {"mean": mean, "m": m, "N": max(0, int(np.ceil(needed)))})
        out = pd.DataFrame(records)
        out.to_csv(args.out / name, index=False)
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
