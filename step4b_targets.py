"""Add the unmeasured target outcome to the synthetic population.

Observed CIS variables are generated in Step 4 by the Bayesian network with
uninformative Dirichlet priors (Martins et al. 2024), population S2. External
evidence enters only here, for food-packaging acceptance, which CIS 3391 does
not measure. Each synthetic individual receives a draw from the transfer model
of transfer_model.py evaluated at that individual's own characteristics.

Parameter uncertainty is propagated per DAG draw: synthetic individuals from
draw m share one Monte Carlo draw of the coefficients and of the base rate.

Usage
-----
    python step4b_targets.py
Inputs : step4_output/synthetic_surveys_S2.csv, metadata/3391_num.csv,
         transfer_coefficients.csv
Output : step4_output/synthetic_surveys_final.csv,
         step4_output/target_summary.json
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from step4 import Config, discretise, load_cis_microdata
from transfer_model import (
    A0,
    P0,
    P0_N,
    PREDICTORS,
    SEED,
    SURVEY_YEAR,
    calibrate_intercept,
    load_design,
    worry_score,
)

WORRY_CODING = "contrast"
TARGET = "acceptance_food_packaging"


def decode(synthetic: pd.DataFrame, binfo: dict) -> pd.DataFrame:
    """Map 0-based synthetic codes back to CIS values for ordinal variables."""
    decoded = {}
    for var in ("SEX", "ESTUDIOS", "V52", "V15"):
        labels = binfo[var]["labels"]
        decoded[var] = synthetic[var].map(lambda i, labels=labels: labels[int(i)])
    decoded["BIRTH_BIN"] = synthetic["BIRTH"].astype(int)
    return pd.DataFrame(decoded, index=synthetic.index)


def under_30_by_bin(cis: pd.DataFrame, binfo: dict) -> np.ndarray:
    """Share of CIS respondents under 30 within each birth-year bin."""
    edges = np.asarray(binfo["BIRTH"]["edges"], dtype=float)
    birth = cis["BIRTH"].dropna()
    bins = np.clip(np.searchsorted(edges, birth, side="right") - 1, 0, len(edges) - 2)
    young = (SURVEY_YEAR - birth) < 30
    return np.array([young[bins == b].mean() for b in range(len(edges) - 1)])


def predictors(decoded: pd.DataFrame, p_young: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Predictor matrix and a mask of individuals with all predictors defined."""
    valid = (
        decoded["SEX"].isin([1, 2])
        & decoded["ESTUDIOS"].between(1, 7)
        & decoded["V52"].isin([1, 2, 3, 4])
        & decoded["V15"].between(1, 5)
    ).to_numpy()
    young = rng.random(len(decoded)) < p_young[decoded["BIRTH_BIN"].to_numpy()]
    x = np.column_stack(
        [
            (decoded["SEX"] == 2).to_numpy(float),
            young.astype(float),
            (decoded["ESTUDIOS"] == 7).to_numpy(float),
            decoded["V52"].isin([1, 2]).to_numpy(float),
            worry_score(decoded["V15"].where(valid, 1).to_numpy(), WORRY_CODING),
        ]
    )
    return x, valid


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(SEED)
    synthetic = pd.read_csv(os.path.join(cfg.OUTDIR, "synthetic_surveys_S2.csv"))
    cis = load_cis_microdata(cfg)
    _, binfo = discretise(cis.copy(), cfg.N_BINS)

    coefficients = (
        pd.read_csv("transfer_coefficients.csv").set_index("predictor").loc[list(PREDICTORS)]
    )
    beta, se = coefficients["coefficient"].to_numpy(), coefficients["std_error"].to_numpy()
    design = load_design(cfg.PATH_CIS, WORRY_CODING)[list(PREDICTORS)].to_numpy()

    decoded = decode(synthetic, binfo)
    x, valid = predictors(decoded, under_30_by_bin(cis, binfo), rng)

    outcome = np.full(len(synthetic), np.nan)
    draws = synthetic["_draw"].to_numpy()
    for m in np.unique(draws):
        beta_m = rng.normal(beta, se)
        p0_m = rng.beta(P0 * P0_N + 0.5, (1 - P0) * P0_N + 0.5)
        intercept = calibrate_intercept(A0 * (design @ beta_m), p0_m)
        rows = (draws == m) & valid
        prob = norm.cdf(intercept + A0 * (x[rows] @ beta_m))
        outcome[rows] = (rng.random(rows.sum()) < prob).astype(float)

    final = synthetic.copy()
    final[TARGET] = outcome
    final.to_csv(os.path.join(cfg.OUTDIR, "synthetic_surveys_final.csv"), index=False)

    frame = pd.DataFrame(x, columns=PREDICTORS).assign(y=outcome, valid=valid)
    frame = frame[frame["valid"]]
    summary = {
        "n_synthetic": int(len(final)),
        "n_with_outcome": int(valid.sum()),
        "share_accepting": float(frame["y"].mean()),
        "by_predictor": {
            name: {
                "with": float(frame.loc[frame[name] == 1, "y"].mean()),
                "without": float(frame.loc[frame[name] == 0, "y"].mean()),
            }
            for name in PREDICTORS[:-1]
        },
        "a0": A0,
        "p0": P0,
        "worry_coding": WORRY_CODING,
        "seed": SEED,
    }
    with open(os.path.join(cfg.OUTDIR, "target_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
