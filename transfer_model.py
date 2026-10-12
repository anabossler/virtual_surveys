"""Transfer model for an outcome that the national survey does not measure.

Projects acceptance of recycled plastic in food packaging onto the Spanish
CIS 3391 sample from two published inputs:

  * a base rate p0 (share judging recycled plastic suitable for food
    packaging; Ruokamo et al. 2022, n = 301), treated as a population
    marginal, and
  * ordered-probit coefficients for respondent characteristics (same study,
    model 2, n = 272), listed in transfer_coefficients.csv.

Model
-----
    P(accept | x) = Phi(c + a0 * sum_j beta_j x_j)

Phi is the standard normal CDF, matching the link of the source model. The
intercept c is calibrated so that the weighted average of P(accept | x) over
the CIS respondents equals p0, using the CIS final weight PESOFIN (stored with a
decimal comma in the microdata file). The transfer factor a0 scales every coefficient.

Assumptions that the data cannot check are explicit arguments: p0, a0 and the
coding of the environmental-worry scale. Uncertainty is propagated by Monte
Carlo over the published standard errors and the binomial error of p0.

Usage
-----
    python transfer_model.py --cis metadata/3391_num.csv --out transfer_output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

SURVEY_YEAR = 2023
P0 = 0.38
P0_N = 301
A0 = 0.80
A0_GRID = (0.50, 0.70, 0.80, 0.90, 1.00)
P0_GRID = (0.25, 0.38, 0.50, 0.60)
N_DRAWS = 5000
SEED = 42
CODINGS = ("centered", "contrast", "per_step")
PREDICTORS = ("female", "under_30", "high_education", "recycles", "env_worry")

# Contrasting profiles used in the paper, as predictor values before coding
# of the worry scale (1 = not at all worried, 5 = very worried).
PROFILES = {
    "most_receptive": {"female": 1, "under_30": 1, "high_education": 1, "recycles": 1, "V15": 5},
    "least_receptive": {"female": 0, "under_30": 0, "high_education": 0, "recycles": 0, "V15": 1},
}


def worry_score(v15: np.ndarray, coding: str) -> np.ndarray:
    """Code the five-point worry scale.

    'centered': the scale is centred on its midpoint and scaled to [-1, 1], so
    the coefficient is the shift from the midpoint to either extreme and each
    scale point moves the index by half the coefficient (the transformation
    used in step2_v5_complete.py). 'contrast': the coefficient is the
    difference between the extremes (score from 0 to 1). 'per_step': the
    coefficient applies to each scale point (score from 0 to 4).
    """
    steps = np.asarray(v15, dtype=float) - 1.0
    if coding == "centered":
        return (steps - 2.0) / 2.0
    if coding == "contrast":
        return steps / 4.0
    if coding == "per_step":
        return steps
    raise ValueError(f"unknown worry coding: {coding}")


def load_design(cis_path: Path, coding: str) -> pd.DataFrame:
    """Build the predictor matrix from CIS 3391, dropping incomplete cases."""
    cis = pd.read_csv(cis_path, sep=";", encoding="latin-1", low_memory=False)
    valid = (
        cis["SEX"].isin([1, 2])
        & cis["BIRTH"].between(1900, SURVEY_YEAR)
        & cis["ESTUDIOS"].between(1, 7)
        & cis["V52"].isin([1, 2, 3, 4])
        & cis["V15"].between(1, 5)
    )
    kept = cis.loc[valid]
    design = pd.DataFrame(
        {
            "female": (kept["SEX"] == 2).astype(float),
            "under_30": ((SURVEY_YEAR - kept["BIRTH"]) < 30).astype(float),
            "high_education": (kept["ESTUDIOS"] == 7).astype(float),
            "recycles": kept["V52"].isin([1, 2]).astype(float),
            "env_worry": worry_score(kept["V15"].to_numpy(), coding),
            "V15": kept["V15"].astype(int),
            "age": SURVEY_YEAR - kept["BIRTH"],
            "weight": pd.to_numeric(kept["PESOFIN"].astype(str).str.replace(",", "."), errors="raise"),
        }
    )
    design.attrs["n_total"] = len(cis)
    return design.reset_index(drop=True)


def calibrate_intercept(linear: np.ndarray, p0: float, weights: np.ndarray | None = None) -> float:
    """Intercept c such that the weighted mean of Phi(c + linear) equals p0."""
    w = np.ones_like(linear) if weights is None else np.asarray(weights, dtype=float)
    return brentq(lambda c: np.average(norm.cdf(c + linear), weights=w) - p0, -12.0, 12.0)


def profile_vector(profile: dict, coding: str) -> np.ndarray:
    values = dict(profile)
    values["env_worry"] = float(worry_score(np.array([values.pop("V15")]), coding)[0])
    return np.array([values[name] for name in PREDICTORS], dtype=float)


def project(design: pd.DataFrame, beta: np.ndarray, p0: float, a0: float, coding: str) -> dict:
    """Respondent-level probabilities and the two contrasting profiles."""
    x = design[list(PREDICTORS)].to_numpy()
    linear = a0 * (x @ beta)
    intercept = calibrate_intercept(linear, p0, design["weight"].to_numpy())
    result = {"intercept": intercept, "respondents": norm.cdf(intercept + linear)}
    for name, profile in PROFILES.items():
        result[name] = float(norm.cdf(intercept + a0 * (profile_vector(profile, coding) @ beta)))
    result["spread"] = result["most_receptive"] - result["least_receptive"]
    return result


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def simulate(design, coefficients, coding, p0=P0, a0=A0, rng=None) -> dict:
    """Point projection plus Monte Carlo draws over published uncertainty."""
    beta = coefficients["coefficient"].to_numpy()
    se = coefficients["std_error"].to_numpy()
    point = project(design, beta, p0, a0, coding)
    draws = {"most_receptive": [], "least_receptive": [], "spread": []}
    if rng is not None:
        betas = rng.normal(beta, se, size=(N_DRAWS, len(beta)))
        p0s = rng.beta(p0 * P0_N + 0.5, (1 - p0) * P0_N + 0.5, size=N_DRAWS)
        for b, p in zip(betas, p0s):
            out = project(design, b, p, a0, coding)
            for key in draws:
                draws[key].append(out[key])
    return {"point": point, "draws": {k: np.array(v) for k, v in draws.items()}}


def profile_table(sim: dict) -> pd.DataFrame:
    rows = []
    for key in ("most_receptive", "least_receptive", "spread"):
        low, high = interval(sim["draws"][key])
        rows.append({"quantity": key, "estimate": sim["point"][key], "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def sensitivity_table(design, coefficients, coding, parameter: str, grid) -> pd.DataFrame:
    beta = coefficients["coefficient"].to_numpy()
    rows = []
    for value in grid:
        kwargs = {"p0": P0, "a0": A0, parameter: value}
        out = project(design, beta, kwargs["p0"], kwargs["a0"], coding)
        rows.append(
            {
                parameter: value,
                "population_mean": float(np.average(out["respondents"], weights=design["weight"])),
                "most_receptive": out["most_receptive"],
                "least_receptive": out["least_receptive"],
                "spread_pp": 100 * out["spread"],
            }
        )
    return pd.DataFrame(rows)


def segment_table(design: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    frame = design.assign(p=probabilities)
    total_weight = frame["weight"].sum()
    frame["age_group"] = pd.cut(frame["age"], [0, 29, 44, 59, 200], labels=["18-29", "30-44", "45-59", "60+"])
    groupings = {
        "gender": frame["female"].map({1.0: "female", 0.0: "male"}),
        "age_group": frame["age_group"].astype(str),
        "education": frame["high_education"].map({1.0: "higher", 0.0: "below higher"}),
        "recycling": frame["recycles"].map({1.0: "always or often", 0.0: "sometimes or never"}),
        "environmental_worry": frame["V15"].astype(str),
    }
    rows = []
    for dimension, labels in groupings.items():
        for level, group in frame.groupby(labels, observed=True):
            rows.append(
                {
                    "dimension": dimension,
                    "level": level,
                    "n": len(group),
                    "share_of_sample": group["weight"].sum() / total_weight,
                    "projected_acceptance": np.average(group["p"], weights=group["weight"]),
                }
            )
    return pd.DataFrame(rows)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(cumulative, 0.5)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cis", type=Path, default=Path("metadata/3391_num.csv"))
    parser.add_argument("--coefficients", type=Path, default=Path("transfer_coefficients.csv"))
    parser.add_argument("--out", type=Path, default=Path("transfer_output"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    coefficients = pd.read_csv(args.coefficients).set_index("predictor").loc[list(PREDICTORS)].reset_index()
    summary = {"p0": P0, "p0_n": P0_N, "a0": A0, "n_draws": N_DRAWS, "seed": SEED, "weights": "PESOFIN", "codings": {}}

    for coding in CODINGS:
        design = load_design(args.cis, coding)
        sim = simulate(design, coefficients, coding, rng=np.random.default_rng(SEED))
        probabilities = sim["point"]["respondents"]

        profile_table(sim).to_csv(args.out / f"profiles_{coding}.csv", index=False)
        sensitivity_table(design, coefficients, coding, "a0", A0_GRID).to_csv(
            args.out / f"sensitivity_a0_{coding}.csv", index=False
        )
        sensitivity_table(design, coefficients, coding, "p0", P0_GRID).to_csv(
            args.out / f"sensitivity_p0_{coding}.csv", index=False
        )
        segment_table(design, probabilities).to_csv(args.out / f"segments_{coding}.csv", index=False)

        weights = design["weight"].to_numpy()
        x = design[list(PREDICTORS[:-1]) + ["V15"]]
        summary["codings"][coding] = {
            "n_cis": design.attrs["n_total"],
            "n_complete": len(design),
            "intercept": sim["point"]["intercept"],
            "population_mean": float(np.average(probabilities, weights=weights)),
            "weighted_median": weighted_median(probabilities, weights),
            "share_below_0.25": float(np.average(probabilities < 0.25, weights=weights)),
            "share_above_0.50": float(np.average(probabilities > 0.50, weights=weights)),
            "n_most_receptive_profile": int((x == pd.Series(PROFILES["most_receptive"])).all(axis=1).sum()),
            "n_least_receptive_profile": int((x == pd.Series(PROFILES["least_receptive"])).all(axis=1).sum()),
        }
        print(f"[{coding}] n = {len(design)}")
        print(profile_table(sim).round(3).to_string(index=False))

    with (args.out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
