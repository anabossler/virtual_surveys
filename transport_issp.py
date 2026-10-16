"""Empirical transport error between countries from ISSP 2020 Environment IV.

The transfer model takes a base rate and probit coefficients from a Finnish
study and applies them to Spain. This script measures, on outcomes that the
ISSP asks in every country, how much a Finnish estimate differs from the
Spanish one, and how much effects vary between countries in general.

For each country c and proxy outcome k, a weighted probit is fitted with the
five predictors of the transfer model, coded as in transfer_model.py:

    P(Y_k = 1 | x) = Phi(c_ck + sum_j beta_ckj x_j)

Outputs (in --out):
    issp_coefficients.csv   beta and SE per country, outcome and predictor
    issp_levels.csv         weighted share of Y_k = 1 per country and outcome
    issp_transport.json     summary used by the manuscript:
        a0_fi_es            slope of Spanish on Finnish coefficients (through
                            the origin), pooled over outcomes and predictors,
                            with a respondent bootstrap interval
        level_shift_fi_es   probit-scale level difference Spain minus Finland
                            per outcome, and its mean and SD
        level_rms_fi_es     root mean square of those differences (level-transport
                            error used by transfer_transport.py)
        coefficient_residual_fi_es
                            Spanish minus a0 x Finnish coefficients: SD, the part
                            expected from sampling, and the excess (transport) SD
        level_sd_pairs      SD of probit-scale level differences over all
                            country pairs (generic level-transport error)
        tau_rel             between-country SD of each coefficient
                            (DerSimonian-Laird) relative to its mean, median
                            over outcomes and predictors

Data: ISSP Research Group (2023), ZA7650 v2.0.0, doi:10.4232/1.14153. Obtain
from GESIS (registration required); the file is not redistributed.

Usage
-----
    python transport_issp.py --issp ZA7650_v2-0-0.dta --out transport_output
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

PREDICTORS = ("female", "under_30", "high_education", "recycles", "env_worry")
# Proxy outcomes: pro-environmental willingness and consumer choices. Value 1
# for the two most favourable categories of each item.
OUTCOMES = {
    "pay_higher_prices": ("v26", (1, 2)),
    "pay_higher_taxes": ("v27", (1, 2)),
    "cut_living_standard": ("v28", (1, 2)),
    "do_right_despite_cost": ("v31", (1, 2)),
    "avoid_products": ("v53", (1, 2)),
}
SOURCE, TARGET = "FI", "ES"
N_BOOT = 500
SEED = 42


def design(frame: pd.DataFrame) -> pd.DataFrame:
    """Predictors coded as in transfer_model.py; non-substantive codes as missing."""
    v15 = frame["v15"].where(frame["v15"].between(1, 5))
    v52 = frame["v52"].where(frame["v52"].between(1, 4))
    out = pd.DataFrame(
        {
            "country": frame["c_alphan"].astype(str).str[:2],
            "weight": frame["WEIGHT"].astype(float),
            "female": (frame["SEX"] == 2).astype(float).where(frame["SEX"].isin([1, 2])),
            "under_30": (frame["AGE"] < 30).astype(float).where(frame["AGE"] > 0),
            "high_education": (frame["EDULEVEL"] >= 6).astype(float).where(frame["EDULEVEL"] >= 0),
            "recycles": v52.isin([1, 2]).astype(float).where(v52.notna()),
            "env_worry": (v15 - 3.0) / 2.0,
        }
    )
    for name, (var, positive) in OUTCOMES.items():
        valid = frame[var] > 0
        out[name] = frame[var].isin(positive).astype(float).where(valid)
    return out


def probit(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted probit by maximum likelihood; returns coefficients and SEs (intercept first)."""
    x1 = np.column_stack([np.ones(len(y)), x])
    w = w * len(w) / w.sum()

    def nll(b):
        eta = x1 @ b
        return -np.sum(w * (y * norm.logcdf(eta) + (1 - y) * norm.logcdf(-eta)))

    def grad(b):
        eta = x1 @ b
        lam1 = np.exp(norm.logpdf(eta) - norm.logcdf(eta))
        lam0 = -np.exp(norm.logpdf(eta) - norm.logcdf(-eta))
        return -x1.T @ (w * np.where(y == 1, lam1, lam0))

    fit = minimize(nll, np.zeros(x1.shape[1]), jac=grad, method="BFGS")
    eta = x1 @ fit.x
    pdf, cdf = norm.pdf(eta), np.clip(norm.cdf(eta), 1e-10, 1 - 1e-10)
    info = (x1 * (w * pdf**2 / (cdf * (1 - cdf)))[:, None]).T @ x1
    return fit.x, np.sqrt(np.diag(np.linalg.inv(info)))


def fit_country(data: pd.DataFrame, outcome: str):
    cols = list(PREDICTORS) + [outcome, "weight"]
    sub = data[cols].dropna()
    if len(sub) < 200 or sub[list(PREDICTORS)].std().min() == 0:
        return None
    return probit(sub[outcome].to_numpy(), sub[list(PREDICTORS)].to_numpy(), sub["weight"].to_numpy())


def slope_through_origin(target: np.ndarray, source: np.ndarray) -> float:
    return float(target @ source / (source @ source))


def dersimonian_laird(beta: np.ndarray, se: np.ndarray) -> tuple[float, float]:
    w = 1 / se**2
    mean_fe = np.sum(w * beta) / w.sum()
    q = np.sum(w * (beta - mean_fe) ** 2)
    tau2 = max(0.0, (q - (len(beta) - 1)) / (w.sum() - np.sum(w**2) / w.sum()))
    w_re = 1 / (se**2 + tau2)
    return float(np.sum(w_re * beta) / w_re.sum()), float(np.sqrt(tau2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--issp", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("transport_output"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = design(pd.read_stata(args.issp, convert_categoricals=False))
    countries = sorted(data["country"].unique())

    coef_rows, level_rows = [], []
    for country in countries:
        part = data[data["country"] == country]
        for outcome in OUTCOMES:
            valid = part[[outcome, "weight"]].dropna()
            if len(valid):
                level_rows.append({"country": country, "outcome": outcome,
                                   "share": float(np.average(valid[outcome], weights=valid["weight"]))})
            fit = fit_country(part, outcome)
            if fit is None:
                continue
            beta, se = fit
            for j, name in enumerate(PREDICTORS, start=1):
                coef_rows.append({"country": country, "outcome": outcome, "predictor": name,
                                  "beta": beta[j], "se": se[j]})
    coefs, levels = pd.DataFrame(coef_rows), pd.DataFrame(level_rows)
    coefs.to_csv(args.out / "issp_coefficients.csv", index=False)
    levels.to_csv(args.out / "issp_levels.csv", index=False)

    def pair(c):
        return coefs[coefs.country == c].set_index(["outcome", "predictor"])["beta"]

    src, tgt = pair(SOURCE), pair(TARGET)
    common = src.index.intersection(tgt.index)
    a0 = slope_through_origin(tgt[common].to_numpy(), src[common].to_numpy())

    rng = np.random.default_rng(SEED)
    parts = {c: data[data["country"] == c].reset_index(drop=True) for c in (SOURCE, TARGET)}
    boot = []
    for _ in range(N_BOOT):
        b = {}
        for c, frame in parts.items():
            sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
            vals = {}
            for outcome in OUTCOMES:
                fit = fit_country(sample, outcome)
                if fit is not None:
                    for j, name in enumerate(PREDICTORS, start=1):
                        vals[(outcome, name)] = fit[0][j]
            b[c] = pd.Series(vals)
        idx = b[SOURCE].index.intersection(b[TARGET].index)
        boot.append(slope_through_origin(b[TARGET][idx].to_numpy(), b[SOURCE][idx].to_numpy()))

    lv = levels.pivot(index="country", columns="outcome", values="share")
    probit_lv = pd.DataFrame(norm.ppf(lv.clip(0.005, 0.995)), index=lv.index, columns=lv.columns)
    shift = (probit_lv.loc[TARGET] - probit_lv.loc[SOURCE]).dropna()
    pair_diffs = [
        probit_lv.loc[a, k] - probit_lv.loc[b, k]
        for a, b in combinations(probit_lv.index, 2) for k in probit_lv.columns
        if pd.notna(probit_lv.loc[a, k]) and pd.notna(probit_lv.loc[b, k])
    ]

    se_pair = coefs[coefs.country.isin([SOURCE, TARGET])].pivot_table(
        index=["outcome", "predictor"], columns="country", values="se").loc[common]
    residual = tgt[common] - a0 * src[common]
    resid_sd = float(residual.std(ddof=1))
    sampling_sd = float(np.sqrt((se_pair[TARGET] ** 2 + a0**2 * se_pair[SOURCE] ** 2).mean()))
    coef_transport_sd = float(np.sqrt(max(0.0, resid_sd**2 - sampling_sd**2)))
    residual_by_predictor = residual.groupby(level="predictor").mean().round(4).to_dict()

    tau_rows = []
    for (outcome, predictor), g in coefs.groupby(["outcome", "predictor"]):
        mean, tau = dersimonian_laird(g["beta"].to_numpy(), g["se"].to_numpy())
        tau_rows.append({"outcome": outcome, "predictor": predictor, "mean": mean, "tau": tau,
                         "n_countries": len(g), "tau_rel": tau / abs(mean) if mean else np.nan})
    taus = pd.DataFrame(tau_rows)
    taus.to_csv(args.out / "issp_heterogeneity.csv", index=False)

    summary = {
        "source": SOURCE, "target": TARGET, "outcomes": list(OUTCOMES),
        "n_countries": len(countries),
        "a0_fi_es": {"estimate": a0, "ci_low": float(np.percentile(boot, 2.5)),
                     "ci_high": float(np.percentile(boot, 97.5)), "sd": float(np.std(boot, ddof=1)),
                     "n_pairs": int(len(common)),
                     "n_boot": N_BOOT},
        "level_shift_fi_es": {"per_outcome": shift.round(4).to_dict(),
                              "mean": float(shift.mean()), "sd": float(shift.std(ddof=1))},
        "level_rms_fi_es": float(np.sqrt(np.mean(shift.to_numpy() ** 2))),
        "level_sd_pairs": float(np.std(pair_diffs, ddof=1)),
        "coefficient_residual_fi_es": {"sd": resid_sd, "sampling_sd": sampling_sd,
                                       "transport_sd": coef_transport_sd,
                                       "mean_by_predictor": residual_by_predictor},
        "tau_rel_median": float(taus["tau_rel"].median()),
        "tau_median": float(taus["tau"].median()),
    }
    with (args.out / "issp_transport.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
