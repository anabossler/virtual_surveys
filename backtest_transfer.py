"""Retrospective test of the transfer procedure on outcomes observed in every country.

The transfer model predicts an outcome that Spain has not measured from a base
rate and probit coefficients estimated in another country (Finland). This
script applies the same procedure to the five ISSP 2020 proxy outcomes of
transport_issp.py, which every country does measure, so that predictions can
be compared with observed values.

For every ordered pair of countries (source s, target t) and outcome k, the
target's own value is treated as unknown and predicted from the source:

    P(Y = 1 | x) = Phi(c + sum_j b_j x_j),  c set so that the weighted mean over
                   the target's respondents equals the base rate

Two variants are compared:
    fixed       the procedure of the main analysis: b = a0 * beta_s with
                a0 = 0.80, beta_s drawn from its sampling distribution, base
                rate drawn from Beta around the source share
    transport   as transfer_transport.py: a0 drawn, per-coefficient noise and
                a probit-scale level shift added. The transport parameters are
                estimated without the target country and without outcome k
                (leave-target-and-outcome-out), so no information about the
                predicted value enters its own prediction

Quantities predicted for the target:
    level       weighted share of Y = 1
    gap_j       share among x_j = 1 minus share among x_j = 0, for each predictor
                (worry: above versus at or below the scale midpoint)

Baselines for the level: the source share copied as is, and the mean share of
all other countries. Baselines for the gaps: no gap, and the source gaps
applied without shrinkage (a0 = 1).

Outputs (in --out):
    backtest_cases_<pool>.csv      one row per source, target, outcome and quantity:
                            observed, predictions, intervals, observed SE
    backtest_summary_<pool>.json   errors and 95% interval coverage by variant,
                            over all pairs of the pool and for Finland to Spain
    backtest_transport_parameters_<pool>.csv
                            transport parameters used for each target and outcome

Usage
-----
    python backtest_transfer.py --issp metadata/ZA7650_v2-0-0.dta --out transport_output --pool europe
    python backtest_transfer.py --issp metadata/ZA7650_v2-0-0.dta --out transport_output --pool all
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from transport_issp import OUTCOMES, PREDICTORS, design, fit_country

A0_FIXED = 0.80
N_DRAWS = 400
MAX_TARGET_N = 1200
SEED = 42
EUROPE = {"AT", "CH", "DE", "DK", "ES", "FI", "FR", "HR", "HU", "IS", "IT", "LT", "NO", "SE", "SI", "SK"}


def calibrate(linear: np.ndarray, weights: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Intercepts c (one per draw) with weighted mean of Phi(c + linear[:, d]) equal to target[d].

    Vectorised Newton iterations from the probit approximation; linear is n x draws.
    """
    w = weights / weights.sum()
    c = norm.ppf(np.clip(target, 1e-6, 1 - 1e-6)) - w @ linear
    for _ in range(30):
        eta = c + linear
        f = w @ norm.cdf(eta) - target
        g = w @ norm.pdf(eta)
        step = f / np.maximum(g, 1e-12)
        c = c - np.clip(step, -2.0, 2.0)
        if np.max(np.abs(f)) < 1e-7:
            break
    return c


def weighted_share(y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted share and its standard error with Kish effective sample size."""
    p = float(np.average(y, weights=w))
    n_eff = w.sum() ** 2 / np.sum(w**2)
    return p, float(np.sqrt(p * (1 - p) / n_eff))


def segments(x: np.ndarray) -> dict[str, np.ndarray]:
    masks = {}
    for j, name in enumerate(PREDICTORS):
        on = x[:, j] > 0 if name == "env_worry" else x[:, j] == 1
        masks[name] = on
    return masks


def transport_parameters(beta: np.ndarray, se: np.ndarray, shares: np.ndarray,
                         t: int, k: int) -> dict:
    """a0 mean and SD, coefficient noise SD and level SD from pairs without country t and outcome k.

    beta, se: countries x outcomes x predictors (NaN where not estimated);
    shares: countries x outcomes.
    """
    n_c, n_o, _ = beta.shape
    keep_c = [c for c in range(n_c) if c != t]
    keep_o = [o for o in range(n_o) if o != k]
    slopes, excess = [], []
    for a, b in permutations(keep_c, 2):
        xa, xb = beta[a, keep_o].ravel(), beta[b, keep_o].ravel()
        sa, sb = se[a, keep_o].ravel(), se[b, keep_o].ravel()
        ok = ~(np.isnan(xa) | np.isnan(xb))
        if ok.sum() < 3:
            continue
        xa, xb, sa, sb = xa[ok], xb[ok], sa[ok], sb[ok]
        slope = float(xb @ xa / (xa @ xa))
        resid = xb - slope * xa
        slopes.append(slope)
        excess.append(resid.var(ddof=1) - np.mean(sb**2 + slope**2 * sa**2))
    z = norm.ppf(np.clip(shares[np.ix_(keep_c, keep_o)], 0.005, 0.995))
    diffs = (z[:, None, :] - z[None, :, :])
    mask = ~np.eye(len(keep_c), dtype=bool)[:, :, None] & ~np.isnan(diffs)
    return {
        "a0_mean": float(np.mean(slopes)), "a0_sd": float(np.std(slopes, ddof=1)),
        "eta_sd": float(np.sqrt(max(0.0, np.mean(excess)))),
        "level_sd": float(np.sqrt(np.mean(diffs[mask] ** 2))),
    }


def summarise(cases: pd.DataFrame) -> dict:
    out = {}
    for quantity, group in (("level", cases[cases.quantity == "level"]),
                            ("gap", cases[cases.quantity != "level"])):
        g = group
        res = {"cases": int(len(g)), "observed_se_median": float(g["observed_se"].median())}
        for col in [c for c in g.columns if c.startswith("pred_")]:
            err = (g[col] - g["observed"]).abs()
            res[f"mae_{col[5:]}"] = float(err.mean())
            res[f"medae_{col[5:]}"] = float(err.median())
        for variant in ("fixed", "transport"):
            inside = (g["observed"] >= g[f"low_{variant}"]) & (g["observed"] <= g[f"high_{variant}"])
            res[f"coverage95_{variant}"] = float(inside.mean())
            res[f"width95_{variant}"] = float((g[f"high_{variant}"] - g[f"low_{variant}"]).median())
        if quantity == "gap":
            nonzero = g[g["observed"].abs() > 2 * g["observed_se"]]
            res["sign_cases"] = int(len(nonzero))
            for variant in ("fixed", "transport"):
                res[f"sign_correct_{variant}"] = float((np.sign(nonzero[f"pred_{variant}"]) == np.sign(nonzero["observed"])).mean())
        out[quantity] = res
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--issp", type=Path, default=Path("metadata/ZA7650_v2-0-0.dta"))
    parser.add_argument("--out", type=Path, default=Path("transport_output"))
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--pool", choices=("europe", "all"), default="europe",
                        help="countries used as sources, targets and for the transport parameters")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    data = design(pd.read_stata(args.issp, convert_categoricals=False))
    countries = sorted(data["country"].unique())
    if args.pool == "europe":
        countries = [c for c in countries if c in EUROPE]
        data = data[data.country.isin(countries)]
    preds = list(PREDICTORS)

    # Per country and outcome: complete cases, probit fit, share
    fits = {}
    for c in countries:
        part = data[data.country == c]
        for k in OUTCOMES:
            sub = part[preds + [k, "weight"]].dropna()
            fit = fit_country(part, k)
            if fit is None:
                continue
            share, se_share = weighted_share(sub[k].to_numpy(), sub["weight"].to_numpy())
            fits[(c, k)] = {"beta": fit[0][1:], "se": fit[1][1:], "share": share, "n": len(sub),
                            "x": sub[preds].to_numpy(), "y": sub[k].to_numpy(), "w": sub["weight"].to_numpy()}
    outcomes = list(OUTCOMES)
    beta_arr = np.full((len(countries), len(outcomes), len(preds)), np.nan)
    se_arr, share_arr = beta_arr.copy(), np.full((len(countries), len(outcomes)), np.nan)
    for (c, k), f in fits.items():
        i, j = countries.index(c), outcomes.index(k)
        beta_arr[i, j], se_arr[i, j], share_arr[i, j] = f["beta"], f["se"], f["share"]
    params = {(t, k): transport_parameters(beta_arr, se_arr, share_arr, countries.index(t), outcomes.index(k))
              for t in countries for k in outcomes}
    pd.DataFrame([{"target": t, "outcome": k, **v} for (t, k), v in params.items()]).to_csv(
        args.out / f"backtest_transport_parameters_{args.pool}.csv", index=False)

    rows = []
    for k in OUTCOMES:
        shares = {c: fits[(c, k)]["share"] for c in countries if (c, k) in fits}
        for s, t in permutations(countries, 2):
            if (s, k) not in fits or (t, k) not in fits:
                continue
            src, tgt = fits[(s, k)], fits[(t, k)]
            x, y, w = tgt["x"], tgt["y"], tgt["w"]
            if len(y) > MAX_TARGET_N:
                idx = np.sort(rng.choice(len(y), MAX_TARGET_N, replace=False))
                x_eval, w_eval = x[idx], w[idx]
            else:
                x_eval, w_eval = x, w
            masks_obs, masks_eval = segments(x), segments(x_eval)
            p = params[(t, k)]
            d = args.draws
            beta_draw = rng.normal(src["beta"], src["se"], size=(d, len(preds)))
            base = rng.beta(src["share"] * src["n"] + 0.5, (1 - src["share"]) * src["n"] + 0.5, size=d)

            variants = {}
            b_fixed = A0_FIXED * beta_draw
            lin = x_eval @ b_fixed.T
            prob = norm.cdf(calibrate(lin, w_eval, base) + lin)
            variants["fixed"] = prob
            a0 = rng.normal(p["a0_mean"], p["a0_sd"], size=(d, 1))
            b_tr = a0 * beta_draw + rng.normal(0.0, p["eta_sd"], size=(d, len(preds)))
            lin = x_eval @ b_tr.T
            prob = norm.cdf(calibrate(lin, w_eval, base) + rng.normal(0.0, p["level_sd"], size=d) + lin)
            variants["transport"] = prob
            lin_full = x_eval @ beta_draw.mean(axis=0)
            c_full = calibrate(lin_full[:, None], w_eval, np.array([src["share"]]))
            prob_full = norm.cdf(c_full + lin_full[:, None])[:, 0]

            def level(pr):
                return np.average(pr, axis=0, weights=w_eval)

            def gap(pr, m):
                return (np.average(pr[m], axis=0, weights=w_eval[m])
                        - np.average(pr[~m], axis=0, weights=w_eval[~m]))

            obs, obs_se = weighted_share(y, w)
            row = {"source": s, "target": t, "outcome": k, "quantity": "level",
                   "observed": obs, "observed_se": obs_se,
                   "pred_copy": src["share"],
                   "pred_other_mean": float(np.mean([v for c, v in shares.items() if c != t]))}
            for name, pr in variants.items():
                lv = level(pr)
                row |= {f"pred_{name}": float(np.median(lv)),
                        f"low_{name}": float(np.percentile(lv, 2.5)),
                        f"high_{name}": float(np.percentile(lv, 97.5))}
            rows.append(row)
            for name in preds:
                m_obs, m_eval = masks_obs[name], masks_eval[name]
                if m_obs.sum() < 30 or (~m_obs).sum() < 30 or m_eval.sum() < 10 or (~m_eval).sum() < 10:
                    continue
                g1, se1 = weighted_share(y[m_obs], w[m_obs])
                g0, se0 = weighted_share(y[~m_obs], w[~m_obs])
                row = {"source": s, "target": t, "outcome": k, "quantity": f"gap_{name}",
                       "observed": g1 - g0, "observed_se": float(np.hypot(se1, se0)),
                       "pred_none": 0.0,
                       "pred_unshrunk": float(gap(prob_full[:, None], m_eval)[0])}
                for vname, pr in variants.items():
                    gp = gap(pr, m_eval)
                    row |= {f"pred_{vname}": float(np.median(gp)),
                            f"low_{vname}": float(np.percentile(gp, 2.5)),
                            f"high_{vname}": float(np.percentile(gp, 97.5))}
                rows.append(row)

    cases = pd.DataFrame(rows)
    cases.to_csv(args.out / f"backtest_cases_{args.pool}.csv", index=False)
    fi_es = cases[(cases.source == "FI") & (cases.target == "ES")]
    summary = {
        "pool": args.pool, "draws": args.draws, "a0_fixed": A0_FIXED,
        "countries": countries,
        "pairs": summarise(cases),
        "fi_to_es": summarise(fi_es),
        "fi_to_es_levels": fi_es[fi_es.quantity == "level"][
            ["outcome", "observed", "pred_copy", "pred_fixed", "low_fixed", "high_fixed",
             "pred_transport", "low_transport", "high_transport"]].round(4).to_dict("records"),
    }
    (args.out / f"backtest_summary_{args.pool}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
