"""Stress tests of the retrospective backtest under conditions closer to the target case.

The published backtest (backtest_transfer.py) transfers between countries
that measured the same outcome with large samples. The Spanish projection
differs in two ways: its coefficients come from a small source (Finland,
n = 272) and from a different item (attractiveness) than the one projected
(suitability for food packaging), whose base rate is used. This script
repeats the backtest under four scenarios:

    same          as published: same outcome, full source sample
    small         source refitted on a random subsample of --source-n respondents
    cross         coefficients from another outcome k' of the source, base rate
                  from the projected outcome k (as attractiveness -> suitability)
    small_cross   both

For each scenario it reports level and gap errors, coverage, interval score,
CRPS and sign accuracy (with 'always positive' and 'no gap' baselines), with
target-country cluster-bootstrap intervals, and the same figures by outcome
and by whether source and target belong to the same European region.

Transport parameters are those of backtest_transfer.py (estimated without the
target country and outcome, same-outcome pairs only), so the 'cross' scenarios
test whether intervals calibrated on same-outcome transfers cover a change of
construct.

Usage: python backtest_stress.py --issp metadata/ZA7650_v2-0-0.dta
Outputs (transport_output): backtest_stress_cases.csv, backtest_stress_summary.json
"""
from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest_diagnostics import cluster_bootstrap, metrics
from backtest_transfer import A0_FIXED, EUROPE, MAX_TARGET_N, calibrate, segments, transport_parameters, weighted_share
from transport_issp import OUTCOMES, PREDICTORS, design, probit

SEED = 7
REGION = {**{c: "nordic" for c in ("DK", "FI", "IS", "NO", "SE")},
          **{c: "western" for c in ("AT", "CH", "DE", "FR")},
          **{c: "southern" for c in ("ES", "IT")},
          **{c: "central_eastern" for c in ("HR", "HU", "LT", "SI", "SK")}}
SCENARIOS = {"same": (None, False), "small": ("n", False), "cross": (None, True), "small_cross": ("n", True)}


def fit(sub: pd.DataFrame, k: str, n: int | None, rng) -> dict | None:
    if n is not None and len(sub) > n:
        sub = sub.iloc[np.sort(rng.choice(len(sub), n, replace=False))]
    x, y, w = sub[list(PREDICTORS)].to_numpy(), sub[k].to_numpy(), sub["weight"].to_numpy()
    if len(sub) < 200 or x.std(axis=0).min() == 0:
        return None
    b, se = probit(y, x, w)
    return {"beta": b[1:], "se": se[1:], "share": weighted_share(y, w)[0], "n": len(sub)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--issp", type=Path, default=Path("metadata/ZA7650_v2-0-0.dta"))
    ap.add_argument("--out", type=Path, default=Path("transport_output"))
    ap.add_argument("--source-n", type=int, default=272)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    preds, outcomes = list(PREDICTORS), list(OUTCOMES)

    data = design(pd.read_stata(args.issp, convert_categoricals=False))
    data = data[data.country.isin(EUROPE)]
    countries = sorted(data.country.unique())
    subs = {(c, k): data[data.country == c][preds + [k, "weight"]].dropna() for c in countries for k in outcomes}

    full = {key: fit(sub, key[1], None, rng) for key, sub in subs.items()}
    small = {key: fit(sub, key[1], args.source_n, rng) for key, sub in subs.items()}
    beta = np.full((len(countries), len(outcomes), len(preds)), np.nan)
    se, share = beta.copy(), np.full((len(countries), len(outcomes)), np.nan)
    for (c, k), f in full.items():
        if f is not None:
            i, j = countries.index(c), outcomes.index(k)
            beta[i, j], se[i, j], share[i, j] = f["beta"], f["se"], f["share"]
    params = {(t, k): transport_parameters(beta, se, share, countries.index(t), outcomes.index(k))
              for t in countries for k in outcomes}

    rows = []
    for scenario, (size, cross) in SCENARIOS.items():
        fits = small if size else full
        for k in outcomes:
            for s, t in permutations(countries, 2):
                tgt, base = subs[(t, k)], fits[(s, k)]
                if full[(t, k)] is None or base is None:
                    continue
                x, y, w = tgt[preds].to_numpy(), tgt[k].to_numpy(), tgt["weight"].to_numpy()
                idx = np.sort(rng.choice(len(y), min(len(y), MAX_TARGET_N), replace=False))
                xe, we = x[idx], w[idx]
                m_obs, m_eval = segments(x), segments(xe)
                others = [o for o in outcomes if o != k] if cross else [k]
                for kc in others:
                    src = fits[(s, kc)]
                    if src is None:
                        continue
                    p, d = params[(t, k)], args.draws
                    bd = rng.normal(src["beta"], src["se"], size=(d, len(preds)))
                    b0 = rng.beta(base["share"] * base["n"] + 0.5, (1 - base["share"]) * base["n"] + 0.5, size=d)
                    lin = xe @ (A0_FIXED * bd).T
                    pr_fixed = norm.cdf(calibrate(lin, we, b0) + lin)
                    a0 = rng.normal(p["a0_mean"], p["a0_sd"], size=(d, 1))
                    lin = xe @ (a0 * bd + rng.normal(0.0, p["eta_sd"], size=bd.shape)).T
                    pr_tr = norm.cdf(calibrate(lin, we, b0) + rng.normal(0.0, p["level_sd"], size=d) + lin)
                    lin_u = xe @ bd.mean(axis=0)
                    pr_u = norm.cdf(calibrate(lin_u[:, None], we, np.array([base["share"]])) + lin_u[:, None])[:, 0]
                    common = {"scenario": scenario, "source": s, "target": t, "outcome": k, "coef_outcome": kc,
                              "same_region": REGION[s] == REGION[t]}
                    obs, obs_se = weighted_share(y, w)
                    row = common | {"quantity": "level", "observed": obs, "observed_se": obs_se,
                                    "pred_copy": base["share"],
                                    "pred_other_mean": float(np.nanmean([full[(c, k)]["share"] for c in countries
                                                                         if c != t and full[(c, k)] is not None]))}
                    for v, pr in (("fixed", pr_fixed), ("transport", pr_tr)):
                        lv = np.average(pr, axis=0, weights=we)
                        row |= {f"pred_{v}": float(np.median(lv)), f"low_{v}": float(np.percentile(lv, 2.5)),
                                f"high_{v}": float(np.percentile(lv, 97.5))}
                    rows.append(row)
                    for j in preds:
                        mo, me = m_obs[j], m_eval[j]
                        if mo.sum() < 30 or (~mo).sum() < 30 or me.sum() < 10 or (~me).sum() < 10:
                            continue
                        g1, s1 = weighted_share(y[mo], w[mo])
                        g0, s0 = weighted_share(y[~mo], w[~mo])
                        row = common | {"quantity": f"gap_{j}", "observed": g1 - g0, "observed_se": float(np.hypot(s1, s0)),
                                        "pred_none": 0.0,
                                        "pred_unshrunk": float(np.average(pr_u[me], weights=we[me]) - np.average(pr_u[~me], weights=we[~me]))}
                        for v, pr in (("fixed", pr_fixed), ("transport", pr_tr)):
                            gp = np.average(pr[me], axis=0, weights=we[me]) - np.average(pr[~me], axis=0, weights=we[~me])
                            row |= {f"pred_{v}": float(np.median(gp)), f"low_{v}": float(np.percentile(gp, 2.5)),
                                    f"high_{v}": float(np.percentile(gp, 97.5))}
                        rows.append(row)
        print(f"scenario {scenario} done")

    cases = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.out / "backtest_stress_cases.csv", index=False)
    summary = {"source_n": args.source_n, "draws": args.draws, "seed": SEED, "regions": REGION, "scenarios": {}}
    for scenario, g in cases.groupby("scenario", sort=False):
        summary["scenarios"][scenario] = {
            "all": metrics(g), "ci95": cluster_bootstrap(g, args.boot, rng),
            "by_outcome": {k: metrics(gk) for k, gk in g.groupby("outcome")},
            "by_region": {("same" if r else "different"): metrics(gr) for r, gr in g.groupby("same_region")},
        }
    (args.out / "backtest_stress_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    keys = [("level", "mae_fixed"), ("level", "coverage_transport"), ("level", "crps_transport"),
            ("gap", "mae_fixed"), ("gap", "mae_none"), ("gap", "coverage_fixed"), ("gap", "coverage_transport"),
            ("gap", "crps_transport"), ("gap", "sign_all_fixed"), ("gap", "sign_all_always_positive")]
    table = pd.DataFrame({sc: {f"{q}:{m}": v["all"][q][m] for q, m in keys} for sc, v in summary["scenarios"].items()})
    print(table.round(1).to_string())
    region = {sc: {f"gap mae {r}": v["by_region"][r]["gap"]["mae_fixed"] for r in v["by_region"]}
              for sc, v in summary["scenarios"].items()}
    print(pd.DataFrame(region).round(1).to_string())
    print("->", args.out / "backtest_stress_summary.json")


if __name__ == "__main__":
    main()
