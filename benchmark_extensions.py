"""Further checks of the retrospective transfer-validity benchmark.

1. construct    Gradient of construct distance. In the cross scenarios the
                coefficients come from another item of the source country.
                Pairs of items are ordered by their respondent-level correlation
                in the source country (higher = closer constructs) and split
                into terciles; the same-item scenarios are distance zero.
                Reported per group: gap MAE, sign accuracy, transport coverage,
                equivalent target sample.

2. competitors  Realistic no-survey competitors for each gap (and level):
                  proxy_own     the target country's own observed gap for the
                                same predictor, averaged over the other four
                                items (available in the application: Spain's
                                gaps on the related CIS items)
                  generic       the mean gap of that predictor over the other
                                countries and the other four items
                  transfer      the published transfer (small cross and same)
                  blend         mean of transfer (small cross) and proxy_own
                The Spanish proxy gaps on the five ISSP items are also printed.

3. omitted      Omitted-predictor stress: the source model is refitted without
                environmental worry, or without higher education, and the gaps of
                the remaining predictors are compared with those of the full
                model on the same cases (same and small-cross scenarios; fixed
                a0 = 0.80; transport intervals with the target-excluded
                parameters).

4. outcomes     Leave-one-outcome-out recalibration: the factor k for gaps in the
                small-cross scenario is estimated on four items and applied to
                the fifth.

Usage:
    python benchmark_extensions.py --issp metadata/ZA7650_v2-0-0.dta
Inputs: transport_output/backtest_stress_cases.csv, backtest_stress_cases_loro.csv,
        audit_extensions.json
Output: transport_output/benchmark_extensions.json
"""
from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from audit_extensions import equivalent
from backtest_stress import fit, region_parameters
from backtest_transfer import EUROPE, MAX_TARGET_N, calibrate, segments, weighted_share
from transport_issp import OUTCOMES, PREDICTORS, design, probit

SEED = 31
A0 = 0.80


def wcorr(a, b, w):
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    return float(cov / np.sqrt(np.average((a - ma) ** 2, weights=w) * np.average((b - mb) ** 2, weights=w)))


def block(g: pd.DataFrame, pred: str, curve: dict) -> dict:
    err = 100 * (g[pred] - g["observed"]).abs()
    out = {"mae": round(float(err.mean()), 2), "sign": round(100 * float((np.sign(g[pred]) == np.sign(g["observed"])).mean()), 1),
           "neq": equivalent(curve, float(err.mean()), True), "cases": int(len(g))}
    if pred == "pred_fixed" and "low_transport" in g:
        out["coverage_transport"] = round(100 * float(((g.low_transport <= g.observed) & (g.observed <= g.high_transport)).mean()), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--issp", type=Path, default=Path("metadata/ZA7650_v2-0-0.dta"))
    ap.add_argument("--cases", type=Path, default=Path("transport_output/backtest_stress_cases.csv"))
    ap.add_argument("--cases-loro", type=Path, default=Path("transport_output/backtest_stress_cases_loro.csv"))
    ap.add_argument("--audit", type=Path, default=Path("transport_output/audit_extensions.json"))
    ap.add_argument("--out", type=Path, default=Path("transport_output/benchmark_extensions.json"))
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--source-n", type=int, default=272)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    preds, outcomes = list(PREDICTORS), list(OUTCOMES)
    curves = {k: {int(n): v for n, v in c.items()} for k, c in json.loads(args.audit.read_text())["curves"]["all"].items()}
    result: dict = {"seed": SEED}

    data = design(pd.read_stata(args.issp, convert_categoricals=False))
    data = data[data.country.isin(EUROPE)]
    countries = sorted(data.country.unique())
    cases = pd.read_csv(args.cases)
    gaps = cases[cases.quantity != "level"].copy()
    gaps["predictor"] = gaps["quantity"].str.replace("gap_", "", regex=False)

    # ------------------------------------------------------------ 1 construct
    corr = {}
    for c in countries:
        sub = data[data.country == c]
        for k, kc in permutations(outcomes, 2):
            s2 = sub[[k, kc, "weight"]].dropna()
            corr[(c, k, kc)] = wcorr(s2[k].to_numpy(), s2[kc].to_numpy(), s2["weight"].to_numpy())
    cr = gaps[gaps.scenario.isin(["cross", "small_cross"])].copy()
    cr["item_corr"] = [corr[(s, k, kc)] for s, k, kc in zip(cr.source, cr.outcome, cr.coef_outcome)]
    cuts = cr["item_corr"].quantile([1 / 3, 2 / 3]).to_numpy()
    cr["distance"] = np.select([cr.item_corr >= cuts[1], cr.item_corr >= cuts[0]], ["near", "middle"], "far")
    grad = {}
    for size, (same_sc, cross_sc) in {"full_source": ("same", "cross"), "small_source": ("small", "small_cross")}.items():
        grad[size] = {"same item": block(gaps[gaps.scenario == same_sc], "pred_fixed", curves["gap_all"])}
        for dist in ("near", "middle", "far"):
            g = cr[(cr.scenario == cross_sc) & (cr.distance == dist)]
            grad[size][dist] = block(g, "pred_fixed", curves["gap_all"]) | {
                "item_corr_range": [round(float(g.item_corr.min()), 2), round(float(g.item_corr.max()), 2)]}
    pair_corr = (cr.groupby(["outcome", "coef_outcome"])["item_corr"].mean().round(2)
                 .reset_index().to_dict(orient="records"))
    result["construct_gradient"] = {"tercile_cuts": [round(float(x), 3) for x in cuts], "groups": grad,
                                    "mean_item_corr_by_pair": pair_corr}
    print("construct gradient (gap MAE, sign, coverage, n_eq):")
    for size, d in grad.items():
        for name, v in d.items():
            print(f"  {size:12s} {name:9s} {v['mae']:5.1f} {v['sign']:5.1f} {v.get('coverage_transport', '')}  n_eq {v['neq']}"
                  + (f"  corr {v['item_corr_range']}" if "item_corr_range" in v else ""))

    # ---------------------------------------------------------- 2 competitors
    obs = gaps.drop_duplicates(["target", "outcome", "predictor"]).set_index(["target", "outcome", "predictor"])["observed"]
    lev = cases[cases.quantity == "level"].drop_duplicates(["target", "outcome"]).set_index(["target", "outcome"])["observed"]

    def proxy_own(t, k, j):
        v = [obs.get((t, o, j), np.nan) for o in outcomes if o != k]
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan

    def generic(t, k, j):
        v = [obs.get((c, o, j), np.nan) for c in countries if c != t for o in outcomes if o != k]
        return float(np.nanmean(v))

    comp = {}
    for sc in ("same", "small_cross"):
        g = gaps[gaps.scenario == sc].copy()
        g["pred_proxy_own"] = [proxy_own(t, k, j) for t, k, j in zip(g.target, g.outcome, g.predictor)]
        g["pred_generic"] = [generic(t, k, j) for t, k, j in zip(g.target, g.outcome, g.predictor)]
        g["pred_blend"] = (g["pred_fixed"] + g["pred_proxy_own"]) / 2
        g = g.dropna(subset=["pred_proxy_own"])
        comp[sc] = {}
        for tag, gg in (("all", g), ("ES", g[g.target == "ES"])):
            comp[sc][tag] = {name: block(gg, col, curves["gap_all"]) for name, col in
                             (("transfer", "pred_fixed"), ("proxy_own", "pred_proxy_own"),
                              ("generic", "pred_generic"), ("blend", "pred_blend"), ("no_gap", "pred_none"))}
            comp[sc][tag]["by_predictor"] = {
                j: {name: round(100 * float((gj[col] - gj.observed).abs().mean()), 2)
                    for name, col in (("transfer", "pred_fixed"), ("proxy_own", "pred_proxy_own"), ("generic", "pred_generic"))}
                for j, gj in gg.groupby("predictor")}
    lv = cases[(cases.scenario == "same") & (cases.quantity == "level")].copy()
    lv["pred_proxy_level"] = [float(np.nanmean([lev.get((t, o), np.nan) for o in outcomes if o != k]))
                              for t, k in zip(lv.target, lv.outcome)]
    comp["level_proxy_own_mae"] = round(100 * float((lv.pred_proxy_level - lv.observed).abs().mean()), 2)
    comp["spain_proxy_gaps_points"] = {j: {o: round(100 * float(obs.get(("ES", o, j), np.nan)), 1) for o in outcomes}
                                       for j in preds}
    comp["spain_proxy_gap_mean_points"] = {j: round(100 * float(np.nanmean([obs.get(("ES", o, j), np.nan) for o in outcomes])), 1)
                                           for j in preds}
    result["competitors"] = comp
    print("\ncompetitors, gap MAE (sign) [n_eq]:")
    for sc in ("same", "small_cross"):
        for tag in ("all", "ES"):
            d = comp[sc][tag]
            print(f"  {sc:11s} {tag:3s}", {n: (v["mae"], v["sign"], v["neq"]) for n, v in d.items() if n != "by_predictor"})
    print("  level MAE with target's own mean on other items:", comp["level_proxy_own_mae"])
    print("  Spanish proxy gaps, mean over the five items (points):", comp["spain_proxy_gap_mean_points"])

    # -------------------------------------------------------------- 3 omitted
    subs = {(c, k): data[data.country == c][preds + [k, "weight"]].dropna() for c in countries for k in outcomes}
    full = {key: fit(sub, key[1], None, rng) for key, sub in subs.items()}
    beta = np.full((len(countries), len(outcomes), len(preds)), np.nan)
    se, share = beta.copy(), np.full((len(countries), len(outcomes)), np.nan)
    for (c, k), f in full.items():
        if f is not None:
            i, j = countries.index(c), outcomes.index(k)
            beta[i, j], se[i, j], share[i, j] = f["beta"], f["se"], f["share"]
    params = {(t, k): region_parameters(beta, se, share, countries, countries.index(t), outcomes.index(k), False)
              for t in countries for k in outcomes}
    small_idx = {key: (np.sort(rng.choice(len(sub), args.source_n, replace=False)) if len(sub) > args.source_n else None)
                 for key, sub in subs.items()}
    cache = {}

    def src_fit(s, k, drop, small):
        key = (s, k, drop, small)
        if key not in cache:
            sub = subs[(s, k)]
            if small and small_idx[(s, k)] is not None:
                sub = sub.iloc[small_idx[(s, k)]]
            keep = [p for p in preds if p != drop]
            x, y, w = sub[keep].to_numpy(), sub[k].to_numpy(), sub["weight"].to_numpy()
            if len(sub) < 200 or x.std(axis=0).min() == 0:
                cache[key] = None
            else:
                b, s_ = probit(y, x, w)
                cache[key] = {"beta": b[1:], "se": s_[1:], "share": weighted_share(y, w)[0], "n": len(sub), "keep": keep}
        return cache[key]

    rows = []
    for sc in ("same", "small_cross"):
        small = sc == "small_cross"
        for k in outcomes:
            for s, t in permutations(countries, 2):
                tgt = subs[(t, k)]
                if full[(t, k)] is None:
                    continue
                kc = k if not small else rng.choice([o for o in outcomes if o != k])
                x, y, w = tgt[preds].to_numpy(), tgt[k].to_numpy(), tgt["weight"].to_numpy()
                idx = np.sort(rng.choice(len(y), min(len(y), MAX_TARGET_N), replace=False))
                xe, we = x[idx], w[idx]
                m_obs, m_eval = segments(x), segments(xe)
                p = params[(t, k)]
                for drop in (None, "env_worry", "high_education"):
                    src, base = src_fit(s, kc, drop, small), src_fit(s, k, drop, small)
                    if src is None or base is None:
                        continue
                    cols = [preds.index(q) for q in src["keep"]]
                    bd = rng.normal(src["beta"], src["se"], size=(args.draws, len(cols)))
                    b0 = rng.beta(base["share"] * base["n"] + 0.5, (1 - base["share"]) * base["n"] + 0.5, size=args.draws)
                    lin = xe[:, cols] @ (A0 * bd).T
                    pr_fixed = norm.cdf(calibrate(lin, we, b0) + lin)
                    a0 = rng.normal(p["a0_mean"], p["a0_sd"], size=(args.draws, 1))
                    lin = xe[:, cols] @ (a0 * bd + rng.normal(0.0, p["eta_sd"], size=bd.shape)).T
                    pr_tr = norm.cdf(calibrate(lin, we, b0) + rng.normal(0.0, p["level_sd"], size=args.draws) + lin)
                    for j in preds:
                        if j in ("env_worry", "high_education"):
                            continue  # evaluate only predictors kept in every model
                        mo, me = m_obs[j], m_eval[j]
                        if mo.sum() < 30 or (~mo).sum() < 30 or me.sum() < 10 or (~me).sum() < 10:
                            continue
                        observed = weighted_share(y[mo], w[mo])[0] - weighted_share(y[~mo], w[~mo])[0]
                        gf = np.average(pr_fixed[me], axis=0, weights=we[me]) - np.average(pr_fixed[~me], axis=0, weights=we[~me])
                        gt = np.average(pr_tr[me], axis=0, weights=we[me]) - np.average(pr_tr[~me], axis=0, weights=we[~me])
                        lo, hi = np.percentile(gt, [2.5, 97.5])
                        rows.append({"scenario": sc, "model": drop or "full", "target": t, "predictor": j,
                                     "observed": observed, "pred_fixed": float(np.median(gf)),
                                     "low_transport": float(lo), "high_transport": float(hi)})
        print(f"omitted-predictor stress: {sc} done")
    om = pd.DataFrame(rows)
    curve_rest = {n: float(np.mean([curves[f"gap_{j}"][n] for j in ("female", "under_30", "recycles")])) for n in curves["gap_all"]}
    result["omitted_predictors"] = {sc: {m: block(g, "pred_fixed", curve_rest) for m, g in om[om.scenario == sc].groupby("model")}
                                    for sc in ("same", "small_cross")}
    result["omitted_predictors"]["evaluated_predictors"] = ["female", "under_30", "recycles"]
    print("\nomitted predictors (gaps of gender, age, recycling):")
    for sc in ("same", "small_cross"):
        for m, v in result["omitted_predictors"][sc].items():
            print(f"  {sc:11s} {m:15s} MAE {v['mae']:5.1f} sign {v['sign']:5.1f} cov {v['coverage_transport']:5.1f} n_eq {v['neq']}")

    # ------------------------------------------- 4 leave-one-outcome-out k
    lo_cases = pd.read_csv(args.cases_loro)
    g = lo_cases[(lo_cases.scenario == "small_cross") & (lo_cases.quantity != "level")].dropna(subset=["low_transport"]).copy()
    g["hw"] = ((g.high_transport - g.low_transport) / 2).clip(lower=1e-4)
    g["z"] = (g.observed - g.pred_transport).abs() / g.hw
    loo = {}
    for k in outcomes:
        kk = float(np.quantile(g.loc[g.outcome != k, "z"], 0.95))
        gi = g[g.outcome == k]
        cov = float(((gi.pred_transport - kk * gi.hw <= gi.observed) & (gi.observed <= gi.pred_transport + kk * gi.hw)).mean())
        loo[k] = {"k": round(kk, 3), "coverage": round(100 * cov, 1)}
    result["leave_one_outcome_out_k"] = loo
    print("\nleave-one-outcome-out recalibration (small cross, gaps):", loo)

    args.out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("->", args.out)


if __name__ == "__main__":
    main()
