"""Extensions of the retrospective audit requested in review.

Five analyses, all on ISSP 2020 Environment IV (16 European countries, five
proxy outcomes) and the case file written by backtest_stress.py:

1. levels   Level estimators the published backtest did not test:
              copy        source share (= published transfer, calibrated to p0)
              other_mean  mean of the other countries (probit scale)
              composition source probit with its own intercept applied to the
                          target's respondents (covariate-shift transportability)
              ce_source   source share shifted by the target-minus-source probit
                          difference on the other four outcomes (country effect)
              ce_pool     mean of other countries shifted by the target's
                          country effect on the other four outcomes
              comp_ce     composition plus the residual country effect left after
                          composition on the other four outcomes
            For the Spanish application, the Finland-to-Spain country effect is
            also applied to the Finnish base rate p0 = 0.38.

2. curves   Error of a simple random sample of n target respondents against
            the disjoint remainder, n = 10, 25, 50, 100, 200, 400: levels, each
            predictor's gap, and the ranking of the five gaps (Spearman rank
            correlation with the reference ranking; whether the largest gap is
            identified). A gap whose segment is empty in the sample is set to 0.

3. by-predictor and ranking equivalents
            Transfer errors by predictor and transfer ranking accuracy for each
            stress scenario (from backtest_stress_cases.csv), converted to
            equivalent target samples by interpolation on the curves of (2):
            log-log for errors, linear in log n for ranking accuracy.

4. Spain    The same equivalents with Spain as the only target (15 sources).

5. regions  Heterogeneity between Spanish regions on the five proxies: SD of
            region levels and gaps net of sampling variance, for regions with at
            least --min-region respondents, compared with the SD between the 16
            European countries.

Usage:
    python audit_extensions.py --issp metadata/ZA7650_v2-0-0.dta \
        --cases transport_output/backtest_stress_cases.csv
Output: transport_output/audit_extensions.json (and a printed summary)
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from backtest_transfer import EUROPE, segments, weighted_share
from transport_issp import OUTCOMES, PREDICTORS, design, probit

SEED = 11
SIZES = (10, 25, 50, 100, 200, 400)
P0, FI_ES = 0.38, ("FI", "ES")


def z(p):
    return norm.ppf(np.clip(p, 0.005, 0.995))


def gaps(y, w, masks):
    out = {}
    for j in PREDICTORS:
        m = masks[j]
        if m.sum() == 0 or (~m).sum() == 0:
            out[j] = 0.0
        else:
            out[j] = float(np.average(y[m], weights=w[m]) - np.average(y[~m], weights=w[~m]))
    return out


def top1(pred: dict, obs: dict) -> float:
    return float(max(pred, key=pred.get) == max(obs, key=obs.get))


def rank_corr(pred: dict, obs: dict) -> float:
    a = [pred[j] for j in PREDICTORS]
    b = [obs[j] for j in PREDICTORS]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(spearmanr(a, b).statistic)


def equivalent(curve: dict, value: float, decreasing: bool) -> str | float:
    """n at which the curve reaches value; log-log for errors, lin-log for accuracy."""
    ns = np.array(sorted(curve)); ys = np.array([curve[n] for n in ns])
    if decreasing:
        if value >= ys[0]:
            return f"<{ns[0]}"
        if value <= ys[-1]:
            return f">{ns[-1]}"
        lx, ly = np.log(ns), np.log(ys)
        i = np.searchsorted(-ly, -np.log(value))
        f = (np.log(value) - ly[i - 1]) / (ly[i] - ly[i - 1])
        return round(float(np.exp(lx[i - 1] + f * (lx[i] - lx[i - 1]))))
    if value <= ys[0]:
        return f"<{ns[0]}"
    if value >= ys[-1]:
        return f">{ns[-1]}"
    i = int(np.searchsorted(ys, value))
    f = (value - ys[i - 1]) / (ys[i] - ys[i - 1])
    return round(float(np.exp(np.log(ns[i - 1]) + f * (np.log(ns[i]) - np.log(ns[i - 1])))))


def find_region_column(raw: pd.DataFrame, override: str | None) -> str | None:
    if override:
        return override
    for col in raw.columns:
        if re.fullmatch(r"es_reg", col, flags=re.I):
            return col
    cands = [c for c in raw.columns if re.match(r"es_", c, flags=re.I) and "reg" in c.lower()]
    return cands[0] if cands else None


def between_sd(est: np.ndarray, se: np.ndarray) -> float:
    """SD of true values net of sampling variance (method of moments), in points."""
    v = np.var(est, ddof=1) - np.mean(se**2)
    return 100 * float(np.sqrt(max(v, 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--issp", type=Path, default=Path("metadata/ZA7650_v2-0-0.dta"))
    ap.add_argument("--cases", type=Path, default=Path("transport_output/backtest_stress_cases.csv"))
    ap.add_argument("--out", type=Path, default=Path("transport_output/audit_extensions.json"))
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--region-var", default=None, help="ISSP variable with the Spanish region (auto-detected)")
    ap.add_argument("--min-region", type=int, default=100)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    preds, outcomes = list(PREDICTORS), list(OUTCOMES)
    result: dict = {"seed": SEED, "sizes": SIZES}

    raw = pd.read_stata(args.issp, convert_categoricals=False)
    data = design(raw)
    keep = data.country.isin(EUROPE)
    data, raw = data[keep], raw[keep]
    countries = sorted(data.country.unique())
    subs = {(c, k): data[data.country == c][preds + [k, "weight"]].dropna() for c in countries for k in outcomes}

    # ---------------------------------------------------------------- 1 levels
    fits, share = {}, {}
    for (c, k), sub in subs.items():
        y, w = sub[k].to_numpy(), sub["weight"].to_numpy()
        share[(c, k)] = weighted_share(y, w)[0]
        if len(sub) >= 200 and sub[preds].std().min() > 0:
            fits[(c, k)] = probit(y, sub[preds].to_numpy(), w)[0]

    def composition(s, t, k):
        if (s, k) not in fits:
            return np.nan
        sub = subs[(t, k)]
        b = fits[(s, k)]
        return float(np.average(norm.cdf(b[0] + sub[preds].to_numpy() @ b[1:]), weights=sub["weight"]))

    comp = {(s, t, k): composition(s, t, k) for s, t in permutations(countries, 2) for k in outcomes}
    rows = []
    for s, t in permutations(countries, 2):
        for k in outcomes:
            oo = [o for o in outcomes if o != k]
            others = [c for c in countries if c != t]
            obs = share[(t, k)]
            ce = np.mean([z(share[(t, o)]) - z(share[(s, o)]) for o in oo])
            pool_k = np.mean([z(share[(c, k)]) for c in others])
            ce_pool = np.mean([z(share[(t, o)]) - np.mean([z(share[(c, o)]) for c in others]) for o in oo])
            resid = np.nanmean([z(share[(t, o)]) - z(comp[(s, t, o)]) for o in oo])
            rows.append({"source": s, "target": t, "outcome": k, "observed": obs,
                         "copy": share[(s, k)], "other_mean": norm.cdf(pool_k),
                         "composition": comp[(s, t, k)],
                         "ce_source": norm.cdf(z(share[(s, k)]) + ce),
                         "ce_pool": norm.cdf(pool_k + ce_pool),
                         "comp_ce": norm.cdf(z(comp[(s, t, k)]) + resid)})
    lev = pd.DataFrame(rows)
    est = ["copy", "other_mean", "composition", "ce_source", "ce_pool", "comp_ce"]
    mae = lambda g: {e: round(100 * float((g[e] - g["observed"]).abs().mean()), 2) for e in est}
    # share of the source-target level gap explained by composition
    expl = 1 - ((lev["composition"] - lev["observed"]) ** 2).mean() / ((lev["copy"] - lev["observed"]) ** 2).mean()
    fi_es_shift = float(np.mean([z(share[("ES", o)]) - z(share[("FI", o)]) for o in outcomes]))
    result["levels"] = {
        "mae_all_pairs": mae(lev), "mae_target_ES": mae(lev[lev.target == "ES"]),
        "mae_FI_to_ES": mae(lev[(lev.source == "FI") & (lev.target == "ES")]),
        "by_outcome": {k: mae(g) for k, g in lev.groupby("outcome")},
        "composition_share_of_squared_level_error_removed": round(float(expl), 3),
        "application": {"fi_es_mean_probit_shift": round(fi_es_shift, 4),
                        "p0_country_effect_adjusted": round(float(norm.cdf(z(P0) + fi_es_shift)), 4),
                        "per_outcome_shift": {o: round(float(z(share[("ES", o)]) - z(share[("FI", o)])), 4)
                                              for o in outcomes}},
    }
    print("level MAE (points):", result["levels"]["mae_all_pairs"])
    print("  Spain as target:", result["levels"]["mae_target_ES"])
    print("  Finland -> Spain:", result["levels"]["mae_FI_to_ES"])
    print("  composition removes", round(100 * expl, 1), "% of squared level error")
    print("  p0 adjusted by FI-ES country effect:", result["levels"]["application"]["p0_country_effect_adjusted"])

    # ---------------------------------------------------------------- 2 curves
    crow = []
    for t in countries:
        for k in outcomes:
            sub = subs[(t, k)]
            x, y, w = sub[preds].to_numpy(), sub[k].to_numpy(), sub["weight"].to_numpy()
            for n in SIZES:
                if n >= len(y) - 100:
                    continue
                for _ in range(args.draws):
                    idx = rng.choice(len(y), n, replace=False)
                    rest = np.setdiff1d(np.arange(len(y)), idx)
                    ms, mr = segments(x[idx]), segments(x[rest])
                    gs, gr = gaps(y[idx], w[idx], ms), gaps(y[rest], w[rest], mr)
                    rec = {"target": t, "outcome": k, "n": n,
                           "level": abs(np.average(y[idx], weights=w[idx]) - np.average(y[rest], weights=w[rest])),
                           "rank_corr": rank_corr(gs, gr), "top1": top1(gs, gr)}
                    rec |= {f"gap_{j}": abs(gs[j] - gr[j]) for j in preds}
                    crow.append(rec)
    cur = pd.DataFrame(crow)
    cols = ["level"] + [f"gap_{j}" for j in preds] + ["rank_corr", "top1"]

    def curves(frame):
        g = frame.groupby("n")[cols].mean()
        g[[c for c in cols if c not in ("rank_corr", "top1")]] *= 100
        g["gap_all"] = g[[f"gap_{j}" for j in preds]].mean(axis=1)
        return {c: {int(n): round(float(v), 3) for n, v in g[c].items()} for c in g.columns}

    cv_all, cv_es = curves(cur), curves(cur[cur.target == "ES"])
    result["curves"] = {"all": cv_all, "ES": cv_es}
    print("SRS curve, all targets:\n", pd.DataFrame(cv_all).round(2).to_string())

    # ------------------------------------------- 3,4 by predictor, ranking, Spain
    cases = pd.read_csv(args.cases)
    gapc = cases[cases.quantity != "level"].copy()
    gapc["predictor"] = gapc["quantity"].str.replace("gap_", "", regex=False)
    key = ["scenario", "source", "target", "outcome", "coef_outcome"]
    rank_rows = []
    for kk, g in gapc.groupby(key):
        if g["predictor"].nunique() < len(preds):
            continue
        pr = dict(zip(g.predictor, g.pred_fixed)); ob = dict(zip(g.predictor, g.observed))
        rank_rows.append(dict(zip(key, kk)) | {"rank_corr": rank_corr(pr, ob), "top1": top1(pr, ob)})
    rk = pd.DataFrame(rank_rows)

    def scenario_block(gc, rkc, lc, cv):
        out = {}
        for sc, g in gc.groupby("scenario", sort=False):
            byp = {j: 100 * float((gj.pred_fixed - gj.observed).abs().mean()) for j, gj in g.groupby("predictor")}
            r = rkc[rkc.scenario == sc]
            lv = lc[lc.scenario == sc]
            lv_mae = 100 * float((lv.pred_fixed - lv.observed).abs().mean())
            gap_mae = 100 * float((g.pred_fixed - g.observed).abs().mean())
            out[sc] = {
                "level_mae": round(lv_mae, 2), "level_neq": equivalent(cv["level"], lv_mae, True),
                "gap_mae": round(gap_mae, 2), "gap_neq": equivalent(cv["gap_all"], gap_mae, True),
                "by_predictor": {j: {"mae": round(v, 2), "neq": equivalent(cv[f"gap_{j}"], v, True)}
                                 for j, v in byp.items()},
                "ranking": {"rank_corr": round(float(r.rank_corr.mean()), 3),
                            "rank_corr_neq": equivalent(cv["rank_corr"], float(r.rank_corr.mean()), False),
                            "top1": round(float(r.top1.mean()), 3),
                            "top1_neq": equivalent(cv["top1"], float(r.top1.mean()), False),
                            "groups": int(len(r))},
            }
        return out

    levc = cases[cases.quantity == "level"]
    result["equivalents_all"] = scenario_block(gapc, rk, levc, cv_all)
    result["equivalents_ES"] = scenario_block(gapc[gapc.target == "ES"], rk[rk.target == "ES"],
                                              levc[levc.target == "ES"], cv_es)
    for tag in ("equivalents_all", "equivalents_ES"):
        print(f"\n{tag}")
        for sc, v in result[tag].items():
            print(f"  {sc:12s} level {v['level_mae']:5.1f} ({v['level_neq']})  gap {v['gap_mae']:5.1f} ({v['gap_neq']})"
                  f"  rank rho {v['ranking']['rank_corr']:.2f} ({v['ranking']['rank_corr_neq']})"
                  f"  top1 {v['ranking']['top1']:.2f} ({v['ranking']['top1_neq']})")
            print("      by predictor:", {j: (d["mae"], d["neq"]) for j, d in v["by_predictor"].items()})

    # ---------------------------------------------------------------- 5 regions
    col = find_region_column(raw, args.region_var)
    if col is None:
        cand = [c for c in raw.columns if "reg" in c.lower()]
        print("\nno Spanish region variable found; candidates:", cand[:40], "-> rerun with --region-var")
        result["regions"] = {"error": "region variable not found", "candidates": cand[:40]}
    else:
        es = data[data.country == "ES"].copy()
        es["region"] = raw.loc[es.index, col]
        es = es[es["region"] > 0]
        sizes = es["region"].value_counts()
        big = sizes[sizes >= args.min_region].index
        reg = {"variable": col, "n_by_region": {str(int(r)): int(v) for r, v in sizes.items()},
               "regions_used": [str(int(r)) for r in big], "by_outcome": {}}
        for k in outcomes:
            sub = es[es.region.isin(big)][preds + [k, "weight", "region"]].dropna()
            lv = [weighted_share(g[k].to_numpy(), g["weight"].to_numpy()) for _, g in sub.groupby("region")]
            cl = [weighted_share(subs[(c, k)][k].to_numpy(), subs[(c, k)]["weight"].to_numpy()) for c in countries]
            row = {"level_sd_regions": round(between_sd(np.array([a for a, _ in lv]), np.array([b for _, b in lv])), 2),
                   "level_sd_countries": round(between_sd(np.array([a for a, _ in cl]), np.array([b for _, b in cl])), 2)}
            for j in preds:
                ests, ses = [], []
                for _, g in sub.groupby("region"):
                    m = segments(g[preds].to_numpy())[j]
                    if m.sum() < 15 or (~m).sum() < 15:
                        continue
                    a, sa = weighted_share(g[k].to_numpy()[m], g["weight"].to_numpy()[m])
                    b, sb = weighted_share(g[k].to_numpy()[~m], g["weight"].to_numpy()[~m])
                    ests.append(a - b); ses.append(np.hypot(sa, sb))
                cg, cs_ = [], []
                for c in countries:
                    s2 = subs[(c, k)]; m = segments(s2[preds].to_numpy())[j]
                    a, sa = weighted_share(s2[k].to_numpy()[m], s2["weight"].to_numpy()[m])
                    b, sb = weighted_share(s2[k].to_numpy()[~m], s2["weight"].to_numpy()[~m])
                    cg.append(a - b); cs_.append(np.hypot(sa, sb))
                row[f"gap_sd_regions_{j}"] = round(between_sd(np.array(ests), np.array(ses)), 2) if len(ests) > 2 else None
                row[f"gap_sd_countries_{j}"] = round(between_sd(np.array(cg), np.array(cs_)), 2)
            reg["by_outcome"][k] = row
        reg["mean"] = {key_: round(float(np.mean([v[key_] for v in reg["by_outcome"].values() if v[key_] is not None])), 2)
                       for key_ in next(iter(reg["by_outcome"].values()))}
        result["regions"] = reg
        print(f"\nSpanish regions ({col}), n by region:", reg["n_by_region"])
        print("between-region vs between-country SD (points), mean over outcomes:", reg["mean"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("->", args.out)


if __name__ == "__main__":
    main()
