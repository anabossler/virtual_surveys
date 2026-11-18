"""Does transferred evidence still help once some local data exist?

Retrospective test of the Beta--Binomial update on the ISSP proxy outcomes.
For every target country and outcome, simple random samples of n target
respondents (n = 25, 50, 100, 200) are drawn and three strategies are compared
against a reference computed from the remaining, disjoint, respondents:

    transfer    the transport-adjusted transfer prediction alone (prior mean)
    local       the local sample alone (posterior mean under a Beta(1/2, 1/2)
                prior, i.e. essentially the sample share)
    combined    the transport-adjusted transfer prediction used as a Beta prior
                (moment-matched, as in the paper) and updated with the same
                local sample, segment by segment
    combined_gap  for gaps only: a normal prior on the gap itself (from the
                transfer draws, in which the common level uncertainty cancels)
                combined by precision weighting with the local gap estimate;
                this keeps the information that segment-wise updating loses

for the national level and for each segment level; a gap is the difference
between the two segment estimates of a predictor. The prior is calibrated
without the target's whole European region (leave-region-out transport
parameters), so the target never informs its own prior.

Two transfer scenarios:
    same          source with the same outcome and its full sample
    small_cross   source refitted on --source-n respondents and coefficients
                  from another outcome (base rate from the projected outcome),
                  the conditions closest to the application

Reported for every scenario, strategy and n: MAE of levels and gaps (points),
share of gaps with the right sign, coverage of 95% intervals, and the gain of
combined over local; 95% intervals from bootstrap resamples of target countries.

Usage: python backtest_update.py --issp metadata/ZA7650_v2-0-0.dta [--inflate k]
With --inflate k every prior keeps its mean and has its SD multiplied by k (the
recalibration factor estimated by recalibrate_intervals.py); outputs then carry
the suffix _k<k>.
Outputs (transport_output): backtest_update_cases.csv (aggregated over
replications) and backtest_update_summary.json
"""
from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from backtest_stress import REGION, fit, region_parameters
from backtest_transfer import EUROPE, MAX_TARGET_N, calibrate, segments, weighted_share
from transport_issp import OUTCOMES, PREDICTORS, design

SEED = 23
SIZES = (25, 50, 100, 200)
JEFFREYS = 0.5


def beta_from_draws(d: np.ndarray, m_max: float = 1e4) -> tuple[float, float]:
    """Moment-matched Beta parameters for a vector of probability draws."""
    p = float(np.clip(d.mean(), 1e-4, 1 - 1e-4))
    v = float(max(d.var(), 1e-8))
    m = min(max(p * (1 - p) / v - 1, 0.5), m_max)
    return p * m, (1 - p) * m


def inflate_beta(ab: tuple[float, float], k: float) -> tuple[float, float]:
    """Beta prior with the same mean and k^2 times the variance (k = 1: unchanged)."""
    a, b = ab
    m = a + b
    m2 = max((m + 1) / k**2 - 1, 0.5)
    return a / m * m2, b / m * m2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--issp", type=Path, default=Path("metadata/ZA7650_v2-0-0.dta"))
    ap.add_argument("--out", type=Path, default=Path("transport_output"))
    ap.add_argument("--source-n", type=int, default=272)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--inflate", type=float, default=1.0,
                    help="multiply the prior SD by k (recalibration factor from recalibrate_intervals.py)")
    args = ap.parse_args()
    k_inf = args.inflate
    tag = "" if k_inf == 1.0 else f"_k{k_inf:g}"
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
    params = {(t, k): region_parameters(beta, se, share, countries, countries.index(t), outcomes.index(k), True)
              for t in countries for k in outcomes}

    # local samples and disjoint references, shared by all sources and scenarios
    local = {}
    for t in countries:
        for k in outcomes:
            if full[(t, k)] is None:
                continue
            sub = subs[(t, k)]
            x, y, w = sub[preds].to_numpy(), sub[k].to_numpy(), sub["weight"].to_numpy()
            reps = []
            for n in SIZES:
                for r in range(args.reps):
                    idx = rng.choice(len(y), n, replace=False)
                    rest = np.setdiff1d(np.arange(len(y)), idx)
                    xs, ys, ws = x[idx], y[idx], w[idx]
                    xr, yr, wr = x[rest], y[rest], w[rest]
                    ms, mr = segments(xs), segments(xr)
                    rec = {"n": n, "rep": r,
                           "level": (weighted_share(ys, ws)[0] * n, n, weighted_share(yr, wr)[0])}
                    for j in preds:
                        s1, s0, r1, r0 = ms[j], ~ms[j], mr[j], ~mr[j]
                        if r1.sum() < 30 or r0.sum() < 30:
                            continue
                        cnt = lambda m: (weighted_share(ys[m], ws[m])[0] * m.sum() if m.sum() else 0.0, int(m.sum()))
                        rec[j] = (cnt(s1), cnt(s0),
                                  weighted_share(yr[r1], wr[r1])[0] - weighted_share(yr[r0], wr[r0])[0])
                    reps.append(rec)
            local[(t, k)] = (x, w, reps)

    rows = []
    for scenario in ("same", "small_cross"):
        fits = small if scenario == "small_cross" else full
        for k in outcomes:
            for s, t in permutations(countries, 2):
                if (t, k) not in local or fits[(s, k)] is None:
                    continue
                kc = k if scenario == "same" else rng.choice([o for o in outcomes if o != k])
                src, base = fits[(s, kc)], fits[(s, k)]
                if src is None:
                    continue
                x, w, reps = local[(t, k)]
                idx = np.sort(rng.choice(len(w), min(len(w), MAX_TARGET_N), replace=False))
                xe, we = x[idx], w[idx]
                p, d = params[(t, k)], args.draws
                bd = rng.normal(src["beta"], src["se"], size=(d, len(preds)))
                b0 = rng.beta(base["share"] * base["n"] + 0.5, (1 - base["share"]) * base["n"] + 0.5, size=d)
                a0 = rng.normal(p["a0_mean"], p["a0_sd"], size=(d, 1))
                lin = xe @ (a0 * bd + rng.normal(0.0, p["eta_sd"], size=bd.shape)).T
                pr = norm.cdf(calibrate(lin, we, b0) + rng.normal(0.0, p["level_sd"], size=d) + lin)
                priors = {"level": inflate_beta(beta_from_draws(np.average(pr, axis=0, weights=we)), k_inf)}
                me = segments(xe)
                for j in preds:
                    if me[j].sum() >= 10 and (~me[j]).sum() >= 10:
                        d1 = np.average(pr[me[j]], axis=0, weights=we[me[j]])
                        d0 = np.average(pr[~me[j]], axis=0, weights=we[~me[j]])
                        priors[j] = (inflate_beta(beta_from_draws(d1), k_inf), inflate_beta(beta_from_draws(d0), k_inf),
                                     (float((d1 - d0).mean()), k_inf * float(max((d1 - d0).std(), 1e-4))))
                acc = {}
                for rec in reps:
                    n = rec["n"]
                    a, b = priors["level"]
                    yl, nl, ref = rec["level"]
                    est = {"transfer": a / (a + b), "local": (yl + JEFFREYS) / (nl + 2 * JEFFREYS),
                           "combined": (a + yl) / (a + b + nl)}
                    lo_hi = {"transfer": beta_dist.ppf([0.025, 0.975], a, b),
                             "local": beta_dist.ppf([0.025, 0.975], yl + JEFFREYS, nl - yl + JEFFREYS),
                             "combined": beta_dist.ppf([0.025, 0.975], a + yl, b + nl - yl)}
                    for st in est:
                        acc.setdefault(("level", n, st), []).append(
                            (abs(est[st] - ref), np.nan, float(lo_hi[st][0] <= ref <= lo_hi[st][1])))
                    for j in preds:
                        if j not in rec or j not in priors:
                            continue
                        (y1, n1), (y0, n0), gref = rec[j]
                        (a1, b1), (a0_, b0_), (gm, gs) = priors[j]
                        pars = {"transfer": ((a1, b1), (a0_, b0_)),
                                "local": ((y1 + JEFFREYS, n1 - y1 + JEFFREYS), (y0 + JEFFREYS, n0 - y0 + JEFFREYS)),
                                "combined": ((a1 + y1, b1 + n1 - y1), (a0_ + y0, b0_ + n0 - y0))}
                        for st, ((u1, v1), (u0, v0)) in pars.items():
                            g = u1 / (u1 + v1) - u0 / (u0 + v0)
                            sim = rng.beta(u1, v1, 200) - rng.beta(u0, v0, 200)
                            lo, hi = np.percentile(sim, [2.5, 97.5])
                            acc.setdefault(("gap", n, st), []).append(
                                (abs(g - gref), float(np.sign(g) == np.sign(gref)), float(lo <= gref <= hi)))
                        # gap-level normal update
                        q1, q0 = (y1 + JEFFREYS) / (n1 + 2 * JEFFREYS), (y0 + JEFFREYS) / (n0 + 2 * JEFFREYS)
                        if n1 > 0 and n0 > 0:
                            ghat, gv = q1 - q0, q1 * (1 - q1) / n1 + q0 * (1 - q0) / n0
                            prec = 1 / gs**2 + 1 / gv
                            g = (gm / gs**2 + ghat / gv) / prec
                            sd = prec ** -0.5
                        else:
                            g, sd = gm, gs
                        acc.setdefault(("gap", n, "combined_gap"), []).append(
                            (abs(g - gref), float(np.sign(g) == np.sign(gref)),
                             float(g - 1.96 * sd <= gref <= g + 1.96 * sd)))
                for (q, n, st), vals in acc.items():
                    v = np.array(vals, dtype=float)
                    rows.append({"scenario": scenario, "source": s, "target": t, "outcome": k, "quantity": q,
                                 "n": n, "strategy": st, "mae": 100 * np.nanmean(v[:, 0]),
                                 "sign": 100 * np.nanmean(v[:, 1]) if q == "gap" else np.nan,
                                 "coverage": 100 * np.nanmean(v[:, 2]), "cases": len(v)})
        print(f"scenario {scenario} done")

    cases = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.out / f"backtest_update_cases{tag}.csv", index=False)

    def agg(g):
        wts = g["cases"]
        return pd.Series({m: np.average(g[m], weights=wts) if g[m].notna().any() else np.nan
                          for m in ("mae", "sign", "coverage")})

    table = cases.groupby(["scenario", "quantity", "n", "strategy"]).apply(agg).reset_index()
    targets = sorted(cases["target"].unique())
    boot = []
    for _ in range(args.boot):
        pick = rng.choice(targets, len(targets), replace=True)
        bc = pd.concat([cases[cases.target == tt] for tt in pick])
        bt = bc.groupby(["scenario", "quantity", "n", "strategy"]).apply(agg)
        boot.append(bt["mae"])
    boot = pd.concat(boot, axis=1)
    table["mae_lo"] = boot.quantile(0.025, axis=1).values
    table["mae_hi"] = boot.quantile(0.975, axis=1).values
    loc = table[table.strategy == "local"].set_index(["scenario", "quantity", "n"])["mae"]
    gains = []
    for st in ("combined", "combined_gap"):
        if st not in boot.index.get_level_values("strategy"):
            continue
        bg = boot.xs("local", level="strategy") - boot.xs(st, level="strategy")
        pg = loc - table[table.strategy == st].set_index(["scenario", "quantity", "n"])["mae"]
        gains.append(pd.DataFrame({"strategy": st, "gain": pg, "gain_lo": bg.quantile(0.025, axis=1),
                                   "gain_hi": bg.quantile(0.975, axis=1)}).dropna(subset=["gain"]).reset_index())
    gains = pd.concat(gains, ignore_index=True)
    summary = {"sizes": list(SIZES), "reps": args.reps, "draws": args.draws, "seed": SEED,
               "source_n": args.source_n, "prior": "transport-adjusted, leave-region-out", "inflate": k_inf,
               "table": table.to_dict(orient="records"), "gain_combined_over_local": gains.to_dict(orient="records")}
    (args.out / f"backtest_update_summary{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    show = table.pivot_table(index=["scenario", "quantity", "n"], columns="strategy", values=["mae", "sign", "coverage"])
    print(show.round(1).to_string())
    print(gains.round(2).to_string(index=False))
    print("->", args.out / f"backtest_update_summary{tag}.json")


if __name__ == "__main__":
    main()
