"""Additional diagnostics for the retrospective test of the transfer procedure.

Reads the case file written by backtest_transfer.py (one row per source,
target, outcome and quantity) and reports, for the levels and for the gaps
between segments:

    sign accuracy       on all gaps and on the gaps with |observed| > 2 SE
    'always positive'   sign baseline: predict a positive gap every time
    interval score      Gneiting and Raftery (2007), alpha = 0.05, in points
    CRPS                Gaussian approximation from the median and the 95%
                        interval, in points
    MAE and coverage    as in backtest_summary_<pool>.json, for checking
    cluster bootstrap   95% intervals for every figure, resampling target
                        countries (cases sharing a target are not independent)

With --issp it also computes the 'small target sample' baseline: the target's
own level and gaps estimated from a simple random sample of n respondents
(n = 100, 200), compared with the full-sample values. This is what a small
pilot survey in the target country would give instead of the transfer.

Usage
-----
    python backtest_diagnostics.py --pool europe
    python backtest_diagnostics.py --pool europe --issp metadata/ZA7650_v2-0-0.dta
Outputs: transport_output/backtest_diagnostics_<pool>.json and .csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

Z = norm.ppf(0.975)
VARIANTS = ("fixed", "transport")
SEED = 42


def crps_gaussian(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def interval_score(y, lo, hi, alpha=0.05):
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def metrics(cases: pd.DataFrame) -> dict:
    """All figures in percentage points or percentages."""
    out = {}
    for quantity, g in (("level", cases[cases.quantity == "level"]),
                        ("gap", cases[cases.quantity != "level"])):
        y = g["observed"].to_numpy()
        res = {"cases": int(len(g))}
        for v in VARIANTS:
            mu, lo, hi = (g[f"{p}_{v}"].to_numpy() for p in ("pred", "low", "high"))
            sd = np.maximum((hi - lo) / (2 * Z), 1e-6)
            res[f"mae_{v}"] = 100 * float(np.mean(np.abs(mu - y)))
            res[f"coverage_{v}"] = 100 * float(np.mean((y >= lo) & (y <= hi)))
            res[f"interval_score_{v}"] = 100 * float(np.mean(interval_score(y, lo, hi)))
            res[f"crps_{v}"] = 100 * float(np.mean(crps_gaussian(y, mu, sd)))
        if quantity == "level":
            res["mae_copy"] = 100 * float(np.mean(np.abs(g["pred_copy"] - y)))
            res["mae_other_mean"] = 100 * float(np.mean(np.abs(g["pred_other_mean"] - y)))
        else:
            clear = np.abs(y) > 2 * g["observed_se"].to_numpy()
            res["clear_cases"] = int(clear.sum())
            res["mae_none"] = 100 * float(np.mean(np.abs(y)))
            res["mae_unshrunk"] = 100 * float(np.mean(np.abs(g["pred_unshrunk"] - y)))
            for v in VARIANTS:
                right = np.sign(g[f"pred_{v}"].to_numpy()) == np.sign(y)
                res[f"sign_all_{v}"] = 100 * float(right.mean())
                res[f"sign_clear_{v}"] = 100 * float(right[clear].mean())
            res["sign_all_always_positive"] = 100 * float(np.mean(y > 0))
            res["sign_clear_always_positive"] = 100 * float(np.mean(y[clear] > 0))
        out[quantity] = res
    return out


def cluster_bootstrap(cases: pd.DataFrame, b: int, rng) -> dict:
    groups = {t: g for t, g in cases.groupby("target")}
    keys = list(groups)
    draws = []
    for _ in range(b):
        pick = rng.choice(keys, len(keys), replace=True)
        draws.append(metrics(pd.concat([groups[k] for k in pick], ignore_index=True)))
    ci = {}
    for quantity in ("level", "gap"):
        frame = pd.DataFrame([d[quantity] for d in draws])
        ci[quantity] = {c: [float(frame[c].quantile(0.025)), float(frame[c].quantile(0.975))]
                        for c in frame.columns if c not in ("cases", "clear_cases")}
    return ci


def small_sample_baseline(issp: Path, countries: list[str], sizes, reps: int, rng) -> dict:
    """MAE (points) of the target's own estimate from n respondents against its full sample."""
    from backtest_transfer import segments, weighted_share
    from transport_issp import OUTCOMES, PREDICTORS, design

    data = design(pd.read_stata(issp, convert_categoricals=False))
    data = data[data.country.isin(countries)]
    preds = list(PREDICTORS)
    err = {n: {"level": [], "gap": []} for n in sizes}
    for c in countries:
        part = data[data.country == c]
        for k in OUTCOMES:
            sub = part[preds + [k, "weight"]].dropna()
            if len(sub) < max(sizes):
                continue
            x, y, w = sub[preds].to_numpy(), sub[k].to_numpy(), sub["weight"].to_numpy()
            full_level = weighted_share(y, w)[0]
            masks = segments(x)
            full_gap = {j: weighted_share(y[m], w[m])[0] - weighted_share(y[~m], w[~m])[0]
                        for j, m in masks.items() if m.sum() >= 30 and (~m).sum() >= 30}
            for n in sizes:
                for _ in range(reps):
                    idx = rng.choice(len(y), n, replace=False)
                    xs, ys, ws = x[idx], y[idx], w[idx]
                    err[n]["level"].append(abs(weighted_share(ys, ws)[0] - full_level))
                    for j, m in segments(xs).items():
                        if j not in full_gap or m.sum() < 2 or (~m).sum() < 2:
                            continue
                        g = weighted_share(ys[m], ws[m])[0] - weighted_share(ys[~m], ws[~m])[0]
                        err[n]["gap"].append(abs(g - full_gap[j]))
    return {f"n{n}": {q: 100 * float(np.mean(v)) for q, v in e.items()} for n, e in err.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", choices=("europe", "all"), default="europe")
    ap.add_argument("--dir", type=Path, default=Path("transport_output"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--issp", type=Path, default=None)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    cases = pd.read_csv(args.dir / f"backtest_cases_{args.pool}.csv")
    point = metrics(cases)
    summary_file = args.dir / f"backtest_summary_{args.pool}.json"
    if summary_file.exists():
        ref = json.loads(summary_file.read_text())["pairs"]
        for q in ("level", "gap"):
            for v in VARIANTS:
                assert abs(point[q][f"mae_{v}"] - 100 * ref[q][f"mae_{v}"]) < 1e-6, "case file does not match summary"
        print(f"case file matches {summary_file.name}")
    ci = cluster_bootstrap(cases, args.boot, rng)
    result = {"pool": args.pool, "bootstrap": f"{args.boot} resamples of target countries",
              "point": point, "ci95": ci}
    if args.issp is not None:
        result["small_target_sample"] = small_sample_baseline(
            args.issp, sorted(cases["target"].unique()), args.sizes, args.reps, rng)

    rows = [{"quantity": q, "metric": m, "estimate": v,
             "low": ci[q].get(m, [np.nan, np.nan])[0], "high": ci[q].get(m, [np.nan, np.nan])[1]}
            for q in ("level", "gap") for m, v in point[q].items()]
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if "small_target_sample" in result:
        print("small target sample (MAE, points):", json.dumps(result["small_target_sample"]))
    (args.dir / f"backtest_diagnostics_{args.pool}.json").write_text(json.dumps(result, indent=2) + "\n")
    table.to_csv(args.dir / f"backtest_diagnostics_{args.pool}.csv", index=False)
    print("->", args.dir / f"backtest_diagnostics_{args.pool}.json")


if __name__ == "__main__":
    main()
