"""Dependence between predictions and the meaning of wide intervals (reviewer checks).

1. Dependence. The benchmark's predictions share target countries, source
   countries and proxy outcomes. The country bootstrap of bn/coverage_ci.py
   resamples targets only. Here errors and coverage of the joint model are
   bootstrapped over targets; targets and outcomes; targets and sources; and all
   three at once (multiway bootstrap with independent multinomial weights for
   each factor, the product of the weights giving each cell's weight).
2. Wide intervals. A 95% coverage with intervals some 57 points wide could be
   reached by any very wide interval. The interval score (width plus 2/alpha times
   the distance by which the observed value falls outside) rewards calibration
   and penalises width at once; it is reported for each inflation k and for the
   own-item comparator. Also reported: the share of gap intervals that exclude
   zero, and how often the sign is right when they do.

Reads the local case files (not committed):
  transport_output/joint_validation_cases_v3.csv   (bn/validate_issp.py)
  transport_output/own_items_cases.csv             (bn/own_items.py)

Usage: python -m bn.dependence_checks [--boot 2000]
Output: transport_output/dependence_checks.json (aggregate; can be committed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_stress import REGION

ALPHA = 0.05


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["err"] = (df.pred - df.observed).abs() * 100
    df["cov"] = ((df.low <= df.observed) & (df.observed <= df.high)).astype(float)
    df["width"] = (df.high - df.low) * 100
    below = np.clip(df.low - df.observed, 0, None) * 100
    above = np.clip(df.observed - df.high, 0, None) * 100
    df["iscore"] = df.width + (2 / ALPHA) * (below + above)
    df["kind"] = np.where(df.quantity == "level", "level", "gap")
    df["region"] = df.target.map(REGION)
    return df


def k_out_of_region(prior_gap: pd.DataFrame) -> dict:
    """Smallest k whose gap coverage reaches 95% in the other regions."""
    out = {}
    for r in sorted(prior_gap.region.unique()):
        cov = prior_gap[prior_gap.region != r].groupby("k")["cov"].mean()
        ok = cov[cov >= 0.95]
        out[r] = float(ok.index.min()) if len(ok) else float(cov.index.max())
    return out


def multiway(df: pd.DataFrame, factors: list[str], rng, B: int) -> dict:
    """Mean error and coverage with percentile intervals from a multiway bootstrap
    over the given factors (each resampled independently, cells weighted by the
    product of their factors' multinomial counts)."""
    keys = [sorted(df[f].unique()) for f in factors]
    idx = [df[f].map({v: i for i, v in enumerate(k)}).to_numpy() for f, k in zip(factors, keys)]
    shape = tuple(len(k) for k in keys)
    E, C, N = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    np.add.at(E, tuple(idx), df.err.to_numpy())
    np.add.at(C, tuple(idx), df["cov"].to_numpy())
    np.add.at(N, tuple(idx), 1.0)
    eb, cb = np.empty(B), np.empty(B)
    for b in range(B):
        w = [rng.multinomial(n, np.full(n, 1.0 / n)).astype(float) for n in shape]
        W = w[0]
        for wi in w[1:]:
            W = np.multiply.outer(W, wi)
        n = (W * N).sum()
        eb[b], cb[b] = (W * E).sum() / n, (W * C).sum() / n
    q = lambda v: [round(float(np.percentile(v, 2.5)), 1), round(float(np.percentile(v, 97.5)), 1)]
    return {"mae": round(float(E.sum() / N.sum()), 2), "mae_ci": q(eb),
            "coverage": round(float(100 * C.sum() / N.sum()), 1), "coverage_ci": q(100 * cb)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=Path("transport_output"))
    ap.add_argument("--tag", default="_v3")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(2020)
    cases = prep(pd.read_csv(a.dir / f"joint_validation_cases{a.tag}.csv"))
    out = {"bootstrap": {}, "interval_score": {}, "excludes_zero": {}}

    sel = cases[cases.scenario == "small_cross"]
    prior = sel[(sel.stage == "prior") & (sel.tau == "cross")]
    kreg = k_out_of_region(prior[prior.kind == "gap"])
    pick = lambda d: d[d.k.to_numpy() == d.region.map(kreg).to_numpy()]
    held_prior = pick(prior[prior.kind == "gap"])
    upd = sel[(sel.stage == "update") & (sel.kind == "gap")]

    # 1. dependence
    schemes = {"targets": ["target"], "targets x outcomes": ["target", "outcome"],
               "targets x sources": ["target", "source"],
               "targets x sources x outcomes": ["target", "source", "outcome"]}
    print("joint model, application-like scenario, gaps, k chosen out of region:", kreg)
    print("error (points) and coverage (%) with 95% bootstrap intervals\n")
    for label, data in [("no local data", held_prior)] + \
            [(f"n = {n}", pick(upd[upd.n_local == n])) for n in (25, 200)]:
        print(f"  {label}")
        out["bootstrap"][label] = {}
        for name, f in schemes.items():
            r = multiway(data, f, rng, a.boot)
            out["bootstrap"][label][name] = r
            print(f"    {name:30s} error {r['mae']:5.1f} {r['mae_ci']}   coverage {r['coverage']:5.1f}% {r['coverage_ci']}")

    # 2. interval score and informativeness, no local data
    print("\ninterval score (points; lower is better), width and coverage, gaps without local data")
    g = prior[prior.kind == "gap"]
    rows = {f"k = {k:g}": g[g.k == k] for k in sorted(g.k.unique())}
    rows["k out of region"] = held_prior
    oc = a.dir / "own_items_cases.csv"
    if oc.exists():
        own = prep(pd.read_csv(oc))
        rows["own items (no local data)"] = own[(own.kind == "gap") & (own.method == "own") & (own.n_local == 0)]
        rows["other countries (no local data)"] = own[(own.kind == "gap") & (own.method == "generic") & (own.n_local == 0)]
    for name, d in rows.items():
        if d.empty:
            continue
        mid = (d.low + d.high) / 2
        excl = (d.low > 0) | (d.high < 0)
        right = (np.sign(mid[excl]) == np.sign(d.observed[excl])).mean() if excl.any() else np.nan
        r = {"interval_score": round(d.iscore.mean(), 1), "width": round(d.width.mean(), 1),
             "coverage": round(100 * d["cov"].mean(), 1), "excludes_zero": round(100 * excl.mean(), 1),
             "sign_right_when_excluding_zero": round(100 * right, 1) if np.isfinite(right) else None}
        out["interval_score"][name] = r
        print(f"  {name:32s} IS {r['interval_score']:6.1f}  width {r['width']:5.1f}  coverage {r['coverage']:5.1f}%  "
              f"excludes zero {r['excludes_zero']:5.1f}%  sign right then {r['sign_right_when_excluding_zero']}%")

    (a.dir / "dependence_checks.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n->", a.dir / "dependence_checks.json")


if __name__ == "__main__":
    main()
