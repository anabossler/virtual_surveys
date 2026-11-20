"""Country-level bootstrap intervals for the coverage and error of the joint model
and of the own-item comparators (reviewer request: 375 gap predictions share each
target country, so coverage needs intervals that resample countries).

Reads the local case files (not committed: they hold per-country observed values)
  transport_output/joint_validation_cases_v3.csv   from bn/validate_issp.py
  transport_output/own_items_cases.csv             from bn/own_items.py
and the inflation chosen out of region from joint_validation_summary_v3.json.
Each bootstrap replicate resamples the 16 target countries with replacement;
within a country all its cases are kept, so the interval reflects dependence
between the predictions for one target.

Usage: python -m bn.coverage_ci [--boot 2000]
Output: transport_output/coverage_ci.json (aggregate; can be committed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_stress import REGION


def boot(df: pd.DataFrame, rng, B: int) -> dict:
    """Mean error and coverage with percentile intervals over target countries."""
    per = df.groupby("target").agg(err=("err", "sum"), cov=("cov", "sum"), n=("err", "size"))
    t = per.index.to_numpy()
    E, C, N = per.err.to_numpy(), per["cov"].to_numpy(), per.n.to_numpy()
    idx = rng.integers(len(t), size=(B, len(t)))
    eb = E[idx].sum(1) / N[idx].sum(1)
    cb = C[idx].sum(1) / N[idx].sum(1)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 1), round(float(np.percentile(v, 97.5)), 1)]
    return {"mae": round(float(E.sum() / N.sum()), 2), "mae_ci": q(eb),
            "coverage": round(float(100 * C.sum() / N.sum()), 1), "coverage_ci": q(100 * cb),
            "cases": int(N.sum()), "countries": len(t)}


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["err"] = (df.pred - df.observed).abs() * 100
    df["cov"] = ((df.low <= df.observed) & (df.observed <= df.high)).astype(float)
    df["kind"] = np.where(df.quantity == "level", "level", "gap")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=Path("transport_output"))
    ap.add_argument("--tag", default="_v3")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(7650)
    out = {"joint": {}, "own_items": {}}

    jc = a.dir / f"joint_validation_cases{a.tag}.csv"
    js = a.dir / f"joint_validation_summary{a.tag}.json"
    if jc.exists() and js.exists():
        cases = prep(pd.read_csv(jc))
        summ = json.loads(js.read_text())
        print("joint model (95% intervals over target countries)")
        for sc, tau in (("same", "same"), ("small_cross", "cross")):
            kreg = summ["k_out_of_region"][f"{sc}|{tau}"]["k_by_region"]
            sel = cases[(cases.scenario == sc)].copy()
            sel["kreg"] = sel.target.map(lambda c: kreg[REGION[c]])
            # prior: chosen k, scenario's own transport variance
            pr = sel[(sel.stage == "prior") & (sel.tau == tau) & (sel.k == sel.kreg)]
            for kind in ("gap", "level"):
                r = boot(pr[pr.kind == kind], rng, a.boot)
                out["joint"][f"{sc}|prior|{kind}"] = r
                print(f"  {sc:12s} prior        {kind:5s}: error {r['mae']:5.1f} {r['mae_ci']}  "
                      f"coverage {r['coverage']:5.1f}% {r['coverage_ci']}")
            up = sel[(sel.stage == "update") & (sel.k == sel.kreg)]
            for n, g in up.groupby("n_local"):
                for kind in ("gap", "level"):
                    r = boot(g[g.kind == kind], rng, a.boot)
                    out["joint"][f"{sc}|n{n}|{kind}"] = r
                    print(f"  {sc:12s} n={n:<4d}       {kind:5s}: error {r['mae']:5.1f} {r['mae_ci']}  "
                          f"coverage {r['coverage']:5.1f}% {r['coverage_ci']}")
    else:
        print(f"missing {jc} or {js}: joint model skipped")

    oc = a.dir / "own_items_cases.csv"
    if oc.exists():
        cases = prep(pd.read_csv(oc))
        print("\nown-item comparators (95% intervals over target countries)")
        for (m, n, kind), g in cases.groupby(["method", "n_local", "kind"]):
            r = boot(g, rng, a.boot)
            out["own_items"][f"{m}|n{n}|{kind}"] = r
            if kind == "gap":
                print(f"  {m:14s} n={n:<4d} gap  : error {r['mae']:5.1f} {r['mae_ci']}  "
                      f"coverage {r['coverage']:5.1f}% {r['coverage_ci']}")
    else:
        print(f"missing {oc}: comparators skipped")

    (a.dir / "coverage_ci.json").write_text(json.dumps(out, indent=2) + "\n")
    print("->", a.dir / "coverage_ci.json")


if __name__ == "__main__":
    main()
