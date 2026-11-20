"""Robustness of the joint-model validation to the choice of proxy outcomes.

Reviewer request: do the results hold when the five ISSP proxy outcomes change?
In bn/validate_issp.py the transport hyperparameters for a target are already
estimated without the target's region and without the target outcome; only the
inflation k was chosen on all five outcomes. This script

  1. chooses k leaving out both the target's region and the target outcome (the
     smallest k on the grid whose prior gap coverage reaches 95% on the other
     regions and the other four outcomes), and reports held-out coverage;
  2. reports every result separately for each held-out outcome: prior and
     updated error and coverage of the joint model, the local sample alone, and
     the equivalent target sample of the prior (read from the local-only error
     curve of the same outcome, local samples of 25-200 against the rest of the
     target);
  3. repeats the comparison with the target's own related items for each outcome
     (bn/own_items.py), to see whether "own items beat the foreign source" holds
     for every outcome or only on average.

No new model runs: it reads the local case files
  transport_output/joint_validation_cases_v3.csv   (bn/validate_issp.py)
  transport_output/own_items_cases.csv             (bn/own_items.py)
which stay local (they hold per-country observed values).

Usage: python -m bn.proxy_robustness
Output: transport_output/proxy_robustness.json (aggregate; can be committed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_stress import REGION

N_LOCAL = (25, 50, 100, 200)
SHORT = {"pay_higher_prices": "prices", "pay_higher_taxes": "taxes",
         "cut_living_standard": "living std", "do_right_despite_cost": "despite cost",
         "avoid_products": "avoid products"}


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["err"] = (df.pred - df.observed).abs() * 100
    df["cov"] = ((df.low <= df.observed) & (df.observed <= df.high)).astype(float)
    df["width"] = (df.high - df.low) * 100
    df["kind"] = np.where(df.quantity == "level", "level", "gap")
    df["region"] = df.target.map(REGION)
    return df


def choose_k(prior_gap: pd.DataFrame, by_outcome: bool) -> dict:
    """Smallest k whose gap coverage reaches 95% without the held-out region (and,
    if by_outcome, without the held-out outcome); the largest k if none does."""
    keys = [(r, o) for r in sorted(prior_gap.region.unique())
            for o in (sorted(prior_gap.outcome.unique()) if by_outcome else [None])]
    out = {}
    for r, o in keys:
        other = prior_gap[(prior_gap.region != r) & ((prior_gap.outcome != o) if o else True)]
        cov = other.groupby("k")["cov"].mean()
        ok = cov[cov >= 0.95]
        out[(r, o)] = float(ok.index.min()) if len(ok) else float(cov.index.max())
    return out


def apply_k(df: pd.DataFrame, kmap: dict, by_outcome: bool) -> pd.DataFrame:
    key = list(zip(df.region, df.outcome if by_outcome else [None] * len(df)))
    return df[df.k.to_numpy() == np.array([kmap[x] for x in key])]


def n_equivalent(err: float, curve: dict) -> str:
    """Local sample size whose error equals err, by log-log interpolation of the
    local-only error curve; outside the computed range, a bound."""
    ns = np.array(sorted(curve)); es = np.array([curve[n] for n in ns])
    if err >= es[0]:
        return f"<{ns[0]}"
    if err <= es[-1]:
        return f">{ns[-1]}"
    i = int(np.searchsorted(-es, -err))          # es is decreasing
    x0, x1, y0, y1 = np.log(ns[i - 1]), np.log(ns[i]), np.log(es[i - 1]), np.log(es[i])
    return str(int(round(np.exp(x0 + (np.log(err) - y0) * (x1 - x0) / (y1 - y0)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=Path("transport_output"))
    ap.add_argument("--tag", default="_v3")
    a = ap.parse_args()
    cases = prep(pd.read_csv(a.dir / f"joint_validation_cases{a.tag}.csv"))
    out = {"k_choice": {}, "by_outcome": {}, "own_items_by_outcome": {}}

    for sc, tau in (("small_cross", "cross"), ("same", "same")):
        sel = cases[cases.scenario == sc]
        prior = sel[(sel.stage == "prior") & (sel.tau == tau)]
        upd = sel[sel.stage == "update"].copy()
        upd["err_local"] = (upd.local - upd.observed).abs() * 100
        pg = prior[prior.kind == "gap"]

        # 1. k chosen without region only, and without region and outcome
        print(f"\n=== {sc} (transport variance: {tau}) ===")
        res_k = {}
        for label, by_o in (("region", False), ("region_and_outcome", True)):
            km = choose_k(pg, by_o)
            held = apply_k(pg, km, by_o)
            held_l = apply_k(prior[prior.kind == "level"], km, by_o)
            r = {"coverage_gap": round(100 * held["cov"].mean(), 1), "width_gap": round(held.width.mean(), 1),
                 "mae_gap": round(held.err.mean(), 2), "coverage_level": round(100 * held_l["cov"].mean(), 1),
                 "k_values": {f"{k:g}": int(v) for k, v in pd.Series(list(km.values())).value_counts().sort_index().items()}}
            for n in N_LOCAL:
                u = apply_k(upd[(upd.kind == "gap") & (upd.n_local == n)], km, by_o)
                r[f"update_{n}"] = {"coverage": round(100 * u["cov"].mean(), 1), "mae": round(u.err.mean(), 2)}
            res_k[label] = r
            print(f"k chosen without {label.replace('_', ' ')}: k values {r['k_values']}; prior gaps "
                  f"coverage {r['coverage_gap']}% (width {r['width_gap']}), levels {r['coverage_level']}%; "
                  + "; ".join(f"n={n}: {r[f'update_{n}']['coverage']}%" for n in N_LOCAL))
        out["k_choice"][sc] = res_k

        # 2. results by held-out outcome, with k chosen without region and outcome
        km = choose_k(pg, True)
        pr_g, pr_l = apply_k(pg, km, True), apply_k(prior[prior.kind == "level"], km, True)
        up_g = apply_k(upd[upd.kind == "gap"], km, True)
        rows = {}
        print(f"\nby held-out outcome (k without region and outcome); gaps, error in points (coverage %)")
        print(f"  {'outcome':15s} {'prior':>13s} {'n_eq':>5s} {'level cov':>9s} "
              + " ".join(f"{'n=' + str(n):>13s}" for n in N_LOCAL) + "   local-only 25/50/100/200")
        for o in sorted(pg.outcome.unique()):
            p, l, u = pr_g[pr_g.outcome == o], pr_l[pr_l.outcome == o], up_g[up_g.outcome == o]
            curve = {n: float(u[u.n_local == n].err_local.mean()) for n in N_LOCAL}
            r = {"prior_mae": round(p.err.mean(), 2), "prior_coverage": round(100 * p["cov"].mean(), 1),
                 "prior_width": round(p.width.mean(), 1), "level_coverage": round(100 * l["cov"].mean(), 1),
                 "level_mae": round(l.err.mean(), 2), "n_eq": n_equivalent(p.err.mean(), curve),
                 "local_curve": {n: round(v, 2) for n, v in curve.items()}}
            for n in N_LOCAL:
                g = u[u.n_local == n]
                r[f"update_{n}"] = {"mae": round(g.err.mean(), 2), "coverage": round(100 * g["cov"].mean(), 1)}
            rows[o] = r
            print(f"  {SHORT.get(o, o):15s} {r['prior_mae']:5.1f} ({r['prior_coverage']:4.1f}) {r['n_eq']:>5s} "
                  f"{r['level_coverage']:8.1f}% "
                  + " ".join(f"{r[f'update_{n}']['mae']:5.1f} ({r[f'update_{n}']['coverage']:4.1f})" for n in N_LOCAL)
                  + "   " + "/".join(f"{curve[n]:.1f}" for n in N_LOCAL))
        maes = [r["prior_mae"] for r in rows.values()]
        covs = [r["prior_coverage"] for r in rows.values()]
        print(f"  range over outcomes: prior error {min(maes):.1f}-{max(maes):.1f}, coverage {min(covs):.1f}-{max(covs):.1f}%")
        out["by_outcome"][sc] = rows

    # 3. own items against the joint model, by outcome (application-like scenario)
    oc = a.dir / "own_items_cases.csv"
    if oc.exists():
        own = prep(pd.read_csv(oc))
        own = own[own.kind == "gap"]
        # each target-outcome weighted equally, as in bn/own_items.py
        tab = own.groupby(["outcome", "method", "n_local", "target"]).err.mean().groupby(
            ["outcome", "method", "n_local"]).mean()
        sel = cases[cases.scenario == "small_cross"]
        km = choose_k(sel[(sel.stage == "prior") & (sel.tau == "cross") & (sel.kind == "gap")], True)
        jp = apply_k(sel[(sel.stage == "prior") & (sel.tau == "cross") & (sel.kind == "gap")], km, True)
        ju = apply_k(sel[(sel.stage == "update") & (sel.kind == "gap")], km, True)
        print("\nown related items against the foreign source, gaps, by outcome (error in points)")
        print(f"  {'outcome':15s} {'own n=0':>8s} {'joint n=0':>9s} {'own-reg 25':>10s} {'joint 25':>9s} "
              f"{'own-reg 200':>11s} {'joint 200':>9s}  own wins")
        wins = 0
        for o in sorted(own.outcome.unique()):
            g = lambda m, n: float(tab.get((o, m, n), np.nan))
            j0 = jp[jp.outcome == o].groupby("target").err.mean().mean()
            j25 = ju[(ju.outcome == o) & (ju.n_local == 25)].groupby("target").err.mean().mean()
            j200 = ju[(ju.outcome == o) & (ju.n_local == 200)].groupby("target").err.mean().mean()
            r = {"own_0": round(g("own", 0), 2), "joint_0": round(j0, 2), "own_mrp_25": round(g("own_mrp", 25), 2),
                 "joint_25": round(j25, 2), "own_mrp_200": round(g("own_mrp", 200), 2), "joint_200": round(j200, 2)}
            w = sum([r["own_0"] < r["joint_0"], r["own_mrp_25"] < r["joint_25"], r["own_mrp_200"] < r["joint_200"]])
            wins += w
            out["own_items_by_outcome"][o] = r
            print(f"  {SHORT.get(o, o):15s} {r['own_0']:8.1f} {r['joint_0']:9.1f} {r['own_mrp_25']:10.1f} "
                  f"{r['joint_25']:9.1f} {r['own_mrp_200']:11.1f} {r['joint_200']:9.1f}  {w}/3")
        print(f"  own items better in {wins} of {3 * own.outcome.nunique()} comparisons")
    else:
        print(f"\n({oc} not found: run python -m bn.own_items first)")

    (a.dir / "proxy_robustness.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print("\n->", a.dir / "proxy_robustness.json")


if __name__ == "__main__":
    main()
