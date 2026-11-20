"""Strong comparators for the joint model: the target's own related items, with
and without a first local sample (reviewer request: MRP/SAE and shrinkage toward
related items combined with local data).

Same benchmark as bn/validate_issp.py: 16 European countries, five ISSP proxy
outcomes; for each target country and outcome, local samples of n = 25, 50,
100, 200 respondents with the outcome observed, compared with the remaining
respondents. Nothing here uses the source country, so each (target, outcome)
is one case, as each target is weighted equally in the joint-model summary.

Methods (gaps between the two groups of each predictor, and the national level):
  local         the local sample alone (Agresti-Caffo intervals), as in validate_issp
  own           prior only: the target's own observed gap (level) on its other four
                items, the competitor 'proxy_own' of benchmark_extensions.py
  generic       prior only: mean gap of the other countries on the other items
  own+local     normal update of each gap (and of the level) with the own-item prior,
                prior variance estimated without the target's region
  generic+local the same with the generic prior
  mrp           logistic regression on the local sample (weak prior), poststratified
                to the target's composition (80 predictor cells)
  own_mrp       the same regression with the prior centred on the target's own
                coefficients for its other items, prior SDs estimated without the
                target's region: hierarchical shrinkage toward related items plus
                local data, poststratified
All intervals predict the statistic of the reference sample, so its sampling error
is added, as in validate_issp.py. Prior variances subtract the sampling variance
of the observed values they are estimated from.

Usage: python -m bn.own_items [--reps 40]
Output: transport_output/own_items_summary.json (aggregate; can be committed),
        transport_output/own_items_cases.csv (ignored by .gitignore).
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_stress import REGION
from backtest_transfer import EUROPE
from bn.validate_issp import (N_LOCAL, XCELL, cell_index, local_estimates, observed,
                              quantities, wquant)
from transport_issp import OUTCOMES, PREDICTORS, design

SEED = 3391
D = 2000                      # Laplace draws for the regression methods
QS = ["level"] + [f"gap_{p}" for p in PREDICTORS]


# ----------------------------------------------------------------- regression
def logit_map(X, y, w, m, s2, iters=50):
    """MAP of a weighted logistic regression with prior N(m, diag s2); returns the
    mode and the posterior precision (Laplace approximation)."""
    b = m.copy()
    P0 = np.diag(1.0 / s2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        g = X.T @ (w * (y - p)) - P0 @ (b - m)
        H = (X * (w * p * (1 - p))[:, None]).T @ X + P0
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-(X @ b)))
    H = (X * (w * p * (1 - p))[:, None]).T @ X + P0
    return b, H


def design_x(x):
    return np.column_stack([np.ones(len(x)), x])


XC = design_x(XCELL)          # 80 x 6


def reg_draws(b, H, rng):
    L = np.linalg.cholesky(np.linalg.inv(H))
    B = b + rng.standard_normal((D, len(b))) @ L.T
    return 1 / (1 + np.exp(-(B @ XC.T)))          # D x 80


def cell_weights(x, w):
    pi = np.bincount(cell_index(x), weights=w, minlength=80)
    return (pi / pi.sum())[None, :]


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--issp", type=Path, default=Path("metadata/ZA7650_v2-0-0.dta"))
    ap.add_argument("--out", type=Path, default=Path("transport_output"))
    ap.add_argument("--reps", type=int, default=40, help="local samples per target, outcome and size")
    ap.add_argument("--joint", type=Path, default=Path("transport_output/joint_validation_summary_v3.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    preds, outcomes = list(PREDICTORS), list(OUTCOMES)

    data = design(pd.read_stata(a.issp, convert_categoricals=False))
    data = data[data.country.isin(EUROPE)]
    countries = sorted(data.country.unique())
    subs = {(c, k): data[data.country == c][preds + [k, "weight"]].dropna() for c in countries for k in outcomes}

    # observed full-sample statistics and own coefficients for every country and item
    obs, coef, cse = {}, {}, {}
    for (c, k), sub in subs.items():
        if len(sub) < 100:
            continue
        x, y, w = sub[preds].to_numpy(), sub[k].to_numpy(), sub["weight"].to_numpy()
        w = w * len(w) / w.sum()
        obs[(c, k)] = observed(x, y, w)
        b, H = logit_map(design_x(x), y, w, np.zeros(6), np.full(6, 100.0))
        coef[(c, k)], cse[(c, k)] = b, np.sqrt(np.diag(np.linalg.inv(H)))

    @lru_cache(maxsize=None)
    def own_stat(c, k, q):
        v = [obs[(c, o)][q][0] for o in outcomes if o != k and (c, o) in obs]
        v = [u for u in v if np.isfinite(u)]
        return float(np.mean(v)) if v else np.nan

    @lru_cache(maxsize=None)
    def generic_stat(c, k, q):
        v = [obs[(d, o)][q][0] for d in countries if d != c for o in outcomes if o != k and (d, o) in obs]
        v = [u for u in v if np.isfinite(u)]
        return float(np.mean(v)) if v else np.nan

    def own_coef(c, k):
        v = [coef[(c, o)] for o in outcomes if o != k and (c, o) in coef]
        return np.mean(v, axis=0)

    # prior variances, without the target's region and without the target outcome
    def prior_sd(t, k):
        keep = [c for c in countries if REGION[c] != REGION[t]]
        sd = {}
        for name, stat in (("own", own_stat), ("generic", generic_stat)):
            sd[name] = {}
            for q in QS:
                e2, s2 = [], []
                for c in keep:
                    for o in outcomes:
                        if o == k or (c, o) not in obs:
                            continue
                        val, se = obs[(c, o)][q]
                        pr = stat(c, o, q)
                        if np.isfinite(val) and np.isfinite(pr):
                            e2.append((val - pr) ** 2); s2.append(se ** 2)
                sd[name][q] = float(np.sqrt(max(np.mean(e2) - np.mean(s2), 1e-4)))
        d2, s2 = [], []
        for c in keep:
            for o in outcomes:
                if o == k or (c, o) not in coef:
                    continue
                d2.append((coef[(c, o)] - own_coef(c, o)) ** 2); s2.append(cse[(c, o)] ** 2)
        sd["coef"] = np.sqrt(np.maximum(np.mean(d2, axis=0) - np.mean(s2, axis=0), 0.01))
        return sd

    rows = []
    for (t, k), tgt in subs.items():
        if (t, k) not in obs:
            continue
        x, y, w = tgt[preds].to_numpy(), tgt[k].to_numpy(), tgt["weight"].to_numpy()
        w = w * len(w) / w.sum()
        sd = prior_sd(t, k)
        pi = cell_weights(x, w)
        own_b = own_coef(t, k)
        base = {"target": t, "outcome": k}
        # priors alone, against the whole target
        for q in QS:
            o, ose = obs[(t, k)][q]
            if not np.isfinite(o):
                continue
            for name, stat in (("own", own_stat), ("generic", generic_stat)):
                mu, s = stat(t, k, q), sd[name][q]
                h = 1.96 * np.hypot(s, ose)
                rows.append(base | {"method": name, "n_local": 0, "quantity": q, "observed": o,
                                    "pred": mu, "low": mu - h, "high": mu + h})
        for n in N_LOCAL:
            for _ in range(a.reps):
                loc = rng.choice(len(y), n, replace=False)
                rest = np.setdiff1d(np.arange(len(y)), loc)
                ref = observed(x[rest], y[rest], w[rest])
                lo = local_estimates(x[loc], y[loc])
                # regression on the local sample, weak and own-item priors
                Xl, yl, wl = design_x(x[loc]), y[loc], np.ones(n)
                reg = {}
                for name, m, s in (("mrp", np.zeros(6), np.array([2.5] + [1.0] * 5)),
                                   ("own_mrp", own_b, sd["coef"])):
                    b, H = logit_map(Xl, yl, wl, m, s ** 2)
                    reg[name] = quantities(reg_draws(b, H, rng), np.repeat(pi, D, axis=0))
                for q in QS:
                    o, ose = ref[q]
                    if not np.isfinite(o):
                        continue
                    common = base | {"n_local": n, "quantity": q, "observed": o}
                    lv, llo, lhi = lo[q]
                    if np.isfinite(lv):
                        mid, half = (llo + lhi) / 2, np.hypot((lhi - llo) / 2, 1.96 * ose)
                        rows.append(common | {"method": "local", "pred": lv, "low": mid - half, "high": mid + half})
                        # normal update of the quantity with each prior; local variance from
                        # the Agresti-Caffo interval
                        v = ((lhi - llo) / (2 * 1.96)) ** 2
                        for name, stat in (("own", own_stat), ("generic", generic_stat)):
                            mu, s = stat(t, k, q), sd[name][q]
                            if not np.isfinite(mu):
                                continue
                            prec = 1 / s**2 + 1 / v
                            pm = (mu / s**2 + mid / v) / prec
                            h = 1.96 * np.sqrt(1 / prec + ose**2)
                            rows.append(common | {"method": f"{name}+local", "pred": pm, "low": pm - h, "high": pm + h})
                    for name, Q in reg.items():
                        vq = Q[q] + rng.normal(0, ose, size=D)
                        rows.append(common | {"method": name, "pred": float(Q[q].mean()),
                                              "low": float(np.percentile(vq, 2.5)), "high": float(np.percentile(vq, 97.5))})
        print(f"{t} {k} done", flush=True)

    cases = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    cases.to_csv(a.out / "own_items_cases.csv", index=False)
    cases["err"] = (cases.pred - cases.observed).abs() * 100
    cases["cov"] = (cases.low <= cases.observed) & (cases.observed <= cases.high)
    cases["width"] = (cases.high - cases.low) * 100
    cases["kind"] = np.where(cases.quantity == "level", "level", "gap")
    # each target-outcome weighted equally, as in the joint-model summary
    tab = (cases.groupby(["kind", "n_local", "method", "target", "outcome"])[["err", "cov", "width"]].mean()
           .groupby(["kind", "n_local", "method"]).mean())
    tab["cov"] *= 100
    summary = {f"{kind}|{n}|{m}": {c: round(float(v), 2) for c, v in r.items()}
               for (kind, n, m), r in tab.iterrows()}
    joint = {}
    if a.joint.exists():
        js = json.loads(a.joint.read_text())
        joint = js.get("k_out_of_region", {}).get("small_cross|cross", {}).get("update", {})
    print("\nerror (points), coverage of 95% intervals (%), width (points); joint model: small cross, k out of region")
    for kind in ("gap", "level"):
        print(f"\n{kind}s")
        for n in [0] + list(N_LOCAL):
            line = f"  n={n:3d}: "
            for m in ("local", "own", "generic", "own+local", "generic+local", "mrp", "own_mrp"):
                if (kind, n, m) in tab.index:
                    r = tab.loc[(kind, n, m)]
                    line += f"{m} {r.err:4.1f} ({r['cov']:4.1f}%)  "
            if kind == "gap" and str(n) in joint:
                line += f"| joint {joint[str(n)]['mae']:4.1f} ({joint[str(n)]['coverage']:4.1f}%)"
            print(line)
    (a.out / "own_items_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("->", a.out / "own_items_summary.json")


if __name__ == "__main__":
    main()
