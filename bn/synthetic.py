"""Fully synthetic population with the outcome node, and combining rules (M4).

Each of M posterior draws (G, theta, transport) generates one synthetic dataset
of size n: the CIS variables from the network and the outcome Y from its
transferred table. A user who receives only the synthetic datasets estimates a
quantity q (here: segment shares of Y and gaps between segments) in each dataset,
with its usual variance u, and combines them with the rule for fully synthetic
data (Raghunathan, Reiter and Rubin 2003; Reiter 2002):

    q_bar = mean q_m,   b = var q_m,   u_bar = mean u_m
    T = (1 + 1/M) b - u_bar   (T = u_bar if negative; Reiter 2002)
    interval q_bar +/- t_nu sqrt(T),  nu = (M - 1)(1 - M u_bar / ((M + 1) b))^2

The combined interval is compared with the posterior predictive interval of the
population quantity computed directly from the draws (bn/transfer_node.py): if the
synthetic datasets carry the uncertainty correctly, the two agree.

Usage
  python -m bn.synthetic --m 100 --n 2133           # combining-rule check
  python -m bn.synthetic --m 10 --release 5         # also write 5 synthetic datasets
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from bn.core import forward_sample
from bn.transfer_node import (GAPS, PARENT_VARS, PREDICTORS, SEGMENTS, calibrate, design,
                              draw_transport, load_transport, predictor_maps, summarise)


def synthetic_draw(post, idx, T, k, n, n_pop, rng):
    """One synthetic dataset (CIS variables plus Y) and the population quantities of
    the same draw."""
    names, arity = post["names"], np.asarray(post["arity"])
    maps = predictor_maps(post["coding"])
    dag, cpts = post["dags"][idx], post["cpts"][idx]
    pop = forward_sample(dag, cpts, arity, n_pop, rng)
    xp = design(pop, names, maps)
    b, delta, p0 = draw_transport(T, k, rng)
    c = calibrate(xp @ b, p0) + delta
    pp = norm.cdf(c + xp @ b)
    truth = {s: float(pp[f(xp, pop, names)].mean()) for s, f in SEGMENTS.items()}
    Y = forward_sample(dag, cpts, arity, n, rng)
    x = design(Y, names, maps)
    y = (rng.random(n) < norm.cdf(c + x @ b)).astype(int)
    return Y, x, y, truth


def estimates(x, y, Y, names):
    out = {}
    for s, f in SEGMENTS.items():
        m = f(x, Y, names)
        k = m.sum()
        p = y[m].mean() if k else np.nan
        out[s] = (p, p * (1 - p) / max(k, 1), k)
    for g, (s1, s2) in GAPS.items():
        out[g] = (out[s1][0] - out[s2][0], out[s1][1] + out[s2][1], np.nan)
    return out


def combine(q: np.ndarray, u: np.ndarray, level=0.95):
    M = len(q)
    qb, b, ub = q.mean(), q.var(ddof=1), u.mean()
    T = (1 + 1 / M) * b - ub
    if T <= 0:
        T, nu = ub, 1e6
    else:
        nu = max(3.0, (M - 1) * (1 - M * ub / ((M + 1) * b)) ** 2)
    h = student_t.ppf(0.5 + level / 2, nu) * np.sqrt(T)
    return qb, qb - h, qb + h, T <= ub


def combine_logit(p: np.ndarray, u: np.ndarray, n_eff: np.ndarray, level=0.95):
    """Combining rule for a share on the logit scale, back-transformed.

    Small groups give skewed sampling distributions near 0 or 1, where the
    normal-scale interval runs outside [0, 1]. Each dataset's share is moved
    away from the boundary by half a case, transformed, and given the
    delta-method variance u / (p (1 - p))^2."""
    pa = (p * n_eff + 0.5) / (n_eff + 1.0)
    z = np.log(pa / (1 - pa))
    uz = u / (pa * (1 - pa)) ** 2
    zb, lo, hi, fallback = combine(z, uz, level)
    inv = lambda v: 1 / (1 + np.exp(-v))
    return inv(zb), inv(lo), inv(hi), fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--posterior", type=Path, default=Path("bn_cis_output/cis_joint_posterior.pkl"))
    ap.add_argument("--coefficients", type=Path, default=Path("transfer_coefficients.csv"))
    ap.add_argument("--issp", type=Path, default=Path("transport_output/issp_transport.json"))
    ap.add_argument("--attenuation", type=Path, default=Path("transport_output/attenuation_by_predictor.json"))
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--hier", type=Path, default=None)
    ap.add_argument("--m", type=int, default=100, help="number of synthetic datasets")
    ap.add_argument("--n", type=int, default=2133, help="size of each synthetic dataset")
    ap.add_argument("--n-pop", type=int, default=20000)
    ap.add_argument("--release", type=int, default=0, help="write this many synthetic datasets")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", type=Path, default=Path("bn_cis_output"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    post = pickle.loads(a.posterior.read_bytes())
    T = load_transport(a.coefficients, a.issp, a.attenuation, a.hier)
    picks = rng.choice(len(post["dags"]), a.m, replace=len(post["dags"]) < a.m)
    est, truths = [], []
    for r, i in enumerate(picks):
        Y, x, y, truth = synthetic_draw(post, i, T, a.k, a.n, a.n_pop, rng)
        est.append(estimates(x, y, Y, post["names"]))
        truths.append(truth)
        if r < a.release:
            df = pd.DataFrame({v: [post["coding"][v][c] for c in Y[:, j]] for j, v in enumerate(post["names"])})
            df["acceptance_food_packaging"] = y
            df.to_csv(a.out / f"synthetic_{r + 1:02d}.csv", index=False)
    td = pd.DataFrame(truths)
    for g, (s1, s2) in GAPS.items():
        td[g] = td[s1] - td[s2]
    direct = summarise(td).set_index("quantity")
    rows = []
    for qn in est[0]:
        q = np.array([e[qn][0] for e in est]); u = np.array([e[qn][1] for e in est])
        ok = np.isfinite(q)
        qb, lo, hi, fallback = combine(q[ok], u[ok])
        if qn in GAPS:
            llo, lhi = np.nan, np.nan
        else:
            ne = np.array([e[qn][2] for e in est], dtype=float)[ok]
            # the delta-method variance needs u > 0: use (p(1-p))/n with p moved off the boundary
            pa = (q[ok] * ne + 0.5) / (ne + 1.0)
            _, llo, lhi, _ = combine_logit(q[ok], pa * (1 - pa) / np.maximum(ne, 1), ne)
        d = direct.loc[qn]
        rows.append({"quantity": qn, "combined": qb, "low": lo, "high": hi,
                     "logit_low": llo, "logit_high": lhi,
                     "direct": d["mean"], "direct_low": d["low"], "direct_high": d["high"],
                     "min_group_n": float(np.nanmin([e[qn][2] for e in est])) if qn not in GAPS else np.nan,
                     "b_over_ubar": q[ok].var(ddof=1) / u[ok].mean(), "T_fallback": fallback})
    out = pd.DataFrame(rows)
    print(f"{a.m} synthetic datasets of n = {a.n}; k = {a.k}")
    print("combined from synthetic data vs posterior predictive of the population quantity (points);")
    print("logit_low/high: the same rule on the logit scale, for shares only:")
    show = out.copy()
    for c in ("combined", "low", "high", "logit_low", "logit_high", "direct", "direct_low", "direct_high"):
        show[c] = (show[c] * 100).round(1)
    print(show.round(2).to_string(index=False))
    out.to_csv(a.out / f"synthetic_combining_k{a.k:g}.csv", index=False)
    if a.release:
        print(f"wrote {a.release} synthetic datasets to {a.out}/ (check disclosure risk before release)")


if __name__ == "__main__":
    main()
