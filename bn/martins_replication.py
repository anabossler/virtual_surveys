"""Replication of the simulation study of Martins, Goncalves and Galletti (2024),
"Generation and analysis of synthetic data via Bayesian networks: a robust
approach for uncertainty quantification via Bayesian paradigm" (arXiv:2402.17915).

Scenarios (their Table 2): binary networks with
    d = 3: X1, X3, X2|X1                                   n = 500, 1000, 5000
    d = 4: X1, X4, X2|X1, X3|X4                            n = 1000, 5000
    d = 7: X1, X3, X6, X7, X2|X1, X5|X6,X7, X4|X3,X5       n = 2000, 5000
true probabilities drawn from U(0.2, 0.8), 10 replications each, Beta(1,1)
priors on the parameters, and two structure priors: uniform (P2) and the
penalising prior exp(-gamma |pa|) (P1) with gamma calibrated as the smallest
value giving the true network posterior probability above 0.85.

Methods:
    S1  full Bayes: posterior predictive of the statistics (their Algorithm 1),
        mean and 98% HPD interval over M posterior draws of (G, theta, Y)
    S2  five synthetic datasets from the highest-posterior network and the MLE of
        theta, combined as mean +/- t_{4, 0.01} s / sqrt(5)
    S3  one dataset from sequential CART synthesis (the default of synthpop;
        Python implementation, donors sampled within leaves)

Statistics: h1 overlap (OIC) of the 95% interval of theta = P(X2=1|X1=0) from
the original and the synthetic data; h2 the MLE of theta; h3 the p-value of the
chi-square test of independence between X1 and X2 (and X1, X5 for d = 7).

Network posterior is summarised by Markov equivalence class (skeleton and
v-structures), which generalises the two-variable redundancies merged in the
paper. Structure MCMC mixes pair moves with two-row block Gibbs steps
(Goudie and Mukherjee 2016, m = 2); see bn/core.py.

In addition to the per-dataset OIC of the paper, the script reports a combined
OIC: the original interval is compared with one interval built from all synthetic
datasets of a method by the combining rule for fully synthetic data (Raghunathan,
Reiter and Rubin 2003; Reiter 2002), T = (1 + 1/M) b - u_bar, with the fallback
T = u_bar when T <= 0. A single posterior predictive dataset carries both
parameter and sampling variability, so its estimate differs from the original one
by about sqrt(2) standard errors on average and its per-dataset OIC is about 0.71
even when the model is correct. The combining rule removes the extra sampling
variability and is the inference a final user would actually make.

Usage: python -m bn.martins_replication [--reps 10] [--quick]
Output: bn_output/martins_replication.json and figures.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, t as student_t
from sklearn.tree import DecisionTreeClassifier

from bn.core import (LocalScore, equivalence_key, forward_sample, sample_cpts,
                     structure_mcmc)

SCENARIOS = {
    3: [(), (0,), ()],
    4: [(), (0,), (3,), ()],
    7: [(), (0,), (), (2, 4), (5, 6), (), ()],
}
SIZES = {3: (500, 1000, 5000), 4: (1000, 5000), 7: (2000, 5000)}
GIBBS_PROB = 0.3
GAMMA_GRID = {3: np.arange(0, 9, 1.0), 4: np.arange(0, 25, 2.0), 7: np.arange(0, 31, 2.0)}
MCMC = {3: (20000, 2000, 36), 4: (30000, 3000, 54), 7: (60000, 10000, 100)}  # iters, burn, thin


# ------------------------------------------------------------------ helpers
MIN_EFFECT = 0.15


def true_model(d, rng):
    """True network with probabilities from U(0.2, 0.8). For a node with parents, the
    draw is repeated until every parent changes the probability by at least
    MIN_EFFECT on average, so that every edge of the true network is identifiable at
    the sample sizes considered (Martins et al. report the true network as the most
    probable one in almost every replication)."""
    dag = tuple(tuple(sorted(p)) for p in SCENARIOS[d])
    cpts = []
    for ps in dag:
        while True:
            th = rng.uniform(0.2, 0.8, 2 ** len(ps))
            ok = True
            for b in range(len(ps)):
                idx = np.arange(2 ** len(ps))
                step = 2 ** (len(ps) - 1 - b)          # ravel order: first parent varies slowest
                lo = idx[(idx // step) % 2 == 0]
                if np.mean(np.abs(th[lo + step] - th[lo])) < MIN_EFFECT:
                    ok = False
            if ok:
                break
        cpts.append(np.column_stack([1 - th, th]))
    return dag, cpts


def theta_ci(y):
    m = y[:, 0] == 0
    n0 = int(m.sum())
    p = float(y[m, 1].mean()) if n0 else np.nan
    se = np.sqrt(p * (1 - p) / n0) if n0 else np.nan
    return p, p - 1.96 * se, p + 1.96 * se


def oic(a, b):
    (l1, u1), (l2, u2) = a, b
    lo, hi = max(l1, l2), min(u1, u2)
    return 0.0 if hi <= lo else 2 * (hi - lo) / ((u1 - l1) + (u2 - l2))


def chi2_p(y, a, b):
    tab = np.zeros((2, 2))
    np.add.at(tab, (y[:, a], y[:, b]), 1)
    if (tab.sum(0) == 0).any() or (tab.sum(1) == 0).any():
        return 1.0
    return float(chi2_contingency(tab, correction=False)[1])


def stats(y, ci_orig, d):
    p, lo, hi = theta_ci(y)
    out = {"oic": oic(ci_orig, (lo, hi)), "mle": p, "se": (hi - lo) / 3.92, "pv12": chi2_p(y, 0, 1)}
    if d == 7:
        out["pv15"] = chi2_p(y, 0, 4)
    return out


def hpd(x, mass=0.98):
    x = np.sort(np.asarray(x)[np.isfinite(x)])
    m = int(np.ceil(mass * len(x)))
    if m >= len(x):
        return float(x[0]), float(x[-1])
    w = x[m - 1:] - x[: len(x) - m + 1]
    i = int(np.argmin(w))
    return float(x[i]), float(x[i + m - 1])


def cart_synthesis(y, rng):
    n, d = y.shape
    out = np.zeros_like(y)
    out[:, 0] = rng.choice(y[:, 0], n, replace=True)
    for j in range(1, d):
        tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=int(rng.integers(1 << 31)))
        tree.fit(y[:, :j], y[:, j])
        leaves = tree.apply(y[:, :j])
        donors = {lf: y[leaves == lf, j] for lf in np.unique(leaves)}
        new_leaves = tree.apply(out[:, :j])
        for lf in np.unique(new_leaves):
            m = new_leaves == lf
            out[m, j] = rng.choice(donors[lf], int(m.sum()), replace=True)
    return out


def posterior_classes(samples):
    keys = [equivalence_key([set(p) for p in s]) for s in samples]
    return Counter(keys), keys


# ------------------------------------------------------------------ one run
def run_one(d, n, gamma, data, true_dag, rng, M=None):
    iters, burn, thin = MCMC[d]
    score = LocalScore(data, np.full(d, 2), alpha=1.0, gamma=gamma, max_parents=min(3, d - 1))
    res = structure_mcmc(score, iters, burn, thin, rng, gibbs_prob=GIBBS_PROB)
    cls, keys = posterior_classes(res.samples)
    tkey = equivalence_key([set(p) for p in true_dag])
    p_true = cls[tkey] / len(keys)
    top_key, top_count = cls.most_common(1)[0]
    # S1: Algorithm 1
    ci_orig = theta_ci(data)[1:]
    draws = res.samples if M is None else res.samples[:M]
    s1 = [stats(forward_sample(g, sample_cpts(score, g, rng), score.arity, n, rng), ci_orig, d) for g in draws]
    # S2: highest-posterior network (most frequent DAG within the top class), MLE theta, 5 datasets
    in_top = [s for s, k in zip(res.samples, keys) if k == top_key]
    map_dag = Counter(in_top).most_common(1)[0][0]
    th_mle = sample_cpts(score, map_dag, rng, mle=True)
    s2 = [stats(forward_sample(map_dag, th_mle, score.arity, n, rng), ci_orig, d) for _ in range(5)]
    # S3: CART synthesis, two replications (labelled like P1 and P2 in the paper)
    s3 = [stats(cart_synthesis(data, rng), ci_orig, d) for _ in range(2)]
    return {"p_true": p_true, "true_is_top": bool(top_key == tkey), "accept": res.accept_rate,
            "n_draws": len(draws), "s1": s1, "s2": s2, "s3": s3}


def summarise_method(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return v


def calibrate_gamma(d, n, rng, reps=None, target=0.85):
    """Smallest gamma on the grid whose mean posterior probability of the true class
    exceeds target; if none does, the smallest gamma within 0.03 of the best mean.
    Each grid value is evaluated on fresh replications (10 for d < 7, 6 for d = 7)."""
    reps = reps or (10 if d < 7 else 6)
    curve = {}
    for g in GAMMA_GRID[d]:
        ps = []
        for _ in range(reps):
            dag, cpts = true_model(d, rng)
            data = forward_sample(dag, cpts, np.full(d, 2), n, rng)
            iters, burn, thin = MCMC[d]
            score = LocalScore(data, np.full(d, 2), gamma=float(g), max_parents=min(3, d - 1))
            res = structure_mcmc(score, iters, burn, thin, rng, gibbs_prob=GIBBS_PROB)
            cls, keys = posterior_classes(res.samples)
            ps.append(cls[equivalence_key([set(p) for p in dag])] / len(keys))
        curve[float(g)] = float(np.mean(ps))
        if curve[float(g)] > target:
            return float(g), curve
    best = max(curve.values())
    return float(min(g for g, v in curve.items() if v >= best - 0.03)), curve


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("bn_output"))
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--quick", action="store_true", help="d = 3 and 4 only, 3 replications")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    dims = (3, 4) if args.quick else (3, 4, 7)
    reps = 3 if args.quick else args.reps
    result = {"seed": args.seed, "reps": reps, "scenarios": {}}
    t0 = time.time()
    for d in dims:
        for n in SIZES[d]:
            gamma, curve = calibrate_gamma(d, n, rng)
            print(f"d={d} n={n}: calibrated gamma {gamma}  ({time.time() - t0:.0f}s)", flush=True)
            entry = {"gamma": gamma, "calibration_curve": curve, "reps": []}
            for r in range(reps):
                dag, cpts = true_model(d, rng)
                data = forward_sample(dag, cpts, np.full(d, 2), n, rng)
                truth = stats(data, theta_ci(data)[1:], d)
                truth["ci"] = list(theta_ci(data)[1:])
                rec = {"truth": truth}
                for label, g in (("P1", gamma), ("P2", 0.0)):
                    rec[label] = run_one(d, n, g, data, dag, rng)
                entry["reps"].append(rec)
                print(f"   rep {r + 1}: P(true) P1 {rec['P1']['p_true']:.2f} P2 {rec['P2']['p_true']:.2f}", flush=True)
            result["scenarios"][f"d{d}_n{n}"] = entry
            (args.out / "martins_replication.json").write_text(json.dumps(result, default=float) + "\n")
    print_summary(result)


def method_summary(rec, prior, stat, d):
    """Point estimate and interval of each method for one replication."""
    r = rec[prior]
    s1 = np.array([x[stat] for x in r["s1"]], dtype=float)
    s2 = np.array([x[stat] for x in r["s2"]], dtype=float)
    s3 = np.array([x[stat] for x in r["s3"]], dtype=float)
    lo1, hi1 = hpd(s1)
    half = student_t.ppf(0.99, 4) * s2.std(ddof=1) / np.sqrt(5)
    return {"S1": (s1.mean(), lo1, hi1), "S2": (s2.mean(), s2.mean() - half, s2.mean() + half),
            "S3": (s3[0 if prior == "P1" else 1],)}


def combined_oic(rec, prior):
    """OIC of the original interval against the interval combined over all synthetic
    datasets of S1 and of S2 by the fully-synthetic combining rule."""
    out = {}
    for m in ("s1", "s2"):
        q = np.array([x["mle"] for x in rec[prior][m]]); u = np.array([x["se"] for x in rec[prior][m]]) ** 2
        M, b, ubar = len(q), q.var(ddof=1), u.mean()
        T = (1 + 1 / M) * b - ubar
        T = T if T > 0 else ubar
        df = max(4.0, (M - 1) * (1 - M * ubar / ((M + 1) * b)) ** 2) if T != ubar else 1e6
        half = student_t.ppf(0.975, df) * np.sqrt(T)
        out[m.upper()] = oic(tuple(rec["truth"]["ci"]), (q.mean() - half, q.mean() + half))
    return out


def print_summary(result):
    print("\nNetwork posterior (cf. Table 3): replications with true class on top - mean P(true) | others")
    for key, e in result["scenarios"].items():
        for prior in ("P1", "P2"):
            top = [r[prior]["p_true"] for r in e["reps"] if r[prior]["true_is_top"]]
            oth = [r[prior]["p_true"] for r in e["reps"] if not r[prior]["true_is_top"]]
            print(f"  {key:10s} {prior}: {len(top):2d} - {np.mean(top) if top else float('nan'):.2f} / "
                  f"{np.mean(oth) if oth else float('nan'):.2f}   gamma={e['gamma'] if prior == 'P1' else 0}")
    print("\nSynthetic-data statistics, mean over replications "
          "(|error| vs original; S1 interval covers original?)")
    for key, e in result["scenarios"].items():
        d = int(key.split("_")[0][1:])
        for stat in ["oic", "mle", "pv12"] + (["pv15"] if d == 7 else []):
            for prior in ("P1", "P2"):
                err = {"S1": [], "S2": [], "S3": []}
                cov = {"S1": [], "S2": []}
                for rec in e["reps"]:
                    t = 1.0 if stat == "oic" else rec["truth"][stat]
                    ms = method_summary(rec, prior, stat, d)
                    for m, v in ms.items():
                        err[m].append(abs(v[0] - t))
                        if m in cov:
                            cov[m].append(v[1] <= t <= v[2])
                print(f"  {key:10s} {stat:5s} {prior}: |err| S1 {np.mean(err['S1']):.4f} S2 {np.mean(err['S2']):.4f} "
                      f"S3 {np.mean(err['S3']):.4f}" + ("" if stat == "oic" else
                      f"   covers truth S1 {np.mean(cov['S1']):.0%} S2 {np.mean(cov['S2']):.0%}"))
                if stat == "oic" and "ci" in e["reps"][0]["truth"]:
                    c = [combined_oic(rec, prior) for rec in e["reps"]]
                    print(f"  {key:10s} oic-combined {prior}: S1 {np.mean([x['S1'] for x in c]):.3f} "
                          f"S2 {np.mean([x['S2'] for x in c]):.3f}")


if __name__ == "__main__":
    main()
