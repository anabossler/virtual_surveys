"""Joint Bayesian network on CIS 3391 with missing values inside the model (M2).

Runs locally on the CIS microdata (metadata/3391_num.csv, not redistributable).
Outputs go to bn_cis_output/ (git-ignored); the printed summary contains only
aggregate quantities and can be pasted.

Coding
  * Missing-value codes. Besides the codes removed by step4 (96-99, ...), a value
    is treated as missing when it is 77 or more, or when it lies after a gap in the
    list of observed values (e.g. 8 and 9 on a 1-5 scale, 7 for SEX, 9 for
    ESTUDIOS). Every code treated as missing is printed.
  * BIRTH is recoded into four age groups; ESTUDIOS into four levels
    (1-2 primary or less, 3-4 lower secondary, 5-6 upper secondary and vocational,
    7 higher), so that "higher education" is the transfer predictor's category.
  * Other variables with more than --max-cats categories are merged into adjacent
    groups of similar size (order kept).

Model
  * Tiers of step4.CAUSAL_TIERS: no edge from a later tier to an earlier one.
  * Candidate parents (sparse-candidate restriction, Friedman, Nachman and Pe'er
    1999): each node may only take parents among the --pool variables with the
    highest mutual information with it. The posterior is exact within that space.
  * Penalising modular prior exp(-gamma |pa|), gamma = 2 log d by default.
  * Tables drawn from the weighted pseudo-posterior with PESOFIN (Savitsky and
    Toth 2016); the network score is unweighted.
  * Two data-augmentation chains (one from the empty network, one from a random
    network), so convergence can be judged by their agreement; then the same
    model on complete cases only.

Usage
  python -m bn.cis_joint --sweeps 200 --test     # timing check, a few minutes
  python -m bn.cis_joint                         # full run
  python -m bn.cis_joint --vars SEX BIRTH ESTUDIOS V15 V52 ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from bn.core import (LocalScore, equivalence_key, forward_sample, reaches, sample_cpts,
                     structure_mcmc)
from bn.missing import da_mcmc

SURVEY_YEAR = 2023
AGE_BREAKS = (18, 30, 45, 60, 200)
TRANSFER_VARS = ("SEX", "BIRTH", "ESTUDIOS", "V52", "V15")
# left out by default: NAT_DEGR duplicates ESTUDIOS with 18 codes; NAT_RINC duplicates
# NAT_INC with 14% missing; V16, V50, V51 have contiguous codes up to 9 or 12, so
# their "don't know" codes cannot be told from the scale without the codebook.
DEFAULT_EXCLUDE = ("NAT_DEGR", "NAT_RINC", "V16", "V50", "V51")


def missing_codes(values: np.ndarray) -> list:
    """Codes after the first gap in the sorted observed values, and codes >= 77."""
    vals = sorted(set(values))
    out = [v for v in vals if v >= 77]
    core = [v for v in vals if v < 77]
    for a, b in zip(core, core[1:]):
        if b - a > 1:
            out += [v for v in core if v >= b]
            break
    return sorted(out)


def merge_adjacent(share: pd.Series, k: int) -> list[list]:
    """Merge ordered categories into at most k adjacent groups of similar size."""
    vals = list(share.index)
    if len(vals) <= k:
        return [[v] for v in vals]
    cum = share.cumsum().to_numpy()
    cuts = [int(np.searchsorted(cum, q, side="left")) for q in np.arange(1, k) / k]
    groups, start = [], 0
    for c in sorted(set(cuts)):
        if c + 1 > start:
            groups.append(vals[start:c + 1]); start = c + 1
    groups.append(vals[start:])
    return [g for g in groups if g]


def load(path: Path, variables, max_cats: int):
    import step4
    cfg = step4.Config()
    cfg.PATH_CIS = str(path)
    raw = step4.load_cis_microdata(cfg)
    tiers = step4.CAUSAL_TIERS
    wanted = variables or [v for v in tiers if v not in DEFAULT_EXCLUDE]
    names = [v for v in wanted if v in raw.columns]
    absent = [v for v in wanted if v not in raw.columns]
    coding, dropped_codes = {}, {}
    X = np.full((len(raw), len(names)), -1, dtype=np.int64)
    arity = []
    for j, v in enumerate(names):
        x = raw[v].copy()
        mc = [] if v == "BIRTH" else missing_codes(x.dropna().to_numpy())
        if mc:
            dropped_codes[v] = [int(c) for c in mc]
            x = x.where(~x.isin(mc))
        if v == "BIRTH":
            x = pd.cut(SURVEY_YEAR - x, AGE_BREAKS, right=False, labels=False)
            groups = [[i] for i in range(4)]
            labels = ["18-29", "30-44", "45-59", "60+"]
        elif v == "ESTUDIOS":
            groups = [[1, 2], [3, 4], [5, 6], [7]]
            labels = ["1-2", "3-4", "5-6", "7 higher"]
        else:
            share = x.value_counts(normalize=True).sort_index()
            groups = merge_adjacent(share, max_cats)
            labels = ["+".join(str(int(a)) for a in g) for g in groups]
        lut = {float(val): g for g, grp in enumerate(groups) for val in grp}
        X[:, j] = [lut.get(float(t), -1) if pd.notna(t) else -1 for t in x]
        arity.append(len(groups))
        coding[v] = labels
    tier = np.array([tiers.get(v, 999) for v in names])
    allowed = tier[:, None] <= tier[None, :]
    w = pd.read_csv(path, sep=";", encoding="latin1", low_memory=False, usecols=["PESOFIN"])["PESOFIN"]
    w = pd.to_numeric(w.astype(str).str.replace(",", "."), errors="coerce").loc[raw.index].to_numpy()
    w = np.where(np.isfinite(w) & (w > 0), w, np.nanmedian(w))
    w = w * len(w) / w.sum()
    return names, X, np.array(arity), allowed, coding, dropped_codes, absent, w


def mutual_information(a: np.ndarray, b: np.ndarray, ra: int, rb: int) -> float:
    ok = (a >= 0) & (b >= 0)
    t = np.bincount(a[ok] * rb + b[ok], minlength=ra * rb).reshape(ra, rb) + 0.5
    p = t / t.sum()
    return float((p * np.log(p / p.sum(1, keepdims=True) / p.sum(0, keepdims=True))).sum())


def candidate_pool(X, arity, allowed, pool):
    d = X.shape[1]
    MI = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            MI[i, j] = MI[j, i] = mutual_information(X[:, i], X[:, j], arity[i], arity[j])
    A = np.zeros_like(allowed)
    for c in range(d):
        cand = [p for p in np.argsort(-MI[:, c]) if allowed[p, c]][:pool]
        A[cand, c] = True
    return A, MI


def random_dag(allowed, max_parents, rng, n_edges):
    d = len(allowed)
    parents = [set() for _ in range(d)]
    edges = [(p, c) for p in range(d) for c in range(d) if allowed[p, c]]
    for k in rng.permutation(len(edges)):
        p, c = edges[k]
        if len(parents[c]) < max_parents and not reaches(parents, c, p):
            parents[c].add(p)
            n_edges -= 1
            if n_edges == 0:
                break
    return parents


def edge_probs(dags, d):
    P = np.zeros((d, d))
    for g in dags:
        for c, ps in enumerate(g):
            for p in ps:
                P[p, c] += 1
    return P / max(len(dags), 1)


def marginals(dags, cpts, arity, idx, rng, n=4000):
    out = {j: [] for j in idx}
    for g, th in zip(dags, cpts):
        y = forward_sample(g, th, arity, n, rng)
        for j in idx:
            out[j].append(np.bincount(y[:, j], minlength=arity[j]) / n)
    return {j: (np.mean(v, 0), np.percentile(v, 5, 0), np.percentile(v, 95, 0)) for j, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cis", type=Path, default=Path("metadata/3391_num.csv"))
    ap.add_argument("--vars", nargs="*", default=None)
    ap.add_argument("--max-cats", type=int, default=5)
    ap.add_argument("--pool", type=int, default=8, help="candidate parents per node")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--max-parents", type=int, default=3)
    ap.add_argument("--sweeps", type=int, default=6000)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--struct-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3391)
    ap.add_argument("--out", type=Path, default=Path("bn_cis_output"))
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        a.burn, a.thin = a.sweeps // 3, 2
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(exist_ok=True)

    names, X, arity, tiers_allowed, coding, codes, absent, weight = load(a.cis, a.vars, a.max_cats)
    d, n = len(names), len(X)
    gamma = a.gamma if a.gamma is not None else round(2 * np.log(d), 2)
    allowed, MI = candidate_pool(X, arity, tiers_allowed, a.pool)
    miss = X < 0
    print(f"\n{n} respondents, {d} variables; gamma = {gamma}; max parents = {a.max_parents}; "
          f"candidate parents per node = {a.pool}")
    if absent:
        print("not in the file:", absent)
    print("codes treated as missing:", json.dumps(codes))
    print("missing share by variable (%):",
          {v: round(float(100 * miss[:, j].mean()), 1) for j, v in enumerate(names) if miss[:, j].any()})
    print(f"complete cases: {(~miss).all(1).sum()} ({100 * (~miss).all(1).mean():.1f}%)")
    print("categories:", json.dumps(coding, ensure_ascii=False))

    chains = []
    starts = [None, random_dag(allowed, a.max_parents, rng, d)]
    for c, init in enumerate(starts):
        print(f"\nchain {c + 1} ({'empty' if init is None else 'random'} start)", flush=True)
        chains.append(da_mcmc(X, arity, sweeps=a.sweeps, burn=a.burn, thin=a.thin, rng=rng,
                              gamma=gamma, max_parents=a.max_parents, allowed=allowed,
                              struct_steps=a.struct_steps, gibbs_prob=0.3, init_dag=init,
                              verbose_every=max(a.sweeps // 10, 1), label=f"chain {c + 1}",
                              weights=weight))
    t_da = sum(ch.seconds for ch in chains)
    dags = chains[0].dags + chains[1].dags
    cpts = chains[0].cpts + chains[1].cpts
    print(f"\ndata augmentation: 2 chains x {len(chains[0].dags)} draws, {t_da / 60:.1f} min "
          f"({1000 * t_da / (2 * a.sweeps):.0f} ms per sweep)")
    # convergence on identified quantities (skeleton, v-structures, predictive);
    # edge directions within an equivalence class are not identified
    from bn.chain_check import colliders, pairwise, skeleton
    ds = np.abs(skeleton(chains[0].dags, d) - skeleton(chains[1].dags, d))
    c1, c2 = colliders(chains[0].dags), colliders(chains[1].dags)
    dc = max((abs(c1.get(k, 0) - c2.get(k, 0)) for k in set(c1) | set(c2)), default=0.0)
    p1 = pairwise(chains[0].dags, chains[0].cpts, arity, rng)
    p2 = pairwise(chains[1].dags, chains[1].cpts, arity, rng)
    tv = [0.5 * np.abs(p1[k] - p2[k]).sum() for k in p1]
    print(f"chain 1 vs chain 2: skeleton max |diff| {ds.max():.2f} ({(ds > 0.2).sum()} pairs > 0.2); "
          f"v-structures max |diff| {dc:.2f}; pairwise predictive TV median {np.median(tv):.3f}, max {max(tv):.3f}")

    t0 = time.time()
    keep = (~miss).all(1)
    cc, wcc = X[keep], weight[keep] * keep.sum() / weight[keep].sum()
    sc = LocalScore(cc, arity, gamma=gamma, max_parents=a.max_parents, allowed=allowed)
    it = a.sweeps * a.struct_steps
    r = structure_mcmc(sc, it, it // 3, max(1, (it - it // 3) // len(chains[0].dags)), rng, gibbs_prob=0.3)
    cc_cpts = [sample_cpts(sc, g, rng, weights=wcc) for g in r.samples]
    print(f"complete cases: {len(r.samples)} draws, {(time.time() - t0) / 60:.1f} min")

    P, Pc = edge_probs(dags, d), edge_probs(r.samples, d)
    rows = sorted(((names[p], names[c], P[p, c], Pc[p, c]) for p in range(d) for c in range(d)
                   if max(P[p, c], Pc[p, c]) >= 0.2), key=lambda t: -t[2])
    print("\nedges with posterior probability >= 0.2 (data augmentation | complete cases):")
    for s, t, p, q in rows:
        print(f"  {s:>9s} -> {t:<9s} {p:.2f} | {q:.2f}")

    cls = Counter(equivalence_key([set(p) for p in g]) for g in dags)
    print("\nmost probable equivalence classes (data augmentation):",
          ", ".join(f"{c / len(dags):.3f}" for _, c in cls.most_common(3)),
          f"; distinct classes visited: {len(cls)}")

    print("\nparents of the transfer predictors (probability >= 0.3):")
    for v in TRANSFER_VARS:
        if v in names:
            c = names.index(v)
            print(f"  {v}: {[(names[p], round(float(P[p, c]), 2)) for p in np.argsort(-P[:, c]) if P[p, c] >= 0.3]}")

    idx = [names.index(v) for v in TRANSFER_VARS if v in names]
    mda = marginals(dags, cpts, arity, idx, rng)
    mcc = marginals(r.samples, cc_cpts, arity, idx, rng)
    print("\nposterior predictive marginals (mean [5%, 95%]); observed weighted share among answers:")
    for j in idx:
        o = ~miss[:, j]
        obs = np.bincount(X[o, j], weights=weight[o], minlength=arity[j]) / weight[o].sum()
        for k in range(arity[j]):
            print(f"  {names[j]:>8s}={coding[names[j]][k]:<8s} observed {obs[k]:.3f} | "
                  f"DA {mda[j][0][k]:.3f} [{mda[j][1][k]:.3f},{mda[j][2][k]:.3f}] | "
                  f"CC {mcc[j][0][k]:.3f} [{mcc[j][1][k]:.3f},{mcc[j][2][k]:.3f}]")

    with open(a.out / "cis_joint_posterior.pkl", "wb") as f:
        pickle.dump({"names": names, "arity": arity, "coding": coding, "gamma": gamma, "weighted": True,
                     "allowed": allowed, "dags": dags, "cpts": cpts,
                     "cc_dags": r.samples, "cc_cpts": cc_cpts}, f)
    pd.DataFrame(P, index=names, columns=names).round(3).to_csv(a.out / "edge_probabilities_da.csv")
    pd.DataFrame(Pc, index=names, columns=names).round(3).to_csv(a.out / "edge_probabilities_cc.csv")
    print(f"\nsaved to {a.out}/ (do not commit: derived from microdata)")


if __name__ == "__main__":
    main()
