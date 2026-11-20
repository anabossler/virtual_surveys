"""Convergence check of the two data-augmentation chains saved by bn/cis_joint.py.

Edge probabilities by direction are not a valid convergence check for a Bayesian
network: networks in the same Markov equivalence class imply the same
distribution, and a chain may settle on either orientation of an edge that the
data cannot orient. This script compares the chains on what is identified:

  * skeleton: probability of an edge between two variables, either direction;
  * v-structures (CPDAG): probability of each compelled collider;
  * the predictive distribution: pairwise joint distributions of the variables,
    compared by total variation distance between the two chains.

Usage: python -m bn.chain_check [bn_cis_output/cis_joint_posterior.pkl]
"""
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from bn.core import forward_sample


def skeleton(dags, d):
    S = np.zeros((d, d))
    for g in dags:
        for c, ps in enumerate(g):
            for p in ps:
                S[min(p, c), max(p, c)] += 1
    return S / len(dags)


def colliders(dags):
    out = {}
    for g in dags:
        par = [set(p) for p in g]
        seen = set()
        for c, ps in enumerate(par):
            for a, b in combinations(sorted(ps), 2):
                if b not in par[a] and a not in par[b]:
                    seen.add((a, c, b))
        for v in seen:
            out[v] = out.get(v, 0) + 1
    return {k: v / len(dags) for k, v in out.items()}


def pairwise(dags, cpts, arity, rng, n=3000, draws=100):
    d = len(arity)
    idx = rng.choice(len(dags), min(draws, len(dags)), replace=False)
    tabs = {}
    for i in idx:
        y = forward_sample(dags[i], cpts[i], arity, n, rng)
        for a, b in combinations(range(d), 2):
            t = np.bincount(y[:, a] * arity[b] + y[:, b], minlength=arity[a] * arity[b]) / n
            tabs.setdefault((a, b), []).append(t)
    return {k: np.mean(v, 0) for k, v in tabs.items()}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bn_cis_output/cis_joint_posterior.pkl")
    post = pickle.loads(path.read_bytes())
    names, arity = post["names"], np.asarray(post["arity"])
    h = len(post["dags"]) // 2
    ch = [(post["dags"][:h], post["cpts"][:h]), (post["dags"][h:], post["cpts"][h:])]
    d = len(names)
    S1, S2 = skeleton(ch[0][0], d), skeleton(ch[1][0], d)
    ds = np.abs(S1 - S2)
    print(f"skeleton (edge in either direction), chain 1 vs 2: max |diff| = {ds.max():.2f}; "
          f"pairs differing by > 0.2: {(ds > 0.2).sum()}")
    for i, j in zip(*np.where(ds > 0.2)):
        print(f"   {names[i]} - {names[j]}: {S1[i, j]:.2f} vs {S2[i, j]:.2f}")
    C1, C2 = colliders(ch[0][0]), colliders(ch[1][0])
    keys = set(C1) | set(C2)
    dc = {k: abs(C1.get(k, 0) - C2.get(k, 0)) for k in keys}
    big = sorted((k for k in keys if dc[k] > 0.2), key=lambda k: -dc[k])
    print(f"v-structures, chain 1 vs 2: max |diff| = {max(dc.values(), default=0):.2f}; differing by > 0.2: {len(big)}")
    for a, c, b in big[:10]:
        print(f"   {names[a]} -> {names[c]} <- {names[b]}: {C1.get((a, c, b), 0):.2f} vs {C2.get((a, c, b), 0):.2f}")
    rng = np.random.default_rng(1)
    P1, P2 = pairwise(*ch[0], arity, rng), pairwise(*ch[1], arity, rng)
    tv = {k: 0.5 * np.abs(P1[k] - P2[k]).sum() for k in P1}
    worst = sorted(tv, key=lambda k: -tv[k])[:5]
    print(f"pairwise predictive distributions, chain 1 vs 2: median TV = {np.median(list(tv.values())):.3f}, "
          f"max TV = {max(tv.values()):.3f}")
    print("   largest:", ", ".join(f"{names[a]}-{names[b]} {tv[(a, b)]:.3f}" for a, b in worst))
    print("(each table averages 100 draws of 3000; a TV below about 0.02 does not change any reported share)")


if __name__ == "__main__":
    main()
