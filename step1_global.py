"""Step 1 (global): posterior samples of the full 90-node DAG by parallel-tempered MCMC.

Why
---
Step 1 learns each thematic block by MC3 and joins the blocks afterwards: GLASSO
proposes undirected cross-block pairs, a pairwise BDeu bootstrap keeps some, their
direction follows column order, cycles are broken, and step4 later drops edges
that violate the causal tiers and samples the rest independently. The held-out
evaluation of network_value.py showed that the resulting population loses most
cross-block associations: several predictor pairs are d-separated in every
sampled graph.

This script samples p(G | X) over all variables at once (Martins et al. 2024,
Algorithm 1, step 2), so cross-block edges and their directions are chosen by
the same BDeu score as within-block edges. It reuses, unchanged, the data
preparation, the available-case BDeu score and the causal constraints of
step1.py, and starts every chain from the Step 1 backbone.

Output
------
  step1_global_samples.json      list of M sampled DAGs (edge lists), for step4 /
                                 network_value.py --structure global
  step1_global_edges.csv         posterior inclusion probability of every edge
  step1_global_diagnostics.json  acceptance, swaps, score trace, ESS, split-half
                                 agreement of edge probabilities, d-connection of
                                 key variable pairs

Usage
-----
    python step1_global.py                      # defaults: 1M iterations, 8 temperatures, M = 500
    python step1_global.py --iters 600000 --seed 7
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from step1 import (
    PATH_NUM,
    ROOTS,
    BlockScorer,
    build_forbidden_edges,
    classify_variables,
    discretize_for_bdeu,
    filter_collinear,
    filter_substantive_variables,
    load_microdata_cis,
)
from step4 import is_edge_forbidden

KEY_PAIRS = [("SEX", "V52"), ("SEX", "V15"), ("BIRTH", "V52"), ("BIRTH", "V15"),
             ("ESTUDIOS", "V15"), ("ESTUDIOS", "V52"), ("V52", "V15"), ("V37", "V15"),
             ("LEFT_RIGHT", "V15"), ("URBRURAL", "V15"), ("V11", "V15"), ("TOPBOT", "ESTUDIOS")]


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def prepare_data():
    """Exactly the data preparation of step1.py (substantive filter, collinearity, discretisation)."""
    df = quiet(load_microdata_cis, PATH_NUM)
    df = quiet(filter_substantive_variables, df)
    df = quiet(filter_collinear, df, threshold=0.95)
    return quiet(discretize_for_bdeu, df, classify_variables(df), n_bins=5)


def coarsen(df_disc, k):
    """Collapse each variable to at most k ordered groups of roughly equal frequency.

    Used only for scoring structures: with n ~ 2,000 and up to ten categories, the
    BDeu parameter penalty can exceed the gain from a weak association, so such edges
    are not selected. Conditional tables are still estimated on the original categories.
    """
    out = df_disc.copy()
    for col in out.columns:
        x = out[col]
        levels = np.sort(x.dropna().unique())
        if len(levels) <= k:
            continue
        freq = x.value_counts(normalize=True).reindex(levels).fillna(0).cumsum().to_numpy()
        group = np.minimum((freq * k - 1e-9).astype(int), k - 1)
        _, group = np.unique(group, return_inverse=True)   # drop empty groups
        out[col] = x.map(dict(zip(levels, group))).astype("Int64")
    return out


def forbidden_rule(variables):
    """Union of the step1 block constraints, the step4 causal tiers, and exogenous roots."""
    block_fb = set().union(*build_forbidden_edges().values())

    def forbidden(u, v):
        return ((u, v) in block_fb or is_edge_forbidden(u, v)
                or (v in ROOTS and u not in ROOTS)
                or {u, v} == {"SEX", "BIRTH"})
    return {(u, v) for u in variables for v in variables if u != v and forbidden(u, v)}


def initial_edges(path, variables, forbidden, max_indeg):
    """Step 1 backbone, restricted to admissible edges, acyclic and within the in-degree bound."""
    if not os.path.exists(path):
        return []
    de = pd.read_csv(path)
    prob = pd.to_numeric(de.get("probability", 0.5), errors="coerce").fillna(0.5)
    rows = sorted(zip(de["source"], de["target"], prob), key=lambda r: (-r[2], r[0], r[1]))
    g = nx.DiGraph()
    g.add_nodes_from(variables)
    for u, v, _ in rows:
        if u in g and v in g and (u, v) not in forbidden and g.in_degree(v) < max_indeg \
                and not nx.has_path(g, v, u):
            g.add_edge(u, v)
    return list(g.edges())


class Chain:
    """One tempered chain over DAGs: parent sets, edge list and reachability graph."""

    def __init__(self, edges, variables, scorer, gamma):
        self.vars, self.scorer, self.gamma = variables, scorer, gamma
        self.g = nx.DiGraph()
        self.g.add_nodes_from(variables)
        self.g.add_edges_from(edges)
        self.edges = list(edges)
        self.pos = {e: i for i, e in enumerate(self.edges)}
        self.score = sum(self.local(v, list(self.g.predecessors(v))) for v in variables)

    def local(self, v, parents):
        return self.scorer.local_score(v, parents) - self.gamma * len(parents)

    def _add(self, e):
        self.g.add_edge(*e)
        self.pos[e] = len(self.edges)
        self.edges.append(e)

    def _remove(self, e):
        self.g.remove_edge(*e)
        i = self.pos.pop(e)
        last = self.edges.pop()
        if i < len(self.edges):
            self.edges[i] = last
            self.pos[last] = i

    def step(self, rng, temp, forbidden, max_indeg):
        """One Metropolis-Hastings move (add, remove or reverse). Returns (proposed, accepted)."""
        r = rng.random()
        if r < 0.4 or not self.edges:                       # add u -> v
            u, v = self.vars[rng.integers(len(self.vars))], self.vars[rng.integers(len(self.vars))]
            if u == v or self.g.has_edge(u, v) or (u, v) in forbidden \
                    or self.g.in_degree(v) >= max_indeg or nx.has_path(self.g, v, u):
                return False, False
            old = list(self.g.predecessors(v))
            delta = self.local(v, old + [u]) - self.local(v, old)
            if np.log(rng.random()) < delta / temp:
                self._add((u, v))
                self.score += delta
                return True, True
            return True, False
        u, v = self.edges[rng.integers(len(self.edges))]
        if r < 0.7:                                          # remove u -> v
            old = list(self.g.predecessors(v))
            delta = self.local(v, [p for p in old if p != u]) - self.local(v, old)
            if np.log(rng.random()) < delta / temp:
                self._remove((u, v))
                self.score += delta
                return True, True
            return True, False
        if (v, u) in forbidden or self.g.in_degree(u) >= max_indeg:   # reverse
            return False, False
        self._remove((u, v))
        if nx.has_path(self.g, u, v):
            self._add((u, v))
            return False, False
        old_v, old_u = list(self.g.predecessors(v)) + [u], list(self.g.predecessors(u))
        delta = (self.local(v, [p for p in old_v if p != u]) - self.local(v, old_v)
                 + self.local(u, old_u + [v]) - self.local(u, old_u))
        if np.log(rng.random()) < delta / temp:
            self._add((v, u))
            self.score += delta
            return True, True
        self._add((u, v))
        return True, False


def effective_size(x):
    x = np.asarray(x, float) - np.mean(x)
    if x.var() == 0:
        return float(len(x))
    ac = np.correlate(x, x, mode="full")[len(x) - 1:] / (x.var() * len(x))
    tau = 1 + 2 * sum(a for a in ac[1:min(200, len(ac))] if a > 0.05)
    return float(len(x) / tau)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=1_000_000)
    ap.add_argument("--burn-in", type=int, default=None, help="default: one third of --iters")
    ap.add_argument("--samples", type=int, default=500, help="DAGs kept from the cold chain (Martins: >= 500)")
    ap.add_argument("--max-indegree", type=int, default=3, help="as in step1.py")
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="penalising prior per parent (Martins eq. 5); Step 1 calibrated 0 for 13 of 15 blocks")
    ap.add_argument("--temps", type=float, nargs="+", default=[1.0, 1.17, 1.37, 1.6, 1.87, 2.19, 2.56, 3.0],
                    help="geometric ladder; widely spaced ladders gave swap rates near 1%%")
    ap.add_argument("--coarsen", type=int, default=0,
                    help="score structures on variables collapsed to at most this many groups (0 = off)")
    ap.add_argument("--init", default="dag_edges_3391.csv")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("BVS_SEED", "42")))
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    burn = args.burn_in if args.burn_in is not None else args.iters // 3
    lag = max(1, (args.iters - burn) // args.samples)
    rng = np.random.default_rng(args.seed)

    print("Step 1 (global) -- MC3 over the full DAG")
    df_disc = prepare_data()
    if args.coarsen:
        df_disc = coarsen(df_disc, args.coarsen)
        print(f"  structure scored on variables collapsed to <= {args.coarsen} groups")
    variables = sorted(df_disc.columns)
    forbidden = forbidden_rule(variables)
    scorer = BlockScorer(df_disc, variables, ess=10.0, gamma=0.0)
    init = initial_edges(args.init, variables, forbidden, args.max_indegree)
    print(f"  {len(df_disc)} respondents, {len(variables)} variables, {len(forbidden)} forbidden ordered pairs")
    print(f"  start: {len(init)} edges from {args.init}; max in-degree {args.max_indegree}; gamma {args.gamma}")
    print(f"  {args.iters} iterations, burn-in {burn}, lag {lag}, temperatures {args.temps}")

    chains = [Chain(init, variables, scorer, args.gamma) for _ in args.temps]
    proposed = np.zeros(len(chains)); accepted = np.zeros(len(chains))
    swaps = [0, 0]
    trace, samples = [], []
    start = time.time()
    for it in range(args.iters):
        for k, ch in enumerate(chains):
            p, a = ch.step(rng, args.temps[k], forbidden, args.max_indegree)
            proposed[k] += p; accepted[k] += a
        if it % 10 == 0 and len(chains) > 1:
            k = rng.integers(len(chains) - 1)
            t1, t2 = args.temps[k], args.temps[k + 1]
            log_swap = (chains[k].score - chains[k + 1].score) * (1 / t2 - 1 / t1)
            swaps[0] += 1
            if np.log(rng.random()) < log_swap:
                chains[k], chains[k + 1] = chains[k + 1], chains[k]
                swaps[1] += 1
        if it % 100 == 0:
            trace.append(chains[0].score)
        if it >= burn and (it - burn) % lag == 0 and len(samples) < args.samples:
            samples.append(sorted(chains[0].edges))
        if (it + 1) % max(1, args.iters // 20) == 0:
            print(f"  {it + 1:>8d} it | cold score {chains[0].score:,.1f} | edges {len(chains[0].edges)} | "
                  f"{time.time() - start:.0f}s", flush=True)

    # Posterior summaries
    counts = Counter(e for s in samples for e in s)
    m = len(samples)
    edges_df = pd.DataFrame([{"source": u, "target": v, "probability": c / m} for (u, v), c in counts.items()])
    edges_df = edges_df.sort_values("probability", ascending=False)
    half = m // 2
    p1 = Counter(e for s in samples[:half] for e in s)
    p2 = Counter(e for s in samples[half:] for e in s)
    keys = set(p1) | set(p2)
    split_diff = max((abs(p1[e] / max(half, 1) - p2[e] / max(m - half, 1)) for e in keys), default=0.0)

    dsep = nx.is_d_separator if hasattr(nx, "is_d_separator") else nx.d_separated
    dconn = {}
    for a, b in KEY_PAIRS:
        if a in variables and b in variables:
            hits = 0
            for s in samples[:: max(1, m // 100)]:
                g = nx.DiGraph(s); g.add_nodes_from(variables)
                hits += not dsep(g, {a}, {b}, set())
            dconn[f"{a}-{b}"] = hits / len(samples[:: max(1, m // 100)])

    diag = {
        "method": "parallel-tempered MCMC over the full DAG, started at the Step 1 backbone",
        "seed": args.seed, "iterations": args.iters, "burn_in": burn, "lag": lag, "samples": m,
        "temperatures": args.temps, "max_indegree": args.max_indegree, "gamma": args.gamma,
        "coarsen": args.coarsen,
        "acceptance": dict(zip(map(str, args.temps), (accepted / np.maximum(proposed, 1)).round(4).tolist())),
        "swap_rate": swaps[1] / max(swaps[0], 1),
        "edges_mean": float(np.mean([len(s) for s in samples])),
        "ess_edge_count": effective_size([len(s) for s in samples]),
        "ess_score_trace_post_burn": effective_size(trace[burn // 100:]),
        "split_half_max_edge_prob_diff": split_diff,
        "edges_with_prob_ge_0.5": int((edges_df["probability"] >= 0.5).sum()),
        "d_connected_share": dconn,
        "runtime_s": round(time.time() - start),
    }
    os.makedirs(args.out, exist_ok=True)
    tag = f"_c{args.coarsen}" if args.coarsen else ""
    with open(os.path.join(args.out, f"step1_global{tag}_samples.json"), "w", encoding="utf-8") as fh:
        json.dump(samples, fh)
    edges_df.to_csv(os.path.join(args.out, f"step1_global{tag}_edges.csv"), index=False)
    with open(os.path.join(args.out, f"step1_global{tag}_diagnostics.json"), "w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2)
    print(json.dumps({k: diag[k] for k in ("acceptance", "swap_rate", "edges_mean", "ess_edge_count",
                                           "split_half_max_edge_prob_diff", "d_connected_share")}, indent=2))
    print(f"Saved step1_global{tag}_samples.json, _edges.csv, _diagnostics.json")


if __name__ == "__main__":
    main()
