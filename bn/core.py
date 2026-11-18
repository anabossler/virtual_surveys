"""Discrete Bayesian networks: marginal likelihood, modular priors, structure MCMC,
conjugate parameter posteriors, and forward sampling.

Variables are integer-coded 0..r_j-1. The parameter prior is Dirichlet with all
hyperparameters equal to `alpha` (alpha = 1 is the uniform Beta(1,1) prior of
Martins et al. 2024 for binary data). The structure prior is modular,
p(G) proportional to prod_j exp(-gamma * |pa(X_j)|**power), the penalising class
proposed by Martins et al. (gamma = 0 gives the uniform prior).

Structure sampling uses a Metropolis-Hastings sampler on DAGs: an unordered pair
{i, j} is chosen uniformly and its state (no edge, i->j, j->i) is proposed to one
of the two other states with equal probability. The proposal is symmetric, so the
acceptance ratio is the posterior ratio; proposals that create a cycle or exceed
the maximum number of parents are rejected. The posterior is the same target as
the Gibbs sampler of Goudie and Mukherjee (2016) used by Martins et al.; only the
mixing differs.

With probability `gibbs_prob` an iteration instead performs a block Gibbs step on
two rows of the adjacency matrix, as in Goudie and Mukherjee (2016) with m = 2: two
nodes i, j are chosen at random and their parent sets are drawn jointly from their
full conditional, over every pair of sets of at most k parents that keeps the graph
acyclic. Acyclicity is checked exactly with reachability bitmasks in the graph
without the edges into i and j. The block step can rewire a v-structure (for
example move the parents {2, 4} from node 3 to the reverse configuration) in one
move, which the pair move can only do by crossing a valley of the posterior when
the structure penalty is strong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.special import gammaln


# ----------------------------------------------------------------------- scores
class LocalScore:
    """Cached log marginal likelihood + log prior of each (node, parent set)."""

    def __init__(self, data: np.ndarray, arity: np.ndarray, alpha: float = 1.0,
                 gamma: float = 0.0, power: float = 1.0, max_parents: int = 3,
                 allowed: np.ndarray | None = None):
        self.data = np.asarray(data, dtype=np.int64)
        self.n, self.d = self.data.shape
        # allowed[p, c] is False for an edge p -> c ruled out a priori (e.g. by tiers)
        self.allowed = (np.ones((self.d, self.d), bool) if allowed is None
                        else np.asarray(allowed, bool).copy())
        np.fill_diagonal(self.allowed, False)
        self.arity = np.asarray(arity, dtype=np.int64)
        self.alpha, self.gamma, self.power, self.k = alpha, gamma, power, max_parents
        self._cache: dict[tuple[int, tuple[int, ...]], float] = {}

    def counts(self, j: int, parents: tuple[int, ...]) -> np.ndarray:
        """Counts N[q, l] for parent configuration q and value l of node j."""
        r = self.arity[j]
        if parents:
            ar = self.arity[list(parents)]
            cfg = np.ravel_multi_index(self.data[:, list(parents)].T, ar)
            q = int(np.prod(ar))
        else:
            cfg, q = np.zeros(self.n, dtype=np.int64), 1
        return np.bincount(cfg * r + self.data[:, j], minlength=q * r).reshape(q, r)

    def log_ml(self, j: int, parents: tuple[int, ...]) -> float:
        n = self.counts(j, parents).astype(float)
        a = self.alpha
        r = n.shape[1]
        return float(np.sum(gammaln(r * a) - gammaln(r * a + n.sum(1)))
                     + np.sum(gammaln(a + n) - gammaln(a)))

    def refresh(self, data: np.ndarray, changed_cols) -> None:
        """Replace the data and drop cached scores that involve a changed column."""
        self.data = np.asarray(data, dtype=np.int64)
        ch = set(int(c) for c in changed_cols)
        if not ch:
            return
        self._cache = {k: v for k, v in self._cache.items() if k[0] not in ch and not ch.intersection(k[1])}
        rows = self.__dict__.get("_rows", {})
        for j in list(rows):
            if j in ch or any(ch.intersection(c) for c in rows[j][0] if c):
                del rows[j]

    def __call__(self, j: int, parents) -> float:
        key = (j, tuple(sorted(parents)))
        if key not in self._cache:
            self._cache[key] = self.log_ml(j, key[1]) - self.gamma * len(key[1]) ** self.power
        return self._cache[key]


# ------------------------------------------------------------------------ DAGs
def reaches(parents: list[set], src: int, dst: int) -> bool:
    """True if there is a directed path src -> ... -> dst."""
    children = [[] for _ in parents]
    for v, ps in enumerate(parents):
        for p in ps:
            children[p].append(v)
    stack, seen = [src], {src}
    while stack:
        u = stack.pop()
        if u == dst:
            return True
        for c in children[u]:
            if c not in seen:
                seen.add(c); stack.append(c)
    return False


def descendants(parents: list[set], j: int) -> set:
    """Nodes reachable from j along directed edges (children, grandchildren, ...)."""
    children = [set() for _ in parents]
    for v, ps in enumerate(parents):
        for u in ps:
            children[u].add(v)
    out, stack = set(), [j]
    while stack:
        for c in children[stack.pop()]:
            if c not in out:
                out.add(c); stack.append(c)
    return out


def _row_candidates(score: LocalScore, j: int):
    """All parent sets of node j with at most k members, and their local scores
    (cached on the score object)."""
    cache = score.__dict__.setdefault("_rows", {})
    if j not in cache:
        others = [v for v in range(score.d) if score.allowed[v, j]]
        c = [cc for m in range(min(score.k, len(others)) + 1) for cc in combinations(others, m)]
        cache[j] = (c, np.array([score(j, cc) for cc in c]))
    return cache[j]


def _mask(nodes) -> int:
    m = 0
    for v in nodes:
        m |= 1 << v
    return m


def _masks(cands) -> np.ndarray:
    return np.array([_mask(c) for c in cands], dtype=np.int64)


def topological_order(parents: list[set]) -> list[int]:
    d = len(parents)
    indeg = [len(p) for p in parents]
    children = [[] for _ in range(d)]
    for v, ps in enumerate(parents):
        for p in ps:
            children[p].append(v)
    order, ready = [], [v for v in range(d) if indeg[v] == 0]
    while ready:
        u = ready.pop()
        order.append(u)
        for c in children[u]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) != d:
        raise ValueError("graph has a cycle")
    return order


def dag_key(parents: list[set]) -> tuple:
    return tuple(tuple(sorted(p)) for p in parents)


def equivalence_key(parents: list[set]) -> tuple:
    """Markov equivalence class: skeleton plus v-structures (Verma and Pearl 1990)."""
    d = len(parents)
    skel = frozenset(frozenset((p, v)) for v in range(d) for p in parents[v])
    vs = set()
    for v in range(d):
        ps = sorted(parents[v])
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                x, y = ps[a], ps[b]
                if frozenset((x, y)) not in skel:
                    vs.add((x, v, y))
    return (tuple(sorted(tuple(sorted(e)) for e in skel)), tuple(sorted(vs)))


@dataclass
class MCMCResult:
    samples: list[tuple]            # dag keys after burn-in and thinning
    log_post: list[float]
    accept_rate: float
    trace: list[float] = field(default_factory=list)
    final: tuple = ()               # dag key of the last state


def structure_mcmc(score: LocalScore, iters: int, burn: int, thin: int,
                   rng: np.random.Generator, init: list[set] | None = None,
                   gibbs_prob: float = 0.0, max_block: int = 250_000) -> MCMCResult:
    d, k = score.d, score.k
    parents = [set() for _ in range(d)] if init is None else [set(p) for p in init]
    local = [score(j, parents[j]) for j in range(d)]
    total = sum(local)
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    acc, samples, lp, trace = 0, [], [], []
    for it in range(iters):
        if gibbs_prob > 0 and rng.random() < gibbs_prob:
            i, j = (int(v) for v in rng.choice(d, 2, replace=False))
            (ci, si), (cj, sj) = _row_candidates(score, i), _row_candidates(score, j)
            if len(ci) * len(cj) > max_block:
                # single-row Gibbs step on node j when the joint table would be too large
                desc = descendants(parents, j)
                mj = _masks(cj)
                ok = (mj & _mask(desc)) == 0
                lw = np.where(ok, sj, -np.inf)
                w = np.exp(lw - lw.max()); w /= w.sum()
                b = int(rng.choice(len(cj), p=w))
                if set(cj[b]) != parents[j]:
                    acc += 1
                total += sj[b] - local[j]
                parents[j] = set(cj[b]); local[j] = float(sj[b])
                if it % 100 == 0:
                    trace.append(total)
                if it >= burn and (it - burn) % thin == 0:
                    samples.append(dag_key(parents)); lp.append(total)
                continue
            g0 = list(parents); g0[i] = set(); g0[j] = set()
            ri, rj = _mask(descendants(g0, i)), _mask(descendants(g0, j))
            mi, mj = _masks(ci), _masks(cj)
            bi, bj = 1 << i, 1 << j
            ok = (((mi & (ri | bi)) == 0)[:, None] & ((mj & (rj | bj)) == 0)[None, :]
                  & ~(((mj & (ri | bi)) != 0)[None, :] & ((mi & (rj | bj)) != 0)[:, None]))
            lw = np.where(ok, si[:, None] + sj[None, :], -np.inf)
            w = np.exp(lw - lw.max()).ravel(); w /= w.sum()
            a, b = divmod(int(rng.choice(w.size, p=w)), len(cj))
            new_i, new_j = set(ci[a]), set(cj[b])
            if new_i != parents[i] or new_j != parents[j]:
                acc += 1
            total += si[a] + sj[b] - local[i] - local[j]
            parents[i], parents[j] = new_i, new_j
            local[i], local[j] = float(si[a]), float(sj[b])
            if it % 100 == 0:
                trace.append(total)
            if it >= burn and (it - burn) % thin == 0:
                samples.append(dag_key(parents)); lp.append(total)
            continue
        i, j = pairs[rng.integers(len(pairs))]
        if i in parents[j]:
            state = 1          # i -> j
        elif j in parents[i]:
            state = 2          # j -> i
        else:
            state = 0
        new = [s for s in (0, 1, 2) if s != state][rng.integers(2)]
        pj, pi = set(parents[j]), set(parents[i])
        pj.discard(i); pi.discard(j)
        ok = True
        if new == 1:
            pj.add(i)
            if len(pj) > k or not score.allowed[i, j]:
                ok = False
        elif new == 2:
            pi.add(j)
            if len(pi) > k or not score.allowed[j, i]:
                ok = False
        if ok:
            trial = list(parents)
            trial[i], trial[j] = pi, pj
            # cycle check: the new edge must not close a path
            if new == 1:
                trial_wo = list(trial); trial_wo[j] = pj - {i}
                ok = not reaches(trial_wo, j, i)
            elif new == 2:
                trial_wo = list(trial); trial_wo[i] = pi - {j}
                ok = not reaches(trial_wo, i, j)
        if ok:
            si, sj = score(i, pi), score(j, pj)
            delta = si + sj - local[i] - local[j]
            if np.log(rng.random()) < delta:
                parents[i], parents[j] = pi, pj
                local[i], local[j] = si, sj
                total += delta
                acc += 1
        if it % 100 == 0:
            trace.append(total)
        if it >= burn and (it - burn) % thin == 0:
            samples.append(dag_key(parents)); lp.append(total)
    return MCMCResult(samples, lp, acc / max(iters, 1), trace, dag_key(parents))


# -------------------------------------------------------------- parameters
def sample_cpts(score: LocalScore, dag: tuple, rng: np.random.Generator, mle: bool = False):
    """Conditional probability tables theta_j[q, :] ~ Dirichlet(alpha + N[q, :]) (or MLE)."""
    cpts = []
    for j, ps in enumerate(dag):
        n = score.counts(j, ps).astype(float)
        if mle:
            tot = n.sum(1, keepdims=True)
            th = np.where(tot > 0, n / np.maximum(tot, 1), 1.0 / n.shape[1])
        else:
            g = rng.gamma(score.alpha + n)
            th = g / g.sum(1, keepdims=True)
        cpts.append(th)
    return cpts


def forward_sample(dag: tuple, cpts: list[np.ndarray], arity: np.ndarray, n: int,
                   rng: np.random.Generator) -> np.ndarray:
    d = len(dag)
    out = np.zeros((n, d), dtype=np.int64)
    for j in topological_order([set(p) for p in dag]):
        ps = dag[j]
        if ps:
            cfg = np.ravel_multi_index(out[:, list(ps)].T, arity[list(ps)])
        else:
            cfg = np.zeros(n, dtype=np.int64)
        probs = cpts[j][cfg]
        u = rng.random(n)[:, None]
        out[:, j] = (u > np.cumsum(probs, axis=1)).sum(1).clip(max=arity[j] - 1)
    return out
