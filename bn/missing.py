"""Bayesian network with missing values handled inside the model.

Missing cells (coded -1) are treated as unknowns and sampled together with the
network and its parameters by data augmentation (Tanner and Wong 1987):

    1. G      | completed data   a block of structure moves (bn.core.structure_mcmc)
    2. theta  | G, completed data   Dirichlet(alpha + N)
    3. X_mis  | G, theta, X_obs  each missing cell from its full conditional, which
                                 involves only the Markov blanket of the variable:
                                 p(x_j | rest) ∝ theta_j[pa_j, x_j] prod_{c in ch(j)} theta_c[pa_c, x_c]

Under missingness at random the stationary distribution is the joint posterior of
(G, theta, X_mis) given X_obs, so the posterior of the network and the posterior
predictive of synthetic data both carry the uncertainty about the missing values.
Complete-case analysis, by contrast, discards every respondent with one missing
answer and is biased unless the missing values are missing completely at random.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bn.core import LocalScore, sample_cpts, structure_mcmc


def _cfg(Y: np.ndarray, cols, arity) -> np.ndarray:
    if not cols:
        return np.zeros(len(Y), dtype=np.int64)
    return np.ravel_multi_index(Y[:, list(cols)].T, arity[list(cols)])


def impute(Y: np.ndarray, miss: np.ndarray, dag: tuple, cpts: list, arity: np.ndarray,
           rng: np.random.Generator) -> np.ndarray:
    """One Gibbs sweep over the missing cells, column by column (random order).
    Returns the indices of the columns that were updated."""
    d = Y.shape[1]
    children = [[c for c in range(d) if j in dag[c]] for j in range(d)]
    cols = [j for j in range(d) if miss[:, j].any()]
    for j in rng.permutation(cols):
        rows = np.flatnonzero(miss[:, j])
        Yr = Y[rows].copy()
        logp = np.empty((len(rows), arity[j]))
        lt = [np.log(cpts[c]) for c in [j] + children[j]]
        for x in range(arity[j]):
            Yr[:, j] = x
            lp = lt[0][_cfg(Yr, dag[j], arity), x]
            for t, c in enumerate(children[j], start=1):
                lp = lp + lt[t][_cfg(Yr, dag[c], arity), Yr[:, c]]
            logp[:, x] = lp
        g = logp + rng.gumbel(size=logp.shape)          # Gumbel-max draw from the conditional
        Y[rows, j] = g.argmax(1)
    return cols


@dataclass
class DAResult:
    dags: list = field(default_factory=list)
    cpts: list = field(default_factory=list)
    log_post: list = field(default_factory=list)
    imputed: list = field(default_factory=list)      # optional completed datasets
    seconds: float = 0.0


def da_mcmc(data: np.ndarray, arity, sweeps: int, burn: int, thin: int, rng: np.random.Generator,
            gamma: float = 0.0, alpha: float = 1.0, max_parents: int = 3, allowed=None,
            struct_steps: int = 20, gibbs_prob: float = 0.3, init_dag=None,
            keep_imputed: int = 0) -> DAResult:
    """Joint posterior of (G, theta, X_mis). `data` uses -1 for a missing cell."""
    import time
    t0 = time.time()
    arity = np.asarray(arity, dtype=np.int64)
    data = np.asarray(data, dtype=np.int64)
    miss = data < 0
    Y = data.copy()
    for j in range(Y.shape[1]):                     # start from draws of the observed marginal
        if miss[:, j].any():
            obs = data[~miss[:, j], j]
            Y[miss[:, j], j] = rng.choice(obs, miss[:, j].sum())
    score = LocalScore(Y, arity, alpha=alpha, gamma=gamma, max_parents=max_parents, allowed=allowed)
    parents = [set() for _ in range(Y.shape[1])] if init_dag is None else [set(p) for p in init_dag]
    out = DAResult()
    for s in range(sweeps):
        res = structure_mcmc(score, struct_steps, struct_steps, 1, rng, init=parents, gibbs_prob=gibbs_prob)
        dag = res.final
        parents = [set(p) for p in dag]
        cpts = sample_cpts(score, dag, rng)
        if s >= burn and (s - burn) % thin == 0:
            out.dags.append(dag); out.cpts.append(cpts)
            out.log_post.append(sum(score(j, dag[j]) for j in range(len(dag))))
            if len(out.imputed) < keep_imputed:
                out.imputed.append(Y.copy())
        changed = impute(Y, miss, dag, cpts, arity, rng)
        score.refresh(Y, changed)
    out.seconds = time.time() - t0
    return out
