"""Diagnose d = 7 failures: is it mixing (chain misses a better-scoring true DAG) or the posterior itself?"""
import sys
from collections import Counter
import numpy as np
from bn.core import LocalScore, structure_mcmc, forward_sample, equivalence_key
from bn.martins_replication import true_model, posterior_classes, MCMC

GP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
n = int(sys.argv[1]); gammas = [float(g) for g in sys.argv[2].split(",")]
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 7)
for rep in range(6):
    dag, cpts = true_model(7, rng)
    data = forward_sample(dag, cpts, np.full(7, 2), n, rng)
    tset = [set(p) for p in dag]; tkey = equivalence_key(tset)
    for g in gammas:
        sc = LocalScore(data, np.full(7, 2), gamma=g, max_parents=3)
        lp_true = sum(sc(j, tset[j]) for j in range(7))
        out = []
        for init in (None, tset):
            res = structure_mcmc(sc, 60000, 10000, 100, rng, init=init, gibbs_prob=GP)
            cls, keys = posterior_classes(res.samples)
            best = max(range(len(res.samples)), key=lambda i: res.log_post[i])
            out.append((cls[tkey] / len(keys), max(res.log_post) - lp_true, res.samples[best]))
        print(f"rep {rep} g={g:4.0f}: P(true) empty-start {out[0][0]:.2f} true-start {out[1][0]:.2f} | "
              f"best-found minus true logpost {out[0][1]:+.1f} / {out[1][1]:+.1f}")
        if out[0][0] < 0.5:
            print("     true:", [tuple(p) for p in dag]); print("     best:", out[0][2])
    # marginal strength of each true edge
    eff = []
    for j, ps in enumerate(dag):
        for p in ps:
            eff.append((p, j, round(abs(data[data[:, p] == 1, j].mean() - data[data[:, p] == 0, j].mean()), 3)))
    print("     marginal effects", eff)
