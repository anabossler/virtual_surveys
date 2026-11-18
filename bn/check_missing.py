"""Check of the data-augmentation sampler on simulated data with MAR missingness.

A categorical network with 8 nodes is simulated (n = 3000). Two attitude-like
variables lose values with a probability that depends on an always-observed
variable (education-like), i.e. missing at random but not completely at random.
The posterior predictive marginals and one conditional are compared with the
truth for: the complete data, data augmentation, and complete-case analysis.
"""
import sys
import numpy as np
from bn.core import LocalScore, structure_mcmc, sample_cpts, forward_sample
from bn.missing import da_mcmc

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
arity = np.array([2, 4, 3, 2, 3, 2, 4, 2])
# 0 sex, 1 age group, 2 education (parent 1), 3 recycles (2), 4 worry (1,2), 5 accepts (3,4), 6 region, 7 urban (6)
dag = ((), (), (1,), (2,), (1, 2), (3, 4), (), (6,))

def rand_cpt(r, q, rng, conc=1.0):
    return rng.dirichlet(np.full(r, conc), size=q)

def draw_cpts(rng):
    return [rand_cpt(arity[j], int(np.prod(arity[list(p)])) if p else 1, rng) for j, p in enumerate(dag)]

cpts = draw_cpts(rng)
n = 3000
X = forward_sample(dag, cpts, arity, n, rng)
# MAR: worry (4) and accepts (5) missing more often at low education (2 == 0)
PM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
pm = np.where(X[:, 2] == 0, PM, 0.05)
Xm = X.copy()
for j in (4, 5):
    Xm[rng.random(n) < pm, j] = -1
print("missing share:", (Xm < 0).mean(0).round(3), " complete cases", (Xm >= 0).all(1).mean().round(3))

def truth_stats(cpts):
    big = forward_sample(dag, cpts, arity, 400_000, np.random.default_rng(0))
    return stats(big)

def stats(Y):
    return {"P(accepts)": (Y[:, 5] == 1).mean(), "P(worry=2)": (Y[:, 4] == 2).mean(),
            "P(accepts|edu=0)": (Y[Y[:, 2] == 0, 5] == 1).mean()}

def posterior_predictive(dags, cptss, rng):
    out = []
    for g, th in zip(dags, cptss):
        out.append(stats(forward_sample(g, th, arity, n, rng)))
    return {k: (np.mean([o[k] for o in out]), np.percentile([o[k] for o in out], [5, 95])) for k in out[0]}

T = truth_stats(cpts)
gamma = 2.0
def complete_run(D):
    sc = LocalScore(D, arity, gamma=gamma)
    r = structure_mcmc(sc, 20000, 2000, 60, rng, gibbs_prob=0.3)
    return r.samples, [sample_cpts(sc, g, rng) for g in r.samples]

full = posterior_predictive(*complete_run(X), rng)
cc = posterior_predictive(*complete_run(Xm[(Xm >= 0).all(1)]), rng)
da = da_mcmc(Xm, arity, sweeps=3000, burn=500, thin=10, rng=rng, gamma=gamma, struct_steps=20)
dap = posterior_predictive(da.dags, da.cpts, rng)
print(f"data augmentation: {len(da.dags)} draws in {da.seconds:.0f}s")
print(f"{'statistic':18s} {'truth':>7s} | {'full data':>20s} | {'data augmentation':>20s} | {'complete cases':>20s}")
for k in T:
    f = lambda r: f"{r[k][0]:.3f} [{r[k][1][0]:.3f},{r[k][1][1]:.3f}]"
    print(f"{k:18s} {T[k]:7.3f} | {f(full):>20s} | {f(dap):>20s} | {f(cc):>20s}")
