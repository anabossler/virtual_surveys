"""Retrospective validation of the joint model on ISSP 2020 (M5).

Same benchmark as backtest_stress.py (16 European countries, five proxy outcomes,
every ordered country pair), with the joint model in place of the transfer alone:

  * Target composition. For each target country a Bayesian network over the five
    transfer predictors (female, under 30, higher education, recycles, worry on
    five levels) is estimated with its missing answers inside the model
    (bn/missing.py) and survey weights. Each posterior draw gives the probability
    of the 80 predictor profiles, so population quantities are exact sums.
  * Transport layer (hierarchical). b_j = a beta_j + eta_j, a ~ N(a_mean, a_sd),
    eta_j ~ N(0, tau_beta^2), level shift delta ~ N(0, tau_level^2); intercept
    calibrated to the source base rate. The hyperparameters are estimated without
    the target's European region (leave-region-out) in two ways:
        same   between countries, same outcome (as backtest_transfer.py)
        cross  between countries and between outcomes, the kind of transfer of
               the application (coefficients of one item used for another)
    No inflation factor is applied.
  * Local update. A local sample of n = 25, 50, 100, 200 target respondents
    with the outcome observed reweights the draws by their likelihood (sampling
    importance resampling) and the resampled particles are then moved by
    random-walk Metropolis steps (resample-move, Gilks and Berzuini 2001), which
    removes the weight degeneracy of plain resampling when the local sample is
    large (applied when the effective sample size of the weights is below 500); the result is compared with the remaining respondents, as is the
    local sample alone.
  * Intervals predict the statistic observed in the reference sample (the whole
    target for the prior, the remaining respondents after a local sample), so the
    sampling error of the reference is added to every draw; the interval of the
    local sample alone is widened in the same way. Without this, coverage falls
    with n for reasons unrelated to the model.
  * Inflation k of the transport layer is evaluated on a grid with the same
    random numbers, and chosen out of region: for each European region, the
    smallest k whose gap coverage reaches 95% in the other regions.

Scenarios: same (full source, same item) and small_cross (source of 272,
coefficients from another item, base rate of the projected item).

Usage: python -m bn.validate_issp --issp metadata/ZA7650_v2-0-0.dta
Outputs (transport_output): joint_validation_cases.csv (aggregate per case),
joint_validation_summary.json, hierarchical_transport.json (hyperparameters
estimated on all pairs, for bn/transfer_node.py).
"""
from __future__ import annotations

import argparse
import json
import pickle
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest_stress import REGION, fit
from backtest_transfer import EUROPE, weighted_share
from bn.missing import da_mcmc
from transport_issp import OUTCOMES, PREDICTORS, design

SEED = 2020
N_LOCAL = (25, 50, 100, 200)
ARITY = np.array([2, 2, 2, 2, 5])
CELLS = np.array(list(product(*[range(r) for r in ARITY])))           # 80 x 5 category codes
XCELL = np.column_stack([CELLS[:, :4], (CELLS[:, 4] - 2) / 2.0]).astype(float)
WORRY_ON = XCELL[:, 4] > 0                                            # V15 in {4, 5}, as segments()


# ----------------------------------------------------------------- target network
def categorical(sub: pd.DataFrame) -> np.ndarray:
    X = np.full((len(sub), 5), -1, dtype=np.int64)
    for j, p in enumerate(PREDICTORS[:4]):
        v = sub[p].to_numpy()
        X[~np.isnan(v), j] = v[~np.isnan(v)].astype(int)
    w = sub["env_worry"].to_numpy()
    X[~np.isnan(w), 4] = np.round(w[~np.isnan(w)] * 2 + 2).astype(int)
    return X


def cell_probs(dag, cpts) -> np.ndarray:
    p = np.ones(len(CELLS))
    for j, ps in enumerate(dag):
        cfg = np.ravel_multi_index(CELLS[:, list(ps)].T, ARITY[list(ps)]) if ps else np.zeros(len(CELLS), int)
        p *= cpts[j][cfg, CELLS[:, j]]
    return p


def target_network(sub: pd.DataFrame, rng, sweeps: int) -> np.ndarray:
    X = categorical(sub)
    w = sub["weight"].to_numpy(float)
    w = w * len(w) / w.sum()
    r = da_mcmc(X, ARITY, sweeps=sweeps, burn=sweeps // 3, thin=max(1, sweeps // 300), rng=rng,
                gamma=2 * np.log(5), max_parents=3, struct_steps=20, weights=w)
    return np.array([cell_probs(g, c) for g, c in zip(r.dags, r.cpts)])


def cell_index(x: np.ndarray) -> np.ndarray:
    cat = np.column_stack([x[:, :4], np.round(x[:, 4] * 2 + 2)]).astype(int)
    return np.ravel_multi_index(cat.T, ARITY)


# ----------------------------------------------------------------- transport layer
def hyper(beta, se, share, keep_c, pairs_of_outcomes):
    """Attenuation, coefficient transport SD and level SD over country pairs in keep_c
    and the given (target outcome, source outcome) pairs."""
    slopes, excess = [], []
    for a, b in permutations(keep_c, 2):
        xa = np.concatenate([beta[a, os_] for _, os_ in pairs_of_outcomes])
        xb = np.concatenate([beta[b, ot] for ot, _ in pairs_of_outcomes])
        sa = np.concatenate([se[a, os_] for _, os_ in pairs_of_outcomes])
        sb = np.concatenate([se[b, ot] for ot, _ in pairs_of_outcomes])
        ok = ~(np.isnan(xa) | np.isnan(xb))
        if ok.sum() < 3:
            continue
        xa, xb, sa, sb = xa[ok], xb[ok], sa[ok], sb[ok]
        s = float(xb @ xa / (xa @ xa))
        slopes.append(s)
        excess.append((xb - s * xa).var(ddof=1) - np.mean(sb**2 + s**2 * sa**2))
    outs = sorted({ot for ot, _ in pairs_of_outcomes})
    z = norm.ppf(np.clip(share[np.ix_(keep_c, outs)], 0.005, 0.995))
    d = z[:, None, :] - z[None, :, :]
    m = ~np.eye(len(keep_c), dtype=bool)[:, :, None] & ~np.isnan(d)
    return {"a_mean": float(np.mean(slopes)), "a_sd": float(np.std(slopes, ddof=1)),
            "tau_beta": float(np.sqrt(max(0.0, np.mean(excess)))),
            "tau_level": float(np.sqrt(np.mean(d[m] ** 2)))}


def hyper_for(beta, se, share, countries, t, k, kind):
    keep_c = [i for i, c in enumerate(countries) if REGION[c] != REGION[countries[t]]] if t is not None \
        else list(range(len(countries)))
    no = beta.shape[1]
    others = [o for o in range(no) if o != k]
    if kind == "same":
        pairs = [(o, o) for o in others]
    else:
        pairs = [(o1, o2) for o1 in others for o2 in others if o1 != o2]
    return hyper(beta, se, share, keep_c, pairs)


# ----------------------------------------------------------------- prediction
def _noise(rng, size, t_df):
    """Standard normal, or Student-t rescaled to unit variance (heavier tails)."""
    if t_df is None:
        return rng.standard_normal(size)
    return rng.standard_t(t_df, size) * np.sqrt((t_df - 2) / t_df)


def _calibrated(pis, b, p0, delta):
    """P(Y=1 | cell) with the intercept calibrated to p0 on the cell probabilities."""
    lin = b @ XCELL.T                                                  # D x 80
    c = norm.ppf(p0) - (pis * lin).sum(1)
    for _ in range(40):
        eta = c[:, None] + lin
        f = (pis * norm.cdf(eta)).sum(1) - p0
        c = c - np.clip(f / np.maximum((pis * norm.pdf(eta)).sum(1), 1e-12), -2, 2)
        if np.abs(f).max() < 1e-8:
            break
    return norm.cdf(c[:, None] + delta + lin)


def _table(lat, h, k):
    b = (h["a_mean"] + k * h["a_sd"] * lat["za"]) * lat["beta"] + k * h["tau_beta"] * lat["ze"]
    return _calibrated(lat["pis"], b, lat["p0"], k * h["tau_level"] * lat["zd"])


def predictive(pi: np.ndarray, src: dict, base: dict, h: dict, D: int, rng, ks=(1.0,), t_df=None):
    """D draws of P(Y=1 | cell) for each inflation k of the transport layer (same
    latent random numbers for every k), the cell probabilities and the latents."""
    lat = {"pis": pi[rng.integers(len(pi), size=D)],
           "beta": rng.normal(src["beta"], src["se"], size=(D, 5)),
           "za": rng.standard_normal((D, 1)), "ze": _noise(rng, (D, 5), t_df), "zd": _noise(rng, (D, 1), t_df),
           "p0": rng.beta(base["share"] * base["n"] + 0.5, (1 - base["share"]) * base["n"] + 0.5, size=D)}
    return {k: _table(lat, h, k) for k in ks}, lat["pis"], lat


def _log_prior(lat, src, base, t_df):
    from scipy.stats import beta as beta_d, t as t_d
    def lz(z):
        if t_df is None:
            return norm.logpdf(z).sum(1)
        sc = np.sqrt((t_df - 2) / t_df)
        return t_d.logpdf(z / sc, t_df).sum(1)
    return (norm.logpdf(lat["za"]).sum(1) + lz(lat["ze"]) + lz(lat["zd"])
            + norm.logpdf(lat["beta"], src["beta"], src["se"]).sum(1)
            + beta_d.logpdf(lat["p0"], base["share"] * base["n"] + 0.5, (1 - base["share"]) * base["n"] + 0.5))


def _log_lik(P, cidx, y):
    pl = np.clip(P[:, cidx], 1e-12, 1 - 1e-12)
    return (y * np.log(pl) + (1 - y) * np.log(1 - pl)).sum(1)


def resample_move(lat, wt, h, k, src, base, cidx, y, rng, t_df, M=1000, steps=6):
    """Resample M particles by their weights, then move each with random-walk
    Metropolis steps that leave the posterior given the local data invariant
    (Gilks and Berzuini 2001). The cell probabilities of a particle are kept."""
    idx = rng.choice(len(wt), M, p=wt)
    cur = {key: v[idx].copy() for key, v in lat.items()}
    P = _table(cur, h, k)
    lp = _log_prior(cur, src, base, t_df) + _log_lik(P, cidx, y)
    acc = 0.0
    for _ in range(steps):
        prop = dict(cur)
        for key in ("za", "ze", "zd"):
            prop[key] = cur[key] + 0.25 * rng.standard_normal(cur[key].shape)
        prop["beta"] = cur["beta"] + 0.25 * src["se"] * rng.standard_normal(cur["beta"].shape)
        lg = np.log(cur["p0"] / (1 - cur["p0"])) + 0.1 * rng.standard_normal(M)
        prop["p0"] = 1 / (1 + np.exp(-lg))
        Pp = _table(prop, h, k)
        # logit random walk on p0: Jacobian p(1-p)
        lpp = _log_prior(prop, src, base, t_df) + _log_lik(Pp, cidx, y)
        jac = np.log(prop["p0"] * (1 - prop["p0"])) - np.log(cur["p0"] * (1 - cur["p0"]))
        ok = np.log(rng.random(M)) < lpp - lp + jac
        acc += ok.mean() / steps
        for key in cur:
            cur[key] = np.where(ok.reshape((-1,) + (1,) * (cur[key].ndim - 1)), prop[key], cur[key])
        P = np.where(ok[:, None], Pp, P)
        lp = np.where(ok, lpp, lp)
    return P, cur["pis"], acc


def quantities(P, pis) -> dict:
    out = {"level": (pis * P).sum(1)}
    for j, name in enumerate(PREDICTORS):
        on = WORRY_ON if name == "env_worry" else CELLS[:, j] == 1
        a = (pis[:, on] * P[:, on]).sum(1) / pis[:, on].sum(1)
        b = (pis[:, ~on] * P[:, ~on]).sum(1) / pis[:, ~on].sum(1)
        out[f"gap_{name}"] = a - b
    return out


def observed(x, y, w) -> dict:
    out = {"level": weighted_share(y, w)}
    for j, name in enumerate(PREDICTORS):
        on = x[:, j] > 0 if name == "env_worry" else x[:, j] == 1
        if on.sum() < 10 or (~on).sum() < 10:
            out[f"gap_{name}"] = (np.nan, np.nan)
            continue
        g1, s1 = weighted_share(y[on], w[on]); g0, s0 = weighted_share(y[~on], w[~on])
        out[f"gap_{name}"] = (g1 - g0, float(np.hypot(s1, s0)))
    return out


def local_estimates(x, y) -> dict:
    """Local sample alone: shares and gaps with Agresti-Caffo intervals (one success
    and one failure added to each group), which hold their level in small groups."""
    def ac(yy):
        n1 = len(yy); pt = (yy.sum() + 1) / (n1 + 2)
        return (yy.mean() if n1 else np.nan), pt, pt * (1 - pt) / (n1 + 2)
    out = {}
    p, pt, v = ac(y)
    out["level"] = (p, pt - 1.96 * np.sqrt(v), pt + 1.96 * np.sqrt(v))
    for j, name in enumerate(PREDICTORS):
        on = x[:, j] > 0 if name == "env_worry" else x[:, j] == 1
        if on.sum() == 0 or (~on).sum() == 0:
            out[f"gap_{name}"] = (np.nan, np.nan, np.nan)
            continue
        p1, t1, v1 = ac(y[on]); p0, t0, v0 = ac(y[~on])
        h = 1.96 * np.sqrt(v1 + v0)
        out[f"gap_{name}"] = (p1 - p0, t1 - t0 - h, t1 - t0 + h)
    return out


def wquant(v, w, q):
    o = np.argsort(v); cw = np.cumsum(w[o]) / w.sum()
    return float(v[o][min(np.searchsorted(cw, q), len(v) - 1)])


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--issp", type=Path, default=Path("metadata/ZA7650_v2-0-0.dta"))
    ap.add_argument("--out", type=Path, default=Path("transport_output"))
    ap.add_argument("--source-n", type=int, default=272)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--ks", type=float, nargs="+", default=[1.0, 1.5, 1.75],
                    help="inflation factors of the transport layer evaluated on the same draws")
    ap.add_argument("--t-df", type=float, default=None,
                    help="Student-t transport errors with these degrees of freedom (default: normal)")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    ap.add_argument("--sweeps", type=int, default=900, help="data-augmentation sweeps per target network")
    ap.add_argument("--quick", action="store_true", help="few countries, for a test")
    ap.add_argument("--scenarios", nargs="+", default=["same", "small_cross"],
                    help="run only these scenarios (for parallel runs)")
    ap.add_argument("--outcomes", nargs="+", default=None,
                    help="run only these outcomes (for parallel runs)")
    ap.add_argument("--networks-only", action="store_true",
                    help="estimate and save the target networks, then stop")
    ap.add_argument("--summarise-only", action="store_true",
                    help="skip computation and summarise the saved parts")
    a = ap.parse_args()
    parts = a.out / f"jv_parts{a.tag}"
    parts.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    preds, outcomes = list(PREDICTORS), list(OUTCOMES)

    data = design(pd.read_stata(a.issp, convert_categoricals=False))
    data = data[data.country.isin(EUROPE)]
    countries = sorted(data.country.unique())
    if a.quick:
        countries = countries[:6]
        data = data[data.country.isin(countries)]
    subs = {(c, k): data[data.country == c][preds + [k, "weight"]].dropna() for c in countries for k in outcomes}
    full = {key: fit(sub, key[1], None, rng) for key, sub in subs.items()}
    small = {key: fit(sub, key[1], a.source_n, rng) for key, sub in subs.items()}
    beta = np.full((len(countries), len(outcomes), 5), np.nan)
    se, share = beta.copy(), np.full((len(countries), len(outcomes)), np.nan)
    for (c, k), f in full.items():
        if f is not None:
            i, j = countries.index(c), outcomes.index(k)
            beta[i, j], se[i, j], share[i, j] = f["beta"], f["se"], f["share"]

    # hyperparameters on all pairs (for the application) and leave-region-out (for the test)
    allpairs = {kind: hyper_for(beta, se, share, countries, None, None, kind) for kind in ("same", "cross")}
    hyp = {(t, k, kind): hyper_for(beta, se, share, countries, countries.index(t), outcomes.index(k), kind)
           for t in countries for k in outcomes for kind in ("same", "cross")}
    print("hyperparameters, all pairs:", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()}
                                                      for k, v in allpairs.items()}))

    # target networks are cached (derived from microdata: keep local, do not commit)
    net_file = parts / "target_networks.pkl"
    if net_file.exists():
        nets = pickle.loads(net_file.read_bytes())
        print(f"target networks loaded from {net_file}", flush=True)
    else:
        print("target networks:", flush=True)
        nets = {}
        for c in countries:
            sub = data[data.country == c][preds + ["weight"]].dropna(subset=["weight"])
            sub = sub[sub[preds].notna().sum(axis=1) >= 3]
            nets[c] = target_network(sub, rng, a.sweeps)
            print(f"  {c}: {len(sub)} respondents, {len(nets[c])} draws", flush=True)
        net_file.write_bytes(pickle.dumps(nets))
    if a.networks_only:
        print("->", net_file)
        return

    todo_out = a.outcomes or outcomes
    for scenario in ("same", "small_cross"):
        if a.summarise_only or scenario not in a.scenarios:
            continue
        fits = small if scenario == "small_cross" else full
        for k in outcomes:
            part = parts / f"{scenario}__{k}.csv"
            if k not in todo_out:
                continue
            if part.exists():
                print(f"scenario {scenario}: outcome {k} already done ({part.name})", flush=True)
                continue
            # one generator per part, so parts can run in any order or in parallel
            rng = np.random.default_rng([SEED, ("same", "small_cross").index(scenario), outcomes.index(k)])
            rows = []
            for s, t in permutations(countries, 2):
                tgt, base = subs[(t, k)], fits[(s, k)]
                if full[(t, k)] is None or base is None:
                    continue
                x, y, w = tgt[preds].to_numpy(), tgt[k].to_numpy(), tgt["weight"].to_numpy()
                obs_full = observed(x, y, w)
                srcs = [fits[(s, kc)] for kc in outcomes if kc != k] if scenario == "small_cross" else [fits[(s, k)]]
                kinds = ("same", "cross") if scenario == "small_cross" else ("same",)
                cidx = cell_index(x)
                for src in srcs:
                    if src is None:
                        continue
                    for kind in kinds:
                      h = hyp[(t, k, kind)]
                      Ps, pis, lat = predictive(nets[t], src, base, h, a.draws, rng, a.ks, a.t_df)
                      for kk, P in Ps.items():
                        Q = quantities(P, pis)
                        common = {"scenario": scenario, "tau": kind, "k": kk, "source": s, "target": t, "outcome": k}
                        for q, v in Q.items():
                            o, ose = obs_full[q]
                            if np.isnan(o):
                                continue
                            # predictive of the observed statistic: population value plus
                            # the sampling error of the reference sample
                            vo = v + rng.normal(0, ose, size=len(v))
                            rows.append(common | {"stage": "prior", "n_local": 0, "quantity": q, "observed": o,
                                                  "pred": float(v.mean()), "low": float(np.percentile(vo, 2.5)),
                                                  "high": float(np.percentile(vo, 97.5))})
                        # local updates (only with the hyperparameters of the scenario)
                        if kind != kinds[-1]:
                            continue
                        for n in N_LOCAL:
                            loc = rng.choice(len(y), n, replace=False)
                            rest = np.setdiff1d(np.arange(len(y)), loc)
                            ref = observed(x[rest], y[rest], w[rest])
                            ll = _log_lik(P, cidx[loc], y[loc])
                            wt = np.exp(ll - ll.max()); wt /= wt.sum()
                            ess = 1.0 / np.sum(wt**2)
                            if ess < 500:   # weights degenerate: resample and move
                                Pm, pim, acc = resample_move(lat, wt, h, kk, src, base, cidx[loc], y[loc], rng, a.t_df)
                                Qm, wm = quantities(Pm, pim), np.full(len(Pm), 1.0 / len(Pm))
                            else:
                                Qm, wm, acc = Q, wt, np.nan
                            lo = local_estimates(x[loc], y[loc])
                            for q, v in Qm.items():
                                o, ose = ref[q]
                                if np.isnan(o):
                                    continue
                                lv, llo, lhi = lo[q]
                                vo = v + rng.normal(0, ose, size=len(v))
                                mid, half = (llo + lhi) / 2, np.hypot((lhi - llo) / 2, 1.96 * ose)
                                rows.append(common | {"stage": "update", "n_local": n, "quantity": q, "observed": o,
                                                      "pred": float(np.sum(wm * v)), "low": wquant(vo, wm, 0.025),
                                                      "high": wquant(vo, wm, 0.975), "ess": ess, "accept": acc,
                                                      "local": lv, "local_low": mid - half, "local_high": mid + half})
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"scenario {scenario}: outcome {k} done -> {part.name}", flush=True)

    expected = [parts / f"{sc}__{k}.csv" for sc in ("same", "small_cross") for k in outcomes]
    missing = [p.name for p in expected if not p.exists()]
    if missing:
        print(f"\nparts still to run: {missing}\n(the summary is written once all ten exist)")
        return
    cases = pd.concat([pd.read_csv(p) for p in expected], ignore_index=True)
    cases.to_csv(a.out / f"joint_validation_cases{a.tag}.csv", index=False)
    cases["err"] = (cases.pred - cases.observed).abs() * 100
    cases["cov"] = (cases.low <= cases.observed) & (cases.observed <= cases.high)
    cases["width"] = (cases.high - cases.low) * 100
    cases["kind"] = np.where(cases.quantity == "level", "level", "gap")
    summary = {"prior": {}, "update": {}, "k_out_of_region": {}}
    print(f"\nprior (no local data), leave-region-out hyperparameters, t_df = {a.t_df}:")
    for (sc, tau, kk, kind), g in cases[cases.stage == "prior"].groupby(["scenario", "tau", "k", "kind"]):
        r = {"mae": round(g.err.mean(), 2), "coverage": round(100 * g["cov"].mean(), 1),
             "width": round(g.width.mean(), 1), "cases": len(g)}
        summary["prior"][f"{sc}|{tau}|k{kk:g}|{kind}"] = r
        print(f"  {sc:12s} tau={tau:5s} k={kk:<4g} {kind:5s}: error {r['mae']:5.1f}  "
              f"coverage {r['coverage']:5.1f}%  width {r['width']:5.1f}")
    print("\nupdate with a local sample (joint model vs local sample alone), against the rest of the target:")
    up = cases[cases.stage == "update"].copy()
    up["err_local"] = (up.local - up.observed).abs() * 100
    up["cov_local"] = (up.local_low <= up.observed) & (up.observed <= up.local_high)
    for (sc, kk, kind, n), g in up.groupby(["scenario", "k", "kind", "n_local"]):
        r = {"mae": round(g.err.mean(), 2), "coverage": round(100 * g["cov"].mean(), 1),
             "mae_local": round(g.err_local.mean(), 2), "coverage_local": round(100 * g.cov_local.mean(), 1),
             "gain": round(g.err_local.mean() - g.err.mean(), 2), "ess_median": round(float(g.ess.median()), 0),
             "accept": round(float(np.nanmean(g.accept)), 2) if g.accept.notna().any() else float("nan")}
        summary["update"][f"{sc}|k{kk:g}|{kind}|{n}"] = r
        print(f"  {sc:12s} k={kk:<4g} {kind:5s} n={n:3d}: joint {r['mae']:5.1f} ({r['coverage']:5.1f}%)  "
              f"local {r['mae_local']:5.1f} ({r['coverage_local']:5.1f}%)  gain {r['gain']:4.1f}  "
              f"ESS before move {r['ess_median']:.0f}, MH acceptance {r['accept']:.2f}")
    # inflation chosen without the target's region: smallest k on the grid whose prior
    # coverage of gaps reaches 95% in the other regions, applied to the held-out region
    print("\ninflation chosen out of region (prior gap coverage >= 95% in the other regions):")
    cases["region"] = cases.target.map(REGION)
    for sc, tau in (("same", "same"), ("small_cross", "cross")):
        pr = cases[(cases.stage == "prior") & (cases.scenario == sc) & (cases.tau == tau) & (cases.kind == "gap")]
        chosen, held = {}, []
        for r in sorted(pr.region.unique()):
            other = pr[pr.region != r]
            cov = other.groupby("k")["cov"].mean()
            ok = cov[cov >= 0.95]
            kk = float(ok.index.min()) if len(ok) else float(cov.index.max())
            chosen[r] = kk
            held.append(pr[(pr.region == r) & (pr.k == kk)])
        held = pd.concat(held)
        res = {"k_by_region": chosen, "coverage": round(float(100 * held["cov"].mean()), 1),
               "width": round(float(held.width.mean()), 1), "mae": round(float(held.err.mean()), 2)}
        upd = {}
        for n in N_LOCAL:
            u = pd.concat([up[(up.scenario == sc) & (up.kind == "gap") & (up.n_local == n)
                              & (up.target.map(REGION) == r) & (up.k == kk)] for r, kk in chosen.items()])
            if len(u):
                upd[n] = {"coverage": round(float(100 * u["cov"].mean()), 1), "mae": round(float(u.err.mean()), 2),
                          "gain": round(float(u.err_local.mean() - u.err.mean()), 2)}
        res["update"] = upd
        summary["k_out_of_region"][f"{sc}|{tau}"] = res
        print(f"  {sc:12s} tau={tau:5s}: k by region {chosen}; held-out coverage {res['coverage']}%, "
              f"width {res['width']}; with local data {upd}")
    summary["hyperparameters_all_pairs"] = allpairs
    summary["t_df"] = a.t_df
    (a.out / f"joint_validation_summary{a.tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    (a.out / "hierarchical_transport.json").write_text(json.dumps(allpairs, indent=2) + "\n")
    print("->", a.out / f"joint_validation_summary{a.tag}.json", "and hierarchical_transport.json")


if __name__ == "__main__":
    main()
