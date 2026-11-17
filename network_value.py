"""What the Bayesian network contributes to BVS queries (revision, point 3).

The profile contrasts of the paper (77% vs 2%, Tables segments and plan) are
computed from the transfer model on CIS respondents and do not use the network.
The network is needed for queries that leave predictors unspecified or involve
other observed variables (institutional trust, ideology, place of residence...).
This script measures, on such queries, what the network adds over simpler ways
of answering them from CIS 3391.

Methods compared (all estimate the same quantity, below):
  direct     PESOFIN-weighted mean over the CIS respondents in the segment,
             normal interval with linearised standard error
  bootstrap  same point estimate, Poisson-weighted bootstrap percentile interval
  ipf2       raking / log-linear model with all two-way margins of the
             predictors and the segment variable (IPF), bootstrap interval
  margins    raking to one-way margins only (independence), bootstrap interval
  bvs        the BVS generator of step4.py: MC3 graphs of Step 1, Dirichlet-BDeu
             tables with PESOFIN-weighted counts and flat prior (population S2),
             one theta draw per graph (Martins et al. 2024, Algorithm 1),
             ancestral sampling; interval = percentiles over graphs

Quantity estimated: the PESOFIN-weighted mean, over the population segment, of
the central transfer-model probability P(accept | x) of transfer_model.py
(a0 = 0.80, p0 = 0.38, centred worry coding). The transfer parameters are held
fixed, so the comparison isolates the generator.

Held-out design: respondents are split at random into two halves. Methods are
fitted on a training subsample of the first half and compared with the direct
estimate on the second half, which none of them has seen. The network structure
is the one learned by Step 1 on the full sample; only the conditional tables are
re-estimated on the training data. This leak favours the network and is reported.

Usage
-----
    python network_value.py                       # full-sample run + held-out run
    python network_value.py --mode full           # where does the network change the answer?
    python network_value.py --mode heldout --reps 20 --n-train 300 600 half
Inputs : metadata/3391_num.csv, step1_mcmc_samples.json, dag_edges_3391.csv,
         transfer_coefficients.csv (the files used by step4.py and transfer_model.py)
Output : network_value_output/ (CSV tables, summary.json, table_network_value.tex)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from step4 import (  # noqa: E402  (the pipeline's own generator)
    Config,
    DirichletBDeuCPD,
    MC3DAGEnsemble,
    Sampler,
    compute_optimal_indegree,
    discretise,
    load_cis_microdata,
    load_dag_structure,
    load_mcmc_samples,
)
from transfer_model import A0, P0, PREDICTORS, SURVEY_YEAR, calibrate_intercept, worry_score  # noqa: E402

WORRY_CODING = "centered"
SOURCE_VARS = ["SEX", "BIRTH", "ESTUDIOS", "V52", "V15"]

# Segment variables and the range of substantive codes (other codes = missing).
# Check these ranges against the CIS 3391 codebook before reporting.
SEGMENT_VARS = {
    "V10": (1, 5),          # trust in people
    "V11": (0, 10),         # trust: parliament
    "V12": (0, 10),         # trust: business and industry
    "V13": (0, 10),         # trust: universities and research
    "V14": (0, 10),         # trust: media
    "LEFT_RIGHT": (0, 10),  # left-right self-placement
    "ATTEND": (1, 8),       # religious attendance
    "URBRURAL": (1, 5),     # place of residence
    "DIFICULT_ECO": (1, 5), # difficulty making ends meet
    "TOPBOT": (1, 10),      # subjective social position
    "V26": (1, 5),          # willingness to pay higher prices
    "V27": (1, 5),          # willingness to pay higher taxes
    "V37": (1, 5),          # danger of air pollution
}
METHODS = ["direct", "bootstrap", "ipf2", "margins", "bvs", "bvs_raked"]
MIN_TRAIN_DIRECT = 5     # below this the direct estimate is not reported
MIN_TEST = 30            # held-out reference needs at least this many respondents


def quiet(fn, *args, **kwargs):
    """Run a pipeline function without its progress printing."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_respondents(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw respondent frame (predictors, segments, weight) and the pipeline frame.

    The respondent frame applies the validity rules of transfer_model.load_design,
    so its complete cases are the 2,133 respondents of the paper. The pipeline
    frame is load_cis_microdata, which the network is trained on.
    """
    raw = pd.read_csv(cfg.PATH_CIS, sep=";", encoding="latin1", low_memory=False)
    pipeline = quiet(load_cis_microdata, cfg)

    num = {v: pd.to_numeric(raw[v], errors="coerce") for v in SOURCE_VARS if v in raw}
    missing = [v for v in SOURCE_VARS if v not in num]
    if missing:
        raise SystemExit(f"CIS file lacks predictor variables: {missing}")
    frame = pd.DataFrame(num, index=raw.index)
    frame["w"] = pd.to_numeric(raw["PESOFIN"].astype(str).str.replace(",", "."), errors="raise")
    for var, (lo, hi) in SEGMENT_VARS.items():
        if var in raw:
            values = pd.to_numeric(raw[var], errors="coerce")
            frame[var] = values.where(values.between(lo, hi))
    add_predictors(frame)
    return frame, pipeline


def add_predictors(frame: pd.DataFrame) -> None:
    """Predictors of the transfer model from raw CIS values (load_design rules)."""
    complete = (
        frame["SEX"].isin([1, 2])
        & frame["BIRTH"].between(1900, SURVEY_YEAR)
        & frame["ESTUDIOS"].between(1, 7)
        & frame["V52"].isin([1, 2, 3, 4])
        & frame["V15"].between(1, 5)
    )
    age = SURVEY_YEAR - frame["BIRTH"]
    frame["complete"] = complete.to_numpy()
    frame["female"] = (frame["SEX"] == 2).astype(int)
    frame["age3"] = np.select([age < 30, age < 60], [0, 1], 2)
    frame["higher"] = (frame["ESTUDIOS"] == 7).astype(int)
    frame["recycles"] = frame["V52"].isin([1, 2]).astype(int)
    frame["worry"] = frame["V15"].where(complete, 1).fillna(1).astype(int)


class Score:
    """Central transfer-model probability, intercept calibrated on all CIS complete cases."""

    def __init__(self, respondents: pd.DataFrame, coef_path: str):
        coef = pd.read_csv(coef_path).set_index("predictor").loc[list(PREDICTORS)]
        self.beta = coef["coefficient"].to_numpy(float)
        cc = respondents[respondents["complete"]]
        self.c = calibrate_intercept(A0 * (self.design(cc) @ self.beta), P0, cc["w"].to_numpy())

    @staticmethod
    def design(frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            frame["female"].to_numpy(float),
            (frame["age3"] == 0).to_numpy(float),
            frame["higher"].to_numpy(float),
            frame["recycles"].to_numpy(float),
            worry_score(frame["worry"].to_numpy(), WORRY_CODING),
        ])

    def __call__(self, frame: pd.DataFrame) -> np.ndarray:
        return norm.cdf(self.c + A0 * (self.design(frame) @ self.beta))


# ---------------------------------------------------------------------------
# Segments and queries
# ---------------------------------------------------------------------------

def segment_groups(respondents: pd.DataFrame, segment_vars: list[str]) -> dict:
    """Up to three groups per segment variable (weighted tertiles of valid values)."""
    cc = respondents[respondents["complete"]]
    groups = {}
    for var in segment_vars:
        valid = cc[var].notna()
        values, weights = cc.loc[valid, var].to_numpy(), cc.loc[valid, "w"].to_numpy()
        levels = np.unique(values)
        if len(levels) <= 3:
            cuts = list(levels[:-1])
        else:
            order = np.argsort(values)
            cum = np.cumsum(weights[order]) / weights.sum()
            cuts = sorted({float(values[order][np.searchsorted(cum, q)]) for q in (1 / 3, 2 / 3)})
            cuts = [c for c in cuts if c < levels.max()]
        edges = [-np.inf] + cuts + [np.inf]
        labels = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            lo_v = levels[levels > lo].min()
            hi_v = levels[levels <= hi].max()
            labels.append(f"{lo_v:g}" if lo_v == hi_v else f"{lo_v:g}-{hi_v:g}")
        groups[var] = {"cuts": cuts, "labels": labels}
    return groups


def assign_groups(frame: pd.DataFrame, groups: dict) -> None:
    for var, spec in groups.items():
        g = np.searchsorted(np.asarray(spec["cuts"], float), frame[var].to_numpy(float), side="left")
        frame["G_" + var] = np.where(frame[var].notna(), g, -1)


def build_queries(groups: dict) -> list[dict]:
    """Partially specified queries: segment alone, segment x sex, segment x age."""
    queries = []
    for var, spec in groups.items():
        for g, label in enumerate(spec["labels"]):
            base = {"S": var, "g": g}
            queries.append({**base, "kind": "segment", "label": f"{var} {label}"})
            for f, name in ((1, "women"), (0, "men")):
                queries.append({**base, "female": f, "kind": "segment x sex",
                                "label": f"{var} {label}, {name}"})
            for a, name in ((0, "under 30"), (2, "60+")):
                queries.append({**base, "age3": a, "kind": "segment x age",
                                "label": f"{var} {label}, {name}"})
    # Predictors partly specified (recycling and worry left free)
    for f, fname in ((1, "women"), (0, "men")):
        for a, aname in ((0, "under 30"), (2, "60+")):
            queries.append({"S": None, "female": f, "age3": a, "kind": "partial profile",
                            "label": f"{fname} {aname}, recycling and worry unspecified"})
        queries.append({"S": None, "female": f, "higher": 1, "kind": "partial profile",
                        "label": f"{fname} with higher education, other predictors unspecified"})
    for i, q in enumerate(queries):
        q["id"] = i
    return queries


def query_matrix(frame: pd.DataFrame, queries: list[dict]) -> np.ndarray:
    """Boolean (rows x queries) membership among complete cases."""
    base = frame["complete"].to_numpy(bool)
    cols = []
    for q in queries:
        m = base.copy()
        if q.get("S"):
            m &= frame["G_" + q["S"]].to_numpy() == q["g"]
        for key in ("female", "age3", "higher"):
            if key in q:
                m &= frame[key].to_numpy() == q[key]
        cols.append(m)
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def direct_estimates(frame, p, queries, rng, n_boot):
    """Direct and bootstrap estimates for every query at once."""
    M = query_matrix(frame, queries).astype(float)
    w = frame["w"].to_numpy(float)
    count = M.sum(axis=0)
    den = w @ M
    with np.errstate(invalid="ignore", divide="ignore"):
        est = ((w * p) @ M) / den
        ss = ((w ** 2) * (p ** 2)) @ M - 2 * est * (((w ** 2) * p) @ M) + est ** 2 * ((w ** 2) @ M)
        se = np.sqrt(np.maximum(ss, 0)) / den
        pois = rng.poisson(1.0, size=(n_boot, len(w)))
        boot = ((pois * (w * p)) @ M) / ((pois * w) @ M)
    lo_b, hi_b = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    out = {
        "direct": dict(est=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se),
        "bootstrap": dict(est=est, se=np.nanstd(boot, axis=0), lo=lo_b, hi=hi_b),
    }
    for res in out.values():
        for key in res:
            res[key] = np.where(count >= MIN_TRAIN_DIRECT, res[key], np.nan)
    return out, count


IPF_DIMS = [("female", 2), ("age3", 3), ("higher", 2), ("recycles", 2), ("worry", 5)]


def ipf(codes: np.ndarray, w: np.ndarray, dims: list[int], order: int, iters=300, tol=1e-7):
    """Table matching all `order`-way weighted margins (iterative proportional fitting)."""
    table = np.full(dims, w.sum() / np.prod(dims))
    margins = []
    for combo in combinations(range(len(dims)), order):
        target = np.zeros([dims[i] for i in combo])
        np.add.at(target, tuple(codes[:, i] for i in combo), w)
        margins.append((combo, target))
    for _ in range(iters):
        worst = 0.0
        for combo, target in margins:
            other = tuple(i for i in range(len(dims)) if i not in combo)
            current = table.sum(axis=other)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(current > 0, target / current, 0.0)
            table *= ratio.reshape([dims[i] if i in combo else 1 for i in range(len(dims))])
            worst = max(worst, float(np.abs(current - target).max()))
        if worst < tol * w.sum():
            break
    return table / table.sum()


def ipf_estimates(frame, score, queries, groups, rng, n_boot):
    """ipf2 and margins estimates; one table per segment variable."""
    out = {m: {k: np.full(len(queries), np.nan) for k in ("est", "se", "lo", "hi")}
           for m in ("ipf2", "margins")}
    cc = frame[frame["complete"]]
    by_s = {}
    for i, q in enumerate(queries):
        by_s.setdefault(q.get("S"), []).append(i)
    for s, idx in by_s.items():
        dims = [k for _, k in IPF_DIMS]
        names = [n for n, _ in IPF_DIMS]
        rows = cc
        if s is not None:
            rows = cc[cc["G_" + s] >= 0]
            dims = dims + [len(groups[s]["labels"])]
            names = names + ["G_" + s]
        codes = np.column_stack([rows[n].to_numpy(int) - (1 if n == "worry" else 0) for n in names])
        w = rows["w"].to_numpy(float)
        # Cells as a weighted pseudo-population, scored like respondents
        grid = np.indices(dims).reshape(len(dims), -1).T
        cells = pd.DataFrame(grid, columns=names)
        cells["worry"] += 1
        cells["complete"] = True
        p_cells = score(cells)
        Mq = query_matrix(cells, [queries[i] for i in idx]).astype(float)
        for method, order in (("ipf2", 2), ("margins", 1)):
            def estimate(weights):
                t = ipf(codes, weights, dims, order).reshape(-1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return ((t * p_cells) @ Mq) / (t @ Mq)
            point = estimate(w)
            boot = np.array([estimate(w * rng.poisson(1.0, len(w))) for _ in range(n_boot)])
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            res = out[method]
            res["est"][idx], res["se"][idx] = point, np.nanstd(boot, axis=0)
            res["lo"][idx], res["hi"][idx] = lo, hi
    return out


class AvailableCaseCPD(DirichletBDeuCPD):
    """Dirichlet-BDeu tables from the respondents who answered the child and its parents.

    Same prior and weights as step4.DirichletBDeuCPD (population S2); the only
    change is that missing answers are left out instead of set to the modal category.
    """

    def __init__(self, df_disc, missing, binfo, ess=10.0, weights=None):
        super().__init__(df_disc, binfo, {}, ess, weights)
        self.observed = ~missing[self.cols].to_numpy(bool)
        for col in self.cols:
            ci, K = self.col_idx[col], self._K(col)
            obs = self.observed[:, ci]
            self.marginals[col] = np.bincount(self.data[obs, ci], weights=self.weights[obs],
                                              minlength=K)[:K].astype(float)

    def get_theta(self, var, parents, parent_vals_dict, rng, sample=True):
        K = self._K(var)
        ci = self.col_idx.get(var)
        if ci is None:
            return np.ones(K) / K
        alpha_base = np.ones(K) * (self.ess / K)
        if not parents:
            alpha_post = alpha_base + self.marginals[var]
        else:
            q = int(np.prod([self._K(p) for p in parents]))
            mask = self.observed[:, ci].copy()
            for p in parents:
                pi = self.col_idx[p]
                mask &= self.observed[:, pi] & (self.data[:, pi] == int(parent_vals_dict[p]))
            counts = np.bincount(self.data[mask, ci], weights=self.weights[mask], minlength=K)[:K]
            alpha_post = alpha_base / max(q, 1) + counts
        alpha_post = np.maximum(alpha_post, 1e-4)
        theta = rng.dirichlet(alpha_post) if sample else alpha_post / alpha_post.sum()
        theta = np.maximum(theta, 1e-10)
        return theta / theta.sum()


class SampleEnsemble:
    """Graphs drawn uniformly from posterior samples of the full DAG (step1_global.py)."""

    def __init__(self, samples, variables):
        vs = set(variables)
        self.samples = [[(u, v) for u, v in s if u in vs and v in vs] for s in samples]

    def sample_dag(self, rng):
        return self.samples[rng.integers(len(self.samples))]


class NetworkGenerator:
    """The step4.py generator refitted on a set of training respondents.

    structure = 'blocks': MC3DAGEnsemble of step4.py (block samples + Bernoulli
    cross-block edges); 'global': posterior samples of the full DAG.
    cpd = 'imputed': step4.DirichletBDeuCPD after modal imputation (as published);
    'available': AvailableCaseCPD.
    """

    def __init__(self, cfg, pipeline, mcmc, dag_edges, emeta, max_indeg, needed,
                 structure="blocks", global_samples=None, cpd="imputed"):
        self.cfg, self.pipeline, self.mcmc = cfg, pipeline, mcmc
        self.dag_edges, self.emeta, self.max_indeg = dag_edges, emeta, max_indeg
        self.needed = needed
        self.structure, self.global_samples, self.cpd_kind = structure, global_samples, cpd

    def fit(self, train_index, weights_raw):
        df = self.pipeline.loc[self.pipeline.index.intersection(train_index)]
        dd, binfo = quiet(discretise, df.copy(), self.cfg.N_BINS)
        # All network variables are generated, including any left without edges in a
        # sampled graph (they are then drawn from their margins).
        dag_vars = {u for e in self.dag_edges for u in e}
        if self.structure == "global":
            dag_vars |= {u for s in self.global_samples for e in s for u in e}
        self.common = [v for v in dd.columns if v in dag_vars]
        w = weights_raw.loc[dd.index].to_numpy(float)
        w = w * len(w) / w.sum()
        if self.cpd_kind == "available":
            missing = df.loc[dd.index, self.common].isna()
            self.cpd = quiet(AvailableCaseCPD, dd[self.common], missing, binfo, self.cfg.ESS, w)
        else:
            self.cpd = quiet(DirichletBDeuCPD, dd[self.common], binfo, {}, self.cfg.ESS, w)
        cs = set(self.common)
        if self.structure == "global":
            self.ensemble = SampleEnsemble(self.global_samples, self.common)
        else:
            self.ensemble = quiet(
                MC3DAGEnsemble, self.mcmc, [(u, v) for u, v in self.dag_edges if u in cs and v in cs],
                self.emeta, self.common, self.max_indeg)
        # Raw values behind each training code, to decode synthetic codes
        self.pools = {}
        for var in self.needed:
            if var not in dd.columns:
                continue
            raw = pd.to_numeric(df[var], errors="coerce")
            self.pools[var] = {k: raw[(dd[var] == k) & raw.notna()].to_numpy()
                               for k in range(binfo[var]["K"])}
        return self

    def generate(self, n_graphs, n_per, rng):
        frames = []
        for m in range(n_graphs):
            dag = self.ensemble.sample_dag(rng)
            graph = nx.DiGraph(dag)
            keep = {v for v in self.needed if v in self.common}
            for v in list(keep):
                if v in graph:
                    keep |= nx.ancestors(graph, v)
            variables = [v for v in self.common if v in keep]
            sampler = Sampler(dag, variables, self.cpd)
            data = sampler.sample(n_per, rng, sample_theta=True)  # theta^(m) ~ Dirichlet
            synth = {}
            for var in self.needed:
                if var not in sampler.ci:
                    synth[var] = np.full(n_per, np.nan)
                    continue
                codes = data[:, sampler.ci[var]]
                values = np.full(n_per, np.nan)
                for k, pool in self.pools[var].items():
                    hit = codes == k
                    if hit.any() and len(pool):
                        values[hit] = rng.choice(pool, size=hit.sum())
                synth[var] = values
            frame = pd.DataFrame(synth)
            frame["_graph"] = m
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def calibration_codes(frame, groups):
    """Integer codes of the calibration variables (predictors and segment groups; -1 -> own level)."""
    cols = [frame["female"].to_numpy(int), frame["age3"].to_numpy(int), frame["higher"].to_numpy(int),
            frame["recycles"].to_numpy(int), frame["worry"].to_numpy(int) - 1]
    dims = [2, 3, 2, 2, 5]
    for var, spec in groups.items():
        cols.append(frame["G_" + var].to_numpy(int) + 1)
        dims.append(len(spec["labels"]) + 1)
    return np.column_stack(cols), dims


def rake_two_way(codes, dims, target_codes, target_w, sweeps=25, tol=1e-4):
    """Weights for synthetic records so that every two-way margin matches the targets."""
    w = np.ones(len(codes))
    pairs = list(combinations(range(len(dims)), 2))
    keys = {(a, b): codes[:, a] * dims[b] + codes[:, b] for a, b in pairs}
    targets = {}
    for a, b in pairs:
        t = np.bincount(target_codes[:, a] * dims[b] + target_codes[:, b], weights=target_w,
                        minlength=dims[a] * dims[b])
        targets[(a, b)] = t / t.sum()
    for _ in range(sweeps):
        worst = 0.0
        for ab in pairs:
            cur = np.bincount(keys[ab], weights=w, minlength=len(targets[ab]))
            cur = cur / cur.sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(cur > 0, targets[ab] / cur, 1.0)
            w *= ratio[keys[ab]]
            worst = max(worst, float(np.abs(cur - targets[ab]).max()))
        if worst < tol:
            break
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.ones(len(codes))      # calibration impossible for this graph: leave unweighted
    return w


def bvs_estimates(generator, train, weights_raw, score, queries, groups, args, rng):
    """Network estimates, raw ('bvs') and calibrated to two-way margins ('bvs_raked').

    Calibration: each graph's synthetic records are raked to the two-way margins of the
    predictors and segment groups among the training respondents, re-weighted by a
    Bayesian bootstrap draw for that graph, so sampling uncertainty is propagated.
    """
    generator.fit(train.index, weights_raw)
    synth = generator.generate(args.n_graphs, args.n_per, rng)
    for var, (lo, hi) in SEGMENT_VARS.items():
        if var in synth:
            synth[var] = synth[var].where(synth[var].between(lo, hi))
    add_predictors(synth)
    assign_groups(synth, groups)
    p = score(synth)
    M = query_matrix(synth, queries).astype(float)
    graphs = synth["_graph"].to_numpy()
    complete = synth["complete"].to_numpy(bool)
    codes, dims = calibration_codes(synth, groups)
    tc = train[train["complete"]]
    t_codes, _ = calibration_codes(tc, groups)
    t_w = tc["w"].to_numpy(float)
    weights = np.zeros(len(synth))
    for m in range(args.n_graphs):
        rows = (graphs == m) & complete
        weights[rows] = rake_two_way(codes[rows], dims, t_codes, t_w * rng.exponential(1.0, len(t_w)))
    out = {}
    for name, w in (("bvs", np.ones(len(synth))), ("bvs_raked", weights)):
        per_graph = np.full((args.n_graphs, len(queries)), np.nan)
        for m in range(args.n_graphs):
            rows = graphs == m
            with np.errstate(invalid="ignore", divide="ignore"):
                per_graph[m] = ((w[rows] * p[rows]) @ M[rows]) / (w[rows] @ M[rows])
        with np.errstate(invalid="ignore", divide="ignore"):
            est = ((w * p) @ M) / (w @ M)
        lo, hi = np.nanpercentile(per_graph, [2.5, 97.5], axis=0)
        out[name] = {"est": est, "se": np.nanstd(per_graph, axis=0), "lo": lo, "hi": hi}
    return out, M.sum(axis=0)


def all_estimates(respondents, train_index, score, queries, groups, generator, args, rng):
    train = respondents.loc[train_index]
    p = score(train)
    res, count = direct_estimates(train, p, queries, rng, args.boot)
    res.update(ipf_estimates(train, score, queries, groups, rng, args.boot_ipf))
    bvs, n_synth = bvs_estimates(generator, train, respondents["w"], score, queries, groups, args, rng)
    res.update(bvs)
    return res, count, n_synth


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def rows_from(res, queries, extra):
    rows = []
    for method in METHODS:
        r = res[method]
        for i, q in enumerate(queries):
            rows.append({**extra(i), "query": q["id"], "kind": q["kind"], "label": q["label"],
                         "method": method, "est": r["est"][i], "se": r["se"][i],
                         "lo": r["lo"][i], "hi": r["hi"][i]})
    return rows


def run_full(respondents, score, queries, groups, generator, args, rng):
    res, count, n_synth = all_estimates(respondents, respondents.index, score, queries,
                                        groups, generator, args, rng)
    table = pd.DataFrame(rows_from(res, queries, lambda i: {"n_cis": int(count[i]),
                                                            "n_synth": int(n_synth[i])}))
    wide = table.pivot_table(index=["query", "kind", "label", "n_cis"], columns="method",
                             values="est").reset_index()
    for m in METHODS:              # a method with no estimate at all keeps an empty column
        if m not in wide:
            wide[m] = np.nan
    se_direct = table[table.method == "direct"].set_index("query")["se"]
    wide["se_direct"] = wide["query"].map(se_direct)
    for m in ("bvs", "bvs_raked", "ipf2", "margins"):
        wide[f"{m}_minus_direct"] = wide[m] - wide["direct"]
        wide[f"{m}_departs"] = (wide[f"{m}_minus_direct"].abs() > 1.96 * wide["se_direct"])
    return table, wide


def run_heldout(respondents, score, queries, groups, generator, args, rng):
    rows = []
    ids = respondents.index.to_numpy()
    start = time.time()
    for rep in range(args.reps):
        perm = rng.permutation(ids)
        half_a, half_b = perm[: len(perm) // 2], perm[len(perm) // 2:]
        test = respondents.loc[half_b]
        p_test = score(test)
        ref, n_test = direct_estimates(test, p_test, queries, rng, 1)
        for n_train in args.n_train:
            size = len(half_a) if n_train == "half" else int(n_train)
            train_index = pd.Index(rng.choice(half_a, size=min(size, len(half_a)), replace=False))
            res, count, _ = all_estimates(respondents, train_index, score, queries, groups,
                                          generator, args, rng)
            rows += rows_from(res, queries, lambda i: {
                "rep": rep, "n_train": size, "n_cis_train": int(count[i]),
                "n_test": int(n_test[i]), "ref": ref["direct"]["est"][i],
                "se_ref": ref["direct"]["se"][i]})
        print(f"  rep {rep + 1}/{args.reps} done ({time.time() - start:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def size_class(n):
    return pd.cut(n, [-1, MIN_TRAIN_DIRECT - 1, 29, 99, np.inf],
                  labels=[f"<{MIN_TRAIN_DIRECT} (direct unavailable)", f"{MIN_TRAIN_DIRECT}-29",
                          "30-99", "100+"])


def summarise(heldout: pd.DataFrame) -> pd.DataFrame:
    d = heldout[(heldout["n_test"] >= MIN_TEST) & heldout["ref"].notna()].copy()
    d["cell"] = size_class(d["n_cis_train"])
    d["err"] = d["est"] - d["ref"]
    d["covered"] = d["err"].abs() <= 1.96 * np.sqrt(d["se"] ** 2 + d["se_ref"] ** 2)
    d["width"] = d["hi"] - d["lo"]

    def agg(g):
        ans = g[g["est"].notna()]
        mse = (ans["err"] ** 2).mean()
        return pd.Series({
            "queries": len(g),
            "answered": len(ans) / max(len(g), 1),
            "rmse_pp": 100 * np.sqrt(mse),
            # test-noise corrected: E(err^2) = MSE(est) + Var(ref)
            "rmse_corrected_pp": 100 * np.sqrt(max(mse - (ans["se_ref"] ** 2).mean(), 0.0)),
            "bias_pp": 100 * ans["err"].mean(),
            "coverage": ans["covered"].mean(),
            "median_width_pp": 100 * ans["width"].median(),
        })

    keys = ["n_train", "cell", "method"]
    by_cell = d.groupby(keys, observed=True).apply(agg).reset_index()
    overall = d.groupby(["n_train", "method"]).apply(agg).reset_index().assign(cell="all")
    return pd.concat([overall, by_cell], ignore_index=True)


CELL_ORDER = [f"<{MIN_TRAIN_DIRECT} (direct unavailable)", f"{MIN_TRAIN_DIRECT}-29", "30-99", "100+", "all"]


def latex_table(summary: pd.DataFrame) -> str:
    """Compact table: rows = training size x segment size, methods in columns."""
    err_methods = ["direct", "ipf2", "margins", "bvs", "bvs_raked"]
    cov_methods = ["direct", "bootstrap", "ipf2", "margins", "bvs", "bvs_raked"]

    def cell_text(cell):
        return {CELL_ORDER[0]: f"$<{MIN_TRAIN_DIRECT}$", "all": "All"}.get(cell, cell.replace("-", "--"))

    def fmt(r, key, pct=False):
        if r is None or pd.isna(r[key]) or r["answered"] == 0:
            return "--"
        return f"{100 * r[key]:.0f}" if pct else f"{r[key]:.1f}"

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Held-out evaluation of the network on partially specified queries.",
        r"Error: root mean squared difference from the direct estimate on the held-out",
        r"half, in percentage points, corrected for the sampling noise of that reference",
        r"(0.0: not distinguishable from it). Coverage: percentage of queries whose 95\%",
        r"interval, combined with the reference standard error, contains the held-out",
        r"estimate. Segment size: training respondents in the segment; -- the CIS",
        r"subset is not computed below " + str(MIN_TRAIN_DIRECT) + r" respondents.",
        r"BVS-cal.: network population calibrated to two-way margins.",
        r"Cases: query--split pairs; means over them. Produced by \texttt{network\_value.py}.}",
        r"\label{tab:network_value}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{rlr rrrrr rrrrrr}", r"\toprule",
        r" & & & \multicolumn{5}{c}{Error (pp)} & \multicolumn{6}{c}{Coverage (\%)} \\",
        r"\cmidrule(lr){4-8}\cmidrule(lr){9-14}",
        r"$n_{\text{train}}$ & Segment & Cases & Subset & IPF & Raking & BVS & BVS-cal."
        r" & Subset & Boot. & IPF & Raking & BVS & BVS-cal. \\", r"\midrule"]
    for n in sorted(summary["n_train"].unique()):
        for cell in CELL_ORDER:
            g = summary[(summary["n_train"] == n) & (summary["cell"].astype(str) == cell)]
            if g.empty:
                continue
            by = {m: (g[g["method"] == m].iloc[0] if (g["method"] == m).any() else None) for m in METHODS}
            nq = int(next(r for r in by.values() if r is not None)["queries"])
            cells = [fmt(by[m], "rmse_corrected_pp") for m in err_methods]
            cells += [fmt(by[m], "coverage", pct=True) for m in cov_methods]
            lines.append(f"{int(n)} & {cell_text(cell)} & {nq} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_results(summary: pd.DataFrame, full: dict | None) -> str:
    """Result sentences for the paper, written from the run (no hand-copied numbers)."""
    names = {"direct": "the CIS subset", "ipf2": "two-way IPF", "margins": "one-way raking",
             "bvs": "the network", "bvs_raked": "the calibrated network", "bootstrap": "the weighted bootstrap"}

    def row(n, cell, method):
        r = summary[(summary["n_train"] == n) & (summary["cell"] == cell) & (summary["method"] == method)]
        return None if r.empty else r.iloc[0]

    out = []
    for n in sorted(summary["n_train"].unique()):
        for cell in ("all", "5-29", "100+"):
            rows = {m: row(n, cell, m) for m in ("bvs", "bvs_raked", "direct", "ipf2", "margins")}
            if any(v is None or pd.isna(v["rmse_corrected_pp"]) for v in rows.values()):
                continue
            best = min(rows, key=lambda m: rows[m]["rmse_corrected_pp"])
            where = "over all queries" if cell == "all" else f"for segments with {cell} training respondents"
            out.append(
                f"With {int(n)} training respondents, {where}, the noise-corrected error was "
                f"{rows['bvs']['rmse_corrected_pp']:.1f} percentage points for the network, "
                f"{rows['bvs_raked']['rmse_corrected_pp']:.1f} for the calibrated network, "
                f"{rows['direct']['rmse_corrected_pp']:.1f} for the CIS subset, "
                f"{rows['ipf2']['rmse_corrected_pp']:.1f} for two-way IPF and "
                f"{rows['margins']['rmse_corrected_pp']:.1f} for one-way raking "
                f"(lowest: {names[best]}); 95\\% intervals covered the held-out estimate in "
                f"{100 * rows['bvs']['coverage']:.0f}\\%, {100 * rows['bvs_raked']['coverage']:.0f}\\%, "
                f"{100 * rows['direct']['coverage']:.0f}\\%, "
                f"{100 * rows['ipf2']['coverage']:.0f}\\% and {100 * rows['margins']['coverage']:.0f}\\% "
                f"of queries, respectively.")
    if full:
        out.append(
            f"On the full sample, the network estimate departed from the CIS subset by more than "
            f"1.96 standard errors in {100 * full['bvs']['share_departing']:.0f}\\% of queries "
            f"(median absolute difference {full['bvs']['median_abs_diff_pp']:.1f} points), against "
            f"{100 * full['ipf2']['share_departing']:.0f}\\% for two-way IPF and "
            f"{100 * full['margins']['share_departing']:.0f}\\% for one-way raking.")
    return "\n".join(out) + "\n"


def self_check(respondents, score, segments_csv):
    """Reproduce Table segments of the paper before trusting anything else."""
    cc = respondents[respondents["complete"]]
    p, w = score(cc), cc["w"].to_numpy()
    print(f"  complete cases: {len(cc)} (paper: 2,133); weighted mean {np.average(p, weights=w):.3f} (p0 {P0})")
    if not os.path.exists(segments_csv):
        print(f"  {segments_csv} not found; segment check skipped")
        return
    paper = pd.read_csv(segments_csv)
    mine = {
        ("gender", "female"): cc["female"] == 1, ("gender", "male"): cc["female"] == 0,
        ("education", "higher"): cc["higher"] == 1, ("education", "below higher"): cc["higher"] == 0,
        ("recycling", "always or often"): cc["recycles"] == 1,
        ("recycling", "sometimes or never"): cc["recycles"] == 0,
    }
    for v in range(1, 6):
        mine[("environmental_worry", str(v))] = cc["worry"] == v
    worst = 0.0
    for _, r in paper.iterrows():
        key = (r["dimension"], str(r["level"]))
        if key in mine:
            m = mine[key].to_numpy()
            worst = max(worst, abs(np.average(p[m], weights=w[m]) - r["projected_acceptance"]))
    print(f"  largest difference from {os.path.basename(segments_csv)}: {100 * worst:.2f} pp")
    if worst > 0.005:
        print("  WARNING: the predictors do not reproduce the paper; check codings before using results")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["both", "full", "heldout"], default="both")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n-train", nargs="+", default=["300", "600", "half"])
    ap.add_argument("--n-graphs", type=int, default=50, help="graph draws (paper: 50)")
    ap.add_argument("--n-per", type=int, default=2000, help="synthetic individuals per graph")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--boot-ipf", type=int, default=50)
    ap.add_argument("--segments", nargs="+", default=list(SEGMENT_VARS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--structure", choices=["blocks", "global"], default="blocks",
                    help="blocks: published step1/step4 ensemble; global: step1_global.py samples")
    ap.add_argument("--global-samples", default=os.path.join(HERE, "step1_global_samples.json"))
    ap.add_argument("--cpd", choices=["imputed", "available"], default="imputed",
                    help="imputed: modal imputation as published; available: available-case tables")
    ap.add_argument("--out", default=None, help="default: network_value_output[_<structure>_<cpd>]")
    args = ap.parse_args()
    if args.out is None:
        suffix = "" if (args.structure, args.cpd) == ("blocks", "imputed") else f"_{args.structure}_{args.cpd}"
        args.out = os.path.join(HERE, "network_value_output" + suffix)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cfg = Config()

    print("1. Data")
    respondents, pipeline = load_respondents(cfg)
    score = Score(respondents, os.path.join(HERE, "transfer_coefficients.csv"))
    self_check(respondents, score, os.path.join(HERE, "transfer_output", "segments_centered.csv"))

    segment_vars = [v for v in args.segments if v in respondents and respondents[v].notna().any()]
    dag_edges, emeta = quiet(load_dag_structure, cfg)
    dag_vars = {u for e in dag_edges for u in e}
    absent = [v for v in segment_vars if v not in dag_vars]
    if absent:
        print(f"  not in the network, dropped: {absent}")
    segment_vars = [v for v in segment_vars if v in dag_vars]
    groups = segment_groups(respondents, segment_vars)
    assign_groups(respondents, groups)
    queries = build_queries(groups)
    print(f"  {len(segment_vars)} segment variables, {len(queries)} queries")

    # Generator with the pipeline's structure and in-degree bound (full sample)
    mcmc = quiet(load_mcmc_samples, cfg)
    if not mcmc:
        raise SystemExit("step1_mcmc_samples.json not found next to this script")
    dd_full, binfo_full = quiet(discretise, pipeline.copy(), cfg.N_BINS)
    common = [v for v in dd_full.columns if v in dag_vars]
    max_indeg, _, _ = quiet(compute_optimal_indegree,
                            quiet(DirichletBDeuCPD, dd_full[common], binfo_full, {}, cfg.ESS), common)
    global_samples = None
    if args.structure == "global":
        if not os.path.exists(args.global_samples):
            raise SystemExit(f"{args.global_samples} not found: run python step1_global.py first")
        with open(args.global_samples, encoding="utf-8") as fh:
            global_samples = [[tuple(e) for e in s] for s in json.load(fh)]
        print(f"  global structure: {len(global_samples)} posterior DAGs from {os.path.basename(args.global_samples)}")
    print(f"  network: {len(common)} variables, structure {args.structure}, tables {args.cpd}"
          + (f", max in-degree {max_indeg}" if args.structure == "blocks" else ""))
    generator = NetworkGenerator(cfg, pipeline, mcmc, dag_edges, emeta, max_indeg,
                                 SOURCE_VARS + segment_vars, structure=args.structure,
                                 global_samples=global_samples, cpd=args.cpd)

    meta = {"a0": A0, "p0": P0, "worry_coding": WORRY_CODING, "seed": args.seed,
            "n_graphs": args.n_graphs, "n_per": args.n_per, "segments": groups,
            "network_structure": args.structure, "conditional_tables": args.cpd,
            "structure": "learned on the full CIS sample (reused in held-out runs)"}

    if args.mode in ("both", "full"):
        print("\n2. Full sample: where does each method depart from the CIS respondents?")
        table, wide = run_full(respondents, score, queries, groups, generator, args, rng)
        table.to_csv(os.path.join(args.out, "full_estimates_long.csv"), index=False)
        wide.to_csv(os.path.join(args.out, "full_estimates.csv"), index=False)
        for m in ("bvs", "bvs_raked", "ipf2", "margins"):
            d = wide[f"{m}_minus_direct"].dropna()
            print(f"  {m:8s}: median |diff| {100 * d.abs().median():.1f} pp, "
                  f"departs beyond 1.96 SE in {wide[f'{m}_departs'].mean():.0%} of queries")
        meta["full"] = {m: {"median_abs_diff_pp": float(100 * wide[f"{m}_minus_direct"].abs().median()),
                            "share_departing": float(wide[f"{m}_departs"].mean())}
                        for m in ("bvs", "bvs_raked", "ipf2", "margins")}

    if args.mode in ("both", "heldout"):
        print(f"\n3. Held-out: {args.reps} splits x n_train {args.n_train}")
        heldout = run_heldout(respondents, score, queries, groups, generator, args, rng)
        heldout.to_csv(os.path.join(args.out, "heldout_estimates.csv"), index=False)
        summary = summarise(heldout)
        summary.to_csv(os.path.join(args.out, "heldout_summary.csv"), index=False)
        with open(os.path.join(args.out, "table_network_value.tex"), "w", encoding="utf-8") as fh:
            fh.write(latex_table(summary))
        overall = summary[summary["cell"] == "all"]
        print(overall[["n_train", "method", "answered", "rmse_corrected_pp", "bias_pp",
                       "coverage", "median_width_pp"]].round(3).to_string(index=False))
        meta["heldout"] = overall.to_dict(orient="records")
        with open(os.path.join(args.out, "results_network_value.tex"), "w", encoding="utf-8") as fh:
            fh.write(latex_results(summary, meta.get("full")))

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"\nOutputs in {args.out}")


if __name__ == "__main__":
    main()
