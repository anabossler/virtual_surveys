# STEP 4 — SYNTHETIC SURVEY GENERATION (MC3 + TOPOLOGICAL + CONDITIONAL CPTs)
# Martins et al. (2024) Algorithm 1 
# Adapted for d>>10 with block-wise MC3 from Step 1
# pi(Y|X) = integral p(Y|G,theta) p(G,theta|X) dtheta dG  [Martins eq. 8]

import os
import json
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import networkx as nx

warnings.filterwarnings("ignore")


# CONFIG

class Config:
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    PATH_CIS   = os.path.join(BASE_DIR, "metadata", "3391_num.csv")

    # Step 1 outputs
    PATH_DAG_CSV     = os.path.join(BASE_DIR, "dag_edges_3391.csv")
    PATH_BACKBONE    = os.path.join(BASE_DIR, "backbone_edges_3391.txt")
    PATH_EPROB       = os.path.join(BASE_DIR, "edge_probabilities_3391.csv")
    PATH_MCMC        = os.path.join(BASE_DIR, "step1_mcmc_samples.json")
    PATH_DIAGNOSTICS = os.path.join(BASE_DIR, "step1_diagnostics.json")

    # Step 3 outputs
    PATH_PRIOR    = os.path.join(BASE_DIR, "step3_fused_priors.json")
    PATH_FDAG     = os.path.join(BASE_DIR, "dag_fused_edges.csv")
    PATH_FDAG_TXT = os.path.join(BASE_DIR, "dag_fused_edges.txt")

    # Step 2 outputs (level mappings)
    PATH_CIS_MAPPINGS = os.path.join(BASE_DIR, "step2_cis_level_mappings.json")

    # Generation parameters
    M               = 50
    N_PER           = 200
    N_FINAL         = 10_000
    MAX_INDEGREE    = 3
    SEED            = 42
    ESS             = 10.0
    N_BINS          = 5
    OUTDIR          = os.path.join(BASE_DIR, "step4_output")
    MISSING_CODES   = {96, 97, 98, 99, 996, 997, 998, 999,
                       9996, 9997, 9998, 9999}

    TVD_EXCELLENT = 0.05
    TVD_GOOD      = 0.10
    TVD_ALARM     = 0.15


# CAUSAL TIERS AND FORBIDDEN EDGES

CAUSAL_TIERS = {
    'SEX': 0, 'BIRTH': 0,
    'URBRURAL': 1,
    'NAT_DEGR': 2, 'ESTUDIOS': 2, 'NAT_INC': 2, 'NAT_RINC': 2,
    'V10': 3, 'V11': 3, 'V12': 3, 'V13': 3, 'V14': 3,
    'V15': 3, 'V16': 3, 'V17': 3, 'V18': 3, 'V19': 3,
    'V37': 3, 'V38': 3, 'V39': 3, 'V40': 3,
    'V26': 4, 'V27': 4,
    'V50': 4, 'V51': 4, 'V52': 4, 'V53': 4,
}

def get_tier(var):
    return CAUSAL_TIERS.get(var, 999)

def is_edge_forbidden(u, v):
    return get_tier(u) > get_tier(v)


CIS_TO_HARM = {
    'demographics_gender':         ['SEX'],
    'demographics_age':            ['BIRTH', 'EDAD'],
    'demographics_education':      ['ESTUDIOS', 'NAT_DEGR'],
    'demographics_income':         ['NAT_INC', 'NAT_RINC'],
    'demographics_urban_rural':    ['URBRURAL'],
    'attitudes_env_concern':       ['V15'],
    'attitudes_wtp_taxes':         ['V27'],
    'attitudes_wtp_prices':        ['V26'],
    'behavior_recycling':          ['V52'],
    'behavior_reduce_consumption': ['V53'],
    'perception_danger_pollution': ['V37', 'V38', 'V39', 'V40'],
    'trust_institutions':          ['V10', 'V11', 'V12', 'V13', 'V14'],
}

HARM_TO_CIS = {}
for h, cs in CIS_TO_HARM.items():
    for c in cs:
        HARM_TO_CIS[c] = h


# 1. LOAD DATA

def load_cis_microdata(cfg):
    print("  1a. CIS microdata...")
    df = pd.read_csv(cfg.PATH_CIS, sep=";", encoding="latin1",
                     low_memory=False, quotechar='"',
                     na_values=['N.P.', 'N.C.', 'N.S.', 'N.D.'])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for code in cfg.MISSING_CODES:
        df = df.replace(code, np.nan)
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, df.std(skipna=True) > 0]
    df = df.loc[:, df.isnull().sum() / len(df) < 0.9]
    df = df.dropna(thresh=int(len(df.columns) * 0.4))
    print(f"      {df.shape[0]} obs x {df.shape[1]} vars")
    return df


def load_cis_level_mappings(cfg):
    # Load CIS numeric -> text level mappings from step2
    mappings = {}

    if os.path.exists(cfg.PATH_PRIOR):
        with open(cfg.PATH_PRIOR, 'r', encoding='utf-8') as f:
            data = json.load(f)
        summary = data.get('cis_level_mappings_summary', {})
        harm_to_cis = summary.get('harmonized_to_cis', {})
        if harm_to_cis:
            mappings['harmonized_to_cis'] = harm_to_cis

    # Fallback: separate step2 file
    if os.path.exists(cfg.PATH_CIS_MAPPINGS):
        with open(cfg.PATH_CIS_MAPPINGS, 'r', encoding='utf-8') as f:
            full_mappings = json.load(f)
        # Convert string keys to int where applicable (JSON serialises ints as strings)
        for cis_var, mapping in full_mappings.items():
            if cis_var == 'harmonized_to_cis':
                mappings['harmonized_to_cis'] = mapping
                continue
            if isinstance(mapping, dict) and 'type' not in mapping:
                int_map = {}
                for k, v in mapping.items():
                    try:
                        int_map[int(k)] = v
                    except (ValueError, TypeError):
                        int_map[k] = v
                mappings[cis_var] = int_map
            else:
                mappings[cis_var] = mapping

    if mappings:
        print(f"      CIS level mappings: {len(mappings)} entries loaded")
    else:
        print(f"      WARNING: No CIS level mappings found")
        print(f"        Expected: {cfg.PATH_CIS_MAPPINGS}")
        print(f"        Run step2 first to generate mappings")

    return mappings


def load_dag_structure(cfg):
    print("  1b. DAG structure...")
    edges = []
    emeta = {}

    for path, label in [
        (cfg.PATH_FDAG,    "fused DAG (Step 3)"),
        (cfg.PATH_DAG_CSV, "DAG edges (Step 1)"),
    ]:
        if os.path.exists(path):
            print(f"      Reading: {label}")
            de = pd.read_csv(path)
            if 'source' in de.columns and 'target' in de.columns:
                for _, r in de.iterrows():
                    u = str(r['source']).strip()
                    v = str(r['target']).strip()
                    try:
                        p = float(str(r.get('probability', 0.5)).split()[0])
                    except ValueError:
                        p = 0.5
                    edges.append((u, v))
                    emeta[(u, v)] = p
                print(f"      {len(edges)} edges loaded")
                break

    if not edges and os.path.exists(cfg.PATH_BACKBONE):
        print(f"      Fallback: backbone_edges_3391.txt")
        with open(cfg.PATH_BACKBONE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '->' in line:
                    parts = line.split('#')[0].strip()
                    arrow = parts.split('->')
                    if len(arrow) == 2:
                        u, v = arrow[0].strip(), arrow[1].strip()
                        p = 1.0
                        if '#' in line and 'p=' in line:
                            try:
                                p = float(line.split('p=')[1].split()[0])
                            except:
                                p = 1.0
                        edges.append((u, v))
                        emeta[(u, v)] = p
        print(f"      {len(edges)} edges loaded")

    return edges, emeta


def load_mcmc_samples(cfg):
    print("  1c. MC3 posterior DAG samples...")
    mcmc_samples = None
    if os.path.exists(cfg.PATH_MCMC):
        with open(cfg.PATH_MCMC, 'r') as f:
            mcmc_samples = json.load(f)
        total = sum(len(v) for v in mcmc_samples.values())
        n_blocks = len(mcmc_samples)
        print(f"      {total} DAG samples across {n_blocks} blocks")
        for block, samples in sorted(mcmc_samples.items()):
            if samples:
                n_distinct = len(set(
                    frozenset(tuple(e) for e in dag) for dag in samples
                ))
                print(f"         {block}: {len(samples)} samples, "
                      f"{n_distinct} distinct DAGs")
    else:
        print(f"      WARNING: step1_mcmc_samples.json not found")
        print(f"         Falling back to backbone perturbation")
    return mcmc_samples


def load_fused_priors(cfg):
    print("  1d. Fused priors (Step 3)...")
    pr = {}
    if os.path.exists(cfg.PATH_PRIOR):
        with open(cfg.PATH_PRIOR, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pr = data.get('variables', {})
        n_cpt = sum(1 for v in pr.values() if 'conditional_cpt' in v)
        print(f"      {len(pr)} variables ({n_cpt} with conditional CPTs)")
    else:
        print(f"      WARNING: step3_fused_priors.json not found, using BDeu flat")
    return pr


def load_edge_prob_matrix(cfg):
    ep = None
    if os.path.exists(cfg.PATH_EPROB):
        ep = pd.read_csv(cfg.PATH_EPROB, index_col=0)
        print(f"  1e. Edge prob matrix: {ep.shape[0]} x {ep.shape[1]}")
    return ep


# 2. DISCRETISE

def discretise(df, n_bins=5):
    print("\n2. Discretising...")
    cols, binfo = {}, {}
    for col in df.columns:
        x = df[col].dropna()
        nu = x.nunique()
        if nu <= 10:
            vals = sorted(x.unique())
            mp = {v: i for i, v in enumerate(vals)}
            cols[col] = df[col].map(mp)
            binfo[col] = {'K': len(vals), 'type': 'ord', 'labels': vals}
        else:
            try:
                b, edges = pd.qcut(df[col], q=n_bins, labels=range(n_bins),
                                   retbins=True, duplicates='drop')
                cols[col] = b
                binfo[col] = {'K': len(edges) - 1, 'type': 'bin',
                               'edges': edges.tolist()}
            except Exception:
                try:
                    cols[col] = pd.cut(df[col], bins=n_bins,
                                       labels=range(n_bins), duplicates='drop')
                    binfo[col] = {'K': n_bins, 'type': 'bin'}
                except Exception:
                    cols[col] = df[col].fillna(0).astype(int) % n_bins
                    binfo[col] = {'K': n_bins, 'type': 'bin'}

    dd = pd.DataFrame(cols, index=df.index)
    for c in dd.columns:
        dd[c] = pd.to_numeric(dd[c], errors='coerce')
    dd = dd.fillna(dd.mode().iloc[0]).astype(int)
    for c in dd.columns:
        K = binfo.get(c, {}).get('K', 2)
        dd[c] = dd[c].clip(0, K - 1)

    print(f"   {len(dd.columns)} vars discretised")
    return dd, binfo


# 3. DIRICHLET-BDeu CPD ENGINE

class DirichletBDeuCPD:
    def __init__(self, df_disc, binfo, fused_priors, ess=10.0):
        self.data    = df_disc.values.astype(int)
        self.cols    = list(df_disc.columns)
        self.col_idx = {c: i for i, c in enumerate(self.cols)}
        self.n       = len(df_disc)
        self.binfo   = binfo
        self.priors  = fused_priors
        self.ess     = ess

        # Pre-compute marginal counts
        self.marginals = {}
        for col in self.cols:
            K = self._K(col)
            ci = self.col_idx[col]
            self.marginals[col] = np.bincount(
                self.data[:, ci], minlength=K)[:K].astype(float)

        print(f"   Dirichlet-BDeu CPD engine: {len(self.cols)} vars, n={self.n}")

    def _K(self, var):
        return max(self.binfo.get(var, {}).get('K', 2), 2)

    def _get_base_prior(self, var):
        K = self._K(var)
        harm = HARM_TO_CIS.get(var)
        for name in [var, harm]:
            if name and name in self.priors:
                vdata = self.priors[name]
                a = np.array(
                    vdata.get('alpha_lit',
                    vdata.get('alpha_posterior',
                    vdata.get('alpha', []))),
                    dtype=float
                )
                if len(a) == K:
                    return a
                if len(a) > 0:
                    if len(a) < K:
                        out = np.ones(K) * (self.ess / K)
                        out[:len(a)] = a
                        return out
                    step = len(a) / K
                    return np.array([
                        a[int(i * step):int((i + 1) * step)].sum()
                        for i in range(K)
                    ])
        return np.ones(K) * (self.ess / K)

    def get_theta(self, var, parents, parent_vals_dict, rng, sample=True):
        # Implements Martins eq. 7: theta | G,X ~ Dirichlet(alpha_post)
        K = self._K(var)
        ci = self.col_idx.get(var)
        if ci is None:
            return np.ones(K) / K

        alpha_base = self._get_base_prior(var)

        if not parents:
            alpha_post = alpha_base + self.marginals.get(var, np.zeros(K))
        else:
            # Scale prior by number of parent configurations
            q_j = 1
            for p in parents:
                q_j *= self._K(p)
            alpha_config = alpha_base / max(q_j, 1)
            mask = np.ones(self.n, dtype=bool)
            for p in parents:
                pi = self.col_idx.get(p)
                pv = parent_vals_dict.get(p)
                if pi is not None and pv is not None:
                    mask &= (self.data[:, pi] == int(pv))
            n_match = mask.sum()
            if n_match > 0:
                child_vals = self.data[mask, ci]
                counts = np.bincount(child_vals, minlength=K)[:K].astype(float)
            else:
                counts = np.zeros(K)
            alpha_post = alpha_config + counts

        alpha_post = np.maximum(alpha_post, 1e-4)
        if sample:
            theta = rng.dirichlet(alpha_post)
        else:
            theta = alpha_post / alpha_post.sum()
        theta = np.maximum(theta, 1e-10)
        theta /= theta.sum()
        return theta


# 4. MC3 DAG ENSEMBLE — TOPOLOGICAL ENFORCEMENT

class MC3DAGEnsemble:
    def __init__(self, mcmc_samples, backbone_edges, emeta,
                 all_vars, max_indeg=3, block_assignment=None):
        self.mcmc_samples = mcmc_samples or {}
        self.all_vars = sorted(all_vars)
        self.vset = set(self.all_vars)
        self.max_indeg = max_indeg
        self.emeta = emeta

        self.backbone = [(u, v) for u, v in backbone_edges
                         if u in self.vset and v in self.vset]

        self.topo_rank = self._compute_topological_order(self.backbone)

        # Separate intra-block and inter-block edges
        block_vars = set()
        for block_name, samples in self.mcmc_samples.items():
            if samples:
                for dag in samples:
                    for e in dag:
                        block_vars.add(e[0])
                        block_vars.add(e[1])

        self.inter_edges = []
        self.inter_probs = []
        for u, v in self.backbone:
            is_intra = False
            for block_name, samples in self.mcmc_samples.items():
                if samples:
                    sample_vars = set()
                    for dag in samples[:1]:
                        for e in dag:
                            sample_vars.add(e[0])
                            sample_vars.add(e[1])
                    if u in sample_vars and v in sample_vars:
                        is_intra = True
                        break
            if not is_intra:
                p = emeta.get((u, v), 0.5)
                self.inter_edges.append((u, v))
                self.inter_probs.append(p)

        self.block_names = list(self.mcmc_samples.keys())
        n_intra = len(self.backbone) - len(self.inter_edges)
        print(f"   MC3 DAG Ensemble (topological):")
        print(f"     Blocks: {len(self.mcmc_samples)}")
        print(f"     Intra-block edges: ~{n_intra}")
        print(f"     Inter-block edges: {len(self.inter_edges)}")
        print(f"     Variables: {len(self.all_vars)}")
        print(f"     Max in-degree: {max_indeg}")

    def _compute_topological_order(self, edges):
        G = nx.DiGraph()
        G.add_edges_from(edges)
        try:
            order = list(nx.topological_sort(G))
            return {v: i for i, v in enumerate(order)}
        except nx.NetworkXError:
            print("      WARNING: Cycle in backbone, using tier-based order")
            vars_by_tier = defaultdict(list)
            for v in self.vset:
                vars_by_tier[get_tier(v)].append(v)
            order = []
            for tier in sorted(vars_by_tier.keys()):
                order.extend(sorted(vars_by_tier[tier]))
            return {v: i for i, v in enumerate(order)}

    def sample_dag(self, rng):
        has_samples = any(len(s) > 0 for s in self.mcmc_samples.values())
        if has_samples:
            return self._sample_from_mc3(rng)
        else:
            return self._sample_perturbation(rng)

    def _sample_from_mc3(self, rng):
        edges = set()
        # Intra-block: sample from posterior DAGs
        for block_name in self.block_names:
            samples = self.mcmc_samples[block_name]
            if not samples:
                continue
            idx = rng.integers(len(samples))
            dag_sample = samples[idx]
            for e in dag_sample:
                u, v = str(e[0]), str(e[1])
                if u in self.vset and v in self.vset:
                    edges.add((u, v))
        # Inter-block: Bernoulli sample from posterior edge probabilities
        for (u, v), p in zip(self.inter_edges, self.inter_probs):
            if rng.random() < p:
                edges.add((u, v))
        return self._enforce_topological(list(edges))

    def _sample_perturbation(self, rng):
        # Fallback when no MCMC samples available
        edges = set(self.backbone)
        if self.backbone:
            n_rm = max(1, int(len(self.backbone) * rng.uniform(0.05, 0.15)))
            probs = np.array([self.emeta.get(e, 0.5) for e in self.backbone])
            w = np.maximum(1.0 - probs, 0.01)
            w /= w.sum()
            for idx in rng.choice(len(self.backbone),
                                  min(n_rm, len(self.backbone)),
                                  replace=False, p=w):
                edges.discard(self.backbone[idx])
        if self.inter_edges:
            n_add = max(1, int(len(self.inter_edges) * rng.uniform(0.1, 0.3)))
            w = np.maximum(np.array(self.inter_probs), 0.01)
            w /= w.sum()
            for idx in rng.choice(len(self.inter_edges),
                                  min(n_add, len(self.inter_edges)),
                                  replace=False, p=w):
                edges.add(self.inter_edges[idx])
        return self._enforce_topological(list(edges))

    def _enforce_topological(self, edges):
        # Remove forbidden edges (violate causal tiers) and enforce max in-degree
        consistent = []
        for u, v in edges:
            if is_edge_forbidden(u, v):
                continue
            ru = self.topo_rank.get(u)
            rv = self.topo_rank.get(v)
            if ru is not None and rv is not None:
                if ru < rv:
                    consistent.append((u, v))
            else:
                if get_tier(u) <= get_tier(v):
                    consistent.append((u, v))

        consistent_p = [(u, v, self.emeta.get((u, v), 0.5))
                        for u, v in consistent]
        consistent_p.sort(key=lambda x: x[2], reverse=True)

        in_deg = defaultdict(int)
        kept = []
        for u, v, p in consistent_p:
            if in_deg[v] >= self.max_indeg:
                continue
            in_deg[v] += 1
            kept.append((u, v))
        return kept

    def get_backbone(self):
        return list(self.backbone)

    def get_posterior_edge_probs(self):
        edge_count = Counter()
        total_samples = 0
        for block_name, samples in self.mcmc_samples.items():
            for dag in samples:
                total_samples += 1
                for e in dag:
                    edge_count[tuple(e)] += 1
        if total_samples == 0:
            return {}
        probs = {}
        for block_name, samples in self.mcmc_samples.items():
            n_block = len(samples)
            if n_block == 0:
                continue
            block_edge_count = Counter()
            for dag in samples:
                for e in dag:
                    block_edge_count[tuple(e)] += 1
            for e, cnt in block_edge_count.items():
                probs[e] = cnt / n_block
        return probs


# 5. ANCESTRAL SAMPLER

class Sampler:
    def __init__(self, dag_edges, variables, cpd):
        self.cpd = cpd
        vs = set(variables)
        self.parents = defaultdict(list)
        for u, v in dag_edges:
            if u in vs and v in vs:
                self.parents[v].append(u)
        self.variables = self._topo(variables, dag_edges)
        self.ci = {v: i for i, v in enumerate(self.variables)}

    def _topo(self, variables, edges):
        # Kahn's algorithm for topological sort
        vs = set(variables)
        ind = defaultdict(int)
        ch = defaultdict(list)
        for u, v in edges:
            if u in vs and v in vs:
                ind[v] += 1
                ch[u].append(v)
        q = [v for v in variables if ind.get(v, 0) == 0]
        order = []
        vis = set()
        while q:
            n = q.pop(0)
            if n in vis:
                continue
            vis.add(n)
            order.append(n)
            for c in ch.get(n, []):
                ind[c] -= 1
                if ind[c] == 0:
                    q.append(c)
        for v in variables:
            if v not in vis:
                order.append(v)
        return order

    def sample(self, n, rng, sample_theta=True, evidence=None):
        nv = len(self.variables)
        data = np.zeros((n, nv), dtype=int)
        ev_set = set()
        if evidence:
            ev_set = set(evidence.keys())
            for var, val in evidence.items():
                j = self.ci.get(var)
                if j is not None:
                    data[:, j] = val

        for var in self.variables:
            if var in ev_set:
                continue
            j = self.ci[var]
            K = self.cpd._K(var)
            pars = self.parents.get(var, [])

            if not pars:
                theta = self.cpd.get_theta(var, [], {}, rng, sample_theta)
                data[:, j] = rng.choice(K, size=n, p=theta)
            else:
                par_ci = [(p, self.ci.get(p)) for p in pars]
                valid = [(p, pi) for p, pi in par_ci if pi is not None]
                if not valid:
                    theta = self.cpd.get_theta(var, [], {}, rng, sample_theta)
                    data[:, j] = rng.choice(K, size=n, p=theta)
                    continue
                vpars = [p for p, _ in valid]
                vpi = [pi for _, pi in valid]
                pdata = data[:, vpi]
                uniq = np.unique(pdata, axis=0)
                for config in uniq:
                    mask = np.all(pdata == config, axis=1)
                    nm = mask.sum()
                    if nm == 0:
                        continue
                    pv_dict = {p: int(v) for p, v in zip(vpars, config)}
                    theta = self.cpd.get_theta(
                        var, vpars, pv_dict, rng, sample_theta)
                    data[mask, j] = rng.choice(K, size=nm, p=theta)
        return data


# 6. MARTINS ALGORITHM 1

def generate_martins(ensemble, cpd, variables, cfg,
                     method_label="S1 (full Bayes MC3)"):
    rng = np.random.default_rng(cfg.SEED)
    M, n_per = cfg.M, cfg.N_PER

    print(f"\n{'=' * 70}")
    print(f"MARTINS ALGORITHM 1 — {method_label}")
    print(f"  M={M} DAG draws, n_per={n_per}, total={M * n_per}")
    print(f"{'=' * 70}")

    all_data = []
    edge_counts = []
    per_draw_tvds = []

    orig_marginals = {}
    for var in variables:
        if var in cpd.col_idx:
            ci = cpd.col_idx[var]
            K = cpd._K(var)
            counts = cpd.marginals.get(var, np.zeros(K))
            orig_marginals[var] = counts / max(counts.sum(), 1)

    for m in tqdm(range(M), desc=method_label):
        # Step 1: G^(m) ~ p(G|X)
        dag = ensemble.sample_dag(rng)
        edge_counts.append(len(dag))

        # Steps 2+3: theta^(m) ~ p(theta|G^(m),X), Y^(m) ~ p(Y|G^(m),theta^(m))
        s = Sampler(dag, variables, cpd)
        Y = s.sample(n_per, rng, sample_theta=True)

        reorder = [s.ci[v] for v in variables if v in s.ci]
        Y_reordered = Y[:, reorder]
        all_data.append(Y_reordered)

        # Step 4: compute h(Y^(m)) — TVD statistic per draw
        draw_tvd = {}
        for var in variables:
            if var not in orig_marginals:
                continue
            vi = variables.index(var)
            if vi >= Y_reordered.shape[1]:
                continue
            K = cpd._K(var)
            synth_counts = np.bincount(Y_reordered[:, vi].astype(int),
                                        minlength=K)[:K].astype(float)
            synth_p = synth_counts / max(synth_counts.sum(), 1)
            orig_p = orig_marginals[var]
            draw_tvd[var] = float(0.5 * np.abs(orig_p - synth_p).sum())
        per_draw_tvds.append(draw_tvd)

    arr = np.vstack(all_data)
    df = pd.DataFrame(arr, columns=variables[:arr.shape[1]])
    df['_draw'] = np.repeat(np.arange(M), n_per)[:len(df)]

    diag = {
        'M': M, 'n_per': n_per, 'total': len(df),
        'method': method_label,
        'edges_mean': float(np.mean(edge_counts)),
        'edges_std': float(np.std(edge_counts)),
        'edges_min': int(np.min(edge_counts)),
        'edges_max': int(np.max(edge_counts)),
        'per_draw_tvds': per_draw_tvds,
    }

    print(f"  OK {len(df)} obs | edges: "
          f"{np.mean(edge_counts):.0f}+-{np.std(edge_counts):.0f} "
          f"[{np.min(edge_counts)}, {np.max(edge_counts)}]")
    return df, diag


# 7. PROFILE PREDICTOR

class Predictor:
    def __init__(self, ensemble, cpd, variables):
        self.ensemble = ensemble
        self.cpd = cpd
        self.variables = variables

    def predict(self, target, profile, M=50, n_sim=500, seed=42):
        rng = np.random.default_rng(seed)
        K = self.cpd._K(target)
        all_probs = np.zeros((M, K))
        for m in range(M):
            dag = self.ensemble.sample_dag(rng)
            s = Sampler(dag, self.variables, self.cpd)
            Y = s.sample(n_sim, rng, sample_theta=True, evidence=profile)
            ti = s.ci.get(target)
            if ti is not None:
                vals = Y[:, ti].astype(int)
                counts = np.bincount(vals, minlength=K)[:K]
                all_probs[m] = counts / max(counts.sum(), 1)
        mean_p = all_probs.mean(axis=0)
        mean_p = np.maximum(mean_p, 1e-10)
        mean_p /= mean_p.sum()
        mode_k = int(np.argmax(mean_p))
        draws = all_probs[:, mode_k]
        labels = [str(l) for l in self.cpd.binfo.get(target, {}).get(
            'labels', list(range(K)))]
        return {
            'target': target, 'profile': profile, 'K': K,
            'probs': mean_p.tolist(),
            'mode': mode_k,
            'mode_label': labels[mode_k] if mode_k < len(labels) else str(mode_k),
            'mode_prob': float(mean_p[mode_k]),
            'ci95': (float(np.percentile(draws, 2.5)),
                     float(np.percentile(draws, 97.5))),
            'labels': labels,
        }

    def predict_batch(self, target, profiles, M=30, n_sim=300):
        rows = []
        for i, prof in enumerate(profiles):
            print(f"    Profile {i + 1}/{len(profiles)}: {prof}")
            r = self.predict(target, prof, M=M, n_sim=n_sim, seed=42 + i)
            row = dict(prof)
            row['mode'] = r['mode_label']
            row['P(mode)'] = f"{r['mode_prob']:.3f}"
            row['CI95'] = f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
            for k, p in enumerate(r['probs']):
                lab = r['labels'][k] if k < len(r['labels']) else str(k)
                row[f'P({lab})'] = f"{p:.3f}"
            rows.append(row)
        return pd.DataFrame(rows)


# 8. VALIDATION

def validate(orig, synth, variables, cfg):
    print(f"\n{'=' * 70}")
    print("VALIDATION (TVD: synthetic vs original)")
    print(f"{'=' * 70}")
    tvds = {}
    for var in variables:
        if var not in orig.columns or var not in synth.columns:
            continue
        if var == '_draw':
            continue
        K = max(int(orig[var].max()), int(synth[var].max())) + 1
        p = np.bincount(orig[var].values.astype(int), minlength=K)[:K].astype(float)
        q = np.bincount(synth[var].values.astype(int), minlength=K)[:K].astype(float)
        p = np.maximum(p, 1e-10); p /= p.sum()
        q = np.maximum(q, 1e-10); q /= q.sum()
        tvds[var] = float(0.5 * np.abs(p - q).sum())
    v = list(tvds.values())
    if not v:
        return tvds
    n_excellent  = sum(1 for t in v if t < cfg.TVD_EXCELLENT)
    n_good       = sum(1 for t in v if cfg.TVD_EXCELLENT <= t < cfg.TVD_GOOD)
    n_acceptable = sum(1 for t in v if t < cfg.TVD_GOOD)
    n_alarm      = sum(1 for t in v if t >= cfg.TVD_ALARM)
    print(f"  TVD mean: {np.mean(v):.4f} | median: {np.median(v):.4f}")
    print(f"  Excellent (<{cfg.TVD_EXCELLENT}): {n_excellent} ({100*n_excellent/len(v):.1f}%)")
    print(f"  Good ({cfg.TVD_EXCELLENT}-{cfg.TVD_GOOD}): {n_good}")
    print(f"  Needs improvement (>={cfg.TVD_ALARM}): {n_alarm}")
    verdict = ("EXCELLENT" if n_excellent/len(v) >= 0.9 else
               "GOOD"      if n_acceptable/len(v) >= 0.9 else
               "ACCEPTABLE" if n_alarm < 0.1*len(v) else
               "NEEDS IMPROVEMENT")
    print(f"  VERDICT: {verdict}")
    sv = sorted(tvds.items(), key=lambda x: x[1])
    print(f"  Best 5:  {', '.join(f'{var}={t:.3f}' for var, t in sv[:5])}")
    print(f"  Worst 5: {', '.join(f'{var}={t:.3f}' for var, t in sv[-5:])}")
    return tvds


# 9. DIRECTIONAL COHERENCE TEST

def check_directional_coherence(ensemble, n_tests=100, seed=42):
    print(f"\n{'=' * 70}")
    print("DIRECTIONAL COHERENCE TEST")
    print(f"{'=' * 70}")
    rng = np.random.default_rng(seed)
    # (downstream_var, upstream_var, description) — paths here would violate causal order
    assertions = [
        ('ESTUDIOS', 'BIRTH',  'education -> birth_year'),
        ('NAT_DEGR', 'BIRTH',  'education -> birth_year'),
        ('NAT_DEGR', 'SEX',    'education -> sex'),
        ('NAT_INC',  'SEX',    'income -> sex'),
        ('V27',      'BIRTH',  'WTP -> birth_year'),
        ('V26',      'BIRTH',  'WTP -> birth_year'),
        ('V52',      'V15',    'recycling -> env_concern'),
        ('V50',      'V15',    'reduce_consumption -> env_concern'),
    ]
    print(f"  Testing {n_tests} sampled DAGs for forbidden paths...")
    violations = defaultdict(int)
    for t in range(n_tests):
        dag = ensemble.sample_dag(rng)
        G = nx.DiGraph(dag)
        for downstream, upstream, desc in assertions:
            if downstream not in G or upstream not in G:
                continue
            if nx.has_path(G, downstream, upstream):
                violations[desc] += 1
    if not violations:
        print(f"  ALL TESTS PASSED — No forbidden paths detected")
    else:
        print(f"  VIOLATIONS DETECTED:")
        for desc, count in sorted(violations.items(), key=lambda x: -x[1]):
            print(f"    {desc}: {count}/{n_tests} ({100*count/n_tests:.1f}%)")
    return violations


# 10. ENRICHMENT

def enrich(df, binfo):
    out = df.copy()
    if 'SEX' in out.columns:
        out['gender'] = out['SEX'].map({0: 'Male', 1: 'Female'}).fillna('?')
    if 'BIRTH' in out.columns:
        bi = binfo.get('BIRTH', {})
        if 'edges' in bi:
            e = bi['edges']
            labels = {i: f"{int(2023 - e[i + 1])}-{int(2023 - e[i])}"
                      for i in range(len(e) - 1)}
            out['age_group'] = out['BIRTH'].map(labels).fillna('?')
    if 'NAT_DEGR' in out.columns:
        edu = {0: 'No formal', 1: 'Primary', 2: 'Secondary',
               3: 'Vocational', 4: 'University', 5: 'Postgrad'}
        out['education'] = out['NAT_DEGR'].map(
            {i: edu.get(i, f'L{i}') for i in range(10)}).fillna('?')
    return out


# 11. DATA-DRIVEN MAX_INDEGREE

def compute_optimal_indegree(cpd, common, tau=5):
    # d* = floor(log(n/(K*tau)) / log(K)) — ensures tau obs per CPT cell
    print("\n4. Computing optimal MAX_INDEGREE...")
    Ks = np.array([cpd._K(v) for v in common])
    K_median = np.median(Ks)
    d_formula = int(np.floor(
        np.log(cpd.n / (K_median * tau)) / np.log(K_median)
    ))
    d_formula = max(1, d_formula)
    configs = int(K_median ** d_formula)
    obs_per_config = cpd.n / configs
    eff_per_cell = cpd.n / (K_median * configs)
    print(f"   n={cpd.n}, K_median={K_median:.0f}, tau={tau}")
    print(f"   d* = {d_formula}  configs={configs}  "
          f"obs/config={obs_per_config:.1f}  eff/cell={eff_per_cell:.1f} "
          f"({'OK' if eff_per_cell >= tau else 'WARN'})")
    return d_formula, K_median, tau


# 12. LITERATURE-ONLY GENERATION — WITH CONDITIONAL CPTs

def _get_individual_config_key(row, parent_variables, parent_cis_vars,
                               parent_levels, cis_level_mappings):
    # Build profile key from CIS row: 'female|university|under_30|...'
    # Used to look up alpha_ab in conditional CPT
    levels_out = []
    for parent_harm in parent_variables:
        cis_var = parent_cis_vars.get(parent_harm)
        available_levels = parent_levels.get(parent_harm, [])
        if not available_levels:
            return None
        default_level = available_levels[len(available_levels) // 2]

        if cis_var is None or cis_var not in row.index:
            levels_out.append(default_level)
            continue

        raw_val = row.get(cis_var)
        if pd.isna(raw_val):
            levels_out.append(default_level)
            continue

        val = int(raw_val)
        mapping = cis_level_mappings.get(cis_var, {})

        # Handle cutpoints (age, birth year)
        if isinstance(mapping, dict) and mapping.get('type') in (
                'ordinal_cutpoints', 'birth_year_cutpoints'):
            mtype    = mapping['type']
            cutpoints = mapping.get('cutpoints', [])
            lvls     = mapping.get('levels', available_levels)
            missing_raw = mapping.get('missing', set())
            if isinstance(missing_raw, (list, set)):
                missing = {int(x) for x in missing_raw if str(x).lstrip('-').isdigit()}
            else:
                missing = set()
            if val in missing:
                levels_out.append(default_level)
                continue
            if mtype == 'birth_year_cutpoints':
                survey_year = mapping.get('survey_year', 2023)
                val = survey_year - val
            idx = int(np.searchsorted(cutpoints, val, side='left'))
            idx = min(idx, len(lvls) - 1)
            level = lvls[idx]
        else:
            # Direct mapping
            level = mapping.get(val) or mapping.get(str(val))

        if level is None or level not in available_levels:
            level = default_level

        levels_out.append(level)

    return '|'.join(levels_out)


def generate_literature_only(pr, n_total, M, seed,
                             cis_df=None, cis_level_mappings=None):
    # Generate variables not observed in CIS using literature priors.
    # Case A (with conditional_cpt + CIS data):
    #   Per individual: lookup profile -> alpha_ab -> theta ~ Dir(alpha_ab)
    #   Real demographic differentiation (Martins Algorithm 1, step 2)
    # Case B (fallback marginal — no CPT or no CIS data):
    #   theta ~ Dir(alpha_marginal) — homogeneous for all individuals
    print(f"\n{'=' * 70}")
    print("LITERATURE-ONLY VARIABLES (with conditional CPTs)")
    print("Martins Algorithm 1, step 2: theta_ab ~ Dir(alpha_ab)")
    print(f"{'=' * 70}")

    rng = np.random.default_rng(seed)
    cis_mappings = cis_level_mappings or {}
    lit_vars = {}
    lit_columns = {}

    for vname, vdata in pr.items():
        is_lit_only = (
            vdata.get('source') == 'literature_only'
            or vdata.get('n_obs_cis', 0) == 0
        )
        if not is_lit_only:
            continue

        alpha = np.array(
            vdata.get('alpha_lit',
            vdata.get('alpha_posterior', [])),
            dtype=float
        )
        if len(alpha) < 2 or alpha.sum() <= 1:
            continue

        lit_vars[vname] = {
            'alpha':     alpha,
            'K':         len(alpha),
            'n_studies': vdata.get('n_studies_lit', 0),
            'note':      vdata.get('note', ''),
            'has_cpt':   'conditional_cpt' in vdata,
            'cpt':       vdata.get('conditional_cpt'),
        }

    if not lit_vars:
        print("  No literature-only variables found.")
        return {}, {}

    n_with_cpt    = sum(1 for v in lit_vars.values() if v['has_cpt'])
    n_without_cpt = len(lit_vars) - n_with_cpt
    print(f"\n  Found {len(lit_vars)} lit-only variables:")
    print(f"    With conditional_cpt (demographic differentiation): {n_with_cpt}")
    print(f"    Without conditional_cpt (marginal, fallback):       {n_without_cpt}")

    n_per_draw = max(1, n_total // M)

    for vname, info in lit_vars.items():
        alpha_marginal = info['alpha']
        K    = info['K']
        cpt  = info.get('cpt')

        # Case A: conditional_cpt + CIS data available
        if cpt is not None and cis_df is not None and len(cis_df) > 0:
            parent_variables = cpt['parent_variables']
            parent_cis_vars  = cpt['parent_cis_vars']
            parent_levels    = cpt['parent_levels']
            alpha_per_config = cpt['alpha_per_parent_config']
            positive_cats    = cpt.get('positive_categories', [3, 4])

            cis_rows = cis_df.reset_index(drop=True)
            n_cis    = len(cis_rows)

            # Sample individuals with replacement if n_total > n_cis
            idx = rng.integers(0, n_cis, size=n_total)
            vals_out   = np.zeros(n_total, dtype=int)
            n_found    = 0
            n_fallback = 0

            for i, cis_idx in enumerate(idx):
                row = cis_rows.iloc[cis_idx]
                config_key = _get_individual_config_key(
                    row, parent_variables, parent_cis_vars,
                    parent_levels, cis_mappings
                )

                if config_key is not None and config_key in alpha_per_config:
                    alpha_ab = np.array(alpha_per_config[config_key],
                                        dtype=float)
                    alpha_ab = np.maximum(alpha_ab, 1e-4)
                    theta    = rng.dirichlet(alpha_ab)
                    n_found += 1
                else:
                    theta = rng.dirichlet(alpha_marginal)
                    n_fallback += 1

                theta = np.maximum(theta, 1e-10)
                theta /= theta.sum()
                vals_out[i] = rng.choice(K, p=theta)

            lit_columns[vname] = vals_out

            props_out  = np.bincount(vals_out, minlength=K)[:K].astype(float)
            props_out /= props_out.sum()
            p_positive = props_out[positive_cats].sum() if positive_cats else 0.0
            fallback_pct = 100 * n_fallback / n_total

            print(f"\n    {vname}")
            print(f"      Method: conditional_cpt")
            print(f"      Configs found: {n_found}/{n_total}  "
                  f"(fallback: {n_fallback}, {fallback_pct:.1f}%)")
            print(f"      P(positive) generated: {p_positive:.1%}  "
                  f"(base_rate={cpt.get('base_rate', '?'):.2f})")
            print(f"      CPT range: "
                  f"[{cpt.get('p_min', 0):.1%}, {cpt.get('p_max', 0):.1%}]")

            if fallback_pct > 20:
                print(f"      WARNING: {fallback_pct:.0f}% marginal fallback")
                print(f"        Review cis_level_mappings for {vname}")

        # Case B: marginal fallback
        else:
            reason = "no conditional_cpt" if cpt is None else "no CIS data"
            all_vals = []
            for m in range(M):
                theta_m = rng.dirichlet(alpha_marginal)
                vals_m  = rng.choice(K, size=n_per_draw, p=theta_m)
                all_vals.append(vals_m)
            all_vals = np.concatenate(all_vals)
            if len(all_vals) < n_total:
                extra = rng.choice(K, size=n_total - len(all_vals),
                                   p=rng.dirichlet(alpha_marginal))
                all_vals = np.concatenate([all_vals, extra])
            all_vals = all_vals[:n_total]
            lit_columns[vname] = all_vals

            props = alpha_marginal / alpha_marginal.sum()
            print(f"\n    {vname}")
            print(f"      Method: marginal (fallback, {reason})")
            print(f"      E[theta]=[{', '.join(f'{p:.2f}' for p in props)}]")

    print(f"\n  DIFFERENTIATION:")
    if n_with_cpt > 0:
        print(f"    CPT variables: real demographic differentiation")
        print(f"    Same person with different profile -> different distribution")
    print(f"    Marginal variables: homogeneous (same for everyone)")

    return lit_vars, lit_columns


# MAIN

def main():
    cfg = Config()
    os.makedirs(cfg.OUTDIR, exist_ok=True)

    print("=" * 70)
    print("STEP 4 — SYNTHETIC SURVEY GENERATION (MC3 + CONDITIONAL CPTs)")
    print("Martins et al. (2024) Algorithm 1")
    print("pi(Y|X) = integral p(Y|G,theta) p(G,theta|X) dtheta dG")
    print("=" * 70)

    # 1. Load
    print(f"\n{'=' * 70}")
    print("1. LOADING DATA")
    print(f"{'=' * 70}")

    df               = load_cis_microdata(cfg)
    dag_edges, emeta = load_dag_structure(cfg)
    mcmc_samples     = load_mcmc_samples(cfg)
    pr               = load_fused_priors(cfg)
    ep               = load_edge_prob_matrix(cfg)

    print("  1f. CIS level mappings...")
    cis_level_mappings = load_cis_level_mappings(cfg)

    # 2. Discretise
    dd, binfo = discretise(df, cfg.N_BINS)

    dag_vars = set()
    for u, v in dag_edges:
        dag_vars.add(u)
        dag_vars.add(v)
    common = [v for v in dd.columns if v in dag_vars]
    print(f"\n   Common variables (CIS n DAG): {len(common)}")

    # 3. CPD engine
    print("\n3. Dirichlet-BDeu CPD engine...")
    cpd = DirichletBDeuCPD(dd[common], binfo, pr, cfg.ESS)

    # 4. Optimal indegree
    d_star, K_median, tau = compute_optimal_indegree(cpd, common, tau=5)
    cfg.MAX_INDEGREE = d_star

    # 5. MC3 DAG ensemble
    print(f"\n5. MC3 DAG ensemble (topological, indeg<={cfg.MAX_INDEGREE})...")
    ensemble = MC3DAGEnsemble(
        mcmc_samples=mcmc_samples,
        backbone_edges=[(u, v) for u, v in dag_edges
                        if u in set(common) and v in set(common)],
        emeta=emeta,
        all_vars=common,
        max_indeg=cfg.MAX_INDEGREE,
    )

    coherence_violations = check_directional_coherence(ensemble, n_tests=100)

    # 6. Generate S1 — full Bayes (Algorithm 1)
    s1, d1 = generate_martins(
        ensemble, cpd, common, cfg,
        method_label="S1 (full Bayes MC3 + intl priors)"
    )

    # 7. Generate S2 — flat BDeu (baseline comparison)
    print("\n   Building S2 CPD (BDeu flat, no intl priors)...")
    cpd_flat = DirichletBDeuCPD(dd[common], binfo, {}, cfg.ESS)
    s2, d2 = generate_martins(
        ensemble, cpd_flat, common, cfg,
        method_label="S2 (MC3 + BDeu flat)"
    )

    # 8. Literature-only variables
    lit_vars, lit_columns = generate_literature_only(
        pr, len(s1), cfg.M, cfg.SEED + 999,
        cis_df=df,
        cis_level_mappings=cis_level_mappings,
    )
    for vname, vals in lit_columns.items():
        s1[vname] = vals
    if lit_vars:
        d1['literature_only_vars'] = list(lit_vars.keys())
        d1['n_new_vars']           = len(lit_vars)
        n_with_cpt = sum(1 for v in lit_vars.values() if v.get('has_cpt'))
        print(f"\n  S1: {len(s1.columns)} cols "
              f"(+{len(lit_vars)} lit-only, {n_with_cpt} with demographic CPT)")
        print(f"     S2: {len(s2.columns)} cols (CIS-only)")

    # 9. Enrich
    s1 = enrich(s1, binfo)

    # 10. Marginal comparison
    print(f"\n{'=' * 70}")
    print("MARGINAL COMPARISON (original vs synthetic)")
    print(f"{'=' * 70}")
    for var in ['SEX', 'V27', 'V26', 'NAT_DEGR', 'V15']:
        if var in dd.columns and var in s1.columns:
            op = dd[var].value_counts(normalize=True).sort_index()
            sp = s1[var].value_counts(normalize=True).sort_index()
            allv = sorted(set(op.index) | set(sp.index))
            print(f"\n  {var}:")
            for v in allv:
                print(f"    {v}: orig={op.get(v, 0):.3f}  "
                      f"synth={sp.get(v, 0):.3f}")

    # 11. Profile predictions
    print(f"\n{'=' * 70}")
    print("PROFILE PREDICTIONS")
    print(f"{'=' * 70}")
    predictor = Predictor(ensemble, cpd, common)

    targets = {}
    for h, cs in CIS_TO_HARM.items():
        for c in cs:
            if c in common:
                targets[h] = c
                break

    demo = [v for v in ['SEX', 'NAT_DEGR', 'URBRURAL', 'BIRTH'] if v in common]
    pred_df = None
    tn = 'attitudes_wtp_prices'
    if tn in targets:
        tv = targets[tn]
        print(f"\n  Predicting: {tn} ({tv})")
        profiles = []
        if 'SEX' in demo and 'NAT_DEGR' in demo:
            for sex in [0, 1]:
                for edu in [1, 3, 4]:
                    profiles.append({'SEX': sex, 'NAT_DEGR': edu})
        if profiles:
            pred_df = predictor.predict_batch(tv, profiles, M=30, n_sim=300)
            print(f"\n{pred_df.to_string(index=False)}")

    # 12. Validate
    val = validate(dd[common], s1, common, cfg)

    # 13. Save
    print(f"\n{'=' * 70}")
    print("SAVING")
    print(f"{'=' * 70}")

    final = s1
    if len(s1) > cfg.N_FINAL:
        final = s1.sample(cfg.N_FINAL, random_state=cfg.SEED).reset_index(drop=True)

    final.to_csv(os.path.join(cfg.OUTDIR, "synthetic_surveys_S1.csv"), index=False)
    print(f"  OK synthetic_surveys_S1.csv ({len(final)} obs)")

    s2.to_csv(os.path.join(cfg.OUTDIR, "synthetic_surveys_S2.csv"), index=False)
    print(f"  OK synthetic_surveys_S2.csv ({len(s2)} obs)")

    if pred_df is not None:
        pred_df.to_csv(os.path.join(cfg.OUTDIR, "profile_predictions.csv"), index=False)
        print(f"  OK profile_predictions.csv")

    v = list(val.values())
    meta = {
        'method':   'Martins et al. (2024) Algorithm 1 — conditional CPTs',
        'formula':  'pi(Y|X) = integral p(Y|G,theta) p(G,theta|X) dtheta dG',
        'coherence_test': {
            'n_tests':    100,
            'violations': coherence_violations,
            'status':     'PASS' if not coherence_violations else 'FAIL',
        },
        'max_indegree': {
            'value':    cfg.MAX_INDEGREE,
            'method':   'd* = floor(log(n/(K*tau)) / log(K))',
            'n':        int(cpd.n),
            'K_median': float(K_median),
            'tau':      int(tau),
        },
        'S1':          d1,
        'S2':          d2,
        'tvd_mean':    float(np.mean(v)) if v else None,
        'tvd_median':  float(np.median(v)) if v else None,
        'lit_only_vars': {
            vname: {
                'has_cpt':    info.get('has_cpt', False),
                'K':          info['K'],
                'n_studies':  info['n_studies'],
                'p_min':      info['cpt'].get('p_min') if info.get('cpt') else None,
                'p_max':      info['cpt'].get('p_max') if info.get('cpt') else None,
            }
            for vname, info in lit_vars.items()
        },
    }

    with open(os.path.join(cfg.OUTDIR, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  OK metadata.json")

    with open(os.path.join(cfg.OUTDIR, "validation.json"), 'w') as f:
        json.dump(val, f, indent=2, default=str)
    print(f"  OK validation.json")

    # Summary
    print(f"\n{'=' * 70}")
    print("STEP 4 COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  S1: {len(final)} obs x {len(final.columns)} vars")
    print(f"  S2: {len(s2)} obs x {len(s2.columns)} vars")

    if coherence_violations:
        print(f"\n  COHERENCE TEST FAILED:")
        for desc, count in coherence_violations.items():
            print(f"    {desc}: {count}/100 violations")
    else:
        print(f"\n  COHERENCE TEST PASSED (100/100)")

    if v:
        n_excellent = sum(1 for t in v if t < cfg.TVD_EXCELLENT)
        print(f"  Quality: {n_excellent}/{len(v)} excellent "
              f"(TVD < {cfg.TVD_EXCELLENT})")

    print(f"\n  Literature-only variables generated with demographic CPT:")
    for vname, info in lit_vars.items():
        if info.get('has_cpt') and info.get('cpt'):
            cpt = info['cpt']
            print(f"    {vname:<45s} "
                  f"p=[{cpt.get('p_min',0):.1%}, {cpt.get('p_max',0):.1%}]")

    print(f"\n  NEXT: python step5_.py")
    print(f"    step5 loads synthetic_surveys_S1.csv")
    print(f"    and generates comparative profiles for Gradio app")


if __name__ == "__main__":
    main()
