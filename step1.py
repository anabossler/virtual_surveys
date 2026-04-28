# step1_improved_v6.py
# Bayesian backbone learning from CIS 3391 microdata.
#
# Learns a DAG over attitudinal and demographic variables using
# parallel-tempered MC3 (Geyer 1991) with the Martins et al. (2024)
# penalizing prior. No hill climbing -- samples are genuine draws
# from the posterior p(G|X).
#
# v6 design: demographic roots are co-located inside thematic blocks
# rather than handled post-hoc via augment_dag_with_roots. This keeps
# indegree bounded (max 3 within MC3 -> q_j <= 5^3 = 125) and lets
# the sampler learn demographic->attitudinal structure directly.
#
# References:
#   Martins et al. (2024), Algorithm 1, eq. 3-7
#   Goudie & Mukherjee (2016), MCMC for DAGs
#   Geyer (1991), parallel tempering
#   Heckerman et al. (1995), BDeu scoring
#   Cooper & Herskovits (1992), K2

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from tqdm import tqdm
from itertools import combinations, product
from collections import defaultdict, Counter
import networkx as nx

from sklearn.covariance import GraphicalLasso
from sklearn.metrics import mutual_info_score
from pgmpy.models import BayesianNetwork


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_NUM = os.path.join(BASE_DIR, "metadata", "3391_num.csv")

# CIS survey missing-value codes (numeric and string variants)
CIS_MISSING_CODES = {
    'N.P.', 'N.C.', 'N.S.', 'N.D.',
    96, 97, 98, 99,
    996, 997, 998, 999,
    9996, 9997, 9998, 9999
}

# demographic root variables -- treated as exogenous w.r.t. attitudinal items
ROOTS = ['SEX', 'BIRTH', 'ESTUDIOS', 'NAT_INC', 'URBRURAL']


# thematic blocks (v6)
#
# Each block contains the demographic roots that are substantively
# relevant to it, so the sampler learns those relationships directly.
# Block sizes are kept <= 12 for MC3 tractability (Martins sec. 3.2).
# Roots appear first in each list (exogenous ordering).

THEMATIC_BLOCKS = {

    # internal structure of the root variables themselves
    'demographics': [
        'SEX', 'BIRTH', 'ESTUDIOS', 'EDUCYRS',
        'NAT_INC', 'NAT_RINC', 'URBRURAL', 'TOPBOT'
    ],

    # willingness to pay / acceptance of recycled plastics
    'wtp_acceptance': [
        'SEX', 'BIRTH', 'ESTUDIOS',
        'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31'
    ],

    # general environmental attitudes, split for block-size reasons
    'env_concern_a': [
        'SEX', 'BIRTH', 'NAT_INC',
        'V15', 'V16', 'V17', 'V18', 'V19'
    ],
    'env_concern_b': [
        'SEX', 'NAT_INC',
        'V19', 'V20', 'V21', 'V22', 'V23', 'V24'
    ],

    # self-reported pro-environmental behaviour
    'behavior': [
        'SEX', 'ESTUDIOS',
        'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57'
    ],

    # ecological footprint items
    'ecological_footprint': [
        'SEX', 'BIRTH', 'NAT_INC',
        'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V58'
    ],

    # pollution and danger perceptions
    'danger_pollution': [
        'SEX', 'NAT_INC',
        'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43'
    ],

    # attitudes toward science and technology
    'science_tech': [
        'SEX', 'ESTUDIOS',
        'V32', 'V33', 'V34', 'V35', 'V36'
    ],

    # institutional trust
    'trust_institutions': [
        'SEX', 'NAT_INC',
        'V10', 'V11', 'V12', 'V13', 'V14'
    ],

    # political issue positions
    'political_issues': [
        'BIRTH', 'NAT_INC',
        'V1', 'V2', 'V3', 'V4', 'V5'
    ],

    # political values (small block, no roots needed)
    'political_values': ['V6', 'V7', 'V8', 'V9'],

    # ideological self-placement and religiosity
    'political_ideology': [
        'SEX', 'BIRTH', 'ESTUDIOS',
        'LEFT_RIGHT', 'IDEOL_CATEG_01',
        'IDEOL_CATEG_02', 'ATTEND', 'NAT_RELIG'
    ],

    # life satisfaction and economic hardship
    'satisfaction': [
        'SEX', 'BIRTH', 'NAT_INC',
        'C_SATISFTRABAJO', 'C_SATISFESTUDIOS',
        'DIFICULT_ECO', 'MERIT', 'PREF_TRABINGRESOS'
    ],

    # labour market variables
    'employment': [
        'SEX', 'BIRTH', 'ESTUDIOS',
        'ISCO08', 'NACE_2', 'TYPORG1',
        'MAINSTAT', 'EMPREL', 'SPEMPREL'
    ],

    # national origin and family background
    'origin_family': [
        'LUGAR_NAC', 'PAIS_NACIMIENTO', 'M_BORN', 'F_BORN',
        'FATH_NAT_DEGR', 'MOTH_NAT_DEGR', 'NAT_ETHN_01', 'SPNAT_DEGR'
    ],
}

MAX_BLOCK_SIZE = 12


# forbidden edges
#
# Encodes prior causal knowledge about exogeneity and variable ordering.
# See Heckerman et al. (1995) sec. 3.

def build_forbidden_edges():
    forbidden = {}

    # demographics block: EDUCYRS (years studied) is determined by
    # ESTUDIOS (qualification attained), not the other way around
    fb = set()
    fb.add(('ESTUDIOS', 'EDUCYRS'))
    tier1 = ['SEX', 'BIRTH']
    tier2 = ['ESTUDIOS', 'NAT_DEGR', 'EDUCYRS']
    tier3 = ['NAT_INC', 'NAT_RINC', 'URBRURAL', 'TOPBOT']
    all_demo = tier1 + tier2 + tier3
    for parent in all_demo:
        for child in tier1:
            if parent != child:
                fb.add((parent, child))
    for e in tier3:
        for x in tier2:
            fb.add((e, x))
    forbidden['demographics'] = fb

    # in mixed blocks: attitudinal items cannot cause demographic roots
    for block_name, block_vars in THEMATIC_BLOCKS.items():
        if block_name == 'demographics':
            continue
        roots_in_block = [v for v in block_vars if v in ROOTS]
        attitudinal_in_block = [v for v in block_vars if v not in ROOTS]

        fb_block = set()

        for att in attitudinal_in_block:
            for root in roots_in_block:
                fb_block.add((att, root))

        # ordering among roots within a block:
        # SEX and BIRTH are fully exogenous; ESTUDIOS is exogenous
        # relative to income/location; NAT_INC cannot cause any of them
        exog_tier1 = ['SEX', 'BIRTH']
        exog_tier2 = ['ESTUDIOS']
        exog_tier3 = ['NAT_INC', 'URBRURAL']

        for r3 in exog_tier3:
            for r1 in exog_tier1 + exog_tier2:
                if r3 in roots_in_block and r1 in roots_in_block:
                    fb_block.add((r3, r1))

        for r2 in exog_tier2:
            for r1 in exog_tier1:
                if r2 in roots_in_block and r1 in roots_in_block:
                    fb_block.add((r2, r1))

        # SEX and BIRTH are mutually independent -- neither causes the other
        if 'SEX' in roots_in_block and 'BIRTH' in roots_in_block:
            fb_block.add(('BIRTH', 'SEX'))
            fb_block.add(('SEX', 'BIRTH'))

        if fb_block:
            forbidden[block_name] = fb_block

    # V51 (number of rooms) is a housing characteristic,
    # not caused by behaviour items in the same block
    if 'behavior' in forbidden:
        for v in ['V50', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57']:
            forbidden['behavior'].add((v, 'V51'))
    else:
        fb = set()
        for v in ['V50', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57']:
            fb.add((v, 'V51'))
        forbidden['behavior'] = fb

    # V15 and V17 are baseline concern items; more specific attitudes flow from them
    for block_name in ['env_concern_a', 'env_concern_b']:
        fb = forbidden.get(block_name, set())
        for dep in ['V16', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24']:
            for root in ['V15', 'V17']:
                fb.add((dep, root))
        forbidden[block_name] = fb

    # V25 is a baseline belief item in the WTP block
    fb = forbidden.get('wtp_acceptance', set())
    for dep in ['V26', 'V27', 'V28']:
        fb.add((dep, 'V25'))
    forbidden['wtp_acceptance'] = fb

    return forbidden


# data loading

def load_microdata_cis(path):
    print("Loading:", path)
    df = pd.read_csv(
        path, sep=";", encoding="latin1",
        low_memory=False, quotechar='"',
        na_values=['N.P.', 'N.C.', 'N.S.', 'N.D.'],
        keep_default_na=True
    )
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for code in CIS_MISSING_CODES:
        if isinstance(code, (int, float)):
            df = df.replace(code, np.nan)
    df = df.dropna(axis=1, how='all')
    variances = df.std(skipna=True)
    df = df.loc[:, variances > 0]
    missing_pct = df.isnull().sum() / len(df)
    df = df.loc[:, missing_pct < 0.9]
    threshold = int(len(df.columns) * 0.4)
    df = df.dropna(thresh=threshold)
    print(f"  Dataset: {df.shape[0]} obs x {df.shape[1]} vars")
    return df


def filter_collinear(df, threshold=0.95):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"  Collinearity filter: dropped {len(to_drop)}, kept {len(df.columns) - len(to_drop)}")
    return df.drop(columns=to_drop)


# variable classification and discretisation

def classify_variables(df):
    var_types = {}
    for col in df.columns:
        nunique = df[col].nunique()
        if nunique == 2:
            var_types[col] = 'binary'
        elif nunique <= 10:
            var_types[col] = 'ordinal'
        else:
            var_types[col] = 'continuous'
    return var_types


def discretize_for_bdeu(df, var_types, n_bins=5):
    # available-case analysis -- no imputation
    print("\nDiscretisation (available case analysis, no imputation)")
    df_disc = df.copy()
    n_discretized = 0

    for col in df.columns:
        vtype = var_types.get(col, 'continuous')
        if vtype == 'continuous':
            try:
                df_disc[col] = pd.qcut(df[col], q=n_bins,
                                        labels=list(range(n_bins)),
                                        duplicates='drop')
                n_discretized += 1
            except Exception:
                try:
                    df_disc[col] = pd.cut(df[col], bins=n_bins,
                                           labels=list(range(n_bins)),
                                           duplicates='drop')
                    n_discretized += 1
                except Exception:
                    pass
        df_disc[col] = pd.to_numeric(df_disc[col], errors='coerce')

    for col in df_disc.columns:
        df_disc[col] = df_disc[col].astype('Int64')

    pct_na = df_disc.isna().sum().sum() / (df_disc.shape[0] * df_disc.shape[1]) * 100
    print(f"  Discretised: {n_discretized} | Already categorical: {len(df.columns) - n_discretized}")
    print(f"  NaN preserved: {pct_na:.1f}% of cells")
    return df_disc


# BDeu local score

def bdeu_local_score(df_disc, child, parents, ess=10.0, gamma=0.0):
    cols = [child] + list(parents)
    mask = df_disc[cols].notna().all(axis=1)
    subset = df_disc.loc[mask, cols]
    if len(subset) < 5:
        return -1e12
    child_vals = subset[child].values.astype(int)
    parent_vals = subset[list(parents)].values.astype(int) if parents else None
    return _bdeu_local_fast(child_vals, parent_vals, ess=ess, gamma=gamma)


def _bdeu_local_fast(child_vals, parent_vals, ess=10.0, gamma=0.0):
    K_j = int(child_vals.max()) + 1
    if K_j < 2:
        K_j = 2

    if parent_vals is None or (hasattr(parent_vals, 'shape') and
                                len(parent_vals.shape) > 1 and
                                parent_vals.shape[1] == 0):
        counts = np.bincount(child_vals, minlength=K_j)[:K_j].astype(float)
        n_total = counts.sum()
        alpha_k = ess / K_j
        score = gammaln(ess) - gammaln(ess + n_total)
        score += np.sum(gammaln(alpha_k + counts) - gammaln(alpha_k))
        return float(score)

    n_parents = parent_vals.shape[1]
    Ks_parents = np.maximum(parent_vals.max(axis=0) + 1, 2)
    q_j = int(np.prod(Ks_parents))

    alpha_c = ess / q_j
    alpha_ck = ess / (K_j * q_j)

    multipliers = np.ones(n_parents, dtype=np.int64)
    for i in range(n_parents - 2, -1, -1):
        multipliers[i] = multipliers[i + 1] * Ks_parents[i + 1]
    config_keys = (parent_vals * multipliers).sum(axis=1)

    unique_configs = np.unique(config_keys)
    score = 0.0
    for cfg in unique_configs:
        mask_c = (config_keys == cfg)
        child_c = child_vals[mask_c]
        n_c = len(child_c)
        counts = np.bincount(child_c, minlength=K_j)[:K_j].astype(float)
        score += gammaln(alpha_c) - gammaln(alpha_c + n_c)
        score += np.sum(gammaln(alpha_ck + counts) - gammaln(alpha_ck))

    score -= gamma * n_parents
    return float(score)


class BlockScorer:
    def __init__(self, df_disc, block_vars, ess=10.0, gamma=0.0):
        self.vars = list(block_vars)
        self.var_idx = {v: i for i, v in enumerate(self.vars)}
        self.ess = ess
        self.gamma = gamma
        self._cache = {}

        self._col_data = {}
        for v in self.vars:
            vals = df_disc[v].values.copy()
            notna = ~pd.isna(vals)
            vals_clean = np.zeros(len(vals), dtype=np.int64)
            if notna.any():
                vals_clean[notna] = vals[notna].astype(np.int64)
            self._col_data[v] = (vals_clean, notna)

    def local_score(self, node, parents):
        node_idx = self.var_idx[node]
        parent_key = frozenset(self.var_idx[p] for p in parents)
        cache_key = (node_idx, parent_key)

        if cache_key in self._cache:
            return self._cache[cache_key]

        child_data, child_notna = self._col_data[node]
        mask = child_notna.copy()
        for p in parents:
            _, p_notna = self._col_data[p]
            mask &= p_notna

        n_valid = mask.sum()
        if n_valid < 5:
            self._cache[cache_key] = -1e12
            return -1e12

        child_vals = child_data[mask]
        parent_vals = (
            None if len(parents) == 0
            else np.column_stack([self._col_data[p][0][mask] for p in parents])
        )

        score = _bdeu_local_fast(child_vals, parent_vals, ess=self.ess, gamma=self.gamma)
        self._cache[cache_key] = score
        return score

    def dag_score(self, edges):
        total = 0.0
        for node in self.vars:
            parents = [u for (u, v) in edges if v == node]
            total += self.local_score(node, parents)
        return total


# random DAG initialisation

def _random_initial_dag(nodes_list, d, rng, max_indegree,
                         edge_prob=None, forbidden=None):
    if edge_prob is None:
        n_possible = d * (d - 1) / 2
        edge_prob = min(0.3, max(0.05, d / max(n_possible, 1)))
    if forbidden is None:
        forbidden = set()

    perm = rng.permutation(nodes_list)
    G = nx.DiGraph()
    G.add_nodes_from(nodes_list)

    for i in range(d):
        n_parents = 0
        for j in range(i):
            if n_parents >= max_indegree:
                break
            if rng.random() < edge_prob:
                u, v = perm[j], perm[i]
                if (u, v) not in forbidden:
                    G.add_edge(u, v)
                    n_parents += 1

    return G


# MC3 parallel-tempered MCMC

def mcmc_structure_block(df_disc, block_vars, ess=10.0, gamma=0.0,
                         max_indegree=3, n_iter=30000, burn_in=5000,
                         lag=30, seed=42, forbidden=None):
    rng = np.random.default_rng(seed)

    block_vars = [v for v in block_vars if v in df_disc.columns]
    if len(block_vars) < 2:
        return [], 0.0, []
    if forbidden is None:
        forbidden = set()

    nodes_list = list(block_vars)
    d = len(nodes_list)

    print(f"\n  MC3 block ({d} vars): {nodes_list[:5]}{'...' if d > 5 else ''}")

    scorer = BlockScorer(df_disc, block_vars, ess=ess, gamma=gamma)

    if d <= 5:
        temperatures = [1.0, 3.0, 10.0, 50.0]
    elif d <= 8:
        temperatures = [1.0, 2.0, 5.0, 20.0, 100.0]
    else:
        temperatures = [1.0, 1.5, 3.0, 8.0, 25.0, 100.0]

    n_chains = len(temperatures)
    swap_interval = 20

    n_iter = max(n_iter, d * 6000)
    burn_in = n_iter // 4
    lag = max(20, d * 5)

    print(f"    Temperatures: {temperatures}")
    print(f"    Iterations: {n_iter} (burn-in: {burn_in}, lag: {lag})")

    chains, chain_scores, chain_indeg = [], [], []
    for k in range(n_chains):
        G = _random_initial_dag(nodes_list, d, rng, max_indegree, forbidden=forbidden)
        score = scorer.dag_score(list(G.edges()))
        chains.append(G)
        chain_scores.append(score)
        chain_indeg.append(dict(G.in_degree()))

    chain_accepts = [0] * n_chains
    chain_proposals = [0] * n_chains
    n_swaps_proposed = 0
    n_swaps_accepted = 0
    cold_scores_trace = []
    collected_dags = []

    for iteration in range(n_iter):

        for k in range(n_chains):
            T = temperatures[k]
            G = chains[k]
            current_score = chain_scores[k]
            indeg = chain_indeg[k]
            n_edges = G.number_of_edges()

            if n_edges == 0:
                move_type = 'add'
            else:
                r_val = rng.random()
                move_type = ('add' if r_val < 0.4 else
                             'remove' if r_val < 0.7 else 'reverse')

            if move_type == 'add':
                for _ in range(10):
                    u = nodes_list[rng.integers(d)]
                    v = nodes_list[rng.integers(d)]
                    if u == v or G.has_edge(u, v):
                        continue
                    if indeg.get(v, 0) >= max_indegree:
                        continue
                    if (u, v) in forbidden:
                        continue
                    G.add_edge(u, v)
                    if nx.is_directed_acyclic_graph(G):
                        old_parents_v = [p for p in G.predecessors(v) if p != u]
                        new_parents_v = list(G.predecessors(v))
                        delta = (scorer.local_score(v, new_parents_v) -
                                 scorer.local_score(v, old_parents_v))
                        new_score = current_score + delta
                        chain_proposals[k] += 1
                        log_ratio = delta / T
                        if log_ratio >= 0 or np.log(rng.random()) < log_ratio:
                            chain_scores[k] = new_score
                            indeg[v] = indeg.get(v, 0) + 1
                            chain_accepts[k] += 1
                        else:
                            G.remove_edge(u, v)
                        break
                    else:
                        G.remove_edge(u, v)

            elif move_type == 'remove' and n_edges > 0:
                edge_list = list(G.edges())
                u, v = edge_list[rng.integers(len(edge_list))]
                old_parents_v = list(G.predecessors(v))
                new_parents_v = [p for p in old_parents_v if p != u]
                delta = (scorer.local_score(v, new_parents_v) -
                         scorer.local_score(v, old_parents_v))
                new_score = current_score + delta
                chain_proposals[k] += 1
                log_ratio = delta / T
                if log_ratio >= 0 or np.log(rng.random()) < log_ratio:
                    G.remove_edge(u, v)
                    chain_scores[k] = new_score
                    indeg[v] = indeg.get(v, 0) - 1
                    chain_accepts[k] += 1

            elif move_type == 'reverse' and n_edges > 0:
                edge_list = list(G.edges())
                u, v = edge_list[rng.integers(len(edge_list))]
                if (v, u) in forbidden:
                    pass
                elif indeg.get(u, 0) < max_indegree:
                    old_parents_v = list(G.predecessors(v))
                    old_parents_u = list(G.predecessors(u))
                    G.remove_edge(u, v)
                    G.add_edge(v, u)
                    if nx.is_directed_acyclic_graph(G):
                        new_parents_v = list(G.predecessors(v))
                        new_parents_u = list(G.predecessors(u))
                        delta = (scorer.local_score(v, new_parents_v) -
                                 scorer.local_score(v, old_parents_v) +
                                 scorer.local_score(u, new_parents_u) -
                                 scorer.local_score(u, old_parents_u))
                        new_score = current_score + delta
                        chain_proposals[k] += 1
                        log_ratio = delta / T
                        if log_ratio >= 0 or np.log(rng.random()) < log_ratio:
                            chain_scores[k] = new_score
                            indeg[v] = indeg.get(v, 0) - 1
                            indeg[u] = indeg.get(u, 0) + 1
                            chain_accepts[k] += 1
                        else:
                            G.remove_edge(v, u)
                            G.add_edge(u, v)
                    else:
                        G.remove_edge(v, u)
                        G.add_edge(u, v)

        if iteration % swap_interval == 0 and iteration > 0:
            k1 = rng.integers(n_chains - 1)
            k2 = k1 + 1
            T1, T2 = temperatures[k1], temperatures[k2]
            S1, S2 = chain_scores[k1], chain_scores[k2]
            log_swap = (S1 - S2) * (1.0 / T2 - 1.0 / T1)
            n_swaps_proposed += 1
            if log_swap >= 0 or np.log(rng.random()) < log_swap:
                chains[k1], chains[k2] = chains[k2], chains[k1]
                chain_scores[k1], chain_scores[k2] = S2, S1
                chain_indeg[k1], chain_indeg[k2] = chain_indeg[k2], chain_indeg[k1]
                n_swaps_accepted += 1

        cold_scores_trace.append(chain_scores[0])
        if iteration >= burn_in and (iteration - burn_in) % lag == 0:
            collected_dags.append(list(chains[0].edges()))

    cold_accept_rate = chain_accepts[0] / max(chain_proposals[0], 1)
    swap_rate = n_swaps_accepted / max(n_swaps_proposed, 1)

    print(f"    --- MC3 diagnostics ---")
    for k in range(n_chains):
        rate = chain_accepts[k] / max(chain_proposals[k], 1)
        print(f"      T={temperatures[k]:5.1f}: {rate:.3f} ({chain_accepts[k]}/{chain_proposals[k]})")
    print(f"    Swap rate: {swap_rate:.3f} ({n_swaps_accepted}/{n_swaps_proposed})")
    print(f"    Samples collected: {len(collected_dags)}")

    if collected_dags:
        edge_counts = Counter()
        for dag in collected_dags:
            for e in dag:
                edge_counts[e] += 1
        n_s = len(collected_dags)
        distinct = len(set(frozenset(tuple(e) for e in dag) for dag in collected_dags))
        print(f"    Distinct DAGs: {distinct}")
        print(f"    Top edges (posterior probability):")
        for (u, v), cnt in edge_counts.most_common(8):
            print(f"      {u} -> {v}: P={cnt/n_s:.3f}")

        n_edges_trace = [len(dag) for dag in collected_dags]
        if len(n_edges_trace) > 10:
            mean_ne = np.mean(n_edges_trace)
            var_ne = np.var(n_edges_trace)
            if var_ne > 0:
                trace_arr = np.array(n_edges_trace[:200]) - mean_ne
                autocorr = np.correlate(trace_arr, trace_arr, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr /= autocorr[0]
                tau = 1 + 2 * np.sum(np.clip(autocorr[1:min(50, len(autocorr))], 0, None))
                ess_est = len(n_edges_trace) / max(tau, 1)
                print(f"    ESS (edge count): {ess_est:.1f}")
    else:
        print("    Warning: no samples collected. Increase n_iter.")

    return collected_dags, cold_accept_rate, cold_scores_trace


# gamma calibration by cross-validated predictive score

def _quick_mc3_dag(df_disc, block_vars, ess=10.0, gamma=0.0,
                    max_indegree=3, n_iter=8000, seed=42, forbidden=None):
    rng = np.random.default_rng(seed)
    block_vars = [v for v in block_vars if v in df_disc.columns]
    if len(block_vars) < 2:
        return []
    if forbidden is None:
        forbidden = set()

    nodes_list = list(block_vars)
    d = len(nodes_list)
    scorer = BlockScorer(df_disc, block_vars, ess=ess, gamma=gamma)
    temperatures = [1.0, 3.0, 10.0]
    n_chains = len(temperatures)
    swap_interval = 10

    chains, chain_scores, chain_indeg = [], [], []
    for k in range(n_chains):
        G = _random_initial_dag(nodes_list, d, rng, max_indegree, forbidden=forbidden)
        chains.append(G)
        chain_scores.append(scorer.dag_score(list(G.edges())))
        chain_indeg.append(dict(G.in_degree()))

    best_score = chain_scores[0]
    best_edges = list(chains[0].edges())

    for iteration in range(n_iter):
        for k in range(n_chains):
            T = temperatures[k]
            G = chains[k]
            current_score = chain_scores[k]
            indeg = chain_indeg[k]
            n_edges = G.number_of_edges()

            if n_edges == 0:
                move_type = 'add'
            else:
                r = rng.random()
                move_type = 'add' if r < 0.4 else ('remove' if r < 0.7 else 'reverse')

            if move_type == 'add':
                for _ in range(10):
                    u = nodes_list[rng.integers(d)]
                    v = nodes_list[rng.integers(d)]
                    if u == v or G.has_edge(u, v) or indeg.get(v, 0) >= max_indegree:
                        continue
                    if (u, v) in forbidden:
                        continue
                    G.add_edge(u, v)
                    if nx.is_directed_acyclic_graph(G):
                        old_pv = [p for p in G.predecessors(v) if p != u]
                        delta = (scorer.local_score(v, list(G.predecessors(v))) -
                                 scorer.local_score(v, old_pv))
                        if delta / T >= 0 or np.log(rng.random()) < delta / T:
                            chain_scores[k] = current_score + delta
                            indeg[v] = indeg.get(v, 0) + 1
                        else:
                            G.remove_edge(u, v)
                        break
                    else:
                        G.remove_edge(u, v)

            elif move_type == 'remove' and n_edges > 0:
                el = list(G.edges())
                u, v = el[rng.integers(len(el))]
                old_pv = list(G.predecessors(v))
                new_pv = [p for p in old_pv if p != u]
                delta = (scorer.local_score(v, new_pv) - scorer.local_score(v, old_pv))
                if delta / T >= 0 or np.log(rng.random()) < delta / T:
                    G.remove_edge(u, v)
                    chain_scores[k] = current_score + delta
                    indeg[v] = indeg.get(v, 0) - 1

            elif move_type == 'reverse' and n_edges > 0:
                el = list(G.edges())
                u, v = el[rng.integers(len(el))]
                if (v, u) not in forbidden and indeg.get(u, 0) < max_indegree:
                    old_pv, old_pu = list(G.predecessors(v)), list(G.predecessors(u))
                    G.remove_edge(u, v)
                    G.add_edge(v, u)
                    if nx.is_directed_acyclic_graph(G):
                        delta = (scorer.local_score(v, list(G.predecessors(v))) -
                                 scorer.local_score(v, old_pv) +
                                 scorer.local_score(u, list(G.predecessors(u))) -
                                 scorer.local_score(u, old_pu))
                        if delta / T >= 0 or np.log(rng.random()) < delta / T:
                            chain_scores[k] = current_score + delta
                            indeg[v] = indeg.get(v, 0) - 1
                            indeg[u] = indeg.get(u, 0) + 1
                        else:
                            G.remove_edge(v, u)
                            G.add_edge(u, v)
                    else:
                        G.remove_edge(v, u)
                        G.add_edge(u, v)

        if iteration % swap_interval == 0 and iteration > 0:
            k1 = rng.integers(n_chains - 1)
            k2 = k1 + 1
            log_swap = ((chain_scores[k1] - chain_scores[k2]) *
                        (1 / temperatures[k2] - 1 / temperatures[k1]))
            if log_swap >= 0 or np.log(rng.random()) < log_swap:
                chains[k1], chains[k2] = chains[k2], chains[k1]
                chain_scores[k1], chain_scores[k2] = chain_scores[k2], chain_scores[k1]
                chain_indeg[k1], chain_indeg[k2] = chain_indeg[k2], chain_indeg[k1]

        if chain_scores[0] > best_score:
            best_score = chain_scores[0]
            best_edges = list(chains[0].edges())

    return best_edges


def calibrate_gamma(df_disc, block_vars, ess=10.0, max_indegree=3,
                    gammas=[0, 1, 3, 5, 10, 15, 20],
                    n_folds=3, seed=42, forbidden=None):
    print(f"\n  Calibrating gamma for block ({len(block_vars)} vars)...")
    block_vars = [v for v in block_vars if v in df_disc.columns]
    if len(block_vars) < 2:
        return 5.0
    if forbidden is None:
        forbidden = set()

    rng = np.random.default_rng(seed)
    n = len(df_disc)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    best_gamma = gammas[0]
    best_score = -np.inf

    for gamma in gammas:
        fold_scores = []
        for fold in range(n_folds):
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, test_idx)
            df_train = df_disc.iloc[train_idx]
            df_test = df_disc.iloc[test_idx]

            edges = _quick_mc3_dag(df_train, block_vars, ess=ess, gamma=gamma,
                                    max_indegree=max_indegree, n_iter=8000,
                                    seed=seed + fold, forbidden=forbidden)
            test_scorer = BlockScorer(df_test, block_vars, ess=ess, gamma=gamma)
            test_score = test_scorer.dag_score(edges)
            fold_scores.append(test_score)

        if fold_scores:
            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_gamma = gamma

    print(f"    Best gamma = {best_gamma} (CV score = {best_score:.1f})")
    return best_gamma


# Gaussian copula transform and GLASSO inter-block skeleton

def gaussian_copula_transform(df, clip_value=5.0):
    Z = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        x = df[col]
        valid_mask = x.notna()
        n_valid = valid_mask.sum()
        if n_valid < 10:
            Z[col] = 0.0
            continue
        x_valid = x[valid_mask]
        ranks = x_valid.rank(method='average')
        u = (ranks - 0.5) / n_valid
        z_valid = stats.norm.ppf(u)
        z_valid = np.clip(z_valid, -clip_value, clip_value)
        Z.loc[valid_mask, col] = z_valid
    Z = Z.fillna(0.0).astype(float)
    return Z


def glasso_interblock(Z, block_assignments, alpha=0.07, edge_eps=0.02):
    print(f"\nGLASSO inter-block (alpha={alpha})")

    variables = list(Z.columns)
    n_vars = len(variables)

    try:
        model = GraphicalLasso(alpha=alpha, max_iter=500, tol=1e-4,
                               assume_centered=False, mode='cd')
        model.fit(Z)
        precision = model.precision_
    except Exception as e:
        print(f"  GLASSO failed ({e}), using correlation fallback")
        precision = np.linalg.inv(Z.corr().values + 0.1 * np.eye(n_vars))

    inter_edges = set()
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            vi, vj = variables[i], variables[j]
            bi = block_assignments.get(vi, 'other')
            bj = block_assignments.get(vj, 'other')
            if bi == bj:
                continue
            if abs(precision[i, j]) > edge_eps:
                inter_edges.add((vi, vj))

    print(f"  GLASSO inter-block edges: {len(inter_edges)}")

    # MI fallback for pairs missed by GLASSO
    mi_added = 0
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            vi, vj = variables[i], variables[j]
            bi = block_assignments.get(vi, 'other')
            bj = block_assignments.get(vj, 'other')
            if bi == bj or (vi, vj) in inter_edges:
                continue
            x = Z[vi].values
            y = Z[vj].values
            x_binned = np.digitize(x, bins=np.percentile(x, [20, 40, 60, 80]))
            y_binned = np.digitize(y, bins=np.percentile(y, [20, 40, 60, 80]))
            mi = mutual_info_score(x_binned, y_binned)
            if mi > 0.03:
                inter_edges.add((vi, vj))
                mi_added += 1
            if mi_added > 50:
                break
        if mi_added > 50:
            break

    print(f"  MI fallback added: {mi_added}")
    print(f"  Total inter-block candidates: {len(inter_edges)}")
    return inter_edges


# pairwise BDeu bootstrap for inter-block edge selection

def bootstrap_interblock(df_disc, inter_candidates, ess=10.0,
                         max_indegree=2, n_bootstrap=50,
                         threshold=0.5, seed=42):
    # threshold=0.5 (raised from 0.3) to reduce spurious edges
    # and keep downstream CPT indegree manageable
    print(f"\nInter-block pairwise BDeu (B={n_bootstrap}, threshold={threshold})")

    rng = np.random.default_rng(seed)
    valid_candidates = [(u, v) for u, v in inter_candidates
                        if u in df_disc.columns and v in df_disc.columns]
    print(f"  Candidate pairs: {len(valid_candidates)}")

    edge_probs = {}
    retained = set()
    n_evaluated = 0

    for u, v in tqdm(valid_candidates, desc="Pairwise inter-block BDeu"):
        mask = df_disc[[u, v]].notna().all(axis=1)
        df_pair = df_disc.loc[mask, [u, v]]
        n_pair = len(df_pair)
        if n_pair < 30:
            continue

        u_vals_full = df_pair[u].values.astype(int)
        v_vals_full = df_pair[v].values.astype(int)

        n_improve = 0
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_pair, size=n_pair)
            v_boot = v_vals_full[idx]
            u_boot = u_vals_full[idx].reshape(-1, 1)
            score_with = _bdeu_local_fast(v_boot, u_boot, ess=ess)
            score_without = _bdeu_local_fast(v_boot, None, ess=ess)
            if score_with > score_without:
                n_improve += 1

        prob = n_improve / n_bootstrap
        edge_probs[(u, v)] = prob
        n_evaluated += 1
        if prob >= threshold:
            retained.add((u, v))

    print(f"  Evaluated: {n_evaluated}")
    print(f"  Edges above threshold: {len(retained)}")

    if retained:
        sorted_edges = sorted(edge_probs.items(), key=lambda x: -x[1])
        print(f"  Top inter-block edges:")
        for (u, v), p in sorted_edges[:10]:
            print(f"    {u} -> {v}: {p:.2f}")

    return edge_probs, retained


# full DAG composition

def compose_full_dag(block_dags, inter_edges, all_vars, edge_probs_inter, df_disc_ref=None):
    print("\nComposing full DAG")

    intra_edge_probs = {}
    for block_name, sampled_dags in block_dags.items():
        if not sampled_dags:
            continue
        n_samples = len(sampled_dags)
        edge_count = Counter()
        for dag in sampled_dags:
            for e in dag:
                edge_count[e] += 1
        for e, cnt in edge_count.items():
            intra_edge_probs[e] = cnt / n_samples

    intra_retained = {e: p for e, p in intra_edge_probs.items() if p >= 0.3}
    print(f"  Intra-block edges (prob >= 0.3): {len(intra_retained)}")

    all_edges = set()
    all_probs = {}

    for e, p in intra_retained.items():
        all_edges.add(e)
        all_probs[e] = p

    # greedy K2-style inter-block parent selection (Cooper & Herskovits 1992)
    # for each child, add inter-block parents in descending BDeu probability
    # order up to K_INTER_MAX -- controls indegree by evidence rather than
    # by an arbitrary hard cap
    K_INTER_MAX = 3

    child_to_candidates = defaultdict(list)
    for (u, v) in inter_edges:
        prob = edge_probs_inter.get((u, v), edge_probs_inter.get((v, u), 0.5))
        child_to_candidates[v].append((prob, u))
    for v in child_to_candidates:
        child_to_candidates[v].sort(reverse=True)

    n_inter_greedy = 0
    n_inter_rejected = 0

    for child, candidates in child_to_candidates.items():
        accepted = 0
        for prob, u in candidates:
            if accepted >= K_INTER_MAX:
                n_inter_rejected += 1
                continue
            all_edges.add((u, child))
            all_probs[(u, child)] = prob
            n_inter_greedy += 1
            accepted += 1
        n_inter_rejected += max(0, len(candidates) - accepted)

    print(f"  Inter-block edges: {n_inter_greedy} accepted, "
          f"{n_inter_rejected} rejected (top-{K_INTER_MAX} per node by BDeu prob)")
    print(f"  Total edges (pre-cycle check): {len(all_edges)}")

    G = nx.DiGraph(list(all_edges))
    G.add_nodes_from(all_vars)
    n_removed = 0

    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation='original')
        except nx.NetworkXNoCycle:
            break
        min_prob = float('inf')
        min_edge = None
        for u, v, _ in cycle:
            prob = all_probs.get((u, v), 0.5)
            if prob < min_prob:
                min_prob = prob
                min_edge = (u, v)
        if min_edge:
            G.remove_edge(*min_edge)
            all_edges.discard(min_edge)
            n_removed += 1

    print(f"  Cycles removed: {n_removed}")
    print(f"  Final DAG: {len(G.edges())} edges, {len(G.nodes())} nodes")
    print(f"  Acyclic: {nx.is_directed_acyclic_graph(G)}")

    indegrees = sorted([G.in_degree(n) for n in G.nodes()], reverse=True)
    print(f"  Max indegree: {indegrees[0]}")
    print(f"  Mean indegree: {np.mean(indegrees):.2f}")
    over_limit = sum(1 for d in indegrees if d > 6)
    if over_limit > 0:
        print(f"  Warning: {over_limit} nodes with indegree > 6 -- check inter-block threshold")

    variables = sorted(all_vars)
    n_vars = len(variables)
    var_to_idx = {v: i for i, v in enumerate(variables)}
    edge_prob_matrix = np.zeros((n_vars, n_vars))

    for (u, v), prob in all_probs.items():
        if (u, v) in G.edges():
            i, j = var_to_idx.get(u), var_to_idx.get(v)
            if i is not None and j is not None:
                edge_prob_matrix[i, j] = prob
                edge_prob_matrix[j, i] = prob

    edge_prob_df = pd.DataFrame(edge_prob_matrix, index=variables, columns=variables)

    dag = BayesianNetwork(list(G.edges()))
    dag.add_nodes_from(variables)

    return dag, edge_prob_df, all_probs, G


# variable block assignment

def assign_blocks(df_disc, max_block_size=MAX_BLOCK_SIZE):
    print("\nAssigning variable blocks")

    available_cols = set(df_disc.columns)
    block_assignment = {}
    final_blocks = {}

    for block_name, vars_list in THEMATIC_BLOCKS.items():
        present = [v for v in vars_list if v in available_cols]
        if len(present) < 2:
            continue

        if len(present) > max_block_size:
            for i in range(0, len(present), max_block_size):
                sub = present[i:i + max_block_size]
                if len(sub) >= 2:
                    sub_name = f"{block_name}_{i // max_block_size}"
                    final_blocks[sub_name] = sub
                    for v in sub:
                        block_assignment[v] = sub_name
        else:
            final_blocks[block_name] = present
            for v in present:
                block_assignment[v] = block_name

    assigned = set(block_assignment.keys())
    remaining = sorted(available_cols - assigned)

    if remaining:
        df_rem = df_disc[remaining].apply(pd.to_numeric, errors='coerce')
        df_rem_filled = df_rem.fillna(df_rem.median())
        corr = df_rem_filled.corr().abs()
        used = set()
        other_idx = 0

        for var in remaining:
            if var in used or var not in corr.columns:
                continue
            cors = corr[var].drop(labels=list(used) + [var], errors='ignore')
            cors = cors.sort_values(ascending=False)
            group = [var]
            used.add(var)
            for candidate in cors.index:
                if candidate in used:
                    continue
                if len(group) >= max_block_size:
                    break
                group.append(candidate)
                used.add(candidate)
            if len(group) >= 2:
                name = f"auto_{other_idx}"
                final_blocks[name] = group
                for v in group:
                    block_assignment[v] = name
                other_idx += 1
            else:
                block_assignment[var] = 'singleton'

    print(f"  Total blocks: {len(final_blocks)}")
    for name, vars_list in sorted(final_blocks.items()):
        roots_in = [v for v in vars_list if v in ROOTS]
        print(f"    {name:25s}: {len(vars_list)} vars "
              f"(roots: {roots_in if roots_in else 'none'}) "
              f"-- {[v for v in vars_list if v not in ROOTS][:4]}{'...' if len(vars_list) > 4 else ''}")
    print(f"  Unassigned: {len(set(available_cols) - set(block_assignment.keys()))}")

    return final_blocks, block_assignment


# variable selection -- substantive criterion

EXCLUDE_ADMINISTRATIVE = {
    'ID_CIS', 'MUN', 'PROV', 'REGISTRO', 'DURACION',
    'IA_E1_MES', 'IA_E4', 'MES_NACIMIENT', 'PARTIALS', 'MODE',
}

ISSP_BACKGROUND_CORE = {
    'SEX', 'BIRTH', 'EDUCYRS', 'NAT_DEGR', 'MAINSTAT', 'ISCO08',
    'EMPREL', 'TYPORG1', 'NACE_2', 'NAT_INC', 'NAT_RINC',
    'TOPBOT', 'URBRURAL', 'ATTEND', 'NAT_RELIG',
}

ISSP_BACKGROUND_SECONDARY = {
    'FATH_NAT_DEGR', 'MOTH_NAT_DEGR', 'F_BORN', 'M_BORN',
    'SPNAT_DEGR', 'SPEMPREL', 'NAT_ETHN_01',
    'PAIS_NACIMIENTO', 'LUGAR_NAC', 'INTERNETHH',
    'LEFT_RIGHT',
}

ESGE_SUBSTANTIVE = {
    'DIFICULT_ECO', 'C_SATISFESTUDIOS', 'C_SATISFTRABAJO',
    'ESTUDIOS', 'MERIT', 'PREF_TRABINGRESOS',
    'IDEOL_CATEG_01', 'IDEOL_CATEG_02',
}


def filter_substantive_variables(df):
    print("\nVariable selection (substantive criterion)")

    all_cols = set(df.columns)
    include = set()

    for col in all_cols:
        if col.startswith('V') and col[1:].isdigit():
            v_num = int(col[1:])
            if 1 <= v_num <= 58:
                include.add(col)

    for v in ISSP_BACKGROUND_CORE:
        if v in all_cols:
            include.add(v)

    for v in ISSP_BACKGROUND_SECONDARY:
        if v in all_cols:
            pct_na = df[v].isna().sum() / len(df)
            if pct_na < 0.30:
                include.add(v)
            else:
                print(f"  Excluded {v}: {pct_na:.0%} missing (threshold 30%)")

    for v in ESGE_SUBSTANTIVE:
        if v in all_cols:
            include.add(v)

    exclude = set(EXCLUDE_ADMINISTRATIVE)
    for col in all_cols:
        if (col.startswith('MAINSTAT_M_') or
                col.startswith('ENC2A_') or col.startswith('ENC3A_') or
                col.startswith('HMP_') or col == 'NSUP'):
            exclude.add(col)

    include = include - exclude

    v_vars = sorted([v for v in include if v.startswith('V') and v[1:].isdigit()])
    bg_vars = sorted([v for v in include if v in ISSP_BACKGROUND_CORE | ISSP_BACKGROUND_SECONDARY])
    esge_vars = sorted([v for v in include if v in ESGE_SUBSTANTIVE])
    excluded = all_cols - include

    print(f"  Total available: {len(all_cols)}")
    print(f"  Included: {len(include)}")
    print(f"    ISSP module (V1-V58): {len(v_vars)}")
    print(f"    ISSP background: {len(bg_vars)}")
    print(f"    ESGE substantive: {len(esge_vars)}")
    print(f"  Excluded: {len(excluded)}")

    return df[sorted(include)]


# main pipeline

if __name__ == "__main__":

    print("Step 1 -- Bayesian backbone v6")
    print("MC3 parallel-tempered MCMC + Martins penalizing prior")
    print("No hill climbing -- genuine posterior samples from p(G|X)")
    print("v6: mixed demographic-attitudinal blocks")

    df = load_microdata_cis(PATH_NUM)
    df = filter_substantive_variables(df)
    df = filter_collinear(df, threshold=0.95)
    var_types = classify_variables(df)
    df_disc = discretize_for_bdeu(df, var_types, n_bins=5)
    blocks, block_assignment = assign_blocks(df_disc)

    print("\nMC3 structure learning by block")

    block_dags = {}
    block_gammas = {}
    all_forbidden = build_forbidden_edges()

    print("\nCausal constraints (forbidden edges)")
    for bn, fb in sorted(all_forbidden.items()):
        if fb:
            print(f"  {bn}: {len(fb)} forbidden edges")
            for (u, v) in sorted(fb)[:5]:
                print(f"    {u} -> {v} [forbidden]")
            if len(fb) > 5:
                print(f"    ... and {len(fb) - 5} more")

    for block_name, block_vars in sorted(blocks.items()):
        if len(block_vars) < 2:
            continue

        fb = all_forbidden.get(block_name, set())

        gamma = calibrate_gamma(
            df_disc, block_vars, ess=10.0, max_indegree=3,
            gammas=[0, 2, 5, 10, 15, 20],
            n_folds=3, seed=42, forbidden=fb
        )
        block_gammas[block_name] = gamma

        d_block = len(block_vars)
        n_iter = max(20000, d_block * 6000)
        burn_in = n_iter // 4
        lag = max(20, d_block * 5)

        sampled, accept_rate, trace = mcmc_structure_block(
            df_disc, block_vars,
            ess=10.0, gamma=gamma,
            max_indegree=3,
            n_iter=n_iter,
            burn_in=burn_in,
            lag=lag,
            seed=42,
            forbidden=fb
        )

        block_dags[block_name] = sampled

    Z = gaussian_copula_transform(df)
    inter_candidates = glasso_interblock(Z, block_assignment, alpha=0.07, edge_eps=0.02)

    inter_edge_probs, inter_retained = bootstrap_interblock(
        df_disc, inter_candidates,
        ess=10.0, max_indegree=2,
        n_bootstrap=50, threshold=0.8, seed=42
    )

    all_vars = sorted(df_disc.columns)
    dag, edge_probs, all_edge_probs, dag_nx = compose_full_dag(
        block_dags, inter_retained, all_vars, inter_edge_probs, df_disc_ref=df_disc
    )

    print("\nExporting for step 3")

    dag_parents = {}
    for node in dag_nx.nodes():
        dag_parents[node] = list(dag_nx.predecessors(node))

    max_indeg_node = max(dag_parents.items(), key=lambda x: len(x[1]))
    print(f"  Max indegree node: {max_indeg_node[0]} ({len(max_indeg_node[1])} parents)")
    print(f"  Parents: {max_indeg_node[1]}")

    nodes_by_indeg = sorted(dag_parents.items(), key=lambda x: -len(x[1]))
    print(f"  Top 5 nodes by indegree:")
    for node, parents in nodes_by_indeg[:5]:
        q_j = 1
        for p in parents:
            n_states = len(df_disc[p].dropna().unique()) if p in df_disc.columns else 5
            q_j *= n_states
        print(f"    {node}: {len(parents)} parents, q_j={q_j:,}")

    with open("dag_parents_3391.json", "w", encoding='utf-8') as f:
        json.dump(dag_parents, f, indent=2)
    print("  dag_parents_3391.json saved")

    disc_schema = {}
    for col in df_disc.columns:
        states = sorted(df_disc[col].dropna().unique().astype(int).tolist())
        disc_schema[col] = {'n_states': len(states), 'states': states}
    with open("disc_schema_3391.json", "w", encoding='utf-8') as f:
        json.dump(disc_schema, f, indent=2)
    print("  disc_schema_3391.json saved")

    print("\nSaving outputs")

    edge_probs.to_csv("edge_probabilities_3391.csv")
    print("  edge_probabilities_3391.csv saved")

    with open("backbone_edges_3391.txt", "w", encoding='utf-8') as f:
        f.write("# Backbone CIS 3391 -- v6\n")
        f.write("# Method: MC3 parallel-tempered MCMC (Martins/Goudie)\n")
        f.write("# v6: mixed demographic-attitudinal blocks, no post-hoc augmentation\n")
        f.write(f"# Observations: {len(df)}\n")
        f.write(f"# Variables: {len(df_disc.columns)}\n")
        f.write(f"# Blocks: {len(blocks)}\n")
        f.write(f"# Edges (final DAG): {len(dag.edges())}\n")
        f.write(f"# Inter-block threshold: 0.8\n\n")
        f.write("# Block gamma calibration:\n")
        for bn, gv in sorted(block_gammas.items()):
            f.write(f"#   {bn}: gamma={gv}\n")
        f.write("\n")
        for u, v in dag.edges():
            prob = all_edge_probs.get((u, v), 0.0)
            bu = block_assignment.get(u, '?')
            bv = block_assignment.get(v, '?')
            source = "intra" if bu == bv else "inter"
            f.write(f"{u} -> {v}  # p={prob:.3f} source={source}\n")
    print("  backbone_edges_3391.txt saved")

    dag_edges_list = []
    for u, v in dag.edges():
        prob = all_edge_probs.get((u, v), 0.0)
        bu = block_assignment.get(u, '?')
        bv = block_assignment.get(v, '?')
        dag_edges_list.append({
            'source': u, 'target': v,
            'probability': prob,
            'edge_type': 'intra' if bu == bv else 'inter',
            'block_source': bu, 'block_target': bv,
        })
    pd.DataFrame(dag_edges_list).to_csv("dag_edges_3391.csv", index=False)
    print("  dag_edges_3391.csv saved")

    diag = {
        'version': 'v6 MC3 -- mixed blocks',
        'method': 'Parallel-tempered MCMC (no hill climbing)',
        'design': 'Mixed demographic-attitudinal blocks (Martins compliant)',
        'inter_block_threshold': 0.8,
        'blocks': {},
        'gamma_calibration': block_gammas,
        'inter_block_edges': len(inter_retained),
        'total_edges': len(dag.edges()),
        'total_variables': len(df_disc.columns),
    }
    for bn, dags in block_dags.items():
        n_dags = len(dags)
        if n_dags > 0:
            edge_counts = Counter()
            for d_sample in dags:
                edge_counts.update(d_sample)
            avg_edges = np.mean([len(d_sample) for d_sample in dags])
            diag['blocks'][bn] = {
                'n_samples': n_dags,
                'n_vars': len(blocks.get(bn, [])),
                'gamma': block_gammas.get(bn, 0),
                'avg_edges': float(avg_edges),
                'top_edges': [(f"{u}->{v}", cnt / n_dags)
                              for (u, v), cnt in edge_counts.most_common(5)],
            }

    with open("step1_diagnostics.json", "w", encoding='utf-8') as f:
        json.dump(diag, f, indent=2, default=str)
    print("  step1_diagnostics.json saved")

    block_samples_out = {}
    for bn, dags in block_dags.items():
        block_samples_out[bn] = [[(u, v) for u, v in d_sample] for d_sample in dags]
    with open("step1_mcmc_samples.json", "w", encoding='utf-8') as f:
        json.dump(block_samples_out, f, indent=2)
    print("  step1_mcmc_samples.json saved (MC3 DAG samples for step 4)")

    print("\nStep 1 v6 complete")
    print(f"  Variables: {df_disc.shape[1]}")
    print(f"  Observations: {len(df)}")
    print(f"  Blocks: {len(blocks)}")
    n_intra = sum(1 for e in dag.edges()
                  if block_assignment.get(e[0], '?') == block_assignment.get(e[1], '?'))
    n_inter = len(dag.edges()) - n_intra
    print(f"  Intra-block edges: {n_intra}")
    print(f"  Inter-block edges: {n_inter}")
    print(f"  Total DAG edges: {len(dag.edges())}")
    density = len(dag.edges()) / (len(df_disc.columns) * (len(df_disc.columns) - 1) / 2)
    print(f"  Density: {density:.4f}")
    print(f"\n  Next: python step3_cpt.py")
