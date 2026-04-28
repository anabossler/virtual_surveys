# step3_bayesian_fusion.py
#
# Stage 3 of the Bayesian network pipeline: Bayesian fusion + conditional CPT.
#
# Reads:
#   - CIS microdata (3391_num.csv)
#   - DAG structure from Stage 1 (dag_edges_3391.csv or backbone_edges_3391.txt)
#   - Discretisation schema from Stage 1 (disc_schema_3391.json)
#   - Literature Dirichlet priors from Stage 2 (step2_dirichlet_priors.json)
#   - Theoretical edge structure from Stage 2 (step2_theoretical_structure.json)
#
# Writes:
#   - step3_cpt.json          (conditional CPT, primary output consumed by Stage 4)
#   - step3_fused_priors.json (alias of step3_cpt.json for Stage 4 loader)
#   - dag_fused_edges.csv     (final edge list with provenance flag)
#   - step3_metadata.json     (summary statistics)
#
# Key formula:
#   alpha_posterior[cfg] = (alpha_lit / q_v) + n_conditional[cfg]
#
#   where q_v is the number of parent configurations and n_conditional[cfg]
#   is the empirical count of node v given parents in configuration cfg.
#   Dividing alpha_lit by q_v spreads the literature prior evenly across
#   all parent configurations, preserving conditional dependencies.

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import product as iterproduct

from pgmpy.models import BayesianNetwork
import networkx as nx


# File paths — all relative to the directory containing this script

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CIS survey microdata (semicolon-delimited, latin-1 encoded)
PATH_NUM = os.path.join(BASE_DIR, "metadata", "3391_num.csv")

# Stage 1 DAG outputs (primary source: CSV; fallback: plain-text backbone)
PATH_DAG_CSV          = os.path.join(BASE_DIR, "dag_edges_3391.csv")
PATH_BACKBONE_TXT     = os.path.join(BASE_DIR, "backbone_edges_3391.txt")
PATH_EDGE_PROB_MATRIX = os.path.join(BASE_DIR, "edge_probabilities_3391.csv")
PATH_MCMC_SAMPLES     = os.path.join(BASE_DIR, "step1_mcmc_samples.json")
PATH_DISC_SCHEMA      = os.path.join(BASE_DIR, "disc_schema_3391.json")

# Stage 2 literature outputs
PATH_LIT_PRIORS      = os.path.join(BASE_DIR, "step2_dirichlet_priors.json")
PATH_THEO_STRUCTURE  = os.path.join(BASE_DIR, "step2_theoretical_structure.json")


# CIS missing-value codes
# These numeric codes indicate "not applicable", "don't know", or refusal.
# They are replaced with NaN before any analysis.

CIS_MISSING_CODES = {96, 97, 98, 99, 996, 997, 998, 999,
                     9996, 9997, 9998, 9999}


# Mapping: harmonised variable name -> list of CIS column names
#
# Each harmonised concept maps to one or more raw CIS columns.
# When multiple columns exist, the first is used as the representative.

CIS_TO_HARMONIZED = {
    'demographics_gender':         ['SEX'],
    'demographics_age':            ['BIRTH'],
    'demographics_education':      ['NAT_DEGR', 'ESTUDIOS'],
    'demographics_income':         ['NAT_INC', 'NAT_RINC'],
    'demographics_urban_rural':    ['URBRURAL'],
    'attitudes_env_concern':       ['V15', 'V16', 'V17', 'V18', 'V19'],
    'attitudes_wtp_taxes':         ['V26'],
    'attitudes_wtp_prices':        ['V27'],
    'behavior_recycling':          ['V52', 'V53'],
    'behavior_reduce_consumption': ['V50', 'V51'],
    'perception_danger_pollution': ['V37', 'V38', 'V39', 'V40'],
    'trust_institutions':          ['V10', 'V11', 'V12', 'V13', 'V14'],
}

# Reverse lookup: CIS column -> harmonised name (built from CIS_TO_HARMONIZED)
CIS_VAR_TO_HARM = {}
for harm_name, cis_vars in CIS_TO_HARMONIZED.items():
    for cv in cis_vars:
        CIS_VAR_TO_HARM[cv] = harm_name

# Primary representative CIS column for each harmonised concept
HARM_TO_CIS = {}
for harm_name, cis_vars in CIS_TO_HARMONIZED.items():
    HARM_TO_CIS[harm_name] = cis_vars[0]


# Data loading

def load_step1_dag():
    """
    Load the DAG structure produced by Stage 1.

    Tries PATH_DAG_CSV first (preferred: has edge probabilities as a column).
    Falls back to PATH_BACKBONE_TXT if the CSV is absent or empty.

    Returns
    -------
    dag_edges      : list of (source, target) tuples
    dag_edge_probs : dict mapping (source, target) -> float probability
    cis_dag_vars   : sorted list of all node names that appear in the DAG
    mcmc_samples   : dict loaded from step1_mcmc_samples.json, or None
    """
    dag_edges, dag_edge_probs = [], {}

    # Primary source: structured CSV with columns [source, target, probability]
    if os.path.exists(PATH_DAG_CSV):
        df = pd.read_csv(PATH_DAG_CSV)
        if {'source', 'target'}.issubset(set(df.columns)):
            for _, row in df.iterrows():
                u, v = str(row['source']).strip(), str(row['target']).strip()
                # The probability cell may contain trailing text; take first token
                prob_raw = str(row.get('probability', 1.0)).strip().split()[0]
                try:
                    prob = float(prob_raw)
                except ValueError:
                    prob = 1.0
                dag_edges.append((u, v))
                dag_edge_probs[(u, v)] = prob

    # Fallback: plain-text backbone file, one edge per line: "A -> B  # p=0.8"
    if not dag_edges and os.path.exists(PATH_BACKBONE_TXT):
        with open(PATH_BACKBONE_TXT) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '->' not in line:
                    continue
                # Strip inline comments, then split on arrow
                parts = line.split('#')[0].strip().split('->')
                if len(parts) == 2:
                    u, v = parts[0].strip(), parts[1].strip()
                    prob = 1.0
                    # Parse optional "p=<value>" annotation in inline comment
                    if '#' in line and 'p=' in line:
                        try:
                            prob = float(line.split('p=')[1].split()[0])
                        except Exception:
                            pass
                    dag_edges.append((u, v))
                    dag_edge_probs[(u, v)] = prob

    if not dag_edges:
        raise FileNotFoundError("No Stage 1 DAG found. "
                                "Ensure dag_edges_3391.csv or backbone_edges_3391.txt exists.")

    # Load MC3 samples if available (used for diagnostics, not for CPT computation)
    mcmc_samples = None
    if os.path.exists(PATH_MCMC_SAMPLES):
        with open(PATH_MCMC_SAMPLES) as f:
            mcmc_samples = json.load(f)

    # Collect the unique set of node names from all edges
    cis_dag_vars = sorted(set(u for u, v in dag_edges) | set(v for u, v in dag_edges))
    return dag_edges, dag_edge_probs, cis_dag_vars, mcmc_samples


def load_disc_schema():
    """
    Load the discretisation schema produced by Stage 1.

    The schema maps each CIS column name to a dict containing at minimum:
        {'states': [list of valid integer values in original CIS coding]}

    Returns an empty dict if the file does not exist (graceful degradation).
    """
    if os.path.exists(PATH_DISC_SCHEMA):
        with open(PATH_DISC_SCHEMA) as f:
            return json.load(f)
    return {}


def load_cis_microdata():
    """
    Load and clean the raw CIS survey microdata.

    Steps applied:
      1. Read the semicolon-delimited CSV with latin-1 encoding.
      2. Coerce all columns to numeric, setting non-parseable cells to NaN.
      3. Replace all CIS missing/refusal codes with NaN.

    Returns a DataFrame with numeric values and NaN where data are absent.
    """
    df = pd.read_csv(PATH_NUM, sep=";", encoding="latin1",
                     low_memory=False, quotechar='"',
                     na_values=['N.P.', 'N.C.', 'N.S.', 'N.D.'])

    # Force numeric dtype; string residuals become NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace CIS-specific sentinel codes with NaN
    for code in CIS_MISSING_CODES:
        df = df.replace(code, np.nan)

    return df


def load_step2_priors():
    """
    Load the Dirichlet priors and theoretical edge structure from Stage 2.

    step2_dirichlet_priors.json must exist and contain a 'priors' key whose
    value maps harmonised variable names to dicts with keys 'alpha' and 'K'.

    step2_theoretical_structure.json is optional. When present it provides
    additional directed edges derived from the literature review.

    Returns
    -------
    priors         : dict  {harmonised_name: {'alpha': [...], 'K': int}}
    theo_structure : dict or None
    """
    with open(PATH_LIT_PRIORS, encoding='utf-8') as f:
        data = json.load(f)
    priors = data.get('priors', {})

    theo_structure = None
    if os.path.exists(PATH_THEO_STRUCTURE):
        with open(PATH_THEO_STRUCTURE, encoding='utf-8') as f:
            theo_structure = json.load(f)

    return priors, theo_structure


# Discretisation helpers

def _clean_states(states):
    """
    Remove CIS missing-value codes from a list of observed category values.

    Large missing codes (96, 97, 98, 99, 996 ...) are always removed.
    Single-digit missing candidates (7, 8, 9, 77, 88) are removed only when
    there is a gap of 2 or more between consecutive valid values, which
    indicates that the value is a sentinel rather than a substantive category.

    Returns the filtered, sorted list of valid state values.
    """
    CIS_MISSING_LARGE = {96, 97, 98, 99, 996, 997, 998, 999,
                         9996, 9997, 9998, 9999}
    SINGLE_MISSING_CANDIDATES = {7, 8, 9, 77, 88}

    s = sorted(x for x in states if x not in CIS_MISSING_LARGE)
    if not s:
        return list(states)

    result = [s[0]]
    for i in range(1, len(s)):
        # A gap >= 2 before a known sentinel suggests the value is missing-coded
        if s[i] - s[i - 1] >= 2 and s[i] in SINGLE_MISSING_CANDIDATES:
            break
        result.append(s[i])
    return result


def discretise_df(df, disc_schema):
    """
    Convert the raw CIS DataFrame to 0-based integer state indices.

    Uses the exact discretisation schema from Stage 1 so that state indices
    in the CPT match those used during structure learning.

    For each column present in disc_schema:
      - Valid category values are cleaned with _clean_states.
      - Each valid value is mapped to its 0-based position in the sorted list.
      - Rows that map to NaN (unrecognised value) are imputed with the column mode.
      - Values are clipped to [0, K-1] to guard against edge cases.

    Returns
    -------
    result     : DataFrame of int state indices, same row index as df
    state_maps : dict {column_name: [real_value_0, real_value_1, ...]}
                 Provides the inverse mapping from index back to original value.
    """
    dd = {}
    state_maps = {}

    for col in df.columns:
        if col not in disc_schema:
            continue

        schema = disc_schema[col]
        states = _clean_states(schema.get('states', []))
        if not states:
            continue

        # Map each original value to its 0-based index
        val_to_idx = {v: i for i, v in enumerate(states)}
        mapped = df[col].map(val_to_idx)

        # Skip columns where no valid values were found in the data
        if mapped.notna().sum() == 0:
            continue

        dd[col] = mapped
        state_maps[col] = states  # preserves original real values for CPT output

    result = pd.DataFrame(dd, index=df.index)
    for c in result.columns:
        result[c] = pd.to_numeric(result[c], errors='coerce')

    # Impute remaining NaN with the column mode (most frequent observed state)
    mode_row = result.mode().iloc[0]
    result = result.fillna(mode_row).astype(int)

    # Clip to valid range in case any residual out-of-range values remain
    for c in result.columns:
        K = len(state_maps.get(c, [1]))
        result[c] = result[c].clip(0, max(K - 1, 0))

    return result, state_maps


def get_states_from_dd(dd, var):
    """
    Return the sorted list of observed 0-based state indices for a variable.

    Used as a fallback when a variable is not in the Stage 1 disc_schema.
    """
    if var not in dd.columns:
        return []
    return sorted(dd[var].dropna().unique().tolist())


# Conditional CPT construction

def build_conditional_cpt(dag_edges, df_disc, disc_schema, lit_priors, state_maps):
    """
    Build a conditional probability table (CPT) for every node in the DAG.

    Algorithm
    ---------
    For each node v with parent set Pa(v):

      1. Compute q_v = product of the cardinalities of all parents.

      2. Determine the Dirichlet prior base vector alpha_base of length K_v:
           - If a literature prior exists for v: blend the literature alpha with
             a uniform Dirichlet(1) using a data-driven mixing weight LAMBDA_LIT
             derived from the effective sample size (ESS) of the literature prior.
           - Otherwise: use a uniform Dirichlet(1) base (alpha_base = ones(K_v)).

      3. Spread the prior evenly across parent configurations:
             alpha_prior_cond = alpha_base / q_v
         This keeps the total prior mass equal to alpha_base.sum() regardless
         of the number of parent configurations, preventing the prior from
         dominating in high-cardinality parent combinations.

      4. For each configuration cfg of Pa(v):
             n[cfg]            = empirical counts of v conditioned on Pa(v) = cfg
             alpha_post[cfg]   = alpha_prior_cond + n[cfg]
             theta[cfg]        = alpha_post[cfg] / alpha_post[cfg].sum()

    Nodes with no parents use the marginal counts and a single '()' table entry.

    Parameters
    ----------
    dag_edges   : list of (source, target) edge tuples defining the DAG
    df_disc     : DataFrame of 0-based integer state indices (from discretise_df)
    disc_schema : raw discretisation schema from Stage 1 (retained for signature
                  consistency with the pipeline)
    lit_priors  : dict {harmonised_name: {'alpha': list, 'K': int}}
    state_maps  : dict {col: [real_value_0, ...]} from discretise_df

    Returns
    -------
    cpts : dict {node_name: {
               'parents':        list of parent node names,
               'states':         list of original CIS values for this node,
               'K_v':            int, number of states,
               'has_lit_prior':  bool,
               'alpha_per_cell': float, mean prior pseudo-count per cell,
               'q_j':            int, number of parent configurations,
               'n_configs':      int, number of entries in table,
               'table':          dict {cfg_key: {
                                      'alpha_posterior': list,
                                      'n':              list,
                                      'theta':          list}},
               'conditional_cpt': same object as 'table' (alias for Stage 4),
           }}
    """
    # Build a directed graph to access predecessor sets
    G = nx.DiGraph()
    G.add_edges_from(dag_edges)
    dag_vars = set(G.nodes())

    cpts = {}
    n_with_lit = 0

    for node in sorted(dag_vars):
        # Skip nodes that were not discretised (absent from the microdata)
        if node not in df_disc.columns:
            continue

        # Original CIS values (e.g. [1, 2] for SEX); df_disc holds 0-based indices
        node_states_real = state_maps.get(node, get_states_from_dd(df_disc, node))
        node_states_idx  = list(range(len(node_states_real)))  # [0, 1, 2, ...]
        K_v = len(node_states_real)
        if K_v == 0:
            continue

        # Only include parents that are present in the discretised data
        parents_all = list(G.predecessors(node))
        parents = [p for p in parents_all if p in df_disc.columns]

        # Attempt to find a literature prior via the harmonised variable mapping
        harm_name  = CIS_VAR_TO_HARM.get(node)
        has_lit    = False
        alpha_base = np.ones(K_v)  # default: uniform Dirichlet(1)

        if harm_name and harm_name in lit_priors:
            lit           = lit_priors[harm_name]
            alpha_lit_raw = np.array(lit['alpha'], dtype=float)
            K_lit         = lit['K']

            # Align literature prior to the number of states observed in the data
            if K_lit != K_v:
                prop = alpha_lit_raw / alpha_lit_raw.sum()
                if K_lit > K_v:
                    # Collapse: sum literature probability mass into fewer bins
                    step     = K_lit / K_v
                    new_prop = np.zeros(K_v)
                    for i in range(K_v):
                        s, e = int(i * step), int((i + 1) * step)
                        new_prop[i] = prop[s:e].sum()
                else:
                    # Expand: assign a small floor probability to extra states
                    new_prop           = np.zeros(K_v)
                    new_prop[:K_lit]   = prop
                    floor              = 0.005
                    new_prop[:K_lit]  *= (1 - floor * (K_v - K_lit))
                    new_prop[K_lit:]   = floor
                alpha_base = new_prop * alpha_lit_raw.sum()
            else:
                alpha_base = alpha_lit_raw

            # Data-driven mixing weight based on literature effective sample size.
            # ESS = sum of alpha values. LAMBDA_LIT approaches 1 as ESS grows,
            # giving more weight to the literature prior; approaches 0 for small ESS.
            # The constant 100 sets the scale: ESS=100 gives LAMBDA_LIT = 0.5.
            ess_lit    = float(alpha_base.sum())
            LAMBDA_LIT = ess_lit / (ess_lit + 100.0)

            # Mix with a uniform prior for robustness against misspecified literature
            alpha_base = LAMBDA_LIT * alpha_base + (1 - LAMBDA_LIT) * np.ones(K_v)

            has_lit    = True
            n_with_lit += 1

        # ------------------------------------------------------------------
        # Root node (no parents): marginal CPT
        # ------------------------------------------------------------------
        if len(parents) == 0:
            counts = np.zeros(K_v)
            for i in node_states_idx:
                counts[i] = (df_disc[node] == i).sum()

            alpha_post = alpha_base + counts
            s          = alpha_post.sum()
            theta      = (alpha_post / s).tolist()

            cpts[node] = {
                'parents':        [],
                'states':         node_states_real,
                'K_v':            K_v,
                'has_lit_prior':  has_lit,
                'alpha_per_cell': float(alpha_base.sum() / K_v),
                'q_j':            1,
                'n_configs':      1,
                'table': {
                    '()': {
                        'alpha_posterior': alpha_post.tolist(),
                        'n':     counts.tolist(),
                        'theta': theta,
                    }
                },
            }
            continue

        # ------------------------------------------------------------------
        # Non-root node: conditional CPT over all parent configurations
        # ------------------------------------------------------------------

        # Build the list of valid 0-based state ranges for each parent
        parent_states_list = [
            list(range(len(state_maps.get(p, get_states_from_dd(df_disc, p)))))
            for p in parents
        ]

        # q_v = total number of parent configurations (product of parent cardinalities)
        q_v = 1
        for ps in parent_states_list:
            q_v *= max(len(ps), 1)

        # Spread literature prior evenly across parent configurations.
        # Without this division the prior would be counted q_v times in
        # aggregate, masking empirical conditional dependencies.
        alpha_prior_cond = alpha_base / max(q_v, 1)
        alpha_per_cell   = float(alpha_prior_cond.mean())

        table = {}
        for cfg in iterproduct(*parent_states_list):
            cfg_key = str(cfg)  # e.g. "(0, 1)" used as JSON-safe dict key

            # Select rows where every parent equals its value in this configuration
            mask = pd.Series([True] * len(df_disc), index=df_disc.index)
            for p, val in zip(parents, cfg):
                mask &= (df_disc[p] == val)

            sub = df_disc.loc[mask, node]

            # Count occurrences of each state of the child node in this subset
            counts = np.zeros(K_v)
            for i in node_states_idx:
                counts[i] = (sub == i).sum()

            # Bayesian update: posterior = prior + empirical counts
            alpha_post = alpha_prior_cond + counts
            total      = alpha_post.sum()
            theta      = (alpha_post / total).tolist() if total > 0 \
                         else (np.ones(K_v) / K_v).tolist()

            table[cfg_key] = {
                'alpha_posterior': alpha_post.tolist(),
                'n':     counts.tolist(),
                'theta': theta,
            }

        cpts[node] = {
            'parents':         parents,
            'states':          node_states_real,
            'K_v':             K_v,
            'has_lit_prior':   has_lit,
            'alpha_per_cell':  alpha_per_cell,
            'q_j':             q_v,
            'n_configs':       len(table),
            'table':           table,
            'conditional_cpt': table,  # alias key expected by Stage 4 loader
        }

    return cpts


# DAG augmentation

def augment_dag(dag_edges, dag_edge_probs, lit_priors, theo_structure, cis_dag_vars):
    """
    Extend the Stage 1 DAG with additional edges from the theoretical structure.

    The theoretical structure (step2_theoretical_structure.json) encodes edges
    from the literature review that were not discovered by MC3 structure search.
    These edges use harmonised variable names that are translated back to their
    representative CIS column names before being added.

    An edge is added only if:
      - It does not already exist in the empirical DAG.
      - At least one endpoint is absent from the current DAG (novel variable).

    After augmentation, acyclicity is enforced by iteratively removing the
    edge with the lowest probability from any detected cycle.

    Returns
    -------
    augmented_edges  : list of (source, target) including new theoretical edges
    augmented_probs  : dict {(source, target): float probability}
    all_vars         : set of all node names in the final DAG
    theo_edges_added : list of dicts describing each added theoretical edge
    """
    augmented_edges  = list(dag_edges)
    augmented_probs  = dict(dag_edge_probs)
    all_vars         = set(cis_dag_vars)
    theo_edges_added = []

    # Build a lookup from harmonised name -> CIS columns present in the DAG
    harm_to_cis_in_dag = {}
    for harm_name, cis_vars in CIS_TO_HARMONIZED.items():
        for cv in cis_vars:
            if cv in cis_dag_vars:
                harm_to_cis_in_dag.setdefault(harm_name, []).append(cv)

    if theo_structure and 'edges' in theo_structure:
        for edge_info in theo_structure['edges']:
            sh       = edge_info[0]                              # harmonised source name
            th       = edge_info[1]                              # harmonised target name
            prob     = edge_info[2] if len(edge_info) > 2 else 0.5
            citation = edge_info[3] if len(edge_info) > 3 else ''

            # Translate harmonised names to representative CIS column names
            src = harm_to_cis_in_dag.get(sh, [sh])[0]
            tgt = harm_to_cis_in_dag.get(th, [th])[0]

            # Skip if both endpoints already exist in the empirical DAG;
            # MC3 would have found this edge if it were strongly supported
            if tgt in cis_dag_vars and src in cis_dag_vars:
                continue

            if (src, tgt) not in augmented_probs:
                augmented_edges.append((src, tgt))
                augmented_probs[(src, tgt)] = prob
                all_vars.update([src, tgt])
                theo_edges_added.append({
                    'source':      src,
                    'target':      tgt,
                    'probability': prob,
                    'citation':    citation,
                })

    # Enforce acyclicity: remove the lowest-probability edge from each cycle
    G = nx.DiGraph(augmented_edges)
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation='original')
            min_edge = min(
                ((u, v) for u, v, _ in cycle),
                key=lambda e: augmented_probs.get(e, 0.5)
            )
            G.remove_edge(*min_edge)
            augmented_edges.remove(min_edge)
            del augmented_probs[min_edge]
        except nx.NetworkXNoCycle:
            break

    return augmented_edges, augmented_probs, all_vars, theo_edges_added


# Output serialisation

def save_outputs(cpts, augmented_edges, augmented_probs, theo_edges_added, cis_dag_vars):
    """
    Write all Stage 3 outputs to disk.

    Files produced
    --------------
    step3_cpt.json
        Full CPT structure consumed by Stage 4. Contains the 'cpts' and
        'variables' keys (the latter is an alias of the former for backward
        compatibility with the Stage 4 load_fused_priors function).

    step3_fused_priors.json
        Identical copy of step3_cpt.json written under the name that the
        Stage 4 loader expects when reading priors.

    dag_fused_edges.csv
        Final edge list with columns [source, target, probability, edge_type].
        edge_type is 'empirical_mc3' for edges from Stage 1 and 'theoretical'
        for edges added from the literature structure.

    step3_metadata.json
        Summary counts for logging and reproducibility checks.
    """
    # Build the top-level output structure expected by Stage 4
    out = {
        'method':        'Conditional Dirichlet CPT: alpha_post[cfg] = (alpha_lit / q_v) + n_cond[cfg]',
        'paper_section': 'Stage 3 — conditional, not marginal',
        'n_nodes':       len(cpts),
        'cpts':          cpts,
        'variables':     cpts,  # alias for Stage 4 compatibility
    }

    with open('step3_cpt.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Duplicate under the name Stage 4 uses when calling load_fused_priors
    with open('step3_fused_priors.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Edge list with provenance flag for each edge
    theo_set = {(e['source'], e['target']) for e in theo_edges_added}
    rows = []
    for u, v in augmented_edges:
        p = augmented_probs.get((u, v), 0.5)
        rows.append({
            'source':      u,
            'target':      v,
            'probability': p,
            'edge_type':   'theoretical' if (u, v) in theo_set else 'empirical_mc3',
        })
    pd.DataFrame(rows).to_csv('dag_fused_edges.csv', index=False)

    # Lightweight metadata for downstream logging and reproducibility
    meta = {
        'method':            'Conditional Dirichlet update per parent configuration',
        'formula':           'alpha_post[cfg] = (alpha_lit / q_v) + n_cond[cfg]',
        'total_edges':       len(augmented_edges),
        'theoretical_edges': len(theo_edges_added),
        'n_nodes_in_cpt':    len(cpts),
        'n_with_lit_prior':  sum(1 for v in cpts.values() if v.get('has_lit_prior')),
    }
    with open('step3_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


# Entry point

def main():
    """
    Execute the full Stage 3 pipeline in order:

      1. Load DAG structure from Stage 1.
      2. Load discretisation schema from Stage 1.
      3. Load and clean CIS microdata.
      4. Load literature priors and theoretical structure from Stage 2.
      5. Discretise the microdata using the Stage 1 schema.
      6. Augment the DAG with theoretical edges from the literature.
      7. Build the conditional CPT with Bayesian fusion.
      8. Write all outputs to disk.
    """
    dag_edges, dag_edge_probs, cis_dag_vars, mcmc_samples = load_step1_dag()
    disc_schema                                            = load_disc_schema()
    df_cis                                                 = load_cis_microdata()
    lit_priors, theo_structure                             = load_step2_priors()
    df_disc, state_maps                                    = discretise_df(df_cis, disc_schema)

    augmented_edges, augmented_probs, all_vars, theo_edges_added = augment_dag(
        dag_edges, dag_edge_probs, lit_priors, theo_structure, cis_dag_vars
    )

    cpts = build_conditional_cpt(
        augmented_edges, df_disc, disc_schema, lit_priors, state_maps
    )

    save_outputs(cpts, augmented_edges, augmented_probs, theo_edges_added, cis_dag_vars)


if __name__ == "__main__":
    main()
