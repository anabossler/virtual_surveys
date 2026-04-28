# ============================================================
# STEP 2 - LITERATURE PRIORS WITH AUTOMATIC DISCOUNT
#
# Pools Dirichlet counts from international surveys (category_mappings.json)
# and computes informative priors for each CIS target variable.
#
# Discount method:
#   Heterogeneity across surveys is quantified via the I2 statistic
#   (Higgins & Thompson 2002).
#   lambda = 0 so discount = 1.0 for all variables; I2 is computed
#   and stored for reporting purposes only (Appendix B of the paper).
#
#   The formula is retained in parametric form so lambda can be
#   adjusted in future analyses without restructuring the code:
#       discount = exp(-lambda * I2)   where I2 in [0, 1]
#
# Outputs:
#   step2_dirichlet_priors.json         - alpha vectors per variable
#   step2_theoretical_structure.json    - DAG tier structure and edges
#   step2_summary_for_paper.csv         - one-row-per-variable summary
#   step2_i2_heterogeneity_analysis.csv - I2 diagnostics per variable
#   step2_input_mappings.json           - flattened input (audit trail)
#
# Dependencies: numpy, pandas
# Input: category_mappings.json (output of harmonize_categories.py)
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict


# ============================================================
# I2 STATISTIC AND DISCOUNT
# ============================================================

def calculate_i2_and_discount(entries, target):
    """
    Compute inter-survey heterogeneity (I2) and a discount factor.

    For each response category k across n_surveys surveys:
        Q_k = sum_i [ w_i * (p_ik - p_pooled_k)^2 ]   where w_i = n_i
        df  = n_surveys - 1
        I2_k = max(0, (Q_k - df) / Q_k)

    I2_variable = mean(I2_1, ..., I2_K)

    Discount function (exponential decay, Cochrane-calibrated):
        discount = exp(-lambda * I2_variable)
        lambda = 0.0 (I2 computed for reporting only; ESS = full pooled N per Section 2.3)

    Parameters
    ----------
    entries : list of dict
        Mappings for one CIS target variable (output of load_category_mappings).
    target : str
        CIS variable name (used only for error messages).

    Returns
    -------
    I2_pct : float
        Heterogeneity percentage in [0, 100].
    discount : float
        Discount factor in [0.1, 1.0].
    metadata : dict
        Diagnostic details for logging and CSV output.
    """
    base_metadata = {
        'I2_pct': 0.0,
        'I2_by_category': [],
        'n_surveys': len(entries),
        'total_n': sum(e.get('total_n', 0) for e in entries),
        'lambda': 0.0,
        'formula': 'discount = exp(-0.0 * I2)  [I2 for reporting; ESS not discounted]',
    }

    if len(entries) < 2:
        base_metadata['reason'] = 'single_survey'
        return 0.0, 1.0, base_metadata

    K = len(entries[0]['dirichlet_counts'])
    n_surveys = len(entries)

    proportions = []
    weights = []

    for e in entries:
        dc = np.array(e['dirichlet_counts'], dtype=float)
        n = e.get('total_n', 0) if e.get('total_n', 0) > 0 else dc.sum()
        if dc.sum() <= 0:
            continue
        proportions.append(dc / dc.sum())
        weights.append(n)

    if len(proportions) < 2:
        base_metadata['reason'] = 'insufficient_valid_surveys'
        base_metadata['n_surveys'] = len(proportions)
        base_metadata['total_n'] = int(np.sum(weights)) if weights else 0
        return 0.0, 1.0, base_metadata

    proportions = np.array(proportions)
    weights = np.array(weights)

    I2_by_category = []
    for k in range(K):
        p_k = proportions[:, k]
        p_pooled = np.average(p_k, weights=weights)
        Q = np.sum(weights * (p_k - p_pooled) ** 2)
        df = n_surveys - 1
        I2_k = max(0.0, (Q - df) / Q) if Q > 0 and df > 0 else 0.0
        I2_by_category.append(I2_k)

    I2_mean = float(np.mean(I2_by_category))
    I2_pct = I2_mean * 100.0

    # I2 is computed for transparency and reported in Appendix B.
    # Per Section 2.3: the full pooled sample contributes to the prior;
    # I2 does not discount ESS (Martins et al. 2024 convention).
    lambda_param = 0.0
    discount = 1.0

    metadata = {
        'I2_pct': I2_pct,
        'I2_by_category': [float(x) for x in I2_by_category],
        'n_surveys': n_surveys,
        'total_n': int(np.sum(weights)),
        'lambda': lambda_param,
        'formula': f'discount = exp(-{lambda_param} * I2)',
    }

    return I2_pct, discount, metadata


# ============================================================
# PART 1: LOAD
# ============================================================

def load_category_mappings(filepath="category_mappings.json"):
    """
    Load and flatten category_mappings.json (output of harmonize_categories.py).

    Returns
    -------
    flat : list of dict
        One entry per valid variable-study mapping.
    by_target : dict
        flat entries indexed by cis_target name.
    """
    print("=" * 60)
    print("PART 1: LOADING CATEGORY MAPPINGS")
    print("=" * 60)

    with open(filepath, 'r', encoding='utf-8') as f:
        all_mappings = json.load(f)

    print(f"\n  Surveys in file: {len(all_mappings)}")

    flat = []
    for survey_mapping in all_mappings:
        study_id = survey_mapping.get('study_id', '?')
        for m in survey_mapping.get('mappings', []):
            dc = m.get('dirichlet_counts', [])
            if not dc or not any(x > 0 for x in dc):
                continue
            flat.append({
                'study_id': study_id,
                'cis_target': m.get('cis_target'),
                'cis_variable': m.get('cis_variable'),
                'cis_K': m.get('cis_K'),
                'source_variable': m.get('source_variable'),
                'source_scale': m.get('source_scale'),
                'transform_type': m.get('transform_type'),
                'dirichlet_counts': dc,
                'total_n': m.get('total_n', sum(dc)),
                'notes': m.get('notes', ''),
            })

    print(f"  Valid mappings: {len(flat)}")

    by_target = defaultdict(list)
    for m in flat:
        by_target[m['cis_target']].append(m)

    print(f"  CIS targets: {len(by_target)}")
    print(f"\n  {'CIS Target':<45s} {'Surveys':<8s} {'Total N'}")
    print(f"  {'-' * 70}")
    for target in sorted(by_target.keys()):
        entries = by_target[target]
        n_surveys = len(set(e['study_id'] for e in entries))
        total_n = sum(e['total_n'] for e in entries)
        print(f"  {target:<45s} {n_surveys:<8d} {total_n}")

    return flat, dict(by_target)


# ============================================================
# PART 2: VALIDATE
# ============================================================

def validate_mappings(by_target):
    """
    Enforce consistent K (number of response categories) per target variable.

    K-mismatch handling:
      - Too few categories: pad with alpha=0.5 (weak uninformative pseudocount).
      - Too many categories: proportionally collapse excess mass into the last
        retained category.

    Parameters
    ----------
    by_target : dict
        Output of load_category_mappings.

    Returns
    -------
    cleaned : dict
        Same structure as by_target, with dirichlet_counts adjusted to target_K.
    """
    print(f"\n{'=' * 60}")
    print("PART 2: VALIDATING K CONSISTENCY")
    print(f"{'=' * 60}")

    issues = []
    cleaned = {}

    for target, entries in by_target.items():
        k_counts = defaultdict(int)
        for e in entries:
            k_counts[len(e['dirichlet_counts'])] += 1
        target_K = max(k_counts, key=k_counts.get)

        valid = []
        for e in entries:
            dc = e['dirichlet_counts']
            original_K = len(dc)

            if len(dc) == target_K:
                valid.append(e)

            elif len(dc) < target_K:
                while len(dc) < target_K:
                    dc.append(0.5)
                e['dirichlet_counts'] = dc
                e['k_mismatch_type'] = 'padded'
                e['k_mismatch_original'] = original_K
                valid.append(e)
                issues.append(
                    f"  {target}: {e['study_id']} padded K={original_K}->{target_K} with alpha=0.5"
                )

            else:
                excess = dc[target_K - 1:]
                excess_mass = sum(excess)
                base_dc = dc[:target_K - 1]
                base_mass = sum(base_dc)

                if base_mass > 0:
                    collapsed = [x + (x / base_mass) * excess_mass for x in base_dc]
                else:
                    collapsed = [x + excess_mass / (target_K - 1) for x in base_dc]

                # last category absorbs rounding residual
                collapsed.append(sum(excess) - (sum(collapsed) - sum(base_dc)))

                e['dirichlet_counts'] = collapsed
                e['k_mismatch_type'] = 'collapsed_proportional'
                e['k_mismatch_original'] = original_K
                valid.append(e)
                issues.append(
                    f"  {target}: {e['study_id']} collapsed K={original_K}->{target_K} (proportional)"
                )

        cleaned[target] = valid

    if issues:
        print(f"  Fixes applied: {len(issues)}")
        for msg in issues[:10]:
            print(msg)
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("  All K values consistent.")

    return cleaned


# ============================================================
# PART 3: POOL WITH AUTOMATIC DISCOUNT
# ============================================================

def pool_dirichlet_counts_autodiscount(cleaned):
    """
    Pool Dirichlet counts across studies and compute alpha vectors.

    For each CIS target variable:
      1. Compute I2 (inter-survey heterogeneity).
      2. discount = 1.0 (lambda=0; I2 stored for Appendix B, does not reduce ESS).
      3. ESS = total_N (full pooled sample, per Section 2.3).
      4. alpha_lit = ESS * pooled_proportions.

    Pooling uses simple n-weighted averaging of proportions.
    An adaptive floor (1 / (2 * K * ESS_approx), clipped to [0.001, 0.01])
    prevents zero counts in sparse categories.

    Parameters
    ----------
    cleaned : dict
        Output of validate_mappings.

    Returns
    -------
    pooled : dict
        Per-variable dict with alpha, ESS, I2 metadata, and study details.
    """
    print(f"\n{'=' * 60}")
    print("PART 3: POOLING WITH AUTOMATIC DISCOUNT (I2 statistic)")
    print(f"{'=' * 60}")
    print(f"\n  Method: Higgins & Thompson (2002) I2 heterogeneity")
    print(f"  I2 computed for reporting (Appendix B); ESS = full pooled N (Section 2.3)")

    pooled = {}

    for target, entries in sorted(cleaned.items()):
        if not entries:
            continue

        target_K = len(entries[0]['dirichlet_counts'])
        I2_pct, discount, i2_metadata = calculate_i2_and_discount(entries, target)

        total_weight = 0.0
        pooled_prop = np.zeros(target_K)
        details = []

        for e in entries:
            dc = np.array(e['dirichlet_counts'], dtype=float)
            n = e['total_n'] if e.get('total_n') and e['total_n'] > 0 else dc.sum()
            dc_sum = dc.sum()
            if dc_sum <= 0:
                continue

            prop = dc / dc_sum
            pooled_prop += n * prop
            total_weight += n

            details.append({
                'study_id': e['study_id'],
                'source_variable': e['source_variable'],
                'dirichlet_counts': e['dirichlet_counts'],
                'proportions': prop.tolist(),
                'total_n': int(n),
                'weight': float(n),
            })

        if total_weight <= 0:
            continue

        pooled_prop /= total_weight

        # Adaptive floor to prevent zero-probability categories
        ess_approx = 0.1 * total_weight
        adaptive_floor = 1.0 / (2 * target_K * max(1.0, ess_approx))
        adaptive_floor = float(np.clip(adaptive_floor, 0.001, 0.01))

        pooled_prop = np.maximum(pooled_prop, adaptive_floor)
        pooled_prop /= pooled_prop.sum()

        # ESS = full pooled N; discount=1.0 per Section 2.3 (I2 does not reduce ESS)
        ESS_var = float(discount * total_weight)
        ESS_var = max(ESS_var, float(target_K))
        alpha_lit = (ESS_var * pooled_prop).tolist()

        bins = np.arange(1, target_K + 1)
        mu = float(np.dot(pooled_prop, bins))
        sigma = float(np.sqrt(np.dot(pooled_prop, (bins - mu) ** 2)))

        pooled[target] = {
            'alpha': alpha_lit,
            'ESS': ESS_var,
            'K': target_K,
            'pooled_proportions': pooled_prop.tolist(),
            'pooled_mean': mu,
            'pooled_std': sigma,
            'n_studies': len(details),
            'total_n': int(total_weight),
            'discount_factor': float(discount),
            'I2_statistic': i2_metadata,
            'source': 'literature_meta_analysis_autodiscount',
            'study_details': details,
        }

    n_cis_approx = 2254  # CIS Study 3391 sample size
    print(f"\n  Pooled priors: {len(pooled)}")
    print(f"\n  {'Target':<35s} {'K':<3s} {'#':<3s} {'N':<7s} {'I2%':<6s} {'disc':<5s} {'ESS':<7s} {'wt%':<6s}")
    print(f"  {'-' * 100}")

    for t in sorted(pooled.keys())[:20]:
        p = pooled[t]
        i2 = p.get('I2_statistic', {}).get('I2_pct', 0.0)
        pct = 100.0 * p['ESS'] / (p['ESS'] + n_cis_approx)
        print(
            f"  {t:<35s} {p['K']:<3d} {p['n_studies']:<3d} {p['total_n']:<7d} "
            f"{i2:<6.1f} {p['discount_factor']:<5.2f} {p['ESS']:<7.0f} {pct:<6.1f}%"
        )

    if len(pooled) > 20:
        print(f"  ... and {len(pooled) - 20} more variables")

    discounts = [p['discount_factor'] for p in pooled.values()]
    i2_values = [p.get('I2_statistic', {}).get('I2_pct', 0.0) for p in pooled.values()]

    print(f"\n  DISCOUNT DISTRIBUTION:")
    print(f"    Mean:   {np.mean(discounts):.3f}")
    print(f"    Median: {np.median(discounts):.3f}")
    print(f"    Range:  [{np.min(discounts):.3f}, {np.max(discounts):.3f}]")

    print(f"\n  I2 HETEROGENEITY DISTRIBUTION:")
    print(f"    Mean:   {np.mean(i2_values):.1f}%")
    print(f"    Median: {np.median(i2_values):.1f}%")
    print(f"    Low (<25%):        {sum(1 for x in i2_values if x < 25)} variables")
    print(f"    Moderate (25-50%): {sum(1 for x in i2_values if 25 <= x < 50)} variables")
    print(f"    High (50-75%):     {sum(1 for x in i2_values if 50 <= x < 75)} variables")
    print(f"    Very high (>=75%): {sum(1 for x in i2_values if x >= 75)} variables")

    return pooled


# ============================================================
# CAUSAL STRUCTURE
# ============================================================

def get_theoretical_structure():
    """
    Return the DAG tier structure and edge list used in step3_fusion.py.

    Causal ordering: Demographics -> Attitudes -> Behaviors -> Acceptance.
    Edge weights are derived from the cited meta-analytic literature.
    """
    return {
        'tiers': {
            0: [
                'demographics_age', 'demographics_gender',
                'demographics_education', 'demographics_income',
                'demographics_urban_rural',
            ],
            1: [
                'attitudes_env_concern', 'attitudes_wtp_prices',
                'attitudes_wtp_taxes', 'attitudes_wtp_lifestyle',
                'attitudes_env_science_optimism',
                'perception_quality', 'perception_safety',
                'perception_danger_pollution', 'trust_institutions',
            ],
            2: ['behavior_recycling', 'behavior_reduce_consumption'],
            3: ['acceptance_general', 'acceptance_food_packaging',
                'price_willingness_to_pay_premium'],
        },
        'edges': [
            ('demographics_age',            'attitudes_env_concern',       0.75, 'Stern 2000'),
            ('demographics_education',       'attitudes_env_concern',       0.85, 'Ajzen 1991'),
            ('demographics_income',          'attitudes_wtp_prices',        0.80, 'Ruokamo 2022'),
            ('demographics_gender',          'attitudes_env_concern',       0.70, 'Stern 2000'),
            ('attitudes_env_concern',        'behavior_recycling',          0.85, 'Ajzen 1991'),
            ('attitudes_env_concern',        'acceptance_general',          0.80, 'Ajzen 1991'),
            ('attitudes_wtp_prices',         'acceptance_general',          0.85, 'Ruokamo 2022'),
            ('perception_quality',           'acceptance_general',          0.80, 'Magnier 2019'),
            ('perception_safety',            'acceptance_food_packaging',   0.85, 'De Marchi 2020'),
            ('behavior_recycling',           'acceptance_general',          0.70, 'Stern 2000'),
            ('acceptance_general',           'acceptance_food_packaging',   0.80, 'hierarchical'),
            ('perception_danger_pollution',  'attitudes_env_concern',       0.75, 'Stern 2000'),
            ('trust_institutions',           'acceptance_general',          0.70, 'institutional'),
            ('attitudes_env_science_optimism', 'attitudes_wtp_lifestyle',   0.65, 'Dunlap 2008'),
        ],
    }


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(pooled, structure, flat):
    """Write all output files for downstream pipeline steps."""
    print(f"\n{'=' * 60}")
    print("SAVING OUTPUTS")
    print(f"{'=' * 60}")

    # 1. Dirichlet priors (primary input for step3_fusion.py)
    out = {
        'method': 'category_mappings.json -> n-weighted pooling -> autodiscount via I2 statistic',
        'paper_section': 'Section 3.2 (Ibrahim & Chen 2000; Higgins & Thompson 2002)',
        'ESS_method': 'variable_specific_autodiscount',
        'discount_formula': 'discount = 1.0 (lambda=0; I2 for reporting only)',
        'n_variables': len(pooled),
        'priors': {
            v: {
                'alpha': p['alpha'],
                'K': p['K'],
                'ESS': p['ESS'],
                'pooled_proportions': p['pooled_proportions'],
                'pooled_mean': p['pooled_mean'],
                'pooled_std': p['pooled_std'],
                'n_studies': p['n_studies'],
                'total_n': p['total_n'],
                'discount_factor': p['discount_factor'],
                'I2_statistic': p['I2_statistic'],
                'source': p['source'],
                'study_details': p['study_details'],
            }
            for v, p in pooled.items()
        },
    }
    with open("step2_dirichlet_priors.json", "w", encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print("  Saved: step2_dirichlet_priors.json")

    # 2. DAG structure
    with open("step2_theoretical_structure.json", "w", encoding='utf-8') as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)
    print("  Saved: step2_theoretical_structure.json")

    # 3. Summary CSV for paper
    rows = [
        {
            'Variable': v,
            'K': p['K'],
            'N_Studies': p['n_studies'],
            'Total_N': p['total_n'],
            'I2_pct': f"{p.get('I2_statistic', {}).get('I2_pct', 0.0):.1f}",
            'Discount': f"{p['discount_factor']:.3f}",
            'ESS': f"{p['ESS']:.0f}",
            'Mean': f"{p['pooled_mean']:.3f}",
            'Std': f"{p['pooled_std']:.3f}",
            'Alpha': ', '.join(f"{a:.1f}" for a in p['alpha']),
        }
        for v, p in sorted(pooled.items())
    ]
    pd.DataFrame(rows).to_csv("step2_summary_for_paper.csv", index=False)
    print("  Saved: step2_summary_for_paper.csv")

    # 4. I2 diagnostics per variable (Appendix B)
    i2_rows = [
        {
            'variable': v,
            'I2_pct': p.get('I2_statistic', {}).get('I2_pct', 0.0),
            'discount': p['discount_factor'],
            'n_surveys': p['n_studies'],
            'total_n': p['total_n'],
            'ESS': p['ESS'],
            'interpretation': (
                'low'       if p.get('I2_statistic', {}).get('I2_pct', 0.0) < 25 else
                'moderate'  if p.get('I2_statistic', {}).get('I2_pct', 0.0) < 50 else
                'high'      if p.get('I2_statistic', {}).get('I2_pct', 0.0) < 75 else
                'very_high'
            ),
        }
        for v, p in sorted(pooled.items())
    ]
    pd.DataFrame(i2_rows).to_csv("step2_i2_heterogeneity_analysis.csv", index=False)
    print("  Saved: step2_i2_heterogeneity_analysis.csv")

    # 5. Raw input mappings (audit trail)
    with open("step2_input_mappings.json", "w", encoding='utf-8') as f:
        json.dump(flat, f, indent=2, ensure_ascii=False, default=str)
    print("  Saved: step2_input_mappings.json")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("STEP 2 - LITERATURE PRIORS WITH AUTOMATIC DISCOUNT")
    print("=" * 60)
    print("\nMethod: I2 heterogeneity statistic (Higgins & Thompson 2002)")
    print("ESS = full pooled N (I2 computed for Appendix B, does not discount ESS)")

    if not os.path.exists("category_mappings.json"):
        print("\nERROR: category_mappings.json not found.")
        print("Run harmonize_categories.py first.")
        return

    flat, by_target = load_category_mappings()
    cleaned = validate_mappings(by_target)
    pooled = pool_dirichlet_counts_autodiscount(cleaned)
    structure = get_theoretical_structure()
    save_outputs(pooled, structure, flat)

    cis_vars = {
        'demographics_gender', 'demographics_age', 'demographics_education',
        'demographics_income', 'demographics_urban_rural',
        'attitudes_env_concern', 'attitudes_wtp_prices', 'attitudes_wtp_taxes',
        'attitudes_wtp_lifestyle', 'behavior_recycling', 'behavior_reduce_consumption',
        'attitudes_env_science_optimism', 'trust_institutions', 'perception_danger_pollution',
    }

    shared = sorted(v for v in pooled if v in cis_vars)
    lit_only = sorted(v for v in pooled if v not in cis_vars)

    print(f"\n{'=' * 60}")
    print("STEP 2 COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Total priors:                         {len(pooled)}")
    print(f"  Shared with CIS (-> conjugate fusion): {len(shared)}")
    print(f"  Literature-only (-> informative prior): {len(lit_only)}")

    print(f"\n  I2 and discount per example variable:")
    for v in ['acceptance_general', 'attitudes_env_concern', 'perception_safety', 'demographics_gender']:
        if v in pooled:
            p = pooled[v]
            i2 = p.get('I2_statistic', {}).get('I2_pct', 0.0)
            print(f"    {v:<35s}  I2={i2:5.1f}%  discount={p['discount_factor']:.3f}  ESS={p['ESS']:.0f}")

    print(f"\nNext: python step3_fusion.py")


if __name__ == "__main__":
    main()
