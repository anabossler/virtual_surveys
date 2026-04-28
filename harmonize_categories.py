"""
harmonize_categories.py
=======================
Survey harmonisation pipeline for Bayesian meta-analysis.

Reads structured JSON files for each international study, calls the
Anthropic API to extract numerical evidence, and produces:

    category_mappings.json          Dirichlet alpha vectors (marginal priors, Step 2)
    regression_priors.json          Pooled regression coefficients per predictor (Step 3)
    extractions_raw.json            Raw Claude output checkpoint
    category_mappings_review.csv    Human-readable marginal review
    regression_priors_review.csv    Human-readable coefficient review

Extraction supports four evidence types:
    regression_coefficient  explicit beta, SE, p from a regression model
    conditional_means       group means converted to beta via Cohen's d
    correlation             Pearson r converted to beta via r * pi/sqrt(3)
    group_percentages       acceptance rates converted to beta via log-OR

Usage:
    python harmonize_categories.py               # full extraction + computation
    python harmonize_categories.py --recompute   # recompute without API calls
"""

import os
import json
import time
import numpy as np
import anthropic
from scipy.stats import norm
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
METADATA_DIR = os.getenv("METADATA_DIR", "metadata")


# Target variable definitions for the CIS survey
# Each entry specifies the number of categories K and the scale type
CIS_TARGETS = {
    "demographics_gender":              {"K": 2,  "type": "categorical"},
    "demographics_age":                 {"K": 5,  "type": "ordinal_bins"},
    "demographics_education":           {"K": 7,  "type": "ordinal"},
    "demographics_income":              {"K": 5,  "type": "ordinal"},
    "demographics_urban_rural":         {"K": 5,  "type": "ordinal"},
    "attitudes_env_concern":            {"K": 5,  "type": "likert"},
    "attitudes_wtp_prices":             {"K": 5,  "type": "likert"},
    "attitudes_wtp_taxes":              {"K": 5,  "type": "likert"},
    "attitudes_wtp_lifestyle":          {"K": 5,  "type": "likert"},
    "behavior_recycling":               {"K": 4,  "type": "likert"},
    "behavior_reduce_consumption":      {"K": 4,  "type": "likert"},
    "attitudes_env_science_optimism":   {"K": 5,  "type": "likert"},
    "trust_institutions":               {"K": 11, "type": "scale_0_10"},
    "perception_danger_pollution":      {"K": 5,  "type": "likert"},
    "acceptance_general":               {"K": 5,  "type": "likert"},
    "acceptance_food_packaging":        {"K": 5,  "type": "likert"},
    "acceptance_non_food_packaging":    {"K": 5,  "type": "likert"},
    "acceptance_beverages":             {"K": 5,  "type": "likert"},
    "acceptance_cleaning_products":     {"K": 5,  "type": "likert"},
    "acceptance_personal_care":         {"K": 5,  "type": "likert"},
    "acceptance_toys":                  {"K": 5,  "type": "likert"},
    "acceptance_clothing_textiles":     {"K": 5,  "type": "likert"},
    "acceptance_electronics":           {"K": 5,  "type": "likert"},
    "acceptance_furniture":             {"K": 5,  "type": "likert"},
    "perception_safety":                {"K": 5,  "type": "likert"},
    "perception_quality":               {"K": 5,  "type": "likert"},
    "price_willingness_to_pay_premium": {"K": 5,  "type": "likert"},
}

# Variables present only in the literature (no CIS equivalent)
# Their priors come entirely from the meta-analysis
LITERATURE_ONLY_TARGETS = {
    "acceptance_food_packaging",
    "acceptance_general",
    "acceptance_non_food_packaging",
    "acceptance_beverages",
    "acceptance_cleaning_products",
    "acceptance_personal_care",
    "acceptance_toys",
    "acceptance_clothing_textiles",
    "acceptance_electronics",
    "acceptance_furniture",
    "perception_safety",
    "perception_quality",
    "price_willingness_to_pay_premium",
}

# Studies excluded from marginal pooling due to construct mismatch,
# experimental contamination, or incompatible scale
# Values are either 'all' (exclude all source variables) or a list of
# specific source_variable names to exclude
MARGINAL_PRIOR_EXCLUSIONS = {
    'neumann_2024_pcr_nudging_germany': {
        'acceptance_general': 'all',
        'price_willingness_to_pay_premium': 'all',
    },
    'cao_lu_zhu_2022_eol_recycled_preferences': {
        'acceptance_general': [
            'eol_scenario_reuse', 'eol_scenario_material_recovery',
            'eol_scenario_remanufacturing', 'eol_scenario_refurbishment',
            'recycled_products_remanufactured', 'recycled_products_refurbished',
        ],
    },
    'ipsos_2022_single_use_plastics_global': {
        'acceptance_general': 'all',
    },
    'zwicker_2020_plastic_attitudes': {
        'acceptance_general': 'all',
    },
    'abella_2022_recycled_plastics_perception': {
        'acceptance_general': [
            'material_identification_accuracy_M1',
            'material_identification_accuracy_M2',
            'material_identification_accuracy_M3',
            'material_identification_accuracy_overall',
        ],
    },
    'leal_filho_2025': {
        'acceptance_general': 'all',
    },
    'plasticircle_valencia_2019': {
        'acceptance_general': 'all',
    },
}

# Maps predictor names found in study JSONs to CIS target dimensions
# Used by compute_regression_priors to connect coefficients to DAG nodes
PREDICTOR_TO_CIS = {
    "female":                                      "demographics_gender",
    "gender":                                      "demographics_gender",
    "sex":                                         "demographics_gender",
    "female_choose_less_plastic":                  "demographics_gender",
    "female_bottle_own_water":                     "demographics_gender",
    "under_30":                                    "demographics_age",
    "under_30_years":                              "demographics_age",
    "over_60":                                     "demographics_age",
    "at_least_60_years":                           "demographics_age",
    "age":                                         "demographics_age",
    "age_recycling_positive":                      "demographics_age",
    "secondary":                                   "demographics_education",
    "secondary_education":                         "demographics_education",
    "tertiary":                                    "demographics_education",
    "tertiary_education":                          "demographics_education",
    "high_education":                              "demographics_education",
    "education":                                   "demographics_education",
    "income":                                      "demographics_income",
    "income_bracket":                              "demographics_income",
    "urban":                                       "demographics_urban_rural",
    "rural":                                       "demographics_urban_rural",
    "countryside":                                 "demographics_urban_rural",
    "living_environment":                          "demographics_urban_rural",
    "env_worry_plastics":                          "attitudes_env_concern",
    "env_concern":                                 "attitudes_env_concern",
    "environmental_concern":                       "attitudes_env_concern",
    "environmental_awareness":                     "attitudes_env_concern",
    "environmental_awareness_direct":              "attitudes_env_concern",
    "env_friendliness_important":                  "attitudes_env_concern",
    "environmental_friendliness_importance":       "attitudes_env_concern",
    "env_concern_recycling_positive":              "attitudes_env_concern",
    "recyclability_important":                     "attitudes_env_concern",
    "recyclability_importance":                    "attitudes_env_concern",
    "recycling":                                   "behavior_recycling",
    "recycles_plastic":                            "behavior_recycling",
    "recycling_frequency":                         "behavior_recycling",
    "wtp":                                         "attitudes_wtp_prices",
    "willingness_to_pay":                          "attitudes_wtp_prices",
    "culpa_guilt":                                 "attitudes_wtp_prices",
    "price_sensitivity":                           "attitudes_wtp_prices",
    "perceived_benefit":                           "perception_quality",
    "perceived_benefit_to_willingness":            "perception_quality",
    "safety_important":                            "perception_safety",
    "safety_importance":                           "perception_safety",
    "perceived_risk":                              "perception_safety",
    "perceived_risk_to_willingness":               "perception_safety",
    "appearance_most_important":                   "perception_quality",
    "appearance_importance":                       "perception_quality",
    "product_ownership":                           "perception_quality",
    "environmental_awareness_to_perceived_benefit": "perception_quality",
    "environmental_awareness_to_perceived_risk":    "perception_safety",
    "pcr_recycled_content_effect":                 "acceptance_food_packaging",
    "pleasantness_effect_on_acceptability":        "acceptance_food_packaging",
    "self_efficacy_cue":                           "acceptance_general",
    "female_choose_less_plastic__female":          "demographics_gender",
    "female_bottle_own_water__female":             "demographics_gender",
    "no_gender_effect_wtp_bioplastics__male":      "demographics_gender",
    "env_concern_recycling_positive__r_predictor": "attitudes_env_concern",
}


# Prompt template used to instruct Claude during extraction
EXTRACTION_PROMPT = """Extract numerical data from this survey JSON for Bayesian meta-analysis.

## CIS TARGETS (what we need priors for):
{cis_targets_list}

## SURVEY JSON:
Study: {study_id}
{survey_json}

## YOUR JOB — TWO TASKS:

### TASK 1: MARGINAL DISTRIBUTIONS
Find any field with mean, std, counts, percentages that describes the distribution
of a variable matching a CIS target. Copy numbers EXACTLY.

IMPORTANT: Always check "bayesian_extraction_for_harmonize" section first if present —
these are pre-mapped entries ready to extract. Copy them AS IS.
Also check "regression_coefficients_for_harmonize" section if present.

data_type options for marginals:
- "counts"            integer counts per category
- "proportions"       proportions summing to 1
- "percentages"       percentages summing to 100
- "mean_std"          mean + std of a scale
- "mean_only"         only mean available
- "single_percentage" one percentage (e.g. "85% agree")
- "review_aggregate"  review paper, no primary data (skip regression too)

### TASK 2: REGRESSION COEFFICIENTS (CRITICAL)
Extract ALL numerical regression evidence. Look in these locations in order:

1. "bayesian_extraction_for_harmonize".extractions  (PRIORITY — already mapped)
2. "regression_coefficients_for_harmonize".entries  (PRIORITY — already mapped)
3. regression_analysis.models[*].significant_variables
4. stratification_for_virtual_surveys.primary_stratification.demographic_strata[*].effect_on_outcome
5. stratification_for_virtual_surveys.behavioral_attitudinal_strata.strata[*].effect_on_outcome
6. statistical_analyses.mediation_analysis.regression_paths
7. key_findings.predictors_of_willingness
8. network_analysis_findings.wtp_connections

Use the MOST SPECIFIC data_type available:

data_type="regression_coefficient" — explicit beta, SE, p from regression model:
  values: {{coefficient, std_error, p_value, significance, model_type, model_name,
            best_model, n_observations, outcome_variable, reference_category}}

data_type="conditional_means" — means by group (no formal regression):
  Use when study reports "female mean=3.8 (n=120), male mean=3.2 (n=100)" on a scale.
  values: {{groups: {{group_name: {{mean, n}}, ...}}, scale_min, scale_max, grouping_variable}}

data_type="correlation" — direct r coefficient:
  values: {{r, n, predictor, outcome, p_value}}

data_type="group_percentages" — acceptance rate by group:
  Use when study reports "45% of university vs 22% of primary school accept recycled".
  values: {{groups: {{group_name: percentage}}, n_total, grouping_variable}}

## OUTPUT FORMAT:

For MARGINAL extractions:
{{
  "cis_target": "acceptance_food_packaging",
  "source_variable": "recycled_plastic_attractiveness",
  "source_scale": "likert_5_1to5",
  "source_K": 5,
  "data_type": "mean_std",
  "values": {{"mean": 3.4, "std": 1.1, "n": 272, "scale_min": 1, "scale_max": 5}},
  "scale_direction": "1=low_5=high",
  "notes": ""
}}

For REGRESSION COEFFICIENT extractions:
{{
  "cis_target": "acceptance_food_packaging",
  "source_variable": "female",
  "source_scale": "binary",
  "source_K": 2,
  "data_type": "regression_coefficient",
  "values": {{
    "coefficient": 0.463,
    "std_error": 0.157,
    "p_value": 0.003,
    "significance": "***",
    "model_type": "ordered_probit",
    "model_name": "Full model with purchasing attitudes",
    "best_model": true,
    "n_observations": 272,
    "outcome_variable": "recycled_plastic_attractiveness",
    "reference_category": null
  }},
  "scale_direction": "positive=higher_acceptance",
  "notes": ""
}}

For CONDITIONAL MEANS:
{{
  "cis_target": "acceptance_food_packaging",
  "source_variable": "acceptance_by_gender",
  "source_scale": "likert_5",
  "source_K": 5,
  "data_type": "conditional_means",
  "values": {{
    "groups": {{"female": {{"mean": 3.8, "n": 120}}, "male": {{"mean": 3.2, "n": 100}}}},
    "scale_min": 1,
    "scale_max": 5,
    "grouping_variable": "gender"
  }},
  "scale_direction": "1=low_5=high",
  "notes": ""
}}

For CORRELATION:
{{
  "cis_target": "behavior_recycling",
  "source_variable": "env_concern_recycling_r",
  "source_scale": "pearson_r",
  "source_K": null,
  "data_type": "correlation",
  "values": {{"r": 0.029, "n": 36, "predictor": "environmental_attitude",
              "outcome": "consumption_habits", "p_value": 0.87}},
  "scale_direction": "positive=more_recycling",
  "notes": ""
}}

## RULES:
1. Copy numbers EXACTLY. Never compute or infer.
2. If "bayesian_extraction_for_harmonize" or "regression_coefficients_for_harmonize"
   exists, copy ALL entries from those sections into the output AS IS.
3. Then also scan the rest of the JSON for any additional evidence not covered.
4. Extract EVERY predictor separately, even if non-significant.
5. Include std_error and p_value when available. Write null if missing.
6. significance: "***" p<0.001, "**" p<0.01, "*" p<0.05, "ns" otherwise.
7. best_model: true only if JSON says best_model=true or lowest AIC.
8. If study is a review (is_review_paper=true or "review_aggregate") skip regression.
9. cis_target mapping for outcome variables:
   acceptance/willingness_to_use/attractiveness -> "acceptance_food_packaging" or "acceptance_general"
   env_concern/env_worry/environmental_awareness -> "attitudes_env_concern"
   wtp/willingness_to_pay                        -> "attitudes_wtp_prices"
   recycling                                     -> "behavior_recycling"
   safety/risk/perceived_risk                    -> "perception_safety"
   quality/benefit/perceived_benefit             -> "perception_quality"
   reduce_consumption/less_plastic               -> "behavior_reduce_consumption"

Return ONLY valid JSON:
{{"study_id": "{study_id}", "extractions": [...], "skipped_variables": [{{"variable": "...", "reason": "..."}}]}}"""


def is_excluded_from_marginal(study_id, cis_target, source_variable):
    """Return True if this extraction should be excluded from marginal pooling."""
    excl = MARGINAL_PRIOR_EXCLUSIONS.get(study_id, {}).get(cis_target, [])
    if excl == 'all':
        return True
    return source_variable in excl


def compute_dirichlet(extraction, cis_info):
    """
    Convert extracted marginal values to a Dirichlet alpha vector.

    Parameters
    ----------
    extraction : dict
        One extraction entry from the Claude output.
    cis_info : dict
        Entry from CIS_TARGETS with keys 'K' and 'type'.

    Returns
    -------
    alpha : list of float or None
        Dirichlet pseudo-counts aligned to target_K bins.
    method : str
        Description of the computation path, or error message prefixed
        with 'skip:' when the entry should not produce a marginal prior.
    """
    target_K = cis_info['K']
    data_type = extraction.get('data_type')
    values = extraction.get('values') or extraction.get('data') or {}
    source_scale = extraction.get('source_scale', '')
    source_K = extraction.get('source_K')
    n = values.get('n')

    skip_types = (
        'review_aggregate', 'regression_coefficient',
        'conditional_means', 'correlation', 'group_percentages',
    )
    if data_type in skip_types:
        return None, f"skip:{data_type}"

    if data_type == 'counts':
        counts = values.get('counts', [])
        if not counts:
            return None, "empty counts"
        counts = [max(c, 0) for c in counts]
        if len(counts) == target_K:
            return counts, "direct_counts"
        return _rescale_counts(counts, target_K, source_K, source_scale)

    if data_type == 'proportions':
        props = values.get('proportions', [])
        if not props:
            return None, "empty proportions"
        total_n = n if n else 1000
        if len(props) == target_K:
            return [max(p * total_n, 0.5) for p in props], f"proportions*n={total_n}"
        counts = [max(p * total_n, 0.5) for p in props]
        return _rescale_counts(counts, target_K, source_K, source_scale)

    if data_type == 'percentages':
        pcts = values.get('percentages', [])
        if not pcts:
            return None, "empty percentages"
        total_n = n if n else 1000
        props = [p / 100.0 for p in pcts]
        if len(props) == target_K:
            return [max(p * total_n, 0.5) for p in props], f"percentages->proportions*n={total_n}"
        counts = [max(p * total_n, 0.5) for p in props]
        return _rescale_counts(counts, target_K, source_K, source_scale)

    if data_type == 'single_percentage':
        pct = values.get('percentage')
        if pct is None:
            return None, "missing percentage"
        total_n = n if n else 1000
        p = pct / 100.0
        if target_K == 5:
            alpha = [
                (1 - p) * 0.3 * total_n,
                (1 - p) * 0.7 * total_n,
                0.05 * total_n,
                p * 0.5 * total_n,
                p * 0.5 * total_n,
            ]
        elif target_K == 4:
            alpha = [
                (1 - p) * 0.4 * total_n,
                (1 - p) * 0.6 * total_n,
                p * 0.6 * total_n,
                p * 0.4 * total_n,
            ]
        elif target_K == 2:
            alpha = [(1 - p) * total_n, p * total_n]
        elif target_K == 11:
            implied_mean = p * 10
            std_est = 2.0
            alpha = []
            for k in range(11):
                pk = (norm.cdf(k + 0.5, implied_mean, std_est) -
                      norm.cdf(k - 0.5, implied_mean, std_est))
                alpha.append(max(pk * total_n, 0.5))
        else:
            mid = target_K // 2
            alpha = []
            for k in range(target_K):
                if k < mid:
                    alpha.append((1 - p) / mid * total_n)
                elif k == mid and target_K % 2 == 1:
                    alpha.append(0.05 * total_n)
                else:
                    remaining = target_K - mid - (1 if target_K % 2 == 1 else 0)
                    alpha.append(p / max(remaining, 1) * total_n)
        alpha = [max(a, 0.5) for a in alpha]
        return alpha, f"single_pct({pct}%)->K={target_K}*n={total_n}"

    if data_type == 'mean_std':
        mean = values.get('mean')
        std = values.get('std') or values.get('se')
        if mean is None or std is None:
            return None, "missing mean or std"
        if std <= 0:
            std = 0.5
        total_n = n if n else 100
        return _normal_to_bins(mean, std, total_n, target_K, source_K,
                               source_scale, values.get('scale_min'), values.get('scale_max'))

    if data_type == 'mean_only':
        mean = values.get('mean')
        if mean is None:
            return None, "missing mean"
        scale_min = values.get('scale_min')
        scale_max = values.get('scale_max')
        if scale_min is not None and scale_max is not None:
            std = (scale_max - scale_min) * 0.25
        elif source_K and source_K > 1:
            std = (source_K - 1) * 0.25
        else:
            std = 1.0
        total_n = n if n else 100
        return _normal_to_bins(mean, std, total_n, target_K, source_K,
                               source_scale, scale_min, scale_max)

    return None, f"unknown data_type: {data_type}"


def _normal_to_bins(mean, std, n, target_K, source_K, source_scale,
                    scale_min=None, scale_max=None):
    """
    Discretise Normal(mean, std) into target_K equal-width bins.

    The source scale is linearly rescaled to [1, target_K] before binning.
    """
    if scale_min is not None and scale_max is not None:
        s_min, s_max = scale_min, scale_max
    elif source_K and source_K > 0:
        s_min, s_max = 1, source_K
    elif 'vas' in str(source_scale).lower() or '100' in str(source_scale):
        s_min, s_max = 0, 100
    elif '10' in str(source_scale) and 'likert' not in str(source_scale).lower():
        s_min, s_max = 0, 10
    else:
        s_min, s_max = 1, 5

    src_range = s_max - s_min
    tgt_range = target_K - 1

    if src_range > 0 and src_range != tgt_range:
        mean_r = 1 + (mean - s_min) * tgt_range / src_range
        std_r = std * tgt_range / src_range
    else:
        mean_r = mean
        std_r = std

    std_r = max(std_r, 0.3)

    alpha = []
    for k in range(1, target_K + 1):
        p_k = (norm.cdf(k + 0.5, mean_r, std_r) -
               norm.cdf(k - 0.5, mean_r, std_r))
        p_k = max(p_k, 0.005)
        alpha.append(p_k * n)

    method = (f"Normal({mean:.2f},{std:.2f})[{s_min}-{s_max}]"
              f"->rescale->K={target_K}bins*n={n}")
    return alpha, method


def _rescale_counts(counts, target_K, source_K, source_scale):
    """Collapse or expand a count vector from len(counts) bins to target_K bins."""
    src_K = len(counts)
    if src_K == target_K:
        return counts, "exact_match"
    if src_K > target_K:
        alpha = [0.0] * target_K
        ratio = src_K / target_K
        for i in range(src_K):
            tb = min(int(i / ratio), target_K - 1)
            alpha[tb] += counts[i]
        return alpha, f"collapse_{src_K}->{target_K}"
    else:
        alpha = [0.0] * target_K
        ratio = target_K / src_K
        for i in range(src_K):
            start = int(i * ratio)
            end = min(int((i + 1) * ratio), target_K)
            per_bin = counts[i] / max(end - start, 1)
            for j in range(start, end):
                alpha[j] += per_bin
        return alpha, f"expand_{src_K}->{target_K}"


def _t_to_p(t_stat, n):
    """Two-tailed p-value from a t-statistic with n-2 degrees of freedom."""
    from scipy.stats import t as t_dist
    return float(2 * t_dist.sf(abs(t_stat), max(n - 2, 1)))


def _pval_to_sig(p):
    """Convert p-value to significance star notation."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def compute_beta_from_conditional(extraction):
    """
    Convert conditional_means, correlation, or group_percentages to an
    equivalent logistic regression coefficient beta.

    Conversion formulas:
        conditional_means : Cohen's d  -> beta = d * pi / sqrt(3)
        correlation       : r          -> beta = r * pi / sqrt(3)
        group_percentages : log-OR     -> beta = log_OR * sqrt(3) / pi

    Parameters
    ----------
    extraction : dict
        One extraction entry of a conditional type.

    Returns
    -------
    betas : dict or None
        Mapping from group / predictor name to coefficient dict.
    method : str
        Description of the conversion path, or error message if None.
    """
    data_type = extraction.get('data_type')
    values = extraction.get('values') or {}

    if data_type == 'conditional_means':
        groups = values.get('groups', {})
        scale_min = values.get('scale_min') or 1
        scale_max = values.get('scale_max') or 5

        sd_pooled = (scale_max - scale_min) / 4.0
        if sd_pooled <= 0:
            sd_pooled = 1.0

        valid_groups = {k: v for k, v in groups.items()
                        if isinstance(v, dict) and 'mean' in v}
        if len(valid_groups) < 2:
            return None, "insufficient groups (need >= 2 with mean)"

        sorted_groups = sorted(valid_groups.items(), key=lambda x: x[1]['mean'])
        ref_name, ref_data = sorted_groups[0]
        ref_mean = ref_data['mean']
        ref_n = ref_data.get('n', 100)

        betas = {}
        for gname, gdata in valid_groups.items():
            g_mean = gdata['mean']
            g_n = gdata.get('n', 100)

            d = (g_mean - ref_mean) / max(sd_pooled, 0.1)
            beta = d * (np.pi / np.sqrt(3))
            se = np.sqrt(1 / max(ref_n, 1) + 1 / max(g_n, 1)) * sd_pooled
            t_stat = d * np.sqrt((ref_n * g_n) / max(ref_n + g_n, 1))
            p_val = _t_to_p(t_stat, ref_n + g_n)

            betas[gname] = {
                'coefficient': round(beta, 4),
                'std_error': round(se, 4),
                'p_value': round(p_val, 4),
                'significance': _pval_to_sig(p_val),
                'method': 'cohen_d_to_beta_logit',
                'reference': ref_name,
                'cohen_d': round(d, 4),
                'n_group': g_n,
                'n_reference': ref_n,
            }

        return betas, f'conditional_means->beta (ref={ref_name})'

    if data_type == 'correlation':
        r = values.get('r')
        n = values.get('n') or 100
        if r is None:
            return None, "missing r value"
        r = float(r)

        beta = r * (np.pi / np.sqrt(3))
        se_r = np.sqrt((1 - r ** 2) ** 2 / max(n - 2, 1))
        se = se_r * (np.pi / np.sqrt(3))
        t_stat = r * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - r ** 2, 1e-10))
        p_val = _t_to_p(t_stat, n)

        pred_name = values.get('predictor', 'r_predictor')
        betas = {
            pred_name: {
                'coefficient': round(beta, 4),
                'std_error': round(se, 4),
                'p_value': round(p_val, 4),
                'significance': _pval_to_sig(p_val),
                'r_original': r,
                'n': n,
                'method': 'r_to_beta_logit',
            }
        }
        return betas, 'correlation->beta_logit'

    if data_type == 'group_percentages':
        groups = values.get('groups', {})
        n_total = values.get('n_total')

        pcts = {}
        for k, v in groups.items():
            if isinstance(v, (int, float)):
                pcts[k] = float(v)
            elif isinstance(v, dict):
                pcts[k] = float(
                    v.get('acceptance_pct', v.get('pct', v.get('percentage', 0)))
                )

        if len(pcts) < 2:
            return None, "insufficient groups (need >= 2 with percentage)"

        if not n_total:
            n_individual = {}
            for k, v in groups.items():
                if isinstance(v, dict) and 'n' in v:
                    n_individual[k] = v['n']
            n_total = sum(n_individual.values()) if n_individual else 1000

        n_per_group = n_total // max(len(pcts), 1)
        ref_name = min(pcts, key=pcts.get)
        ref_p = pcts[ref_name] / 100.0

        betas = {}
        for gname, pct in pcts.items():
            p = np.clip(pct / 100.0, 1e-6, 1 - 1e-6)
            ref_p_c = np.clip(ref_p, 1e-6, 1 - 1e-6)

            log_or = np.log(p / (1 - p)) - np.log(ref_p_c / (1 - ref_p_c))
            beta = log_or * np.sqrt(3) / np.pi
            se = np.sqrt(
                1 / max(n_per_group * p + 1, 1) +
                1 / max(n_per_group * (1 - p) + 1, 1)
            )
            t_stat = beta / max(se, 0.01)
            p_val = _t_to_p(t_stat, n_per_group)

            betas[gname] = {
                'coefficient': round(beta, 4),
                'std_error': round(se, 4),
                'p_value': round(p_val, 4),
                'significance': _pval_to_sig(p_val),
                'pct_original': pct,
                'log_or': round(log_or, 4),
                'reference': ref_name,
                'method': 'log_OR_to_beta_probit',
            }

        return betas, f'group_percentages->beta (ref={ref_name})'

    return None, f"unknown conditional data_type: {data_type}"


def compute_regression_priors(all_results):
    """
    Aggregate regression evidence from all studies into pooled beta coefficients.

    Direct regression_coefficient entries are used as-is.
    conditional_means, correlation, and group_percentages are converted via
    compute_beta_from_conditional before pooling.

    Pooling uses inverse-variance weighting (1/SE^2) when multiple studies
    report the same predictor for the same CIS target.

    Returns
    -------
    dict
        Nested structure: cis_target -> predictors -> dimensions -> levels,
        ready to be written to regression_priors.json and consumed by step3.
    """
    print("\nCOMPUTING REGRESSION PRIORS")
    print("Sources: regression_coefficient + conditional_means + correlation + group_percentages")

    CONDITIONAL_TYPES = ('conditional_means', 'correlation', 'group_percentages')

    raw_coefs = defaultdict(list)

    for result in all_results:
        sid = result.get('study_id', '?')

        for ext in result.get('extractions', []):
            data_type = ext.get('data_type')
            cis_target = ext.get('cis_target')

            if cis_target not in CIS_TARGETS:
                continue

            if data_type == 'regression_coefficient':
                values = ext.get('values') or {}
                coef = values.get('coefficient')
                if coef is None:
                    continue

                source_var = ext.get('source_variable', '').lower().strip()
                entry = {
                    'study_id': sid,
                    'source_variable': source_var,
                    'coefficient': float(coef),
                    'std_error': values.get('std_error'),
                    'p_value': values.get('p_value'),
                    'significance': values.get('significance', 'unknown'),
                    'model_type': values.get('model_type', 'unknown'),
                    'model_name': values.get('model_name', ''),
                    'best_model': values.get('best_model', False),
                    'n_observations': values.get('n_observations'),
                    'outcome_variable': values.get('outcome_variable', ''),
                    'reference_category': values.get('reference_category'),
                    'notes': ext.get('notes', ''),
                    'cis_predictor': PREDICTOR_TO_CIS.get(source_var),
                    'derived_method': 'direct_regression',
                    'derived_level': None,
                }
                raw_coefs[(cis_target, source_var)].append(entry)
                print(f"  {sid:<25s} {cis_target:<30s} <- {source_var:<35s} "
                      f"b={coef:+.3f} {entry['significance']}  [direct]")

            elif data_type in CONDITIONAL_TYPES:
                betas, method = compute_beta_from_conditional(ext)
                if betas is None:
                    print(f"  SKIP {sid}: {cis_target} <- "
                          f"{ext.get('source_variable', '?')}: {method}")
                    continue

                source_var_base = ext.get('source_variable', '').lower().strip()
                values = ext.get('values') or {}
                n_obs = values.get('n')

                for level_name, beta_info in betas.items():
                    src_var = f"{source_var_base}__{level_name}"

                    entry = {
                        'study_id': sid,
                        'source_variable': src_var,
                        'coefficient': beta_info['coefficient'],
                        'std_error': beta_info.get('std_error'),
                        'p_value': beta_info.get('p_value'),
                        'significance': beta_info.get('significance', 'ns'),
                        'model_type': data_type,
                        'model_name': f"{method} -- {ext.get('source_variable', '')}",
                        'best_model': False,
                        'n_observations': n_obs,
                        'outcome_variable': cis_target,
                        'reference_category': beta_info.get('reference'),
                        'notes': ext.get('notes', '') + f' [derived: {method}]',
                        'cis_predictor': (
                            PREDICTOR_TO_CIS.get(src_var) or
                            PREDICTOR_TO_CIS.get(source_var_base) or
                            PREDICTOR_TO_CIS.get(level_name.lower())
                        ),
                        'derived_method': method,
                        'derived_level': level_name,
                    }
                    raw_coefs[(cis_target, src_var)].append(entry)

                    sig = beta_info.get('significance', '?')
                    print(f"  {sid:<25s} {cis_target:<30s} <- {src_var:<35s} "
                          f"b={beta_info['coefficient']:+.3f} {sig}  [{data_type}]")

    if not raw_coefs:
        print("  No regression coefficients found in any study.")
        return {}

    by_target = defaultdict(lambda: defaultdict(list))
    for (cis_target, source_var), entries in raw_coefs.items():
        by_target[cis_target][source_var].extend(entries)

    regression_priors = {}

    for cis_target, predictors in sorted(by_target.items()):
        print(f"\n  TARGET: {cis_target}")

        all_entries_flat = [e for entries in predictors.values() for e in entries]
        best_entries = [e for e in all_entries_flat if e.get('best_model')]
        ref_entry = (best_entries[0] if best_entries else
                     max(all_entries_flat, key=lambda e: e.get('n_observations') or 0))

        pred_dict = {}

        for source_var, entries in sorted(predictors.items()):
            cis_pred = (
                entries[0].get('cis_predictor') or
                PREDICTOR_TO_CIS.get(source_var.lower()) or
                _infer_dim_from_name(source_var)
            )

            if len(entries) == 1:
                e = entries[0]
                pooled_coef = e['coefficient']
                pooled_se = e['std_error']
                pooled_sig = e['significance']
                pooled_n = e['n_observations']
            else:
                weights, coefs = [], []
                for e in entries:
                    se = e.get('std_error')
                    w = (1.0 / se ** 2 if se and se > 0
                         else (e.get('n_observations') or 100) / 1000.0)
                    weights.append(w)
                    coefs.append(e['coefficient'])

                w_arr = np.array(weights)
                c_arr = np.array(coefs)
                pooled_coef = float(np.sum(w_arr * c_arr) / np.sum(w_arr))
                has_se = sum(1 for e in entries if e.get('std_error'))
                pooled_se = float(1.0 / np.sqrt(np.sum(w_arr))) if has_se > 0 else None
                pooled_sig = entries[0]['significance']
                pooled_n = sum(e.get('n_observations') or 0 for e in entries)

            dim = cis_pred or 'unknown'

            if dim not in pred_dict:
                pred_dict[dim] = {
                    'cis_variable': _dim_to_cis_var(dim),
                    'levels': {},
                    'reference_level': None,
                }

            level_name = entries[0].get('derived_level') or source_var

            pred_dict[dim]['levels'][level_name] = {
                'coefficient': round(pooled_coef, 4),
                'std_error': round(pooled_se, 4) if pooled_se else None,
                'significance': pooled_sig,
                'n_studies': len(entries),
                'n_observations': pooled_n,
                'studies': [e['study_id'] for e in entries],
                'derived_method': entries[0].get('derived_method', 'direct'),
            }

            for e in entries:
                if e.get('reference_category'):
                    pred_dict[dim]['reference_level'] = e['reference_category']

            print(f"    {level_name:<40s} -> {dim:<30s} "
                  f"b={pooled_coef:+.3f} {pooled_sig} (n_studies={len(entries)})")

        regression_priors[cis_target] = {
            'model_type': ref_entry.get('model_type', 'unknown'),
            'model_name': ref_entry.get('model_name', ''),
            'study_id': ref_entry.get('study_id', ''),
            'n_observations': ref_entry.get('n_observations'),
            'outcome_variable': ref_entry.get('outcome_variable', ''),
            'predictors': pred_dict,
            'n_predictor_dimensions': len(pred_dict),
            'meta_analysis_note': (
                f"Pooled from {len(all_entries_flat)} entries across "
                f"{len(set(e['study_id'] for e in all_entries_flat))} studies"
            ),
        }

    print(f"\n  Variables with regression priors: {len(regression_priors)}")
    return regression_priors


def _infer_dim_from_name(source_var):
    """Infer a CIS dimension from a predictor name using keyword matching."""
    sv = source_var.lower()
    if any(x in sv for x in ['female', 'gender', 'sex', 'male']):
        return 'demographics_gender'
    if any(x in sv for x in ['age', 'under_30', 'over_60', 'young', 'older']):
        return 'demographics_age'
    if any(x in sv for x in ['edu', 'secondary', 'tertiary', 'primary', 'university']):
        return 'demographics_education'
    if any(x in sv for x in ['income', 'salary']):
        return 'demographics_income'
    if any(x in sv for x in ['urban', 'rural', 'countryside']):
        return 'demographics_urban_rural'
    if any(x in sv for x in ['env', 'worry', 'concern', 'recycle', 'recycl', 'friendliness']):
        return 'attitudes_env_concern'
    if any(x in sv for x in ['benefit', 'quality', 'appearan', 'product', 'ownership']):
        return 'perception_quality'
    if any(x in sv for x in ['risk', 'safety', 'danger']):
        return 'perception_safety'
    if any(x in sv for x in ['guilt', 'culpa', 'wtp', 'willing', 'price', 'sensitivity']):
        return 'attitudes_wtp_prices'
    return 'unknown'


def _dim_to_cis_var(dim):
    """Map a harmonised dimension name to the corresponding CIS variable code."""
    mapping = {
        'demographics_gender': 'SEX',
        'demographics_age': 'BIRTH',
        'demographics_education': 'ESTUDIOS',
        'demographics_income': 'NAT_INC',
        'demographics_urban_rural': 'URBRURAL',
        'attitudes_env_concern': 'V15',
        'behavior_recycling': 'V52',
        'attitudes_wtp_prices': 'V27',
        'perception_safety': None,
        'perception_quality': None,
    }
    return mapping.get(dim)


def validate_regression_consistency(regression_priors, pooled_marginals):
    """
    Check coherence between marginal priors and regression coefficients.

    Prints warnings when the implied probability range across profiles is
    narrower than 5% (weak coefficients) or wider than 60% (inflated).
    Does not modify any data structures.
    """
    print("\nVALIDATING REGRESSION vs MARGINAL CONSISTENCY")

    def logit(p):
        return np.log(p / (1 - p + 1e-10))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for target, reg in regression_priors.items():
        if target not in pooled_marginals:
            print(f"  {target}: no marginal prior -- skip")
            continue

        marginal = pooled_marginals[target]
        alpha = np.array(marginal['alpha'])
        prop = alpha / alpha.sum()
        K = len(prop)
        threshold = K // 2
        p_baseline = float(prop[threshold:].sum())

        print(f"\n  {target}:")
        print(f"    Marginal baseline: {p_baseline:.1%}")

        base_logit = logit(p_baseline)

        max_adj = sum(
            max((lvl['coefficient'] for lvl in dim['levels'].values()), default=0)
            for dim in reg['predictors'].values()
            if dim['levels']
        )
        min_adj = sum(
            min((lvl['coefficient'] for lvl in dim['levels'].values()), default=0)
            for dim in reg['predictors'].values()
            if dim['levels']
        )

        p_max = sigmoid(base_logit + max_adj)
        p_min = sigmoid(base_logit + min_adj)
        spread = p_max - p_min

        print(f"    Max profile (all positive): {p_max:.1%}")
        print(f"    Min profile (all negative): {p_min:.1%}")
        print(f"    Spread: {spread:.1%}")

        if spread < 0.05:
            print(f"    WARNING: spread < 5% -- coefficients are very weak")
        elif spread > 0.60:
            print(f"    WARNING: spread > 60% -- coefficients may be inflated")
        else:
            print(f"    OK: spread is reasonable")


def extract_from_survey(study_id, filename, data, client):
    """
    Send one survey JSON to Claude for extraction.

    Returns a dict with keys 'study_id', 'extractions', and optionally 'error'.
    Retries up to 5 times on rate-limit errors with exponential back-off.
    """
    survey_json = json.dumps(data, indent=2, ensure_ascii=False)

    cis_list = '\n'.join(
        f"  - {name}: {info['type']}, K={info['K']}"
        for name, info in CIS_TARGETS.items()
    )

    prompt = EXTRACTION_PROMPT.format(
        cis_targets_list=cis_list,
        study_id=study_id,
        survey_json=survey_json,
    )

    print(f"  Extracting: {filename} ({len(prompt) // 1000}k chars)")

    for attempt in range(5):
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_json(study_id, msg.content[0].text)
        except Exception as e:
            if '429' in str(e) or 'rate_limit' in str(e).lower():
                wait = 60 * (2 ** attempt)
                print(f"  Rate limit -- waiting {wait}s (attempt {attempt + 1}/5)")
                time.sleep(wait)
            else:
                print(f"  Error: {e}")
                return {"study_id": study_id, "extractions": [], "error": str(e)}

    return {"study_id": study_id, "extractions": [], "error": "rate_limit_exceeded"}


def _parse_json(study_id, text):
    """Strip Markdown code fences and parse Claude's JSON response."""
    if "```json" in text:
        s = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        s = text.split("```")[1].split("```")[0].strip()
    else:
        s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return {"study_id": study_id, "extractions": [], "error": str(e)}


def _vals_summary(vals):
    """Return a compact summary string for a values dict (used in logging)."""
    if not vals:
        return ""
    parts = []
    for k in ['mean', 'std', 'n', 'r', 'coefficient']:
        if k in vals and vals[k] is not None:
            parts.append(f"{k}={vals[k]}")
    if 'counts' in vals:
        parts.append(f"counts={vals['counts'][:5]}")
    if 'proportions' in vals:
        parts.append(f"props={[round(p, 3) for p in vals['proportions'][:5]]}")
    if 'percentages' in vals:
        parts.append(f"pcts={vals['percentages'][:5]}")
    if 'groups' in vals:
        parts.append(f"groups={list(vals['groups'].keys())[:3]}")
    return ', '.join(parts)


def main():
    """
    Entry point for the harmonisation pipeline.

    Modes:
        default     Extract from all JSON files in METADATA_DIR, then compute.
        --recompute Read existing extractions_raw.json and recompute outputs.
    """
    import sys
    recompute_only = '--recompute' in sys.argv

    print("HARMONIZE CATEGORIES -> CIS SCALES")
    print("Claude extracts: marginals + regression_coefficient")
    print("                 + conditional_means + correlation + group_percentages")
    print("Python computes: Dirichlet + beta_from_conditional + meta-analysis")
    if recompute_only:
        print("RECOMPUTE MODE: using existing extractions_raw.json")

    if not recompute_only and not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY not set")
        return

    # Load or extract raw data
    if recompute_only:
        raw_path = "extractions_raw.json"
        if not os.path.exists(raw_path):
            print(f"{raw_path} not found")
            return
        with open(raw_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} surveys from {raw_path}")

    else:
        surveys = {}
        for fname in sorted(os.listdir(METADATA_DIR)):
            if not fname.endswith('.json') or fname == 'cis_2023.json':
                continue
            fpath = os.path.join(METADATA_DIR, fname)
            with open(fpath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            sid = (
                data.get('survey_metadata', {}).get('study_id') or
                data.get('study_metadata', {}).get('study_id') or
                data.get('review_metadata', {}).get('study_id') or
                fname.replace('.json', '')
            )
            surveys[sid] = {'filename': fname, 'filepath': fpath, 'data': data}

        print(f"Surveys found: {len(surveys)}")
        for sid, s in surveys.items():
            print(f"  {sid:<50s}  {os.path.getsize(s['filepath']) // 1024}KB")

        # Resume from checkpoint if available
        all_results = []
        done = set()
        partial_path = "extractions_partial.json"
        if os.path.exists(partial_path):
            with open(partial_path) as f:
                all_results = json.load(f)
            done = {r.get('study_id') for r in all_results}
            print(f"Resuming: {len(done)} studies already done")

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        for sid, survey in surveys.items():
            if sid in done:
                print(f"  done: {sid}")
                continue

            print(f"\n{sid}")
            result = extract_from_survey(sid, survey['filename'], survey['data'], client)
            all_results.append(result)

            n_ext = len(result.get('extractions', []))
            n_reg = sum(1 for e in result.get('extractions', [])
                        if e.get('data_type') == 'regression_coefficient')
            n_cond = sum(1 for e in result.get('extractions', [])
                         if e.get('data_type') in ('conditional_means', 'correlation', 'group_percentages'))
            n_skip = len(result.get('skipped_variables', []))
            print(f"  Extracted: {n_ext} ({n_reg} regression, {n_cond} conditional), "
                  f"Skipped: {n_skip}")

            for ext in result.get('extractions', []):
                dt = ext.get('data_type', '?')
                vals = ext.get('values') or {}
                if dt == 'regression_coefficient':
                    coef = vals.get('coefficient', '?')
                    sig = vals.get('significance', '')
                    print(f"    [COEF] {ext.get('cis_target', '?'):<35s} "
                          f"<- {ext.get('source_variable', '?'):<25s} b={coef} {sig}")
                elif dt in ('conditional_means', 'correlation', 'group_percentages'):
                    print(f"    [{dt.upper()[:5]}] {ext.get('cis_target', '?'):<35s} "
                          f"<- {ext.get('source_variable', '?'):<25s} {_vals_summary(vals)}")
                else:
                    print(f"    [DATA] {ext.get('cis_target', '?'):<35s} "
                          f"<- {ext.get('source_variable', '?'):<25s} "
                          f"[{dt}] {_vals_summary(vals)}")

            with open(partial_path, "w", encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"  Saved: {len(all_results)}/{len(surveys)}")

            remaining = len(surveys) - len(all_results)
            if remaining > 0:
                print("  Waiting 90s before next call...")
                time.sleep(90)

        with open("extractions_raw.json", "w", encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print("extractions_raw.json saved")

    # Compute Dirichlet alpha vectors from marginal extractions
    print("\nCOMPUTING DIRICHLET COUNTS (marginals)")

    all_mappings = []

    for result in all_results:
        sid = result.get('study_id', '?')
        mappings = []

        for ext in result.get('extractions', []):
            cis_target = ext.get('cis_target')
            if not cis_target or cis_target not in CIS_TARGETS:
                continue

            if is_excluded_from_marginal(sid, cis_target, ext.get('source_variable', '')):
                print(f"  EXCLUDED  {sid}: {cis_target} <- {ext.get('source_variable', '')}")
                continue

            cis_info = CIS_TARGETS[cis_target]
            alpha, method = compute_dirichlet(ext, cis_info)

            if alpha is None:
                if not method.startswith('skip:'):
                    print(f"  SKIP {sid}: {cis_target} "
                          f"<- {ext.get('source_variable', '?')}: ({method})")
                continue

            alpha = [max(round(a), 1) for a in alpha]
            ext_vals = ext.get('values') or ext.get('data') or {}
            total_n = ext_vals.get('n') or sum(alpha)

            mappings.append({
                'cis_target': cis_target,
                'cis_K': cis_info['K'],
                'source_variable': ext.get('source_variable'),
                'source_scale': ext.get('source_scale'),
                'data_type': ext.get('data_type'),
                'extracted_values': ext_vals,
                'dirichlet_counts': alpha,
                'total_n': total_n,
                'computation_method': method,
                'notes': ext.get('notes', ''),
            })

            alpha_str = str(alpha)
            print(f"  OK {sid}: {cis_target:<40s} "
                  f"alpha={alpha_str:<40s} [{method[:40]}]")

        all_mappings.append({
            'study_id': sid,
            'mappings': mappings,
            'skipped_variables': result.get('skipped_variables', []),
        })

    # Compute pooled regression coefficients
    regression_priors = compute_regression_priors(all_results)

    # Save outputs
    import pandas as pd

    with open("category_mappings.json", "w", encoding='utf-8') as f:
        json.dump(all_mappings, f, indent=2, ensure_ascii=False, default=str)
    print("category_mappings.json saved")

    reg_output = {
        'method': 'Fixed-effects meta-analysis: regression_coefficient + conditional->beta',
        'version': 'harmonize_categories_v6',
        'paper_section': 'Section 2.2 Stage 2b',
        'description': (
            'Regression coefficients (beta) per predictor per CIS target. '
            'Sources: explicit regression models, '
            'conditional_means->Cohen_d->beta, correlations->r_to_beta, '
            'group_percentages->log_OR->beta. '
            'Pooled via 1/SE^2 weighting when multiple studies report the same predictor. '
            'Used by step3 to construct P(acceptance | demographics).'
        ),
        'predictor_to_cis_mapping': PREDICTOR_TO_CIS,
        'variables': regression_priors,
    }
    with open("regression_priors.json", "w", encoding='utf-8') as f:
        json.dump(reg_output, f, indent=2, ensure_ascii=False, default=str)
    print("regression_priors.json saved (input for step3)")

    rows = []
    for sm in all_mappings:
        for m in sm.get('mappings', []):
            rows.append({
                'study_id': sm['study_id'],
                'cis_target': m['cis_target'],
                'cis_K': m['cis_K'],
                'source_variable': m['source_variable'],
                'source_scale': m['source_scale'],
                'data_type': m['data_type'],
                'extracted_values': json.dumps(m['extracted_values']),
                'dirichlet_counts': json.dumps(m['dirichlet_counts']),
                'total_n': m['total_n'],
                'computation_method': m['computation_method'],
                'notes': m['notes'],
            })
    pd.DataFrame(rows).to_csv("category_mappings_review.csv", index=False)
    print(f"category_mappings_review.csv saved ({len(rows)} mappings)")

    reg_rows = []
    for target, reg in regression_priors.items():
        for dim, pred in reg['predictors'].items():
            for level, coef_info in pred['levels'].items():
                reg_rows.append({
                    'cis_target': target,
                    'predictor_dimension': dim,
                    'cis_variable': pred.get('cis_variable', ''),
                    'level': level,
                    'reference_level': pred.get('reference_level', ''),
                    'coefficient': coef_info['coefficient'],
                    'std_error': coef_info.get('std_error', ''),
                    'significance': coef_info.get('significance', ''),
                    'n_studies': coef_info.get('n_studies', 1),
                    'n_observations': coef_info.get('n_observations', ''),
                    'studies': ', '.join(coef_info.get('studies', [])),
                    'model_type': reg.get('model_type', ''),
                    'derived_method': coef_info.get('derived_method', 'direct'),
                })
    if reg_rows:
        pd.DataFrame(reg_rows).to_csv("regression_priors_review.csv", index=False)
        print(f"regression_priors_review.csv saved ({len(reg_rows)} coefficients)")

    # Summary
    total_mappings = sum(len(m.get('mappings', [])) for m in all_mappings)
    n_reg_targets = len(regression_priors)
    n_reg_coefs = sum(
        sum(len(d['levels']) for d in r['predictors'].values())
        for r in regression_priors.values()
    )
    n_direct = sum(
        1 for r in regression_priors.values()
        for d in r['predictors'].values()
        for coef in d['levels'].values()
        if coef.get('derived_method') == 'direct_regression'
    )
    n_derived = n_reg_coefs - n_direct

    print("\nHARMONISATION COMPLETE")
    print(f"  Surveys processed:            {len(all_results)}")
    print(f"  Marginal mappings:            {total_mappings}")
    print(f"  Regression priors (targets):  {n_reg_targets}")
    print(f"  Total coefficients:           {n_reg_coefs}")
    print(f"    direct regression:          {n_direct}")
    print(f"    derived (cond/corr/pct):    {n_derived}")
    print(f"\n  Outputs:")
    print(f"    category_mappings.json       -> step2 (marginals)")
    print(f"    regression_priors.json       -> step3_cpt_conditional")
    print(f"    category_mappings_review.csv -> marginal review")
    if reg_rows:
        print(f"    regression_priors_review.csv -> coefficient review")
    print("\nNext: python step2_informed_v3.py")


if __name__ == "__main__":
    main()
