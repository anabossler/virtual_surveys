"""
app_gradio.py
ASAP v6 — Acceptance of recycled plastic: CPT-based inference engine.

Loads fused Bayesian network priors from step3_fused_priors.json and
computes posterior acceptance probabilities for 13 product categories.

Five primary indicators (food packaging, general acceptance, perceived
safety, perceived quality, WTP premium) use real CPTs estimated from
the full pipeline (CIS 3391 + 14 international studies).

Eight extended categories (non-food packaging, beverages, cleaning
products, personal care, toys, clothing, electronics, furniture) use
inline CPTs built from Ruokamo 2022 base rates scaled to Spain and
demographic betas from Athanasios 2022.

Uncertainty is quantified via Dirichlet posterior variance, consistent
with Martins et al. (2024) Algorithm 1, step 2:
    theta ~ Dir(alpha_ab)
    E[theta_k] = alpha_k / sum(alpha)
    Var[theta_k] = alpha_k * (A - alpha_k) / (A^2 * (A + 1))

Usage:
    python encuesta_bayesiana.py          # runs CLI differentiation test + Gradio UI
"""

import os
import sys
import json
import re
import warnings
import base64
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit as sigmoid

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent

# File paths
PRIORS_PATH = BASE / "step3_fused_priors.json"
META_PATH   = BASE / "step4_output" / "metadata.json"
S1_PATH     = BASE / "step4_output" / "synthetic_surveys_S1.csv"
LOGO_PATH   = BASE / "logo.png"

# Load fused priors and conditional probability tables
PRIORS    = {}
CPT_CACHE = {}
N_STUDIES = 14
N_VARS    = 0
N_CIS     = 2254

if PRIORS_PATH.exists():
    with open(PRIORS_PATH, 'r', encoding='utf-8') as f:
        _data = json.load(f)
    PRIORS = _data.get('variables', {})
    for vname, vdata in PRIORS.items():
        if 'conditional_cpt' in vdata:
            CPT_CACHE[vname] = vdata['conditional_cpt']
    print(f"Loaded step3_fused_priors.json: {len(PRIORS)} variables, "
          f"{len(CPT_CACHE)} CPTs")
    for vname, cpt in CPT_CACHE.items():
        print(f"  {vname}: {cpt.get('n_configs', 0)} configs "
              f"p=[{cpt.get('p_min', 0):.1%}, {cpt.get('p_max', 0):.1%}]")
else:
    print(f"WARNING: {PRIORS_PATH} not found — run step3 first")

if META_PATH.exists():
    with open(META_PATH) as f:
        _meta = json.load(f)
    N_CIS  = _meta.get('n_cis', 2254)
    N_VARS = _meta.get('n_variables', 0)

DF_S1 = None
if S1_PATH.exists():
    DF_S1 = pd.read_csv(S1_PATH)
    if N_VARS == 0:
        N_VARS = len([c for c in DF_S1.columns
                      if c not in ('_draw', 'gender', 'age_group')])


# Mappings from UI labels to CPT level names
AGE_MAP = {
    '18-29': 'under_30',
    '30-44': '30_44',
    '45-59': '45_59',
    '60-74': '60_plus',
    '75+':   '60_plus',
}

GENDER_MAP = {
    'F':  'female',
    'M':  'male',
    'NB': 'other',
}

EDU_MAP = {
    'none':       'no_studies',
    'primary':    'primary',
    'secondary':  'secondary',
    'vocational': 'secondary',
    'university': 'university',
    'postgrad':   'postgraduate',
}

INCOME_MAP = {
    'very_low':  'very_low',
    'low':       'low',
    'middle':    'middle',
    'high':      'high',
    'very_high': 'very_high',
}

ENV_MAP = {
    'low':    'low',
    'medium': 'medium',
    'high':   'very_high',
}

RECYCLING_MAP = {
    'always':        'always',
    'nearly_always': 'nearly_always',
    'sometimes':     'sometimes',
    'rarely':        'rarely',
    'never':         'never',
}

WTP_PRICES_MAP = {
    'strongly_disagree': 'strongly_disagree',
    'disagree':          'disagree',
    'neutral':           'neutral',
    'agree':             'agree',
    'strongly_agree':    'strongly_agree',
}

# Default profile values used when NLP extraction leaves a field empty
DEFAULTS = {
    'age':       '30-44',
    'gender':    'F',
    'edu':       'secondary',
    'income':    'middle',
    'product':   'general',
    'env':       'medium',
    'urban':     'urban',
    'recycling': 'sometimes',
    'wtp_prices':'neutral',
}

# Functions that read a profile dict and return the CPT-level name for each parent
PARENT_LEVEL_GETTERS = {
    'demographics_gender':    lambda p: GENDER_MAP.get(p['gender'], 'female'),
    'demographics_education': lambda p: EDU_MAP.get(p['edu'], 'secondary'),
    'demographics_age':       lambda p: AGE_MAP.get(p['age'], '30_44'),
    'demographics_income':    lambda p: INCOME_MAP.get(p['income'], 'middle'),
    'behavior_recycling':     lambda p: RECYCLING_MAP.get(p['recycling'], 'sometimes'),
    'attitudes_env_concern':  lambda p: ENV_MAP.get(p['env'], 'medium'),
    'attitudes_wtp_prices':   lambda p: WTP_PRICES_MAP.get(p['wtp_prices'], 'neutral'),
}


def build_config_key(profile_dict, parent_variables, parent_levels):
    """
    Build the CPT lookup key for a given profile.

    Parameters
    ----------
    profile_dict : dict
        Keys: age, gender, edu, income, env, recycling, wtp_prices.
    parent_variables : list of str
        Harmonised parent names in CPT order.
    parent_levels : dict
        Mapping from parent name to list of valid level strings.

    Returns
    -------
    str or None
        Pipe-separated level string, e.g. 'female|university|under_30',
        or None if any parent cannot be resolved.
    """
    parts = []
    for parent_harm in parent_variables:
        getter = PARENT_LEVEL_GETTERS.get(parent_harm)
        if getter is None:
            return None
        level = getter(profile_dict)
        valid = parent_levels.get(parent_harm, [])
        if level not in valid:
            level = valid[len(valid) // 2] if valid else None
        if level is None:
            return None
        parts.append(level)
    return '|'.join(parts)


def _build_inline_cpt(base_rate, scale, positive_cats=None, parents=None,
                      ess=60.0, extra_coefs=None):
    """
    Build an inline CPT for extended product categories.

    Base rates come from Ruokamo 2022 (Finland, n=301) scaled by 0.85
    to approximate Spanish acceptance levels (Ibrahim & Chen 2000).
    Demographic beta coefficients are derived from the same source and
    attenuated by a perceived-risk scale factor following Athanasios 2022.

    Parameters
    ----------
    base_rate : float
        Prior acceptance probability for a reference profile.
    scale : float
        Perceived-risk attenuation factor (1.0 = food packaging risk level).
    positive_cats : list of int, optional
        Category indices counted as acceptance. Defaults to [3, 4].
    parents : list of str, optional
        Parent variable names. Defaults to six demographic/attitudinal parents.
    ess : float
        Effective sample size used to set Dirichlet concentration.
    extra_coefs : dict, optional
        Override or extend any parent coefficient dict.

    Returns
    -------
    dict
        CPT structure compatible with _compute_from_cpt_dict.
    """
    if positive_cats is None:
        positive_cats = [3, 4]
    if parents is None:
        parents = ['demographics_gender', 'demographics_education',
                   'demographics_age', 'demographics_income',
                   'behavior_recycling', 'attitudes_env_concern']

    ATT = 0.8  # cultural attenuation factor Finland -> Spain

    base_coefs = {
        'demographics_gender': {
            'male':   0.0,
            'female': round(0.463 * ATT * scale, 4),
            'other':  round(0.463 * ATT * scale / 2, 4),
        },
        'demographics_education': {
            'no_studies':  round(-0.200 * ATT * scale, 4),
            'primary':     0.0,
            'secondary':   round(0.572 * ATT * scale, 4),
            'university':  round(0.707 * ATT * scale, 4),
            'postgraduate':round(0.707 * ATT * scale, 4),
        },
        'demographics_age': {
            'under_30': round(0.468 * ATT * 0.7 * scale, 4),
            '30_44':    0.0,
            '45_59':    0.0,
            '60_plus':  round(-0.162 * ATT * 0.7 * scale, 4),
        },
        'demographics_income': {
            'very_low': round(-0.15 * scale, 4),
            'low':      round(-0.07 * scale, 4),
            'middle':   0.0,
            'high':     round(0.10 * scale, 4),
            'very_high':round(0.15 * scale, 4),
        },
        'behavior_recycling': {
            'always':        round(0.352 * ATT * scale, 4),
            'nearly_always': round(0.352 * ATT * 0.75 * scale, 4),
            'sometimes':     0.0,
            'rarely':        round(-0.352 * ATT * 0.5 * scale, 4),
            'never':         round(-0.352 * ATT * scale, 4),
        },
        'attitudes_env_concern': {
            'very_low': round(-0.761 * scale, 4),
            'low':      round(-0.380 * scale, 4),
            'medium':   0.0,
            'high':     round(0.380 * scale, 4),
            'very_high':round(0.761 * scale, 4),
        },
        'attitudes_wtp_prices': {
            'strongly_disagree': round(-0.50 * scale, 4),
            'disagree':          round(-0.25 * scale, 4),
            'neutral':           0.0,
            'agree':             round(0.30 * scale, 4),
            'strongly_agree':    round(0.55 * scale, 4),
        },
    }
    if extra_coefs:
        for k, v in extra_coefs.items():
            base_coefs[k] = v

    parent_levels = {
        'demographics_gender':    ['male', 'female', 'other'],
        'demographics_education': ['no_studies', 'primary', 'secondary',
                                   'university', 'postgraduate'],
        'demographics_age':       ['under_30', '30_44', '45_59', '60_plus'],
        'demographics_income':    ['very_low', 'low', 'middle', 'high', 'very_high'],
        'behavior_recycling':     ['always', 'nearly_always', 'sometimes',
                                   'rarely', 'never'],
        'attitudes_env_concern':  ['very_low', 'low', 'medium', 'high', 'very_high'],
        'attitudes_wtp_prices':   ['strongly_disagree', 'disagree', 'neutral',
                                   'agree', 'strongly_agree'],
    }

    K = 5
    base_logit = float(np.log(base_rate / (1 - base_rate)))
    n_pos = len(positive_cats)
    n_neg = K - n_pos

    def p_to_alpha(p):
        props = np.ones(K) / K
        for i in positive_cats:
            props[i] = p / n_pos
        for i in range(K):
            if i not in positive_cats:
                props[i] = (1 - p) / max(n_neg, 1)
        props = np.maximum(props, 1e-6)
        props /= props.sum()
        return ess * props

    alpha_per_config = {}
    level_lists = [parent_levels[p] for p in parents]
    p_vals = []
    for config in itertools.product(*level_lists):
        delta = sum(base_coefs.get(par, {}).get(lv, 0.0)
                    for par, lv in zip(parents, config))
        p_ab = float(np.clip(sigmoid(base_logit + delta), 0.02, 0.98))
        alpha_per_config['|'.join(config)] = p_to_alpha(p_ab).tolist()
        p_vals.append(p_ab)

    return {
        'parent_variables':        parents,
        'parent_levels':           {p: parent_levels[p] for p in parents},
        'alpha_per_parent_config': alpha_per_config,
        'positive_categories':     positive_cats,
        'base_rate':               base_rate,
        'n_configs':               len(alpha_per_config),
        'p_min':                   float(np.min(p_vals)),
        'p_max':                   float(np.max(p_vals)),
        'inline':                  True,
    }


# Build inline CPTs for extended categories
# scale values follow the perceived-risk taxonomy of Athanasios 2022
print("Building inline CPTs for extended categories...")
_INLINE_CPTS = {}

_INLINE_CPTS['non_food_packaging'] = _build_inline_cpt(0.625 * 0.85, scale=0.7)
_INLINE_CPTS['beverages']          = _build_inline_cpt(0.45,          scale=0.65)
_INLINE_CPTS['cleaning']           = _build_inline_cpt(0.57 * 0.85,   scale=0.7)
_INLINE_CPTS['personal_care']      = _build_inline_cpt(
    0.40, scale=0.5,
    parents=['demographics_gender', 'demographics_education',
             'demographics_age', 'attitudes_env_concern'],
    extra_coefs={'attitudes_env_concern': {
        'very_low': -0.30, 'low': -0.15, 'medium': 0.0,
        'high': +0.15,     'very_high': +0.25,
    }},
)
_INLINE_CPTS['toys']        = _build_inline_cpt(0.31 * 0.85, scale=0.4)
_INLINE_CPTS['clothing']    = _build_inline_cpt(0.25 * 0.85, scale=0.5)
_INLINE_CPTS['electronics'] = _build_inline_cpt(
    0.24 * 0.85, scale=0.5,
    extra_coefs={'demographics_income': {
        'very_low': -0.18, 'low': -0.09, 'middle': 0.0,
        'high': +0.18,     'very_high': +0.30,
    }},
)
_INLINE_CPTS['furniture'] = _build_inline_cpt(0.22 * 0.85, scale=0.5)

for k, cpt in _INLINE_CPTS.items():
    print(f"  {k:20s}: {cpt['n_configs']:5d} configs "
          f"p=[{cpt['p_min']:.1%}, {cpt['p_max']:.1%}]")


# Product target registry
# 'tier' distinguishes pipeline-derived priors (primary) from
# inline extended estimates (extended)
TARGETS = {
    'food_packaging': {
        'cpt_var': 'acceptance_food_packaging',
        'label':   'Food packaging',
        'base':    0.38,
        'tier':    'primary',
    },
    'general': {
        'cpt_var': 'acceptance_general',
        'label':   'General acceptance',
        'base':    0.55,
        'tier':    'primary',
    },
    'perception_safety': {
        'cpt_var': 'perception_safety',
        'label':   'Perceived safety',
        'base':    0.50,
        'tier':    'primary',
    },
    'perception_quality': {
        'cpt_var': 'perception_quality',
        'label':   'Perceived quality',
        'base':    0.45,
        'tier':    'primary',
    },
    'wtp_premium': {
        'cpt_var': 'price_willingness_to_pay_premium',
        'label':   'WTP price premium',
        'base':    0.35,
        'tier':    'primary',
    },
    'non_food_packaging': {
        'inline_key': 'non_food_packaging',
        'label':      'Non-food packaging',
        'base':       round(0.625 * 0.85, 3),
        'tier':       'extended',
    },
    'beverages': {
        'inline_key': 'beverages',
        'label':      'Beverage bottles',
        'base':       0.45,
        'tier':       'extended',
    },
    'cleaning': {
        'inline_key': 'cleaning',
        'label':      'Cleaning products',
        'base':       round(0.57 * 0.85, 3),
        'tier':       'extended',
    },
    'personal_care': {
        'inline_key': 'personal_care',
        'label':      'Personal care / cosmetics',
        'base':       0.40,
        'tier':       'extended',
    },
    'toys': {
        'inline_key': 'toys',
        'label':      'Toys',
        'base':       round(0.31 * 0.85, 3),
        'tier':       'extended',
    },
    'clothing': {
        'inline_key': 'clothing',
        'label':      'Clothing / textiles',
        'base':       round(0.25 * 0.85, 3),
        'tier':       'extended',
    },
    'electronics': {
        'inline_key': 'electronics',
        'label':      'Electronics / appliances',
        'base':       round(0.24 * 0.85, 3),
        'tier':       'extended',
    },
    'furniture': {
        'inline_key': 'furniture',
        'label':      'Furniture',
        'base':       round(0.22 * 0.85, 3),
        'tier':       'extended',
    },
    # Aliases for NLP resolution
    'cosmetics':         {'alias': 'personal_care'},
    'appliances':        {'alias': 'electronics'},
    'general_packaging': {'alias': 'non_food_packaging'},
    'shampoo':           {'alias': 'personal_care'},
    'bottle':            {'alias': 'beverages'},
    'water_bottle':      {'alias': 'beverages'},
}


def _compute_from_cpt_dict(profile_dict, cpt, label='', inline=False):
    """
    Compute P(positive) from a CPT dict using Dirichlet posterior mean.

    Falls back to base_rate with a fixed standard error when the
    config key is not found in the table.

    Returns
    -------
    tuple : (central, lo, hi, debug_dict)
    """
    parent_variables = cpt['parent_variables']
    parent_levels    = cpt['parent_levels']
    alpha_per_config = cpt['alpha_per_parent_config']
    positive_cats    = cpt.get('positive_categories', [3, 4])

    config_key = build_config_key(profile_dict, parent_variables, parent_levels)

    if config_key is None or config_key not in alpha_per_config:
        base = cpt.get('base_rate', 0.5)
        se   = 0.07 if inline else 0.06
        debug = {'method': 'fallback base_rate', 'label': label,
                 'config_key': config_key, 'inline': inline}
        return (base,
                float(np.clip(base - 1.96 * se, 0, 1)),
                float(np.clip(base + 1.96 * se, 0, 1)),
                debug)

    alpha = np.maximum(np.array(alpha_per_config[config_key], dtype=float), 1e-6)
    A     = alpha.sum()
    theta = alpha / A

    p_pos = float(sum(theta[i] for i in positive_cats if i < len(theta)))

    # Dirichlet variance for a sum of categories
    var_pos = 0.0
    for i in positive_cats:
        if i < len(alpha):
            var_pos += alpha[i] * (A - alpha[i]) / (A ** 2 * (A + 1))
    for idx_i, i in enumerate(positive_cats):
        for j in positive_cats[idx_i + 1:]:
            if i < len(alpha) and j < len(alpha):
                var_pos += 2 * (-alpha[i] * alpha[j]) / (A ** 2 * (A + 1))
    var_pos = max(var_pos, 1e-8)
    if inline:
        var_pos = max(var_pos, 0.002)  # extended CPTs carry additional uncertainty
    se = np.sqrt(var_pos)

    debug = {
        'method':             'inline CPT (Ruokamo+Athanasios)' if inline else 'CPT step3 (Bayesian)',
        'label':              label,
        'config_key':         config_key,
        'alpha_sum':          round(float(A), 1),
        'K':                  len(alpha),
        'positive_categories':positive_cats,
        'p_range':            f"[{cpt.get('p_min', 0):.1%}, {cpt.get('p_max', 0):.1%}]",
        'n_configs':          cpt.get('n_configs', 0),
        'inline':             inline,
    }
    return (p_pos,
            float(np.clip(p_pos - 1.96 * se, 0, 1)),
            float(np.clip(p_pos + 1.96 * se, 0, 1)),
            debug)


def compute_from_cpt(profile_dict, cpt_var, inline_key=None):
    """
    Route a computation to either the pipeline CPT or an inline CPT.

    Returns
    -------
    tuple : (central, lo, hi, debug_dict)
    """
    if inline_key is not None:
        cpt = _INLINE_CPTS.get(inline_key)
        if cpt is not None:
            return _compute_from_cpt_dict(profile_dict, cpt,
                                          label=inline_key, inline=True)

    cpt = CPT_CACHE.get(cpt_var)
    if cpt is None:
        return _fallback_from_marginal(cpt_var, profile_dict)
    return _compute_from_cpt_dict(profile_dict, cpt, label=cpt_var, inline=False)


def _fallback_from_marginal(cpt_var, profile_dict, config_key=None, note=''):
    """Use the marginal alpha_lit when no CPT is available for a variable."""
    vdata = PRIORS.get(cpt_var, {})
    alpha = np.array(
        vdata.get('alpha_lit', vdata.get('alpha_posterior', [])),
        dtype=float,
    )
    base_rate = 0.5
    for tinfo in TARGETS.values():
        if tinfo.get('cpt_var') == cpt_var:
            base_rate = tinfo.get('base', 0.5)
            break

    if len(alpha) >= 2:
        alpha = np.maximum(alpha, 1e-6)
        theta = alpha / alpha.sum()
        positive_cats = vdata.get('positive_categories', [3, 4])
        p_pos = float(sum(theta[i] for i in positive_cats if i < len(theta)))
    else:
        p_pos = base_rate

    se = 0.06
    debug = {
        'method':     f'marginal fallback ({note or "no CPT available"})',
        'cpt_var':    cpt_var,
        'config_key': config_key,
        'alpha_sum':  float(alpha.sum()) if len(alpha) > 0 else 0,
    }
    return (p_pos,
            float(np.clip(p_pos - 1.96 * se, 0, 1)),
            float(np.clip(p_pos + 1.96 * se, 0, 1)),
            debug)


def compute(age, gender, edu, income, product, env, urban,
            recycling='sometimes', wtp_prices='neutral'):
    """
    Main inference entry point.

    Resolves product aliases, selects the appropriate CPT, and returns
    the posterior acceptance probability with a 95% credibility interval.

    Parameters
    ----------
    age, gender, edu, income, product, env, urban : str
        Profile fields using UI-level labels (see DEFAULTS for valid values).
    recycling, wtp_prices : str
        Behavioural/attitudinal fields.

    Returns
    -------
    tuple : (central, lo, hi, debug_dict)
    """
    profile = {
        'age':       age,
        'gender':    gender,
        'edu':       edu,
        'income':    income,
        'env':       env,
        'urban':     urban,
        'recycling': recycling,
        'wtp_prices':wtp_prices,
    }

    tinfo = TARGETS.get(product, TARGETS['general'])
    if 'alias' in tinfo:
        tinfo = TARGETS[tinfo['alias']]

    cpt_var    = tinfo.get('cpt_var')
    inline_key = tinfo.get('inline_key')

    c, lo, hi, debug = compute_from_cpt(profile, cpt_var, inline_key=inline_key)
    debug['product']       = product
    debug['product_label'] = tinfo['label']
    debug['base_rate']     = tinfo['base']
    debug['tier']          = tinfo.get('tier', 'primary')
    debug['profile']       = profile

    return c, lo, hi, debug


def compute_all_targets(age, gender, edu, income, env, urban,
                        recycling='sometimes', wtp_prices='neutral'):
    """Compute all canonical targets for a single profile."""
    results = {}
    canonical = [k for k, v in TARGETS.items() if 'alias' not in v]
    for product in canonical:
        c, lo, hi, debug = compute(age, gender, edu, income, product,
                                   env, urban, recycling, wtp_prices)
        results[product] = {
            'central': c, 'lo': lo, 'hi': hi,
            'label':   TARGETS[product]['label'],
            'debug':   debug,
        }
    return results


def test_differentiation():
    """
    CLI sanity check: print acceptance probabilities for contrasting profiles.

    A spread below 10% across profiles for any target suggests the CPT
    coefficients may be too weak. Below 5% means no effective differentiation.
    """
    print("\nDIFFERENTIATION TEST")

    profiles = [
        ("18-29", "F", "university", "high",    "high",   "urban",  "always",   "agree"),
        ("18-29", "F", "primary",    "low",      "low",    "rural",  "never",    "disagree"),
        ("60-74", "M", "primary",    "very_low", "low",    "rural",  "never",    "strongly_disagree"),
        ("60-74", "M", "university", "high",     "high",   "urban",  "always",   "strongly_agree"),
        ("30-44", "F", "secondary",  "middle",   "medium", "urban",  "sometimes","neutral"),
        ("30-44", "M", "secondary",  "middle",   "medium", "urban",  "sometimes","neutral"),
    ]

    targets = ['food_packaging', 'general', 'perception_safety',
               'perception_quality', 'wtp_premium']

    for target in targets:
        print(f"\n  TARGET: {target}")
        vals = []
        for age, gen, edu, inc, env, urb, rec, wtp in profiles:
            c, lo, hi, debug = compute(age, gen, edu, inc, target,
                                       env, urb, rec, wtp)
            key    = debug.get('config_key', '?')
            method = debug.get('method', '?')
            label  = f"{gen} {age:5s} {edu:11s} {inc:8s} {env:6s} {rec:13s}"
            print(f"    {label} -> {c:.1%} [{lo:.1%}, {hi:.1%}]  key={key}  ({method})")
            vals.append(c)
        spread = max(vals) - min(vals)
        status = ("OK" if spread > 0.10
                  else "LOW" if spread > 0.05
                  else "NO DIFFERENTIATION")
        print(f"    Spread: {spread:.1%}  [{status}]")


# NLP layer (Anthropic API — optional)
try:
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    HAS_API = bool(os.getenv("ANTHROPIC_API_KEY", ""))
except ImportError:
    HAS_API = False

LABELS = {
    "age":     {"18-29": "18-29", "30-44": "30-44", "45-59": "45-59",
                "60-74": "60-74", "75+": "75+"},
    "gender":  {"F": "Female", "M": "Male", "NB": "Other"},
    "edu":     {"none": "No studies", "primary": "Primary",
                "secondary": "Secondary", "vocational": "Vocational",
                "university": "University", "postgrad": "Postgraduate"},
    "income":  {"very_low": "Very low", "low": "Low", "middle": "Middle",
                "high": "High", "very_high": "Very high"},
    "product": {
        "food_packaging":    "Food packaging",
        "general":           "General acceptance",
        "perception_safety": "Perceived safety",
        "perception_quality":"Perceived quality",
        "wtp_premium":       "WTP price premium",
        "non_food_packaging":"Non-food packaging",
        "beverages":         "Beverage bottles",
        "cleaning":          "Cleaning products",
        "personal_care":     "Personal care / cosmetics",
        "toys":              "Toys",
        "clothing":          "Clothing / textiles",
        "electronics":       "Electronics / appliances",
        "furniture":         "Furniture",
    },
    "env":      {"low": "Low", "medium": "Moderate", "high": "High"},
    "urban":    {"rural": "Rural", "suburban": "Suburban", "urban": "Urban"},
    "recycling":{"always": "Always recycles", "nearly_always": "Nearly always",
                 "sometimes": "Sometimes", "rarely": "Rarely",
                 "never": "Never recycles"},
    "wtp_prices":{"strongly_disagree": "Strongly disagree to pay more",
                  "disagree":          "Disagree to pay more",
                  "neutral":           "Neutral",
                  "agree":             "Agree to pay more",
                  "strongly_agree":    "Strongly agree to pay more"},
}

# System prompt for profile extraction (sent to Claude)
PARSE_SYS = """Extract consumer profile JSON from a question about recycled plastic acceptance.
Output ONLY valid JSON, no markdown.
age: 18-29 or teens->"18-29" | 30-44 or thirties/forties->"30-44" | 45-59->"45-59" | 60-74 or senior->"60-74" | 75+->"75+"
gender: woman/female->"F"|man/male->"M"|other->"NB"
edu: none->"none"|primary->"primary"|secondary->"secondary"|vocational->"vocational"|university->"university"|postgrad->"postgrad"
income: very low->"very_low"|low->"low"|middle->"middle"|high->"high"|very high->"very_high"
product: food/alimentary->"food_packaging"|non_food/general_packaging->"non_food_packaging"|beverage/bottle->"beverages"|cleaning/detergent->"cleaning"|cosmetic/personal_care/shampoo->"personal_care"|toy/infant->"toys"|clothing/textile->"clothing"|electronics/appliance->"electronics"|furniture->"furniture"|safety/risk->"perception_safety"|quality->"perception_quality"|premium/wtp->"wtp_premium"|else->"general"
env_concern: low->"low"|moderate->"medium"|high->"high"
urban: rural->"rural"|suburban->"suburban"|city/urban->"urban"
recycling: always->"always"|nearly always->"nearly_always"|sometimes->"sometimes"|rarely->"rarely"|never->"never"
wtp_prices: strongly disagree->"strongly_disagree"|disagree->"disagree"|neutral->"neutral"|agree->"agree"|strongly agree->"strongly_agree"
If comparison requested: is_comparison=true, compare_products=[list of product keys]
Null for fields not mentioned.
{"age":null,"gender":null,"edu":null,"income":null,"product":null,"env_concern":null,"urban":null,"recycling":null,"wtp_prices":null,"is_comparison":false,"compare_products":[],"confidence":"high"}"""

# System prompt for narrative generation
_NARR_SYS = (
    f"Analyst for a recycled plastic acceptance study (University of Alicante).\n"
    f"Audience: brand managers, policy makers, non-technical.\n"
    f"Data: CIS 3391 (n={N_CIS:,}) + {N_STUDIES} international studies "
    f"via Bayesian network, {N_VARS} variables.\n"
    f"Rules:\n"
    f"- Same language as the user\n"
    f"- Max 3 sentences\n"
    f"- Do not use technical terms like 'Bayesian' or 'Dirichlet'\n"
    f"- Use ONLY the numbers provided, do not invent\n"
    f"- State the probability, the main driver, and one actionable insight\n"
    f"- For comparisons: highlight the biggest gap and its implication\n"
    f"- End with a brief source note mentioning CIS 3391 and international studies"
)


def _client():
    """Return an Anthropic client using the API key from the environment."""
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))


def parse_question(text):
    """
    Extract a structured profile dict from a free-text question via Claude.

    Returns a dict with keys: age, gender, edu, income, product, env_concern,
    urban, recycling, wtp_prices, is_comparison, compare_products, confidence.
    Unknown fields are set to None.
    """
    r = _client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=PARSE_SYS,
        messages=[{"role": "user", "content": text}],
    )
    raw = re.sub(r"```(?:json)?", "", r.content[0].text.strip()).strip().strip("`")
    m   = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        return ({k: None for k in ["age", "gender", "edu", "income", "product",
                                   "env_concern", "urban", "recycling", "wtp_prices"]}
                | {"is_comparison": False, "compare_products": [], "confidence": "low"})


def narrative(data, question):
    """Generate a plain-language interpretation of computed results via Claude."""
    r = _client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=_NARR_SYS,
        messages=[{"role": "user",
                   "content": f"Q: {question}\nData:\n{json.dumps(data, ensure_ascii=False)}"}],
    )
    return r.content[0].text.strip()


# HTML card rendering utilities
def _pct(x):  return f"{x:.0%}"
def _pct1(x): return f"{x:.1%}"


def _cc(v):
    """Map a probability to a colour class key."""
    return "high" if v >= 0.65 else ("medium" if v >= 0.45 else "low")


CC = {
    "high":   {"bg": "#f0fdf4", "border": "#bbf7d0", "bbg": "#dcfce7",
               "bfg": "#166534", "badge": "High",     "bar": "#22c55e"},
    "medium": {"bg": "#fffbeb", "border": "#fde68a", "bbg": "#fef9c3",
               "bfg": "#854d0e", "badge": "Moderate", "bar": "#f59e0b"},
    "low":    {"bg": "#fff1f2", "border": "#fecdd3", "bbg": "#ffe4e6",
               "bfg": "#9f1239", "badge": "Low",      "bar": "#f43f5e"},
}


def _src():
    """Build the data-source attribution string."""
    parts = [f"CIS 3391 (n={N_CIS:,})", f"{N_STUDIES} international studies"]
    if DF_S1 is not None:
        parts.append(f"{len(DF_S1):,} virtual surveys")
    return " | ".join(parts)


def _method_badge(debug):
    """Return a small HTML badge describing the computation method."""
    m = debug.get('method', '')
    if 'CPT' in m or 'Dirichlet' in m or 'Bayesian' in m:
        return ('<span style="background:#dbeafe;color:#1e40af;font-size:9px;'
                'padding:2px 7px;border-radius:5px;">Bayesian network + literature</span>')
    if 'marginal' in m:
        return ('<span style="background:#fef3c7;color:#92400e;font-size:9px;'
                'padding:2px 7px;border-radius:5px;">Marginal prior</span>')
    return ('<span style="background:#f1f5f9;color:#64748b;font-size:9px;'
            'padding:2px 7px;border-radius:5px;">Base estimate</span>')


def _profile_str(age, gender, edu, income, env, urban,
                 recycling='sometimes', wtp_prices='neutral'):
    parts = [
        LABELS['gender'].get(gender, gender),
        LABELS['age'].get(age, age),
        LABELS['edu'].get(edu, edu),
        LABELS['income'].get(income, income),
        LABELS['urban'].get(urban, urban),
        LABELS['env'].get(env, env) + " env. concern",
    ]
    if recycling != 'sometimes':
        parts.append(LABELS['recycling'].get(recycling, recycling))
    return ", ".join(parts)


def card_single(central, lo, hi, debug, profile_str, product_label):
    """Render a single-target result as an HTML card."""
    c   = CC[_cc(central)]
    pct = round(central * 100)
    key = debug.get('config_key', 'N/A')
    alpha_sum = debug.get('alpha_sum', '?')
    p_range   = debug.get('p_range', '')
    debug_line = f"CPT key: {key} | ESS={alpha_sum} | {p_range} | {debug.get('method', '')}"

    tier_badge = (
        '<div style="background:#fef3c7;border:1px solid #fbbf24;'
        'border-radius:6px;padding:6px 10px;margin-bottom:10px;'
        'font-size:10px;color:#92400e;line-height:1.4;">'
        '&#9888; <strong>Extended estimate</strong> &mdash; '
        'Base rate: Ruokamo 2022 (n=301, Finland x0.85). '
        'Demographic effects derived from Athanasios 2022. '
        'Greater uncertainty than primary indicators.</div>'
        if debug.get('tier') == 'extended' else ''
    )

    return f"""<div style="font-family:'DM Sans',sans-serif;max-width:620px;">
{tier_badge}<div style="background:{c['bg']};border:1px solid {c['border']};
     border-radius:14px;padding:22px 24px;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:start;">
    <div>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:4px;">
        <span style="font-size:9px;text-transform:uppercase;letter-spacing:.1em;
              color:#94a3b8;">Acceptance probability</span>
        {_method_badge(debug)}
      </div>
      <div style="font-size:52px;font-weight:200;color:#0f172a;line-height:1;">{_pct(central)}</div>
      <div style="font-size:11px;color:#94a3b8;margin-top:4px;">95% CI: {_pct1(lo)} – {_pct1(hi)}</div>
    </div>
    <div style="background:{c['bbg']};color:{c['bfg']};font-size:10px;
                font-weight:600;padding:5px 12px;border-radius:99px;
                border:1px solid {c['border']};">{c['badge']}</div>
  </div>
  <div style="background:#e2e8f0;border-radius:99px;height:7px;width:100%;margin:10px 0 0;">
    <div style="background:{c['bar']};border-radius:99px;height:7px;width:{pct}%;"></div>
  </div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;">
  <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;">
    <div style="font-size:9px;text-transform:uppercase;letter-spacing:.08em;
          color:#94a3b8;margin-bottom:4px;">Profile</div>
    <div style="font-size:11px;color:#334155;line-height:1.6;">{profile_str}</div>
  </div>
  <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;">
    <div style="font-size:9px;text-transform:uppercase;letter-spacing:.08em;
          color:#94a3b8;margin-bottom:4px;">Outcome</div>
    <div style="font-size:12px;font-weight:600;color:#0f172a;">{product_label}</div>
    <div style="font-size:10px;color:#94a3b8;margin-top:3px;">CPT range: {p_range}</div>
  </div>
</div>
<div style="font-size:9px;color:#b0b8c4;text-align:right;margin-bottom:3px;">{_src()}</div>
<div style="font-size:8px;color:#cbd5e1;text-align:right;">{debug_line}</div>
</div>"""


def card_all_targets(results, profile_str):
    """Render all targets for a profile as a ranked HTML card."""
    rows = ""
    for product, r in sorted(results.items(), key=lambda x: -x[1]['central']):
        c   = r['central']
        lo  = r['lo']
        hi  = r['hi']
        lbl = r['label']
        cc  = CC[_cc(c)]
        pct = round(c * 100)
        rows += f"""<div style="padding:11px 0;border-bottom:1px solid #f1f5f9;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;">
    <span style="font-size:12px;font-weight:500;color:#1e293b;">{lbl}</span>
    <span style="font-size:24px;font-weight:200;color:#0f172a;">{_pct(c)}</span>
  </div>
  <div style="background:#e2e8f0;border-radius:99px;height:5px;width:100%;margin:5px 0 3px;">
    <div style="background:{cc['bar']};border-radius:99px;height:5px;width:{pct}%;"></div>
  </div>
  <div style="font-size:10px;color:#94a3b8;">95% CI: {_pct1(lo)} – {_pct1(hi)}</div>
</div>"""

    return f"""<div style="font-family:'DM Sans',sans-serif;max-width:620px;">
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;
     padding:22px 24px;margin-bottom:10px;">
  <div style="font-size:9px;text-transform:uppercase;letter-spacing:.1em;
        color:#94a3b8;margin-bottom:4px;">All indicators</div>
  <div style="font-size:11px;color:#64748b;margin-bottom:14px;">{profile_str}</div>
  {rows}
</div>
<div style="font-size:9px;color:#b0b8c4;text-align:right;">{_src()}</div>
</div>"""


def card_comparison(products, cs, ls, hs, profile_str):
    """Render a product comparison as an HTML card."""
    rows = ""
    for p, c, lo, hi in sorted(zip(products, cs, ls, hs), key=lambda x: -x[1]):
        cc  = CC[_cc(c)]
        pct = round(c * 100)
        lbl = LABELS['product'].get(p, p)
        rows += f"""<div style="padding:11px 0;border-bottom:1px solid #f1f5f9;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;">
    <span style="font-size:12px;font-weight:500;color:#1e293b;">{lbl}</span>
    <span style="font-size:24px;font-weight:200;color:#0f172a;">{_pct(c)}</span>
  </div>
  <div style="background:#e2e8f0;border-radius:99px;height:5px;width:100%;margin:5px 0 3px;">
    <div style="background:{cc['bar']};border-radius:99px;height:5px;width:{pct}%;"></div>
  </div>
  <div style="font-size:10px;color:#94a3b8;">95% CI: {_pct1(lo)} – {_pct1(hi)}</div>
</div>"""

    return f"""<div style="font-family:'DM Sans',sans-serif;max-width:620px;">
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;
     padding:22px 24px;margin-bottom:10px;">
  <div style="font-size:9px;text-transform:uppercase;letter-spacing:.1em;
        color:#94a3b8;margin-bottom:4px;">Product comparison</div>
  <div style="font-size:11px;color:#64748b;margin-bottom:14px;">{profile_str}</div>
  {rows}
</div>
<div style="font-size:9px;color:#b0b8c4;text-align:right;">{_src()}</div>
</div>"""


# Gradio interface
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');
* { box-sizing: border-box }
body, .gradio-container { font-family: 'DM Sans', sans-serif !important }
footer { display: none !important }
textarea { font-family: 'DM Sans', sans-serif !important; font-size: 13px !important }
.ask-btn button { background: #0f172a !important; color: #fff !important;
  border: none !important; border-radius: 10px !important;
  font-weight: 600 !important; font-size: 13px !important; padding: 10px 18px !important }
.ask-btn button:hover { background: #1e293b !important }
.ex-btn button { background: #fff !important; border: 1px solid #e2e8f0 !important;
  border-radius: 8px !important; color: #475569 !important; font-size: 11px !important;
  text-align: left !important; padding: 6px 10px !important;
  margin: 2px 0 !important; width: 100% !important }
.ex-btn button:hover { border-color: #0ea5e9 !important;
  color: #0284c7 !important; background: #f0f9ff !important }
"""

EXAMPLES = [
    "Would a 30-year-old woman with a university degree accept recycled plastic in food packaging?",
    "Man, 65 years old, high income, university -- food packaging",
    "Compare food packaging vs WTP premium vs perceived quality for urban women aged 30-44",
    "Young person, 22 years old, low income, rural, primary education, never recycles -- general",
    "All indicators for a 35-year-old woman, university, high income, very concerned about the environment, always recycles",
]


def chat_fn(msg, hist):
    """Gradio chat callback: parse question, compute, render result and narrative."""
    if not msg or not msg.strip():
        return hist, "", gr.update()
    hist = hist or []

    try:
        parsed = parse_question(msg)
    except Exception as e:
        hist2 = hist + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": f"[API error] {e}"},
        ]
        return hist2, "", gr.update()

    age        = parsed.get("age")         or DEFAULTS["age"]
    gender     = parsed.get("gender")      or DEFAULTS["gender"]
    edu        = parsed.get("edu")         or DEFAULTS["edu"]
    income     = parsed.get("income")      or DEFAULTS["income"]
    product    = parsed.get("product")     or DEFAULTS["product"]
    env        = parsed.get("env_concern") or DEFAULTS["env"]
    urban      = parsed.get("urban")       or DEFAULTS["urban"]
    recycling  = parsed.get("recycling")   or DEFAULTS["recycling"]
    wtp_prices = parsed.get("wtp_prices")  or DEFAULTS["wtp_prices"]

    prof_str = _profile_str(age, gender, edu, income, env, urban,
                            recycling, wtp_prices)

    ask_all = (
        product == 'general'
        and not parsed.get("is_comparison")
        and any(kw in msg.lower() for kw in
                ['all', 'todos', 'complete', 'indicators', 'dashboard'])
    )

    if ask_all:
        results = compute_all_targets(age, gender, edu, income, env, urban,
                                      recycling, wtp_prices)
        html = card_all_targets(results, prof_str)
        data = {
            "profile": prof_str,
            "results": {
                p: {"probability": _pct(r['central']),
                    "label": r['label'],
                    "ci_95": f"{_pct1(r['lo'])}-{_pct1(r['hi'])}"}
                for p, r in results.items()
            },
        }

    elif parsed.get("is_comparison") and parsed.get("compare_products"):
        prods = [p for p in parsed["compare_products"] if p in TARGETS]
        if not prods:
            prods = ["food_packaging", "general", "wtp_premium"]
        cs, ls, hs = [], [], []
        for p in prods:
            c, lo, hi, _ = compute(age, gender, edu, income, p,
                                   env, urban, recycling, wtp_prices)
            cs.append(c); ls.append(lo); hs.append(hi)
        html = card_comparison(prods, cs, ls, hs, prof_str)
        data = {
            "profile": prof_str,
            "comparison": [
                {"product": LABELS['product'].get(p, p),
                 "probability": _pct(c),
                 "ci_95": f"{_pct1(lo)}-{_pct1(hi)}"}
                for p, c, lo, hi in zip(prods, cs, ls, hs)
            ],
        }

    else:
        c, lo, hi, debug = compute(age, gender, edu, income, product,
                                   env, urban, recycling, wtp_prices)
        prod_label = LABELS['product'].get(product, product)
        html = card_single(c, lo, hi, debug, prof_str, prod_label)
        data = {
            "profile":     prof_str,
            "product":     prod_label,
            "probability": _pct(c),
            "ci_95":       f"{_pct1(lo)}-{_pct1(hi)}",
            "config_key":  debug.get('config_key', ''),
            "cpt_range":   debug.get('p_range', ''),
        }

    try:
        narr = narrative(data, msg)
    except Exception as e:
        narr = f"*(Narrative unavailable: {e})*"

    if parsed.get("confidence", "high") == "low":
        narr += "\n\n> Low extraction confidence — some fields use default values."

    hist2 = hist + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": narr},
    ]
    return hist2, "", gr.update(value=html)


LOGO_B64 = ""
if LOGO_PATH.exists():
    with open(LOGO_PATH, 'rb') as f:
        LOGO_B64 = base64.b64encode(f.read()).decode()


def build_demo():
    """Construct and return the Gradio Blocks application."""
    n_virtual = len(DF_S1) if DF_S1 is not None else 10000

    def _header():
        logo = (f'<img src="data:image/png;base64,{LOGO_B64}" '
                f'style="height:52px;width:auto;"/>'
                ) if LOGO_B64 else ""
        return f"""<div style="background:#0f172a;color:#fff;padding:18px 28px;
    display:flex;align-items:center;gap:18px;">
    {logo}
    <div style="flex:1;">
      <div style="font-size:18px;font-weight:600;color:#e2e8f0;">
        ASAP — Virtual Survey: Recycled Plastic Acceptance</div>
      <div style="font-size:12px;color:#94a3b8;margin-top:3px;">
        E4CE · Engineering for Circular Economy | University of Alicante</div>
    </div>
    <div style="font-size:10px;color:#475569;text-align:right;line-height:1.6;">
      CIS 3391 (n={N_CIS:,}) · {N_STUDIES} studies · {n_virtual:,} virtual surveys<br>
      <span style="color:#334155;">5 primary indicators + 8 extended</span>
    </div>
    </div>"""

    with gr.Blocks(
        title="ASAP Virtual Survey",
        theme=gr.themes.Base(
            primary_hue="sky", secondary_hue="emerald", neutral_hue="slate",
            font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif"],
        ),
        css=CSS,
    ) as demo:
        gr.HTML(_header())
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    label="", height=460,
                    show_copy_button=True, type="messages",
                    bubble_full_width=False, show_label=False,
                    placeholder=(
                        "**ASAP Virtual Survey** — available indicators:\n"
                        "Primary: food packaging, general acceptance, "
                        "perceived safety, perceived quality, WTP premium\n"
                        "Extended: non-food packaging, beverages, cleaning, "
                        "personal care, toys, clothing, electronics, furniture\n\n"
                        "Ask about any indicator or request **all indicators** for a profile."
                    ),
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="E.g.: all indicators for a 30-year-old woman, university, high income...",
                        show_label=False, scale=5, container=False,
                        lines=1, max_lines=2,
                    )
                    btn = gr.Button("Query", variant="primary",
                                   scale=1, min_width=90,
                                   elem_classes=["ask-btn"])
                with gr.Accordion("Example questions", open=False):
                    for ex in EXAMPLES:
                        b = gr.Button(ex, size="sm", elem_classes=["ex-btn"])
                        b.click(fn=lambda x=ex: x, outputs=[msg])

            with gr.Column(scale=4):
                gr.HTML('<div style="font-size:9px;text-transform:uppercase;'
                        'letter-spacing:.1em;color:#94a3b8;font-weight:600;'
                        'padding:16px 16px 8px;">Result</div>')
                result = gr.HTML(
                    value='<div style="text-align:center;padding:40px;color:#94a3b8;">'
                          '<div style="font-size:24px;margin-bottom:8px;">...</div>'
                          '<div style="font-size:12px;">Results appear here</div>'
                          '</div>'
                )

        hist_state = gr.State([])

        def _go(m, h):
            nh, cl, ru = chat_fn(m, h)
            return nh, cl, ru, nh

        btn.click(fn=_go, inputs=[msg, hist_state],
                  outputs=[chatbot, msg, result, hist_state])
        msg.submit(fn=_go, inputs=[msg, hist_state],
                   outputs=[chatbot, msg, result, hist_state])

    return demo


if __name__ == "__main__":
    test_differentiation()

    if HAS_GRADIO:
        api_ok  = bool(os.getenv("ANTHROPIC_API_KEY", ""))
        cpt_ok  = len(CPT_CACHE) == 5
        print(f"\n  API key:  {'set' if api_ok else 'NOT SET -- NLP disabled'}")
        print(f"  CPTs:     {'5/5 OK' if cpt_ok else f'{len(CPT_CACHE)}/5 loaded'}")
        print(f"  Priors:   {'found' if PRIORS_PATH.exists() else 'NOT FOUND'}\n")
        demo = build_demo()
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            allowed_paths=[str(BASE)],
        )
    else:
        print("gradio not installed -- CLI test only")ƒ
