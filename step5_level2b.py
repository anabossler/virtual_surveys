"""
step5_validation.py
-------------------
Three-level validation for the Bayesian survey fusion pipeline.

Level 1  -- Machinery check
           Does S2 (national data only) reproduce observed marginals?
           Pass criterion: median TVD < 0.02.

Level 2  -- Fusion coherence (Martins et al. 2024 §2.2)
           Is the fused posterior pi_S1 a convex combination of the
           literature prior pi_lit and the national survey pi_CIS?
           pi_posterior = w_lit * pi_lit + w_CIS * pi_CIS,
           where w_lit = ESS_lit / (ESS_lit + n_CIS).
           Pass criterion: >= 60% of categories satisfy betweenness.

Level 3b -- Subsample convergence
           Does S1 converge to its full-data reference faster than S2,
           confirming that literature priors regularise the estimate
           when national observations are scarce?
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colour palette
C_S1   = "#1A1A1A"  # fused (S1)
C_S2   = "#2563EB"  # national-only (S2)
C_LIT  = "#D97706"  # literature prior
C_CIS  = "#10B981"  # national survey
C_POST = "#8B5CF6"  # expected posterior

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "step4_output")
FIG  = os.path.join(OUT,  "figures_validation")
os.makedirs(FIG, exist_ok=True)

# Survey missing-value codes to recode as NaN
MISSING_CODES = {96, 97, 98, 99, 996, 997, 998, 999}

# Mapping from harmonised variable names to raw survey column names
HARM_MAP = {
    "attitudes_env_concern":        "V15",
    "attitudes_wtp_taxes":          "V26",
    "behavior_recycling":           "V52",
    "behavior_reduce_consumption":  "V50",
    "demographics_age":             "BIRTH",
    "demographics_education":       "ESTUDIOS",
    "demographics_gender":          "SEX",
    "demographics_urban_rural":     "URBRURAL",
    "perception_danger_pollution":  "V37",
    "trust_institutions":           "V10",
}


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def total_variation_distance(p: dict, q: dict) -> float:
    """Total variation distance between two categorical distributions.

    Both inputs are {category: probability} dicts; missing categories
    are treated as probability zero.
    """
    keys = sorted(set(p) | set(q))
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    # Normalise defensively
    pv = np.maximum(pv, 1e-12); pv /= pv.sum()
    qv = np.maximum(qv, 1e-12); qv /= qv.sum()
    return 0.5 * np.abs(pv - qv).sum()


def load_survey_csv(path: str) -> pd.DataFrame:
    """Load a semicolon-delimited survey file and recode missing values."""
    df = pd.read_csv(
        path, sep=";", encoding="latin1", low_memory=False,
        quotechar='"', na_values=["N.P.", "N.C.", "N.S.", "N.D."]
    )
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for code in MISSING_CODES:
        df = df.replace(code, np.nan)
    return df


def discretise_to_synthetic_scale(df_raw: pd.DataFrame,
                                   df_synth: pd.DataFrame,
                                   common_cols: list) -> pd.DataFrame:
    """Re-encode raw survey columns to match the integer scale of the
    synthetic data, using quantile binning for continuous variables."""
    recoded = {}
    for col in common_cols:
        if col not in df_raw.columns or col not in df_synth.columns:
            continue
        x = df_raw[col].dropna()
        synth_cats = sorted(df_synth[col].dropna().unique())
        if len(synth_cats) <= 1:
            continue
        if x.nunique() <= 15:
            levels = sorted(x.unique())
            level_map = {v: i for i, v in enumerate(levels)}
            recoded[col] = df_raw[col].map(level_map)
        else:
            K = len(synth_cats)
            try:
                recoded[col] = pd.qcut(
                    df_raw[col], q=K, labels=range(K), duplicates="drop"
                )
            except Exception:
                recoded[col] = pd.cut(
                    df_raw[col], bins=K, labels=range(K), duplicates="drop"
                )
    out = pd.DataFrame(recoded, index=df_raw.index)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ---------------------------------------------------------------
# Level 1: Machinery check
# ---------------------------------------------------------------

def level1_machinery(df_orig: pd.DataFrame,
                     df_s2: pd.DataFrame,
                     common_cols: list) -> dict:
    """Verify that the Bayesian network sampler faithfully reproduces
    the national survey marginals (S2 uses only national data).

    Implements the OIC / TVD comparison described in Martins et al. (2024)
    Section 3.3, adapted to the TVD metric.
    """
    print("\nLEVEL 1 -- MACHINERY CHECK")
    print("Does S2 (national data only) reproduce observed marginals?")

    tvd_by_var = {}
    for var in common_cols:
        if var.startswith("_"):
            continue
        orig_dist = df_orig[var].dropna().value_counts(normalize=True).to_dict()
        s2_dist   = df_s2[var].value_counts(normalize=True).to_dict()
        tvd_by_var[var] = total_variation_distance(orig_dist, s2_dist)

    values  = list(tvd_by_var.values())
    n_good  = sum(1 for t in values if t < 0.05)
    n_fair  = sum(1 for t in values if 0.05 <= t < 0.15)
    n_poor  = sum(1 for t in values if t >= 0.15)
    mean    = float(np.mean(values))
    median  = float(np.median(values))

    print(f"\n  Variables analysed : {len(values)}")
    print(f"  TVD mean           : {mean:.4f}")
    print(f"  TVD median         : {median:.4f}")
    print(f"  Good (<0.05)       : {n_good}")
    print(f"  Fair (0.05-0.15)   : {n_fair}")
    print(f"  Poor (>=0.15)      : {n_poor}")

    ranked = sorted(tvd_by_var.items(), key=lambda x: x[1])
    print(f"\n  Best 5 : {', '.join(f'{v}={t:.3f}' for v, t in ranked[:5])}")
    print(f"  Worst 5: {', '.join(f'{v}={t:.3f}' for v, t in ranked[-5:])}")

    verdict = "PASS" if median < 0.02 else ("MARGINAL" if median < 0.05 else "FAIL")
    print(f"\n  Verdict: {verdict}  (median TVD = {median:.4f})")

    return {
        "tvds": tvd_by_var,
        "mean": mean, "median": median,
        "good": n_good, "fair": n_fair, "poor": n_poor,
        "verdict": verdict,
    }


# ---------------------------------------------------------------
# Level 2: Fusion coherence
# ---------------------------------------------------------------

def level2_fusion_coherence(df_orig: pd.DataFrame,
                             df_s1: pd.DataFrame,
                             fused_priors: dict,
                             common_cols: list) -> list:
    """Check that the fused posterior is a proper convex combination.

    For each harmonised variable v, the fused posterior satisfies
        alpha_posterior = alpha_lit + counts_CIS
    so its mean satisfies
        pi_posterior = w_lit * pi_lit + w_CIS * pi_CIS
    where w_lit = ESS_lit / (ESS_lit + n_CIS).

    Betweenness: the fraction of categories k for which pi_S1[k] lies
    between pi_lit[k] and pi_CIS[k] (with a 2 pp tolerance).

    Reference: Martins et al. (2024), Eq. (7) and Section 2.1.
    """
    print("\nLEVEL 2 -- FUSION COHERENCE")
    print("Is pi_S1 a convex combination of pi_lit and pi_CIS?")

    results = []

    for harm_name, cis_var in sorted(HARM_MAP.items()):
        if cis_var not in common_cols:
            continue
        var_data = fused_priors.get(harm_name)
        if not var_data:
            continue

        alpha_lit = np.array(
            var_data.get("alpha_lit", var_data.get("alpha", [])),
            dtype=float
        )
        if len(alpha_lit) < 2:
            continue

        pi_lit = alpha_lit / alpha_lit.sum()

        # National survey distribution
        orig_series = df_orig[cis_var].dropna().astype(int)
        K_cis = int(orig_series.max()) + 1
        counts_cis = np.bincount(orig_series, minlength=K_cis).astype(float)
        pi_cis = counts_cis / counts_cis.sum()

        # Synthetic (S1) distribution
        s1_series = df_s1[cis_var].dropna().astype(int)
        K_s1 = int(s1_series.max()) + 1
        counts_s1 = np.bincount(s1_series, minlength=K_s1).astype(float)
        pi_s1 = counts_s1 / counts_s1.sum()

        # Align to the smallest common number of categories
        K = min(len(pi_lit), len(pi_cis), len(pi_s1))
        pi_lit = pi_lit[:K] / pi_lit[:K].sum()
        pi_cis = pi_cis[:K] / pi_cis[:K].sum()
        pi_s1  = pi_s1[:K]  / pi_s1[:K].sum()

        # Mixing weights from effective sample sizes
        ess_lit = float(alpha_lit.sum())
        n_cis   = float(counts_cis.sum())
        w_lit = ess_lit / (ess_lit + n_cis)
        w_cis = n_cis   / (ess_lit + n_cis)
        pi_expected = w_lit * pi_lit + w_cis * pi_cis

        # Betweenness test (2 pp tolerance)
        n_between = sum(
            1 for k in range(K)
            if (min(pi_lit[k], pi_cis[k]) - 0.02
                <= pi_s1[k]
                <= max(pi_lit[k], pi_cis[k]) + 0.02)
        )
        pct_between  = n_between / K * 100.0
        tvd_expected = 0.5 * np.abs(pi_s1 - pi_expected).sum()

        print(f"\n  {harm_name} ({cis_var}):  K={K}  "
              f"w_lit={w_lit:.1%}  w_CIS={w_cis:.1%}")
        fmt = lambda a: "[" + ", ".join(f"{p:.3f}" for p in a[:6]) + "]"
        print(f"    pi_lit      : {fmt(pi_lit)}")
        print(f"    pi_CIS      : {fmt(pi_cis)}")
        print(f"    pi_expected : {fmt(pi_expected)}")
        print(f"    pi_S1       : {fmt(pi_s1)}")
        print(f"    Betweenness : {pct_between:.0f}%   "
              f"TVD to expected : {tvd_expected:.4f}")

        results.append({
            "variable": harm_name, "cis_var": cis_var, "K": K,
            "ess_lit": ess_lit, "n_cis": n_cis,
            "w_lit": w_lit, "w_cis": w_cis,
            "pct_between": pct_between,
            "tvd_to_expected": float(tvd_expected),
            "pi_lit": pi_lit.tolist(),
            "pi_cis": pi_cis.tolist(),
            "pi_s1":  pi_s1.tolist(),
            "pi_expected": pi_expected.tolist(),
        })

    if results:
        mean_between = float(np.mean([r["pct_between"] for r in results]))
        mean_tvd     = float(np.mean([r["tvd_to_expected"] for r in results]))
        verdict = (
            "PASS"     if mean_between >= 60 else
            "MARGINAL" if mean_between >= 40 else
            "FAIL"
        )
        print(f"\n  Mean betweenness    : {mean_between:.0f}%")
        print(f"  Mean TVD to expected: {mean_tvd:.4f}")
        print(f"  Verdict             : {verdict}")

    return results


# ---------------------------------------------------------------
# Level 3a: Literature-only variable predictions
# ---------------------------------------------------------------

def level3a_literature_only(df_s1: pd.DataFrame,
                             fused_priors: dict) -> dict:
    """Report distributions of variables that exist only in the literature
    prior (no national survey analogue), broken down by demographic profile.

    These variables can only be generated by S1 (the fused model); S2 cannot
    produce them. This demonstrates the added value of the fusion step.
    """
    print("\nLEVEL 3a -- LITERATURE-ONLY VARIABLES")
    print("Variables that only the fused model (S1) can generate.")

    lit_vars = {}
    for vname, vdata in fused_priors.items():
        is_lit_only = (
            vdata.get("source") == "literature_only"
            or vdata.get("n_obs_cis", 0) == 0
        )
        if not is_lit_only:
            continue
        alpha = np.array(
            vdata.get("alpha_lit", vdata.get("alpha_posterior", [])),
            dtype=float
        )
        if len(alpha) < 2:
            continue
        lit_vars[vname] = {
            "alpha": alpha,
            "K": len(alpha),
            "n_studies": vdata.get("n_studies_lit", 0),
            "pi": (alpha / alpha.sum()).tolist(),
        }

    if not lit_vars:
        print("  No literature-only variables found.")
        return {}

    print(f"\n  Literature-only variables: {len(lit_vars)}")
    for vn, info in lit_vars.items():
        pi_str = ", ".join(f"{p:.3f}" for p in info["pi"])
        print(f"    {vn}: K={info['K']}, "
              f"n_studies={info['n_studies']}, "
              f"E[theta]=[{pi_str}]")

    # Demographic stratification profiles
    profiles = {
        "gender": {
            "var": "SEX",
            "labels": {0: "Male", 1: "Female"},
        },
        "age": {
            "var": "BIRTH",
            "labels": {
                0: "Young", 1: "Mid-young", 2: "Middle",
                3: "Mid-old",  4: "Old",
            },
        },
        "education": {
            "var": "ESTUDIOS",
            "labels": {
                0: "No formal",  1: "Primary",
                2: "Secondary",  3: "Vocational",
                4: "Bachillerato", 5: "FP Superior",
                6: "University",  7: "Postgrad",
            },
        },
    }

    results = {}
    for lit_name in lit_vars:
        if lit_name not in df_s1.columns:
            continue
        print(f"\n  {lit_name}")
        var_results = {}
        K = lit_vars[lit_name]["K"]

        for prof_name, prof_info in profiles.items():
            pvar = prof_info["var"]
            if pvar not in df_s1.columns:
                continue
            print(f"\n    By {prof_name} ({pvar}):")

            for val, label in sorted(prof_info["labels"].items()):
                sub = df_s1[df_s1[pvar] == val][lit_name].dropna()
                if len(sub) < 30:
                    continue
                counts = np.bincount(
                    sub.astype(int), minlength=K
                )[:K].astype(float)
                pi = counts / counts.sum()
                pi_str = ", ".join(f"{p:.3f}" for p in pi)
                print(f"      {label:15s} (n={len(sub):5d}): [{pi_str}]")
                var_results[f"{prof_name}_{label}"] = pi.tolist()

        results[lit_name] = var_results

    return results


# ---------------------------------------------------------------
# Level 3b: Subsample convergence experiment
# ---------------------------------------------------------------

def level3b_subsample(df_disc: pd.DataFrame,
                       binfo: dict,
                       fused_priors: dict,
                       backbone_edges: list,
                       edge_meta: dict,
                       mcmc_samples: dict,
                       common_cols: list) -> list:
    """Demonstrate that S1 converges to its full-data reference faster
    than S2 when national observations are scarce.

    Design:
      - Reference: generate synthetic data using the full national sample.
      - Subsamples of size n in {100, 200, 500, 1000}.
      - For each n, generate S1(n) and S2(n), then compute TVD against
        the respective full-sample reference.
      - A positive delta (TVD_S2 - TVD_S1) means S1 is closer to its
        reference, i.e. the literature prior regularises estimation.

    Reference: Martins et al. (2024), Section 3.3 and Table 3.
    """
    print("\nLEVEL 3b -- SUBSAMPLE CONVERGENCE EXPERIMENT")
    print("Does the fused model converge faster with scarce national data?")
    print("Ground truth: S1 / S2 generated at n = full national sample.")

    sys.path.insert(0, BASE)
    try:
        from step4_improved import (
            DirichletBDeuCPD, MC3DAGEnsemble, Sampler,
            generate_martins, Config, HARM_TO_CIS,
        )
    except ImportError as exc:
        print(f"  Cannot import step4_improved: {exc}")
        return []

    df_full = df_disc[common_cols].dropna().reset_index(drop=True)
    n_full  = len(df_full)

    eval_vars = [
        v for v in
        ["SEX", "V15", "V27", "V52", "V37", "V10",
         "BIRTH", "URBRURAL", "V50", "V26"]
        if v in common_cols
    ]

    cfg = Config()
    cfg.M = 30; cfg.N_PER = 100; cfg.N_FINAL = 3000; cfg.SEED = 42

    ensemble = MC3DAGEnsemble(
        mcmc_samples, backbone_edges, edge_meta, common_cols, max_indeg=2
    )

    # Reference distributions at n = full
    print(f"\n  Generating reference at n = {n_full}...")
    cpd_s1_full = DirichletBDeuCPD(df_full[common_cols], binfo, fused_priors, ess=10.0)
    cpd_s2_full = DirichletBDeuCPD(df_full[common_cols], binfo, {},           ess=10.0)

    df_s1_ref, _ = generate_martins(ensemble, cpd_s1_full, common_cols, cfg,
                                    method_label=f"S1 ref n={n_full}")
    df_s2_ref, _ = generate_martins(ensemble, cpd_s2_full, common_cols, cfg,
                                    method_label=f"S2 ref n={n_full}")

    gt_s1 = {v: df_s1_ref[v].value_counts(normalize=True).to_dict()
             for v in eval_vars}
    gt_s2 = {v: df_s2_ref[v].value_counts(normalize=True).to_dict()
             for v in eval_vars}

    # Subsample experiment
    subsample_sizes = [100, 200, 500, 1000]
    results = []

    for n_sub in subsample_sizes:
        print(f"\n  n = {n_sub}")
        rng = np.random.default_rng(42)
        idx = rng.choice(n_full, size=n_sub, replace=False)
        df_sub = df_full.iloc[idx].reset_index(drop=True)

        cpd_s1 = DirichletBDeuCPD(df_sub[common_cols], binfo, fused_priors, ess=10.0)
        cpd_s2 = DirichletBDeuCPD(df_sub[common_cols], binfo, {},           ess=10.0)

        df_s1_sub, _ = generate_martins(ensemble, cpd_s1, common_cols, cfg,
                                         method_label=f"S1 n={n_sub}")
        df_s2_sub, _ = generate_martins(ensemble, cpd_s2, common_cols, cfg,
                                         method_label=f"S2 n={n_sub}")

        tvd_s1 = [total_variation_distance(
                     gt_s1[v],
                     df_s1_sub[v].value_counts(normalize=True).to_dict())
                  for v in eval_vars]
        tvd_s2 = [total_variation_distance(
                     gt_s2[v],
                     df_s2_sub[v].value_counts(normalize=True).to_dict())
                  for v in eval_vars]

        m1 = float(np.mean(tvd_s1))
        m2 = float(np.mean(tvd_s2))
        delta = m2 - m1  # positive => S1 converges faster

        results.append({
            "n":            n_sub,
            "tvd_s1_mean":  m1,
            "tvd_s2_mean":  m2,
            "delta":        delta,
            "winner":       "S1" if m1 < m2 else "S2",
            "tvd_s1_vars":  dict(zip(eval_vars, tvd_s1)),
            "tvd_s2_vars":  dict(zip(eval_vars, tvd_s2)),
        })
        winner = "S1" if m1 < m2 else "S2"
        print(f"    S1(n={n_sub}) vs S1(full): TVD = {m1:.4f}")
        print(f"    S2(n={n_sub}) vs S2(full): TVD = {m2:.4f}")
        print(f"    Faster convergence: {winner}  (delta = {delta:+.4f})")

    # Summary table
    print(f"\n  Convergence summary")
    print(f"  {'n':>6}  {'S1 TVD':>10}  {'S2 TVD':>10}  {'delta':>10}  {'Winner':>6}")
    print(f"  {'-'*50}")
    for r in results:
        print(f"  {r['n']:6d}  {r['tvd_s1_mean']:10.4f}  "
              f"{r['tvd_s2_mean']:10.4f}  "
              f"{r['delta']:+10.4f}  {r['winner']:>6s}")

    s1_wins = sum(1 for r in results if r["winner"] == "S1")
    print(f"\n  S1 converges faster in {s1_wins}/{len(results)} subsample sizes.")
    if s1_wins > len(results) / 2:
        print("  Literature priors stabilise the fused distribution "
              "when national data is scarce.")
    elif s1_wins == 0:
        print("  S2 converges faster at all sample sizes. "
              "Literature priors add variability rather than stability.")
    else:
        for i in range(len(results) - 1):
            if results[i]["winner"] != results[i + 1]["winner"]:
                print(f"  Crossover between n={results[i]['n']} "
                      f"and n={results[i + 1]['n']}.")
                break

    return results


# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------

def plot_level2(results: list, figdir: str) -> None:
    """Bar chart: pi_lit, pi_CIS, pi_expected, pi_S1 per variable."""
    n_vars = min(len(results), 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, r in enumerate(results[:n_vars]):
        ax = axes[i]
        K = min(r["K"], 7)
        x = np.arange(K)
        w = 0.18

        ax.bar(x - 1.5*w, r["pi_lit"][:K],      w, label="Literature", color=C_LIT,  alpha=0.8)
        ax.bar(x - 0.5*w, r["pi_cis"][:K],      w, label="National",   color=C_CIS,  alpha=0.8)
        ax.bar(x + 0.5*w, r["pi_expected"][:K], w, label="Expected",   color=C_POST, alpha=0.8)
        ax.bar(x + 1.5*w, r["pi_s1"][:K],       w, label="S1 fused",   color=C_S1,   alpha=0.8)

        ax.set_title(
            f"{r['variable']}\n"
            f"w_lit={r['w_lit']:.0%}  w_nat={r['w_cis']:.0%}",
            fontsize=10,
        )
        ax.set_xlabel("Category")
        ax.set_ylabel("Proportion")
        ax.set_xticks(x)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Level 2: Fusion Coherence", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(figdir, "level2_fusion_coherence.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def plot_level3b(results: list, figdir: str) -> None:
    """Convergence curves for S1 and S2 as a function of n."""
    if not results:
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    ns = [r["n"] for r in results]
    s1 = [r["tvd_s1_mean"] for r in results]
    s2 = [r["tvd_s2_mean"] for r in results]

    ax.plot(ns, s1, "o-",  color=C_S1, lw=2, ms=8, label="S1 (fused)")
    ax.plot(ns, s2, "s--", color=C_S2, lw=2, ms=8, label="S2 (national only)")

    ax.set_xlabel("National sample size (n)", fontsize=12)
    ax.set_ylabel("TVD to own full-sample reference", fontsize=12)
    ax.set_title("Convergence: Fused vs National-only", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])

    plt.tight_layout()
    path = os.path.join(figdir, "level3b_subsample_curve.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def plot_summary(l1: dict, l2: list, l3b: list, figdir: str) -> None:
    """Three-panel summary figure for the paper."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: TVD distribution (Level 1)
    if l1 and "tvds" in l1:
        vals = list(l1["tvds"].values())
        axes[0].hist(vals, bins=30, color=C_S2, alpha=0.7, edgecolor="white")
        axes[0].axvline(x=0.05, color="red", ls="--", alpha=0.6, label="0.05")
        axes[0].set_xlabel("TVD")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Level 1: Machinery\nmedian = {l1['median']:.4f}")
        axes[0].legend()

    # Panel 2: Betweenness by variable (Level 2)
    if l2:
        names  = [r["cis_var"] for r in l2]
        pcts   = [r["pct_between"] for r in l2]
        colors = [C_CIS if p >= 60 else C_LIT for p in pcts]
        axes[1].barh(range(len(names)), pcts, color=colors, alpha=0.8)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names, fontsize=9)
        axes[1].set_xlabel("% categories between literature and national")
        axes[1].set_title("Level 2: Fusion Coherence")
        axes[1].axvline(x=60, color="red", ls="--", alpha=0.5, label="60% threshold")
        axes[1].legend(fontsize=8)

    # Panel 3: Convergence curves (Level 3b)
    if l3b:
        ns = [r["n"] for r in l3b]
        axes[2].plot(ns, [r["tvd_s1_mean"] for r in l3b], "o-",
                     color=C_S1, lw=2, label="S1")
        axes[2].plot(ns, [r["tvd_s2_mean"] for r in l3b], "s--",
                     color=C_S2, lw=2, label="S2")
        axes[2].set_xlabel("n")
        axes[2].set_ylabel("TVD to reference")
        axes[2].set_title("Level 3b: Convergence")
        axes[2].set_xscale("log")
        axes[2].legend()

    fig.suptitle("Three-Level Validation Summary", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(figdir, "validation_summary.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    print("STEP 5 -- THREE-LEVEL VALIDATION")
    print("Reference: Martins et al. (2024), Sections 2.2 and 3.3")

    # Load national survey (raw)
    path_cis = os.path.join(BASE, "metadata", "3391_num.csv")
    df_raw   = load_survey_csv(path_cis)
    print(f"\n  National survey (raw): {df_raw.shape}")

    # Load synthetic datasets
    df_s1 = pd.read_csv(
        os.path.join(OUT, "synthetic_surveys_S1.csv"), low_memory=False
    )
    df_s2 = pd.read_csv(
        os.path.join(OUT, "synthetic_surveys_S2.csv"), low_memory=False
    )
    print(f"  S1 (fused)           : {df_s1.shape}")
    print(f"  S2 (national only)   : {df_s2.shape}")

    # Load fused priors
    with open(os.path.join(BASE, "step3_fused_priors.json"), "r", encoding="utf-8") as f:
        fused_data   = json.load(f)
    fused_priors = fused_data.get("variables", {})

    # Load Bayesian network metadata
    path_meta = os.path.join(OUT, "metadata.json")
    binfo = {}
    if os.path.exists(path_meta):
        with open(path_meta, "r") as f:
            binfo = json.load(f).get("binfo", {})

    # Load DAG edges
    backbone_edges = []
    edge_meta      = {}
    for dag_path in [
        os.path.join(BASE, "dag_fused_edges.csv"),
        os.path.join(BASE, "dag_edges_3391.csv"),
    ]:
        if not os.path.exists(dag_path):
            continue
        for _, row in pd.read_csv(dag_path).iterrows():
            u, v = str(row["source"]).strip(), str(row["target"]).strip()
            try:
                p = float(str(row.get("probability", 0.5)).split()[0])
            except Exception:
                p = 0.5
            backbone_edges.append((u, v))
            edge_meta[(u, v)] = p
        break

    # Load MCMC samples
    mcmc_samples = {}
    mcmc_path    = os.path.join(BASE, "step1_mcmc_samples.json")
    if os.path.exists(mcmc_path):
        with open(mcmc_path, "r") as f:
            mcmc_samples = json.load(f)

    # Discretise raw survey to match synthetic scale
    common_all = sorted(
        set(df_s1.columns) & set(df_s2.columns) - {"_draw", "Unnamed: 0"}
    )
    sys.path.insert(0, BASE)
    from step4_improved import load_cis_microdata, discretise, Config
    cfg_tmp  = Config()
    df_clean = load_cis_microdata(cfg_tmp)
    df_disc, binfo = discretise(df_clean)

    common = sorted(set(common_all) & set(df_disc.columns))
    print(f"\n  Common variables     : {len(common)}")

    df_orig = df_disc[common].dropna()
    df_s1_c = df_s1[[c for c in common if c in df_s1.columns]].dropna()
    df_s2_c = df_s2[[c for c in common if c in df_s2.columns]].dropna()
    print(f"  National (disc.)     : {len(df_orig)}")
    print(f"  S1                   : {len(df_s1_c)}")
    print(f"  S2                   : {len(df_s2_c)}")

    # Run validation levels
    l1  = level1_machinery(df_orig, df_s2_c, common)
    l2  = level2_fusion_coherence(df_orig, df_s1_c, fused_priors, common)
    l3a = level3a_literature_only(df_s1, fused_priors)
    l3b = level3b_subsample(
        df_disc, binfo, fused_priors,
        backbone_edges, edge_meta, mcmc_samples, common
    )

    # Plots
    print("\nSaving figures...")
    if l2:  plot_level2(l2, FIG)
    if l3b: plot_level3b(l3b, FIG)
    plot_summary(l1, l2, l3b, FIG)

    # Serialise results
    summary = {
        "level1": {
            "verdict":    l1["verdict"],
            "tvd_mean":   l1["mean"],
            "tvd_median": l1["median"],
            "good":       l1["good"],
            "fair":       l1["fair"],
            "poor":       l1["poor"],
        },
        "level2": [
            {
                "variable":       r["variable"],
                "w_lit":          r["w_lit"],
                "w_cis":          r["w_cis"],
                "pct_between":    r["pct_between"],
                "tvd_to_expected": r["tvd_to_expected"],
            }
            for r in l2
        ] if l2 else [],
        "level3a": l3a or {},
        "level3b": [
            {
                "n":       r["n"],
                "tvd_s1":  r["tvd_s1_mean"],
                "tvd_s2":  r["tvd_s2_mean"],
                "delta":   r["delta"],
                "winner":  r["winner"],
            }
            for r in l3b
        ] if l3b else [],
    }

    json_path = os.path.join(OUT, "validation_3level.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if l3b:
        pd.DataFrame([
            {"n": r["n"], "TVD_S1": r["tvd_s1_mean"],
             "TVD_S2": r["tvd_s2_mean"], "Delta": r["delta"],
             "Winner": r["winner"]}
            for r in l3b
        ]).to_csv(os.path.join(OUT, "subsample_experiment.csv"), index=False)

    if l2:
        pd.DataFrame([
            {"Variable": r["variable"], "CIS_var": r["cis_var"],
             "K": r["K"], "ESS_lit": r["ess_lit"], "n_CIS": r["n_cis"],
             "w_lit": r["w_lit"], "w_CIS": r["w_cis"],
             "Pct_between": r["pct_between"],
             "TVD_to_expected": r["tvd_to_expected"]}
            for r in l2
        ]).to_csv(os.path.join(OUT, "fusion_coherence.csv"), index=False)

    # Final summary
    print("\nVALIDATION COMPLETE")
    print(f"\n  Level 1 (Machinery)  : {l1['verdict']}  "
          f"(median TVD = {l1['median']:.4f})")

    if l2:
        mb = float(np.mean([r["pct_between"] for r in l2]))
        mt = float(np.mean([r["tvd_to_expected"] for r in l2]))
        print(f"  Level 2 (Coherence)  : {mb:.0f}% between,  "
              f"TVD to expected = {mt:.4f}")

    if l3a:
        print(f"  Level 3a (Lit-only)  : {len(l3a)} variables with "
              f"demographic predictions")

    if l3b:
        s1_wins = sum(1 for r in l3b if r["winner"] == "S1")
        print(f"  Level 3b (Convergence): S1 faster in {s1_wins}/{len(l3b)} "
              f"subsample sizes")
        for r in l3b:
            print(f"    n={r['n']:5d}: "
                  f"S1={r['tvd_s1_mean']:.4f}  "
                  f"S2={r['tvd_s2_mean']:.4f}  "
                  f"winner={r['winner']}")


if __name__ == "__main__":
    main()
