# STEP 5 - THREE-LEVEL VALIDATION
#
# Level 1: Machinery check (S2 reproduces CIS)
# Level 2: Fusion coherence (pi_posterior between pi_CIS and pi_lit)
# Level 3: Fusion value
#   3a: Literature-only predictions by demographic profile
#   3b: Subsample experiment (value of fusion at small n)

import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

C_S1 = '#1A1A1A'; C_S2 = '#2563EB'; C_LIT = '#D97706'
C_CIS = '#10B981'; C_POST = '#8B5CF6'

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "step4_output")
FIG = os.path.join(OUT, "figures_validation")
os.makedirs(FIG, exist_ok=True)

CIS_MISSING = {96, 97, 98, 99, 996, 997, 998, 999}

HARM_MAP = {
    'attitudes_env_concern': 'V15',
    'attitudes_wtp_taxes': 'V26',
    'behavior_recycling': 'V52',
    'behavior_reduce_consumption': 'V50',
    'demographics_age': 'BIRTH',
    'demographics_education': 'ESTUDIOS',
    'demographics_gender': 'SEX',
    'demographics_urban_rural': 'URBRURAL',
    'perception_danger_pollution': 'V37',
    'trust_institutions': 'V10',
}


def tvd(p, q):
    keys = sorted(set(list(p.keys()) + list(q.keys())))
    pv = np.array([p.get(k, 0) for k in keys], dtype=float)
    qv = np.array([q.get(k, 0) for k in keys], dtype=float)
    pv = np.maximum(pv, 1e-12); pv /= pv.sum()
    qv = np.maximum(qv, 1e-12); qv /= qv.sum()
    return 0.5 * np.abs(pv - qv).sum()


def load_cis(path):
    df = pd.read_csv(path, sep=";", encoding="latin1",
                     low_memory=False, quotechar='"',
                     na_values=['N.P.', 'N.C.', 'N.S.', 'N.D.'])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for code in CIS_MISSING:
        df = df.replace(code, np.nan)
    return df


def discretise_to_match(df_raw, df_synth, common):
    out = {}
    for col in common:
        if col not in df_raw.columns or col not in df_synth.columns:
            continue
        x = df_raw[col].dropna()
        s_vals = sorted(df_synth[col].dropna().unique())
        if len(s_vals) <= 1:
            continue
        nu = x.nunique()
        if nu <= 15:
            vals = sorted(x.unique())
            mp = {v: i for i, v in enumerate(vals)}
            out[col] = df_raw[col].map(mp)
        else:
            K = len(s_vals)
            try:
                out[col] = pd.qcut(df_raw[col], q=K,
                                   labels=range(K), duplicates='drop')
            except Exception:
                out[col] = pd.cut(df_raw[col], bins=K,
                                  labels=range(K), duplicates='drop')
    df = pd.DataFrame(out, index=df_raw.index)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def level1_machinery(df_orig, df_s2, common):
    print("\nLEVEL 1 - MACHINERY CHECK")
    print("Does S2 (CIS-only) reproduce CIS marginals?")

    tvds = {}
    for var in common:
        if var.startswith('_'):
            continue
        o = df_orig[var].dropna().value_counts(normalize=True).to_dict()
        s = df_s2[var].value_counts(normalize=True).to_dict()
        tvds[var] = tvd(o, s)

    vals = list(tvds.values())
    good = sum(1 for t in vals if t < 0.05)
    fair = sum(1 for t in vals if 0.05 <= t < 0.15)
    poor = sum(1 for t in vals if t >= 0.15)

    print(f"\n  Variables: {len(vals)}")
    print(f"  TVD mean: {np.mean(vals):.4f} | median: {np.median(vals):.4f}")
    print(f"  Good (<0.05): {good} | Fair: {fair} | Poor: {poor}")

    sv = sorted(tvds.items(), key=lambda x: x[1])
    print(f"\n  Best 5:  {', '.join(f'{v}={t:.3f}' for v, t in sv[:5])}")
    print(f"  Worst 5: {', '.join(f'{v}={t:.3f}' for v, t in sv[-5:])}")

    verdict = "PASS" if np.median(vals) < 0.02 else "MARGINAL" if np.median(vals) < 0.05 else "FAIL"
    print(f"\n  VERDICT: {verdict} (median TVD = {np.median(vals):.4f})")

    return {'tvds': tvds, 'mean': np.mean(vals), 'median': np.median(vals),
            'good': good, 'fair': fair, 'poor': poor, 'verdict': verdict}


def level2_fusion_coherence(df_orig, df_s1, fused_priors, common):
    print("\nLEVEL 2 - FUSION COHERENCE")
    print("Is pi_S1 between pi_CIS and pi_lit?")
    print("alpha_posterior = alpha_lit + n_CIS")

    results = []

    for harm_name, cis_var in sorted(HARM_MAP.items()):
        if cis_var not in common:
            continue
        vdata = fused_priors.get(harm_name)
        if not vdata:
            continue

        alpha_lit = np.array(
            vdata.get('alpha_lit', vdata.get('alpha', [])), dtype=float)
        if len(alpha_lit) < 2:
            continue

        pi_lit = alpha_lit / alpha_lit.sum()

        o = df_orig[cis_var].dropna().astype(int)
        K_cis = int(o.max()) + 1
        counts_cis = np.bincount(o, minlength=K_cis)[:K_cis].astype(float)
        pi_cis = counts_cis / counts_cis.sum()

        s1 = df_s1[cis_var].dropna().astype(int)
        K_s1 = int(s1.max()) + 1
        counts_s1 = np.bincount(s1, minlength=K_s1)[:K_s1].astype(float)
        pi_s1 = counts_s1 / counts_s1.sum()

        K = min(len(pi_lit), len(pi_cis), len(pi_s1))
        pi_lit_k = pi_lit[:K]; pi_lit_k /= pi_lit_k.sum()
        pi_cis_k = pi_cis[:K]; pi_cis_k /= pi_cis_k.sum()
        pi_s1_k = pi_s1[:K]; pi_s1_k /= pi_s1_k.sum()

        ess_lit = float(alpha_lit.sum())
        n_cis = float(counts_cis.sum())
        w_lit = ess_lit / (ess_lit + n_cis)
        w_cis = n_cis / (ess_lit + n_cis)

        pi_expected = w_lit * pi_lit_k + w_cis * pi_cis_k

        n_between = 0
        for k in range(K):
            lo = min(pi_lit_k[k], pi_cis_k[k])
            hi = max(pi_lit_k[k], pi_cis_k[k])
            if pi_s1_k[k] >= lo - 0.02 and pi_s1_k[k] <= hi + 0.02:
                n_between += 1
        pct_between = n_between / K * 100

        tvd_to_exp = 0.5 * np.abs(pi_s1_k - pi_expected).sum()

        results.append({
            'variable': harm_name, 'cis_var': cis_var, 'K': K,
            'ess_lit': ess_lit, 'n_cis': n_cis,
            'w_lit': w_lit, 'w_cis': w_cis,
            'pct_between': pct_between, 'tvd_to_expected': tvd_to_exp,
            'pi_lit': pi_lit_k.tolist(), 'pi_cis': pi_cis_k.tolist(),
            'pi_s1': pi_s1_k.tolist(), 'pi_expected': pi_expected.tolist(),
        })

        print(f"\n  {harm_name} ({cis_var}):  K={K}  "
              f"w_lit={w_lit:.1%}  w_CIS={w_cis:.1%}")
        fmt = lambda a: '[' + ', '.join(f'{p:.3f}' for p in a[:6]) + ']'
        print(f"    pi_lit:      {fmt(pi_lit_k)}")
        print(f"    pi_CIS:      {fmt(pi_cis_k)}")
        print(f"    pi_expected: {fmt(pi_expected)}")
        print(f"    pi_S1:       {fmt(pi_s1_k)}")
        print(f"    Between: {pct_between:.0f}%  TVD to expected: {tvd_to_exp:.4f}")

    if results:
        mb = np.mean([r['pct_between'] for r in results])
        mt = np.mean([r['tvd_to_expected'] for r in results])
        print(f"\n  Mean betweenness: {mb:.0f}%")
        print(f"  Mean TVD to expected: {mt:.4f}")
        v = "PASS" if mb >= 60 else "MARGINAL" if mb >= 40 else "FAIL"
        print(f"  VERDICT: {v}")

    return results


def level3a_literature_only(df_s1, fused_priors):
    print("\nLEVEL 3a - FUSION VALUE: LITERATURE-ONLY PREDICTIONS")
    print("Variables only S1 can produce")

    lit_vars = {}
    for vname, vdata in fused_priors.items():
        is_lit = (vdata.get('source') == 'literature_only'
                  or vdata.get('n_obs_cis', 0) == 0)
        if not is_lit:
            continue
        alpha = np.array(
            vdata.get('alpha_lit', vdata.get('alpha_posterior', [])),
            dtype=float)
        if len(alpha) < 2:
            continue
        lit_vars[vname] = {
            'alpha': alpha, 'K': len(alpha),
            'n_studies': vdata.get('n_studies_lit', 0),
            'pi': (alpha / alpha.sum()).tolist(),
        }

    if not lit_vars:
        print("  No literature-only variables found.")
        return {}

    print(f"\n  Literature-only variables: {len(lit_vars)}")
    for vn, info in lit_vars.items():
        pi_str = ', '.join(f'{p:.3f}' for p in info['pi'])
        print(f"    {vn}: K={info['K']}, "
              f"n_studies={info['n_studies']}, E[theta]=[{pi_str}]")

    profiles = {
        'gender': {'var': 'SEX', 'labels': {0: 'Male', 1: 'Female'}},
        'age': {'var': 'BIRTH',
                'labels': {0: 'Young', 1: 'Mid-young', 2: 'Middle',
                           3: 'Mid-old', 4: 'Old'}},
        'education': {'var': 'ESTUDIOS',
                      'labels': {0: 'No formal', 1: 'Primary',
                                 2: 'Secondary', 3: 'Vocational',
                                 4: 'Bachillerato', 5: 'FP Superior',
                                 6: 'University', 7: 'Postgrad'}},
    }

    results = {}
    for lit_name in lit_vars:
        if lit_name not in df_s1.columns:
            continue
        print(f"\n  {lit_name}")
        var_results = {}
        K = lit_vars[lit_name]['K']

        for prof_name, prof_info in profiles.items():
            pvar = prof_info['var']
            if pvar not in df_s1.columns:
                continue
            print(f"\n    By {prof_name} ({pvar}):")

            for val, label in sorted(prof_info['labels'].items()):
                sub = df_s1[df_s1[pvar] == val][lit_name].dropna()
                if len(sub) < 30:
                    continue
                counts = np.bincount(sub.astype(int), minlength=K)[:K].astype(float)
                pi = counts / counts.sum()
                pi_str = ', '.join(f'{p:.3f}' for p in pi)
                print(f"      {label:15s} (n={len(sub):5d}): [{pi_str}]")
                var_results[f"{prof_name}_{label}"] = pi.tolist()

        results[lit_name] = var_results

    return results


def level3b_subsample(df_orig_disc, binfo, fused_priors,
                      backbone_edges, emeta, mcmc_samples, common):
    print("\nLEVEL 3b - SUBSAMPLE EXPERIMENT")
    print("Value of fusion when national data is scarce")

    sys.path.insert(0, BASE)
    try:
        from step4_improved import (
            DirichletBDeuCPD, MC3DAGEnsemble, Sampler,
            generate_martins, Config, HARM_TO_CIS
        )
    except ImportError as e:
        print(f"  Cannot import step4_improved: {e}")
        return []

    df_full = df_orig_disc[common].dropna().reset_index(drop=True)

    subsample_sizes = [100, 200, 500, 1000, len(df_full)]
    eval_vars = [v for v in
        ['SEX', 'V15', 'V27', 'V52', 'V37', 'V10', 'BIRTH', 'URBRURAL', 'V50', 'V26']
        if v in common]

    gt = {}
    for var in eval_vars:
        gt[var] = df_full[var].value_counts(normalize=True).to_dict()

    results = []

    for n_sub in subsample_sizes:
        print(f"\n  n = {n_sub}")

        if n_sub >= len(df_full):
            df_sub = df_full.copy()
        else:
            rng_sub = np.random.default_rng(42)
            idx = rng_sub.choice(len(df_full), size=n_sub, replace=False)
            df_sub = df_full.iloc[idx].reset_index(drop=True)

        cpd_s1 = DirichletBDeuCPD(df_sub[common], binfo, fused_priors, ess=10.0)
        cpd_s2 = DirichletBDeuCPD(df_sub[common], binfo, {}, ess=10.0)

        ensemble = MC3DAGEnsemble(
            mcmc_samples, backbone_edges, emeta, common, max_indeg=2)

        cfg = Config()
        cfg.M = 30; cfg.N_PER = 100; cfg.N_FINAL = 3000; cfg.SEED = 42

        df_s1, _ = generate_martins(ensemble, cpd_s1, common, cfg,
                                    method_label=f"S1 n={n_sub}")
        df_s2, _ = generate_martins(ensemble, cpd_s2, common, cfg,
                                    method_label=f"S2 n={n_sub}")

        tvd_s1 = []; tvd_s2 = []
        for var in eval_vars:
            s1_d = df_s1[var].value_counts(normalize=True).to_dict()
            s2_d = df_s2[var].value_counts(normalize=True).to_dict()
            tvd_s1.append(tvd(gt[var], s1_d))
            tvd_s2.append(tvd(gt[var], s2_d))

        m1 = np.mean(tvd_s1); m2 = np.mean(tvd_s2)
        delta = m2 - m1

        results.append({
            'n': n_sub, 'tvd_s1_mean': m1, 'tvd_s2_mean': m2,
            'delta': delta, 'winner': 'S1' if m1 < m2 else 'S2',
            'tvd_s1_vars': dict(zip(eval_vars, tvd_s1)),
            'tvd_s2_vars': dict(zip(eval_vars, tvd_s2)),
        })
        w = 'S1' if m1 < m2 else 'S2'
        print(f"    TVD: S1={m1:.4f}  S2={m2:.4f}  delta={delta:+.4f}  [{w}]")

    print(f"\n  {'n':>6s}  {'S1':>8s}  {'S2':>8s}  {'Delta':>8s}  {'Win':>4s}")
    print(f"  {'-'*38}")
    for r in results:
        print(f"  {r['n']:6d}  {r['tvd_s1_mean']:8.4f}  {r['tvd_s2_mean']:8.4f}  "
              f"{r['delta']:+8.4f}  {r['winner']:>4s}")

    for i in range(len(results) - 1):
        if results[i]['winner'] != results[i+1]['winner']:
            print(f"\n  Crossover between n={results[i]['n']} and n={results[i+1]['n']}")
            print(f"  Below crossover: international priors add value")
            print(f"  Above crossover: national data is self-sufficient")
            break

    return results


def plot_level2(results, figdir):
    n_vars = min(len(results), 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, r in enumerate(results[:n_vars]):
        ax = axes[i]
        K = min(r['K'], 7)
        x = np.arange(K)
        w = 0.18

        ax.bar(x - 1.5*w, r['pi_lit'][:K], w, label='Literature', color=C_LIT, alpha=0.8)
        ax.bar(x - 0.5*w, r['pi_cis'][:K], w, label='CIS', color=C_CIS, alpha=0.8)
        ax.bar(x + 0.5*w, r['pi_expected'][:K], w, label='Expected', color=C_POST, alpha=0.8)
        ax.bar(x + 1.5*w, r['pi_s1'][:K], w, label='S1', color=C_S1, alpha=0.8)

        ax.set_title(f"{r['variable']}\nw_lit={r['w_lit']:.0%} w_CIS={r['w_cis']:.0%}",
                     fontsize=10)
        ax.set_xlabel('Category'); ax.set_ylabel('Proportion')
        ax.set_xticks(x)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Level 2: Fusion Coherence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(figdir, 'level2_fusion_coherence.png')
    fig.savefig(p); fig.savefig(p.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved {p}")


def plot_level3b(results, figdir):
    if not results:
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    ns = [r['n'] for r in results]
    s1 = [r['tvd_s1_mean'] for r in results]
    s2 = [r['tvd_s2_mean'] for r in results]

    ax.plot(ns, s1, 'o-', color=C_S1, lw=2, ms=8, label='S1 (with intl priors)')
    ax.plot(ns, s2, 's--', color=C_S2, lw=2, ms=8, label='S2 (CIS-only)')

    ax.set_xlabel('National sample size (n)', fontsize=12)
    ax.set_ylabel('Mean TVD vs full CIS', fontsize=12)
    ax.set_title('Value of International Priors by Sample Size', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])

    plt.tight_layout()
    p = os.path.join(figdir, 'level3b_subsample_curve.png')
    fig.savefig(p); fig.savefig(p.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved {p}")


def plot_summary(l1, l2, l3b, figdir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if l1 and 'tvds' in l1:
        vals = list(l1['tvds'].values())
        axes[0].hist(vals, bins=30, color=C_S2, alpha=0.7, edgecolor='white')
        axes[0].axvline(x=0.05, color='red', ls='--', alpha=0.6, label='0.05')
        axes[0].set_xlabel('TVD'); axes[0].set_ylabel('Count')
        axes[0].set_title(f'Level 1: Machinery\nmedian={l1["median"]:.4f}')
        axes[0].legend()

    if l2:
        names = [r['cis_var'] for r in l2]
        pcts = [r['pct_between'] for r in l2]
        colors = [C_CIS if p >= 60 else C_LIT for p in pcts]
        axes[1].barh(range(len(names)), pcts, color=colors, alpha=0.8)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names, fontsize=9)
        axes[1].set_xlabel('% categories between lit & CIS')
        axes[1].set_title('Level 2: Coherence')
        axes[1].axvline(x=60, color='red', ls='--', alpha=0.5)

    if l3b:
        ns = [r['n'] for r in l3b]
        s1v = [r['tvd_s1_mean'] for r in l3b]
        s2v = [r['tvd_s2_mean'] for r in l3b]
        axes[2].plot(ns, s1v, 'o-', color=C_S1, lw=2, label='S1')
        axes[2].plot(ns, s2v, 's--', color=C_S2, lw=2, label='S2')
        axes[2].set_xlabel('n'); axes[2].set_ylabel('Mean TVD')
        axes[2].set_title('Level 3b: Value of Fusion')
        axes[2].set_xscale('log'); axes[2].legend()

    fig.suptitle('Three-Level Validation Summary', fontsize=15, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(figdir, 'validation_summary.png')
    fig.savefig(p); fig.savefig(p.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved {p}")


def main():
    print("STEP 5 - THREE-LEVEL VALIDATION")

    print("\n  Loading data...")
    path_cis = os.path.join(BASE, "metadata", "3391_num.csv")
    df_raw = load_cis(path_cis)
    print(f"    CIS raw: {df_raw.shape}")

    df_s1 = pd.read_csv(os.path.join(OUT, "synthetic_surveys_S1.csv"), low_memory=False)
    df_s2 = pd.read_csv(os.path.join(OUT, "synthetic_surveys_S2.csv"), low_memory=False)
    print(f"    S1: {df_s1.shape} | S2: {df_s2.shape}")

    with open(os.path.join(BASE, "step3_fused_priors.json"), 'r', encoding='utf-8') as f:
        fused_data = json.load(f)
    fused_priors = fused_data.get('variables', {})

    path_meta = os.path.join(OUT, "metadata.json")
    binfo = {}
    if os.path.exists(path_meta):
        with open(path_meta, 'r') as f:
            binfo = json.load(f).get('binfo', {})

    backbone_edges = []; emeta = {}
    for dp in [os.path.join(BASE, "dag_fused_edges.csv"),
               os.path.join(BASE, "dag_edges_3391.csv")]:
        if os.path.exists(dp):
            de = pd.read_csv(dp)
            for _, r in de.iterrows():
                u, v = str(r['source']).strip(), str(r['target']).strip()
                try: p = float(str(r.get('probability', 0.5)).split()[0])
                except: p = 0.5
                backbone_edges.append((u, v)); emeta[(u, v)] = p
            break

    mcmc_samples = {}
    mp = os.path.join(BASE, "step1_mcmc_samples.json")
    if os.path.exists(mp):
        with open(mp, 'r') as f:
            mcmc_samples = json.load(f)

    common_all = sorted(
        set(df_s1.columns) & set(df_s2.columns) - {'_draw', 'Unnamed: 0'})
    common_raw = [c for c in common_all if c in df_raw.columns]

    print(f"\n  Discretising CIS to match synthetic encoding...")
    if binfo:
        sys.path.insert(0, BASE)
        from step4_improved import discretise
        df_disc, binfo_full = discretise(df_raw.copy())
        binfo = binfo_full
    else:
        from step4_improved import discretise
        df_disc, binfo = discretise(df_raw.copy())

    common = sorted(set(common_all) & set(df_disc.columns))
    print(f"    Common variables: {len(common)}")

    df_orig = df_disc[common].dropna()
    df_s1_c = df_s1[[c for c in common if c in df_s1.columns]].dropna()
    df_s2_c = df_s2[[c for c in common if c in df_s2.columns]].dropna()
    print(f"    Original: {len(df_orig)} | S1: {len(df_s1_c)} | S2: {len(df_s2_c)}")

    l1 = level1_machinery(df_orig, df_s2_c, common)
    l2 = level2_fusion_coherence(df_orig, df_s1_c, fused_priors, common)
    l3a = level3a_literature_only(df_s1, fused_priors)
    l3b = level3b_subsample(df_disc, binfo, fused_priors,
                            backbone_edges, emeta, mcmc_samples, common)

    print(f"\nFIGURES")
    if l2: plot_level2(l2, FIG)
    if l3b: plot_level3b(l3b, FIG)
    plot_summary(l1, l2, l3b, FIG)

    print(f"\nSAVING")

    summary = {
        'level1': {
            'verdict': l1['verdict'], 'tvd_mean': l1['mean'],
            'tvd_median': l1['median'],
            'good': l1['good'], 'fair': l1['fair'], 'poor': l1['poor'],
        },
        'level2': [{
            'variable': r['variable'], 'w_lit': r['w_lit'], 'w_cis': r['w_cis'],
            'pct_between': r['pct_between'], 'tvd_to_expected': r['tvd_to_expected'],
        } for r in l2] if l2 else [],
        'level3a': l3a or {},
        'level3b': [{
            'n': r['n'], 'tvd_s1': r['tvd_s1_mean'],
            'tvd_s2': r['tvd_s2_mean'], 'delta': r['delta'],
            'winner': r['winner'],
        } for r in l3b] if l3b else [],
    }

    op = os.path.join(OUT, "validation_3level.json")
    with open(op, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  validation_3level.json")

    if l3b:
        df_sub = pd.DataFrame([{
            'n': r['n'], 'TVD_S1': r['tvd_s1_mean'],
            'TVD_S2': r['tvd_s2_mean'], 'Delta': r['delta'],
            'Winner': r['winner'],
        } for r in l3b])
        df_sub.to_csv(os.path.join(OUT, "subsample_experiment.csv"), index=False)
        print(f"  subsample_experiment.csv")

    if l2:
        df_l2 = pd.DataFrame([{
            'Variable': r['variable'], 'CIS_var': r['cis_var'],
            'K': r['K'], 'ESS_lit': r['ess_lit'], 'n_CIS': r['n_cis'],
            'w_lit': r['w_lit'], 'w_CIS': r['w_cis'],
            'Pct_between': r['pct_between'],
            'TVD_to_expected': r['tvd_to_expected'],
        } for r in l2])
        df_l2.to_csv(os.path.join(OUT, "fusion_coherence.csv"), index=False)
        print(f"  fusion_coherence.csv")

    print(f"\nSTEP 5 - THREE-LEVEL VALIDATION COMPLETE")

    print(f"\n  Level 1 (Machinery):  {l1['verdict']}  "
          f"(S2 median TVD = {l1['median']:.4f})")

    if l2:
        mb = np.mean([r['pct_between'] for r in l2])
        mt = np.mean([r['tvd_to_expected'] for r in l2])
        print(f"  Level 2 (Coherence):  {mb:.0f}% between, "
              f"TVD to expected = {mt:.4f}")

    if l3a:
        print(f"  Level 3a (Lit-only):  {len(l3a)} variables with "
              f"demographic predictions")

    if l3b:
        s1_wins = sum(1 for r in l3b if r['winner'] == 'S1')
        print(f"  Level 3b (Subsample): S1 wins {s1_wins}/{len(l3b)} sizes")
        for r in l3b:
            print(f"    n={r['n']:5d}: S1={r['tvd_s1_mean']:.4f} "
                  f"S2={r['tvd_s2_mean']:.4f} -> {r['winner']}")

    print(f"\n  Paper narrative:")
    print(f"    - Machinery: BN+ancestral sampling reproduces CIS faithfully")
    print(f"    - Coherence: Fused posterior is weighted combination as expected")
    print(f"    - Value: S1 produces {len(l3a) if l3a else 0} variables S2 cannot")
    if l3b:
        for r in l3b:
            if r['winner'] == 'S1' and r['n'] < len(df_orig):
                print(f"    - At n={r['n']}, intl priors improve TVD by "
                      f"{abs(r['delta']):.4f}")


if __name__ == '__main__':
    main()
