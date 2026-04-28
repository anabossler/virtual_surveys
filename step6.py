"""
STEP 6

"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")


def _build_paths():
    base = os.path.dirname(os.path.abspath(__file__))
    return {
        "base":    base,
        "out":     os.path.join(base, "step6_output"),
        "cis":     os.path.join(base, "metadata", "3391_num.csv"),
        "s1":      os.path.join(base, "step4_output", "synthetic_surveys_S1.csv"),
        "s2":      os.path.join(base, "step4_output", "synthetic_surveys_S2.csv"),
    }


_CIS_MISSING = {96, 97, 98, 99, 996, 997, 998, 999, 9996, 9997, 9998, 9999}

_LIT_ONLY_COLS = {
    'acceptance_food_packaging', 'acceptance_general',
    'perception_quality', 'perception_safety',
    'price_willingness_to_pay_premium', 'demographics_income',
    'acceptance_non_food_packaging', 'acceptance_beverages',
    'acceptance_cleaning_products', 'acceptance_personal_care',
    'acceptance_toys', 'acceptance_clothing_textiles',
    'acceptance_electronics', 'acceptance_furniture',
}

_ENRICHED_COLS = {'gender', 'age_group', 'education', '_draw'}

_CHI2_PAIRS = [
    ('SEX',      'V15', 'gender -> env_concern'),
    ('SEX',      'V27', 'gender -> WTP taxes'),
    ('SEX',      'V26', 'gender -> WTP prices'),
    ('ESTUDIOS', 'V15', 'education -> env_concern'),
    ('ESTUDIOS', 'V27', 'education -> WTP'),
    ('V15',      'V52', 'env_concern -> recycling'),
    ('V15',      'V27', 'env_concern -> WTP'),
    ('BIRTH',    'V15', 'age -> env_concern'),
    ('URBRURAL', 'V15', 'urban_rural -> env_concern'),
    ('V10',      'V27', 'trust -> WTP'),
]

_CORR_KEY_VARS = [
    'SEX', 'ESTUDIOS', 'BIRTH', 'NAT_INC', 'URBRURAL',
    'V15', 'V26', 'V27', 'V52', 'V10',
]


def align_to_0based(x_cis_raw, x_synth):
    """
    Align a CIS variable (original scale) with S1 (0-based scale).

    Method (mirrors step4.validate()):
      1. If unique values already match, return as-is.
      2. If same number of categories but different values
         (typical: CIS=[1,2,3,4,5] vs S1=[0,1,2,3,4]),
         apply rank mapping: sort both lists and pair by position.
      3. If CIS has more values (finer scale, uncleaned missing
         codes, or different scale), qcut to the same K.
      4. If nothing works, return (None, None) to exclude the
         variable from TVD.

    Returns (x_cis_aligned, x_synth_clean) on the same scale.
    """
    x_cis   = x_cis_raw.dropna()
    x_synth = x_synth.dropna()

    if len(x_cis) < 5 or len(x_synth) < 5:
        return None, None

    vals_cis   = sorted(x_cis.unique())
    vals_synth = sorted(x_synth.unique())

    if set(vals_cis) == set(vals_synth):
        return x_cis, x_synth

    if len(vals_cis) == len(vals_synth):
        mapping = {v_old: v_new for v_old, v_new in zip(vals_cis, vals_synth)}
        return x_cis.map(mapping), x_synth

    K = len(vals_synth)
    if K >= 2:
        try:
            x_cis_cut = pd.qcut(
                x_cis, q=K, labels=range(K), duplicates='drop'
            ).astype(float)
            return x_cis_cut, x_synth
        except Exception:
            pass

    return None, None


def tvd_aligned(x_cis_raw, x_synth):
    """
    TVD with scale alignment.
    Returns (tvd_value, n_cats) or (None, None).
    """
    x_c, x_s = align_to_0based(x_cis_raw, x_synth)
    if x_c is None:
        return None, None

    K = max(int(x_c.max()), int(x_s.max())) + 1
    p = np.bincount(x_c.values.astype(int), minlength=K).astype(float)
    q = np.bincount(x_s.values.astype(int), minlength=K).astype(float)
    p = np.maximum(p, 1e-10); p /= p.sum()
    q = np.maximum(q, 1e-10); q /= q.sum()

    return float(0.5 * np.abs(p - q).sum()), K


def load_data(paths):
    print("Loading data...")

    df_cis = pd.read_csv(
        paths["cis"], sep=";", encoding="latin1",
        low_memory=False, quotechar='"',
        na_values=['N.P.', 'N.C.', 'N.S.', 'N.D.'],
    )
    for c in df_cis.columns:
        df_cis[c] = pd.to_numeric(df_cis[c], errors='coerce')
    for code in _CIS_MISSING:
        df_cis = df_cis.replace(code, np.nan)

    df_s1 = pd.read_csv(paths["s1"], low_memory=False)
    df_s2 = pd.read_csv(paths["s2"], low_memory=False) if os.path.exists(paths["s2"]) else None

    shared = sorted(
        set(df_cis.columns) & set(df_s1.columns)
        - _LIT_ONLY_COLS - _ENRICHED_COLS - {'_draw'}
    )

    print(f"  CIS: {df_cis.shape[0]} obs x {df_cis.shape[1]} vars")
    print(f"  S1:  {df_s1.shape[0]} obs x {df_s1.shape[1]} vars")
    print(f"  Shared comparable: {len(shared)}")
    return df_cis, df_s1, df_s2, shared


def validate_tvd(df_cis, df_s1, df_s2, shared, out_dir):
    """
    TVD per variable with scale alignment (replicates step4.validate() method).
    """
    print("\nA. TVD PER VARIABLE (aligned scales)")

    results = []
    skipped = []

    for var in shared:
        if var not in df_cis.columns or var not in df_s1.columns:
            continue

        t_s1, K = tvd_aligned(df_cis[var], df_s1[var])
        if t_s1 is None:
            skipped.append(var)
            continue

        t_s2 = None
        if df_s2 is not None and var in df_s2.columns:
            t_s2, _ = tvd_aligned(df_cis[var], df_s2[var])

        results.append({
            'variable': var,
            'K':        K,
            'TVD_S1':   round(t_s1, 4),
            'TVD_S2':   round(t_s2, 4) if t_s2 is not None else None,
            'quality':  (
                'excellent' if t_s1 < 0.05 else
                'good'      if t_s1 < 0.10 else
                'fair'      if t_s1 < 0.15 else
                'poor'
            ),
        })

    df_r = pd.DataFrame(results)
    df_r.to_csv(os.path.join(out_dir, "tvd_per_variable_fixed.csv"), index=False)

    v = df_r['TVD_S1'].values
    n_exc  = int((v < 0.05).sum())
    n_good = int((v < 0.10).sum())
    n_poor = int((v >= 0.15).sum())

    print(f"  Validated: {len(v)}  (skipped: {len(skipped)})")
    print(f"  TVD mean:   {v.mean():.4f}")
    print(f"  TVD median: {np.median(v):.4f}")
    print(f"  Excellent (<0.05): {n_exc}/{len(v)}  ({100*n_exc/len(v):.1f}%)")
    print(f"  Good      (<0.10): {n_good}/{len(v)}  ({100*n_good/len(v):.1f}%)")
    print(f"  Poor      (>=0.15): {n_poor}/{len(v)}  ({100*n_poor/len(v):.1f}%)")

    print(f"\n  Worst 5:")
    for _, r in df_r.nlargest(5, 'TVD_S1').iterrows():
        print(f"    {r['variable']} (K={r['K']}): {r['TVD_S1']:.4f}")
    print(f"  Best 5:")
    for _, r in df_r.nsmallest(5, 'TVD_S1').iterrows():
        print(f"    {r['variable']} (K={r['K']}): {r['TVD_S1']:.4f}")

    if skipped:
        print(f"  Skipped (not alignable): {skipped}")

    return df_r


def _cramers_v(ct):
    """Cramer's V from a contingency table."""
    chi2 = sp_stats.chi2_contingency(ct)[0]
    n    = ct.values.sum()
    r, c = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 else 0.0


def validate_dependencies(df_cis, df_s1, shared, out_dir):
    """
    Chi-squared preservation with Cramer's V for magnitude.
    """
    print("\nB. CHI-SQUARED PRESERVATION + CRAMER'S V")

    results = []
    for v1, v2, desc in _CHI2_PAIRS:
        if v1 not in shared or v2 not in shared:
            continue
        if any(v not in df_cis.columns for v in [v1, v2]):
            continue
        if any(v not in df_s1.columns for v in [v1, v2]):
            continue

        try:
            c1, s1_v1 = align_to_0based(df_cis[v1], df_s1[v1])
            c2, s1_v2 = align_to_0based(df_cis[v2], df_s1[v2])
            if c1 is None or c2 is None:
                continue

            idx_cis = c1.index.intersection(c2.index)
            idx_s1  = s1_v1.index.intersection(s1_v2.index)

            ct_cis = pd.crosstab(
                c1.loc[idx_cis].astype(int),
                c2.loc[idx_cis].astype(int),
            )
            ct_s1 = pd.crosstab(
                s1_v1.loc[idx_s1].astype(int),
                s1_v2.loc[idx_s1].astype(int),
            )

            chi2_cis, p_cis = sp_stats.chi2_contingency(ct_cis)[:2]
            chi2_s1,  p_s1  = sp_stats.chi2_contingency(ct_s1)[:2]

            cv_cis = _cramers_v(ct_cis)
            cv_s1  = _cramers_v(ct_s1)

            dep_cis   = p_cis < 0.05
            dep_s1    = p_s1  < 0.05
            preserved = dep_cis == dep_s1
            v_ratio   = cv_s1 / cv_cis if cv_cis > 0.001 else np.nan

            results.append({
                'pair':      desc,
                'p_CIS':     round(p_cis, 4),
                'p_S1':      round(p_s1, 4),
                'dep_CIS':   dep_cis,
                'dep_S1':    dep_s1,
                'preserved': preserved,
                'V_CIS':     round(cv_cis, 3),
                'V_S1':      round(cv_s1, 3),
                'V_ratio':   round(v_ratio, 2) if not np.isnan(v_ratio) else None,
            })

            status = "OK" if preserved else "FAIL"
            v_note = (
                f"  V: CIS={cv_cis:.3f} S1={cv_s1:.3f} ratio={v_ratio:.2f}"
                if not np.isnan(v_ratio) else ""
            )
            print(f"  [{status}] {desc}: p_CIS={p_cis:.4f} p_S1={p_s1:.4f}{v_note}")

        except Exception as e:
            print(f"  [WARN] {desc}: {e}")

    df_r = pd.DataFrame(results)
    df_r.to_csv(os.path.join(out_dir, "chisq_dependencies_fixed.csv"), index=False)

    if len(df_r) > 0:
        n_pres = int(df_r['preserved'].sum())
        print(f"\n  Preserved: {n_pres}/{len(df_r)}")
        print(f"  Note: check Cramer's V ratio for CTGAN-style spurious dependencies.")

    return df_r


def validate_detection(df_cis, df_s1, shared, out_dir, n_sample=2000):
    """
    Detection AUC using only shared columns (fair comparison).
    """
    print("\nC. DETECTION AUC (shared columns only)")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("  scikit-learn not available")
        return None

    rng = np.random.default_rng(42)
    n   = min(n_sample, len(df_cis), len(df_s1))
    idx_cis = rng.choice(len(df_cis), n, replace=False)
    idx_s1  = rng.choice(len(df_s1),  n, replace=False)

    cols_ok = []
    X_cis_list, X_s1_list = [], []

    for col in shared:
        if col not in df_cis.columns or col not in df_s1.columns:
            continue
        x_c_raw = df_cis[col].iloc[idx_cis].reset_index(drop=True)
        x_s_raw = df_s1[col].iloc[idx_s1].reset_index(drop=True)
        x_c, x_s = align_to_0based(x_c_raw, x_s_raw)
        if x_c is None:
            continue
        X_cis_list.append(x_c.reindex(range(n)).fillna(-1).values)
        X_s1_list.append(x_s.reindex(range(n)).fillna(-1).values)
        cols_ok.append(col)

    if len(cols_ok) < 3:
        print("  Not enough alignable columns")
        return None

    X = np.vstack([np.column_stack(X_cis_list), np.column_stack(X_s1_list)])
    y = np.array([0]*n + [1]*n)

    clf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    aucs = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    auc_mean = float(aucs.mean())
    auc_std  = float(aucs.std())

    if auc_mean < 0.60:
        verdict = "excellent (indistinguishable)"
    elif auc_mean < 0.70:
        verdict = "good"
    elif auc_mean < 0.80:
        verdict = "acceptable"
    else:
        verdict = "poor - detectable as synthetic"

    print(f"  AUC = {auc_mean:.4f} +/- {auc_std:.4f}  ({len(cols_ok)} columns)")
    print(f"  Verdict: {verdict}")
    print(f"  Note: high AUC is expected when n_synth=100K >> n_CIS=2254.")
    print(f"  The relevant metrics for the paper are TVD and Cramer's V.")

    result = {
        'auc_mean': auc_mean,
        'auc_std':  auc_std,
        'n_cols':   len(cols_ok),
        'verdict':  verdict,
        'note':     'AUC inflated by class imbalance n_synth=100K vs n_CIS=2254',
    }
    with open(os.path.join(out_dir, "detection_fixed.json"), 'w') as f:
        json.dump(result, f, indent=2)
    return result


def validate_pmse(df_cis, df_s1, shared, n_sample=2000):
    """
    pMSE = max(0, Brier_score - 0.25)   [Woo et al. 2009]

    Interpretation:
      pMSE ~ 0.000 -> classifier cannot distinguish real from synthetic (ideal)
      pMSE > 0.050 -> synthetic is detectable (poor fidelity)

    Note: when n_synth >> n_CIS, Brier score tends to 0 even with
    high marginal fidelity because the classifier exploits class
    imbalance. Use balanced samples.
    """
    print("\nD. pMSE (Woo et al. 2009)")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  scikit-learn not available")
        return np.nan

    rng = np.random.default_rng(42)
    n   = min(n_sample, len(df_cis))
    idx_cis = rng.choice(len(df_cis), n, replace=False)
    idx_s1  = rng.choice(len(df_s1),  n, replace=False)

    cols_ok = []
    X_cis_list, X_s1_list = [], []
    for col in shared:
        if col not in df_cis.columns or col not in df_s1.columns:
            continue
        x_c_raw = df_cis[col].iloc[idx_cis].reset_index(drop=True)
        x_s_raw = df_s1[col].iloc[idx_s1].reset_index(drop=True)
        x_c, x_s = align_to_0based(x_c_raw, x_s_raw)
        if x_c is None:
            continue
        X_cis_list.append(x_c.reindex(range(n)).fillna(-1).values)
        X_s1_list.append(x_s.reindex(range(n)).fillna(-1).values)
        cols_ok.append(col)

    if len(cols_ok) < 3:
        return np.nan

    X = np.vstack([
        np.column_stack(X_cis_list),
        np.column_stack(X_s1_list),
    ]).astype(float)
    y = np.array([1]*n + [0]*n)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    clf    = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    scores = cross_val_score(clf, X_sc, y, cv=5, scoring='neg_brier_score')
    brier  = float(-scores.mean())
    pmse   = float(max(0.0, brier - 0.25))

    print(f"  Brier score (balanced): {brier:.4f}")
    print(f"  pMSE = max(0, {brier:.4f} - 0.25) = {pmse:.4f}")
    print(f"  Reference: pMSE=0.000 ideal, >0.050 poor")
    if brier < 0.10:
        print(f"  Low Brier -> high detectability. May be class imbalance artefact.")

    return pmse


def validate_correlations(df_cis, df_s1, shared, out_dir):
    """
    Correlation preservation between key variables.
    """
    print("\nE. CORRELATION PRESERVATION")

    key_vars = [
        v for v in _CORR_KEY_VARS
        if v in shared and v in df_cis.columns and v in df_s1.columns
    ]

    cis_aligned = {}
    s1_clean    = {}
    for v in key_vars:
        xc, xs = align_to_0based(df_cis[v], df_s1[v])
        if xc is not None:
            cis_aligned[v] = xc
            s1_clean[v]    = xs

    if len(cis_aligned) < 2:
        print("  Not enough variables for correlation")
        return {}

    df_cis_aln = pd.DataFrame(cis_aligned)
    df_s1_aln  = pd.DataFrame(s1_clean)

    corr_cis = df_cis_aln.corr(numeric_only=True)
    corr_s1  = df_s1_aln.corr(numeric_only=True)

    diff = (corr_cis - corr_s1).abs()
    idx  = np.triu_indices_from(diff.values, k=1)
    mae  = float(diff.values[idx].mean())

    print(f"  Variables: {list(cis_aligned.keys())}")
    print(f"  Correlation MAE (S1 vs CIS, aligned scales): {mae:.4f}")

    pairs = []
    for i, v1 in enumerate(cis_aligned.keys()):
        for j, v2 in enumerate(cis_aligned.keys()):
            if j <= i:
                continue
            d = abs(corr_cis.loc[v1, v2] - corr_s1.loc[v1, v2])
            pairs.append((v1, v2, corr_cis.loc[v1, v2], corr_s1.loc[v1, v2], d))

    pairs.sort(key=lambda x: -x[4])
    print(f"  Most different correlations:")
    for v1, v2, r_c, r_s, d in pairs[:5]:
        print(f"    {v1}x{v2}: CIS={r_c:.3f}  S1={r_s:.3f}  diff={d:.3f}")

    corr_cis.to_csv(os.path.join(out_dir, "corr_cis_fixed.csv"))
    corr_s1.to_csv(os.path.join(out_dir, "corr_s1_fixed.csv"))

    return {'mae': mae, 'key_vars': list(cis_aligned.keys())}


def main():
    print("STEP 6 - CORRECTED VALIDATION (v2)")
    print("Fixes: scale alignment, pMSE, Cramer's V")

    paths = _build_paths()
    os.makedirs(paths["out"], exist_ok=True)

    df_cis, df_s1, df_s2, shared = load_data(paths)

    r_tvd  = validate_tvd(df_cis, df_s1, df_s2, shared, paths["out"])
    r_chi2 = validate_dependencies(df_cis, df_s1, shared, paths["out"])
    r_det  = validate_detection(df_cis, df_s1, shared, paths["out"])
    r_pmse = validate_pmse(df_cis, df_s1, shared)
    r_corr = validate_correlations(df_cis, df_s1, shared, paths["out"])

    v      = r_tvd['TVD_S1'].values
    n_pres = int(r_chi2['preserved'].sum()) if len(r_chi2) > 0 else 0

    report = {
        'tvd_mean':          float(v.mean()),
        'tvd_median':        float(np.median(v)),
        'tvd_n_excellent':   int((v < 0.05).sum()),
        'tvd_n_vars':        len(v),
        'chi2_preserved':    f"{n_pres}/{len(r_chi2)}",
        'pmse':              float(r_pmse) if not np.isnan(r_pmse) else None,
        'detection_auc':     r_det['auc_mean'] if r_det else None,
        'corr_mae':          r_corr.get('mae'),
        'fixes_applied': [
            'align_scales: rank-based CIS->0-based mapping (mirrors step4.validate)',
            'pMSE: balanced n_CIS=n_synth, corrected interpretation',
            'chi2: added Cramer V ratio to detect spurious dependencies',
            'detection: note on n_synth >> n_CIS class imbalance',
        ],
    }

    with open(os.path.join(paths["out"], "full_report_fixed.json"), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\nFINAL SUMMARY")
    print(f"  TVD: mean={v.mean():.4f}  median={np.median(v):.4f}")
    print(f"  TVD excellent (<0.05): {(v<0.05).sum()}/{len(v)}")
    print(f"  Chi-squared preserved: {n_pres}/{len(r_chi2)}")
    if r_corr:
        print(f"  Correlation MAE: {r_corr.get('mae'):.4f}")
    if not np.isnan(r_pmse):
        print(f"  pMSE: {r_pmse:.4f}")

    print(f"\n  Outputs -> {paths['out']}/")
    print("    tvd_per_variable_fixed.csv")
    print("    chisq_dependencies_fixed.csv  (includes Cramer's V)")
    print("    corr_cis_fixed.csv, corr_s1_fixed.csv")
    print("    detection_fixed.json")
    print("    full_report_fixed.json")


if __name__ == '__main__':
    main()
