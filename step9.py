"""
STEP 9 - EXTERNAL HOLDOUT VALIDATION: BVS S1 vs IPSOS SPAIN

Semantic remapping (corrected):
    - Ban Support -> acceptance_general (closer conceptual match)
    - wtp_taxes excluded due to documented construct mismatch

Reference: Ipsos (2022), Spain, n=1000, representative sample.
"""

import os
import json
import numpy as np
import pandas as pd


def _build_paths():
    base = os.path.dirname(os.path.abspath(__file__))
    return {
        "base": base,
        "s1":   os.path.join(base, "step4_output", "synthetic_surveys_S1.csv"),
        "out":  os.path.join(base, "step9_output"),
    }


# Ground truth: BVS variable -> (Ipsos label, Spain value, note)
_GROUND_TRUTH = {
    'attitudes_env_concern': (
        'International Treaty Support',
        0.90,
        'Direct semantic match: env. concern ~ treaty importance',
    ),
    'attitudes_wtp_prices': (
        'Manufacturer Responsibility Support',
        0.86,
        'Direct semantic match: price WTP ~ manufacturer responsibility',
    ),
    'acceptance_general': (
        'Single-Use Plastics Ban Support',
        0.78,
        'Remapped: ban support ~ general acceptance of plastic regulation '
        '(closer than wtp_taxes; ban support correlates negatively with '
        'tax WTP per Carattini et al. 2018)',
    ),
}

# Excluded variable with documented reason
_EXCLUDED = {
    'attitudes_wtp_taxes': (
        'Single-Use Plastics Ban Support',
        0.78,
        'EXCLUDED - construct mismatch: "ban support" (regulatory preference) '
        '!= "wtp taxes" (economic sacrifice). Negative correlation documented '
        'in Steg et al. (2015) and Carattini et al. (2018). Prior distribution '
        'reflects CIS V26 (tax WTP) which is conceptually distinct.',
    ),
}


def compute_validation(s1, rng, n_boot=2000):
    """
    For each ground-truth variable, compute top-3 proportion,
    bootstrap 95% CI, and absolute error against Ipsos Spain.
    """
    results = []

    for var, (ipsos_name, ipsos_val, note) in _GROUND_TRUTH.items():
        if var not in s1.columns:
            print(f"  [WARN] {var}: not found in S1")
            continue

        vals   = s1[var].dropna().astype(int)
        K      = int(vals.max()) + 1
        counts = np.bincount(vals.values, minlength=K)[:K].astype(float)
        props  = counts / counts.sum()

        # Top-3 categories = "agree" in Ipsos (strongly + agree + somewhat agree)
        top3 = float(props[-3:].sum())
        top2 = float(props[-2:].sum())
        err  = float(abs(top3 - ipsos_val))

        boot_top3 = []
        for _ in range(n_boot):
            sample = rng.choice(vals.values, size=len(vals), replace=True)
            bc     = np.bincount(sample, minlength=K)[:K].astype(float)
            bp     = bc / bc.sum()
            boot_top3.append(float(bp[-3:].sum()))

        ci_low  = float(np.percentile(boot_top3, 2.5))
        ci_high = float(np.percentile(boot_top3, 97.5))
        covered = bool(ci_low <= ipsos_val <= ci_high)

        results.append({
            'bvs_variable':   var,
            'ipsos_variable': ipsos_name,
            'ipsos_spain':    float(ipsos_val),
            'bvs_top3':       round(top3, 4),
            'bvs_top2':       round(top2, 4),
            'error_abs':      round(err,  4),
            'ci_95':          [round(ci_low, 4), round(ci_high, 4)],
            'covered':        covered,
            'K':              int(K),
            'props':          [round(float(p), 4) for p in props],
            'note':           note,
        })

        status  = "OK" if err < 0.10 else "WARN"
        cov_str = "covered" if covered else "not covered"
        print(f"\n  [{status}] {ipsos_name}")
        print(f"    Ipsos Spain:  {ipsos_val:.0%}")
        print(f"    BVS top-3:    {top3:.1%}  (top-2: {top2:.1%})")
        print(f"    Abs error:    {err:.1%}")
        print(f"    CI 95%:       [{ci_low:.1%}, {ci_high:.1%}]  {cov_str}")
        print(f"    Distribution: {[round(float(p), 3) for p in props]}")

    return results


def report_excluded(s1):
    """Print excluded variable diagnostics for the paper footnote."""
    print("\n  EXCLUDED - construct mismatch:")
    for var, (ipsos_name, ipsos_val, note) in _EXCLUDED.items():
        if var not in s1.columns:
            continue
        vals   = s1[var].dropna().astype(int)
        K      = int(vals.max()) + 1
        counts = np.bincount(vals.values, minlength=K)[:K].astype(float)
        props  = counts / counts.sum()
        top3   = float(props[-3:].sum())
        print(f"    {var}: BVS={top3:.1%} vs Ipsos={ipsos_val:.0%} "
              f"(err={abs(top3 - ipsos_val):.1%})")
        print(f"    Reason: {note[:120]}...")


def print_summary_table(results):
    """Print a formatted summary table."""
    print("\nVALIDATION TABLE: BVS S1 vs IPSOS SPAIN")
    print(f"\n  {'Ipsos Variable':<38} {'Ipsos':>6} {'BVS':>6} "
          f"{'Err':>5} {'CI 95%':>18} {'Cov':>4}")
    print(f"  {'-'*76}")
    for r in results:
        ci  = f"[{r['ci_95'][0]:.1%},{r['ci_95'][1]:.1%}]"
        cov = "yes" if r['covered'] else "no"
        print(f"  {r['ipsos_variable']:<38} {r['ipsos_spain']:>5.0%} "
              f"{r['bvs_top3']:>6.1%} {r['error_abs']:>4.1%} {ci:>18} {cov:>4}")


def compute_aggregate_metrics(results):
    errors   = [r['error_abs'] for r in results]
    rmse     = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    mae      = float(np.mean(errors))
    coverage = float(np.mean([r['covered'] for r in results]))
    return rmse, mae, coverage


def save_outputs(results, rmse, mae, coverage, paths):
    os.makedirs(paths["out"], exist_ok=True)

    output = {
        'method':          'External holdout validation: BVS S1 vs Ipsos 2022 Spain',
        'source_bvs':      'step4_output/synthetic_surveys_S1.csv',
        'source_ipsos':    'Ipsos 2022, Spain, n=1000, representative=True',
        'scale_conversion': 'Ipsos "agree" = top-3 of K=5 BVS Likert scale',
        'semantic_remapping': {
            'acceptance_general': 'Remapped from Ban Support (closer than wtp_taxes)',
            'rationale': 'Ban support ~ regulatory acceptance; '
                         'negative correlation with tax WTP (Carattini 2018)',
        },
        'excluded_variables': {
            var: {'reason': note}
            for var, (_, _, note) in _EXCLUDED.items()
        },
        'metrics': {
            'RMSE':           rmse,
            'MAE':            mae,
            'coverage_95ci':  coverage,
            'n_variables':    len(results),
        },
        'per_variable': results,
    }

    json_path = os.path.join(paths["out"], "ipsos_validation.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    df_out = pd.DataFrame([{
        'Variable (Ipsos)': r['ipsos_variable'],
        'Variable (BVS)':   r['bvs_variable'],
        'Ipsos Spain':      f"{r['ipsos_spain']:.0%}",
        'BVS (top-3)':      f"{r['bvs_top3']:.1%}",
        'Error abs.':       f"{r['error_abs']:.1%}",
        'CI 95%':           f"[{r['ci_95'][0]:.1%}, {r['ci_95'][1]:.1%}]",
        'Covered':          r['covered'],
    } for r in results])

    csv_path = os.path.join(paths["out"], "ipsos_validation_table.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def main():
    print("STEP 9 - EXTERNAL HOLDOUT VALIDATION: BVS S1 vs IPSOS SPAIN")

    paths = _build_paths()

    s1 = pd.read_csv(paths["s1"])
    print(f"\n  S1 shape: {s1.shape}")

    rng     = np.random.default_rng(42)
    results = compute_validation(s1, rng)

    report_excluded(s1)

    print_summary_table(results)

    rmse, mae, coverage = compute_aggregate_metrics(results)

    print(f"\n  Aggregate metrics (n={len(results)} variables):")
    print(f"    RMSE:     {rmse:.4f}  ({rmse:.1%})")
    print(f"    MAE:      {mae:.4f}  ({mae:.1%})")
    print(f"    Coverage: {coverage:.0%}")

    if rmse < 0.05:
        verdict = "EXCELLENT"
    elif rmse < 0.10:
        verdict = "GOOD"
    else:
        verdict = "REVIEW"
    print(f"    Verdict: {verdict}")

    print(f"\n  Paper footnote:")
    print(f"    attitudes_wtp_taxes excluded: construct mismatch.")
    print(f"    'Ban support' (Ipsos) measures regulatory preference,")
    print(f"    not willingness to pay taxes (CIS V26).")
    print(f"    Carattini et al. (2018) document negative correlation")
    print(f"    between both attitudes. Expected divergence, not model failure.")

    save_outputs(results, rmse, mae, coverage, paths)

    print(f"\n  RMSE={rmse:.4f} | MAE={mae:.4f} | Coverage={coverage:.0%} | {verdict}")


if __name__ == '__main__':
    main()
