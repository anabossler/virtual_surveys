"""
STEP 9 - EXTERNAL COMPARISON: BVS vs IPSOS SPAIN

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


# Ground truth: CIS variable -> (Ipsos label, Spain value, note).
# Only one Ipsos item has a counterpart in CIS 3391. The scale of V15 runs from
# 1 (not at all worried) to 5 (very worried); 8 and 9 are non-substantive.
_GROUND_TRUTH = {
    'V15': (
        'International Treaty Support',
        0.90,
        'Loose match: environmental concern (top three of five) vs treaty support',
    ),
}
_TOP3_CODES = {3, 4, 5}
_VALID_CODES = {1, 2, 3, 4, 5}

# Items excluded because no CIS variable measures the same construct.
_EXCLUDED = {
    'Manufacturer Responsibility Support': 'no CIS counterpart; V27 is willingness to pay higher taxes',
    'Single-Use Plastics Ban Support': 'no CIS counterpart',
    'Minimal Packaging Preference': 'no CIS counterpart',
}


def compute_validation(s1, rng, n_boot=2000):
    """Share in the top three valid categories, bootstrap 95% CI and error."""
    from step4 import Config, discretise, load_cis_microdata

    cfg = Config()
    _, binfo = discretise(load_cis_microdata(cfg), cfg.N_BINS)
    results = []

    for var, (ipsos_name, ipsos_val, note) in _GROUND_TRUTH.items():
        labels = binfo[var]['labels']
        codes = s1[var].dropna().astype(int).map(lambda i: labels[i])
        valid = codes[codes.isin(_VALID_CODES)].to_numpy()
        top3 = float(np.isin(valid, list(_TOP3_CODES)).mean())
        err = float(abs(top3 - ipsos_val))
        boot = [
            float(np.isin(rng.choice(valid, size=len(valid), replace=True), list(_TOP3_CODES)).mean())
            for _ in range(n_boot)
        ]
        ci_low, ci_high = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        results.append({
            'bvs_variable':   var,
            'ipsos_variable': ipsos_name,
            'ipsos_spain':    float(ipsos_val),
            'bvs_top3':       round(top3, 4),
            'error_abs':      round(err, 4),
            'ci_95':          [round(ci_low, 4), round(ci_high, 4)],
            'covered':        bool(ci_low <= ipsos_val <= ci_high),
            'n_valid':        int(len(valid)),
            'note':           note,
        })
        print(f"  {ipsos_name}: Ipsos {ipsos_val:.0%} | BVS top-3 {top3:.1%} | error {err:.1%}")

    return results


def report_excluded():
    """Print the Ipsos items that have no CIS counterpart."""
    print("\n  Excluded Ipsos items:")
    for item, reason in _EXCLUDED.items():
        print(f"    {item}: {reason}")


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
        'method':           'External comparison: BVS synthetic population vs Ipsos 2022 Spain',
        'source_bvs':       'step4_output/synthetic_surveys_S1.csv',
        'source_ipsos':     'Ipsos 2022, Spain, n=1000',
        'scale_conversion': 'Ipsos agreement vs share of V15 in categories 3-5 among valid answers (1-5)',
        'excluded_items':   _EXCLUDED,
        'metrics': {
            'RMSE':          rmse,
            'MAE':           mae,
            'coverage_95ci': coverage,
            'n_variables':   len(results),
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
    print("STEP 9 - EXTERNAL COMPARISON: BVS vs IPSOS SPAIN")

    paths = _build_paths()

    s1 = pd.read_csv(paths["s1"])
    print(f"\n  S1 shape: {s1.shape}")

    rng     = np.random.default_rng(42)
    results = compute_validation(s1, rng)

    report_excluded()

    print_summary_table(results)

    rmse, mae, coverage = compute_aggregate_metrics(results)

    print(f"\n  Aggregate metrics (n={len(results)} variables):")
    print(f"    RMSE:     {rmse:.4f}  ({rmse:.1%})")
    print(f"    MAE:      {mae:.4f}  ({mae:.1%})")
    print(f"    Coverage: {coverage:.0%}")

    save_outputs(results, rmse, mae, coverage, paths)


if __name__ == "__main__":
    main()
