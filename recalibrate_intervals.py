"""Recalibrated transport intervals, validated out of region.

The transport intervals of the retrospective tests cover fewer observed values
than their nominal 95% (about 80-87% of gaps and 85-90% of levels). This script
recalibrates them and checks the recalibration on regions not used to estimate
it.

Recalibration. For every prediction, the scaled error is
    z = |observed - median| / half-width of the 95% transport interval.
The factor k is the 95th percentile of z, and the recalibrated interval is
median +/- k * half-width. k is estimated separately for levels and gaps in each
stress scenario.

Validation. Leave-one-region-out over the four European regions (Nordic,
Western, Southern, Central-Eastern): k is estimated on targets outside a region
and applied to targets inside it. Reported: coverage, median width, and
interval score (alpha = 0.05) before and after recalibration, all out of region.
Input: backtest_stress_cases_loro.csv (transport parameters also estimated
without the target's region), so neither the intervals nor k use the target's
region.

Spain. The Spanish projected gaps (women - men, higher - below higher education,
recycles often - rarely, extremely - not at all worried, under 30 - 30 and over)
are recomputed with transport error from the coefficient file given, and their
intervals are widened by the small-cross gap factor k, the scenario closest to
the application.

Usage:
    python recalibrate_intervals.py --cases transport_output/backtest_stress_cases_loro.csv \
        --cis metadata/3391_num.csv --coefficients transfer_coefficients_pp.csv
Output: transport_output/recalibration.json; the factor for backtest_update.py
--inflate is printed at the end.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest_stress import REGION

ALPHA = 0.05
SEED = 5


def interval_score(lo, hi, y):
    return (hi - lo) + (2 / ALPHA) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))


def summarise(g: pd.DataFrame, k: np.ndarray | float) -> dict:
    lo, hi = g["med"] - k * g["hw"], g["med"] + k * g["hw"]
    y = g["observed"]
    return {"coverage": round(100 * float(((lo <= y) & (y <= hi)).mean()), 1),
            "median_width": round(100 * float((hi - lo).median()), 1),
            "interval_score": round(100 * float(interval_score(lo, hi, y).mean()), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", type=Path, default=Path("transport_output/backtest_stress_cases_loro.csv"))
    ap.add_argument("--cis", type=Path, default=Path("metadata/3391_num.csv"))
    ap.add_argument("--coefficients", type=Path, default=Path("transfer_coefficients_pp.csv"))
    ap.add_argument("--transport", type=Path, default=Path("transport_output/issp_transport.json"))
    ap.add_argument("--out", type=Path, default=Path("transport_output/recalibration.json"))
    args = ap.parse_args()

    cases = pd.read_csv(args.cases)
    cases = cases.dropna(subset=["low_transport", "high_transport", "pred_transport"])
    cases["med"] = cases["pred_transport"]
    cases["hw"] = ((cases["high_transport"] - cases["low_transport"]) / 2).clip(lower=1e-4)
    cases["z"] = (cases["observed"] - cases["med"]).abs() / cases["hw"]
    cases["kind"] = np.where(cases["quantity"] == "level", "level", "gap")
    cases["region"] = cases["target"].map(REGION)

    result = {"input": str(args.cases), "alpha": ALPHA, "scenarios": {}}
    rows = []
    for (sc, kind), g in cases.groupby(["scenario", "kind"], sort=False):
        k_all = float(np.quantile(g["z"], 1 - ALPHA))
        k_cv = pd.Series(np.nan, index=g.index)
        for reg in g["region"].unique():
            inside = g["region"] == reg
            k_cv[inside] = float(np.quantile(g.loc[~inside, "z"], 1 - ALPHA))
        raw, cal = summarise(g, 1.0), summarise(g, k_cv)
        rec = {"k": round(k_all, 3), "k_by_held_out_region": {r: round(float(k_cv[g.region == r].iloc[0]), 3)
                                                             for r in g["region"].unique()},
               "raw": raw, "recalibrated_out_of_region": cal, "cases": int(len(g))}
        result["scenarios"].setdefault(sc, {})[kind] = rec
        rows.append({"scenario": sc, "quantity": kind, "k": k_all,
                     "cov_raw": raw["coverage"], "cov_recal": cal["coverage"],
                     "width_raw": raw["median_width"], "width_recal": cal["median_width"],
                     "IS_raw": raw["interval_score"], "IS_recal": cal["interval_score"]})
    table = pd.DataFrame(rows)
    print(table.round(2).to_string(index=False))

    # transfer of k across scenarios: k from 'same' applied to the others
    cross = {}
    for kind in ("level", "gap"):
        k_same = result["scenarios"]["same"][kind]["k"]
        for sc, g in cases[cases.kind == kind].groupby("scenario", sort=False):
            cross.setdefault(kind, {})[sc] = summarise(g, k_same)["coverage"]
    result["coverage_with_k_from_same"] = cross
    print("coverage with k estimated on 'same':", cross)

    # Spain: recalibrated gap intervals
    try:
        from measurement_plan import segments
        from transfer_model import N_DRAWS, P0, P0_N, PREDICTORS, calibrate_intercept, load_design
        tr = json.loads(args.transport.read_text())
        a0_mean, a0_sd = tr["a0_fi_es"]["estimate"], tr["a0_fi_es"]["sd"]
        eta_sd, level_sd = tr["coefficient_residual_fi_es"]["transport_sd"], tr["level_rms_fi_es"]
        tab = pd.read_csv(args.coefficients).set_index("predictor").loc[list(PREDICTORS)]
        beta, se = tab["coefficient"].to_numpy(), tab["std_error"].to_numpy()
        des = load_design(args.cis, "centered")
        x, w = des[list(PREDICTORS)].to_numpy(), des["weight"].to_numpy()
        m = segments(des)
        young = des["under_30"].to_numpy() == 1
        pairs = {"women - men": (m["Women"], m["Men"]),
                 "higher - below higher education": (m["Higher education"], m["Below higher education"]),
                 "recycles often - rarely": (m["Recycles always or often"], m["Recycles sometimes or never"]),
                 "extremely - not at all worried": (m["Extremely worried (V15 = 5)"], m["Not at all worried (V15 = 1)"]),
                 "under 30 - 30 and over": (young, ~young)}
        rng = np.random.default_rng(SEED)
        draws = {k: [] for k in pairs}
        for _ in range(N_DRAWS):
            a0 = rng.normal(a0_mean, a0_sd)
            b = a0 * rng.normal(beta, se) + rng.normal(0.0, eta_sd, len(beta))
            p0 = rng.beta(P0 * P0_N + 0.5, (1 - P0) * P0_N + 0.5)
            lin = x @ b
            pr = norm.cdf(calibrate_intercept(lin, p0, w) + rng.normal(0.0, level_sd) + lin)
            for k, (s1, s0) in pairs.items():
                draws[k].append(np.average(pr[s1], weights=w[s1]) - np.average(pr[s0], weights=w[s0]))
        k_gap = result["scenarios"]["small_cross"]["gap"]["k"]
        spain = {}
        for k, v in draws.items():
            v = np.array(v)
            med, lo, hi = np.median(v), *np.percentile(v, [2.5, 97.5])
            hw = (hi - lo) / 2
            spain[k] = {"median": round(100 * med, 1), "transport": [round(100 * lo, 1), round(100 * hi, 1)],
                        "recalibrated": [round(100 * (med - k_gap * hw), 1), round(100 * (med + k_gap * hw), 1)]}
        result["spain_gaps"] = {"k_small_cross_gap": k_gap, "coefficients": str(args.coefficients), "gaps": spain}
        print(f"\nSpanish gaps (points), transport and recalibrated (k = {k_gap:.2f}):")
        for k, v in spain.items():
            print(f"  {k:36s} {v['median']:6.1f}  {v['transport']}  {v['recalibrated']}")
    except FileNotFoundError as e:
        print("Spanish part skipped:", e)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("->", args.out)
    print(f"\nfor backtest_update.py use --inflate {result['scenarios']['small_cross']['gap']['k']:.2f}")


if __name__ == "__main__":
    main()
