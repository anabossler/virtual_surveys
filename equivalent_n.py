"""Equivalent target sample size of transferred (synthetic) evidence.

For each stress scenario (backtest_stress.py), finds the size n of a simple
random sample from the target country whose error equals the transfer error,
for levels and for gaps between segments.

The small-sample error is computed by backtest_diagnostics.py in two ways that
bracket the truth:
    overlapping  the small sample is part of the reference sample, which makes
                 it look too good as n approaches the target's sample size;
                 only sizes up to --max-overlap (default 200, at most about
                 15% of a national ISSP sample) are used
    disjoint     the reference is computed from the remaining respondents,
                 whose own sampling error makes the small sample look too bad
The equivalent n is read from each curve by log-log interpolation between
adjacent sizes and reported as a range [overlapping, disjoint]. Values beyond
the sizes computed are reported as bounds. The interval uses the 95% bootstrap
interval of the transfer error.

Usage:
    python equivalent_n.py
Output: transport_output/equivalent_n.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def curve(path: Path, max_n: int | None) -> tuple[np.ndarray, dict]:
    small = json.loads(path.read_text())["small_target_sample"]
    ns = np.array(sorted(int(k[1:]) for k in small if max_n is None or int(k[1:]) <= max_n))
    return ns, {q: np.array([small[f"n{n}"][q] for n in ns]) for q in ("level", "gap")}


def invert(ns: np.ndarray, err: np.ndarray, e: float):
    """n at which the (decreasing) error curve equals e; None and a bound outside the range."""
    if e > err[0]:
        return None, f"< {ns[0]}"
    if e < err[-1]:
        return None, f"> {ns[-1]}"
    n = float(np.exp(np.interp(-np.log(e), -np.log(err), np.log(ns))))
    return n, f"{n:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--overlap-json", type=Path, default=Path("transport_output/backtest_diagnostics_europe_sizes.json"))
    ap.add_argument("--disjoint-json", type=Path,
                    default=Path("transport_output/backtest_diagnostics_europe_sizes_disjoint.json"))
    ap.add_argument("--stress-json", type=Path, default=Path("transport_output/backtest_stress_summary.json"))
    ap.add_argument("--max-overlap", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("transport_output/equivalent_n.json"))
    args = ap.parse_args()

    curves = {"overlapping": curve(args.overlap_json, args.max_overlap), "disjoint": curve(args.disjoint_json, None)}
    stress = json.loads(args.stress_json.read_text())["scenarios"]
    result = {"curves": {k: {"sizes": v[0].tolist(), **{q: e.round(2).tolist() for q, e in v[1].items()}}
                         for k, v in curves.items()},
              "scenarios": {}}
    for sc, v in stress.items():
        result["scenarios"][sc] = {}
        for q in ("level", "gap"):
            e = v["all"][q]["mae_fixed"]
            lo, hi = v["ci95"][q]["mae_fixed"]
            row = {"mae": e, "mae_ci95": [lo, hi]}
            for name, (ns, err) in curves.items():
                n, lab = invert(ns, err[q], e)
                _, lab_hi_err = invert(ns, err[q], lo)   # lower error -> larger n
                _, lab_lo_err = invert(ns, err[q], hi)
                row[name] = {"n": n, "label": lab, "ci95": [lab_lo_err, lab_hi_err]}
            result["scenarios"][sc][q] = row
    for sc, v in result["scenarios"].items():
        line = [f"{sc:12s}"]
        for q in ("level", "gap"):
            o, d = v[q]["overlapping"], v[q]["disjoint"]
            line.append(f"{q}: {o['label']}-{d['label']} (CI {o['ci95'][0]}..{d['ci95'][1]})")
        print("  ".join(line))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("->", args.out)


if __name__ == "__main__":
    main()
