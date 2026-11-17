"""Equivalent target sample size of transferred evidence.

Fits the error of a simple random target sample as a power law in n,
log MAE = a + b log n, separately for levels and gaps, on the sizes computed by
backtest_diagnostics.py --sizes ..., and inverts it for the transfer error of
each stress scenario (backtest_stress.py). Values outside the range of sizes
fitted are reported as bounds. Intervals combine the bootstrap interval of the
transfer error with the uncertainty of the fitted curve (residual bootstrap
over sizes).

Usage:
    python equivalent_n.py --sizes-json transport_output/backtest_diagnostics_europe_sizes.json
Output: transport_output/equivalent_n.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 11


def fit(ns, err):
    b, a = np.polyfit(np.log(ns), np.log(err), 1)
    return a, b


def invert(a, b, e):
    return float(np.exp((np.log(e) - a) / b))


def label(n, lo, hi):
    if n < lo:
        return f"< {lo}"
    if n > hi:
        return f"> {hi}"
    return f"{n:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes-json", type=Path, default=Path("transport_output/backtest_diagnostics_europe_sizes.json"))
    ap.add_argument("--stress-json", type=Path, default=Path("transport_output/backtest_stress_summary.json"))
    ap.add_argument("--out", type=Path, default=Path("transport_output/equivalent_n.json"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    small = json.loads(args.sizes_json.read_text())["small_target_sample"]
    ns = np.array(sorted(int(k[1:]) for k in small))
    stress = json.loads(args.stress_json.read_text())["scenarios"]
    result = {"sizes": ns.tolist(), "curves": {}, "scenarios": {}}
    for q in ("level", "gap"):
        err = np.array([small[f"n{n}"][q] for n in ns])
        a, b = fit(ns, err)
        resid = np.log(err) - (a + b * np.log(ns))
        boots = [fit(ns, np.exp(a + b * np.log(ns) + rng.choice(resid, len(ns)))) for _ in range(args.boot)]
        result["curves"][q] = {"a": float(a), "b": float(b), "errors": dict(zip(ns.tolist(), err.round(2).tolist()))}
        for sc, v in stress.items():
            e = v["all"][q]["mae_fixed"]
            lo_e, hi_e = v["ci95"][q]["mae_fixed"]
            draws = [invert(ab, bb, rng.uniform(lo_e, hi_e)) for ab, bb in boots]
            n_hat = invert(a, b, e)
            ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
            result["scenarios"].setdefault(sc, {})[q] = {
                "mae": e, "n_equivalent": n_hat, "ci95": ci,
                "label": label(n_hat, ns.min(), ns.max()),
                "ci_label": [label(c, ns.min(), ns.max()) for c in ci],
            }
    for sc, v in result["scenarios"].items():
        print(f"{sc:12s} level: {v['level']['label']:>6} ({v['level']['ci_label'][0]}-{v['level']['ci_label'][1]})"
              f" | gap: {v['gap']['label']:>6} ({v['gap']['ci_label'][0]}-{v['gap']['ci_label'][1]})")
    print("curve exponents:", {q: round(float(c["b"]), 2) for q, c in result["curves"].items()}, "(-0.5 = square-root law)")
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("->", args.out)


if __name__ == "__main__":
    main()
