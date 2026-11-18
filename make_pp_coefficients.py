"""Predictor-specific attenuation for the Spanish projection.

The central specification multiplies every published coefficient by a single
transfer factor a0 = 0.80. The ISSP proxies show that attenuation differs by
predictor: it is close to 0.80 for environmental worry, higher education and
recycling, but about 0.40 for gender and age, whose effects are small and
change sign between countries. This script estimates the per-predictor slope
of target on source coefficients over every ordered pair of the 16 European
countries and all five proxy outcomes (as in backtest_stress.predictor_slopes,
here without leaving anything out), and writes a coefficient file whose
coefficients and standard errors are rescaled by a_j / 0.80.

Every application script multiplies the coefficients by a0 = 0.80, so running
it with --coefficients transfer_coefficients_pp.csv applies a_j to predictor j
without changing any code. The a0 sweeps then scale the vector proportionally,
and the transport draws of a0 around the Finland-Spain slope (0.84) scale it by
about 1.05.

Inputs:  transport_output/issp_coefficients.csv (aggregate, from transport_issp.py)
Outputs: transfer_coefficients_pp.csv, transport_output/attenuation_by_predictor.json

Usage: python make_pp_coefficients.py
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

A0 = 0.80
EUROPE = ["AT", "CH", "DE", "DK", "ES", "FI", "FR", "HR", "HU", "IS", "IT", "LT", "NO", "SE", "SI", "SK"]


def main() -> None:
    coefs = pd.read_csv("transport_output/issp_coefficients.csv")
    coefs = coefs[coefs.country.isin(EUROPE)]
    slopes = {}
    for predictor, g in coefs.groupby("predictor"):
        b = g.pivot(index="country", columns="outcome", values="beta")
        num = den = 0.0
        for s, t in permutations(b.index, 2):
            xs, xt = b.loc[s].to_numpy(), b.loc[t].to_numpy()
            ok = ~(np.isnan(xs) | np.isnan(xt))
            num += float(xs[ok] @ xt[ok]); den += float(xs[ok] @ xs[ok])
        slopes[predictor] = num / den
    table = pd.read_csv("transfer_coefficients.csv")
    factor = table["predictor"].map(slopes) / A0
    table["coefficient"] = (table["coefficient"] * factor).round(4)
    table["std_error"] = (table["std_error"] * factor).round(4)
    table["status"] = table["status"] + "; rescaled by a_j/0.80 (predictor-specific attenuation)"
    table.to_csv("transfer_coefficients_pp.csv", index=False)
    out = {"a0_central": A0, "attenuation": {k: round(v, 3) for k, v in slopes.items()},
           "pairs": "all ordered pairs of 16 European countries, 5 ISSP proxy outcomes"}
    Path("transport_output/attenuation_by_predictor.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["attenuation"], indent=2))
    print("-> transfer_coefficients_pp.csv")


if __name__ == "__main__":
    main()
