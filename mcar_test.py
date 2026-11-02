"""Little's MCAR test for each variable block of the structure learned in step1.

The blocks and the analysis variables are rebuilt with the functions of
step1.py (same loader, substantive filter, collinearity filter and block
assignment), without running the MCMC. Each block is tested on the original
CIS codes, treated as numeric, after removing the missing codes; a block with
no missing values, or with a single missing-data pattern, has no test.

Little, R. J. A. (1988). A test of missing completely at random for
multivariate data with missing values. JASA 83, 1198-1202.

Output: mcar_blocks.csv (block, n_vars, n_obs, n_patterns, statistic, df,
p_value, rejected) and mcar_summary.json (blocks tested, blocks rejected at
alpha). Reported in the discussion of missing data.

Usage
-----
    python mcar_test.py [--alpha 0.05] [--out .]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

import step1

EM_MAX_ITER = 500
EM_TOL = 1e-8
RIDGE = 1e-9


def em_mvnorm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood mean and covariance of a normal model with missing values."""
    n, p = x.shape
    observed = ~np.isnan(x)
    keys = pd.Series([tuple(r) for r in observed])
    groups = [(np.array(k), np.asarray(rows)) for k, rows in keys.groupby(keys).groups.items()]
    mu = np.nanmean(x, axis=0)
    sigma = np.diag(np.nanvar(x, axis=0)) + RIDGE * np.eye(p)
    for _ in range(EM_MAX_ITER):
        sum_x = np.zeros(p)
        sum_xx = np.zeros((p, p))
        for o, rows in groups:
            m = ~o
            xg = x[rows].copy()
            cg = np.zeros((p, p))
            if m.any():
                s_oo_inv = np.linalg.inv(sigma[np.ix_(o, o)])
                s_mo = sigma[np.ix_(m, o)]
                xg[:, m] = mu[m] + (xg[:, o] - mu[o]) @ (s_mo @ s_oo_inv).T
                cg[np.ix_(m, m)] = sigma[np.ix_(m, m)] - s_mo @ s_oo_inv @ s_mo.T
            sum_x += xg.sum(axis=0)
            sum_xx += xg.T @ xg + len(rows) * cg
        mu_new = sum_x / n
        sigma_new = sum_xx / n - np.outer(mu_new, mu_new) + RIDGE * np.eye(p)
        converged = (np.max(np.abs(mu_new - mu)) < EM_TOL
                     and np.max(np.abs(sigma_new - sigma)) < EM_TOL)
        mu, sigma = mu_new, sigma_new
        if converged:
            break
    return mu, sigma


def little_mcar(data: pd.DataFrame) -> dict:
    """Little's chi-squared statistic, degrees of freedom and p-value."""
    data = data.dropna(how="all")
    x = data.to_numpy(dtype=float)
    n, p = x.shape
    patterns = pd.Series([tuple(r) for r in ~np.isnan(x)])
    n_patterns = patterns.nunique()
    result = {"n_vars": p, "n_obs": n, "n_patterns": int(n_patterns),
              "statistic": np.nan, "df": 0, "p_value": np.nan}
    if n_patterns < 2:
        return result
    mu, sigma = em_mvnorm(x)
    statistic, dof = 0.0, 0
    for pattern, rows in patterns.groupby(patterns).groups.items():
        o = np.array(pattern)
        diff = x[np.asarray(rows)][:, o].mean(axis=0) - mu[o]
        statistic += len(rows) * diff @ np.linalg.inv(sigma[np.ix_(o, o)]) @ diff
        dof += int(o.sum())
    dof -= p
    result.update(statistic=float(statistic), df=dof, p_value=float(chi2.sf(statistic, dof)))
    return result


def blocks_from_step1() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Analysis data (codes, missing as NaN) and the variables of each block."""
    with contextlib.redirect_stdout(io.StringIO()):
        df = step1.load_microdata_cis(step1.PATH_NUM)
        df = step1.filter_substantive_variables(df)
        df = step1.filter_collinear(df, threshold=0.95)
        var_types = step1.classify_variables(df)
        df_disc = step1.discretize_for_bdeu(df, var_types, n_bins=5)
        blocks, _ = step1.assign_blocks(df_disc)
    return df, {name: sorted(v for v in variables if v in df.columns)
                for name, variables in sorted(blocks.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()

    df, blocks = blocks_from_step1()
    rows = []
    for name, variables in blocks.items():
        rows.append({"block": name} | little_mcar(df[variables]))
    table = pd.DataFrame(rows)
    table["tested"] = table["n_patterns"] >= 2
    table["rejected"] = table["tested"] & (table["p_value"] < args.alpha)
    table.to_csv(args.out / "mcar_blocks.csv", index=False)

    summary = {
        "alpha": args.alpha,
        "blocks": int(len(table)),
        "blocks_tested": int(table["tested"].sum()),
        "blocks_rejected": int(table["rejected"].sum()),
    }
    (args.out / "mcar_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(table.round(4).to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
