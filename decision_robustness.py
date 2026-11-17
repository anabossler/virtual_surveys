"""Stability of the decision-relevant conclusions (Table tab:decisions).

For the 80 profile types (sex x under 30 x higher education x recycling x five
worry levels), draws the transfer inputs and records, per draw, the ordering
of the types, which predictor leads, and each type's acceptance class.

Two sets of draws, as in the paper:
    sampling    b = a0 * N(beta, se) with a0 = 0.80, p0 ~ Beta around 0.38
    transport   a0 ~ N(a0_FI-ES, sd), b = a0 * N(beta, se) + N(0, eta_sd),
                p0 ~ Beta around 0.38 and a probit level shift N(0, level_sd);
                a0, the coefficient noise and the level shift are drawn in
                the same iteration as beta and p0 (transfer_transport.py)

The profile and segment quantities do not depend on the Bayesian network, so
G and theta do not enter these draws.

Reported:
    rank        median (and mean) Spearman correlation of the 80 types with the central
                ordering; share of the central top quartile kept in the top
                quartile of each draw
    lead        share of draws in which worry has the largest effect over its
                coded range (centred, full contrast, per scale point);
                education leading among the four binary predictors; female
                above recycling; all five coefficients positive
    class       mean share of types that stay in their central class
                (<25%, 25-50%, >50%) under transport draws
    proxies     Spearman correlation between the central ordering and the
                ordering implied by the Spanish ISSP coefficients of each
                proxy outcome; Finland-Spain agreement on the same outcome;
                number of proxies with a positive Spanish effect per predictor

Usage: python decision_robustness.py --cis metadata/3391_num.csv
Output: transfer_output/decision_robustness.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from transfer_model import P0, P0_N, PREDICTORS, calibrate_intercept, load_design, worry_score

A0 = 0.80
N_DRAWS = 20_000
SEED = 2026
TYPES = list(itertools.product([0, 1], [0, 1], [0, 1], [0, 1], [1, 2, 3, 4, 5]))
WORRY_RANGE = {"centered": 2.0, "contrast": 1.0, "per_step": 4.0}


def type_matrix(coding: str) -> np.ndarray:
    return np.array([[f, u, h, r, worry_score(np.array([v]), coding)[0]] for f, u, h, r, v in TYPES])


def classes(p: np.ndarray) -> np.ndarray:
    return np.where(p < 0.25, 0, np.where(p <= 0.50, 1, 2))


def spearman_rows(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of a with ref."""
    ra = a.argsort(axis=1).argsort(axis=1).astype(float)
    rr = ref.argsort().argsort().astype(float)
    ra -= ra.mean(axis=1, keepdims=True)
    rr -= rr.mean()
    return (ra @ rr) / np.sqrt((ra**2).sum(axis=1) * (rr**2).sum())


def draw(beta, se, transport, rng, n, mode):
    b = rng.normal(beta, se, size=(n, len(beta)))
    p0 = rng.beta(P0 * P0_N + 0.5, (1 - P0) * P0_N + 0.5, size=n)
    if mode == "sampling":
        return A0 * b, p0, np.zeros(n)
    a0 = rng.normal(transport["a0_fi_es"]["estimate"], transport["a0_fi_es"]["sd"], size=(n, 1))
    eta = rng.normal(0.0, transport["coefficient_residual_fi_es"]["transport_sd"], size=b.shape)
    shift = rng.normal(0.0, transport["level_rms_fi_es"], size=n)
    return a0 * b + eta, p0, shift


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cis", type=Path, default=Path("metadata/3391_num.csv"))
    ap.add_argument("--coefficients", type=Path, default=Path("transfer_coefficients.csv"))
    ap.add_argument("--transport", type=Path, default=Path("transport_output/issp_transport.json"))
    ap.add_argument("--issp-coefficients", type=Path, default=Path("transport_output/issp_coefficients.csv"))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--out", type=Path, default=Path("transfer_output/decision_robustness.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    table = pd.read_csv(args.coefficients).set_index("predictor").loc[list(PREDICTORS)]
    beta, se = table["coefficient"].to_numpy(), table["std_error"].to_numpy()
    transport = json.loads(args.transport.read_text(encoding="utf-8"))
    design = load_design(args.cis, "centered")
    w = design["weight"].to_numpy()
    print(f"complete cases: {len(design)} (paper: 2,133)")

    result = {"draws": args.draws, "seed": SEED}
    t_central = type_matrix("centered")
    x_central = design[list(PREDICTORS)].to_numpy()
    c0 = calibrate_intercept(A0 * (x_central @ beta), P0, w)
    p_central = norm.cdf(c0 + A0 * (t_central @ beta))
    lin_central = t_central @ beta
    top = lin_central >= np.quantile(lin_central, 0.75)
    cls_central = classes(p_central)
    result["central_classes"] = [int((cls_central == k).sum()) for k in range(3)]

    for mode in ("sampling", "transport"):
        b, p0, shift = draw(beta, se, transport, rng, args.draws, mode)
        lin = b @ t_central.T                                     # draws x 80
        rho = spearman_rows(lin, lin_central)
        q75 = np.quantile(lin, 0.75, axis=1, keepdims=True)
        kept_top = ((lin >= q75) & top).sum(axis=1) / top.sum()
        res = {"rank_median": float(np.median(rho)), "rank_mean": float(rho.mean()), "rank_p05": float(np.percentile(rho, 5)),
               "top_quartile_kept": float(kept_top.mean()),
               "all_positive": float((b > 0).all(axis=1).mean())}
        for coding, rng_w in WORRY_RANGE.items():
            effect = np.column_stack([b[:, :4], b[:, 4] * rng_w])
            res[f"worry_leads_{coding}"] = float((effect.argmax(axis=1) == 4).mean())
        res["education_leads_binary"] = float((b[:, :4].argmax(axis=1) == 2).mean())
        res["female_above_recycling"] = float((b[:, 0] > b[:, 3]).mean())
        if mode == "transport":
            x_lin = b @ x_central.T                               # draws x n
            stay = np.empty(args.draws)
            for d in range(args.draws):
                c = calibrate_intercept(x_lin[d], p0[d], w) + shift[d]
                stay[d] = (classes(norm.cdf(c + lin[d])) == cls_central).mean()
            res["class_kept_mean"] = float(stay.mean())
        result[mode] = res

    issp = pd.read_csv(args.issp_coefficients)
    proxies = {}
    for outcome, g in issp.groupby("outcome"):
        es = g[g.country == "ES"].set_index("predictor").reindex(list(PREDICTORS))["beta"].to_numpy()
        fi = g[g.country == "FI"].set_index("predictor").reindex(list(PREDICTORS))["beta"].to_numpy()
        proxies[outcome] = {
            "rank_vs_central": float(spearmanr(t_central @ es, lin_central)[0]),
            "rank_fi_vs_es": float(spearmanr(t_central @ es, t_central @ fi)[0]),
            "same_sign_fi_es": int((np.sign(es) == np.sign(fi)).sum()),
            "es_positive": {p: bool(v > 0) for p, v in zip(PREDICTORS, es)},
        }
    result["proxies"] = proxies
    result["proxies_summary"] = {
        "rank_vs_central_mean": float(np.mean([v["rank_vs_central"] for v in proxies.values()])),
        "rank_vs_central_range": [float(min(v["rank_vs_central"] for v in proxies.values())),
                                  float(max(v["rank_vs_central"] for v in proxies.values()))],
        "rank_fi_vs_es_mean": float(np.mean([v["rank_fi_vs_es"] for v in proxies.values()])),
        "same_sign_fi_es": f"{sum(v['same_sign_fi_es'] for v in proxies.values())} of {5 * len(proxies)}",
        "es_positive_count": {p: int(sum(v["es_positive"][p] for v in proxies.values())) for p in PREDICTORS},
    }
    print(json.dumps({k: v for k, v in result.items() if k != "proxies"}, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("->", args.out)


if __name__ == "__main__":
    main()
