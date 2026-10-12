"""Summarise the external evidence for the target outcomes after step2.py.

Reads step2_input_mappings.json and step2_dirichlet_priors.json and writes
target_evidence.csv: studies, entries, pooled n, independent respondents
(each study counted once, at its largest entry), I2 and the share of the
pooled prior in the low (1-2), middle (3) and high (4-5) categories.
"""

from __future__ import annotations

import json

import pandas as pd

TARGETS = [
    "acceptance_general",
    "acceptance_food_packaging",
    "perception_safety",
    "perception_quality",
    "price_willingness_to_pay_premium",
]
STUDY_ALIASES = {
    "widiastuti_2025_recycled_plastic_acceptance": "widiastuti_2025",
    "widiastuti_2025_recycled_plastics": "widiastuti_2025",
}


def main() -> None:
    with open("step2_input_mappings.json", encoding="utf-8") as handle:
        entries = json.load(handle)
    with open("step2_dirichlet_priors.json", encoding="utf-8") as handle:
        priors = json.load(handle)["priors"]

    rows = []
    for target in TARGETS:
        subset = [e for e in entries if e["cis_target"] == target]
        largest = {}
        for e in subset:
            study = STUDY_ALIASES.get(e["study_id"], e["study_id"])
            largest[study] = max(largest.get(study, 0), int(e["total_n"]))
        row = {
            "construct": target,
            "studies": len(largest),
            "entries": len(subset),
            "pooled_n": sum(int(e["total_n"]) for e in subset),
            "independent_n": sum(largest.values()),
            "I2_pct": "",
            "low_pct": "",
            "mid_pct": "",
            "high_pct": "",
            "study_ids": ";".join(sorted(largest)),
        }
        if target in priors and subset:
            alpha = priors[target]["alpha"]
            total = sum(alpha)
            row["I2_pct"] = round(priors[target]["I2_statistic"]["I2_pct"], 1)
            row["low_pct"] = round(100 * sum(alpha[:2]) / total)
            row["mid_pct"] = round(100 * alpha[2] / total)
            row["high_pct"] = round(100 * sum(alpha[3:]) / total)
        rows.append(row)
    pd.DataFrame(rows).to_csv("target_evidence.csv", index=False)
    print(pd.DataFrame(rows).drop(columns="study_ids").to_string(index=False))


if __name__ == "__main__":
    main()
