"""Audit of the language-model extraction against the human review.

Compares the entries proposed by the language model (category_mappings.json)
for the five target outcomes with the corrections recorded by the human
reviewer (evidence_rules.csv), and classifies the corrections by error type.

Outputs (in --out):
    extraction_audit.csv    one row per error type: entries, share of proposed
                            entries, action taken, example
    extraction_audit.json   totals: entries extracted, entries proposed for the
                            target outcomes, accepted as proposed, corrected,
                            excluded

Usage
-----
    python extraction_audit.py --out transfer_output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TARGETS = {
    "acceptance_general", "acceptance_food_packaging", "perception_safety",
    "perception_quality", "price_willingness_to_pay_premium",
}
ERROR_LABELS = {
    "construct": "Construct mismatch",
    "polarity": "Reversed polarity",
    "double_counting": "Double counting",
    "scale": "Scale handling",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mappings", type=Path, default=Path("category_mappings.json"))
    parser.add_argument("--rules", type=Path, default=Path("evidence_rules.csv"))
    parser.add_argument("--out", type=Path, default=Path("transfer_output"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    studies = json.loads(args.mappings.read_text(encoding="utf-8"))
    entries = [(s["study_id"], m["cis_target"], m["source_variable"])
               for s in studies for m in s["mappings"]]
    proposed = [e for e in entries if e[1] in TARGETS]

    rules = pd.read_csv(args.rules)
    rules = rules[[tuple(k) in set(proposed) for k in rules[["study_id", "cis_target", "source_variable"]].values]]

    rows = []
    for error_type, group in rules.groupby("error_type", sort=False):
        rows.append({
            "error_type": ERROR_LABELS[error_type],
            "entries": len(group),
            "share_of_proposed": len(group) / len(proposed),
            "action": "; ".join(sorted(group["action"].unique())),
            "example": group["reason"].iloc[0],
        })
    table = pd.DataFrame(rows).sort_values("entries", ascending=False)
    table.to_csv(args.out / "extraction_audit.csv", index=False)

    excluded = int((rules["action"] == "exclude").sum())
    corrected = int((rules["action"] != "exclude").sum())
    summary = {
        "entries_extracted": len(entries),
        "source_records": len(studies),
        "proposed_for_targets": len(proposed),
        "accepted_as_proposed": len(proposed) - excluded - corrected,
        "corrected": corrected,
        "excluded": excluded,
        "retained": len(proposed) - excluded,
    }
    (args.out / "extraction_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
