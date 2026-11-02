"""Independent second coding of the language-model extraction and inter-rater agreement.

Two steps:

    python second_coder.py sheet --out coding_sheet.xlsx
        Writes a blind coding sheet with the entries the language model proposed
        for the five target outcomes (category_mappings.json). The first
        reviewer's decisions (evidence_rules.csv) are not included.

    python second_coder.py kappa --sheet coding_sheet_filled.xlsx --out transfer_output
        Reads the sheet filled by the second coder and compares it with the
        first reviewer. Writes second_coder_agreement.json (percentage agreement
        and Cohen's kappa for the decision and for accept/change) and
        second_coder_disagreements.csv (entries coded differently).

Decision codes
    accept     use the entry as proposed
    reverse    use it with the scale reversed (polarity error)
    rescale    use it with a different scale handling (scale error)
    exclude    do not use it (construct mismatch or double counting)

Cohen, J. (1960). A coefficient of agreement for nominal scales.
Educ. Psychol. Meas. 20, 37-46.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

TARGETS = {
    "acceptance_general": "Acceptance of products made from recycled plastic in general. "
                          "Higher category = more accepting.",
    "acceptance_food_packaging": "Acceptance of recycled plastic used as food packaging. "
                                 "Higher category = more accepting.",
    "perception_safety": "Perceived safety of recycled plastic (health, contamination). "
                         "Higher category = perceived as safer.",
    "perception_quality": "Perceived quality of products made from recycled plastic. "
                          "Higher category = higher perceived quality.",
    "price_willingness_to_pay_premium": "Willingness to pay a higher price for products with "
                                        "recycled plastic. Higher category = more willing.",
}
DECISIONS = ["accept", "reverse", "rescale", "exclude"]
REASONS = ["none", "construct", "polarity", "scale", "double_counting"]
ACTION_TO_DECISION = {"reverse": "reverse", "binary_share": "rescale", "exclude": "exclude"}
KEY = ["study_id", "cis_target", "source_variable"]

INSTRUCTIONS = [
    "Purpose: check whether each entry proposed by the language model measures the target "
    "construct, with the right direction and scale, without counting the same respondents twice.",
    "Work alone. Do not consult the first reviewer or the manuscript.",
    "For each row, read the source item, the notes and the extracted statistic, and the "
    "definition of the proposed target (sheet 'targets').",
    "decision: accept = use as proposed; reverse = the item measures the target in the "
    "opposite direction (e.g. a risk statement for safety); rescale = right construct but the "
    "statistic must be mapped to the 5-point scale differently; exclude = wrong construct, or "
    "the same respondents and items already enter through another row.",
    "reason: none for accept; otherwise construct, polarity, scale or double_counting.",
    "comment: optional, one line.",
    "You may open the source article if the item wording is unclear (column study_id).",
]


def proposed_entries(mappings: Path) -> pd.DataFrame:
    studies = json.loads(mappings.read_text(encoding="utf-8"))
    rows = []
    for study in studies:
        for m in study["mappings"]:
            if m["cis_target"] not in TARGETS:
                continue
            values = m.get("extracted_values", {})
            rows.append({
                "study_id": study["study_id"],
                "cis_target": m["cis_target"],
                "source_variable": m["source_variable"],
                "notes": m.get("notes", ""),
                "source_scale": m.get("source_scale", ""),
                "data_type": m.get("data_type", ""),
                "extracted_statistic": json.dumps({k: v for k, v in values.items() if k != "n"}),
                "n": values.get("n", m.get("total_n", "")),
            })
    table = pd.DataFrame(rows).sort_values(KEY).reset_index(drop=True)
    table.insert(0, "entry_id", [f"E{i + 1:02d}" for i in range(len(table))])
    return table


def write_sheet(entries: pd.DataFrame, out: Path) -> None:
    from openpyxl.worksheet.datavalidation import DataValidation

    coding = entries.assign(decision="", reason="", comment="")
    targets = pd.DataFrame({"cis_target": list(TARGETS), "definition": list(TARGETS.values())})
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame({"instructions": INSTRUCTIONS}).to_excel(writer, sheet_name="instructions", index=False)
        targets.to_excel(writer, sheet_name="targets", index=False)
        coding.to_excel(writer, sheet_name="coding", index=False)
        sheet = writer.sheets["coding"]
        last = len(coding) + 1
        for column, options in (("decision", DECISIONS), ("reason", REASONS)):
            letter = chr(ord("A") + list(coding.columns).index(column))
            rule = DataValidation(type="list", formula1='"' + ",".join(options) + '"', allow_blank=False)
            sheet.add_data_validation(rule)
            rule.add(f"{letter}2:{letter}{last}")
        for name in ("instructions", "targets", "coding"):
            for col in writer.sheets[name].columns:
                writer.sheets[name].column_dimensions[col[0].column_letter].width = 28
    print(f"{len(coding)} entries written to {out}")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def agreement(entries: pd.DataFrame, rules: Path, sheet: Path, out: Path) -> None:
    first = pd.read_csv(rules)
    first["first_decision"] = first["action"].map(ACTION_TO_DECISION)
    first["first_reason"] = first["error_type"]
    second = pd.read_excel(sheet, sheet_name="coding")
    second["decision"] = second["decision"].astype(str).str.strip().str.lower()
    second["reason"] = second["reason"].astype(str).str.strip().str.lower()
    missing = second[~second["decision"].isin(DECISIONS)]
    if len(missing):
        raise SystemExit(f"invalid or empty decision in {list(missing['entry_id'])}")

    merged = (entries[["entry_id", *KEY]]
              .merge(first[[*KEY, "first_decision", "first_reason"]], on=KEY, how="left")
              .merge(second[["entry_id", "decision", "reason", "comment"]], on="entry_id", how="left"))
    merged["first_decision"] = merged["first_decision"].fillna("accept")
    merged["first_reason"] = merged["first_reason"].fillna("none")

    d1, d2 = merged["first_decision"].tolist(), merged["decision"].tolist()
    c1 = ["accept" if x == "accept" else "change" for x in d1]
    c2 = ["accept" if x == "accept" else "change" for x in d2]
    result = {
        "entries": len(merged),
        "decision_agreement": sum(x == y for x, y in zip(d1, d2)) / len(d1),
        "decision_kappa": cohen_kappa(d1, d2),
        "change_agreement": sum(x == y for x, y in zip(c1, c2)) / len(c1),
        "change_kappa": cohen_kappa(c1, c2),
        "first_changed": sum(x == "change" for x in c1),
        "second_changed": sum(x == "change" for x in c2),
        "confusion": pd.crosstab(merged["first_decision"], merged["decision"]).to_dict(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "second_coder_agreement.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    merged[merged["first_decision"] != merged["decision"]].to_csv(
        out / "second_coder_disagreements.csv", index=False)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--out", type=Path, default=Path("coding_sheet.xlsx"))
    k = sub.add_parser("kappa")
    k.add_argument("--sheet", type=Path, required=True)
    k.add_argument("--out", type=Path, default=Path("transfer_output"))
    for p in (s, k):
        p.add_argument("--mappings", type=Path, default=Path("category_mappings.json"))
    k.add_argument("--rules", type=Path, default=Path("evidence_rules.csv"))
    args = parser.parse_args()

    entries = proposed_entries(args.mappings)
    if args.command == "sheet":
        write_sheet(entries, args.out)
    else:
        agreement(entries, args.rules, args.sheet, args.out)


if __name__ == "__main__":
    main()
