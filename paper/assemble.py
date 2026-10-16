"""Assemble main.tex from section parts, filling table rows from the deposit.

Usage:
    python assemble.py <parts_dir> <deposit_dir> <audit_dir> <output_tex> <transfer_dir> <run_dir>
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

PART_ORDER = [
    "00_front.tex",
    "01_intro.tex",
    "02_theory.tex",
    "03_system_data.tex",
    "05_method.tex",
    "06_results.tex",
    "07_scenarios.tex",
    "08_implications.tex",
    "09_discussion.tex",
    "11_back.tex",
]
TARGET_OUTCOMES = [
    "acceptance_general",
    "acceptance_food_packaging",
    "perception_safety",
    "perception_quality",
    "price_willingness_to_pay_premium",
]
CONSTRUCT_LABELS = {
    "acceptance_general": "General acceptance",
    "acceptance_food_packaging": "Food-packaging acceptance",
    "perception_safety": "Perceived safety",
    "perception_quality": "Perceived quality",
    "price_willingness_to_pay_premium": "WTP premium",
    "attitudes_env_concern": "Environmental concern",
    "attitudes_wtp_prices": "WTP higher prices",
    "attitudes_wtp_taxes": "WTP higher taxes",
    "behavior_recycling": "Recycling",
    "behavior_reduce_consumption": "Reduced consumption",
    "trust_institutions": "Institutional trust",
    "demographics_age": "Age",
    "demographics_education": "Education",
    "demographics_gender": "Gender",
    "demographics_income": "Income",
    "demographics_urban_rural": "Urban/rural",
}
CIS_CODES = {
    "attitudes_env_concern": "V15",
    "attitudes_wtp_taxes": "V26",
    "behavior_recycling": "V52",
    "behavior_reduce_consumption": "V50",
    "demographics_age": "BIRTH",
    "demographics_education": "ESTUDIOS",
    "demographics_gender": "SEX",
    "demographics_urban_rural": "URBRURAL",
    "trust_institutions": "V10",
}
STUDIES = {
    "abella_2022_recycled_plastics_perception": (r"\citet{abella2025}", "Spain"),
    "cabrera_2025_plastic_apparel_stpb": (r"\citet{cabrera2025}", "Philippines"),
    "cao_lu_zhu_2022_eol_recycled_preferences": (r"\citet{cao2022}", "China"),
    "dicicco_2024_pcr_odor_acceptance": (r"\citet{dicicco2020}", "Netherlands"),
    "leal_filho_2025": (r"\citet{lealfilho2025}", "Europe (16)"),
    "neumann_2024_pcr_nudging_germany": (r"\citet{neumann2024}", "Germany"),
    "plasticircle_valencia_2019": (r"\citet{plasticircle2023}", "Spain"),
    "reijonen_2020": (r"\citet{reijonen2020}", "Finland"),
    "ruokamo_2022_recycled_plastics": (r"\citet{ruokamo2022}", "Finland"),
    "widiastuti_2025": (r"\citet{widiastuti2025}", "Indonesia"),
    "zwicker_2020_plastic_attitudes": (r"\citet{zwicker2020}", "Netherlands, UK"),
}
MODELS = [
    ("BVS (S1)", r"\textbf{BVS (proposed)}", "Yes"),
    ("BN-MLE (S2)", "BN-MLE", "No"),
    ("CTGAN", r"CTGAN \citep{xu2019}", "No"),
    ("TVAE", r"TVAE \citep{xu2019}", "No"),
    ("INDEP", "Independent", "No"),
]
PAIR_LABELS = {
    "gender": "Gender",
    "education": "Education",
    "age": "Age",
    "urban_rural": "Urban/rural",
    "env_concern": "env.\\ concern",
    "WTP taxes": "WTP taxes",
    "WTP prices": "WTP prices",
    "WTP": "WTP",
    "recycling": "recycling",
    "trust": "Trust",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def lower_initial(label: str) -> str:
    """Lower-case the first letter unless the label starts with an acronym."""
    return label if label[:2].isupper() else label[0].lower() + label[1:]


def i2_rows(constructs: list[dict]) -> str:
    ordered = [c for name in TARGET_OUTCOMES for c in constructs if c["construct"] == name]
    lines = []
    for index, c in enumerate(ordered):
        i2 = "n.e." if int(c["entries"]) == 1 else c["I2_pct"]
        lines.append(
            f"{CONSTRUCT_LABELS[c['construct']]} & {c['studies']} & {c['entries']} & "
            f"{thousands(int(c['pooled_n']))} & {thousands(int(c['independent_n']))} & {i2} \\\\"
        )
    return "\n".join(lines)


def comparison_rows(deposit: Path) -> str:
    table = {r["Model"]: r for r in read_csv(deposit / "04_validation/baselines/comparison_table.csv")}
    chi = read_csv(deposit / "04_validation/baselines/chisq_by_model.csv")
    model_based = [m for m, _, _ in MODELS if m != "INDEP"]
    best = {
        "TVD (mean)": min(float(table[m]["TVD (mean)"]) for m in model_based),
        "TVD (median)": min(float(table[m]["TVD (median)"]) for m in model_based),
        "TVD Exc. (%)": max(float(table[m]["TVD Exc. (%)"]) for m in model_based),
        "Corr MAE": min(float(table[m]["Corr MAE"]) for m in model_based),
    }

    def cell(model: str, column: str, fmt: str) -> str:
        value = float(table[model][column])
        text = format(value, fmt)
        return f"\\textbf{{{text}}}" if model in model_based and value == best[column] else text

    lines = []
    for model, label, external in MODELS:
        ratios = [float(r["V_ratio"]) for r in chi if r["model"] == model]
        kept = table[model]["Chi² pres."]
        lines.append(
            f"{label} & {cell(model, 'TVD (mean)', '.3f')} & {cell(model, 'TVD (median)', '.3f')} & "
            f"{cell(model, 'TVD Exc. (%)', '.1f')} & {cell(model, 'Corr MAE', '.4f')} & {kept} & "
            f"{min(ratios):.2f}--{max(ratios):.2f} & {external} \\\\"
        )
    return "\n".join(lines)


def dependency_rows(deposit: Path) -> str:
    lines = []
    for row in read_csv(deposit / "step6_output/chisq_dependencies_fixed.csv"):
        source, target = [part.strip() for part in re.split(r"→|->", row["pair"])]
        label = f"{PAIR_LABELS[source][0].upper()}{PAIR_LABELS[source][1:]} $\\to$ {PAIR_LABELS[target]}"
        kept = "Yes" if row["preserved"] == "True" else "No"
        lines.append(
            f"{label} & {float(row['V_CIS']):.3f} & {float(row['V_S1']):.3f} & {float(row['V_ratio']):.2f} & {kept} \\\\"
        )
    return "\n".join(lines)


def fusion_rows(deposit: Path) -> str:
    rows = read_csv(deposit / "04_validation/three_level/fusion_coherence.csv")
    lines = [
        f"{CONSTRUCT_LABELS[r['Variable']]} ({CIS_CODES[r['Variable']]}) & {float(r['Pct_between']):.0f}\\% & "
        f"{float(r['TVD_to_expected']):.3f} \\\\"
        for r in rows
    ]
    mean = sum(float(r["TVD_to_expected"]) for r in rows) / len(rows)
    lines += ["\\midrule", f"\\textbf{{Mean}} & \\textbf{{100\\%}} & \\textbf{{{mean:.3f}}} \\\\"]
    return "\n".join(lines)


def novel_rows(constructs: list[dict]) -> str:
    notes = {"acceptance_food_packaging": "a", "perception_safety": "b", "perception_quality": "c"}
    lines = []
    for name in TARGET_OUTCOMES:
        c = next(x for x in constructs if x["construct"] == name)
        label = CONSTRUCT_LABELS[name] + (f"\\tnote{{{notes[name]}}}" if name in notes else "")
        head = f"{label} & {c['studies']} & {c['entries']} & {thousands(int(c['independent_n']))}"
        if name == "acceptance_food_packaging":
            lines.append(f"{head} & \\multicolumn{{3}}{{c}}{{38\\% judge it suitable}} \\\\")
        else:
            low, mid, high = c["low_mid_high_pct"].split("/")
            lines.append(f"{head} & {low}\\% & {mid}\\% & {high}\\% \\\\")
    return "\n".join(lines)


def study_rows(studies: list[dict], run: Path) -> str:
    """One row per corpus study; the last column lists retained target evidence."""
    retained: dict[str, list[str]] = {}
    for c in read_csv(run / "target_evidence.csv"):
        for sid in filter(None, c["study_ids"].split(";")):
            retained.setdefault(sid, []).append(c["construct"])
    lines = []
    for s in studies:
        cite, country = STUDIES[s["study"]]
        targets = [t for t in TARGET_OUTCOMES if t in retained.get(s["study"], [])]
        text = ", ".join(lower_initial(CONSTRUCT_LABELS[t]) if i else CONSTRUCT_LABELS[t] for i, t in enumerate(targets))
        lines.append(f"{cite} & {country} & {thousands(int(s['n']))} & {s['entries']} & {text or 'None'} \\\\")
    return "\n".join(lines)


def lambda_rows(deposit: Path) -> str:
    ess = {r["variable"]: r for r in read_csv(deposit / "05_sensitivity/lambda/lambda_sensitivity_ess.csv")}
    summary = (deposit / "05_sensitivity/lambda/lambda_sensitivity_summary.txt").read_text(encoding="utf-8")
    rho = dict(re.findall(r"lambda=([\d.]+): Spearman rho vs lambda=0 = ([\d.]+)", summary))
    lines = []
    for lam in ("0.0", "0.5", "1.0", "2.0"):
        values = [
            thousands(round(float(ess[v][f"ESS_lam{lam}"])))
            for v in ("acceptance_general", "perception_safety", "demographics_gender")
        ]
        lines.append(f"{float(lam):.1f} & {' & '.join(values)} & {float(rho[lam]):.2f} \\\\")
    return "\n".join(lines)


SEGMENT_LABELS = {
    "gender": "Gender",
    "age_group": "Age",
    "education": "Education",
    "recycling": "Recycling",
    "environmental_worry": "Environmental worry (1--5)",
}


def sensitivity_rows(transfer: Path, parameter: str, label: str) -> str:
    rows = read_csv(transfer / f"sensitivity_{parameter}_centered.csv")
    central = {"a0": "0.8", "p0": "0.38"}[parameter]
    lines = []
    for index, r in enumerate(rows):
        cells = [
            f"{float(r[parameter]):.2f}",
            f"{float(r['most_receptive']):.2f}",
            f"{float(r['least_receptive']):.2f}",
            f"{float(r['spread_pp']):.0f}~pp",
        ]
        if r[parameter] == central:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(f"{label if index == 0 else ''} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def segment_rows(transfer: Path) -> str:
    lines, previous = [], None
    for r in read_csv(transfer / "segments_centered.csv"):
        label = SEGMENT_LABELS[r["dimension"]] if r["dimension"] != previous else ""
        if previous is not None and r["dimension"] != previous:
            lines.append("\\addlinespace")
        previous = r["dimension"]
        level = r["level"][0].upper() + r["level"][1:]
        lines.append(
            f"{label} & {level} & {thousands(int(r['n']))} & {100 * float(r['share_of_sample']):.0f}\\% & "
            f"{100 * float(r['projected_acceptance']):.0f}\\% \\\\"
        )
    return "\n".join(lines)


def segment_a0_rows(transfer: Path) -> str:
    rows = read_csv(transfer / "segments_a0.csv")
    grid = sorted({r["a0"] for r in rows}, key=float)
    lines = []
    for segment in dict.fromkeys(r["segment"] for r in rows):
        values = {r["a0"]: float(r["projected_acceptance"]) for r in rows if r["segment"] == segment}
        n = next(int(r["n"]) for r in rows if r["segment"] == segment)
        cells = [f"{100 * values[a]:.0f}\\%" for a in grid]
        lines.append(f"{segment} & {thousands(n)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def plan_rows(transfer: Path) -> str:
    transport = {r["quantity"]: r for r in read_csv(transfer / "transport_segments.csv")}
    lines = []
    for r in read_csv(transfer / "measurement_plan.csv"):
        t = transport[r["segment"]]
        lines.append(
            f"{r['segment']} & {100 * float(r['mean']):.0f} ({100 * float(r['ci_low']):.0f}--{100 * float(r['ci_high']):.0f}) & "
            f"{100 * float(t['median']):.0f} ({100 * float(t['ci_low']):.0f}--{100 * float(t['ci_high']):.0f}) & "
            f"{float(r['m']):.0f} & {float(t['m']):.0f} & {thousands(int(r['N_delta_1']))} & {thousands(int(t['N']))} \\\\"
        )
    return "\n".join(lines)


def transport_scalars(transfer: Path, transport: Path) -> dict[str, str]:
    t = json.loads((transport / "issp_transport.json").read_text())
    prof = {r["quantity"]: r for r in read_csv(transfer / "transport_profiles.csv")}
    seg = {r["quantity"]: r for r in read_csv(transfer / "transport_segments.csv")}

    def pct(row):
        return f"{100 * float(row['median']):.0f}\\% ({100 * float(row['ci_low']):.0f}--{100 * float(row['ci_high']):.0f})"

    res = t["coefficient_residual_fi_es"]
    return {
        "%%TA0%%": f"{t['a0_fi_es']['estimate']:.2f}",
        "%%TA0CI%%": f"{t['a0_fi_es']['ci_low']:.2f}--{t['a0_fi_es']['ci_high']:.2f}",
        "%%TLEVEL%%": f"{t['level_rms_fi_es']:.2f}",
        "%%TETA%%": f"{res['transport_sd']:.2f}",
        "%%TFEMALE%%": f"{res['mean_by_predictor']['female']:.2f}",
        "%%TNC%%": str(t["n_countries"]),
        "%%TTAU%%": f"{t['tau_rel_median']:.2f}",
        "%%TPOP%%": pct(seg["All residents"]),
        "%%TMOST%%": pct(prof["most_receptive"]),
        "%%TLEAST%%": pct(prof["least_receptive"]),
        "%%TSPREAD%%": f"{100 * float(prof['spread']['median']):.0f} ({100 * float(prof['spread']['ci_low']):.0f}--{100 * float(prof['spread']['ci_high']):.0f})",
        "%%TMLOW%%": f"{min(float(r['m']) for r in seg.values()):.0f}",
        "%%TMHIGH%%": f"{max(float(r['m']) for r in seg.values()):.0f}",
    }


def run_comparison_rows(deposit: Path, run: Path) -> str:
    """BVS row from this run; CTGAN, TVAE and independent rows from the deposit."""
    report = json.loads((run / "step6_output/full_report_fixed.json").read_text())
    tvd = [float(r["TVD_S1"]) for r in read_csv(run / "step6_output/tvd_per_variable_fixed.csv")]
    chi = read_csv(run / "step6_output/chisq_dependencies_fixed.csv")
    ratios = [float(r["V_ratio"]) for r in chi]
    kept = sum(r["preserved"] == "True" for r in chi)
    lines = [
        f"\\textbf{{BVS (proposed)}} & \\textbf{{{report['tvd_mean']:.3f}}} & \\textbf{{{report['tvd_median']:.3f}}} & "
        f"\\textbf{{{100 * sum(x < 0.05 for x in tvd) / len(tvd):.1f}}} & \\textbf{{{report['corr_mae']:.4f}}} & "
        f"{kept}/{len(chi)} & {min(ratios):.2f}--{max(ratios):.2f} & Yes \\\\"
    ]
    table = {r["Model"]: r for r in read_csv(deposit / "04_validation/baselines/comparison_table.csv")}
    dchi = read_csv(deposit / "04_validation/baselines/chisq_by_model.csv")
    for model, label, _ in MODELS[2:]:
        r = table[model]
        rr = [float(x["V_ratio"]) for x in dchi if x["model"] == model]
        lines.append(
            f"{label} & {float(r['TVD (mean)']):.3f} & {float(r['TVD (median)']):.3f} & {float(r['TVD Exc. (%)']):.1f} & "
            f"{float(r['Corr MAE']):.4f} & {r['Chi² pres.']} & {min(rr):.2f}--{max(rr):.2f} & No \\\\"
        )
    return "\n".join(lines)


def evidence_rows(run: Path, with_shares: bool) -> str:
    notes = {"acceptance_food_packaging": "a", "perception_safety": "b", "perception_quality": "c"}
    lines = []
    for r in read_csv(run / "target_evidence.csv"):
        name = r["construct"]
        label = CONSTRUCT_LABELS[name]
        if with_shares:
            label += f"\\tnote{{{notes[name]}}}" if name in notes else ""
            if int(r["entries"]) == 0:
                lines.append(f"{label} & 0 & 0 & --- & \\multicolumn{{3}}{{c}}{{no usable entry}} \\\\")
                continue
            lines.append(
                f"{label} & {r['studies']} & {r['entries']} & {thousands(int(r['independent_n']))} & "
                f"{r['low_pct']}\\% & {r['mid_pct']}\\% & {r['high_pct']}\\% \\\\"
            )
        else:
            if int(r["entries"]) == 0:
                lines.append(f"{label} & 0 & 0 & --- & --- & --- \\\\")
                continue
            i2 = "n.e." if int(r["entries"]) == 1 else r["I2_pct"]
            lines.append(
                f"{label} & {r['studies']} & {r['entries']} & {thousands(int(r['pooled_n']))} & "
                f"{thousands(int(r['independent_n']))} & {i2} \\\\"
            )
    return "\n".join(lines)


def missing_rows(run: Path) -> tuple[str, str]:
    rows = read_csv(run / "missingness_network_vars.csv")
    values = [(r[""] if "" in r else list(r.values())[0], float(r["pct_missing"])) for r in rows]
    high = [(v, m) for v, m in values if m > 10]
    n_missing = sum(m > 0 for _, m in values)
    table = "\n".join(f"\\texttt{{{v.replace('_', chr(92) + '_')}}} & {m:.1f} \\\\" for v, m in high)
    summary = (f"{n_missing} of the {len(values)} variables have missing values; "
               f"the remaining {n_missing - len(high)} have 10\\% or less.")
    return table, summary


def run_scalars(run: Path, cis_path: Path) -> dict:
    import numpy as np
    import pandas as pd

    tvd = [float(r["TVD_S1"]) for r in read_csv(run / "step6_output/tvd_per_variable_fixed.csv")]
    chi = read_csv(run / "step6_output/chisq_dependencies_fixed.csv")
    rec = next(r for r in chi if "recycling" in r["pair"])
    ipsos = json.loads((run / "step9_output/ipsos_validation.json").read_text())["per_variable"][0]
    bvs_share = ipsos["bvs_top3"]
    cis = pd.read_csv(cis_path, sep=";", encoding="latin-1", usecols=["V15", "PESOFIN"])
    w = pd.to_numeric(cis["PESOFIN"].astype(str).str.replace(",", "."))
    valid = cis["V15"].between(1, 5)
    cis_share = (w[valid & cis["V15"].between(3, 5)].sum()) / w[valid].sum()
    seeds = json.loads((run / "structure_variability.json").read_text()) if (run / "structure_variability.json").exists() else None
    target = json.loads((run / "step4_output/target_summary.json").read_text())
    pct = lambda x: f"{100 * x:.0f}\\%"
    bp = target["by_predictor"]
    return {
        "%%TVDMEAN%%": f"${sum(tvd) / len(tvd):.3f}$",
        "%%TVDMEDIAN%%": f"${float(np.median(tvd)):.3f}$",
        "%%TVD05%%": f"{100 * sum(x < 0.05 for x in tvd) / len(tvd):.1f}\\%",
        "%%TVD10%%": f"{100 * sum(x < 0.10 for x in tvd) / len(tvd):.1f}\\%",
        "%%VREC%%": f"$V$ ratio {float(rec['V_ratio']):.2f}",
        "%%KEPT%%": str(sum(r["preserved"] == "True" for r in chi)),
        "%%IPSOSBVS%%": f"{100 * bvs_share:.1f}\\%",
        "%%IPSOSERR%%": f"{100 * abs(bvs_share - 0.90):.1f}~percentage points",
        "%%IPSOSCIS%%": f"{100 * cis_share:.1f}\\%",
        "%%NWITH%%": thousands(target["n_with_outcome"]),
        "%%SHAREACC%%": pct(target["share_accepting"]),
        "%%FEMW%%": pct(bp["female"]["with"]), "%%FEMWO%%": pct(bp["female"]["without"]),
        "%%EDUW%%": pct(bp["high_education"]["with"]), "%%EDUWO%%": pct(bp["high_education"]["without"]),
        "%%RECW%%": pct(bp["recycles"]["with"]), "%%RECWO%%": pct(bp["recycles"]["without"]),
        "%%SEEDTEXT%%": seed_text(seeds),
    }


def seed_text(seeds) -> str:
    if not seeds:
        return "\\todo{Structural variability across seeds pending.}"
    return seeds["text"]


def main() -> None:
    parts, deposit, audit, output, transfer, run = (Path(a) for a in sys.argv[1:7])
    constructs = read_csv(audit / "audit_constructs.csv")
    studies = read_csv(audit / "audit_studies.csv")
    fillers = {
        "%%I2ROWS%%": evidence_rows(run, with_shares=False),
        "%%CMPROWS%%": run_comparison_rows(deposit, run),
        "%%VROWS%%": dependency_rows(run),
        "%%NOVELROWS%%": evidence_rows(run, with_shares=True),
        "%%MISSROWS%%": missing_rows(run)[0],
        "%%MISSSUMMARY%%": missing_rows(run)[1],
        "%%STUDYROWS%%": study_rows(studies, run),
        "%%A0ROWS%%": sensitivity_rows(transfer, "a0", "$a_0$"),
        "%%P0ROWS%%": sensitivity_rows(transfer, "p0", "$p_0$"),
        "%%SEGROWS%%": segment_rows(transfer),
        "%%SEGA0ROWS%%": segment_a0_rows(transfer),
        "%%PLANROWS%%": plan_rows(transfer),
    }
    text = "".join((parts / name).read_text(encoding="utf-8") for name in PART_ORDER)
    for marker, rows in fillers.items():
        if text.count(marker) != 1:
            raise SystemExit(f"marker {marker} found {text.count(marker)} times")
        text = text.replace(marker, rows)
    scalars = run_scalars(run, run / "metadata/3391_num.csv") | transport_scalars(transfer, transfer.parent / "transport_output")
    for marker, value in scalars.items():
        if marker not in text:
            raise SystemExit(f"marker {marker} not found")
        text = text.replace(marker, value)
    leftover = re.findall(r"%%[A-Z0-9]+%%", text)
    if leftover:
        raise SystemExit(f"unfilled markers: {leftover}")
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
