"""Accuracy of the query-parsing call of the interface on a labelled question set.

Sends each question of interface_parsing_gold.jsonl to the same call as
app_gradio.parse_question (system prompt PARSE_SYS read from app_gradio.py,
same model, temperature and output limit, same post-processing) and compares
the returned profile with the gold profile, field by field.

A field is correct when the returned value equals the gold value, including
null when the question does not mention it. For comparison questions the
product field is not scored ("*" in the gold) and compare_products is scored
as a set. Each question is sent --repeats times to measure stability.

Outputs (in --out):
    parsing_predictions.jsonl   raw text and parsed profile per call
    parsing_fields.csv          accuracy per field
    parsing_summary.json        exact-profile accuracy, field accuracy, JSON
                                failures, stability across repeats, model

Requires ANTHROPIC_API_KEY in the environment (or a .env file).

Usage
-----
    python eval/evaluate_parsing.py --out eval/results --repeats 3
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 500
TEMPERATURE = 0
FIELDS = ["age", "gender", "edu", "income", "product", "env_concern", "urban",
          "recycling", "wtp_prices", "is_comparison", "compare_products"]
ROOT = Path(__file__).resolve().parents[1]


def parse_system_prompt(app: Path) -> str:
    """PARSE_SYS from app_gradio.py, read without importing the app."""
    tree = ast.parse(app.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PARSE_SYS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"PARSE_SYS not found in {app}")


def to_profile(text: str) -> dict | None:
    """Same post-processing as app_gradio.parse_question; None if the JSON is invalid."""
    raw = re.sub(r"```(?:json)?", "", text.strip()).strip().strip("`")
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def correct(field: str, gold, predicted) -> bool | None:
    """None when the field is not scored for this question."""
    if gold == "*":
        return None
    if field == "compare_products":
        return set(gold or []) == set(predicted or [])
    if field == "is_comparison":
        return bool(gold) == bool(predicted)
    return gold == predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gold", type=Path, default=ROOT / "eval/interface_parsing_gold.jsonl")
    parser.add_argument("--app", type=Path, default=ROOT / "app_gradio.py")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/results")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    import anthropic
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic()
    system = parse_system_prompt(args.app)
    items = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    predictions, scores = [], defaultdict(list)
    exact, failures, stable = [], 0, []
    for item in items:
        profiles = []
        for repeat in range(args.repeats):
            response = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, system=system,
                messages=[{"role": "user", "content": item["question"]}],
            )
            text = response.content[0].text
            profile = to_profile(text)
            failures += profile is None
            profiles.append(profile)
            predictions.append({"id": item["id"], "repeat": repeat, "raw": text, "profile": profile})
        first = profiles[0] or {}
        results = {f: correct(f, item["gold"][f], first.get(f)) for f in FIELDS}
        for f, ok in results.items():
            if ok is not None:
                scores[f].append(ok)
        exact.append(profiles[0] is not None and all(ok is not False for ok in results.values()))
        stable.append(all(p == profiles[0] for p in profiles))
        print(f"{item['id']} {'ok ' if exact[-1] else 'ERR'} "
              + " ".join(f for f, ok in results.items() if ok is False))

    with open(args.out / "parsing_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    fields = pd.DataFrame([{"field": f, "scored": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)}
                           for f, v in scores.items()])
    fields.to_csv(args.out / "parsing_fields.csv", index=False)
    summary = {
        "model": MODEL, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "questions": len(items), "repeats": args.repeats,
        "exact_profile_accuracy": sum(exact) / len(exact),
        "field_accuracy": float(fields["correct"].sum() / fields["scored"].sum()),
        "json_failures": failures,
        "stable_across_repeats": sum(stable) / len(stable),
        "exact_by_language": {lang: float(pd.Series([e for e, i in zip(exact, items) if i["lang"] == lang]).mean())
                              for lang in sorted({i["lang"] for i in items})},
    }
    (args.out / "parsing_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(fields.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
