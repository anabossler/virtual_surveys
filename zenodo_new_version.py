"""Create a new version of a Zenodo record from the terminal (legacy deposit API).

Steps: new-version draft -> remove the files copied from the previous version ->
upload the given files -> update title, description and version -> stop, so the
draft can be checked in the browser. With --publish it also publishes (a
published version cannot be deleted).

Token: create one at https://zenodo.org/account/settings/applications/tokens/new
with scopes deposit:write and deposit:actions, and put it in the environment:
    export ZENODO_TOKEN=...          (never in the code, never in a commit)

Usage
  python zenodo_new_version.py --record 23824923 \
      --files ~/Desktop/virtual_surveys_v3.zip ~/Desktop/virtual_surveys_outputs_v3.zip
  python zenodo_new_version.py --record 23824923 --publish     # after checking the draft
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import requests

API = "https://zenodo.org/api/deposit/depositions"

TITLE = ("How much is synthetic evidence worth before measurement? Code and aggregate "
         "outputs of a joint Bayesian model and transfer audit")
DESCRIPTION = """
<p>Companion to a manuscript under double-anonymised review. Version 3 replaces the
Bayesian Virtual Survey outputs of versions 1&ndash;2 with the code and aggregate
outputs of the revised study.</p>
<p>The study attaches an unmeasured outcome, acceptance of recycled plastic in
food-contact packaging, to a Bayesian network estimated on CIS Study 3391 (Spain,
n = 2,254), with missing answers inside the model. The outcome's coefficients and
base rate come from a Finnish study, and a hierarchical transport layer is
estimated on ISSP 2020 Environment IV (16 European countries). A retrospective
benchmark audits the transferred evidence and expresses its value as the size of a
target-country sample with the same error. It compares the foreign source with the
target country's own related survey items.</p>
<p><b>virtual_surveys_v3.zip</b>: Python source code. It includes the network
sampler with data augmentation for missing answers, a replication of Martins et al.
(2024, arXiv:2402.17915), the outcome node and transport layer, synthetic datasets
with combining rules, the validation on ISSP 2020, the comparators, and the
robustness and bootstrap checks.</p>
<p><b>virtual_surveys_outputs_v3.zip</b>: aggregate outputs (JSON/CSV) behind the
tables and figures. Supplementary Table S17 of the manuscript maps each result to
the script that produces it.</p>
<p>No microdata are included. CIS Study 3391 (https://www.cis.es) and ISSP 2020
ZA7650 v2.0.0 (GESIS, https://doi.org/10.4232/1.14153) must be obtained from their
providers. The scripts that read them are run locally, and only aggregate outputs
are deposited.</p>
""".strip()


def call(method, url, token, **kw):
    r = requests.request(method, url, headers={"Authorization": f"Bearer {token}"}, timeout=600, **kw)
    if r.status_code >= 400:
        sys.exit(f"{method} {url} -> {r.status_code}\n{r.text[:2000]}")
    return r.json() if r.content else {}


def get_draft(record: int, token: str) -> dict:
    """Open (or reuse) the new-version draft of the record."""
    dep = call("GET", f"{API}/{record}", token)
    latest = dep.get("links", {}).get("latest_draft")
    if latest and not latest.rstrip("/").endswith(str(record)):
        draft = call("GET", latest, token)
        if not draft.get("submitted", True):
            print("reusing the existing new-version draft")
            return draft
    out = call("POST", f"{API}/{record}/actions/newversion", token)
    return call("GET", out["links"]["latest_draft"], token)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", type=int, required=True, help="id of the current published version")
    ap.add_argument("--files", nargs="*", type=Path, default=[])
    ap.add_argument("--version", default="v3")
    ap.add_argument("--publish", action="store_true", help="publish the draft (cannot be undone)")
    a = ap.parse_args()
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("set ZENODO_TOKEN first (export ZENODO_TOKEN=...)")
    for f in a.files:
        if not f.expanduser().exists():
            sys.exit(f"file not found: {f}")

    draft = get_draft(a.record, token)
    did = draft["id"]
    print(f"draft {did}: {draft['links'].get('html')}")

    if a.files:
        for f in call("GET", f"{API}/{did}/files", token) or []:
            call("DELETE", f"{API}/{did}/files/{f['id']}", token)
            print(f"  removed {f['filename']}")
        bucket = draft["links"]["bucket"]
        for f in a.files:
            f = f.expanduser()
            with open(f, "rb") as fh:
                call("PUT", f"{bucket}/{f.name}", token, data=fh)
            print(f"  uploaded {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

        md = draft["metadata"]
        md.update({"title": TITLE, "description": DESCRIPTION, "version": a.version,
                   "publication_date": dt.date.today().isoformat()})
        md.pop("doi", None)
        call("PUT", f"{API}/{did}", token, json={"metadata": md})
        print("  metadata updated (creators, licence and keywords kept from the previous version)")
        print("\ncheck the draft in the browser, then run again with --publish")

    if a.publish:
        out = call("POST", f"{API}/{did}/actions/publish", token)
        print(f"\npublished: DOI {out.get('doi')}  {out.get('links', {}).get('html')}")


if __name__ == "__main__":
    main()
