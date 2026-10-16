"""Share of missing values for the network variables of CIS 3391.

Uses the loader of step4.py (same missing codes and filters), restricted to
the variables of the learned structure (dag_edges_3391.csv).

Output: missingness_network_vars.csv (variable, pct_missing), sorted
descending; reported in Section 4 and Appendix C of the paper.

Usage
-----
    python missingness.py
"""

from __future__ import annotations

import pandas as pd

import step4


def main() -> None:
    cfg = step4.Config()
    data = step4.load_cis_microdata(cfg)
    edges = pd.read_csv("dag_edges_3391.csv")
    variables = sorted(set(edges["source"]) | set(edges["target"]))
    pct = (data[variables].isna().mean() * 100).round(2).sort_values(ascending=False)
    pct.rename("pct_missing").to_csv("missingness_network_vars.csv")
    affected = pct[pct > 0]
    print(f"{len(variables)} variables, {len(affected)} with missing values; "
          f"mean {pct.mean():.2f}%, mean among affected {affected.mean():.2f}%, max {pct.max():.2f}%")


if __name__ == "__main__":
    main()
