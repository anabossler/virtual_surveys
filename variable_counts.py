"""Number of CIS 3391 variables at each selection stage of step1.

Uses the functions of step1.py on the released file and records:
    columns_in_file      columns of metadata/3391_num.csv
    usable               after removing empty, constant and >=90% missing columns
    substantive          after the substantive criterion (ISSP module, background, ESGE)
    analysis             after the collinearity filter (|r| > 0.95)

Output: variable_counts.json, read by paper/assemble.py.

Usage
-----
    python variable_counts.py
"""

from __future__ import annotations

import contextlib
import io
import json

import pandas as pd

import step1


def main() -> None:
    header = pd.read_csv(step1.PATH_NUM, sep=";", encoding="latin1", nrows=0, quotechar='"')
    with contextlib.redirect_stdout(io.StringIO()):
        usable = step1.load_microdata_cis(step1.PATH_NUM)
        substantive = step1.filter_substantive_variables(usable)
        analysis = step1.filter_collinear(substantive, threshold=0.95)
    counts = {
        "columns_in_file": int(header.shape[1]),
        "usable": int(usable.shape[1]),
        "substantive": int(substantive.shape[1]),
        "analysis": int(analysis.shape[1]),
        "respondents": int(usable.shape[0]),
    }
    with open("variable_counts.json", "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
        f.write("\n")
    print(counts)


if __name__ == "__main__":
    main()
