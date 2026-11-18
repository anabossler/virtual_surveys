"""Figures in the style of Martins et al. (2024), Figs 2-4, from martins_replication.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bn.martins_replication import method_summary, combined_oic

res = json.loads(Path("bn_output/martins_replication.json").read_text())
rows = [("d3_n5000", 3), ("d4_n5000", 4), ("d7_n5000", 7)]
for stat, label in (("oic", "OIC"), ("mle", "MLE of P(X2=1|X1=0)"), ("pv12", "p-value X1-X2")):
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharey="row")
    for i, (key, d) in enumerate(rows):
        e = res["scenarios"][key]
        for j, prior in enumerate(("P1", "P2")):
            ax = axes[i, j]
            for r, rec in enumerate(e["reps"], start=1):
                ms = method_summary(rec, prior, stat, d)
                m, lo, hi = ms["S1"]
                ax.errorbar(r - 0.15, m, yerr=[[max(m - lo, 0)], [max(hi - m, 0)]], fmt="o", color="black", ms=4, capsize=2)
                m2, lo2, hi2 = ms["S2"]
                ax.errorbar(r + 0.05, m2, yerr=[[max(m2 - lo2, 0)], [max(hi2 - m2, 0)]], fmt="*", color="tab:blue", ms=7, capsize=2)
                ax.plot(r + 0.2, ms["S3"][0], "^", color="tab:red", ms=5)
                if stat == "oic":
                    c = combined_oic(rec, prior)
                    ax.plot(r - 0.15, c["S1"], "D", mfc="white", mec="black", ms=6)
                    ax.plot(r + 0.05, c["S2"], "D", mfc="white", mec="tab:blue", ms=6)
                if stat != "oic":
                    ax.plot([r - 0.35, r + 0.35], [rec["truth"][stat]] * 2, color="green", lw=2)
            ax.set_title(f"d = {d}, n = 5000, {prior}", fontsize=9)
            ax.set_xticks(range(1, 11))
            if j == 0:
                ax.set_ylabel(label)
    axes[-1, 0].set_xlabel("Replication"); axes[-1, 1].set_xlabel("Replication")
    extra = "; open diamonds: OIC of the interval combined over all synthetic datasets" if stat == "oic" else ""
    fig.suptitle(f"{label}: S1 mean and 98% HPD (black), S2 mean and 98% CI (blue), S3 (red); original in green{extra}", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"bn_output/martins_fig_{stat}.png", dpi=110)
print("ok")
