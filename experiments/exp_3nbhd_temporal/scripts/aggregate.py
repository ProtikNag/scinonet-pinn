"""Aggregate the per-level metrics into the availability comparison figure + table.

Reads every ``outputs/keep_*/metrics.json`` produced by ``run_level.py`` and writes:

    outputs/availability_comparison.{png,svg}   median relL2 vs availability, three
                                                lines (seen / neighborhood / far)
    outputs/results_table.md                    markdown table of medians
    outputs/results.json                        machine-readable aggregate

    python experiments/exp_3nbhd_temporal/scripts/aggregate.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from scinonet import viz  # noqa: E402

OUT = os.path.join(ROOT, "experiments/exp_3nbhd_temporal/outputs")
SETTINGS = [("seen", "Seen points (temporal infill)", viz.AC["blue"]),
            ("neighborhood", "Neighborhood (spatial holdout)", viz.AC["amber"]),
            ("far", "Far (extrapolation)", viz.AC["green"])]


def main():
    import matplotlib.pyplot as plt
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, "keep_*", "metrics.json"))):
        rows.append(json.load(open(path)))
    rows.sort(key=lambda r: r["keep"])
    if not rows:
        print("no metrics found"); return

    pct = [r["pct"] for r in rows]
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for key, label, color in SETTINGS:
        med = [r[key]["median"] for r in rows]
        ax.plot(pct, med, color=color, lw=1.8, marker="o", ms=5, label=label)
    ax.set_xlabel("Temporal data availability [% of timesteps]", fontsize=12)
    ax.set_ylabel("Median held-out relative L2", fontsize=12)
    ax.set_title("Temporal prediction vs data availability (3 neighborhoods, Lz=1, no gauge)",
                 fontsize=13, fontweight=600)
    ax.set_xticks(pct)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout(); viz._save(fig, os.path.join(OUT, "availability_comparison")); plt.close(fig)

    # markdown table
    lines = ["| Availability | Seen (temporal) | Neighborhood (spatial) | Far (extrap.) | train rows |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['pct']:.0f}% | {r['seen']['median']:.3f} | "
                     f"{r['neighborhood']['median']:.3f} | {r['far']['median']:.3f} | "
                     f"{r['train_rows']:,} |")
    table = "\n".join(lines)
    open(os.path.join(OUT, "results_table.md"), "w").write(table + "\n")
    json.dump({"levels": rows}, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print(table)
    print(f"\nsaved: {OUT}/availability_comparison.png/.svg, results_table.md, results.json")


if __name__ == "__main__":
    main()
