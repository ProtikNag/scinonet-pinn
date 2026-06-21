"""Render the Experiment-1 (layer-wise random sampling) results table.

Same visual style as the Experiment-2 neighborhood table, but with the columns
that fit this experiment: there is no inside/outside split here, only

    Recon error (seen, held timesteps)  vs  Unseen error (held spatial points)

Reads metrics.json + config.json from a run dir; writes results_table.{png,svg,tex,csv}.

    python experiments/exp2_layerwise/scripts/make_results_table.py \
        --out experiments/exp2_layerwise/outputs/sp20_t10_silu_F256_h256x256x256_a1_hpc
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import scinonet_pinn as P  # noqa: E402  (palette + save_fig + style)


def _cell(s):
    return "--" if not s or s.get("median") is None else f"{s['median']:.3f} ({s['mean']:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run dir with metrics.json + config.json")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.out, "config.json")))
    ev = json.load(open(os.path.join(args.out, "metrics.json")))["eval"]
    seen = ev.get("seen_temporal"); unseen = ev.get("unseen_spatial")

    name = (f"PINN ({cfg.get('activation')}, F{cfg.get('num_freq')}, "
            f"{'x'.join(str(h) for h in cfg.get('hidden', []))}, a={cfg.get('balance_alpha')})")
    rows = [{
        "name": name,
        "e_r": "wave (phi,psi) + gauge + IC",
        "e_b": "Dirichlet u=v=w=0" if cfg.get("bdry") else "None",
        "recon": seen, "unseen": unseen,
    }]

    cols = ["Name", "$E_r$ (PDE residual)", "$E_b$ (boundary)",
            "Recon error\n(seen, held t)", "Unseen error\n(held spatial)"]
    col_w = [0.27, 0.23, 0.17, 0.165, 0.165]
    table = [[r["name"], r["e_r"], r["e_b"], _cell(r["recon"]), _cell(r["unseen"])] for r in rows]

    fig, ax = plt.subplots(figsize=(15, 2.0 + 0.8 * len(rows)))
    ax.axis("off")
    tb = ax.table(cellText=table, colLabels=cols, colWidths=col_w, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(13)
    for (i, j), c in tb.get_celld().items():
        c.set_height(0.30 if i == 0 else 0.20)
        c.set_edgecolor("#C9CED3")
        if i == 0:
            c.set_text_props(weight="bold", color="white"); c.set_facecolor(P.AC["axis"])
        else:
            c.set_facecolor("#F4F6F8" if i % 2 else "white")
            if j == 0:
                c.set_text_props(ha="left"); c.PAD = 0.04
            if j >= 3:                              # highlight the error columns
                c.set_text_props(weight="bold")
    ax.set_title("Table 1: Reconstruction vs unseen-spatial error (relative-L2)",
                 fontsize=21, pad=24)
    caption = ("Relative-L2 of the full-signal reconstruction. Recon = held timesteps at "
               "seen points; Unseen = held-out spatial points (random per-layer sampling). "
               "Median over points (mean in parentheses).")
    fig.text(0.5, 0.04, caption, ha="center", va="bottom", fontsize=12,
             color=P.AC["muted"], wrap=True)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.18)
    stem = os.path.join(args.out, "results_table")
    P.save_fig(fig, stem)
    plt.close(fig)

    # LaTeX + CSV siblings
    def tex_cell(s):
        return "--" if not s or s.get("median") is None else f"${s['median']:.3f}$ (${s['mean']:.3f}$)"
    with open(f"{stem}.tex", "w") as f:
        f.write("\\begin{center}\n\\begin{tabular}{lll cc}\n\\toprule\n")
        f.write("Name & $E_r$ & $E_b$ & Recon (seen) & Unseen (spatial) \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['name']} & {r['e_r']} & {r['e_b']} & "
                    f"{tex_cell(r['recon'])} & {tex_cell(r['unseen'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{center}\n")
    with open(f"{stem}.csv", "w") as f:
        f.write("name,E_r,E_b,recon_median,recon_mean,unseen_median,unseen_mean\n")
        for r in rows:
            def md(s): return ("", "") if not s or s.get("median") is None else (
                f"{s['median']:.6f}", f"{s['mean']:.6f}")
            rc, uc = md(r["recon"]), md(r["unseen"])
            f.write(f"\"{r['name']}\",\"{r['e_r']}\",\"{r['e_b']}\","
                    f"{rc[0]},{rc[1]},{uc[0]},{uc[1]}\n")
    print(f"saved: {stem}.tex / .csv")


if __name__ == "__main__":
    main()
