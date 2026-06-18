"""Write a readable README.md card into every run folder under outputs/.

Each card is generated from that run's config.json + metrics.json so it stays
accurate. Also refreshes the catalog table in the experiment README.

    python experiments/exp2_layerwise/scripts/make_run_cards.py
"""

from __future__ import annotations

import glob
import json
import os

HERE = os.path.dirname(__file__)
EXP = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(EXP, "outputs")


def card(tag, cfg, met):
    ev = met.get("eval", {})
    us = ev.get("unseen_spatial", {}); se = ev.get("seen_temporal", {})
    dataset = os.path.basename(cfg.get("csv", "?"))
    return f"""# Run `{tag}`

Layer-wise PINN run. **{cfg.get('pct_spatial')}% spatial** sampling x
**{int(cfg.get('temporal_keep', 0)*100)}% temporal**, activation
**{cfg.get('activation')}**, physics weight (alpha) **{cfg.get('balance_alpha')}**.

## Setup
- **Spatial availability:** {cfg.get('pct_spatial')}% of the per-layer grid (3 plies),
  {cfg.get('train_rows'):,} training rows.
- **Temporal:** {int(cfg.get('temporal_keep', 0)*100)}% of timesteps kept per trained point.
- **Model:** Fourier leaves F={cfg.get('num_freq')}, hidden {cfg.get('hidden')},
  activation {cfg.get('activation')}, Dirichlet BC {'on' if cfg.get('bdry') else 'off'}.
- **Physics weight:** balance alpha = {cfg.get('balance_alpha')} (gradient-norm balancing).
- **Stopping:** training-loss early stop, patience {cfg.get('patience')}, epoch cap
  {cfg.get('epochs_cap')}; ran {met.get('epochs_run')} epochs.
- **Non-dim:** beta_y={cfg.get('beta_y'):.3f}, beta_z={cfg.get('beta_z'):.3f},
  gamma={cfg.get('gamma'):.3f}.

## Data
- **File:** `{dataset}`
- Built by `scripts/gen_layerwise_dataset.py --pct {cfg.get('pct_spatial')}`: random
  per-layer sample at all 3 plies, with a reserved fraction of perimeter
  (`is_boundary=1`) points that are kept in training (never spatially held out).

## Results (median relative L2)
| evaluation | median | mean | n |
|---|---|---|---|
| unseen spatial (no timestep seen) | {us.get('median')} | {us.get('mean')} | {us.get('n')} |
| seen points (held-out timesteps) | {se.get('median')} | {se.get('mean')} | {se.get('n')} |

## Figures (PNG + SVG, current viz style)
`loss_{{data,wave,gauge,ic,bdry,total}}` (loss parameters vs training),
`reconstruction` (seen spatial points, 10% temporal seen),
`heldout_prediction` (unseen spatial points). `model.pt`, `metrics.json`,
`config.json` saved for re-running prediction without retraining.
"""


def main():
    rows = []
    n = 0
    for cfgp in sorted(glob.glob(os.path.join(OUT, "*", "config.json"))):
        d = os.path.dirname(cfgp)
        tag = os.path.basename(d)
        cfg = json.load(open(cfgp))
        metp = os.path.join(d, "metrics.json")
        met = json.load(open(metp)) if os.path.exists(metp) else {}
        open(os.path.join(d, "README.md"), "w").write(card(tag, cfg, met))
        ev = met.get("eval", {})
        rows.append((tag, cfg, ev))
        print(f"wrote {tag}/README.md")
        n += 1

    # refresh the catalog table in the experiment README
    readme = os.path.join(EXP, "README.md")
    if os.path.exists(readme) and rows:
        lines = ["| run tag | data | temporal | activation | F | hidden | alpha | unseen | seen |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for tag, cfg, ev in rows:
            us = ev.get("unseen_spatial", {}).get("median")
            se = ev.get("seen_temporal", {}).get("median")
            h = "x".join(str(w) for w in cfg.get("hidden", []))
            lines.append(f"| {tag} | {cfg.get('pct_spatial')}% | "
                         f"{int(cfg.get('temporal_keep',0)*100)}% | {cfg.get('activation')} | "
                         f"{cfg.get('num_freq')} | {h} | {cfg.get('balance_alpha')} | "
                         f"{us if us is None else round(us,3)} | {se if se is None else round(se,3)} |")
        table = "\n".join(lines)
        txt = open(readme).read()
        import re
        txt = re.sub(r"\| run tag \|.*?(?=\n\n|\Z)", table, txt, flags=re.S)
        open(readme, "w").write(txt)
        print("refreshed catalog table in README.md")
    print(f"\n{n} cards written.")


if __name__ == "__main__":
    main()
