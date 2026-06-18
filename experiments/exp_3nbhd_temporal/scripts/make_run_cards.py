"""Write a human-readable README.md card into every run folder under outputs/.

Each card is generated from that run's ``metrics.json`` so it cannot drift from the
actual settings. It records the data used, how the data was made, what changed
relative to the baseline sweep, and the headline results. Re-run any time to
refresh all cards (idempotent).

    python experiments/exp_3nbhd_temporal/scripts/make_run_cards.py
"""

from __future__ import annotations

import glob
import json
import os

HERE = os.path.dirname(__file__)
OUT = os.path.abspath(os.path.join(HERE, "..", "outputs"))

# ── dataset descriptions (keyed by how the run folder is named) ─────────────────
DATA_SPARSE = """\
- **File:** `experiments/exp_3nbhd_temporal/data/dataset_3nbhd_50pts_r15_3ply_fullsignal_6001steps.csv`
- **Three neighborhoods** on the line y = -99.5 mm: `near_source` (-49.5, -99.5,
  the measured excitation point), `in_between` (38, -99.5), `near_boundary`
  (125, -99.5, ~25 mm from the right edge).
- Each neighborhood is a **15 mm-radius disk**; **50 points are sampled at random**
  from the dense 1 mm grid in the disk (not the nearest 50), so points are ~3-4 mm
  apart.
- Split **45 train + 5 spatially held-out test** per neighborhood; all three
  through-thickness plies (z = 0, -1, -2 mm).
- **Built by:** `python experiments/exp_3nbhd_temporal/scripts/gen_3nbhd_dataset.py`"""

DATA_TIGHT = """\
- **File:** `experiments/exp_3nbhd_temporal/data/dataset_3nbhd_tight_50pts_3ply_fullsignal_6001steps.csv`
- **Three tight, contiguous 1 mm-grid clusters** (the **nearest 50** points to each
  center, r ~ 4-5 mm) on the line y = -99.5 mm: `near_source` (-49.5, -99.5),
  `in_between` (50, -99.5), `near_boundary` (149.5, -99.5, **on the right plate
  edge** so most of the cluster touches the boundary).
- Split **45 train + 5 spatially held-out test** per neighborhood; the 5 test
  points are taken from the cluster **interior** (all four 1 mm neighbors present)
  so each held-out point is flanked by training data. All three plies (z = 0, -1, -2 mm).
- **Built by:** `python experiments/exp_3nbhd_temporal/scripts/gen_3nbhd_tight.py`"""


def lz_line(m):
    rho = m["rho"]
    lz = m.get("lz_mode")
    if lz is None:
        lz = "physical" if rho > 10 else "inplane"
    if lz == "physical":
        return f"**Through-thickness scale:** Lz = 1 mm (physical ply spacing, rho ~ {rho:.0f})"
    return f"**Through-thickness scale:** Lz = L (in-plane change of variables, rho = {rho:.1f})"


def gauge_on(name, m):
    return ("_gauge" in name) or (m.get("w_gauge") or 0) > 0


def changed_lines(name, m):
    """Bullet list of what differs from the baseline 6-level sweep."""
    lz = m.get("lz_mode") or ("physical" if m["rho"] > 10 else "inplane")
    out = []
    if "_tight" in name:
        out.append("Tight contiguous 1 mm clusters instead of the 15 mm random-sampled disks.")
    if gauge_on(name, m):
        out.append("Coulomb-gauge loss `div(psi)` re-enabled in the objective (w = 1.0).")
    if lz == "inplane":
        out.append("Reverted to the previous Lz = L scaling (rho = 1, well-conditioned).")
    if not out:
        out.append(f"This is part of the baseline availability sweep "
                   f"(Lz = 1, gauge off); only the data fraction ({m['pct']:.0f}%) varies.")
    return out


def results_table(m):
    rows = [("seen (temporal infill at training points)", m["seen"]["median"]),
            ("neighborhood (spatial holdout, 15 unseen pts)", m["neighborhood"]["median"]),
            ("far (extrapolation, away from all clusters)", m["far"]["median"])]
    lines = ["| setting | median rel-L2 |", "|---|---|"]
    for label, val in rows:
        lines.append(f"| {label} | {val:.3f} |")
    return "\n".join(lines)


def per_nbhd_line(m):
    pn = m.get("per_neighborhood", {})
    parts = []
    for nm, v in pn.items():
        parts.append(f"{nm} {v['seen_median']:.3f} / {v['neighborhood_median']:.3f}")
    return ", ".join(parts)


def card(name, m):
    g = "ON (w = 1.0)" if gauge_on(name, m) else "OFF"
    data = DATA_TIGHT if "_tight" in name else DATA_SPARSE
    changed = "\n".join(f"- {c}" for c in changed_lines(name, m))
    return f"""# Run `{name}`

Helmholtz-potential PINN, **{m['pct']:.0f}% temporal availability**, gauge {g}.

## Setup
- **Temporal availability:** {m['pct']:.0f}% of timesteps kept per training point
  ({m['train_rows']:,} training rows).
- {lz_line(m)}
- **Gauge loss:** {g}
- **Training:** {m.get('epochs', '?')} epochs, physics = wave residual + initial
  condition, CPU / float64. Seed {m.get('seed', 42)}.

## Data
{data}

## What changed vs the baseline sweep
{changed}

## Results (median held-out relative L2)
{results_table(m)}

Per-neighborhood (seen / neighborhood spatial): {per_nbhd_line(m)}

## Figures (PNG + SVG)
`plate_layout` (point map{', full plate + zoomed neighborhood' if '_tight' in name else ''}),
`loss_curves`, `temporal_seen_grid` (seen timesteps as pale blue vertical lines),
`neighborhood_holdout_grid`, `far_holdout_grid`. Metrics in `metrics.json`,
weights in `model.pt`.
"""


def main():
    n = 0
    for path in sorted(glob.glob(os.path.join(OUT, "keep_*", "metrics.json"))):
        d = os.path.dirname(path)
        name = os.path.basename(d)
        m = json.load(open(path))
        open(os.path.join(d, "README.md"), "w").write(card(name, m))
        print(f"wrote {name}/README.md")
        n += 1
    print(f"\n{n} cards written.")


if __name__ == "__main__":
    main()
