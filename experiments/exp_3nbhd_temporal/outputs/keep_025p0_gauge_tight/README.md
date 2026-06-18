# Run `keep_025p0_gauge_tight`

Helmholtz-potential PINN, **25% temporal availability**, gauge ON (w = 1.0).

## Setup
- **Temporal availability:** 25% of timesteps kept per training point
  (607,078 training rows).
- **Through-thickness scale:** Lz = 1 mm (physical ply spacing, rho ~ 94)
- **Gauge loss:** ON (w = 1.0)
- **Training:** 50 epochs, physics = wave residual + initial
  condition, CPU / float64. Seed 42.

## Data
- **File:** `experiments/exp_3nbhd_temporal/data/dataset_3nbhd_tight_50pts_3ply_fullsignal_6001steps.csv`
- **Three tight, contiguous 1 mm-grid clusters** (the **nearest 50** points to each
  center, r ~ 4-5 mm) on the line y = -99.5 mm: `near_source` (-49.5, -99.5),
  `in_between` (50, -99.5), `near_boundary` (149.5, -99.5, **on the right plate
  edge** so most of the cluster touches the boundary).
- Split **45 train + 5 spatially held-out test** per neighborhood; the 5 test
  points are taken from the cluster **interior** (all four 1 mm neighbors present)
  so each held-out point is flanked by training data. All three plies (z = 0, -1, -2 mm).
- **Built by:** `python experiments/exp_3nbhd_temporal/scripts/gen_3nbhd_tight.py`

## What changed vs the baseline sweep
- Tight contiguous 1 mm clusters instead of the 15 mm random-sampled disks.
- Coulomb-gauge loss `div(psi)` re-enabled in the objective (w = 1.0).

## Results (median held-out relative L2)
| setting | median rel-L2 |
|---|---|
| seen (temporal infill at training points) | 0.476 |
| neighborhood (spatial holdout, 15 unseen pts) | 0.485 |
| far (extrapolation, away from all clusters) | 1.916 |

Per-neighborhood (seen / neighborhood spatial): near_source 0.114 / 0.346, in_between 0.476 / 0.485, near_boundary 11.844 / 14.408

## Figures (PNG + SVG)
`plate_layout` (point map, full plate + zoomed neighborhood),
`loss_curves`, `temporal_seen_grid` (seen timesteps as pale blue vertical lines),
`neighborhood_holdout_grid`, `far_holdout_grid`. Metrics in `metrics.json`,
weights in `model.pt`.
