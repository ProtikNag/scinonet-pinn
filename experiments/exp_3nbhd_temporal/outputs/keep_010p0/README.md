# Run `keep_010p0`

Helmholtz-potential PINN, **10% temporal availability**, gauge OFF.

## Setup
- **Temporal availability:** 10% of timesteps kept per training point
  (242,576 training rows).
- **Through-thickness scale:** Lz = 1 mm (physical ply spacing, rho ~ 86)
- **Gauge loss:** OFF
- **Training:** 50 epochs, physics = wave residual + initial
  condition, CPU / float64. Seed 42.

## Data
- **File:** `experiments/exp_3nbhd_temporal/data/dataset_3nbhd_50pts_r15_3ply_fullsignal_6001steps.csv`
- **Three neighborhoods** on the line y = -99.5 mm: `near_source` (-49.5, -99.5,
  the measured excitation point), `in_between` (38, -99.5), `near_boundary`
  (125, -99.5, ~25 mm from the right edge).
- Each neighborhood is a **15 mm-radius disk**; **50 points are sampled at random**
  from the dense 1 mm grid in the disk (not the nearest 50), so points are ~3-4 mm
  apart.
- Split **45 train + 5 spatially held-out test** per neighborhood; all three
  through-thickness plies (z = 0, -1, -2 mm).
- **Built by:** `python experiments/exp_3nbhd_temporal/scripts/gen_3nbhd_dataset.py`

## What changed vs the baseline sweep
- This is part of the baseline availability sweep (Lz = 1, gauge off); only the data fraction (10%) varies.

## Results (median held-out relative L2)
| setting | median rel-L2 |
|---|---|
| seen (temporal infill at training points) | 0.948 |
| neighborhood (spatial holdout, 15 unseen pts) | 1.608 |
| far (extrapolation, away from all clusters) | 2.915 |

Per-neighborhood (seen / neighborhood spatial): near_source 0.259 / 1.049, in_between 0.948 / 1.608, near_boundary 5.892 / 3.322

## Figures (PNG + SVG)
`plate_layout` (point map),
`loss_curves`, `temporal_seen_grid` (seen timesteps as pale blue vertical lines),
`neighborhood_holdout_grid`, `far_holdout_grid`. Metrics in `metrics.json`,
weights in `model.pt`.
