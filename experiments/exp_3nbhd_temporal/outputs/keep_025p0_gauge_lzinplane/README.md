# Run `keep_025p0_gauge_lzinplane`

Helmholtz-potential PINN, **25% temporal availability**, gauge ON (w = 1.0).

## Setup
- **Temporal availability:** 25% of timesteps kept per training point
  (607,132 training rows).
- **Through-thickness scale:** Lz = L (in-plane change of variables, rho = 1.0)
- **Gauge loss:** ON (w = 1.0)
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
- Coulomb-gauge loss `div(psi)` re-enabled in the objective (w = 1.0).
- Reverted to the previous Lz = L scaling (rho = 1, well-conditioned).

## Results (median held-out relative L2)
| setting | median rel-L2 |
|---|---|
| seen (temporal infill at training points) | 0.188 |
| neighborhood (spatial holdout, 15 unseen pts) | 0.951 |
| far (extrapolation, away from all clusters) | 1.588 |

Per-neighborhood (seen / neighborhood spatial): near_source 0.056 / 1.090, in_between 0.188 / 0.714, near_boundary 0.778 / 1.070

## Figures (PNG + SVG)
`plate_layout` (point map),
`loss_curves`, `temporal_seen_grid` (seen timesteps as pale blue vertical lines),
`neighborhood_holdout_grid`, `far_holdout_grid`. Metrics in `metrics.json`,
weights in `model.pt`.
