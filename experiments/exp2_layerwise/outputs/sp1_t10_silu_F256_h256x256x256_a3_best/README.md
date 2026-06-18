# Run `sp1_t10_silu_F256_h256x256x256_a3_best`

Layer-wise PINN run. **1.0% spatial** sampling x
**10% temporal**, activation
**silu**, physics weight (alpha) **3.0**.

## Setup
- **Spatial availability:** 1.0% of the per-layer grid (3 plies),
  1,061,316 training rows.
- **Temporal:** 10% of timesteps kept per trained point.
- **Model:** Fourier leaves F=256, hidden [256, 256, 256],
  activation silu, Dirichlet BC on.
- **Physics weight:** balance alpha = 3.0 (gradient-norm balancing).
- **Stopping:** training-loss early stop, patience 40, epoch cap
  100; ran 52 epochs.
- **Non-dim:** beta_y=1.424, beta_z=112.044,
  gamma=0.508.

## Data
- **File:** `dataset_layerwise_1pct_3ply_fullsignal_6001steps.csv`
- Built by `scripts/gen_layerwise_dataset.py --pct 1.0`: random
  per-layer sample at all 3 plies, with a reserved fraction of perimeter
  (`is_boundary=1`) points that are kept in training (never spatially held out).

## Results (median relative L2)
| evaluation | median | mean | n |
|---|---|---|---|
| unseen spatial (no timestep seen) | 1.2740619402092714 | 4.843749511599354 | 10 |
| seen points (held-out timesteps) | 0.2734016311931164 | 2.033143821772152 | 40 |

## Figures (PNG + SVG, current viz style)
`loss_{data,wave,gauge,ic,bdry,total}` (loss parameters vs training),
`reconstruction` (seen spatial points, 10% temporal seen),
`heldout_prediction` (unseen spatial points). `model.pt`, `metrics.json`,
`config.json` saved for re-running prediction without retraining.
