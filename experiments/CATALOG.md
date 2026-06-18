# Experiment Catalog

Catalog of the controlled experiments built on top of the Helmholtz-potential PINN
(`src/scinonet/potential.py`). All runs in this folder use the two code changes
requested on 2026-06-03:

1. **`Lz = 1`** — the through-thickness scale is fixed to the physical 1 mm ply
   spacing (`PotentialScalers.fit`), so `rho = L / Lz = L` (large, ~85) instead of
   the earlier `rho = 1` change-of-variables.
2. **Gauge loss removed** — the Coulomb-gauge term `div(psi)` no longer enters the
   training objective (`w_gauge = 0`); it is still computed and logged for
   monitoring. Physics = wave residual + initial condition.

Environments on this machine:
- dataset generation (reads `3D_Pristine.mat`): `~/anaconda3/bin/python` (has h5py)
- training / evaluation (needs torch): `~/anaconda3/envs/rlclass/bin/python`

---

## exp_3nbhd_temporal — temporal availability x spatial generalization

**Question.** How does temporal prediction quality change as the fraction of
retained timesteps drops from 25% to 1%, measured at three spatial settings:
*seen* training points (temporal infill), *neighborhood* held-out points (spatial
interpolation), and *far* points (extrapolation)?

### Dataset (Experiment 3)

`data/dataset_3nbhd_50pts_r15_3ply_fullsignal_6001steps.csv` + `_meta.json`,
built by `scripts/gen_3nbhd_dataset.py`.

- Three neighborhoods, each a **15 mm-radius disk** (larger than the earlier ~4 mm
  clusters), **50 points sampled at random** from the dense 1 mm grid in the disk,
  split **45 train + 5 spatially held-out test**.
- Placement along `y = -99.5 mm`:
  | name | center (mm) | rationale |
  |---|---|---|
  | `near_source` | (-49.5, -99.5) | the measured excitation point (peak \|w\|) |
  | `in_between` | (38.0, -99.5) | midway between source and right edge |
  | `near_boundary` | (125.0, -99.5) | ~25 mm from the right plate edge x=149.5 |
- All three through-thickness plies (z = 0, -1, -2 mm) for every (x, y).
- 150 unique (x, y) = 135 train + 15 neighborhood-test; 450 spatial rows; 2.70 M
  signal rows.

### Runs (Experiment 1)

`scripts/run_level.py --keep <frac>` trains one physics-on model per availability
level and evaluates the three settings. Six levels: **1, 5, 10, 15, 20, 25 %**.
Far points are random plate points > 28 mm from every neighborhood center.

Per-level outputs under `outputs/keep_<pct>/`: `metrics.json`, `model.pt`,
`loss_curves`, `plate_layout`, `temporal_seen_grid`, `neighborhood_holdout_grid`,
`far_holdout_grid` (each PNG + SVG).

### Visualization (Experiment 2)

`temporal_seen_grid` marks the seen timesteps as **pale blue vertical lines**
(`ax.vlines`, alpha 0.18) instead of blue dots, overlaid on ground truth (ink) and
the PINN reconstruction (red dashed).

### Aggregate

`scripts/aggregate.py` writes `outputs/availability_comparison.{png,svg}` (median
relL2 vs availability, three lines) and `outputs/results_table.md`.

### Results (Lz=1, rho~85, gauge removed, 50 epochs/level)

Median held-out relative L2 (`outputs/results_table.md`,
`outputs/availability_comparison.{png,svg}`):

| Availability | Seen (temporal) | Neighborhood (spatial) | Far (extrap.) | train rows |
|---|---|---|---|---|
| 1%  | 25.385 | 40.986 | 28.503 | 24,388 |
| 5%  | 3.318  | 5.383  | 4.435  | 121,197 |
| 10% | 0.948  | 1.608  | 2.915  | 242,576 |
| 15% | 0.699  | 1.188  | 2.560  | 363,673 |
| 20% | 0.411  | 1.306  | 2.436  | 485,350 |
| 25% | 0.332  | 1.145  | 2.484  | 607,132 |

**Reading.** `Lz=1` makes the standardized speed ratio `rho ~ 85`, which heavily
weights the through-thickness (z) derivatives. The model fits the seen training
timesteps but generalizes poorly: even at 25% availability the seen-point temporal
error is ~0.33 (vs ~0.27 for the prior `rho=1` config at 1%), the neighborhood
spatial error plateaus near ~1.1, and far-field extrapolation stays ~2.5. This
reproduces the failure the original `PotentialScalers.fit` comment warned about.
Error falls monotonically with availability for seen points; spatial/far
generalization is largely data-availability-insensitive (the bottleneck is the
`rho=85` ill-conditioning, not the temporal sampling).

Per-neighborhood at 25% (`metrics.json["per_neighborhood"]`): `near_source`
seen=0.108 (high amplitude, easiest), `in_between` seen=0.332, `near_boundary`
seen=1.423 (low amplitude near the edge, hardest); neighborhood-spatial errors are
~1.0-1.3 across all three.

### Runtime

Measured wall-clock for the full sweep (CPU, float64, double-backward):
**~1 h 35 min** total (six levels run sequentially). Per level (training only):
1% 2.2 min, 5% 6.6 min, 10% 12.1 min, 15% 18.1 min, 20% 23.4 min, 25% 29.5 min,
plus ~1-1.5 min each for evaluation and figures. Runtime scales linearly with the
retained-row count.

---

## exp2_layerwise — layer-wise spatial sampling x temporal (from `Experiment details.docx`)

Separate, config-driven port of `june16_ffn_signal_reconstruction_(1).py`. Per-layer
random sampling at {1,10,20,30}% x 3 plies, 10% temporal, evaluate seen + unseen
points. See `exp2_layerwise/README.md`, `exp2_layerwise/HANDOFF.md` (full state),
and `exp2_layerwise/HPC_20PCT_COMMANDS.md` (RCI git workflow).

Best config: silu, alpha=3.0, F=256, 256x3, physics@30. Trainer is crash-safe
(checkpoint + `--resume`) and supports float32 for GPU speed. 20% runs on RCI via
`exp2_layerwise/run20.sbatch`.
