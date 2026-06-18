# Experiment 2 — Layer-wise spatial sampling x temporal holdout (from `Experiment details.docx`)

Physics-informed Helmholtz-potential reconstruction, following the current setup
in `june16_ffn_signal_reconstruction_(1).py` (ported to a config-driven module
here). The non-dimensionalization, Fourier features, four-potential network,
Helmholtz displacement, wave + Coulomb-gauge + IC + Dirichlet-BC physics,
gradient-norm balancing, and the **visualization style are kept intact**.

## What the spec asks (and where it lives)

**Experiment 1 — data availability.** Per z-layer, randomly sample {1, 10, 20, 30}%
of the 60,000 first-ply (x, y) grid points, taken at all 3 plies (so 1% -> 600 x 3
= 1800 spatial points), with **10% temporal** subsampling per trained point. The
training set must keep **enough boundary points**. Evaluate (a) unseen spatial
points (no timestep seen) and (b) seen-point reconstruction (10% temporal seen).
- Dataset: `scripts/gen_layerwise_dataset.py --pct {1,10,20,30}`
  (forces a `--boundary-frac` of the budget onto the plate perimeter, flags them
  `is_boundary=1`; those points are never spatially held out).
- Run: `scripts/run_exp.py --csv <dataset> --temporal 0.10 ...`

**Ablations.**
- Activation: `--activation {tanh,sin,gelu,silu}`. `tanh` is the working default;
  `sin` (SIREN-style) is added. *Recommended to also try:* `gelu`/`silu` (already
  present); for periodic wave fields `sin` is the most principled alternative, and
  a sine first layer with tanh interior is a common robust compromise.
- Architecture: `--hidden 256 256 256` (width/depth) and `--num-freq 160`
  (Fourier leaves) are sweepable from the CLI.
- Early stopping: training stops when the **training loss** has not improved for
  `--patience` epochs (spec: **180**), with a hard `--epochs` cap.
- **Physics weight (important).** The physics weight is the gradient-balancing
  factor `BALANCE_ALPHA`. `scripts/run_phys_sweep.py` sweeps it and recommends the
  **highest** value whose val-data MSE stays within `--tol`x the best, so the model
  is as physics-aware as possible without collapsing the data fit.

**Outputs / saving.** Every dataset is kept under `data/` (CSV + `_meta.json`).
Every run writes `outputs/<tag>/` with the loss-parameter curves, the seen-point
reconstruction, the unseen-point prediction (all PNG+SVG, current style),
`model.pt` (state_dict + history + metrics + scalers + config), `metrics.json`,
and `config.json`, so prediction code can be re-run later without retraining.

## Environments (this machine)
- dataset generation (reads `3D_Pristine.mat`): `~/anaconda3/bin/python` (h5py)
- training / eval (torch): `~/anaconda3/envs/rlclass/bin/python`

## Local vs HPC

CPU/float64 with the heavy double-backward physics makes the larger percentages
slow locally. Plan:
- **Local (here):** the 1% dataset (1800 points, 1.06M training rows) and short
  validation runs / a short physics sweep, to prove the pipeline and produce
  sample figures.
- **HPC (GPU):** the full-spec runs (`--patience 180`), the 10/20/30% datasets,
  the full physics sweep, and the activation/architecture ablations.

See `HPC_RUN_GUIDE.md` for copy-paste commands.

## Catalog

| run tag | data | temporal | activation | F | hidden | alpha | unseen | seen |
|---|---|---|---|---|---|---|---|---|
| sp1_t10_silu_F256_h256x256x256_a3_best | 1.0% | 10% | silu | 256 | 256x256x256 | 3.0 | 1.274 | 0.273 |
| sp1_t10_silu_F256_h256x256x256_a3_warm30 | 1.0% | 10% | silu | 256 | 256x256x256 | 3.0 | 1.238 | 0.137 |

(Per-run cards are written into each `outputs/<tag>/README.md` by
`scripts/make_run_cards.py`.)
