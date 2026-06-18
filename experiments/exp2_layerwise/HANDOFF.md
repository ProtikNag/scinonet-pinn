# Experiment 2 (layer-wise sampling) — HANDOFF

State and decisions for this experiment, so any next session (local or on HPC) can
continue without re-deriving anything. Last updated 2026-06-18.

---

## 1. What this experiment is

Physics-informed Helmholtz-potential reconstruction of guided-wave displacement,
following `june16_ffn_signal_reconstruction_(1).py` (the "current setup"), ported
to a config-driven module here. Spec lives in `Experiment details.docx`. Goal:
sample a fraction of the plate's points per z-layer, train on 10% of timesteps,
and measure reconstruction at **seen** points (temporal infill) and **unseen**
spatial points (full prediction).

The physics, non-dimensionalization (`beta_y`, `beta_z`, `gamma`), Fourier
features, four-potential network, wave + Coulomb-gauge + IC + Dirichlet-BC losses,
gradient-norm balancing, and the **visualization style** are kept identical to the
source script.

## 2. Code map (`experiments/exp2_layerwise/scripts/`)

- `scinonet_pinn.py` — config-driven core (model, physics, training, viz). Set
  module globals, then call `build_dataset` / `make_net` / `train` / `plot_*`.
- `gen_layerwise_dataset.py` — build a dataset from the `.mat` for a spatial
  percentage; per-layer random sample at 3 plies, reserves perimeter points
  (`is_boundary=1`, never spatially held out). Chunked writer (bounded memory).
- `run_exp.py` — one configuration: train, evaluate (seen/unseen), save figures +
  `model.pt` + `metrics.json` + `config.json` + `checkpoint.pt` + `progress.csv`.
- `run_search.py` — config search (activation x alpha, then architecture).
- `run_phys_sweep.py` — physics-weight (alpha) sweep.
- `plot_plate.py` — plate-layout figure for a run or dataset.
- `make_run_cards.py` — per-run README cards + catalog table refresh.
- `run20.sbatch` — RCI SLURM job for the 20% run (crash-safe, resume).

## 3. What we determined (config search on the 0.2% set, 16 trials)

- **Activation:** gelu/silu beat tanh/sin on the data fit; **silu** chosen.
- **Physics weight:** higher `alpha` lowers the wave residual (more physics-aware)
  at the cost of the seen fit. Per the spec ("choose the highest physics-aware"),
  **alpha = 3.0**.
- **Architecture:** **num_freq = 256** beats 160/96; **256x256x256** beats 128x3
  and 256x4.
- Results in `outputs/search_*/search.{csv,json}`.

**Best config (use this everywhere):**
`--activation silu --num-freq 256 --hidden 256 256 256 --balance-alpha 3.0
--temporal 0.10 --data-only-epochs 30` (physics turns on after a 30-epoch
data-only warmup, which roughly halved the seen error vs warmup=12).

## 4. Runs done (local, 1% set)

- `outputs/sp1_t10_silu_F256_h256x256x256_a3_best/` — physics@12: seen 0.273,
  unseen 1.274.
- `outputs/sp1_t10_silu_F256_h256x256x256_a3_warm30/` — physics@30: seen **0.137**,
  unseen 1.238 (better seen fit). u, v reconstruct well; w (out-of-plane, ~10x
  smaller) is poorly fit. Unseen-spatial stays poor at 1% (sparse sampling →
  spatial-undersampling / PINN propagation failure; not a code bug).

Datasets generated locally: `data/dataset_layerwise_{0p2,1}pct_3ply_*.csv`
(gitignored). 10/20/30% are generated on HPC from the `.mat`.

## 5. HPC status and the fixes that matter

- **20% job timed out at 24 h and lost everything** because the old trainer saved
  only at the end. FIXED:
  - `train()` now **checkpoints every `--ckpt-every` epochs** to
    `outputs/<tag>/checkpoint.pt` (model + optimizer + history + counters), saves
    on SIGTERM/SIGINT, and `--resume` continues from it. Re-submitting the job
    resumes automatically.
  - **`model.pt` is saved right after training** (before eval/figures), so an
    interruption during plotting no longer loses the model.
- **Speed:** `--dtype float32` (~2x on a V100; validate the wave residual),
  training tensors kept resident on the device with index batching (no per-batch
  host->device copies), `--batch-size` configurable.
- **Progress:** every epoch is flushed to stdout (epoch time + ETA) and appended
  to `outputs/<tag>/progress.csv`. Run python with `-u` in jobs.
- **Memory:** `MAX_TEST_ROWS` caps the validation tensors (the 18 GB / 216M-row
  20% set otherwise materializes ~10 GB of unused test tensors). Loading the CSV
  still needs a >=64 GB node (`--mem=120G`).

## 6. How to run the 20% on HPC (RCI)

The `.mat` is already at `/work/pnag/data/3D_Pristine.mat`. Put the repo at
`/work/pnag/PINN` (git clone — see `HPC_20PCT_COMMANDS.md`) and symlink the .mat
into it: `ln -s /work/pnag/data/3D_Pristine.mat /work/pnag/PINN/data/3D_Pristine.mat`.
Then:
```bash
cd /work/pnag/PINN
sbatch experiments/exp2_layerwise/run20.sbatch     # generates data if absent, trains, resumes
```
If it hits the wall again, just `sbatch` the same script — it resumes from the
checkpoint. Watch progress: `tail -f logs/pinn20-*.out` or
`column -s, -t outputs/sp20_*_hpc/progress.csv | tail`.

## 7. Open / next

- Confirm float32 keeps the wave residual sane on the V100; fall back to float64
  if not.
- 20% expected runtime: ~hours on a V100 (float32); resume covers overruns.
- Unseen-spatial generalization is the hard part; needs structural help (denser
  collocation near held-out points, locality prior, or a source term), not just
  more data. See the chat discussion on propagation failure.
- 10% and 30% datasets/runs are the same commands with `--pct {10,30}`.
