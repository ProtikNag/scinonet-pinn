# SciNoNet — Fourier-Feature Reconstruction of Elastic-Wave Signals

Reconstruct the full 6001-timestep displacement signal of an elastic wave on a
200x300 mm plate from a temporally partial observation, using a Fourier-feature
network. This is the data-only baseline; the physics-informed (PINN) extension is
scaffolded and documented in [`HANDOFF.md`](HANDOFF.md).

> If you are picking this up cold, read [`HANDOFF.md`](HANDOFF.md) first. It
> records the data situation, the bugs fixed from the original notebooks, current
> results, and the exact steps to add the PDE residual.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate the dense 100-point neighbor dataset from the .mat (one time).
python scripts/generate_neighbor_dataset.py --n-points 100

# 2. Data-only baseline: reconstruct full signals from a temporal subsample.
python scripts/run_experiment.py --config configs/default.yaml \
    --override data.split_mode=subsample data.subsample_keep=0.10 --tag subsample10

# 3. PINN deliverables: temporal-holdout success + spatial-holdout A/B (~25 min, CPU).
python scripts/pinn_final.py --mode both --comp w
```

Figures (PNG + SVG), metrics, and checkpoints are written under `outputs/`.

The data-only model auto-selects a device (CUDA > Apple MPS > CPU); MPS runs in
float32. The PINN runs on CPU in float64 because it needs second-order
double-backward (unavailable in MPS float64).

---

## Repository layout

```
PINN/
├── README.md                 # this file
├── HANDOFF.md                # project log + road to the full PINN  <-- read this
├── requirements.txt
├── configs/
│   └── default.yaml          # all hyperparameters; no magic numbers in code
├── data/                     # 3D_Pristine.mat + generated neighbor CSV
├── notebooks/                # original notebooks, preserved untouched
├── scripts/
│   ├── generate_neighbor_dataset.py  # extract N closest points from the .mat
│   ├── run_experiment.py     # data-only: load -> train -> evaluate -> visualize
│   ├── sweep.py              # bandwidth / regularization sweep
│   ├── spatial_holdout.py    # data-only spatial-generalization probe
│   ├── pinn_final.py         # PINN temporal + spatial deliverables (figures)
│   ├── pinn_ab.py            # controlled physics-off vs physics-on A/B
│   └── run_pinn.py           # single-mode PINN vs data-only compare
├── src/scinonet/
│   ├── seed.py               # set_seed, device/dtype resolution
│   ├── data.py               # load, standardize, train/test split + holdout modes
│   ├── features.py           # Random Fourier Features (+ physics bandwidth)
│   ├── models.py             # FourierMLP (data-only baseline)
│   ├── pinn.py               # PINN: per-component wave eq, learnable speed,
│   │                         #   gradient-norm balancing, train/eval/reconstruct
│   ├── physics.py            # reference wave-residual notes
│   ├── train.py              # data-only training loop
│   ├── evaluate.py           # relative-L2 metrics + full-signal reconstruction
│   ├── config.py             # YAML -> objects
│   └── viz.py                # academic-style PNG+SVG plots
└── outputs/
    ├── data_only/            # baseline runs
    ├── pinn/                 # temporal/, spatial/, spatial_ab/
    └── comparisons/          # curated cross-run figures
```

---

## The task and the three split modes

The full-signal CSV is complete ground truth. Training uses a temporally partial
view of it; the held-out timesteps measure reconstruction quality honestly.
`data.split_mode` selects how the partial view is built:

| Mode | What is kept for training | Difficulty |
|------|---------------------------|------------|
| `subsample` | a random fraction of each point's timesteps | easy — small gaps |
| `shared` | the same time windows for every point | hard — wide data-free gaps |
| `mixed` | half the points keep the windows, half the complement | hard — needs spatial inference |

## Results

**Data-only baseline** (held-out relative L2):

| Mode | train relL2 | held-out relL2 |
|------|-------------|----------------|
| `subsample` | ~0.004 | ~0.004 |
| `mixed` / `shared` | ~0.01–0.05 | ~1.0 |

**PINN** (physics-informed: per-component wave equation + IC + BC, specialized
dispersion-informed Fourier B, `pinn_final.py`, 4 neighborhoods x 50 points):

| Setting | metric | data-only | PINN |
|---|---|---|---|
| Temporal, 1% samples | median (200 pts) | 0.56 | **0.27** |
| Temporal, 1% samples | demo point | 1.07 | **0.38** |
| Spatial, unseen points | median (8 pts) | 0.36 | **0.06** |
| Spatial, unseen points | demo point | 0.85 | **0.10** |

Two changes unlocked spatial generalization: the amplitude moved outside the
sinusoid (B stays a pure clamped frequency) and the wavenumber band was read off
the Lamb dispersion plot (kappa <= 0.8 rad/mm). The physically-correct band stops
the network indexing points with non-physical high spatial frequencies, so it
interpolates as a smooth wave field; the PDE+IC+BC then clean up the residual.
At never-seen points the held-out error drops to ~6% (every point 5-11%); at 1%
temporal samples to ~0.27.

Each run saves `model.pt`, `history.json`, `metrics.json`, and `loss_curves`
(data / PDE / IC / BC / total, train vs val) under
`outputs/pinn/<mode>/<phys_off|phys_on>/`; each mode also writes `plate_layout`
(point selection), `compare_<mode>_w`, and (spatial) `all_holdout_uvw` (every
held-out point x u,v,w). A concise write-up of the model and equations is in
[docs/report.pdf](docs/report.pdf). See [HANDOFF.md](HANDOFF.md) §5–6.

The Fourier network reconstructs the waveform near-perfectly wherever it has
temporal samples. Wide windows with no local data cannot be filled by a
data-only model — that is the role of the wave-equation prior, and the
motivation for the PINN stage. See [`HANDOFF.md`](HANDOFF.md) §5–6.

## Reproducibility

Every run seeds `random`, `numpy`, and `torch` (`scinonet.seed.set_seed`) and
forces deterministic cuDNN. All hyperparameters live in `configs/default.yaml`;
override any of them from the CLI with `--override a.b.c=value`.
