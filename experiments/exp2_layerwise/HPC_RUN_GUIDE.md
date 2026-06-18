# HPC run guide — Experiment 2 (layer-wise sampling)

The 10/20/30% datasets and the full-spec runs (`--patience 180`, the activation /
architecture ablations, the full physics sweep) are GPU/HPC work. Everything is
config-driven; the same scripts run unchanged on a CUDA box (the code auto-selects
`cuda` when available and trains in float64).

## 0. Environment
```bash
pip install torch numpy pandas matplotlib h5py
```
Point the scripts at the same repo layout (they use paths relative to the repo
root). `3D_Pristine.mat` must be under `data/`.

## 1. Generate the datasets (needs h5py + the .mat)
```bash
for p in 1 10 20 30; do
  python experiments/exp2_layerwise/scripts/gen_layerwise_dataset.py --pct $p
done
```
Sizes scale linearly: 1% ~ 1800 points / 10.8M rows / ~0.9 GB CSV;
30% ~ 54,000 points / 324M rows / ~28 GB CSV. For 20/30% prefer a fast scratch
disk; if CSV is unwieldy, switch the generator to Parquet (one-line change in
`gen_layerwise_dataset.py`).

## 2. Physics-weight sweep (choose the highest workable alpha), per dataset
```bash
python experiments/exp2_layerwise/scripts/run_phys_sweep.py \
  --csv experiments/exp2_layerwise/data/dataset_layerwise_10pct_3ply_fullsignal_6001steps.csv \
  --alphas 0.3 0.5 1.0 2.0 4.0 8.0 --epochs 60 --data-only-epochs 10
# -> outputs/phys_sweep_<dataset>/sweep.json  ("recommended_alpha")
```

## 3. Experiment 1 — full runs (spec: patience 180)
```bash
A=<recommended_alpha>
for p in 1 10 20 30; do
  CSV=experiments/exp2_layerwise/data/dataset_layerwise_${p}pct_3ply_fullsignal_6001steps.csv
  python experiments/exp2_layerwise/scripts/run_exp.py \
    --csv "$CSV" --activation tanh --num-freq 160 --hidden 256 256 256 \
    --balance-alpha $A --epochs 2000 --patience 180 --temporal 0.10
done
```

## 4. Ablations (vary one axis at a time on a fixed dataset, e.g. 10%)
```bash
CSV=experiments/exp2_layerwise/data/dataset_layerwise_10pct_3ply_fullsignal_6001steps.csv
A=<recommended_alpha>
# activation: tanh (baseline), sin (SIREN), gelu, silu
for act in tanh sin gelu silu; do
  python .../run_exp.py --csv "$CSV" --activation $act --num-freq 160 --hidden 256 256 256 \
    --balance-alpha $A --epochs 2000 --patience 180 --tag-suffix _abl
done
# Fourier leaves
for F in 96 160 256; do
  python .../run_exp.py --csv "$CSV" --activation tanh --num-freq $F --hidden 256 256 256 \
    --balance-alpha $A --epochs 2000 --patience 180 --tag-suffix _abl
done
# width / depth
python .../run_exp.py --csv "$CSV" --hidden 128 128 128 --tag-suffix _abl ...
python .../run_exp.py --csv "$CSV" --hidden 256 256 256 256 --tag-suffix _abl ...
```

## 5. Collect
```bash
python experiments/exp2_layerwise/scripts/make_run_cards.py   # per-run cards + catalog table
```

## Activation recommendations (spec asked)
- `tanh` — current working baseline.
- `sin` (SIREN) — most principled for oscillatory wave fields; pair with the usual
  SIREN init if it is unstable (first-layer omega_0 scaling). Added here.
- `gelu` / `silu` — smooth, second-derivative-friendly (the physics uses
  double-backward), good robust alternatives; both already supported.
- A sine first layer with tanh interior layers is a common stable compromise; can
  be added if pure `sin` underperforms.

## Notes
- Each run saves `model.pt` (+ history/metrics/scalers/config), so prediction code
  can be modified and re-run without retraining.
- The visualization style is intentionally identical to the source script.
- Early stopping is on the **training** loss (patience epochs without improvement),
  with the best checkpoint chosen by validation data MSE.
