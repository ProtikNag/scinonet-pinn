#!/usr/bin/env bash
# Local (CPU) Experiment-1 sequence on the 1% dataset: a lean physics-weight sweep
# to choose alpha, then a full 1% run with that alpha. The larger percentages and
# the full-spec (patience 180) runs go to HPC (see HPC_RUN_GUIDE.md).
set -e
PY=/Users/protiknag/anaconda3/envs/rlclass/bin/python
ROOT=/Users/protiknag/Desktop/PINN
cd "$ROOT"
S=experiments/exp2_layerwise/scripts
CSV=experiments/exp2_layerwise/data/dataset_layerwise_1pct_3ply_fullsignal_6001steps.csv
BASE=$(basename "$CSV" .csv)

echo "===== PHYSICS-WEIGHT SWEEP  $(date +%H:%M:%S) ====="
$PY $S/run_phys_sweep.py --csv "$CSV" --alphas 0.3 1.0 3.0 \
    --epochs 25 --data-only-epochs 6 --num-freq 160 --hidden 256 256 256

ALPHA=$($PY -c "import json;print(json.load(open('experiments/exp2_layerwise/outputs/phys_sweep_${BASE}/sweep.json'))['recommended_alpha'])")
echo "===== CHOSEN ALPHA = $ALPHA  $(date +%H:%M:%S) ====="

echo "===== FULL 1% RUN (tanh, F160, 256x3, alpha=$ALPHA)  $(date +%H:%M:%S) ====="
$PY $S/run_exp.py --csv "$CSV" --activation tanh --num-freq 160 --hidden 256 256 256 \
    --balance-alpha "$ALPHA" --epochs 120 --patience 40 --data-only-epochs 12

echo "===== RUN CARDS  $(date +%H:%M:%S) ====="
$PY $S/make_run_cards.py || true
echo "===== LOCAL DONE  $(date +%H:%M:%S) ====="
