#!/bin/sh
# 20% layer-wise experiment on RCI (gpu-v100-32gb). Generates the data if absent,
# then trains with the best config. Crash-safe: checkpoints every 5 epochs and
# --resume picks up outputs/<tag>/checkpoint.pt, so re-submitting continues.
#   sbatch experiments/exp2_layerwise/run20.sbatch
#SBATCH --job-name=pinn20
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH -p gpu-v100-32gb
#SBATCH --signal=B:TERM@120
#SBATCH --output logs/pinn20-%j.out
#SBATCH --error  logs/pinn20-%j.err

export CUDA_VISIBLE_DEVICES=0
# Run from the repo root = the directory you ran `sbatch` from. Location-independent,
# so the clone dir name (PINN, scinonet-pinn, ...) does not matter. Submit with:
#   cd /work/pnag/scinonet-pinn && sbatch experiments/exp2_layerwise/run20.sbatch
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
mkdir -p logs
hostname; date; echo "repo root: $(pwd)"

module load cuda/11.0
module load python3/anaconda/2023.9
source activate /work/pnag/envs/ml_env
python --version

CSV=experiments/exp2_layerwise/data/dataset_layerwise_20pct_3ply_fullsignal_6001steps.csv
if [ ! -f "$CSV" ]; then
  echo "[job] generating 20% dataset..."
  python -u experiments/exp2_layerwise/scripts/gen_layerwise_dataset.py --pct 20
fi

# Config: silu, F=256, 256x3, 10% temporal, physics@30, patience 50 (early-stop on val).
# alpha=1.0 (not 3.0): the gradient-norm balancer scales the physics gradient to
#   alpha x the data gradient each step. At alpha=3.0 physics dominated 3:1 and drove
#   the data fit *uphill* on the 20% set (val 1.6e-2 -> 8e-2 over 16 epochs); 1.0
#   balances them so physics regularizes without destroying the fit.
# float64 (not float32): the wave residual uses 2nd-order autograd (q_tt, q_xx, ...);
#   float32 noise there amplified the imbalance above. float64 is ~2x slower but sane
#   (the HANDOFF flagged exactly this fallback). Resume covers any 48h wall overrun.
python -u experiments/exp2_layerwise/scripts/run_exp.py \
    --csv "$CSV" --activation silu --num-freq 256 --hidden 256 256 256 \
    --balance-alpha 1.0 --temporal 0.10 \
    --data-only-epochs 30 --epochs 2000 --patience 50 \
    --dtype float64 --ckpt-every 5 --resume --tag-suffix _hpc
date
