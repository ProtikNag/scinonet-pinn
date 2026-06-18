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
REPO=/work/pnag/PINN              # <-- adjust if the repo lives elsewhere
cd "$REPO" || exit 1
mkdir -p logs
hostname; date

module load cuda/11.0
module load python3/anaconda/2023.9
source activate /work/pnag/envs/ml_env
python --version

CSV=experiments/exp2_layerwise/data/dataset_layerwise_20pct_3ply_fullsignal_6001steps.csv
if [ ! -f "$CSV" ]; then
  echo "[job] generating 20% dataset..."
  python -u experiments/exp2_layerwise/scripts/gen_layerwise_dataset.py --pct 20
fi

# Best config: silu, alpha=3.0, F=256, 256x3, 10% temporal, physics@30, patience 180.
# float32 for speed on the V100 (switch to float64 if the wave residual misbehaves).
python -u experiments/exp2_layerwise/scripts/run_exp.py \
    --csv "$CSV" --activation silu --num-freq 256 --hidden 256 256 256 \
    --balance-alpha 3.0 --temporal 0.10 \
    --data-only-epochs 30 --epochs 2000 --patience 180 \
    --dtype float32 --ckpt-every 5 --resume --tag-suffix _hpc
date
