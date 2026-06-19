#!/bin/sh
# Experiment 2 (neighborhood-confined sampling) at 10% training budget, alpha=1.0.
# Same config as the 1% / 5% runs otherwise. Crash-safe: checkpoints every 5 epochs
# and --resume picks up outputs/<tag>/checkpoint.pt, so re-submitting continues.
#   cd /work/pnag/scinonet-pinn && sbatch experiments/exp3_neighborhood/run_nbhd_10pct.sbatch
#SBATCH --job-name=pinnNB10
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH -p gpu-v100-32gb
#SBATCH --signal=B:TERM@120
#SBATCH --output logs/pinnNB10-%j.out
#SBATCH --error  logs/pinnNB10-%j.err

export CUDA_VISIBLE_DEVICES=0
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
mkdir -p logs
hostname; date; echo "repo root: $(pwd)"

module load cuda/11.0
module load python3/anaconda/2023.9
source activate /work/pnag/envs/ml_env
python --version

# 10% TRAIN budget (6000 (x,y)/layer) confined to 6 neighborhoods (r~20mm); 20% of
# each held out (unseen-inside) + 150 points/layer far outside (unseen-outside).
# ~1250 inside pts/nbhd; ~10.8M train rows after 10% temporal subsample.
CSV=experiments/exp3_neighborhood/data/dataset_nbhd_10pct_n6_3ply_fullsignal_6001steps.csv
if [ ! -f "$CSV" ]; then
  echo "[job] generating 10% neighborhood dataset..."
  python -u experiments/exp3_neighborhood/scripts/gen_neighborhood_dataset.py \
      --pct 10 --n-nbhd 6 --inside-held-frac 0.20 --n-outside 150
fi

# Same working config as the 1% / 5% runs: silu, F=256, 256x3, 10% temporal,
# physics@30, patience 50 (early-stop on val), balanced alpha=1.0, float64.
python -u experiments/exp3_neighborhood/scripts/run_nbhd.py \
    --csv "$CSV" --activation silu --num-freq 256 --hidden 256 256 256 \
    --balance-alpha 1.0 --temporal 0.10 \
    --data-only-epochs 30 --epochs 2000 --patience 50 \
    --dtype float64 --ckpt-every 5 --resume
date
