#!/bin/sh
# Evaluate the 20% layer-wise run from its wall-limit checkpoint (no further training).
#   cd /work/pnag/scinonet-pinn && sbatch experiments/exp2_layerwise/eval20.sbatch
#SBATCH --job-name=eval20
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH -p gpu-v100-32gb
#SBATCH --output logs/eval20-%j.out
#SBATCH --error  logs/eval20-%j.err

export CUDA_VISIBLE_DEVICES=0
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
mkdir -p logs
hostname; date; echo "repo root: $(pwd)"

module load cuda/11.0
module load python3/anaconda/2023.9
source activate /work/pnag/envs/ml_env
python --version

python -u experiments/exp2_layerwise/scripts/eval_from_ckpt.py \
    --out experiments/exp2_layerwise/outputs/sp20_t10_silu_F256_h256x256x256_a1_hpc \
    --which best
date
