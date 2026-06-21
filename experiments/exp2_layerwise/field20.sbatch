#!/bin/sh
# Full-field layer-1 reconstruction (actual vs predicted) at fixed timesteps
# for the 20% layer-wise run.
#   cd /work/pnag/scinonet-pinn && sbatch experiments/exp2_layerwise/field20.sbatch
#SBATCH --job-name=field20
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH -p gpu-v100-32gb
#SBATCH --output logs/field20-%j.out
#SBATCH --error  logs/field20-%j.err

export CUDA_VISIBLE_DEVICES=0
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
mkdir -p logs
hostname; date

module load cuda/11.0
module load python3/anaconda/2023.9
source activate /work/pnag/envs/ml_env

python -u experiments/exp2_layerwise/scripts/reconstruct_field.py \
    --out experiments/exp2_layerwise/outputs/sp20_t10_silu_F256_h256x256x256_a1_hpc \
    --mat data/3D_Pristine.mat --timesteps 300 3000 5800
date
