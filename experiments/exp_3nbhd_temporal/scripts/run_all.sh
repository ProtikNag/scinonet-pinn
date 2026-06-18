#!/usr/bin/env bash
# Run the six temporal-availability levels one by one (Lz=1, gauge removed, 50 epochs),
# then aggregate. Each level prints a one-line result on completion.
set -e
PY=/Users/protiknag/anaconda3/envs/rlclass/bin/python
ROOT=/Users/protiknag/Desktop/PINN
cd "$ROOT"
LEVELS="0.01 0.05 0.10 0.15 0.20 0.25"
for k in $LEVELS; do
  echo "===== LEVEL keep=$k  $(date +%H:%M:%S) ====="
  $PY experiments/exp_3nbhd_temporal/scripts/run_level.py --keep "$k" --epochs 50 --n-far 10 --n-demo 6 \
    2>&1 | grep -E "^\[lvl|best val|done"
done
echo "===== AGGREGATE  $(date +%H:%M:%S) ====="
$PY experiments/exp_3nbhd_temporal/scripts/aggregate.py
echo "===== ALL DONE  $(date +%H:%M:%S) ====="
