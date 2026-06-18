# HPC workflow (RCI) — git-based, crash-safe, 20% run

No more repeated scp. The repo is a git repo with large files ignored
(`.mat`, dataset CSVs, `*.pt`); code + results + state travel via git. The 16 GB
`.mat` already lives on RCI at `/work/pnag/data/3D_Pristine.mat`.

Best config (determined by the search): `silu`, `alpha=3.0`, `num_freq=256`,
`256x256x256`, `10% temporal`, physics after a 30-epoch warmup, early stop on the
training loss (patience 180). float32 for speed on the V100.

--------------------------------------------------------------------------------
## One-time: put the repo on a remote, clone on HPC
--------------------------------------------------------------------------------
On the Mac (create the GitHub repo via the website, then):
```bash
cd /Users/protiknag/Desktop/PINN
git remote add origin git@github.com:<YOU>/PINN.git     # or https URL
git push -u origin master
```
On RCI (clone into its own dir; symlink the existing .mat so it isn't re-downloaded):
```bash
cd /work/pnag
git clone git@github.com:<YOU>/PINN.git PINN     # or https
mkdir -p PINN/data
ln -s /work/pnag/data/3D_Pristine.mat PINN/data/3D_Pristine.mat
```
Thereafter sync with `git pull` / `git push` from either side.

--------------------------------------------------------------------------------
## Environment (RCI ml_env)
--------------------------------------------------------------------------------
```bash
module load cuda/11.0 python3/anaconda/2023.9
source activate /work/pnag/envs/ml_env
python -c "import torch,h5py,numpy,pandas,matplotlib,scipy; print('torch', torch.__version__)"
# install anything missing into the env, e.g.:  pip install h5py
```

--------------------------------------------------------------------------------
## Run the 20% experiment (generate-if-absent + train, crash-safe)
--------------------------------------------------------------------------------
```bash
cd /work/pnag/PINN
sbatch experiments/exp2_layerwise/run20.sbatch
squeue -u pnag
tail -f logs/pinn20-*.out                                   # live progress (python -u)
column -s, -t outputs/sp20_t10_silu_F256_h256x256x256_a3_hpc/progress.csv | tail
```
The job checkpoints every 5 epochs to `outputs/<tag>/checkpoint.pt` and passes
`--resume`, so **if it hits the wall again just `sbatch` it again** and it
continues from the last checkpoint. `run20.sbatch` writes its SLURM logs to
`logs/` (housekeeping); adjust `REPO=` inside it if the clone path differs.

--------------------------------------------------------------------------------
## Bring results back
--------------------------------------------------------------------------------
On RCI, commit the (small) outputs — figures, metrics.json, config.json,
progress.csv are tracked; weights/datasets are gitignored:
```bash
cd /work/pnag/PINN
git add experiments/exp2_layerwise/outputs experiments/exp2_layerwise/*.md
git commit -m "20% HPC run results"
git push
```
Then `git pull` on the Mac.

--------------------------------------------------------------------------------
## 10% / 30%
--------------------------------------------------------------------------------
Same as 20% with `--pct {10,30}` in the generate step and the matching CSV path in
`run_exp.py` (copy `run20.sbatch` to `run10.sbatch` / `run30.sbatch` and edit the
pct + CSV name).

## Notes
- float32 is the default in `run20.sbatch` for speed; if the wave-residual curve
  looks wrong, switch to `--dtype float64`.
- The 18 GB CSV load needs a >=64 GB node (`--mem=120G`).
- Disk: ~18 GB (20% CSV) + the symlinked 16 GB .mat.
- Fallback (no remote): scp a code tarball as before — see git history of this file.
