"""Generate a plate-spanning dataset: 6 larger domains + boundary points, 3 plies.

The full plate (x in [-149.5,149.5], y in [-199.5,-0.5]) is split into a 3x2 grid
of six ~100x100 mm domains. From each domain ``--per-domain`` random first-ply
points are drawn, plus ``--n-boundary`` points on the plate perimeter. Every
selected (x,y) is taken at all three through-thickness plies (z=0,-1,-2 mm).

    python scripts/generate_domains.py --per-domain 100 --n-boundary 10
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import pandas as pd

MAT = "data/3D_Pristine.mat"
N_PLY = 60000
MM = 1000.0
X_EDGES = [-149.5, -49.5, 49.5, 149.5]   # 3 columns
Y_EDGES = [-199.5, -99.5, -0.5]          # 2 rows
EDGE_TOL = 0.6                            # within 0.6 mm of a plate edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=100)
    ap.add_argument("--n-boundary", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    out = args.out or f"data/dataset_6domain_{args.per_domain}pts_3ply_fullsignal_6001steps.csv"

    with h5py.File(MAT, "r") as f:
        dt = float(np.array(f["dt"]).ravel()[0])
        x = np.array(f["X_zero_coord_ply"]).ravel() * MM
        y = np.array(f["Y_zero_coord_ply"]).ravel() * MM
        z = np.array(f["Z_zero_coord_ply"]).ravel() * MM
        xf, yf = x[:N_PLY], y[:N_PLY]

        chosen = []
        for cx in range(3):
            for ry in range(2):
                m = ((xf >= X_EDGES[cx]) & (xf <= X_EDGES[cx + 1])
                     & (yf >= Y_EDGES[ry]) & (yf <= Y_EDGES[ry + 1]))
                pool = np.where(m)[0]
                pick = rng.choice(pool, min(args.per_domain, len(pool)), replace=False)
                chosen.extend(pick.tolist())
                print(f"[gen] domain (col{cx},row{ry}): {len(pick)} pts in "
                      f"x[{X_EDGES[cx]},{X_EDGES[cx+1]}] y[{Y_EDGES[ry]},{Y_EDGES[ry+1]}]")

        on_edge = ((np.abs(xf - (-149.5)) < EDGE_TOL) | (np.abs(xf - 149.5) < EDGE_TOL)
                   | (np.abs(yf - (-199.5)) < EDGE_TOL) | (np.abs(yf - (-0.5)) < EDGE_TOL))
        bpool = np.where(on_edge)[0]
        bpick = rng.choice(bpool, args.n_boundary, replace=False)
        chosen.extend(bpick.tolist())
        print(f"[gen] boundary: {len(bpick)} points on the plate perimeter")

        base = sorted(set(chosen))
        sel = np.array(sorted(set(base) | {i + N_PLY for i in base} | {i + 2 * N_PLY for i in base}))
        print(f"[gen] {len(base)} (x,y) points x 3 plies = {len(sel)} spatial points")

        u = np.array(f["Disp_x"][sel, :]) * MM
        v = np.array(f["Disp_y"][sel, :]) * MM
        w = np.array(f["Disp_z"][sel, :]) * MM

    n_pts, n_t = u.shape
    t_full = np.arange(1, n_t + 1) * dt
    df = pd.DataFrame({
        "x": np.repeat(x[sel], n_t), "y": np.repeat(y[sel], n_t),
        "z": np.repeat(z[sel], n_t), "t": np.tile(t_full, n_pts),
        "u": u.ravel(), "v": v.ravel(), "w": w.ravel(),
    })
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[gen] wrote {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
