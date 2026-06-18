"""Layer-wise spatial-sampling dataset generator for Experiment 1.

Per the experiment spec: for a chosen spatial percentage ``--pct`` (e.g. 1, 10, 20,
30), randomly select that fraction of the 60,000 unique first-ply (x, y) grid
points, take each selected point at all three through-thickness plies
(z = 0, -1, -2 mm), and write the full 6001-step signals long-format
``[x, y, z, t, u, v, w, is_boundary]``.

So 1% -> 600 (x, y) points -> 600 x 3 plies = 1800 spatial points.

Boundary coverage: the spec requires the training set to contain enough boundary
points. A fraction ``--boundary-frac`` of the budget is reserved for points on the
plate perimeter (within ``--edge-tol`` mm of x=+/-149.5 or y in {-0.5, -199.5}),
the rest filled with random interior points. Selected perimeter points are flagged
``is_boundary=1`` so the trainer can keep them in training (never spatially held
out).

Only coordinate arrays and the selected rows are read from the 16 GB .mat.

    # local, smallest
    python experiments/exp2_layerwise/scripts/gen_layerwise_dataset.py --pct 1
    # larger ones (run on HPC)
    python experiments/exp2_layerwise/scripts/gen_layerwise_dataset.py --pct 10
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd

MAT_PATH = "data/3D_Pristine.mat"
N_PLY_ROWS = 60000
MM = 1000.0
X_MIN, X_MAX = -149.5, 149.5
Y_MIN, Y_MAX = -199.5, -0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, required=True,
                    help="percent of the 60,000 per-layer (x,y) points to sample")
    ap.add_argument("--boundary-frac", type=float, default=0.20,
                    help="fraction of the budget reserved for perimeter points")
    ap.add_argument("--edge-tol", type=float, default=0.6,
                    help="mm distance from a plate edge counted as boundary")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mat", default=MAT_PATH)
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk-points", type=int, default=2000,
                    help="spatial points per CSV write chunk (bounds memory for large pct)")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    tag = f"{args.pct:g}pct".replace(".", "p")
    out = args.out or (f"experiments/exp2_layerwise/data/"
                       f"dataset_layerwise_{tag}_3ply_fullsignal_6001steps.csv")
    meta_path = out.replace(".csv", "_meta.json")

    with h5py.File(args.mat, "r") as f:
        dt = float(np.array(f["dt"]).ravel()[0])
        x = np.array(f["X_zero_coord_ply"]).ravel() * MM
        y = np.array(f["Y_zero_coord_ply"]).ravel() * MM
        z = np.array(f["Z_zero_coord_ply"]).ravel() * MM
        xf, yf = x[:N_PLY_ROWS], y[:N_PLY_ROWS]

        budget = int(round(args.pct / 100.0 * N_PLY_ROWS))
        on_edge = ((np.abs(xf - X_MIN) <= args.edge_tol) | (np.abs(xf - X_MAX) <= args.edge_tol)
                   | (np.abs(yf - Y_MIN) <= args.edge_tol) | (np.abs(yf - Y_MAX) <= args.edge_tol))
        edge_idx = np.where(on_edge)[0]
        interior_idx = np.where(~on_edge)[0]

        n_bound = min(len(edge_idx), int(round(args.boundary_frac * budget)))
        bound_pick = rng.choice(edge_idx, n_bound, replace=False)
        n_int = max(0, budget - n_bound)
        int_pick = rng.choice(interior_idx, min(n_int, len(interior_idx)), replace=False)
        base = np.sort(np.concatenate([bound_pick, int_pick]))
        is_bound_base = np.isin(base, edge_idx)
        print(f"[gen] pct={args.pct}: budget={budget} -> {len(base)} (x,y) points "
              f"({n_bound} boundary + {len(int_pick)} interior)")

        # all 3 plies
        sel = np.array(sorted(set(base.tolist())
                              | {i + N_PLY_ROWS for i in base}
                              | {i + 2 * N_PLY_ROWS for i in base}))
        print(f"[gen] {len(base)} (x,y) x 3 plies = {len(sel)} spatial points; "
              f"reading signals (this is the slow part)...", flush=True)

        u = np.array(f["Disp_x"][sel, :]) * MM
        v = np.array(f["Disp_y"][sel, :]) * MM
        w = np.array(f["Disp_z"][sel, :]) * MM

    # is_boundary per selected row (map base flag across plies)
    base_flag = {int(i): bool(b) for i, b in zip(base, is_bound_base)}
    bound_per_sel = np.array([base_flag[int(s % N_PLY_ROWS)] for s in sel], dtype=int)

    n_pts, n_t = u.shape
    t_full = np.arange(1, n_t + 1) * dt
    xs, ys, zs = x[sel], y[sel], z[sel]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # chunked write so a 200M+ row dataset never builds one giant DataFrame
    ch = max(1, args.chunk_points)
    n_chunks = (n_pts + ch - 1) // ch
    for ci, start in enumerate(range(0, n_pts, ch)):
        end = min(start + ch, n_pts)
        sl = slice(start, end)
        m = end - start
        sub = pd.DataFrame({
            "x": np.repeat(xs[sl], n_t), "y": np.repeat(ys[sl], n_t),
            "z": np.repeat(zs[sl], n_t), "t": np.tile(t_full, m),
            "u": u[sl].ravel(), "v": v[sl].ravel(), "w": w[sl].ravel(),
            "is_boundary": np.repeat(bound_per_sel[sl], n_t),
        })
        sub.to_csv(out, mode="w" if ci == 0 else "a", header=(ci == 0), index=False)
        if n_chunks > 1:
            print(f"[gen] wrote chunk {ci + 1}/{n_chunks} ({end}/{n_pts} points)", flush=True)
    total_rows = n_pts * n_t
    meta = {"pct": args.pct, "seed": args.seed, "budget_points": budget,
            "n_unique_xy": int(len(base)), "n_boundary": int(n_bound),
            "n_interior": int(len(int_pick)), "n_plies": 3,
            "n_spatial_points": int(len(sel)), "rows": int(total_rows),
            "boundary_frac": args.boundary_frac, "edge_tol_mm": args.edge_tol,
            "csv": out}
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[gen] wrote {total_rows:,} rows -> {out}")
    print(f"[gen] metadata -> {meta_path}")


if __name__ == "__main__":
    main()
