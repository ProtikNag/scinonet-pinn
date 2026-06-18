"""Generate dense neighborhood full-signal datasets from 3D_Pristine.mat.

Selects, for each requested neighborhood center, the ``--per-center`` first-ply
points closest to that center (a dense 1 mm cluster), and writes their full
6001-timestep signals as a long-format CSV ``[x, y, z, t, u, v, w]``.

Only coordinate arrays and the selected rows are read from the 16 GB file.

Examples:
    # single neighborhood at the origin (100 points)
    python scripts/generate_neighbor_dataset.py --centers "0,0" --per-center 100

    # four neighborhoods, 50 points each (200 points)
    python scripts/generate_neighbor_dataset.py \
        --centers "-70,-60;70,-60;-70,-140;70,-140" --per-center 50
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import pandas as pd

MAT_PATH = "data/3D_Pristine.mat"
N_PLY_ROWS = 60000
MM = 1000.0


def parse_centers(text: str) -> list[tuple[float, float]]:
    centers = []
    for chunk in text.split(";"):
        cx, cy = chunk.split(",")
        centers.append((float(cx), float(cy)))
    return centers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--centers", default="0,0",
                        help='Neighborhood centers "x1,y1;x2,y2;..." in mm')
    parser.add_argument("--per-center", type=int, default=50)
    parser.add_argument("--all-plies", action="store_true",
                        help="include all 3 through-thickness layers (z=0,-1,-2 mm)")
    parser.add_argument("--mat", default=MAT_PATH)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    centers = parse_centers(args.centers)
    suffix = "_3ply" if args.all_plies else ""
    out = args.out or (
        f"data/dataset_{len(centers)}nbhd_{args.per_center}pts{suffix}_fullsignal_6001steps.csv"
    )

    with h5py.File(args.mat, "r") as f:
        dt = float(np.array(f["dt"]).ravel()[0])
        # full coordinate arrays (all plies) so cross-ply indices resolve
        x = np.array(f["X_zero_coord_ply"]).ravel() * MM
        y = np.array(f["Y_zero_coord_ply"]).ravel() * MM
        z = np.array(f["Z_zero_coord_ply"]).ravel() * MM

        chosen: list[int] = []
        for cx, cy in centers:
            # select clusters on the first ply only
            r = np.sqrt((x[:N_PLY_ROWS] - cx) ** 2 + (y[:N_PLY_ROWS] - cy) ** 2)
            idx = np.argsort(r)[: args.per_center]
            chosen.extend(idx.tolist())
            print(f"[gen] center ({cx},{cy}): {len(idx)} points, "
                  f"r_max={r[idx].max():.2f} mm")
        # de-duplicate (clusters should not overlap)
        base = sorted(set(chosen))
        if args.all_plies:
            # add the same (x,y) at plies 2 and 3 (rows +60000, +120000)
            sel = np.array(sorted(set(base) | {i + N_PLY_ROWS for i in base}
                                  | {i + 2 * N_PLY_ROWS for i in base}))
            print(f"[gen] all plies: {len(base)} (x,y) points x 3 layers = {len(sel)} rows")
        else:
            sel = np.array(base)

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
    print(f"[gen] wrote {len(df):,} rows ({n_pts} unique points x {n_t} steps) -> {out}")


if __name__ == "__main__":
    main()
