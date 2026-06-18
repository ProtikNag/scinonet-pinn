"""Generate the 3-neighborhood full-signal dataset for the temporal-availability study.

Three larger neighborhoods are placed along the line y = -99.5 mm, spanning the
plate from the excitation source to the right edge:

    near_source   center (-49.5,  -99.5)   the measured wave source (peak |w|)
    in_between    center ( 38.0,  -99.5)   midway between source and boundary
    near_boundary center (125.0,  -99.5)   close to the right plate edge x=149.5

Each neighborhood is a disk of radius ``--radius`` mm (larger than the earlier
~4 mm clusters). From the dense 1 mm grid inside each disk we draw 50 points at
random (not the nearest 50): 45 are training points and 5 are spatially held-out
test points. Every selected (x, y) is taken at all three through-thickness plies
(z = 0, -1, -2 mm), and the full 6001-step signal is written long-format.

A sidecar JSON records, per neighborhood, the chosen (x, y), the train/test split,
and the global row indices into the .mat first ply (so far-field evaluation can
exclude them). Only coordinate arrays and the selected rows are read from the
16 GB file.

    python experiments/exp_3nbhd_temporal/scripts/gen_3nbhd_dataset.py
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

# (name, center_x, center_y) in mm. Source measured at (-49.5, -99.5).
NEIGHBORHOODS = [
    ("near_source", -49.5, -99.5),
    ("in_between", 38.0, -99.5),
    ("near_boundary", 125.0, -99.5),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=15.0,
                        help="neighborhood disk radius in mm (spatial extent)")
    parser.add_argument("--per-nbhd", type=int, default=50,
                        help="points sampled per neighborhood (45 train + 5 test)")
    parser.add_argument("--n-test", type=int, default=5,
                        help="spatially held-out test points per neighborhood")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mat", default=MAT_PATH)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    out = args.out or (
        f"experiments/exp_3nbhd_temporal/data/"
        f"dataset_3nbhd_{args.per_nbhd}pts_r{int(args.radius)}_3ply_fullsignal_6001steps.csv"
    )
    meta_path = out.replace(".csv", "_meta.json")

    with h5py.File(args.mat, "r") as f:
        dt = float(np.array(f["dt"]).ravel()[0])
        x = np.array(f["X_zero_coord_ply"]).ravel() * MM
        y = np.array(f["Y_zero_coord_ply"]).ravel() * MM
        z = np.array(f["Z_zero_coord_ply"]).ravel() * MM
        xf, yf = x[:N_PLY_ROWS], y[:N_PLY_ROWS]

        meta = {"radius_mm": args.radius, "per_nbhd": args.per_nbhd,
                "n_test": args.n_test, "seed": args.seed, "neighborhoods": []}
        chosen: list[int] = []
        for name, cx, cy in NEIGHBORHOODS:
            r = np.sqrt((xf - cx) ** 2 + (yf - cy) ** 2)
            pool = np.where(r <= args.radius)[0]
            if len(pool) < args.per_nbhd:
                raise RuntimeError(f"{name}: only {len(pool)} points within r={args.radius}")
            # random selection from the disk (not nearest-50)
            pick = rng.choice(pool, args.per_nbhd, replace=False)
            # random train/test split inside the neighborhood
            perm = rng.permutation(args.per_nbhd)
            test_local = perm[: args.n_test]
            train_local = perm[args.n_test:]
            test_rows = pick[test_local].tolist()
            train_rows = pick[train_local].tolist()
            chosen.extend(pick.tolist())
            meta["neighborhoods"].append({
                "name": name, "center": [cx, cy], "n_in_disk": int(len(pool)),
                "r_max_mm": float(r[pick].max()),
                "train_rows": [int(i) for i in train_rows],
                "test_rows": [int(i) for i in test_rows],
                "train_xy": [[float(xf[i]), float(yf[i])] for i in train_rows],
                "test_xy": [[float(xf[i]), float(yf[i])] for i in test_rows],
            })
            print(f"[gen] {name:14s} center=({cx},{cy}) disk={len(pool)} pts "
                  f"-> {len(train_rows)} train + {len(test_rows)} test, r_max={r[pick].max():.1f} mm")

        base = sorted(set(chosen))
        # 3 plies: same (x,y) at rows +0, +60000, +120000
        sel = np.array(sorted(set(base) | {i + N_PLY_ROWS for i in base}
                              | {i + 2 * N_PLY_ROWS for i in base}))
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
    meta["csv"] = out
    meta["n_unique_xy"] = len(base)
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[gen] wrote {len(df):,} rows ({n_pts} spatial x {n_t} steps) -> {out}")
    print(f"[gen] metadata -> {meta_path}")


if __name__ == "__main__":
    main()
