"""Generate a 3-neighborhood dataset of TIGHT, dense 1 mm clusters (like the original).

Unlike ``gen_3nbhd_dataset.py`` (random points in a 15 mm disk), this takes the
**50 nearest** first-ply points to each center, i.e. a contiguous ~4 mm 1 mm-grid
patch where the 45 train and 5 test points sit immediately adjacent. The 5 test
points are chosen from the cluster *interior* (all four 1 mm neighbors present in
the cluster) so each held-out point is flanked by training data on every side.

Centers along y = -99.5 mm:

    near_source   (-49.5, -99.5)   the excitation point
    in_between    ( 50.0, -99.5)   midway between source and the right edge
    near_boundary (149.5, -99.5)   ON the right plate edge x=149.5, so most of the
                                   cluster touches the boundary

All three through-thickness plies (z = 0, -1, -2 mm). Writes the long-format CSV
and a sidecar JSON with the per-neighborhood train/test split.

    python experiments/exp_3nbhd_temporal/scripts/gen_3nbhd_tight.py
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

MAT_PATH = "data/3D_Pristine.mat"
N_PLY_ROWS = 60000
MM = 1000.0

NEIGHBORHOODS = [
    ("near_source", -49.5, -99.5),
    ("in_between", 50.0, -99.5),
    ("near_boundary", 149.5, -99.5),
]


def interior_mask(xy: np.ndarray) -> np.ndarray:
    """True where all four 1 mm grid neighbors are present in the given cluster."""
    present = {tuple(np.round(p, 3)) for p in xy}
    out = np.zeros(len(xy), bool)
    for i, (x, y) in enumerate(np.round(xy, 3)):
        out[i] = all(n in present for n in [(round(x + 1, 3), y), (round(x - 1, 3), y),
                                            (x, round(y + 1, 3)), (x, round(y - 1, 3))])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-nbhd", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mat", default=MAT_PATH)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    out = args.out or ("experiments/exp_3nbhd_temporal/data/"
                       f"dataset_3nbhd_tight_{args.per_nbhd}pts_3ply_fullsignal_6001steps.csv")
    meta_path = out.replace(".csv", "_meta.json")

    import pandas as pd
    with h5py.File(args.mat, "r") as f:
        dt = float(np.array(f["dt"]).ravel()[0])
        x = np.array(f["X_zero_coord_ply"]).ravel() * MM
        y = np.array(f["Y_zero_coord_ply"]).ravel() * MM
        z = np.array(f["Z_zero_coord_ply"]).ravel() * MM
        xf, yf = x[:N_PLY_ROWS], y[:N_PLY_ROWS]

        meta = {"per_nbhd": args.per_nbhd, "n_test": args.n_test, "seed": args.seed,
                "tight": True, "neighborhoods": []}
        chosen: list[int] = []
        for name, cx, cy in NEIGHBORHOODS:
            r = np.sqrt((xf - cx) ** 2 + (yf - cy) ** 2)
            idx = np.argsort(r)[: args.per_nbhd]                 # the 50 nearest (dense)
            cluster_xy = np.stack([xf[idx], yf[idx]], 1)
            interior = np.where(interior_mask(cluster_xy))[0]
            if len(interior) < args.n_test:
                raise RuntimeError(f"{name}: only {len(interior)} interior points")
            test_local = rng.choice(interior, args.n_test, replace=False)
            train_local = np.setdiff1d(np.arange(args.per_nbhd), test_local)
            test_rows = idx[test_local].tolist()
            train_rows = idx[train_local].tolist()
            chosen.extend(idx.tolist())
            meta["neighborhoods"].append({
                "name": name, "center": [cx, cy], "r_max_mm": float(r[idx].max()),
                "n_interior": int(len(interior)),
                "train_rows": [int(i) for i in train_rows],
                "test_rows": [int(i) for i in test_rows],
                "train_xy": [[float(xf[i]), float(yf[i])] for i in train_rows],
                "test_xy": [[float(xf[i]), float(yf[i])] for i in test_rows],
            })
            print(f"[gen] {name:14s} center=({cx},{cy}) nearest-{args.per_nbhd} "
                  f"r_max={r[idx].max():.2f} mm, interior={len(interior)} "
                  f"-> 45 train + 5 test")

        base = sorted(set(chosen))
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
    print(f"[gen] wrote {len(df):,} rows -> {out}")
    print(f"[gen] metadata -> {meta_path}")


if __name__ == "__main__":
    main()
