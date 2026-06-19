"""Neighborhood-confined spatial-sampling dataset generator (Experiment 2).

Difference from Experiment 1 (`gen_layerwise_dataset.py`): instead of scattering the
sampled (x, y) points randomly across the whole 60,000-point per-layer grid, the
training points are *confined to n compact neighborhoods*. The result has three
disjoint roles, written to a ``role`` column so the trainer can split them:

    role=train         points inside a neighborhood, used for training
    role=inside_held   points inside a neighborhood, NOT used for training
                       -> "unseen point INSIDE the neighborhood" test
    role=outside_held  points outside every neighborhood, never trained
                       -> "unseen point OUTSIDE the neighborhood" test

The spec keeps 1% as the *training* budget: 1% of 60,000 = 600 (x, y) points/layer.
Each neighborhood holds ``inside_per_nbhd`` nearest grid points; a fraction
``--inside-held-frac`` of those is held out (inside_held), the rest train. The
neighborhoods are sized so the kept (train) points total the 1% budget. A separate
pool of ``--n-outside`` grid points far from every neighborhood forms outside_held.

Every selected (x, y) is taken at all three plies (z = 0, -1, -2 mm) with the full
6001-step signal, long-format ``[x, y, z, t, u, v, w, is_boundary, nbhd, role]``.

    python experiments/exp3_neighborhood/scripts/gen_neighborhood_dataset.py --pct 1
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


def _k_nearest(center, pts, k, taken):
    """Indices of the k nearest still-available points to `center`."""
    d2 = (pts[:, 0] - center[0]) ** 2 + (pts[:, 1] - center[1]) ** 2
    d2[taken] = np.inf
    return np.argpartition(d2, k)[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, default=1.0,
                    help="percent of the 60,000 per-layer points used for TRAINING")
    ap.add_argument("--n-nbhd", type=int, default=6, help="number of neighborhoods")
    ap.add_argument("--inside-held-frac", type=float, default=0.20,
                    help="fraction of each neighborhood held out (inside-unseen test)")
    ap.add_argument("--n-outside", type=int, default=150,
                    help="(x,y) points sampled far from every neighborhood (outside-unseen test)")
    ap.add_argument("--edge-tol", type=float, default=0.6,
                    help="mm distance from a plate edge counted as boundary")
    ap.add_argument("--center-margin", type=float, default=18.0,
                    help="keep neighborhood centers this many mm inside the plate edges")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mat", default=MAT_PATH)
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk-points", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    tag = f"{args.pct:g}pct".replace(".", "p")
    out = args.out or (f"experiments/exp3_neighborhood/data/"
                       f"dataset_nbhd_{tag}_n{args.n_nbhd}_3ply_fullsignal_6001steps.csv")
    meta_path = out.replace(".csv", "_meta.json")

    budget = int(round(args.pct / 100.0 * N_PLY_ROWS))            # train target / layer
    inside_per_nbhd = int(np.ceil(budget / (1.0 - args.inside_held_frac) / args.n_nbhd))
    # typical neighborhood radius (uniform 1mm grid -> ~1 point/mm^2) and a center
    # separation that keeps neighborhoods disjoint with a buffer.
    rho = N_PLY_ROWS / (abs(X_MAX - X_MIN) * abs(Y_MAX - Y_MIN))   # points per mm^2
    r_typ = float(np.sqrt(inside_per_nbhd / (np.pi * rho)))
    min_sep = 2.6 * r_typ

    with h5py.File(args.mat, "r") as f:
        dt = float(np.array(f["dt"]).ravel()[0])
        x = np.array(f["X_zero_coord_ply"]).ravel() * MM
        y = np.array(f["Y_zero_coord_ply"]).ravel() * MM
        z = np.array(f["Z_zero_coord_ply"]).ravel() * MM
        xf, yf = x[:N_PLY_ROWS], y[:N_PLY_ROWS]
        pts = np.stack([xf, yf], axis=1)

        # ── choose n_nbhd well-separated centers in the interior ──
        cand = np.where((xf > X_MIN + args.center_margin) & (xf < X_MAX - args.center_margin)
                        & (yf > Y_MIN + args.center_margin) & (yf < Y_MAX - args.center_margin))[0]
        centers_idx = []
        tries = 0
        while len(centers_idx) < args.n_nbhd and tries < 20000:
            c = cand[rng.randint(len(cand))]
            cp = pts[c]
            if all(np.hypot(cp[0] - pts[j, 0], cp[1] - pts[j, 1]) > min_sep for j in centers_idx):
                centers_idx.append(c)
            tries += 1
        if len(centers_idx) < args.n_nbhd:
            raise RuntimeError(f"only placed {len(centers_idx)}/{args.n_nbhd} centers; "
                               f"lower --n-nbhd or --center-margin")
        centers = pts[centers_idx]

        # ── grow each neighborhood by nearest points (disjoint via `taken`) ──
        taken = np.zeros(N_PLY_ROWS, bool)
        nbhd_members = []      # list of arrays of base indices, one per neighborhood
        for ci in range(args.n_nbhd):
            members = _k_nearest(centers[ci], pts, inside_per_nbhd, taken)
            taken[members] = True
            nbhd_members.append(members)

        # ── split each neighborhood into train / inside_held (spread per nbhd) ──
        base_role = {}     # base idx -> role
        base_nbhd = {}     # base idx -> neighborhood id
        for ci, members in enumerate(nbhd_members):
            m = members.copy()
            rng.shuffle(m)
            n_held = int(round(args.inside_held_frac * len(m)))
            held, train = m[:n_held], m[n_held:]
            for i in train:
                base_role[int(i)] = "train"; base_nbhd[int(i)] = ci
            for i in held:
                base_role[int(i)] = "inside_held"; base_nbhd[int(i)] = ci

        # ── outside pool: grid points far from EVERY neighborhood ──
        cdist = np.full(N_PLY_ROWS, np.inf)
        for ci in range(args.n_nbhd):
            d = np.hypot(pts[:, 0] - centers[ci, 0], pts[:, 1] - centers[ci, 1])
            cdist = np.minimum(cdist, d)
        far = np.where((cdist > r_typ + min_sep) & (~taken))[0]
        n_out = min(args.n_outside, len(far))
        out_pick = rng.choice(far, n_out, replace=False)
        for i in out_pick:
            base_role[int(i)] = "outside_held"; base_nbhd[int(i)] = -1

        base = np.array(sorted(base_role.keys()))
        on_edge = ((np.abs(xf[base] - X_MIN) <= args.edge_tol) | (np.abs(xf[base] - X_MAX) <= args.edge_tol)
                   | (np.abs(yf[base] - Y_MIN) <= args.edge_tol) | (np.abs(yf[base] - Y_MAX) <= args.edge_tol))
        base_bound = {int(b): bool(e) for b, e in zip(base, on_edge)}

        n_train = sum(v == "train" for v in base_role.values())
        n_in_held = sum(v == "inside_held" for v in base_role.values())
        print(f"[gen] pct={args.pct} budget(train)={budget} | n_nbhd={args.n_nbhd} "
              f"inside_per_nbhd={inside_per_nbhd} r_typ={r_typ:.1f}mm min_sep={min_sep:.1f}mm")
        print(f"[gen] per-layer (x,y): train={n_train} inside_held={n_in_held} "
              f"outside_held={n_out} (total {len(base)})")

        # all 3 plies
        sel = np.array(sorted(set(base.tolist())
                              | {i + N_PLY_ROWS for i in base}
                              | {i + 2 * N_PLY_ROWS for i in base}))
        print(f"[gen] {len(base)} (x,y) x 3 plies = {len(sel)} spatial points; "
              f"reading signals (slow part)...", flush=True)
        u = np.array(f["Disp_x"][sel, :]) * MM
        v = np.array(f["Disp_y"][sel, :]) * MM
        w = np.array(f["Disp_z"][sel, :]) * MM

    # per-selected-row role / nbhd / boundary (map base-point attributes across plies)
    sel_base = (sel % N_PLY_ROWS).astype(int)
    role_sel = np.array([base_role[int(b)] for b in sel_base])
    nbhd_sel = np.array([base_nbhd[int(b)] for b in sel_base], dtype=int)
    bound_sel = np.array([int(base_bound[int(b)]) for b in sel_base], dtype=int)

    n_pts, n_t = u.shape
    t_full = np.arange(1, n_t + 1) * dt
    xs, ys, zs = x[sel], y[sel], z[sel]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ch = max(1, args.chunk_points)
    n_chunks = (n_pts + ch - 1) // ch
    for ci, start in enumerate(range(0, n_pts, ch)):
        end = min(start + ch, n_pts)
        sl = slice(start, end); m = end - start
        sub = pd.DataFrame({
            "x": np.repeat(xs[sl], n_t), "y": np.repeat(ys[sl], n_t),
            "z": np.repeat(zs[sl], n_t), "t": np.tile(t_full, m),
            "u": u[sl].ravel(), "v": v[sl].ravel(), "w": w[sl].ravel(),
            "is_boundary": np.repeat(bound_sel[sl], n_t),
            "nbhd": np.repeat(nbhd_sel[sl], n_t),
            "role": np.repeat(role_sel[sl], n_t),
        })
        sub.to_csv(out, mode="w" if ci == 0 else "a", header=(ci == 0), index=False)
        if n_chunks > 1:
            print(f"[gen] wrote chunk {ci + 1}/{n_chunks} ({end}/{n_pts} points)", flush=True)

    total_rows = n_pts * n_t
    meta = {"experiment": "neighborhood", "pct_train": args.pct, "seed": args.seed,
            "budget_train_points": budget, "n_nbhd": args.n_nbhd,
            "inside_per_nbhd": inside_per_nbhd, "inside_held_frac": args.inside_held_frac,
            "r_typ_mm": r_typ, "min_sep_mm": min_sep, "center_margin_mm": args.center_margin,
            "n_train_xy": int(n_train), "n_inside_held_xy": int(n_in_held),
            "n_outside_held_xy": int(n_out), "n_unique_xy": int(len(base)),
            "n_plies": 3, "n_spatial_points": int(len(sel)), "rows": int(total_rows),
            "centers_xy": centers.tolist(), "edge_tol_mm": args.edge_tol, "csv": out}
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[gen] wrote {total_rows:,} rows -> {out}")
    print(f"[gen] metadata -> {meta_path}")


if __name__ == "__main__":
    main()
