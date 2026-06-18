"""Far-field spatial extrapolation for the Helmholtz-potential model.

Loads the trained spatial models (phys_off / phys_on) and evaluates them at
random plate points far from the training neighborhood (true extrapolation, no
nearby data, outside the collocation domain). Reconstructs u, v, w at the z=0
surface and compares to ground truth from the .mat.

    python scripts/potential_far.py --n 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scinonet.potential import PotentialNet, displacement, relative_l2  # noqa: E402
from scinonet import viz  # noqa: E402
from run_potential import make_features, CP_MM_PER_S, CS_MM_PER_S  # noqa: E402

MAT = "data/3D_Pristine.mat"
N_PLY = 60000
MM = 1000.0
DT = 1e-8
N_T = 6001
CPU = torch.device("cpu")


def load_model(path):
    ck = torch.load(path, weights_only=False)
    sc = ck["scalers"]
    net = PotentialNet(make_features(sc, 0), [256, 256, 256],
                       chatp_init=sc.chat(CP_MM_PER_S), chats_init=sc.chat(CS_MM_PER_S))
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, sc


def reconstruct_at(net, sc, x, y, z=0.0, n_t=N_T):
    t = np.arange(1, n_t + 1) * DT
    X = sc.encode(np.full(n_t, x), np.full(n_t, y), np.full(n_t, z), t)
    Xg = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    return sc.decode_fields(displacement(net, Xg, sc.rho).detach().cpu().numpy())


def pick_far(n, seed, min_dist=20.0, margin=8.0):
    with h5py.File(MAT, "r") as f:
        x = np.array(f["X_zero_coord_ply"]).ravel()[:N_PLY] * MM
        y = np.array(f["Y_zero_coord_ply"]).ravel()[:N_PLY] * MM
    far = (np.sqrt(x**2 + y**2) > min_dist) & (x > -149.5 + margin) & (x < 149.5 - margin) \
        & (y > -199.5 + margin) & (y < -0.5 - margin)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(np.where(far)[0], n, replace=False))


def fetch_gt(idx):
    with h5py.File(MAT, "r") as f:
        x = np.array(f["X_zero_coord_ply"]).ravel()[:N_PLY] * MM
        y = np.array(f["Y_zero_coord_ply"]).ravel()[:N_PLY] * MM
        u = np.array(f["Disp_x"][idx, :]) * MM
        v = np.array(f["Disp_y"][idx, :]) * MM
        w = np.array(f["Disp_z"][idx, :]) * MM
    return np.stack([x[idx], y[idx]], 1), np.stack([u, v, w], 2)


def main():
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7); args = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    net_on, sc = load_model("outputs/potential/spatial/phys_on/model.pt")
    net_off, _ = load_model("outputs/potential/spatial/phys_off/model.pt")
    idx = pick_far(args.n, args.seed)
    pts, gt = fetch_gt(idx)
    print(f"[far] points (mm):\n{np.round(pts, 1)}")

    out = "outputs/potential/spatial_far"; os.makedirs(out, exist_ok=True)
    viz.apply_style()
    fig, axes = plt.subplots(args.n, 3, figsize=(16, 2.5 * args.n), squeeze=False)
    t = np.arange(N_T); rows = []
    for i in range(args.n):
        po = reconstruct_at(net_on, sc, pts[i, 0], pts[i, 1])
        pf = reconstruct_at(net_off, sc, pts[i, 0], pts[i, 1])
        rows.append({"x": float(pts[i, 0]), "y": float(pts[i, 1]),
                     "data_only_relL2": relative_l2(pf, gt[i]), "pinn_relL2": relative_l2(po, gt[i])})
        for ci, comp in enumerate(["u", "v", "w"]):
            ax = axes[i][ci]
            ax.plot(t, gt[i, :, ci], color=viz.AC["ink"], lw=1.0, label="Ground truth")
            ax.plot(t, pf[:, ci], color=viz.AC["muted"], lw=1.0, ls=":", alpha=0.8, label="Data only")
            ax.plot(t, po[:, ci], color=viz.AC["red"], lw=1.2, ls="--", alpha=0.85, label="PINN")
            if ci == 0: ax.set_ylabel(f"(x={pts[i,0]:.0f},\ny={pts[i,1]:.0f})", fontsize=9)
            if i == 0: ax.set_title(f"component {comp}", fontsize=12, fontweight=600)
            ax.tick_params(labelsize=8)
            if i == 0 and ci == 2: ax.legend(fontsize=8, frameon=False, loc="upper right")
    for ci in range(3): axes[-1][ci].set_xlabel("Timestep index", fontsize=10)
    fig.suptitle("Helmholtz-potential PINN - far-field extrapolation (points OUTSIDE the neighborhood)",
                 fontsize=14, fontweight=600, y=1.005)
    fig.tight_layout(); viz._save(fig, os.path.join(out, "far_holdout_uvw")); plt.close(fig)

    train_xy = np.array([[0, 0]])
    viz.plot_plate_layout(np.vstack([pts, train_xy]), holdout_indices=list(range(args.n)),
                          save_stem=os.path.join(out, "far_layout"))
    summary = {"points": rows,
               "data_only_median": float(np.median([r["data_only_relL2"] for r in rows])),
               "pinn_median": float(np.median([r["pinn_relL2"] for r in rows]))}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("[far] " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
