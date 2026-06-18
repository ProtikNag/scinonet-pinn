"""Small-data spatial holdout for the Helmholtz-potential PINN (option B).

Two dense neighborhoods (50 points each, 3 plies). Five interior points are held
out from EACH neighborhood (10 unseen points total) and the model never sees
them during training. Two prediction settings are written to separate folders:

    outputs/potential/spatial_2nbhd/neighborhood/  held-out points INSIDE the
                                                   trained neighborhoods
    outputs/potential/spatial_2nbhd/far/           random points FAR from both
                                                   neighborhoods (extrapolation)

    python scripts/run_potential_2nbhd.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scinonet.potential import (  # noqa: E402
    PotentialTrainConfig, build_potential_dataset, train_potential,
    evaluate_holdout, reconstruct_xy, displacement, relative_l2,
    CP_MM_PER_S, CS_MM_PER_S,
)
from scinonet.seed import set_seed  # noqa: E402
from scinonet import viz  # noqa: E402
from run_potential import make_net, plot_losses, holdout_grid  # noqa: E402

CSV = "data/dataset_2nbhd_50pts_3ply_fullsignal_6001steps.csv"
CENTERS = np.array([[-70.0, -60.0], [70.0, -140.0]])   # the two neighborhood centers
MAT = "data/3D_Pristine.mat"
N_PLY = 60000
MM = 1000.0
DT = 1e-8
N_T = 6001
CPU = torch.device("cpu")
SURFACE_Z = 0.0
OUT = os.path.join("outputs", "potential", "spatial_2nbhd")


def interior_indices(points):
    """xy indices that have all four 1 mm neighbors present (safe to hold out)."""
    xy = np.round(points, 3)
    present = {tuple(p) for p in xy}
    return [i for i, (x, y) in enumerate(xy)
            if all(n in present for n in [(round(x + 1, 3), y), (round(x - 1, 3), y),
                                          (x, round(y + 1, 3)), (x, round(y - 1, 3))])]


def pick_holdout(points, per_nbhd=5, seed=42):
    """Five interior held-out points from EACH neighborhood (nearest-center split)."""
    inter = np.array(interior_indices(points))
    owner = np.argmin(((points[inter, None, :] - CENTERS[None]) ** 2).sum(-1), axis=1)
    rng = np.random.RandomState(seed)
    held = []
    for c in range(len(CENTERS)):
        pool = inter[owner == c]
        rng.shuffle(pool)
        held.extend(pool[:per_nbhd].tolist())
    return sorted(int(i) for i in held)


# ── Far-field extrapolation (points outside BOTH neighborhoods) ────────────────
def pick_far(n, seed, min_dist=25.0, margin=10.0):
    with h5py.File(MAT, "r") as f:
        x = np.array(f["X_zero_coord_ply"]).ravel()[:N_PLY] * MM
        y = np.array(f["Y_zero_coord_ply"]).ravel()[:N_PLY] * MM
    far = (x > -149.5 + margin) & (x < 149.5 - margin) & (y > -199.5 + margin) & (y < -0.5 - margin)
    for cx, cy in CENTERS:
        far &= np.sqrt((x - cx) ** 2 + (y - cy) ** 2) > min_dist
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


def reconstruct_at(net, sc, x, y, z=0.0):
    t = np.arange(1, N_T + 1) * DT
    X = sc.encode(np.full(N_T, x), np.full(N_T, y), np.full(N_T, z), t)
    Xg = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    return sc.decode_fields(displacement(net, Xg, sc.rho).detach().cpu().numpy())


def far_grid(net_off, net_on, sc, n=6, seed=7):
    import matplotlib.pyplot as plt
    out = os.path.join(OUT, "far"); os.makedirs(out, exist_ok=True)
    idx = pick_far(n, seed)
    pts, gt = fetch_gt(idx)
    viz.apply_style()
    fig, axes = plt.subplots(n, 3, figsize=(16, 2.5 * n), squeeze=False)
    t = np.arange(N_T); rows = []
    for i in range(n):
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
    fig.suptitle("Helmholtz-potential PINN - far-field extrapolation (points OUTSIDE both neighborhoods)",
                 fontsize=14, fontweight=600, y=1.005)
    fig.tight_layout(); viz._save(fig, os.path.join(out, "far_holdout_uvw")); plt.close(fig)

    viz.plot_plate_layout(np.vstack([pts, CENTERS]),
                          holdout_indices=list(range(n)),
                          save_stem=os.path.join(out, "far_layout"))
    summary = {"points": rows,
               "data_only_median": float(np.median([r["data_only_relL2"] for r in rows])),
               "pinn_median": float(np.median([r["pinn_relL2"] for r in rows]))}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("[far] " + json.dumps(summary, indent=2))


def main(seed=42):
    set_seed(seed); torch.set_default_dtype(torch.float64)
    nbhd = os.path.join(OUT, "neighborhood"); os.makedirs(nbhd, exist_ok=True)

    base = build_potential_dataset(CSV, subsample_keep=1.0, seed=seed)
    held = pick_holdout(base.xy_points, per_nbhd=5, seed=seed)
    pdata = build_potential_dataset(CSV, subsample_keep=0.15, seed=seed, holdout_indices=held)
    print(f"[2nbhd] held={held} rho={pdata.scalers.rho:.2f} train_rows={len(pdata.Xtr)}", flush=True)

    viz.plot_plate_layout(pdata.xy_points, holdout_indices=held,
                          save_stem=os.path.join(nbhd, "plate_layout"))

    common = dict(epochs=55, batch_size=16384, n_colloc=2048, n_ic=1024,
                  data_only_epochs=12, early_stop_patience=55, log_every=10)
    net0 = make_net(pdata.scalers, seed)
    net_off, net_on = copy.deepcopy(net0), copy.deepcopy(net0)

    print("\n##### 2NBHD spatial - PHYSICS OFF #####")
    cfg_off = PotentialTrainConfig(data_only_epochs=90,
                                   **{k: v for k, v in common.items() if k != "data_only_epochs"})
    h_off, _ = train_potential(net_off, pdata, cfg_off, CPU)
    print("\n##### 2NBHD spatial - PHYSICS ON (wave + gauge + IC) #####")
    h_on, _ = train_potential(net_on, pdata, PotentialTrainConfig(balance_alpha=0.3, **common), CPU)

    m_off = evaluate_holdout(net_off, pdata, CPU, held, z=SURFACE_Z)
    m_on = evaluate_holdout(net_on, pdata, CPU, held, z=SURFACE_Z)
    for tag, net, h, m in [("phys_off", net_off, h_off, m_off), ("phys_on", net_on, h_on, m_on)]:
        d = os.path.join(nbhd, tag); os.makedirs(d, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "scalers": pdata.scalers,
                    "history": h, "metrics": m}, os.path.join(d, "model.pt"))
        json.dump(h, open(os.path.join(d, "history.json"), "w"))
        json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=2)
    plot_losses(h_off, os.path.join(nbhd, "phys_off", "loss_curves"), 0)
    plot_losses(h_on, os.path.join(nbhd, "phys_on", "loss_curves"), common["data_only_epochs"])

    recs_off = [reconstruct_xy(net_off, pdata, i, SURFACE_Z, CPU) for i in held]
    recs_on = [reconstruct_xy(net_on, pdata, i, SURFACE_Z, CPU) for i in held]
    holdout_grid(recs_off, recs_on, os.path.join(nbhd, "all_holdout_uvw"),
                 title="Helmholtz-potential PINN - 2 neighborhoods, held-out points INSIDE the "
                       "neighborhoods (u/v/w, z=0 surface)")

    summary = {"held": held, "phys_off_median": m_off["median"], "phys_on_median": m_on["median"],
               "phys_off_mean": m_off["mean"], "phys_on_mean": m_on["mean"]}
    json.dump(summary, open(os.path.join(nbhd, "summary.json"), "w"), indent=2)
    print(f"[2nbhd:neighborhood] " + json.dumps(summary, indent=2))

    # Far-field extrapolation from the same trained models.
    far_grid(net_off, net_on, pdata.scalers, n=6, seed=7)


if __name__ == "__main__":
    main()
