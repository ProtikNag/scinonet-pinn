"""Gap-interpolation spatial holdout for the Helmholtz-potential PINN (option B).

Two CLOSE neighborhoods (50 points each, 3 plies) flank a held-out test cluster
sitting in the gap between them. The model trains only on the two neighborhoods
and is evaluated on the gap points -- they are not neighborhood points, but they
are flanked by training data on both sides (interpolation, not far extrapolation).

    outputs/potential/spatial_2nbhd_gap/   plate layout, loss curves, models,
                                           and the gap held-out u/v/w grid

    python scripts/run_potential_gap.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scinonet.potential import (  # noqa: E402
    PotentialTrainConfig, build_potential_dataset, train_potential,
    evaluate_holdout, reconstruct_xy,
)
from scinonet.seed import set_seed  # noqa: E402
from scinonet import viz  # noqa: E402
from run_potential import make_net, plot_losses, holdout_grid  # noqa: E402

CSV = "data/dataset_2nbhd_gap_50pts_3ply_fullsignal_6001steps.csv"
TRAIN_CENTERS = np.array([[-18.0, -100.0], [18.0, -100.0]])   # the two neighborhoods
GAP_CENTER = np.array([0.0, -100.0])                          # held-out test region
CPU = torch.device("cpu")
SURFACE_Z = 0.0
OUT = os.path.join("outputs", "potential", "spatial_2nbhd_gap")


def split_gap(points):
    """Indices whose nearest center is the gap center -> the held-out test set."""
    centers = np.vstack([TRAIN_CENTERS, GAP_CENTER[None]])
    owner = np.argmin(((points[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    gap_id = centers.shape[0] - 1
    return sorted(int(i) for i in np.where(owner == gap_id)[0])


def main(seed=42):
    set_seed(seed); torch.set_default_dtype(torch.float64)
    os.makedirs(OUT, exist_ok=True)

    base = build_potential_dataset(CSV, subsample_keep=1.0, seed=seed)
    held = split_gap(base.xy_points)
    pdata = build_potential_dataset(CSV, subsample_keep=0.20, seed=seed, holdout_indices=held)
    gp = base.xy_points[held]
    print(f"[gap] n_held={len(held)} rho={pdata.scalers.rho:.2f} train_rows={len(pdata.Xtr)} "
          f"gap_x=[{gp[:,0].min():.1f},{gp[:,0].max():.1f}] "
          f"gap_y=[{gp[:,1].min():.1f},{gp[:,1].max():.1f}]", flush=True)

    viz.plot_plate_layout(pdata.xy_points, holdout_indices=held,
                          save_stem=os.path.join(OUT, "plate_layout"))

    common = dict(epochs=70, batch_size=16384, n_colloc=2048, n_ic=1024,
                  data_only_epochs=12, early_stop_patience=70, log_every=10)
    net0 = make_net(pdata.scalers, seed)
    net_off, net_on = copy.deepcopy(net0), copy.deepcopy(net0)

    print("\n##### GAP spatial - PHYSICS OFF #####", flush=True)
    cfg_off = PotentialTrainConfig(data_only_epochs=90,
                                   **{k: v for k, v in common.items() if k != "data_only_epochs"})
    h_off, _ = train_potential(net_off, pdata, cfg_off, CPU)
    print("\n##### GAP spatial - PHYSICS ON (wave + gauge + IC) #####", flush=True)
    h_on, _ = train_potential(net_on, pdata, PotentialTrainConfig(balance_alpha=0.3, **common), CPU)

    m_off = evaluate_holdout(net_off, pdata, CPU, held, z=SURFACE_Z)
    m_on = evaluate_holdout(net_on, pdata, CPU, held, z=SURFACE_Z)
    for tag, net, h, m in [("phys_off", net_off, h_off, m_off), ("phys_on", net_on, h_on, m_on)]:
        d = os.path.join(OUT, tag); os.makedirs(d, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "scalers": pdata.scalers,
                    "history": h, "metrics": m}, os.path.join(d, "model.pt"))
        json.dump(h, open(os.path.join(d, "history.json"), "w"))
        json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=2)
    plot_losses(h_off, os.path.join(OUT, "phys_off", "loss_curves"), 0)
    plot_losses(h_on, os.path.join(OUT, "phys_on", "loss_curves"), common["data_only_epochs"])

    # plot a representative spread of the gap test points (sorted by x), evaluate all
    demo = [held[i] for i in np.linspace(0, len(held) - 1, 10).round().astype(int)]
    recs_off = [reconstruct_xy(net_off, pdata, i, SURFACE_Z, CPU) for i in demo]
    recs_on = [reconstruct_xy(net_on, pdata, i, SURFACE_Z, CPU) for i in demo]
    holdout_grid(recs_off, recs_on, os.path.join(OUT, "gap_holdout_uvw"),
                 title="Helmholtz-potential PINN - held-out points BETWEEN two close "
                       "neighborhoods (gap interpolation, u/v/w, z=0 surface)")

    summary = {"train_centers": TRAIN_CENTERS.tolist(), "gap_center": GAP_CENTER.tolist(),
               "n_held": len(held), "phys_off_median": m_off["median"],
               "phys_on_median": m_on["median"], "phys_off_mean": m_off["mean"],
               "phys_on_mean": m_on["mean"]}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("[gap] " + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
