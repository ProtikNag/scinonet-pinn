"""Full run of the 3D Helmholtz-potential PINN (option B): physics off vs on.

Spatial holdout of interior (x,y) points on the 3-ply neighborhood data. Saves
checkpoints, loss histories (data / wave / gauge / IC / total), the plate layout,
the physics-off-vs-on comparison, and the all-holdout u/v/w grid.

    python scripts/run_potential.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scinonet.features import SpecializedFourierFeatures  # noqa: E402
from scinonet.potential import (  # noqa: E402
    PotentialNet, PotentialTrainConfig, build_potential_dataset, train_potential,
    evaluate_holdout, reconstruct_xy, relative_l2, CP_MM_PER_S, CS_MM_PER_S,
)
from scinonet.seed import set_seed  # noqa: E402
from scinonet import viz  # noqa: E402

CSV = "data/dataset_6domain_100pts_3ply_fullsignal_6001steps.csv"
CPU = torch.device("cpu")
K_MAX_SPATIAL = 0.13          # cyc/mm, in-plane (kappa<=0.8 rad/mm)
SPATIAL_SCALE = 2.0
K_MAX_Z = 0.5                  # cyc/mm, through-thickness
F_MAX_HZ = 300e3
NUM_FREQ = 160
SURFACE_Z = 0.0


def interior_xy(points, k, seed):
    xy = np.round(points, 3)
    present = {tuple(p) for p in xy}
    inter = [i for i, (x, y) in enumerate(xy)
             if all(n in present for n in [(round(x + 1, 3), y), (round(x - 1, 3), y),
                                           (x, round(y + 1, 3)), (x, round(y - 1, 3))])]
    rng = np.random.RandomState(seed)
    rng.shuffle(inter)
    return sorted(inter[:k])


def make_features(sc, seed):
    # z is scaled by the in-plane L (rho=1), so it shares the in-plane band.
    L, st = sc.L, sc.s_t
    ks = K_MAX_SPATIAL * SPATIAL_SCALE
    lo = torch.tensor([-ks * L, -ks * L, -ks * L, 0.0], dtype=torch.float64)
    hi = torch.tensor([ks * L, ks * L, ks * L, F_MAX_HZ * st], dtype=torch.float64)
    return SpecializedFourierFeatures(lo, hi, NUM_FREQ, seed=seed)


def make_net(sc, seed):
    net = PotentialNet(make_features(sc, seed), [256, 256, 256],
                       chatp_init=sc.chat(CP_MM_PER_S), chats_init=sc.chat(CS_MM_PER_S))
    net.log_chatp.requires_grad_(False)   # fixed bulk speeds keep cp/cs = sqrt(3)
    net.log_chats.requires_grad_(False)
    return net


def plot_losses(hist, save_stem, data_only_epochs):
    import matplotlib.pyplot as plt
    viz.apply_style()
    comps = [("data", "Data loss"), ("wave", "Wave residual"),
             ("gauge", "Gauge div(psi)"), ("ic", "Initial condition"),
             ("total", "Total loss")]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.0))
    for ax, (key, title) in zip(axes, comps):
        tr = np.maximum(hist[f"train_{key}"], 1e-30)
        ax.plot(range(1, len(tr) + 1), tr, color=viz.AC["blue"], lw=1.8, label="Train")
        if f"val_{key}" in hist:
            va = np.maximum(hist[f"val_{key}"], 1e-30)
            ax.plot(range(1, len(va) + 1), va, color=viz.AC["amber"], lw=1.8, label="Validation")
        if 0 < data_only_epochs < len(tr):
            ax.axvline(data_only_epochs, color=viz.AC["muted"], lw=1.0, ls=":", label="physics on")
        ax.set_yscale("log"); ax.set_xlabel("Epoch", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight=600); ax.legend(fontsize=9, frameon=False)
    axes[0].set_ylabel("MSE (standardized)", fontsize=11)
    fig.tight_layout(); viz._save(fig, save_stem); plt.close(fig)


def holdout_grid(recs_off, recs_on, save_stem,
                 title="Helmholtz-potential PINN - spatial holdout, all unseen points (u/v/w, z=0 surface)",
                 mark_seen=False):
    import matplotlib.pyplot as plt
    viz.apply_style()
    n = len(recs_off); comps = ["u", "v", "w"]
    fig, axes = plt.subplots(n, 3, figsize=(16, 2.5 * n), squeeze=False)
    t = np.arange(len(recs_on[0]["pred"]))
    for i in range(n):
        px, py, _ = recs_on[i]["point"]
        seen = recs_on[i].get("train_idx", np.array([], int))
        for ci, comp in enumerate(comps):
            ax = axes[i][ci]
            gt = recs_on[i]["gt"][:, ci]
            ax.plot(t, gt, color=viz.AC["ink"], lw=1.0, label="Ground truth")
            ax.plot(t, recs_off[i]["pred"][:, ci], color=viz.AC["muted"], lw=1.0, ls=":", alpha=0.8, label="Data only")
            ax.plot(t, recs_on[i]["pred"][:, ci], color=viz.AC["red"], lw=1.2, ls="--", alpha=0.85, label="PINN")
            if mark_seen and len(seen):
                # pale blue vertical strips at seen timesteps (not disruptive dots)
                ax.vlines(seen, *ax.get_ylim(), color=viz.AC["blue"], lw=0.6,
                          alpha=0.18, zorder=0,
                          label="Seen samples" if (i == 0 and ci == 2) else None)
            if ci == 0: ax.set_ylabel(f"(x={px:.1f},\ny={py:.1f})", fontsize=9)
            if i == 0: ax.set_title(f"component {comp}", fontsize=12, fontweight=600)
            ax.tick_params(labelsize=8)
            if i == 0 and ci == 2: ax.legend(fontsize=8, frameon=False, loc="upper right")
    for ci in range(3): axes[-1][ci].set_xlabel("Timestep index", fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight=600, y=1.005)
    fig.tight_layout(); viz._save(fig, save_stem); plt.close(fig)


def run_mode(mode, seed=42):
    import copy
    set_seed(seed); torch.set_default_dtype(torch.float64)
    out = os.path.join("outputs", "potential", mode); os.makedirs(out, exist_ok=True)

    base = build_potential_dataset(CSV, subsample_keep=1.0, seed=seed)
    n_xy = base.xy_points.shape[0]
    rng = np.random.RandomState(seed)
    if mode == "spatial":
        # random held-out points (domain sampling is ~10 mm apart, no 1 mm neighbors)
        held = sorted(rng.choice(n_xy, 24, replace=False).tolist())
        pdata = build_potential_dataset(CSV, subsample_keep=0.10, seed=seed, holdout_indices=held)
        eval_xy = held
        demo_xy = held[:8]
        epochs = 70
    else:  # temporal: 5% of timesteps, no spatial holdout
        held = []
        pdata = build_potential_dataset(CSV, subsample_keep=0.05, seed=seed)
        eval_xy = list(range(n_xy))
        demo_xy = sorted(rng.choice(n_xy, 8, replace=False).tolist())
        epochs = 80
    print(f"[pot:{mode}] n_held={len(held)} rho={pdata.scalers.rho:.2f} train_rows={len(pdata.Xtr)}")

    viz.plot_plate_layout(pdata.xy_points, holdout_indices=held,
                          save_stem=os.path.join(out, "plate_layout"))

    common = dict(epochs=epochs, batch_size=16384, n_colloc=2048, n_ic=1024,
                  data_only_epochs=12, early_stop_patience=epochs, log_every=15)
    net0 = make_net(pdata.scalers, seed)
    net_off = copy.deepcopy(net0); net_on = copy.deepcopy(net0)

    print(f"\n##### POTENTIAL {mode} — PHYSICS OFF #####")
    cfg_off = PotentialTrainConfig(data_only_epochs=90, **{k: v for k, v in common.items() if k != "data_only_epochs"})
    h_off, _ = train_potential(net_off, pdata, cfg_off, CPU)
    print(f"\n##### POTENTIAL {mode} — PHYSICS ON (wave + gauge + IC) #####")
    h_on, _ = train_potential(net_on, pdata, PotentialTrainConfig(balance_alpha=0.3, **common), CPU)

    m_off = evaluate_holdout(net_off, pdata, CPU, eval_xy, z=SURFACE_Z)
    m_on = evaluate_holdout(net_on, pdata, CPU, eval_xy, z=SURFACE_Z)
    for tag, net, h, m in [("phys_off", net_off, h_off, m_off), ("phys_on", net_on, h_on, m_on)]:
        d = os.path.join(out, tag); os.makedirs(d, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "scalers": pdata.scalers,
                    "history": h, "metrics": m}, os.path.join(d, "model.pt"))
        json.dump(h, open(os.path.join(d, "history.json"), "w"))
        json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=2)
    plot_losses(h_off, os.path.join(out, "phys_off", "loss_curves"), 0)
    plot_losses(h_on, os.path.join(out, "phys_on", "loss_curves"), common["data_only_epochs"])

    recs_off = [reconstruct_xy(net_off, pdata, i, SURFACE_Z, CPU) for i in demo_xy]
    recs_on = [reconstruct_xy(net_on, pdata, i, SURFACE_Z, CPU) for i in demo_xy]
    grid_title = ("Helmholtz-potential PINN - temporal holdout (5% samples), seen points "
                  "(blue dots = seen timesteps; u/v/w, z=0)"
                  if mode == "temporal"
                  else "Helmholtz-potential PINN - spatial holdout, unseen points (u/v/w, z=0 surface)")
    holdout_grid(recs_off, recs_on, os.path.join(out, "all_holdout_uvw"),
                 title=grid_title, mark_seen=(mode == "temporal"))

    summary = {"mode": mode, "held": held, "phys_off_median": m_off["median"],
               "phys_on_median": m_on["median"], "phys_off_mean": m_off["mean"],
               "phys_on_mean": m_on["mean"]}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(f"[pot:{mode}] " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["spatial", "temporal"], default="spatial")
    a = p.parse_args()
    run_mode(a.mode)
