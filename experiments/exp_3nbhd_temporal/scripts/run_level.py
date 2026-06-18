"""Train + evaluate the Helmholtz-potential PINN at one temporal-availability level.

For a given fraction ``--keep`` of timesteps retained per training point, the model
is trained on the 3-neighborhood dataset (135 training points, temporally
subsampled) with 15 neighborhood points held out spatially. It is then evaluated
in three prediction settings:

    seen          held-out timesteps at the 135 *training* points (temporal infill)
    neighborhood  the 15 spatially held-out points inside the neighborhoods
    far           random points far from all three neighborhoods (extrapolation)

Outputs (under outputs/keep_<pct>/): metrics.json, loss_curves, plate_layout,
temporal_seen_grid (seen timesteps drawn as pale blue vertical lines), and the
neighborhood / far reconstruction grids.

    python experiments/exp_3nbhd_temporal/scripts/run_level.py --keep 0.10 --epochs 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import scinonet.potential as potential  # noqa: E402
from scinonet.potential import (  # noqa: E402
    PotentialTrainConfig, build_potential_dataset, train_potential,
    reconstruct_xy, displacement, relative_l2,
)
from scinonet.seed import set_seed  # noqa: E402
from scinonet import viz  # noqa: E402
from run_potential import make_net  # noqa: E402

CSV = os.path.join(ROOT, "experiments/exp_3nbhd_temporal/data/"
                   "dataset_3nbhd_50pts_r15_3ply_fullsignal_6001steps.csv")
META = CSV.replace(".csv", "_meta.json")
MAT = os.path.join(ROOT, "data/3D_Pristine.mat")
N_PLY = 60000
MM = 1000.0
DT = 1e-8
N_T = 6001
CPU = torch.device("cpu")
SURFACE_Z = 0.0
CENTERS = np.array([[-49.5, -99.5], [38.0, -99.5], [125.0, -99.5]])


# ── test-point -> xy index mapping ──────────────────────────────────────────────
def map_test_indices(meta, xy_points):
    """Return {nbhd_name: [xy_index,...]} for the held-out test points."""
    lut = {tuple(np.round(p, 3)): i for i, p in enumerate(xy_points)}
    out = {}
    for nb in meta["neighborhoods"]:
        out[nb["name"]] = [lut[tuple(np.round(p, 3))] for p in nb["test_xy"]]
    return out


def train_xy_indices(meta, xy_points):
    lut = {tuple(np.round(p, 3)): i for i, p in enumerate(xy_points)}
    out = {}
    for nb in meta["neighborhoods"]:
        out[nb["name"]] = [lut[tuple(np.round(p, 3))] for p in nb["train_xy"]]
    return out


# ── far-field points (outside all neighborhoods) ────────────────────────────────
def pick_far(n, seed, centers, min_dist=28.0, margin=10.0):
    with h5py.File(MAT, "r") as f:
        x = np.array(f["X_zero_coord_ply"]).ravel()[:N_PLY] * MM
        y = np.array(f["Y_zero_coord_ply"]).ravel()[:N_PLY] * MM
    far = ((x > -149.5 + margin) & (x < 149.5 - margin)
           & (y > -199.5 + margin) & (y < -0.5 - margin))
    for cx, cy in centers:
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


# ── evaluation ──────────────────────────────────────────────────────────────────
def eval_seen_temporal(net, pdata, xy_indices):
    """relL2 on each train point's held-out timesteps (temporal infill)."""
    errs = []
    for i in xy_indices:
        rec = reconstruct_xy(net, pdata, i, SURFACE_Z, CPU)
        ti = rec["test_idx"]
        if len(ti) == 0:
            continue
        valid = ~np.isnan(rec["gt"][ti]).any(axis=1)
        if valid.sum() == 0:
            continue
        errs.append(relative_l2(rec["pred"][ti][valid], rec["gt"][ti][valid]))
    return np.array(errs)


def eval_spatial(net, pdata, xy_indices):
    """relL2 over the full signal of spatially held-out neighborhood points."""
    errs = []
    for i in xy_indices:
        rec = reconstruct_xy(net, pdata, i, SURFACE_Z, CPU)
        valid = ~np.isnan(rec["gt"]).any(axis=1)
        errs.append(relative_l2(rec["pred"][valid], rec["gt"][valid]))
    return np.array(errs)


def eval_far(net, sc, n_far, seed, centers):
    idx = pick_far(n_far, seed, centers)
    pts, gt = fetch_gt(idx)
    errs, recs = [], []
    for i in range(len(idx)):
        pred = reconstruct_at(net, sc, pts[i, 0], pts[i, 1])
        errs.append(relative_l2(pred, gt[i]))
        recs.append({"point": (pts[i, 0], pts[i, 1]), "pred": pred, "gt": gt[i]})
    return np.array(errs), recs, pts


# ── visualizations ──────────────────────────────────────────────────────────────
def plot_layout(train_xy, test_xy, far_xy, centers, names, save_stem,
                source_xy=(-49.5, -99.5), zoom_idx=2, zoom_half=6.5):
    """Full plate (left) + one zoomed-in neighborhood (right)."""
    import matplotlib.pyplot as plt
    viz.apply_style()
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(14, 6.2),
                                  gridspec_kw={"width_ratios": [1.45, 1]})

    # ── full plate ──
    ax.add_patch(plt.Rectangle((-149.5, -199.5), 299.0, 199.0, fill=False,
                               edgecolor=viz.AC["axis"], lw=1.2))
    ax.scatter(train_xy[:, 0], train_xy[:, 1], s=14, color=viz.AC["blue"],
               label=f"train ({len(train_xy)})", zorder=3)
    ax.scatter(test_xy[:, 0], test_xy[:, 1], s=42, color=viz.AC["red"], marker="D",
               edgecolor="white", lw=0.5, label=f"neighborhood test ({len(test_xy)})", zorder=5)
    ax.scatter(far_xy[:, 0], far_xy[:, 1], s=46, color=viz.AC["green"], marker="X",
               edgecolor="white", lw=0.5, label=f"far ({len(far_xy)})", zorder=4)
    for (cx, cy), nm in zip(centers, names):
        ax.annotate(nm, (cx, cy + 15), fontsize=9, color=viz.AC["muted"], ha="center")
    ax.scatter(*source_xy, s=130, marker="*", color=viz.AC["amber"],
               edgecolor=viz.AC["axis"], lw=0.5, label="source", zorder=6)
    # mark the zoom window
    zc = centers[zoom_idx]
    ax.add_patch(plt.Rectangle((zc[0] - zoom_half, zc[1] - zoom_half), 2 * zoom_half,
                               2 * zoom_half, fill=False, edgecolor=viz.AC["red"],
                               lw=1.0, ls="--", zorder=7))
    ax.set_xlabel("x [mm]", fontsize=12); ax.set_ylabel("y [mm]", fontsize=12)
    ax.set_title("3-neighborhood layout on the 300x200 mm plate", fontsize=13, fontweight=600)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=9, frameon=False, loc="lower left")

    # ── zoomed neighborhood ──
    def near(pts):
        if len(pts) == 0:
            return pts
        return pts[(np.abs(pts[:, 0] - zc[0]) <= zoom_half) & (np.abs(pts[:, 1] - zc[1]) <= zoom_half)]
    ztr, zte = near(train_xy), near(test_xy)
    # draw the plate edge if the window touches it
    for ex in (-149.5, 149.5):
        if zc[0] - zoom_half <= ex <= zc[0] + zoom_half:
            axz.axvline(ex, color=viz.AC["axis"], lw=1.4, label="plate edge")
    for ey in (-199.5, -0.5):
        if zc[1] - zoom_half <= ey <= zc[1] + zoom_half:
            axz.axhline(ey, color=viz.AC["axis"], lw=1.4)
    axz.scatter(ztr[:, 0], ztr[:, 1], s=80, color=viz.AC["blue"], edgecolor="white",
                lw=0.6, label="train", zorder=3)
    axz.scatter(zte[:, 0], zte[:, 1], s=150, color=viz.AC["red"], marker="D",
                edgecolor="white", lw=0.7, label="test (held out)", zorder=4)
    axz.set_xlim(zc[0] - zoom_half, zc[0] + zoom_half)
    axz.set_ylim(zc[1] - zoom_half, zc[1] + zoom_half)
    axz.set_xlabel("x [mm]", fontsize=12); axz.set_ylabel("y [mm]", fontsize=12)
    axz.set_title(f"zoom: {names[zoom_idx]} (1 mm grid)", fontsize=13, fontweight=600)
    axz.set_aspect("equal", adjustable="box")
    axz.legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout(); viz._save(fig, save_stem); plt.close(fig)


def temporal_seen_grid(recs, save_stem, keep_pct):
    """Seen points: GT vs PINN, with seen timesteps as pale blue vertical lines."""
    import matplotlib.pyplot as plt
    viz.apply_style()
    n = len(recs); comps = ["u", "v", "w"]
    fig, axes = plt.subplots(n, 3, figsize=(16, 2.5 * n), squeeze=False)
    t = np.arange(N_T)
    for i in range(n):
        px, py, _ = recs[i]["point"]
        seen = recs[i].get("train_idx", np.array([], int))
        for ci, comp in enumerate(comps):
            ax = axes[i][ci]
            ax.plot(t, recs[i]["gt"][:, ci], color=viz.AC["ink"], lw=1.0, label="Ground truth")
            ax.plot(t, recs[i]["pred"][:, ci], color=viz.AC["red"], lw=1.2, ls="--",
                    alpha=0.85, label="PINN")
            if len(seen):
                ax.vlines(seen, *ax.get_ylim(), color=viz.AC["blue"], lw=0.6, alpha=0.18,
                          zorder=0, label="Seen samples" if (i == 0 and ci == 2) else None)
            if ci == 0:
                ax.set_ylabel(f"(x={px:.1f},\ny={py:.1f})", fontsize=9)
            if i == 0:
                ax.set_title(f"component {comp}", fontsize=12, fontweight=600)
            ax.tick_params(labelsize=8)
            if i == 0 and ci == 2:
                ax.legend(fontsize=8, frameon=False, loc="upper right")
    for ci in range(3):
        axes[-1][ci].set_xlabel("Timestep index", fontsize=10)
    fig.suptitle(f"Temporal holdout at seen points - {keep_pct:.0f}% of timesteps retained "
                 "(pale blue lines = seen timesteps; u/v/w, z=0)", fontsize=14, fontweight=600, y=1.003)
    fig.tight_layout(); viz._save(fig, save_stem); plt.close(fig)


def recon_grid(recs, save_stem, title):
    import matplotlib.pyplot as plt
    viz.apply_style()
    n = len(recs); comps = ["u", "v", "w"]
    fig, axes = plt.subplots(n, 3, figsize=(16, 2.5 * n), squeeze=False)
    t = np.arange(N_T)
    for i in range(n):
        px, py = recs[i]["point"][0], recs[i]["point"][1]
        for ci, comp in enumerate(comps):
            ax = axes[i][ci]
            ax.plot(t, recs[i]["gt"][:, ci], color=viz.AC["ink"], lw=1.0, label="Ground truth")
            ax.plot(t, recs[i]["pred"][:, ci], color=viz.AC["red"], lw=1.2, ls="--",
                    alpha=0.85, label="PINN")
            if ci == 0:
                ax.set_ylabel(f"(x={px:.0f},\ny={py:.0f})", fontsize=9)
            if i == 0:
                ax.set_title(f"component {comp}", fontsize=12, fontweight=600)
            ax.tick_params(labelsize=8)
            if i == 0 and ci == 2:
                ax.legend(fontsize=8, frameon=False, loc="upper right")
    for ci in range(3):
        axes[-1][ci].set_xlabel("Timestep index", fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight=600, y=1.003)
    fig.tight_layout(); viz._save(fig, save_stem); plt.close(fig)


def plot_losses(hist, save_stem, data_only_epochs, include_gauge=False):
    import matplotlib.pyplot as plt
    viz.apply_style()
    comps = [("data", "Data loss"), ("wave", "Wave residual"),
             ("ic", "Initial condition")]
    if include_gauge:
        comps.append(("gauge", "Gauge div(psi)"))
    comps.append(("total", "Total loss"))
    fig, axes = plt.subplots(1, len(comps), figsize=(3.7 * len(comps), 4.0))
    for ax, (key, title) in zip(axes, comps):
        tr = np.maximum(hist[f"train_{key}"], 1e-30)
        ax.plot(range(1, len(tr) + 1), tr, color=viz.AC["blue"], lw=1.8, label="Train")
        va = np.maximum(hist.get(f"val_{key}", []), 1e-30)
        if len(va):
            ax.plot(range(1, len(va) + 1), va, color=viz.AC["amber"], lw=1.8, label="Validation")
        if 0 < data_only_epochs < len(tr):
            ax.axvline(data_only_epochs, color=viz.AC["muted"], lw=1.0, ls=":", label="physics on")
        ax.set_yscale("log"); ax.set_xlabel("Epoch", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight=600); ax.legend(fontsize=9, frameon=False)
    axes[0].set_ylabel("MSE (standardized)", fontsize=11)
    fig.tight_layout(); viz._save(fig, save_stem); plt.close(fig)


def run(keep, epochs, seed, n_far, n_demo, w_gauge=0.0, lz_mode="physical",
        csv=CSV, tag_suffix=""):
    set_seed(seed); torch.set_default_dtype(torch.float64)
    potential.LZ_MODE = lz_mode
    meta_path = csv.replace(".csv", "_meta.json")
    pct = keep * 100
    tag = f"keep_{pct:05.1f}".replace(".", "p")
    if w_gauge > 0:
        tag += "_gauge"
    if lz_mode != "physical":
        tag += f"_lz{lz_mode}"
    tag += tag_suffix
    out = os.path.join(ROOT, "experiments/exp_3nbhd_temporal/outputs", tag)
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(meta_path))
    centers = np.array([nb["center"] for nb in meta["neighborhoods"]], float)
    names = [nb["name"] for nb in meta["neighborhoods"]]

    base = build_potential_dataset(csv, subsample_keep=1.0, seed=seed)
    test_map = map_test_indices(meta, base.xy_points)
    train_map = train_xy_indices(meta, base.xy_points)
    held = sorted(i for v in test_map.values() for i in v)

    pdata = build_potential_dataset(csv, subsample_keep=keep, seed=seed, holdout_indices=held)
    print(f"[lvl {pct:.0f}%] held={len(held)} rho={pdata.scalers.rho:.2f} "
          f"L={pdata.scalers.L:.1f} train_rows={len(pdata.Xtr):,}", flush=True)

    net = make_net(pdata.scalers, seed)
    cfg = PotentialTrainConfig(epochs=epochs, batch_size=16384, n_colloc=2048, n_ic=1024,
                               data_only_epochs=12, balance_alpha=0.3, w_gauge=w_gauge,
                               early_stop_patience=epochs, log_every=max(5, epochs // 6))
    gtxt = f"gauge w={w_gauge}" if w_gauge > 0 else "gauge removed"
    print(f"\n##### LEVEL {pct:.0f}% - PHYSICS ON (wave + IC, {gtxt}) #####", flush=True)
    hist, _ = train_potential(net, pdata, cfg, CPU)
    plot_losses(hist, os.path.join(out, "loss_curves"), cfg.data_only_epochs,
                include_gauge=w_gauge > 0)

    # ── evaluation: seen / neighborhood / far ──
    all_train = sorted(i for v in train_map.values() for i in v)
    seen_err = eval_seen_temporal(net, pdata, all_train)
    nbhd_err = eval_spatial(net, pdata, held)
    far_err, far_recs, far_xy = eval_far(net, pdata.scalers, n_far, seed + 7, centers)

    # per-neighborhood breakdown
    per_nbhd = {}
    for nm in test_map:
        s = eval_seen_temporal(net, pdata, train_map[nm])
        n = eval_spatial(net, pdata, test_map[nm])
        per_nbhd[nm] = {"seen_median": float(np.median(s)), "neighborhood_median": float(np.median(n))}

    metrics = {
        "keep": keep, "pct": pct, "epochs": epochs, "seed": seed, "w_gauge": w_gauge,
        "lz_mode": lz_mode, "Lz": float(pdata.scalers.Lz),
        "rho": float(pdata.scalers.rho), "L": float(pdata.scalers.L),
        "train_rows": int(len(pdata.Xtr)),
        "seen": {"median": float(np.median(seen_err)), "mean": float(np.mean(seen_err)),
                 "n": int(len(seen_err))},
        "neighborhood": {"median": float(np.median(nbhd_err)), "mean": float(np.mean(nbhd_err)),
                         "n": int(len(nbhd_err)), "per_point": nbhd_err.tolist()},
        "far": {"median": float(np.median(far_err)), "mean": float(np.mean(far_err)),
                "n": int(len(far_err)), "per_point": far_err.tolist()},
        "per_neighborhood": per_nbhd,
    }
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    torch.save({"state_dict": net.state_dict(), "scalers": pdata.scalers,
                "history": hist, "metrics": metrics}, os.path.join(out, "model.pt"))

    # ── figures ──
    rng = np.random.RandomState(seed)
    train_xy = base.xy_points[all_train]
    test_xy = base.xy_points[held]
    src = tuple(centers[0]) if "near_source" in names else (-49.5, -99.5)
    zoom_idx = names.index("near_boundary") if "near_boundary" in names else len(names) - 1
    plot_layout(train_xy, test_xy, far_xy, centers, names,
                os.path.join(out, "plate_layout"), source_xy=src, zoom_idx=zoom_idx)

    # one demo seen point per neighborhood
    demo_seen = [train_map[nm][0] for nm in test_map]
    seen_recs = [reconstruct_xy(net, pdata, i, SURFACE_Z, CPU) for i in demo_seen]
    temporal_seen_grid(seen_recs, os.path.join(out, "temporal_seen_grid"), pct)

    # all 15 neighborhood test points (or first n_demo per request)
    nbhd_recs = [reconstruct_xy(net, pdata, i, SURFACE_Z, CPU) for i in held]
    recon_grid(nbhd_recs, os.path.join(out, "neighborhood_holdout_grid"),
               f"Neighborhood spatial holdout - 15 unseen points, {pct:.0f}% timesteps (u/v/w, z=0)")
    recon_grid(far_recs[:n_demo], os.path.join(out, "far_holdout_grid"),
               f"Far-field extrapolation - {pct:.0f}% timesteps (u/v/w, z=0)")

    print(f"[lvl {pct:.0f}%] seen={metrics['seen']['median']:.3f} "
          f"nbhd={metrics['neighborhood']['median']:.3f} far={metrics['far']['median']:.3f}", flush=True)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=float, required=True, help="fraction of timesteps retained")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-far", type=int, default=10)
    ap.add_argument("--n-demo", type=int, default=6)
    ap.add_argument("--w-gauge", type=float, default=0.0,
                    help="Coulomb-gauge loss weight (>0 re-enables the gauge term)")
    ap.add_argument("--lz-mode", choices=["physical", "inplane"], default="physical",
                    help="physical -> Lz=1mm (rho=L); inplane -> Lz=L (rho=1, previous)")
    ap.add_argument("--csv", default=CSV, help="dataset CSV (meta JSON inferred)")
    ap.add_argument("--tag-suffix", default="", help="appended to the output folder name")
    a = ap.parse_args()
    run(a.keep, a.epochs, a.seed, a.n_far, a.n_demo, a.w_gauge, a.lz_mode, a.csv, a.tag_suffix)
