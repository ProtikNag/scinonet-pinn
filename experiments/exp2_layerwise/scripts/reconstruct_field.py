"""Full-field spatial reconstruction at fixed timesteps for layer 1 (z=0).

For a trained layer-wise run, predict the entire top-ply (60,000-point, 300x200 mm)
displacement field at a few chosen timesteps and compare to the full-resolution
ground truth pulled directly from the source .mat (NOT the subsampled CSV, which
only holds the training points).

Output: one figure, rows = {u, v, w}, columns = [Actual, Predicted] per timestep,
with the plate aspect ratio preserved, a diverging (zero-centered) colormap shared
per field row, large fonts, and academic labels.

    python experiments/exp2_layerwise/scripts/reconstruct_field.py \
        --out  experiments/exp2_layerwise/outputs/sp20_t10_silu_F256_h256x256x256_a1_hpc \
        --mat  data/3D_Pristine.mat --timesteps 300 3000 5800
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import scinonet_pinn as P  # noqa: E402

# plate geometry (mm) and source layout — same constants as the dataset generator
MM = 1000.0
N_PLY_ROWS = 60000
X_MIN, X_MAX = -149.5, 149.5
Y_MIN, Y_MAX = -199.5, -0.5
NX, NY = 300, 200


def _to_grid(vals, x, y):
    """Scatter per-point values onto the regular 1 mm (NY, NX) plate grid."""
    g = np.full((NY, NX), np.nan)
    ix = np.rint(x - X_MIN).astype(int)
    iy = np.rint(y - Y_MIN).astype(int)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    g[iy[ok], ix[ok]] = vals[ok]
    return g


def predict_field(model, sc, x, y, z, t_phys, device, dtype, batch=16384):
    """Predicted (u, v, w) at every (x, y) for a single physical time t_phys."""
    n = len(x)
    out = np.empty((n, 3))
    tcol = np.full(n, t_phys)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        X = sc.encode(x[s:e], y[s:e], z[s:e], tcol[s:e])
        Xg = torch.tensor(X, dtype=dtype, device=device).requires_grad_(True)
        uvw = P.displacement(model, Xg, sc.beta_y, sc.beta_z)
        out[s:e] = sc.decode_fields(uvw.detach().cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run dir with model.pt + config.json")
    ap.add_argument("--mat", default="data/3D_Pristine.mat")
    ap.add_argument("--timesteps", type=int, nargs="+", default=[300, 3000, 5800])
    ap.add_argument("--which", choices=["best", "current"], default="best")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.out, "config.json")))
    blob = torch.load(os.path.join(args.out, "model.pt"), map_location="cpu", weights_only=False)

    # rebuild the core configuration so make_net reproduces the trained architecture
    P.set_dtype(cfg["dtype"])
    P.set_seed(cfg["seed"])
    P.ACTIVATION = cfg["activation"]
    P.NUM_FREQ = cfg["num_freq"]
    P.HIDDEN_SIZES = list(cfg["hidden"])
    P.BALANCE_ALPHA = cfg["balance_alpha"]
    P.BDRY_ENABLE = bool(cfg["bdry"])
    dtype = torch.float64 if cfg["dtype"] == "float64" else torch.float32

    sc = P.PotentialScalers.__new__(P.PotentialScalers)
    sc.__dict__.update(blob["scalers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = P.make_net(sc, cfg["seed"])
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    print(f"[field] device={device} dtype={cfg['dtype']} timesteps={args.timesteps}", flush=True)

    # layer-1 coordinates + full-resolution ground truth straight from the .mat
    Ts = list(args.timesteps)
    with h5py.File(args.mat, "r") as f:
        dt = float(np.array(f["dt"]).ravel()[0])
        x = np.array(f["X_zero_coord_ply"]).ravel()[:N_PLY_ROWS] * MM
        y = np.array(f["Y_zero_coord_ply"]).ravel()[:N_PLY_ROWS] * MM
        z = np.array(f["Z_zero_coord_ply"]).ravel()[:N_PLY_ROWS] * MM
        gt = {comp: np.empty((N_PLY_ROWS, len(Ts))) for comp in "uvw"}
        for j, T in enumerate(Ts):
            gt["u"][:, j] = f["Disp_x"][:N_PLY_ROWS, T] * MM
            gt["v"][:, j] = f["Disp_y"][:N_PLY_ROWS, T] * MM
            gt["w"][:, j] = f["Disp_z"][:N_PLY_ROWS, T] * MM
    print(f"[field] read ground truth: {N_PLY_ROWS} pts x {len(Ts)} steps", flush=True)

    # model prediction at each timestep (column k <-> physical time (k+1)*dt)
    pred = {comp: np.empty((N_PLY_ROWS, len(Ts))) for comp in "uvw"}
    for j, T in enumerate(Ts):
        p = predict_field(model, sc, x, y, z, (T + 1) * dt, device, dtype)
        pred["u"][:, j], pred["v"][:, j], pred["w"][:, j] = p[:, 0], p[:, 1], p[:, 2]
        print(f"[field]  predicted step {T}", flush=True)

    # ── figure: rows = u/v/w, cols = [Actual, Predicted] per timestep ──────────────
    fields = ["u", "v", "w"]
    field_lbl = {"u": "$u$  (x-disp.)", "v": "$v$  (y-disp.)", "w": "$w$  (z-disp., out-of-plane)"}
    ncol = 2 * len(Ts)
    fig, axes = plt.subplots(3, ncol, figsize=(4.6 * ncol, 11.0), squeeze=False)
    cmap = plt.get_cmap("RdBu_r")
    extent = [X_MIN - 0.5, X_MAX + 0.5, Y_MIN - 0.5, Y_MAX + 0.5]

    for r, comp in enumerate(fields):
        # symmetric, zero-centered scale shared across actual & predicted, all timesteps
        vmax = np.nanmax(np.abs([gt[comp], pred[comp]]))
        vmax = vmax if vmax > 0 else 1.0
        im = None
        for j, T in enumerate(Ts):
            for k, (src, tag) in enumerate(((gt, "Actual"), (pred, "Predicted"))):
                ax = axes[r][2 * j + k]
                g = _to_grid(src[comp][:, j], x, y)
                im = ax.imshow(g, origin="lower", extent=extent, cmap=cmap,
                               vmin=-vmax, vmax=vmax, aspect="equal", interpolation="nearest")
                if r == 0:
                    t_us = (T + 1) * dt * 1e6
                    ax.set_title(f"step {T}  ({t_us:.2f} µs)\n{tag}", fontsize=18, pad=10)
                if r == 2:
                    ax.set_xlabel("x [mm]", fontsize=16)
                if 2 * j + k == 0:
                    ax.set_ylabel(f"{field_lbl[comp]}\n\ny [mm]", fontsize=17)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=13)
        # one colorbar per field row
        cax = fig.add_axes([0.92, 0.655 - 0.305 * r, 0.013, 0.22])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=12)
        fmt = ScalarFormatter(useMathText=True); fmt.set_powerlimits((-2, 2))
        cb.ax.yaxis.set_major_formatter(fmt)
        cb.set_label("displacement [mm]", fontsize=13)

    name = (f"{cfg.get('pct_spatial', '?')}% layer-wise PINN  —  layer 1 (z = 0) full-field "
            f"reconstruction:  actual vs predicted")
    fig.suptitle(name, fontsize=22, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.07, wspace=0.08, hspace=0.20)

    stem = os.path.join(args.out, "field_reconstruction")
    fig.savefig(f"{stem}.png", dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {stem}.png / .svg", flush=True)

    # report per-field relative-L2 over the full layer at each timestep
    for comp in fields:
        for j, T in enumerate(Ts):
            a, p = gt[comp][:, j], pred[comp][:, j]
            rel = np.linalg.norm(p - a) / (np.linalg.norm(a) + 1e-30)
            print(f"[field] {comp} step {T:>5}: rel-L2 = {rel:.4f}", flush=True)


if __name__ == "__main__":
    main()
