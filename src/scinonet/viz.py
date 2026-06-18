"""Publication-quality plotting in the academic palette.

Every figure is written as both PNG and SVG. White background, Tufte spine,
faint horizontal grid, Inter/Source Serif typography per the project style.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

AC_SERIES = ["#2563EB", "#D97706", "#059669", "#DC2626",
             "#7C3AED", "#0891B2", "#BE185D", "#92400E"]
AC = {
    "blue": "#2563EB", "amber": "#D97706", "green": "#059669", "red": "#DC2626",
    "axis": "#495057", "grid": "#E9ECEF", "text": "#212529", "muted": "#6C757D",
    "ink": "#212529",
}


def apply_style() -> None:
    """Inject the academic matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": AC["grid"],
        "grid.linewidth": 0.6,
        "axes.edgecolor": AC["axis"],
        "axes.labelcolor": AC["text"],
        "axes.titlecolor": AC["text"],
        "xtick.color": AC["muted"],
        "ytick.color": AC["muted"],
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _save(fig, save_stem: str | None) -> None:
    if save_stem is None:
        return
    os.makedirs(os.path.dirname(save_stem) or ".", exist_ok=True)
    fig.savefig(f"{save_stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{save_stem}.svg", bbox_inches="tight", facecolor="white")
    print(f"saved: {save_stem}.png / .svg")


def plot_loss_curves(history, save_stem: str | None = None) -> None:
    """Train/val loss curves on a log-y axis."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, len(history.train_loss) + 1)
    ax.plot(epochs, history.train_loss, color=AC["blue"], lw=1.8, label="Train (kept windows)")
    ax.plot(epochs, history.val_loss, color=AC["amber"], lw=1.8, label="Val (held-out gaps)")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("MSE (standardized)", fontsize=13)
    ax.set_title("Training dynamics", fontsize=14, fontweight=600)
    ax.legend(fontsize=11, frameon=False)
    _save(fig, save_stem)
    plt.close(fig)


def plot_reconstruction(
    recon: dict,
    component: str = "w",
    save_stem: str | None = None,
    shade_train: bool = True,
) -> None:
    """Overlay predicted vs ground-truth full signal for one point/component.

    Args:
        recon: Output of :func:`scinonet.evaluate.reconstruct_point`.
        component: ``"u"``, ``"v"`` or ``"w"``.
        save_stem: Path stem (without extension) for PNG/SVG output.
        shade_train: Shade the kept training windows.
    """
    apply_style()
    comp_idx = {"u": 0, "v": 1, "w": 2}[component]
    pred = recon["pred"][:, comp_idx]
    gt = recon["gt"][:, comp_idx]
    n_t = len(pred)
    t = np.arange(n_t)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, gt, color=AC["ink"], lw=1.3, label="Ground truth", zorder=3)
    ax.plot(t, pred, color=AC["red"], lw=1.5, ls="--", alpha=0.85,
            label="Predicted (full)", zorder=4)

    if shade_train:
        train_idx = recon["train_idx"]
        # shade contiguous kept runs
        if len(train_idx):
            splits = np.where(np.diff(train_idx) > 1)[0]
            starts = np.concatenate([[train_idx[0]], train_idx[splits + 1]])
            ends = np.concatenate([train_idx[splits], [train_idx[-1]]])
            for s, e in zip(starts, ends):
                ax.axvspan(s, e, color=AC["blue"], alpha=0.06, zorder=0)

    px, py = recon["point"][0], recon["point"][1]
    ax.set_title(f"(x={px:.1f}, y={py:.1f}) — component {component}   "
                 f"[shaded = training windows]",
                 fontsize=13, fontweight=600)
    ax.set_xlabel("Timestep index", fontsize=13)
    ax.set_ylabel(component, fontsize=13)
    ax.legend(fontsize=11, ncol=2, loc="upper center", frameon=False)
    _save(fig, save_stem)
    plt.close(fig)


def plot_point_layout(points: np.ndarray, save_stem: str | None = None) -> None:
    """Scatter of the spatial points over the plate footprint."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[:, 0], points[:, 1], s=40, color=AC["blue"],
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlabel("x [mm]", fontsize=13)
    ax.set_ylabel("y [mm]", fontsize=13)
    ax.set_title(f"{len(points)} observation points", fontsize=14, fontweight=600)
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig, save_stem)
    plt.close(fig)


def plot_pinn_losses(history: dict, save_stem: str | None = None,
                     data_only_epochs: int = 0) -> None:
    """Loss-component curves: data, PDE residual, IC, BC, total — train vs val.

    Components that were not used (all-zero) are skipped automatically.

    Args:
        history: Dict with ``train_<k>`` / ``val_<k>`` lists for k in
            {data, pde, ic, bc, total} (older runs may only have phys/total).
        save_stem: Output path stem (PNG + SVG).
        data_only_epochs: If > 0, mark where the physics term switches on.
    """
    apply_style()
    candidates = [("data", "Data loss"), ("pde", "PDE residual"),
                  ("ic", "Initial condition"), ("bc", "Boundary condition"),
                  ("total", "Total loss")]
    # keep components that exist and are not identically zero
    panels = []
    for key, title in candidates:
        tr = history.get(f"train_{key}")
        if tr is None:
            if key == "pde":  # backward-compat: older runs stored "phys"
                tr, key = history.get("train_phys"), "phys"
                title = "Physics residual"
            else:
                continue
        if tr is None or np.allclose(np.nan_to_num(tr), 0.0):
            continue
        panels.append((key, title))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.3))
    if n == 1:
        axes = [axes]
    for ax, (key, title) in zip(axes, panels):
        tr = history.get(f"train_{key}", [])
        va = history.get(f"val_{key}", [])
        ax.plot(range(1, len(tr) + 1), np.maximum(tr, 1e-30), color=AC["blue"],
                lw=1.8, label="Train")
        if va:
            ax.plot(range(1, len(va) + 1), np.maximum(va, 1e-30), color=AC["amber"],
                    lw=1.8, label="Validation")
        if data_only_epochs > 0 and data_only_epochs < len(tr):
            ax.axvline(data_only_epochs, color=AC["muted"], lw=1.0, ls=":",
                       label="physics on")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight=600)
        ax.legend(fontsize=9, frameon=False)
    axes[0].set_ylabel("MSE (standardized)", fontsize=12)
    fig.tight_layout()
    _save(fig, save_stem)
    plt.close(fig)


def plot_plate_layout(selected_points: np.ndarray,
                      holdout_indices: list[int] | None = None,
                      plate_xlim: tuple[float, float] = (-149.5, 149.5),
                      plate_ylim: tuple[float, float] = (-199.5, -0.5),
                      save_stem: str | None = None) -> None:
    """Map of the selected observation points on the full plate footprint.

    Train points are blue; spatially held-out points are red. A zoomed inset
    shows the dense neighbor cluster.

    Args:
        selected_points: ``[P, >=2]`` physical coordinates of the chosen points.
        holdout_indices: Indices (into ``selected_points``) held out spatially.
        plate_xlim, plate_ylim: Full plate extent in mm.
        save_stem: Output path stem (PNG + SVG).
    """
    apply_style()
    holdout = set(holdout_indices or [])
    is_held = np.array([i in holdout for i in range(len(selected_points))])

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(13, 6),
                                  gridspec_kw={"width_ratios": [1.3, 1]})

    # full plate
    ax.add_patch(plt.Rectangle((plate_xlim[0], plate_ylim[0]),
                               plate_xlim[1] - plate_xlim[0],
                               plate_ylim[1] - plate_ylim[0],
                               fill=False, edgecolor=AC["axis"], lw=1.2))
    ax.scatter(selected_points[~is_held, 0], selected_points[~is_held, 1],
               s=18, color=AC["blue"], label="train points", zorder=3)
    if is_held.any():
        ax.scatter(selected_points[is_held, 0], selected_points[is_held, 1],
                   s=40, color=AC["red"], marker="D", label="held-out points", zorder=4)
    ax.set_xlabel("x [mm]", fontsize=12)
    ax.set_ylabel("y [mm]", fontsize=12)
    ax.set_title(f"{len(selected_points)} selected points on the 300x200 mm plate",
                 fontsize=13, fontweight=600)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=10, frameon=False, loc="lower right")

    # zoomed cluster
    pad = 1.5
    axz.scatter(selected_points[~is_held, 0], selected_points[~is_held, 1],
                s=70, color=AC["blue"], edgecolor="white", lw=0.6, zorder=3)
    if is_held.any():
        axz.scatter(selected_points[is_held, 0], selected_points[is_held, 1],
                    s=110, color=AC["red"], marker="D", edgecolor="white", lw=0.6, zorder=4)
    axz.set_xlim(selected_points[:, 0].min() - pad, selected_points[:, 0].max() + pad)
    axz.set_ylim(selected_points[:, 1].min() - pad, selected_points[:, 1].max() + pad)
    axz.set_xlabel("x [mm]", fontsize=12)
    axz.set_ylabel("y [mm]", fontsize=12)
    axz.set_title("Neighbor cluster (1 mm grid)", fontsize=13, fontweight=600)
    axz.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    _save(fig, save_stem)
    plt.close(fig)


def plot_error_summary(metrics_by_point: np.ndarray, save_stem: str | None = None) -> None:
    """Histogram of per-point held-out relative-L2 errors."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(metrics_by_point, bins=20, color=AC["blue"], alpha=0.85,
            edgecolor="white")
    ax.axvline(np.median(metrics_by_point), color=AC["red"], lw=1.8, ls="--",
               label=f"median = {np.median(metrics_by_point):.3f}")
    ax.set_xlabel("Held-out relative L2 error", fontsize=13)
    ax.set_ylabel("Number of points", fontsize=13)
    ax.set_title("Reconstruction error across points", fontsize=14, fontweight=600)
    ax.legend(fontsize=11, frameon=False)
    _save(fig, save_stem)
    plt.close(fig)
