"""Build the headline comparison figure: subsample success vs mixed failure.

Loads the two saved checkpoints, reconstructs the same point/component under
each, and stacks them so the contrast (data-only fills small gaps, not wide
ones) is visible at a glance. Output as PNG + SVG.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scinonet.config import build_dataset_from_config, build_model_from_config  # noqa: E402
from scinonet.evaluate import reconstruct_point, relative_l2  # noqa: E402
from scinonet.seed import resolve_device, resolve_dtype  # noqa: E402
from scinonet import viz  # noqa: E402


def load_run(tag: str, device, dtype):
    ckpt = torch.load(f"outputs/checkpoints/model_{tag}.pt", weights_only=False)
    cfg = ckpt["config"]
    data = build_dataset_from_config(cfg)
    for attr in ["X_train", "y_train", "X_test", "y_test"]:
        setattr(data, attr, getattr(data, attr).to(dtype))
    model = build_model_from_config(cfg, data).to(dtype).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, data, ckpt["metrics"]


def main() -> None:
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    # Each panel is "tag:Label text"; default reproduces the original figure.
    parser.add_argument("--panels", nargs="*", default=[
        "subsample:Data-only, 50% random temporal samples",
        "mixed:Data-only, windowed split (wide data-free gaps)",
    ])
    parser.add_argument("--point", type=int, default=10)
    parser.add_argument("--comp", default="w")
    parser.add_argument("--out", default="outputs/figures/comparison_subsample_vs_mixed")
    args = parser.parse_args()

    device = resolve_device("auto")
    dtype = resolve_dtype(device)
    torch.set_default_dtype(dtype)

    point_index, comp = args.point, args.comp
    ci = {"u": 0, "v": 1, "w": 2}[comp]

    viz.apply_style()
    fig, axes = plt.subplots(len(args.panels), 1, figsize=(14, 3.5 * len(args.panels)),
                             sharex=True)
    if len(args.panels) == 1:
        axes = [axes]
    panels = [(p.split(":", 1)[0], p.split(":", 1)[1], ax)
              for p, ax in zip(args.panels, axes)]
    for tag, title, ax in panels:
        model, data, metrics = load_run(tag, device, dtype)
        rec = reconstruct_point(model, data, point_index, device)
        t = np.arange(len(rec["pred"]))
        gt, pred = rec["gt"][:, ci], rec["pred"][:, ci]

        ti = rec["test_idx"]
        valid = ~np.isnan(rec["gt"][ti]).any(axis=1)
        err = relative_l2(rec["pred"][ti][valid], rec["gt"][ti][valid])

        train_idx = rec["train_idx"]
        if len(train_idx):
            splits = np.where(np.diff(train_idx) > 1)[0]
            starts = np.concatenate([[train_idx[0]], train_idx[splits + 1]])
            ends = np.concatenate([train_idx[splits], [train_idx[-1]]])
            for s, e in zip(starts, ends):
                ax.axvspan(s, e, color=viz.AC["blue"], alpha=0.06, zorder=0)

        ax.plot(t, gt, color=viz.AC["ink"], lw=1.3, label="Ground truth", zorder=3)
        ax.plot(t, pred, color=viz.AC["red"], lw=1.5, ls="--", alpha=0.85,
                label="Predicted (full)", zorder=4)
        ax.set_ylabel(comp, fontsize=13)
        ax.set_title(f"{title}   |   held-out relL2 = {err:.3f}",
                     fontsize=13, fontweight=600)
        ax.legend(fontsize=10, ncol=2, loc="upper right", frameon=False)

    panels[-1][2].set_xlabel("Timestep index", fontsize=13)
    px, py = data.points[point_index][0], data.points[point_index][1]
    fig.suptitle(f"Full-signal reconstruction at (x={px:.1f}, y={py:.1f}) — "
                 f"shaded = training samples",
                 fontsize=15, fontweight=600, y=1.0)
    fig.tight_layout()
    viz._save(fig, args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
