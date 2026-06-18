"""End-to-end experiment: load -> train -> evaluate -> visualize.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --config configs/default.yaml \
        --override features.temporal_scale=1.5 train.epochs=200 --tag tscale1p5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scinonet.config import (  # noqa: E402
    build_dataset_from_config,
    build_model_from_config,
    build_train_config,
    load_config,
    report_bandwidth,
)
from scinonet.evaluate import reconstruct_point, relative_l2  # noqa: E402
from scinonet.seed import resolve_device, resolve_dtype, set_seed  # noqa: E402
from scinonet.train import train_model  # noqa: E402
from scinonet import viz  # noqa: E402


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply ``a.b.c=value`` dotted overrides in place (value parsed as YAML scalar)."""
    import yaml
    for item in overrides:
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--tag", default="default")
    parser.add_argument("--outdir", default=None,
                        help="Self-contained run directory (default outputs/data_only/<tag>)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        apply_overrides(cfg, args.override)

    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    dtype = resolve_dtype(device)
    torch.set_default_dtype(dtype)
    print(f"[setup] device={device} dtype={dtype}")

    data = build_dataset_from_config(cfg)
    report_bandwidth(cfg, data)
    # Match dataset tensors to the device dtype (MPS/CUDA -> float32).
    data.X_train = data.X_train.to(dtype)
    data.y_train = data.y_train.to(dtype)
    data.X_test = data.X_test.to(dtype)
    data.y_test = data.y_test.to(dtype)

    model = build_model_from_config(cfg, data).to(dtype)
    train_cfg = build_train_config(cfg)
    history, metrics = train_model(model, data, train_cfg, device)

    # Per-point held-out reconstruction error (component w by default for summary).
    per_point_err = []
    for pidx in range(data.n_points):
        rec = reconstruct_point(model, data, pidx, device)
        test_idx = rec["test_idx"]
        gt = rec["gt"][test_idx]
        pred = rec["pred"][test_idx]
        valid = ~np.isnan(gt).any(axis=1)
        per_point_err.append(relative_l2(pred[valid], gt[valid]))
    per_point_err = np.array(per_point_err)
    metrics["heldout_relL2_median"] = float(np.median(per_point_err))
    metrics["heldout_relL2_mean"] = float(np.mean(per_point_err))

    # Self-contained run directory: figures, checkpoint, metrics, history together.
    run_dir = args.outdir or os.path.join("outputs", "data_only", args.tag)
    os.makedirs(run_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "metrics": metrics},
               os.path.join(run_dir, "model.pt"))

    print("\n[summary]")
    print(json.dumps(metrics, indent=2))

    if not args.no_figures:
        viz.plot_loss_curves(history, os.path.join(run_dir, "loss_curves"))
        viz.plot_point_layout(data.points, os.path.join(run_dir, "point_layout"))
        viz.plot_error_summary(per_point_err, os.path.join(run_dir, "error_summary"))
        for pidx in cfg["output"]["recon_points"]:
            if pidx >= data.n_points:
                continue
            rec = reconstruct_point(model, data, pidx, device)
            for comp in cfg["output"]["recon_components"]:
                viz.plot_reconstruction(
                    rec, component=comp,
                    save_stem=os.path.join(run_dir, f"recon_p{pidx}_{comp}"),
                )

    with open(os.path.join(run_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w") as handle:
        json.dump({"train_loss": history.train_loss, "val_loss": history.val_loss,
                   "lr": history.lr}, handle)


if __name__ == "__main__":
    main()
