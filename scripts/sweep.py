"""Quick hyperparameter sweep over Fourier bandwidth and regularization.

Builds the dataset once and trains several configs, reporting held-out error.
Used to find a setting that interpolates the held-out gaps rather than
overfitting the training windows.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scinonet.config import build_dataset_from_config, load_config  # noqa: E402
from scinonet.evaluate import reconstruct_point, relative_l2  # noqa: E402
from scinonet.features import RandomFourierFeatures, bandwidth_from_physics  # noqa: E402
from scinonet.models import FourierMLP  # noqa: E402
from scinonet.seed import resolve_device, resolve_dtype, set_seed  # noqa: E402
from scinonet.train import TrainConfig, train_model  # noqa: E402


def heldout_median(model, data, device) -> float:
    errs = []
    for pidx in range(data.n_points):
        rec = reconstruct_point(model, data, pidx, device)
        ti = rec["test_idx"]
        gt, pred = rec["gt"][ti], rec["pred"][ti]
        valid = ~np.isnan(gt).any(axis=1)
        errs.append(relative_l2(pred[valid], gt[valid]))
    return float(np.median(errs))


def main() -> None:
    cfg = load_config("configs/default.yaml")
    device = resolve_device(cfg["device"])
    dtype = resolve_dtype(device)
    torch.set_default_dtype(dtype)

    data = build_dataset_from_config(cfg)
    for attr in ["X_train", "y_train", "X_test", "y_test"]:
        setattr(data, attr, getattr(data, attr).to(dtype))

    temporal_scales = [0.5, 0.7, 1.0]
    weight_decays = [0.0, 1e-5, 1e-4]
    num_freqs = [256]
    epochs = 150

    print(f"{'tscale':>7} {'wd':>8} {'nfreq':>6} | {'train_L2':>9} {'test_L2':>9} {'heldout_med':>11}")
    results = []
    for tscale, wd, nfreq in itertools.product(temporal_scales, weight_decays, num_freqs):
        set_seed(cfg["seed"])
        sigma = bandwidth_from_physics(
            data.coord_scaler.std,
            float(cfg["features"]["f_max_temporal_hz"]),
            float(cfg["features"]["f_max_spatial_per_mm"]),
            float(cfg["features"]["spatial_scale"]),
            tscale,
        )
        feats = RandomFourierFeatures(4, nfreq, sigma, seed=cfg["seed"])
        model = FourierMLP(feats, cfg["model"]["hidden_sizes"],
                           activation=cfg["model"]["activation"],
                           concat_raw=cfg["model"]["concat_raw"]).to(dtype)
        tcfg = TrainConfig(epochs=epochs, batch_size=16384, lr=2e-3,
                           weight_decay=wd, early_stop_patience=epochs,
                           log_every=epochs + 1)
        _, metrics = train_model(model, data, tcfg, device, verbose=False)
        hm = heldout_median(model, data, device)
        results.append((tscale, wd, nfreq, metrics["train_relL2_all"],
                        metrics["test_relL2_all"], hm))
        print(f"{tscale:>7.2f} {wd:>8.0e} {nfreq:>6d} | "
              f"{metrics['train_relL2_all']:>9.3f} {metrics['test_relL2_all']:>9.3f} {hm:>11.3f}")

    best = min(results, key=lambda r: r[5])
    print(f"\nbest by heldout median: tscale={best[0]} wd={best[1]:.0e} "
          f"nfreq={best[2]} -> heldout_med={best[5]:.3f}")


if __name__ == "__main__":
    main()
