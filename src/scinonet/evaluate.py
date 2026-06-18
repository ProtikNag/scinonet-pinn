"""Evaluation metrics for signal reconstruction."""

from __future__ import annotations

import numpy as np
import torch

from .data import FIELD_COLS, N_T, Standardizer, WaveDataset, full_grid_inputs


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    """Relative L2 error ``||pred - target|| / ||target||``."""
    denom = np.linalg.norm(target)
    if denom == 0:
        return float(np.linalg.norm(pred - target))
    return float(np.linalg.norm(pred - target) / denom)


@torch.no_grad()
def predict_fields(model: torch.nn.Module, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Forward pass returning standardized field predictions on CPU."""
    model.eval()
    dtype = next(model.parameters()).dtype
    return model(X.to(device=device, dtype=dtype)).cpu().double()


def reconstruct_point(
    model: torch.nn.Module,
    data: WaveDataset,
    point_index: int,
    device: torch.device,
    n_t: int = N_T,
) -> dict[str, np.ndarray]:
    """Reconstruct the full signal at one point in physical units.

    Returns a dict with ``pred`` (``[n_t, 3]``), ``gt`` (``[n_t, 3]``) and
    ``train_idx`` / ``test_idx`` timestep indices for that point.
    """
    point = data.points[point_index]
    grid = full_grid_inputs(data.coord_scaler, point, n_t=n_t)
    pred_std = predict_fields(model, grid, device)
    pred = data.field_scaler.inverse(pred_std).numpy()

    df = data.df
    mask = (
        (np.abs(df["x"].to_numpy() - point[0]) < 1e-6)
        & (np.abs(df["y"].to_numpy() - point[1]) < 1e-6)
    )
    grp = df[mask].sort_values("t")
    gt = np.full((n_t, 3), np.nan)
    idx = grp["t_idx"].to_numpy()
    gt[idx] = grp[FIELD_COLS].to_numpy()

    # Per-point train/test timestep indices from the recorded split mask.
    train_keep = np.zeros(n_t, dtype=bool)
    if "is_train" in grp.columns:
        train_keep[grp["t_idx"].to_numpy()] = grp["is_train"].to_numpy()
    return {
        "pred": pred,
        "gt": gt,
        "train_idx": np.where(train_keep)[0],
        "test_idx": np.where(~train_keep)[0],
        "point": point,
    }


def evaluate_split_metrics(
    model: torch.nn.Module, data: WaveDataset, device: torch.device
) -> dict[str, float]:
    """Per-component and overall relative-L2 on train and held-out test rows.

    Metrics are computed in physical units after inverse-standardization.
    """
    out: dict[str, float] = {}
    for split, X, y in [
        ("train", data.X_train, data.y_train),
        ("test", data.X_test, data.y_test),
    ]:
        pred = data.field_scaler.inverse(predict_fields(model, X, device)).numpy()
        target = data.field_scaler.inverse(y).numpy()
        for ci, comp in enumerate(FIELD_COLS):
            out[f"{split}_relL2_{comp}"] = relative_l2(pred[:, ci], target[:, ci])
        out[f"{split}_relL2_all"] = relative_l2(pred, target)
    return out
