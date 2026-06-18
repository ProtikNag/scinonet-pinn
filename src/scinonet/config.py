"""Config loading and object construction from a YAML file."""

from __future__ import annotations

from typing import Any

import torch
import yaml

from .data import WaveDataset, build_dataset
from .features import RandomFourierFeatures, bandwidth_from_physics
from .models import FourierMLP
from .train import TrainConfig


def load_config(path: str) -> dict[str, Any]:
    """Read a YAML config into a nested dict."""
    with open(path) as handle:
        return yaml.safe_load(handle)


def build_dataset_from_config(cfg: dict[str, Any]) -> WaveDataset:
    """Build the standardized dataset from the ``data`` section."""
    data_cfg = cfg["data"]
    time_ranges = [tuple(r) for r in data_cfg["time_ranges"]]
    return build_dataset(
        data_cfg["csv"],
        time_ranges=time_ranges,
        split_mode=data_cfg.get("split_mode", "shared"),
        subsample_keep=float(data_cfg.get("subsample_keep", 0.5)),
        seed=int(cfg.get("seed", 42)),
    )


def build_model_from_config(cfg: dict[str, Any], data: WaveDataset) -> FourierMLP:
    """Build the Fourier-feature model, deriving bandwidth from the data stats."""
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]

    sigma = bandwidth_from_physics(
        coord_std=data.coord_scaler.std,
        f_max_temporal_hz=float(feat_cfg["f_max_temporal_hz"]),
        f_max_spatial_per_mm=float(feat_cfg["f_max_spatial_per_mm"]),
        spatial_scale=float(feat_cfg["spatial_scale"]),
        temporal_scale=float(feat_cfg["temporal_scale"]),
    )
    features = RandomFourierFeatures(
        in_features=4,
        num_frequencies=int(feat_cfg["num_frequencies"]),
        sigma_per_dim=sigma,
        seed=int(cfg.get("seed", 0)),
    )
    return FourierMLP(
        features=features,
        hidden_sizes=list(model_cfg["hidden_sizes"]),
        out_features=3,
        activation=model_cfg["activation"],
        concat_raw=bool(model_cfg["concat_raw"]),
    )


def build_train_config(cfg: dict[str, Any]) -> TrainConfig:
    """Build a :class:`TrainConfig` from the ``train`` section."""
    t = cfg["train"]
    return TrainConfig(
        epochs=int(t["epochs"]),
        batch_size=int(t["batch_size"]),
        lr=float(t["lr"]),
        weight_decay=float(t["weight_decay"]),
        grad_clip=float(t["grad_clip"]),
        scheduler_factor=float(t["scheduler_factor"]),
        scheduler_patience=int(t["scheduler_patience"]),
        min_lr=float(t["min_lr"]),
        early_stop_patience=int(t["early_stop_patience"]),
        log_every=int(t["log_every"]),
    )


def report_bandwidth(cfg: dict[str, Any], data: WaveDataset) -> None:
    """Print the resolved per-dimension RFF bandwidth for sanity checking."""
    feat_cfg = cfg["features"]
    sigma = bandwidth_from_physics(
        coord_std=data.coord_scaler.std,
        f_max_temporal_hz=float(feat_cfg["f_max_temporal_hz"]),
        f_max_spatial_per_mm=float(feat_cfg["f_max_spatial_per_mm"]),
        spatial_scale=float(feat_cfg["spatial_scale"]),
        temporal_scale=float(feat_cfg["temporal_scale"]),
    )
    std = data.coord_scaler.std
    print(f"[bandwidth] coord std (phys) = {std.tolist()}")
    print(f"[bandwidth] sigma_per_dim (cyc/std-unit) = "
          f"x={sigma[0]:.2f} y={sigma[1]:.2f} z={sigma[2]:.2f} t={sigma[3]:.2f}")
