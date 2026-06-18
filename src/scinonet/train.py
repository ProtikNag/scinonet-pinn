"""Training loop for the data-only Fourier baseline.

The held-out gaps act as the validation signal here: they are never used to
update weights, only to track generalization and drive the LR scheduler and
early stopping. This is honest because the gaps are pure ground truth the model
never sees during optimization.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import WaveDataset
from .evaluate import evaluate_split_metrics


@dataclass
class TrainConfig:
    """Hyperparameters for a training run."""

    epochs: int = 300
    batch_size: int = 16384
    lr: float = 2e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 12
    min_lr: float = 1e-5
    early_stop_patience: int = 60
    log_every: int = 10


@dataclass
class TrainHistory:
    """Per-epoch training history."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)


def train_model(
    model: nn.Module,
    data: WaveDataset,
    config: TrainConfig,
    device: torch.device,
    verbose: bool = True,
) -> tuple[TrainHistory, dict[str, float]]:
    """Train ``model`` on the kept windows, validate on the held-out gaps.

    Args:
        model: The network to train (moved to ``device`` internally).
        data: Standardized dataset container.
        config: Training hyperparameters.
        device: Compute device.
        verbose: Whether to print a run header and per-epoch logs.

    Returns:
        A ``(history, final_metrics)`` tuple. The model is left holding the
        best (lowest val-loss) weights.
    """
    model = model.to(device)
    train_loader = DataLoader(
        TensorDataset(data.X_train, data.y_train),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    X_val = data.X_test.to(device)
    y_val = data.y_test.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_lr,
    )
    mse = nn.MSELoss()

    history = TrainHistory()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    if verbose:
        print(
            f"[run] device={device} | points={data.n_points} | "
            f"train_rows={len(data.X_train)} | val_rows={len(data.X_test)} | "
            f"params={sum(p.numel() for p in model.parameters()):,}"
        )

    start = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = mse(model(xb), yb)
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(1, n_batches)

        model.eval()
        with torch.no_grad():
            val_loss = mse(model(X_val), y_val).item()

        scheduler.step(val_loss)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.lr.append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % config.log_every == 0 or epoch == 1):
            print(
                f"epoch {epoch:04d} | train {train_loss:.3e} | val {val_loss:.3e} | "
                f"best {best_val:.3e} | lr {history.lr[-1]:.2e}"
            )

        if epochs_no_improve >= config.early_stop_patience:
            if verbose:
                print(f"early stop at epoch {epoch} (no val improvement for "
                      f"{config.early_stop_patience} epochs)")
            break

    model.load_state_dict(best_state)
    metrics = evaluate_split_metrics(model, data, device)
    if verbose:
        elapsed = time.time() - start
        print(f"[done] {elapsed/60:.2f} min | best_val {best_val:.3e}")
        print("[metrics] " + " ".join(f"{k}={v:.3e}" for k, v in metrics.items()))
    return history, metrics
