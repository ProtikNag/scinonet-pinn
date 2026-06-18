"""Reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every relevant RNG and force deterministic cuDNN.

    Args:
        seed: Integer seed applied to ``random``, ``numpy`` and ``torch``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(prefer: str = "auto") -> torch.device:
    """Pick a compute device.

    Args:
        prefer: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``. ``"auto"`` chooses
            CUDA, then Apple MPS, then CPU.

    Returns:
        The selected ``torch.device``.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device: torch.device) -> torch.dtype:
    """Pick a float dtype. MPS lacks float64, so use float32 off CPU.

    Args:
        device: The compute device.

    Returns:
        ``torch.float64`` on CPU (best for autograd-heavy PINN residuals),
        ``torch.float32`` on MPS/CUDA.
    """
    return torch.float64 if device.type == "cpu" else torch.float32
