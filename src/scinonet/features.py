"""Random Fourier Feature (RFF) embedding.

The embedding maps standardized coordinates ``x`` to
``[sin(2*pi * x @ B), cos(2*pi * x @ B)]`` with a frozen Gaussian frequency
matrix ``B`` whose per-dimension bandwidth is set explicitly. This is the
Tancik et al. (2020) "Fourier Features" construction.

Two design points that matter for this project:

1. The frequency bandwidth is the single most important knob. The temporal
   signal carries content up to roughly 4 cycles per standardized time unit, so
   the temporal bandwidth must reach that band. The original notebooks either
   set it ~1000x too high (raw Hz applied to standardized inputs, causing
   aliasing) or too low (covering ~2 of the needed ~9 cycles).

2. ``B`` is a registered buffer (not a learnable parameter) and the forward pass
   is a plain differentiable op, so autograd can propagate PDE-residual
   gradients back to the raw coordinates when the PINN stage is added.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RandomFourierFeatures(nn.Module):
    """Gaussian Random Fourier Feature embedding with per-dimension bandwidth.

    Args:
        in_features: Number of input coordinate dims (4 for ``x, y, z, t``).
        num_frequencies: Number of Fourier bases ``L``; output dimension is ``2L``.
        sigma_per_dim: Per-dimension frequency std (cycles per standardized unit),
            shape ``[in_features]``. A dimension with ``sigma == 0`` contributes no
            frequency content (useful for the constant ``z`` axis).
        seed: Seed for drawing ``B`` so the embedding is reproducible.
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int,
        sigma_per_dim: torch.Tensor,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies

        generator = torch.Generator().manual_seed(seed)
        base = torch.randn(in_features, num_frequencies, generator=generator, dtype=torch.float64)
        sigma = sigma_per_dim.to(torch.float64).unsqueeze(1)
        self.register_buffer("B", base * sigma)

    @property
    def out_features(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * (x @ self.B.to(x))
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class SpecializedFourierFeatures(nn.Module):
    """Dispersion-informed Fourier features with a learnable, clamped B.

    This is the corrected version of the notebooks' ``FourierFeatures``: instead
    of a random Gaussian frequency matrix, ``B`` is *learnable* and *clamped* to
    per-dimension frequency ranges derived from the Lamb-wave dispersion curve.

    The embedding is ``[A * sin(2*pi * x @ B), A * cos(2*pi * x @ B)]`` with the
    amplitude ``A`` applied *outside* the sinusoids (one learnable amplitude per
    frequency). This separates the role of ``A`` (amplitude) from ``B``
    (frequency); the notebooks instead used ``a*B`` inside the sinusoid, which
    rescales the frequency and lets it escape the clamped physical band.

    The notebooks also expressed the ranges in raw Hz/rad-per-mm but applied them
    to standardized inputs (off by ~1000x, causing aliasing); here the ranges are
    given directly in standardized cycles-per-unit.

    Args:
        lo: Per-dimension lower frequency bounds, shape ``[in_features]``.
        hi: Per-dimension upper frequency bounds, shape ``[in_features]``.
        num_frequencies: Number of bases ``L`` (output dim ``2L``).
        seed: Seed for the uniform initialization of ``B``.
    """

    def __init__(self, lo: torch.Tensor, hi: torch.Tensor, num_frequencies: int,
                 seed: int = 0) -> None:
        super().__init__()
        self.in_features = lo.numel()
        self.num_frequencies = num_frequencies
        lo = lo.to(torch.float64).unsqueeze(1)
        hi = hi.to(torch.float64).unsqueeze(1)

        gen = torch.Generator().manual_seed(seed)
        u = torch.rand(self.in_features, num_frequencies, generator=gen, dtype=torch.float64)
        self.B = nn.Parameter(lo + u * (hi - lo))                       # frequencies (clamped)
        self.A = nn.Parameter(torch.ones(num_frequencies, dtype=torch.float64))  # amplitudes
        self.register_buffer("B_min", lo)
        self.register_buffer("B_max", hi)

    @property
    def out_features(self) -> int:
        return 2 * self.num_frequencies

    def clamp_B(self) -> None:
        """Clamp B back into the per-dimension ranges. Call after optimizer.step()."""
        with torch.no_grad():
            self.B.clamp_(self.B_min, self.B_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * (x @ self.B.to(x))          # [N, L], B = pure frequency
        amp = self.A.to(x)                                # [L], amplitude outside
        return torch.cat([amp * torch.sin(proj), amp * torch.cos(proj)], dim=-1)

    @classmethod
    def from_dispersion(
        cls,
        coord_std: torch.Tensor,
        f_max_temporal_hz: float,
        k_max_spatial_per_mm: float,
        num_frequencies: int,
        in_features: int = 3,
        spatial_scale: float = 1.0,
        temporal_scale: float = 1.0,
        seed: int = 0,
    ) -> "SpecializedFourierFeatures":
        """Build standardized per-dimension ranges from the dispersion curve.

        A physical frequency ``f`` maps to ``f * std`` cycles per standardized
        unit. Spatial ranges are symmetric (forward/backward traveling waves);
        the temporal range is non-negative. For ``in_features == 3`` the dims are
        ``[x, y, t]``; for 4 they are ``[x, y, z, t]`` with ``z`` fixed at 0.

        Args:
            coord_std: Physical std per coordinate dim (``[L, L, s_t]`` or
                ``[sx, sy, sz, st]``).
            f_max_temporal_hz: Max temporal frequency to resolve [Hz].
            k_max_spatial_per_mm: Max spatial frequency to resolve [cycles/mm].
            num_frequencies: Number of bases.
            in_features: 3 (x,y,t) or 4 (x,y,z,t).
            spatial_scale, temporal_scale: Tuning multipliers on the ranges.
            seed: Init seed.
        """
        std = coord_std.to(torch.float64)
        ks = spatial_scale * k_max_spatial_per_mm
        ft = temporal_scale * f_max_temporal_hz
        if in_features == 3:
            sx, sy, st = float(std[0]), float(std[1]), float(std[2])
            lo = torch.tensor([-ks * sx, -ks * sy, 0.0], dtype=torch.float64)
            hi = torch.tensor([ks * sx, ks * sy, ft * st], dtype=torch.float64)
        else:
            sx, sy, st = float(std[0]), float(std[1]), float(std[3])
            lo = torch.tensor([-ks * sx, -ks * sy, 0.0, 0.0], dtype=torch.float64)
            hi = torch.tensor([ks * sx, ks * sy, 0.0, ft * st], dtype=torch.float64)
        return cls(lo, hi, num_frequencies, seed=seed)


def bandwidth_from_physics(
    coord_std: torch.Tensor,
    f_max_temporal_hz: float,
    f_max_spatial_per_mm: float,
    spatial_scale: float = 1.0,
    temporal_scale: float = 1.0,
) -> torch.Tensor:
    """Per-dimension RFF bandwidth derived from physical max frequencies.

    A physical frequency ``f`` (cycles per physical unit) maps to
    ``f * std`` cycles per standardized unit after z-scoring, because the
    standardized coordinate compresses the physical axis by ``std``. This is the
    unit conversion the original notebooks got wrong.

    Args:
        coord_std: Physical std of ``[x, y, z, t]`` used for standardization.
        f_max_temporal_hz: Max temporal frequency to resolve [Hz].
        f_max_spatial_per_mm: Max spatial frequency to resolve [cycles/mm].
        spatial_scale: Multiplier on the spatial bandwidth (tuning knob).
        temporal_scale: Multiplier on the temporal bandwidth (tuning knob).

    Returns:
        ``sigma_per_dim`` tensor of shape ``[4]`` for ``x, y, z, t``.
    """
    std = coord_std.to(torch.float64)
    sigma_x = spatial_scale * f_max_spatial_per_mm * float(std[0])
    sigma_y = spatial_scale * f_max_spatial_per_mm * float(std[1])
    sigma_z = 0.0  # constant axis, no frequency content
    sigma_t = temporal_scale * f_max_temporal_hz * float(std[3])
    return torch.tensor([sigma_x, sigma_y, sigma_z, sigma_t], dtype=torch.float64)
