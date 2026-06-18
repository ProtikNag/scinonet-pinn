"""Neural network models.

``FourierMLP`` is the data-only baseline: a Fourier-feature embedding followed by
an MLP that regresses the standardized displacement field directly. Predicting
the field directly avoids the magnitude-floor bug in the original notebooks,
where ``sign * 10**Softplus(mag)`` could never output a magnitude below 1 and so
could not represent ~85% of the standardized targets.

The model keeps the embedding inside ``forward`` so the same class can later feed
a physics-residual term that differentiates the output w.r.t. the raw inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import RandomFourierFeatures


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


class FourierMLP(nn.Module):
    """Fourier-feature MLP regressing the standardized displacement field.

    Args:
        features: The Fourier-feature embedding module.
        hidden_sizes: Widths of the hidden layers.
        out_features: Number of output fields (3 for ``u, v, w``).
        activation: One of ``"gelu"``, ``"tanh"``, ``"silu"``.
        concat_raw: If ``True``, append the raw coordinates to the embedding so
            the network can also represent slow trends cheaply.
    """

    def __init__(
        self,
        features: RandomFourierFeatures,
        hidden_sizes: list[int],
        out_features: int = 3,
        activation: str = "gelu",
        concat_raw: bool = True,
    ) -> None:
        super().__init__()
        self.features = features
        self.concat_raw = concat_raw

        act_cls = _ACTIVATIONS[activation]
        in_dim = features.out_features + (features.in_features if concat_raw else 0)

        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(prev, width), act_cls()])
            prev = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.features(x)
        if self.concat_raw:
            embed = torch.cat([embed, x], dim=-1)
        return self.head(self.backbone(embed))
