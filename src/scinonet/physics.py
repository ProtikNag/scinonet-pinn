"""Elastic-wave PDE residual for the future PINN stage (NOT used by the baseline).

This module records the *correct* non-dimensionalized residual so the migration
to a full PINN does not repeat the unit bugs in the original notebooks. It is
intentionally unused by the data-only baseline; see ``HANDOFF.md``.

Physical model. The plate displacement is decomposed via Helmholtz potentials
(scalar ``phi`` for the dilatational part, vector ``psi`` for the rotational
part). Each potential obeys a scalar wave equation with the appropriate speed::

    cp^2 (phi_xx + phi_yy) - phi_tt = 0          (cp: longitudinal speed)
    cs^2 (psi_xx + psi_yy) - psi_tt = 0          (cs: shear speed)

The two unit corrections relative to the notebooks:

1. Speeds in consistent units. With ``x`` in mm and ``t`` in s,
   ``cp = 6300 m/s = 6.3e6 mm/s`` (the notebooks used ``6.3``).

2. Derivatives taken w.r.t. *standardized* coordinates must be rescaled by the
   standardization std before entering the physical PDE::

       d/dx_phys   = (1/std_x) d/dx_std
       d^2/dx_phys^2 = (1/std_x^2) d^2/dx_std^2

   Likewise the network output is standardized, so multiply field second
   derivatives by ``std_field`` to return to physical units. Equivalently, work
   fully in physical units inside the residual.

The cleanest implementation differentiates the *physical* field w.r.t. the
*physical* coordinates by composing the inverse-standardization with the
network, letting autograd carry the chain rule. A reference implementation is
sketched below; wire it in when the PINN stage begins.
"""

from __future__ import annotations

import torch

CP_MM_PER_S = 6.3e6   # longitudinal wave speed [mm/s]
CS_MM_PER_S = 3.2e6   # shear wave speed [mm/s]


def second_derivative(field: torch.Tensor, coords: torch.Tensor, dim: int) -> torch.Tensor:
    """Second partial derivative of ``field`` w.r.t. ``coords[:, dim]``.

    ``coords`` must have ``requires_grad=True`` and ``field`` must be a scalar
    column produced from it within the active autograd graph.

    Args:
        field: Scalar field values, shape ``[N]``.
        coords: Input coordinates, shape ``[N, D]``, requiring grad.
        dim: Coordinate index to differentiate against.

    Returns:
        Second derivative, shape ``[N]``.
    """
    grad = torch.autograd.grad(
        field, coords, grad_outputs=torch.ones_like(field), create_graph=True
    )[0][:, dim]
    grad2 = torch.autograd.grad(
        grad, coords, grad_outputs=torch.ones_like(grad), create_graph=True
    )[0][:, dim]
    return grad2


def wave_residual(
    potential: torch.Tensor,
    coords_phys: torch.Tensor,
    speed_mm_per_s: float,
    x_dim: int = 0,
    y_dim: int = 1,
    t_dim: int = 3,
) -> torch.Tensor:
    """Scalar wave-equation residual ``c^2 (p_xx + p_yy) - p_tt`` in physical units.

    Args:
        potential: Scalar potential evaluated at ``coords_phys``, shape ``[N]``.
        coords_phys: Physical coordinates ``[x, y, z, t]`` (grad-enabled), ``[N, 4]``.
        speed_mm_per_s: Wave speed for this potential [mm/s].
        x_dim, y_dim, t_dim: Column indices of the respective axes.

    Returns:
        Residual values, shape ``[N]`` (zero for an exact solution).
    """
    p_xx = second_derivative(potential, coords_phys, x_dim)
    p_yy = second_derivative(potential, coords_phys, y_dim)
    p_tt = second_derivative(potential, coords_phys, t_dim)
    return speed_mm_per_s**2 * (p_xx + p_yy) - p_tt
