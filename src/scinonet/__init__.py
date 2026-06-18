"""SciNoNet: Fourier-feature networks for elastic-wave signal reconstruction.

The package is organised so that the data-only Fourier baseline and the future
physics-informed (PINN) extension share the same data pipeline, feature
embedding, and training scaffolding. See ``HANDOFF.md`` at the repo root for the
roadmap from the data-only model to the full PINN.
"""

from .seed import set_seed

__all__ = ["set_seed"]
