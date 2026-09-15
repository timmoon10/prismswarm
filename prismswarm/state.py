"""Particle state: positions, velocities, emission wavelengths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import Field
from .spectra import Spectrum


@dataclass
class ParticleState:
    positions: np.ndarray  # (n, dim) float32
    velocities: np.ndarray  # (n, dim) float32
    wavelengths: np.ndarray  # (n,) float32, nm

    @property
    def n(self) -> int:
        return self.positions.shape[0]

    @property
    def dim(self) -> int:
        return self.positions.shape[1]

    @classmethod
    def gaussian(
        cls,
        n: int,
        rng: np.random.Generator,
        spectrum: Spectrum,
        dim: int = 3,
        scale: float = 0.3,
    ) -> "ParticleState":
        """Particles distributed as an isotropic Gaussian cloud centered on
        the origin, with per-axis standard deviation ``scale``, and
        wavelengths drawn from ``spectrum`` (see ``spectra.py``)."""
        positions = (scale * rng.standard_normal((n, dim))).astype(np.float32)
        velocities = np.zeros((n, dim), dtype=np.float32)
        wavelengths = spectrum(n, rng).astype(np.float32)
        return cls(positions, velocities, wavelengths)

    def step(self, field: Field, t: float, dt: float, rng: np.random.Generator) -> None:
        """Advance one Euler step: velocity is recomputed fresh from the
        field (it's a prescribed quantity, not an accumulator — see
        ``fields.py``), then position is advected by it."""
        self.velocities = field(self.positions, self.velocities, self.wavelengths, t, dt, rng)
        self.positions = self.positions + self.velocities * dt
