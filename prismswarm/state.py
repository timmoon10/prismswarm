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
    def uniform_ball(
        cls,
        n: int,
        rng: np.random.Generator,
        spectrum: Spectrum,
        dim: int = 3,
        radius: float = 1.0,
    ) -> "ParticleState":
        """Particles distributed uniformly by volume in a ``dim``-ball, with
        wavelengths drawn from ``spectrum`` (see ``spectra.py``)."""
        directions = rng.standard_normal((n, dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = radius * rng.random(n) ** (1.0 / dim)
        positions = (directions * radii[:, None]).astype(np.float32)
        velocities = np.zeros((n, dim), dtype=np.float32)
        wavelengths = spectrum(n, rng).astype(np.float32)
        return cls(positions, velocities, wavelengths)

    def step(self, field: Field, t: float, dt: float, rng: np.random.Generator) -> None:
        """Advance one Euler step: velocity is recomputed fresh from the
        field (it's a prescribed quantity, not an accumulator — see
        ``fields.py``), then position is advected by it."""
        self.velocities = field(self.positions, self.velocities, t, dt, rng)
        self.positions = self.positions + self.velocities * dt
