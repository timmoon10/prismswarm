"""The mutable simulation state shared between the render loop (main thread)
and the REPL (background thread).

Field-catalog membership, exposure, and dt are all plain attributes so the
REPL can rebind them (e.g. ``sim.active_field_name = "brownian"`` or
``sim.fields["radial"] = fields.radial_inward(speed=0.6)``) between frames
without any locking: each attribute read/write is a single Python
reference assignment, which is atomic under the GIL, and the render loop
only ever reads a consistent snapshot once per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import Detector
from .fields import Field
from .spectra import Spectrum
from .state import ParticleState


@dataclass
class Simulation:
    state: ParticleState
    detector: Detector
    fields: dict[str, Field]
    active_field_name: str
    rng: np.random.Generator
    spectrum: Spectrum
    dt: float = 1.0 / 30.0
    exposure: float = 1.0
    adaptive_exposure: bool = True
    adaptive_percentile: float = 100.0
    t: float = 0.0
    running: bool = True

    @property
    def active_field(self) -> Field:
        return self.fields[self.active_field_name]

    def step(self) -> None:
        self.state.step(self.active_field, self.t, self.dt, self.rng)
        self.t += self.dt

    def reset(self, n: int | None = None, radius: float = 1.0) -> None:
        """Reinitialize the particle population from a fresh uniform ball,
        discarding current positions/velocities/wavelengths. The recovery
        path for a swarm that has wandered far off-screen or gone
        non-finite (see ``fields.py``'s ``exponential_confinement`` for one
        source of that) without restarting the process. Reuses ``rng`` and
        ``spectrum`` as originally configured; ``n`` defaults to the
        current particle count.
        """
        n = self.state.n if n is None else n
        self.state = ParticleState.uniform_ball(
            n=n, rng=self.rng, spectrum=self.spectrum, dim=self.state.dim, radius=radius
        )
        self.detector.clear()
        self.t = 0.0
