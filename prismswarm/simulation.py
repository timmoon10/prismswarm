"""The mutable simulation state shared between the render loop (main thread)
and the REPL (background thread).

Field-catalog membership, exposure, and dt are all plain attributes so the
REPL can rebind them (e.g. ``sim.active_field_name = "white_noise"`` or
``sim.fields["radial"] = fields.radial_field(profile=fields.constant(-0.6))``)
between frames
without any locking: each attribute read/write is a single Python
reference assignment, which is atomic under the GIL, and the render loop
only ever reads a consistent snapshot once per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import export
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
    adaptive_percentile: float = 99.5
    t: float = 0.0
    running: bool = True

    @property
    def active_field(self) -> Field:
        return self.fields[self.active_field_name]

    def random_direction(self) -> np.ndarray:
        """A random unit vector sized to this simulation's current
        dimensionality (``self.state.dim``), drawn from ``self.rng``. The
        convenience `fields.axial_field`'s ``axis`` and
        `fields.constant_field`'s ``velocity`` need but deliberately don't
        provide themselves — those constructors are plain functions with no
        notion of "the current simulation," so sizing a random default is
        this object's job, not theirs (see their docstrings in
        ``fields.py``). Draws a fresh vector each call; assign the result
        to a local if you need the same direction across multiple field
        constructions.
        """
        v = self.rng.standard_normal(self.state.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def step(self) -> None:
        self.state.step(self.active_field, self.t, self.dt, self.rng)
        self.t += self.dt

    def reset(self, n: int | None = None, dim: int | None = None, sigma: float = 0.3) -> None:
        """Reinitialize the particle population from a fresh Gaussian cloud,
        discarding current positions/velocities/wavelengths. The recovery
        path for a swarm that has wandered far off-screen or gone
        non-finite (see ``fields.py``'s ``exponential_ramp`` profile for
        one source of that) without restarting the process. Reuses ``rng``
        and ``spectrum`` as originally configured; ``n`` defaults to the
        current particle count, ``dim`` to the current dimensionality.

        Passing a different ``dim`` is how the REPL changes dimensionality
        live — every geometry/profile/gain resolves its shape from ``pos``
        at call time (see ``fields.py``), so nothing here needs to know
        ``dim`` beyond this one rebuild. The one thing this can't fix up
        for you: any field already sitting in ``sim.fields`` that was built
        with an explicit fixed-length vector for the *old* dimensionality
        (an ``axis``/``center``/``velocity`` argument, or one auto-drawn
        from the old ``dim`` at construction) will raise a shape-mismatch
        error on the next step against the new ``pos``. Fields built with
        their vector arguments left at their dimension-agnostic defaults
        (e.g. ``radial_field()`` with no ``center``) carry over fine —
        rebuild anything else for the new ``dim`` before switching to it.
        """
        n = self.state.n if n is None else n
        dim = self.state.dim if dim is None else dim
        if dim < 2:
            raise ValueError(f"dim must be >= 2 (orthographic projection needs at least an xy-plane), got {dim}")
        self.state = ParticleState.gaussian(n=n, rng=self.rng, spectrum=self.spectrum, dim=dim, sigma=sigma)
        self.detector.clear()
        self.t = 0.0

    def record(
        self,
        duration_s: float,
        path: str,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
        progress: bool = True,
    ) -> None:
        """Render ``duration_s`` seconds of simulated time to an MP4 at
        ``path``, starting from a snapshot of the current state/field/rng —
        see ``export.record`` for the full docstring, including why this is
        safe to call from the REPL thread while the render loop is live on
        the main thread (it never steps or mutates ``self`` directly).
        """
        export.record(self, duration_s, path, fps=fps, width=width, height=height, progress=progress)
