"""Velocity fields.

A field is a plain callable ``field(pos, vel, t, dt, rng) -> velocity``,
where the returned array has the same shape as ``pos``. Fields are
*kinematic*: they prescribe the particle velocity directly, like a fluid
flow field advecting passive tracers, rather than a force that accumulates
into velocity over time. This is what makes stochastic and deterministic
fields interoperable through a single interface: a deterministic field
(e.g. radial-inward) ignores ``rng``, a stochastic field (e.g. Brownian
motion) ignores ``pos``/``vel``, and both return "the velocity for this
step." ``dt`` is passed through explicitly (rather than baked into the
field at construction time) because stochastic fields need it to produce
a correctly scaled discretization of their underlying SDE — see
``brownian`` below.

Multiple fields compose by addition (``sum_fields``), since summing
prescribed velocities is exactly how independent flows superpose.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Field = Callable[[np.ndarray, np.ndarray, float, float, np.random.Generator], np.ndarray]


def radial_inward(
    center: Sequence[float] = (0.0, 0.0, 0.0),
    speed: float = 1.0,
    eps: float = 1e-6,
) -> Field:
    """A field of constant magnitude pointing toward ``center``.

    "Uniform" refers to speed, not direction: every particle moves toward
    the center at the same rate regardless of its distance from it. ``eps``
    guards the direction normalization for particles exactly at the center.
    """
    center_arr = np.asarray(center, dtype=np.float32)

    def field(pos: np.ndarray, vel: np.ndarray, t: float, dt: float, rng: np.random.Generator) -> np.ndarray:
        offset = center_arr - pos
        dist = np.linalg.norm(offset, axis=-1, keepdims=True)
        direction = offset / np.maximum(dist, eps)
        return (direction * speed).astype(pos.dtype)

    return field


def brownian(sigma: float = 1.0) -> Field:
    """Brownian motion, discretized via Euler-Maruyama.

    The Wiener process gives a position increment ``dx = sigma * sqrt(dt) *
    randn()`` per step. To fit the velocity-field interface (where the
    integrator computes ``x += v * dt``), the field must return
    ``v = sigma / sqrt(dt) * randn()`` so that ``v * dt`` recovers the
    correct increment. This velocity grows without bound as ``dt -> 0``,
    which is expected: it reflects the non-differentiability of Brownian
    paths, not a bug.
    """

    def field(pos: np.ndarray, vel: np.ndarray, t: float, dt: float, rng: np.random.Generator) -> np.ndarray:
        return (sigma / np.sqrt(dt) * rng.standard_normal(pos.shape)).astype(pos.dtype)

    return field


def sum_fields(*components: Field) -> Field:
    """Compose fields by summing their prescribed velocities."""

    def combined(pos: np.ndarray, vel: np.ndarray, t: float, dt: float, rng: np.random.Generator) -> np.ndarray:
        total = np.zeros_like(pos)
        for component in components:
            total += component(pos, vel, t, dt, rng)
        return total

    return combined
