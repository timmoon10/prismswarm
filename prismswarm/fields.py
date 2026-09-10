"""Velocity fields.

A field is a plain callable ``field(pos, vel, wavelength, t, dt, rng) ->
velocity``, where the returned array has the same shape as ``pos``. Fields
are *kinematic*: they prescribe the particle velocity directly, like a
fluid flow field advecting passive tracers, rather than a force that
accumulates into velocity over time. This is what makes stochastic,
deterministic, and wavelength-coupled fields interoperable through a
single interface: a deterministic field (e.g. radial-inward) ignores
``rng``, a stochastic field (e.g. Brownian motion) ignores ``pos``/``vel``,
a wavelength-independent field ignores ``wavelength`` — and every default
field in this module does exactly that, so wavelength coupling is strictly
opt-in (see ``wavelength_coupled`` below). ``dt`` is passed through
explicitly (rather than baked into the field at construction time) because
stochastic fields need it to produce a correctly scaled discretization of
their underlying SDE — see ``brownian`` below.

Multiple fields compose by addition (``sum_fields``), since summing
prescribed velocities is exactly how independent flows superpose — this
is also how you'd combine several differently wavelength-tuned fields
(e.g. one favoring short wavelengths, another favoring long ones) into a
single scene.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Field = Callable[
    [np.ndarray, np.ndarray, np.ndarray, float, float, np.random.Generator],
    np.ndarray,
]
WavelengthWeight = Callable[[np.ndarray], np.ndarray]


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

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
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

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        return (sigma / np.sqrt(dt) * rng.standard_normal(pos.shape)).astype(pos.dtype)

    return field


def sum_fields(*components: Field) -> Field:
    """Compose fields by summing their prescribed velocities."""

    def combined(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        total = np.zeros_like(pos)
        for component in components:
            total += component(pos, vel, wavelength, t, dt, rng)
        return total

    return combined


def wavelength_coupled(base_field: Field, weight: WavelengthWeight) -> Field:
    """Scale a base field's velocity by a per-particle wavelength-dependent
    weight. The base field itself stays wavelength-agnostic; coupling is
    entirely in ``weight``. Composes with ``sum_fields`` like any other
    field, which is how multiple differently-tuned couplings combine (e.g.
    a short-wavelength-favoring and a long-wavelength-favoring instance of
    the same base field, added together).
    """

    def coupled(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        return base_field(pos, vel, wavelength, t, dt, rng) * weight(wavelength)[:, None]

    return coupled


def power_law_weight(reference_nm: float = 530.0, exponent: float = -1.0) -> WavelengthWeight:
    """``(wavelength / reference_nm) ** exponent``.

    ``exponent = -1`` favors short wavelengths: photon momentum
    ``p = h/lambda`` is inversely proportional to wavelength, so if the
    coupled field represents radiation-pressure-like forcing, shorter
    wavelengths physically do push harder. ``exponent = +1`` favors long
    wavelengths instead — not backed by the same fundamental law, but a
    principled and equally tunable choice on its own terms (e.g. as a
    diffraction-flavored metaphor: diffraction angle scales with
    wavelength, so longer wavelengths could be read as coupling more
    strongly to a field's spatial structure). ``exponent = 0`` recovers an
    uncoupled field (weight is 1 everywhere).
    """

    def weight(wavelength: np.ndarray) -> np.ndarray:
        return (wavelength / reference_nm) ** exponent

    return weight
