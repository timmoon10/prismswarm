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

    Because speed never decays near ``center``, explicit Euler integration
    does not converge a particle to the center — it overshoots. Once a
    particle is closer than ``speed * dt``, each step sends it clean
    through to the opposite side, where the direction flips and the next
    step sends it back: a permanent period-2 limit cycle at exactly
    ``speed * dt`` from center, not decaying noise or a numerical blow-up
    (``eps`` never even engages here; nothing overflows). Composed with
    other fields — e.g. `rotational`, `brownian` — each bounce lands at a
    different angle, which is what turns this from a static back-and-forth
    into the chaotic-looking pinballing seen around confined centers.
    This is a deliberate consequence of "uniform" meaning non-decaying
    speed, not a bug to fix — see the README for the same category of
    artifact in ``rotational``.
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


def rotational(angular_velocity: float = 1.0) -> Field:
    """Rigid-body rotation about the origin, in the xy-plane (the view
    plane the orthographic projection uses) — "around the view axis" means
    only the projected coordinates rotate; any further coordinates (z, and
    any higher dimensions from a future extension) are left untouched,
    which is what makes this definition dimension-agnostic rather than
    hard-coded to 3D cross products.

    Unlike ``radial_inward``'s "uniform" (distance-independent) speed, this
    is genuine rigid-body rotation: speed grows linearly with distance from
    the axis, i.e. ``v = angular_velocity * (-y, x, 0, ...)``, since that's
    what a constant angular velocity actually means.

    No discretization correction is applied: explicit Euler integration of
    pure circular motion is unconditionally unstable and drifts outward
    over time (each step's straight-line displacement along the tangent
    lands slightly farther from the axis than it started). Deferred
    deliberately — see the README roadmap.
    """

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        v = np.zeros_like(pos)
        v[:, 0] = -angular_velocity * pos[:, 1]
        v[:, 1] = angular_velocity * pos[:, 0]
        return v

    return field


def exponential_confinement(
    center: Sequence[float] = (0.0, 0.0, 0.0),
    length_scale: float = 1.0,
    amplitude: float = 1.0,
    eps: float = 1e-6,
    max_exponent: float = 0.5 * float(np.log(np.finfo(np.float32).max)),
) -> Field:
    """A radially-symmetric field pulling toward ``center`` whose speed
    grows exponentially with distance: ``speed(r) = amplitude *
    (exp(r / length_scale) - 1)`` (``expm1`` for numerical stability near
    ``r = 0``, where it vanishes rather than a hard boundary — particles
    near the center are left to whatever other fields are active, e.g.
    Brownian, and only get pulled back once they wander roughly beyond
    ``length_scale``. A soft confinement boundary, not a wall.

    A large ``dt`` combined with a particle far past ``length_scale`` can
    otherwise produce a step large enough to overshoot the center before
    the exponential growth brakes it, sending ``r`` even farther out next
    step — a runaway that reaches ``inf`` in a handful of steps and ``nan``
    shortly after (once a position update involves ``inf - inf``). To keep
    that from ever producing non-finite state, the exponent ``r /
    length_scale`` is clamped to ``max_exponent`` before ``expm1``, which
    bounds ``speed`` to a large-but-finite value instead of letting it
    overflow. The default, ``ln(float32 max) / 2``, keeps ``exp(exponent)``
    itself far from float32 overflow, leaving headroom for the subsequent
    multiply by ``amplitude`` and the direction vector. This bounds the
    field's own output but doesn't prevent a large step from a *different*
    field or an oversized ``dt`` from still producing a bad step —
    ``dt`` modest relative to ``length_scale / amplitude`` remains good
    practice.
    """
    center_arr = np.asarray(center, dtype=np.float32)

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        offset = center_arr - pos
        dist = np.linalg.norm(offset, axis=-1, keepdims=True)
        direction = offset / np.maximum(dist, eps)
        exponent = np.minimum(dist / length_scale, max_exponent)
        speed = amplitude * np.expm1(exponent)
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
