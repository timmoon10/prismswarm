"""Velocity fields.

A field is a plain callable ``field(pos, vel, wavelength, t, dt, rng) ->
velocity``, where the returned array has the same shape as ``pos``. Fields
are *kinematic*: they prescribe the particle velocity directly, like a
fluid flow field advecting passive tracers, rather than a force that
accumulates into velocity over time. This is what makes stochastic,
deterministic, and wavelength-coupled fields interoperable through a
single interface: a deterministic field ignores ``rng``, a stochastic
field (e.g. Brownian motion) ignores ``pos``/``vel``, a
wavelength-independent field ignores ``wavelength`` — every field in this
module that doesn't explicitly couple to wavelength does exactly that, so
wavelength coupling is strictly opt-in. ``dt`` is passed through
explicitly (rather than baked into a field at construction time) because
stochastic fields need it to produce a correctly scaled discretization of
their underlying SDE — see ``brownian`` below.

Multiple fields compose by addition (``sum_fields``), since summing
prescribed velocities is exactly how independent flows superpose.

Most fields in this module are built from three independent pieces rather
than each being a one-off constructor — see the README's "Structured
fields: geometry × profile × gain" section for the design rationale. None
of the three is required to build a ``Field``; they're a convenient
factoring of the common case, not a shape every field must fit:

- A *geometry* (``radial_field``, ``tangential_field``, ``axial_field``)
  turns position into a scalar coordinate plus a direction vector.
- A *profile* (``constant``, ``linear``, ``exponential``, ``sinusoidal``)
  is a scalar-to-scalar shape function applied to that coordinate to get
  a magnitude.
- A *gain* (``Gain``) is an optional dimensionless multiplier, resolved
  from wavelength/time/randomness, that a profile folds into one of its
  own parameters — never into position, since a position-dependent
  multiplier would just duplicate what the geometry step already does.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Field = Callable[
    [np.ndarray, np.ndarray, np.ndarray, float, float, np.random.Generator],
    np.ndarray,
]
Profile = Callable[[np.ndarray, np.ndarray, float, float, np.random.Generator], np.ndarray]
Gain = Callable[[np.ndarray, float, np.random.Generator], "np.ndarray | float"]

_DEFAULT_SOFTENING = 1e-6
_MAX_EXPONENT = 0.5 * float(np.log(np.finfo(np.float32).max))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _as_velocity(direction: np.ndarray, magnitude: "np.ndarray | float", dtype: np.dtype) -> np.ndarray:
    """Combine a per-particle ``direction`` (n, dim) with a ``magnitude``
    that's either a bare scalar (a profile with no gain, or a gain that
    ignores wavelength — the common, cheap case) or a per-particle (n,)
    array, without forcing the scalar case through an (n,)-sized
    allocation. ``np.asarray(scalar)[..., None]`` is a 0-d-to-(1,) reshape
    (broadcasts against any ``direction`` for free); ``np.asarray((n,)
    array)[..., None]`` is the (n, 1) reshape needed to broadcast
    correctly against (n, dim).
    """
    return (direction * np.asarray(magnitude)[..., None]).astype(dtype)


# --- Profiles: scalar coordinate -> scalar magnitude -----------------------


def constant(value: float = -1.0, gain: Gain | None = None) -> Profile:
    """A profile that ignores its coordinate entirely: ``value`` (times
    ``gain``, if given). With a ``radial_field``, ``value = -1`` (the
    default) reproduces the old ``radial_inward``: constant inward speed
    regardless of distance from center.
    """
    if gain is None:
        def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
            return value

        return profile

    def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
        return value * gain(wavelength, t, rng)

    return profile


def linear(slope: float = 1.0, gain: Gain | None = None) -> Profile:
    """``slope * coordinate`` (times ``gain``, if given). With a
    ``tangential_field``, this is genuine rigid-body rotation: ``slope`` is
    the angular velocity, since constant angular velocity means speed
    grows linearly with distance from the rotation axis.
    """
    if gain is None:
        def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
            return slope * coordinate

        return profile

    def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
        return slope * coordinate * gain(wavelength, t, rng)

    return profile


def exponential(
    rate: float = 1.0,
    amplitude: float = -1.0,
    gain: Gain | None = None,
    max_exponent: float = _MAX_EXPONENT,
) -> Profile:
    """``amplitude * expm1(min(rate * coordinate, max_exponent))`` (times
    ``gain``, if given) — with a ``radial_field``, this reproduces the old
    ``exponential_confinement``: speed vanishes (``expm1(0) = 0``) at
    ``coordinate = 0`` and grows exponentially with distance. Since
    ``radial_field``'s direction is outward, ``amplitude`` defaults to
    negative (matching ``constant``'s default) so the profile pulls inward
    — confining — rather than pushing particles out; a positive
    ``amplitude`` gives exponential repulsion instead. A large
    ``dt`` combined with a coordinate far past ``1 / rate`` can otherwise
    produce a step large enough to overshoot before the exponential growth
    brakes it, sending the coordinate even farther out next step — a
    runaway that reaches ``inf`` in a handful of steps and ``nan`` shortly
    after. Clamping the exponent to ``max_exponent`` before ``expm1``
    bounds this profile's own output to a large-but-finite value instead;
    the default, ``ln(float32 max) / 2``, leaves headroom for the
    subsequent multiply by ``amplitude`` and by the geometry's direction
    vector. This doesn't prevent a large step from a *different* field or
    an oversized ``dt`` from still producing a bad step — ``dt`` modest
    relative to ``1 / (rate * amplitude)`` remains good practice.
    """
    if gain is None:
        def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
            return amplitude * np.expm1(np.minimum(rate * coordinate, max_exponent))

        return profile

    def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
        return amplitude * np.expm1(np.minimum(rate * coordinate, max_exponent)) * gain(wavelength, t, rng)

    return profile


def sinusoidal(frequency: float = 1.0, phase: float = 0.0, amplitude: float = 1.0, gain: Gain | None = None) -> Profile:
    """``amplitude * sin(2*pi*frequency*coordinate + phase)``, a single
    scalar wave along whatever coordinate the geometry provides — combined
    with ``axial_field``, this is a plane wave with wavevector ``axis``
    (see the README's "Structured fields" section for how this differs
    from the deleted lattice-forming sinusoidal field). Output is
    unconditionally bounded to ``[-amplitude, amplitude]`` regardless of
    how extreme ``frequency``, ``phase``, or ``gain`` get.

    Unlike the other profiles, ``gain`` here scales the *entire argument*
    to ``sin`` — frequency and phase together — rather than the output
    magnitude: that's the only way a modulator can shift *where* the wave's
    zeros land (e.g. a per-particle wavelength setting the lattice
    spacing), which a magnitude-only gain can't reproduce.
    """
    two_pi_f = 2.0 * np.pi * frequency
    if gain is None:
        def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
            return amplitude * np.sin(two_pi_f * coordinate + phase)

        return profile

    def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
        return amplitude * np.sin((two_pi_f * coordinate + phase) * gain(wavelength, t, rng))

    return profile


# --- Geometries: position -> (coordinate, direction), folded into a Field --


def radial_field(
    profile: Profile = constant(-1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    softening: float = _DEFAULT_SOFTENING,
) -> Field:
    """``direction(x) * profile(coordinate(x))`` where ``coordinate`` is
    the true (unsoftened) distance from ``center`` and ``direction`` is
    the outward unit-ish vector ``offset / sqrt(|offset|^2 +
    softening^2)``. ``profile`` never sees a singularity — ``constant``
    and ``exponential`` are already well-defined at ``coordinate = 0`` —
    so softening lives entirely here: it's ``direction`` that's ill-defined
    (``0/0``) at ``center`` without it, and the softened denominator makes
    ``direction`` (and so the whole field) smoothly vanish there instead,
    rather than picking an arbitrary direction. See the README for how
    this removes ``radial_inward``'s old center-overshoot artifact.
    """
    center_arr = np.asarray(center, dtype=np.float32)

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        offset = pos - center_arr
        r = np.linalg.norm(offset, axis=-1)
        r_safe = np.sqrt(r**2 + softening**2)
        direction = offset / r_safe[..., None]
        magnitude = profile(r, wavelength, t, dt, rng)
        return _as_velocity(direction, magnitude, pos.dtype)

    return field


def tangential_field(
    profile: Profile = linear(1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    plane_axes: tuple[Sequence[float], Sequence[float]] | None = None,
    softening: float = _DEFAULT_SOFTENING,
) -> Field:
    """Like ``radial_field``, but ``direction`` is tangential (the in-plane
    radial direction rotated 90 degrees) rather than radial, and
    ``coordinate`` is distance from ``center`` measured within the
    rotation plane only. The plane is spanned by two arbitrary orthonormal
    ``plane_axes`` (default: the first two coordinate axes, i.e. the
    orthographic view plane); components of ``offset`` outside that plane
    are left untouched, which is what keeps this dimension-agnostic rather
    than hard-coded to a 3D cross product. ``tangential_field(profile=
    linear(w))`` reproduces the old ``rotational(angular_velocity=w)``.

    No discretization correction is applied: explicit Euler integration of
    pure circular motion is unconditionally unstable and drifts outward
    over time. Deferred deliberately — see the README roadmap.
    """
    center_arr = np.asarray(center, dtype=np.float32)
    fixed_axes = None if plane_axes is None else tuple(_unit(np.asarray(a, dtype=np.float32)) for a in plane_axes)

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        if fixed_axes is None:
            dim = pos.shape[-1]
            u = np.zeros(dim, dtype=pos.dtype)
            u[0] = 1.0
            v = np.zeros(dim, dtype=pos.dtype)
            v[1] = 1.0
        else:
            u, v = fixed_axes
        offset = pos - center_arr
        a = offset @ u
        b = offset @ v
        r = np.sqrt(a**2 + b**2)
        r_safe = np.sqrt(a**2 + b**2 + softening**2)
        direction = (-b[..., None] * u + a[..., None] * v) / r_safe[..., None]
        magnitude = profile(r, wavelength, t, dt, rng)
        return _as_velocity(direction, magnitude, pos.dtype)

    return field


def axial_field(
    profile: Profile = linear(1.0),
    axis: Sequence[float] | None = None,
    direction: Sequence[float] | None = None,
    anchor: Sequence[float] | None = None,
    dim: int = 3,
    rng: np.random.Generator | None = None,
) -> Field:
    """``direction * profile(coordinate)`` where ``coordinate = dot(axis,
    x - anchor)`` and ``direction`` is a fixed unit vector — unlike the
    radial geometries, this direction doesn't depend on position at all,
    so there's no singularity to soften. ``axis`` and ``direction`` are
    independent: leaving ``direction`` unset aligns it to ``axis`` (the
    common case — e.g. ``axial_field(profile=sinusoidal(...))`` is a plane
    wave traveling along and oscillating along the same line), but setting
    them differently gives a shear flow — velocity pointing along one
    direction while varying with position along another. If ``axis`` isn't
    given, one is drawn from ``rng`` (a fresh unseeded generator if none is
    passed) as a reasonable default direction, using ``dim`` since a
    position array isn't available yet at construction time.
    """
    if axis is None:
        axis_arr = _unit((rng or np.random.default_rng()).standard_normal(dim).astype(np.float32))
    else:
        axis_arr = _unit(np.asarray(axis, dtype=np.float32))
    direction_arr = axis_arr if direction is None else _unit(np.asarray(direction, dtype=np.float32))
    anchor_arr = np.zeros(axis_arr.shape[0], dtype=np.float32) if anchor is None else np.asarray(anchor, dtype=np.float32)

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        coordinate = (pos - anchor_arr) @ axis_arr
        magnitude = profile(coordinate, wavelength, t, dt, rng)
        return _as_velocity(direction_arr, magnitude, pos.dtype)

    return field


# --- Standalone fields (no meaningful position dependence) -----------------


def constant_field(velocity: Sequence[float] | None = None, dim: int = 3, rng: np.random.Generator | None = None) -> Field:
    """A uniform drift: every particle gets the same fixed velocity every
    step, regardless of position. If ``velocity`` isn't given, one is
    drawn from ``rng`` (a fresh unseeded generator if none is passed).
    """
    if velocity is None:
        velocity_arr = (rng or np.random.default_rng()).standard_normal(dim).astype(np.float32)
    else:
        velocity_arr = np.asarray(velocity, dtype=np.float32)

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        return np.broadcast_to(velocity_arr, pos.shape).astype(pos.dtype)

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


# --- Composition -------------------------------------------------------


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


def modulated(field: Field, gain: Gain) -> Field:
    """Rescale an already-built field's total output velocity by ``gain``.
    The complement to giving a profile its own ``gain`` parameter: this
    applies from the outside, to a field whose internals you don't need to
    (or can't) reach into — e.g. an entire ``sum_fields(...)`` composition.
    Composes with ``sum_fields`` like any other field, which is how
    several differently-tuned modulations combine into one scene (e.g. a
    short-wavelength-favoring and a long-wavelength-favoring instance of
    the same base field, added together).
    """

    def coupled(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        base = field(pos, vel, wavelength, t, dt, rng)
        return _as_velocity(base, gain(wavelength, t, rng), pos.dtype)

    return coupled


def power_law_weight(reference_nm: float = 530.0, exponent: float = -1.0) -> Gain:
    """``(wavelength / reference_nm) ** exponent`` — a ``Gain``: it
    evaluates to exactly ``1`` at ``reference_nm``, which is what makes it
    usable as a profile's ``gain`` parameter or with ``modulated()``
    interchangeably, without either needing to know its scale.

    ``exponent = -1`` is physically grounded: photon momentum ``p = h/λ``
    is inversely proportional to wavelength, so if the modulated field
    represents radiation-pressure-like forcing, shorter wavelengths
    physically do push harder. ``exponent = +1`` favors long wavelengths
    instead — not backed by the same fundamental law, but a principled and
    equally tunable choice on its own terms (e.g. as a diffraction-flavored
    metaphor: diffraction angle scales with wavelength, so longer
    wavelengths could be read as coupling more strongly to a field's
    spatial structure). ``exponent = 0`` recovers a gain of ``1``
    everywhere, i.e. no modulation.
    """

    def gain(wavelength: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
        return (wavelength / reference_nm) ** exponent

    return gain
