"""Velocity fields.

A field is a plain callable ``field(pos, vel, wavelength, t, dt, rng) ->
velocity``, where the returned array has the same shape as ``pos``. Fields
are *kinematic*: they prescribe the particle velocity directly, like a
fluid flow field advecting passive tracers, rather than a force that
accumulates into velocity over time. This is what makes stochastic,
deterministic, and wavelength-coupled fields interoperable through a
single interface: a deterministic field ignores ``rng``, a stochastic
field (e.g. white noise) ignores ``pos``/``vel``, a
wavelength-independent field ignores ``wavelength`` — every field in this
module that doesn't explicitly couple to wavelength does exactly that, so
wavelength coupling is strictly opt-in. ``dt`` is passed through
explicitly (rather than baked into a field at construction time) because
stochastic fields need it to produce a correctly scaled discretization of
their underlying SDE — see ``white_noise_field`` below.

Multiple fields compose by addition (``sum_fields``), since summing
prescribed velocities is exactly how independent flows superpose.

Most fields in this module are built from three independent pieces rather
than each being a one-off constructor — see the README's "Structured
fields: geometry × profile × gain" section for the design rationale. None
of the three is required to build a ``Field``; they're a convenient
factoring of the common case, not a shape every field must fit:

- A *geometry* (``radial_field``, ``tangential_field``, ``axial_field``,
  ``twist_field``) turns position into a scalar coordinate plus a
  direction vector.
- A *profile* (``constant``, ``linear``, ``exponential``,
  ``exponential_ramp``, ``sinusoidal``) is a scalar-to-scalar shape
  function applied to that coordinate to get a magnitude.
- A *gain* (``Gain``, see ``gains.py``) is an optional dimensionless
  multiplier, resolved from wavelength/time/randomness, that a profile
  folds into one of its own parameters — never into position, since a
  position-dependent multiplier would just duplicate what the geometry
  step already does. The gain catalog and composition rules live in
  ``gains.py``, not here, since it's a large enough vocabulary (spectral,
  deterministic-temporal, stochastic, and stateful gains) to warrant its
  own module.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .gains import Gain

Field = Callable[
    [np.ndarray, np.ndarray, np.ndarray, float, float, np.random.Generator],
    np.ndarray,
]
Profile = Callable[[np.ndarray, np.ndarray, float, float, np.random.Generator], np.ndarray]

_DEFAULT_SOFTENING = 1e-6
_MAX_EXPONENT = 0.5 * float(np.log(np.finfo(np.float32).max))
_TWO_PI = 2.0 * np.pi


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


def constant(value: float = 1.0, gain: Gain | None = None) -> Profile:
    """A profile that ignores its coordinate entirely: ``value`` (times
    ``gain``, if given). ``value`` defaults to ``1`` — this profile
    unscaled — with no bias toward either sign: a radially-symmetric field
    is, in general, ``f(r) * direction`` for *any* signed ``f`` (gravity is
    ``f(r) < 0``, Coulomb repulsion between like charges is ``f(r) > 0``),
    so a negative ``value`` isn't a special case here, just the other half
    of the ordinary range. With a ``radial_field`` (whose ``direction`` is
    outward), a negative ``value`` pulls inward and a positive one pushes
    outward.
    """
    if gain is None:
        def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
            return value

        return profile

    def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
        return value * gain(wavelength, t, dt, rng)

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
        return slope * coordinate * gain(wavelength, t, dt, rng)

    return profile


def exponential(
    rate: float = 1.0,
    amplitude: float = 1.0,
    gain: Gain | None = None,
    max_exponent: float = _MAX_EXPONENT,
) -> Profile:
    """``amplitude * exp(min(rate * coordinate, max_exponent))`` (times
    ``gain``, if given): plain exponential growth, equal to ``amplitude``
    at ``coordinate = 0`` and growing (or, for negative ``rate``, decaying)
    exponentially from there — never zero, unlike ``exponential_ramp``
    below. ``amplitude`` defaults to ``1`` — this profile unscaled, same
    convention as ``constant`` — with no bias toward either sign: with a
    ``radial_field`` (whose ``direction`` is outward), a negative
    ``amplitude`` points inward and a positive one outward; see
    ``constant`` for why neither sign is a special case.

    A large ``dt`` combined with a coordinate far past ``1 / rate`` can
    otherwise produce a step large enough to overshoot before the
    exponential growth brakes it, sending the coordinate even farther out
    next step — a runaway that reaches ``inf`` in a handful of steps and
    ``nan`` shortly after. Clamping the exponent to ``max_exponent``
    bounds this profile's own output to a large-but-finite value instead;
    the default, ``ln(float32 max) / 2``, leaves headroom for the
    subsequent multiply by ``amplitude`` and by the geometry's direction
    vector. This doesn't prevent a large step from a *different* field or
    an oversized ``dt`` from still producing a bad step — ``dt`` modest
    relative to ``1 / (rate * amplitude)`` remains good practice.
    """
    if gain is None:
        def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
            return amplitude * np.exp(np.minimum(rate * coordinate, max_exponent))

        return profile

    def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
        return amplitude * np.exp(np.minimum(rate * coordinate, max_exponent)) * gain(wavelength, t, dt, rng)

    return profile


def exponential_ramp(
    rate: float = 1.0,
    amplitude: float = 1.0,
    gain: Gain | None = None,
    max_exponent: float = _MAX_EXPONENT,
) -> Profile:
    """``amplitude * expm1(min(rate * coordinate, max_exponent))`` (times
    ``gain``, if given): like ``exponential``, but shifted down by
    ``amplitude`` so it vanishes (``expm1(0) = 0``) at ``coordinate = 0``
    instead of starting from ``amplitude`` — the exponential analogue of
    ``linear``, which also passes through the origin, rather than of
    ``constant``. That matters for a confining ``radial_field``: with
    plain ``exponential``, a particle sitting exactly at ``center`` still
    gets pulled at full ``amplitude``, which is a discontinuity-flavored
    surprise for a field meant to gently confine around that point;
    ``exponential_ramp`` instead ramps up from zero the farther out a
    particle sits, only reaching exponential strength once ``coordinate``
    is well past ``1 / rate``. ``amplitude``'s sign convention (inward vs.
    outward on a ``radial_field``) matches ``exponential``; see
    ``constant`` for why neither sign is a special case.

    Shares the ``max_exponent`` overflow clamp and its rationale with
    ``exponential`` — see that profile's docstring: a large ``dt`` past
    ``1 / rate`` can otherwise produce a runaway step that reaches ``inf``
    within a handful of steps.
    """
    if gain is None:
        def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
            return amplitude * np.expm1(np.minimum(rate * coordinate, max_exponent))

        return profile

    def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
        return amplitude * np.expm1(np.minimum(rate * coordinate, max_exponent)) * gain(wavelength, t, dt, rng)

    return profile


def sinusoidal(
    frequency: float = 1.0,
    phase: float = 0.0,
    amplitude: float = 1.0,
    amplitude_gain: Gain | None = None,
    frequency_gain: Gain | None = None,
    phase_gain: Gain | None = None,
) -> Profile:
    """``amplitude * sin(2*pi*(frequency*coordinate + phase))``, a single
    scalar wave along whatever coordinate the geometry provides — combined
    with ``axial_field``, this is a plane wave with wavevector ``axis``.
    Output is unconditionally bounded to ``[-amplitude, amplitude]``
    regardless of how extreme ``frequency``, ``phase``, or any gain get.

    Both ``frequency`` and ``phase`` are in *cycles*, not radians:
    ``frequency`` is periods per unit coordinate, and ``phase`` is a
    fractional offset of one period (``phase=0.25`` shifts the wave a
    quarter-turn), so the two combine by plain addition
    (``frequency*coordinate + phase``) before the one conversion to
    radians that ``sin`` needs. This is deliberate, not incidental:
    radians are canonical for the trig primitive computing the wave, but
    cycles are canonical for the periodic *quantity* being modulated, and
    modulating with a ``Gain`` (below) requires that quantity, not the
    primitive's units — a phase modulator should scale "how much of a
    period," which only means what it says if ``phase`` is already
    expressed that way.

    Unlike the other profiles, ``sinusoidal`` has three independent scalar
    knobs worth modulating rather than one, so each gets its own named
    ``Gain`` hook instead of sharing a single ambiguous ``gain``:
    ``amplitude_gain`` scales the output, same as every other profile's
    ``gain``; ``frequency_gain`` and ``phase_gain`` scale ``frequency`` and
    ``phase`` respectively, *before* they enter ``sin`` — the only way a
    modulator can shift *where* the wave's zeros land (e.g. a per-particle
    wavelength setting the lattice spacing), which scaling the output can't
    reproduce. Because both are cycles-valued, the two hooks are on equal
    footing: passing the *same* ``Gain`` to both scales frequency and phase
    together, as one wavelength-dependent factor; passing it to only one
    modulates that one alone.
    """
    omega = _TWO_PI * frequency
    phase_rad = _TWO_PI * phase
    if amplitude_gain is None and frequency_gain is None and phase_gain is None:
        def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
            return amplitude * np.sin(omega * coordinate + phase_rad)

        return profile

    def profile(coordinate: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator):
        freq = frequency if frequency_gain is None else frequency * frequency_gain(wavelength, t, dt, rng)
        ph = phase if phase_gain is None else phase * phase_gain(wavelength, t, dt, rng)
        out = amplitude * np.sin(_TWO_PI * (freq * coordinate + ph))
        return out if amplitude_gain is None else out * amplitude_gain(wavelength, t, dt, rng)

    return profile


# --- Geometries: position -> (coordinate, direction), folded into a Field --


def radial_field(
    profile: Profile = constant(),
    center: Sequence[float] | None = None,
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
    rather than picking an arbitrary direction.

    ``center`` defaults to the origin in whatever dimension ``pos`` turns
    out to be, resolved per call rather than baked in as a fixed-length
    vector at construction time — that's what keeps this dimension-agnostic
    (usable at any ``dim >= 2``, not just R^3) without a ``dim`` parameter.

    The bare default, ``profile=constant()``, is the ``direction`` field
    itself unscaled — pure unit-speed outward flow — since that's the
    neutral composition (profile identity, no sign bias) rather than a
    choice tuned to any particular intended use. For inward/confining
    behavior, negate the profile's scale explicitly, e.g.
    ``radial_field(profile=constant(-1.0))`` or
    ``radial_field(profile=exponential_ramp(amplitude=-1.0))`` — see those
    profiles' docstrings for why the sign isn't a special case.
    """
    center_arr = None if center is None else np.asarray(center, dtype=np.float32)

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        offset = pos if center_arr is None else pos - center_arr
        r = np.linalg.norm(offset, axis=-1)
        r_safe = np.sqrt(r**2 + softening**2)
        direction = offset / r_safe[..., None]
        magnitude = profile(r, wavelength, t, dt, rng)
        return _as_velocity(direction, magnitude, pos.dtype)

    return field


def tangential_field(
    profile: Profile = linear(1.0),
    center: Sequence[float] | None = None,
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
    than hard-coded to a 3D cross product. ``center`` defaults to the
    origin in whatever dimension ``pos`` turns out to be, resolved per
    call rather than a fixed-length vector, same reasoning as
    ``radial_field``. ``tangential_field(profile=linear(w))`` is
    rigid-body rotation at angular velocity ``w``.

    No discretization correction is applied: explicit Euler integration of
    pure circular motion is unconditionally unstable and drifts outward
    over time. Deferred deliberately — see the README roadmap.
    """
    center_arr = None if center is None else np.asarray(center, dtype=np.float32)
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
        offset = pos if center_arr is None else pos - center_arr
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
    *,
    axis: Sequence[float],
    direction: Sequence[float] | None = None,
    center: Sequence[float] | None = None,
) -> Field:
    """``direction * profile(coordinate)`` where ``coordinate = dot(axis,
    x - center)`` and ``direction`` is a fixed unit vector — unlike the
    radial geometries, this direction doesn't depend on position at all,
    so there's no singularity to soften. ``axis`` and ``direction`` are
    independent: leaving ``direction`` unset aligns it to ``axis`` (the
    common case — e.g. ``axial_field(profile=sinusoidal(...))`` is a plane
    wave traveling along and oscillating along the same line), but setting
    them differently gives a shear flow — velocity pointing along one
    direction while varying with position along another. ``center`` names
    the same reference-point role ``radial_field``/``tangential_field``
    give that name to — here it's the point the coordinate plane passes
    through, rather than a point of rotational symmetry, but it's the same
    kind of knob: where ``coordinate = 0`` is.

    ``axis`` is a required keyword-only argument: unlike
    ``radial_field``/``tangential_field``'s ``center``, there's no
    dimension-agnostic default direction to fall back to (a "reasonable
    random default" needs to know how many components to draw, which
    needs a dimensionality this constructor
    otherwise has no reason to know — fields aren't meant to carry that
    context themselves; the running ``Simulation`` already does, via
    ``sim.state.dim``). Use ``sim.random_direction()`` for a random unit
    vector sized to the simulation's current dimensionality, e.g.
    ``fields.axial_field(profile=fields.sinusoidal(), axis=sim.random_direction())``.
    """
    axis_arr = _unit(np.asarray(axis, dtype=np.float32))
    direction_arr = axis_arr if direction is None else _unit(np.asarray(direction, dtype=np.float32))
    center_arr = np.zeros(axis_arr.shape[0], dtype=np.float32) if center is None else np.asarray(center, dtype=np.float32)

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        coordinate = (pos - center_arr) @ axis_arr
        magnitude = profile(coordinate, wavelength, t, dt, rng)
        return _as_velocity(direction_arr, magnitude, pos.dtype)

    return field


def twist_field(
    axis: Sequence[float],
    plane_axes: tuple[Sequence[float], Sequence[float]] | None = None,
    angle: Profile = linear(1.0),
    magnitude: Profile = constant(1.0),
    center: Sequence[float] | None = None,
) -> Field:
    """Direction rotates within a fixed plane as a function of position
    along ``axis``, while staying constant across the whole plane spanned
    by ``plane_axes`` — velocity has no dependence on in-plane position at
    all, unlike ``tangential_field``, where direction depends on where you
    sit *within* the rotation plane. ``direction(coordinate) =
    cos(theta)*u + sin(theta)*v`` where ``theta = 2*pi*angle(coordinate)``
    and ``coordinate = dot(offset, axis)`` — exactly ``axial_field``'s
    coordinate. No singularity to soften: unlike the radial geometries,
    direction never depends on ``offset`` within the plane, so there's no
    ``0/0`` at any point.

    This is the cholesteric liquid-crystal director field / the spatial
    snapshot of a circularly-or-elliptically-polarized plane wave: freeze
    a circularly polarized EM wave at one instant and its field vector
    traces exactly this helix along the propagation axis, with ``axis`` as
    the propagation direction, ``angle``'s rate as the wavenumber, and
    each wavelength free to have its own twist rate — chromatic optical
    activity / circular birefringence is a real, wavelength-dependent
    effect, achieved here by giving ``angle`` its own ``gain`` (e.g.
    ``linear(rate, gain=wavelength_power_law())``), not an invented one.

    Reuses ``Profile`` for two independent scalar roles instead of one:
    ``angle`` (interpreted as *cycles*, exactly like ``sinusoidal``'s
    ``frequency``/``phase``, converted to radians once here) sets the
    rotation rate — ``angle=linear(rate)`` (the default, ``rate=1``) is
    the canonical constant-pitch helix, one full twist per unit
    ``coordinate``; a nonlinear ``angle`` (e.g. ``sinusoidal(...)``) gives
    an accelerating or oscillating twist instead of a fixed pitch, a
    principled but non-physical extension of the base case above.
    ``magnitude`` (interpreted as an ordinary magnitude, like every other
    geometry's ``profile``) is the amplitude envelope along ``axis`` —
    ``constant()`` (the default) is a uniform helix;
    ``exponential_ramp(amplitude=-1.0)`` would give one that decays away
    from ``center``. Both accept the full ``Profile``/``Gain`` machinery
    independently, including wavelength coupling through either.

    ``axis`` has no dimension-agnostic default, for the same reason
    ``axial_field``'s doesn't (see its docstring): in 3D the orthogonal
    complement of a 2-plane is a unique line, but for ``dim > 3`` it's
    ``(dim - 2)``-dimensional, so there's no canonical "the other axis" to
    fall back to past 3D. ``plane_axes`` defaults to the first two
    coordinate axes (dimension-agnostic, like ``tangential_field``'s
    default), resolved from ``axis``'s own length at construction time
    since ``axis`` already fixes the dimension. Correctness of a custom
    ``axis``/``plane_axes`` pairing — they should be mutually orthogonal,
    or ``coordinate`` and in-plane position stop being independent and
    "constant across the plane" no longer holds — is the caller's
    responsibility, the same convention as ``tangential_field``'s custom
    ``plane_axes`` and ``axial_field``'s ``axis``/``direction``. Requires
    ``dim >= 3`` in practice (one dimension for ``axis``, two more for the
    plane); nothing here checks that explicitly, consistent with the rest
    of this module.
    """
    axis_arr = _unit(np.asarray(axis, dtype=np.float32))
    dim = axis_arr.shape[0]
    if plane_axes is None:
        u = np.zeros(dim, dtype=np.float32)
        u[0] = 1.0
        v = np.zeros(dim, dtype=np.float32)
        v[1] = 1.0
    else:
        u, v = (_unit(np.asarray(a, dtype=np.float32)) for a in plane_axes)
    center_arr = np.zeros(dim, dtype=np.float32) if center is None else np.asarray(center, dtype=np.float32)

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        coordinate = (pos - center_arr) @ axis_arr
        theta = _TWO_PI * angle(coordinate, wavelength, t, dt, rng)
        direction = np.cos(theta)[..., None] * u + np.sin(theta)[..., None] * v
        mag = magnitude(coordinate, wavelength, t, dt, rng)
        return _as_velocity(direction, mag, pos.dtype)

    return field


# --- Standalone fields (no meaningful position dependence) -----------------


def constant_field(velocity: Sequence[float]) -> Field:
    """A uniform drift: every particle gets the same fixed ``velocity``
    every step, regardless of position. ``velocity`` is required for the
    same reason ``axial_field``'s ``axis`` is (see its docstring) — no
    dimensionality to draw a default from here; use
    ``sim.random_direction()`` for a random one sized to the simulation's
    current dimensionality, e.g. ``fields.constant_field(sim.random_direction())``.
    """
    velocity_arr = np.asarray(velocity, dtype=np.float32)

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        return np.broadcast_to(velocity_arr, pos.shape).astype(pos.dtype)

    return field


def white_noise_field(sigma: float = 1.0, gain: Gain | None = None) -> Field:
    """Velocity drawn fresh each step as i.i.d. Gaussian noise: uncorrelated
    in time, so this is white noise at the velocity level. Integrating a
    white-noise velocity produces Brownian motion in position — an
    Euler-Maruyama discretization of the Wiener process, whose position
    increment is ``dx = sigma * sqrt(dt) * randn()`` per step. To fit the
    velocity-field interface (where the integrator computes ``x += v *
    dt``), the field must return ``v = sigma / sqrt(dt) * randn()`` so that
    ``v * dt`` recovers the correct increment. This velocity grows without
    bound as ``dt -> 0``, which is expected: it reflects the
    non-differentiability of Brownian paths, not a bug.

    ``sigma`` is this field's one scalar knob, so ``gain`` (if given) scales
    it exactly like ``constant``/``linear``/``exponential``'s ``gain``
    scales theirs — e.g. ``gain=wavelength_power_law(exponent=-1)`` makes
    shorter wavelengths diffuse faster, matching the photon-momentum
    framing used elsewhere in the gain catalog.
    """

    def field(
        pos: np.ndarray, vel: np.ndarray, wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        magnitude = sigma / np.sqrt(dt) if gain is None else sigma / np.sqrt(dt) * gain(wavelength, t, dt, rng)
        return _as_velocity(rng.standard_normal(pos.shape), magnitude, pos.dtype)

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
        return _as_velocity(base, gain(wavelength, t, dt, rng), pos.dtype)

    return coupled
