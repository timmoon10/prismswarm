"""Gains: dimensionless multipliers a `fields.Profile` can fold into one of
its own scalar parameters (or that `fields.modulated()` can apply to an
already-built `Field`'s total output), resolved from wavelength, time, and
randomness rather than position — a position-dependent multiplier would
just duplicate what a `fields.py` geometry's coordinate step already does.

``Gain = Callable[[wavelength, t, dt, rng], array | float]``. ``dt`` is
carried for the same reason `fields.Field` carries it: a stateful gain
(`ornstein_uhlenbeck`, `telegraph`) needs it to Euler-Maruyama-discretize
its underlying SDE correctly, exactly like `fields.brownian` does — a
stateless gain just ignores it, the same "ignore what you don't need"
convention every field/profile already follows.

Every gain here defaults to its own mathematically canonical form —
`sine_gain`/`square_gain` zero-centered, `ornstein_uhlenbeck`/`telegraph`
resting at/switching around `0`, `wavelength_gaussian` peaking at
`amplitude` and decaying to `0` — rather than one pre-tuned to "neutral at
1", which is a property of *using* a gain multiplicatively, not a property
of the shape itself. Pass `center=1.0` (or `mu=1.0`, or `low`/`high`
straddling `1`) explicitly to get that. `lognormal_noise` is the one
exception: median-1 isn't a tuned default there, it's a structural
consequence of exponentiating a zero-mean Gaussian, so it needs no
`center` parameter at all. `wavelength_power_law` also evaluates to
exactly `1` at its reference wavelength by construction, for the same
reason — see its docstring.

Multiple gains combine via `gain_product`, multiplying their outputs —
the natural composition rule for dimensionless multipliers (matching
Beer-Lambert absorption, where stacked absorbers multiply transmittances),
and how a profile's single gain slot can be driven by more than one
independent effect at once (e.g. `gain_product(wavelength_power_law(),
sine_gain(frequency=0.5, center=1.0))` for a field that's both
wavelength- and time-modulated).

Stateful gains (`ornstein_uhlenbeck`, `telegraph`) hold their state in a
closure, which only works correctly if each call corresponds to a distinct
forward step in time. That holds by default — every `Field`/`Profile` in
`fields.py` calls its children exactly once per `Simulation.step()` — with
one documented exception: `fields.sinusoidal` explicitly supports passing
the *same* `Gain` instance to both `frequency_gain` and `phase_gain`,
which calls that one instance twice within a single step. `_stateful`
handles this by caching on `t` (see its docstring) rather than assuming
one call per step, so stateful gains are safe under that reuse too.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

Gain = Callable[[np.ndarray, float, float, np.random.Generator], "np.ndarray | float"]


# --- Statefulness helper -----------------------------------------------


def _stateful(
    step: Callable[[Any, float, float, np.random.Generator], tuple[Any, "np.ndarray | float"]],
    init: Any,
) -> Gain:
    """Wrap a ``step(state, t, dt, rng) -> (new_state, value)`` function
    into a ``Gain``. Caches the last ``t`` it was called with and only
    invokes ``step`` when ``t`` has changed, replaying the cached ``value``
    on a repeat call at the same ``t`` instead of advancing again — see the
    module docstring for why that matters (the same instance can
    legitimately be called more than once per simulation step). Comparing
    ``t`` by exact equality is safe here: within one `Simulation.step()`,
    the same `float` is threaded unmodified through the whole
    field/profile/gain call tree, so no floating-point drift can occur
    between the two calls being deduplicated.
    """
    cache: dict[str, Any] = {"t": None, "state": init, "value": None}

    def gain(wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator) -> "np.ndarray | float":
        if t != cache["t"]:
            cache["state"], cache["value"] = step(cache["state"], t, dt, rng)
            cache["t"] = t
        return cache["value"]

    return gain


# --- Spectral: wavelength only -------------------------------------------


def wavelength_power_law(reference_nm: float = 530.0, exponent: float = -1.0) -> Gain:
    """``(wavelength / reference_nm) ** exponent`` — evaluates to exactly
    ``1`` at ``reference_nm``, which is what makes it pluggable into any
    profile's gain hook (or into ``fields.modulated()``) without that hook
    needing to know its scale.

    ``exponent = -1`` is physically grounded: photon momentum ``p = h/λ``
    is inversely proportional to wavelength, so if the modulated field
    represents radiation-pressure-like forcing, shorter wavelengths
    physically do push harder. ``exponent = +1`` favors long wavelengths
    instead — not backed by the same fundamental law, but a principled and
    equally tunable choice on its own terms (e.g. a diffraction-flavored
    metaphor: diffraction angle scales with wavelength). ``exponent = 0``
    recovers a gain of ``1`` everywhere, i.e. no modulation.
    """

    def gain(wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator) -> np.ndarray:
        return (wavelength / reference_nm) ** exponent

    return gain


def wavelength_gaussian(reference_nm: float = 530.0, sigma_nm: float = 50.0, amplitude: float = 1.0) -> Gain:
    """``amplitude * exp(-0.5 * ((wavelength - reference_nm) / sigma_nm) **
    2)`` — a resonance/bandpass gain, peaked at ``reference_nm`` and
    decaying to ``0`` away from it: the standard lineshape for a single
    absorption/emission resonance, the same mechanism that gives colored
    glass its color (a dopant ion's electronic transition).

    Unlike ``wavelength_power_law``, there's no wavelength at which this
    evaluates to a fixed ``1`` independent of parameters — "no coupling far
    from resonance" (``0``) is this shape's natural neutral state, not "no
    modulation" (``1``), so ``amplitude`` directly sets the peak gain
    rather than a departure from an identity value.
    """

    def gain(wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator) -> np.ndarray:
        return amplitude * np.exp(-0.5 * ((wavelength - reference_nm) / sigma_nm) ** 2)

    return gain


# --- Temporal, deterministic: t only --------------------------------------


def sine_gain(frequency: float, amplitude: float = 1.0, phase: float = 0.0, center: float = 0.0) -> Gain:
    """``center + amplitude * sin(2*pi*frequency*t + phase)``. Defaults to
    the canonical sine wave — zero-centered, unit amplitude — rather than
    one pre-tuned to a gain's "neutral at 1" convention; pass ``center=1.0``
    explicitly to idle at 1 and swing symmetrically around it.
    """
    two_pi_f = 2.0 * np.pi * frequency

    def gain(wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator) -> float:
        return center + amplitude * np.sin(two_pi_f * t + phase)

    return gain


def square_gain(frequency: float, amplitude: float = 1.0, phase: float = 0.0, center: float = 0.0) -> Gain:
    """``center +/- amplitude``, switching at ``sine_gain``'s zero
    crossings. Defaults to the canonical +/-1 square wave (``center=0``,
    ``amplitude=1``) — zero-mean, matching ``sine_gain``'s convention —
    rather than the 0/1 rectified form; pass ``center=amplitude=0.5`` for
    that instead.
    """
    two_pi_f = 2.0 * np.pi * frequency

    def gain(wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator) -> float:
        sign = 1.0 if np.sin(two_pi_f * t + phase) >= 0.0 else -1.0
        return center + amplitude * sign

    return gain


# --- Stochastic, memoryless: rng only -------------------------------------


def gaussian_noise(sigma: float = 1.0, center: float = 0.0) -> Gain:
    """``center + sigma * randn()`` — i.i.d. per call, no memory.
    Zero-centered by default, like ``sine_gain``/``square_gain``; pass
    ``center=1.0`` for gain use. Note that for ``sigma`` large relative to
    ``center`` the result can go negative, flipping the sign of whatever it
    multiplies — ``lognormal_noise`` is the safer default for a strictly
    positive multiplicative gain; use this one when sign flips are a
    deliberate effect.
    """

    def gain(wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator) -> float:
        return center + sigma * rng.standard_normal()

    return gain


def lognormal_noise(sigma: float = 0.1) -> Gain:
    """``exp(sigma * randn())`` — i.i.d. per call, always positive, median
    exactly ``1`` regardless of ``sigma`` (mean is ``exp(sigma**2 / 2)``,
    slightly above 1). The multiplicative analogue of ``gaussian_noise``:
    a factor of ``x`` is as likely as a factor of ``1/x`` at any ``sigma``,
    and it can never flip the sign of whatever it multiplies — the
    canonical choice for a "neutral at 1" stochastic gain, the same way
    geometric Brownian motion is the multiplicative analogue of ordinary
    Brownian motion.
    """

    def gain(wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator) -> float:
        return np.exp(sigma * rng.standard_normal())

    return gain


# --- Stateful: hold memory across calls -----------------------------------


def ornstein_uhlenbeck(theta: float, sigma: float, mu: float = 0.0) -> Gain:
    """The canonical continuous-time mean-reverting stochastic process —
    the stochastic generalization of ``sine_gain``'s oscillation, and in
    gain-space the same shape as an ``exponential_confinement`` field plus
    ``brownian`` noise. Euler-Maruyama discretized exactly like
    ``fields.brownian``: ``x += -theta*(x - mu)*dt + sigma*sqrt(dt) *
    randn()`` — the ``sqrt(dt)`` diffusion term is what makes this converge
    to the right SDE as ``dt -> 0``, not a plain ``randn()`` scaled by
    ``dt``.

    Rests at ``mu=0`` by default — the process's own natural resting
    point, not backward-engineered from gain usage — so pass ``mu=1.0`` to
    idle at a gain's neutral value. Starts at ``x = mu``.

    Stateful (see the module docstring and ``_stateful``): safe to call
    more than once at the same ``t``, but don't share one instance across
    two fields stepped at different rates — a call at a new ``t`` always
    advances the process by exactly one step.
    """

    def step(x: float, t: float, dt: float, rng: np.random.Generator) -> tuple[float, float]:
        x_new = x - theta * (x - mu) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        return x_new, x_new

    return _stateful(step, init=mu)


def telegraph(rate: float, low: float = -1.0, high: float = 1.0) -> Gain:
    """The stochastic generalization of ``square_gain``: switches between
    ``low`` and ``high`` at Poisson-process arrival times (rate ``rate``,
    i.e. mean dwell time ``1/rate``) instead of a fixed period — the random
    telegraph process / dichotomous Markov noise used to model e.g. ion
    channel gating. Zero-centered +/-1 by default, matching
    ``square_gain``'s convention; pass ``low=0.0, high=1.0`` (or
    ``low=0.5, high=1.5``) for gain use.

    Stateful like ``ornstein_uhlenbeck`` — same same-``t`` caching, same
    caveat about sharing one instance across differently-stepped fields.
    Uses a ``while`` loop rather than a single check so a ``dt`` long
    enough to span multiple switches (large relative to ``1/rate``) still
    lands on the correct number of flips.
    """

    def step(
        state: tuple[float, float | None], t: float, dt: float, rng: np.random.Generator
    ) -> tuple[tuple[float, float], float]:
        value, next_switch = state
        if next_switch is None:
            next_switch = t + rng.exponential(1.0 / rate)
        while t >= next_switch:
            value = low if value == high else high
            next_switch += rng.exponential(1.0 / rate)
        return (value, next_switch), value

    return _stateful(step, init=(low, None))


# --- Composition -----------------------------------------------------------


def gain_product(*gains: Gain) -> Gain:
    """Combine multiple gains into one by multiplying their outputs — the
    natural composition rule for dimensionless multipliers (matching
    Beer-Lambert absorption, where stacked absorbers multiply
    transmittances), and how a profile's single gain slot can be driven by
    more than one independent effect at once, e.g.
    ``gain_product(wavelength_power_law(), sine_gain(frequency=0.5,
    center=1.0))`` for a field that's both wavelength- and time-modulated.
    A gain of ``1`` (the identity) is a no-op under this product, matching
    every gain's own "neutral at 1" convention when centered accordingly.
    """

    def gain(wavelength: np.ndarray, t: float, dt: float, rng: np.random.Generator) -> "np.ndarray | float":
        result: "np.ndarray | float" = 1.0
        for g in gains:
            result = result * g(wavelength, t, dt, rng)
        return result

    return gain
