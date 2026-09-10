"""Emission spectra: how a particle's wavelength is chosen at initialization.

A spectrum is a callable ``spectrum(n, rng) -> wavelengths_nm``, mirroring
the shape of the ``Field`` interface in ``fields.py`` — a plain function
that scene setup code swaps in, rather than a class hierarchy. Spectra
currently only govern initialization; per-particle wavelength dynamics
(random walks, explicit spectral conversion) are deferred.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Spectrum = Callable[[int, np.random.Generator], np.ndarray]

_H = 6.62607015e-34  # Planck constant, J s
_C = 2.99792458e8  # speed of light, m/s
_KB = 1.380649e-23  # Boltzmann constant, J/K


def monochrome(wavelength_nm: float = 530.0) -> Spectrum:
    """Every particle emits the same pure wavelength."""

    def spectrum(n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(n, wavelength_nm, dtype=np.float64)

    return spectrum


def blackbody(
    temperature_k: float = 5778.0,
    lo_nm: float = 380.0,
    hi_nm: float = 780.0,
    table_size: int = 2048,
) -> Spectrum:
    """Particles sample wavelengths from Planck's law at ``temperature_k``,
    restricted to ``[lo_nm, hi_nm]`` — defaults to the visible range and
    the sun's ~5778K effective temperature (whose Planck peak, by Wien's
    law, falls at ~502nm — green — well inside that range).

    Sampling is inverse-transform: the Planck spectral radiance is
    evaluated once over a fine wavelength grid, normalized into a CDF, and
    inverted by interpolation against uniform draws. The table is built
    once per call to this factory, not per particle or per frame.
    """
    wavelengths_m = np.linspace(lo_nm, hi_nm, table_size) * 1e-9
    # Planck's law spectral radiance by wavelength, up to a constant factor
    # that cancels once normalized into a probability distribution.
    exponent = (_H * _C) / (wavelengths_m * _KB * temperature_k)
    radiance = (1.0 / wavelengths_m**5) / np.expm1(exponent)
    cdf = np.cumsum(radiance)
    cdf /= cdf[-1]
    wavelengths_nm = wavelengths_m * 1e9

    def spectrum(n: int, rng: np.random.Generator) -> np.ndarray:
        return np.interp(rng.random(n), cdf, wavelengths_nm)

    return spectrum
