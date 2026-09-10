"""Wavelength -> XYZ -> sRGB color pipeline.

Wavelength-to-XYZ uses the Wyman/Sloan/Shirley multi-Gaussian analytic fit
to the CIE 1931 standard observer color-matching functions [1], evaluated
directly rather than interpolated from a tabulated CMF, so it's cheap,
vectorized, and needs no bundled data file. These are the same detector
sensitivities behind human color vision (and, by design, behind calibrated
camera sensors): a pure 530nm green legitimately registers on the x-bar
curve, which is why it reads as slightly yellow-green rather than pure
green — that's not a mistake to correct, it's the physically correct
outcome of expressing wavelength in a 3-primary color space.

[1] Wyman, Sloan, Shirley, "Simple Analytic Approximations to the CIE XYZ
    Color Matching Functions", JCGT 2013.
"""

from __future__ import annotations

import numpy as np

# Rows: (amplitude, mu, sigma_lo, sigma_hi), summed per channel.
_X_TERMS = ((1.056, 599.8, 37.9, 31.0), (0.362, 442.0, 16.0, 26.7), (-0.065, 501.1, 20.4, 26.2))
_Y_TERMS = ((0.821, 568.8, 46.9, 40.5), (0.286, 530.9, 16.3, 31.1))
_Z_TERMS = ((1.217, 437.0, 11.8, 36.0), (0.681, 459.0, 26.0, 13.8))

# CIE XYZ (D65) -> linear sRGB.
_XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)


def _asymmetric_gaussian(wavelength: np.ndarray, mu: float, sigma_lo: float, sigma_hi: float) -> np.ndarray:
    sigma = np.where(wavelength < mu, sigma_lo, sigma_hi)
    t = (wavelength - mu) / sigma
    return np.exp(-0.5 * t * t)


def wavelength_to_xyz(wavelength_nm: np.ndarray) -> np.ndarray:
    """Map wavelengths (nm) to CIE XYZ tristimulus values, shape ``(..., 3)``."""
    w = np.asarray(wavelength_nm, dtype=np.float64)
    channels = []
    for terms in (_X_TERMS, _Y_TERMS, _Z_TERMS):
        channels.append(sum(amp * _asymmetric_gaussian(w, mu, lo, hi) for amp, mu, lo, hi in terms))
    return np.stack(channels, axis=-1)


def xyz_to_linear_srgb(xyz: np.ndarray) -> np.ndarray:
    return xyz @ _XYZ_TO_LINEAR_SRGB.T


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def tonemap(
    xyz_buffer: np.ndarray,
    exposure: float = 1.0,
    adaptive: bool = False,
    percentile: float = 100.0,
) -> np.ndarray:
    """Detector XYZ accumulation buffer -> displayable uint8 sRGB.

    Gamut handling is the simplest possible policy: negative linear-sRGB
    components (monochromatic light falls outside the sRGB gamut) are
    clipped to zero rather than desaturated. This is a placeholder — see
    the README's open design questions — chosen because it's cheap and
    doesn't hide the pipeline's correctness while more interesting policies
    are explored later.

    When ``adaptive`` is set, brightness is normalized per frame: the
    ``percentile`` (of nonzero, gamut-clipped linear channel values —
    zeros from empty background pixels would otherwise swamp anything
    below roughly the image's fill fraction) is scaled to hit full
    brightness. ``percentile=100`` is a literal "brightest pixel channel
    maps to white"; a lower value (e.g. 99.5) trades a few blown-out
    outlier pixels for a brighter overall image. ``exposure`` is always
    applied too, as a manual gain on top of whatever normalization (or
    lack of it) precedes it — the two combine rather than being
    alternatives.
    """
    linear = xyz_to_linear_srgb(xyz_buffer)
    linear = np.clip(linear, 0.0, None)
    if adaptive:
        nonzero = linear[linear > 0]
        if nonzero.size > 0:
            reference = float(nonzero.max()) if percentile >= 100.0 else float(np.percentile(nonzero, percentile))
            if reference > 0:
                linear = linear / reference
    linear = linear * exposure
    srgb = _linear_to_srgb(linear)
    return (srgb * 255.0 + 0.5).astype(np.uint8)
