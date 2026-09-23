"""The detector: a pixel accumulation buffer, independent of display resolution.

The detector owns a world-space window mapped onto a pixel grid (``width``
x ``height``) that need not match the on-screen display size. Running a
coarse detector grid upscaled to a large window, or a fine grid
downsampled to a small one, is purely a presentation-layer choice made
downstream in ``render.py``.

``half_extent`` is the world-space half-*width* the detector covers (the x
half-range); the half-*height* is derived from it via the pixel aspect
ratio (``height / width``) rather than reusing ``half_extent`` directly for
both axes, so pixels are always square and a non-square detector (e.g. a
16:9 video export) doesn't stretch the image.
"""

from __future__ import annotations

import numpy as np


class Detector:
    def __init__(self, width: int, height: int, half_extent: float):
        self.width = width
        self.height = height
        self.half_extent = half_extent
        # float64, not float32: accumulating millions of small per-particle
        # contributions benefits from the extra precision; the rest of the
        # pipeline stays float32 for speed/memory.
        self.buffer = np.zeros((height, width, 3), dtype=np.float64)

    def clear(self) -> None:
        self.buffer.fill(0.0)

    def splat(self, xy: np.ndarray, tristimulus: np.ndarray) -> None:
        """Accumulate per-particle tristimulus contributions into detector
        pixels.

        ``xy`` is the orthographically projected position, ``tristimulus``
        the per-particle CIE XYZ contribution. Particles landing outside
        the detector's world-space window are dropped. Accumulation uses
        ``np.bincount`` on flattened pixel indices rather than
        ``np.add.at``, which is dramatically faster at particle counts in
        the millions.
        """
        scale = self.width / (2.0 * self.half_extent)
        half_extent_y = self.height / (2.0 * scale)  # world half-height at the same pixel scale as x

        col = np.floor((xy[:, 0] + self.half_extent) * scale).astype(np.int64)
        row_from_bottom = np.floor((xy[:, 1] + half_extent_y) * scale).astype(np.int64)
        row = self.height - 1 - row_from_bottom  # flip: array row 0 is the top, +y is up

        valid = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        col = col[valid]
        row = row[valid]
        weights = tristimulus[valid]

        flat_index = row * self.width + col
        size = self.height * self.width
        for channel in range(3):
            self.buffer[:, :, channel] += np.bincount(
                flat_index, weights=weights[:, channel], minlength=size
            ).reshape(self.height, self.width)
