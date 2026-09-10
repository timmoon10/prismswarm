"""The detector: a pixel accumulation buffer, independent of display resolution.

The detector owns a world-space window (``half_extent``) mapped onto a
pixel grid (``width`` x ``height``) that need not match the on-screen
display size. Running a coarse detector grid upscaled to a large window,
or a fine grid downsampled to a small one, is purely a presentation-layer
choice made downstream in ``render.py``.
"""

from __future__ import annotations

import numpy as np


class Detector:
    def __init__(self, width: int, height: int, half_extent: float):
        self.width = width
        self.height = height
        self.half_extent = half_extent
        self.buffer = np.zeros((height, width, 3), dtype=np.float64)

    def clear(self) -> None:
        self.buffer.fill(0.0)

    def splat(self, xy: np.ndarray, xyz: np.ndarray) -> None:
        """Accumulate per-particle XYZ contributions into detector pixels.

        ``xy`` is the orthographically projected position, ``xyz`` the
        per-particle tristimulus contribution. Particles landing outside
        the detector's world-space window are dropped. Accumulation uses
        ``np.bincount`` on flattened pixel indices rather than
        ``np.add.at``, which is dramatically faster at particle counts in
        the millions.
        """
        scale = self.width / (2.0 * self.half_extent)
        col = np.floor((xy[:, 0] + self.half_extent) * scale).astype(np.int64)
        scale_y = self.height / (2.0 * self.half_extent)
        row_from_bottom = np.floor((xy[:, 1] + self.half_extent) * scale_y).astype(np.int64)
        row = self.height - 1 - row_from_bottom  # flip: array row 0 is the top, +y is up

        valid = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        col = col[valid]
        row = row[valid]
        weights = xyz[valid]

        flat_index = row * self.width + col
        size = self.height * self.width
        for channel in range(3):
            self.buffer[:, :, channel] += np.bincount(
                flat_index, weights=weights[:, channel], minlength=size
            ).reshape(self.height, self.width)
