"""The pygame display window and main render loop. Owns the main thread —
SDL/pygame windowing needs to run on the thread it was created on,
particularly on macOS — while the REPL runs separately (see ``repl.py``).

The REPL mutates live simulation state with no validation (see
``simulation.py``), so a bad edit — wrong dtype, wrong shape, a field that
returns garbage — can make the per-frame step or render raise. Since this
loop owns the main thread, an unhandled exception here would take the
whole process down, killing the REPL's daemon thread with it (from the
user's perspective, "the REPL died"). The frame body is therefore wrapped
in a try/except: on failure the traceback prints once to stderr (not every
frame, while the same error persists) and the window stays up showing the
last good frame, with the error surfaced in the title bar, so the user can
fix state from the still-live REPL — ``sim.reset()`` is the usual recovery
for a population that's gone non-finite or wandered off-screen — and the
loop picks back up next frame once state is valid again.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import pygame

from . import color
from .simulation import Simulation


def run(sim: Simulation, display_size: tuple[int, int] = (900, 900), target_fps: int = 30) -> None:
    pygame.init()
    pygame.display.set_caption("prismswarm")
    screen = pygame.display.set_mode(display_size)
    clock = pygame.time.Clock()

    last_error: str | None = None

    while sim.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sim.running = False

        try:
            sim.step()

            sim.detector.clear()
            xy = sim.state.positions[:, :2]
            xyz = color.wavelength_to_xyz(sim.state.wavelengths)
            sim.detector.splat(xy, xyz)

            rgb = color.tonemap(
                sim.detector.buffer,
                exposure=sim.exposure,
                adaptive=sim.adaptive_exposure,
                percentile=sim.adaptive_percentile,
            )
            # surfarray expects (width, height, 3); the detector buffer is (row, col, 3) = (height, width, 3).
            surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            if surface.get_size() != display_size:
                surface = pygame.transform.smoothscale(surface, display_size)
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            last_error = None
        except Exception:
            error_text = traceback.format_exc()
            if error_text != last_error:
                print(error_text, file=sys.stderr)
                last_error = error_text

        clock.tick(target_fps)
        caption = f"prismswarm — {sim.active_field_name} — {clock.get_fps():.1f} fps"
        if last_error is not None:
            caption += " — ERROR, see console — try sim.reset()"
        pygame.display.set_caption(caption)

    pygame.quit()
