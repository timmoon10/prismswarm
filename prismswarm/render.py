"""The pygame display window and main render loop. Owns the main thread —
SDL/pygame windowing needs to run on the thread it was created on,
particularly on macOS — while the REPL runs separately (see ``repl.py``).
"""

from __future__ import annotations

import numpy as np
import pygame

from . import color
from .simulation import Simulation


def run(sim: Simulation, display_size: tuple[int, int] = (900, 900), target_fps: int = 30) -> None:
    pygame.init()
    pygame.display.set_caption("prismswarm")
    screen = pygame.display.set_mode(display_size)
    clock = pygame.time.Clock()

    field_keys = list(sim.fields.keys())

    while sim.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sim.running = False
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    index = event.key - pygame.K_1
                    if index < len(field_keys):
                        sim.active_field_name = field_keys[index]

        sim.step()

        sim.detector.clear()
        xy = sim.state.positions[:, :2]
        xyz = color.wavelength_to_xyz(sim.state.wavelengths)
        sim.detector.splat(xy, xyz)

        rgb = color.tonemap(sim.detector.buffer, exposure=sim.exposure)
        # surfarray expects (width, height, 3); the detector buffer is (row, col, 3) = (height, width, 3).
        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        if surface.get_size() != display_size:
            surface = pygame.transform.smoothscale(surface, display_size)
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        clock.tick(target_fps)
        pygame.display.set_caption(f"prismswarm — {sim.active_field_name} — {clock.get_fps():.1f} fps")

    pygame.quit()
