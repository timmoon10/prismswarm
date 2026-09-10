"""The live control REPL: a plain terminal-embedded IPython shell (not a
notebook or kernel — no Jupyter client needed) running on a background
thread, with direct references to the running simulation.

This module owns the REPL's usage guidance (the banner and ``sim_help()``)
rather than leaving callers to document `sim.*` conventions themselves —
the set of controllable attributes lives on ``Simulation``, but how you're
meant to talk to them from the REPL is a REPL concern.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

HELP_TEXT = """\
prismswarm REPL guide
======================
The simulation is running live in another thread; changes here take
effect on the next frame. Nothing is locked, so read-modify-write your
own state if you need a consistent snapshot across multiple attributes.

Fields
  sim.active_field_name        key into sim.fields that's currently driving the sim
  sim.fields                   dict[str, Field]; assign to add or replace a field, e.g.:
                                  sim.fields['radial'] = fields.radial_inward(speed=0.6)
                                  sim.active_field_name = 'radial'
  fields.sum_fields(*fs)       compose fields by summing their velocities

Exposure / display
  sim.exposure                 manual brightness gain (float, default 1.0),
                                always applied on top of adaptive normalization
  sim.adaptive_exposure        if True (default), auto-normalize each frame so
                                a percentile of the detector's brightest nonzero
                                pixels maps to full brightness
  sim.adaptive_percentile      percentile in [0, 100] used when adaptive
                                (default 100 = the single brightest pixel channel;
                                lower e.g. 99.5 trades a few blown-out outliers
                                for a brighter overall image)

Simulation state
  sim.t, sim.dt                simulation time and timestep
  sim.state.positions/velocities/wavelengths
                                raw NumPy arrays, shape (n, dim) / (n, dim) / (n,)
  sim.detector.half_extent     world-space half-width mapped to the detector's pixel grid
  sim.rng                      shared numpy.random.Generator

Call sim_help() to print this again.
"""


def start(namespace: dict[str, Any]) -> threading.Thread | None:
    """Start the REPL on a background thread, or skip it if stdin isn't a
    real terminal. Without this guard, an embedded IPython shell reading
    from a non-tty stdin sees immediate EOF on every read and busy-loops
    re-prompting "Do you really want to exit?" at 100% CPU instead of
    exiting once — worth checking explicitly rather than letting it happen.
    """
    if not sys.stdin.isatty():
        print("prismswarm: stdin is not a terminal, skipping the REPL.", file=sys.stderr)
        return None

    namespace = dict(namespace)
    namespace.setdefault("sim_help", lambda: print(HELP_TEXT))

    def _run() -> None:
        from IPython.terminal.embed import InteractiveShellEmbed

        banner = "prismswarm REPL — sim is running live. Call sim_help() for a usage guide.\n"
        shell = InteractiveShellEmbed(user_ns=namespace, banner1=banner)
        shell()

    thread = threading.Thread(target=_run, name="prismswarm-repl", daemon=True)
    thread.start()
    return thread
