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
                                  sim.fields['radial'] = fields.radial_field(profile=fields.constant(-0.6))
                                  sim.active_field_name = 'radial'
  fields.sum_fields(*fs)       compose fields by summing their velocities

  Fields are built from a geometry × profile × gain (see README "Structured fields"):
  fields.radial_field(profile, center, softening)
                                direction is outward from center; profile(distance) sets magnitude,
                                e.g. fields.radial_field(profile=fields.constant(-1.0)) pulls inward
  fields.tangential_field(profile, center, plane_axes, softening)
                                direction is tangential within plane_axes (default: view plane);
                                e.g. fields.tangential_field(profile=fields.linear(1.0)) is rigid rotation
  fields.axial_field(profile, axis, direction, anchor, dim, rng)
                                direction is fixed (defaults to axis); coordinate is dot(axis, x - anchor)
  fields.constant(value, gain), fields.linear(slope, gain),
  fields.exponential(rate, amplitude, gain)
                                profiles: coordinate -> magnitude. gain (a Gain, see below) scales
                                the profile's one scalar knob (value/slope/amplitude)
  fields.sinusoidal(frequency, phase, amplitude, amplitude_gain, frequency_gain, phase_gain)
                                has three knobs, so each gets its own named hook instead of one
                                ambiguous gain: amplitude_gain scales the output like the others do;
                                frequency_gain/phase_gain scale their parameter before it enters sin
                                (the same Gain in both reproduces the old lattice field's
                                frequency-and-phase-together wavelength coupling — see fields.py)

  fields.power_law_weight(reference_nm, exponent)
                                a Gain: (wavelength / reference_nm) ** exponent, equal to 1 at
                                reference_nm; exponent<0 favors short wavelengths, >0 favors long,
                                0 is no modulation. Plug into any profile's gain hook, e.g.:
                                  short_favored = fields.radial_field(
                                      profile=fields.constant(-0.6, gain=fields.power_law_weight(exponent=-1.0)))
  fields.modulated(field, gain)
                                rescale an already-built field's total output by gain, for when
                                you don't/can't reach into its profile's own gain parameter

Exposure / display
  sim.exposure                 manual brightness gain (float, default 1.0),
                                always applied on top of adaptive normalization
  sim.adaptive_exposure        if True (default), auto-normalize each frame so
                                a percentile of the detector's brightest nonzero
                                pixels maps to full brightness
  sim.adaptive_percentile      percentile in [0, 100] used when adaptive
                                (default 99.5, trading a few blown-out outlier pixels
                                for a brighter overall image; 100 = the single
                                brightest pixel channel maps to white exactly)

Simulation state
  sim.t, sim.dt                simulation time and timestep
  sim.state.positions/velocities/wavelengths
                                raw NumPy arrays, shape (n, dim) / (n, dim) / (n,)
  sim.detector.half_extent     world-space half-width mapped to the detector's pixel grid
  sim.rng                      shared numpy.random.Generator
  sim.reset(n=None, radius=1.0)
                                reinitialize the particle population from a fresh uniform
                                ball (reusing sim.rng and sim.spectrum); recovers a swarm
                                that has wandered off-screen or gone non-finite (inf/nan)
                                without restarting the process. If the render loop hits an
                                error, its traceback prints to the console and the window
                                title shows ERROR until state is valid again — sim.reset()
                                is usually the fix.

Wavelengths (spectra.py)
  spectra.monochrome(nm), spectra.blackbody(temperature_k)
                                Spectrum factories: spectrum(n, rng) -> wavelengths_nm.
                                Reassign the whole population directly, e.g.:
                                  sim.state.wavelengths[:] = spectra.blackbody(3000.0)(sim.state.n, sim.rng)

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
