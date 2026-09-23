"""Video export: render a run of the simulation to an MP4 file.

Deliberately does *not* step the live `Simulation` passed to `record` — see
that function's docstring for why. It's normally called from the REPL's
background thread while the render loop is stepping that same `sim` on the
main thread (see `simulation.py`'s threading-model note), and a tight loop
of `sim.step()` calls from a second thread would race the render loop's
own stepping, corrupting both the live view and the recording. Instead,
`record` clones just enough of `sim`'s current state to start an
independent run and steps *that* clone as fast as this thread can compute.

No new Python dependency: frames are piped to the `ffmpeg` binary as raw
RGB24, which must already be on `PATH` (checked up front, rather than
left to fail deep inside a subprocess pipe with a cryptic error).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

from . import color
from .detector import Detector
from .state import ParticleState

if TYPE_CHECKING:
    from .simulation import Simulation


def record(
    sim: "Simulation",
    duration_s: float,
    path: str,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
    progress: bool = True,
) -> None:
    """Render ``duration_s`` seconds of simulated time, starting from
    ``sim``'s current state and driven by its currently-active field, to
    an MP4 at ``path``. Frames are generated as fast as this thread can
    compute them — there's no real-time throttling — so wall-clock capture
    time can be faster or slower than ``duration_s``; the output always
    plays back at exactly ``duration_s`` seconds.

    ``fps`` defaults to ``round(1 / sim.dt)`` — one video frame per
    simulation step, matching the live render loop's cadence. Passing a
    different ``fps`` steps the recording's own clone at ``dt = 1 / fps``
    instead, so a higher ``fps`` also means finer-grained (more accurate)
    integration, not just a smoother video — ``sim.dt`` itself is never
    touched. ``width``/``height`` default to ``sim.detector``'s
    resolution; pass larger ones for a higher-resolution export rendered
    directly at that resolution (not upscaled after the fact), without
    touching the live detector.

    Safe to call from the REPL's background thread while the render loop
    is live on the main thread: this never steps, splats into, or
    otherwise mutates ``sim.state``/``sim.detector``/``sim.rng`` — it
    copies ``sim.state``'s arrays, spawns an independent child of
    ``sim.rng`` (so it can't race the draws the live render loop is making
    against the same shared generator), and builds its own ``Detector``.
    ``sim.fields``/``sim.spectrum`` are read-only callables, safe to share
    directly. The recording therefore starts from what's visible right
    now but diverges from the live view from that point on — it is not a
    screen capture of the live window.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("record() needs the `ffmpeg` binary on PATH to encode video")
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")

    fps = round(1.0 / sim.dt) if fps is None else fps
    dt = 1.0 / fps
    n_frames = max(1, round(duration_s * fps))
    width = sim.detector.width if width is None else width
    height = sim.detector.height if height is None else height

    rng = sim.rng.spawn(1)[0]
    state = ParticleState(
        positions=sim.state.positions.copy(),
        velocities=sim.state.velocities.copy(),
        wavelengths=sim.state.wavelengths.copy(),
    )
    detector = Detector(width=width, height=height, half_extent=sim.detector.half_extent)
    field = sim.active_field
    t = sim.t
    exposure, adaptive_exposure, adaptive_percentile = sim.exposure, sim.adaptive_exposure, sim.adaptive_percentile

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(fps), "-i", "-",
            "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None

    if progress:
        print(f"record: {n_frames} frames ({duration_s}s @ {fps}fps, {width}x{height}) -> {path}")
    try:
        report_every = max(1, n_frames // 20)
        for i in range(n_frames):
            state.step(field, t, dt, rng)
            t += dt
            detector.clear()
            xyz = color.wavelength_to_xyz(state.wavelengths)
            detector.splat(state.positions[:, :2], xyz)
            rgb = color.tonemap(
                detector.buffer, exposure=exposure, adaptive=adaptive_exposure, percentile=adaptive_percentile
            )
            try:
                ffmpeg.stdin.write(rgb.tobytes())
            except BrokenPipeError:
                # ffmpeg has already exited (bad path, bad codec, disk full,
                # ...) — stop feeding it and let the returncode check below
                # report why, instead of this raw pipe error masking it.
                break
            if progress and (i + 1) % report_every == 0:
                print(f"\rrecord: {i + 1}/{n_frames}", end="", flush=True)
    finally:
        try:
            ffmpeg.stdin.close()
        except BrokenPipeError:
            pass
        returncode = ffmpeg.wait()

    if progress:
        print()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {returncode} while writing {path}")
    if progress:
        print(f"record: saved {path}")
