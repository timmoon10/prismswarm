# prismswarm

A digital art project for exploring emergent mathematical phenomena through
a large-scale particle light simulation, rather than hand-crafting a fixed
visual outcome. Priorities, in order: flexibility for exploration; a
principled, generalizable implementation (especially of the velocity-field
abstraction); aesthetic quality (pleasant, striking, sensual). Secondary,
but still required: real-time rendering with no GPU acceleration, live
control via a terminal REPL, and room to stumble onto interesting math
along the way.

## Simulation model

### Particle state

Each particle carries a position, a velocity, and an emission wavelength.
Positions currently live in R^3; the state representation is kept general
enough to extend to higher dimensions (see Roadmap).

### Velocity fields

The core abstraction is a velocity field: a function of
`(positions, velocities, t, rng)` returning a velocity increment. Fields
compose by addition — combining two fields is just summing their
contributions, with no special-casing required in the integrator. A
deterministic field (e.g. radial-inward) ignores `rng`; a stochastic field
(e.g. Brownian motion) ignores position. Both fit the same interface.

Initial fields: a radially-inward uniform field, and Brownian motion
(an Euler–Maruyama noise term, `Δv = sqrt(dt) * σ * randn()` per particle
per step). Although the interface supports summing them, the first
validation experiments run them independently, alternating between the two
to check the visualization pipeline in isolation before exercising
composition.

### Integration

Explicit Euler: `v += field(...)`, `x += v * dt`. Force-based integration
(e.g. Verlet) is an explicitly lower-priority future direction — it implies
switching from a velocity-field abstraction to a force-field one, so it's
being deferred rather than retrofitted early.

### Projection & detector

Particle positions are projected orthographically onto the xy-plane
(a pinhole camera projection is a future extension). The projected points
land on a **detector**: a per-pixel accumulation buffer of CIE XYZ
tristimulus values.

The detector's resolution is an independent parameter from the on-screen
display resolution. This matters for both performance and aesthetics: you
can run a coarse detector grid and upscale to a large window (cheap), or a
fine detector grid downsampled to a small window, without touching the
physics or the color pipeline. Detector→display resizing is purely a
presentation-layer concern.

### Photon → color pipeline

1. Each particle's wavelength maps to an XYZ tristimulus contribution via
   the CIE 1931 standard observer color-matching functions
   (x̄(λ), ȳ(λ), z̄(λ)). This is the "physically-inspired, wavelength-dependent
   detector sensitivity," and it reproduces cross-channel spillage for free
   — a pure 530nm green legitimately produces nonzero x̄, the same way it
   would stimulate human L-cones somewhat, or a calibrated camera's red
   channel.
2. Per frame, particle contributions are splatted into the detector's XYZ
   buffer, accumulated per detector pixel (via `np.bincount` on flattened
   pixel indices — avoids the slow `np.add.at` path).
3. The accumulated XYZ buffer is converted to linear sRGB via the standard
   CIE XYZ→sRGB (D65) matrix.
4. Out-of-gamut handling: monochromatic spectral-locus colors fall outside
   the sRGB gamut, producing negative components. Policy (clip vs.
   desaturate toward white) is an open question — see below.
5. Exposure normalization / tonemapping (particle density varies hugely
   across a frame and across scenes), then the sRGB OETF (gamma) for the
   final display buffer, which may sit at a different resolution than the
   detector.

## Software architecture

Proposed module layout:

- `prismswarm/fields.py` — velocity field interface, initial
  implementations (radial-inward, Brownian), composition helper.
- `prismswarm/state.py` — particle state container (positions, velocities,
  wavelengths as NumPy float32 arrays) and the integration step.
- `prismswarm/color.py` — CIE color-matching-function tables, XYZ
  accumulation, XYZ→sRGB conversion, gamut handling, tonemapping.
- `prismswarm/detector.py` — the detector buffer (independent resolution),
  splatting (bincount-based accumulation).
- `prismswarm/render.py` — pygame window, detector→display resize/blit,
  the main render loop.
- `prismswarm/repl.py` — the control REPL: an `IPython.terminal.embed`
  shell (a plain terminal REPL, not a notebook or kernel) running on a
  background thread, with live references to the active field list,
  detector, and display parameters for interactive editing while the sim
  runs.
- `prismswarm/main.py` — entry point wiring the above together and holding
  the initial scene setup.

**Threading model:** the main thread owns the pygame window and the
render/integration loop (SDL/pygame windowing needs to own the thread it
was created on, particularly on macOS). The REPL runs on a background
thread and mutates shared state — e.g. swapping the active field list —
through simple reference/list reassignment rather than fine-grained
locking, relying on the GIL for atomicity of those individual ops. This
keeps the REPL responsive without stalling the render loop, and vice
versa.

**Numeric approach:** NumPy (float32), vectorized over the whole particle
array for field evaluation and integration — no per-particle Python loop.
If profiling at ~1M particles shows this isn't fast enough for real-time,
the plan is to JIT just the fixed inner loop (integration + splat) with
Numba, keeping field *definitions* in vectorized Python so they stay
REPL-editable regardless of which acceleration path is active. A native
pybind11 extension is the fallback if Numba turns out to be insufficient.

## Roadmap

**M1 — Core loop (current target)**
- Particle state in R^3, Euler integration
- Radial-inward and Brownian fields, independently selectable at runtime
  (not summed yet)
- Orthographic projection; detector buffer decoupled from display
  resolution
- CIE XYZ→sRGB pipeline with basic gamut clipping and exposure
  normalization
- pygame display window, real-time loop at ~1M particles
- Terminal REPL for live control (swap active field, tweak parameters)

**M2 — Field catalog & composition**
- Perlin noise, rectilinear, sinusoidal, stereographic projections of Hopf
  fibers, exponentially-growing confining fields
- Exercise field composition (addition) now that multiple fields exist
- Discretization correction for the radial field to prevent outward
  spiraling

**M3 — Color catalog & wavelength dynamics**
- Color distributions: white, blackbody, discrete RGB, monochrome
- Per-particle wavelength random walks; explore convergence to target
  spectra (e.g. a uniform visible-spectrum distribution via
  Brownian motion confined to the visible range)

**M4 — Performance scaling**
- Profile M1–M3 at 1M+ particles; introduce Numba JIT on the
  integration/splat hot path if needed
- pybind11 native extension as a fallback if Numba proves insufficient

**Future extensions (not scheduled)**
- Wavelength-dependent field coupling (a particle's emitted wavelength
  affects its response to a field)
- Higher-dimensional particle space (4D/8D projected down to R^2)
- Pinhole camera projection with distance-based brightness falloff
  (`d^(1-n)` is natural for a pinhole model; the orthographic falloff law,
  if any, is TBD)

**Lower priority (not excluded)**
- Spectral (non-pure-wavelength) emission per particle
- Particle interactions via a Barnes-Hut-style approximation (O(n²) is a
  non-starter at this scale)
- Force-based dynamics (e.g. Verlet) as an alternative to velocity-field
  integration

## Open design questions

- Gamut-mapping policy for out-of-gamut spectral colors (clip vs.
  desaturate) — likely needs visual experimentation to settle.
- Exposure/normalization strategy for the detector buffer (fixed vs.
  adaptive/auto-exposure) as particle density varies across a frame and
  across scenes.
