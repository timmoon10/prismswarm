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
`(positions, velocities, wavelengths, t, dt, rng)` returning *the*
velocity for this step — not an increment to accumulate. Fields are
kinematic: like a fluid flow field advecting passive tracers, they
prescribe velocity directly rather than exerting a force that builds up
over time. Fields compose by addition — combining two fields is just
summing their prescribed velocities, with no special-casing required in
the integrator. A deterministic field (e.g. radial-inward) ignores `rng`;
a stochastic field (e.g. Brownian motion) ignores position; a
wavelength-independent field ignores `wavelengths` — every field shipped
so far does exactly that, so wavelength coupling is strictly opt-in, never
implicit. `dt` is passed through explicitly so stochastic fields can
discretize correctly (see below).

Initial fields: a radially-inward uniform field, and Brownian motion. The
latter is an Euler–Maruyama discretization: the Wiener process gives a
position increment `dx = σ * sqrt(dt) * randn()` per step, so the field
returns `v = σ / sqrt(dt) * randn()` — velocity that grows unboundedly as
`dt -> 0`, reflecting the non-differentiability of Brownian paths — so
that the integrator's `x += v * dt` recovers the correct increment.
Although the interface supports summing fields, the first validation
experiments run them independently, alternating between the two to check
the visualization pipeline in isolation before exercising composition.

### Wavelength-coupled fields

`wavelength_coupled(base_field, weight)` scales a base field's velocity by
a per-particle `weight(wavelength)`, leaving the base field itself
wavelength-agnostic. It composes with `sum_fields` like any other field —
that's the intended way to combine several differently-tuned couplings
(e.g. a short-wavelength-favoring and a long-wavelength-favoring instance
of the same base field, added together) into one scene, rather than
building a separate multi-weight mechanism.

The only weight shipped so far is `power_law_weight(reference_nm,
exponent)`: `(wavelength / reference_nm) ** exponent`. `exponent = -1` is
physically grounded — photon momentum `p = h/λ` means shorter wavelengths
genuinely carry more momentum, so if the coupled field represents
radiation-pressure-like forcing, favoring short wavelengths this way is
the physically correct direction. `exponent = +1` favors long wavelengths
instead; it isn't backed by an equally fundamental law the way `-1` is,
but it's still a principled, tunable choice (readable e.g. as a
diffraction-flavored metaphor, where longer wavelengths couple more
strongly to a field's spatial structure). `exponent = 0` recovers an
uncoupled field.

A resonant/bandpass weight (Gaussian or Lorentzian, peaked at a target
wavelength) was considered and is a legitimate physical model — it's the
standard lineshape for a single absorption/emission resonance, the same
mechanism that gives colored glass its color (a dopant ion's electronic
transition). It's deliberately deferred: modeling multiple resonances
well requires weight functions to compose by *multiplication* (matching
Beer-Lambert absorption, where stacked absorbers multiply transmittances),
which is a different composition rule than the addition used for fields
themselves — worth its own design pass rather than bolting on now.

### Integration

Explicit Euler: velocity is recomputed each step as the sum of active
fields (`v = Σ field_i(...)`), then `x += v * dt`. Velocity is not
persistent, accumulated state — it's a quantity fully determined each
instant by the fields currently acting on the particle. Force-based
integration (e.g. Verlet) is an explicitly lower-priority future direction
— it implies switching from this velocity-field abstraction to a
force-field one, so it's being deferred rather than retrofitted early.

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
5. Exposure normalization / tonemapping, then the sRGB OETF (gamma) for
   the final display buffer, which may sit at a different resolution than
   the detector. Particle density varies hugely across a frame and across
   scenes, so normalization defaults to adaptive: a percentile of the
   detector's brightest nonzero linear-sRGB channel values (default the
   100th, i.e. the single brightest pixel channel) is scaled to hit full
   brightness each frame. A manual `exposure` gain is always applied on
   top of that, so the two aren't alternatives — adaptive picks a
   reasonable per-frame baseline, manual pushes from there.

## Software architecture

Proposed module layout:

- `prismswarm/fields.py` — velocity field interface, implementations
  (radial-inward, rotational, exponential confinement, Brownian), the
  additive composition helper (`sum_fields`), and wavelength coupling
  (`wavelength_coupled`, `power_law_weight`).
- `prismswarm/spectra.py` — the emission-spectrum interface (mirrors
  `fields.py`'s shape: `spectrum(n, rng) -> wavelengths_nm`), used at
  particle initialization. Implementations: monochrome, blackbody
  (Planck's law, inverse-transform sampled).
- `prismswarm/state.py` — particle state container (positions, velocities,
  wavelengths as NumPy float32 arrays; wavelengths are drawn from a
  `Spectrum` at construction) and the integration step.
- `prismswarm/color.py` — wavelength→XYZ (an analytic multi-Gaussian fit
  to the CIE 1931 color-matching functions, not a tabulated lookup),
  XYZ→sRGB conversion, gamut handling, tonemapping.
- `prismswarm/detector.py` — the detector buffer (independent resolution),
  splatting (bincount-based accumulation).
- `prismswarm/simulation.py` — the mutable `Simulation` object shared
  between the render loop and the REPL: particle state, detector, the
  field catalog and which one is active, the spectrum used to (re)seed
  wavelengths, `dt`, and the exposure knobs (`exposure`, `adaptive_exposure`,
  `adaptive_percentile`). Plain attributes, no locking — each read/write is
  a single, GIL-atomic reference assignment, and the render loop reads a
  consistent snapshot once per frame. `Simulation.reset()` reinitializes
  the particle population from a fresh uniform ball using the stored `rng`
  and `spectrum` — the recovery path for a swarm that has wandered
  off-screen or gone non-finite.
- `prismswarm/render.py` — pygame window, detector→display resize/blit,
  the main render loop. The only input it handles is window close /
  `Esc`; everything else is REPL-only (no ad hoc keyboard shortcuts for
  simulation parameters — those don't scale past a couple of options and
  the REPL already covers it). Since the REPL can push the sim into a bad
  state with no validation, the per-frame body is wrapped in a
  try/except: an exception prints its traceback once (not every frame,
  while the same error persists) and surfaces in the window title, rather
  than crashing the process — which would otherwise take the REPL's
  daemon thread down with it. `sim.reset()` (see `simulation.py`) is the
  usual recovery.
- `prismswarm/repl.py` — the control REPL: an `IPython.terminal.embed`
  shell (a plain terminal REPL, not a notebook or kernel) running on a
  background thread, with a live reference to the `Simulation` object for
  interactive editing while the sim runs. Owns the REPL's usage guidance
  (banner + `sim_help()`) — skips starting itself if stdin isn't a real
  terminal, rather than spinning on repeated EOF.
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

**M1 — Core loop (done)**
- Particle state in R^3, Euler integration
- Radial-inward and Brownian fields, independently selectable at runtime
  (not summed yet)
- Orthographic projection; detector buffer decoupled from display
  resolution
- CIE XYZ→sRGB pipeline with gamut clipping and manual + adaptive
  exposure normalization
- pygame display window, real-time loop at ~1M particles
- Terminal REPL for live control, with a `sim_help()` usage guide

**M2 — Color & wavelength-coupled dynamics (current target)**

Reordered ahead of the field catalog: color dynamics are expected to carry
more of the piece's visual interest than particle motion, so it's worth
settling this before scaling out M3. Also where wavelength-dependent field
coupling — previously a "not scheduled" extension — moved to, since it's
naturally part of the same question (how wavelength does and doesn't
influence the rest of the system).

- Static emission spectra at initialization: monochrome (done), blackbody
  (done — defaults to the sun's ~5778K effective temperature, restricted
  to the visible range); white and discrete-RGB catalog entries later
- Wavelength-dependent velocity field coupling (done): the `Field`
  interface now carries `wavelengths`; `wavelength_coupled` +
  `power_law_weight` give short- or long-wavelength-favoring coupling,
  composable with `sum_fields`. Resonant/bandpass coupling (deferred, see
  "Wavelength-coupled fields" above) intentionally left for later.
- Explicitly deferred within this milestone: dynamic per-particle spectra
  (random walks, explicit spectral conversion) — a later milestone once
  static spectra and field coupling are both working

**M3 — Field catalog & composition**
- Rotational (rigid-body rotation about the view axis) and
  exponentially-growing radial confinement fields: done, ahead of the
  rest of this milestone, alongside the M2 wavelength-coupling work
- Still to add: Perlin noise, rectilinear, sinusoidal, stereographic
  projections of Hopf fibers
- Exercise field composition (addition) now that multiple fields exist
- Discretization correction for `rotational` to prevent outward spiraling:
  explicit Euler applied to pure circular motion is unconditionally
  unstable and drifts outward every step (verified — after 200 steps at
  `angular_velocity=2.0`, a particle starting at r=1 drifts to r≈1.56).
  Deferred deliberately for now.

**M4 — Dynamic spectra**
- Per-particle wavelength random walks; explore convergence to target
  spectra (e.g. a uniform visible-spectrum distribution via Brownian
  motion confined to the visible range)
- Explicit spectral conversion between distributions

**M5 — Performance scaling**
- Profile M1–M4 at 1M+ particles; introduce Numba JIT on the
  integration/splat hot path if needed
- pybind11 native extension as a fallback if Numba proves insufficient

**Future extensions (not scheduled)**
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
