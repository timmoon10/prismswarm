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

### Structured fields: geometry × profile × gain

The field catalog (`radial_inward`, `rotational`, `exponential_confinement`,
the old lattice-forming `sinusoidal`, and the ad hoc `wavelength_coupled`
wrapper) was a set of one-off constructors named after the *use case* they
were built for, each re-deriving its own direction/magnitude/singularity
handling. It's been replaced with a small, general system built from three
independent pieces, none of which is required to construct a `Field` (the
raw `Field` callable — `(pos, vel, wavelength, t, dt, rng) -> velocity` —
is still the actual interface; this is one convenient way to build one, not
a shape every field must fit. Perlin noise and the Hopf-fibration
projections on the roadmap won't decompose this way and will implement
`Field` directly when they land):

- A **geometry** factory (`radial_field`, `tangential_field`, `axial_field`)
  turns position into a scalar *coordinate* and a unit-ish *direction*
  vector: radial distance from a center with the outward direction,
  tangential distance/direction within a rotation plane, or a linear
  projection onto an axis with a fixed direction. `radial_field` and
  `tangential_field` accept an arbitrary `center`; `axial_field` an
  arbitrary `axis` and (optionally) a separate `direction` — none of these
  default to or assume the coordinate axes, so a linear field pushing along
  one direction while varying with position along a *different* one (a
  shear flow) is a first-class case, not a special one.
- A **profile** (`constant`, `linear`, `exponential`, `sinusoidal`) is a
  plain scalar→scalar shape function applied to the coordinate, giving the
  field's magnitude. `radial_field(profile=constant(-1.0))` is the old
  `radial_inward`; `radial_field(profile=exponential(...))` is the old
  `exponential_confinement`; `tangential_field(profile=linear(w))` is the
  old `rotational`; `axial_field(profile=sinusoidal(...))` is a
  single-wavevector plane wave.
- A **gain** (`Gain = Callable[[wavelength, t, rng], array | float]`) is a
  dimensionless multiplier a profile can fold into one of its own
  parameters, always normalized so a gain of `1` (the default when none is
  given) leaves the profile unchanged — `power_law_weight` already has this
  property (it evaluates to exactly `1` at its reference wavelength), which
  is what makes it pluggable into *any* profile's gain slot without needing
  to know that slot's scale. Deliberately excluded from `Gain`'s inputs is
  position: a position-dependent scalar would be redundant with the
  geometry/coordinate step, which is already a function of position.

Which parameter a gain multiplies is decided by each profile, not by a
single generic mechanism — `constant`, `linear`, and `exponential` apply
their `gain` to the output magnitude, but `sinusoidal` applies it to the
*entire pre-sine argument* (frequency and phase together), because that's
what the deleted lattice field actually needed (a wavelength-dependent
lattice spacing) and a magnitude-only gain can't reproduce it — scaling
frequency and phase happens inside the `sin`, not as a multiply on its
output. Each profile factory resolves `gain is None` once, at construction
time, into one of two closures with no gain-related branching or
array allocation in the per-step, per-frame hot path; the "no modulation"
case costs exactly what it did before this system existed.

Softening the radial singularity (Plummer-style, avoiding the `1/r`-type
blowup as a particle approaches `center`) belongs in the geometry step, not
the profile: a profile like `constant` or `exponential` is already a
well-defined, finite function at a coordinate of `0` — there's nothing to
soften there. The actual singularity is the *direction* vector,
`offset / |offset|`, which is `0/0` exactly at `center`. `radial_field`
and `tangential_field` fix this by normalizing direction against
`sqrt(|offset|^2 + softening^2)` instead of `|offset|`, while leaving the
coordinate handed to `profile` as the true, unsoftened distance. The
softened direction's own magnitude smoothly shrinks to `0` exactly at
`center` (rather than being clamped to a unit vector all the way in), so
`radial_field() * constant(-1.0)` — the old `radial_inward` — no longer
has a non-decaying speed near the center: velocity is `direction *
profile`, and `direction -> 0` there regardless of what `profile` returns.
That eliminates the permanent period-2 overshoot bounce this field used to
have at `center` (a direct consequence of the direction magnitude no
longer being pinned to `1` all the way to `r = 0`, not something that
needed separate verification). `tangential_field`'s outward-spiral drift
under explicit Euler is unrelated to this and still applies — see the
Roadmap.

`modulated(field, gain)` is the complementary, coarser tool: it rescales an
*already-built* field's total output by a gain, for when you want to tune
wavelength/time dependence from outside without reaching into a profile's
own parameters — e.g. scaling an entire `sum_fields(...)` composition, or
a field you didn't construct yourself. `power_law_weight` (the only
`Gain` shipped so far, `(wavelength / reference_nm) ** exponent`) works
identically whether it's plugged into a profile's `gain` parameter or into
`modulated()`, since both consume the same `Gain` type. `exponent = -1` is
physically grounded (photon momentum `p = h/λ` — shorter wavelengths
genuinely push harder under radiation-pressure-like forcing); `exponent =
+1` favors long wavelengths on a principled but not equally fundamental
basis (e.g. a diffraction-flavored reading); `exponent = 0` recovers no
modulation at all (same as omitting `gain`/`modulated` entirely).

A resonant/bandpass gain (Gaussian or Lorentzian, peaked at a target
wavelength) was considered and is a legitimate physical model — it's the
standard lineshape for a single absorption/emission resonance, the same
mechanism that gives colored glass its color (a dopant ion's electronic
transition). It's deliberately deferred: modeling multiple resonances well
requires gain functions to compose by *multiplication* (matching
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
   99.5th, trading a few blown-out outlier pixels for a brighter overall
   image; the 100th would map the single brightest pixel channel exactly
   to full brightness) is scaled to hit full brightness each frame. A
   manual `exposure` gain is always applied on top of that, so the two
   aren't alternatives — adaptive picks a reasonable per-frame baseline,
   manual pushes from there.

## Software architecture

Proposed module layout:

- `prismswarm/fields.py` — the `Field` interface; geometry factories
  (`radial_field`, `tangential_field`, `axial_field`) and profiles
  (`constant`, `linear`, `exponential`, `sinusoidal`) that combine into
  structured fields; `Gain`-based modulation (`power_law_weight`,
  `modulated`); the standalone `constant_field` and `brownian`; and the
  additive composition helper (`sum_fields`). See "Structured fields:
  geometry × profile × gain" above.
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
  interface carries `wavelengths`; `power_law_weight` (a `Gain`) plugs into
  any profile's gain parameter or into `modulated()` for short- or
  long-wavelength-favoring coupling, composable with `sum_fields`.
  Resonant/bandpass gain (deferred, see "Structured fields" above)
  intentionally left for later.
- Explicitly deferred within this milestone: dynamic per-particle spectra
  (random walks, explicit spectral conversion) — a later milestone once
  static spectra and field coupling are both working

**M3 — Field catalog & composition**
- Reworked the catalog from one-off named constructors into the
  geometry × profile × gain system (done — see "Structured fields" above):
  `radial_field`, `tangential_field`, `axial_field` geometries; `constant`,
  `linear`, `exponential`, `sinusoidal` profiles; `power_law_weight` +
  `modulated()` for wavelength/time-dependent gain. The old lattice-forming
  sinusoidal field (a per-axis product of sines, geometrically distinct
  from `axial_field`'s single-wavevector plane wave) was deleted rather
  than kept alongside the new system — a deliberate prototype, not a
  regression.
- Still to add: Perlin noise, rectilinear, stereographic projections of
  Hopf fibers
- Exercise field composition (addition) now that multiple fields exist
- Discretization correction for `tangential_field` to prevent outward
  spiraling: explicit Euler applied to pure circular motion is
  unconditionally unstable and drifts outward every step (verified — after
  200 steps at `angular_velocity=2.0`, a particle starting at r=1 drifts to
  r≈1.56). Deferred deliberately for now.

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
