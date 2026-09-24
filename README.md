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
Positions live in R^dim for any `dim >= 2` (3 by default, set via
`main.py --dim`) — not just R^3. Every geometry, profile, and gain in
`fields.py`/`gains.py` operates on whatever dimensionality `pos` carries at
call time rather than assuming 3, and orthographic projection (`positions[:,
:2]`) always takes the first two axes regardless of `dim`, so a `dim > 3`
scene is a genuine higher-dimensional simulation projected down to the 2D
detector, not a special case. The one thing that does hard-code a
dimension is `tangential_field`'s rotation plane when `plane_axes` isn't
given explicitly (defaults to the first two coordinate axes) — a
consequence of "tangential" needing a 2D plane to rotate within, not a
limitation on `dim` itself.

### Velocity fields

The core abstraction is a velocity field: a function of
`(positions, velocities, wavelengths, t, dt, rng)` returning *the*
velocity for this step — not an increment to accumulate. Fields are
kinematic: like a fluid flow field advecting passive tracers, they
prescribe velocity directly rather than exerting a force that builds up
over time. Fields compose by addition — combining two fields is just
summing their prescribed velocities, with no special-casing required in
the integrator. A deterministic field (e.g. `radial_field`) ignores `rng`;
a stochastic field (e.g. white noise) ignores position; a
wavelength-independent field ignores `wavelengths` — every field shipped
so far does exactly that, so wavelength coupling is strictly opt-in, never
implicit. `dt` is passed through explicitly so stochastic fields can
discretize correctly (see below).

`white_noise_field` (velocity drawn fresh each step as i.i.d. Gaussian
noise) is the one field outside the geometry × profile × gain system
below — see its docstring in `fields.py` for its Euler–Maruyama
discretization. The rest of the catalog is covered in "Structured fields"
next.

### Structured fields: geometry × profile × gain

The field catalog is built from a small, general system of three
independent pieces, rather than one-off constructors each re-deriving
their own direction/magnitude/singularity handling for a specific use
case. None of the three is required to construct a `Field` (the raw
`Field` callable — `(pos, vel, wavelength, t, dt, rng) -> velocity` — is
still the actual interface; this is one convenient way to build one, not a
shape every field must fit. Perlin noise and the Hopf-fibration
projections on the roadmap won't decompose this way and will implement
`Field` directly when they land):

- A **geometry** factory (`radial_field`, `tangential_field`, `axial_field`,
  `twist_field`) turns position into a scalar *coordinate* and a unit-ish
  *direction* vector: radial distance from a center with the outward
  direction, tangential distance/direction within a rotation plane, a
  linear projection onto an axis with a fixed direction, or (`twist_field`)
  a linear projection onto an axis with a direction that itself rotates
  within a separate plane as that projection grows — see below. `radial_field` and
  `tangential_field` accept an arbitrary `center` (default: the origin, in
  whatever dimension `pos` turns out to be at call time — resolved lazily
  rather than baked in, which is what makes these usable at any `dim >= 2`
  with no `dim` parameter of their own); `axial_field` an arbitrary `axis`
  and (optionally) a separate `direction` — none of these default to or
  assume the coordinate axes, so a linear field pushing along one direction
  while varying with position along a *different* one (a shear flow) is a
  first-class case, not a special one. Unlike `center`, `axis` has no
  dimension-agnostic default to fall back to — a "reasonable random
  direction" needs to know how many components to draw, and that's the
  running `Simulation`'s job (`sim.random_direction()`, sized to
  `sim.state.dim`), not this constructor's; `axial_field` requires `axis`
  explicitly rather than accepting a `dim` to draw one itself, since a
  field constructor has no business knowing which simulation it'll run in.
  `constant_field`'s `velocity` is required for the same reason.
- A **profile** (`constant`, `linear`, `exponential`, `exponential_ramp`,
  `sinusoidal`) is a plain scalar→scalar shape function applied to the
  coordinate, giving the field's magnitude. Every profile's scale
  parameter (`constant`'s `value`, `linear`'s `slope`, `exponential`'s and
  `exponential_ramp`'s `amplitude`, `sinusoidal`'s `amplitude`) defaults
  to `1` — the profile unscaled, with no bias toward either sign. That
  matters because a radially-symmetric vector field is, in general, `f(r)
  * direction` for *any* signed `f` (gravity is `f(r) < 0`, Coulomb
  repulsion between like charges is `f(r) > 0`), so with `radial_field`
  (outward `direction`), a negative scale pulls inward and a positive one
  pushes outward — neither is a special case, and the default
  deliberately doesn't favor one over the other just because one of them
  happens to match a familiar use case (confinement, in this case) — see
  `constant`'s and `exponential`'s docstrings. `exponential` and
  `exponential_ramp` are the same growth curve at two different anchor
  points: `exponential` equals `amplitude` at `coordinate = 0` and grows
  (or decays) from there, never crossing zero; `exponential_ramp` is
  shifted down by `amplitude` so it vanishes at `coordinate = 0` instead —
  the exponential analogue of `linear` (which also passes through the
  origin) rather than of `constant`. For example: `radial_field(profile=
  constant(-1.0))` is inward confinement at constant speed;
  `radial_field(profile=exponential_ramp(amplitude=-1.0))` is exponential
  confinement — its center-anchored zero is what makes it suitable for
  confinement in the first place, since plain `exponential` would instead
  pull at full `amplitude` on a particle sitting exactly at `center`;
  `tangential_field(profile=linear(w))` is rigid-body rotation at angular
  velocity `w`; `axial_field(profile=sinusoidal(...))` is a
  single-wavevector plane wave.
- A **gain** (`Gain = Callable[[wavelength, t, dt, rng], array | float]`,
  defined in `gains.py`) is an optional dimensionless multiplier a profile
  can fold into one of its own parameters. Deliberately excluded from
  `Gain`'s inputs is position: a position-dependent scalar would be
  redundant with the geometry/coordinate step, which is already a function
  of position. See "Gain catalog" below for the full vocabulary and its
  design rationale.

Which parameter a gain multiplies is decided by each profile, not by a
single generic mechanism. `constant`, `linear`, `exponential`, and
`exponential_ramp` each have exactly one scalar knob, so its gain is
unambiguously named `gain`. `sinusoidal` has three independent knobs
worth modulating — amplitude, frequency, phase — so it names each hook
after the parameter it touches (`amplitude_gain`, `frequency_gain`,
`phase_gain`) instead of overloading a single `gain` to mean something
different from what it means everywhere else in the module:
`amplitude_gain` scales the output like the other profiles' `gain`, while
`frequency_gain`/`phase_gain` scale their parameter *before* it enters
`sin`, since that's the only way a modulator can shift *where* the wave's
zeros land (e.g. a per-particle wavelength setting the lattice spacing) —
scaling the output can't reproduce that. `frequency` and `phase` are both
expressed in cycles rather than radians (`phase = 0.25` is a
quarter-period shift), so the two combine by plain addition before the
one conversion to radians `sin` needs, and `frequency_gain`/`phase_gain`
scale genuinely equivalent, same-unit quantities rather than one already-
converted value and one not. Passing the same `Gain` to both
`frequency_gain` and `phase_gain` scales frequency and phase together, as
one wavelength-dependent factor; passing it to only one modulates that
one alone. Each profile factory
resolves its gain parameters once, at construction time, into a closure
with no gain-related branching or array allocation when none are given;
the "no modulation" case costs exactly what it did before this system
existed.

The same convention extends to `white_noise_field`, which isn't built
from a geometry × profile at all — it's a standalone `Field`, like
`constant_field` — but does have a single scalar knob (`sigma`), so it
takes a `gain` that scales it exactly like `constant`/`linear`/
`exponential`'s does, e.g. `white_noise_field(sigma, gain=
wavelength_power_law(exponent=-1))` for shorter wavelengths that diffuse
faster. `constant_field` doesn't get the same treatment: its one
parameter is a fixed vector, not a scalar magnitude, so there's no single
obvious knob for a `gain` to multiply — `modulated()` already covers
scaling its output identically.

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
`radial_field(profile=constant(-1.0))` has no non-decaying speed near the
center: velocity is `direction * profile`, and `direction -> 0` there
regardless of what `profile` returns — a direction magnitude pinned to `1`
all the way to `r = 0` would instead produce a permanent period-2
overshoot bounce at `center`. `tangential_field`'s outward-spiral drift
under explicit Euler is unrelated to this and still applies — see the
Roadmap.

`twist_field(axis, plane_axes, angle, magnitude, phase)` is a fourth
geometry with a different shape from the other three: direction rotates
within `plane_axes` as a function of position along `axis`, but is
constant across the whole plane itself — unlike `tangential_field`,
velocity has no dependence on position *within* the rotation plane at
all. This is the cholesteric liquid-crystal director field, and (frozen
at one instant) the spatial helix a circularly- or elliptically-polarized
plane wave's field vector traces along its propagation axis — a real
structure, not an invented one, with `axis` as the propagation direction
and each wavelength free to twist at its own rate (chromatic optical
activity/circular birefringence, achieved by giving `angle` its own
`gain`) rather than a fixed one. It reuses `Profile` for two independent
roles instead of one, the same reason `sinusoidal` needed three named
gain hooks instead of a single `gain`: `angle` is interpreted as *cycles*
(exactly like `sinusoidal`'s `frequency`/`phase`) and sets the rotation
rate — `angle=linear(rate)` (the default, `rate=1`) is the canonical
constant-pitch helix; a nonlinear `angle` (e.g. `sinusoidal(...)`) gives
an accelerating or oscillating twist instead, a principled but
non-physical extension of the base case. `magnitude` is an ordinary
profile-as-magnitude, the amplitude envelope along `axis` (`constant()`
by default). Unlike the other three geometries, `twist_field` takes no
`center` — a spatial center's only meaningful effect here would be on
`theta`, and for the default linear `angle` that's exactly what `phase`
(in cycles, `sinusoidal`'s convention) already does directly, while
`center`'s component orthogonal to `axis` would silently do nothing at
all (the same latent oddity `axial_field`'s own `center` has, just
starker here); `phase` says "what direction the field points at the
origin" without a mostly-inert vector parameter. `phase_gain`, mirroring
`sinusoidal`'s own `phase_gain`, scales `phase` multiplicatively (so it
needs a nonzero `phase` baseline to do anything, like every profile's
`gain` against a zero-valued parameter) — a time-varying `phase_gain`
(`sine_gain`, `ornstein_uhlenbeck`) spins the whole helix pattern about
its own axis over time, independent of any wavelength coupling on
`angle` itself. The one cost of dropping `center`: no longer being able
to shift where `magnitude`'s envelope is anchored — a niche
enough need, given `magnitude` defaults to `constant()`, that it's better
folded into a custom `magnitude` profile if it's ever needed than
reintroduced as a geometry-level parameter. `axis` has no
dimension-agnostic default, for the same reason `axial_field`'s doesn't:
in 3D the orthogonal complement of a 2-plane is a unique line, but for
`dim > 3` it's `(dim - 2)`-dimensional, so there's no canonical "the
other axis" past 3D. `plane_axes` defaults to the first two coordinate
axes, like `tangential_field`'s default. Correctness of a custom
`axis`/`plane_axes` pairing — they should be mutually orthogonal, or
`coordinate` and in-plane position stop being independent — is the
caller's responsibility, same convention as `tangential_field`'s custom
`plane_axes` and `axial_field`'s `axis`/`direction`; this makes `dim >=
3` a practical requirement, though
nothing checks it explicitly. No singularity to soften here, unlike the
radial geometries: direction never depends on `offset` within the plane,
so there's no `0/0` at any point.

**General rotation generators, and the Hopf fibration, come for free by
composition.** `tangential_field(profile=linear(w))`'s output on its own
plane is exactly `w` times a 90-degree rotation of the in-plane offset,
untouched outside that plane — so summing several instances over disjoint
orthogonal `plane_axes` (via `sum_fields`) reconstructs the action of a
general block-diagonal rotation generator (an element of `so(dim)`) on
the full position vector, with no new geometry needed: `sum_fields`'s
addition *is* the matrix's block-diagonal addition. Giving every plane
the *same* rate is the isoclinic/Clifford case; on any even `dim`,
pairing up all `dim / 2` axes at one shared rate generates the general
unitary Hopf fibration `S^(dim-1) -> CP^(dim/2 - 1)` natively in whatever
dimension the sim is running at — the classical `S^3 -> S^2` Hopf
fibration (`dim = 4`) is just its smallest case, e.g.
`sum_fields(tangential_field(profile=linear(w), plane_axes=(e0, e1)),
tangential_field(profile=linear(w), plane_axes=(e2, e3)))`. No
stereographic projection or subspace-limiting hack is needed for this —
particles trace genuine Hopf fibers at whatever radius they sit, and the
existing orthographic (first-two-axes) projection displays it like
anything else in this project. `twist_field` composes the same way for
the same reason (its direction is likewise confined to its own plane):
summing several instances sharing one `axis` over disjoint orthogonal
`plane_axes`, with each plane's `magnitude` weight `w_i` satisfying
`sum(w_i^2) = 1`, keeps the combined direction unit-length — equal
`angle` rates trace one (diagonally embedded, still one-dimensional)
great circle, while different rates trace a genuinely higher-dimensional
torus/braid.

The literal "linked circles" picture most people mean by "the Hopf
fibration" is a different, dimension-*locked* thing: the stereographic
projection of the S^3 flow above into a field on R^3 specifically — see
`fields.hopf_r3_field` for that one. It's kept separate from the
dimension-general construction above deliberately: it's a specific,
famous artifact of one arbitrary choice of projection pole, not a
canonical structure to generalize from.

`modulated(field, gain)` is the complementary, coarser tool: it rescales an
*already-built* field's total output by a gain, for when you want to tune
wavelength/time dependence from outside without reaching into a profile's
own parameters — e.g. scaling an entire `sum_fields(...)` composition, or
a field you didn't construct yourself. It consumes the same `Gain` type as
every profile's gain parameter, so any constructor from the gain catalog
below works identically in either place.

`normalized(field, softening)` rescales an already-built field's output to
unit magnitude at every point (softened the same way `radial_field`'s
direction is, so it vanishes rather than dividing by zero exactly where
`field` itself is zero). Rescaling a vector field by a positive scalar
function of position never changes its integral curves as geometric
paths — a standard ODE fact — only the speed they're traced at, so
`normalized()` keeps a field's orbits identical while making that speed
uniform everywhere. This is what turns the rigid-body rotation generators
above into genuinely unit-speed ones: `normalized(sum_fields(...))` over
the same disjoint orthogonal `tangential_field` planes preserves their
orbits exactly (including, in the equal-rate case, the Hopf fibers
themselves), while fixing rigid rotation's `speed = w * r` into a
constant particles move at everywhere in frame, rather than crawling
near the origin and shooting through the edges. It has to wrap the whole
sum, not each plane individually: swapping each `tangential_field`'s
`profile` for `constant(...)` instead would make each plane's own
*angular* rate position-dependent (`speed / r_i`), which breaks the fixed
relative phase rate between planes that makes the sum's orbits Hopf
fibers in the first place, whereas normalizing the combined output
leaves every plane's relative phase rate untouched.

### Gain catalog

`gains.py` holds the `Gain` type and its constructors, kept separate from
`fields.py` since the vocabulary (spectral, deterministic-temporal,
stochastic, and stateful gains) is large enough to warrant its own module.
`Gain = Callable[[wavelength, t, dt, rng], array | float]` — `dt` is
carried for the same reason `Field` carries it: a stateful gain needs it
to Euler-Maruyama-discretize its underlying SDE correctly (see below); a
stateless gain just ignores it. Position is excluded, same reasoning as
in "Structured fields" above.

The catalog is organized around which of the three inputs — wavelength,
t, rng — a gain actually reads, which also predicts what it's good for:

- **Spectral** (wavelength only): `wavelength_power_law(reference_nm,
  exponent)` — `(wavelength / reference_nm) ** exponent`, physically
  grounded at `exponent = -1` (photon momentum `p = h/λ`: shorter
  wavelengths push harder under radiation-pressure-like forcing);
  `exponent = +1` favors long wavelengths on a principled but not equally
  fundamental basis. `wavelength_gaussian(reference_nm, sigma_nm,
  amplitude)` — a resonance/bandpass bump, the standard lineshape for a
  single absorption/emission resonance (the mechanism that gives colored
  glass its color); combine multiple via `gain_product` (below) to stack
  resonances.
- **Temporal, deterministic** (t only): `sine_gain(frequency, amplitude,
  phase, center)` and `square_gain(frequency, amplitude, phase, center)`
  — a smooth oscillation and a hard 50%-duty switch between two levels,
  `center + amplitude*sin(2*pi*(frequency*t + phase))` and its duty-cycle
  analogue. `frequency` (periods per unit `t`) and `phase` (fraction of a
  period) are both in cycles for the same reason `fields.sinusoidal`'s
  are — see "Structured fields" above — which matters especially for
  `square_gain`: a square wave's phase is a fraction-of-period offset by
  definition, not an angle, so radians would be a borrowed unit rather
  than the natural one. `square_gain` is computed directly from the
  fractional part of `frequency*t + phase`, not from the sign of `sin`, so
  it has no dependency on trigonometry at all. `linear_gain(rate, center)`
  — `center + rate*t`, matching `fields.linear`'s `slope * coordinate`
  with `t` as the coordinate — grows (or, for negative `rate`, decays)
  unboundedly; unlike `fields.exponential`, there's no overflow risk to
  clamp against.
- **Stochastic, memoryless** (rng only): `gaussian_noise(sigma, center)`
  and `lognormal_noise(sigma)` — i.i.d. per call, no state. `lognormal_noise`
  (`exp(sigma * randn())`) is the canonical choice for a multiplicative
  gain: always positive, median exactly `1` at any `sigma`, and a factor
  of `x` is as likely as `1/x` — the same relationship geometric Brownian
  motion has to ordinary Brownian motion. `gaussian_noise` can flip the
  sign of whatever it multiplies once `sigma` is large relative to
  `center`; that's a deliberate opt-in, not this catalog's default.
- **Stateful** (hold memory across calls): `ornstein_uhlenbeck(theta,
  sigma, center)` and `telegraph(rate, low, high)` — the stochastic
  generalizations of `sine_gain` and `square_gain` respectively.
  Ornstein-Uhlenbeck is the canonical continuous-time mean-reverting SDE
  (`dx = -theta*(x - center)*dt + sigma*sqrt(dt)*dW` — `center` plays the
  role the standard SDE notation calls `mu`, renamed to match `sine_gain`/
  `square_gain`/`gaussian_noise`'s convention for the same resting-value
  role — discretized exactly like `fields.white_noise_field`) — in
  gain-space, the same shape as an `exponential_ramp`-confined field plus
  white noise, composed instead around a resting gain value. `telegraph`
  is the random telegraph process / dichotomous Markov noise: switches
  between `low` and `high` at Poisson-arrival times (rate `rate`) instead
  of a fixed period, the standard model for e.g. ion channel gating.

Every constructor defaults to its own mathematically canonical shape —
`sine_gain`/`square_gain`/`linear_gain` zero-centered,
`ornstein_uhlenbeck` resting at `center=0`, `telegraph` switching around
`0` — rather than one pre-tuned
to "neutral at 1", which is a property of *using* a gain multiplicatively,
not of the shape itself. Pass `center=1.0` (or `low`/`high` straddling
`1`) explicitly to get that. `lognormal_noise` and
`wavelength_power_law` are the exceptions: their neutral-at-1 behavior is
a structural consequence of their formulas (exponentiating a zero-mean
Gaussian; evaluating a power law at its own reference point), not a tuned
default, so neither needs a `center` parameter.

Multiple gains combine via `gain_product(*gains)` (multiplying their
outputs — the natural composition rule for dimensionless multipliers,
matching Beer-Lambert absorption where stacked absorbers multiply
transmittances) or `gain_sum(*gains)` (adding their outputs — for
independent additive signals rather than multiplicative factors, the same
rule `sum_fields` uses to compose `Field`s). Either lets a profile's
single gain slot be driven by more than one independent effect, e.g.
`gain_product(wavelength_power_law(), sine_gain(frequency=0.5,
center=1.0))` for a field that's both wavelength- and time-modulated, or
`gain_sum(linear_gain(rate=0.1, center=1.0), sine_gain(frequency=0.5,
amplitude=0.2))` for a gain that trends upward while oscillating around
that trend. A
generic pair of "lift" combinators (an affine `center + scale * signal`
and a multiplicative `exp(scale * signal)`) were considered as a way to
build every centered/rescaled variant from one canonical zero-centered
atom, but rejected as premature abstraction: expressive, but verbose for
what's actually needed today. Each constructor takes `center`/`amplitude`
(or `low`/`high`) directly instead; a generic lift can be added later if
enough constructors end up wanting one to justify it.

Stateful gains hold their state in a closure, which only works correctly
if each call corresponds to a distinct forward step in time. That holds
by default — every `Field`/`Profile` calls its children exactly once per
`Simulation.step()`, and `Simulation.t` only ever advances — with one
documented exception: `sinusoidal` explicitly supports passing the *same*
`Gain` instance to both `frequency_gain` and `phase_gain`, which calls
that instance twice within a single step. `gains._stateful` (the shared
helper both `ornstein_uhlenbeck` and `telegraph` are built on) handles
this by caching on `t`: a repeat call at a `t` it's already seen replays
the cached value instead of advancing the state again, so sharing a
stateful gain across multiple hooks is safe. Comparing `t` by exact
float equality is safe here specifically because the same `float` is
threaded unmodified through the whole call tree within one step — no
floating-point drift can occur between the calls being deduplicated.

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

`Detector`'s `half_extent` is the world-space half-*width* it covers; the
half-*height* is derived from it via the pixel aspect ratio
(`height / width`) rather than reused directly for both axes, so pixels
are always square — a non-square detector (e.g. a 16:9 `sim.record()`
export) doesn't stretch the image.

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
   the sRGB gamut, producing negative linear-sRGB components — no triangle
   spanned by 3 real (non-negative-power) primaries can contain the full,
   convex curve of spectral colors, so some part of that curve always
   falls outside any 3-primary gamut, sRGB's included. Rather than
   clipping the negative component away (discarding real color
   information), it's desaturated toward white: mixed with white by just
   enough to bring every channel to `>= 0`, which is the standard
   technique for rendering the spectral locus into a limited gamut. This
   narrows the practical cost (measured peak-channel contrast between
   "primary-like" and "in-between" wavelengths across the visible range
   drops from ~5.9x to ~4.0x) but can't eliminate it: the rendered colors
   are still less saturated than the true spectral colors an eye sees
   directly, the same way a photograph of a rainbow looks on an sRGB
   monitor. See `color._desaturate_to_white`.
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

Module layout:

- `prismswarm/fields.py` — the `Field` interface; geometry factories
  (`radial_field`, `tangential_field`, `axial_field`, `twist_field`) and profiles
  (`constant`, `linear`, `exponential`, `exponential_ramp`, `sinusoidal`)
  that combine into structured fields; `modulated` for `Gain`-based
  modulation of a whole field and `normalized` for rescaling one to unit
  magnitude; the standalone `constant_field`, `white_noise_field`, and
  `hopf_r3_field`; and the additive composition helper (`sum_fields`).
  See "Structured fields: geometry ×
  profile × gain" above.
- `prismswarm/gains.py` — the `Gain` type and its constructor catalog
  (spectral, deterministic-temporal, stochastic, and stateful), plus the
  `gain_product` composition helper and the `_stateful` memoization helper
  stateful gains are built on. See "Gain catalog" above.
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
  the particle population from a fresh Gaussian cloud using the stored
  `rng` and `spectrum` — the recovery path for a swarm that has wandered
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
- `prismswarm/export.py` — `record(sim, duration_s, path, ...)` (exposed
  as `sim.record(...)`) renders a run to an MP4 by piping raw RGB24 frames
  to the `ffmpeg` binary (required on `PATH`; not a pip dependency, so not
  in `pyproject.toml`). Explicitly does not step `sim` itself — see
  "Threading model" below for why — instead cloning `sim.state` (a plain
  array copy), spawning an independent child of `sim.rng`
  (`Generator.spawn()`, so it can't race draws the live render loop makes
  against the same shared generator), and building its own `Detector`,
  then stepping that clone as fast as the calling thread can compute,
  uncoupled from real-time. `sim.fields`/`sim.spectrum` are read-only
  callables, shared directly with no cloning needed.
- `prismswarm/main.py` — entry point wiring the above together and holding
  the initial scene setup.

**Threading model:** the main thread owns the pygame window and the
render/integration loop (SDL/pygame windowing needs to own the thread it
was created on, particularly on macOS). The REPL runs on a background
thread and mutates shared state — e.g. swapping the active field list —
through simple reference/list reassignment rather than fine-grained
locking, relying on the GIL for atomicity of those individual ops. This
keeps the REPL responsive without stalling the render loop, and vice
versa. That model covers single attribute reads/writes, not a tight loop
of calls — which is exactly what recording a video needs, since it must
call `state.step()` and `detector.splat()` many times in a row. Since the
REPL is where `sim.record()` is meant to be called from, and the render
loop is concurrently doing exactly that same kind of tight-loop stepping
on `sim` itself, `export.record()` never touches the live `sim.state` /
`sim.detector` / `sim.rng` at all — it works from clones, by construction
outside this model's guarantees rather than straining them.

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
- `radial_field` and `white_noise_field`, independently selectable at
  runtime
- Orthographic projection; detector buffer decoupled from display
  resolution
- CIE XYZ→sRGB pipeline with gamut mapping and manual + adaptive
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
  interface carries `wavelengths`; the `gains.py` catalog (`Gain`
  constructors — spectral, temporal, stochastic, and stateful, see "Gain
  catalog" above) plugs into any profile's gain parameter or into
  `modulated()`, composable with `sum_fields` and, within a single gain
  slot, with `gain_product`.
- Explicitly deferred within this milestone: dynamic per-particle spectra
  (random walks, explicit spectral conversion) — a later milestone once
  static spectra and field coupling are both working

**M3 — Field catalog & composition**
- Reworked the catalog from one-off named constructors into the
  geometry × profile × gain system (done — see "Structured fields" above):
  `radial_field`, `tangential_field`, `axial_field` geometries; `constant`,
  `linear`, `exponential`, `exponential_ramp`, `sinusoidal` profiles; the
  `gains.py` catalog +
  `modulated()` for wavelength/time/stochastic gain (see "Gain catalog"
  above). The old lattice-forming sinusoidal field (a per-axis product of
  sines, geometrically distinct from `axial_field`'s single-wavevector
  plane wave) was deleted rather than kept alongside the new system — a
  deliberate prototype, not a regression.
- Added `twist_field` (see "Structured fields" above): a fourth geometry
  whose direction rotates within a plane as a function of position along
  a separate axis, rather than depending on in-plane position like
  `tangential_field` — the cholesteric liquid-crystal director field /
  the spatial helix a frozen circularly-polarized wave traces.
- Added the Hopf fibration, in two forms (see "Structured fields" above):
  the dimension-general isoclinic rotation flow — free by composing
  `sum_fields` over disjoint orthogonal `tangential_field` planes at
  equal rate, generalizing to `S^(dim-1) -> CP^(dim/2 - 1)` on any even
  `dim` — and `hopf_r3_field`, the famous stereographically-projected
  "linked circles" picture, dimension-locked to R^3 and kept separate as
  the celebrity case rather than blended into the canonical one.
- Still to add: Perlin noise, rectilinear projections
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
- Pinhole camera projection with distance-based brightness falloff
  (`d^(1-n)` is natural for a pinhole model; the orthographic falloff law,
  if any, is TBD)

**Lower priority (not excluded)**
- Spectral (non-pure-wavelength) emission per particle
- Particle interactions via a Barnes-Hut-style approximation (O(n²) is a
  non-starter at this scale)
- Force-based dynamics (e.g. Verlet) as an alternative to velocity-field
  integration
