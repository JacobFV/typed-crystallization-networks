# External environments under the generator contract

Research track `external-environments`, 2026-09-09. Everything below was
produced in this workspace with
`/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python`
(Python 3.13.15, `mujoco` 3.12.0). **Nothing under `tcn/` was modified and no
existing generator was changed.** The three core changes this track would ask
for are written as diffs in section 7 and have not been applied. No package was
installed.

New files: `generators/control/` (the working environment),
`tests/test_control_generator.py` (25 tests), and this directory. The full
suite is **164 passed** with the new generator in the tree.

Reproduce with:

```
.venv/bin/python research/external-environments/operator_legality.py
.venv/bin/python research/external-environments/replay_check.py
.venv/bin/python -m pytest tests/test_control_generator.py -q
```

---

## 0. Verdict

**An external environment fits the contract, and the fit is not free.**

A MuJoCo control task now satisfies `ARCHITECTURE.md` section 6 end to end:
typed observations, an observation/latent/probe split enforced at the actor
boundary, named random streams, a logical clock that drives the simulator's
physical clock, recorded exogenous inputs, and **complete state restoration
verified bit-for-bit across a process boundary**, not assumed.

Three things it cost, all measured:

1. **The typing is authored, not derived.** Gym gives a `Box` — a shape and a
   dtype. Every unit, frame, encoding and bound in section 4 was written by
   hand, and a wrong one fails silently rather than loudly.
2. **77% of a step is contract, not physics.** 0.394 ms of MuJoCo integration
   inside a 1.68 ms typed step.
3. **The contract is only as strong as the engine.** Restoration works because
   MuJoCo exposes `mjSTATE_INTEGRATION`. For an environment whose engine does
   not serialize (Box2D through gym, for one), no wrapper fixes it.

And one thing it exposed that is not specific to this track: `source_fingerprint()`
hashes all of `tcn/` and `generators/`, so **adding any generator invalidates
every previously recorded episode**. `artifacts/system/*/episode.json.gz` already
fails `Host.restore` with `episode source revision mismatch` — verified to be
already failing *before* `generators/control` existed, so this track did not
cause it, but this track would have. Diff in section 7.3.

NES is specified, not built (section 6). It is a good fit for a reason this
repository has already measured: `eq` is the only predicate available on pixels,
and NES output is flat palettized colour with no anti-aliasing, which is exactly
the regime where `eq` is informative. OAM supplies the object identity that
`research/object-identity/RESULTS.md` proved is *structurally* unrecoverable
from this repo's own renderer.

---

## 1. The contract audit: what a gym-style environment does not provide

`gym.Env` is `reset(seed) -> obs, info` and `step(action) -> obs, reward,
terminated, truncated, info`. Section 6 asks for strictly more. Each row is a
gap, its severity, and whether it is bridgeable.

| # | Section 6 requirement | What gym provides | Bridgeable? |
|---|---|---|---|
| G1 | Complete state restoration | Nothing. `reset()` returns to a *sampled* initial state, not an arbitrary one. There is no `get_state`/`set_state` in the API. | **Only if the engine serializes.** MuJoCo: yes (`mj_getState`/`mj_setState`, `mjSTATE_INTEGRATION`) — verified bit-identical here. ALE: yes (`cloneSystemState`). Classic control: yes (state is a 4-vector). Box2D (LunarLander, BipedalWalker): **no** — contact and island state are internal and gym exposes no serialization. libretro/stable-retro: yes (core serialization). **Must be verified per environment by a round-trip test; never assumed from the fact that a `state` attribute exists.** |
| G2 | Explicit random streams | One `self.np_random` per env, consumed by `reset` and by any stochastic dynamics. All draws share one stream. | **Yes, but only by reimplementing the sampling**, not by wrapping. A wrapper cannot separate streams it does not own. Measured counterfactual in section 3.5: with one shared stream, drawing a goal before the initial pose *moves the pose*; with named streams it does not. Where the randomness is inside an opaque engine (an emulator's own PRNG), the only available guarantee is pinning the whole seed and capturing it in the state. |
| G3 | Logical clock | None. The timestep is baked into the env and `step()` takes no `dt`. Worse, common wrappers read host time (`RecordVideo`, `Monitor`, real-time rendering). This is the exact failure `docs/LESSONS.md` records for the computer runtime: *"Seeded network randomness does not make the entire episode replayable."* | **Yes.** Take `dt` from the host and derive everything from it. Here the substep count is a pure function of `dt` (`test_substep_count_is_a_function_of_dt_alone`), and `data.time` is driven only by accumulated `dt` (`test_the_logical_clock_drives_the_physical_one`). For a fixed-rate engine (NES at 60.0988 Hz) `dt` must be quantized to whole frames and a non-integral `dt` rejected, not rounded. |
| G4 | Typed observations | `Box(low, high, shape, dtype)`. A shape and a dtype. No unit, no frame, no role, no declared encoding contract, no refinement distinct from clipping. | **Yes, but the typing is authored.** Nothing in `Box(-inf, inf, (3,), float32)` says which entry is an angle and which is an angular velocity. The schema is written by hand per environment and a wrong entry is silently wrong. This is the largest real cost of the whole exercise and it does not go away with tooling. |
| G5 | Observation / latent / probe split | `obs`, plus an **untyped `info` dict**. Privileged truth has exactly one place to go, and nothing marks it privileged or stops a training loop reading it. This is `docs/LESSONS.md`'s `payload: Any` finding verbatim: *"A uniform envelope alone does not establish uniform semantics."* | **Yes for engines that expose their state**, no in general. Here the split is real: `qpos`/`qvel`/`physical_time` are latents at float64, `site`/`kinetic_energy`/`potential_energy`/`upright` are probes, and `StepRecord.actor_view` cannot reach either (`test_privileged_truth_is_not_in_the_actor_view`). For a generic gym env, the privileged fields usually do not exist to be split off. |
| G6 | Typed reward components | One float. The component structure exists in the env's source and is discarded at the boundary. | **Only by reimplementing the reward.** Done here: `upright` / `rate` / `effort` are three separate typed scalars, which is what section 8's `L_RL` and objective-conflict reporting need. |
| G7 | `available_actions`, per actor, state-dependent | `action_space`, fixed for the episode. Action masking is an ad-hoc `info` key with no contract. | **Yes.** Cheap. |
| G8 | Pinned versions | Env ids carry a version suffix (`v4`, `v5`); the **engine** version is not pinned or recorded. MuJoCo trajectories change between minor versions. | **Not by wrapping, and not currently by this repo either.** `source_fingerprint()` covers repository code only. An episode recorded under `mujoco` 3.12.0 restores without complaint under a different build. Diff in section 7.1. |
| G9 | Declared numerical reproducibility limits | Nothing. | **Yes**, and it must be stated per engine: MuJoCo is deterministic for a fixed model, state, `ctrl` and timestep *on one build and one platform*. Cross-platform bit-identity is **not** claimed here and was not tested. |
| G10 | Held-out generating structure (section 9) | Not applicable — a gym env *is* one fixed structure. Seeds vary the initial state, not the mechanism. | **Not bridgeable.** This is a provenance limit, not an API limit, and it is why section 5 proposes an evidence tier. A generalization claim over held-out *structures* cannot be made on a fixed external environment at all. |

Two gaps are load-bearing and the rest are labour: **G1** decides whether an
environment can enter this repository at all, and **G10** decides what a result
on it is allowed to claim.

---

## 2. The environment: `generators/control`

A MuJoCo control generator with two tasks, chosen at pendulum/reacher scale as
directed. It is a sibling under `generators/` with the standard
`manifest.json`, and the trainer needs no branch for it.

| | `pendulum` | `reacher` |
|---|---|---|
| Mechanism | one hinge, gravity, `damping=0.05` | two planar hinges, no gravity |
| Integration state width | 18 floats | 29 floats |
| Actuator | 1 motor, declared limit ±2.0 N·m | 2 motors, declared limit ±1.0 N·m |
| Observations | `angles`, `orientation`, `rates` | `angles`, `orientation`, `rates`, `to_target` |
| Flat observation width | **4** | **10** |
| Latents (float64) | `qpos`, `qvel`, `physical_time` | same |
| Probes (float64) | `site`, `kinetic_energy`, `potential_energy`, `upright` | `site`, energies, `target`, `distance` |
| Rewards | `upright`, `rate`, `effort` | `proximity`, `rate`, `effort` |

Design points worth naming, because each is a place where the obvious
implementation would have broken the contract:

- **The action limit lives in the type, not in the MJCF.** `torque_type(limit)`
  returns `floating(32, unit="N*m", frame="joint", bounds=(-limit, limit))`. The
  model's `ctrlrange` is deliberately wider (±100), so a torque outside the
  declared bound raises from `Value.of` instead of being silently clipped by the
  simulator. `tcn/policy.py:numeric_bounds` reads exactly this field, so a typed
  policy emits in-range torques with **no environment-specific action head**
  (`test_typed_policy_emits_in_range_torques_with_no_domain_action_head`).
- **The reaching target is generator state, not a body**, so the compiled model
  does not vary per episode and the model cache is a pure function of the XML.
- **The disturbance is an exogenous input, recorded, not a hidden draw.** It is
  drawn from the host's per-tick stream, clipped, stored in `state["exogenous"]`
  and emitted in the typed `transition` record. It is deliberately *not* bounded
  by the actuator limit, because it is not a command.
- **The angle is wrapped for observation and unwrapped in state.** A joint winds;
  its observation must not. Verified both ways.
- **No host clock is read anywhere.** `data.time` is advanced only by the host's
  `dt`.

The pendulum is a genuine under-actuated swing-up, not a stub: a two-line
energy-pumping controller reaches `upright = 0.9994` from rest, and the
zero-torque arm never exceeds `-0.99` (`test_the_pendulum_is_a_real_swing_up_problem`).
Mean `upright` over the first ten steps is −0.645 and the peak is 0.999.

---

## 3. Replay, verified

`research/external-environments/replay_check.py`, raw output in
`out/replay_check.json`. Every boolean below is `true`.

### 3.1 Recorded-input replay
`Host.replay()` re-runs the episode from the recorded action/`dt`/`random_input`
rows and asserts digest equality itself. It passes with the disturbance stream
active.

### 3.2 Snapshot round trip, in process
`Host.restore(host.snapshot())` reproduces the digest, and **the restored
episode continues identically**: after eight further steps,
`max |state_i - state_i'| = 0.0` over the 18-float integration vector, and the
lists compare equal by `==`, i.e. bit-identical, not close.

`tests/test_control_generator.py:test_snapshot_round_trip_continues_bit_identically`
strengthens this: ten further steps at *varying* `dt` (0.02 … 0.11), comparing
the raw integration state after every single step, on both tasks.

### 3.3 Cross-process
The strongest arm, and the one that would catch a cached-object or
warm-start-residue bug that an in-process restore would not: the episode is
saved to `out/episode.json.gz`, reloaded by `continue_episode.py` in a **fresh
interpreter**, continued eight steps, and compared. Digest matches; integration
vector matches element for element.

### 3.4 Address determinism
Two hosts at the same `Address` are digest-identical; a different seed differs.

### 3.5 Named stream separation, with its counterfactual
`initialize` draws the goal from `address.rng("layout")` and the initial pose
from `address.rng("start")`. Supplying an explicit target through `objective`
changes the goal and leaves the initial integration state **exactly** unchanged.
The counterfactual is measured in the same script: with one shared stream,
drawing the goal first changes the pose that follows. That is gap G2 made
concrete — a single-stream environment cannot vary its goal distribution without
silently varying its initial-state distribution too.

### 3.6 Visibility
`actor_sees = ['angles', 'orientation', 'rates']`. `latent_states` and `probes`
are absent from `ActorView` by construction, and the visible set is disjoint
from both. On `reacher` the *absolute* target is a probe and only the relative
`to_target` vector is observable — the goal coordinate is privileged truth, the
egocentric offset is not.

### 3.7 Cost

| measurement | seconds |
|---|---|
| raw MuJoCo integration, 10 substeps of 5 ms | 3.94e-4 |
| one `physics.readout` (fresh `MjData` + `mj_forward`) | 3.79e-4 |
| full `Host.step` (pendulum) | 1.68e-3 |
| full `Host.step` (reacher) | 2.01e-3 |
| **contract overhead fraction** | **0.766** |

Two `readout` calls per step (before/after, for the typed `displacement`
transition) account for most of it, each allocating a fresh `MjData`. This is
the same shape as the finding in FINDINGS section 3 about interpreter overhead
dominating a `cost = 4` proxy: the honest number for an environment step here is
1.7 ms, not 0.4 ms. Caching `MjData` per generator instance would recover most
of it and was **not** done, because a cached mutable handle is exactly how a
restore-continuation bug gets introduced, and this track's whole point was to
verify restoration.

### 3.8 What is *not* claimed
Cross-platform and cross-`mujoco`-version bit-identity were not tested. The
declared reproducibility limit is: identical on one build of MuJoCo 3.12.0 on
this host. G8 is why that is currently unenforced.

---

## 4. What the observation typing costs, measured against the registry

`operator_legality.py` asks `tcn.operators.Registry.resolve` about every
candidate encoding. `y` = the signature resolves; `.` = it raises. Full output
in `out/operator_legality.json`.

```
                                       sin  cos  exp  log sqrt  add  sub  mul  div  pow atan2  mod  eq   lt  sum mean pack
float32 plain (orientation, shipped)     y    y    y    y    y    y    y    y    y    y     y    .   y    y    y    y    y
float32 unit="rad"  (the mistake)        .    .    .    .    y    y    y    y    y    .     .    .   y    y    y    y    y
float32 unit="rad/s" (rates, shipped)    .    .    .    .    y    y    y    y    y    .     .    .   y    y    y    y    y
float32 bounds=(-1,1)                    y    y    y    y    y    y    y    y    y    y     y    .   y    y    y    y    y
fixed int[16] scale 4096                 y    y    y    y    y    y    y    y    y    y     y    .   y    y    y    y    y
int[16] plain integer                    .    .    .    .    .    y    y    y    .    .     .    y   y    y    y    .    y
int[8] role="byte" (pixels)              .    .    .    .    .    .    .    .    .    .     .    .   y    .    .    .    y
```

Read the last row first. **A byte-roled pixel admits `eq` and `pack` and nothing
else** — no arithmetic, no ordering, no reduction. That is the boundary
`research/perception-ladder/RESULTS.md` hit and `research/object-identity/RESULTS.md`
worked around with `pack`. This generator does not go near it: every leaf of
every observation is asserted `role == ""`, `numeric == True`, and
`encoding.kind == "float"` (`test_observations_are_numeric_and_never_byte_roled`).

### 4.1 Why floating, and not fixed

`floating(32)`. The `fixed int[16] scale 4096` row shows fixed-point is
**equally legal** — it is not in `{"category", "symbol", "byte"}`, so it is
`numeric`, and the analytic guard is `encoding.kind != "integer"`, which
`"fixed"` passes. So the choice is not about legality. It is:

- **Range.** `fixed(16, 4096)` covers ±8.0 with `overflow="error"`. Measured:
  the energy-pumping swing-up reaches **|qvel| = 40.49 rad/s**, five times that
  range. The episode would die on an `OverflowError` raised inside `encode`, in
  the observation channel, mid-rollout — not a clipped observation, a crashed
  episode. Floating has no such cliff. (Widening the scale to fit 40 rad/s costs
  the angular resolution that matters near upright; that is the fixed-point
  trade, and it is a real one, not a formality.)
- **Precision uniformity.** Fixed-point resolution is absolute (2.4e-4 here);
  float32 resolution is relative (2.6e-8 measured against the float64 latent).
  For a quantity that spans small angles near upright and large rates during a
  swing, relative precision is the right contract.
- **A plain `int[16]` is not an option at all**: `integer` encoding fails the
  analytic guard, so `sin`, `cos`, `div`, `atan2` and `mean` are all illegal.
  Quantising a joint angle to a plain integer would silently delete
  trigonometry from the algebra.

Fixed-point remains the right answer for a bounded, coarsely-resolved channel —
a normalized command, a tile coordinate — and section 6 recommends it for NES
sub-tile scroll.

### 4.2 What declaring a unit costs, and why it is still right

Rows 1 and 3 differ only by `unit`. Declaring `unit="rad/s"` removes `sin`,
`cos`, `exp`, `log`, `pow` and `atan2`. That is correct: those operators require
a dimensionless argument and `sin(6.27 rad/s)` is nonsense. But the registry
enforces it *strictly*, and the consequences are sharper than they first look:

- `add(rate, angle)` → `TypeError`. Types must be equal, and unit is part of the
  type.
- `mul(rate, rate)` → unit `(rad/s)^2` and `bounds` dropped. So a quadratic cost
  term does not live in the same type as a linear one.
- **No conversion operator can change a unit.** `encode`/`decode`/`quantize`/
  `dequantize` all `require(ts[0].unit == output.unit and ts[0].frame == output.frame and ts[0].role == output.role)`.
  A united channel is therefore a one-way room.
- **The single legal exit is division by the same unit.** Measured:
  `div(rate, rate)` resolves, and its output type is **exactly equal** to the
  shipped `OBS_ANGLE`. So the idiomatic normalization — divide an angular
  velocity by a trainable constant of type `floating(32, unit="rad/s", frame="joint")` —
  lands the result in the same dimensionless joint-frame algebra the angle
  lives in, where it can be added to a cosine. `tcn/graph.py` accepts such a
  constant (it only requires `numeric`).

This is the type system doing exactly what section 1 asks of it, and the cost is
that a scaffold author must *know* to put a `div` there. It is worth saying
plainly: **angles are declared dimensionless (`unit=""`), because radians are
dimensionless.** Tagging them `unit="rad"` — the intuitive thing — would have
made `sin` and `cos` on a joint angle type-illegal, which is row 2.

### 4.3 What declaring `bounds` costs

The `bounds=(-1,1)` row resolves everything, so the cost is invisible at
`resolve` time and appears at execution:

```
add(0.8, 0.7) on bounds=(-1,1)  ->  ValueError: value outside semantic bounds
add(0.8, 0.7) unbounded         ->  1.5
```

A refinement turns a total operator into a partial one. `tcn/search.py` treats
that as an unusable candidate (correct); a gradient run hits it as a mid-rollout
exception. So **bounds belong on values the mechanism guarantees, not on values
a program might compute**: they are declared on the *action* type, where the
actuator limit is a real physical constraint and where `policy.numeric_bounds`
consumes them, and left off the observation types, where a downstream `add`
would be legal but fatal. `frame` is used the same way, as a free separation:
`unit="m", frame="world"` lengths and `frame="joint"` angles cannot be mixed by
accident, at no legality cost within each family.

---

## 5. Provenance: the conflict, stated explicitly

`AGENTS.md` line 42 says **"All initial training/development data is synthetic."**
`ARCHITECTURE.md` section 6 says **"All content is synthetic; generated language
and images are projections of known state/programs."** Section 9 closes with
**"Synthetic success does not establish natural-data transfer."**

`generators/control` is a deliberate departure and should be accepted or
rejected as one. Two observations first, because they change what is actually
being decided:

1. **The departure is already latent.** `generators/world_3d/physics.py` imports
   `mujoco` and `generators/computer` shells out to a Node/TypeScript engine.
   Neither is content this repository generates or audits. So the line between
   "synthetic" and "external" was already crossed for *mechanism*; what
   `control` adds is an environment whose external mechanism is the point rather
   than the scenery.
2. **NES is a different kind of departure from MuJoCo, and collapsing them
   would be the mistake.** A MuJoCo episode is drawn from a distribution
   declared here — the initial pose, the goal, the horizon and the disturbance
   are all this repository's. A NES episode's *content* is a fixed third-party
   artifact that cannot be regenerated, cannot be held out by construction, and
   cannot be redistributed. One is synthetic data from a borrowed simulator; the
   other is a found object.

Hence a three-tier proposal rather than a yes/no.

### 5.1 Proposed amendment to `AGENTS.md`

Replace the last sentence of the bullet at lines 40–42 (**not edited here**):

> - Use explicit logical time and randomness, versioned semantic configuration,
>   and complete replay state. Check copied code for host-clock and live-data
>   dependencies. All initial training/development data is synthetic. **A pinned
>   deterministic external simulator may serve as a generator mechanism when it
>   exposes complete state restoration, reads no host clock and no live data,
>   and has that restoration verified by a replay round-trip test; its episodes
>   remain synthetic because the distribution is declared here. Third-party
>   authored content — game ROMs, recorded corpora, captured sensor data — is
>   not synthetic under this rule. It is user-supplied, never redistributed with
>   this repository, verified by digest, and declared in the episode record; a
>   result obtained on it may not be reported as a synthetic-data result.**

### 5.2 Proposed amendment to `ARCHITECTURE.md` section 6

Replace *"All content is synthetic; generated language and images are
projections of known state/programs."* with:

> All content is generated rather than captured, and every episode declares its
> **provenance tier** (section 9). Generated language and images are projections
> of known state/programs. A generator whose mechanism is an external engine
> declares that engine and its pinned version in its manifest, states its
> numerical reproducibility limits, and is admissible only once a replay
> round-trip has been verified against the engine's own state, not merely
> against the record format.

### 5.3 Proposed amendment to `ARCHITECTURE.md` section 9

Append to the final paragraph:

> Evidence is graded by provenance in three tiers, declared per episode.
> **Tier 1, generated here:** mechanism and distribution are both owned by this
> repository; held-out generating structures are available and a
> structural-generalization claim is meaningful. **Tier 2, externally
> simulated:** a pinned deterministic external engine supplies the dynamics
> under a distribution declared here; the content is synthetic, but the dynamics
> are unaudited by this repository and a fault in them is a fault in the result.
> **Tier 3, third-party authored:** the content is a fixed human-authored
> artifact this repository cannot regenerate, hold out by construction, or
> redistribute. Tier 3 admits held-out initial states and goals only, never a
> held-out generating structure, so a generalization claim on tier 3 is strictly
> weaker than the same number on tier 1 and must be reported with its tier.
> Synthetic success does not establish natural-data transfer, and neither tier 2
> nor tier 3 success establishes it: an emulated game is not the natural world.

Under this, `logic`/`arithmetic`/`geometry`/`language` are tier 1;
`control`, `world_3d`, `embodied_world` and `computer` are tier 2; a NES
generator would be tier 3. **This is a proposal. The two documents were not
edited.**

---

## 6. NES: design specification (not built)

Specified only, as directed. Nothing under `generators/` implements this and no
package for it is installed.

### 6.1 Why NES specifically suits this substrate

This is not a preference, it follows from two measured results in this
repository.

- **`eq` is the only predicate available on a pixel** (section 4, last row), and
  `research/discrete-perception/RESULTS.md` certified four candidate `eq`
  families empty on the `geometry` renderer at rung 4. The reason is in
  `research/object-identity/RESULTS.md`: the renderer paints
  `clip(base_colour * shade)`, so one object carries **3.31 distinct RGBs on
  average** and 65.4% of objects are multi-coloured. Equality between two pixels
  of one object is simply false. **NES has no shading and no anti-aliasing**:
  every pixel of a background tile or a sprite is one of at most four palette
  entries, and two pixels of the same tile with the same palette are *exactly*
  equal. The predicate the algebra actually offers becomes informative.
- **OAM supplies exact object identity.** `research/object-identity/RESULTS.md`
  proves the integer label `object_ids` is unrecoverable from the `geometry`
  image *at any context size, structurally*: permuting the object list leaves
  the image bit-identical and changes the labels in 8/8 episodes, so the image
  determines the partition and nothing about the names. NES sprite OAM gives
  sprite index, tile id, x, y and attributes directly — the names, not just the
  partition. That is the missing privileged channel, and it is exactly the sort
  of label section 7 wants: rich, exact, and *not* an agent input.

### 6.2 Package and the state-restoration argument

Two candidates:

| package | savestate | verdict |
|---|---|---|
| `nes-py` (+ `gym-super-mario-bros`) | `_backup()`/`_restore()`, a **single slot**, RAM-oriented | Not sufficient. One slot cannot express an arbitrary snapshot, and the backup is not a documented complete serialization of PPU/APU/mapper state. |
| **`stable-retro`** (maintained fork of `gym-retro`), libretro NES core | `em.get_state()` / `em.set_state()` returning the core's **full serialized state** | **This one.** Complete CPU + PPU + APU + RAM + mapper state as an opaque byte string, which is precisely `mjSTATE_INTEGRATION`'s role here. |

So G1 is satisfiable the same way it was satisfied for MuJoCo: the generator
state carries the emulator's own serialized state, and `advance` is
`set_state -> run N frames -> get_state`. It must be verified, not assumed, by
the same test as section 3.3 — save, reload in a fresh interpreter, continue,
compare byte for byte.

Two engineering consequences:

- **Size.** A NES savestate is tens of kilobytes. `Host.snapshot` JSON-encodes
  the whole state, so a base64 savestate per step is unaffordable. The design:
  store **one savestate at episode start** plus the recorded input sequence
  (deterministic core ⇒ exact replay), with periodic checkpoint savestates at a
  declared interval for random access. This is a real deviation from `control`,
  where the whole state is 18 floats, and it should be declared as such rather
  than discovered.
- **ROM identity is not in the fingerprint.** `source_fingerprint()` hashes
  repository code. A different ROM under the same configuration would restore
  silently. Fix without a core change: put `rom_sha1` in the *configuration*,
  which is inside the digest, and verify it in `configure()`. **ROMs are never
  committed, never distributed, and never downloaded by this repository.** They
  are user-supplied by path, digest-checked, and their absence is a skipped
  test, not a failure.

### 6.3 The logical clock

The NES runs at 60.0988 Hz and accepts one controller state per frame.
`dt` is therefore **counted in frames**, not seconds: `dt = 1.0` means one
frame, `frame_seconds = 1/60.0988` is a declared constant, and wall-clock
seconds is a latent, not the clock. A `dt` that is not a positive integer number
of frames is **rejected**, not rounded — rounding is how G3 becomes a silent
replay divergence.

### 6.4 Channel assignment

**Observations (agent-visible).** The framebuffer only.

- `framebuffer`: 256 × 240 palette indices. The NES emits an index into a 64-entry
  master palette, not a colour, so the honest carrier is
  `integer(6, signed=False, role="category")` — an unordered ID, and section 1 is
  explicit that a category ID must not be reinterpreted as a scalar measurement.
  Cost, measured with `Type.width`: **368,640** flattened floats, because
  `Value.flat` expands a `category` to one float per bit. `role="byte"` (what
  `tcn.generation.image_value` uses) gives **61,440** instead, at the price of
  encoding an unordered index as a magnitude. Neither is usable raw — for
  comparison, `world_3d` at resolution 24 is 1,731 and this track's `control`
  observation is **4**.
- Therefore the generator declares a **`decimation`** configuration exactly as
  `world_3d` declares `resolution`: one sample per 8×8 tile cell gives a 32 × 30
  grid, **960** (byte) or **5,760** (category) floats. This is a declared
  representation choice in versioned configuration, not a learned feature
  extractor, and it is the same precedent `world_3d` already sets. It must be
  recorded in the episode, and a claim measured at one decimation is not a claim
  at another.
- `palette`: the 32 active bytes at PPU `$3F00–$3F1F`. On screen, therefore
  observable, and without it the palette indices are not interpretable.
- Optionally `fine_scroll`: the sub-tile scroll offsets (0–7 each). Tile
  alignment holds only modulo scroll, so a program that exploits tile structure
  needs this; it is on screen in the sense that it is a rendering parameter, and
  making it a *latent* instead is the more conservative choice. Recommend
  latent, and note the choice.

**Latents (privileged, never model input).** The PPU's own truth:

- `nametable`: 32 × 30 tile indices, `integer(8, signed=False, role="category")`.
- `attributes`: the 64-byte attribute table, 2 bits of palette per 16×16 quad.
- `scroll_x`, `scroll_y`, `ppu_ctrl`, `ppu_mask`, `sprite0_hit`, `frame`.

**Probes (supervision targets).** OAM, as a relation:

- `sprites`: `set[(index, tile, x, y, attributes)]` at capacity 64 — five
  `int[8]` fields. `Type.width` = **384** with plain integer fields, 2,624 if
  every field is `category`-roled. The **`index` field is load-bearing**: two
  sprites can be byte-identical and a set is duplicate-free. This is exactly the
  construction `generators/logic`'s `gates` channel uses for the same reason
  (`research/depth-generalization/RESULTS.md`), and it is why a varying number
  of on-screen objects does not change the observation *type*.
- `sprite_count`, and per-pixel `sprite_index` for a segmentation probe.

The split is not a labelling convention: it is the whole point. The nametable
and OAM are the generator's authoritative state, they make a rich exact
supervision target, and a training loop that fed them to the actor would be
solving a different problem. `StepRecord.actor_view` already enforces this and
needs no change.

**Actions.** `press` with `{"buttons": product(BOOL × 8)}` in the fixed order
A, B, Select, Start, Up, Down, Left, Right; plus `wait`. Eight Booleans is the
controller, exactly. `available_actions` is `("wait", "press")` throughout —
NES has no state-dependent action menu, and pretending otherwise (a
game-specific "menu of useful moves") would be the oracle action menu section 6
forbids.

**Rewards.** This is where a NES generator needs per-game knowledge, and it must
be *declared configuration*, not code: a list of
`{name, address, width, signed, scale, sense}` RAM readings (score, x-position,
lives, timer). The generator implements "read these declared addresses and
difference them"; it must not contain a `if game == "smb"` branch, which
`AGENTS.md` forbids at the trainer level and which is no better here.

### 6.5 Remaining risks

- **Determinism across core versions.** libretro savestate layout is
  core-version-specific. The core name and version must be pinned in the
  manifest and checked at restore (section 7.1 again).
- **Frame-timing artefacts.** Some titles are sensitive to precise NMI timing;
  a core in an accuracy-reduced mode may diverge. Choose the accurate core, and
  verify by the round trip rather than trusting the label.
- **Tier 3 evidence.** Per section 5, no NES result can support a
  structural-generalization claim. It can support closed-loop control,
  perception-under-`eq`, and probe-supervision results, which is the point of
  proposing it.

---

## 7. Proposed core changes, as diffs. None applied.

`tcn/` was not modified, per the constraint that four agents are running against
this checkout and a core branch is queued.

### 7.1 External engine versions are not pinned in an episode (gap G8)

`Host.snapshot` records `source_fingerprint()`, which hashes `.py`/`.ts`/`.json`
under `tcn/` and `generators/`. It does **not** record the version of `mujoco`,
of Node, or of a libretro core. Section 6 requires "pinned
generator/configuration versions"; today an episode recorded on `mujoco` 3.12.0
restores silently on any other build.

Preferred form — per-generator, so only generators with an external mechanism
pay for it:

```diff
--- a/generators/control/manifest.json
+++ b/generators/control/manifest.json
@@
   "entrypoint": "generators.control.generator:Implementation",
-  "contract": "tcn.generator/1"
+  "contract": "tcn.generator/1",
+  "engines": ["mujoco"]

--- a/tcn/generation.py
+++ b/tcn/generation.py
@@
+def engine_versions(name):
+    """Installed versions of the external engines a generator declares."""
+    declared=manifests().get(name,{}).get("engines",())
+    out={}
+    for module in declared:
+        out[module]=getattr(importlib.import_module(module),"__version__","unknown")
+    return out
@@ class Host:
     def snapshot(self):
-        return copy.deepcopy({"format":"tcn.episode/1","source":source_fingerprint(),...
+        return copy.deepcopy({"format":"tcn.episode/2","source":source_fingerprint(),
+                              "engines":engine_versions(self.address.generator),...
@@     def restore(cls,d):
-        if d.get("format")!="tcn.episode/1": raise ValueError("unknown episode format")
+        if d.get("format")!="tcn.episode/2": raise ValueError("unknown episode format")
+        if d.get("engines",{})!=engine_versions(Address(**d["address"]).generator):
+            raise ValueError("episode external-engine version mismatch")
```

Cost: an episode-format bump, which invalidates every recorded episode. That is
why it is proposed and not applied.

### 7.2 Declare reproducibility limits where they are enforced

Section 6 says "Declare numerical reproducibility limits" but nothing in the
record carries them. Adding `"reproducibility"` to the manifest — e.g.
`"bit-identical for one build of the declared engines on one platform"` — and
copying it into the snapshot would put the claim next to the data instead of in
a docstring. No behavioural change; one field.

### 7.3 The fingerprint is repository-wide, so any new generator invalidates
every recorded episode

`source_fingerprint()` walks all of `tcn/` and `generators/`. Adding
`generators/control/` therefore changes the fingerprint of `logic` episodes,
and `Host.restore` refuses them. Confirmed here: `artifacts/system/logic/episode.json.gz`
raises `episode source revision mismatch` — and confirmed to *already* be failing
with `generators/control` moved out of the tree, so some earlier change on main
had done this first. It is a live papercut regardless of who tripped it.

```diff
--- a/tcn/generation.py
+++ b/tcn/generation.py
@@
-def source_fingerprint():
+def source_fingerprint(name=None):
+    """Hash `tcn/` plus, when named, one generator: a sibling must not invalidate."""
     root=Path(__file__).resolve().parent.parent;digest=hashlib.sha256()
-    for base in ('tcn','generators'):
+    for base in ('tcn',) if name is None else ('tcn', f'generators/{name}'):
```

with `Host.snapshot`/`restore` passing `self.address.generator`. This narrows
what invalidates an episode from "any change anywhere" to "a change to the core
or to that generator", which is what the check is actually for. It is a
behaviour change to a shared invariant and belongs to whoever owns the queued
core branch, not to this track.

---

## 8. Tests

`tests/test_control_generator.py`, 25 tests, all passing; full suite **164
passed**. The set deliberately mirrors the checks
`tests/test_generators.py::test_common_contract_replay_restore_and_visibility`
makes for every other generator, rather than editing that file, because four
agents share this checkout.

| test | what it pins |
|---|---|
| `test_common_contract_replay_restore_and_visibility` | replay, restore, save/load, `ActorView` isolation, objective immutability — both tasks |
| `test_snapshot_round_trip_continues_bit_identically` | **the replay contract**: raw integration state equal after every one of ten steps at varying `dt` |
| `test_recorded_exogenous_input_is_replayable` | the disturbance is typed, recorded, and reproduced |
| `test_named_streams_keep_the_start_pose_independent_of_the_goal` | G2 |
| `test_address_determines_the_episode` | seeded reproducibility and seed sensitivity |
| `test_action_schema_rejects_wrong_types_and_conflicting_commands` | typed action schema, simultaneous-command conflict |
| `test_actuator_limit_lives_in_the_type` | out-of-range torque is a type error, not a clip |
| `test_typed_policy_emits_in_range_torques_with_no_domain_action_head` | `tcn/policy.py` drives this environment with no domain code |
| `test_configuration_pins_the_task_and_the_actuator_limit` | versioned semantic configuration; restore rebuilds the same schema |
| `test_observations_are_numeric_and_never_byte_roled` | the section-4 trap, as an assertion |
| `test_declared_encodings_admit_the_arithmetic_the_task_needs` | the registry legality table, executable |
| `test_a_byte_roled_observation_would_lose_the_arithmetic` | the contrast, executable |
| `test_privileged_channels_carry_more_precision_than_observations` | float64 latents against float32 observations |
| `test_privileged_truth_is_not_in_the_actor_view` | the observation/latent/probe split, including the reacher goal |
| `test_reward_components_are_separate_typed_scalars` | G6 |
| `test_the_pendulum_is_a_real_swing_up_problem` | the mechanism is a control problem, not a stub |
| `test_the_logical_clock_drives_the_physical_one` | G3 |
| `test_substep_count_is_a_function_of_dt_alone` | replay determinism under varying `dt` |
| `test_angles_are_wrapped_for_observation_but_not_in_state` | the wrap is an observation projection, not state loss |

Not tested, and stated as limitations: cross-platform determinism, cross-version
MuJoCo determinism, any *learning* result on this environment. This track
delivers a working environment and a verified replay contract. It does not
demonstrate that anything can be learned in it, and per FINDINGS section 6
("policy learning is the weak half") that is the open question, not a formality.
