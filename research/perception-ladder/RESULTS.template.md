# The perception ladder: where supervised typed synthesis stops working

Research track "perception-ladder". Everything below was produced in this
workspace on 2026-09-08 with `/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python`
(Python 3.13.15, torch 2.14.0+cpu, `torch.set_num_threads(1)`).
**Nothing under `tcn/` or `generators/` was modified.** All code, logs and raw
JSON are in this directory; `tables.py` renders every table below directly from
`out/*.json` so no number here is transcribed by hand.

---

## 0. Verdict

**The highest rung that works is rung 3: foreground/background segmentation of
raw `geometry` pixels.** Two measurements, both 12/12 seeds with **zero error on
48 held-out episodes**:

* the per-pixel mask over the whole image at 2x2 and at 3x3 (section 6, the
  `pinned` rows), where the program learns each pixel's two channel comparisons
  and their Boolean combination;
* the background colour itself, searched over the **entire 256-value byte
  alphabet** at 2x2, where enumeration certifies the solution is **unique**
  among 65,536 programs (section 4.4) -- so there is no partial credit and no
  ambiguity about what was learned.

It is a small capability but it is a real one: nothing was pre-digested, the
only input is the raw byte tuple, and the supervision is an equivalence class of
the generator's own `depth` probe.

**It works only because the scaffold supplies the byte addresses.** The instant
the program has to choose *which* byte to read, the same task collapses: 1/12 at
576 programs, 0/12 from 46,656 up. That is the wall, and it is sharp.

Three walls, in increasing order of how fundamental they are.

1. **Input-binding relaxation is worse than chance (section 6.4).** Measured at
   initialisation over 8 perturbed starts: when a free node's choice is *which
   constant or operator* to apply at a fixed input, the steepest-descent
   direction picks the reference candidate 0.67-0.81 of the time against a
   chance rate of 0.004-0.33 -- up to 208x chance. When the choice is *which
   input to bind*, it picks the reference **0.25 of the time against a chance
   rate of 0.29**. A convex mixture of candidate operators is a blend of
   functions at a valid input; a convex mixture of candidate input bindings is
   a blend of unrelated values that denotes nothing, and its gradient
   misinforms. This single measurement explains rung 3c, rung 4, and the sample
   indices in rung 1.
2. **An optimisation wall at ~10^2 programs for analytic operator chains
   (section 2).** On a task whose exact solution is in the library, gradient
   synthesis goes 16/16 -> 12/16 -> 8/16 -> **0/16** as the space grows
   4 -> 36 -> 144 -> 288. This is *not* a relaxation-tightness failure: the
   reference program **is** the global optimum of the relaxed objective (relaxed
   loss 3.0e-16) and the optimiser settles 1.5e8 to 3.2e14 times above it.
   Per node the relaxation gives only a ~1.3x-over-chance hint, which cannot
   deliver six simultaneously correct nodes. Exhaustive enumeration solves every
   instance -- the largest, 829,440 programs, in 77 s, with held-out error 1e-7 --
   and uniform random sampling of the same space also beats gradient descent
   from 288 programs upward.
3. **A representational wall at the image boundary (section 4.1).**
   `image_value` types every channel as `int[8]` with `role="byte"`, and
   `Type.numeric` excludes role `"byte"`. So `sum`, `mean`, `reduce_min/max`,
   `fft` and *every* arithmetic operator and ordering comparison are
   type-illegal on pixels; the only operators that consume an image are
   `project`, `index`, `eq`, `tuple` and `pack`. `pack` is the sole route to
   arithmetic and declares `gradient="none"`. `eq` is the only differentiable
   predicate and its surrogate `exp(-(a-b)^2/tau)` at `tau=1` **underflows to
   exactly 0.0 in float32 for |a-b| >= 12** on 0..255 data. There is no
   convolution, no window, no pooling, and you cannot even count how many
   Booleans fired (`sum` over a `bool` tuple is illegal too). **The algebra
   contains no learnable transformation of an image other than per-byte
   equality.**

Two generator targets are not reachable at all, for reasons the type checker
settles without any experiment:

* **`relations.closure`** -- `join`/`pair` are the only set-composition
  operators and they multiply capacity (64 -> 4096) while no conversion narrows
  a set back and `union` requires identical types. No program can have the
  closure's type as its output, so the probe cannot be attached (section 3.1).
* **`raster_text.text`** -- each of its 64 byte fields needs a `BYTE`-valued
  node, and with no aggregation operator to couple them that is `448^64 ~ 5e166`
  programs (section 5.1). Reduced to a two-symbol discrimination, enumeration
  **exhausts the 49,152-program differentiable sub-algebra and certifies that no
  solution exists** there, while the solution that does exist runs through
  `pack` (section 5.2).

**Track 3's dense-probe result replicates on non-Boolean tasks, with a condition
that should be attached to it.** Dense probes help exactly when the output loss
does not already decompose over the probed quantities:

| task | output | dense | output-only |
|---|---|---|---|
| rung 3 `allbg`, 2x2, 8.4e6 programs | one bit, AND-folding 4 per-pixel predicates | **12/12 generalise** | 3/12 exact on train, **0/12 generalise** |
| rung 3 `allbg`, 3x3, 9.0e15 programs | one bit, AND-folding 9 predicates | **12/12 generalise** | 9/12 exact on train, **1/12 generalise** |
| rung 2 `target`, 65,536 programs | one bit, OR-folding 4 conjunctions | **8/8**, held-out accuracy 0.943 | **1/8**, held-out accuracy 0.873 |
| rung 3 `mask`, 2x2 and 3x3 | a concatenation of the per-pixel predicates | 12/12 | 12/12 -- **identical** |
| rung 1 `future`, 2,880 up | a 4-step recurrence | 0/12 | 0/12 (fit 210x worse) |

The 3x3 `allbg` row is the sharpest form of the effect: without the probes the
optimiser fits nine out of twelve training sets and generalises on **one**, so
output-only supervision is not failing to optimise, it is finding programs that
agree on 24 episodes and disagree on the next 48. With the per-pixel probes it
finds the right program every time. Note also what dense probes do *not* do:
with the byte addresses left free, every one of these eight arms is 0/12,
because no probe can make a mixture over candidate addresses mean anything.

The first two rows are the same effect track 3 measured (19-38% -> 88-94%),
reproduced at 25% -> 100% and 12.5% -> 100% on an image task and a set task.
The third row is the negative control that pins down the condition: when the
output signal is already the concatenation of the intermediates, `probe_loss`'s
elementwise BCE *is* the mean of the per-element BCEs, the extra signals are
arithmetically redundant, and they change nothing at all. The fourth row is the
limit: dense supervision changes the fit, not the reachability, so once a task
is past wall 2 it does not rescue it.

---

## 1. Protocol

**Inputs are raw observations only.** Every program's input ports are filled
from `StepRecord.actor_view().observations`. `latent_states` and `probes` are
used *only* to build `example["targets"]`. No detector, tokeniser, parsed text,
object list or oracle action menu is supplied anywhere. Where a target is a
deterministic function of a privileged value (an equivalence class of the
`depth` probe, one field of the `frequencies` latent) that is stated at the
point of use; where a target is derived from *observables* instead, that is
flagged explicitly and its weaker status noted.

**Success criterion.** A run succeeds only if `SoftProgram.export()` -- the
argmax program, hardened and validated -- reproduces the targets exactly, to the
same 1e-3 tolerance `tcn.synthesis.fit` uses. Relaxed loss is reported
separately and never treated as evidence (finding F-soft, `research/FINDINGS.md`).
Held-out error is measured by executing the exported program on episodes from a
disjoint seed range.

**Discrete reference.** Following recommendation 9 of `research/FINDINGS.md`,
every synthesis number is reported alongside `tcn.search.enumerate_fit` over the
identical candidate space, plus a uniform-random-sampling control over the same
space.

**One harness change, verified inert.** `tcn.synthesis.fit` builds a
`SoftProgram` whose choice logits are initialised to **all zeros**, so repeated
runs are bit-identical and a seed changes nothing (`audit.json`:
`softprogram_two_instances_identical = true`). Track 1 hit the same wall.
`common.local_fit` is a faithful copy of `fit`'s control flow with an added
`init_noise` parameter (sigma = 0.5 on the logits). `check_local_fit.py` verifies
it is **bit-identical to `tcn.synthesis.fit` at `init_noise=0`** on the
`examples/mixed.py` fixture at two budgets.

---

## 2. Rung 1 -- `signal`: frequency recovery from raw samples

### 2.1 The task and the exact reference program

The agent sees only `observations['samples']`, a window of the last W samples of
`sum_i a_i sin(2 pi f_i t + p_i)`. With `components=1` the signal obeys a linear
recurrence, and the following program -- built entirely from operators already
in `tcn/operators.py` -- recovers the frequency and extrapolates exactly:

```
c      = (x[t] + x[t-2]) / (2 * x[t-1])          =  cos(2 pi f)
f      = atan2(sqrt(1 - c^2), c) / (2 pi)        =  acos(c) / (2 pi)
x[t+1] = 2*c*x[t] - x[t-1]                        (iterate for t+2 .. t+4)
```

Measured against the generator's own `frequencies` latent and `future` probe on
every one of the 45 kept episodes: maximum error **9.9e-8** for the
frequency program and **1.2e-6** for the four-step extrapolation. The task
is genuinely solvable, exactly, inside the declared library.

`amplitude` is deliberately not needed: the recurrence extrapolates without it,
which matters because `amplitude` is *not* one of the generator's latents.

**The `phases` latent is also expressible, and I initially got this wrong.** The
first draft of this report claimed phase was inexpressible because unwrapping
`phi_t = 2 pi f t + p` into `[-pi, pi]` needs a modulus, and `mod`/`idiv`/`shl`/`shr`
all `require(t.encoding.kind=="integer")` and so are illegal on a floating
scalar. That is true of `mod` and false of the conclusion: `atan2` wraps to
`[-pi, pi]` by construction, and
`p = atan2(A sin(phi) cos(wt) - A cos(phi) sin(wt), A cos(phi) cos(wt) + A sin(phi) sin(wt))`
with `A sin(phi) = x[t]`, `A cos(phi) = (x[t] cos w - x[t-1]) / sin w` recovers it
without ever needing the amplitude (`atan2` is invariant to a positive scale).
Verified numerically: **worst error 2.5e-06 over 30 episodes.** I did not run it
as a synthesis rung, because its prefix is the frequency chain of 2.2 and that
alone is already past the wall; the point of recording it is that the *type
algebra* is less of a constraint here than it first appears, and the binding
constraint on `signal` is the optimiser.

### 2.2 The wall, measured

W = 4, 10 training episodes, 24 held-out episodes from a disjoint seed range,
600 Adam steps at lr .05, 16 seeds per cell. Scaffolds are graded: `free` nodes
keep their full candidate menu, the rest are pinned to the reference choice, so
the *only* thing changing down the table is the size of the discrete space.

TABLE_RUNG1_LADDER

Read the first two columns against the fourth. Gradient synthesis is perfect at
12 programs, has lost a quarter of its seeds at 36, half at 144, and **all of
them at 288**, where exhaustive enumeration finishes in 14 milliseconds.

From `L2_trig` (288 programs) onward, **uniform random sampling of the same
space is strictly better than gradient descent**: at a solution density of
6.4e-3 a random draw succeeds in ~156 attempts, each of which costs about 0.15 ms,
against 0/16 for a 600-step gradient run costing 1.7 s.

### 2.3 The failure is optimisation, not relaxation tightness

The obvious hypothesis is that the relaxed optimum sits away from any vertex, so
the argmax rounds to a different program. That is measurably **not** what
happens here. Setting the choice logits one-hot at the reference program and
evaluating the same relaxed objective:

TABLE_RUNG1_GAP

The reference program *is* the global optimum of the relaxed objective, at
3.0e-16. Adam settles between 1.5e8 and 3.2e14 times above it (8.2 to 14.5 orders of magnitude, over all 72 runs of the sweep) and stops
moving: in the traced runs the argmax selection freezes by step ~50 and does not
change over the following 1,450 steps while the loss continues to fall. This is
premature entropy collapse into a function-space attractor -- the same mechanism
track 3 identified on Boolean targets -- now observed on an analytic task where
the target is a continuous scalar rather than a bit.

Two further contributors, both measured:

* **Scale weighting does not help** (`weights.py`, `out/rung1_signal_weight.json`).
  The frequency target lives in [0.02, 0.25] while the intermediate values are
  O(1), so the obvious suspect is the loss scale, and ARCHITECTURE section 8
  prescribes normalising field scales. Raising the signal's `weight` from 1 to
  20 to 100 to 400 changes **nothing**: 0/8 at every weight on both the
  288-program and the 829,440-program scaffold, and on the 288-program scaffold
  the eight per-seed exact errors are **bit-identical at all four weights**
  (0.1842, 0.1891, 0.1842, 0.1891, 0.1842, abort, 0.8082, 0.0387). Adam's
  per-parameter normalisation absorbs the rescaling entirely.
* **Declared numeric domains abort runs.** `relaxed()` raises on
  `sqrt` of a negative mixture and on `log`/`div` domain violations. Across the
  `gap` sweep, **10 of 72 runs (13.9%) died this way** rather than returning a
  wrong program. This is the specification behaving as written -- a candidate
  without a valid relaxation is an exact operator at a gradient boundary -- but
  it means the relaxed search cannot even traverse the interior of a scaffold
  containing `sqrt`.

### 2.4 Future prediction, and dense vs output-only

The `future` probe (values at t+1, t+2, t+4) is produced by a four-step
recurrence sharing the prefix `a, b, d, num, den, c` with the frequency branch.
That makes an honest hierarchical-supervision test: the `frequencies` **latent**
supervises an interior node whose subgraph the `future` **probe** also depends
on. Both arms are judged on the `future` signal alone.

TABLE_RUNG1_FUTURE

**Nothing succeeds.** The task is exactly solvable (reference error 1.1e-6) and
the smallest scaffold here is 2,880 programs -- already an order of magnitude
past the wall located in 2.2. What the dense probe does buy is a better *fit*:
at the full 1.36e10-program scaffold the median held-out error is 5.24 with the
`frequencies` probe attached and 1104 without it, a 210x improvement that still
does not reach exactness. Dense supervision is helping; it is not helping enough
to matter, because the binding constraint is the optimisation wall in 2.2.

---

## 3. Rung 2 -- `relations`: the set-operator gradient boundary

### 3.1 The `closure` latent is unattachable

The generator emits `latent_states['closure'] : set[tuple[int[3],int[3]], 64]`,
the exact transitive closure. **No program in this algebra can produce a value
of that type equal to the closure**, for a reason that is purely structural
(`closure_wall()` in `rung2_relations.py`, raw output in `out/rung2_closure_wall.json`):

| step | result |
|---|---|
| the only set-composition operators are `pair` and `join` | `join(EDGES, EDGES)` has capacity **4096**, width 53,248, `gradient="none"` |
| rejoin the composed relation with the original | `union(EDGES, join(...))` -> `TypeError: union: operator signature mismatch` (`union` requires `ts[0]==ts[1]`, and capacity is part of the type) |
| narrow the capacity back | `encode`, `decode`, `quantize`, `dequantize`, `pack`, `unpack` all -> `TypeError: signature mismatch` (every conversion requires `ts[0].kind in {bool,int}`) |
| attach the probe elsewhere | `Program.validate_signals` -> `TypeError: probe signature mismatch` |

So one composition step is expressible, a *second* one is not, and a fixpoint
iteration never is. The generator's richest supervision target on this domain is
not merely hard to learn; it is outside the image of the operator algebra.
This is the sharpest single finding in the track and needs no experiment to
establish -- only the type checker.

### 3.2 Every set operator is a gradient boundary; the Boolean layer above it decays

`member`, `insert`, `remove`, `union`, `intersection`, `pair` and `join` all
declare `gradient="none"`, so `relaxed()` falls through to `exact_tensor`, which
`.detach()`es its inputs. Measured choice-logit gradients on a reachability
scaffold with 8 free wiring nodes below the set layer and 8 free Boolean nodes
above it (`out/rung2_gradients.json`, logits perturbed sigma=0.5 to break the
uniform truth-table symmetry):

| node | region | candidates | \|grad\|_1 |
|---|---|---|---|
| `q0`, `q1`, `l0..l3`, `rr0..rr3` | below / into a set operator | 1-4 | **`None`** -- not reachable in the autograd graph at all |
| `D`, `L0..L3`, `R0..R3` (`member`) | set | 1 | 0.0 |
| `P0` / `P1` / `P2` / `P3` | Boolean, depth 4 | 16 | 1.06e-05 / 3.63e-05 / 1.56e-03 / 3.07e-02 |
| `O0` / `O1` / `O2` / `O3` | Boolean, depth 5-8 | 16 | 8.97e-05 / 9.53e-04 / 7.27e-03 / **2.58e-01** |

Two separate effects. **Hard:** 65,536 wiring choices (8 nodes x 4 mediators)
receive literally no gradient, because they feed `member`. **Soft:** above the
boundary the gradient decays by roughly a factor of ten per layer, so the
deepest Boolean node sees a signal **24,000x weaker** than the output node. The
second effect is precisely the case for dense intermediate probes; the first is
beyond their reach, since a probe cannot restore a severed autograd path.

### 3.3 The reachable task

What *is* reachable is the `target` probe (is the query pair in the closure?).
The scaffold computes `direct OR (OR_k [edge(q0,m_k) AND edge(m_k,q1)])` --
membership tests over the raw observed `edges` and `query`, with the mediator
identifiers supplied as constant-valued input ports (see **P1** in
section 7; they carry no episode information). This expresses reachability in at
most two steps, which at `entities=4` agrees with the generator's `target` on
**99.5%** of episodes (1-step agrees on 94.0%); the residual 0.5% are queries
needing a three-step path, and they are the reason held-out *accuracy* rather
than held-out *max error* is the informative statistic here.

TABLE_RUNG2

The **ceiling** for every row is the reference program itself, whose held-out
accuracy is **0.9844** -- the residual is the 0.5% of episodes needing a
three-step path, which the scaffold cannot express. Read the rows against that.

* `R0_fold_only` pins the conjunctions and searches only the OR-fold. Both arms
  solve it; there is nothing for a probe to add.
* `R1_pool4` frees the conjunctions too, over the same 65,536 programs. **Dense
  8/8 at 0.943 held-out accuracy against output-only 1/8 at 0.873**, which is
  74% of the way from the output-only arm to the ceiling. This is the rung-2
  form of track 3's result.
* `R2_pool16` (4.3e9) and `R3_free_wiring` (2.8e14) defeat both arms. In
  `R3_free_wiring` that is expected and not a matter of budget: the 65,536
  mediator-wiring choices sit below `member` and receive `grad = None` (section
  3.2), so no amount of supervision above the set layer can reach them.
* Enumeration solves both 65,536-program arms (33 s and 46 s) and cannot be run
  above that. Random sampling of `R1_pool4` has solution density 9e-05, i.e. one
  hit per ~11,000 draws -- so at this size the discrete methods are still ahead
  of the output-only gradient arm, and behind the dense one.

---

## 4. Rung 3 -- `geometry`: what an image is, to this operator algebra

### 4.1 The expressibility audit (`out/audit.json`)

`image_value` builds `product(int[16], int[16], int[8], product(BYTE x 3*R*R))`
with `BYTE = int[8]` and `role="byte"`. `Type.numeric` returns
`kind == "int" and role not in {"category","symbol","byte"}`, so a pixel is
**not numeric**, and that single line decides almost everything:

| operator on image bytes | legal? | gradient |
|---|---|---|
| `sum`, `mean`, `reduce_min`, `reduce_max` | **no** -- `sum: operator signature mismatch` | -- |
| `fft` | **no** | -- |
| `add`, `sub`, `mul`, `div`, ... | **no** | -- |
| `lt`, `le`, `gt`, `ge` | **no** (`COMPARE` requires `numeric` for everything but `eq`) | -- |
| `eq(byte, byte)` | yes | **surrogate** |
| `project(image, i) -> byte` | yes | exact |
| `index(image, int) -> byte` | yes | surrogate |
| `decode(byte) -> scalar` | **no** -- conversions require `role` to match, and a float32 with `role="byte"` is still non-numeric | -- |
| `pack((byte,)) -> uint8` | yes | **none** |
| `decode(uint8) -> scalar` | yes | surrogate |
| `count(image bytes)` | yes, but returns the *static tuple length* -- a constant | exact |
| `sum(bool tuple)` | **no** -- `BOOL` is not numeric either, so you cannot count how many predicates fired | -- |

Three consequences, all load-bearing:

1. **There is no aggregation over an image.** Not a convolution, not a window,
   not a pooling operator, not even a mean. Every program that reads pixels must
   name individual bytes by index. The number of legal `project` candidates for
   one byte-reading node is exactly `3*R*R` -- 3,072 at the shipped `resolution=32`.
2. **The only route from a pixel to arithmetic is `pack`, which is a declared
   gradient boundary.** Anything reached through it -- thresholds, differences,
   sums of channels -- is invisible to gradient descent.
3. **A byte constant cannot be trained.** `Program.validate` requires
   `type.numeric` for trainable constants, so a colour value must be found by
   discrete search over a supplied alphabet, never by descent.

And the one differentiable predicate is numerically blind. `eq`'s declared
relaxation is `exp(-((a-b)^2).sum(-1)/tau)` at `tau=1`:

| \|a-b\| | 0 | 1 | 2 | 3 | 5 | 8 | 10 | **12** | 32 | 255 |
|---|---|---|---|---|---|---|---|---|---|---|
| relaxed `eq` | 1.0 | 3.7e-1 | 1.8e-2 | 1.2e-4 | 1.4e-11 | 1.6e-28 | 3.8e-44 | **0.0** | 0.0 | 0.0 |
| d/d input | 0 | 7.4e-1 | 7.3e-2 | 7.4e-4 | 1.4e-10 | 2.6e-27 | 7.6e-43 | **0.0** | 0.0 | 0.0 |

On data with a 0..255 range, the surrogate and its gradient are **exactly zero**
in float32 beyond 11 grey levels. Comparisons in image space are informative to
the search only if they are already almost right.

### 4.2 The generator's own probes are mostly not functions of the observation

Two measurements from `out/audit.json` (`geometry_information`), 60 episodes per cell:

* **At the shipped camera the scene is empty.** `focal = 0.5*R/tan(30 deg)` and
  the objects sit ~6.5 units from `eye=[3,3,5]`, so a 0.5-unit object projects
  to ~0.07*R pixels. The background fraction is **0.98 at every resolution from
  3 to 32** (1.00 at R=2 -- no object renders at all). Any accuracy figure on
  `depth`, `object_ids` or `normals` at the shipped configuration is a report on
  a constant. Everything below therefore issues one `camera` action (part of the
  generator's declared action schema, not a code change) to move the eye in,
  which gives a 0.71 / 0.29 background/foreground split.
* **`depth` is not a function of a pixel.** With the approach camera at R=32,
  **688 of 859** distinct RGB values (80%) map to more than one depth; at R=8,
  164 of 357. **`object_ids` is worse than ambiguous: it is unlearnable in
  principle**, because object colours are re-randomised every episode while ids
  are list positions, so no fixed function of colour can name them.
  The apparent absence of id collisions is memorisation, not structure. The
  measurement that shows it: the fraction of held-out pixels whose exact RGB
  triple was seen in training is 0.726 at R=8 and 0.728 at R=32 -- against
  background fractions of 0.717 and 0.719. Subtracting the background, **only
  3.2% of held-out foreground pixels have a colour that ever occurred in
  training**, at both resolutions. An RGB lookup table -- the matched-information
  baseline for a per-pixel `object_ids` predictor -- has essentially no coverage
  of the foreground.

What *is* an exact function of a pixel is the background indicator: `render()`
paints untouched pixels `(24,30,43)` and sets their depth to 0, so
`depth[i] == 0` is an equivalence class of the `depth` probe and
`eq(red,24) AND eq(green,30)` computes it exactly (measured 0 errors over 200
episodes x every resolution tested). That is the rung.

### 4.3 The width sweep: where addressing stops converging

Predict whether the centre pixel is background. The two byte-address choices are
free over all `3*R*R` bytes; the two colour constants are supplied. 24 training
episodes, 48 held-out, 400 steps, 12 seeds.

TABLE_RUNG3_WIDTH

Gradient synthesis is already broken at a 15-float observation, and the
enumerated program is **exactly right on held-out episodes at every width**.
The apparent recovery at 4x4 is noise; the pattern from 6x6 on is unambiguous.

### 4.4 Searching a constant works; searching an address does not

The same scaffold with the **addresses pinned** and the background colour
searched over the **whole 256-value byte alphabet** (65,536 programs -- 114x
larger than the 2x2 addressing task of 4.3, which gradient descent fails 11/12,
and with a **unique** solution, so there is no partial credit):

TABLE_RUNG3_COLOUR

Both enumeration runs return the same unique selection,
`hit_eq0 = 24`, `hit_eq1 = 30` -- which is exactly `render()`'s background
`(24, 30, 43)`, recovered from raw bytes with nothing supplied but the 256-value
alphabet. Gradient synthesis finds the same program in 12/12 seeds at both
resolutions.

This is the cleanest statement of the mechanism. Relaxing a node's *operator or
constant choice* while its inputs are fixed is benign: exactly one of the 256
`eq` candidates fires, the mixture is a clean one-hot signal, and descent finds
it. Relaxing a node's *input binding* is not, because a convex mixture of
addresses is a blend of unrelated pixel values -- a quantity that is not a legal
value of the type at all, and whose `eq` against any constant is (by 4.1) zero
with zero gradient. **The wall is input-binding relaxation, not space size.**

---

## 5. Rung 4 -- `raster_text`: pixels to symbol

### 5.1 The `text` probe is not addressable

`probes['text']` has type `product(int[32] bounds(0,64), product(BYTE x 64))`.
Each of its 64 byte fields must be produced by a node of type `BYTE`, and the
only `BYTE`-valued operators are `project` (copy an image byte), `index` and
constants. On an 8x8 canvas with the full 256-value byte alphabet that is
`448^64 ~ 5e166` discrete programs for the byte head alone, with **no
aggregation operator** available to condition the 64 heads jointly (section 4.1:
`count` returns the static length, `sum` over bytes or bools is illegal). Full
OCR is not "hard" in this system; the candidate algebra has no shape that could
express it.

The rung is therefore reduced to a two-symbol discrimination ('a' vs 'b'),
8x8 canvas, font size still randomised by the generator over 10..16, with the
target an equivalence class of the `text` probe.

### 5.2 The task is solvable, and solvable only where gradients cannot go

Over 60 seeds x 2 symbols on the 192-byte canvas:

* **single-byte equality classifiers that separate the classes: 0.** There is no
  `(index, value)` pair for which `eq` is a perfect predictor. Anti-aliasing
  spreads each glyph over many grey levels, and `eq` is exact.
* **single-byte threshold classifiers: 3** -- byte indices 180, 181, 182 (one
  pixel's R, G and B) with a threshold at 68. Verified: `lt(decode(pack(px)), 68)`
  is exact (max error 0.0) on 40 examples (20 seeds x 2 symbols) at all three addresses.

The reachable solution therefore runs `project -> tuple -> pack -> decode -> lt`,
and `pack` declares `gradient="none"`.

### 5.3 Measured

TABLE_RUNG4

The `eq` arm is the important one: enumeration **exhausted all 49,152 programs
and returned no solution**. That is a completeness certificate -- something a
gradient run cannot produce -- and it says the differentiable sub-algebra
provably cannot express this task, rather than merely failing to find it.

In the threshold program, gradient reaches nothing: the address node's logits
are `None` (unreachable past `pack`), and the trainable threshold's gradient is
exactly `0.0`. The last one deserves its mechanism spelled out, because it is a
compound failure: the relaxed `project` collapses the image to a weighted mean
byte (~230, since 71% of the canvas is near-white), `lt`'s surrogate
`sigmoid((thr - value)/tau)` at `tau=1` is saturated at a distance of 100 grey
levels, the prediction is therefore constant, and on a class-balanced training
set the gradient of a constant prediction is identically zero.

Pinning the address to byte 180 -- a byte on which an exact threshold solution
is known to exist -- does **not** rescue it: 0/8 at lr .05 and 0/8 at lr 2.0.
Descent cannot fit even a single scalar threshold on byte-ranged data, because
`lt`'s sigmoid at `tau = 1` is saturated across the whole 0..255 range. Whereas
enumerating that same one-dimensional constant over 256 values, alongside the
address and the comparison direction, settles the task outright with held-out
accuracy 1.0. This is recommendation P3 and P6 in one instance: the continuous
parameter is the part relaxation is supposed to be good at, and at the shipped
temperature it is not even good at that.

---

## 6. Dense probes versus output-only supervision

Track 3 measured dense intermediate probes taking free-wiring Boolean synthesis
from 19-38% to 88-94%. This track ran the same comparison on three non-Boolean
tasks. The result is a **conditional replication**, and the condition is worth
stating precisely because it is easy to get wrong.

**Dense probes help exactly when the output loss does not already decompose over
the probed quantities, and only when the search is otherwise inside its working
range.**

| rung | probed intermediates | is the output loss separable over them? | dense | output-only |
|---|---|---|---|---|
| 3, `allbg`, 2x2 | per-pixel `hit_i` | no -- one bit AND-folding 4 pixels | **12/12** generalise | 3/12 train, **0/12** generalise |
| 3, `allbg`, 3x3 | per-pixel `hit_i` | no -- one bit AND-folding 9 pixels | **12/12** generalise | 9/12 train, **1/12** generalise |
| 2, `target`, `R1_pool4` | per-mediator conjunctions `P_k` | no -- one bit OR-folding four conjunctions; section 3.2 shows `P_0` sees a gradient 24,000x weaker than the output | **8/8**, held-out accuracy 0.943 | **1/8**, held-out accuracy 0.873 |
| 3, `mask`, 2x2 and 3x3 | per-pixel `hit_i` | **yes** -- `probe_loss` uses elementwise BCE over the output tuple, which *is* the mean of the per-pixel BCEs | 12/12 | 12/12, **identical** |
| 1, `future` | `frequencies` latent on a shared prefix | no -- `future` is a 4-step recurrence over `c`, `frequencies` is `acos(c)` | 0/12; median held-out error 5.24 | 0/12; median held-out error 1104 |

**Addresses pinned** (so the comparison isolates supervision from the
input-binding wall of 6.4); pool of 4 byte constants, `AND`/`OR` menu:

TABLE_RUNG3_DENSE

The same six arms with **addresses free**, for completeness -- everything fails,
as 4.3 predicts:

TABLE_RUNG3_DENSE_FREE

The rung-3 `mask` row is the useful negative control and it should discipline
how track 3's headline is quoted. "Probe every intermediate node" is not a
general-purpose accelerator; it is a way of *re-separating a loss that the
output entangles*. Where the output signal is already a concatenation of the
intermediates, the extra signals are arithmetically redundant and buy nothing.
Where the output entangles them -- an OR-fold, an AND-fold, a recurrence -- the
probes restore a gradient that decays by an order of magnitude per layer.

The second, harsher observation: **dense probes do not cross either wall.**
With the byte addresses free, all eight arms above are 0/12 -- dense and
output-only alike -- because a probe on `hit_i` cannot make the mixture over
candidate addresses mean anything (section 6.4). And on rung 1, past the
optimisation wall of 2.2, both arms are 0/12 at every scaffold from 2,880
programs up. Dense supervision changes the *fit*, not the *reachability*. It is
a real and large effect -- 0/12 to 12/12 on the entangled geometry head, 1/8 to
8/8 on relations -- operating strictly inside a much smaller envelope than the
one `ARCHITECTURE.md` section 7 assumes.

### 6.4 What actually separates the cases: does the gradient point at the answer?

Two hypotheses were tested against the data and one survived.

**Rejected: "the relaxed optimum lies outside the achievable set."** A convex
mixture is always inside the elementwise range of its candidates, and
`excursion.py` confirms it directly: across every scaffold in this track, **zero**
free nodes propagate a mixture outside the range spanned by their own
candidates. The relaxed values are not wild; section 2.3 already showed the
reference vertex *is* the relaxed global optimum. The problem is not where the
optimum is.

**Supported: the local gradient carries almost no information about the discrete
choice, except at nodes whose inputs are fixed.** `pointing.py` measures, at
initialisation and over 8 perturbed starts, how often the most-negative logit
gradient at a free node is the reference candidate -- i.e. how often the first
Adam step pushes toward the right program -- against the chance rate
`1/|candidates|`.

| scaffold | free-node type | picks the reference | chance | advantage | gradient success |
|---|---|---|---|---|---|
| `rung3:centre_colour_search_R2` | constant choice, inputs fixed (256 candidates) | **0.812** | 0.004 | **208x** | 12/12 at 65,536 programs |
| `rung1:L1_om_s` | operator choice, shallow | 0.812 | 0.292 | 2.8x | 16/16 at 12 |
| `rung3:mask_pinned_address_R2` | constant + Boolean choice, inputs fixed | 0.667 | 0.333 | 2.0x | 12/12 at 1,048,576 |
| `rung1:L0_om` | operator + argument order | 0.500 | 0.250 | 2.0x | 16/16 at 4 |
| `rung1:L1b_+t2` | + one unary operator | 0.583 | 0.306 | 1.9x | 12/16 at 36 |
| `rung1:L1d_+csq` | + two more | 0.520 | 0.383 | 1.4x | 8/16 at 144 |
| `rung1:L2_trig` | six analytic operator choices | 0.533 | 0.403 | 1.3x | **0/16** at 288 |
| `rung1:L3_trig_alg` | nine, incl. sample indices | 0.347 | 0.365 | **0.95x** | 0/16 at 12,960 |
| `rung3:centre_free_address_R2` | **input binding** (which byte to read) | 0.250 | 0.292 | **0.86x** | 1/12 at 576 |

Two things fall out.

* **Input-binding relaxation is worse than useless.** Choosing which byte to read
  scores *below chance*: the mixture of candidate addresses is a blend of
  unrelated pixel values, and its gradient actively misinforms. This is the
  single clearest mechanism in the track, and it is why the same rung-3 task
  succeeds 12/12 with addresses pinned and fails 1/12 with them free.
  **Caveat, stated because it matters:** the largest space solved here
  (3.5e13 programs, rung 3 `mask` at 3x3 with pinned addresses) *factorises* --
  the output is a concatenation of per-pixel bits, so it is really nine
  independent 32-way choices. The non-factorising results are the ones to quote:
  the `allbg` head, one bit AND-folding four pixels over 8,388,608 programs,
  **12/12 generalising with dense probes and 0/12 without**; and the 65,536-program colour
  search of section 4.4, whose two `eq` choices are coupled through an `AND` and
  a single output bit and whose solution enumeration certifies **unique** --
  12/12 at both 2x2 and 4x4.
* **Even a positive advantage does not survive conjunction.** A per-node
  advantage of 1.3x cannot deliver six simultaneously correct nodes, which is
  exactly the `L1d` (1.4x, 50%) -> `L2` (1.3x, 0%) transition. Relaxation is
  giving a weak per-node hint, and success needs every node right at once.

This is the same conclusion track 8 reached from timing, arrived at from
gradient geometry: relaxation is a heuristic worth having where its hint is
strong (constants and operators at fixed inputs) and worth discarding where it
is not (input bindings), which is precisely the NEAR-shaped split.

---

## 7. Proposed changes to `tcn/`, as diffs. None applied.

Ordered by measured payoff. Nothing under `tcn/` or `generators/` was touched by
this track; each item below names the measurement that motivates it.

### P1 (bug). `relaxed("tuple", ...)` does not broadcast, so a constant cannot be tupled with a batched value

`tcn/learning.py:49`. `torch.cat` requires equal rank, and `SoftProgram.forward`
gives program constants shape `(width,)` while node values have shape
`(batch, width)`. Every other relaxation broadcasts (`mul` on the same pair
returns `[5,1]` correctly); `tuple` raises
`RuntimeError: Tensors must have same number of dimensions: got 2 and 1`.
This is why rung 2 has to pass its mediator identifiers in as constant-valued
*input ports*. Verified fix:

```python
-    if n=="tuple": return torch.cat(xs,dim=-1) if xs else torch.empty(0)
+    if n=="tuple":
+        if not xs: return torch.empty(0)
+        shape=torch.broadcast_shapes(*(x.shape[:-1] for x in xs))
+        return torch.cat([x.expand(*shape,x.shape[-1]) for x in xs],dim=-1)
```

Checked on `[(5,3),(3,)] -> (5,6)`, `[(5,3),(5,2)] -> (5,5)` and the unbatched
`[(3,),(2,)] -> (5,)`, so the fix is inert where `cat` already worked.

### P2 (bug, and it silently invalidated every "N seeds" claim on this repo)

`SoftProgram.__init__` initialises every choice logit with `torch.zeros`, so two
`SoftProgram`s over the same program are bit-identical and `torch.manual_seed`
has no effect on synthesis. Track 1 worked around this locally; so did this
track (`common.local_fit`, verified bit-identical to `tcn.synthesis.fit` at zero
noise). It belongs in the core:

```python
-        self.choices=nn.ParameterList([nn.Parameter(torch.zeros(len(n.candidates)),...)])
+        g=None if seed is None else torch.Generator().manual_seed(int(seed))
+        self.choices=nn.ParameterList([nn.Parameter(init_noise*torch.randn(len(n.candidates),generator=g),...)])
```
with `init_noise` defaulting to 0 to preserve every recorded result, and
`tcn.synthesis.fit` gaining a matching pass-through. Without this, "success rate
over N seeds" is not a measurable quantity for pure synthesis.

### P3. Relaxation temperatures are in the units of the data and are unreachable from a scaffold

`SoftProgram.temperatures` is initialised to `1.0` for every node and is written
only by `Crystallizer`. `Node` has no temperature field, so a scaffold cannot
declare one, and `synthesis.fit` never sets one. Measured consequence
(section 4.1): on byte-valued data `eq`'s surrogate needs `tau ~ 1.6e4` -- about
`(255/2)^2` -- to produce a gradient at all; at the shipped `tau=1` it is
`0.0` with gradient `0.0` beyond 11 grey levels. `lt`'s `sigmoid(d/tau)` is
saturated over the same range. ARCHITECTURE section 5 already says temperature
is "experiment configuration, not a settled constant"; there is currently no
interface through which to configure it. Minimal change: a `temperature: float = 1.0`
field on `Node`, consumed in `SoftProgram.__init__`.

### P4. `role="byte"` makes image intensities non-numeric, which removes every aggregation

`tcn/types.py`: `numeric` excludes `{"category","symbol","byte"}`. Excluding
category and symbol is exactly right and section 1 of ARCHITECTURE argues for
it. Excluding `byte` is a different claim: a channel intensity **is** a
measurement, not an identifier, and the exclusion is what forbids `sum`, `mean`,
`reduce_max`, `fft` and every arithmetic operator on images (section 4.1). The
present escape hatch, `pack`, is a `gradient="none"` boundary, so the numeric
view of an image is reachable only by a search that does not use gradients.

Two candidate resolutions, both requiring a spec decision rather than a patch:
either drop `"byte"` from the non-numeric set and let intensities be ordinary
`int[8]` measurements; or keep it and add an explicit, differentiable
`dequantize(BYTE) -> scalar` contract with a declared scale, which is what the
representation family in ARCHITECTURE section 2 already promises
("encode/decode name a representation"). The second is more in keeping with the
constitution. Either way it should be a deliberate decision, because as shipped
**no learnable transformation of an image exists other than per-byte equality**.

### P5. Set capacity is part of the type, so no fixpoint over sets is expressible

Section 3.1. `join`/`pair` multiply capacity and nothing narrows it, so
`generators/relations`' own `closure` latent is not the output type of any
program. The minimal fix is a declared narrowing conversion
`restrict(set[T,m]) -> set[T,n]` with an explicit overflow contract (error /
truncate-by-canonical-order), which is the set analogue of the existing
`quantize` for integers. Without it, section 9's `S -> P` curriculum edge
("sets and relations" -> "predicate logic") has no expressible target.

### P6. Give the search a discrete backend for input-binding choices

This is the same recommendation track 8 reached from a different direction, and
this track localises it. Section 4.4 measures that relaxing a node's *operator
or constant* choice is benign (12/12 successes searching a background colour
over 256 candidates) while relaxing its *input binding* is not (0/12 at 108
address candidates). The reason is structural: a convex mixture of candidate
*operators* is a blend of functions evaluated at a valid input, whereas a convex
mixture of candidate *input bindings* is a blend of unrelated values that is
generally not a meaningful value of the type -- an average of pixel 3 and pixel
57 denotes nothing. `tcn/search.py` already enumerates exactly this space. A
NEAR-shaped split -- enumerate or solve the binding choices, relax only the
operator and continuous parameters -- follows directly from the measurement.

---

## 8. Threats to validity

* **Scaffold authorship.** Every scaffold here was written by me with the
  reference program in mind, so these are *best-case* difficulty measurements:
  the answer is always in the space, usually with several equivalent forms
  (enumeration reports `unique=False` on most cells). A scaffold discovered
  rather than supplied would be harder, not easier.
* **`local_fit`.** The seed variation needed to report a success *rate* does not
  exist in `tcn.synthesis.fit` (P2). `check_local_fit.py` shows the copy is
  bit-identical at `init_noise=0` on the shipped `examples/mixed.py` fixture at
  two budgets, but it is still a copy.
* **Budgets.** 400-600 Adam steps at lr .05, `polish` where constants exist,
  crystallisation off (`freeze=False`) because track 1 measured the scheduler
  contributing nothing at working budgets. Track 3 measured a 32x step increase
  moving success by <= 6 pp on its tasks; I did not re-establish that here
  beyond the 300 vs 1500 step check in section 2.3, which changed nothing.
* **Derived targets.** Rung 3 and rung 4 supervise on equivalence classes of the
  `depth` and `text` probes rather than on the probes themselves; this is
  privileged supervision and section 4.2/5.1 justify why the raw probes are not
  usable. Rung 2's dense targets are derived from *observables*, which is a
  weaker warrant, and is flagged in the code and the table.
* **The `camera` action in rung 3** changes the data distribution relative to
  the shipped configuration. It is a declared action, not a code change, and
  section 4.2 gives the measurement that motivates it (0.98 background at every
  shipped resolution). Numbers for the shipped camera are in `out/audit.json`.
* **Two-step reachability** in rung 2 is not the closure: it agrees with the
  `target` probe on 99.5% of episodes at `entities=4`. Exact-conformance figures
  there are therefore capped slightly below 1 by the task, not by the method,
  which is why held-out accuracy is reported alongside.

---

## 9. The ladder, end to end

| rung | task | agent input | supervision | largest space gradient synthesis solves | discrete reference | verdict |
|---|---|---|---|---|---|---|
| **1a** | `signal`: frequency of a sinusoid | 4 raw samples | `frequencies` latent | **144** programs (8/16); 0/16 from 288 | enumeration solves 829,440 in 77 s, held-out error 1e-7 | works up to ~150 programs -- the analytic ceiling |
| 1b | `signal`: phase | 4 raw samples + `time` | `phases` latent | not run | reference program exists, worst error 2.5e-6 over 30 episodes | expressible via `atan2`; its prefix is 1a, so it is already past the wall |
| 1c | `signal`: `future` (t+1, t+2, t+4) | 4 raw samples | `future` probe (+ `frequencies`) | 0/12 at every scaffold from 2,880 up | exact reference exists (error 1.1e-6) | fails; smallest scaffold is already past the wall |
| **2a** | `relations`: transitive closure | `edges`, `query` | `closure` latent | -- | -- | **inexpressible**: `join` inflates set capacity 64 -> 4096 and nothing narrows it |
| 2b | `relations`: `target` (<=2-step reachability) | `edges`, `query` | `target` probe | **65,536** with dense probes (8/8, held-out accuracy 0.943 against a ceiling of 0.984); 1/8 output-only; 0/8 from 4.3e9 | enumeration solves 65,536 in 33-46 s; random density 9e-05 | works above the set layer only; the 65,536 wiring choices below it get `grad = None` |
| **3a** | `geometry`: `depth`, `object_ids`, `normals` | pixels | probes | -- | -- | **not functions of the observation**; and at the shipped camera 98% of every image is background |
| 3b | `geometry`: background, addresses pinned | pixels | equivalence class of `depth` | per-pixel mask **3.5e13** (12/12, held-out error 0, but factorises); entangled `allbg` head **8.4e6** and **9.0e15** (12/12 generalising with dense probes, 0/12 and 1/12 without); colour searched over the full byte alphabet **65,536** (12/12 at 2x2 and 4x4) | enumeration certifies the colour solution **unique** in 7.9 s (2x2) and 14.1 s (4x4) | **works -- the highest rung that does** |
| 3c | `geometry`: background, addresses free | pixels | equivalence class of `depth` | **fails from 576** (1/12); 0/12 from 46,656 | enumeration solves 147,456 in 71 s, held-out error 0 at every width | fails |
| **4a** | `raster_text`: `text` probe | pixels | `text` probe | -- | -- | **not addressable**: 448^64 byte-head programs, no aggregation operator |
| 4b | `raster_text`: 'a' vs 'b', `eq` route | 192 pixel bytes | equivalence class of `text` | 0/8 | enumeration **exhausts 49,152 programs and certifies no solution exists** | provably inexpressible in the differentiable sub-algebra |
| 4c | `raster_text`: 'a' vs 'b', threshold route | 192 pixel bytes | equivalence class of `text` | **0/8** in all five configurations, including with the address pinned to a byte that works; every logit gradient is `None` or `0.0` | enumeration over address x comparison x 256 thresholds (98,304 programs) solves it in **203 s** with **held-out accuracy 1.0** | solvable, and solvable only through `pack`, a gradient boundary |

**Where the ceiling is.** Two ceilings, and they are not the same one. The
highest *rung* that works is 3b; the largest *non-factorising* discrete space
any of these tasks was solved in is 8,388,608 (rung 3 `allbg` at 2x2, with dense
probes), and the smallest one that defeats the method is 288 (rung 1 `L2_trig`)
-- a five-order-of-magnitude spread that depends entirely on what kind of choice
is being relaxed, not on how many choices there are.

*For continuous/analytic content* the ceiling is a **discrete-space ceiling of
roughly 150 candidate programs**, and it is caused by the relaxation carrying
only a ~1.3x-per-node hint about the right operator -- enough for two or three
nodes, never enough for six. Exhaustive enumeration over the same space is four
orders of magnitude past it and finishes in under two minutes.

*For perceptual content* the ceiling is **representational and sits before any
search question is asked**. Images enter the algebra as tuples of non-numeric
bytes. There is no aggregation of any kind over them; the only differentiable
predicate is an equality whose surrogate is identically zero beyond 11 grey
levels; the only route to arithmetic is `pack`, which declares no gradient; and
the address of the byte you want is exactly the kind of choice for which relaxed
search performs below chance. A perception rung is therefore not blocked by
scale or by budget. It is blocked by the fact that **the operator algebra
contains no learnable transformation of an image other than per-byte equality.**

## 10. What to run next

1. **Apply P1-P3.** They are small, they are bugs, and P2 means no existing
   per-seed synthesis number on this repo was measuring what it claimed.
2. **Settle P4 as a specification question before any further perception work.**
   Until an image intensity is either numeric or has a declared differentiable
   dequantisation, "learn a detector rather than being handed one" is not a
   thing this algebra can do, at any budget.
3. **Take P6 seriously and split the backend by choice kind.** The measurement
   is unusually clean: constants and operators at fixed inputs, relax; input
   bindings, enumerate. `tcn/search.py` already enumerates the right space.
4. **Re-run the dense-probe result on tasks whose output loss actually entangles
   the intermediates.** Section 6 shows the mechanism is real but conditional,
   and the condition is not stated anywhere in `ARCHITECTURE.md` section 7.
5. **Do not use `generators/geometry` at its shipped camera** for anything that
   reads its probes; 98% background makes every accuracy number a report on a
   constant.
