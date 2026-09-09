# Object identity: bound the context first, then search it

Research track `object-identity`, continuing directly from
`research/discrete-perception/RESULTS.md` rung 3.5.  Everything below was
produced in this workspace with
`/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python`.
**Nothing under `tcn/` or `generators/` was modified**; the two changes this
track would ask for are written as diffs in section 9.  `tables.py` renders
every table here from `out/*.json`, so no number is transcribed by hand.

---

## 0. Verdict

**The integer label `object_ids` is not a function of the observation at any
context size, up to and including the entire image, and this is proved
structurally rather than estimated.  The permutation-invariant content of that
label -- the same-object *relation* -- is learnable from a two-pixel context,
exactly, and the program that computes it was found by exhaustive search: held-out
max error 0.0 over 393216 programs, applied afterwards at
every position of images up to 3,072 bytes with 1 wrong slot
in 5520.**

Five results, in the order they should change what the project does.

1. **The ceiling is not statistical, it is a symmetry.**  Permuting the
   generator's object list leaves the rendered image bit-identical in
   8/8 episodes and changes `object_ids` in
   8/8; re-drawing every object's colour leaves `object_ids`
   identical in 8/8 and changes the image in all of them.
   So the observation determines the labels' *partition* and nothing about their
   *names*.  No context bound can be above baseline for `object_ids` and no
   search over any context can succeed, and the ceiling table in section 2.2
   confirms that at every context measured.  The previous track's per-pixel
   certificate is a special case of this.

2. **The same-object relation is determined by two pixels, and the discriminator
   is collinearity, not equality.**  The renderer paints
   `clip(base_colour * shade)`, so every pixel of one object lies on that
   object's colour ray.  Over the training episodes the integer cross product
   `|r1*g2 - r2*g1|` is at most **140** across
   282 adjacent same-object pairs and at least
   **231** across 34 adjacent different-object
   pairs -- a clean margin.  Every equality-based context in the table is at
   advantage 0.0000 on the same target; adding collinearity takes it to
   **1.0000** held-out against a 0.8842 baseline.

3. **`pack` is the whole difference, and it is legal.**  Pixels are
   `role="byte"` and therefore non-numeric, so equality is all the algebra can
   do to them directly -- which is exactly why the previous track's four
   candidate families were certified empty.  `pack` on a one-field tuple is the
   registered conversion out of the byte carrier, `encode` widens the result,
   and the collinearity test is then ordinary `mul`/`sub`/`abs`/`le`.  No
   operator was added and no type rule bent.

4. **Enumeration settles this rung, and the reason the relaxed path does not is
   now three measured mechanisms rather than one number.**  Every gradient arm
   in section 3 was run on a surrogate that is **exactly 0.0** at the operating
   distance: `relaxed` computes `le` as `sigmoid(d/tau)` at `tau = 1`, which
   underflows in float32 at a gap of 89, and this rung's gaps have
   median 192 with 0.906 of records past it.  Those
   `0/4` and `1/4` figures are therefore not evidence
   about learnability.  Re-run with a live surrogate under three temperature
   policies the arms are 0/4, 0/4 and
   1/4, and the landscape measurement in section 4.4 explains why:
   the **shipped** temperature puts the loss minimum on a *correct* threshold
   with zero gradient, and every temperature large enough to restore the
   gradient moves the minimum onto a *wrong* one -- at `tau = 2^bits` the whole
   loss spread collapses to 5.96e-08.  This rung's relaxed
   landscape is a plateau with cliffs: enumeration walks it and gradient descent
   cannot.  One boundary survives all of that and is genuine: `shifted`, the
   neighbour offset, is `grad = None` under every policy because `pack` declares
   `gradient="none"`.

5. **A certificate about a family is not a certificate about a target, and the
   candidate pool is part of the family.**  The coarse eight-value threshold
   pool omits the entire separating interval; the search correctly exhausts it,
   reports 10 conforming programs, and returns one that is
   wrong on exactly two held-out positions.  Widening the pool to 512 values
   fixes it: **858** programs are conforming on training and exact at
   every validation position, held-out max error **0.0**.  The two
   wrong positions are a foreground pixel whose shaded colour lies on the
   background's ray, and `residual.py` names them.

---

## 1. Protocol

Same as the previous track, and stated again because two things changed.

* Program inputs are filled **only** from `StepRecord.actor_view().observations`
  (`pixels`, the raw byte tuple).  `latent_states` and `probes` build `targets`
  and nothing else.  Probes are supervision, never model inputs.
* Episodes come from disjoint seed ranges: 12 training episodes
  from seed 0, validation from seed 50, held-out from seed 100 on the `test`
  split, application from seed 200.
* One declared `camera` action moves the eye in, for the reason the previous
  track measured (98% background at the shipped camera).  It is an action from
  the generator's own schema, not a code change.
* The two information-bound certificates in section 2.1 call the generator's
  own `render()` as an **audit oracle on generator state**.  That is not an
  agent input and no program in this track sees it.
* A run succeeds only if the hardened program reproduces the targets exactly at
  1e-6, and the selection rule is exactness at **every** position of a
  validation split, which is what the previous track found necessary.
* **`SoftProgram` zero-initialises every choice logit, so `torch.manual_seed`
  does not vary synthesis.**  Every gradient number here comes from
  `research/discrete-perception/common.py:local_fit` with `init_noise = 0.5`,
  which that track verified bit-identical to `tcn.synthesis.fit` at zero noise.
* **The host is shared and loaded** (load average 28-81 on 20 cores, other agents
  included), and up to four of this track's runs overlapped deliberately.  Every
  wall clock is an upper bound; programs evaluated, node evaluations and
  choice-logit gradients are the load-independent measures and are reported
  beside every time.
* **Every gradient arm reports the surrogate's value at its operating
  distance.**  Section 4.2 is why: a `0/n` next to a surrogate of 0.0 is a
  statement about the relaxation, not about the task.

---

## 2. The information bounds, computed before any search

### 2.1 Two structural certificates

These are statements about the renderer, not estimates from samples.

| certificate | measurement |
|---|---|
| permute the generator's object list, re-render | rgb bit-identical in **8/8** episodes, depth identical in **8/8**, `object_ids` identical in **0/8**, `object_ids` identical *after remapping through the permutation* in **8/8**; 265 of the 265 foreground pixels across those 8 episodes change label -- all of them |
| re-draw every object's colour, re-render | `object_ids` identical in **8/8**, rgb identical in **0/8**, 265 pixels changed across the 8 episodes |

Read together they say: the observation is invariant to the transformation that
changes the labels, and the labels are invariant to the transformation that
changes the observation.  `object_ids` is therefore **not a function of the
image**, so it is not a function of any context inside the image either --
a pixel, a 3x3 window, a global aggregate, or all 6,912 bytes.  A search over
any of them is a search for a function that does not exist.

The supporting statistic: of 208 distinct foreground colours over
12 training and 12 held-out episodes,
**0** occur in more than one episode (fraction
0.0000).  A colour essentially never recurs, which is why every
lookup-table ceiling below memorises and transfers nothing.

The obstruction inside a single image, restated from the previous track and
re-measured here at R=8: mean **2.89** distinct RGB values
per object, max 18, only 0.342 of object instances
single-coloured.

The consequence for equality is worth spelling out, because it is why every
equality-based context in section 2.3 is at advantage exactly zero.  Among
adjacent **different-object** pairs, 0.0000 have the same
colour -- so equal colours *do* imply the same object.  But among adjacent
**same-object** pairs, **0.224** have different colours,
because the renderer shades per face.  The rule "equal colour means same object,
otherwise different" therefore misses about a fifth of the same-object pairs,
which costs more than the constant `same` predictor gains.  Colour equality is
sound and badly incomplete, and there is no threshold to tune: the predicate has
no free parameter.

### 2.2 The ceiling table: `object_ids` and its relabellings

The best possible predictor of each target from exactly each context, fitted as
a lookup table on the training episodes and scored on held-out episodes.  It is
computed outside the operator algebra on purpose: it upper-bounds every program
that reads only that context and is required to fit the training records.
"key recurrence" is the fraction of held-out records whose context key was seen
in training, and it is the diagnostic that explains the whole table.

| context | target | keys | key recurrence | train acc | held-out acc | majority baseline | advantage |
|---|---|---|---|---|---|---|---|
| one pixel (3 bytes) | object_ids | 111 | 0.469 | 1.0000 | 0.4688 | 0.4688 | +0.0000 |
| one pixel (3 bytes) | raster_rank | 111 | 0.469 | 1.0000 | 0.4688 | 0.4688 | +0.0000 |
| one pixel (3 bytes) | is_object_0 | 111 | 0.469 | 1.0000 | 0.9701 | 0.9701 | +0.0000 |
| pixel + its row/column | object_ids | 459 | 0.469 | 1.0000 | 0.4688 | 0.4688 | +0.0000 |
| pixel + its row/column | raster_rank | 459 | 0.469 | 1.0000 | 0.4688 | 0.4688 | +0.0000 |
| pixel + its row/column | is_object_0 | 459 | 0.469 | 1.0000 | 0.9701 | 0.9701 | +0.0000 |
| pixel + right neighbour (6 bytes) | object_ids | 208 | 0.422 | 1.0000 | 0.4688 | 0.4688 | +0.0000 |
| pixel + right neighbour (6 bytes) | raster_rank | 208 | 0.422 | 1.0000 | 0.4688 | 0.4688 | +0.0000 |
| pixel + right neighbour (6 bytes) | is_object_0 | 208 | 0.422 | 1.0000 | 0.9701 | 0.9701 | +0.0000 |
| pixel + 4-neighbourhood (15 bytes) | object_ids | 424 | 0.301 | 1.0000 | 0.4688 | 0.4688 | +0.0000 |
| pixel + 4-neighbourhood (15 bytes) | raster_rank | 424 | 0.301 | 1.0000 | 0.4688 | 0.4688 | +0.0000 |
| pixel + 4-neighbourhood (15 bytes) | is_object_0 | 424 | 0.301 | 1.0000 | 0.9701 | 0.9701 | +0.0000 |
| 3x3 window (27 bytes) | object_ids | 498 | 0.264 | 1.0000 | 0.4688 | 0.4688 | +0.0000 |
| 3x3 window (27 bytes) | raster_rank | 498 | 0.264 | 1.0000 | 0.4688 | 0.4688 | +0.0000 |
| 3x3 window (27 bytes) | is_object_0 | 498 | 0.264 | 1.0000 | 0.9701 | 0.9701 | +0.0000 |
| pixel + global colour rank | object_ids | 116 | 0.349 | 1.0000 | 0.4688 | 0.4688 | +0.0000 |
| pixel + global colour rank | raster_rank | 116 | 0.349 | 1.0000 | 0.4688 | 0.4688 | +0.0000 |
| pixel + global colour rank | is_object_0 | 116 | 0.349 | 1.0000 | 0.9701 | 0.9701 | +0.0000 |
| global colour rank alone | object_ids | 21 | 1.000 | 0.6016 | 0.3880 | 0.4688 | **-0.0807** |
| global colour rank alone | raster_rank | 21 | 1.000 | 0.6576 | 0.4036 | 0.4688 | **-0.0651** |
| global colour rank alone | is_object_0 | 21 | 1.000 | 0.9310 | 0.9701 | 0.9701 | +0.0000 |
| = to right neighbour (1 bit) | object_ids | 2 | 1.000 | 0.4857 | 0.4688 | 0.4688 | +0.0000 |
| = to right neighbour (1 bit) | raster_rank | 2 | 1.000 | 0.4857 | 0.4688 | 0.4688 | +0.0000 |
| = to right neighbour (1 bit) | is_object_0 | 2 | 1.000 | 0.9310 | 0.9701 | 0.9701 | +0.0000 |
| = to each of 4 neighbours (4 bits) | object_ids | 16 | 1.000 | 0.5169 | 0.4583 | 0.4688 | **-0.0104** |
| = to each of 4 neighbours (4 bits) | raster_rank | 16 | 1.000 | 0.5404 | 0.5208 | 0.4688 | **+0.0521** |
| = to each of 4 neighbours (4 bits) | is_object_0 | 16 | 1.000 | 0.9310 | 0.9701 | 0.9701 | +0.0000 |
| = to each of 8 neighbours (8 bits) | object_ids | 86 | 0.975 | 0.5664 | 0.4531 | 0.4688 | **-0.0156** |
| = to each of 8 neighbours (8 bits) | raster_rank | 86 | 0.975 | 0.5911 | 0.5013 | 0.4688 | **+0.0326** |
| = to each of 8 neighbours (8 bits) | is_object_0 | 86 | 0.975 | 0.9310 | 0.9701 | 0.9701 | +0.0000 |
| is-background + 4 equalities | object_ids | 32 | 1.000 | 0.6758 | 0.5143 | 0.4688 | **+0.0456** |
| is-background + 4 equalities | raster_rank | 32 | 1.000 | 0.7422 | 0.6706 | 0.4688 | **+0.2018** |
| is-background + 4 equalities | is_object_0 | 32 | 1.000 | 0.9375 | 0.9206 | 0.9701 | **-0.0495** |
| is-background + 8 equalities | object_ids | 132 | 0.953 | 0.7292 | 0.5404 | 0.4688 | **+0.0716** |
| is-background + 8 equalities | raster_rank | 132 | 0.953 | 0.7839 | 0.6393 | 0.4688 | **+0.1706** |
| is-background + 8 equalities | is_object_0 | 132 | 0.953 | 0.9518 | 0.9115 | 0.9701 | **-0.0586** |
| is-background x2 + one equality | object_ids | 5 | 1.000 | 0.6107 | 0.5130 | 0.4688 | **+0.0443** |
| is-background x2 + one equality | raster_rank | 5 | 1.000 | 0.7135 | 0.6523 | 0.4688 | **+0.1836** |
| is-background x2 + one equality | is_object_0 | 5 | 1.000 | 0.9310 | 0.9701 | 0.9701 | +0.0000 |
| is-background x2 + collinearity | object_ids | 7 | 0.992 | 0.6107 | 0.5312 | 0.4688 | **+0.0625** |
| is-background x2 + collinearity | raster_rank | 7 | 0.992 | 0.7174 | 0.6224 | 0.4688 | **+0.1536** |
| is-background x2 + collinearity | is_object_0 | 7 | 0.992 | 0.9310 | 0.9701 | 0.9701 | +0.0000 |
| is-background + 4 collinearities | object_ids | 92 | 0.975 | 0.6680 | 0.5599 | 0.4688 | **+0.0911** |
| is-background + 4 collinearities | raster_rank | 92 | 0.975 | 0.7734 | 0.6432 | 0.4688 | **+0.1745** |
| is-background + 4 collinearities | is_object_0 | 92 | 0.975 | 0.9323 | 0.9701 | 0.9701 | +0.0000 |

**Every raw-byte context is at advantage 0.0000, and the reason is visible in
the recurrence column**: a wider window makes the key *rarer*, not more
informative -- recurrence falls from 0.469 at one pixel to
0.264 at a 3x3 window.  Enlarging the context strictly *lowers* what a fitted predictor
transfers, which is the opposite of what "more context" is supposed to buy.

Restricting to foreground pixels removes the free background component and makes
the same point without the baseline doing the work:

| context | target | keys | key recurrence | train acc | held-out acc | majority baseline | advantage |
|---|---|---|---|---|---|---|---|
| one pixel (3 bytes) | object_ids_fg | 110 | 0.000 | 1.0000 | 0.2500 | 0.3113 | **-0.0613** |
| one pixel (3 bytes) | raster_rank_fg | 110 | 0.000 | 1.0000 | 0.2696 | 0.4265 | **-0.1569** |
| pixel + its row/column | object_ids_fg | 395 | 0.000 | 1.0000 | 0.2500 | 0.3113 | **-0.0613** |
| pixel + its row/column | raster_rank_fg | 395 | 0.000 | 1.0000 | 0.2696 | 0.4265 | **-0.1569** |
| pixel + right neighbour (6 bytes) | object_ids_fg | 182 | 0.000 | 1.0000 | 0.2500 | 0.3113 | **-0.0613** |
| pixel + right neighbour (6 bytes) | raster_rank_fg | 182 | 0.000 | 1.0000 | 0.2696 | 0.4265 | **-0.1569** |
| pixel + 4-neighbourhood (15 bytes) | object_ids_fg | 316 | 0.000 | 1.0000 | 0.2500 | 0.3113 | **-0.0613** |
| pixel + 4-neighbourhood (15 bytes) | raster_rank_fg | 316 | 0.000 | 1.0000 | 0.2696 | 0.4265 | **-0.1569** |
| 3x3 window (27 bytes) | object_ids_fg | 340 | 0.000 | 1.0000 | 0.2500 | 0.3113 | **-0.0613** |
| 3x3 window (27 bytes) | raster_rank_fg | 340 | 0.000 | 1.0000 | 0.2696 | 0.4265 | **-0.1569** |
| pixel + global colour rank | object_ids_fg | 110 | 0.000 | 1.0000 | 0.2500 | 0.3113 | **-0.0613** |
| pixel + global colour rank | raster_rank_fg | 110 | 0.000 | 1.0000 | 0.2696 | 0.4265 | **-0.1569** |
| global colour rank alone | object_ids_fg | 21 | 1.000 | 0.5013 | 0.1985 | 0.3113 | **-0.1127** |
| global colour rank alone | raster_rank_fg | 21 | 1.000 | 0.7139 | 0.3505 | 0.4265 | **-0.0760** |
| = to right neighbour (1 bit) | object_ids_fg | 2 | 1.000 | 0.2380 | 0.0760 | 0.3113 | **-0.2353** |
| = to right neighbour (1 bit) | raster_rank_fg | 2 | 1.000 | 0.4152 | 0.3358 | 0.4265 | **-0.0907** |
| = to each of 4 neighbours (4 bits) | object_ids_fg | 16 | 1.000 | 0.3696 | 0.0858 | 0.3113 | **-0.2255** |
| = to each of 4 neighbours (4 bits) | raster_rank_fg | 16 | 1.000 | 0.4987 | 0.3799 | 0.4265 | **-0.0466** |
| = to each of 8 neighbours (8 bits) | object_ids_fg | 71 | 0.958 | 0.4734 | 0.1520 | 0.3113 | **-0.1593** |
| = to each of 8 neighbours (8 bits) | raster_rank_fg | 71 | 0.958 | 0.5797 | 0.3211 | 0.4265 | **-0.1054** |
| is-background + 4 equalities | object_ids_fg | 16 | 1.000 | 0.3696 | 0.0858 | 0.3113 | **-0.2255** |
| is-background + 4 equalities | raster_rank_fg | 16 | 1.000 | 0.4987 | 0.3799 | 0.4265 | **-0.0466** |
| is-background + 8 equalities | object_ids_fg | 71 | 0.958 | 0.4734 | 0.1520 | 0.3113 | **-0.1593** |
| is-background + 8 equalities | raster_rank_fg | 71 | 0.958 | 0.5797 | 0.3211 | 0.4265 | **-0.1054** |
| is-background x2 + one equality | object_ids_fg | 3 | 1.000 | 0.2430 | 0.0833 | 0.3113 | **-0.2279** |
| is-background x2 + one equality | raster_rank_fg | 3 | 1.000 | 0.4430 | 0.3456 | 0.4265 | **-0.0809** |
| is-background x2 + collinearity | object_ids_fg | 4 | 0.988 | 0.2430 | 0.1299 | 0.3113 | **-0.1814** |
| is-background x2 + collinearity | raster_rank_fg | 4 | 0.988 | 0.4506 | 0.2892 | 0.4265 | **-0.1373** |
| is-background + 4 collinearities | object_ids_fg | 49 | 0.993 | 0.3544 | 0.1716 | 0.3113 | **-0.1397** |
| is-background + 4 collinearities | raster_rank_fg | 49 | 0.993 | 0.5595 | 0.3358 | 0.4265 | **-0.0907** |

Every advantage is zero or negative.  This is section 2.1's certificate showing
up as a measurement.

### 2.3 The same-object relation, and the one context that clears the baseline

`same_right(i)` is `object_ids[i] == object_ids[i+1]` (baseline
0.8170), and `obj_edge_fg` is its restriction to pairs where
**both** pixels are foreground (baseline 0.8842) -- the genuinely new
part of segmentation, with the foreground/background boundary of rung 3.5
removed.

| context | target | keys | key recurrence | train acc | held-out acc | majority baseline | advantage |
|---|---|---|---|---|---|---|---|
| one pixel (3 bytes) | same_right | 102 | 0.461 | 0.8884 | 0.8170 | 0.8170 | +0.0000 |
| one pixel (3 bytes) | obj_edge_fg | 87 | 0.000 | 0.9557 | 0.8842 | 0.8842 | +0.0000 |
| pixel + its row/column | same_right | 421 | 0.461 | 0.9524 | 0.8140 | 0.8170 | -0.0030 |
| pixel + its row/column | obj_edge_fg | 316 | 0.000 | 1.0000 | 0.8842 | 0.8842 | +0.0000 |
| pixel + right neighbour (6 bytes) | same_right | 193 | 0.408 | 1.0000 | 0.8170 | 0.8170 | +0.0000 |
| pixel + right neighbour (6 bytes) | obj_edge_fg | 135 | 0.000 | 1.0000 | 0.8842 | 0.8842 | +0.0000 |
| pixel + 4-neighbourhood (15 bytes) | same_right | 382 | 0.289 | 1.0000 | 0.8170 | 0.8170 | +0.0000 |
| pixel + 4-neighbourhood (15 bytes) | obj_edge_fg | 247 | 0.000 | 1.0000 | 0.8842 | 0.8842 | +0.0000 |
| 3x3 window (27 bytes) | same_right | 446 | 0.247 | 1.0000 | 0.8170 | 0.8170 | +0.0000 |
| 3x3 window (27 bytes) | obj_edge_fg | 268 | 0.000 | 1.0000 | 0.8842 | 0.8842 | +0.0000 |
| pixel + global colour rank | same_right | 106 | 0.339 | 0.8884 | 0.8170 | 0.8170 | +0.0000 |
| pixel + global colour rank | obj_edge_fg | 87 | 0.000 | 0.9557 | 0.8842 | 0.8842 | +0.0000 |
| global colour rank alone | same_right | 20 | 1.000 | 0.8229 | 0.8051 | 0.8170 | **-0.0119** |
| global colour rank alone | obj_edge_fg | 20 | 1.000 | 0.8924 | 0.8842 | 0.8842 | +0.0000 |
| = to right neighbour (1 bit) | same_right | 2 | 1.000 | 0.8229 | 0.8170 | 0.8170 | +0.0000 |
| = to right neighbour (1 bit) | obj_edge_fg | 2 | 1.000 | 0.8924 | 0.8842 | 0.8842 | +0.0000 |
| = to each of 4 neighbours (4 bits) | same_right | 16 | 1.000 | 0.8467 | 0.8318 | 0.8170 | **+0.0149** |
| = to each of 4 neighbours (4 bits) | obj_edge_fg | 16 | 1.000 | 0.8924 | 0.8842 | 0.8842 | +0.0000 |
| = to each of 8 neighbours (8 bits) | same_right | 83 | 0.969 | 0.8780 | 0.8408 | 0.8170 | **+0.0238** |
| = to each of 8 neighbours (8 bits) | obj_edge_fg | 67 | 0.949 | 0.8987 | 0.8842 | 0.8842 | +0.0000 |
| is-background + 4 equalities | same_right | 31 | 0.994 | 0.8884 | 0.8735 | 0.8170 | **+0.0565** |
| is-background + 4 equalities | obj_edge_fg | 16 | 1.000 | 0.8924 | 0.8842 | 0.8842 | +0.0000 |
| is-background + 8 equalities | same_right | 121 | 0.939 | 0.9003 | 0.8571 | 0.8170 | **+0.0402** |
| is-background + 8 equalities | obj_edge_fg | 67 | 0.949 | 0.8987 | 0.8842 | 0.8842 | +0.0000 |
| is-background x2 + one equality | same_right | 5 | 1.000 | 0.9494 | 0.9464 | 0.8170 | **+0.1295** |
| is-background x2 + one equality | obj_edge_fg | 2 | 1.000 | 0.8924 | 0.8842 | 0.8842 | +0.0000 |
| is-background x2 + collinearity | same_right | 5 | 0.991 | 0.9970 | 0.9911 | 0.8170 | **+0.1741** |
| is-background x2 + collinearity | obj_edge_fg | 2 | 1.000 | 0.9937 | 1.0000 | 0.8842 | **+0.1158** |
| is-background + 4 collinearities | same_right | 70 | 0.979 | 0.9970 | 0.9777 | 0.8170 | **+0.1607** |
| is-background + 4 collinearities | obj_edge_fg | 35 | 0.997 | 0.9937 | 0.9968 | 0.8842 | **+0.1125** |

Read the `obj_edge_fg` rows first.  **Every equality-based context is at
advantage exactly 0.0000** -- one pixel, a pixel pair, a 4-neighbourhood, a 3x3
window, the equality patterns, and the equality patterns with a background test.
The gains that the raw `same_right` rows show for `bg_pair_right` and
`eqbg_cross4` are the foreground/background part, which rung 3.5 already had.

The two `chroma` rows are the exception, and they are the result: adding one
colour-invariant bit -- do these two pixels lie on a common ray through the
origin -- takes `obj_edge_fg` from advantage 0.0000 to **1.0000 held-out**
against a 0.8842 baseline, with key recurrence 1.000.

The controls behave as the previous track reports -- `fg` is the rung-3 target
and is exact from one pixel, `fg_edge` is rung 3.5 and needs the pair:

| context | target | keys | key recurrence | train acc | held-out acc | majority baseline | advantage |
|---|---|---|---|---|---|---|---|
| one pixel (3 bytes) | fg | 111 | 0.469 | 1.0000 | 1.0000 | 0.5312 | **+0.4688** |
| one pixel (3 bytes) | fg_edge | 102 | 0.461 | 0.9018 | 0.8705 | 0.8705 | +0.0000 |
| one pixel (3 bytes) | near_fg | 110 | 0.000 | 0.9544 | 0.3799 | 0.6201 | **-0.2402** |
| pixel + its row/column | fg | 459 | 0.469 | 1.0000 | 1.0000 | 0.5312 | **+0.4688** |
| pixel + its row/column | fg_edge | 421 | 0.461 | 0.9524 | 0.8676 | 0.8705 | -0.0030 |
| pixel + its row/column | near_fg | 395 | 0.000 | 1.0000 | 0.3799 | 0.6201 | **-0.2402** |
| pixel + right neighbour (6 bytes) | fg | 208 | 0.422 | 1.0000 | 0.9531 | 0.5312 | **+0.4219** |
| pixel + right neighbour (6 bytes) | fg_edge | 193 | 0.408 | 1.0000 | 0.8705 | 0.8705 | +0.0000 |
| pixel + right neighbour (6 bytes) | near_fg | 182 | 0.000 | 0.9595 | 0.3799 | 0.6201 | **-0.2402** |
| pixel + 4-neighbourhood (15 bytes) | fg | 424 | 0.301 | 1.0000 | 0.8320 | 0.5312 | **+0.3008** |
| pixel + 4-neighbourhood (15 bytes) | fg_edge | 382 | 0.289 | 1.0000 | 0.8705 | 0.8705 | +0.0000 |
| pixel + 4-neighbourhood (15 bytes) | near_fg | 316 | 0.000 | 0.9848 | 0.3799 | 0.6201 | **-0.2402** |
| 3x3 window (27 bytes) | fg | 498 | 0.264 | 1.0000 | 0.7956 | 0.5312 | **+0.2643** |
| 3x3 window (27 bytes) | fg_edge | 446 | 0.247 | 1.0000 | 0.8705 | 0.8705 | +0.0000 |
| 3x3 window (27 bytes) | near_fg | 340 | 0.000 | 0.9924 | 0.3799 | 0.6201 | **-0.2402** |
| pixel + global colour rank | fg | 116 | 0.349 | 1.0000 | 0.8802 | 0.5312 | **+0.3490** |
| pixel + global colour rank | fg_edge | 106 | 0.339 | 0.9018 | 0.8705 | 0.8705 | +0.0000 |
| pixel + global colour rank | near_fg | 110 | 0.000 | 0.9544 | 0.3799 | 0.6201 | **-0.2402** |
| global colour rank alone | fg | 21 | 1.000 | 0.7695 | 0.6250 | 0.5312 | **+0.0938** |
| global colour rank alone | fg_edge | 20 | 1.000 | 0.8735 | 0.8705 | 0.8705 | +0.0000 |
| global colour rank alone | near_fg | 21 | 1.000 | 0.7646 | 0.4730 | 0.6201 | **-0.1471** |
| = to right neighbour (1 bit) | fg | 2 | 1.000 | 0.6615 | 0.6367 | 0.5312 | **+0.1055** |
| = to right neighbour (1 bit) | fg_edge | 2 | 1.000 | 0.8735 | 0.8705 | 0.8705 | +0.0000 |
| = to right neighbour (1 bit) | near_fg | 2 | 1.000 | 0.6127 | 0.5686 | 0.6201 | **-0.0515** |
| = to each of 4 neighbours (4 bits) | fg | 16 | 1.000 | 0.7201 | 0.6810 | 0.5312 | **+0.1497** |
| = to each of 4 neighbours (4 bits) | fg_edge | 16 | 1.000 | 0.8765 | 0.8586 | 0.8705 | **-0.0119** |
| = to each of 4 neighbours (4 bits) | near_fg | 16 | 1.000 | 0.6684 | 0.6569 | 0.6201 | **+0.0368** |
| = to each of 8 neighbours (8 bits) | fg | 86 | 0.975 | 0.7539 | 0.6966 | 0.5312 | **+0.1654** |
| = to each of 8 neighbours (8 bits) | fg_edge | 83 | 0.969 | 0.9018 | 0.8690 | 0.8705 | -0.0015 |
| = to each of 8 neighbours (8 bits) | near_fg | 71 | 0.958 | 0.7620 | 0.6471 | 0.6201 | **+0.0270** |
| is-background + 4 equalities | fg | 32 | 1.000 | 1.0000 | 1.0000 | 0.5312 | **+0.4688** |
| is-background + 4 equalities | fg_edge | 31 | 0.994 | 0.9271 | 0.9182 | 0.8705 | **+0.0476** |
| is-background + 4 equalities | near_fg | 16 | 1.000 | 0.6684 | 0.6569 | 0.6201 | **+0.0368** |
| is-background + 8 equalities | fg | 132 | 0.953 | 1.0000 | 0.9753 | 0.5312 | **+0.4440** |
| is-background + 8 equalities | fg_edge | 121 | 0.939 | 0.9360 | 0.8973 | 0.8705 | **+0.0268** |
| is-background + 8 equalities | near_fg | 71 | 0.958 | 0.7620 | 0.6471 | 0.6201 | **+0.0270** |
| is-background x2 + one equality | fg | 5 | 1.000 | 1.0000 | 1.0000 | 0.5312 | **+0.4688** |
| is-background x2 + one equality | fg_edge | 5 | 1.000 | 1.0000 | 1.0000 | 0.8705 | **+0.1295** |
| is-background x2 + one equality | near_fg | 3 | 1.000 | 0.6127 | 0.5686 | 0.6201 | **-0.0515** |
| is-background x2 + collinearity | fg | 7 | 0.992 | 1.0000 | 0.9987 | 0.5312 | **+0.4674** |
| is-background x2 + collinearity | fg_edge | 5 | 0.991 | 1.0000 | 0.9911 | 0.8705 | **+0.1205** |
| is-background x2 + collinearity | near_fg | 4 | 0.988 | 0.6228 | 0.5882 | 0.6201 | **-0.0319** |
| is-background + 4 collinearities | fg | 92 | 0.975 | 1.0000 | 0.9792 | 0.5312 | **+0.4479** |
| is-background + 4 collinearities | fg_edge | 70 | 0.979 | 0.9658 | 0.9315 | 0.8705 | **+0.0610** |
| is-background + 4 collinearities | near_fg | 49 | 0.993 | 0.7671 | 0.7108 | 0.6201 | **+0.0907** |


### 2.4 The bound for the family the algebra can actually express

`same_chroma` above uses floating ratios, which `role="byte"` pixels cannot
produce.  What the algebra *can* produce, once `pack` strips the byte role, is
the integer cross product.  Exact equality is too strict because the renderer
truncates `base * shade` to uint8: over the training episodes the exact cross
product vanishes for only **0.5603** of same-object pairs (and
0.0000 of different-object pairs).  The expressible predicate is
therefore `|r1*g2 - r2*g1| <= T` with `T` a searched constant, and it separates
the classes with a margin:

* max cross product over adjacent **same-object** pairs: **140**
* min cross product over adjacent **different-object** pairs: **231**

so every `T` in [140, 231) is exact on the
training episodes.  Swept over `T`, on both offsets:

| T | right: train | right: held-out | down: train | down: held-out |
|---|---|---|---|---|
| 0 | 0.8155 | 0.8780 | 0.9092 | 0.9033 |
| 1 | 0.8155 | 0.8780 | 0.9092 | 0.9033 |
| 2 | 0.8155 | 0.8780 | 0.9092 | 0.9033 |
| 4 | 0.8155 | 0.8780 | 0.9092 | 0.9033 |
| 8 | 0.8185 | 0.8780 | 0.9092 | 0.9033 |
| 16 | 0.8393 | 0.8795 | 0.9152 | 0.9077 |
| 32 | 0.8646 | 0.9048 | 0.9211 | 0.9196 |
| 64 | 0.9390 | 0.9375 | 0.9658 | 0.9509 |
| 96 | 0.9554 | 0.9732 | 0.9851 | 0.9821 |
| 128 | 0.9851 | 0.9955 | 0.9955 | 0.9911 |
| 192 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 256 | 0.9985 | 0.9985 | 1.0000 | 0.9985 |
| 384 | 0.9985 | 0.9955 | 1.0000 | 0.9955 |
| 512 | 0.9955 | 0.9896 | 0.9985 | 0.9881 |
| 768 | 0.9807 | 0.9807 | 0.9836 | 0.9777 |
| 1024 | 0.9762 | 0.9732 | 0.9821 | 0.9717 |
| 2048 | 0.9196 | 0.9524 | 0.9330 | 0.9449 |
| 4096 | 0.8661 | 0.8824 | 0.8884 | 0.8943 |

Majority baseline 0.8170 (right) and 0.8467 (down).
**This is the number that authorised the search in section 3**, and it is the
only row in the whole bounds exercise that clears baseline on the target that
matters.

### 2.5 The generator's other probes, for completeness

`geometry` also probes `depth` and `normals`.  Restricted to foreground pixels,
so the background class cannot do the work:

| context | target | keys | key recurrence | train acc | held-out acc | majority baseline | advantage |
|---|---|---|---|---|---|---|---|
| one pixel (3 bytes) | near_fg | 110 | 0.000 | 0.9544 | 0.3799 | 0.6201 | **-0.2402** |
| one pixel (3 bytes) | shade_bin_fg | 110 | 0.000 | 1.0000 | 0.4559 | 0.4559 | +0.0000 |
| one pixel (3 bytes) | normal_up_fg | 110 | 0.000 | 0.9873 | 0.8529 | 0.8529 | +0.0000 |
| pixel + right neighbour (6 bytes) | near_fg | 182 | 0.000 | 0.9595 | 0.3799 | 0.6201 | **-0.2402** |
| pixel + right neighbour (6 bytes) | shade_bin_fg | 182 | 0.000 | 1.0000 | 0.4559 | 0.4559 | +0.0000 |
| pixel + right neighbour (6 bytes) | normal_up_fg | 182 | 0.000 | 0.9924 | 0.8529 | 0.8529 | +0.0000 |
| pixel + 4-neighbourhood (15 bytes) | near_fg | 316 | 0.000 | 0.9848 | 0.3799 | 0.6201 | **-0.2402** |
| pixel + 4-neighbourhood (15 bytes) | shade_bin_fg | 316 | 0.000 | 1.0000 | 0.4559 | 0.4559 | +0.0000 |
| pixel + 4-neighbourhood (15 bytes) | normal_up_fg | 316 | 0.000 | 1.0000 | 0.8529 | 0.8529 | +0.0000 |
| 3x3 window (27 bytes) | near_fg | 340 | 0.000 | 0.9924 | 0.3799 | 0.6201 | **-0.2402** |
| 3x3 window (27 bytes) | shade_bin_fg | 340 | 0.000 | 1.0000 | 0.4559 | 0.4559 | +0.0000 |
| 3x3 window (27 bytes) | normal_up_fg | 340 | 0.000 | 1.0000 | 0.8529 | 0.8529 | +0.0000 |
| pixel + global colour rank | near_fg | 110 | 0.000 | 0.9544 | 0.3799 | 0.6201 | **-0.2402** |
| pixel + global colour rank | shade_bin_fg | 110 | 0.000 | 1.0000 | 0.4559 | 0.4559 | +0.0000 |
| pixel + global colour rank | normal_up_fg | 110 | 0.000 | 0.9873 | 0.8529 | 0.8529 | +0.0000 |
| is-background + 8 equalities | near_fg | 71 | 0.958 | 0.7620 | 0.6471 | 0.6201 | **+0.0270** |
| is-background + 8 equalities | shade_bin_fg | 71 | 0.958 | 0.5873 | 0.4289 | 0.4559 | **-0.0270** |
| is-background + 8 equalities | normal_up_fg | 71 | 0.958 | 0.9494 | 0.8603 | 0.8529 | +0.0074 |
| is-background + 4 collinearities | near_fg | 49 | 0.993 | 0.7671 | 0.7108 | 0.6201 | **+0.0907** |
| is-background + 4 collinearities | shade_bin_fg | 49 | 0.993 | 0.5747 | 0.4289 | 0.4559 | **-0.0270** |
| is-background + 4 collinearities | normal_up_fg | 49 | 0.993 | 0.9241 | 0.8407 | 0.8529 | **-0.0123** |

Every advantage is zero or negative at every context, and the foreground-key
recurrence is 0.000 throughout: these are functions of the *base colour and the
facing*, and the observation carries only their product.  `depth` reproduces the
previous track's rung-5 result; `normals` behaves the same way and is reported
here for the first time.  The one thing collinearity does recover about shading
is a *ratio* -- two pixels of one object differ by the scalar `shade_i/shade_j`
-- which is exactly why the same-object relation is determined while the shading
itself is not.


---

## 3. The searched rungs

**Read every `gradient` column in this section together with section 4.**  They
were all measured with the shipped `le` surrogate, which section 4.2 shows is
*exactly* 0.0 at 91-100% of this rung's operating distances.  A gradient arm run
on a dead surrogate is not evidence about learnability, and section 4.4 re-runs
them with a live one.

Target: `same(i, i+k) = object_ids[i] == object_ids[i+k]` at every position,
supervised from the `object_ids` probe, with the offset `k` searched over
{+1 pixel, +2 pixels, +1 row}.  448 training records,
448 validation, 448 held-out
(0.759 positive).

### 3.1 Direct: the collinearity module, coarse threshold pool

| arm | space | evaluated | exhausted | conforming on train | + exact at every validation position | unique | enum s | held-out max error | held-out accuracy | gradient |
|---|---|---|---|---|---|---|---|---|---|---|
| direct (8-value pool) | 6144 | 6144 | True | 10 | 10 | no | 8.0 | 1.0 | 0.995536 | 0/4 (dead surrogate; see 4.4) |

10 of 6144 programs conform, and **all
10 of them are also exact at every position of the validation
split**, so the validation tie-break does not discriminate here -- unlike rung
3, where it was the difference between right and wrong.  The conforming set is
10/6144 against rung 3's 2,464/32,000, so this
probe identifies the program roughly 40x more tightly than the foreground probe
does.  A uniform random program from the same space conforms
0.0025 of the time.

Both selection rules return held-out accuracy 0.995536 with a max
error of 1.0: **two positions out of 448 are wrong**,
and section 3.2 shows that is the candidate pool, not the search.

### 3.2 The residual, and why it is the pool rather than the data

| split | pairs | majority baseline | exact interval for T | best accuracy | accuracy at T=192 | errors at T=192 |
|---|---|---|---|---|---|---|
| train | 448 | 0.7589 | [140, 230] | 1.000000 | 1.000000 | 0 |
| validation | 448 | 0.7478 | [144, 800] | 1.000000 | 1.000000 | 0 |
| test | 448 | 0.8013 | [155, 158] | 1.000000 | 0.995536 | 2 |

Every split has a threshold interval on which the predicate is *exact*, and the
intervals differ: [140, 230] on training, [144, 800] on
validation, [155, 158] on test, intersecting in [155, 158].  The
coarse pool `(0, 16, 48, 96, 192, 384, 1024, 4096)` contains **no value in that
intersection**, so no member of the coarse space can be exact on all three
splits.  The search exhausted its space and returned the best member of it.  The
two errors are all of one kind:

{"object vs background, collinear": 2}

and the offending pair is a foreground pixel whose shaded colour happens to lie
on the background's ray:

    {"seed": 105, "r": 3, "c": 4, "a": [24, 30, 43], "b": [75, 93, 128], "ia": -1, "ib": 3, "same": false, "cross": [18, 159, 153]}

**And the search improves on the analyst.**  The bound above is derived for
`max(cross products) <= T`, the conjunction of all three collinearity tests,
because that is the textbook form of the invariant.  The searched space contains
every Boolean combination of the three, and the program the wide arm returns is
`(rg or gb) and rb` -- `m1 = truth_1` is NOR and `same = truth_2` is
`not m1 and rb`.  It is not the shape the bound was computed for and it is
strictly more robust:

| predicate shape | exact T on train | on validation | on test | exact on all three | width | coarse-pool values inside it |
|---|---|---|---|---|---|---|
| AND of all three (the shape the bound was derived for) | [140, 230] | [144, 800] | [155, 158] | [155, 158] | 4 | **none** |
| (rg OR gb) AND rb -- the program the wide search returned | [135, 225] | [142, 533] | [114, 152] | [142, 152] | 11 | **none** |

The conjunction is exact on all three splits only for 4
threshold values ([155, 158]); the searched disjunction-then-conjunction
for 11 ([142, 152]).  Neither interval contains a
coarse-pool value, which is why section 3.3 is not optional.  **A bound bounds
the predicate shape it was written for, not the family** -- a caveat the
previous track's ceiling tables have as well, and this is the first case in
either track where the search lands outside the shape the bound assumed and does
better.

### 3.3 Widening the pool to every value 0..511

This is the case the previous track's section 8 sets up -- a choice among many
**constants at a fixed input**, where relaxation was measured 208x better than
chance and enumeration lost its certificate.  The difference here is that the
constant sits behind `pack`.

| method | space | evaluated | exhausted | conforming | + validation exact | held-out max error | held-out accuracy | s |
|---|---|---|---|---|---|---|---|---|
| `enumerate_fit`, capped | 393216 | 40,000 | no | -- | -- | -- | -- | rate 323 prog/s, projects to 1217 s |
| `enumerate_fit`, stop-at-first | 393216 | 34579 | no | -- | -- | -- | -- | 65.1 |
| prefix-reusing exhaustive walk | 393216 | 486291 node evals | yes | 910 | 858 | 0.0 | 1.000000 | 196.1 |
| gradient descent (init_noise 0.5, dead surrogate -- see 4.4) | 393216 | -- | no | 1/4 conforming on train | -- | -- | 0/4 exact on held-out | 350.6 s median |

**858 programs conform on training and are exact at every validation
position, and their held-out max error is 0.0.**
86 of the 910 training-conforming programs are
also exact on the held-out split.  The thresholds among the survivors are
142..230 (89 values, contiguous) -- the separating interval, recovered by search.

Two things to note about cost.  The prefix-reusing walk from the previous
track's `incremental.py` exhausts 393216 programs in
196.1 s (486291 node evaluations) where
`enumerate_fit`'s own loop projects to 1217 s -- the same
conclusion that track reached, on a space 12x larger.  And stop-at-first needed
34579 evaluations here rather than the 21 it needed at rung 3,
because a wide constant pool is dense in *near*-solutions and sparse in exact
ones.

### 3.4 Staged on the frozen foreground module

The previous track's rung 3.5 shape, applied to this target.  Stage 1 is the
rung-3 foreground module searched exhaustively over its 32000
programs and supervised at every position: 1952 conform, and
the one returned is accurate 1.000000 at every position of every
training episode and 1.000000 on held-out -- the diagnostic that
track showed a staged search stands or falls on.  Stage 2 freezes it, registers
it as an operator, calls it at both positions, and searches how the foreground
agreement combines with the collinearity predicate.

| arm | space | conforming | + validation exact | enum s | held-out accuracy | gradient |
|---|---|---|---|---|---|---|
| staged (frozen fg module + collinearity) | 98304 | 64 | 64 | 340.1 | 0.995536 | 0/4 (dead surrogate) |

Staging is not what makes this rung reachable -- the direct arm is already only
6,144 programs, because the *collinearity* predicate does the work the frozen
foreground module would have done.  What staging buys here is the same thing
section 3.2 buys: it separates a failure of the frozen stage from a failure of
the search, and its cost is that the composed space inherits the frozen module's
errors.

### 3.5 Applying it at every position, at four widths

The module is rebuilt at each width and the **selections are reused** -- it
never names an absolute address, so only the input type changes.  Applied by the
three-node `insert`/`pair`/`map` caller at every position of three fresh
episodes, with T = 142:

| R | observation bytes | positions | caller nodes | wrong slots | accuracy | `tcn.scaffold` agrees | s / 3 images |
|---|---|---|---|---|---|---|---|
| 8 | 192 | 56 | 3 | 1/168 | 0.994048 | yes | 0.37 |
| 16 | 768 | 240 | 3 | 0/720 | 1.000000 | yes | 3.69 |
| 24 | 1728 | 552 | 3 | 0/1656 | 1.000000 | yes | 17.74 |
| 32 | 3072 | 992 | 3 | 0/2976 | 1.000000 | yes | 58.81 |

**1 wrong slot in 5520**, three caller
nodes at every width, and the merged `tcn.scaffold.positional_scaffold` produces
bit-identical output to the copied caller at every width.  The one error is at
R=8 and is the same phenomenon as section 3.2: a fresh episode whose separating
interval does not contain the chosen threshold.

A module hardened at one width **cannot** be registered against another: its
input type names the observation, so `registry.resolve("map", ...)` raises
`map: operator signature mismatch`.  Rebuilding the scaffold and reusing the
selections is the supported route and is what makes the search space
resolution-independent in the first place.

### 3.6 The caller itself is now discovered, not written

The previous track's D4: `legal_candidates` resolved every candidate with empty
parameters, so `project`, `map`, `filter` and `join` were never proposed and
every scaffold in that track supplied its candidates by hand.
`operator_parameters` has since been merged, and this is the before/after over
the same port set (`observation`, `positions`, `empty`), with the "before"
reproduced faithfully by forcing empty parameter settings:

| | candidates proposed |
|---|---|
| with empty parameters (the old behaviour) | {"insert": 1, "intersection": 2, "member": 1, "pair": 4, "remove": 1, "tuple": 12, "union": 2} |
| now | {"insert": 1, "intersection": 2, "join": 36864, "member": 1, "pair": 4, "project": 192, "remove": 1, "tuple": 12, "union": 2} |

The operators that appear are `join`, `project`.  Searched rather than written,
the three caller nodes have [4, 2, 1] candidates -- a space of
**8 programs** -- and `enumerate_fit` returns
solved=True, exhausted=True, **unique=True**
in 8 evaluations and 0.10 s, supervised only by the
dense probe over the whole image.  The discovered wiring produces bit-identical
output to `tcn.scaffold.positional_scaffold`'s 3 nodes
(yes).

So the positional-reuse pattern is no longer hand-wired by construction.  That
closes D4, and it is the first measurement in either track where the caller is
*found*.

---

## 4. Gradient versus enumeration, and what a zero gradient is evidence of

### 4.1 The probe needed fixing before anything could be read off it

`SoftProgram.choices` has one entry per node, free or not, so zipping it against
only the free nodes silently misreports every entry.  The first version of this
measurement did that and reported `None` for all four choices.  Corrected, and
this is the table the rest of the section is about:

| scaffold | choice logit L1 gradient (`None` = unreachable in autograd) |
|---|---|
| rung 3 foreground (control, no `pack`) | {"cmp_r": 0.0006820035632699728, "cmp_g": 0.0014736297307536006, "cmp_b": 0.017924290150403976, "rg": 0.03237934783101082, "foreground": 0.055888641625642776} |
| rung 4 collinearity, as built | {"shifted": null, "thr": null, "m1": 0.008697787299752235, "same": 0.13985049724578857} |
| rung 4 collinearity, deterministic nodes unfrozen | {"shifted": null, "thr": 1.8977138334229368e-22, "m1": 0.006204310804605484, "same": 0.1962541788816452} |
| rung 4, wide pool, as built | {"shifted": null, "thr": null, "m1": 0.012470545247197151, "same": 0.14114171266555786} |
| rung 4, comparison nodes unfrozen **and** the `le` surrogate scaled | {"shifted": null, "thr": 0.005835290066897869, "m1": 0.008797964081168175, "same": 0.1539536416530609} |

### 4.2 The shipped `le` surrogate is exactly 0.0 at this rung's operating distance

`relaxed` computes `lt`/`le`/`gt`/`ge` as `torch.sigmoid(d/temperature)` with
`temperature = 1`.  In float32 both the value and its derivative are **exactly
0.0 at `|d| >= 89`**:

| gap \|d - T\| | sigmoid((T-d)/tau) | d/dT |
|---|---|---|
| 0 | 0.5 | -0.25 |
| 4 | 0.018 | -0.0177 |
| 16 | 1.13e-07 | -1.13e-07 |
| 32 | 1.27e-14 | -1.27e-14 |
| 60 | 8.76e-27 | -8.76e-27 |
| 88 | 6.05e-39 | -6.05e-39 |
| 89 | 0 | -0 |
| 128 | 0 | -0 |
| 4096 | 0 | -0 |
| 65025 | 0 | -0 |

And the operands here are products of two bytes.  Measured at the three `le`
nodes over the training records, with the threshold at the middle of the coarse
pool:

| `le` node | median gap | mean gap | min | max | fraction at or past underflow | shipped surrogate there | shipped derivative there |
|---|---|---|---|---|---|---|---|
| rg_le | 192 | 609 | 34 | 9024 | 0.969 | 0 | -0 |
| gb_le | 192 | 1053 | 5 | 21820 | 0.938 | 0 | -0 |
| rb_le | 192 | 991 | 53 | 13980 | 0.906 | 0 | -0 |

**The surrogate's value and its derivative are exactly 0.0 at the median
operating gap at all three nodes**, and essentially every record is past the
underflow point.  So the `0/4` and `1/4` gradient arms recorded in section 3 are
not evidence about whether this rung is learnable by relaxation; they are
evidence that the relaxation was invalid there.  They are reported as such and
section 4.4 is the arm that means something.

This is the same fault the perception ladder found in `eq`
(`exp(-(a-b)^2/tau)` reaching exactly 0.0 at `|a-b| >= 11`, which made that
track's address-relaxation benchmark a measurement of dead gradients rather than
misdirected ones), reproduced in the other comparison operator and at three
orders of magnitude larger operands.

### 4.3 Three mechanisms sever the gradient here, and they are not the same kind

| choice | severed by | kind | still severed with a live surrogate? |
|---|---|---|---|
| `shifted` (the neighbour offset) | `pack` declares `gradient="none"`, so `relaxed` routes it through `exact_tensor`, which detaches | **declared** | **yes** -- `grad = None` in every variant measured |
| `thr` (the threshold) | `SoftProgram` treats every node with `selected is not None` as frozen and detaches it, and `Builder`/`tcn.scaffold` set `selected = 0` on every *deterministic* node | **implementation** | no -- unfreezing makes it reachable at 1.9e-22 |
| `thr`, again | the `le` surrogate underflows | **numerical** | no -- scaling takes it to 0.00584 |

The distinction matters and it is the correction's real content.  A choice
behind a *declared* `gradient="none"` boundary is genuinely outside the relaxed
backend, and the offset is such a choice: `pack` is the only route from a
`role="byte"` pixel to arithmetic and it declares no gradient, so relaxation
cannot see which neighbour the module reads however the surrogate is scaled.
A choice that is dead because a surrogate underflowed, or because a
single-candidate node was detached, is not outside anything.

There is a fourth invalidity in the same neighbourhood, found while separating
the other three.  Unfreezing *every* deterministic node also relaxes `index`,
whose relaxation is `softmax(-(address - arange(n))^2 / tau)` at `tau = 1`.  Over
a 192-byte observation that puts only **0.564** of its weight on the true
address and **0.208** on each immediate neighbour, so the relaxed forward
pass reads a blur of about five bytes rather than a pixel.  That is why section
4.4 unfreezes only the three comparison nodes: it restores the gradient to `thr`
while leaving the addressing exact.

### 4.4 A scaled surrogate buys a gradient and can cost the answer

Before running anything, the landscape.  With every other choice pinned to a
program that conforms exactly, and the three comparison nodes unfrozen so
`relaxed` is actually used, the relaxed loss was evaluated at each candidate
threshold under seven temperatures.  The question is whether the minimum sits on
a threshold that is genuinely exact.

| comparison temperature | argmin threshold | is it exact? | loss spread |
|---|---|---|---|
| shipped tau=1 | 160 | **yes** | 3.28 |
| tau=8 | 192 | **yes** | 2.18 |
| tau=32 | 208 | **yes** | 1.07 |
| tau=128 | 496 | **no** | 0.76 |
| tau=1024 | 496 | **no** | 0.243 |
| operand (mean\|operand\|) | 496 | **no** | 0.241 |
| carrier (2^bits) | 400 | **no** | 5.96e-08 |

The thresholds exact on training in this grid are [144, 160, 176, 192, 208, 224].

**The shipped `tau = 1` puts its minimum on a correct threshold.**  Its gradient
is exactly zero, but its *loss* is right: `sigmoid` underflowing to 0.0 or 1.0
saturates to the correct hard answer, so the landscape is a plateau with cliffs
rather than a misleading slope.  **Scaling the temperature up restores the
gradient and moves the minimum onto a wrong threshold.**  At the carrier rule
`tau = 2^bits` the loss spread collapses to 5.96e-08 -- the
surrogate is so flat that every threshold looks the same -- and the minimum is at
400, which is not exact.

So the two failure modes are different and the fix for one is not the fix for the
other:

* **`eq` on bytes** (the perception ladder, and the coordinator's correction):
  the operands and the decision margin are the same scale, `exp(-(a-b)^2/tau)`
  underflows for the true *and* the false case alike, and the surrogate carries
  no information at all.  Scaling by the carrier restores it.
* **`le` on products of bytes** (this rung): the operands span 0..65,025 while
  the decision margin is about 80 wide, so the underflow saturates
  to the *correct* value and the loss stays informative while the gradient dies.
  Scaling by the carrier -- 2^32 here -- flattens a correct landscape into a wrong
  one.

What both cases actually want is a temperature matched to the **decision
margin**, not to the operand range.  Here that is the spacing of the candidate
constants, and `tau = 32` is the one policy measured that has *both* a correct
minimum and a live derivative (7.71e-05 at the median gap).

### 4.5 The corrected gradient arms

All arms leave `tcn/` untouched and replace `relaxed` at runtime in one process.

| arm | surrogate at the median gap | `thr` gradient | landscape minimum correct | conforming on train | exact on held-out |
|---|---|---|---|---|---|
| as shipped (section 3) | **0.0** | `None` | yes | 0/4 | 0/4 |
| deterministic nodes unfrozen, shipped surrogate | 0.0 | 1.9e-22 | yes | 0/4 | 0/4 |
| comparison nodes unfrozen, `tau = mean operand` | alive | 0.00584 | **no** | 0/4 | 0/4 |
| comparison nodes unfrozen, `tau = 2^bits` (carrier) | alive | 7.26e-09 | **no** | 0/4 | 0/4 |
| comparison nodes unfrozen, `tau = 32` (decision margin) | alive | 0.0319 | yes | 1/4 | 0/4 |
| `tau = mean operand`, **offset pinned** to the right neighbour | alive | 0.00882 | no | 0/4 | 0/4 |
| *every* deterministic node unfrozen, `tau = mean operand` | alive | 0.00892 | -- | 0/4 | 0/4 |
| *every* deterministic node unfrozen, `tau = 2^bits` | alive | 7.26e-09 | -- | 0/4 | 0/4 |

**Reading the table.**  Every arm run with a surrogate that is 0.0 at the
operating distance is 0/4, and so is every arm run with a surrogate
scaled to the *operand* range -- 0/4 for `mean|operand|` and
0/4 for the carrier rule -- which is what section 4.4 predicts,
because those two policies move the loss minimum off the correct threshold.  The
one policy whose minimum stays correct while its derivative comes alive,
`tau = 32`, is **1/4 conforming on training** with held-out accuracy
0.995536 -- which is the coarse pool's own ceiling, the same number
enumeration returns from that pool (section 3.1).  So with a temperature matched
to the decision margin the relaxed path does reach the best program the coarse
space contains, in 1 seed of 4, where enumeration reaches it in
8.0 s with an exhaustion certificate and a conforming count.

Pinning the offset -- removing the one choice behind `pack`'s declared boundary
-- does not rescue the operand policy (0/4), so the declared boundary
is not the only thing in the way; the temperature is.  That ordering is the
correction's content applied to this rung: the recorded failures were invalid
relaxations, and the valid measurement is that this landscape is hard for
relaxation and trivial for enumeration, not that relaxation is excluded from it.

Enumeration, on the same space and the same records, exhausts 6144
programs in 8.0 s and 393216 in 196.1 s
with a prefix-reusing walk, returning a program with held-out max error
0.0 and a conforming count.

### 4.6 What this says about the method boundary

The previous track drew the boundary as "constants at a fixed input: relaxation;
behind `gradient="none"`: enumeration".  Two of the three failures here were not
on that boundary at all -- they were an implementation detail and a numerical
underflow -- and only the offset is genuinely behind a declared boundary.  The
honest statement after this track is:

* a choice behind a **declared** `gradient="none"` operator -- here the offset,
  behind `pack` -- is outside the relaxed backend, and no surrogate fixes it;
* a choice at a **surrogate comparison** is only as good as that surrogate's
  dynamic range at the *operating distance*, which must be measured and reported,
  not assumed -- and the temperature that restores the dynamic range must be
  matched to the **decision margin**, because a temperature matched to the
  operand range flattens the landscape instead (section 4.4);
* a choice at a node the scaffold made deterministic is severed by
  `SoftProgram`, which is a bug (E1) and not a property of anything;
* **space size predicts nothing.**  The 393,216-program space and the
  6,144-program space behave identically here.

## 5. Is a different generator's segmentation probe determined?

`geometry` also probes `depth` and `normals`; `world_2d` and `world_3d` probe
`agent_0/visible_ids`.  Both worlds call the same renderer, and their object
list is built differently -- the agent body and its arm carry colours written
into the generator, and only the free bodies are re-drawn per episode -- so the
probe could in principle be partly determined.

| generator | permuted list: rgb identical | ids identical | `visible_ids` held-out acc | majority baseline | advantage | fg key recurrence | same-object held-out acc, AND-of-three at T=144 | same-object held-out acc, searched shape at T=144 |
|---|---|---|---|---|---|---|---|---|
| world_3d | 4/4 | 1/4 | 0.7743 | 0.7743 | +0.0000 | 0.000 | 1.000000 | 1.000000 |
| world_2d | 4/4 | 2/4 | 0.7318 | 0.7318 | +0.0000 | 0.000 | 1.000000 | 1.000000 |

It is not.  The permutation certificate holds there too -- the two episodes
where reversing the list leaves the labels unchanged are ones where too few
distinct objects are in view for the reversal to move any visible index -- `visible_ids` is at
advantage 0.0000 with foreground key recurrence 0.000, and the one fixed-colour
object -- `arm_agent_0`, index 0 of ["arm_agent_0", "body_0", "body_1", "body_2", "body_3", "body_4"] -- **never appears in the
agent's own view**: the held-out label histogram is {"-1": 892, "1": 5, "2": 20, "3": 78, "4": 65, "5": 92}, with no
class 0 at all.  So the only object whose identity the generator makes
determinable is the one the camera is mounted on.

**The same-object relation transfers.**  The collinearity predicate searched on
`geometry` -- both the conjunction shape and the `(rg or gb) and rb` shape the
wide search actually returned, at the same threshold -- applied unchanged to
`world_3d` and `world_2d` pixels, is exact on held-out episodes of both.  Nothing in it is
`geometry`-specific; it is a property of the shared renderer.

---

## 6. The ladder, end to end

| rung | target | context | bound before search | discrete result | gradient | verdict |
|---|---|---|---|---|---|---|
| 3 | per-pixel foreground | one pixel | 1.0000 vs 0.4492 | 32,000 exhausted, 2,464 conform, not unique | 6/6 exact | works (previous track) |
| 3.5 | `fg(i) != fg(i+k)` | two pixels | -- | 48 exhausted, **unique**, held-out 0.0 | 1/4 | works (previous track) |
| **4a** | `object_ids`, any context | pixel .. 3x3 .. global | **advantage 0.0000 everywhere; structurally impossible** | not searched, by design | -- | **certified unreachable** |
| **4b** | `object_ids` restricted to foreground | pixel .. 3x3 | advantage <= 0 everywhere | not searched | -- | **certified unreachable** |
| **4c** | `same(i, i+1)` restricted to two foreground pixels, equality only | two pixels, equality | advantage **0.0000** at every equality context | not searched | -- | **certified unreachable** |
| **4d** | `same(i, i+k)`, collinearity | two pixels + `pack` | held-out **1.0000**, margin [140, 231) | 6144 exhausted, 10 conform, held-out 0.995536 | 0/4 | **works** |
| **4e** | the same, threshold pool 0..511 | two pixels + `pack` | as above | 393216 exhausted, 910 conform, 858 validation-exact, held-out max error **0.0** | 1/4 | **highest rung reached** |
| 5 | `depth` threshold | pixel .. 3x3 | advantage 0.0000 everywhere | -- | -- | unreachable (previous track, reproduced) |

---

## 7. The precise verdict

**What context makes object identity learnable:** two pixels, one relative
offset apart, compared for *collinearity* rather than equality -- which needs
`pack` to lift them out of `role="byte"` and costs one searched integer
threshold.  At that context the same-object relation is exact: held-out max
error 0.0, and 858 of 393216 programs conform on
training and are exact at every validation position.  One pixel is not enough,
and neither is a 3x3 window, a global aggregate, or a position, when the only
predicate available on bytes is equality: every one of those is at advantage
exactly 0.0000 on foreground-to-foreground object boundaries.

**What no context makes learnable:** the labels themselves.  `object_ids` is the
index of an object in the generator's list, the image is invariant to permuting
that list, and the labels are invariant to re-drawing the colours the image is
made of.  That is a certificate, not a budget failure, and it holds for
`world_2d`/`world_3d`'s `agent_0/visible_ids` for the same reason.  What the
observation determines is the *partition*; what it cannot determine is the
*naming*.  A generator that wanted identity to be learnable would have to make
appearance carry it -- fixed per-object colours, or a persistent texture -- and
`world_3d` almost does, for exactly one object, which its own camera cannot see.

---

## 8. Threats to validity

### 8.0 Hand-initialization, stated, ablated and certified

`AGENTS.md` requires every hand-initialization to be named, ablated and, where
possible, certified.  Everything supplied by hand in this track, and what
happens without it:

| supplied | where | ablation / certificate |
|---|---|---|
| the module *shape*: read your own pixel and one at a searched offset, form three cross products, combine with two searched truth tables | `collinear_scaffold` docstring | not ablated; this is the scaffold, and section 8's first bullet is the caveat.  Its free choices are searched from nothing: enumeration has no initialization at all. |
| the *threshold pool* | `THRESHOLDS` in `rung4_segment.py` | **ablated directly** -- section 3.3 replaces the 8-value pool with all 512 values and the answer changes, from held-out max error 1.0 to 0.0 |
| the *offset pool* {+1 pixel, +2 pixels, +1 row} | `offsets()` | the search picks `+1 pixel` from three, exhaustively; not pinned in any reported arm except the one row of section 4.5 that says it is |
| the *frozen stage-1 module* | section 3.4 | the direct arm freezes nothing and reaches the same target, so staging is not load-bearing here |
| **no** initialization of any choice logit | -- | every discrete result is from exhaustive enumeration, which starts from nothing and reports whether the space was exhausted.  That is the certificate `AGENTS.md` asks for: the answer provably lies in the declared space |

The gradient arms are the only ones with an initialization at all
(`init_noise = 0.5`, because the shipped `fit` cannot vary a seed), and none of
them is the source of a positive claim.

* **Scaffold authorship.**  The collinearity module was written knowing the
  renderer multiplies a base colour by a scalar shade.  The *bound* in section
  2.4 was computed before the search and does not depend on the scaffold, but
  the shape of the module -- three cross products and two combinators -- is
  supplied, and section 3.2 shows how much the *candidate pool* alone can cost.
  A discovered scaffold would be harder.  Section 3.6 is the one place where the
  wiring is discovered rather than written, and only for the caller.
* **A bound bounds a predicate shape, not a family.**  Section 3.2 measures this
  directly: the ceiling was derived for `max(cross) <= T` and the search
  returned `(rg or gb) and rb`, whose exact-threshold interval is
  11 values wide against the conjunction's
  4.  Every "advantage 0.0000" row in section 2 is a statement
  about the *context*, which is shape-free, but section 2.4's margin is a
  statement about one shape.
* **The bound is a generalisation bound, not an information bound.**  A lookup
  table fitted on training keys upper-bounds every program required to fit those
  keys, but on *unseen* keys any function is admissible, so a low held-out
  number is evidence that no *learnable* function transfers, not a proof that
  none exists.  This is why section 2.1's structural certificates are stated
  first and carry the negative result; the table corroborates them.
* **The chromaticity contexts in section 2.2 are not expressible** on
  `role="byte"` pixels.  They are reported as a bound on a family the algebra
  reaches only through `pack`, and section 2.4 re-measures that family exactly.
* **Derived targets.**  Every rung supervises on an equivalence class of a probe
  (`object_ids >= 0`, `object_ids[i] == object_ids[j]`) rather than the probe
  itself.  Section 2.1 is the measurement that motivates it.
* **A shared, loaded host,** with up to four of this track's runs overlapping
  deliberately and other agents on the same 20 cores throughout (load average
  28-81).  Wall clocks are upper bounds, are not comparable across hours, and the
  section 4.5 gradient arms in particular range from 350 s to 1,160 s per seed
  for identical work.  Programs, node evaluations and choice-logit gradients are
  the load-independent measures and are reported beside every time; no
  qualitative claim here rests on a wall clock.
* **Supervision budgets are small**: 448 records at R=8.
* **`local_fit`** is a copy of `tcn.synthesis.fit` with an `init_noise`
  argument, because the shipped one cannot vary a seed.  0.5 is arbitrary.

---

## 9. Proposed changes to `tcn/`, as diffs.  None applied.

Two agents are changing core concurrently, so nothing here was edited.  Both
items are new; the previous track's D1 (the `IndexError` catch) is already
merged and this track depended on it.

### E1 (bug).  `SoftProgram` detaches every single-candidate node, which severs gradients through deterministic parts of any hand-written scaffold

`SoftProgram.__init__` builds `self.frozen = {n.name: n.selected for n in
program.nodes if n.selected is not None}`, and `forward` evaluates those with
`exact_tensor(...).detach()`.  But `Node.selected` is set to `0` by
`tcn.scaffold` and by every hand-written builder whenever a node has exactly one
candidate -- that is *determinism*, not crystallisation.  Measured here: the
`thr` node feeds `le`, a `surrogate` operator, and has `grad = None`; restoring
`selected = None` on the single-candidate nodes -- an identical search space,
6144 programs either way -- makes it reachable, at |grad| = 1.9e-22.

```python
-        self.frozen={n.name:n.selected for n in program.nodes if n.selected is not None}
+        # A node with one candidate is deterministic, not crystallized: relaxing
+        # it is a no-op on the mixture and keeps the autograd path alive for the
+        # choices downstream of it.  Only genuinely frozen nodes should detach.
+        self.frozen={n.name:n.selected for n in program.nodes
+                     if n.selected is not None and len(n.candidates)>1}
```

The cost is that a deterministic node is relaxed rather than executed exactly,
which for a `gradient="none"` operator is the same call either way.

### E2.  The `le`/`lt` surrogate has no dynamic range on integer operands

This is `eq`'s underflow, in the other comparison operator.  `relaxed` computes
`lt`/`le`/`gt`/`ge` as `torch.sigmoid(d/temperature)` with `temperature = 1`; in
float32 both the value and its derivative are **exactly 0.0 at
`|d| >= 89`**, and this rung's operating gaps have median
192 with 0.906 of records at or past the underflow
point (section 4.2).  Any comparison on operands wider than about two decimal
digits -- a product of two bytes, a squared distance, a pixel count -- is
invisible to the optimiser, and every gradient arm over such a comparison is a
measurement of a dead relaxation rather than of learnability.

**But the obvious scaling is the wrong one here, and section 4.4 measures that.**
`tau = 2^bits` is 2^32 for these operands; it restores the derivative and
flattens the loss to a spread of 5.96e-08, moving the minimum onto
a threshold that is not exact.  `tau = mean|operand|` does the same.  Only a
temperature matched to the **decision margin** -- the spacing of the candidate
constants at that node, about 80 here -- keeps the minimum correct
*and* the derivative alive.  So the fix is not "divide by the carrier"; it is
"divide by something the caller can relate to the decision", and the honest
version exposes it:

```python
-def relaxed(registry,op,xs,temperature=1.):
+def relaxed(registry,op,xs,temperature=1.,scales=None):
...
     if n in COMPARE:
-        if n=="eq": return torch.exp(-((a-b)**2).sum(-1,keepdim=True)/temperature)
         d=b-a if n in {"lt","le"} else a-b
-        return torch.sigmoid(d/temperature)
+        # A fixed temperature makes a comparison blind past |a-b| ~ 11 for `eq`
+        # and |d| ~ 89 for the orderings, in float32; a temperature scaled to the
+        # OPERAND range flattens the landscape instead. The informative scale is
+        # the decision margin, which only the caller knows, so it is passed in
+        # and defaults to the carrier width rather than to 1.
+        tau=temperature*float((scales or {}).get(op.name) or 2**op.inputs[0].bits)
+        if n=="eq": return torch.exp(-((a-b)**2).sum(-1,keepdim=True)/tau)
+        return torch.sigmoid(d/tau)
```

Whatever the scaling, the requirement that does not depend on choosing it is:
**a gradient arm must report the surrogate's value at its operating distance.**
A `0/n` recorded next to a surrogate of 0.0 says nothing, and a `0/n` recorded
next to a loss spread of 5.96e-08 says nothing either.

### E3.  `index`'s address relaxation is not sharp enough to read a pixel

`relaxed` computes `index` as `softmax(-(address - arange(n))^2 / temperature)`
at `temperature = 1`.  Over a 192-byte observation that puts only
**0.564** of its weight on the true address and 0.208 on each
immediate neighbour, so a relaxed read returns a blur of about five bytes.  A
scaffold whose deterministic nodes are relaxed (which E1 would make the default)
therefore reads a blurred pixel, which is a second invalid relaxation in the
same place.  Sharpening the temperature, or using a straight-through estimator
on the argmax address as `relaxed` already does for `idiv`, would fix it; this
track measured the sharpness rather than choosing between them.

### The previous track's D2 is merged; D3 and D6 are not

`SearchResult` now carries `conforming`, and `enumerate_fit` takes `rank` --
that is D2, and this track would have needed it.  D3 (a validation split as the
tie-break) and D6 (walk the choice tree instead of re-executing per candidate)
are still open, and this track needed both: it filters by validation itself
(section 3.1 and 3.3, where 910 conforming becomes
858), and it used `research/discrete-perception/incremental.py` for the
393216-program sweep, which `enumerate_fit`'s own loop projects at
1217 s against 196.1 s.
