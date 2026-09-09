# Hierarchical screen parsing: the generator, the bounds, and rung one

Research track `gui-hierarchy`.  Everything below was produced in this workspace
on 2026-09-09 with
`/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python`
(Python 3.13.15, torch 2.14.0+cpu, Pillow 12.3.0).  **Nothing under `tcn/` is
modified and no existing generator is modified.**  `generators/gui/` is a new
peer under the same lifecycle and record contract as every other generator; the
trainer does not know it exists.  Every table is rendered from `out/*.json` by
`report.py render`, so no number here is transcribed by hand.

**Wall clock caveat.**  This host was shared with three other agents throughout
(load average 20 on 20 cores).  Every timing is an upper bound and noisy at the
tens-of-percent level.  Programs evaluated and node evaluations are the
load-independent cost measures and are reported alongside every time.

---

## 0. Verdict

**The generator exists and replays; the hierarchy is recoverable, with two
measured exceptions that are certificates rather than failures; and rung one
learns, exhaustively, with a uniqueness statement and a registered module.**

Six results, in the order they should change what the project does.

1. **`generators/gui` renders a widget tree to raw pixels and emits the tree as
   `set[(id, parent, kind, x, y, w, h)]` at a declared capacity.**  The probe's
   type is the same at 2 widgets and at 24, at nesting 1 and at nesting 6 --
   cardinality moves, width does not.  That is `generators/logic`'s `gates`
   channel applied to a second domain, and it is the shape a depth-independent
   parser output has to have.  **13 tests** in
   `tests/test_gui_generator.py`; the full suite passes (164 at the time of
   writing, which includes tests other agents added to this working tree
   concurrently).

2. **The recoverability bounds were computed before any search, and two of them
   came back negative.**  A widget boundary is exactly determined by a two-pixel
   neighbourhood in the flat configuration (oracle ceiling 1.0000 against a
   majority of 0.8261) and is **not** determined once borders are drawn (ceiling
   0.8483) or the palette is too small to give every widget its own colour
   (0.8963).  Those are ceilings on *every* function of that context, so no
   program over it can exist there, and the search confirms it: 0 conforming
   programs at exactly those settings.

3. **A tighter bound, specific to this algebra, is the one that is actually a
   certificate.**  The only legal predicate on a `role="byte"` value is `eq`, so
   any two-pixel rung-1 program is a Boolean function of exactly three equality
   bits.  Over *that* context the flat configuration's ceiling is 1.0000 and its
   transfer is also **1.0000** -- the equality pattern is the episode-independent
   representation, which is why the rung transfers where a colour lookup does not.
   With borders on, the same bound is **exactly the majority baseline, advantage
   0.0000**: no program over two pixels can beat a constant there, which is a
   sharper statement than the raw-neighbourhood ceiling of 0.8483.  With labels
   on the raw ceiling is 1.0000 while the pattern ceiling is 0.9441, so the target
   is a function of the neighbourhood that this algebra cannot write -- and the
   search duly returns 0 conforming programs at a setting the coarse bound calls
   reachable.

4. **Rung one learns, and conformance alone would have shipped the wrong
   program.**  The 1,280-program space over the true ownership boundary is
   exhausted and several programs conform on the training records, denoting more
   than one distinct Boolean function; requiring exactness at *every* position of
   a validation split drawn from different episodes reduces them to a single
   function that is exact on held-out episodes.  The program `enumerate_fit`
   returns -- the lexicographically first conforming one -- is not that program.
   That is the discrete-perception track's tie-break fault reproduced
   independently on a new generator with a different target, and the same fix
   works.  Exact counts are in the section 4 table.

5. **Rung one is packaged as an earned abstraction, not a scaffold.**  The
   selected program is hardened, registered with `Registry.register_module`, and
   then *chosen* by a second program from two same-shaped candidates -- the edge
   module and a distractor reading the row below.  Three caller nodes apply it at
   every position of a fresh screen with max error 0.0.

6. **The gradient arm fails for two independent and separately measured
   reasons, neither of them "the search is hard".**  `SoftProgram` treats any
   `selected` node as frozen and evaluates it through
   `exact_tensor(...).detach()`, so a single-candidate node between a choice and
   the loss makes that choice's logit `grad is None` -- measured directly, and
   independent of the data.  With that removed the address logit is reachable
   but starved by six orders of magnitude, because `eq`'s surrogate slope is
   **exactly 0.0 at every colour separation this generator can render** at the
   shipped `tau = 1`.  Both numbers are in section 7, and neither is an inference.

---

## 1. The generator and its contract

`generators/gui/` is an ordinary peer: `initialize` / `advance` / `observe`,
seeded from the episode `Address`, no host clock, all content synthetic,
complete state in the snapshot.

```text
observation   pixels     the raster, as image_value: (height, width, channels, bytes)

latent        focus      the focused widget id

probes        hierarchy  set[(id, parent, kind, x, y, w, h)]   capacity C, int[8] fields
              owner      per-pixel id of the deepest widget covering that pixel
              glyphs     set[(id, slot, code, x, y, w, h)]     only when `labels` is set

actions       wait; focus{id}; press{id}
reward        goal       1.0 when the objective's named widget is pressed
```

**The hierarchy probe's type does not depend on the screen.**  `ARCHITECTURE.md`
section 1 says "relations are sets of tuples", and this is that.  `id` is
load-bearing twice: a set is duplicate-free, so without it two widgets of the
same kind and geometry would collapse, and `parent` would have nothing to name.
`hierarchy_value` / `widgets_from_set` are inverse, verified on a list
containing two rows identical apart from their id.  Capacity is a property of
the declared type, not of the episode: a request for more widgets than
`hierarchy_capacity` is refused rather than truncated, and a screen larger than
255 is refused because the coordinate fields are `int[8]`.

**Probes are supervision, never model inputs.**  Every program input in this
track is filled only from `StepRecord.actor_view().observations`, which contains
exactly `pixels`; `probes` and `latent_states` build targets and nothing else,
and `common.episode` asserts the observation set on every draw.

**Why the rendering is flat and sharp, and why that is a contract.**
`role="byte"` excludes a pixel from `Type.numeric`, so the entire legal algebra
over a raw pixel is `eq(byte, byte) -> bool`, `pack` (gradient `none`), `index`,
and the structural operators; there is no conversion out of byte at all, and
every arithmetic and ordering operator is signature-illegal.  Any perception
program over these pixels is therefore Boolean and relational.  The generator is
rendered so that `eq` is sufficient: filled axis-aligned rectangles at integer
coordinates with one flat colour per widget, and glyphs thresholded to a hard
mask before painting rather than antialiased, because an antialiased edge would
put a blend of two palette entries on the screen and there is no operator that
can read a blend.  **No numeric view of the pixels is emitted.**  Where a rung
would need arithmetic on bytes, this report says so and stops.

**Replay.**  `Host.replay`, `snapshot`/`restore` and `save`/`load` all reproduce
the digest, including after a `focus` and a `press`; verified in
`tests/test_gui_generator.py`.

---

## 2. The difficulty dial, and the F-bench trap it walks into

`generators/logic`'s `depth` looked like a difficulty axis and was not, which
invalidated a 97%-at-every-depth result.  So every dial here is measured on four
axes: the structure it achieves, the appearance it produces, the recoverability
ceiling, and the number of programs that conform.

<!-- TABLE: dial -->

**Read the first three rows before anything else.**  Requesting 12 widgets at
`min_size 6` and `resolution 16` achieves **3.0**, exactly what requesting 6
achieves, and `nesting 4` achieves the same tree depth as `nesting 1`.  At that
resolution the layout saturates: a container that cannot be split into parts of
at least `min_size` is a leaf whatever the request says.  **The request is not
the achievement**, which is the F-bench lesson in miniature, hit inside this
track rather than found afterwards.  The generator reports the achieved count
and depth in the probe itself, and every number in this report is stated against
the achieved screen.  The dial becomes real once the resolution or `min_size`
allows it: the rung-1 screen used below (`widgets 12, nesting 4, min_size 4`)
achieves a mean of 8.5 widgets in 768 bytes.

Reading the rest:

* **`palette` and `borders` move the ceiling**, and where the ceiling falls below
  1.0 the search returns **0** conforming programs.  Bound and search agree.
* **`labels` leaves the raw-neighbourhood ceiling at 1.0 and still returns 0
  conforming programs.**  That is not a contradiction; see section 3.
* **`widgets`, `nesting` and `resolution` move the majority baseline and the
  conforming count** without moving the ceiling.  Those are search-difficulty
  dials: fewer widgets means weaker supervision and more programs survive it
  (12 conforming at 2 widgets, 2 at the baseline), and `colour_mode=kind`
  reduces the distinct colours on screen and lets 16 programs through.
* **The rung-1 space is 1,280 at every resolution**, from 768 to 6,912 bytes,
  because the module computes its addresses from its own position argument
  rather than choosing them from a menu over the image.

---

## 3. Recoverability, computed before any search

The method is the object-identity track's.  A lookup table keyed by exactly the
context in question is an upper bound on *every* function of that context, so if
it does no better than the majority label, no program over that context can
exist.  Two numbers per rung: the **oracle** ceiling, fitted on the evaluation
set itself and therefore a hard ceiling, and the **transfer** number, fitted on
training episodes, which is what a learned table would achieve.

<!-- TABLE: bounds -->

### What each block says

**R1.** In the flat configuration a widget boundary is *exactly* determined by
the two-pixel neighbourhood, on both axes, at every resolution and widget count
tested.  With `borders` on the ceiling drops to 0.8483 against a majority of
0.8261: a certificate that **no** program over two pixels can be exact there,
because a widget's own border-to-fill transition and a real ownership boundary
present the same ordered colour pair.  With a palette too small to give every
widget its own colour the ceiling is 0.8963.  Those are the two ways this
generator destroys the information its probe claims to describe, and both are
explicit configuration.

**The transfer column is the geometry trap in miniature, and it is not a
failure of the rung.**  A raw colour-pair lookup scores about 0.83 held-out with
40% of its keys never seen, because the widget-to-colour assignment is redrawn
every episode.  The program that rung one actually learns is not a table over
colour pairs; it is *colour inequality*, which is episode-independent.  The
oracle says a function exists; the transfer number says a table is the wrong
function class.  Where colour is fixed (`colour_mode=kind`) the table transfers
perfectly, which is the control.

**R1b is the bound that matters for this algebra, and it is sharper in both
directions.**  Since `eq` is the only predicate on a byte, any two-pixel program
is a Boolean function of three equality bits, and the ceiling over *that* context
is the ceiling of the whole expressible family.

* On the flat configuration the pattern ceiling is 1.0000 **and its transfer is
  1.0000**, against 0.83 for the raw-byte table.  That is the cleanest available
  statement of why the rung generalises: the answer is a function of the
  equality pattern, which is episode-independent, and not of the colours, which
  are redrawn.  The algebra can only express the former.
* With `borders` on the pattern ceiling equals the majority baseline **exactly**
  -- advantage 0.0000 on both axes -- so no two-pixel program can beat a
  constant.  The raw-neighbourhood bound only said "not exact"; this one says
  "not better than nothing".
* With `labels` on the two bounds separate: the raw neighbourhood determines the
  answer (1.0000) and the equality pattern does not (0.9441), because a glyph
  pixel against its own widget's fill and a real boundary between two widgets
  both present "all three channels differ".  A search returning zero there is
  reporting a property of the algebra, not a budget failure.

**R2.** A glyph **is** determined by its bounding box -- oracle 0.9625 against a
majority of 0.0625 -- but only through an ink reduction.  The raw-byte lookup
transfers at 0.0875 with 85% of its keys unseen, because the box carries the
widget's fill colour behind the glyph and that colour is redrawn every episode.
Binarising the box to ink/not-ink takes transfer to 0.9250 with 1.3% unseen.
The reduction is `eq(pixel, ink)`, which is expressible, so the glyph rung is
reachable and section 8 says what it needs.

**R3.** A widget's kind is **not** determined by its colour in the default
`random` colour mode (ceiling 0.4620 against a majority of 0.2982) -- the
deliberate analogue of `geometry`'s object identity, and the trap this generator
was built to be able to fall into on purpose.  It is largely determined by its
size (ceiling 0.8480, transfer 0.5322), and exactly determined by colour in
`colour_mode=kind` (1.0000 / 1.0000).  The "fill + size" row is the honest
warning: its oracle of 0.9942 in random mode is memorisation, since that key is
nearly unique per widget, and its transfer collapses to 0.3509.

**R4.** The parent relation is **exactly determined by the geometry alone**:
taking the smallest rectangle strictly containing a widget recovers its true
parent for 32/32, 155/155 and 368/368 widgets at three complexities, with zero
ties.  That is the cleanest statement in this report about what the ladder can
reach once rung one and widget extents exist.

---

## 4. Rung one, measured

**Target.**  `owner(i) != owner(i + 3)` in bytes -- the true ownership boundary
between a pixel and its right neighbour, taken from the `owner` probe.  Not a
chromatic edge: a colour change inside a widget is not a boundary, and the
distinction is exactly what makes `borders` and `labels` break the rung.

**Screen.**  `resolution 16, widgets 12, nesting 4, min_size 4, palette 32`;
achieved mean 8.5 widgets, 768 observation bytes, 225 supervised positions per
screen, 37.0% positive.

**Staging.**  One per-position module `(position, image bytes) -> (position,
edge)` searched discretely, then applied at every position by the three-node
`insert`/`pair`/`map` caller.  Addresses inside the module are computed from its
own position argument with `add`; the *neighbour offset* is a searched choice
among a declared pool.

<!-- TABLE: rung1 -->

<!-- TABLE: selections -->

### Reading it

**Conformance is not the acceptance criterion; uniqueness is.**  Arm A exhausts
its 1,280-program space and finds 4 conforming programs denoting 2 distinct
Boolean functions -- so `unique` is `False`, and the lexicographically first
conforming program, which is exactly what `enumerate_fit` returns, is **wrong on
held-out episodes**.  Requiring exactness at every position of a validation split
drawn from different episodes leaves selections denoting a single function, and
that function is exact on held-out episodes.  This is the discrete-perception
track's finding reproduced on a different generator with a different target, and
it should now be treated as the default expectation rather than a surprise.

**Non-uniqueness here is a statement about the scaffold, not only the probe.**
Counting distinct *functions* rather than distinct *selections* separates the
two: where two conforming selections denote one function, the scaffold is
redundant (`rg` and `edge` can write the same composite two ways); where they
denote two functions, the supervision genuinely underdetermines the target and
the validation filter is what removes the wrong one.  Both cases occur in the
table and are reported separately.  Arm A's 4 conforming selections denote 2
functions and the filter removes one of them; arm C's 16 denote 8 and the filter
again leaves exactly 1.

**Neither hand-supplied structure turned out to be load-bearing, which is the
result the ablations exist to produce.**  Ablating H2 -- replacing the five
hand-chosen offsets with all 51 byte offsets up to one row -- takes the space
from 1,280 to 13,056 and returns the *same* conforming count, the same number of
functions, the same filtered answer and the same zero held-out error.  Ablating
H3 -- letting each channel comparison take a constant byte instead of its
neighbour, so the operand binding is searched -- takes the space to 81,920 and
still filters to one function with zero held-out error.  Rung one is therefore a
capability of the staged discrete search, not a consequence of the scaffold's
generosity.  The random-draw control is the reason to care: at the wider two arms
a uniform random program from the same space conforms 0/400 times, so finding a
conforming program is not easy by accident.

---

## 5. Hand-initialisation, declared and ablated

`examples/joint.py` shipped a result compromised by initialising a decoder to
the exact solution.  Everything this scaffold supplies is listed here, and every
item that can be ablated is, with both numbers side by side.

| # | what is supplied | is it an answer? | ablation | where |
|---|---|---|---|---|
| H1 | one `core` region, fixed node depths, fixed predecessor pool | no -- this is the coarse structure ARCHITECTURE section 4 exists to supply | not ablated: without a graph there is no scaffold | — |
| H2 | the offset pool `shifted` chooses from (5 plausible spatial steps) | no -- the correct offset is one of five, unmarked | arm B offers **every** byte offset from 1 to 3W+3 (51 candidates) | table in section 4 |
| H3 | the channel correspondence: `cmp_c` compares channel c here to channel c there | no operator or constant is fixed, but the operand binding is | arm C offers each comparison a constant byte from a pool instead, so the binding is searched | table in section 4 |
| H4 | choice-logit initialisation noise for the gradient arm | no -- `SoftProgram` zero-initialises every logit, so a seed alone changes nothing; noise breaks symmetry without supplying an answer | `init_noise = 0`, reported as the single outcome it is | table in section 4 |
| H5 | single-candidate nodes written `selected=0` | no, but it silently severs the gradient | `relax_single=True` leaves them relaxed at no change to the discrete space | section 7 |
| H6 | candidate declaration order | **it would be, if left alone** | removed rather than ablated: see below | — |

**H6 deserves its own paragraph, because the first run of this experiment had
it wrong.**  `SoftProgram` zero-initialises every choice logit, so `argmax` at
initialisation is candidate 0, and `enumerate_fit`'s tie-break prefers candidate
0 as well.  The offset pool was first written `(3, 6, 9, 3W, 3W+3)` -- the
correct answer at index 0 -- which means a zero-init gradient run that never
moves that logit would have been scored a success for reading its own
declaration order, exactly the shape of the `F-init` fault.  The pool is now
`(3W, 6, 3, 9, 3W+3)` and the free arm's operand pool likewise places the
correct binding at index 2.  This is the removal of a hand-initialisation, so
there is no "with/without" pair to report; the discarded first run is noted here
because the diagnostic -- a zero-init run succeeding where noisy runs fail -- is
what exposed it.

Nothing is initialised to, or restricted to, the correct answer.  Both
combinator nodes offer all 16 two-input truth tables in every arm.

---

## 6. Stage B and Stage C: the module is earned, then consumed

Rung one is delivered as a **registered module**, not a scaffold.  A parser the
system learns from pixels against probes and then crystallises is the earned
abstraction of ARCHITECTURE section 4 -- registered with
`Registry.register_module`, it becomes one typed candidate operator, its
internals charged to description size and its execution charged per call.  This
is the opposite of the hand-written detector `AGENTS.md` forbids as a model
input: the agent never sees a parse, it sees pixels, and the parser is a program
the search found.

Stage B applies it at every position of fresh screens with three caller nodes.
Stage C offers a single `map` node two same-shaped candidates -- the edge module
and a distractor that is the identical program reading the row below instead of
the pixel to the right -- and lets the search choose, scored against the whole
dense edge map of held-out screens.

---

## 7. The gradient arm, and the two things that stop it

The brief said not to assume either backend wins, and not to report a failed
address search as confirming a known wall without checking whether the surrogate
is alive at the mixture.  Both checks were run.

<!-- TABLE: surrogate -->

**Finding one: the graph is severed, and it has nothing to do with the data.**
`SoftProgram.forward` treats any node with `selected is not None` as frozen and
evaluates it through `exact_tensor(...).detach()`.  Every single-candidate node
in a hand-wired scaffold is written `selected=0` by the `Builder` this track
copied from `research/discrete-perception`, and `add`, `index` and `eq` between
the `shifted` choice and the loss are all single-candidate.  The result is
`grad is None` on the address logit at **every** palette and both pool sizes --
not a small gradient, no gradient.  Writing those nodes `selected=None` costs
nothing discretely (a product over 1-element choices) and restores the edge.
This independently reproduces D1 from `research/depth-generalization`.

**Finding two: with the graph intact, the address logit is starved, and the
number that says so is the surrogate's slope.**  `eq`'s relaxation is
`exp(-(a-b)^2/tau)`; what a choice upstream of it receives is the *slope*,
`|2d/tau| * exp(-d^2/tau)`, which is exactly 0 at `d = 0` (the peak) as well as
once the exponential underflows.  At the shipped `tau = 1` the slope at this
generator's colour separations is 0.00e+00 at every `palette_levels` from 2 to
16 and 7.34e-21 at 32 -- **there is no configuration of this generator in which
a comparison between two different colours passes gradient on shipped code.**
Measured end to end: with single-candidate nodes relaxed, the address logit's
gradient is ~1e-9 against ~3e-2 at the combinator that sits downstream of the
comparison rather than upstream of it.

At the proposed carrier-width `tau = 2^bits = 256` the slope becomes 6.97e-03 at
`palette_levels 8` and 5.09e-02 at 16, and 13.2% of the comparisons actually
made at the rung-1 operating point acquire a live slope.  That is the setting in
which this question should be re-asked, and this generator has the dial for it.

**One caveat on any such fix.**  `SoftProgram` keeps one temperature per node and
uses it for both the candidate softmax and the operator relaxation, so widening
the surrogate also flattens that node's choice distribution.  They are not
independently tunable on current main.

---

## 8. What the next rung needs

Stated precisely, because each item is a measured prerequisite rather than a
guess.

1. **Rung two, widget extents.**  Rung one gives a per-position boundary flag.
   Turning that into widget rectangles needs an aggregation over positions that
   groups them -- `filter` and `count` over the mapped set exist and are used by
   the discrete-perception track, but there is no operator that returns the
   *extent* of a run.  The honest next step is a rung-2 module
   `(position, image) -> (position, is_left_edge)` composed from the frozen rung-1
   module at two offsets, which is expressible today, followed by a bound
   computed on whether a rectangle is determined by its edge flags.

2. **The glyph rung is reachable and needs one reduction.**  R2 measures the
   glyph determined by its bounding box at 0.9625 and transferring at 0.9250 --
   but only after binarising the box to ink/not-ink.  That reduction is
   `eq(pixel, ink_constant)`, which is legal, so the rung is a per-position
   module over a *box* rather than a pair.  What it needs from the caller is the
   box, which is rung two's output; this is a genuine ordering constraint, not a
   convenience.

3. **The kind rung needs a probe decision, not a bigger search.**  In the default
   colour mode a widget's kind is not determined by its appearance (ceiling
   0.4620 against 0.2982) -- the same informational wall as `geometry`'s
   `object_ids`.  It is determined by size at 0.8480 and by colour in
   `colour_mode=kind` at 1.0000.  So the ladder should either run the kind rung
   in `colour_mode=kind` and say so, or take size as its context, or accept the
   ceiling.  Searching the default mode would be wasted.

4. **The parent rung is free once extents exist.**  R4 measures the smallest
   strictly-containing rectangle recovering the true parent at 1.0000 for every
   widget at three complexities with zero ties.  It needs no pixel information at
   all, only the rectangle set -- which is what makes the
   `set[(id, parent, kind, x, y, w, h)]` probe the right target type: the parent
   field is derivable from the geometry fields, so a program that produces the
   rectangles produces the tree.

5. **Anything requiring arithmetic on pixels is blocked, not hard.**  There is no
   conversion out of `role="byte"`.  Rungs that would need an intensity
   comparison, a gradient magnitude, or any weighted sum over pixels cannot be
   written at all.  Nothing in this track needed one -- edges, ink and colour
   identity are all `eq`-shaped -- and no such rung is proposed here.  If one is
   wanted later, the missing piece is a declared byte-to-integer conversion with
   an explicit contract, which is a core change and is not made here.

---

## 9. Proposed changes to `tcn/`, none applied

Two agents are measuring against `main`, so nothing outside `generators/gui/`,
`research/gui-hierarchy/` and `tests/` was touched.  These are the diffs this
track's measurements support.

### D1 (restated, independently reproduced) — a pre-selected node severs the graph

`SoftProgram.forward` evaluates any node with `selected is not None` through
`exact_tensor(...).detach()`.  For a genuinely frozen module that is correct.
For a single-candidate node written by a scaffold builder it silently makes every
upstream choice unreachable, which is measured here as `grad is None` on the
address logit in four separate configurations.

```diff
--- a/tcn/learning.py
+++ b/tcn/learning.py
@@
-            if n.name in self.frozen:
+            # A node with exactly one candidate is not a frozen module: there is
+            # nothing to freeze. Detaching it severs every upstream choice from
+            # the loss while changing nothing about the discrete program.
+            if n.name in self.frozen and len(n.candidates) > 1:
                 c=n.candidates[self.frozen[n.name]]
                 values[n.name]=exact_tensor(self.registry,c.operator,[values[s] for s in c.sources]).detach()
                 continue
```

A scaffold can work around this today by writing `selected=None`, which is what
`common.Builder(relax_single=True)` does, so this is a usability and
correctness-of-measurement fix rather than a blocker.

### D2 — scale `eq`'s surrogate by the declared carrier width

```diff
--- a/tcn/learning.py
+++ b/tcn/learning.py
@@
     if n in COMPARE:
-        if n=="eq": return torch.exp(-((a-b)**2).sum(-1,keepdim=True)/temperature)
+        # The surrogate's width must be in the units of the carrier being
+        # compared. At tau=1 on int[8] data it is exactly 0.0 in float32 for
+        # |a-b| >= 11, so no comparison between two distinct colours passes any
+        # gradient at all. 2**bits is derived from the declared type, not tuned.
+        if n=="eq":
+            width=max(1.,float(2**op.inputs[0].bits)) if op.inputs[0].kind=="int" else 1.
+            return torch.exp(-((a-b)**2).sum(-1,keepdim=True)/(temperature*width))
```

Measured consequence on this generator: the slope at `palette_levels 16` goes
from 0.00e+00 to 5.09e-02.  This does not on its own make the address search
succeed here and is not claimed to; it makes the question askable.

### D3 — separate the candidate temperature from the relaxation temperature

`SoftProgram` keeps one `temperatures[node]` and passes it both to
`torch.softmax(logits/tau, 0)` and to `relaxed(..., tau)`.  Any fix to D2 that is
applied through the temperature therefore also flattens that node's choice
distribution.  A second field is the minimal change; it is an interface change to
`Crystallizer` as well, so it is named rather than drafted.

### D4 — `enumerate_fit` should report the conforming count

`SearchResult` carries `unique` but discards the conforming set, so a caller
cannot tell 2 conforming from 2,500 without re-running the sweep.  This track,
like `discrete-perception`, had to reimplement the loop to get the number that
mattered.  Returning `len(found)` alongside `unique` is a one-line change.

---

## 10. Files, tests and limitations

| file | what it does |
|---|---|
| `generators/gui/generator.py`, `render.py`, `manifest.json` | The generator: contract, probes, dials. |
| `tests/test_gui_generator.py` | 13 tests: contract/replay/restore/visibility, probe typing across screens, round trip and the load-bearing id, capacity and field-width enforcement, the owner probe against the tree, colour injectivity and the settings that break it, actions and illegal targets, the objective, the dials moving the achieved screen, `colour_mode=kind`, and the glyph channel. |
| `research/gui-hierarchy/common.py` | Episode/probe access, hand-wiring helpers, discrete and gradient references. Names the three copies from `research/discrete-perception`. |
| `bounds.py` -> `out/bounds.json` | Every recoverability bound, before any search. |
| `dial.py` -> `out/dial.json` | Whether each dial moves structure, appearance, ceiling and search. |
| `rung1_edges.py` -> `out/rung1.json`, `out/rung1.log` | Rung one, the ablations, stage B and stage C. |
| `surrogate.py` -> `out/surrogate.json` | Surrogate value, slope, operating distances and the measured gradient at each choice node. |
| `report.py` | Renders this document from `out/*.json`. |

**Limitations.**

* Rung one is a two-pixel horizontal boundary detector.  It is one rung; nothing
  above it is claimed.
* Every gradient number is from `common.local_fit`, which is `tcn.synthesis.fit`
  plus an `init_noise` argument, because `SoftProgram` zero-initialises every
  choice logit and `torch.manual_seed` does not vary synthesis.  Runs at
  `init_noise = 0` are reported as single outcomes.
* Wall clock is an upper bound; the host was shared throughout.  Node evaluations
  are reported alongside.
* The bounds are computed on 16 training and 16 held-out episodes per setting
  (8 and 8 in `dial.py`), which is enough to separate a ceiling of 1.0000 from
  one of 0.85 but not enough to resolve differences in the third decimal.
* `tcn.search` cannot score a recurrent program, so nothing here uses one.
