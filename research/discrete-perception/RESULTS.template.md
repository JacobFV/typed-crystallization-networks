# Discrete perception: search the structure discretely, apply it positionally

Research track `discrete-perception`.  Everything below was produced in this
workspace on 2026-09-08/09 with
`/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python`
(Python 3.13.15, torch 2.14.0+cpu).  **Nothing under `tcn/` or `generators/` was
modified**, and neither the `positional-reuse` nor the `perturbation-selection`
branch was merged; the three-node caller is copied into `common.py` and named as
a copy.  All code, logs and raw JSON are in this directory, and `tables.py`
renders every table below directly from `out/*.json`, so no number here is
transcribed by hand.

---

## 0. Verdict

**The convergence's architecture works, and it moves the perception ceiling up
by one rung -- from a per-pixel predicate to a learned two-position spatial
operator, with a uniqueness certificate.**  It also produces what appears to
be the first measurement in this repository where **gradient descent beats the
discrete backend on a pure synthesis task** -- track 8's one clean win for
relaxation was environment sample efficiency, not synthesis -- reported in full
in sections 3.1, 4 and 8.

Five results, in the order they should change what the project does.

1. **Searching the structure discretely and applying it positionally reproduces
   rung 3 and makes the search space independent of image resolution.**  The
   per-position module computes its addresses from its own position argument, so
   the discrete space is **32,000 programs at every width from 48 to
   12,288 bytes**, and the selections found at one resolution are
   the selections at every resolution.  The perception ladder's free-address
   scaffold had a space of `(3R^2)^2` and failed from 576 programs upward; there
   is no address search here at all.  Three caller nodes apply the module at
   every position, and `filter`+`count` aggregates the result -- both of which the
   ladder's operator audit concluded the algebra did not have.

2. **The highest rung reached is a learned two-position window (rung 3.5), and
   it is the only rung with a unique solution.**  Freezing the rung-3 foreground
   module and searching a 48-program window scaffold on top recovers the right
   neighbour offset and XOR, exhaustively, uniquely, with **zero held-out error**.
   The undecomposed version of the same task is
   49,152,000,000 programs and is not exhaustible.  This is track 3's
   dense-supervision result in its sharpest form: **staging behind an intermediate
   probe turned an unexhaustible space into 48 programs with a certificate.**

3. **`object_ids` and `depth` are not reachable, and the reason is
   informational, not algorithmic.**  The best possible per-pixel predictor scores
   *exactly* the majority baseline on held-out episodes for `object_ids`, for its
   easiest binary reduction, for a canonical relabelling, and for a binary depth
   threshold -- advantage 0.000 in every case -- while fitting the training pixels
   perfectly.  Object colours are re-drawn every episode and the renderer never
   attenuates with distance.  `enumerate_fit` **exhausts** the corresponding
   candidate families and certifies that no program in them fits.  That is a
   ceiling no search moves.

4. **A "solved, exhausted" certificate is not a correctness certificate, and
   this bit.**  At every rung-3 arm thousands of programs conform; ~5% of them
   disagree with the renderer on fresh episodes; `enumerate_fit` returns the
   lexicographically first, and at R=4 that program is **wrong on 3 of 384 test
   slots** -- while all six gradient runs over the same space return programs that
   are **exactly right**.  Filtering the conforming set through a second
   supervision split does not fix it.  Section 12's D2 and D3 are the two-line
   changes that would at least make the ambiguity visible.

5. **The two backends split by the *kind* of choice, and this track measures
   both halves of the split in the same architecture.**  Where the choice is one
   of many constants at a fixed *differentiable* input -- the 256-value byte
   alphabet, 4.3e9 programs -- relaxation is 6/6 with zero held-out error while
   exhaustive enumeration projects to 107 days and loses its certificate
   (section 8).  Where the choice sits **behind a `gradient="none"` boundary** --
   the window offset, which feeds a frozen module -- relaxation is 1/4 and its
   logits are measurably `grad = None`, while enumeration settles all 48 programs
   with a unique answer (section 5).  And where the choice is *which input to
   bind*, neither works, which is exactly why this architecture computes
   addresses instead of choosing them.  A NEAR-shaped backend split should be
   made on this axis, not on space size.

   The practical blocker on making the discrete half the default is not the
   algorithm but `enumerate_fit` re-executing whole programs: reusing the prefix
   is **17.8x** faster at R=8 and, unlike the current implementation, **flat in
   the width of the observation** (section 8.1).

---

## 1. What was built, and why this shape

Three measurements converged on one architecture: enumeration settles small
discrete spaces in milliseconds (track 8), relaxing an **input binding** is worse
than chance while relaxing a **constant or operator at a fixed input** is 208x
better than chance (perception ladder section 6.4), and one crystallized module
can be applied at every position of an arbitrarily wide observation with three
caller nodes (`positional-reuse`).  Put together:

```
stage A  (structure, discretely)
    module scaffold   (position, image bytes) -> (position, label)
        pos   = project(rec, 0)                     the module's own position
        obs   = project(rec, 1)                     the whole observation
        red   = index(obs, pos)                     addresses COMPUTED, not chosen
        green = index(obs, pos + 1)
        blue  = index(obs, pos + 2)
        cmp_c = eq(channel, k)        <- searched: which byte value      (5 each)
        rg    = truth_i(cmp_r, cmp_g) <- searched: how they combine      (16)
        label = truth_j(rg, cmp_b)    <- searched                        (16)
    searched exhaustively by `tcn.search.enumerate_fit` over
    (episode, position) records: |space| = 5^3 * 16^2 = 32,000

stage B  (application, positionally)
    held    = insert(empty, observation)        set[Observation], capacity 1
    records = pair(positions, held)             set[(Index, Observation)]
    mapped  = map(records; module = m)          set[(Index, Label)]
    three caller nodes, for any number of positions

stage C  (aggregation)
    kept  = filter(mapped; module = flag)
    total = count(kept)
```

The load-bearing difference from the perception ladder's rung 3 is in one line
of stage A: **`index(obs, pos)` computes the address from the module's own
position argument instead of choosing it from a menu over the whole image.**
The ladder's free-address scaffold had a space of `(3R^2)^2` and gradient
synthesis failed from 576 programs upward; this scaffold's space is **32,000 at
every resolution**, because widening the image adds candidates to nothing.

Everything else is unchanged: pixels are `role="byte"` and so non-numeric, `eq`
is the only differentiable predicate on them, `map`/`pair`/`insert` declare
`gradient="none"`, and the module is trained on single-position records, so the
whole pattern sits behind a declared gradient boundary exactly as the
`positional-reuse` branch reported.

**Protocol.**  Program inputs are filled only from
`StepRecord.actor_view().observations` (`pixels`, a raw byte tuple);
`latent_states` and `probes` build `targets` and nothing else.  A run succeeds
only if the hardened program reproduces the targets exactly at 1e-6.  Held-out
episodes come from disjoint seed ranges (train 0..7, held-out 100..107, apply
200..207).  One declared `camera` action moves the eye in, because at the
shipped camera 98% of every image is background at every resolution and any
accuracy figure there is a report on a constant; the resulting split is
51.2% foreground / 48.8% background at every resolution from 4 to 32 (`out/audit.json`).

**`SoftProgram` zero-initialises every choice logit, so `torch.manual_seed` does
not vary synthesis.**  Every gradient number here comes from `common.local_fit`,
which is `tcn.synthesis.fit`'s control flow plus an `init_noise` argument, and
is verified bit-identical to `fit` at `init_noise = 0`:

{{check_local_fit}}

**Wall clock caveat.**  This host was shared with other agents throughout (load
average 55-65 on 20 cores).  Every timing below is therefore an upper bound and
noisy at the tens-of-percent level.  Programs evaluated is the load-independent
cost measure and is reported alongside every time.

---

## 2. What the geometry probes are, before any search

{{audit}}

Read the last four columns.  **`object_ids >= 0` is an exact function of the
pixel and transfers perfectly**: the best RGB lookup table fitted on the training
episodes scores 1.0000 on held-out episodes at every resolution from 4 to 32.
**`object_ids` itself is not**: the same table fits the training pixels perfectly
(there are *no* colliding RGB values, because a colour rarely recurs) and then
scores exactly the majority baseline on held-out episodes, to four decimal
places, at every resolution.  That is memorisation with zero transfer, and it is
the whole of rung 4 in one row.  Exact `depth` is not even a function of the
pixel: the fraction of training RGB values that map to more than one depth rises
from 0.254 at R=4 to 0.839 at R=32.

The single exactly-learnable thing an image in this generator contains is
therefore **which pixels the renderer never painted**, and `render()` sets both
`depth = 0` and `object_ids = -1` there, which `audit.py` confirms agree on
every pixel of every episode measured.  That equivalence class is rung 3.

---

## 3. Rung 3, reproduced through this route, and the resolution climb

{{rung3}}

Both rows are the same 32,000-program space -- the module never names an absolute
address, so widening the image from 48 to 192 bytes adds no candidates -- and
both are **exhausted** with `solved = True` and `unique = False`.  Held-out error
on the supervised *records* is 0.0 at both resolutions.

**Read the last three columns before believing the first ones.**  Applying the
returned module at every position of three fresh episodes gives a max error of
**1.0** and a foreground-count error of **3** at R=4 and **8** at R=8.  The
certificate is sound -- the search really did see all 32,000 programs and the
returned one really does fit the training records exactly -- and the delivered
program is still wrong on pixels it was never shown.  Section 4 takes that
apart; section 3.2 shows what selection rule fixes it.

The cheap number is worth keeping in view: `enumerate_fit` with
`stop_at_first = True` finds a conforming program in **21 evaluations** -- 0.04 s
at R=4 and 1.35 s at R=8 -- against 61.8 s and 1,087.9 s for the sweep that earns
the certificate.  Almost the entire cost of the discrete path here is buying a
uniqueness verdict that comes back "no".

### 3.1 The gradient path on the identical space and data

{{rung3_gradient}}

**Gradient descent succeeds 4/4 at both resolutions, and the programs it returns
are exact on held-out episodes**, while the exhaustive sweep's returned program
is not (section 4).  It is also faster than the sweep at both rows -- 22.0 s
against 61.8 s at R=4, 178.5 s against 1,087.9 s at R=8 -- though slower than
stop-at-first enumeration, which settles it in well under two seconds.

That ordering is what the perception ladder predicts for this scaffold and it is
worth restating why: **every free choice here is a constant or an operator at a
fixed input**, which is the case the ladder measured relaxation to pick correctly
0.67-0.81 of the time against chance rates of 0.004-0.33.  There is no address
relaxation anywhere in this module, because the addresses are computed.  The
architecture that makes enumeration cheap is the same architecture that makes
relaxation work.

The random-draw control puts both in context: a uniform random program from the
same space conforms about 8% of the time, so *finding* a conforming program is
easy by any method.  The whole difficulty of this rung is finding the *right*
one, and that is a question about the probe, not the search.

### 3.2 The resolution climb: search once, apply at every width

{{resolution_transfer_search}}

{{resolution_transfer}}

The module is searched **once**, at R=8, and the resulting selections are used
verbatim at every width: only the input *type* changes, because the program never
names an absolute address.  The space is 32,000 programs at every row, the caller
is **3 nodes** at every row, and the applied module is **exact against the whole
dense probe** and against the `filter`+`count` foreground total at every
resolution from 8 to 48 -- **2,304 positions and a 6,912-byte observation** at
the top row.

Two costs behave differently, and the difference is the point.  **Application**
grows with the image, as it must: 0.08 s per image at R=8 to 64.0 s at R=48,
which is 64 to 2,304 module calls, and the `positional-reuse` branch's own
caveat applies -- 73% of that is `Value.of` re-encoding, because `pair`
replicates the observation per position.  **Search** does not grow with the
image *in the size of the space*, only in the cost of executing one candidate:
the last column projects what a fresh `enumerate_fit` sweep would cost at each
width, and it rises from 385 s to 2,602 s purely because `Value.decoded` walks a
wider tuple.  Section 8.1 removes that too.

**The selection rule matters more than the resolution.**  The first version of
this table used the same rule `enumerate_fit` uses -- the first conforming
program, filtered through a validation split of subsampled positions -- and was
**wrong at every resolution**: max error 1.0 and foreground-count errors of 8,
28 and 65 at R=8, 16 and 24 (`out/resolution_transfer_naive.json`).  The table
above selects the first program that is exact at *every* position of the
validation episodes.  That is the section 4 lesson and the section 5 lesson in
one line: what has to be checked is the module's accuracy everywhere it will be
applied, not its conformance on the records it was searched with.

### 3.3 How much of the rung is the scaffold?

{{offsets_free}}

The rung-3 scaffold supplies one thing beyond the positional pattern: that a
pixel's three channels sit at `pos`, `pos+1`, `pos+2`.  Freeing both offsets over
`{1,2,3}` makes the module discover the interleaving of an RGB raster as well as
the background colour and the Boolean combination: **288,000 programs**, nine
times rung 3's space.

Exhaustive search still settles it -- 288,000 programs swept in 109 s with
prefix reuse, 10,432 conforming, 9,360 of those also exact on held-out episodes,
and the returned program exact on held-out.  `enumerate_fit`'s stop-at-first
finds one in **21 evaluations and 0.90 s**.  The address of every read is still
*computed*: this is a wider constant search, not an address search, which is why
it costs 9x rather than the `(3R^2)^2` the perception ladder's free-address
scaffold cost.

**The gradient arm on this scaffold is 1/4, and three of the four failures are
not learning failures.**  They are `IndexError('index outside tuple')`: offset 3
on the last pixel of the image reads one byte past the end, and neither
`tcn.search.evaluate` nor `SoftProgram`'s export path treats an out-of-range
address as an unusable candidate.  The prefix-reusing enumerator catches it and
carries on, which is the only reason the exhaustive column exists at all.  This
is D1 in section 12, found by accident and worth fixing before any window
operator is searched at an image border.

---

## 4. Non-uniqueness is a statement about the probe

{{rung3_identification}}

The conforming set is monotone in the training set, so the curve is produced by
one exhaustive sweep at the smallest budget and filtering the survivors upward.

Two things are worth separating.  **The supervision identifies the *function*
almost completely and the *program* not at all.**  Roughly 95% of the conforming
programs are also exact on held-out episodes -- they are extensionally equal on
this distribution, differing only in which redundant comparison they ignore --
and that fraction barely moves as supervision grows by 8x.  No amount of data
from this generator will separate `not(r==24 and g==30)` from
`not(r==24 and g==30 and b==43)`, because the renderer never paints a foreground
pixel whose red and green happen to be 24 and 30.

**The remaining 5% are wrong, and `enumerate_fit` can return one of them.**  That
is the part that matters, because a caller sees `solved = True`,
`exhausted = True` and a program, and `unique = False` is the only signal that
the program it just received might be one of a few thousand, of which one in
twenty disagrees with the renderer on episodes it has not seen.

{{tiebreak}}

`enumerate_fit` returns the first conforming program in candidate order.  At
R=4 that program is **wrong on 3 of 384 test slots** -- three pixels of three
fresh episodes -- and filtering the conforming set through a second, disjoint
supervision split of the same size does not fix it: 2,464 programs conform on
training records, 2,336 also conform on validation, and the first of those 2,336
is the same program with the same three errors.

Gradient descent over the identical space and the identical training records
delivers a program that is **exactly right on every test slot**, and does so in
6 of 6 seeds.  Those six runs converge to **five distinct programs**, and
`out/gradient_spread.json` records that **all six are exact on the test
episodes** -- so this is not one lucky seed.

This is a genuine win for the relaxed path, and it is worth being precise about
what it is a win at.  Both methods search the same 32,000 programs; both find
conforming programs; enumeration additionally certifies that it has seen all of
them.  What differs is the tie-break among an under-determined conforming set:
lexicographic order is an arbitrary prior that happens to prefer a degenerate
program here, while the softmax relaxation is biased towards the choice that
lowers the loss fastest and lands on the renderer's own comparisons.  This is
exactly the regime the perception ladder measured relaxation to be good at --
every free choice is a constant or an operator at a *fixed* input, where the
steepest-descent direction picks the reference candidate 0.67-0.81 of the time
against chance rates of 0.004-0.33.

The fix is not to abandon enumeration; it is to stop pretending the returned
program is the answer.  D2 and D3 in section 12 are the two-line changes.

---

## 5. Rung 3.5 -- a learned two-position window, and what decomposition buys

{{rung35}}

**This is the highest rung reached, and the only one anywhere in this track or
the perception ladder whose solution is certified unique.**

The target is `fg(i) != fg(i + offset/3)` -- does the foreground mask change
between raster position `i` and its neighbour -- and the module has to discover
*which* neighbour: the offset is a searched choice over `{+1 pixel, +2 pixels,
+1 row}`.  Positive fraction 16.7%.

Stage 1 is the rung-3 foreground module, searched exhaustively over the 32,000
programs, this time supervised at **every** position of every training episode
(512 records).  1,952 programs conform -- still not unique -- but the one
returned is now measured to be **exact at every position of every training and
held-out episode** (accuracy 1.000000 on both), which is what the second stage
needs and what the subsampled supervision of section 3 did not give.

Stage 2 then searches 48 programs -- three offsets by sixteen combinators --
exhausts them in 4.6 s, and returns **exactly one**: offset
`+3` (the next pixel in raster order) combined by `truth_6`, which is XOR.  Held-out
max error **0.0**, held-out accuracy **1.0**.

**And the gradient column is the mirror image of section 3.1.**  On this
scaffold `local_fit` succeeds **1 of 4** seeds, and the reason is structural
rather than statistical: the offset node feeds a frozen module, `map`-style
operators declare `gradient="none"`, and a direct measurement of the choice
logits confirms it -- `shifted` has `grad = None` (unreachable in the autograd
graph) while `edge` has `|grad|_1 = 0.771` (`out/staged_gradients.json`).  Three
of the four runs settled on the same wrong offset and then fitted the combinator
to it.  So **relaxation cannot learn the neighbour at all here, and enumeration
settles it in 48 programs with a certificate** -- the exact converse of the
full-alphabet result in section 8, and between them they draw the line the
project has been looking for:

* a choice **behind a frozen module or any `gradient="none"` operator**: only the
  discrete backend can make it;
* a choice over **many constants at a fixed differentiable input**: the relaxed
  path scales where the discrete one loses its certificate;
* a choice of **which input to bind**: neither, which is why this architecture
  computes addresses instead of choosing them.

Two things follow.

*The operator algebra does have a window.*  The perception ladder's audit
concluded that "every program that reads pixels must name individual bytes by
index" and that there is "no convolution, no window, no pooling".  Once the
observation is a set of positional records, a shared module addresses relative to
its own position with `add`, and the relative offset is an ordinary searched
constant.  That is a two-tap convolution, learned, with no operator added.  The
same applies to aggregation: section 3's stage C runs `filter` then `count` over
the mapped set at every resolution tested.

*The decomposition is what makes it searchable.*  The flat arm searches the same
predicate with nothing frozen -- both pixels' three channel comparisons and all
five combinators -- and that is 49,152,000,000 programs.  A capped sweep of it runs at 205 programs/s, which projects the exhaustive
version at **2.4e8 s -- about 7.6 years**.  It is not exhaustible, and the staged
arm's 48-program uniqueness certificate has no counterpart there.  The staged
space is **1.0e9 times smaller** than the flat one for the same target.

This is the concrete instance of track 3's dense-supervision result in this
architecture: **staging the search behind an intermediate probe turned an
unexhaustible space into 48 programs with a uniqueness certificate.**  It is also where the
failure mode lives, and that variant is recorded in full
(`out/rung35_window_subsampled.json`).

Supervise stage 1 on 48 of the 64 positions per image instead of all 64 and it
still exhausts its 32,000 programs, still reports `solved`, and still has **zero
held-out error on the records it was scored on** -- but its module is only
**0.998047** accurate at *every* position of the training episodes and
**0.997396** on held-out episodes.  Two pixels in a thousand.  Stage 2 then
exhausts all 48 programs and correctly certifies that **none of them fit**, and
the best-accuracy program in that space is still the right one -- offset `+3`,
`truth_6` -- at 0.99479 train and 0.99306 held-out accuracy.

So a staged search is exactly as good as the stage it froze, its certificate is
about the composed space rather than about the target, and the diagnostic that
distinguishes the two cases is not the search's own report but the frozen
module's accuracy at *every* position, which is what
`accuracy_all_positions_train` measures.

---

## 6. Rung 4 -- `object_ids`, the multi-class rung

### 6.1 The information ceiling

{{rung4_ceilings}}

A shared per-position module that reads only its own three bytes cannot beat the
best RGB lookup table on those bytes, so this table is an upper bound on that
entire family of programs, computed outside the operator algebra on purpose.

`foreground` is the row that works: **1.0000 held-out against a 0.4492 majority
baseline.**  Every other row has an advantage over the majority baseline of
**exactly zero**.  `object_ids` fits the training pixels perfectly with *no*
colliding RGB values -- a colour essentially never recurs, so the table simply
memorises -- and then scores the majority baseline on held-out episodes.
`is_object_0`, the easiest possible binary reduction, does the same.  So does
`raster_rank`, which relabels the ids canonically by first appearance in raster
order and therefore removes the "ids are arbitrary list positions" objection: it
does not help, because the *colours* are re-drawn every episode, so no fixed
function of appearance can name an instance.

The failure is not resolution, budget, or search.  It is that the generator
re-randomises object colours per episode while ids are list positions, so
`object_ids` is not a function of appearance at all -- it is a function of
appearance *and the episode*, and the episode is not an input.

{{rung4_colours}}

### 6.2 Exhaustive search in every family the algebra allows

{{rung4_families}}

Every row is **exhausted**.  Only the control is solved.

* `colour-constant / foreground` is the positive control -- the rung-3 family on
  the rung-3 target -- and it behaves: 32,000 programs swept, solved, held-out
  accuracy 1.0, and (as everywhere) not unique.
* `colour-constant / is_object_0` sweeps the same 32,000 programs against the
  easiest possible reduction of the multi-class target and **certifies that no
  program in the family fits**.
* `relational / *` searches the only episode-adaptive handle the algebra has: the
  module compares its own pixel against the pixel at a **searched reference
  address**, over all 64 addresses of the image -- the free-address choice the
  perception ladder measured relaxation to be *worse than chance* at.
  Enumeration settles it in both cases, exhaustively, with **no solution**, and
  the gradient arm on the same space is **0/4**.  Note that this family does not
  contain the rung-3 solution either (`relational / foreground` is also
  unsolved), because no *fixed* address is background in every episode; so its
  failure on `is_object_0` is corroboration rather than proof.
* `multi-class relational` is the genuine article: the module's output is an
  integer class label built by `mux` over three searched reference pixels, at
  R=6 with three objects, and its 46,656 programs are exhausted in 19 s with **no
  solution**.

Two caveats, stated because they bound what the certificates mean.  `is_object_0`
is only **1.04%** positive at R=8 with six objects, so exact conformance is a
demanding criterion and a constant-`false` program would already be 98.96%
accurate; the informative statistic for this target is the ceiling table above,
where the best possible per-pixel predictor has **advantage 0.000** over the
majority baseline.  And every family here is one I wrote; a certificate says
"not in this space", never "not in any space".

The underlying obstruction is measurable directly: {{rung4_colours}}  An object
is not one colour, so colour equality cannot even separate instances *within* an
image, let alone name them across episodes where the colours are re-drawn.

---

## 7. Rung 5 -- `depth`

{{rung5_ceilings}}

{{rung5_families}}

{{rung5_best}}

`near` is `(depth > 0) and (depth < 0.573)`, the median foreground depth over the
training episodes; 25.8% of training pixels are positive and the held-out
majority baseline is 0.8105.

**No information source in reach beats that baseline.**  A perfect RGB lookup
table fitted on the training pixels reaches 0.9883 on them and **0.8105** on
held-out episodes -- the majority baseline, to four decimal places.  Adding the
position to the key takes training accuracy to **1.0000** and held-out accuracy
to **0.8105**, unchanged.  The renderer shades by surface normal and never
attenuates with distance, so a pixel's colour carries object identity and facing,
not range, and the objects move every episode, so position carries nothing
either.

The searches agree.  `enumerate_fit` **exhausts** the 32,000-program
colour-constant family -- the family that solves rung 3 -- and certifies that no
program in it computes `near`.  It exhausts the 480-program position-threshold
family with the colour comparisons pinned to the (already certified) background
test and certifies the same.  The best-*accuracy* program in that family reaches
0.8203 on training records and **0.8086 held-out, below the 0.8105 majority
baseline**.  Gradient descent over the joint 15,360,000-program family is
0/4 at 273 s median.

This is the cleanest kind of negative result available: not "the search failed"
but "the search finished, and the answer is that nothing in the space computes
this".  The exception proves the shape of the rung -- `depth == 0` *is*
computable, and is exactly rung 3, because it is the renderer's untouched-pixel
marker rather than a range measurement.

---

## 8. Where brute force stops, and what replaces it

{{full_alphabet}}

Widening the byte pool from five values to all 256 makes the same rung-3
scaffold **4,294,967,296** programs.  Four things happen, and only one of them is
the one the convergence story predicts.

**Finding a solution stays easy; certifying it becomes impossible.**  Brute force
in candidate order finds a conforming program inside its first 100,000
evaluations, because the space is dense in solutions -- `not eq(blue, 43)` alone
separates the classes on this data, and the enumerator reaches it early.  What is
out of reach is exhaustion: at the measured rate the full sweep projects to
**9.2e6 s**, about 107 days.  So `enumerate_fit` would return `solved = True`,
`exhausted = False`, `unique = None`, which is precisely the state in which the
uniqueness certificate -- the thing the discrete path is supposed to add -- is
unavailable.

**Scoping the candidate set by the observed alphabet does not rescue it.**
Restricting each comparison's constants to byte values that actually occur in the
training observations (a data-derived scope using no probe and no renderer
knowledge) leaves 129 values and a 7.8x smaller space.  The projection is still
1.9e6 s.  Scoping helps by a constant factor against an exponential.

**Coordinate descent with random restarts is unreliable here.**  Over six
restarts it solved once, in 801 evaluations, and plateaued the other five times
at the majority-class accuracy.  The landscape is the reason: from a random
start every single-coordinate move is neutral, because a comparison node only
becomes informative once the two combinators downstream of it are also right.
It is a genuine search algorithm and it is not the answer.

**Gradient descent is the arm that does not care about the width of the pool.**
Over the same 4,294,967,296 programs, with initialisation noise and six seeds,
`local_fit` succeeds **6/6** at a median of 601 s, and **all six programs have
zero held-out error**; one of them (seed 1) recovers the renderer's own
`(24, 30, 43)` exactly, the others land on equivalent programs that ignore a
redundant channel.  A 256-way choice of a
*constant at a fixed input* is precisely the case the perception ladder measured
the relaxation to be 208x better than chance at, and it is unaffected by the pool
being 5 values or 256.  So the honest summary of this section is: **at the point
where exhaustive enumeration loses its certificate, the relaxed path is the arm
that still works**, and enumeration's remaining contribution is that
stop-at-first is 465 programs/s and finds a solution in the first 100,000.

### 8.1 What actually makes exhaustive search affordable: reuse the prefix

{{incremental}}

`tcn.search.evaluate` re-executes the *whole* program for every discrete
selection.  In a per-position module almost nothing depends on the choices: the
two `project`s, the two `add`s and the three `index` reads are identical for
every candidate program, and at the shipped `Value.decoded` each `index` walks
the entire byte tuple.  So a 32,000-program sweep pays that walk 32,000 times,
and its cost grows with the width of the *observation* even though the space is
fixed -- which is exactly the asymmetry section 3.2 is about.

`incremental.py` runs the identical search -- same candidates, same order, same
conformance test -- as a depth-first walk of the choice tree, computing each
node's value once per prefix instead of once per leaf, and checking a probed
node example by example so a candidate that already disagrees costs one record
rather than all of them.  It returns the **identical conforming set**, which is
asserted against `common.all_conforming` (the loop `enumerate_fit` uses) at
R=8.

At R=8, over 384 records and the identical 32,000 programs: `enumerate_fit`'s
loop takes **1,067.9 s**, the prefix-reusing walk takes **60.0 s** -- an **17.8x**
speedup -- and the two conforming sets are asserted equal.  What matters more
than the constant is the shape: the prefix-reusing walk performs about **36,000
node evaluations at every width**, so its cost is **flat in the size of the
observation** (60.0, 71.6, 47.8 and 74.2 s at 192, 768, 3,072 and 12,288 bytes,
the spread being host load), while `enumerate_fit`'s cost is proportional to it.

That closes the asymmetry section 3.2 opens.  The *space* is already independent
of the image, because the positional module computes its addresses; with prefix
reuse the *search* is too.

This is not a better search algorithm; it is the same algorithm with the
redundancy removed, and it is the single change that would move the discrete
backend from "affordable on toys" to "the default for perception".  It is
written up as a proposed diff in section 12 rather than applied.

---

## 9. The ladder, end to end

| rung | target | agent input | supervision | discrete result | gradient result | verdict |
|---|---|---|---|---|---|---|
| **3a** | per-pixel foreground mask, R=4 | raw byte tuple | `object_ids >= 0` | 32,000 programs exhausted; **not unique** (2,464 conform); the returned program is wrong on 3 of 384 test slots | **6/6 seeds exact on every test slot** (5 distinct programs; the main arm is 4/4 at median 22.0 s) | **works, and the gradient path picks better** |
| **3b** | the same at R=8 | raw byte tuple | `object_ids >= 0` | 32,000 exhausted in 1,087.9 s, **not unique** (2,608 conform); applied module wrong on fresh episodes (count error 8) | **4/4 seeds exact**, median 178.5 s | works |
| **3c** | the same, channel offsets also searched | raw byte tuple | `object_ids >= 0` | 288,000 exhausted in 109.1 s (prefix-reusing); 10,432 conform, 9,360 also on held-out; stop-at-first 21 evaluations / 0.90 s | 1/4 -- and 3 of the 4 aborted with `IndexError` at the image border, not a learning failure (D1) | **works; the raster layout is discovered, not supplied** |
| **3d** | apply the module at every position, R=8..48 | raw byte tuple | -- (search reused) | search once at R=8 (1,952 conform, 1,640 also exact at every validation position); **3 caller nodes and max error 0.0 at R=8, 16, 24, 32 and 48**, up to 2,304 positions and 6,912 bytes | -- | **works; search cost is resolution-independent** |
| **3e** | `filter`+`count` foreground total | raw byte tuple | -- (search reused) | exact at every resolution applied | -- | **works -- an aggregate over an image** |
| **3.5** | two-position edge (`fg(i) != fg(i+k)`), offset searched | raw byte tuple | `object_ids >= 0`, then the edge | staged: 48 programs exhausted, **unique**, held-out error 0.0. flat: 49,152,000,000 programs, projected 2.4e8 s | staged **1/4** (the offset node has `grad = None`); flat **0/3** | **highest rung reached; the only unique certificate** |
| **4a** | `object_ids`, multi-class | raw byte tuple | `object_ids` | multi-class `mux` over three searched reference pixels, 46,656 programs **exhausted, no solution** | -- | **unreachable: lookup ceiling = majority baseline** |
| 4b | `object_ids == 0`, colour-constant family | raw byte tuple | `object_ids == 0` | 32,000 **exhausted, no solution** | -- | unreachable |
| 4c | `object_ids == 0`, relational family (reference address searched) | raw byte tuple | `object_ids == 0` | 16,384 **exhausted, no solution** (the same family also fails to contain the rung-3 solution) | 0/4 | unreachable |
| **5a** | `near = 0 < depth < median` | raw byte tuple | `depth` | colour family: 32,000 exhausted, **no solution**. position family: 480 exhausted, **no solution**; best accuracy 0.8203 train / 0.8086 held-out against a 0.8105 majority baseline | 0/4 | **unreachable: colour and position both carry nothing** |
| 5b | `depth == 0` | raw byte tuple | `depth` | identical to rung 3 -- it is the renderer's untouched-pixel marker | -- | works (= rung 3) |

---

## 10. The ceiling and its cause

**The highest rung reached is a learned two-position spatial operator over raw
pixels -- rung 3.5 -- and it is the only rung anywhere in this track or the
perception ladder whose solution enumeration certifies as unique.**  Everything
below it (the per-pixel foreground mask, the positional application at every
resolution tested, the `filter`/`count` aggregate) works; everything above it
does not, and the reason changes as you go up.

**Where the ladder stops, and why -- three separate causes, in the order they
bite.**

1. **`object_ids` and `depth` stop for an information reason, before any search
   question is asked.**  The best possible per-pixel predictor -- an RGB lookup
   table, an upper bound on every program in the family -- scores *exactly* the
   majority baseline on held-out episodes for `object_ids`, for the one-vs-rest
   reduction, for a canonical raster-order relabelling, and for a binary depth
   threshold.  The generator re-randomises object colours every episode, so
   appearance does not name an instance; and the renderer shades by surface
   normal without distance attenuation, so appearance does not encode range.
   `enumerate_fit` **exhausts** the relevant candidate families and certifies
   that no program in them fits.  This is the ceiling, and no better search
   algorithm, larger budget or higher resolution moves it.  What would move it is
   a different target -- one that is a function of the observation.

2. **Non-uniqueness stops the *certificate* long before it stops the *program*.**
   At every rung-3 arm measured, thousands of programs conform; about 95% of them
   are extensionally equal on this distribution and 5% are not; and
   `enumerate_fit` returns whichever comes first in candidate order, which at R=4
   was wrong on fresh episodes.  More supervision shrinks the conforming set
   slowly and never to one, because the surviving ambiguity is real: the
   generator never produces a pixel that separates the equivalent programs.  The
   only rung that broke through was the staged one, where the *second* stage's
   48-program space had a unique answer.

3. **Exhaustive enumeration stops at a space size that is much smaller than it
   should be, for an implementation reason rather than a fundamental one.**
   `enumerate_fit` re-executes the whole program for every candidate, so a
   32,000-program sweep over a few hundred image records costs minutes and grows
   with the *width of the observation* even though the space does not.  Reusing
   the prefix -- the same search, the same candidates, the same conforming set --
   removes both.  That is the difference between "the discrete backend is the
   default for perception" and "the discrete backend is affordable on toys".

---

## 11. Threats to validity

* **Scaffold authorship.**  Every scaffold here was written with a reference
  program in mind, so these are best-case difficulty measurements.  The
  positional pattern supplies "read your own three channels at `pos`, `pos+1`,
  `pos+2`", which is the raster layout; section 3.3 frees those two offsets and
  re-measures, but the *shape* of the module -- three comparisons and two
  combinators -- is still supplied.  A scaffold discovered rather than written
  would be harder, not easier, and on main it cannot be discovered at all,
  because `legal_candidates` never proposes `project`, `map`, `filter` or `join`
  (D4).
* **A shared host.**  Load average was 55-80 on 20 cores throughout, from other
  agents.  Wall clocks are upper bounds, they are not comparable across hours,
  and three long runs were killed by memory pressure part-way and had to be
  re-run with incremental dumping.  Every cost claim is therefore stated in
  programs (or node evaluations) as well as seconds, and the qualitative
  comparisons -- exhaustive versus stop-at-first, prefix-reusing versus
  re-executing, gradient versus enumeration -- were each measured within a
  single run under the same load.
* **`local_fit`.**  The seed variation needed to report a success *rate* does
  not exist in `tcn.synthesis.fit`, so every gradient arm uses a local copy with
  `init_noise = 0.5`.  It is verified bit-identical to `fit` at zero noise on
  the shipped `examples/mixed.py` fixture at two budgets and both polish
  settings (section 1), but it is still a copy, and 0.5 is an arbitrary scale.
* **Derived targets.**  Every rung supervises on an equivalence class of a
  probe (`object_ids >= 0`, `depth < T`) rather than the probe itself.  This is
  privileged supervision, stated at each point of use, and section 2 gives the
  measurement that motivates it.
* **The `camera` action** changes the data distribution relative to the shipped
  configuration.  It is a declared action from the generator's own action
  schema, not a code change, and section 2 gives the measurement that motivates
  it (98% background at the shipped camera at every resolution).
* **Supervision budgets are small.**  Rung 3 is searched over 128-512
  (episode, position) records at R=4-8.  The identification curve in section 4
  is the direct measurement of what that costs, and it is the main caveat on
  every "solved" verdict in this report.
* **The information ceilings in sections 6 and 7 are computed outside the
  operator algebra**, as ordinary Python lookup tables.  That is deliberate --
  they are upper bounds on a whole family of programs, not programs themselves --
  but it means they are evidence about the *data*, and the corresponding
  evidence about the *algebra* is the exhaustive-search certificate beside them.

---

## 12. Proposed changes to `tcn/`, as diffs.  None applied.

Nothing under `tcn/` or `generators/` was touched by this track.  Each item names
the measurement that motivates it.

### D1 (bug).  `tcn.search.evaluate` does not catch `IndexError`, so one
out-of-range address aborts the whole sweep

`index(tuple, i)` raises `IndexError("index outside tuple")` when `i` is outside
the tuple, and `evaluate`'s `except` clause lists
`(ValueError, TypeError, OverflowError, ZeroDivisionError, ArithmeticError)`.
An out-of-range address is exactly the case its own docstring describes -- "a
property of that candidate, not an error in the search" -- and it is the *normal*
case for a positional module with a relative offset, because the offset runs off
the end of the observation at the image border.  Verified in
`out/index_bug.json`: a two-candidate scaffold whose second candidate reads one
byte past the end makes `enumerate_fit` raise instead of returning a result.

```python
-        except (ValueError, TypeError, OverflowError, ZeroDivisionError, ArithmeticError):
+        except (ValueError, TypeError, IndexError, OverflowError, ZeroDivisionError,
+                ArithmeticError):
             return None
```

### D2.  `SearchResult` should report how many programs conform, not only whether
one does

`enumerate_fit` returns `found[0]` and `unique = len(found) == 1`.  When the
solution is not unique the caller learns nothing about *how* under-determined the
problem is, and silently receives the lexicographically first program, which is
a prior.  Measured at R=4 on rung 3: **2,464 of 32,000** programs conform on the training
records, about one in twenty of them disagrees with the renderer on episodes it
has not seen, and the one `enumerate_fit` hands back is wrong on 3 of 384 test
slots (section 4).

```python
-    started = time.perf_counter(); evaluated = 0; found = []; best = float('inf')
+    started = time.perf_counter(); evaluated = 0; found = []; best = float('inf'); hits = 0
     for combination in itertools.product(*(range(c) for c in counts)):
         ...
         if error <= tolerance:
-            found.append(selections)
+            hits += 1
+            if len(found) < keep: found.append(selections)
             if stop_at_first: break
```

with `keep: int = 1` a new argument, `conforming: int = 0` a new
`SearchResult` field, and `unique` derived from `hits == 1` as before.  This is
information the sweep already computes and throws away.

### D3.  `enumerate_fit` should accept a validation set and use it as the
tie-break

The one-line consequence of D2's measurement: when several programs conform, the
useful selection rule is "also conforms on episodes not used to search".
Measured at R=4, filtering the conforming set through a second disjoint
supervision split takes it from 2,464 to 2,336 programs -- and the first of those
2,336 is the *same* program with the *same* three errors, so on this task the
tie-break has to be reported rather than trusted.  That is the honest form of the
change: a caller that is told "2,336 programs conform, here is one of them"
behaves differently from a caller told "solved".

```python
-def enumerate_fit(program, examples, signals, registry=None, tolerance=.001, ...):
+def enumerate_fit(program, examples, signals, registry=None, tolerance=.001,
+                  validation=(), ...):
...
-        if error <= tolerance:
+        if error <= tolerance and (not validation or (lambda e: e is not None and e <= tolerance)(
+                evaluate(program, selections, validation, signals, r, tolerance))):
             found.append(selections)
```

Reporting both counts -- conforming on the search set, and conforming on both --
keeps the certificate meaningful: exhausting the space still certifies
*something*, just not what the unqualified word "unique" suggests.

### D4.  Merge `positional-reuse`, and say in `ARCHITECTURE.md` section 2 what it
buys

Two of the perception ladder's three walls are properties of *tuple-shaped*
observations, not of the type algebra:

* "there is no aggregation over an image" -- `filter` then `count` over the
  mapped record set is an aggregate, and this track runs it at every resolution
  tested (section 3);
* "every program that reads pixels must name individual bytes by index" -- a
  shared module addresses relative to its own position with `add`, which is a
  window, and section 5 searches the window offset.

Neither needs an operator.  Both need `insert`/`pair`/`map`, which needs the
branch's `legal_candidates` fix to be *discoverable* rather than hand-written --
on main, `legal_candidates` resolves every candidate with empty parameters, so
`project`, `map`, `filter` and `join` are never proposed.  Every scaffold in this
track supplies its candidates explicitly for that reason.

### D6 (largest measured payoff).  `enumerate_fit` should walk the choice tree,
not re-execute the program per candidate

`evaluate` calls `Program.execute`, which recomputes **every** node for **every**
discrete selection.  In a per-position module the projections, the offset
arithmetic and the three `index` reads are identical across the whole space, and
each `index` walks the entire observation through `Value.decoded`.  Measured at
R=8 over 384 records and the same 32,000 programs: **1,067.9 s** for the current
loop against **60.0 s** for a depth-first walk that computes each node once per
prefix and checks a probed node example by example -- **17.8x**, with the
conforming sets asserted equal.  More importantly the walk performs ~36,000 node
evaluations at 192, 768, 3,072 and 12,288 bytes alike, so its cost is **flat in
the width of the observation** where the present one is linear in it.

`incremental.py` in this directory is a working prototype (about 50 lines) and
the shape of the change is:

```python
-    for combination in itertools.product(*(range(c) for c in counts)):
-        selections = dict(zip(names, combination)); evaluated += 1
-        error = evaluate(program, selections, examples, signals, r, tolerance)
+    # depth-first over nodes in program order; values[e][node] is computed once
+    # per prefix, and a node that a signal names is checked as it is computed so
+    # a losing candidate costs one example rather than all of them.
+    def descend(i, values, selections): ...
```

The interface need not change: `SearchResult` keeps its fields, and `evaluated`
becomes the number of *node* evaluations (or the space size, since the walk is
exhaustive by construction).  It is the single change that would move the
discrete backend from "affordable on toy widths" to "the default for perception",
and it composes with D2 and D3 rather than competing with them.

### D5.  `SoftProgram` initialisation noise (the ladder's P2), again

Restated because it bit here too: `torch.manual_seed` does not vary synthesis, so
every gradient number in this track comes from a local copy of `fit` with an
`init_noise` argument, verified bit-identical at zero noise (section 1).  Without
this, none of the "n of m seeds" comparisons between the gradient and discrete
paths in this report could have been made at all.
