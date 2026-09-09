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
max error {{wide_val_err}} over {{wide_space}} programs, applied afterwards at
every position of images up to 3,072 bytes with {{apply_total_wrong}} wrong slot
in {{apply_total_slots}}.**

Five results, in the order they should change what the project does.

1. **The ceiling is not statistical, it is a symmetry.**  Permuting the
   generator's object list leaves the rendered image bit-identical in
   {{perm_rgb}}/{{perm_n}} episodes and changes `object_ids` in
   {{perm_n}}/{{perm_n}}; re-drawing every object's colour leaves `object_ids`
   identical in {{col_ids_same}}/{{col_n}} and changes the image in all of them.
   So the observation determines the labels' *partition* and nothing about their
   *names*.  No context bound can be above baseline for `object_ids` and no
   search over any context can succeed, and the ceiling table in section 2.2
   confirms that at every context measured.  The previous track's per-pixel
   certificate is a special case of this.

2. **The same-object relation is determined by two pixels, and the discriminator
   is collinearity, not equality.**  The renderer paints
   `clip(base_colour * shade)`, so every pixel of one object lies on that
   object's colour ray.  Over the training episodes the integer cross product
   `|r1*g2 - r2*g1|` is at most **{{max_cross_same}}** across
   {{n_same_pairs}} adjacent same-object pairs and at least
   **{{min_cross_diff}}** across {{n_diff_pairs}} adjacent different-object
   pairs -- a clean margin.  Every equality-based context in the table is at
   advantage 0.0000 on the same target; adding collinearity takes it to
   **{{obj_edge_chroma}}** held-out against a {{obj_edge_base}} baseline.

3. **`pack` is the whole difference, and it is legal.**  Pixels are
   `role="byte"` and therefore non-numeric, so equality is all the algebra can
   do to them directly -- which is exactly why the previous track's four
   candidate families were certified empty.  `pack` on a one-field tuple is the
   registered conversion out of the byte carrier, `encode` widens the result,
   and the collinearity test is then ordinary `mul`/`sub`/`abs`/`le`.  No
   operator was added and no type rule bent.

4. **Enumeration settles this rung; the relaxed path's recorded failures were
   invalid relaxations, and one real boundary survives the correction.**  The gradient arm is {{direct_grad}} on the 6,144-program space
   and {{wide_grad}} on the 393,216-program one -- and **{{wide_grad_held}}
   exact on held-out episodes**, so its one training success does not
   generalise -- where enumeration exhausts both and returns a program with
   held-out max error {{wide_val_err}}.  But **those gradient numbers were
   measured on a surrogate that is exactly 0.0** at the operating distance:
   `relaxed` computes `le` as `sigmoid(d/tau)` at `tau = 1`, which underflows in
   float32 at a gap of {{le_zero}}, and this rung's gaps have median
   {{gap_median}}.  Re-run with a live surrogate the arms are
   {{sfix_operand}} and {{sfix_carrier}}, and the reason is now a *declared*
   boundary rather than an accident: `shifted`, the neighbour offset, is
   `grad = None` under every temperature policy because `pack` declares
   `gradient="none"`, while `thr` goes from `None` to {{unfreeze_thr}} to
   {{sfix_thr}} as the two accidents are removed.  Section 4 separates the three
   mechanisms.

5. **A certificate about a family is not a certificate about a target, and the
   candidate pool is part of the family.**  The coarse eight-value threshold
   pool omits the entire separating interval; the search correctly exhausts it,
   reports {{direct_conforming}} conforming programs, and returns one that is
   wrong on exactly two held-out positions.  Widening the pool to 512 values
   fixes it: **{{wide_val}}** programs are conforming on training and exact at
   every validation position, held-out max error **{{wide_val_err}}**.  The two
   wrong positions are a foreground pixel whose shaded colour lies on the
   background's ray, and `residual.py` names them.

---

## 1. Protocol

Same as the previous track, and stated again because two things changed.

* Program inputs are filled **only** from `StepRecord.actor_view().observations`
  (`pixels`, the raw byte tuple).  `latent_states` and `probes` build `targets`
  and nothing else.  Probes are supervision, never model inputs.
* Episodes come from disjoint seed ranges: {{train_seeds}} training episodes
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
* **The host is shared and loaded.**  Every wall clock is an upper bound and two
  of the long runs overlapped deliberately; programs evaluated and node
  evaluations are the load-independent measures and are reported beside every
  time.

---

## 2. The information bounds, computed before any search

### 2.1 Two structural certificates

These are statements about the renderer, not estimates from samples.

| certificate | measurement |
|---|---|
| permute the generator's object list, re-render | rgb bit-identical in **{{perm_rgb}}/{{perm_n}}** episodes, depth identical in **{{perm_depth}}/{{perm_n}}**, `object_ids` identical in **{{perm_ids_same}}/{{perm_n}}**, `object_ids` identical *after remapping through the permutation* in **{{perm_remap}}/{{perm_n}}**; {{perm_changed}} of the {{perm_fg}} foreground pixels across those {{perm_n}} episodes change label -- all of them |
| re-draw every object's colour, re-render | `object_ids` identical in **{{col_ids_same}}/{{col_n}}**, rgb identical in **{{col_rgb_same}}/{{col_n}}**, {{col_changed}} pixels changed across the {{col_n}} episodes |

Read together they say: the observation is invariant to the transformation that
changes the labels, and the labels are invariant to the transformation that
changes the observation.  `object_ids` is therefore **not a function of the
image**, so it is not a function of any context inside the image either --
a pixel, a 3x3 window, a global aggregate, or all 6,912 bytes.  A search over
any of them is a search for a function that does not exist.

The supporting statistic: of {{reuse_distinct}} distinct foreground colours over
{{train_seeds}} training and {{held_seeds}} held-out episodes,
**{{reuse_multi}}** occur in more than one episode (fraction
{{reuse_frac}}).  A colour essentially never recurs, which is why every
lookup-table ceiling below memorises and transfers nothing.

The obstruction inside a single image, restated from the previous track and
re-measured here at R=8: mean **{{mean_rgb_per_object}}** distinct RGB values
per object, max {{max_rgb_per_object}}, only {{single_frac}} of object instances
single-coloured.

The consequence for equality is worth spelling out, because it is why every
equality-based context in section 2.3 is at advantage exactly zero.  Among
adjacent **different-object** pairs, {{adj_diff_same_colour}} have the same
colour -- so equal colours *do* imply the same object.  But among adjacent
**same-object** pairs, **{{adj_same_diff_colour}}** have different colours,
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

{{bounds_ids}}

**Every raw-byte context is at advantage 0.0000, and the reason is visible in
the recurrence column**: a wider window makes the key *rarer*, not more
informative -- recurrence falls from {{recur_pixel}} at one pixel to
{{recur_3x3}} at a 3x3 window.  Enlarging the context strictly *lowers* what a fitted predictor
transfers, which is the opposite of what "more context" is supposed to buy.

Restricting to foreground pixels removes the free background component and makes
the same point without the baseline doing the work:

{{bounds_fg_restricted}}

Every advantage is zero or negative.  This is section 2.1's certificate showing
up as a measurement.

### 2.3 The same-object relation, and the one context that clears the baseline

`same_right(i)` is `object_ids[i] == object_ids[i+1]` (baseline
{{same_right_base}}), and `obj_edge_fg` is its restriction to pairs where
**both** pixels are foreground (baseline {{obj_edge_base}}) -- the genuinely new
part of segmentation, with the foreground/background boundary of rung 3.5
removed.

{{bounds_same}}

Read the `obj_edge_fg` rows first.  **Every equality-based context is at
advantage exactly 0.0000** -- one pixel, a pixel pair, a 4-neighbourhood, a 3x3
window, the equality patterns, and the equality patterns with a background test.
The gains that the raw `same_right` rows show for `bg_pair_right` and
`eqbg_cross4` are the foreground/background part, which rung 3.5 already had.

The two `chroma` rows are the exception, and they are the result: adding one
colour-invariant bit -- do these two pixels lie on a common ray through the
origin -- takes `obj_edge_fg` from advantage 0.0000 to **{{obj_edge_chroma}} held-out**
against a {{obj_edge_base}} baseline, with key recurrence 1.000.

The controls behave as the previous track reports -- `fg` is the rung-3 target
and is exact from one pixel, `fg_edge` is rung 3.5 and needs the pair:

{{bounds_control}}


### 2.4 The bound for the family the algebra can actually express

`same_chroma` above uses floating ratios, which `role="byte"` pixels cannot
produce.  What the algebra *can* produce, once `pack` strips the byte role, is
the integer cross product.  Exact equality is too strict because the renderer
truncates `base * shade` to uint8: over the training episodes the exact cross
product vanishes for only **{{cross_zero_same}}** of same-object pairs (and
{{cross_zero_diff}} of different-object pairs).  The expressible predicate is
therefore `|r1*g2 - r2*g1| <= T` with `T` a searched constant, and it separates
the classes with a margin:

* max cross product over adjacent **same-object** pairs: **{{max_cross_same}}**
* min cross product over adjacent **different-object** pairs: **{{min_cross_diff}}**

so every `T` in [{{max_cross_same}}, {{min_cross_diff}}) is exact on the
training episodes.  Swept over `T`, on both offsets:

{{chroma_table}}

Majority baseline {{chroma_base_right}} (right) and {{chroma_base_down}} (down).
**This is the number that authorised the search in section 3**, and it is the
only row in the whole bounds exercise that clears baseline on the target that
matters.

### 2.5 The generator's other probes, for completeness

`geometry` also probes `depth` and `normals`.  Restricted to foreground pixels,
so the background class cannot do the work:

{{bounds_other_probes}}

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
{+1 pixel, +2 pixels, +1 row}.  {{train_records}} training records,
{{val_records}} validation, {{held_records}} held-out
({{positive_fraction}} positive).

### 3.1 Direct: the collinearity module, coarse threshold pool

| arm | space | evaluated | exhausted | conforming on train | + exact at every validation position | unique | enum s | held-out max error | held-out accuracy | gradient |
|---|---|---|---|---|---|---|---|---|---|---|
| direct (8-value pool) | {{direct_space}} | {{direct_evaluated}} | {{direct_exhausted}} | {{direct_conforming}} | {{direct_val}} | no | {{direct_seconds}} | {{direct_lex_err}} | {{direct_val_acc}} | {{direct_grad}} (dead surrogate; see 4.4) |

{{direct_conforming}} of {{direct_space}} programs conform, and **all
{{direct_val}} of them are also exact at every position of the validation
split**, so the validation tie-break does not discriminate here -- unlike rung
3, where it was the difference between right and wrong.  The conforming set is
{{direct_conforming}}/{{direct_space}} against rung 3's 2,464/32,000, so this
probe identifies the program roughly 40x more tightly than the foreground probe
does.  A uniform random program from the same space conforms
{{direct_density}} of the time.

Both selection rules return held-out accuracy {{direct_val_acc}} with a max
error of {{direct_lex_err}}: **two positions out of {{held_records}} are wrong**,
and section 3.2 shows that is the candidate pool, not the search.

### 3.2 The residual, and why it is the pool rather than the data

{{residual_table}}

Every split has a threshold interval on which the predicate is *exact*, and the
intervals differ: {{interval_train}} on training, {{interval_validation}} on
validation, {{interval_test}} on test, intersecting in {{interval_all}}.  The
coarse pool `(0, 16, 48, 96, 192, 384, 1024, 4096)` contains **no value in that
intersection**, so no member of the coarse space can be exact on all three
splits.  The search exhausted its space and returned the best member of it.  The
two errors are all of one kind:

{{residual_kinds}}

and the offending pair is a foreground pixel whose shaded colour happens to lie
on the background's ray:

    {{residual_example}}

**And the search improves on the analyst.**  The bound above is derived for
`max(cross products) <= T`, the conjunction of all three collinearity tests,
because that is the textbook form of the invariant.  The searched space contains
every Boolean combination of the three, and the program the wide arm returns is
`(rg or gb) and rb` -- `m1 = truth_1` is NOR and `same = truth_2` is
`not m1 and rb`.  It is not the shape the bound was computed for and it is
strictly more robust:

{{rule_table}}

The conjunction is exact on all three splits only for {{rule_and_width}}
threshold values ({{rule_and_all}}); the searched disjunction-then-conjunction
for {{rule_found_width}} ({{rule_found_all}}).  Neither interval contains a
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
| `enumerate_fit`, capped | {{wide_space}} | 40,000 | no | -- | -- | -- | -- | rate {{wide_enum_rate}} prog/s, projects to {{wide_enum_proj}} s |
| `enumerate_fit`, stop-at-first | {{wide_space}} | {{wide_first_evaluated}} | no | -- | -- | -- | -- | {{wide_first_s}} |
| prefix-reusing exhaustive walk | {{wide_space}} | {{wide_inc_nodes}} node evals | yes | {{wide_conforming}} | {{wide_val}} | {{wide_val_err}} | {{wide_val_acc}} | {{wide_inc_seconds}} |
| gradient descent (init_noise 0.5, dead surrogate -- see 4.4) | {{wide_space}} | -- | no | {{wide_grad}} conforming on train | -- | -- | {{wide_grad_held}} exact on held-out | {{wide_grad_s}} s median |

**{{wide_val}} programs conform on training and are exact at every validation
position, and their held-out max error is {{wide_val_err}}.**
{{wide_held_exact}} of the {{wide_conforming}} training-conforming programs are
also exact on the held-out split.  The thresholds among the survivors are
{{wide_thresholds}} -- the separating interval, recovered by search.

Two things to note about cost.  The prefix-reusing walk from the previous
track's `incremental.py` exhausts {{wide_space}} programs in
{{wide_inc_seconds}} s ({{wide_inc_nodes}} node evaluations) where
`enumerate_fit`'s own loop projects to {{wide_enum_proj}} s -- the same
conclusion that track reached, on a space 12x larger.  And stop-at-first needed
{{wide_first_evaluated}} evaluations here rather than the 21 it needed at rung 3,
because a wide constant pool is dense in *near*-solutions and sparse in exact
ones.

### 3.4 Staged on the frozen foreground module

The previous track's rung 3.5 shape, applied to this target.  Stage 1 is the
rung-3 foreground module searched exhaustively over its {{stage1_space}}
programs and supervised at every position: {{stage1_conforming}} conform, and
the one returned is accurate {{stage1_acc_train}} at every position of every
training episode and {{stage1_acc_held}} on held-out -- the diagnostic that
track showed a staged search stands or falls on.  Stage 2 freezes it, registers
it as an operator, calls it at both positions, and searches how the foreground
agreement combines with the collinearity predicate.

| arm | space | conforming | + validation exact | enum s | held-out accuracy | gradient |
|---|---|---|---|---|---|---|
| staged (frozen fg module + collinearity) | {{staged_space}} | {{staged_conforming}} | {{staged_val}} | {{staged_seconds}} | {{staged_val_acc}} | {{staged_grad}} (dead surrogate) |

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
episodes, with T = {{apply_T}}:

{{apply_table}}

**{{apply_total_wrong}} wrong slot in {{apply_total_slots}}**, three caller
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
| with empty parameters (the old behaviour) | {{cand_was}} |
| now | {{cand_now}} |

The operators that appear are {{cand_new_ops}}.  Searched rather than written,
the three caller nodes have {{disc_per_node}} candidates -- a space of
**{{disc_space}} programs** -- and `enumerate_fit` returns
solved={{disc_solved}}, exhausted={{disc_exhausted}}, **unique={{disc_unique}}**
in {{disc_evaluated}} evaluations and {{disc_seconds}} s, supervised only by the
dense probe over the whole image.  The discovered wiring produces bit-identical
output to `tcn.scaffold.positional_scaffold`'s {{disc_nodes}} nodes
({{disc_agrees}}).

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
| rung 3 foreground (control, no `pack`) | {{rung3_grads}} |
| rung 4 collinearity, as built | {{unfreeze_built}} |
| rung 4 collinearity, deterministic nodes unfrozen | {{unfreeze_free}} |
| rung 4, wide pool, as built | {{wide_grads}} |
| rung 4, comparison nodes unfrozen **and** the `le` surrogate scaled | {{sfix_grads}} |

### 4.2 The shipped `le` surrogate is exactly 0.0 at this rung's operating distance

`relaxed` computes `lt`/`le`/`gt`/`ge` as `torch.sigmoid(d/temperature)` with
`temperature = 1`.  In float32 both the value and its derivative are **exactly
0.0 at `|d| >= {{le_zero}}`**:

{{le_table}}

And the operands here are products of two bytes.  Measured at the three `le`
nodes over the training records, with the threshold at the middle of the coarse
pool:

{{gap_table}}

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
| `thr` (the threshold) | `SoftProgram` treats every node with `selected is not None` as frozen and detaches it, and `Builder`/`tcn.scaffold` set `selected = 0` on every *deterministic* node | **implementation** | no -- unfreezing makes it reachable at {{unfreeze_thr}} |
| `thr`, again | the `le` surrogate underflows | **numerical** | no -- scaling takes it to {{sfix_thr}} |

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
a 192-byte observation that puts only **{{addr_true}}** of its weight on the true
address and **{{addr_nb}}** on each immediate neighbour, so the relaxed forward
pass reads a blur of about five bytes rather than a pixel.  That is why section
4.4 unfreezes only the three comparison nodes: it restores the gradient to `thr`
while leaving the addressing exact.

### 4.4 The corrected gradient arms

Two temperature policies, both leaving `tcn/` untouched and replacing `relaxed`
at runtime in one process: the coordinator's **carrier** rule
(`tau = 2^bits` of the compared type) and this track's **operand** rule
(`tau = mean |operand|` over the batch).

| arm | surrogate at the median gap | `thr` gradient | conforming on train | exact on held-out |
|---|---|---|---|---|
| as shipped (section 3) | **0.0** | `None` | {{direct_grad}} | 0/4 |
| deterministic nodes unfrozen, shipped surrogate | 0.0 | {{unfreeze_thr}} | {{unfreeze_grad}} | 0/4 |
| comparison nodes unfrozen, `tau = mean operand` | alive | {{sfix_thr}} | {{sfix_operand}} | {{sfix_operand_held}} |
| comparison nodes unfrozen, `tau = 2^bits` | alive | {{sfix_carrier_thr}} | {{sfix_carrier}} | {{sfix_carrier_held}} |
| *every* deterministic node unfrozen, `tau = mean operand` | alive | {{full_operand_thr}} | {{full_operand}} | {{full_operand_held}} |
| *every* deterministic node unfrozen, `tau = 2^bits` | alive | {{full_carrier_thr}} | {{full_carrier}} | {{full_carrier_held}} |

{{sfix_conclusion}}

Enumeration, on the same space and the same records, exhausts {{direct_space}}
programs in {{direct_seconds}} s and {{wide_space}} in {{wide_inc_seconds}} s
with a prefix-reusing walk, returning a program with held-out max error
{{wide_val_err}} and a uniqueness report.

### 4.5 What this says about the method boundary

The previous track drew the boundary as "constants at a fixed input: relaxation;
behind `gradient="none"`: enumeration".  Two of the three failures here were not
on that boundary at all -- they were an implementation detail and a numerical
underflow -- and only the offset is genuinely behind a declared boundary.  The
honest statement after this track is:

* a choice behind a **declared** `gradient="none"` operator -- here the offset,
  behind `pack` -- is outside the relaxed backend, and no surrogate fixes it;
* a choice at a **surrogate comparison** is only as good as that surrogate's
  dynamic range at the *operating distance*, which must be measured and
  reported, not assumed;
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

{{world_table}}

It is not.  The permutation certificate holds there too -- the two episodes
where reversing the list leaves the labels unchanged are ones where too few
distinct objects are in view for the reversal to move any visible index -- `visible_ids` is at
advantage 0.0000 with foreground key recurrence 0.000, and the one fixed-colour
object -- `arm_agent_0`, index 0 of {{world_ids}} -- **never appears in the
agent's own view**: the held-out label histogram is {{world_hist}}, with no
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
| **4d** | `same(i, i+k)`, collinearity | two pixels + `pack` | held-out **1.0000**, margin [{{max_cross_same}}, {{min_cross_diff}}) | {{direct_space}} exhausted, {{direct_conforming}} conform, held-out {{direct_val_acc}} | {{direct_grad}} | **works** |
| **4e** | the same, threshold pool 0..511 | two pixels + `pack` | as above | {{wide_space}} exhausted, {{wide_conforming}} conform, {{wide_val}} validation-exact, held-out max error **{{wide_val_err}}** | {{wide_grad}} | **highest rung reached** |
| 5 | `depth` threshold | pixel .. 3x3 | advantage 0.0000 everywhere | -- | -- | unreachable (previous track, reproduced) |

---

## 7. The precise verdict

**What context makes object identity learnable:** two pixels, one relative
offset apart, compared for *collinearity* rather than equality -- which needs
`pack` to lift them out of `role="byte"` and costs one searched integer
threshold.  At that context the same-object relation is exact: held-out max
error {{wide_val_err}}, and {{wide_val}} of {{wide_space}} programs conform on
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
  {{rule_found_width}} values wide against the conjunction's
  {{rule_and_width}}.  Every "advantage 0.0000" row in section 2 is a statement
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
* **A shared, loaded host,** with two long runs overlapping deliberately.  Wall
  clocks are upper bounds and are not comparable across hours; programs and node
  evaluations are reported beside every one.
* **Supervision budgets are small**: {{train_records}} records at R=8.
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
{{unfreeze_space}} programs either way -- makes it reachable, at |grad| = {{unfreeze_thr}}.

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
`|d| >= {{le_zero}}`**, and this rung's operating gaps have median
{{gap_median}} with {{gap_worst_fraction}} of records at or past the underflow
point (section 4.2).  Any comparison on operands wider than about two decimal
digits -- a product of two bytes, a squared distance, a pixel count -- is
invisible to the optimiser, and every gradient arm over such a comparison is a
measurement of a dead relaxation rather than of learnability.

Two scalings were measured (section 4.4): `tau = 2^bits` of the compared carrier,
which is the rule that fixed the `eq` benchmark, and `tau = mean|operand|` over
the batch.  Both take the threshold logit's gradient from {{unfreeze_thr}} to
{{sfix_thr}}.

```python
     if n in COMPARE:
-        if n=="eq": return torch.exp(-((a-b)**2).sum(-1,keepdim=True)/temperature)
         d=b-a if n in {"lt","le"} else a-b
-        return torch.sigmoid(d/temperature)
+        # A fixed temperature makes a comparison blind past |a-b| ~ 11 for `eq`
+        # and |d| ~ 89 for the orderings, in float32.  Scale by the carrier so
+        # the surrogate is informative across the representable range; the exact
+        # semantics are unchanged.
+        tau=temperature*float(2**op.inputs[0].bits)
+        if n=="eq": return torch.exp(-((a-b)**2).sum(-1,keepdim=True)/tau)
+        return torch.sigmoid(d/tau)
```

Whatever the scaling, the requirement is that a gradient arm reports the
surrogate's value at its operating distance; a `0/n` recorded next to a
surrogate of 0.0 says nothing.

### E3.  `index`'s address relaxation is not sharp enough to read a pixel

`relaxed` computes `index` as `softmax(-(address - arange(n))^2 / temperature)`
at `temperature = 1`.  Over a 192-byte observation that puts only
**{{addr_true}}** of its weight on the true address and {{addr_nb}} on each
immediate neighbour, so a relaxed read returns a blur of about five bytes.  A
scaffold whose deterministic nodes are relaxed (which E1 would make the default)
therefore reads a blurred pixel, which is a second invalid relaxation in the
same place.  Sharpening the temperature, or using a straight-through estimator
on the argmax address as `relaxed` already does for `idiv`, would fix it; this
track measured the sharpness rather than choosing between them.

### Re-statement of the previous track's D2/D3/D6, all still unmerged

This track needed all three again and implemented them locally: it reports the
conforming count itself (D2), filters by a validation split and reports both
counts (D3), and used `research/discrete-perception/incremental.py` for the
393,216-program sweep (D6), which `enumerate_fit`'s own loop projects at
{{wide_enum_proj}} s.
