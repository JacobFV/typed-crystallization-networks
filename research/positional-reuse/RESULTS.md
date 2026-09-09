# Positional reuse — 2026-09-08

**Verdict: positional reuse is expressible today. No operator addition is
justified, and none is proposed.** One crystallized module can be applied at
every position of an arbitrarily wide tuple using three caller nodes, built
entirely from `insert`, `pair` and `map`, which are already in
`tcn/operators.py`. The caller does not grow with the input width or the number
of positions.

The blocker as stated — "`map` and `filter` require `set[T]` inputs and all
perceptual observations are `tuple[...]`, so there is no way to form a set of
windows" — is real as far as it goes: there is indeed no operator that chunks a
tuple into a set of windows. But forming a set *of windows* is not what
positional reuse needs. It needs a set of **positions**, which is an ordinary
constant, and one bridge from the tuple into set-land, which `insert` supplies
in one node. `pair` then does the rest.

| headline | measurement |
|---|---|
| caller size | **3 nodes**, independent of positions and width |
| description, 128 positions | **17 structural symbols** vs 397 per-position, 1,025 inlined |
| description crossover | **2 positions** against per-position calls |
| execution cost | `N x body + 2` — charged per use, 0.93x the per-position program |
| end to end on `geometry` | shared module learned from the dense probe, **0.0 max error** over 640 held-out pixels |
| widest input that works | **27,648 values over 9,216 positions**; `focus_pixels` scale (18,723) applies in 621 s |
| added to the core | no operator, no type; one enumerator generalization and one scaffold helper |

Everything below was measured on this worktree with the repository `.venv`.

---

## 1. The pattern

```text
held    = insert(empty_set_constant, observation)   # set[Wide],  capacity 1
records = pair(position_constant_set, held)         # set[(Index, Wide)]
mapped  = map(records; module = m)                  # set[(Index, Out)]
```

`pair` is the cartesian product. Pairing an N-element constant set of positions
with a *singleton* set holding the whole observation yields exactly one record
per position, each carrying the entire observation. `map` then applies one
crystallized module to every record; the module reads its own position out of
the record with `project` and indexes the observation there with `index`, adding
constant offsets for a window wider than one element.

Three caller nodes and two constants, for any N and any width.

The module returns `(position, result)` rather than `result` alone. This is
required, not decorative: a set is duplicate-free, so untagged results would
collapse wherever two positions computed the same value, and the position would
be unrecoverable. Tagged, the output is a *relation of indexed values*, which
`ARCHITECTURE.md` section 1 already names as the way sequences are represented
("Sequences are indexed values plus length").

Reference implementation: `poc_pair_map.py` (standalone, no repository helpers) and
`tcn/scaffold.py:positional_scaffold`.

### What the composition does *not* give you

Three limits, all measured, none of them semantic gaps:

1. **No set → tuple.** There is no operator taking a set back to an ordered
   tuple, so a shared positional computation cannot produce a `tuple[N]`-typed
   output with O(1) nodes. Reading position *i* back out costs a `filter` plus a
   reduction — O(1) nodes but O(N) execution — so recovering all N as a tuple is
   O(N) nodes and O(N²) execution. For supervision this does not bite, because
   the indexed set is a lossless sequence representation and can be compared
   directly against a re-encoded probe. It bites when the per-position results
   must feed a tuple-consuming operator such as `fft`.
2. **The whole pattern sits behind a gradient boundary.** `insert`, `pair`,
   `map` and `filter` all declare `gradient="none"` in `Registry.resolve`, so
   `SoftProgram` routes them through `exact_tensor` and no gradient reaches
   anything upstream of the map. The shared module therefore cannot be trained
   *through* its own application. This is declared behaviour, not a bug —
   section 2 says "a candidate without a valid relaxation remains an exact
   operator at an explicit gradient boundary" — and the workaround is natural:
   train the module on single-position records, where **every position of every
   episode is one i.i.d. training example**. That is what section 3 below does,
   and it is arguably the better formulation, because the dense probe is used
   unaggregated. What is genuinely unreachable by gradient is a module whose only
   supervision arrives *after* aggregation; that case needs discrete search.
3. **`pair` replicates the observation N times.** The record set holds N·W
   values. This is the cost wall, quantified in section 5.

### The sharpest objection, and the answer to it

*Is the constant set of positions smuggled-in domain knowledge — a window
handed to the agent, which section 2 forbids?*

No, and the distinction is worth stating precisely because it is the one thing
that could invalidate the result. The constant is a set of plain integers with a
declared `int[n]` encoding. It asserts only that the program is allowed to look
at those positions — the same kind of statement `arithmetic_scaffold` makes when
it declares how many hidden terms exist, and the same kind section 4 authorises
("The supplied coarse structure defines admissible regions, predecessor pools,
capacities, and boundary placements"). The agent still has to learn to consume
it: nothing tells the module that a position indexes an observation, that
offsets 0/1/2 are colour channels, or that adjacent indices are adjacent pixels.
The module has to select `index` against that position, choose its own offsets,
and choose what to compute — and in section 3 below it does, from a 32,000
program space. What is *not* present anywhere is a window, a patch, a channel
split, a neighbourhood, or a stride as an operator or as a preprocessed input.

Two honest caveats. First, the position set is structural and cannot be learned:
constants are trainable only when numeric and scalar, so which positions exist is
declared, never discovered. Second, the observation reaches the module whole, so
what the module reads is entirely its own choice — a stronger position than a
windowing operator would give it, not a weaker one.

### Compositions that were tried and rejected

| Attempt | Outcome |
|---|---|
| N `insert` nodes building a set of window tuples, then one `map` | Works, shares the transformation, but costs 2N caller nodes — the thing to avoid. |
| `map` over a constant set of positions alone | Illegal: a module's inputs must equal the set element type, and a module cannot close over an outer port. The observation must travel inside the element. |
| Untagged module output | Legal but wrong: the result set collapses duplicates and loses position. |
| `join` instead of `pair` | Needs both operands to be sets of tuples, so it does not remove the need for the `insert` bridge; strictly more machinery for the same result. |
| `index` with a learned index to fake sharing | One node per position either way; shares nothing the module route does not already share. |

---

## 2. Cost of sharing versus doing it position by position

`measure_cost.py`, three programs computing the same window function
(`mean` of a 3-wide window) over the same input:

* **shared** — `hold / pair / map`, one module definition.
* **per_position** — the *same* module, called once per position: `2N + 1` nodes.
* **inlined** — no module at all, the body written out at every position.

`structural_symbols` counts nodes plus constants with each module definition
counted once. It is reported alongside `description_bits` because the latter
measures serialized JSON length, which for a wide input is dominated by the
observation type repeated in every node signature — track 6 of `FINDINGS.md`
already flagged that metric.

| width | N | caller nodes<br>shared / per-pos / inlined | structural symbols<br>shared / per-pos / inlined | description bits<br>shared / per-pos / inlined | execution cost<br>shared / per-pos / inlined |
|---:|---:|---|---|---|---|
| 6 | 4 | 3 / 9 / 21 | **17** / 25 / 33 | 179,776 / 277,568 / 217,368 | 42 / 45 / 21 |
| 10 | 8 | 3 / 17 / 41 | **17** / 37 / 65 | 243,808 / 588,384 / 514,488 | 82 / 89 / 41 |
| 18 | 16 | 3 / 33 / 81 | **17** / 61 / 129 | 371,976 / 1,551,344 / 1,365,408 | 162 / 177 / 81 |
| 34 | 32 | 3 / 65 / 161 | **17** / 109 / 257 | 628,232 / 4,841,456 / 4,090,784 | 322 / 353 / 161 |
| 66 | 64 | 3 / 129 / 321 | **17** / 205 / 513 | 1,140,744 / 16,877,552 / 13,633,440 | 642 / 705 / 321 |
| 130 | 128 | 3 / 257 / 641 | **17** / 397 / 1,025 | 2,166,048 / 62,774,800 / 49,090,648 | 1,282 / 1,409 / 641 |

Read the structural-symbol column first: **the shared program is 17 symbols at
every size**, while both alternatives grow linearly in N. At N = 128 that is a
23x reduction against per-position calls and 60x against inlining, and the ratio
keeps growing. On the repository's own `description_bits`, sharing wins 29x at
N = 128 — a smaller factor only because the shared program's JSON still repeats
the wide observation type inside the module body.

Execution cost tells the opposite and expected story. `map` charges
`capacity x module_cost`, so **sharing does not make calls cheaper**: shared is
`N x body + 2`, per-position is `N x (body + 1) + 1`, and inlining is cheapest of
all because it skips the record bookkeeping. That is exactly what
`ARCHITECTURE.md` section 4 requires ("Sharing counts a module definition once
plus its call sites, with execution cost charged per use"), and it is now
honoured in both directions: shared and per-position are within 10% of each
other, so the description-size saving is not bought with execution.

Measured wall time follows the cost proxy: shared and per-position are within
noise of each other, inlining is roughly 4x faster, at every size.

The three columns fit exactly:

```text
shared        17          (3 caller nodes + 2 constants + a 12-symbol module body)
per_position  3N + 13     (2N + 1 nodes, N constants, the same body)
inlined       8N + 1      (5 nodes and 3 constants per position, no body to share)
```

so the **description crossover against per-position calls is N = 2**, and against
inlining N = 1. That matches the 2-call-site crossover the recent single-output
module fix produced for a 3-gate body, and it now holds for *any* number of
positions rather than being eaten back as N grows.

---

## 3. End-to-end demonstration on `generators/geometry`

`demo_geometry.py`. Downscaled deliberately: `resolution=8` (64 pixels, a 192
byte observation), six objects, one `camera` action to bring the objects close
enough that the foreground fraction is not degenerate. The camera step is an
ordinary generator action, so the episode stays replayable and nothing is
preprocessed.

**Supervision.** The agent sees only `pixels`. The dense probe `object_ids`
supplies a per-pixel label; the predicate used is `object_ids >= 0`.
`inspect_geometry.py` and `inspect_geometry2.py` verify that this predicate is
*exactly* determined by the pixel value: **0 disagreements** with the renderer's
background colour over 768 pixels at the default camera and over every one of 18
resolution/object/camera configurations swept (512 to 2,048 pixels each). So an
exact program exists, and a non-conforming result cannot be excused as label
noise. The chosen configuration is 51.8% foreground, so a constant predicate
scores nothing.

**Type discipline.** `image_value` gives pixel bytes the semantic role `"byte"`,
which makes `Type.numeric` false, so the type system already forbids arithmetic
on a pixel. The learned module uses `eq` only. The byte field is reached with an
ordinary `project` node inside the graph; nothing is unpacked outside it.

**Stage A — learn the shared sub-program.** A module scaffold
`(position, pixel bytes) -> (position, foreground)` chooses, per colour channel,
which of five bytes to compare against, and chooses both combining gates from all
16 two-input truth tables: 5³ x 16² = **32,000 discrete programs**. Training
examples are (episode, position) pairs — every pixel of every training episode
is one example, which is the dense probe doing the work.

| stage A, 512 training pixels (51.8% foreground) | result |
|---|---|
| discrete space | 32,000 programs |
| exhaustive search, all 32,000 evaluated | 1,952 conform exactly, 668 s |
| `synthesis.fit` (400 steps, freeze on) | `exact_max_error` **0.0**, `fully_frozen` true, 1,174 s |
| gradient and enumeration pick the same program | no |

Both methods reach an exactly conforming program and the gradient path
crystallizes it fully. They pick *different* programs, and the reason is a
property of the task worth stating plainly rather than hiding: **1,952 of the
32,000 programs fit all 512 training pixels exactly, and 1,584 of those still fit
all 640 held-out pixels.** The renderer's own three-channel test is among the
survivors, and so is the gradient's pick, but the supervision does not identify
it — at this scene density a foreground pixel almost never shares any single
background channel value, so `blue != 43` alone already separates the classes.
Enumeration's first solution is exactly that one-comparison program.

That is a limitation of this dense probe on this generator configuration, not of
positional reuse. It is reported because a "we learned the right program" claim
here would be false, while the claim this track actually needs — *a shared
sub-program was learned from dense per-pixel supervision, crystallized, and
applied at every position* — is unaffected, and is checked against the
renderer's own test as a reference module in stage B.

**Stage B — apply it everywhere.** The exported module is frozen, registered, and
applied at all 64 positions by `positional_scaffold`, with **3 caller nodes**.
The result is compared against the whole dense probe of held-out episodes at
once, re-encoded as an indexed set.

| stage B | result |
|---|---|
| caller nodes | **3**, for 64 positions |
| held-out episodes | 10, 640 pixels, 45.3% foreground |
| max error against the dense probe | **0.0** |
| the renderer's own test, mapped the same way | asserted equal to the probe on every held-out episode |

The comparison is against the *whole* dense probe at once, re-encoded as an
indexed set — every pixel of every held-out episode, not a sampled readout. A
constant predicate would score 0.453, not 0.0.

**Stage C — composition after the map.** `filter` (with a one-node module that
projects the flag) followed by `count` reduces the same result set to a
foreground-pixel count. Max error against the true count on all 10 held-out
episodes: **0**. The pattern composes, and an aggregate readout needs no
set-to-tuple conversion.

**Stage D — cost, on the real task.**

| | caller nodes | structural symbols | description bits | execution cost | ms/apply |
|---|---:|---:|---:|---:|---:|
| shared (`hold / pair / map`) | **3** | **25** | 3,656,064 | 834 | 163 |
| per-position (same module, 64 call sites) | 129 | 213 | 53,623,400 | 897 | 63 |

**8.5x fewer structural symbols and 14.7x fewer description bits, at 0.93x the
execution cost.** The per-position program is faster in wall time here (63 ms
against 163 ms) even though its charged cost is higher, because `pair` builds and
revalidates the 64 x 192 record set while the per-position calls build 64 small
records — the representational overhead analysed in section 5, not a difference
in the work done.

### Search discovers the shared module, it is not handed over

`demo_search.py`. Four sub-programs with the same interface are registered: the
renderer's own background test and three plausible wrong ones. The map node's
candidates are produced by `tcn.graph.legal_candidates` over the registry —
nothing about positional reuse is wired in advance — and `enumerate_fit` picks
one against the whole dense probe of three episodes.

| variant | train max error | held-out max error (640 pixels) |
|---|---:|---:|
| `renderer_background` | 0.0 | **0.0** |
| `or_instead_of_and` | 0.0 | **0.0** |
| `wrong_blue_byte` | 1.0 | 1.0 |
| `unnegated` | 1.0 | 1.0 |

Search selects `renderer_background` and reports `unique=False`, which is
correct and worth stating plainly: `or_instead_of_and` computes a different
Boolean function that happens to agree on all 832 pixels seen, because no
foreground pixel in this sample has blue = 43 together with red = 24 or
green = 30. The mechanism claim holds; a uniqueness claim would not, and
`enumerate_fit` says so rather than letting it pass.

---

## 4. What was added to the core

No operator, no type, no relaxation, no loss term. Two changes, both of which
expose compositions of operations that already existed.

### `tcn/graph.py` — `legal_candidates` can enumerate parametric operators

`legal_candidates` resolved every candidate with **empty parameters**, so
`project`, `map`, `filter` and `join` could never be proposed. That is precisely
the family that expresses recursive abstraction, which meant a search could
never discover positional reuse — it could only ever be hand-wired. A new
`operator_parameters(registry, name, types)` derives the legal settings from the
bound input types and the registry (indices for `project`, registered modules for
`map`/`filter`, field pairs for `join`), and `legal_candidates` ranges over them.
A `parameters=` argument lets a caller narrow a wide family without changing the
operator. The existing enumeration budget still applies and now counts
parameterizations, so an unbounded family raises rather than silently truncating.

This implements the spec rather than extending it: section 2 already says an
operator's contract includes its *parameter domains*, and section 3 already says
"full signatures must unify". Parameters are part of the signature.

### `tcn/scaffold.py` — `positional_scaffold`

The `hold / pair / map` construction as a named helper, alongside the existing
`arithmetic_scaffold`, built from universal operators only. It is domain-neutral
by construction: `observation` is any nonempty uniform tuple and positions are
ordinary integers, so it serves a signal, a symbol sequence or a coordinate list
as readily as a raster. Several modules give the node one candidate each, so
*which* shared sub-program runs at every position is a searched or learned
choice; they must agree on their output interface, because a node has one output
type. Illegal configurations (non-tuple observation, empty positions, no module,
width mismatch, a category-role index, disagreeing interfaces) raise.

### Tests

`tests/test_positional_reuse.py`, 9 tests: exact semantics of the pattern; caller
size independent of the position count; agreement between the shared and
per-position programs; the module definition charged once while execution is
charged per use; the mapped module's interface enforced; the indexed record set
being a lossless, alignment-stable sequence representation; parametric
enumeration proposing `project` indices and registered modules and never
proposing an unregistered one; the enumeration budget still bounding parametric
families; and `positional_scaffold` matching the hand-wired pattern, offering a
choice of modules, and refusing illegal configurations.

Full suite: **96 passed, 4 failed**. The four failures are all
`generators/computer`, which shells out to `node --import tsx` and fails with
`ERR_MODULE_NOT_FOUND` because `generators/computer/engine/node_modules` is
gitignored and absent in a fresh worktree. They fail identically before any
change in this branch.

---

## 5. Scaling — how wide an input actually works

`measure_scaling.py` and `measure_wide.py`, on synthetic rasters with the same
byte semantics, walking out to and past the shipped observation widths.
`geometry` ships `pixels` at 3,075 wide with 1,024 pixels; `embodied_world`
ships `focus_pixels` at 18,435.

The scaffold searched here is the same shape as the demonstration's but with four
candidate bytes rather than five, so the space is 4³ x 16² = **16,384 programs**,
and `enumerate_fit` runs with `stop_at_first=True` — a first-solution search over
the whole space, not an exhaustive sweep. It solved at every width.

| resolution | width W | positions N | module search (`enumerate_fit`, first solution) | soft forward, 48 examples | exact apply, all N positions |
|---:|---:|---:|---:|---:|---:|
| 2 | 12 | 4 | 0.01 s | 0.51 s | 0.00 s |
| 4 | 48 | 16 | 0.03 s | 0.01 s | 0.01 s |
| 6 | 108 | 36 | 0.02 s | 0.05 s | 0.03 s |
| 8 | 192 | 64 | 0.05 s | 0.09 s | 0.04 s |
| 12 | 432 | 144 | 0.08 s | 0.12 s | 0.20 s |
| 16 | 768 | 256 | 0.13 s | 0.22 s | 0.59 s |
| 24 | 1,728 | 576 | 0.54 s | 0.93 s | 3.22 s |
| 32 | **3,072** | **1,024** | 1.25 s | 2.90 s | 26.2 s |
| 48 | 6,912 | 2,304 | 1.63 s | 2.13 s | 113 s |
| 64 | 12,288 | 4,096 | 4.77 s | 6.16 s | 232 s |
| 79 | **18,723** | **6,241** | 10.7 s | 16.8 s | 621 s |
| 96 | 27,648 | 9,216 | 10.3 s | 13.4 s | 2,479 s |

Every row is exact: the applied program reproduces the per-position labels at
every one of the N positions. The walk stopped at resolution 96 because the apply
exceeded its 900 s budget, not because anything failed.

Widths 3,072 and 18,723 are marked because they bracket what the generators
actually ship: `geometry`'s `pixels` is 3,075 wide over 1,024 pixels, and
`embodied_world`'s `focus_pixels` is 18,435. **Both are inside the range that
works.**

Three different walls, and they are far apart:

* **Learning the shared sub-program scales essentially for free.** The module
  sees one `(position, observation)` record per example, so its search is O(W)
  per example and *independent of N*. Enumeration over the 16,384 program space
  solves it in **1.25 s at the full geometry resolution**
  (W = 3,072), 10.7 s at the `focus_pixels` width (W = 18,723), and 10.3 s at
  W = 27,648; the relaxed forward pass over 48 examples stays under 17 s
  throughout. This is the part that has to converge, and it converged at every
  width tested — there is no width in this range at which the search degrades.
* **Applying the frozen program at every position is the slow part**, and it is
  slow for an avoidable reason. `profile_apply.py` at W = 768 / N = 256 spends
  2.65 s of 3.61 s (**73%**) inside `Value.of` — that is `types.encode` plus
  `types.validate_raw` — with `types.decode` accounting for most of the rest,
  against a small remainder in the 256 module runs that do the work. `pair`
  attaches the whole observation to every position, and every stage then decodes,
  re-encodes and revalidates all N·W values. The cost is representational
  bookkeeping over a replicated cartesian product, not computation.
* **The relaxed path through the map node is the hard wall.** `map`, `pair` and
  `insert` have no relaxation, so `SoftProgram` materialises the record set as a
  flat tensor of N·(2 + W) values per batch row: 0.49 s at N = 64, 5.2 s at
  N = 256, 19.6 s at N = 576, **71 s at N = 1,024**. That is a soft choice
  *between* shared modules at full resolution, which is only worth paying when
  the choice cannot be settled discretely — and per `FINDINGS.md` section 8 it
  usually can.

**Answer to "largest input width that works": 27,648 values over 9,216
positions, which is past every observation the generators ship.** The practical
recommendation is width **18,723 over 6,241 positions** — the `focus_pixels`
scale — where the shared sub-program is learned in 10.7 s and applied exactly at
every position in 621 s; at the `geometry` scale of 3,072 over 1,024 the same
figures are 1.25 s and 26 s. The limit is wall time on the *apply* path, not
convergence. The search itself does not degrade at any width measured, which is
the part of the question that matters: the wide observation never enters the
learning problem, only one position of it at a time does.

The obvious lever, not pulled here because it is an implementation optimisation
and not the question asked: `Registry.exact` decodes both operands of `pair`,
re-encodes the product, and `map` then decodes and re-encodes every record
again. Constructing these from `Value.raw` instead would remove two of roughly
three full passes over N·W. A windowing operator would attack the same cost from
the other side (O(N·k) instead of O(N·W)) — but the semantics are already
complete without one, so on `AGENTS.md`'s rule ("Operator additions need a
stated *semantic* necessity ... domain-specific shortcuts are not justified by
implementation ease") that is not a justification, and it is not proposed.

---

## 6. Proposed `ARCHITECTURE.md` amendment

**No section 2 operator amendment is proposed.** The task's condition for one —
that positional reuse be inexpressible — is not met, and section 2's existing
sentence ("Connected components, edge detectors, tokenizers, objects, words,
**windows**, and buttons are not initial learned primitives. Richer
transformations are composed or learned.") is confirmed rather than contradicted
by this work: windows are composed, and the composition costs three nodes.

One **clarifying** amendment to section 3 is proposed, because the implementation
silently disagreed with the specification and the disagreement is what made
recursive abstraction unreachable by search. Section 3 currently reads:

> Candidate input bindings are drawn from bounded predecessor pools; full
> signatures must unify, including arity, input and output types,
> representations, shapes, and refinements.

Proposed replacement:

> Candidate input bindings are drawn from bounded predecessor pools; full
> signatures must unify, including arity, input and output types,
> representations, shapes, and refinements. An operator's parameter domain is
> part of that signature, so a node's candidate space ranges over the legal
> parameter settings as well as the bindings, derived from the bound input types
> and the module registry. Fixing parameters at enumeration excludes `project`,
> `map`, `filter` and `join` outright, which would make recursive abstraction
> reachable only by hand-wiring. Parameter families are bounded by the same
> enumeration budget as bindings.

That is a clarification of an existing requirement, not a new capability, and
`tcn/graph.py` now implements it.

Two further recommendations that need no spec change:

* Section 4 says a crystallized module is registered "as one typed candidate
  operator for later synthesis: this is recursive abstraction." It is worth
  recording that the *positional* form of this — one module across many
  positions — is `hold / pair / map`, three nodes, and that its output is a
  relation of indexed values rather than a tuple. `positional_scaffold` is the
  reference.
* `FINDINGS.md` track 5 concluded "recursive abstraction helps: no measurable
  benefit", with the module on the output path in 0 of 20 runs. Two of the three
  causes it identified have since been fixed. The third is visible here: a caller
  that must name every call site can only ever save the *body*, and for a small
  body that saving is marginal. Positional reuse changes the shape of the
  question — the caller stops growing at all — and the measured saving is 23x in
  structural symbols at 128 positions and unbounded in N. That question is worth
  re-running.

---

## 7. Files

| file | what it does |
|---|---|
| `poc_pair_map.py` | Minimal standalone proof that `insert / pair / map` gives positional reuse in 3 nodes. |
| `shared.py` | Construction helpers; `shared_map_program` now delegates to `tcn.scaffold.positional_scaffold`. |
| `measure_cost.py` → `cost.json` | Description size and execution cost, shared vs per-position vs inlined. |
| `inspect_geometry.py`, `inspect_geometry2.py` | Verify the dense probe's per-pixel predicate is exactly determined by the observation, and find a non-degenerate downscaled configuration. |
| `demo_geometry.py` → `demo_geometry.json` | End to end: learn the module, crystallize it, apply it at every pixel, compose `filter`/`count`, compare costs. |
| `demo_search.py` → `demo_search.json` | `legal_candidates` proposes the map candidates and search picks the shared module. |
| `measure_scaling.py` → `scaling.json` | Search, relaxed forward and exact apply against width and position count. |
| `measure_wide.py` → `wide.json` | The same, past the shipped observation widths. |
| `profile_apply.py` | Where the exact apply time actually goes. |
