# A persistent module library, artifact flow, and a three-stage chain

**Track: module-library. Branch `module-library`, not merged.**

The gap: ten demonstrated capabilities, each a separately authored scaffold; no
learned artifact outlives the script that produced it; no registry persists; and
`curricula/system.json`'s two learning stages pass nothing to each other. The
composition machinery is built and measured working. Nothing wires it together.

This track builds the wiring and then measures whether it buys anything.

---

## 0. The answer, first

**Yes, a stage gets easier given a prior stage's module, and the size of the
effect depends entirely on what the module removes from the search.**

| stage | with the prior module | without it | difference |
|---|---|---|---|
| 2 — two-position boundary | 48 programs, **1.52 s**, exhausted, **unique**, held-out max error 0.0 | 4.9152e10 programs, 0 conforming in 20,000 evaluated, **431.7 days projected** | **1.02e9x** the space, **2.4e7x** the time, and the difference between solved and not |
| 3 — region predicate | 60 programs, **7.23 s**, 3 conforming, held-out 1.000 | *stage 1 only:* 960 programs, 32.66 s, also solves | **16x** the space, **4.5x** the time; both succeed |
| 3 — region predicate | 60 programs, 7.23 s | *no library at all:* **inexpressible** | `map` takes a module; with an empty library no program in this algebra aggregates over positions |

Two of those three rows are honest about the limit of the claim. Stage 2 without
stage 1 is not merely slower, it is out of reach: the flat space is 4.9e10
programs and 20,000 of them were swept in 15.2 s with none conforming, which
projects to 431.7 days to exhaust. Stage 3 given only stage 1 still succeeds,
16x more expensive in space and 4.5x in wall clock — a real saving, not a
categorical one, and I report it as such rather than as another billion-fold
headline.

And one cost, measured rather than assumed: **composition propagates the
parent's residual error, and the interface needs exactness, not accuracy.** A
first attempt at this chain failed with a stage-1 module that was 99.74% correct
at every position. One wrong pixel in 384 made stage 2's exactly-derived target
unreachable and the stage found zero conforming programs out of 48. Section 4
has the numbers.

---

## 1. The persistent library — `tcn/library.py`

### Layout

```text
<root>/manifest.json           index: every published version, in publication order
<root>/modules/<digest>.json   the pruned frozen program, content-addressed
<root>/fixtures/<digest>.json  recorded conformance cases for that program
```

All plain JSON: an artifact is readable without this package, and a module file
is named by the same content address `Registry.register_module` computes, so
storing and registering are one identity.

### What a manifest entry records

`name` (logical, e.g. `perception.edge`), `version`, `digest`, `operator`
(`module:<digest>`), the typed `inputs` and `outputs`, `requires` (the operator
names of every module it calls), `source` (the `source_fingerprint()` it was
built against), `nodes`, `execution_cost`, `description_bits`, free-form
`provenance`, the number of recorded fixture cases, `stale` marks, and
`revalidated` stamps.

### How it satisfies ARCHITECTURE section 4

- **A definition is counted once.** Storage is content-addressed. Two logical
  names that crystallize to the same program share one file and one operator;
  publishing the same program twice under one name is a no-op that returns the
  existing version rather than manufacturing one. Measured in
  `tests/test_module_library.py::test_content_addressing_shares_one_definition`.
- **Execution is charged per use.** A loaded module resolves to the same
  `Operator` with the same `cost` as a freshly registered one. Section 6.
- **Description cost is transitive.** An entry records its callees and `load`
  registers dependencies first, so `Program.description_bits(registry)` charges
  a caller its callees' internals. Measured exactly, section 6.
- **Relearning creates a new version and revalidates dependents.** Publishing a
  *different* program under an existing name allocates the next version and
  marks every entry that calls the superseded digest `stale`. `load` refuses a
  stale entry. `Library.revalidate` clears the mark by re-executing that
  dependent's own recorded fixture.
- **What is registered is the program, not the scaffold.** `register_module`
  already prunes; the library stores the pruned, registered form, so
  re-registering a stored module is a fixed point and the recorded digest is the
  one a later run computes. A registration that disagrees with the manifest is
  an error, not a warning.

### The source-revision policy, stated

`source_fingerprint()` hashes all of `tcn/` and `generators/`, so *any* edit to
either invalidates every stored module's provenance — the `external-environments`
track already measured that adding one generator invalidates artifacts from all
the others. A stored module records the fingerprint it was built against and the
loader refuses to guess:

| policy | behaviour |
|---|---|
| `strict` (default) | raises `SourceRevisionMismatch` naming both fingerprints. A module never crosses a code revision silently. |
| `revalidate` | loads **only after** the module's recorded conformance fixture re-executes and agrees exactly under the current source. One disagreement raises `FixtureMismatch`; a module with no recorded fixture raises too, because its behaviour cannot be checked. A success is written back to the manifest as a `revalidated` stamp, so the crossing is recorded rather than forgotten. |

There is deliberately no third policy that ignores the mismatch. The fixture is
what makes `revalidate` a check rather than a bypass, which is why `publish`
takes example inputs and records the outputs the module produced at publication
time. Four tests cover this: strict refusal, revalidation plus its stamp,
revalidation refused when a fixture no longer reproduces, and revalidation
refused outright when no fixture exists.

The practical consequence is not hypothetical. This branch's own core changes
moved the fingerprint, so the shipped library was published against the final
tree; any later edit to `tcn/` or `generators/` will make `tcn library verify`
report `source_current: false` while still passing, because verify goes through
the fixture. That is the intended behaviour.

### `tcn library`

```sh
tcn library list   --root <dir>                     # digest, nodes, cost, bits, deps, staleness, source
tcn library show   <name[@version]> --root <dir>    # the manifest entry and the stored program
tcn library verify [name] --root <dir>              # digest, dependencies, source, fixture; non-zero exit on failure
```

`verify` on the shipped chain library: 5 of 5 entries load, every dependency
resolves, every recorded fixture reproduces, no entry stale.

---

## 2. Curriculum artifact flow — `tcn/curriculum.py`

Prerequisites already formed a DAG. What was missing is that nothing travelled
along its edges. Three additions, all explicit configuration:

```json
{
  "name": "stage2_edge",
  "requires":  ["stage1_foreground"],
  "inherits":  ["stage1_foreground"],
  "publishes": ["perception.edge"]
}
```

- **`requires`** is ordering, unchanged.
- **`inherits`** is artifacts, validated to be a *subset* of `requires`. A stage
  that lists a prerequisite but does not inherit from it receives nothing. There
  is no implicit inheritance and no transitive inheritance: a stage that wants
  its grandparent's module names its grandparent too. Naming a stage that
  publishes nothing is a configuration error, caught at load.
- **`publishes`** is what the stage owes the next one, and it is an evidence
  gate: a stage that declares a module and does not leave it in the library
  fails, exactly like a missed metric threshold.

`Curriculum.run` resolves each stage's `inherits` into an `Artifacts` record —
the library path plus the exact list of references that stage may bind — and
passes it to the runner. A runner cannot reach a module the stage did not
declare, because it is never told about one. This changed the runner signature
from `runner(stage, path)` to `runner(stage, path, artifacts)`; the two shipped
tests that define runners were updated to match.

Publishing curricula are refused at `workers > 1`: one manifest, one writer.

Nothing branches on a domain or generator name. `stage_runner` branches on
`operation`, as it already did.

### The shipped curriculum now flows

`curricula/system.json` gains `publishes: ["mixed.answer"]` on `typed_synthesis`
and a new `module_reuse` stage that `requires` and `inherits` it. The reuse
stage loads the library under the **strict** source policy, offers the inherited
module as one typed candidate beside `identity` and `sin` on the same ports, and
enumerates. Run end to end with `tcn curriculum curricula/system.json`:

| | |
|---|---|
| published | `mixed.answer@1`, digest `d332eb48f8f56b181d47975b`, 4 nodes, execution cost 4.0, 17,728 description bits |
| reuse space | 3 programs — 1 module candidate, 2 same-typed distractors |
| selected | `module:d332eb48f8f56b181d47975b`, exact max error 0.0, conforming 1, unique |
| gates | `module_selected >= 1`, `solved >= 1`, `exact_max_error <= 0.005` — all passed |

That is the first time in this repository that a program crystallized by one
stage has been a candidate operator for another. The distractors matter: with a
single legal candidate the "selection" would be plumbing. All fifteen stages of
`curricula/system.json` pass end to end in the same run, including `computer`
once its gitignored `node_modules` is present.

Note the composed program costs 25,424 description bits against the module's own
17,728 — **one call site never pays**, which is the crossover of section 6
showing up in the shipped path.

---

## 3. The chain

Three stages, each one's output the next one's primitive. Stages 1 and 2 are the
demonstrated capability from `research/discrete-perception`, imported rather
than reimplemented (`rung3_mask.module_scaffold`,
`rung35_window.staged_scaffold` and `flat_scaffold`). Stage 3 is new.

```text
stage 1  perception.foreground   raw geometry pixels          -> foreground/background
stage 2  perception.edge         + stage 1's frozen module     -> fg(i) != fg(i+1)
stage 3  perception.region       + stage 2's frozen module     -> "this 2x2 tile straddles a boundary"
```

Stage 3's program, six nodes, is the positional-reuse pattern plus a readout:

```text
held    = insert(empty, rec)             set[(origin, image)], capacity 1
records = pair(offsets, held)            set[(offset, (origin, image))]
mapped  = map(records; wrapper)          set[(offset, edge)]      <- the free choice
kept    = filter(mapped; keep_flag)      the edges that fired
total   = count(kept)
region  = <comparison>(total, <k>)       <- the other free choice, 5 x 4 = 20
```

The `wrapper` is a crystallized module that computes its own absolute address
from the origin it was handed and calls the inherited edge module there. Which
wrapper is mapped is a *choice*, so the inherited module is selected on evidence
rather than supplied: arm A offers the stage-2 wrapper against two same-shaped
distractors built on the same stage-1 module with a wrong offset and a wrong
combinator, and the search picks the right one (`mapped = 0`).

Data: `geometry` at R=8, six training episodes, six held-out test episodes, two
to eight validation episodes in their own split. Supervision is the generator's
`object_ids` probe reduced to `>= 0` for stage 1 and derived from it for stages 2
and 3; only `record.actor_view().observations` reaches a program input.
Tolerance 1e-6 throughout. Every arm is settled by exhaustive enumeration over
its declared space, so the results are certificates rather than arguments.

### Stage 1 — foreground from raw pixels (nothing to inherit)

| | |
|---|---|
| space | 32,000 programs, **exhausted** in 522.5 s |
| supervision | 384 records — every position of 6 episodes |
| conforming | 1,952 — **not unique** |
| after 512 validation records at every position | 1,520 survive |
| pick | `blue != 43`; held-out max error **0.0**, accuracy **1.000** at every position of 6 test episodes |
| baselines | majority class 0.565; uniform draw from the space conforms 0.070 of the time |
| published | `perception.foreground@1`, 13 nodes, execution cost 13.0 |

### Stage 2 — the two-position boundary relation

| arm | space | evaluated | wall | conforming | held-out max error | held-out accuracy |
|---|---|---|---|---|---|---|
| **inherits stage 1** | **48** | 48, exhausted | **1.52 s** | **1, unique** | **0.0** | **1.000** |
| flat, nothing inherited | 4.9152e10 | 20,000 (budget) | 15.18 s | 0 | — | 0.000 |

Majority class on the same held-out set is 0.802. The flat arm's measured rate
is 7.589e-4 s/program, so exhausting it projects to **3.73e7 s = 431.7 days**.
The two arms use the identical data, supervision, tolerance and search
procedure; the only difference is whether stage 1's module is a candidate.

The inherited module was re-checked at the interface before use: exact at
384/384 training positions, 256/256 validation positions and 384/384 test
positions.

Published `perception.edge@1`, 11 nodes, execution cost 35.0,
`requires: [module:91b7e8e8862c0ebf333d8bf6]` — the dependency is recorded, so
the description cost stays transitive.

### Stage 3 — a region-level predicate over stage 2's output

| arm | what it inherits | space | wall | conforming | validated | held-out |
|---|---|---|---|---|---|---|
| A | stage 1 **and** stage 2 | **60** | **7.23 s** | 3 | 3 | max error 0.0, accuracy **1.000** |
| B | stage 1 only | 960 | 32.66 s | 3 | 3 | max error 0.0, accuracy **1.000** |
| C | nothing | **inexpressible** | — | — | — | — |

Majority class 0.556; a uniform draw conforms 0.040 of the time in arm A and
0.005 in arm B.

- **Arm A** returns the inherited edge wrapper with the readout
  `count(kept) > 0`. What the evidence actually establishes is narrower than
  "the module was chosen", and `conformers.py` enumerates it rather than leaving
  it to inference (`out/conformers.json`):

  | wrapper | readout | conforms | validation exact | held-out exact |
  |---|---|---|---|---|
  | inherited stage-2 edge | `gt(total, 0)` | yes | yes | yes |
  | inherited stage-2 edge | `ge(total, 1)` | yes | yes | yes |
  | distractor: offset 6, xor | — | **no** | — | — |
  | distractor: offset 3, xnor | `le(total, 3)` | yes | yes | yes |

  So the supervision **excludes the genuinely wrong module** — the offset-6
  distractor conforms at no threshold — but it does **not** discriminate between
  logically equivalent routes: complementing the combinator and complementing
  the readout is De Morgan, and that program computes the identical predicate.
  The inherited module is returned because it comes first in enumeration order,
  which is a tie-break, not evidence. `module_selected` in the stage's gates
  should be read as "a module is on the output path and the wrong one is
  excluded", not as "the inherited one was preferred".
- **Arm B** puts stage 2's content back into stage 3's search: one candidate
  wrapper per (offset, combinator) pair, 3 x 16 = 48 of them, against arm A's 3.
  It still solves, at 16x the space and 4.5x the wall clock. The wall-clock
  ratio is smaller than the space ratio because `evaluate` exits at the first
  disagreeing record, so wrong programs are cheap.
- **Arm C** is a structural result rather than a budget failure. `map` takes a
  module parameter; with an empty library there is no registered module, so no
  program in this algebra aggregates over a set of positions at all. The
  smallest library that admits one is a module drawn from stage 2's own flat
  space, which makes the joint space 9.83e11 and projects to 3.34e10 s — about
  1,060 years. **Composition is not an optimisation here; it is the only route
  to the aggregate.**

Published `perception.tile_call@1` (9 nodes, cost 43), `perception.keep_flag@1`
(1 node, cost 1) and `perception.region@1` (6 nodes, cost 180,
`requires` both of the former).

---

## 4. The cost of inheritance: exactness, not accuracy

The first two attempts at this chain both failed at stage 2, and the reason is
worth more than the eventual success.

Stage 1's supervision does not identify its program. 2,608 of 32,000 programs
conform on 288 subsampled training records; 2,464 still conform after 128
validation records. `enumerate_fit` returns the first in enumeration order, and
that program is 99.74% correct at every position of held-out episodes — one
wrong pixel in 384.

That one pixel is fatal downstream. Stage 2's target is `fg(i) != fg(i+1)`
computed from the generator's probe, so it is exact by construction; a parent
that is wrong at one position makes the child's target unreachable at two. The
measurement, with the stage-1 module exact on 512 validation records and 384
test records but wrong at **1 of 384 training positions**:

```text
best achievable stage-2 error over all 48 staged programs: 1.0   (i.e. none conform)
```

Zero of 48. Not harder — impossible.

`identify.py` sweeps the validation budget over the cached conforming set
(`out/identification.json`):

| validation episodes | records | survivors | pick's max error at every position of 6 test episodes | pick's accuracy |
|---|---|---|---|---|
| 1 | 64 | 2,608 | 1.0 | 0.9974 |
| 2 | 128 | 2,464 | 1.0 | 0.9974 |
| 4 | 256 | 2,464 | 1.0 | 0.9974 |
| 8 | 512 | 1,976 | **0.0** | **1.0000** |
| 16 | 1,024 | 1,672 | **0.0** | **1.0000** |

Two things to read off this. Validation buys a *correct pick*, not
identification: 1,672 programs still conform at the largest budget. And the
budget that fixes the pick is about eight times the budget the
`discrete-perception` track used, which is why that track's own tie-break
experiment recorded the validation-filtered rule still landing on a program with
three wrong slots in 384.

The chain as shipped fixes it in the parent rather than the child: stage 1 is
supervised at **every** position of its training episodes rather than a 48/64
subsample, so training conformance already means exactness on those episodes,
and 8 validation episodes then filter the rest. That leaves 1,952 conforming and
1,520 validated, with the returned pick exact everywhere it was measured — and
it is *cheaper*, 522.5 s against the subsampled arm's 940.9 s for the same
sweep-plus-filter, because a smaller conforming set makes the filter cheap. More
supervision in the parent bought both correctness and time.

The library records this as an obligation rather than a hope: `runner.py`
re-checks the inherited module at every position of the consuming stage's own
train, validation and test episodes and reports `inherited_interface` before
searching. For the shipped chain it reads 0 wrong of 384, 256 and 384.

**The general statement: a chain needs exactness at the interface, not accuracy
at the interface.** A 99.7%-accurate module is a fine classifier and a useless
primitive when the child's supervision is exact. Nothing in the repository
measured this before, because nothing had ever consumed a learned module.

---

## 5. What was touched

| file | change |
|---|---|
| `tcn/library.py` | **new.** `Library`, `Entry`, `module_dependencies`, the source policy and its exceptions. |
| `tcn/curriculum.py` | `Artifacts`; `Stage.inherits` / `Stage.publishes`; validation that `inherits` is a subset of `requires` and that the source publishes; `Curriculum.inherited`; `run(..., library=)`; publication as an evidence gate; runner signature. |
| `tcn/cli.py` | `tcn library list/show/verify`; `--library` on `tcn curriculum`; `synthesize` publishes when the stage declares it; the `reuse` operation. |
| `curricula/system.json` | `typed_synthesis` publishes; new `module_reuse` stage. |
| `tests/test_module_library.py` | **new**, 14 tests. |
| `tests/test_training_curriculum.py`, `tests/test_integration_complete.py` | runner signature (3 lines). |

**Deliberately not touched**: `tcn/learning.py` and `tcn/select.py`, which have
branches queued for merge, and `tcn/graph.py`, `tcn/operators.py`,
`tcn/runtime.py`, `tcn/search.py`, `tcn/synthesis.py` and `tcn/training.py`,
which the `abstraction-preference` and `positional-reuse` branches change. The
library needed no change to any of them: `register_module` already prunes and
content-addresses, `save_program`/`load_program` already round-trip a module
list, and `description_bits` already charges transitively. That is the useful
finding about the existing core — **the composition machinery really was
complete, and the missing piece really was only storage and flow.**

`tcn/cli.py` is also changed by the queued `perturbation-selection` branch; the
changes here are additive (one new subcommand, one new stage operation, a
publish hook inside the existing `synthesize` branch) and should merge cleanly,
but `git merge-tree` was not run against it.

Test suite: **193 passed** with the four `generators/computer` tests enabled via
a temporary `node_modules` symlink, removed before committing.

---

## 6. Cost accounting

`out/accounting.json`. The flat side of every comparison is produced by a generic
inliner that walks a frozen module's selected candidates and re-emits them into
the caller, recursing through nested module calls; each flat program is checked
to agree with its composed counterpart on real episodes before its cost is
reported, so both sides are two spellings of one function.

**Caveat first, because it is load-bearing.** `description_bits` is the
serialized length of the program's JSON, not its learned content
(`FINDINGS` section 3). A module call site carries the operator's full input and
output type dictionaries, and at R=8 the observation type alone is a 192-field
product, so both sides of every comparison are inflated by type serialization.
Node counts are given beside every bit count.

### Transitive charging, exactly

| module | nodes | own bits | callee bits | transitive bits | execution cost |
|---|---|---|---|---|---|
| `perception.foreground@1` | 13 | 1,650,712 | 0 | 1,650,712 | 13 |
| `perception.edge@1` | 11 | 2,631,760 | 1,650,712 | 4,282,472 | 35 |
| `perception.tile_call@1` | 9 | 2,625,464 | 4,282,472 | 6,907,936 | 43 |
| `perception.keep_flag@1` | 1 | 5,936 | 0 | 5,936 | 1 |
| `perception.region@1` | 6 | 2,034,312 | 6,913,872 | 8,948,184 | 180 |

Each row's `callee bits` is exactly the sum of its callees' *transitive* bits
(4,282,472 = edge's own 2,631,760 + foreground's 1,650,712; 6,913,872 =
6,907,936 + 5,936). The definition is charged once and the charge is transitive
through three levels of nesting, as section 4 requires.

### The composed region program against the flat equivalent

| region | positions | composed nodes | composed bits | flat nodes | flat bits | bits ratio | composed cost | flat cost | cost ratio |
|---|---|---|---|---|---|---|---|---|---|
| 2x2 | 4 | **6** | 8,948,184 | 183 | 31,255,432 | **0.286** | 180 | 183 | 0.984 |
| 3x3 | 9 | **6** | 8,948,336 | 413 | 70,077,792 | **0.128** | 400 | 413 | 0.969 |
| 4x4 | 16 | **6** | 8,948,640 | 735 | 124,429,624 | **0.072** | 708 | 735 | 0.963 |

The composed program is six nodes at every region size and its description grows
by 152 bits per step — the constant set naming the positions. The flat program
grows linearly in both, 183 to 735 nodes and 31 Mbit to 124 Mbit. This is the
positional-reuse result restated in the description-cost metric: **the definition
is charged once and the caller does not grow with the number of positions.**

Execution cost is at parity, 0.96-0.98, with the composed program marginally
*cheaper*: `map` charges `capacity x body`, which is 4 x 43 = 172 of the composed
program's 180, while the flat spelling pays the same body 4 times plus the fold.
A call and its inlined body pay equally, as section 4 says they should.

### The crossover

`research/abstraction-preference` measured description size crossing over at **2
call sites for a 3-gate body**. Re-measured here on the chain's own bodies, over
a 192-byte observation — two orders of magnitude larger on both axes:

| body | nodes | transitive bits | crossover |
|---|---|---|---|
| `perception.foreground` | 13 | 1,650,712 | **2 call sites** |
| `perception.edge` | 11 (35 with callees) | 4,282,472 | **1 call site** |

| call sites | foreground abstracted bits | foreground inlined bits | pays? |
|---|---|---|---|
| 1 | 2,658,584 | 2,254,792 | no |
| 2 | 3,468,712 | 4,311,848 | **yes** |
| 4 | 5,088,968 | 8,425,960 | yes |
| 8 | 8,329,776 | 16,655,816 | yes |

**The 2-call-site crossover holds at this scale.** It moves to 1 for the edge
module, because a body that itself contains module calls carries its callees'
definitions into every inlined copy while the abstracted spelling charges them
once. Execution cost is identical on both sides at every call count (15/15,
31/31, ... 127/127 for foreground), which is the section 4 claim measured
directly.

---

## 7. Limits, and what a reader should not conclude

- **This is a perceptual chain, chosen because those modules were already known
  to transfer.** Nothing here shows the same effect on the language, control or
  computer-use tracks.
- **Stage 3's arm-B result is the honest one to quote for "how much does
  inheritance help".** Stage 2's billion-fold number is real but it is the
  difference between a reachable space and an unreachable one, which is a
  statement about the flat space's size as much as about composition. Arm B is a
  case where both arms succeed and inheritance still buys 16x in space and 4.5x
  in time.
- **Non-uniqueness is everywhere and it is not fixed.** Stage 1 has 1,520
  validated conforming programs out of 32,000; stage 3 has 3, one of them over a
  distractor module that computes the same predicate by De Morgan. The pick is
  lexicographic among survivors. `enumerate_fit(rank='description'|'cost')` does
  not separate them because they are all the same size. So this track shows a
  module being *used* and a wrong module being *excluded*; it does not show a
  module being preferred over an equally good alternative.
- **`description_bits` is JSON length.** Every bit number here inherits that
  fault. The ratios are meaningful because both sides are measured the same way;
  the absolute magnitudes are not information content.
- **No gradient arm was run.** The whole chain is discrete enumeration, which is
  what `FINDINGS` section 8 recommends for spaces this size and shape; the
  address-relaxation and dead-surrogate results say the relaxed path would not
  reach these targets without the core fixes now queued.
- **Two writers, one manifest.** Publication is serialized by refusing
  `workers > 1`, not by locking. A distributed curriculum would need more.
- **Fingerprint granularity is unchanged.** The policy makes a mismatch loud and
  checkable; it does not make `source_fingerprint()` finer-grained. That remains
  the open problem `external-environments` named.

---

## 8. Reproducing

```sh
.venv/bin/python research/module-library/run_chain.py --fresh    # ~10 min; builds library/ and out/chain
.venv/bin/python research/module-library/identify.py             # caches out/conforming_r8.json (~25 min)
.venv/bin/python research/module-library/accounting.py           # out/accounting.json
.venv/bin/python research/module-library/conformers.py           # out/conformers.json
.venv/bin/python -m tcn library verify --root research/module-library/library
.venv/bin/python -m tcn curriculum curricula/system.json --out artifacts/curriculum
.venv/bin/python -m pytest tests/ -q
```
