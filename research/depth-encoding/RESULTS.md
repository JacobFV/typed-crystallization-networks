# Depth encoding — is width polymorphism reachable inside the current algebra?

**Verdict: yes, and it needs no new type machinery. The width barrier is an
ergonomic gap, not a type-system one — but only because a *schema* is what
crosses widths, and the schema is currently Python source rather than a stored
artifact.**

Three numbers carry it.

1. **Width enters in exactly 31 places, 16 of them a type equality, and every
   one of them is a comparison of `Type`, never a separate width mechanism.**
   `Type` is a frozen dataclass whose `items` and `capacity` fields *are* the
   width (`tcn/types.py:58,64`); there is no width variable anywhere to
   generalize. The barrier is one structural `==`, reached from 16 sites.
2. **Selections transfer perfectly.** The selection vector found by exhausting
   the depth-1 space — 2 integers, **8.09 bits** — scores **4.00 / 4 on 64
   held-out episodes at depths 1, 2, 3, 4, 6 and 8** (`program` observation
   widths 3, 6, 9, 12, 18, 24), against a best constant of 2.06–2.50 and a
   whole-space mean of 2.00. Independently exhausting all 272 programs at each
   depth certifies that same vector **`unique`** at every depth, second-best
   3.06–3.44. Fitting at depth 2 instead returns the identical vector.
3. **The hardened artifact does not transfer, and must not.** The same frozen
   program handed a depth-2 observation raises `TypeError: input representation
   mismatch` at `tcn/graph.py:99` — §30 on the width axis, executable. Six
   depths produce **six distinct digests**, 169 kbit to 1.30 Mbit. The 8.09-bit
   vector is the width-invariant part; the megabit is not.

And one control that keeps the headline honest:

4. **A schema that is wrong is invisible at one width.** An interpreter variant
   differing by a single edge — its answer node reads `gate_0` rather than
   `gate_{d-1}` — is *indistinguishable* at depth 1: same space, same
   exhaustion, same certificate `unique`, same vector, 4.00. It then collapses
   to **1.63–2.69** at depths 2–8, and exhausting its own 272-program space at
   each of those depths returns **0 conforming, certificate `complete`** — so
   the collapse is a proved non-existence in that family, not a search failure.
   **Transfer is a property of the schema, and depth-1 evidence cannot certify
   a schema.**

Nothing in `tcn/` or `generators/` was modified, none is proposed, and no type
check was relaxed at any point. `PREREGISTRATION.md` was committed at `1e439a1`
**before any arm**; amendment 1 (arms F, G, H) at `5f8aa4f`, **after** the
pre-registered arms reported and **before** F/G/H were written or run.

Environment: repository `.venv`. Baseline before writing anything and again
after every arm: **324 passed / 13 failed** of 337 (the documented
`node_modules` failures — verified identical with this directory moved out of
the tree), and `python -m tcn train --episodes 160` reproducing
**0.248835613951087 → 0.0022308224288281053**, `fully_frozen: true`,
evaluation **4.0**, frozen evaluation **4.0**.

---

## 1. Step 1 — where width actually enters

`audit.py` → `out/audit.json`. Every citation is resolved at run time by
matching the recorded source text, so a line number cannot silently drift; every
load-bearing site is also demonstrated executably, and the exception below is
the one the interpreter actually raised.

**The single root.** `Type` is a frozen dataclass (`tcn/types.py:54`), so `==`
is structural over every field. A tuple's arity lives in `items`
(`tcn/types.py:58`) and a set's bound in `capacity` (`tcn/types.py:64`). There
is no width *parameter* in the system to make polymorphic — the width is the
type. That is why the audit is narrower than the problem sounded: 16 type
equalities, all of them the same `==` reached from different callers.

### width-entry sites

| # | site | class | what it compares |
|---:|---|---|---|
| 1 | `tcn/types.py:58` | definition | `Type` is a frozen dataclass, so `==` is structural over every field; `items` carries a tuple's arity and `capacity` a set's bound. This is the single root of every comparison below. |
| 2 | `tcn/types.py:64` | definition | A set's declared bound is a field of its type, so two capacities are two types. |
| 3 | `tcn/types.py:113` | derivation | `Type.width` — the flat relaxation width, derived from arity/capacity. |
| 4 | `tcn/types.py:141` | value equality | `encode` refuses a value whose arity disagrees with the declared tuple. |
| 5 | `tcn/types.py:194` | value equality | `validate_raw` — every `Value` construction re-checks arity. |
| 6 | `tcn/types.py:146` | value equality | `encode` refuses a set larger than the declared capacity. |
| 7 | `tcn/graph.py:75` | type equality | `Program.validate` — a candidate's declared operator inputs must equal the types of the ports it reads. This is where a depth-1 candidate spliced into a depth-2 scaffold dies. |
| 8 | `tcn/graph.py:76` | type equality | `Program.validate` — every candidate at a node must agree with the node's output type, so a width-carrying output pins the node. |
| 9 | `tcn/graph.py:77` | type equality | `Program.validate` re-resolves each operator from its own dict and demands the identical `Operator`; widths inside `inputs`/`output` are part of that. |
| 10 | `tcn/graph.py:82` | type equality | `Program.validate` — a recurrence carrier's type is fixed, width included. |
| 11 | `tcn/graph.py:87` | type equality | `validate_signals` — a supervision signal's type must equal the port's. |
| 12 | `tcn/graph.py:99` | type equality | `Program.execute` — **the runtime gate**. A frozen program handed a `program` observation of another depth fails here, before any node runs. |
| 13 | `tcn/graph.py:104` | type equality | `Program.execute` — the same for a supplied state carrier. |
| 14 | `tcn/graph.py:184` | derivation | `operator_parameters` — the `project` parameter family is derived from the tuple's arity, so the *search space itself* is width-derived. |
| 15 | `tcn/operators.py:47` | type equality | `Registry.resolve`, module branch — **§30's error**. A hardened module's input type names the observation, so it cannot be resolved at another width. |
| 16 | `tcn/operators.py:94` | type equality | `project` — the index bound is the arity, so a `project index=3` operator is illegal on a 3-field tuple. |
| 17 | `tcn/operators.py:98` | type equality | `index` — dynamic tuple indexing needs a uniform tuple; its *input* type still carries the arity even though its output does not. |
| 18 | `tcn/operators.py:101` | type equality | `member`/`insert`/`remove` — element type equality, capacity carried in `ts[0]`. |
| 19 | `tcn/operators.py:104` | type equality | `union`/`intersection` — full set-type equality, **capacity included**. |
| 20 | `tcn/operators.py:107` | derivation | `pair`/`join` — the output capacity is the product of the inputs', so a capacity change propagates downstream as a type change. |
| 21 | `tcn/operators.py:113` | type equality | `map`/`filter` — the mapped module's input type must equal the set's element type. |
| 22 | `tcn/operators.py:117` | derivation | `map` — **the declared capacity is copied into the output type**, so a `map` at capacity 8 and one at capacity 16 are different operators with different output types. |
| 23 | `tcn/operators.py:118` | derivation | `map`/`filter` — declared capacity is also the charged execution cost, so it is not a free parameter to raise. |
| 24 | `tcn/operators.py:171` | type equality | `Registry.resolve` — the final gate; a requested output type must equal the inferred one, widths included. |
| 25 | `tcn/operators.py:176` | type equality | `Registry.exact` — re-checked at every single exact execution. |
| 26 | `tcn/learning.py:225` | type equality | `SoftProgram` — the relaxed path checks the flat width, the numeric shadow of the same fact. |
| 27 | `tcn/compile.py:338` | derivation | `compile.py` bakes the arity into the generated boundary check, so a compiled artifact is width-specific in emitted source too. |
| 28 | `tcn/compile.py:542,555` | derivation | the compiled backend likewise bakes the declared set capacity. |
| 29 | `tcn/library.py:241` | storage | `Library.publish` — a stored entry records the module's input **type dicts**, which contain `items`/`capacity`; two widths are two entries with two digests. |
| 30 | `generators/logic/generator.py:128` | origin | **where the width is created**: `program` is a tuple of `3 × depth` scalars, so depth moves the type. |
| 31 | `generators/logic/generator.py:129` | origin | §15's second typed view: one `set` type at a declared capacity, emitted only when configured. Depth moves cardinality, not width. |

Counts: 16 type equality, 7 derivation, 3 value equality, 2 definition,
1 storage, 2 origin.

### executable demonstrations

Each row is a real call and a real exception, recorded with the frame that
raised it. (`operators.py:43` is the body of `resolve`'s `require` helper; the
check that failed is the site cited in the table above.)

| probe | outcome |
|---|---|
| `execute_wrong_width` — depth-1 artifact, depth-2 observation | `TypeError: input representation mismatch` at `tcn/graph.py:99` |
| `execute_right_width` | no error |
| `validate_spliced_candidate` — depth-1 candidate, depth-2 port | `TypeError: candidate input mismatch` at `tcn/graph.py:75` |
| `module_resolved_at_other_width` — §30 | `TypeError: module:b10add3e…: operator signature mismatch` (`tcn/operators.py:47`) |
| `module_resolved_at_own_width` | no error |
| `project_index_past_arity` | `TypeError: project: operator signature mismatch` (`tcn/operators.py:94`) |
| `map_wrong_element_width` | `TypeError: map: operator signature mismatch` (`tcn/operators.py:113`) |
| `union_across_capacities` | `TypeError: union: operator signature mismatch` (`tcn/operators.py:104`) |
| `exact_wrong_width` | `TypeError: input type mismatch` at `tcn/operators.py:176` |
| `encode_wrong_arity` | `ValueError: tuple arity mismatch` at `tcn/types.py:141` |
| `validate_raw_wrong_arity` | `TypeError: invalid tuple` at `tcn/types.py:194` |
| `encode_over_capacity` | `OverflowError: set capacity exceeded` at `tcn/types.py:146` |
| `soft_wrong_width` — the relaxed path | `TypeError: input width mismatch` at `tcn/learning.py:225` |
| library publish/load round trip, then resolve at the other width | two digests (`b10add3e…`, `29bd8074…`), two versions of one name, `stored_input_types_equal: false`, and `operator signature mismatch` on the cross-width call |

Three measured facts worth naming separately.

* **The search space itself is width-derived.** `operator_parameters` offers
  `project` **3** settings at depth 1 and **6** at depth 2
  (`tcn/graph.py:184`). Width is not only a check at the end; it sizes the
  family an enumerator walks.
* **`map`'s declared capacity is copied into its output type and into its
  charged cost.** `map` at capacity 8 and at capacity 16 have unequal outputs
  and costs 8.0 vs 16.0 (`tcn/operators.py:117,118`). Raising a capacity to
  cover more widths is therefore not free — it is charged at every call site.
* **§15's `gates` channel is confirmed here from the shipped generator**:
  one `set` type at every depth, flat width **40**, cardinality 1 at depth 1 and
  2 at depth 2, while `program` is 3 and 6 fields with `types_equal: false`.

---

## 2. Step 2 — the schema, and what crosses widths

`scaffold.py` builds an interpreter over the **`program`** channel — the
width-varying one, not §15's `gates` view — and unrolls the circuit: gate `i`
reads fields `3i, 3i+1, 3i+2`, `encode`s them to `int[8]`, looks its two
operands up in the wire tuple built so far with `index`, and selects its truth
table with a second `index` over the sixteen possible gate outputs.

Choice lives at exactly two nodes and nowhere else, with the **same candidate
counts (17, 16) at every depth**: `relation` (16 fixed `truth_j` over the last
gate's operands, plus `identity(gate_{d-1})`) and `goal_relation` (16
`truth_j(relation, goal)`). Space **272 at every depth**. Everything else is
single-candidate. The policy tail, its five constants and the action schema are
`examples/joint.py` verbatim, so returns are on the same 0–4 scale as §15.

**No type check is relaxed anywhere.** Each instantiated `Program` is exactly
typed for its own `program` width and passes `Program.validate`; the six
artifacts have six different digests and different node counts. What crosses
widths is only the integer vector.

Configuration `{'inputs': 4, 'nondegenerate': True, 'min_relevant_inputs': 2}`,
random gate wiring, horizon 4, whole episode scored. Fit on `range(16)`
`split='train'`; held out on `range(10000, 10064)` `split='test'` — 64 episodes,
disjoint addresses and a different split.

### held-out return, 64 episodes, 0–4 scale

| arm | d1 | d2 | d3 | d4 | d6 | d8 |
|---|---|---|---|---|---|---|
| **A1 — interpreter, selections fitted at d=1** | **4.0000** | **4.0000** | **4.0000** | **4.0000** | **4.0000** | **4.0000** |
| **A2 — interpreter, selections fitted at d=2** | **4.0000** | **4.0000** | **4.0000** | **4.0000** | **4.0000** | **4.0000** |
| F — wrong schema, same vector, fitted at d=1 | 4.0000 | 2.5000 | 1.6250 | 2.3125 | 1.6250 | 2.6875 |
| B — record scaffold, best of 4 carried | 1.8750 | 2.1875 | 2.4375 | 2.0625 | 2.0625 | 2.5000 |
| best constant | 2.1250 | 2.1250 | 2.3750 | 2.5000 | 2.0625 | 2.4375 |
| uniform random | 1.8438 | 2.0625 | 1.8750 | 1.9062 | 2.0000 | 1.8125 |
| mean over the whole 272-program space | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 |
| second best of the 272 | 3.0625 | 3.1250 | 3.2500 | 3.3125 | 3.0625 | 3.4375 |

`program` fields at those depths: 3, 6, 9, 12, 18, 24. Five of the six widths
are unseen by the A1 fit.

Both fits return the **identical** vector `{relation: 16, goal_relation: 6}` —
the table-conditioned lookup, and `xor` against the objective bit. Fitting at
the larger width does not change what is found.

### certificates

Every sweep is scored with `tcn.search.program_return` and certified with
`tcn.search.certificate_of` — the shipped enumerator's own scoring function and
its own certificate rule. `sweep` exists only because
`enumerate_environment` reports the *size* of the conforming set and not its
members, and this track needs the members (§47: a transfer that works only for
whichever member enumeration reaches first is under-determined). The
`crosscheck` condition runs the same depth-1 space through
`tcn.search.enumerate_environment` itself and **agrees on all of** evaluated,
space size, exhausted, conforming, certificate, best return, and chosen
selection.

| sweep | space | evaluated | exhausted | conforming @ 4.0 | certificate |
|---|---:|---:|---|---:|---|
| A fit, interpreter, depth 1, 16 train episodes | 272 | 272 | true | 1 | `unique` |
| A fit, interpreter, depth 2, 16 train episodes | 272 | 272 | true | 1 | `unique` |
| E held-out exhaustion, interpreter, depth 1 | 272 | 272 | true | 1 | `unique` |
| E held-out exhaustion, interpreter, depth 2 | 272 | 272 | true | 1 | `unique` |
| E held-out exhaustion, interpreter, depth 3 | 272 | 272 | true | 1 | `unique` |
| E held-out exhaustion, interpreter, depth 4 | 272 | 272 | true | 1 | `unique` |
| E held-out exhaustion, interpreter, depth 6 | 272 | 272 | true | 1 | `unique` |
| E held-out exhaustion, interpreter, depth 8 | 272 | 272 | true | 1 | `unique` |
| F fit, wrong schema, depth 1 | 272 | 272 | true | 1 | `unique` |
| F held-out exhaustion, wrong schema, depth 1 | 272 | 272 | true | 1 | `unique` |
| F held-out exhaustion, wrong schema, depth 2 | 272 | 272 | true | **0** | `complete` |
| F held-out exhaustion, wrong schema, depth 3 | 272 | 272 | true | **0** | `complete` |
| F held-out exhaustion, wrong schema, depth 4 | 272 | 272 | true | **0** | `complete` |
| F held-out exhaustion, wrong schema, depth 6 | 272 | 272 | true | **0** | `complete` |
| F held-out exhaustion, wrong schema, depth 8 | 272 | 272 | true | **0** | `complete` |
| B fit, record scaffold, depth 1 | 256 | 256 | true | **0** | `complete` |

Arm E is what makes the headline a transfer result rather than a search result:
the transferred vector is not merely *good* at each unseen width, it is the
**unique** optimum of that width's whole 272-program space, with 271 members
below it and the runner-up 0.56–0.94 behind.

### arm D — the hardened artifact does not cross, and should not

Depth-1 frozen artifact `aa29e3e3bad03fb8546f199d`, offered each depth's own
observation:

| depth | outcome |
|---:|---|
| 1 | executes |
| 2 | `TypeError: input representation mismatch` at `tcn/graph.py:99` |
| 3 | `TypeError: input representation mismatch` at `tcn/graph.py:99` |
| 4 | `TypeError: input representation mismatch` at `tcn/graph.py:99` |
| 6 | `TypeError: input representation mismatch` at `tcn/graph.py:99` |
| 8 | `TypeError: input representation mismatch` at `tcn/graph.py:99` |

This is §30 reproduced on the width axis rather than the raster axis, and it is
the behaviour the project wants: the artifact is honest about what it accepts.

### arm B — the negative control that separates "applicable" from "capable"

The record scaffold reads `bits` and `goal` only, never the width-varying
channel, so its vector is width-free trivially. Exhausting its 256-program space
on the fit episodes finds **0 conforming, certificate `complete`**, best 2.75.
Carrying its four argmax selections to every depth gives **1.50–2.50** across
the 24 (selection, depth) cells, against a best constant of 2.06–2.50 and a
ceiling of 4.00. Applying a vector at another width is not the same thing as a
capability surviving there.

**Pre-registered criterion 4 said "arm B stays at or below its best constant at
every `d′`", and taken literally that is false**: 4 of the 24 cells sit above
the constant, by **0.0625 to 0.1875** on a 0–4 scale at n = 64 — 2.1875 vs
2.1250 at d2, 2.5000 and 2.4375 vs 2.3750 at d3, 2.2500 (twice) vs 2.0625 at d6,
2.5000 vs 2.4375 at d8. Those margins are a single episode's reward or two, and
arm B's spread straddles the constant in both directions at every depth. The
criterion was written too tightly for a control with this much per-episode
variance; the substantive claim it was meant to test — that arm B carries no
capability — holds with room to spare, since 4.00 is 1.5 return above anything
arm B reaches anywhere. Recorded here rather than smoothed over, and the
criterion is marked failed-as-written in section 4.

### arm F — a wrong schema is invisible at one width

Amendment 1. The variant differs from A by **one edge**: `relation`'s lookup
candidate reads `gate_0` instead of `gate_{d-1}`. At depth 1 those are the same
node, so the depth-1 evidence is *necessarily* identical — and it is: 272
evaluated, exhausted, 1 conforming, certificate `unique`, the same vector
`{relation: 16, goal_relation: 6}`, 4.00 held out.

At depths 2–8 it scores **2.50, 1.63, 2.31, 1.63, 2.69**, below or around the
best constant, and exhausting its own space at each of those depths returns
**0 of 272 conforming, certificate `complete`** — a proved non-existence, not a
search failure, and exactly the distinction the discipline asks for.

**Consequence for the finding.** "Selections transfer" is a statement about a
schema, and it is not self-certifying: evidence at one width cannot distinguish
a schema that generalizes from one that does not. Certifying a schema requires
evidence at more than one width. That is a real and cheap requirement — arm E's
per-depth exhaustion costs 26–95 s — but it must be stated, and it is the reason
this result is *not* "the depth-1 artifact just works everywhere".

### arm G — achieved difficulty, not the requested configuration

| depth | distinct circuits | distinct final tables | relevant-input histogram | majority fraction |
|---:|---:|---:|---|---:|
| 1 | 52/64 | 10 | `{"2": 64}` | 0.5312 |
| 2 | 64/64 | 10 | `{"2": 53, "3": 11}` | 0.5312 |
| 3 | 64/64 | 10 | `{"2": 50, "3": 13, "4": 1}` | 0.5938 |
| 4 | 64/64 | 10 | `{"2": 40, "3": 18, "4": 6}` | 0.6250 |
| 6 | 64/64 | 10 | `{"2": 32, "3": 25, "4": 7}` | 0.5156 |
| 8 | 64/64 | 10 | `{"2": 35, "3": 24, "4": 5}` | 0.6094 |

All ten non-degenerate truth tables appear at the final gate at every depth, and
the circuits are essentially all distinct from depth 2 on, so a single frozen
gate cannot be the answer: the program really is reading the episode's own
program text. The majority answer is 0.52–0.63, which is the 2.06–2.50 best
constant in the return table.

### arm H — what the width-invariant part costs

| depth | `program` fields | scaffold nodes | frozen digest | description bits | execution cost |
|---:|---:|---:|---|---:|---:|
| 1 | 3 | 42 | `aa29e3e3bad03fb8546f199d` | 169,264 | 42 |
| 2 | 6 | 69 | `2ecc0b362563175302d25e34` | 280,944 | 69 |
| 3 | 9 | 96 | `a6a183ea4f6c7266ae58ba73` | 409,528 | 96 |
| 4 | 12 | 123 | `98f03890535115eb55e526a9` | 554,912 | 123 |
| 6 | 18 | 177 | `ddbedead8876e31a0f888768` | 896,048 | 177 |
| 8 | 24 | 231 | `2ffc8dad769d5f35dc2b4bf3` | 1,304,320 | 231 |

Six distinct artifacts; `pruned()` removes nothing, so these are not padded
scaffolds. The selection vector is **2 integers, 8.09 bits**, byte-identical at
all six depths. That ratio — 8 bits invariant against 169 kbit–1.30 Mbit
specific — is the whole ergonomic claim, measured.

Cost: 237,184 environment episodes across all sweeps and evaluations; 12.2 s for
the two fits, 313.6 s for the six held-out exhaustions, 411.5 s for arm F.

---

## 3. What this means for the §12 directions

Step 3 was **not entered**: the step-2 falsification did not fire. That is a
finding rather than an omission, and the directions can now be judged against
evidence instead of intuition.

* **Type variables / shape variables / bounded dependent dimensions.** No
  measurement in this track calls for them. The audit shows there is no width
  *parameter* to abstract over — width is the identity of the type
  (`tcn/types.py:58,64`), so introducing a variable means introducing a second,
  weaker notion of type equality reachable from all 16 sites. It would have to
  be honoured in `Program.validate`, `Registry.resolve`, `Registry.exact`,
  `SoftProgram`, `compile.py`'s emitted boundary code and `Library`'s stored
  entries, and each of those is a place where exactness is currently free. The
  measured benefit it would buy over the schema route is **8.09 bits of
  bookkeeping**. That is not a trade worth making, and this track does not
  propose it.
* **Generic schema + concrete instantiated artifacts** (DESIGN.md §7's
  reading) is what was measured, and it works exactly as that section
  predicted: the class holds the schema, the artifacts hold the instantiations,
  and no type-system weakening is needed. Its cost is honest and small: one
  parametric builder, and one artifact per width.

**The gap that is left is storage and provenance, not typing.** `tcn/library.py`
stores *artifacts* (`Entry.inputs` is a tuple of type dicts,
`tcn/library.py:241`), and there is no place to put a schema or a selection
vector. Today a schema is a Python function in a research directory and the
vector is a dict printed in a log; §30's `apply.py` and this track's
`scaffold.py` each re-implement the same rebuild-and-reapply move. A minimal,
additive shape follows directly from what was measured, and is stated here as a
proposal only — **nothing was implemented**:

* a library entry that already has `semantic_id` as class identity (DESIGN.md
  §7) gains, per class, a **selection vector** and the **identifier of the
  builder** that instantiates it, plus the set of widths at which the vector has
  been certified;
* "certified at a width" means what arm E measured: the vector is conforming in
  that width's own instantiated space, with a certificate;
* arm F is the reason the *set* of certified widths must be stored rather than a
  single flag. One width's certificate says nothing about another, and the
  system currently has no way to record that distinction.

This is off by default in the strongest sense — it does not exist. The
recommendation is that it stay that way until an experiment needs to reuse a
schema across process boundaries, which this one did not.

## 4. The pre-registered criteria, one by one

| # | criterion | verdict |
|---:|---|---|
| 1 | A1's depth-1 enumeration exhausted, certificate `unique` or `complete` | **held** — 272/272, `unique` |
| 2 | every conforming depth-1 selection ≥ 3.9/4 at each `d′ ∈ {2,3,4,6,8}` | **held** — 4.0000 at all five, one conforming selection |
| 3 | strictly above the best constant and above arm C at every `d′` | **held** — 4.00 vs 2.06–2.50 and vs a whole-space mean of 2.00 |
| 4 | arm B at or below its best constant at every `d′` | **failed as written** — 4 of 24 cells above it by 0.06–0.19; see §2, arm B. The claim it tested (arm B carries no capability) holds |
| 5 | arm D raises a type error | **held** — `TypeError` at `tcn/graph.py:99` at all five unseen depths |
| 6 | no `tcn/` file modified; 337-test result and shipped fixture unchanged | **held** — `git diff` touches nothing outside `research/depth-encoding/`; 324/13 before and after; 0.248835613951087 → 0.0022308224288281053 at 4.0/4.0 |

## 5. Falsifications, honoured

* *Selections transfer across depths → the width barrier is an ergonomic
  problem, not a type-system one. Report it that way and do not build
  machinery.* **This is what happened.** No machinery was built and none is
  proposed beyond the storage note above, which is explicitly not implemented.
* *A prototype works only by relaxing a type check → that is the forbidden
  solution.* No type check was relaxed at any point; every instantiated program
  passes the unmodified `Program.validate`, and arm D shows the artifact still
  refuses the wrong width.
* The under-determination clause did not fire: both fits returned a
  **single** conforming member, certificate `unique`, and it was the same member.
* Amendment 1's arm F fired **as pre-registered**: the wrong schema was
  indistinguishable at depth 1 and collapsed at every unseen depth, with its own
  exhaustion certifying 0 conforming. The alternative outcome named in the
  amendment — F also scoring 4.00 everywhere, which would have weakened A1/A2 —
  did not occur.

## 6. Limits, stated

* **One generator, one task.** Everything here is `logic`. The claim
  generalizes only as far as the observation shape does: a `tuple` whose arity
  is a linear function of one configuration parameter.
* **Width generalization is up to a declared capacity in the same sense §15's
  was.** The schema instantiates at any depth, but each instantiation is a
  separate, growing artifact — 231 nodes at depth 8 — and the wire index must
  fit the `int[8]` field. This is unrolling, not unbounded polymorphism.
* **The schema is not itself a typed artifact.** It is Python. Nothing in the
  system validates that a builder is semantics-preserving across widths, and
  arm F is a two-character demonstration that a broken one passes every check
  available at a single width.
* **A gradient arm was not run.** Enumeration is the primary method here by
  design (§16: `eq` is exactly 0.0 past |a−b| ≥ 11, and this scaffold's `index`
  and `encode` nodes sit well past that). No claim is made about what the
  differentiable path would do, and none should be read into the absence.
* **The 13 test failures are environmental**, verified at baseline before this
  directory existed and again with it moved out of the tree: 324 passed / 13
  failed both times.

## 7. Files

| file | what |
|---|---|
| `PREREGISTRATION.md` | committed `1e439a1` before any arm; amendment 1 at `5f8aa4f` before arms F/G/H |
| `audit.py` → `out/audit.json` | step 1: 31 cited sites, resolved by source matching, plus executable demonstrations |
| `scaffold.py` | the depth-parametric schema, and the `record` and `first_gate` controls |
| `run.py` | `baselines fit exhaust transfer record hardened crosscheck wrong_schema difficulty size` |
| `report.py` | regenerates every table above from `out/*.json`; nothing is transcribed by hand |
