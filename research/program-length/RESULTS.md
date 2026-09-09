# Shorter programs, and the boundary: one lever is provably inert, the other is 11x

Research track `program-length`.  Branched from `origin/compiled-runtime`
(`68db06a`), not merged.  Produced with the repository `.venv` (Python 3.13.15).
`tcn/types.py`, `tcn/graph.py` and `tcn/operators.py` are untouched; the two
changed files are `tcn/synthesis.py` (one keyword, passed through to backends
that already accepted it) and `tcn/compile.py` (one recogniser at the external
boundary).  The generic interpreter remains the oracle for every equivalence
assertion.

> **Load-independent counts beside every wall clock.**  Executed CPython
> bytecodes (`sys.monitoring` INSTRUCTION events), enumerated program counts,
> node counts and `program_cost` figures do not depend on this host; milliseconds
> do, and this host is shared.  Where a wall clock was unstable it is reported
> with its spread and the count is the claim.

---

## 0. The verdict, stated first

**Q1 — a cost term does not shorten these programs, and the reason is a
certificate rather than a search failure.**  Every artifact this repository ships
was searched over a scaffold whose node count is fixed before the search starts.
Exhausting all five declared spaces and measuring every conforming program:

| artifact / stage | space | evaluated | conforming | certificate | distinct node counts | distinct bytecode counts |
|---|---|---|---|---|---|---|
| mixed, demo synthesis | 96 | 96 | **1** | **unique** | {4} | {118} |
| language, stage B | 45,375 | 45,375 | **10** | complete | **{84}** | {41,417, 41,441, **41,465**} |
| visual S0 `same` | 256 | 256 | 2 | complete | **{15}** | {1,456} |
| visual S1 `corner` | 400 | 400 | 2 | complete | **{7}** | {2,984} |
| visual S2 `rect` | 25 | 25 | **1** | **unique** | **{585}** | {60,858} |

Bytecode counts are totals over the held-out cases each stage was measured on —
1 for mixed, 24 for language, 8 for S0 and S1, 4 for S2 — and the per-case
figures are in section 3.  What matters is the *cardinality* of each set.

Three of the five have exactly one conforming program, so no preference can
change the answer.  The two that have more are **size-degenerate**: every member
has the identical node count, the identical `execution_cost` and, in the visual
stages, the identical bytecode count.  On the one artifact where the conforming
set differs at all — language, 10 programs — the whole spread is **48 bytecodes
in 41,465, or 0.116%**, and:

> **`rank='description'` returns the program with the *most* executed bytecodes
> in the conforming set.**  Its description is minimal (4,043,552 bits, the unique
> minimum) and its bytecode count is maximal (41,465).  The bytecode minimum
> (41,417) costs 8 more description bits, so description ranking selects against
> it.

That is exactly the falsification the brief named — *"a cost term that shortens
the description without reducing executed bytecodes"* — and it is reported rather
than tuned away.  Task quality is unchanged because it **cannot** change: all ten
conforming programs score **1.000** on the same 242 held-out unseen-length
episodes (majority-constant **0.5661**, random 0.500) and produce bit-identical
outputs on every compiled case.

**Q1, the part ranking cannot reach.**  Program length in this repository is set
by a scaffold parameter, not by the objective.  `rect_scaffold`'s `span` unrolls
`span - 1` prefix-conjunction terms per axis at nine nodes apiece.  Sweeping it
with the space exhausted at each value:

| span | scaffold nodes | conforming | certificate | bytecodes / call | held-out accuracy |
|---|---|---|---|---|---|
| 4 … 28 | 81 … 513 | **0** | **no conforming program exists** | — | — |
| 29 | 531 | **0** | **no conforming program exists** | — | — |
| **30** | **549** | 1 | **unique** | **14,197** | **1.000** |
| 31 | 567 | 1 | unique | 14,692 | 1.000 |
| 32 (shipped) | 585 | 1 | unique | 15,248 | 1.000 |

So a shorter program does exist — **span 30, 6.2% fewer nodes, 6.9% fewer
bytecodes, held-out max error 0.0 and accuracy unchanged at 1.000** — and span 29
is certified impossible.  Assembled into the whole parse, with the rectangle set
asserted identical to the hand-written reference on all three held-out screens:

| parse | bytecodes | ÷ hand-written | of the 27.6x |
|---|---|---|---|
| shipped, span 32 | **650,571** | **11.05x** | — |
| shortest certified, span 30 | **630,626** | **10.71x** | **3.1% closed** |

**Q2 — the boundary is 11x cheaper and every check is still performed.**  On the
computer artifact the external boundary falls from **0.5776 ms to 0.0516 ms**
(**11.20x**; three independent runs give 11.26 / 11.18 / 11.20) around a
**1.06 µs** program.  The 547x asymmetry becomes **48.8x**.  It is not a removed check: it is the
difference between performing the contract once per element inside a Python frame
and *recognising in bulk, at C speed, that the container already satisfies it*.
No check was deleted, and the honest answer to "is all of it required" is **yes —
all of it is required, and none of it needed a Python frame**.

---

## 1. A recorded premise that does not hold, checked before anything was built

The brief states that `tcn/synthesis.fit` "has **no cost term at all**".  On
`origin/compiled-runtime` that is false, and the correction matters because it
changes what had to be built.

`fit` has carried `mdl_weight` since FINDINGS section 14 was merged (section 25).
It defaults to zero and adds `mdl_weight * model.description_cost()` to the
relaxation loss — ARCHITECTURE section 8's `L_program_description`, exact at any
one-hot selection.  `SoftProgram.description_cost()`, `Program.description_bits`
and `enumerate_fit(rank=...)` in `order`/`description`/`cost` are all present and
tested (`tests/test_preference.py`, 15 tests before this track).

**What was actually missing is narrower and was verified in the source.**  `fit`
never passed `rank` to the discrete backends:

```python
result = (solve(problem) if route(problem) not in (None, 'fit')
          else enumerate_fit(program, scored, signals, registry, tolerance))
```

so `mdl_weight` reached only the gradient path, and every discrete run took the
first conforming program in enumeration order however large it was.  Since three
of the five artifacts above were searched discretely and the language artifact
records `ranked_by: order` over ten conforming programs, that gap is exactly the
one the brief's hypothesis needs closed.

### 1.1 What this branch changes in `tcn/`

* `tcn/synthesis.py`: `fit(..., rank='order')`, passed to `enumerate_fit` and to
  `solve`, recorded in the returned report, rejected for `mode='hybrid'` (which
  stops at the first conforming structure and never builds the set a ranking
  needs) and for an unknown value.  Default `'order'`, so nothing shipped moves —
  asserted by `test_ranking_defaults_to_order_so_nothing_shipped_moves`.
* `tcn/compile.py`: `_Compiler._identity_interval` and `_identity_guard`, plus
  one emitted line per homogeneous wide tuple at the external boundary.  Section 7.

Nothing else.  No operator, no type, no semantic change.

---

## 2. Q1: how the family was measured

For every stage with an exhaustible space the whole space is walked with
`tcn.search.evaluate` — the same scorer `enumerate_fit` uses — collecting *every*
conforming selection rather than the first.  Each conforming program is then
hardened, pruned and measured in the four units ARCHITECTURE section 8.1 insists
on keeping apart:

| unit | what it is | source |
|---|---|---|
| nodes / distinct operators | the pruned program's size | `Program.pruned()` |
| `description_bits` | ARCHITECTURE **size 3**, the serialized artifact | `Program.description_bits` |
| `execution_cost` | ARCHITECTURE **cost 1**, the modelled operator sum | `Program.execution_cost` |
| **bytecodes** | CPython instructions the **compiled** program executes on real held-out input | `sys.monitoring`, `tcn.compile` |

Only the last is a measurement of work.  It is the unit that connects to
`research/compiled-runtime/RESULTS.md` section 7's 11.1x, and it is the unit the
brief requires beside any claim that a program got shorter.

**Cross-check against the recorded number.**  This harness measures the shipped
language program at **41,465 bytecodes over 24 cases = 1,727.7 per case**, where
`research/compiled-runtime` records **1,727** for one case; and the shipped parse
at **650,571** against its recorded **650,567**, with arm D at **58,859** against
its recorded **58,862** and the ratio at **11.05x** against **11.1x**.  The
harness reproduces the track it is extending before it reports anything new.

---

## 3. Q1: the enumeration certificates, in full

Before = the shipped artifact, which is `rank='order'` in every case.
After = `rank='description'` and `rank='cost'`, each over the full conforming set.

### 3.1 visual — the artifact that carries the 27.6x

| | S0 `same` | S1 `corner` | S2 `rect` |
|---|---|---|---|
| declared space | 256 | 400 | 25 |
| evaluated / exhausted | 256 / yes | 400 / yes | 25 / yes |
| conforming on training | 2 | 2 | **1** |
| survivors after validation | 2 | 2 | 1 |
| certificate | complete | complete | **unique** |
| pruned nodes, every survivor | **15** | **7** | **585** |
| distinct operators | 4 or 5 | 4 | 16 |
| `description_bits`, every survivor | 22,262,976 | 44,492,816 | 256,364,352 |
| `execution_cost`, every survivor | 15.0 | 35.0 | 1,453.0 |
| **bytecodes, every survivor** | **182 / case** | **373 / case** | **15,248 / case** † |
| held-out max error | 0.0 | 0.0 | 0.0 |
| held-out accuracy | 1.000 | 1.000 | 1.000 |
| `rank='order'` picks | #0 | #0 | #0 |
| `rank='description'` picks | #0 | #0 | #0 |
| `rank='cost'` picks | #0 | #0 | #0 |
| minimum-bytecode member | #0 | #0 | #0 |
| outputs identical across survivors | yes | yes | yes |
| sweep wall clock (this run) | 15.1 s | 123.7 s | 108.9 s |

† S0 and S1 are exactly constant across all 8 held-out cases measured; S2 is
15,248 / 15,256 / 15,244 / 15,110 on its 4, and identical between the two
survivors case for case.  The row quotes the first.

S0's two survivors differ in *which* truth-table operators they use — one spells
the same Boolean function with 4 distinct operators and the other with 5 — and
that difference moves **nothing**: same nodes, same bits, same cost, same
bytecodes, same outputs.  S2, the 585-node stage, has exactly one conforming
program in its declared space.

**Where the parse's work actually is.**  Attributed, not separately monitored:
each module's standalone per-call bytecode count from the tables above times the
assembly's call count (961 interior positions, 19 rectangles on seed 200).  The
residual row is what the attribution does not account for, and it is 0.4%, which
is why the attribution is quotable.

| module | calls per parse | bytecodes / call | total | share of 650,571 |
|---|---|---|---|---|
| `corner` (S1, via S0) | **961** | 373 | 358,453 | **55.1%** |
| `rect` (S2, via S0) | 19 | 15,248 | 289,712 | 44.5% |
| assembly + boundary-free scaffolding | — | — | ~2,400 | 0.4% |

This is worth stating because it is not where one would guess: the 585-node
module is *not* the majority of the work.  Evaluating the 7-node `corner` module
at every one of 961 interior positions is, and that count is fixed by the
assembly, not by any searched choice.  It also predicts section 5.1's result
exactly — shortening `rect` from span 32 to span 30 can only remove
`19 x (15,248 - 14,197) = 19,969` bytecodes, and the measured whole-parse
difference is **19,945**.

**There is no shorter program in this family, and the search did not fail to find
one.**  That distinction is the whole of the result for the visual artifact.

### 3.2 language — the one live case, and the falsification

45,375 programs, exhausted, **10 conforming**.  Stream pinned to
`hardening='none'`, which FINDINGS section 39 established reproduces the
pre-audit stream the artifact was measured on bit-identically; the module digest
rebuilds to `module:8063993bfeee7393683e47b3`, matching `stage_b.json`.  All
figures below are on that stream and it is named beside every one of them.

| # | `symbols` / `plus` / `minus` / `answer` | nodes | `description_bits` | `execution_cost` | **bytecodes (24 cases)** | unseen accuracy |
|---|---|---|---|---|---|---|
| **0** | 99 / 1 / 3 / 12 | 84 | **4,043,552** ← min | 148.0 | **41,465** ← max | 1.000 |
| 1 | 99 / 3 / 1 / 0 | 84 | 4,043,560 | 148.0 | 41,465 | 1.000 |
| 2 | 100 / 0 / 4 / 12 | 84 | 4,043,560 | 148.0 | 41,441 | 1.000 |
| 3 | 100 / 1 / 3 / 9 | 84 | 4,043,560 | 148.0 | 41,441 | 1.000 |
| 4 | 100 / 3 / 1 / 3 | 84 | 4,043,568 | 148.0 | 41,441 | 1.000 |
| 5 | 100 / 4 / 0 / 0 | 84 | 4,043,568 | 148.0 | 41,441 | 1.000 |
| **6** | 101 / 0 / 4 / 6 | 84 | 4,043,560 | 148.0 | **41,417** ← min | 1.000 |
| 7 | 101 / 1 / 3 / 6 | 84 | 4,043,560 | 148.0 | 41,417 | 1.000 |
| 8 | 101 / 3 / 1 / 6 | 84 | 4,043,560 | 148.0 | 41,417 | 1.000 |
| 9 | 101 / 4 / 0 / 6 | 84 | 4,043,560 | 148.0 | 41,417 | 1.000 |

| | before (`rank='order'`, shipped) | after (`rank='description'`) | after (`rank='cost'`) |
|---|---|---|---|
| program returned | **#0** | **#0** | **#0** |
| nodes | 84 | 84 | 84 |
| `description_bits` | 4,043,552 | 4,043,552 | 4,043,552 |
| `execution_cost` | 148.0 | 148.0 | 148.0 |
| **bytecodes** | **41,465** | **41,465** | **41,465** |
| bytecodes of the best available | 41,417 | 41,417 | 41,417 |
| train accuracy (n=24, majority 0.625) | 1.000 | 1.000 | 1.000 |
| held-out seen lengths (n=120, majority 0.5667) | 1.000 | 1.000 | 1.000 |
| **held-out unseen lengths (n=242, majority 0.5661, random 0.500)** | **1.000** | **1.000** | **1.000** |

**Identical-output assertion.**  All ten conforming programs produce
bit-identical answers on the 24 compiled held-out cases
(`outputs_identical_across_conforming: true`), and each of the ten is 1.000 at
every length 8/10/12/14 individually.

Two things follow, and they point in opposite directions:

1. **Quality is not at risk here**, because the conforming set is a set of
   *spellings* of one function.  #0 counts with `+1/-1` and tests `acc == 2`; #6
   counts with `+2/-2` and tests `acc == 0`.  Nothing to lose.
2. **`description_bits` is anti-correlated with executed work on this set.**  The
   description-minimal program is the bytecode-maximal one.  ARCHITECTURE section
   8.1 already warns that `description_bits` measures *size 3* — the serialized
   artifact, 99.68% of which is repeated type declarations — and this is that
   warning arriving as a measurement: 4 million bits of which the ten programs
   differ by 16, deciding a choice whose real spread is 48 bytecodes.

The available prize even under a perfect bytecode oracle is **41,465 → 41,417, a
factor of 1.0012**, against a 7.1x gap to hand-written Python.

### 3.3 mixed — unique, so there is nothing to prefer

`artifacts/demo/synthesis/result.json`, regenerated on this branch: space **96**,
evaluated 96, **conforming 1**, `unique: true`, `description_bits` 17,728,
held-out max error 5.93e-08 on 584 unfitted points, agrees with enumeration.  The
compiled program is 4 nodes and 118 bytecodes.  Any ranking returns the same
program.

---

## 4. Q1: what preference costs the search

`rank='order'` and `rank='description'` walk the **same** space — a completeness
or uniqueness certificate already requires exhausting it, and every artifact here
carries one — so evaluations are identical.  The only extra work is one
`program_cost` call per conforming program, and it forbids `stop_at_first`, which
none of these searches used.

| stage | evaluations, before and after | conforming | sweep, this run | `program_cost` each | ranking overhead | fraction of the sweep |
|---|---|---|---|---|---|---|
| visual S0 `same` | 256 | 2 | 15.1 s | 48.0 ms | 96 ms | 0.64% |
| visual S1 `corner` | 400 | 2 | 123.7 s | 103.3 ms | 207 ms | 0.17% |
| visual S2 `rect` | 25 | 1 | 108.9 s | 596.1 ms | 596 ms | 0.55% |
| language stage B | 45,375 | 10 | 218.6 s | 8.1 ms | 81 ms | 0.037% |

(`out/rank_cost.json` divides by `rung3.json`'s recorded `search.seconds`
instead, which is a different and faster walk; the column above uses this
track's own measured enumeration times, so it is self-consistent.)

Ranking is essentially free on every space large enough to need a certificate,
and it buys nothing on any of them.

---

## 5. Q1: the lever that ranking cannot reach, and its price

Length here is a **scaffold** property.  `rect_scaffold(..., span)` emits, per
axis, `span - 1` terms of nine nodes each: `mul`, `add`, `min`, a `same` module
call, `add`, `lt`, `and`, the prefix `and`, and an `encode`.  `span` defaults to
`max(width, height)`.  It is a declared bound on a fixed-depth unrolling — the
same parameter appears in any feed-forward formulation of a bounded run length —
so sweeping it is a generic question, not a pixel-specific one.

Every span was exhausted (25 programs each), so a span with no conforming program
is a certificate rather than a budget failure.

| span | scaffold nodes | pruned nodes | `execution_cost` | conforming | certificate | bytecodes / call | train acc | held acc | held max err |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 81 | — | — | 0 | **none exists** | — | — | — | — |
| 8 | 153 | — | — | 0 | **none exists** | — | — | — | — |
| 12 | 225 | — | — | 0 | **none exists** | — | — | — | — |
| 16 | 297 | — | — | 0 | **none exists** | — | — | — | — |
| 20 | 369 | — | — | 0 | **none exists** | — | — | — | — |
| 24 | 441 | — | — | 0 | **none exists** | — | — | — | — |
| 28 | 513 | — | — | 0 | **none exists** | — | — | — | — |
| 29 | 531 | — | — | 0 | **none exists** | — | — | — | — |
| **30** | **549** | **549** | **1,361.0** | 1 | **unique** | **14,197** | 1.000 | **1.000** | **0.0** |
| 31 | 567 | 567 | 1,407.0 | 1 | unique | 14,692 | 1.000 | 1.000 | 0.0 |
| 32 (shipped) | 585 | 585 | 1,453.0 | 1 | unique | 15,248 | 1.000 | 1.000 | 0.0 |

The widest and tallest widget in the training and held-out episodes is **30**
pixels, reported here for the reader and not consulted by the search: span 30
covers extents to 30 and span 29 does not, which is why the certificate flips
exactly there.  **The shipped artifact is two spans longer than the shortest one
that exists, and shortening it to the minimum costs no task quality at all.**

### 5.1 Assembled: how much of the 27.6x that closes

The whole parse, both spans, output asserted identical to
`research/compiled-runtime/fixtures.py`'s hand-written arm D before any count is
reported — `frozenset == frozenset` on the six-tuples, 19 / 19 / 16 rectangles on
seeds 200 / 201 / 202, identical on all three for both spans.

| | bytecodes, seed 200 | ÷ arm D (58,859) | source bytes | program-only median |
|---|---|---|---|---|
| shipped, span 32 | **650,571** | **11.053x** | 171,232 | 5.160 ms |
| shortest, span 30 | **630,626** | **10.714x** | 162,128 | 5.209 ms |
| arm D, hand-written | 58,859 | 1.0 | — | — |

The work factor falls **11.053 → 10.714, which is 3.1% of it**.  Carrying
`research/compiled-runtime` section 7's decomposition through unchanged, the
27.6x becomes about **26.8x**.  The wall clock does not move at all — 5.160 ms
against 5.209 ms, the shorter program measuring marginally *slower* on a shared
host, which is noise and is reported as such rather than as a regression.

**Why the ceiling is this low, stated mechanically.**  Arm D's advantage is not
its length; it is an early `continue` and a `while` that stops at the first
colour change.  A fixed-depth feed-forward graph has no data-dependent exit, so
it must evaluate all `span - 1` terms at all 961 interior positions whatever the
image contains.  Shortening the unrolling can only remove terms that are
*provably never needed on any input in the declared type* — here 2 of 31 — and
the remaining 29 are needed on some input.  Closing the 11x would require a
formulation with a data-dependent exit, which is a change to the program family
(recurrence, or a fold), not to the objective that selects within it.

---

## 6. Q2: what the external boundary must check

The generated boundary for the computer artifact's 4,097-element observation, as
the compiler emitted it before this branch:

```python
def _c1(x):
    if not _isfinite(x): _nonfinite()
    n = int(x)
    if n != x: raise ValueError('fractional value requires explicit quantization')
    if not 0 <= n <= 255: raise OverflowError('%s outside [0, 255]' % n)
    return n

def _b1(x):
    if not isinstance(x, (int, float)) or isinstance(x, bool): _nonfinite()
    return _c1(x)

def _b2(x):
    if len(x) != 4096: raise ValueError('tuple arity mismatch')
    return tuple(map(_b1, x))
```

Seven obligations, each attributable to a clause of the contract:

| # | check | contract | removable? |
|---|---|---|---|
| 1 | `isinstance(x, (int, float))` | carrier admission — `TypeError` | **no** |
| 2 | `not isinstance(x, bool)` | `bool` is a distinct carrier — `TypeError` | **no** |
| 3 | `math.isfinite(x)` | finite numeric — `TypeError` | **no** |
| 4 | refinement pre-bound | `ValueError` (only where `bounds` declared) | **no** |
| 5 | `n = int(x); n != x` | fractional in an integer encoding — `ValueError` | **no** |
| 6 | encoded-range overflow | `OverflowError` / `wrap` / `saturate` | **no** |
| 7 | refinement post-bound | `ValueError` (only where `bounds` declared) | **no** |

**Nothing here is redundant, and nothing was removed.**  What *is* redundant is
performing them.  For a Python `int` inside `I = [encoded range] ∩ [refinement
bounds]`, checks 1–7 are all provably satisfied and the function returns `x`
itself, so the whole chain is a no-op on that branch — and the branch can be
recognised by one type test and one interval test.  `Type.__post_init__` caps
`bits` at 64, so the interval endpoints are exactly representable and check 3 can
never itself raise on a member.

Two further obligations are structural: the container's arity check, which is one
`len`, and re-materialisation, which is **not** an obligation at all — a Python
tuple is immutable, so a container whose every element is already canonical *is*
its own canonical form and can be returned by reference.  That is the same
argument `research/compiled-runtime` section 8 used for internal edges, applied
at the boundary.

### 6.1 The hazard that makes the guard non-obvious, and the measurement that caught it

`hash(True) == hash(1.0) == hash(1)`.  A membership or interval test *alone*
therefore accepts a `bool` and an integral `float` whose value happens to be
admissible.  Both must be declined, for two different reasons the type graph
knows:

* `Value.of` **rejects** a `bool` for a numeric type with `TypeError`;
* `Value.of` **accepts** an integral `float` and canonicalises it to `int`, so
  returning the container by reference would hand the program a non-canonical
  carrier.

`research/program-length/micro_boundary.py` measured the naive spelling —
`frozenset(range(256)).issuperset(x)` — at **0.0165 ms**, 45x faster than the
shipped boundary and **wrong**: it returns the container unchanged when element
2,000 is `True`, where every other spelling raises `TypeError`.  Adding
`set(map(type, x)) == frozenset((int,))`, a second C-level pass, closes it and
still runs at **0.0459 ms**.  Three sound spellings, on a 4,096-element uint8
tuple:

| spelling | median | correct on `True` / `1.0` |
|---|---|---|
| shipped: `tuple(map(_b1, x))` | 0.744 ms | yes |
| element-wise `all(type(v) is int and lo <= v <= hi ...)` | 0.190 ms | yes |
| `set(map(type,·)) == {int}` and `lo <= min(x) and max(x) <= hi` | 0.111 ms | yes |
| **`set(map(type,·)) == {int}` and `okset.issuperset(x)`** | **0.072 ms** | **yes** |
| `okset.issuperset(x)` alone | 0.0165 ms | **NO** — accepts `True` |

The compiler emits the last sound form where the admissible set has at most
**8,192** members and the `min`/`max` form otherwise, which carries no
cardinality cap.  8,192 is a *cold-start allocation* choice, not a correctness
one: the `frozenset` is built at import, and
`research/compiled-runtime/RESULTS.md` section 11 measures cold start at 38-90 ms
for these artifacts, which a multi-megabyte set would dominate.  It covers every
byte type and every refinement-bounded address type in the repository — `uint8`
at 256 members and the `(0, 4096)` terminal offset at 4,097 — at roughly 300 KB.  Both are keyed only on `t.kind`, `t.encoding.kind`, `t.encoding.overflow`,
`t.bits` and `t.bounds` — there is no mention of pixels, bytes, rasters or
terminals anywhere in the change, and `mixed`, whose inputs are scalars, gets no
guard at all (section 8).

---

## 7. Q2: the minimal correct boundary, measured

`before` is this same compiler with `_identity_guard` disabled, so the two arms
differ by exactly one emitted line rather than by a file checkout.  `oracle` is
`Value.of`.  Timing is reported only after equivalence passes.

**Identical-output assertion, all four artifacts.**  Program outputs identical
across before / after / interpreter on every case; the boundary's own canonical
value identical for every input port on every case; and a ten-case adversarial
suite on the widest port, corrupting one element:

| corrupted element | before | after | oracle (`Value.of`) | identical |
|---|---|---|---|---|
| `0.5` | `ValueError: fractional value requires explicit quantization` | same | same | yes |
| `1 << 40` | `OverflowError: 1099511627776 outside [0, 255]` | same | same | yes |
| `True` | `TypeError: finite numeric value required` | same | same | yes |
| `inf` | `TypeError: finite numeric value required` | same | same | yes |
| `nan` | `TypeError: finite numeric value required` | same | same | yes |
| `10**400` | `OverflowError: int too large to convert to float` | same | same | yes |
| `'x'` | `TypeError: finite numeric value required` | same | same | yes |
| `None` | `TypeError: finite numeric value required` | same | same | yes |
| `-1` | `OverflowError: -1 outside [0, 255]` | same | same | yes |
| `3.0` | accepted, canonicalised to `3` | same | same | yes |

Exception **type and message** match in all ten, including the two that are easy
to get wrong: `10**400` raises inside `math.isfinite` with a different message
than the range check would give, and `3.0` is *accepted*.  26 pre-existing
differential tests in `tests/test_compile.py` still pass, and 20 new ones in
`tests/test_boundary_guard.py` hold this line directly.

### 7.1 Cost

| artifact | widest port | boundary **before** | boundary **after** | **ratio** | program only | asymmetry before | asymmetry after |
|---|---|---|---|---|---|---|---|
| **computer** | `terminal`, 4,097 | **0.57755 ms** | **0.05155 ms** | **11.20x** | 0.00106 ms | **547x** | **48.8x** |
| **visual** | `observation`, 3,072 | 0.36896 ms | 0.03176 ms | **11.62x** | 4.9068 ms | 0.075x | 0.0065x |
| **language** | `text`, 129 | 0.01608 ms | 0.00184 ms | **8.74x** | 0.00397 ms | 4.05x | 0.46x |
| **mixed** | scalars only | 0.00034 ms | 0.00032 ms | 1.05x — noise | 0.00040 ms | 0.84x | 0.80x |

Complete path (`validate=True`), and executed bytecodes:

| artifact | complete before | complete after | ratio | bytecodes before | bytecodes after |
|---|---|---|---|---|---|
| computer | 0.57686 ms | 0.05294 ms | **10.90x** | 198,809 | **697** |
| visual | 5.21076 ms | 4.87540 ms | 1.07x | 782,682 | 650,602 |
| language | 0.02006 ms | 0.00574 ms | **3.49x** | 7,329 | **1,841** |
| mixed | 0.00062 ms | 0.00062 ms | 1.00x | **190** | **190** |

`mixed`'s boundary row differs by 20 ns on a 340 ns measurement, which is host
noise; its **bytecode counts are identical, 103 and 190**, and its generated
source is byte-for-byte the same in both arms.  That is the real statement, and
it is the control: `mixed`'s inputs are two `bool`s and a `float32`, so the type
graph offers the guard nothing and `boundary_identity_guards` is **0**.

**Stability.**  Three independent runs of the computer row gave
0.56662 / 0.57417 / 0.57755 ms before and 0.05032 / 0.05134 / 0.05155 ms after,
ratios **11.26 / 11.18 / 11.20**.  The bytecode counts are identical across runs.
The table's row is the third, which is the one recorded in
`out/boundary_computer.json`.

**The bytecode ratio overstates the win and is reported anyway.**  Boundary
bytecodes fall 198,514 → 402, a factor of 494, but the wall clock falls only
11.2x, because the work did not disappear — it moved into C, where
`set`, `map`, `type` and `frozenset.issuperset` execute no Python bytecodes.  The
wall clock is the honest figure here and the bytecode count is not; on the parse
in section 5 it is the other way round.  Which unit is load-bearing depends on
whether the work is in Python.

**For reference, the interpreter's own boundary** (`Value.of`, arm A) on the same
inputs: computer 2.928 ms, visual 1.724 ms, language 0.0748 ms, mixed 0.00203 ms
— so the compiled boundary after this change is **56.8x** cheaper than the
interpreter's on the computer artifact, **54.3x** on visual and **40.6x** on
language.

### 7.2 Attribution

`cProfile`, one complete path with the boundary on, `tottime` by function.  The
profiler inflates absolute seconds; the shares are the measurement.

| computer, **before** | share | | computer, **after** | share |
|---|---|---|---|---|
| `_b1` (per element) | **42.4%** | | `_b2` (the guard) | 42.2% |
| `_c1` (per element) | **24.1%** | | `frozenset.issuperset` | 26.4% |
| `builtins.isinstance` | 14.0% | | `_c4` (a float scalar) | 7.6% |
| `_b2` (the container) | 10.5% | | `_b7` (the 512-tuple) | 5.9% |
| `math.isfinite` | 7.2% | | `run` (the program) | 3.7% |
| **per-element work, total** | **87.7%** | | **profiled total** | **0.0069 s** |
| profiled total | 0.4278 s | | | |

Before, 87.7% of the boundary is per-element Python frames and their builtins.
After, the entire boundary is one Python frame per container plus two C-level
passes, and the profiled total falls **62x**.  On the visual artifact the same change
moves `run` — the program itself — from **37.3% to 54.1%** of the profiled path
and removes `_b0`/`_c0` (20.8% combined) entirely.

---

## 8. Q2: where it does not help, measured rather than omitted

A recogniser that fails has done work for nothing.  The computer artifact's
4,097-element port, with the observation in each shape:

| input shape | before | after | after ÷ before | returned by reference |
|---|---|---|---|---|
| canonical tuple | 0.51595 ms | **0.04472 ms** | **0.087** | inner tuple yes |
| **JSON-decoded list** | 0.51433 ms | 0.51491 ms | **1.001** | no |
| tuple of integral floats | 0.54566 ms | 0.57161 ms | 1.048 | no |
| canonical except the last element | 0.50824 ms | 0.53693 ms | 1.056 | no |

Three consequences, all of them limits on the headline:

1. **The deployment path gets nothing.**
   `research/compiled-runtime/RESULTS.md` section 11 sends every artifact through
   a JSON line, and `json.loads` produces `list`, not `tuple`.  `type(x) is tuple`
   fails on its first test, so the guard costs 0.1% and buys 0%.  The 11.2x is an
   **in-process** figure for a caller that already holds a tuple.  Making the
   deployment path benefit means changing the transport to build tuples, which is
   a separate change this track did not make.
2. **A non-canonical container costs about 5% more**, because both C-level passes
   run before failing.  That is the price of the recogniser and it is paid on
   exactly the inputs that were going to pay for a full re-encode anyway.
3. **`mixed` is unchanged in every column**, which is the control: the
   optimisation is derived from the type graph and does nothing where the type
   graph offers it nothing.

---

## 9. Negative results, preserved

**9.1 The brief's premise about `fit` is false, and was checked in the source
before anything was built.**  `mdl_weight` has been wired to
`SoftProgram.description_cost()` since section 14 merged.  Section 1.

**9.2 A cost term cannot shorten any program this repository ships, and that is
certified, not conjectured.**  Three of five spaces have one conforming program;
the other two are size-degenerate.  Sections 0 and 3.

**9.3 Description ranking selects *against* executed work on the only set where
it has a choice.**  Section 3.2.  The description-minimal program is the
bytecode-maximal one.  This is the brief's stated falsification condition and it
is met.

**9.4 The shortest program that exists closes 3.1% of the visual work gap and
0% of its wall clock.**  Section 5.1.  It cost no quality — held-out accuracy
1.000 either way, max error 0.0 — so it is worth taking, and it is nowhere near
enough.

**9.5 The boundary's 11.2x does not reach the deployment path.**  Section 8.

**9.6 The 500x asymmetry does not go away.**  It goes 547x → 48.8x.  What remains
is irreducible O(n) inspection of untrusted input: two C-level passes over 4,608
elements at about 11 ns each.  Any boundary that admits arbitrary Python objects
must look at every element at least once.

**9.7 The language artifact's 1.000 is quoted with its stream, always.**  Every
accuracy in section 3.2 is on `hardening='none'`, the pre-audit stream, because
FINDINGS section 39 established that the post-audit default produces **zero**
episodes at the training lengths and that running the frozen program on it is an
undesigned out-of-distribution measurement.  This track pins the stream
explicitly rather than inheriting the generator default.  The capability has
still never been measured on the post-audit stream and this track did not measure
it either.

**9.8 `python -m tcn demo`'s `language` step fails on this branch, and fails
identically on `68db06a`.**  It raises `ValueError: training examples required`,
which is the empty-training-split consequence of the same section 39 defect.  9 of
10 demonstrations reproduce.  Not caused here, not fixed here.

---

## 10. What this does NOT show

1. **No native backend.**  Nothing here touches the 2.25x per-bytecode residue.
2. **No new program family.**  The 11x work factor is a property of fixed-depth
   feed-forward formulation.  Whether a recurrent or folded formulation of the
   run scan would conform, and at what search cost, is untested — this track
   states the mechanism (section 5.1) and stops there.
3. **The guard covers homogeneous tuples of integer-encoded scalars.**  Sets are
   not covered; float and fixed-point elements are excluded by construction
   because their round trip is not the identity.  A wider guard is possible and
   was not built.
4. **`_IDENTITY_SET_MAX` is a start-up-allocation choice, not a correctness
   one.**  Both spellings are sound; the threshold decides which is emitted.
5. **Quality is inherited, not improved.**  `research/visual-ladder/out/rung3.json`
   records the parse recovering every rectangle on all six screens but only
   **89 of 106** parent links, with **2 of 6** trees exact.  This track changed
   neither and measured neither beyond the rectangle equivalence its own
   assertions need.
6. **Still no matched neural baseline** for visual, computer or language.
   Unchanged from section 36 and from `research/compiled-runtime` section 12.
7. **Absolute milliseconds are host-specific and this host is shared.**  Bytecode
   counts, node counts, enumerated program counts and conforming counts are not.

---

## 11. Test status

`tests/`: **324 passed, 13 failed**.  The same 13 fail on the pristine
`68db06a` in this worktree with `tcn/synthesis.py` and `tcn/compile.py` checked
out from that commit — verified by doing exactly that and re-running — so they
are environmental (`tests/test_generators.py` × 4 and
`tests/test_panel_interface.py` × 9, all failing with an empty subprocess reply
that `json.loads` rejects).  The brief predicted **one** such failure
(`test_panel_episode_replays_and_restores`); there are **thirteen**, and the
discrepancy is recorded here rather than absorbed.  Passing tests went 301 → 324:
+20 in `tests/test_boundary_guard.py`, +3 in `tests/test_preference.py`.

Shipped fixture, re-verified on this branch:
`python -m tcn train --episodes 160` gives **0.248836 → 0.0022308**,
`fully_frozen: true`, mean return **4.0 / 4**; `tcn demo --only synthesis`
gives enumeration **unique of 96 in 96**, held-out max error 5.9e-08 on 584
unfitted points.

---

## 12. Reproduce

```bash
.venv/bin/python -m pytest tests/test_compile.py tests/test_boundary_guard.py tests/test_preference.py -q
.venv/bin/python -m tcn demo --out artifacts/demo                       # mixed fixture, 96/96 unique

# Q1 -- the families
.venv/bin/python research/program-length/visual_family.py --stages s0,s1   # ~2.5 min
.venv/bin/python research/program-length/visual_family.py --stages s2 --tag visual_family_s2
.venv/bin/python research/program-length/language_family.py               # ~4 min, 45,375 programs
.venv/bin/python research/program-length/rank_cost.py

# Q1 -- the lever ranking cannot reach
.venv/bin/python research/program-length/span_sweep.py                    # spans 4..32
.venv/bin/python research/program-length/span_sweep.py --spans 29 30 31 32 --tag span_sweep_fine
.venv/bin/python research/program-length/parse_endgame.py --spans 30 32 --tag parse_endgame_30_32

# Q2 -- the boundary
.venv/bin/python research/program-length/inspect_boundary.py computer
.venv/bin/python research/program-length/micro_boundary.py
.venv/bin/python research/program-length/micro_bulk.py
.venv/bin/python research/program-length/boundary.py computer --reps 401
.venv/bin/python research/program-length/boundary.py visual --reps 21
.venv/bin/python research/program-length/boundary.py language --reps 301
.venv/bin/python research/program-length/boundary.py mixed --reps 401
.venv/bin/python research/program-length/boundary_worstcase.py computer
```

| file | what it does |
|---|---|
| `measure.py` | the four size units, including compiled-bytecode counting |
| `visual_family.py` | exhausts S0/S1/S2 and measures every conforming program |
| `language_family.py` | exhausts stage B's 45,375 and measures all 10, stream pinned |
| `rank_cost.py` | what ranking costs the search |
| `span_sweep.py` | shortens the family and certifies each length |
| `parse_endgame.py` | the whole parse against arm D, both spans |
| `inspect_boundary.py` | prints the emitted boundary verbatim |
| `micro_boundary.py`, `micro_bulk.py` | the guard spellings, including the unsound one |
| `boundary.py` | before / after / oracle, equivalence then cost then attribution |
| `boundary_worstcase.py` | what the guard costs when it does not fire |
| `table.py` | renders any `out/*.json` row set as a table |

`out/*.json` holds every figure above and is small enough to commit.
