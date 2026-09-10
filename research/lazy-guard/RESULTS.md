# Lazy guard: can a resynthesizer recover early exit it was not given?

Branch `research/lazy-guard`, from `main` at `ee76353` (which carries
`tcn/compile.py` from FINDINGS §48).  Nothing under `tcn/`, `generators/` or any
other track is modified.  This directory is a pure addition.

This is the §9 experiment of `research/algorithm-resynthesis/DESIGN.md`, run in
the order that document's §0.3 demands: **cost instrumentation first**, then the
pre-registered criteria, then the staged arms.

Every number below is printed by `report.py` from `out/*.json` and can be
re-checked against those files without re-running anything.

---

## 0. Headline

| | stage 1 (miniature) | stage 2 (miniature) | stage 3 (real visual S2 subroutine) |
|---|---|---|---|
| recovered | `guard` | `find` | `find`, twice |
| **expected executed primitives** | 84.0 → **16.5** = **5.09×** | 72.0 → **17.93** = **4.02×** | 1453.0 → **328.69** = **4.42×** |
| **worst-case executed primitives** | 84 → 84 = **1.00×** | 72 → 72 = **1.00×** | 1453 → 1431 = **1.015×** held out, 1453 → 1461 = **0.995×** on the synthesis domain |
| exactness | exhaustive, 65,536/65,536 | exhaustive, 65,536/65,536 | 2,883/2,883 of a *declared enumerated* domain |
| ablation | removing `Guard` alone blocks it | removing `Find` **and** `Scan` blocks it; either alone suffices | removing `Find` alone blocks it |
| supplied verbatim? | no | no | no |

**The resynthesizer recovered a lazy conditional without being given it**, and
recovered a bounded early-exit scan on the shipped visual parser's own S2
subroutine.  **The worst case does not improve, and in executed CPython
bytecodes it gets worse** — 0.67× on stage 1, 0.63× on stage 2.  Both facts are
reported below with equal weight, because a win quoted without the second is a
false win.

---

## 1. PRECONDITION — the cost instrumentation (DESIGN §0.3, §11)

`cost.py`; run by `instrument.py`; raw data `out/instrumentation.json`.

DESIGN §0.3 states the blocker exactly: `filter` is charged at declared
*capacity* (`tcn/operators.py:118`), `Program.execution_cost` is a static sum
over selected candidates, and §41 measured it constant at 148.0 across ten
programs differing by 48 executed bytecodes.  Early exit **never** improves a
worst case, so an objective that reports only a worst case cannot reward it.

The instrument reports three quantities, always separately:

* **executed primitive operations** — one application of a leaf `tcn` operator.
  A module call is not itself a primitive; the leaves inside it are.  Reported
  as *worst*, *expected over the declared distribution*, and *best*.
* **executed CPython bytecodes** — counted with `sys.monitoring`, exactly as
  `research/compiled-runtime/attribute.py` counts them.
* the old scalar, `Program.execution_cost`, alongside, so the two can be
  compared rather than conflated.

Both domains are **exhausted**, so `expected` is an exact expectation under the
declared distribution and `worst` is a true maximum, not a bound.

### 1.1 The instrument is certified, not asserted

`cost.NativeEval` is a generic decoded-value evaluator over any frozen
`Program` — it dispatches on operator name and knows nothing about either
miniature.  `certify_native` checks it against `Registry.exact`, the shipped
typed interpreter, on **every** input of the declared domain:

| miniature | domain | `NativeEval` == `Registry.exact` on |
|---|---|---|
| `sparse_guard` | 16⁴ = 65,536 | 65,536 / 65,536 |
| `find_first` | 4⁸ = 65,536 | 65,536 / 65,536 |

### 1.2 The instrument shows why it was needed

| miniature | `execution_cost` | spec primitives worst / expected / best | distinct values over the whole domain |
|---|---|---|---|
| `sparse_guard` | 84.0 | 84 / 84.0 / 84 | **1** |
| `find_first` | 72.0 | 72 / 72.0 / 72 | **1** |

The specification's executed primitive count is *identically constant* across
all 65,536 inputs — the histogram has a single bin.  That is DESIGN §0.1 made
measurable: `Program.execute` (`tcn/graph.py:105-110`) is an unconditional loop
over `self.nodes`, so there is no input on which it does less work.  Any metric
computed from the specification alone therefore cannot distinguish an early-exit
algorithm from an eager one, and **criterion 2 was literally unmeasurable before
this file existed.**

The compiled specification's bytecodes are near-constant too (998/998/1002 and
815/823/823 on three probe inputs), varying only in `frozenset` insertion.

**The precondition is met: expected and worst are separated, both are exact over
the declared domain, and bytecodes are reported alongside.**

---

## 2. PRE-REGISTRATION

Fixed before any arm was run; reproduced verbatim from DESIGN §9.

1. **Exactness** — agrees with the frozen spec on **every** input of the declared
   finite domain, exhaustively, yielding a certificate.
2. **Expected work** — ≥ **3×** fewer executed primitive operations than the
   frozen spec over the declared distribution.
3. **Not supplied** — the target algorithm does not appear verbatim in the
   candidate set or rule set; checked by ablation — removing the specific
   construct must be the only thing that prevents the result.
4. **Held out** — exactness survives structural cases not used during synthesis.
5. **Deployable** — generated standalone code runs stdlib-only and is
   bit-identical, via the `research/compiled-runtime` harness.
6. **Honest worst case** — the report states worst-case executed operations,
   which is expected **not** to improve.

**Declared distribution: uniform over the declared finite domain.**  Nothing is
weighted or tuned; the predicate's hit rate falls out of the predicate.

**Falsification, also fixed in advance:** expected gain < 3× ⇒ the gap is not
primarily laziness; result only when the construct is supplied verbatim ⇒
template instantiation, report as negative; exhaustive equivalence unobtainable
⇒ shrink the domain rather than sample; gain traceable to DCE/CSE/constant
folding ⇒ report as level 1.

---

## 3. What was built (and what was deliberately not)

`ir.py` — the algorithm IR of DESIGN §3 and nothing more:

```
Lit / Ref / Var / Prim / Call     the specification's own vocabulary
Let   name = e in body            sharing
Guard c then a else b             LAZY: only the taken branch runs
Find  v in [lo,hi) where p -> t | d
Scan  v in [lo,hi) from a0 by f   with an optional early-stop `while`
```

`Find` and `Scan` carry an explicit finite bound, so termination is structural
rather than proof-obligated.  **No Lean.  No e-graph** — DESIGN §2C explains why
a rewrite rule that turns "evaluate every position then filter" into "scan with
early exit" *is* the answer, and supplying it would be supplying the answer.

`synth.py` — two proposal engines, both generic over the grammar handed to them:

* `rewrite_search` — typed bottom-up enumeration of a **replacement for one
  binding**, chosen by expected cost among candidates exact on the example set,
  each acceptance then verified **exhaustively**, iterated to a fixed point.
  Counterexamples enter the example set permanently (CEGIS).
* `toplevel_search` — typed bottom-up enumeration of a **whole output
  expression** over components obtained by anti-unifying the specification's
  repeated sub-DAGs.

`wide_vocabulary` deliberately gives the enumerator **more** operators than the
specification contains — every universal operator `Registry.resolve` admits on
the available types — because a search restricted to the operators the answer
happens to use has been given a hint.

**Cost accounting is chosen so the worst case comes out equal, not better.**  A
`Guard` costs 1, the same as the `mux` it can replace.  One `Find` or `Scan`
iteration test costs 1, the same as the `mux` in an unrolled chain.

### 3.1 The anti-unifier

DESIGN §6 items 2 and 3 (port/constant parameterization, anti-unification), in
their smallest useful form.  It groups sub-DAGs by a structural key in which
operator `parameters` and constant leaves are holes; a group generalizes only
when every varying slot carries the *same* integer sequence and that sequence is
a contiguous run.  A varying `project` index generalizes to the dynamic `index`
operator — which the algebra already has — and only when the tuple is
homogeneous; a varying constant leaf generalizes to the loop variable.

It is not told that any miniature has "positions".  On `find_first` it returns

```
x8  index(y, i)                       [param family 0..7 -> loop variable]
x8  module:8175c6(index(y, i))        [param family 0..7 -> loop variable]
x9  Var(i)                            [const family 0..8 -> loop variable]
```

and **the loop bounds the search uses come from these families**, not from the
fixture.

---

## 4. STAGE 1 — recover a lazy conditional

`spec.py:sparse_guard`, `stage1.py`; raw data `out/stage1.json`, generated code
`out/stage1_algorithm.py`.

**The miniature.**  N = 4 positions over a 4-bit carrier; domain 16⁴ = **65,536**,
exhausted.  Per position: a cheap predicate `p_i = (x_i == 15)` (1 primitive) and
an expensive value `e_i = E(x_i)` where `E` is a registered module of **16** leaf
primitives; output `{(i, e_i) : p_i}` as a `frozenset` built with `insert` and
selected with `mux`.  Under the uniform distribution `p_i` holds at **1/16**,
against the parse's 20-of-961 ≈ 1/48 — the same shape, a little denser.

Written in today's algebra this is 24 nodes and **84 primitives on every input**.

### 4.1 The control that matters

| arm | worst | expected | best |
|---|---|---|---|
| frozen `Program`, eager | 84 | 84.0 | 84 |
| **the same program transcribed 1:1 into the IR, run call-by-need** | 84 | **84.0** | 84 |
| resynthesized | 84 | **16.5** | 12 |

Call-by-need **binding** changes nothing, because every `Prim` argument is
strict: `mux(p, j, s)` forces `j`, which forces `e`.  The gain is therefore
attributable to the `Guard` *construct*, not to the evaluation strategy.  This
is DESIGN §0.2 measured: `tcn/compile.py:471` emits `a if c else b` over SSA
names already assigned, which is syntactically lazy and semantically eager.

### 4.2 What the search found, unaided

107,260 candidates enumerated over a **320-signature** vocabulary, 5 rounds,
**3 counterexamples** — `(0,0,15,15)`, `(15,15,0,0)`, `(0,15,15,0)` — each of
which killed an unsound `insert` re-association the sampled example set had
accepted.  Accepted rewrites, in the order the greedy search took them:

```
s2 := guard(p2, j2, s1)
s1 := guard(p1, j1, s0)
s0 := guard(p0, j0, EMPTY)
s3 := guard(p3, j3, s2)
```

All four `mux` nodes became `guard`.  Nothing else changed.

### 4.3 Criteria

| criterion | result |
|---|---|
| 1 exactness | **exact on 65,536 / 65,536**, oracle `tcn.graph.Program.run`, output digests identical |
| 2 expected work | 84.0 → 16.5 = **5.09×** ≥ 3× ✓ |
| 3 not supplied | see §7 |
| 4 held out | see §8 |
| 5 deployable | `/usr/bin/python3 -I`, 2,147 bytes, `tcn`/`torch`/`numpy` absent from `sys.modules`, **bit-identical on all 65,536 cases** |
| 6 honest worst case | 84 → 84 = **1.00×**; in bytecodes **0.67×, i.e. worse** |

### 4.4 Bytecodes — where the honesty is

Sixteen control-flow classes (which positions satisfy the predicate); the count
is constant within every class on both sides, so the expectation is exact.

| | expected | worst |
|---|---|---|
| compiled specification (`tcn/compile.py`) | 1003.25 | 1007 |
| resynthesized standalone Python | **157.00** | **1507** |
| ratio | **6.39×** | **0.67× — the resynthesized code is 50% slower** |

The guarded program executes fewer bytecodes than primitives suggest when the
predicate fails (67 bytecodes when no position hits) and **more** when it always
hits, because four branch tests and four extra jumps are paid for nothing.
Early exit is not free, and on this miniature it is a 1.50× worst-case penalty.

The generated code is the hand-written shape:

```python
def run(x):
    v1 = x[3]; k2 = 15; v3 = (v1 == k2)
    k5 = frozenset(); v6 = x[0]; v7 = (v6 == k2)
    if v7:
        k9 = 0; v10 = _m0(v6); v27 = (k9, v10,); g8 = _ins(k5, v27, 4)
    else:
        g8 = k5
    ...
```

---

## 5. STAGE 2 — recover find-first from a uniformly unrolled family

`spec.py:find_first`, `stage2.py`; raw data `out/stage2.json`.

**The miniature.**  N = 8 positions over a 2-bit carrier; domain 4⁸ = **65,536**,
exhausted.  Each position calls a registered predicate module `P` of **7** leaf
primitives whose truth rate is measured (not assumed) at exactly **0.5**.  The
output is the first index satisfying `P`, else 8, expressed the only way the
algebra can express it: all eight predicates evaluated, then folded by a `mux`
chain.  **72 primitives on every input.**

This is not a local rewrite of any node.  Recovering the loop requires
recognising that eight structurally distinct sub-DAGs are one parametric family.

### 5.1 What the search found

**10,135,809** top-level candidates; 372 exact on 128 sampled inputs; the
cheapest survivor verified exhaustively on the first attempt:

```
find i in [0,8) where module:8175c6(index(y, i)) -> i else 8
```

Loop bounds `[0,8)` came from the anti-unified family's arity, not from the
fixture.  Lowered:

```python
def run(y):
    for i1 in range(0, 8):
        v3 = _ix(y, i1)
        v4 = _m0(v3)
        if v4:
            f2 = i1
            break
    else:
        f2 = 8
    return f2
```

### 5.2 Criteria

| criterion | result |
|---|---|
| 1 exactness | **exact on 65,536 / 65,536**, oracle `tcn.graph.Program.run`, digests identical |
| 2 expected work | 72.0 → 17.93 = **4.02×** ≥ 3× ✓ |
| 3 not supplied | see §7 |
| 4 held out | see §8 |
| 5 deployable | `/usr/bin/python3 -I`, 1,165 bytes, no `tcn`/`torch`/`numpy`, **bit-identical on 65,536 cases** |
| 6 honest worst case | 72 → 72 = **1.00×**; bytecodes **0.63×, i.e. worse** |

Primitive histogram: 9 ops on half the domain, 72 on 512 inputs (0.78%).

| bytecodes | expected | worst |
|---|---|---|
| compiled specification | 821.00 | 821 |
| resynthesized | **338.76** | **1303** |
| ratio | **2.42×** | **0.63×** |

*Caveat, from the raw data:* the specification's bytecode count is not perfectly
constant within a class (821 vs 822 on seven of nine classes), so its expectation
carries ±1.  The resynthesized program's count *is* constant within class,
because its control flow is exactly the first-hit index.

---

## 6. STAGE 3 — the same constructs on a real visual-parser subroutine

`stage3.py`; raw data `out/stage3.json`, generated code `out/stage3_algorithm.py`.

The subroutine is `rect_scaffold` from `research/visual-ladder/rung3_widgets.py`
— the **S2 module of the shipped 32×32 parse**, rebuilt from `rung3` with the
selections the track froze, 585 nodes, digest `c79e712699a5aa1716f3f65c`.  Its
own docstring says what it is: *"Still a fixed-depth feedforward graph; no
accumulator and no recurrence."*  It computes a widget's extent as

```
extent = 1 + sum_{k=1..31} [ t_1 and t_2 and ... and t_k ]
```

in each of two directions, where each `t_k` calls the S0 same-colour module.
FINDINGS §48 attributed 11.1× of the parse's 27.6× gap to exactly this: *"the S2
module is a fixed-depth 30-term formulation with no early exit"*.  The
hand-written reference in `research/compiled-runtime/fixtures.py` instead writes
`while x + w < W and ...: w += 1`.

### 6.1 The domain is weaker here, and is reported as weaker

The carrier is a 3,072-byte raster, so the *type's* domain cannot be exhausted
and stages 1–2's certificate is unavailable.  What is declared and then
exhausted is a **finite enumerated domain**: every interior position of every
episode in a declared seed list — 1,922 records for synthesis (train seeds 0, 1),
**2,883 held-out records** (test seeds 200, 201, 202) never seen by the search.

The oracle is also weaker by necessity.  The typed interpreter costs ~270 µs per
operator on this carrier (§48), which makes 2,883 × 1,453 operations impossible;
the oracle is therefore `tcn/compile.py`'s output, which §48 certified
bit-identical to the interpreter, and **that identity is re-established here on
a sample before it is used**.

### 6.2 What the search found

The anti-unifier recovered **14 parametric components** from the 585-node frozen
artifact, including the per-offset term family with bound `1..31`:

```
x31  and(module:b409b7(min(add(pos, mul(step_w, i)), last), pos, obs),
         lt(add(mod(idiv(pos, three), width), i), width))
```

62,034 candidates per extent, 294 exact on 96 sampled records, first survivor
exhaustively verified.  Both extents became:

```
w_extent := find i in [1,32) where not(t_w(i)) -> i else 1
h_extent := find i in [1,32) where not(t_h(i)) -> i else 1
```

Lowered, this is the reference's loop:

```python
for i7 in range(1, 32):
    v12 = 3 * i7; v13 = rec[0] + v12; v15 = min(v13, 3069)
    v18 = _m0(v15, rec[0], rec[1])          # the S0 same-colour module
    v37 = (rec[0] // 3) % 32 + i7
    if not (v18 and (v37 < 32)):
        f8 = i7; break
else:
    f8 = 1
```

### 6.3 Numbers

| distribution | records | spec worst / expected | resynth worst / expected | expected ratio | worst ratio |
|---|---|---|---|---|---|
| synthesis domain (seeds 0,1) | 1,922 | 1453 / 1453.00 | 1461 / **320.38** | **4.54×** | 0.995× |
| **held out (seeds 200,201,202)** | 2,883 | 1453 / 1453.00 | 1431 / **328.69** | **4.42×** | 1.015× |
| **corner positions only** — what the assembly's `filter` actually passes | **54** | 1453 / 1453.00 | 1431 / **585.44** | **2.48×** | 1.015× |

Exactness: **2,883 / 2,883** held-out records, no counterexample.  Deployment:
3,291 bytes of standalone Python, matching the compiled oracle on all 2,883.

**The corner row is the honest deployment number and it is below the
pre-registered 3×.**  Corners are 54 of 2,883 interior positions (**1.87%**,
against the parse's 20-of-961 = 2.08% — the sparsity the miniature was built to
mirror), but they are precisely the positions where the extent is *large*, so
the loop runs longer there than at a random interior pixel.  On the distribution
the shipped parse actually feeds this subroutine, early exit is worth **2.48×**,
not 4.4×.  Reporting only the 4.42× would be the false win criterion 6 warns of.

### 6.4 Level check (DESIGN §9, brief §9)

| | value |
|---|---|
| worst-case ratio | **0.995×** |
| expected ratio | 4.54× |
| level-1 component (the bookkeeping the loop removes) | **0.995× — none** |
| level-3 component (iterations the data lets it skip) | **4.56×** |

The worst-case ratio is 1.0 to within a rounding of the loop's own overhead, so
**none** of the gain is DCE, CSE or constant folding.  The whole of it is
iterations not run.  This is a level-3 result on this subroutine, and the same
holds for stages 1 and 2, whose worst-case ratios are exactly 1.00×.

### 6.5 A limitation the declared domain cannot close

The `else` branch of both loops came out as `Lit(1)`, an arbitrary value.  It is
**unreachable on the declared domain**: the in-bounds mask `lt(x + k, width)`
makes `t_k` false at `k = 32 - x ≤ 31` for every interior `x ≥ 1`, so the loop
always breaks.  The search had no input that could constrain it, and exactness
over an *enumerated* domain cannot notice.  Stages 1–2, which exhaust the type's
domain, have no such hole.  This is the concrete cost of the weaker domain, and
it is the argument DESIGN §10 makes for a parametric equivalence backend.

---

## 7. CRITERION 3 — the ablation matrix

Each construct removed **one at a time**, everything else unchanged.  `ablate.py`.

### Stage 1

| grammar | solved | expected ops | ratio | constructs in the result |
|---|---|---|---|---|
| full | yes | 16.5 | **5.09×** | Guard |
| **− Guard** | yes | **84.0** | **1.00×** | — |
| − Find | yes | 16.5 | 5.09× | Guard |
| − Scan | yes | 16.5 | 5.09× | Guard |
| − Let | yes | 16.5 | 5.09× | Guard |

Removing `Guard` is **the only** thing that prevents the result.  With it gone
the search still enumerates 20,300 candidates and still finds exact ones — it
just cannot find a cheaper one.  This is the literal pre-registered reading, and
it is satisfied.

### Stage 2

| grammar | solved | expected ops | ratio | result |
|---|---|---|---|---|
| full | yes | 17.93 | **4.02×** | `find` |
| − Guard | yes | 17.93 | 4.02× | `find` |
| **− Find** | yes | 20.92 | **3.44×** | **`scan i in [0,8) from 0 by (a -> i+1) while not P(index(y,i))`** |
| − Scan | yes | 17.93 | 4.02× | `find` |
| − Let | yes | 17.93 | 4.02× | `find` |
| **− Find and − Scan** | **no exact candidate at all** (9 candidates remain) | — | **1.00×** | — |

**No single removal blocks stage 2, and that is a stronger result than one
that does.**  Denied `find`, the search invented the same early exit as a
`scan` with a `while` — a *different* construct reaching the same win — and only
when both data-dependent iteration constructs are gone does the space contain no
exact non-trivial candidate.  The pre-registration's phrasing presumes a unique
culprit; what the data shows is that the necessary thing is the **capability**,
not a particular spelling of it.  `ablate.summarize` reports both verdicts
(`single_blocking`, `capability_blocking`) rather than forcing one.

### Stage 3

| grammar | solved | expected ops | ratio |
|---|---|---|---|
| full | yes | 166.08 | **4.34×** |
| − Guard | yes | 166.08 | 4.34× |
| **− Find** | **no** | — | — |
| − Scan | yes | 166.08 | 4.34× |
| − Let | yes | 166.08 | 4.34× |
| − Find and − Scan | **no** (42 candidates) | — | — |

Here `Scan` did **not** rescue the result as it did in stage 2.  That is a
property of the achieved search size, not a proof of impossibility: stage 3 ran
at `body_size=2`, where the accumulator grammar is too small to express the
step.  Reported as achieved, not as certified absence.

### The template-instantiation control

Both stage 1 and stage 2 were re-run with **the exact winning expression handed
to a grammar that lacks the construct**, as a seeded candidate.  Both reproduce
the full ratio (5.09×, 4.02×).  That is what template instantiation looks like,
and it is indistinguishable in *outcome* from discovery — which is precisely why
the unseeded runs above are the load-bearing ones.  In the reported runs
`seed_candidates` is empty; the search receives a grammar (constructs, operator
signatures, terminals) and composes the target itself.

**Verdict on criterion 3: discovery, not template instantiation** — on all three
stages the win disappears when the construct class is removed from the grammar
and survives every other single removal.

---

## 8. CRITERION 4 — held-out structural cases

`heldout.py`; raw data `out/heldout.json`.  The engines are re-run **unchanged**
on problems that differ in the number of positions, the carrier width, the hit
constant, and the module coefficients.  Every row is certified exhaustively
against the typed interpreter.

| family | variant | domain | expected | worst | exact | construct found |
|---|---|---|---|---|---|---|
| sparse_guard | n=3, hit=15, 4-bit | 4,096 | 63.0 → 12.38 = **5.09×** | 1.00× | 4,096/4,096 | `Guard` |
| sparse_guard | n=5, hit=7, 3-bit | 32,768 | 105.0 → 26.25 = **4.00×** | 1.00× | 32,768/32,768 | `Guard` |
| sparse_guard | n=4, hit=2, 3-bit, 3 coefficients | 4,096 | 60.0 → 18.00 = **3.33×** | 1.00× | 4,096/4,096 | `Guard` |
| sparse_guard | n=4, hit=0, 4-bit, **1 coefficient** | 65,536 | 36.0 → 13.50 = *2.67×* | 1.00× | 65,536/65,536 | `Guard` |
| find_first | **n=4** | 256 | 36.0 → 16.88 = *2.13×* | 1.00× | 256/256 | `Find` |
| find_first | n=6 | 4,096 | 54.0 → 17.72 = **3.05×** | 1.00× | 4,096/4,096 | `Find` |
| find_first | n=6, different predicate module | 4,096 | 54.0 → 17.72 = **3.05×** | 1.00× | 4,096/4,096 | `Find` |
| find_first | n=7, different predicate module | 16,384 | 63.0 → 17.86 = **3.53×** | 1.00× | 16,384/16,384 | `Find` |

**Exactness survives all eight, exhaustively.**  The construct is recovered in
all eight.  The *ratio* clears 3× on six of eight; the two that fall short are
exactly where the analysis says they should — the variant whose "expensive"
continuation is only 4 primitives instead of 16, and the variant with only 4
positions to scan.  The gain is a function of how much work the guard skips and
how many iterations the scan avoids, which is the correct dependency and not a
failure of the method.

---

## 9. Verdict against the pre-registration

| criterion | stage 1 | stage 2 | stage 3 |
|---|---|---|---|
| 1 exactness, exhaustive, with certificate | ✅ 65,536/65,536 | ✅ 65,536/65,536 | ⚠️ 2,883/2,883 of a **declared enumerated** domain, not the type domain |
| 2 expected work ≥ 3× | ✅ **5.09×** | ✅ **4.02×** | ✅ 4.42× held out / ⚠️ **2.48×** on the deployed corner distribution |
| 3 not supplied (ablation) | ✅ `Guard` uniquely necessary | ✅ {`Find`,`Scan`} jointly necessary, each alone sufficient | ✅ `Find` uniquely necessary at the achieved search size |
| 4 held out | ✅ 8/8 exact; 6/8 above 3× | ✅ | ✅ 3 unseen episodes |
| 5 deployable, stdlib-only, bit-identical | ✅ 65,536 cases | ✅ 65,536 cases | ✅ 2,883 records vs the compiled oracle |
| 6 honest worst case | ✅ **1.00×**, bytecodes **0.67×** | ✅ **1.00×**, bytecodes **0.63×** | ✅ **1.015×** |

**None of the pre-registered falsifications fired.**  Expected-work gain is
≥ 3× on both miniatures; the result does **not** require the construct to be
supplied verbatim; exhaustive equivalence was obtained at the chosen size for
stages 1–2 and the domain was *declared and shrunk*, not sampled, for stage 3;
and the level check attributes **none** of the gain to DCE, CSE or constant
folding — the worst-case ratios are 1.00×, 1.00× and 0.995×.

### What this does and does not establish

**It establishes** that the representational gap DESIGN §0.2 identified is real
and is the binding one: the faster algorithm is *inexpressible* in the
specification algebra, an ordinary enumerative search finds it as soon as the
algorithm IR can state it, and exhaustive execution certifies it for free on a
finite carrier.  On the shipped visual parser's own S2 subroutine the same
search turns a fixed-depth 31-term formulation into the reference's `while`
loop.

**It does not establish** that early exit is worth 11×.  On the distribution the
parse actually feeds S2 it is worth **2.48×** in executed primitives, and in
executed *bytecodes* the early-exit form is **worse** in the worst case on both
miniatures.  §48's 11.1× excess is not all extent computation; this addresses
one named part of it.

**Difficulty achieved, stated rather than requested.**  Stage 1's discovery is
shallow: once `Guard` is in the grammar, the target is a size-4 expression one
step from the specification, and the transformation is recognisably lazy code
motion — a known compiler pass, made unavailable to any within-algebra rewriter
only because the specification language has no branch to sink into.  Stage 2 is
deeper: 10.1M candidates, and the target cannot even be *stated* without
anti-unifying an eight-member family and deriving the loop bound from it.
Stage 3 is the same engine on a 585-node frozen artifact with 14 discovered
components.  The honest reading is that **stage 1 is a level-2 result and stages
2 and 3 are level-3 results.**

---

## 10. Reproduction

```bash
.venv/bin/python research/lazy-guard/instrument.py   # the precondition, ~4 min
.venv/bin/python research/lazy-guard/stage1.py       # ~20 min
.venv/bin/python research/lazy-guard/stage2.py       # ~15 min
.venv/bin/python research/lazy-guard/stage3.py 32    # ~10 min, needs research/visual-ladder
.venv/bin/python research/lazy-guard/heldout.py      # ~10 min
.venv/bin/python research/lazy-guard/resummarize.py
.venv/bin/python research/lazy-guard/report.py       # every number in this file
```

| file | what |
|---|---|
| `cost.py` | the instrumentation of §1; certified `NativeEval`, meters, `sys.monitoring` |
| `spec.py` | the two miniatures as real frozen `tcn.graph.Program`s |
| `ir.py` | the algorithm IR, its evaluator, and the lowering to standalone Python |
| `synth.py` | anti-unification, typed enumeration, CEGIS, exhaustive verification |
| `ablate.py` | the criterion-3 matrix |
| `runner.py` | shared profiling, certification and deployment |
| `stage1/2/3.py`, `heldout.py` | the arms |
| `report.py` | prints every number quoted above from `out/*.json` |

### Repository state

* **`.venv/bin/python -m pytest -q` → 336 passed, 1 failed of 337.**  The
  failure is
  `tests/test_panel_interface.py::test_panel_episode_replays_and_restores`, the
  known worktree-only replay divergence already recorded in
  `research/MERGE-QUEUE.md` and in FINDINGS §41.  The twelve further failures
  the brief warned about appear only without a `node_modules` symlink from the
  main checkout; with it, the count is 336/1, matching §41's verification
  exactly.  The symlink is gitignored.  This track adds no tests and imports
  nothing from any tested module, so removing `research/lazy-guard/` changes
  neither number — the only files outside this directory that changed are two
  `.gitignore` lines.
* **Shipped fixture reproduces exactly**: `.venv/bin/python -m tcn train
  --episodes 160` gives `initial_prediction_loss 0.248835613951087`,
  `final_prediction_loss 0.0022308224288281053`, `fully_frozen true`,
  `evaluation_mean_return 4.0`, `frozen_evaluation_mean_return 4.0` —
  **0.248836 → 0.002231 at 4/4 frozen.**
* Large generated artifacts (`out/*_inputs.json`, `out/*_deploy.py`) are
  gitignored; the reports and the generated algorithms are not.
* `research/scaffold-diagnosis/` was not touched.
