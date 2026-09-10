# Interpreter overhead or representational cost? Compiled, and settled

Research track `compiled-runtime`.  Produced with the repository `.venv`
(Python 3.13.15) for the in-process rows and the *system* interpreter
`/usr/bin/python3` 3.12.3 with `-I` for the deployment rows — no repository on
the path, no virtualenv, no torch, no numpy, and for arms C and D no `tcn`
either.  **`tcn/types.py`, `tcn/graph.py` and `tcn/operators.py` are not
modified.**  The only addition to the package is `tcn/compile.py`, which reads a
frozen `Program` and writes Python; the interpreter is untouched and stays the
oracle.

> **The host is shared.**  Every wall clock is reported beside a load-independent
> count — typed operator applications, `encode`/`decode`/`validate_raw` element
> operations, and executed CPython bytecodes — so a reader on another machine can
> reprice every row.  Wall clocks are medians with the garbage collector
> disabled; minima are in `out/*.json`.

---

## 0. The verdict, stated first

`research/inference-cost/RESULTS.md` measured exact frozen programs at **146× to
90,400×** slower than the same function hand-written in Python, and attributed
**97.7%** of it to `Type.decode`, `Type.encode` and `validate_raw` against
**0.56%** for the operator semantics and the graph walk.  The open question was
whether that is *interpreter overhead* — removable by compiling — or
*representational cost* — inherent to carrying typed values, in which case native
codegen would not rescue it either.

**It is interpreter overhead, and it is now removed.**  Compiling the frozen
program to straight-line Python, with output asserted bit-identical to the
interpreter on every case before any timing was taken:

| artifact | arm A interpreter | arm C generated Python | A ÷ C | arm D hand-written | **C ÷ D** |
|---|---|---|---|---|---|
| **mixed** (4 operations) | 0.01358 ms | **0.000496 ms** | **27.4×** | 0.000160 ms | **3.1×** |
| **language** (164) | 2.6610 ms | **0.004176 ms** | **637×** | 0.000592 ms | **7.1×** |
| **visual** (64,346, 3,072-wide edges) | 17,254 ms | **5.137 ms** | **3,359×** | 0.1861 ms | **27.6×** |
| **computer** (23, 4,097-wide) | 15.94 ms | **0.001120 ms** | **14,232×** | — | — |

Three consequences, in order of how much they matter.

**1. The measured cost was interpreter overhead, essentially all of it.**  On the
parse, `encode`/`decode`/`validate_raw` perform **103,487,972** element
operations in arm A and **6,144** in arm C — and every one of arm C's 6,144 is at
the external boundary; internal frozen edges perform **zero**.  Cost per typed
operator application falls from **268 µs** to **81 ns**, a factor of 3,300.  The
profile attribution inverts: 97.7% typed-value layer / 0.60% operator semantics
becomes 24.4% boundary encode-and-validate / 66.7% straight-line operator work.

**2. The residue is low single-digit to low double-digit, and it is *work*, not
representation.**  Counting executed CPython bytecodes with `sys.monitoring`:

| artifact | arm C bytecodes | arm D bytecodes | work ratio | time ratio | ns/bytecode C | ns/bytecode D |
|---|---|---|---|---|---|---|
| mixed | 118 | 26 | 4.5× | 2.6× | 3.9 | 6.8 |
| language | 1,727 | 219 | 7.9× | **7.1×** | 2.4 | 2.6 |
| visual | 650,567 | 58,862 | 11.1× | 24.9× | 8.0 | 3.5 |

(Bytecode counts and the time ratio beside them come from the same `attribute.py`
run, so they differ by a few percent from `bench.py`'s ratios in §2-§4.)

For language the two ratios agree to within 11% and the *per-bytecode* cost of
generated code is **below** the hand-written reference's.  There is no per-value
overhead left to remove: arm C is slower than arm D because the frozen program
executes more elementary operations than the hand-written parse does, which is a
property of the program the search found, not of how values are carried.

**3. Native compilation is therefore a second-stage engineering question, not a
gamble — and on this evidence not the next thing to do.**  Two of three artifacts
land inside 10×; the third lands at 27.6×, of which 11.1× is extra work and 2.25×
is per-bytecode cost that a native target would attack.  A native backend could
plausibly buy the ~8 ns/bytecode, i.e. roughly one order; it cannot buy the 11×
work factor, because that is what the program says to compute.  The cheaper
lever, by a wide margin, is a search that finds shorter programs.

Section 10 states what this does *not* show, at the same volume.

---

## 1. The four arms

All four run **the same frozen program** on **the same inputs**.

| arm | what it is | representation on an internal edge |
|---|---|---|
| **A** | the generic interpreter exactly as shipped: `Value.of` → `Program.run` → `.decoded` | a `Value`, validated on construction, re-decoded on every read |
| **B** | the same interpreter with three caches monkeypatched in for the duration of a `with` block and undone after: memoized `Value.decoded`, no revalidation of interpreter-produced carriers, interned `Type.from_dict` | a `Value`, decoded once |
| **C** | Python source emitted by `tcn/compile.py`, executed as an ordinary module | a native `bool`/`int`/`float`/`tuple`/`frozenset`, passed by reference |
| **D** | the same function written directly in plain Python | native values |

Arm B is `research/inference-cost/RESULTS.md` §3.1's two-cache experiment, plus
§4.3.1's type interning, reproduced here so the "cheap fix" and the "compile it"
answers can be read off the same table.

Arm D is the honest hand-written reference: it takes and returns the **same
native values from the same typed observation** as arm C.  For `language` that
meant rewriting the reference to read the 129-element prompt tuple rather than a
pre-parsed Python string; the earlier track compared against `common.balanced` on
an already-extracted string, which is not the same function of the same input.

---

## 2. mixed — the control, where little is available to gain

4 typed nodes, 4 operators, 4 operator applications, widest declared value **1
element**.  `answer = sin(xor(a, b) + x)` at `floating(32)`.

| | arm A | arm B | arm C | arm D |
|---|---|---|---|---|
| batch-one warm, program only | 0.013584 ms | 0.013168 ms | **0.000496 ms** | 0.000160 ms |
| batch-one warm, complete path | 0.016192 ms | 0.015008 ms | **0.000720 ms** | 0.000160 ms |
| external boundary (encode + decode) | 0.002608 ms | 0.001840 ms | 0.000224 ms | — |
| ratio to arm D, program only | 84.9× | 82.3× | **3.1×** | 1.0 |
| peak Python allocation | 0.00526 MB | 0.00526 MB | 0.00047 MB | 0.00023 MB |
| operator applications | 4 | 4 | 4 | — |
| encode/decode/validate element ops | 29 | 18 | **4** (1 boundary, 3 internal) | 0 |
| bytecodes executed | 2,880 | 2,551 | **118** | 26 |
| static nodes / distinct operators | 4 / 4 | 4 / 4 | 4 / 4 | — |
| widest declared live value | 1 element | 1 | 1 | 1 |
| artifact bytes on disk (`.pyz`) | 45,322 | 45,322 | **1,713** | 876 |
| artifact bytes gzipped | 12,140 | 12,140 | 1,697 | 819 |

**Identical-output assertion.**  Arm B and arm C reproduce arm A **exactly**
(`==` on the decoded output) on all 4 cases, and 2,000 further random `(a, b, x)`
draws agree bit-for-bit including the error contract: `x = 1e39` and `x = 1e300`
raise `OverflowError('float too large to pack with f format')` in both arms,
`x = inf` raises `TypeError('finite numeric value required')` in both, and
`x = 3.4e38` succeeds in both.  Arm D is **not** bit-identical: it computes in
float64 while the program's declared output type is `floating(32)`, so it differs
by **1.5e-9 to 2.6e-8** on the four cases.  That is a declared-type difference, not
an error — `research/inference-cost/RESULTS.md` §4.1's "max abs error 0.0" for
this row is not reproduced and is corrected here.

The three internal canonicalisations arm C keeps are real: the program declares a
32-bit float carrier, so each of the three float-producing nodes must round to
float32, and that is what `_c1` does.

---

## 3. language — 164 operations over a 129-element prompt

84 caller nodes plus one registered module (89 static nodes), 9 distinct
operators, 164 operator applications, widest declared value **129 elements**.

| | arm A | arm B | arm C | arm D |
|---|---|---|---|---|
| batch-one warm, program only | 2.660991 ms | 1.504511 ms | **0.004176 ms** | 0.000592 ms |
| batch-one warm, complete path | 2.739342 ms | 1.550478 ms | **0.021328 ms** | 0.000592 ms |
| external boundary | 0.078351 ms | 0.045967 ms | 0.017152 ms | — |
| ratio to arm D, program only | 4,495× | 2,541× | **7.1×** | 1.0 |
| peak Python allocation | 0.0492 MB | 0.0475 MB | 0.0015 MB | 0.00018 MB |
| operator applications | 164 | 164 | 146 (18 folded) | — |
| encode/decode/validate element ops | 13,556 | 4,618 | **276** (259 boundary, 17 internal) | 0 |
| bytecodes executed | 755,329 | 368,590 | **1,727** | 219 |
| static nodes / distinct operators | 89 / 9 | 89 / 9 | 86 emitted, 3 folded | — |
| widest declared live value | 129 elements | 129 | 129 | 129 |
| artifact bytes on disk (`.pyz`) | 1,557,558 | 1,557,558 | **5,585** | 1,183 |
| artifact bytes gzipped | 31,677 | 31,677 | 5,179 | 1,144 |

**Identical-output assertion.**  Arms B and C reproduce arm A **exactly** on all
12 held-out episodes.  Arm D agrees with arm A on 9 of 12 — see §9.1, which is a
fixture defect this track reproduced rather than anything the compiler did.

Note the shape of arm C's complete path: **0.0042 ms of program inside 0.0213 ms
of typed boundary**.  Once the interpreter is out of the way, validating the
observation costs four times what computing on it costs.

---

## 4. visual — 64,346 operations with 3,072-wide edges

This is where the thesis lives.  4 caller nodes over 3 registered modules
(611 static nodes), 24 distinct operators, **64,346 operator applications**, and
a `records` set whose declared width is **2,954,114 elements** — 961 positions
each paired with the whole 3,072-byte raster.

| | arm A | arm B | arm C | arm D |
|---|---|---|---|---|
| batch-one warm, program only | 17,254.14 ms | 4,665.42 ms | **5.137 ms** | 0.18611 ms |
| batch-one warm, complete path | 17,255.95 ms | 4,666.49 ms | **5.569 ms** | 0.18611 ms |
| external boundary | 1.8146 ms | 1.0647 ms | 0.4327 ms | — |
| ratio to arm D, program only | 92,708× | 25,068× | **27.6×** | 1.0 |
| µs per operator application | 268.1 | 72.5 | **0.0814** | — |
| peak Python allocation | 48.39 MB | 49.09 MB | **0.121 MB** | 0.0068 MB |
| operator applications | 64,346 | 64,346 | 63,130 (1,216 folded) | — |
| encode/decode/validate element ops | **103,487,972** | 18,212,868 | **6,144** (6,144 boundary, **0 internal**) | 0 |
| bytecodes executed | not traceable (§6) | not traceable | **650,567** | 58,862 |
| static nodes / distinct operators | 611 / 24 | 611 / 24 | 547 emitted, 64 folded | — |
| widest declared live value | 2,954,114 elements | 2,954,114 | 2,954,114 | 2,954,114 |
| artifact JSON bytes on disk | **123,384,155** | 123,384,155 | **171,074** (`.py`), 22,617 (`.pyz`) | 1,423 |
| artifact bytes gzipped | 586,925 | 586,925 | 21,069 | 1,374 |

**Identical-output assertion.**  On all three held-out screens (seeds 200, 201,
202) arms B, C **and** D produce the byte-identical rectangle set — the same 19,
19 and 16 six-tuples `(x, y, w, h, key, parent_key)` as arm A, compared as
`frozenset == frozenset`.  Timing was recorded only after that assertion passed;
`bench.py` refuses to time an arm that differs.

**Arm A's absolute wall clock is the least stable number here.**  Four
independent runs on this host gave 15,006, 16,615, 17,254 and 18,648 ms;
`research/inference-cost` measured 15,364 ms on the same fixture.  Arm C over the
same four runs gave 5.120, 5.185, 5.137 and 5.158 ms — a 1.3% spread.  The A ÷ C
factor is therefore **2,900× to 3,600×**, and the load-independent counts
(103,487,972 element operations against 6,144) do not move at all.  The table
above is the last of the four, which is the run recorded in `out/bench_visual.json`.

### 4.1 The 2.95-million-element value, and why arm C never materializes it

`pair(positions, held)` builds 961 records, each a `(position, observation)`
pair.  In the interpreter every one of those records is a `Value`, so `Value.of`
**re-encodes all 961 × 3,072 raster elements and `validate_raw` walks them
again**, and every subsequent `.decoded` on a record decodes 3,072 elements to
read one byte.  In arm C the same `pair` is

```python
v3 = frozenset((v4, v5) for v4 in _k0 for v5 in v2)
```

— 961 tuples each holding a **reference** to the one immutable 3,072-element
observation.  Peak Python allocation for the whole parse falls from **48.4 MB to
0.121 MB**, a factor of 400, and that number is the direct measurement behind the
representation decision in §7.

---

## 5. computer — the widest observation, program only

The frozen agent program loaded from `research/computer-capability/out/agent_program.json`:
23 typed nodes, 14 distinct operators, 23 operator applications, a 4,097-element
terminal observation and a 512-element action tail.  There is **no arm D**: the
plain-Python reference in `research/inference-cost/refs.py` decides between three
action templates from a parsed document rather than from this program's typed
observation, so it is not the same function of the same input and would not be an
honest reference.  The live OS round trip is not modelled here at all; it is
967.85 ms and it is measured in `research/inference-cost/RESULTS.md` §2.

| | arm A | arm B | arm C |
|---|---|---|---|
| batch-one warm, program only | 15.9399 ms | 11.7457 ms | **0.001120 ms** |
| batch-one warm, complete path | 19.2495 ms | 13.4628 ms | **0.586433 ms** |
| external boundary | 3.3096 ms | 1.7170 ms | 0.5853 ms |
| peak Python allocation | 0.1297 MB | 0.1548 MB | 0.0407 MB |
| operator applications | 23 | 23 | 22 (1 folded) |
| encode/decode/validate element ops | 48,856 | 20,079 | **9,235** (9,228 boundary, 7 internal) |
| widest declared live value | 4,097 elements | 4,097 | 4,097 |
| artifact JSON bytes on disk | 12,150,388 | 12,150,388 | **30,393** (`.py`) |
| artifact bytes gzipped | 58,667 | 58,667 | 2,652 |

Arms B and C reproduce arm A exactly on all three synthetic observations.

**This artifact makes the boundary visible in isolation.**  The frozen program is
**1.12 µs**; validating one typed observation at the external boundary is
**585 µs**, 523× the program.  For any artifact with a wide input and a small
program, the cost after compilation is *entirely* the typed boundary, and that is
the next thing worth engineering — not the program.

The synthetic observation is of the declared type with a plausible non-empty
document; a zero-length terminal correctly drives `pos := sub(length, 1)` below
its declared bound `(0, 4096)` and both arms raise `ValueError` at that same node,
which is itself an error-contract check.

---

## 6. Profile attribution, before and after

`cProfile` over one screenshot parse, `tottime` grouped by role.  Arm A's roles
are exactly `research/inference-cost/profile_path.py`'s, so the "before" column is
directly comparable with that track's table.  The profiler inflates absolute
seconds; the shares are the measurement.

| role | **arm A** | **arm B** | |
|---|---|---|---|
| `Type.decode` (types.py:171,175) | 37.36 s — **61.15%** | 4.39 s — **24.15%** | |
| `Type.encode` (types.py:134,142) | 10.74 s — **17.58%** | 10.93 s — **60.12%** | |
| `validate_raw` (types.py:185) | 5.88 s — **9.63%** | 0.00 s — 0.00% | |
| numeric/type guards called by both | 3.45 s — 5.64% | 2.43 s — 13.37% | |
| `Type`/`Value` dict round-trip | 2.26 s — 3.69% | 0.00 s — 0.00% | |
| operator semantics + graph walk | 0.37 s — **0.60%** | 0.24 s — **1.34%** | |
| other | 1.04 s — 1.71% | 0.19 s — 1.02% | |
| **typed value layer, total** | **97.7%** | **97.6%** | |
| profiled total | 61.09 s | 18.18 s | |

Arm A reproduces the prior track's 97.7% / 0.60% split exactly.  **Arm B does not
change the attribution at all** — caching removes 4× of wall clock and leaves the
typed value layer at 97.6%, with `Type.encode` simply promoted from second place
to first.  That is the decisive negative for the cheap fix: memoizing decode
cannot help with encoding, because `Registry.exact` ends every operator by
constructing a fresh `Value`, and no cache can skip work that has not happened
yet.

| role | **arm C** |
|---|---|
| straight-line operator work (`_m*`, `run`, generator expressions) | 6.28 ms — **66.71%** |
| boundary encode/validate (`_b*`) | 1.60 ms — 17.03% |
| surviving scalar canonicalisers (`_c*`, all reached from the boundary) | 0.69 ms — 7.33% |
| builtin guards (`min`/`max`/`len`/`sorted`) | 0.21 ms — 2.23% |
| other | 0.63 ms — 6.71% |
| **typed value layer, total** | **24.4%, all of it at the external boundary** |
| profiled total | 9.41 ms |

| role | **arm D** |
|---|---|
| hand-written body | **95.72%** |
| other | 4.28% |

---

## 7. Where the residual arm-C-versus-arm-D gap goes

The brief asks: if generated Python stays more than 10× slower than hand-written,
profile *that* gap, because it would mean the cost is representational and native
codegen would not rescue it.

Two artifacts land inside 10× (mixed 2.9×, language 6.9×).  One does not: the
parse, at 27.5×.  So the parse's gap was profiled, with `sys.monitoring`
INSTRUCTION events counting every bytecode each arm executes — a load-independent
unit finer than an operator application.

| artifact | C bytecodes | D bytecodes | **work ratio** | time ratio | ns/bytecode C | ns/bytecode D |
|---|---|---|---|---|---|---|
| mixed | 118 | 26 | 4.5× | 2.6× | 3.93 | 6.77 |
| language | 1,727 | 219 | 7.9× | 7.1× | 2.37 | 2.63 |
| visual | 650,567 | 58,862 | **11.1×** | 24.9× | 7.97 | 3.54 |

Read three ways:

* **Language settles the representational question.**  Work ratio 7.9× against a
  time ratio of 7.1×, and generated code is *cheaper per bytecode* than the
  hand-written reference (2.37 ns against 2.63 ns).  There is no per-value
  overhead left; the arm-C program is slower only because it performs more
  elementary steps.
* **The parse's 27.5× decomposes as 11.1× work × 2.25× per-bytecode cost.**  The
  11.1× is the program: `research/visual-ladder`'s S2 module is a *fixed-depth*
  formulation — 30 prefix-conjunction terms per axis, evaluated at every corner,
  with the `corner` module evaluated at all 961 interior positions and no early exit — whereas the
  hand-written parse breaks out of its run scan as soon as the colour changes.
  No representation change and no target language removes that; only a search
  that finds a shorter program does.
* **The 2.25× per-bytecode residue is real and is the only part a native backend
  would attack.**  Its two identified components are Python call frames
  (the `same` module `_m1` is invoked 3,100 times per parse, the `corner` module
  `_m0` 961 times and the `rect` module `_m2` 19 times, at roughly 40-60 ns a
  call, ≈4% of arm C's 5.16 ms) and the emitted range and index guards, which are
  two extra comparisons per arithmetic and indexing node.  Arm C's mean
  **7.97 ns per bytecode** against arm D's 3.54 ns is the whole of it — an
  ordinary CPython dispatch cost, not a typed-value cost.

**Arms A and B are not bytecode-traceable on the parse.**  At ~1 µs of monitoring
overhead per instruction, arm A's run would take hours.  Their comparable
load-independent figure is the element-operation count in §4: 103,487,972 against
arm C's 6,144, a factor of 16,844.

---

## 8. The representation decision, justified from measurement

The brief asks whether large structured values should be immutable references,
typed views or slices, or indexed carriers, rather than materialized copies.

**Immutable references.  No view object, no slice type, no index carrier.**  The
justification is one property plus one measurement.

The property is that `encode(t, decode(t, raw)) == raw` for any valid carrier, so
`decode ∘ encode` is *idempotent on canonical values*.  Every SSA name in
generated code is canonical by construction, which makes re-encoding a value that
only *rearranges* already-canonical parts provably a no-op.  So `tuple`,
`project`, `index`, `mux`, `identity`, `interpret`, `member`, the whole set
algebra, `map`, `filter` and a module call emit **no** encode, decode or validate
at all — only the O(1) checks that can genuinely fail (set capacity, a dynamic
index).  Only an operator that computes a *new* scalar re-applies the encoding,
and for a statically known scalar type that is one or two inline comparisons.
Across the parse, 89 structural canonicalisations are elided and 266 scalar ones
are kept, and the dynamic count of internal encode/decode/validate operations is
**zero**.

The measurement is §4.1: a Python tuple is already immutable, so 961 records can
share one 3,072-element observation for free, and peak allocation for the parse
falls **48.4 MB → 0.121 MB**.  A view or slice type would add an object and an
indirection per edge to buy a property — O(1) element access with free structural
sharing — that the language's own immutable tuple already provides.  An indexed
carrier would buy the same property at the cost of making the generated code stop
looking like the computation.  Neither is worth its complexity here, and the
400× allocation reduction is the evidence.

The one place a different representation *would* pay is the external boundary
(§5: 585 µs of validation around a 1.12 µs program), and the fix there is not a
view but a cheaper boundary encoder — which is engineering on a path this track
has now isolated.

---

## 9. What `tcn/compile.py` does, and the exact scope of its equivalence

Pure addition.  `compile_program(program, registry) -> CompileResult` with
`.source`, `.provenance`, `.types`, `.stats`.  Pipeline:

1. **validate once** — `Program.validate` against the registry, then reject
   anything not fully crystallized;
2. **prune** — `Program.pruned()`, the same normalization `save_program` already
   applies to every shipped artifact;
3. **intern types** — one canonical table keyed by the type's own canonical JSON,
   with nested types stored *by index*.  On the parse this takes the emitted
   source from 1,750,054 bytes to **171,074**, because `Type.to_dict` otherwise
   writes a 3,072-field tuple as 3,072 separate objects at every mention — the
   same fault `research/inference-cost` §5.1 measured as 99.68% of the exported
   artifact;
4. **lower to straight-line SSA** — one destination name per node, in the
   program's own topological order, with module bodies emitted as Python
   functions so module boundaries stay observable;
5. **constant-fold and DCE where exact** — a node whose sources are all constants
   is evaluated at compile time *through the interpreter itself*, so folding
   cannot disagree with it; a node that raises is caught and left in place.
   64 nodes fold on the parse, 3 on language, 1 on computer.  Post-fold dead-code
   elimination can only remove nodes that became unreachable because their
   consumers folded, and those were themselves constants;
6. **emit** — with each statement carrying its typed node and operator as a
   source comment and as an entry in `PROVENANCE`, keyed by the generated
   variable name and referencing `TYPES` by index.

Generated code for the parse's `same` module, verbatim:

```python
    # module:b409b71ea0e41d77abb7cb9c/b_g := add(b, one)
    v21 = v14 + _k7
    if not 0 <= v21 <= 65535: _ovf(v21, 0, 65535)
    # module:b409b71ea0e41d77abb7cb9c/b_g_v := index(obs, b_g)
    if not 0 <= v21 < 3072: raise IndexError('index outside tuple')
    v24 = v15[v21]
    # module:b409b71ea0e41d77abb7cb9c/cmp_g := eq(a_g_v, b_g_v)
    v27 = v19 == v24
    # module:b409b71ea0e41d77abb7cb9c/rg := truth_7(cmp_r, cmp_g)
    v29 = _k9[2 * v26 + v27]
```

### 9.1 What is preserved, and what is not

**Preserved exactly, and tested** (`tests/test_compile.py`, 26 tests, each
asserting the generated module returns the identical value *or* raises the
identical exception type at the identical edge):

* `OverflowError` on integer range, on float32 packing, and on set capacity;
* all three overflow policies — `error`, `wrap`, `saturate` — round-tripping
  identically at the declared width;
* `ValueError` on a fractional value in an integer encoding, a zero denominator,
  a shift outside the bit width, an empty reduction, and a refinement bound
  violated either before or after encoding;
* `IndexError` on a dynamic index outside a tuple;
* `TypeError` on a non-finite intermediate, at the same edge;
* fixed-point quantization and float32 rounding at every edge that declares them;
* module boundaries: each registered module becomes its own Python function,
  named in `PROVENANCE` by its content digest, and inlining through it is visible
  in the source comments;
* semantic type, representation and operator stay distinct: `interpret` compiles
  to an identity on the value because `Registry.resolve` already requires it to
  preserve carrier, encoding, unit, frame and bounds and to move only `role` —
  the compiler does not conflate them, it *uses* the distinction to prove the
  no-op.

**Not re-checked on internal edges, by design and stated plainly:** errors that
`Program.validate` proves unreachable on a frozen program — an operand of the
wrong carrier kind, a tuple of the wrong arity, a constant `project` index
outside a statically known tuple.  Values crossing an *input* port are validated
in full, exactly as `Value.of` does, including the `isinstance` guard.

**No domain-specific fast path exists anywhere in `tcn/compile.py`.**  There is no
mention of pixels, rasters, text, bytes, widgets or terminals; every optimization
is keyed on the generic frozen program and type graph — the operator name, the
declared encoding, whether a source is a compile-time constant, and whether the
output type is a scalar.  The parse's 3,072-wide handling comes out of the
generic tuple rule.

**Generated artifacts run with no torch and no `tcn` installed.**  `import` lines
in the generated module are limited to `math`, `cmath` and `struct`, asserted by
a test, and §11's deployment rows execute every artifact under
`/usr/bin/python3 -I` with nothing else on the path.

---

## 10. Negative results and defects reproduced, reported plainly

**10.1 The cheap fix does not deliver, and here is the number.**  Arm B is the
two-cache-plus-interning fix that `research/inference-cost` §3.1 and §4.3.1
recommended, and it is worth **3.70×** on the parse (17,254 → 4,665 ms),
**1.77×** on language, **1.36×** on computer and **1.03×** on mixed.  It leaves
the parse **25,068×** slower than hand-written Python and leaves the profile
attribution unchanged at 97.6% typed value layer (§6).  Memoizing `decoded`
cannot touch `Type.encode`, which is 60% of arm B, because every operator ends by
constructing a new `Value`.  **A cache cannot fix a cost that is paid on
construction.**

**10.2 Interning `Type` is a no-op on a live program and a real fix on a reloaded
one, and now there is a count.**  Distinct `Type` objects against distinct
canonical types, measured on each fixture as it is actually built:

| artifact | how it is built | distinct `Type` objects | distinct canonical types |
|---|---|---|---|
| visual | live, in memory | 20 | 13 |
| language | live, in memory | 6 | 6 |
| mixed | `load_program` | 15 | 2 |
| computer | `load_program` | **28,318** | **15** |

`Type.from_dict` allocates a fresh object per occurrence, so a reloaded artifact
carries thousands of structurally equal, non-identical types and
`Registry.exact`'s opening `tuple(v.type for v in args) != op.inputs` becomes a
deep compare.  This confirms §4.3.1's mechanism directly with an object count
rather than a timing.  It is moot for arm C, whose type table is interned at
compile time by construction.

**10.3 The `language` fixture does not reproduce its own recorded accuracy, and
this track reproduces the failure rather than papering over it.**  Rebuilding
stage B exactly as `final_eval.py` does — same `build_module()`, whose module
digest `module:8063993bfeee7393683e47b3` and cost 5.0 match `stage_b.json`
exactly; same `scaffolds.stage_b`; same 84 scaffold nodes; the same selection,
verified to be the same indices *by name* from `stage_b.json`'s `selected` record
as well as by its `enumeration.selections` — yields **0.44** accuracy on 300
held-out unseen-length episodes where `final_eval.json` records **0.9986**.  The
rebuilt program answers `True` on every one of the twelve cases used here.
`research/inference-cost` hit the same wall: its own `out/inproc.json` records
`all_agree: false` while its RESULTS.md §4.1 states the program "agrees with the
program and the label on 12/12".  The defect is upstream of this track, is
independent of the compiler, and does not affect anything measured here: arms A,
B and C are bit-identical on the same program and the same inputs, and the
comparison is a cost comparison either way.  It should be fixed by whoever owns
that track.

**10.4 `mixed`'s hand-written reference cannot be bit-identical.**  Reported at
its true discrepancy, 1.5e-9 to 2.6e-8, in §2, rather than as the "0.0" the prior
track recorded.

---

## 11. Deployment: what each arm actually ships

`/usr/bin/python3 -I` 3.12.3, nothing else on the path.  `cold start` is the
median of five spawns with **empty stdin**, so no inference runs: process spawn,
import, artifact load.  `inference` is amortized as the *difference* between a
2N-line and an N-line batch on one warm process, so spawn and load cancel rather
than being subtracted noisily; it **includes** parsing the JSON input line and
serializing the JSON reply.  Peak RSS is the child's own, via `/usr/bin/time -f %M`.

| artifact | arm | bytes on disk | gzip | cold start | amortized inference | peak RSS |
|---|---|---|---|---|---|---|
| mixed | A (`export_executable`) | 45,322 | 12,140 | 83.3 ms | 0.0333 ms | 18.5 MB |
| mixed | **C** | **1,713** | 1,697 | **38.4 ms** | **0.00464 ms** | **10.8 MB** |
| mixed | D | 876 | 819 | 36.8 ms | 0.00650 ms | 10.5 MB |
| language | A | 1,557,558 | 31,677 | 139.7 ms | 2.612 ms | 23.4 MB |
| language | **C** | **5,585** | 5,179 | **39.4 ms** | **0.0412 ms** | **13.2 MB** |
| language | D | 1,183 | 1,144 | 54.8 ms | 0.0684 ms | 10.8 MB |
| visual | A | **123,424,313** † | 586,925 † | 5,179 ms † | 66,557 ms † | 377.5 MB † |
| visual | **C** | **22,617** | 20,152 | **90.0 ms** | **18.72 ms** | **30.2 MB** |
| visual | D | 1,423 | 1,374 | 46.5 ms | 1.108 ms | 13.0 MB |

† inherited from `research/inference-cost/RESULTS.md` §4.3, not re-measured here;
building the 117.7 MB `.pyz` and running its 66.6 s inference was judged not worth
the budget once §4 had settled the in-process question.  The `123,384,155`-byte
`program.json` inside it *was* re-measured here, in §4.

Three things a reader should take from this table.

* **The 117.7 MB envelope becomes a 22.6 KB zipapp**, a factor of **5,458**, and
  peak RSS falls from 377.5 MB to 30.2 MB.  §5.1 of the prior track called for a
  shared type table; §9 step 3 is that table, and this is what it buys.
* **Transport, not the program, dominates arm C at deployment.**  Visual arm C is
  5.60 ms in process and 18.72 ms through the `.pyz`; the difference is parsing
  13,964 bytes of JSON and rebuilding the typed value from it.  For mixed and
  language, arms C and D are within noise of each other precisely because both
  are transport bound.
* **Both C and D beat arm A's cold start and RSS**, and arm C keeps the
  dependency-free property intact: no torch, no numpy, no repository, no `tcn`.

---

## 12. What this does NOT show

At the same volume as the results, because the credibility of the claim depends
entirely on its scope.

1. **This is not a native compiler and does not measure one.**  Arm C is Python.
   The 2.25× per-bytecode residue in §7 is what a native backend would attack;
   nothing here demonstrates that it would get it.
2. **The programs are the same small, specialised, synthetic ones.**  Every
   caveat in `research/inference-cost/RESULTS.md` §7 applies unchanged: a 32×32
   flat-fill screen, a 46-operation agent on a one-line file, a Dyck-word
   grammaticality check.  Making them 3,000× cheaper does not make them general.
3. **Quality is untouched and unmeasured here.**  Arm C computes exactly what
   arm A computes, so it inherits the parse's 173/215 parent links and the
   language program's reproduction failure (§10.3) in full.  A fast wrong answer
   is measured nowhere in this document.
4. **Still no matched neural baseline.**  This track did not build one.  "Cheaper
   than a model" remains unsupported for visual, computer and language, exactly as
   before.
5. **`computer` has no arm D**, so no hand-written ratio is claimed for it, and
   its observations are synthetic values of the declared type rather than live
   kernel output.
6. **The compiler covers the operators these four artifacts use, plus the rest of
   the registry under test, but it is not proven exhaustive.**  It raises
   `KeyError` on an operator it cannot compile rather than emitting something
   approximate, and `tests/test_compile.py` exercises the logic, comparison,
   arithmetic, analytic, reduction, structural, set, spectral, temporal,
   conversion, pack/unpack, module, map/filter and state families against the
   interpreter — but a differential fuzzer over the whole registry would be a
   stronger statement than 26 tests.
7. **Absolute milliseconds are host-specific and arm A's are unstable** (§4).
   Operator applications, element operations, bytecode counts, node counts and
   byte counts are not.

---

## 13. Reproduce

```bash
.venv/bin/python -m tcn demo --out artifacts/demo                    # writes the mixed fixture
.venv/bin/python -m pytest tests/test_compile.py -q                  # 26 differential tests
.venv/bin/python research/compiled-runtime/check.py mixed            # compile + equivalence
.venv/bin/python research/compiled-runtime/differential_mixed.py     # 2000 random draws + error contract
.venv/bin/python research/compiled-runtime/check.py language
.venv/bin/python research/compiled-runtime/check.py visual           # ~1 min
.venv/bin/python research/compiled-runtime/bench.py mixed            # four arms
.venv/bin/python research/compiled-runtime/bench.py language
.venv/bin/python research/compiled-runtime/bench.py computer
.venv/bin/python research/compiled-runtime/bench.py visual           # ~12 min (arm A is 18 s a call)
.venv/bin/python research/compiled-runtime/attribute.py mixed        # bytecode attribution
.venv/bin/python research/compiled-runtime/attribute.py language
.venv/bin/python research/compiled-runtime/attribute.py visual       # ~3 min
.venv/bin/python research/compiled-runtime/profile_arms.py visual    # ~2 min
.venv/bin/python research/compiled-runtime/deploy.py mixed acd       # python3 -I deployment rows
.venv/bin/python research/compiled-runtime/deploy.py language acd
.venv/bin/python research/compiled-runtime/deploy.py visual cd
```

| file | what it does |
|---|---|
| `fixtures.py` | rebuilds the four frozen artifacts and their hand-written references |
| `harness.py` | operator counter, encode/decode/validate counter, arm-B patches, timing, type-identity report |
| `check.py` | compiles one fixture and asserts exact equivalence with the interpreter |
| `differential_mixed.py` | 2,000 random draws on the shipped mixed artifact, plus its error contract |
| `bench.py` | the four-arm table, work counters and byte counts; refuses to time an arm that differs from A |
| `attribute.py` | `sys.monitoring` bytecode counts per arm |
| `profile_arms.py` | `cProfile` attribution by role, before and after |
| `deploy.py` | zipapps and the `python3 -I` cold start / RSS / transport rows |

`out/*.json` holds every figure above.  `out/*_compiled.py`, `out/*_artifact.json`
and `out/*.pyz` are gitignored (visual's artifact JSON alone is 123 MB);
regenerate them with the commands above.
