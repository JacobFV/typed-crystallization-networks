# Typed-guard elimination in `tcn/compile.py`, measured on all four artifacts

Branch `research/emitter-guards`, from `main` at `ee7d63c`.  **This is a core
change**: `tcn/compile.py` is modified, and `tests/test_compile.py` gains seven
tests.  Nothing else under `tcn/` or `generators/` is touched, and
`research/cross-domain/` was not touched.

`PREREGISTRATION.md` was committed at **`79778d8`, before the first edit to
`tcn/compile.py`** (`ee15cb4`).  Every number below is printed by `report.py`
from `out/*.json` and can be re-checked without re-running anything.

This is the job FINDINGS §56 relocated: *"the next gain is in `tcn/compile.py`'s
emitter (guard elimination where the type system already proves the bound), not
in the resynthesizer."*

---

## 0. The verdict, stated first

**The change is sound, it is worth 1.10× on one artifact, and §56's 2.834× does
not transfer to the compiler at artifact level.**

* **No gate failed.**  1,252 differential comparisons against the compiler at
  `ee7d63c` — every artifact, every case, both `validate=True` and
  `validate=False`, value *or* exception type at the identical edge — and 608
  independent checks against the typed interpreter.  **Zero mismatches.**
* **206 of 282** emitted range guards (G1) across the four artifacts are proved
  unreachable from the declared types and removed.  **0 of 14** dynamic index
  guards (G2) are.  G3 and G4 never occur on these artifacts.
* **Wall clock, three independent runs:** `language` **1.10×** (no CI contains
  1 in any run); `visual` **1.017× / 1.020× / 1.003×** — at the edge of what this
  instrument resolves; `mixed` and `computer` **unchanged, and provably so** —
  their generated source is byte-identical between the arms, which makes them
  exact null controls.
* **On the `visual` parser's S2 subroutine in isolation — the thing §56 actually
  measured — the same elimination is 1.136×**, CI [1.131, 1.138].  It does not
  reach the artifact because S2 runs **19 times per screenshot** against
  `_m1`'s **3,100**, and *not one guard in `_m1` is provable*.
* **The guards that survive are the informative part.**  All 12 surviving index
  guards on `visual` read a 3,072-element raster through an address declared
  `int16u` with no refinement bound.  **The type algebra can express the bound
  and this artifact does not declare it** — and §56's load-bearing
  `min(·, 3069)` clamp is the value-level shadow of exactly that missing bound.

**F1 did not fire** (one artifact improves, well clear of noise).  **F2 did not
fire** (73 % of G1 discharges).  **F3 did not fire** (no gate failed).
**F4 fired in reverse**: the win is on the *narrow* artifact, not the wide one.
**F5 did not fire.**  **F6 fired**: on `visual` the bytecode reduction is exact
and the wall-clock effect is not resolvable.

---

## 1. What changed

One idea, in `tcn/compile.py` only: an **interval lattice over integer-encoded
scalars**, computed during lowering, starting from the declared type of every
binding site and refined — never widened — by the expression the emitter is
about to write.  Where the interval discharges a guard's proof obligation the
guard is not emitted.  Where it does not, the guard is emitted exactly as before.

Invariant **IV**, its establishment at each binding site, and the scope of
equivalence are in `PREREGISTRATION.md` §1 and repeated in the module's own
comments.  Two properties are worth restating here:

* **Only emitter-inserted checks are removed.**  No operator's value semantics is
  touched.  `research/residual-gap`'s R5a deleted a `min(·, 3069)` *clamp* and
  failed on 243 of 2,883 records; this pass structurally cannot do that, and
  `tests/test_compile.py::test_a_load_bearing_clamp_is_never_deleted_and_is_what_proves_the_index`
  is the standing regression — the clamp survives, and the interval it
  establishes is precisely what discharges the index obligation downstream.
* **The external boundary is untouched.**  `_IN`, `_ST`, every `_b*` and every
  `_c*` still run the full `Value.of` contract.
  `test_guard_elimination_never_changes_the_boundary` pins it.

### 1.1 The one qualification on "semantics unchanged"

Registered in advance (`PREREGISTRATION.md` §1.1).  IV at an input port depends
on `validate=True`, or on the caller honouring the documented contract of
`validate=False`.  A caller that violates that contract may now receive a wrong
answer where it might previously have received an `OverflowError` from an
internal check.  For every **legal** call the value and the exception, its type
and its edge, are unchanged — and the gate exercises **both** flag settings on
every case.

---

## 2. The guard classes and their proof obligations

| # | class | emitted by | obligation | eliminated / emitted by the baseline, all four artifacts |
|---|---|---|---|---|
| **G1** | integer range check `if not lo <= v <= hi: _ovf(...)` | `_emit_scalar` | `iv_expr(op, srcs) ⊆ [lo, hi]` | **206 / 282** |
| **G2** | dynamic tuple index `if not 0 <= i < N: raise IndexError` | `_lower`, `index` | `iv(i) ⊆ [0, N)` | **0 / 14** |
| **G3** | zero denominator | `_lower`, `div`/`mod`/`idiv` | `0 ∉ iv(d)` | 0 / 0 — the class does not occur |
| **G4** | shift width | `_lower`, `shl`/`shr` | `iv(s) ⊆ [0, bits)` | 0 / 0 — the class does not occur |

Retained by design, unchanged: the `float64` `_isfinite` check, every `_canon_fn`
body and `_b*` boundary encoder, the set-capacity `OverflowError`, the
empty-reduction `ValueError`, and the `pack`/`unpack` refinement-bound checks.

Per artifact (`out/gate.json`):

| artifact | G1 eliminated / total | G2 eliminated / total |
|---|---|---|
| mixed | 0 / 0 | 0 / 0 |
| language | **15 / 15** | 0 / 1 |
| computer | 0 / 1 | 0 / 1 |
| visual | **191 / 266** | 0 / 12 |

---

## 3. The gates — every result, including that none failed

`gate.py`, raw data `out/gate.json`.  The reference compiler is read out of git
at `ee7d63c` (source SHA-256 `bf2115e66ba3…`) rather than copied, so it cannot
drift from the thing this track claims to be identical to.

| artifact | gate domain | cases | comparisons (gate A) | A mismatches | interpreter checks (gate B) | B mismatches |
|---|---|---|---|---|---|---|
| mixed | 2 × 2 booleans × 101 values of `x` on [−1, 1] | 404 | **808** | **0** | **404** | **0** |
| language | every test episode of declared length 8–14 in a 400-episode draw | 188 | **376** | **0** | **188** | **0** |
| computer | 10 synthetic terminal documents, including empty and out-of-range | 10 | **20** | **0** | **10** | **0** |
| visual | 24 held-out screenshots — **23,064 interior positions** | 24 | **48** | **0** | **6** | **0** |
| | | | **1,252** | **0** | **608** | **0** |

`all_passed: true`.  A SHA-256 digest over each arm's canonical outcomes is
recorded per artifact and the two digests are equal in every case, so the claim
is checkable from the JSON alone.

The domains are deliberately far wider than the timing cases:
`research/residual-gap/RESULTS.md` §2.1 records a transform that passed on 54
deployment records and failed on 243 of 2,883 held-out ones.

**Gate C — the differential error-contract suite.**  §48's 26 tests are the
floor; `tests/test_compile.py` now has **33**.  The seven added drive each class
to its edge in both directions: the guard is proved and gone (value identical to
the interpreter, and the check verifiably absent from the emitted source), or it
is unproved and kept (the identical exception at the identical edge).  All pass.

**Gate D — the repository.**  §10 below.

**No elimination was dropped, because no gate failed.**  This section would have
carried the record count in §56's R5a form (243 / 2,883) had one.

---

## 4. Three cost measures, per artifact

`measure.py`, raw data `out/measure_run1.json`, `run2`, `run3`.  Timing under
`research/lazy-latency/latency.py`'s protocol verbatim — pinned to core 19, gc
collected then disabled around each sweep, a discarded warm-up sweep per arm,
21 repeats with the two arms **interleaved round-robin inside each repeat**,
medians with non-parametric 95 % CIs, 10,000-resample bootstrap CI on the ratio.
**Bit-identity is re-asserted on the exact cases about to be timed, and nothing
is timed if it fails.**

### 4.1 Executed primitives — unchanged, exactly as registered

| artifact | worst | expected | ratio old ÷ new |
|---|---|---|---|
| mixed | 4 | 4.0 | **1.00×** |
| language | 148 | 148.0 | **1.00×** |
| computer | 23 | 23.0 | **1.00×** |
| visual | 61,246 | 59,793.0 | **1.00×** |

The pass changes emitted Python and not the program, so this is arm-independent
by construction; **F5 made any other value a bug, and it did not fire.**  The
instrument is `Registry.exact` applications with module invocations excluded —
`research/lazy-guard/cost.py`'s definition, metered on the shipped interpreter
because that file's `NativeEval` has not been taught `sin`, `decode` or `pair`
and cannot run three of the four artifacts.  Where both instruments run
(`language`) they agree exactly at 148.

### 4.2 Executed bytecodes — exact, deterministic, and the honest core of the result

| artifact | old | new | **old ÷ new** |
|---|---|---|---|
| mixed | 120.0 | 120.0 | 1.0000× |
| **language** | 1,731.8 | **1,596.8** | **1.0845×** |
| computer | 328.0 | 328.0 | 1.0000× |
| **visual** | 635,400.3 | **602,748.3** | **1.0542×** |

### 4.3 Wall clock, batch-one warm, µs per case

| artifact | | old | new | **old ÷ new** | 95 % CI |
|---|---|---|---|---|---|
| **mixed** *(null control)* | run1 | 0.4210 | 0.4185 | 1.0061 | [1.0003, 1.0113] |
| | run2 | 0.4206 | 0.4200 | 1.0013 | [0.9904, 1.0093] |
| | run3 | 0.4293 | 0.4298 | 0.9988 | [0.9896, 1.0032] |
| **language** | run1 | 3.8784 | **3.5147** | **1.1035** | [1.0982, 1.1093] |
| | run2 | 4.0512 | 3.6851 | **1.0993** | [1.0915, 1.1062] |
| | run3 | 4.0622 | 3.6684 | **1.1073** | [1.0993, 1.1092] |
| **computer** *(null control)* | run1 | 1.0843 | 1.0857 | 0.9987 | [0.9878, 1.0085] |
| | run2 | 1.7772 | 1.5132 | 1.1745 | [0.9700, 1.4302] |
| | run3 | 1.0867 | 1.0821 | 1.0042 | [0.9925, 1.0140] |
| **visual** | run1 | 4,998.26 | 4,914.63 | 1.0170 | [1.0071, 1.0280] |
| | run2 | 4,993.10 | 4,894.80 | 1.0201 | [1.0057, 1.0274] |
| | run3 | 4,985.22 | 4,971.33 | **1.0028** | **[0.9921, 1.0175]** |

**`mixed` and `computer` are exact null controls, and this is checkable:** the
two arms' generated source has the *same* SHA-256 (`out/measure_run*.json`,
`source_sha256`), because no guard on either artifact discharged its obligation.
Any difference measured on them is instrument bias — and **run1's `mixed` row is
a demonstrated false positive: 1.0061× with a CI that excludes 1.0 on
byte-identical code.**

That fixes the resolution of this instrument at roughly **0.6 %** on this shared
host, which is why **`visual`'s 1.7–2.0 % is reported as "at the edge of
resolution" rather than as a win**: run3's CI contains 1.0.  `language`'s 10 %
is an order of magnitude clear of it in all three runs.

**F6 fired, exactly as §51 warned.**  On `visual` the bytecode reduction is
5.42 %, exact and deterministic, and the wall-clock effect is not resolvable.
On `language` bytecodes fall 8.45 % and wall clock 10.3 % — the *time* moves
more than the *bytecodes*, the opposite direction, because a removed inline
guard is three cheap comparisons and a not-taken branch.  Three measures, three
different answers, none of them collapsible into the others.

### 4.4 Bytes on disk and cold start

| artifact | source B old → new | zipapp B old → new | cold start ms old → new |
|---|---|---|---|
| mixed | 2,629 → 2,629 | 1,713 → 1,713 | 24.5 → 14.0 |
| language | 24,255 → **23,234** (−4.2 %) | 5,661 → **5,515** (−2.6 %) | 23.1 → 20.3 |
| computer | 30,645 → 30,645 | 3,353 → 3,353 | 31.5 → 30.4 |
| visual | 171,232 → **161,352** (−5.8 %) | 22,692 → **21,344** (−5.9 %) | 61.5 → 53.8 |

**Cold start is not resolvable here and no claim is made from it.**  The `mixed`
row moves 24.5 → 14.0 ms on *byte-identical* code; the null control disqualifies
the column at these magnitudes.  Source and zipapp bytes are exact.

---

## 5. Why `visual` barely moves — checked against raw data, not explained away

`attribute.py`, raw data `out/attribute.json`.  A 1.02× where §56 measured
2.834× for the same transform is a surprising headline, so it is attributed at
the opcode level rather than argued.

**Calls per screenshot:** `_m1` **3,100**, `_m0` **961**, `_m2` **19**.

**Where the eliminations are:** all **191** are in `_m2`.  **Not one** is in
`_m1` or `_m0` — those keep 4 G1 + 6 G2 and 2 G1 respectively.

**Executed bytecodes by code object, one screenshot:**

| function | calls | old | new | change |
|---|---|---|---|---|
| `_m1` | 3,100 | 468,100 | **468,100** | **0** |
| `_m2` | 19 | 109,147 | **74,681** | **−34,466 (−31.6 %)** |
| `_m0` | 961 | 51,894 | 51,894 | 0 |
| `<genexpr>` | — | 21,363 | 21,363 | 0 |
| `run` | 1 | 57 | 57 | 0 |

The pass removes **1,814 bytecodes per `_m2` call** and `_m2` runs 19 times, so
34,466 of 635,400 — **5.4 %** — which is what the wall clock then fails to
resolve.  `_m1`, which is **73.7 %** of the artifact's bytecodes, is untouched
because every one of its guards is unprovable.

### 5.1 The subroutine in isolation — the number comparable to §56

Same protocol, on the arguments the artifact actually passes to `_m2`, with
bit-identity asserted on exactly those arguments first:

| | old | new | ratio | 95 % CI | bytecodes/call |
|---|---|---|---|---|---|
| `_m2` (the S2 `rect_scaffold`) | 35.965 µs | **31.672 µs** | **1.1355×** | [1.1290, 1.1390] | 15,114.1 → 13,300.1 |

Reproduced in a second independent run at **1.1357×**, CI [1.1309, 1.1381].
Note the denominator: 15,114 bytecodes per call is `_m2` *and everything it
calls*, while the 109,147 in the table above is `_m2`'s own frame only — the
elimination removes 1,814 either way, which is **12.0 %** of the call and
**31.6 %** of the frame.

**1.136×, not 2.834×**, and the difference is measurable rather than mysterious:

1. **`tcn/compile.py` already inlines its guards.**  §56 §4.4 says so itself —
   52 % of the resynthesizer's bytecodes were inside guard *frames*, 378 Python
   calls per record against 2.  Here a guard is three comparisons and a
   not-taken branch: the 191 eliminated guards are 1,814 of 15,114 bytecodes per
   call, **12 %**, not 52 %.  §56 priced R1 (inlining) separately at 1.184× and
   then asserted the remaining 2.834× is *"the checks themselves, which both
   emitters pay"*.  **On this evidence that assertion is too strong**: the
   compiled emitter's checks are much cheaper than the resynthesizer's, and the
   two figures are not interchangeable.
2. **§56's R5 baseline was already hoisted and short-circuited** (R3, R4), so the
   guards were a larger share of a smaller program.
3. **§56's R5 also copy-propagated the temporaries its guards had kept alive**
   (their AMENDMENT 3), which is dead-code elimination.  This track does not do
   it — `PREREGISTRATION.md` §0 excludes it — so part of their 2.834× is not
   guard elimination at all, by their own record.

---

## 6. The guards the declared type does not bound — the finding about the type algebra

`residue.py`, raw data `out/residue.json`.  75 G1 and all 14 G2 survive.

| artifact | count | shape | why the obligation does not discharge |
|---|---|---|---|
| visual | **70** | `G1 add(int16u, int16u) -> int16u` | 65535 + 65535 overflows `int16u`.  The declared carrier is full-width, so `add` is unbounded **by construction** |
| visual | **12** | `G2 index(tuple[3072], int16u)` | the raster address is declared `int16u` with **no refinement bound**; [0, 65535] ⊄ [0, 3072) |
| visual | 3 | `G1 sub(int16u, int16u)` | 0 − 65535 underflows |
| visual | 2 | `G1 sum(tuple[31]) -> int16u` | 31 × 65535 overflows |
| computer | 1 | `G1 add(int8u, int8u)` | as above |
| computer | 1 | `G2 index(tuple[4096], int32u bounds(0, 4096))` | **the bound is closed and the index is half-open**: 4096 is admissible and out of range |
| language | 1 | `G2 index(tuple[128], int32u bounds(0, 128))` | the same off-by-one |

Three separate statements about the type algebra fall out, and they are the most
transferable part of this track:

1. **A closed refinement bound `(0, N)` can never prove an index into an
   `N`-tuple.**  Two of the three surviving index shapes are exactly this
   off-by-one, on two different artifacts.  `bounds=(0, N-1)` would discharge
   both.  This is pinned as a test
   (`test_g2_index_guard_goes_only_when_the_bound_is_inside_the_tuple`).
2. **`visual`'s hot function loses every guard to a missing refinement bound.**
   `_m1(a, b, obs)` reads `obs[a]`, `obs[a+1]`, `obs[a+2]` with `a` declared
   `int16u`.  A declared `bounds=(0, 3069)` on the address type would discharge
   all six index guards and all four range guards in the function that is 73.7 %
   of the artifact — **and `(0, 3069)` is exactly §56's load-bearing
   `min(·, 3069)` clamp**.  The clamp is the value-level shadow of a bound the
   type does not carry.
3. **But the bound is not free, and I checked before claiming it was.**
   `Registry.resolve` propagates a refinement bound through `add`, so declaring
   the address `bounds=(0, 3069)` gives `a + 1` the *same* bound — which changes
   the arithmetic's error contract (3070 would now raise) and pushes the node off
   `_fast_kind`'s inline path onto a `_canon_fn` call.  **The type algebra
   couples "what bounds the index" to "what the arithmetic may produce".**  That
   coupling, not the compiler, is what stands between `visual` and the rest of
   this win.

---

## 7. Falsification, condition by condition

| pre-registered condition | outcome |
|---|---|
| **F1 — no artifact improves measurably** | **DID NOT FIRE.**  `language` is 1.10× in all three runs, no CI containing 1, against a 0.6 % instrument floor established by a byte-identical null control. |
| **F2 — fewer than half of emitted G1 guards discharge** | **DID NOT FIRE**: 206 / 282 = **73.0 %**.  It would not have fired before AMENDMENT 1 either, but only just — 144 / 282 = **51.1 %**, one percentage point from the threshold.  Both numbers are stated so the reader can apply their own standard. |
| **F3 — any gate fails** | **DID NOT FIRE.**  1,252 comparisons + 608 interpreter checks, 0 mismatches.  No elimination was dropped. |
| **F4 — `visual` improves but `mixed` does not** | **FIRED, in reverse.**  The measurable win is on `language` (164 operations); `visual` (64,346) is at the edge of resolution and `mixed` and `computer` are exactly zero.  **This is a win that appears at *narrowness*, not at width** — because on `visual` the provable guards sit in a function called 19 times. |
| **F5 — executed primitives change** | **DID NOT FIRE.**  1.00× on all four. |
| **F6 — bytecodes fall but wall clock does not, or the reverse** | **FIRED, both directions on the same table.**  `visual`: −5.42 % bytecodes, no resolvable time.  `language`: −8.45 % bytecodes, −10.3 % time.  Reported separately, never collapsed. |

---

## 8. Amendments to the pre-registration

Each with the number it replaced and its direction relative to this track's own
hypothesis, to the standard of `research/lazy-latency/RESULTS.md` §9 and
`research/residual-gap/RESULTS.md` §8.

**AMENDMENT 1 — the interval lattice gained a `bool → int` conversion case after
the first residue was inspected, and it moves *for* the hypothesis, which is why
it is recorded most loudly.**  `PREREGISTRATION.md` §2 defines `iv_expr` over a
named operator list that did not include a conversion from `bool`.  The first
residue run showed **62 guards of the shape `G1 encode(bool) -> int16u`** kept on
`visual` — where `round(float(b))` is 0 or 1 and the declared type `bool` admits
nothing else, so the obligation discharges trivially.  It was added.  **The
effect is 144 → 206 eliminated of 282, 51.1 % → 73.0 %**; F2 would not have
fired either way, but 51.1 % is one percentage point from its threshold, so the
amendment moved a marginal number to a comfortable one and that is precisely why
it is disclosed here rather than folded into §2.  Both figures are in §7.
Nothing else about the lattice was changed after seeing any residue or any
timing.

**AMENDMENT 2 — three timing runs, not one.**  `PREREGISTRATION.md` §4 fixes 21
repeats but not a number of runs.  After run1 reported `mixed` at 1.0061× with a
CI excluding 1.0 on **byte-identical source**, two further runs were taken.  This
moves against the hypothesis: it is what demoted `visual`'s 1.017× / 1.020× from
a result to an edge case, since run3 gives 1.0028× with a CI containing 1.

**AMENDMENT 3 — the null control was not pre-registered and is the strongest
methodological thing here.**  `PREREGISTRATION.md` predicted `mixed` "may be
unmeasurable"; it did not anticipate that `mixed` and `computer` would emit
**byte-identical source** in both arms and so serve as exact null controls.  The
`source_sha256` field was added to `measure.py` after run1 to make the claim
checkable.  It moves against the hypothesis: it is the reason §4.3 refuses to
bank `visual`'s 2 %.

**AMENDMENT 4 — the primitive meter is not `NativeEval`.**
`PREREGISTRATION.md` §4 names `research/lazy-guard/cost.py`'s
`NativeEval`/`profile`.  That evaluator is a miniature and raises on `sin`,
`decode` and `pair`, so it runs only `language` of the four.  The portable meter
counts `Registry.exact` applications on the shipped interpreter with module
invocations excluded — `cost.py`'s own definition of a primitive, on a different
instrument.  Where both run they agree exactly (148 / 148).  Neutral.

**AMENDMENT 5 — `residue.py`'s first attribution double-counted.**  Its
counter-delta wrapper attributed a module's inner guards to the outer `module:`,
`map` or `filter` node as well as to the node that emitted them, inflating the
`visual` and `language` G2 residue.  Fixed by excluding those three operators
from the wrapper; the corrected totals agree with `CompileResult.stats`.  This
moves against the hypothesis: it *reduced* the count of guards attributed to the
declared-type residue.

---

## 9. What this does NOT establish

At the same volume as the results.

1. **§56's 2.834× is not reproduced, and this track does not claim it is
   wrong.**  It is a different baseline (post-LICM, post-short-circuit), a
   different emitter (guards as calls, not inline) and a different program (a
   `while` loop, not a 521-node straight-line formulation), and it bundled a
   copy-propagation this pass does not do.  §5.1 gives the compiler's own number
   for the same subroutine, 1.136×, and that is the number that should be
   quoted for `tcn/compile.py`.
2. **`visual`'s 1.7–2.0 % is not banked.**  Two of three runs exclude 1.0 and one
   does not, on a host where a byte-identical null control produced a CI
   excluding 1.0.  The bytecode figure (−5.42 %) is exact; the time figure is not
   established.
3. **Nothing here says the pass is worthless on programs it has not seen.**  It
   also says nothing about programs whose declared types carry tighter bounds,
   where §6 predicts it does much more — untested, because building such a
   variant of `visual` changes the arithmetic's error contract (§6.3) and was out
   of scope.
4. **G3 and G4 are tested but never exercised by an artifact.**  Their correctness
   rests on `tests/test_compile.py`, not on the artifact gate, and no performance
   claim is made for them.
5. **The `validate=False` qualification of §1.1 is a real narrowing of the error
   contract on a contract-violating call.**  It is not hypothetical; it is
   untested by construction, because there is no correct behaviour to test
   against.
6. **Cold start and peak RSS are reported and disqualified**, not used.
7. **Nothing here touches quality.**  Every arm computes the identical function
   on 1,252 gated comparisons; a fast wrong answer is measured nowhere.

### 9.1 The next step, named and priced, and deliberately not taken

The single largest guard bundle left is **the `_canon_fn` call on a
refinement-bounded scalar**.  On `language` `_c0` is **731 of the 1,586
bytecodes executed on the case attributed in §5 — 46 % of the whole artifact**,
across 17 calls per case (`out/attribute.json`), and its
body is nothing but a finiteness test, two bounds tests, a fractionality test and
a range test.  Where the interval proves all five, the call is provably the
identity.  That is a fifth guard class with the same shape of obligation, and it
is **not** done here because `PREREGISTRATION.md` §0 and §2 explicitly retain
`_canon_fn`, and adding a class after seeing the data would be precisely the
scope creep §56 §9.2 warns about.  It should be its own pre-registered change.

**Difficulty achieved, stated rather than requested.**  The hard part was not the
lattice; it was refusing three easy overstatements — quoting §56's 2.834× as
transferred, banking `visual`'s 2 % before the null control disqualified it, and
adding the `_canon_fn` class once it was obvious it would help.  The trap avoided
is R5a's: this pass deletes checks and never operators, and the regression test
that pins it is written from §56's own failure. The trap not fully avoided is
AMENDMENT 1 — a lattice case added after inspecting the residue, which moved F2
across its own threshold; both numbers are reported.

---

## 10. Repository state and reproduction

* **`.venv/bin/python -m pytest -q` → 343 passed, 1 failed of 344.**  The suite
  was 337; this track adds **7** tests to `tests/test_compile.py` (26 → 33).  The
  single failure is
  `tests/test_panel_interface.py::test_panel_episode_replays_and_restores`, the
  known worktree-only replay divergence (FINDINGS §41,
  `research/MERGE-QUEUE.md`).  **Verified environmental by reverting this
  track's core change**: with `tcn/compile.py` and `tests/test_compile.py`
  restored to `ee7d63c` the same single test still fails.  The thirteen-failure
  mode is the *missing* `generators/computer/engine/node_modules` symlink, not
  its presence; with it restored the count is as above.
* **Shipped fixture reproduces exactly**, on this core change:
  `.venv/bin/python -m tcn train --episodes 160` gives
  `initial_prediction_loss 0.248835613951087`,
  `final_prediction_loss 0.0022308224288281053`, `fully_frozen true`,
  `evaluation_mean_return 4.0`, `frozen_evaluation_mean_return 4.0` —
  **0.248836 → 0.002231 at 4/4 frozen.**
* Generated zipapps and their staging trees are gitignored; every report JSON is
  committed.  `research/cross-domain/` was not touched.

```bash
.venv/bin/python research/emitter-guards/gate.py                 # PREREG sec 3, ~95 s
.venv/bin/python research/emitter-guards/residue.py              # the unprovable residue, ~5 s
taskset -c 19 .venv/bin/python research/emitter-guards/measure.py run1   # ~3 min
taskset -c 19 .venv/bin/python research/emitter-guards/measure.py run2
taskset -c 19 .venv/bin/python research/emitter-guards/measure.py run3
taskset -c 19 .venv/bin/python research/emitter-guards/attribute.py visual   # sec 5, ~1 min
.venv/bin/python research/emitter-guards/report.py               # every number above
```

| file | what |
|---|---|
| `PREREGISTRATION.md` | the protocol, committed at `79778d8` before any change to `tcn/compile.py` |
| `gate.py` | gates A and B against the compiler at `ee7d63c` and against the typed interpreter |
| `measure.py` | the three cost measures, plus disk and cold start |
| `attribute.py` | §5 — bytecodes by code object, calls per case, the subroutine timing |
| `residue.py` | §6 — every guard that stayed, by operator and declared type |
| `report.py` | prints every number above from `out/*.json` |

`research/compiled-runtime/harness.py` supplies the gc discipline, the call
counter and `deploy.py`'s zipapp builder; `research/lazy-guard/cost.py` supplies
the `sys.monitoring` bytecode meter (extended with a bucket key, not
re-implemented); `research/lazy-latency/latency.py` supplies the timing protocol.
**No third harness was built.**
