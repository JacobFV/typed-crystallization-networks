# Pre-registration — typed-guard elimination in `tcn/compile.py`

Committed **before any change to `tcn/compile.py`** and before any timing arm.
Branch `research/emitter-guards`, from `main` at `ee7d63c`.

FINDINGS §56 (`research/residual-gap/RESULTS.md`) attributed the residual 6.2×
between the compiled specification and hand-written Python.  Its largest single
named cause is **typed-guard elimination, 2.834×, 56.3 % of the gap**, and its
closing line relocates the work: *"the next gain is in `tcn/compile.py`'s
emitter (guard elimination where the type system already proves the bound), not
in the resynthesizer."*  This track implements that, and only that.

§56 also supplies the cautionary case this document is built around: its
pre-registered rung **R5a failed the bit-identity gate on 243 of 2,883 records**
because a `min(·, 3069)` clamp was load-bearing rather than dead.  Ungated, that
would have shipped as a silent semantic change on 8.4 % of records.

---

## 0. What is *not* in scope

Stated first, because §56's own §9.2 warns that scope creep is how a core change
becomes unreviewable.

* **No loop-invariant code motion, no CSE** (§56's R3, 1.844×).
* **No guard-call inlining** (§56's R1, 1.184×) — `tcn/compile.py` already emits
  its guards inline; that rung is a defect of the *resynthesizer's* emitter.
* **No module-call inlining** (R2), **no loop-shape or strength-reduction
  changes** (R6, R7), **no short-circuiting conjunction** (R4 — the synthesizer's
  1.067×, not the compiler's).
* **Nothing in `tcn/types.py`, `tcn/graph.py`, `tcn/operators.py`.**  The
  interpreter remains the oracle.
* **No change to the external boundary.**  `_IN`, `_ST`, the `_b*` boundary
  encoders, the `_c*` scalar canonicalisers and the boundary identity guard are
  untouched.  Every value entering at an input port is still validated in full.

The only file under `tcn/` that this track modifies is `tcn/compile.py`.

---

## 1. The invariant the whole pass rests on

**IV.**  At every point in generated code, the value bound to an SSA name whose
declared type is an `int` with `encoding.kind == "integer"` lies in that type's
closed interval

    lo = -(2**(bits-1)) if signed else 0
    hi =  2**(bits - signed) - 1
    if bounds is not None:  lo = max(lo, ceil(bounds[0])); hi = min(hi, floor(bounds[1]))

IV is established, not assumed, at every binding site the emitter controls:

| binding site | why IV holds |
|---|---|
| input port, state port | `_IN[k]` / `_ST[k]` runs the full `Value.of` contract |
| constant, constant-folded node | decoded from a validated `Value`; the exact value is known, so the interval is a singleton |
| scalar arithmetic node | the emission *ends* in either the range check (raises otherwise), or `_canon_fn` (wraps/saturates/raises into range) |
| `pack` | `Registry.resolve` requires `scalar.bits == sum(item bits)`, so the packed raw fits the width exactly; a refinement bound is separately checked |
| `unpack` | each field is masked to its own width and sign-adjusted; a refinement bound is separately checked |
| `project`, `index`, `mux`, `tuple`, `delay`, module return, set algebra | structural: they only move values that already satisfy IV |

**IV is inductive**, so eliminating a check *after proving the value is in range*
preserves it.

### 1.1 Scope of equivalence, stated plainly

IV at an input port depends on `run(..., validate=True)`, or on the caller
honouring the documented contract of `validate=False` ("the record is already
canonical").  `validate=False` **already** skips arity, carrier-kind and range
validation at the boundary today; after this change, a caller that violates that
contract may additionally receive a wrong answer where it might previously have
received an `OverflowError` from an internal check.  This is a change in
behaviour on a **contract-violating** call only, it is recorded here in advance,
and it is the one place where "exact semantics unchanged" is qualified.  For
every legal call — `validate=True`, or `validate=False` on a canonical record —
the value returned and the exception raised, its type and its edge, are
unchanged.  The gate below exercises both.

---

## 2. Guard classes to be eliminated, and the proof obligation for each

An interval lattice over integer-encoded scalars is computed during lowering.
`iv(v)` is the declared-type interval of `v` (IV), refined to a tighter interval
where the emitter itself computed the value; a constant's interval is a
singleton.  `iv_expr(op, srcs)` is the interval of the *expression* the emitter
is about to assign, computed from the operand intervals by ordinary interval
arithmetic over `add sub mul neg abs min max mod idiv shl shr count sum
reduce_min reduce_max` and the integer conversions.  `iv_expr` returns ⊤
(unknown) for anything it has not been taught, and ⊤ never discharges an
obligation.

| # | guard class | emitted today by | proof obligation to eliminate |
|---|---|---|---|
| **G1** | **integer range check** — `if not lo <= v <= hi: _ovf(v, lo, hi)` | `_emit_scalar`, `_fast_kind == "int"` | `iv_expr(op, srcs) ⊆ [lo, hi]` |
| **G2** | **dynamic tuple index** — `if not 0 <= i < N: raise IndexError` | `_lower`, `index` | `iv(i) ⊆ [0, N)` |
| **G3** | **zero denominator** — `if d == 0: raise ValueError` | `_lower`, `div`/`mod`/`idiv` | `0 ∉ iv(d)` |
| **G4** | **shift width** — `if not 0 <= s < bits: raise ValueError` | `_lower`, `shl`/`shr` | `iv(s) ⊆ [0, bits)` |

G1–G4 are exactly the four classes §56 named (`_ck`, `_ix`, `_dz`, `_dzi` in the
resynthesizer's terms).

**Explicitly retained, with the reason:**

* the `float64` `_isfinite` check — a float interval lattice is not built, and a
  float operation can reach infinity from finite operands;
* every `_canon_fn` body and every `_b*` boundary encoder;
* the set-capacity `OverflowError` — needs a cardinality analysis, not an
  interval;
* the empty-reduction `ValueError`;
* the refinement-`bounds` domain checks after `pack` and `unpack`;
* the shift/denominator/index checks whose operand interval does not discharge
  the obligation — **a guard that cannot be proved redundant stays**, and is
  reported as a guard the declared type does not bound.

**This pass cannot repeat R5a.**  R5a deleted a `min(·, 3069)` *clamp*, which is
an operator, not a guard.  G1–G4 delete only emitter-inserted checks whose
failure branch raises; no operator's value semantics is touched by any of them.
An interval analysis that sees `min(a, 3069)` *gains* the bound `≤ 3069` from
the clamp rather than deleting it.

---

## 3. Gates — nothing is timed before all of these pass

For each of the four artifacts `mixed`, `language`, `visual`, `computer`:

* **Gate A — bit-identity against the current compiler.**  The unmodified
  `tcn/compile.py` at `ee7d63c` is loaded as a frozen reference module; the new
  emitter's output is compared to it on every case, both `validate=True` and
  `validate=False`, output dict and state dict, **equal value or equal exception
  type at the identical edge**.  Compared by a canonical SHA-256 digest recorded
  per artifact, so the claim is checkable without re-running.
* **Gate B — against the typed interpreter.**  `Program.run` on
  `Value(t, case)`, decoded, compared to the new module's output on every case
  the interpreter can afford (all cases for `mixed`, `language`, `computer`; the
  `visual` interpreter costs minutes per case, so a stratified subset with the
  count reported).
* **Gate C — a differential error-contract suite.**  `tests/test_compile.py`'s
  26 differential tests are the floor.  New tests are added that drive each of
  G1–G4 to its edge: a program where the guard **is** provably redundant (the
  value must be identical) and a program where it is **not** (the exception type
  and edge must be identical to the interpreter's).
* **Gate D — the repository.**  All 337 tests, and the shipped fixture
  reproducing `0.248836 → 0.002231` at 4/4 frozen.

**A failing gate drops that elimination**, and the failure is recorded with its
record count, in the form §56 used for R5a (243 / 2,883).

---

## 4. Measures — three, never one scalar

§51 established that executed primitives, executed bytecodes and wall clock
**disagree in both directions**.  All three are reported per artifact.

1. **Executed primitives** — leaf operator applications, worst and expected over
   the declared distribution, metered by `research/lazy-guard/cost.py`'s
   `NativeEval`/`profile`.  *Prediction, registered:* **unchanged, exactly
   1.00×**, because the pass changes emitted Python and not the program.  A
   change here would mean the pass altered the program and is a **bug**.
2. **Executed CPython bytecodes** per case, worst and expected, counted with
   `research/lazy-guard/cost.py`'s `count_opcodes`.
3. **Wall clock**, batch-one warm, per case, under
   `research/lazy-latency/latency.py`'s protocol verbatim: pinned to one core,
   gc collected then disabled around each sweep, one discarded warm-up sweep per
   arm per artifact, 21 repeats with the two arms **interleaved round-robin
   inside each repeat**, medians with non-parametric 95 % CIs and a 10,000-
   resample bootstrap CI on every ratio.

Plus, per artifact: **bytes of generated source on disk**, **zipapp bytes**, and
**cold start** (`/usr/bin/python3 -I`, no venv, no `tcn`), reusing
`research/compiled-runtime/deploy.py`'s method.

And: **static guard counts** per class per artifact, before and after, plus the
count of guards that **could not** be proved redundant, by class.

---

## 5. Falsification conditions, registered in advance

| # | condition | what it means |
|---|---|---|
| F1 | **No artifact improves measurably** — every wall-clock ratio's 95 % CI contains 1.0 | the guards were not the cost outside the one subroutine §56 measured; §56's relocation was too optimistic for the general case.  Report plainly. |
| F2 | **The provable fraction is small** — fewer than half the emitted G1 guards discharge their obligation | the declared type algebra does not bound these values.  Report the residue by class; this is a finding about the type algebra, not a failure of the pass. |
| F3 | **Any gate fails** | that elimination is dropped and recorded with its record count. |
| F4 | **`visual` improves but `mixed` does not** | a win that only appears at width.  Say so; the claim narrows to wide artifacts. |
| F5 | **Executed primitives change** | the pass altered the program.  Bug, not result. |
| F6 | **Bytecodes fall but wall clock does not** (or the reverse) | §51's disagreement recurs; report all three and do not collapse them. |

**Registered hypothesis:** G1 dominates the static counts (267 of 281 emitted
guards on `visual`); the wall-clock gain is largest on `visual` and `language`
and may be unmeasurable on `mixed` (4 operations, 0 emitted G1 guards).  I do
**not** predict §56's 2.834×: that figure is for a subroutine whose loop index
gives a tighter bound than the declared type does, and F2 is registered
precisely because the declared type may not reproduce it.

---

## 6. Discipline

* Achieved difficulty is reported, never the requested configuration.
* Every amendment is recorded with the number it replaced, in the form of
  `research/lazy-latency/RESULTS.md` §9 and `research/residual-gap/RESULTS.md`
  §8.
* Negative results are preserved.
* Large generated artifacts stay out of git.
* `research/cross-domain/` is not touched.
