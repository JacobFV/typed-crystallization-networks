# Pre-registration — a declared refinement bound on the visual raster address

Track `research/refinement-bounds`, branch `worktree-agent-a0350f1fdeac805b2`,
**from `main` at `59252fc` with §59's three commits (`79778d8`, `ee15cb4`,
`16e50df`) cherry-picked**.  Baseline test count is therefore **344**, not 337.

Committed **before any file under `research/refinement-bounds/`,
`research/visual-ladder/` or `tcn/` is touched.**  House standard since §44.

---

## 0. The question, and what is and is not being claimed

FINDINGS §59 §6.2 records that `visual`'s hot module `_m1` — **3,100 calls per
screenshot, 73.7 % of the artifact's bytecodes** — loses **every** guard (4 G1
range + 6 G2 index) because the raster address is declared `IDX =
integer(16, signed=False)` with **no refinement bound**, and that
`bounds = (0, 3069)` "would discharge all six index guards and all four range
guards".  §56's R5a failed its bit-identity gate on 243 of 2,883 records for
deleting the `min(·, 3069)` clamp that is the value-level shadow of that bound.

`Type.bounds` already exists (`tcn/types.py:63`), is validated at construction
(`:76`) and enforced in `encode` (`:150`) and `validate_raw` (`:191`).  §58's
inventory shows `language` and `computer` already declaring `u32<=128` and
`u32<=4096`.  **So this is not "add refinement types to the algebra".**  It is:
the algebra has them, the visual scaffold does not declare one, and the dominant
function pays for it 3,100 times per screenshot.

**Claimed here:** whether a bound can be declared *soundly*, how many of `_m1`'s
guards it discharges, what it costs on three measures, and whether any
certificate moves.  **Not claimed:** that §59's or §56's factors transfer.

---

## 1. Step 1 — what the clamp guarantees, and what the type may declare

The soundness obligation is stated **before** it is discharged, because §56's
R5a is what happens when a plausible-looking redundancy is not one.

### 1.1 The obligation

Let `ADDR` be the type that would replace `IDX` at every site that flows into the
`same` module's address ports.  A declared `bounds = (lo, hi)` on `ADDR` is
**sound** iff, for every input the artifact can be given and every candidate the
scaffold's search can select, every value ever assigned to an `ADDR`-typed node
satisfies `lo <= v <= hi`.

This is stronger than "the clamp keeps the index in range", for a reason that is
the whole point of step 1: `Registry.resolve` gives a `BINARY` operator the
**same** output type as its (identical) operand types (`tcn/operators.py:68-73`),
so a bound declared on an address is inherited by **every** arithmetic result
derived from it — `a + 1`, `a + 2`, `pos + k·step` — and each of those becomes a
*runtime-enforced* obligation, not merely a static fact.

### 1.2 The three registered predictions for step 1

Registered now, before measuring:

* **P1.1 — `bounds = (0, 3069)` is UNSOUND.**  §59 §6.2 names `(0, 3069)` as the
  bound to declare.  `_m1` computes `a + 1` and `a + 2`, whose declared type is
  `a`'s, and the largest reachable interior address is
  `3·(31·32 + 31) = 3069`, so `a + 2 = 3071` violates `(0, 3069)` on a
  **reachable** input.  Declaring it would be R5a shipped rather than caught.
* **P1.2 — `bounds = (0, 3071)` is UNSOUND for the scaffold as written**, because
  `rect_scaffold` computes `wa_k = add(pos, mul(step, k))` **before** the
  `min(·, 3069)` clamp, and that reaches `3069 + 96·31 = 6045`.  This is exactly
  why the clamp is load-bearing.
* **P1.3 — `bounds = (0, 3071)` becomes sound if the clamp is moved inside the
  addition**: `wc_k = add(pos, min(mul(step,k), sub(last, pos)))` computes the
  identical function and never leaves `[0, 3069]`.

`3071 = 3·W·H − 1` is the last raster byte; `(0, 3071)` closed is inside
`[0, 3072)` half-open, so it is the bound that discharges the G2 index
obligation *and* admits `a + 2`.  §59 §6.1's "a closed bound `(0, N)` can never
prove a half-open index into an `N`-tuple" is honoured, not repeated.

### 1.3 How soundness is established, and what counts as failure

1. **Static.**  Every node in `same_scaffold`, `corner_scaffold`,
   `rect_scaffold`, `assembly`, and `rung3_root.py`'s masked/clamped variants
   whose declared type becomes `ADDR` is enumerated with a hand-checked
   reachable range, and the reformulated clamp's identity
   `min(p + d, L) == p + min(d, L − p)` for `0 <= p <= L`, `d >= 0` is verified
   **exhaustively** over the full reachable product (every interior/all position
   × every `step ∈ offsets` × every `k ∈ [1, span)`), not argued.
2. **Dynamic.**  Every `ADDR`-typed node's value is recorded over a full run of
   the artifact on **24 held-out screenshots** (the same domain §59's gate used)
   and the observed min/max compared with `(0, 3071)`.
3. **FALSIFICATION F1 — if either check shows a reachable violation that cannot
   be removed by a semantics-preserving reformulation, the bound is not sound,
   this is reported, and the track STOPS at step 1.**  Declaring an unsound
   bound is the forbidden outcome.

---

## 2. Step 2 — declare the bound and re-measure

### 2.1 Arms

All arms use the **same emitter** (§59's, at `ee15cb4`).  The variable is the
*declared type*, so the baseline is §59's `new` visual arm exactly.

| arm | address type | clamp | emitter |
|---|---|---|---|
| **A0** | `IDX` (no bound) | `min(pos + d, last)` | §59 |
| **A1** | `IDX` (no bound) | `pos + min(d, last − pos)` | §59 |
| **B**  | `ADDR = bounds(0, 3071)` | `pos + min(d, last − pos)` | §59 |
| **N**  | A0 compiled twice — **byte-identical source** | | §59 |

`A1 − A0` isolates the reformulation; `B − A1` isolates the bound; **N is the
null control** and fixes the instrument floor at `visual`'s own magnitude.

A fifth arm **B+** is registered **conditionally** — see §2.4.

The scaffold change is **flag-gated and off by default** (`addr=None`,
`reformulated_clamp=False`), so every existing track, test and artifact that
imports `research/visual-ladder/` is byte-identical to `main`.

### 2.2 Gates — nothing is timed before these pass

Verbatim from §59, because the domains are already known to be wide enough:

* **Gate A — differential against the compiler.**  Arm B's compiled artifact vs
  arm A0's, on **24 held-out screenshots** (23,064 interior positions), both
  `validate=True` and `validate=False`, value *or* exception type at the
  identical edge.  Digests recorded per arm.
* **Gate B — against the typed interpreter.**  `Program.run` on the same cases.
* **Gate C — the repository.**  `.venv/bin/python -m pytest -q`, and the shipped
  fixture reproducing `0.248836 → 0.002231` at `4.0 / 4.0` frozen.
* **FALSIFICATION F2 — any gate failure means the declaration is unsound in
  practice.**  It is reported with the record count in R5a's form
  (`k / N` records), and the arm is dropped, not repaired after the fact.

### 2.3 The three cost measures, all reported (§51)

1. **Executed primitives**, worst and expected — `Registry.exact` applications,
   module invocations excluded, §59 AMENDMENT 4's portable meter.
   **Registered prediction: 1.00× — the declared type does not change the
   program's operator count.  Any other value is a bug.**
2. **Executed CPython bytecodes** per case, exact and deterministic, by code
   object as well as in total, using §59's `attribute.py`.
3. **Wall clock**, batch-one warm, `research/lazy-latency/latency.py`'s protocol
   verbatim: pinned to core 19, gc collected then disabled around each sweep, a
   discarded warm-up sweep, 21 repeats with arms interleaved round-robin inside
   each repeat, medians with non-parametric 95 % CIs, 10,000-resample bootstrap
   CI on the ratio.  **Three independent runs.**
   **§59 established this host's instrument floor at ~0.6 % with a
   byte-identical null control that read 1.0061× with a CI excluding 1.0.  No
   wall-clock claim below ~0.6 % is believable and none will be made.**

### 2.4 Registered predictions for step 2

* **P2.1 — the 6 G2 index guards in `_m1` discharge; the 4 G1 range guards do
  not.**  `add`'s output type is its input type, so `a + 1` can never be proved
  inside a non-degenerate bound on `a`.  §59 §6.2's "would discharge … all four
  range guards" is predicted to be **wrong**, and this is registered *before*
  measuring.
* **P2.2 — arm B is predicted to be SLOWER than A0 in bytecodes and wall clock**,
  despite discharging guards, because `_fast_kind` (`tcn/compile.py:566`) returns
  `None` for **any** type carrying `bounds`, so every bounded scalar node leaves
  the inline range-test path for a `_canon_fn` **Python call** with five checks
  in it.  In `_m1` that trades 6 inline index checks for 4 function calls.
* **P2.3 — conditional arm B+, and the only core change registered here.**  If
  and only if P2.2 is confirmed by measurement, one minimal change to
  `tcn/compile.py` is made: `_fast_kind` accepts an integer-encoded carrier that
  carries `bounds`, and `_emit_scalar`'s inline path tests the **intersected**
  interval `_int_interval(t)` and raises the exception `_canon_body` would have
  raised at that value, in `_canon_body`'s order.  It is **behind a keyword
  argument `inline_bounded=False` on `compile_program`, off by default**, so no
  existing behaviour changes.  It is gated by A, B and C above with
  `inline_bounded=True`, and by new tests in `tests/test_compile.py` that drive
  a bounded carrier to both edges in both directions.
  **This arm is not run at all if P2.2 is falsified.**
* **FALSIFICATION F3 — the bound is sound but no `_m1` guard discharges.**  Then
  the obstruction is the emitter's lattice, not the type declaration, and §59's
  residue is mis-diagnosed.  Reported as such.
* **FALSIFICATION F4 — guards discharge but no cost measure moves above the
  instrument floor.**  Then `_m1`'s cost is not its guards; the bytecode
  attribution by code object is used to say what it is.
* **FALSIFICATION F5 — the three measures disagree** (§51 says they will).  All
  three are reported separately and none is collapsed into another.

---

## 3. Step 3 — what the declaration costs elsewhere

A declared bound narrows the type.  The following are measured, not assumed.

* **Search space.**  `tcn.search.space_size` for S0, S1, S2 in both arms, and
  for `rung3_root.py`'s S1'/S2'.  Registered prediction: **unchanged** — the
  choice nodes and their candidate lists are untouched.
* **Enumeration certificates.**  The exhaustive sweeps are re-run in the bounded
  arm and compared field by field with `out/rung3.json` and `out/rung3_root.json`:
  `space_size`, `evaluated`, `conforming`, `certificate`, `exhausted`, `unique`,
  and the selected `chosen`.  §59's baseline: S0 256/256 → 2 conforming
  (`complete`), S1 400/400 → 2 (`complete`), S2 25/25 → **1, `unique`**.
* **§33's parse.**  `rung3_root.py`'s scored parse must still be **227 / 227
  parent links (1.000)**, **12 / 12 exact trees**, **227 / 227 rectangles with 0
  spurious and 0 missing**, on the 12 held-out flat screens.
* **The `rung3_widgets` parse** (`out/rung3.json`) rectangle exactness.
* **FALSIFICATION F6 — if any of these moves, that is a REGRESSION**, reported as
  one however fast the artifact runs, and the bound is not recommended.

---

## 4. Discipline

* Enumeration certificates beside every synthesis number.
* Achieved difficulty reported, never the requested configuration.
* Every surprising headline re-derived from the raw JSON before it is written.
* Negative results preserved; "the algebra can express the bound, the bound is
  sound, and it still buys nothing" is a first-class deliverable.
* **Every amendment recorded with the number it replaced and its direction
  relative to this track's own hypothesis** — §51, §56 and §59 each recorded
  five.
* Large artifacts (zipapps, staging trees) stay out of git; every report JSON is
  committed.

## 5. What this track will not do

* It will not touch `research/motif-unification/` or `research/emitter-guards/`.
* It will not add a guard class, a copy-propagation, or any transform §59's
  `PREREGISTRATION.md` §0 excluded.
* It will not re-declare bounds on the `language` or `computer` artifacts; §58
  already records those carriers as bounded, and §59 §6.1's off-by-one on them is
  a separate change.
* It will not bank any wall-clock figure inside the ~0.6 % instrument floor.
