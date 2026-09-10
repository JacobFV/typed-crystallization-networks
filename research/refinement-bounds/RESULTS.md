# A declared refinement bound on the visual raster address

Track `research/refinement-bounds`, branch `worktree-agent-a0350f1fdeac805b2`.

**Baseline: `main` at `59252fc` with §59's three commits cherry-picked**
(`79778d8`, `ee15cb4`, `16e50df`) — so the emitter already has §59's interval
lattice and the test suite starts at **344**, not 337.  `PREREGISTRATION.md` was
committed at **`a99373f`, before any change to `research/visual-ladder/`,
`research/refinement-bounds/` or `tcn/`.**

Every number below is printed by `report.py` from `out/*.json` and can be
re-checked without re-running anything.

This is the job FINDINGS §59 §6.2 left: *"`visual`'s hot function loses every
guard to a missing refinement bound … a declared `bounds=(0, 3069)` on the
address type would discharge all six index guards and all four range guards in
the function that is 73.7 % of the artifact."*

---

## 0. The verdict, stated first

**The bound is sound at `(0, 3071)` and only after the clamp is moved inside the
addition; it is *not* sound at `(0, 3069)`, the value §59 named.  Declaring it
discharges 6 of the hot module's 10 guards and exchanges the other 4 — and makes
the artifact take 28–29 % MORE wall clock, because `tcn/compile.py` refused the
inline path to any carrier that declares a bound.  One pre-registered,
flag-gated, off-by-default change to that refusal turns the same declaration into
1.3222× fewer bytecodes and 1.084× wall clock.  No certificate moves.**

* **Soundness, proved not asserted.**  `min(p + d, L) == p + min(d, L − p)` is
  verified with **0 mismatches on 9,424,900 pairs** (the whole rectangle
  `[0, L]²`) plus both reachable products; the reachable range of **every one of
  the 331–340 address-typed nodes** is enumerated exhaustively over the whole
  candidate product; and the frozen program is executed on **24 held-out
  screenshots**.  All three agree.
* **§59's `(0, 3069)` is unsound and would have been R5a shipped rather than
  caught.**  `_m1` computes `a + 2`, whose declared type is `a`'s, and the
  largest reachable address is 3069, so `a + 2 = 3071` violates it on a
  **reachable** input.
* **`(0, 3071)` is also unsound for the scaffold as written**, and not marginally:
  `rect_scaffold` evaluates `pos + k·step` **before** the `min(·, 3069)` clamp,
  which reaches **6,138** in principle and **5,529 measured on real
  screenshots**, where `ha31`'s *minimum* over 24 screenshots is already 3,075
  (2,976 in the root variant) — i.e. **out of range on essentially every call**.
  **The clamp is load-bearing exactly here.**  This is §56's R5a in a new dress.
* **Gates: 3 arms × 48 differential comparisons against arm A0 on 24 held-out
  screenshots (961 interior positions each, 23,064 in total), both `validate`
  settings, value or exception at the identical edge, plus 6 typed-interpreter
  checks per arm — zero mismatches, digests equal.**
* **Six of `_m1`'s ten guards discharge; the other four are exchanged, one for
  one, for a refinement-bound check of the same shape.  §59's prediction is half
  right**: the six index guards go, the four range guards do not — `add`'s output
  type *is* its input type, so `a + 1` can never be proved inside a
  non-degenerate bound on `a`.  This was pre-registered as P2.1 before measuring.
* **Certificates: unchanged.**  Search spaces 256 / 400 / 25 in every arm, every
  sweep exhausted with the same certificate and the same conforming count, and
  §33's parse still **227 / 227 parent links and 12 / 12 exact trees**.
* **The instrument floor is re-established in this session, not cited**: arm `N`
  is arm A0 compiled twice, byte-identical source, and reads 0.9994 / 0.9967 /
  0.9972 with every CI containing 1.  Nothing inside ~0.6 % is claimed.

---

## 1. Step 1 — what the clamp guarantees, and what the type may declare

`soundness.py`, `static_ranges.py`; raw data `out/soundness.json`,
`out/static_ranges.json`.

### 1.1 Why the bound is not simply the clamp

`Registry.resolve` gives a `BINARY` operator the **same** output type as its
(identical) operand types — `tcn/operators.py:68-73`, `inferred = t = ts[0]`.  So
a bound declared on an address is inherited by every arithmetic result derived
from it, and each of those becomes a **runtime-enforced obligation**, not merely
a static fact: `tcn/compile.py` emits a bounds test for every bounded output.

That is the whole difficulty, and it cuts both ways.

| candidate bound | what breaks | verdict |
|---|---|---|
| **`(0, 3069)`** — what the clamp enforces, and what §59 named | `_m1` reads `obs[a]`, `obs[a+1]`, `obs[a+2]`; `a+1` and `a+2` carry `a`'s type; the largest reachable `a` is **3069**, so `a+2 = 3071` **violates the bound on a reachable input** | **UNSOUND** |
| **`(0, 3071)`** = `3WH − 1`, the last raster **byte** | admits `a + 2`; and `[0, 3071]` **closed** sits inside `[0, 3072)` **half-open**, so it still discharges the index obligation | sound **only after §1.3** |
| `(0, 3071)` on the scaffold **as written** | `rect_scaffold` computes `wa_k = add(pos, mul(step, k))` **before** the clamp, reaching **6,138** | **UNSOUND** |

§59 §6.1 records that *"a closed refinement bound `(0, N)` can never prove an
index into an `N`-tuple"* and costs `language` and `computer` a guard each.  Here
the same arithmetic works in our favour: the correct bound is `(0, N−1)`, and
`N−1` is also exactly the largest value `a + 2` can take.  **The two constraints
meet at one value, and it is not the clamp's.**

### 1.2 The clamp is load-bearing, measured

`out/soundness.json`, from the frozen program run node by node on 24 held-out
screenshots — **23,064 corner calls and 431 rect calls** in the widgets variant,
**24,576 and 455** in the root variant:

| | widgets variant | root variant |
|---|---|---|
| addresses handed to `same` | **[3, 3069]** | **[0, 3069]** |
| derived `a + 2`, max | **3071** | **3071** |
| **every `IDX`-typed node, observed** | **[0, 5529]** | **[0, 5529]** |
| `ha31 = pos + 96·31`, observed | min **3075**, max **5529** | min **2976**, max 5529 |

The pre-clamp address is out of range on essentially every call, not at a corner
case.  §56's R5a failed on 243 of 2,883 records for deleting this clamp; the
same fact makes a naive bound on the address unsound.

### 1.3 The reformulation, and its exhaustive certificate

For `0 ≤ p ≤ L` and `d ≥ 0`:

```
min(p + d, L)  ==  p + min(d, L − p)
```

Both compute the clamped address; only the *intermediate* differs, and the right
spelling never leaves `[0, L]`.  `soundness.py::clamp_identity` checks it
**exhaustively, on three domains**:

| domain | pairs | mismatches | max pre-clamp | max reformulated |
|---|---|---|---|---|
| reachable, 961 interior positions × 155 deltas | 117,242 | **0** | 6,138 | **3,069** |
| reachable, all 1,024 positions × 155 deltas | 124,928 | **0** | 6,138 | **3,069** |
| the whole rectangle `[0, 3069]²` | **9,424,900** | **0** | 6,138 | **3,069** |

It is **one extra node** — `room = sub(last, pos)`, hoisted out of both loops.
Per multiplier, `{tag}a{k} = add(pos, {tag}d{k})` then `{tag}c{k} = min(·, last)`
becomes `{tag}p{k} = min({tag}d{k}, room)` then `{tag}c{k} = add(pos, ·)`: the
same two nodes, in the other order.  It is **off by default**
(`reformulated_clamp=False`), so every other track that imports this scaffold
builds the program it always built.

### 1.4 The static certificate: every address-typed node, every candidate

`static_ranges.py` enumerates the reachable range of **every** node whose
declared type is the address carrier, over the **whole** candidate product
(every position × every offset × every multiplier), with the data-dependent nodes
taken at their *type-level* extremes rather than anything observed.  The node set
is read off the actual `Program` and the enumeration must cover it name for
name — `coverage_complete` is `true` in all four rows below or the file raises.

| scaffold | address nodes | reachable range | `(0, 3071)` sound? | `(0, 3069)` sound? |
|---|---|---|---|---|
| `rung3_widgets`, as written | 331 | **[0, 6138]** | **no** | no |
| `rung3_widgets`, reformulated | 332 | **[0, 3071]** | **YES** | no |
| `rung3_root`, as written | 339 | **[0, 6138]** | **no** | no |
| `rung3_root`, reformulated | 340 | **[0, 3071]** | **YES** | no |

The six largest nodes in the reformulated arm are `own_b`, `b_b`, `a_b` at
**3071** and `own_g`, `b_g`, `a_g` at **3070** — i.e. **the bound is tight**: it
is attained, by `a + 2`, and by nothing else.  A bound of `(0, 3070)` would be
unsound and `(0, 3072)` would stop proving the index.  **There is exactly one
admissible bound.**

**P1.1, P1.2 and P1.3 are all confirmed as registered.  F1 did not fire: the
bound is sound, and the soundness argument is a proof, not a plausibility.**

---

## 2. Step 2 — declare the bound and re-measure

### 2.1 The arms

All four use **the same emitter**.  The variable is the declared type.

| arm | address carrier | clamp | emitter |
|---|---|---|---|
| **A0** | `IDX = u16`, no bound | `min(pos + d, last)` | §59, as shipped — **§59's `visual` artifact byte for byte** |
| **A1** | `IDX`, no bound | `pos + min(d, last − pos)` | §59 |
| **B** | `ADDR = u16 bounds(0, 3071)` | reformulated | §59 |
| **B+** | `ADDR` | reformulated | §59 **+ `inline_bounded=True`** (§2.5) |
| **N** | A0 compiled a second time — **byte-identical source** | | the **null control** |

`A1 − A0` isolates the reformulation; `B − A1` isolates the bound; `B+ − B`
isolates the one emitter line.

### 2.2 Gates — nothing was timed before these passed

`gate.py`, raw data `out/gate.json` and `out/gate_inline.json`.  Reference is
arm A0's compiled artifact (source SHA-256 `20e45f6b5e87…`).  Domain: **24
held-out screenshots, seeds 200–223, 961 interior positions each**, both
`validate=True` and `validate=False`, value **or** exception type and message at
the identical edge.

| arm | gate A | mismatches | digests equal | gate B (typed interpreter) |
|---|---|---|---|---|
| A1 | **PASS** | **0 / 48** | yes | **PASS**, 6 checked |
| B | **PASS** | **0 / 48** | yes | **PASS**, 6 checked |
| B+ | **PASS** | **0 / 48** | yes | **PASS**, 6 checked |

`all_passed: true` in both files.  **F2 did not fire.**  Bit-identity was then
re-asserted on the three timed cases before any timing (`identity_before_timing`,
`all_identical: true` in every run).

Gate C — the repository — is §5.

### 2.3 Guards, per emitted function

`attribute.py`, raw data `out/attribute.json`; one screenshot (seed 200).
**Calls per screenshot: `_m1` 3,100, `_m0` 961, `_m2` 19** — §59's figures,
reproduced.

| | A0 (and A1) | B | B+ |
|---|---|---|---|
| `_m1` guards | **4 G1 + 6 G2 = 10 inline checks** | **0 inline checks, 4 `_c3` calls** | **4 G1b = 4 inline checks** |
| `_m0` guards | 2 G1 | 0 inline, 2 `_c3` calls | 2 G1 + 2 G1b |
| `_m2` guards | 69 G1 + 6 G2 kept, 191 G1 gone | 0 inline, 261 `_c3` calls | 4 G1 + 70 G1b kept; 257 G1 + 191 G1b + 6 G2 gone |
| whole artifact, G2 index | **0 of 12 eliminated** | **12 of 12 eliminated** | **12 of 12 eliminated** |
| `_c3` calls per screenshot | 0 | **19,281** | 0 |

**`_m1`'s six index guards discharge in both B and B+.  Its four range guards do
not discharge in either** — in B they leave the inline path for a function call,
in B+ they are replaced one-for-one by a bounds check.  **P2.1 confirmed;
§59 §6.2's "all four range guards" is wrong, and was pre-registered as wrong.**

The whole emitted body of `_m1` in each arm is in `out/emitted_m1.txt`.  Excerpt
— the same two nodes, `a_g := add(a, one)` and `a_r_v := index(obs, a)`, in
emission order with the generated comments stripped:

```python
# A0                                  10 inline checks in the function
v16 = v13 + _k7
if not 0 <= v16 <= 65535: _ovf(v16, 0, 65535)
if not 0 <= v13 < 3072: raise IndexError('index outside tuple')
v18 = v15[v13]

# B    the index checks are gone -- and every arithmetic node is now a CALL
v16 = _c3(v13 + _k7)
v18 = v15[v13]

# B+   the index checks are gone and the arithmetic stays inline
v16 = v13 + _k7
if not 0 <= v16 <= 3071: raise ValueError('value outside semantic bounds')
v18 = v15[v13]
```

`_c3` is the thing arm B pays 19,281 times per screenshot:

```python
def _c3(x):
    if not _isfinite(x): _nonfinite()
    if not 0 <= x <= 3071: raise ValueError('value outside semantic bounds')
    n = int(x)
    if n != x: raise ValueError('fractional value requires explicit quantization')
    if not 0 <= n <= 65535: raise OverflowError('%s outside [0, 65535]' % n)
    d = n
    if not 0 <= d <= 3071: raise ValueError('encoded value violates domain')
    return d
```

### 2.4 Three cost measures

`measure.py`, raw data `out/measure_run1.json`, `run2`, `run3`.  Timing protocol
is `research/lazy-latency/latency.py`'s verbatim, reached through §59's
`measure.py` so no third harness exists: core 19, gc collected then disabled
around each sweep, a discarded warm-up sweep per arm, **21 repeats with all five
arms interleaved round-robin inside each repeat**, medians with non-parametric
95 % CIs, 10,000-resample bootstrap CI on the ratio.

#### 2.4.1 Executed primitives — and a registered prediction that was wrong

| arm | worst | expected | A0 ÷ arm |
|---|---|---|---|
| A0, N | 61,246 | **59,793.0** | 1.0000 |
| A1, B, B+ | 61,265 | **59,811.0** | **0.99970** |

**AMENDMENT 1 (below) — this was registered as exactly 1.00× and it is not.**
The **bound** costs nothing (B and B+ are identical to A1 at 59,811.0), but the
**reformulation** adds one `sub` node, executed once per `_m2` call, and `_m2`
runs 19 times: **+18 primitives per screenshot expected, +19 worst, exactly
accounted for.**  It is +0.030 %.

#### 2.4.2 Executed bytecodes — exact and deterministic

| arm | expected / screenshot | worst | **A0 ÷ arm** |
|---|---|---|---|
| A0 | **602,748.3** | 616,103 | 1.0000 |
| N | 602,748.3 | 616,103 | 1.0000 |
| A1 | 603,000.3 | 616,369 | 0.9996 |
| **B** | **1,143,954.3** | 1,173,922 | **0.5269** |
| **B+** | **455,868.3** | 465,868 | **1.3222** |

A0's 602,748.3 is **§59's `visual` figure to the decimal**, which is the
cross-check that these are the same artifact.

Two derived figures, so nobody has to compute them:

* **against arm A1** — i.e. the bound alone, with the reformulation already in
  both arms — B+ is **1.3228×** in bytecodes and **1.0866 / 1.0860 / 1.0863×**
  in wall clock.  The reformulation contributes nothing either way, so the whole
  effect is the declared bound.
* **against the emitter before §59** — §59 §4.2 records `visual` at
  **635,400.3** bytecodes without its guard elimination, so A0 → B+ compounds to
  **635,400.3 ÷ 455,868.3 = 1.394×** on that measure.  §59's own elimination is
  1.0542× of that; **the declared bound is the larger half.**

Bytecodes by code object, one screenshot:

| function | calls | A0 | B | B+ | B+ ÷ A0 |
|---|---|---|---|---|---|
| **`_m1`** | 3,100 | **468,100** (151.0/call) | 213,900 **+ 533,200 in `_c3`** | **300,700** (97.0/call) | **−35.8 %** |
| `_m2` | 19 | 74,681 | 71,071 + `_c3` | 74,548 | −0.2 % |
| `_m0` | 961 | 51,894 | 38,440 + `_c3` | **69,192** | **+33.3 %** |
| `_c3` | 0 / 19,281 / 0 | — | **829,083** | — | — |
| `<genexpr>` | — | 21,363 | 21,363 | 21,363 | 0 |

`_m0`'s **+33 %** is the honest cost of the bound, and it is worth stating
plainly.  `back_a = sub(pos, off3)` has an interval of `[−3, 3068]` and
`back_b = sub(pos, off96)` one of `[−96, 3068]`; a negative lower endpoint
discharges **neither** the width check nor the bound check, so each of those two
nodes pays **two** inline tests where it used to pay one:

```python
# B+, _m0                             two checks where A0 emitted one
v10 = v8 - _k4
if not 0 <= v10 <= 3071: raise ValueError('value outside semantic bounds')
if not 0 <= v10 <= 65535: _ovf(v10, 0, 65535)
```

`_m0` is 961 calls against `_m1`'s 3,100, so the net is still strongly positive —
but see §7.8: the second of those two lines is **provably unreachable given the
first**, and this pass does not notice, by design.

#### 2.4.3 Wall clock, batch-one warm, µs per screenshot

| arm | run | median µs | **A0 ÷ arm** | 95 % CI |
|---|---|---|---|---|
| **N** *(null control, byte-identical source)* | 1 | 4,939.88 | 0.9994 | [0.9895, 1.0075] ∋ 1 |
| | 2 | 4,820.65 | 0.9967 | [0.9932, 1.0032] ∋ 1 |
| | 3 | 4,811.12 | 0.9972 | [0.9863, 1.0079] ∋ 1 |
| **A1** *(reformulation only)* | 1 | 4,935.24 | 1.0003 | [0.9896, 1.0129] ∋ 1 |
| | 2 | 4,818.95 | 0.9970 | [0.9926, 1.0043] ∋ 1 |
| | 3 | 4,806.20 | 0.9982 | [0.9873, 1.0084] ∋ 1 |
| **B** *(the bound, declared)* | 1 | 6,342.17 | **0.7784** | [0.7689, 0.7823] |
| | 2 | 6,190.04 | **0.7762** | [0.7723, 0.7812] |
| | 3 | 6,211.78 | **0.7723** | [0.7630, 0.7784] |
| **B+** *(the bound + one emitter line)* | 1 | 4,541.81 | **1.0870** | [1.0727, 1.1024] |
| | 2 | 4,437.23 | **1.0828** | [1.0768, 1.0917] |
| | 3 | 4,424.30 | **1.0844** | [1.0715, 1.0924] |

A0's medians are 4,936.77 / 4,804.73 / 4,797.53 µs.

**The null control is what licenses the claims, and it is this session's own.**
`N` is arm A0 compiled a second time; the two modules have the **same source
SHA-256** (`out/measure_run*.json`, `null_control_pairs: [["A0", "N"]]`), so any
difference on it is instrument bias.  It reads within **0.33 %** in all three
runs with every CI containing 1, which reproduces §59's ~0.6 % floor
independently.  **Therefore:**

* **A1 is not distinguishable from A0** — the reformulation is free.  Its
  bytecode cost is exact and tiny (+252, +0.04 %); its wall-clock effect sits
  inside the floor and **no claim is made from it**.
* **B runs at 0.772–0.778× A0's speed — 28.5 %, 28.8 % and 29.5 % more wall
  clock in the three runs, ~35× the floor.  Declaring the bound, on its own, is a
  regression.**  P2.2 confirmed.
* **B+ is 1.0870× / 1.0828× / 1.0844× — 8.3–8.7 % in §59's convention, 7.7–8.0 %
  less wall clock — ~14× the floor, and no CI contains 1 in any run.**

Source bytes: A0 161,352 → A1 161,570 → **B 158,465** → **B+ 163,219**.  Note
that arm B is the *smallest* artifact and the *slowest*: a `_c3(…)` call is
fewer characters than an inline test.  **Bytes on disk say the opposite of wall
clock here, which is a fourth measure disagreeing with the other three.**

### 2.5 The one core change, registered in advance and fired by measurement

`PREREGISTRATION.md` §2.4 P2.3 registered this **conditionally**, before any
measurement: *"if and only if P2.2 is confirmed."*  It was.

`tcn/compile.py:566` read

```python
if t.kind != "int" or t.bounds is not None:
    return None
```

`_fast_kind` refused the inline range-test path to **any** carrier declaring a
refinement bound, sending it to `_canon_fn` — a Python call with **five tests and
an `int()`** in it, and a frame to build and tear down.
**That one line is why a declared bound cost 1.90× the bytecodes it saved.**

The change is **52 added and 11 removed lines** in `tcn/compile.py` — about half
of them comment — and is **behind `compile_program(..., inline_bounded=False)`,
off by default**:

* `_fast_kind` accepts a bounded integer-encoded carrier when the flag is set;
* `_emit_scalar` emits **`_canon_body`'s two tests, in `_canon_body`'s order,
  with `_canon_body`'s exceptions** — the refinement bound first
  (`ValueError('value outside semantic bounds')`, which is also what the *typed
  interpreter* raises, via `Value.of` → `encode`), then the width
  (`_ovf` → `OverflowError`) — and **drops each one only where the expression's
  interval already discharges it**;
* the interval recorded for the result is the expression's interval met with
  `_int_interval(t)`, which for an unbounded carrier is exactly `(lo, hi)`, so
  §59's bookkeeping is unchanged byte for byte.

**With the flag off, every byte of emitted source is what §59 shipped**, and this
is checked rather than asserted: `inert.py` reads `tcn/compile.py` out of git at
§59's `ee15cb4` — not a copy — compiles all three arms with it, and compares
SHA-256 against the patched compiler at `inline_bounded=False`.
**`all_identical: true`** (`out/inert.json`; A0 161,352 B, A1 161,570 B,
B 158,465 B), arm B included deliberately, since a bounded carrier is the only
thing the patch can reach.
`test_inline_bounded_is_off_by_default_and_the_default_source_is_unchanged` pins
the same fact as a unit test.

Guard counts it unlocks, whole artifact:

| class | A0 | B | **B+** |
|---|---|---|---|
| G1 integer range | 191 eliminated / 266 | — (class does not occur; every node is a `_canon_fn` call) | **261 eliminated / 267** |
| G1b refinement bound | n/a | n/a | **191 eliminated / 267** |
| G2 dynamic index | **0 / 12** | **12 / 12** | **12 / 12** |

**Five new tests** in `tests/test_compile.py` (33 → 38) drive a bounded carrier to
both edges in both directions, in both compilations, against the interpreter:
the flag is off by default and the source is unchanged; the identical exception
at the identical edge; the index guard discharged only because the bound is
re-established by the check on the addition; a bound the carrier cannot itself
enforce, where both tests survive **and the bound must be tested first**; and the
external boundary unchanged.

### 2.6 Falsification, condition by condition

| pre-registered condition | outcome |
|---|---|
| **F1 — the bound cannot be proved sound → report and stop** | **DID NOT FIRE.**  §1: exhaustive on 9.4 M pairs, exhaustive over the candidate product, and executed on 24 held-out screenshots.  But it fired *at* `(0, 3069)`, the bound §59 named — see §4. |
| **F2 — any gate fails** | **DID NOT FIRE.**  0 / 48 in three arms, digests equal, plus 18 interpreter checks. |
| **F3 — the bound is sound but no `_m1` guard discharges** | **DID NOT FIRE.**  6 of 10 discharge. |
| **F4 — guards discharge but no cost measure moves above the floor** | **FIRED FOR ARM B, in the wrong direction.**  Twelve index guards discharged, six of them in `_m1`, and the artifact took **29 % more wall clock**.  The cost of `_m1` was not only its guards; it was what the emitter does to a bounded carrier. |
| **F5 — the three measures disagree** | **FIRED, in both directions.**  For B+: primitives **+0.030 %** (slightly *more* work), bytecodes **−24.4 %**, wall clock **−7.8 %**.  For B: bytecodes **+89.8 %** against wall clock **+28.9 %** — the same change, two magnitudes that differ threefold.  Source bytes disagree with all three: **arm B is the smallest artifact and the slowest.** |
| **F6 — any certificate moves** | **DID NOT FIRE.**  §3. |

---

## 3. Step 3 — what the bound costs elsewhere

`certificates.py`, raw data `out/certificates_space.json`,
`out/certificates_search.json`, `out/certificates_parse.json`.

### 3.1 The search space

| | S0 `same` | S1 `corner` | S2 `rect` |
|---|---|---|---|
| `out/rung3.json` (reference) | 256 | 400 | 25 |
| `rung3_widgets`, `IDX` | 256 | 400 | 25 |
| `rung3_widgets`, bounded | **256** | **400** | **25** |
| `out/rung3_root.json` (reference) | — | 400 | 25 |
| `rung3_root`, `IDX` | 256 | 400 | 25 |
| `rung3_root`, bounded | **256** | **400** | **25** |

**Unchanged, as registered.**  A refinement bound narrows the *carrier*; it does
not touch a choice node or its candidate list.  Node counts move by **exactly
one**, and only in S2 — `rung3_widgets` 585 → 586, `rung3_root` 586 → 587 — which
is `room`, i.e. the reformulation, not the bound.  S0 (15) and S1 (7 / 22) are
unchanged.

### 3.2 The exhaustive sweeps and their certificates

Every sweep re-run in both arms, on both scaffolds, with
`common.sweep` → `tcn.search.enumerate_prefix`; **12 sweeps, ~48 minutes**.

| scaffold / arm | stage | space | evaluated | conforming | certificate | exhausted | unique | chosen |
|---|---|---|---|---|---|---|---|---|
| **reference** `out/rung3.json` | S0 | 256 | 256 | 2 | complete | — | false | `rg 7, same 2` |
| `rung3_widgets`, `IDX` | S0 | 256 | 256 | 2 | complete | true | false | `rg 7, same 2` |
| `rung3_widgets`, **bounded** | S0 | **256** | **256** | **2** | **complete** | **true** | **false** | **`rg 7, same 2`** |
| **reference** `out/rung3.json` | S1 | 400 | 400 | 2 | complete | — | false | `back_a 2, back_b 3, corner 1` |
| `rung3_widgets`, `IDX` | S1 | 400 | 400 | 2 | complete | true | false | same |
| `rung3_widgets`, **bounded** | S1 | **400** | **400** | **2** | **complete** | **true** | **false** | **same** |
| **reference** `out/rung3.json` | S2 | 25 | 25 | **1** | **unique** | — | true | `step_w 2, step_h 3` |
| `rung3_widgets`, `IDX` | S2 | 25 | 25 | 1 | unique | true | true | same |
| `rung3_widgets`, **bounded** | S2 | **25** | **25** | **1** | **unique** | **true** | **true** | **same** |
| `rung3_root`, `IDX` | S1 / S2 | 400 / 25 | 400 / 25 | 2 / **1** | complete / **unique** | true | false / true | as `out/rung3_root.json` |
| `rung3_root`, **bounded** | S1 / S2 | **400 / 25** | **400 / 25** | **2 / 1** | **complete / unique** | **true** | **false / true** | **as `out/rung3_root.json`** |

**Nothing moves.**  `node_evaluations` is identical to the reference to the unit
in every stage but S2, where it is **1,605,195 → 1,605,306 (+111)** on
`rung3_widgets` and **1,694,889 → 1,695,006 (+117)** on `rung3_root`.  Those are
exactly the **111** and **117** training records: the reformulation's single
`room` node is evaluated once per record in the shared prefix.  **The whole
difference in the enumeration is one node, and it is accounted for to the unit.**

### 3.3 §33's parse — the certificate that matters most

`rung3_root.py`'s parser, re-frozen in each arm from the same stored selections
and run through the **typed interpreter** (not the compiler) on the 12 held-out
flat screens, scored with `rung3_root.score_with_root`:

| arm | screens | rectangles | screens exact | roots | parent links | links wrong | **exact trees** |
|---|---|---|---|---|---|---|---|
| `IDX` (as `main`) | 12 | **227 / 227** | 12 / 12 | 12 | **227 / 227** | **0** | **12 / 12** |
| **bounded + reformulated** | 12 | **227 / 227** | 12 / 12 | 12 | **227 / 227** | **0** | **12 / 12** |

**§33's result is reproduced exactly and is unmoved by the declared bound.**
`out/certificates_parse.json` carries the per-screen rows; every one has
`rects_exact: true`, `parent_links_wrong: 0`, `tree_exact: true`,
`roots_predicted: 1` in both arms.

**F6 did not fire.  No certificate moves.**

---

## 4. §59's residue, re-diagnosed

§59 §6.2 wrote: *"A declared `bounds=(0, 3069)` on the address type would
discharge all six index guards and all four range guards in the function that is
73.7 % of the artifact — and `(0, 3069)` is exactly §56's load-bearing
`min(·, 3069)` clamp."*

Three separate corrections, all of them pre-registered before measurement:

1. **`(0, 3069)` is unsound.**  It is violated by `a + 2` on a reachable input.
   The clamp bounds the *address*; the type must also admit the two *derived*
   addresses the module reads.  The admissible bound is `(0, 3071)`, and §1.4
   shows it is attained — there is exactly one.
2. **The four range guards do not discharge.**  `add`'s output type is its input
   type, so `a + 1` is outside any non-degenerate bound on `a` by construction.
   What happens instead is an *exchange*: the width check is discharged and a
   bound check takes its place.  The hot function goes from **10 inline checks to
   4**, not from 10 to 0.
3. **§59 §6.3 named the right obstruction but priced it as an error contract
   rather than a compiler one.**  It says the coupling "changes the arithmetic's
   error contract (3070 would now raise) and pushes the node off `_fast_kind`'s
   inline path onto a `_canon_fn` call", and concludes *"that coupling, not the
   compiler, is what stands between `visual` and the rest of this win."*  On
   measurement it is **the other way round**: the error contract is *identical*
   at `(0, 3071)` — 48 differential comparisons on 24 screenshots, both validate
   settings, zero mismatches — and the `_canon_fn` fallback is **the whole of the
   cost**.  Removing it moves the same declared bound from **0.776× to 1.084×**
   wall clock, a **1.40× swing**, and from 1.90× the bytecodes to 0.756×.  It is
   a compiler problem, and 52 flag-gated lines in `_fast_kind` and
   `_emit_scalar` fix it.

---

## 5. Repository state and reproduction

* **`.venv/bin/python -m pytest -q` → 348 passed, 1 failed of 349.**  The
  baseline is **344** (§59's branch); this track adds **5** tests to
  `tests/test_compile.py` (33 → 38).  The single failure is
  `tests/test_panel_interface.py::test_panel_episode_replays_and_restores`, the
  known worktree-only replay divergence (FINDINGS §41, §59,
  `research/MERGE-QUEUE.md`).  **Verified environmental here, not assumed**: with
  `tcn/compile.py` restored to `main` at `59252fc` — i.e. without even §59's
  lattice — `pytest tests/test_panel_interface.py` still gives `1 failed,
  9 passed`.  The thirteen-failure mode is the *missing*
  `generators/computer/engine/node_modules` symlink, not its presence; with it
  restored the count is as above.
* **Shipped fixture reproduces exactly.**  `.venv/bin/python -m tcn train
  --episodes 160` gives `initial_prediction_loss 0.248835613951087`,
  `final_prediction_loss 0.0022308224288281053`, `fully_frozen true`,
  `evaluation_mean_return 4.0`, `frozen_evaluation_mean_return 4.0` —
  **0.248836 → 0.002231 at 4.0 / 4.0 frozen.**
* Every scaffold change is **flag-gated and off by default** (`addr=None`,
  `reformulated_clamp=False`), so every other track that imports
  `research/visual-ladder/` builds the identical program.  The core change is
  `inline_bounded=False` by default.
* `research/motif-unification/` and `research/emitter-guards/` were not touched.
* No large artifact is committed; every report JSON is.

```bash
.venv/bin/python research/refinement-bounds/soundness.py        # sec 1.2, ~20 min
.venv/bin/python research/refinement-bounds/static_ranges.py    # sec 1.4, ~1 min
.venv/bin/python research/refinement-bounds/gate.py A1 B        # sec 2.2, ~4 min
.venv/bin/python research/refinement-bounds/gate.py B --inline-bounded
.venv/bin/python research/refinement-bounds/attribute.py        # sec 2.3, ~2 min
.venv/bin/python research/refinement-bounds/emitted.py          # sec 2.3, the source
.venv/bin/python research/refinement-bounds/inert.py            # sec 2.5, ~2 min
taskset -c 19 .venv/bin/python research/refinement-bounds/measure.py run1   # sec 2.4
taskset -c 19 .venv/bin/python research/refinement-bounds/measure.py run2
taskset -c 19 .venv/bin/python research/refinement-bounds/measure.py run3
.venv/bin/python research/refinement-bounds/certificates.py space
.venv/bin/python research/refinement-bounds/certificates.py search   # ~50 min
.venv/bin/python research/refinement-bounds/certificates.py parse    # ~15 min
.venv/bin/python research/refinement-bounds/report.py           # every number above
```

| file | what |
|---|---|
| `PREREGISTRATION.md` | the protocol, at `a99373f`, before any change |
| `arms.py` | the four arms, built from one place |
| `soundness.py` | §1.2 — the clamp identity and the observed ranges |
| `static_ranges.py` | §1.4 — every address node, every candidate, exhaustively |
| `gate.py` | §2.2 — differential against arm A0 and against the typed interpreter |
| `attribute.py` | §2.3 — guards and bytecodes by emitted function |
| `emitted.py` | §2.3 — the generated source of `_m1` in each arm |
| `inert.py` | §2.5 — the core change is byte-inert with the flag off, checked against the emitter read out of git at `ee15cb4`; and `harden_all == harden(chosen)` |
| `measure.py` | §2.4 — three cost measures with the null control |
| `certificates.py` | §3 — space, sweeps, and §33's parse |
| `report.py` | prints every number above from `out/*.json` |

---

## 6. Amendments to the pre-registration

Each with the number it replaced and its direction relative to this track's own
hypothesis, to the standard of §51, §56 and §59.

**AMENDMENT 1 — the registered "executed primitives = exactly 1.00×" is wrong,
and it moves against this track.**  `PREREGISTRATION.md` §2.3(1) registered
*"1.00× — the declared type does not change the program's operator count.  Any
other value is a bug."*  Measured: **59,793.0 → 59,811.0 expected, 61,246 →
61,265 worst — 1.00030×, not 1.00000×.**  The *bound* is exactly 1.00× (arms B
and B+ are identical to A1); the **reformulation** adds one `sub` node per `_m2`
call and `_m2` runs 19 times, which accounts for the difference to the unit.
The registered sentence conflated "declaring a bound" with "the arm that
declares it", and the arm also carries a program change.  Both figures are in
§2.4.1; the prediction is recorded as **falsified**, not reinterpreted.

**AMENDMENT 2 — the candidate bound moved from `(0, 3069)` to `(0, 3071)` before
any measurement, and it moves *for* this track, which is why it is stated
loudly.**  `PREREGISTRATION.md` §1.2 P1.1 registered `(0, 3069)` — §59's value —
as *predicted unsound*, and §1.2 P1.2/P1.3 registered `(0, 3071)` as the
candidate.  That reasoning was done **before** `soundness.py` was written and is
recorded in the pre-registration commit, so it is a prediction rather than a
post-hoc repair; but a reader should know that the bound actually measured is
**not** the one §59 named, and that the change was made by this track.

**AMENDMENT 3 — the core change was conditional and the condition was checked
before it was written.**  `PREREGISTRATION.md` §2.4 P2.3 makes `inline_bounded`
contingent on P2.2 being confirmed by measurement.  Arm B was built, gated and
attributed **first** (`out/attribute.json`: 19,281 `_c3` calls per screenshot,
829,083 bytecodes), and only then was `tcn/compile.py` touched.  This is the
opposite of §59's AMENDMENT 1, where a lattice case was added *after* inspecting
a residue; here the residue was inspected against a rule written in advance.

**AMENDMENT 4 — the null control is arm A0 compiled twice, not a second
artifact.**  §59's null controls were the `mixed` and `computer` artifacts, whose
two arms happened to emit byte-identical source.  Those artifacts are ~0.4 µs and
~1.1 µs per case; `visual` is ~4,800 µs, four orders of magnitude away, so
§59's floor is not automatically this measurement's floor.  Arm `N` puts the null
control **at `visual`'s own magnitude**, and it reproduces the same ~0.6 % answer.
Neutral to the hypothesis; it is a strengthening of the instrument, added because
citing §59's number across four orders of magnitude would not have been honest.

**AMENDMENT 5 — `harden_all` replaced `Program.harden(chosen)`.**
`research/visual-ladder`'s `out/rung3.json` stores a `chosen` dict keyed by every
node name, so a scaffold that adds a node (`room`) cannot be hardened from it.
`arms.harden_all` freezes every non-choice node at its only candidate and takes
the searched nodes from `chosen`.  It is verified **identical** to
`harden(chosen)` on all three unmodified scaffolds by program digest —
`inert.py`, `out/inert.json`, `harden_all.{S0,S1,S2}.identical: true` — before
being used anywhere.  Neutral.

---

## 7. What this does NOT establish

At the same volume as the results.

1. **It does not establish that refinement bounds are worth declaring in
   general.**  It establishes it for **one** carrier on **one** artifact, where
   the same address is indexed 3,100 times per case.  On `_m0` the same bound is
   a **33 % bytecode regression**, and `_m0` is in the same artifact.
2. **It does not establish that `inline_bounded` should be on by default.**  It
   is one flag-gated path, gated on one artifact plus five unit tests.  §59's
   own verdict — that 247 lines of emitter for 1.10× on one artifact is a
   judgement call rather than a fact — applies here at 52 lines and 1.084×, and
   the judgement is again recorded rather than made silently.
3. **It does not re-open §56's R5a.**  R5a deleted the clamp; this moves it.  The
   clamp is still there, still executed, and `tests/test_compile.py::
   test_a_load_bearing_clamp_is_never_deleted_and_is_what_proves_the_index`
   still passes.
4. **It says nothing about the `language` and `computer` artifacts.**  §58
   records their carriers as already bounded (`u32<=128`, `u32<=4096`) and §59
   §6.1 records a closed-bound off-by-one costing each of them one index guard.
   Changing `(0, N)` to `(0, N−1)` on those is a different change, not made here.
   **But `inline_bounded` changes how *their* bounded nodes compile too**, and
   this track measured only `visual`.  That is a gap, and it is why the flag is
   off.
5. **The `validate=False` qualification of §59 §1.1 is inherited unchanged.**
6. **Nothing here touches quality.**  Every arm computes the identical function
   on 144 gated comparisons and 18 interpreter checks; a fast wrong answer is
   measured nowhere.
7. **Cold start, peak RSS and zipapp bytes were not measured.**  §59 measured and
   then disqualified them on this host; repeating a disqualified column would add
   nothing.
8. **The obvious next improvement is left undone on purpose, named and priced.**
   Where both checks are emitted the second is redundant: after
   `if not 0 <= v <= 3071: ValueError`, the test `if not 0 <= v <= 65535: _ovf`
   cannot fire, because `[0, 3071] ⊂ [0, 65535]`.  `_emit_scalar` decides the two
   independently from the *expression's* interval and never from the state after
   the first check, so both are written.  **Price: only 6 nodes in the whole
   artifact emit the width check at all — 2 in `_m0` at 961 calls per screenshot
   and 4 in `_m2` at 19 — so at most 1,998 redundant tests per screenshot, on the
   order of 3 % of arm B+'s bytecodes.**  It is not done here because
   `PREREGISTRATION.md` §2.4 P2.3 registered exactly two independently discharged
   tests, and widening a rule after seeing its residue is precisely §59's
   AMENDMENT 1 and §56 §9.2's warning.  It should be its own change.

**Difficulty achieved, stated rather than requested.**  The hard part was not the
emitter line.  It was noticing, before writing any code, that the bound the clamp
enforces is **not** the bound the type can declare — because the type algebra
propagates it to `a + 1` and `a + 2` — and then that the bound the type *can*
declare is violated **before** the clamp by a value the clamp exists to remove.
Both are §56's R5a wearing different clothes, and both are the kind of thing that
passes a plausibility check and fails a gate on 8 % of records.  The trap not
avoided is AMENDMENT 1: a registered prediction of exactly 1.00× that measurement
falsified at the fourth decimal, because the arm under test carried a program
change as well as a type change.
