# The residual 6.2×, attributed

Branch `worktree-agent-ae2285740a6b89153`, from `main` at `2b9cb2b`.  Nothing
under `tcn/`, `generators/`, `research/lazy-guard/`, `research/lazy-latency/`,
`research/compiled-runtime/` or `research/second-family/` is modified.  This
directory is a pure addition.

`PREREGISTRATION.md` was committed at **`6d875b8`, before any arm ran**.  Every
number below is printed by `report.py` from `out/*.json` and can be re-checked
against those files without re-running anything.

This is the question FINDINGS §51 left open.  §48 settled that the original
146×–90,400× was interpreter overhead and that compiling removes it.  §41
certified that ranking inside a fixed scaffold cannot shorten executed work.
§49 and §51 showed control flow is discoverable and worth ~2×.  §51 then
measured, on the shipped visual parser's S2 subroutine over its deployment
distribution, that the hand-written reference is **12.908×** faster than the
compiled specification while laziness recovers only **2.079×** of it, and
concluded: *"whatever closes the remaining gap is not a lazy conditional."*
Nobody had diagnosed what it is.  **≈6.2× was the last unexplained number in the
performance line.**

---

## 0. The verdict, stated first

**The residual is representational, not structural, and by a wide margin.**

On the deployment distribution, arm B (resynthesized) ÷ arm C (hand-written) is
**6.364×** on this host.  A cumulative chain of seven semantics-preserving
source transforms carries arm B's source to arm C's, each rung gated
bit-identical on all 2,883 held-out + 54 corner + worst-case records **and**
against the shipped typed interpreter on a 41-record stratified oracle.  The
log-time deltas therefore sum to the whole gap by construction:

| class | share of the 6.364× in log terms | factor |
|---|---|---|
| **representational** — four ordinary compiler passes | **93.7 %** | **5.666×** |
| **structural** — the one rung that changes executed operation count | **3.5 %** | **1.067×** |
| **unaccounted** — what still separates the transformed program from the human's | **2.8 %** | 1.052× |

Under the strictest possible reading — counting *every* rung that changes the
executed operation count as structural, which sweeps loop-invariant code motion
into the structural column — structural rises to **36.6 %** and
representational falls to **60.7 %**.  **Representational dominates on either
reading, on all three distributions, and in a second interpreter.**

**So the resynthesis programme's ceiling is near 2×, and that is a positive
finding for the compiler and a bounding one for the synthesizer.**  Of the full
13.61× that separates the compiled specification from the hand-written
reference (bare entry, deployment):

| cause | factor | log share | who owns it |
|---|---|---|---|
| **early exit / control-flow discovery** (§49, §51) | **2.139×** | **29.1 %** | the resynthesizer — *already banked* |
| short-circuiting a learned truth table | 1.067× | 2.5 % | the resynthesizer — the only headroom left |
| **four compiler passes** | **5.666×** | **66.4 %** | `tcn/compile.py` and `research/lazy-guard/ir.py`'s emitters |
| unaccounted | 1.052× | 2.0 % | — |

**About one third of the gap is structural and laziness has already claimed
nearly all of it.  Two thirds is ordinary code generation.**  A resynthesizer
that got everything else right would buy at most a further **1.067×** on this
subroutine.  Any plan that proposes to close the visual gap by making the
synthesizer smarter is bounded above by that number.

Two named engineering changes account for **89 %** of the residual between
them, and neither is research:

1. **Range- and bounds-check elimination** — **2.834×**, 56.3 % of the residual.
2. **Loop-invariant code motion and CSE in the emitter** — **1.844×**, 33.1 %.

---

## 1. Step 0 — does 12.908× reproduce?

It does.  Envelope-matched exactly as §51 timed it, µs per record, batch-one
warm, two full independent runs:

| distribution | | A compiled spec | B resynthesized | C hand-written | A ÷ B | **A ÷ C** | **B ÷ C** |
|---|---|---|---|---|---|---|---|
| **deployment** (54 corner) | run 1 | 36.749 | 17.288 | 2.802 | 2.126 | **13.114** | **6.169** |
| | run 2 | 37.495 | 17.415 | 2.927 | 2.153 | 12.812 | 5.951 |
| | **§51** | 39.043 | 18.781 | 3.025 | 2.079 | **12.908** | 6.208 |
| uniform held out (2,883) | run 1 | 35.635 | 9.612 | 1.420 | 3.707 | 25.094 | 6.769 |
| | §51 | 38.031 | 10.576 | 1.532 | 3.596 | 24.820 | 6.903 |
| **worst case** (`w+h` = 47) | run 1 | 34.728 | 37.775 | 5.304 | **0.919** | 6.547 | 7.122 |
| | §51 | 37.025 | 40.892 | 5.749 | **0.905** | 6.440 | 7.113 |

All three arms are 3–6 % faster here than in §51 — a whole-table shift, the
host being less loaded (`loadavg` 2.5–3.8 against §51's 4.6–5.9) — and every
ratio reproduces: 12.908 → 13.114/12.812, 2.079 → 2.126/2.153, and **6.208 →
6.169/5.951**.  §51's worst-case sign reproduces too: B is 1.09× *slower* than
A there.

**The residual is nearly distribution-independent, and that is itself
evidence.**  B ÷ C is 6.17× on deployment, 6.77× on uniform and 7.12× at worst,
where A ÷ B swings 2.13× / 3.71× / **0.92×** across the same three rows.  A
cause that is a constant factor per unit of work looks like the first row; a
cause that is algorithmic looks like the second.

---

## 2. The gate — nothing was timed before this passed

`gate.py`, raw data `out/gate.json`; `interp.py`, raw data `out/interp.json`.

| rung | uniform 2,883 | deploy 54 | worst | verdict |
|---|---|---|---|---|
| R0 … R7 (all eight) | 0 mismatches | 0 | 0 | **PASS** |
| **R5a** (guard elimination *including* the index clamp) | **243 mismatches** | 0 | 0 | **FAIL — `IndexError: tuple index out of range`** |

`ladder_gate_passed: true`.  A SHA-256 digest over each rung's canonical outputs
is recorded per distribution in `out/gate.json` and every digest is equal, so
the claim is checkable without re-running.

**The interpreter link is re-run here, not merely inherited.**  §51's own gate
sampled 32 records because the typed interpreter costs 0.454 s per record.  This
track ran its own: **41 records — one per achieved `w + h` stratum across all 39
strata, six corner records and the worst case** — through
`Program.run` on `Value(input_type, rec)`, and **all eight rungs plus arm A
agree with the interpreter on all 41** (`all_agree: true`).

### 2.1 The failed rung is a finding, and it was pre-registered as the one at risk

R5a removes the `min(·, 3069)` index clamp along with the other typed guards.
It fails on **243 of 2,883** held-out records.  The reason is exact: the height
loop's last iteration evaluates `obs[pos + 96·i]` at `i = 32 − y`, which is
`3x + 3072 ≥ 3075`, outside the 3,072-element raster.  **The clamp is not dead
code.**  It is load-bearing *because the synthesizer chose to evaluate the pixel
test before the bounds test*, and no compiler can delete it without reordering
that conjunction — which is exactly what rung R6 does, and only then does the
clamp become removable.  A cost the synthesizer's operand ordering imposes on
the compiler, measured.

It passes on all 54 corner records, so a track that gated only on the
deployment distribution would have shipped a program that crashes on 8.4 % of
the held-out set.

---

## 3. The ladder — the attribution

`rungs.py` builds every rung; `timing.py` times them on
`research/lazy-latency/latency.py`'s protocol verbatim (CPU-pinned under
`taskset -c 19`, gc collected then disabled around each sweep, one discarded
warm-up sweep per rung per distribution, 21 repeats with the rungs **interleaved
round-robin inside each repeat**, medians with non-parametric 95 % CIs,
bootstrap CIs with 10,000 resamples on every ratio).  Raw data
`out/latency_run1.json`, `out/latency_run2.json`.

**R1 and R2 are produced mechanically** from the committed
`research/lazy-guard/out/stage3_algorithm.py` by an `ast`-driven rewrite, so the
largest rewrite carries no transcription risk.  R3–R6 are written out and are
proved by the gate, not by inspection.  R7 is arm C verbatim.

### 3.1 Deployment distribution — the row that matters

Bare entry, µs per record.  `ratio` is the previous rung ÷ this rung.

| rung | transformation | run 1 | run 2 | **ratio** | 95 % CI | log share | class |
|---|---|---|---|---|---|---|---|
| **R0** | arm B verbatim | **17.181** | 17.393 | — | — | — | baseline |
| R1 | guard-call inlining (`_ck`/`_ix`/`_dz`/`_dzi`) | 14.508 | 14.564 | **1.184×** | [1.17, 1.20] | 9.1 % | repr |
| R2 | module-call inlining (`_m0`) | 14.196 | 14.348 | 1.022× | [1.01, 1.03] | 1.2 % | repr |
| **R3** | **loop-invariant code motion + CSE** | 7.699 | 7.724 | **1.844×** | [1.81, 1.87] | **33.1 %** | repr |
| **R4** | **short-circuit the truth-table conjunction** | 7.217 | 7.289 | **1.067×** | [1.05, 1.09] | **3.5 %** | **STRUCTURAL** |
| **R5** | **typed-guard elimination** (clamp retained) | **2.546** | 2.679 | **2.834×** | [2.78, 2.88] | **56.3 %** | repr |
| R6 | loop-bound strength reduction (`while` form) | 2.841 | 2.925 | **0.896×** | [0.87, 0.92] | −5.9 % | repr |
| R7 | **arm C verbatim** | **2.700** | 2.759 | 1.052× | [1.03, 1.08] | 2.8 % | *unaccounted* |

**R0 ÷ R7 = 6.364× (run 1), 6.304× (run 2).**  No CI on any rung contains 1.0.

**Two of the rungs deserve their own sentence.**

* **R5 alone is 2.834×** — more than a third of the whole 6.36× — and it is
  *nothing but* deleting integer range checks and tuple index guards that the
  declared types make unreachable on this domain.  It removes **1,909 of R4's
  2,904 bytecodes per record, 65.7 %**.
* **R5 overshoots arm C.**  At 2.546 µs the mechanically-transformed arm B is
  **6 % faster than the hand-written reference's 2.700 µs**.  There is no
  residual left to explain at that point; the last two rungs, which only bend
  the code toward arm C's exact source shape, cost time rather than saving it.

### 3.2 The same ladder on the other two distributions

Per §51's caution that worst case and expected case behave differently.

| rung | deployment | uniform | worst case |
|---|---|---|---|
| R1 guard-call inlining | 1.184× | 1.181× | 1.162× |
| R2 module-call inlining | 1.022× | 1.028× | 1.024× |
| R3 LICM + CSE | 1.844× | 1.837× | 1.926× |
| **R4 short-circuit (structural)** | **1.067×** | **1.085×** | **1.061×** |
| R5 typed-guard elimination | 2.834× | 3.063× | 3.435× |
| R6 strength reduction | 0.896× | 0.931× | 0.807× |
| R7 unaccounted | 1.052× | 1.010× | 1.079× |
| **R0 ÷ R7** | **6.364×** | **6.972×** | **7.260×** |
| **structural share (log)** | **3.5 %** | **4.2 %** | **3.0 %** |
| **representational share (log)** | **93.7 %** | **95.3 %** | **93.2 %** |
| unaccounted share (log) | 2.8 % | 0.5 % | 3.8 % |

**The verdict does not depend on the distribution.**  This is the opposite of
what §51 found for laziness, whose value swung from 3.60× to 0.905× across the
same three rows.

### 3.3 Out-of-process replication, Python 3.12.3

`outproc.py`, raw data `out/outproc.json`.  A fresh `/usr/bin/python3 -I` with
`PATH` only, no venv, no repository on the path — `leaked_modules: []`, no
`tcn`, no torch, no numpy — importing the committed `out/R*.py` and re-checking
every rung's output against R0 inside the child before timing.

| rung | 3.12.3 µs | ratio | log share | in-process 3.13.15 ratio |
|---|---|---|---|---|
| R0 | 14.945 | — | — | — |
| R1 | 10.256 | **1.457×** | 20.1 % | 1.184× |
| R2 | 9.727 | 1.054× | 2.8 % | 1.022× |
| R3 | 5.521 | 1.762× | 30.2 % | 1.844× |
| **R4** | 4.813 | **1.147×** | **7.3 %** | 1.067× |
| R5 | 2.786 | 1.727× | 29.2 % | 2.834× |
| R6 | 2.008 | **1.388×** | 17.5 % | **0.896×** |
| R7 | 2.296 | 0.875× | −7.2 % | 1.052× |
| **R0 ÷ R7** | | **6.509×** | | 6.364× |

Same total (6.51 against 6.36), same verdict — structural **7.3 %**,
representational **99.8 %**, unaccounted **−7.2 %** — and under the strict
reading structural is **37.5 %** against 36.6 % in process, which is closer
agreement than the individual rungs manage.

**Two rungs move materially between interpreters and neither may be quoted as a
constant.**  R1 (guard-call inlining) is worth 1.46× on 3.12 against 1.18× on
3.13, because 3.13's call sequence is cheaper.  **R6 changes sign**: adopting
the reference's `while` shape is 1.39× *faster* on 3.12 and 1.12× *slower* on
3.13.  The aggregate is stable; those two rungs are not.

---

## 4. The five candidates, each falsified or confirmed by its own measurement

`counts.py`, raw data `out/counts.json`; `fit.py`, raw data `out/fit.json`.

### 4.1 Candidate 1 — data representation: **FALSIFIED, in the strongest direction**

The brief's hypothesis was that the typed path materialises `frozenset` and
`tuple` where the reference uses native ints.  Measured per record on the
deployment distribution:

| | arm B (R0) | arm C (R7) | arm A |
|---|---|---|---|
| `BUILD_TUPLE` executed **per record** | **3.0** | **20.8** | — |
| `BUILD_TUPLE` executed **per scan step** | **0.00** | **1.00** | — |
| peak traced allocation | 480 B | **160 B** | 2,880 B |

**Arm C allocates a tuple on every scan step and arm B allocates none, and arm
C is 6.4× faster.**  Arm B's peak live footprint is 320 bytes larger than arm
C's — three hundred and twenty bytes, against a gap of 14.5 µs.  Allocation
cannot explain a gap in which the slower arm allocates seven times less.

This does not contradict §48: the `frozenset`/`tuple` materialisation §48
measured (48.4 MB → 0.121 MB) is what separates the *interpreter* from
*compiled code*, and it was already removed before arm B exists.  Nothing of it
survives into the residual.

### 4.2 Candidate 2 — boundary cost: **FALSIFIED**

Measured three ways, all agreeing.

**(a) Envelope minus bare.**  The dict-in/dict-out envelope costs **+0.029 µs
on arm B (+0.17 %)** and **+0.089 µs on arm C (+3.28 %)**.  It is charged to the
side under test and it makes the ratio *smaller*: 6.169× envelope-matched
against 6.364× bare.

**(b) and (c) The linear fit** `t(k) = a + b·k` against executed scan steps
`k = w + h`, eleven padded bins from `k = 2` to `k = 47`:

| rung | intercept a (µs) | slope b (µs/step) | body share at `k` = 18.8 |
|---|---|---|---|
| **R0 arm B** | **1.545** | **0.7902** | **90.6 %** |
| R3 | 0.874 | 0.3323 | 87.7 % |
| R5 | 0.386 | 0.0908 | 81.6 % |
| **R7 arm C** | **0.233** | **0.1123** | **90.1 %** |
| A compiled spec | 35.564 | −0.0198 | −1.1 % (constant work, as §51 found) |

R0's fitted slope 0.790 and intercept 1.545 reproduce §51's 0.787 and 1.887;
arm C's 0.1123 reproduces its 0.1095.

**Intercept ratio 6.64×, slope ratio 7.03×, overall time ratio 6.36×.**  The
non-loop prologue and epilogue are the same ~10 % of both arms and are
over-represented in the gap by exactly nothing.  The pre-registered
falsification threshold was 20 % in log terms; the boundary accounts for **0 %
of the gap** because it scales with it.

§48's *585 µs of validation around a 1.12 µs program* is a real cost and is not
in scope here: S2 is entered with `validate=False` on an already-native record,
so the external typed boundary is upstream of every arm in this document.

### 4.3 Candidate 3 — algorithmic difference beyond early exit: **FALSIFIED**

Every rung returns the same `w` and `h` on every gated record, and the executed
scan count confirms it directly at the opcode level — measured as the slope of
each opcode against `k`, deployment (`k` = 18.81) to worst (`k` = 47):

| opcode | R0 per record | **R0 per scan step** | R7 per record | **R7 per scan step** |
|---|---|---|---|---|
| `FOR_ITER` | 18.8 | **1.00** | 0.0 | 0.00 |
| `JUMP_BACKWARD` | 16.8 | **1.00** | 14.8 | **1.00** |
| **`BINARY_SUBSCR`** | 239.8 | **12.00** | 65.4 | **3.00** |
| `CALL` | 525.2 | **26.00** | 3.0 | **0.00** |
| `COMPARE_OP` | 750.0 | 38.00 | 37.6 | 2.00 |
| `POP_JUMP_IF_FALSE` | 376.5 | 19.00 | 37.6 | 2.00 |

**Exactly one loop iteration per scan step in both arms.**  Both are contiguous
run scans over the same number of steps; the resynthesized version does *not*
recompute per position.  §49's discovery result is intact and this candidate is
dead.

What differs is the cost *of* a step: **12 subscripts against 3**, and **26
Python calls against zero**.  Six of arm B's twelve are raster reads — three for
the candidate pixel and three for the anchor pixel, which it re-reads on every
step because nothing hoisted it — two are the learned truth table's own tuple
lookups, and the rest are `rec[0]`/`rec[1]` re-subscripts.  That is rung R3, and
it is a compiler pass, not an algorithm.

### 4.4 Candidate 4 — interpreter overhead surviving compilation: **CONFIRMED, and it is the largest single named cause**

Executed bytecodes per record on the deployment distribution, bucketed by code
object with `sys.monitoring` — the profile diff, at a resolution cProfile cannot
reach on a 3 µs function:

| | arm B (R0), 7,563.5 total | arm C (R7), 805.7 total | arm A, 15,145.7 total |
|---|---|---|---|
| `run` | 2,056.6 — 27.2 % | **790.7 — 98.1 %** | 5,760.7 — 38.0 % |
| **`_ck`** (integer range guard) | **1,981.5 — 26.2 %** | — | — |
| **`_ix`** (tuple index guard) | **1,664.4 — 22.0 %** | — | — |
| `_m0` (the learned module) | 1,561.6 — 20.6 % | — | 9,362.0 — 61.8 % |
| `_dzi` / `_dz` (division guards) | 284.4 — 3.8 % | — | — |
| **guard-helper frames, total** | **3,930.3 — 52.0 %** | **0** | 0 |
| **Python calls per record** | **378.5** | **2.0** | 65.0 |

**Fifty-two per cent of arm B's executed bytecodes are inside guard helper
frames that have no counterpart in the reference, and it makes 378 Python calls
per record where the reference makes two.**  The pre-registered falsification
threshold was 25 %.  This is the exact analogue of the `Value` construction §48
removed from internal edges: a per-node dynamic check that the frozen program
already proves unnecessary.

**And the fix is half-known already.**  `tcn/compile.py` emits its guards
**inline** — `if not 0 <= v13 <= 65535: _ovf(v13, 0, 65535)` and
`if not 0 <= v10 < 3072: raise IndexError(...)`, calling only on the failure
path, and using the statically known tuple width rather than `len(t)`.
`research/lazy-guard/ir.py` emits **calls**.  Rung R1's 1.184× (1.457× on
3.12) is therefore a defect of the *resynthesizer's* emitter that the
*specification* compiler has already solved, and closing it is a port, not a
design.  The remaining 2.834× (R5) is the checks themselves, which both
emitters pay.

### 4.5 Candidate 5 — constant factors in the emitted Python: **CONFIRMED, 33 % of the residual**

Rungs R1 + R2 + R3 — repeated `rec[0]` subscripts, the unhoisted anchor pixel,
the module call frame — are **1.184 × 1.022 × 1.844 = 2.232×**, i.e. **43.4 %**
of the residual in log terms.  R3 alone, which is textbook loop-invariant code
motion and common-subexpression elimination inside a loop body the emitter lays
out one SSA statement per node in topological order, is **1.844×**.

---

## 5. Three cost measures, and they disagree — again, and in a new direction

§51's caution, honoured: primitives, bytecodes and wall clock are reported
separately and no claim mixes them.

| measure | arm B | arm C | **B ÷ C** |
|---|---|---|---|
| **scan steps executed** (primitives, the algorithm) | 18.81 | 18.81 | **1.00×** |
| **raster subscripts per scan step** (primitives, the step) | 12.00 | 3.00 | **4.00×** |
| **executed CPython bytecodes per record** | 7,563.5 | 805.7 | **9.39×** |
| **executed bytecodes per scan step** | 381.5 | 40.5 | **9.42×** |
| **wall clock** | 17.181 µs | 2.700 µs | **6.36×** |
| ns per bytecode | **2.27** | **3.35** | 0.68× |

**The residual is 1.00× in scan steps, 9.4× in bytecodes and 6.4× in wall
clock.**  Arm C costs **48 % more per bytecode** than arm B, which is the same
inversion §48 measured for the language fixture (2.37 ns against 2.63 ns) and
§51 measured for arm C on the uniform distribution (3.213 ns against 2.518).
Quoting the bytecode ratio as the gap would overstate it by 48 %; quoting the
scan count would say there is no gap at all.

| rung | bytecodes/record | bytecodes/step | Python calls | µs | ns/bytecode | peak alloc |
|---|---|---|---|---|---|---|
| R0 | 7,563.5 | 381.5 | 378.5 | 17.181 | 2.27 | 480 B |
| R1 | 5,970.3 | 301.5 | 20.8 | 14.508 | 2.43 | 480 B |
| R2 | 6,036.5 | 304.7 | **2.0** | 14.196 | 2.35 | 736 B |
| R3 | 3,286.9 | 160.2 | 2.0 | 7.699 | 2.34 | 572 B |
| R4 | 2,904.1 | 142.9 | 2.0 | 7.217 | 2.49 | 572 B |
| R5 | **995.4** | **47.6** | 2.0 | **2.546** | 2.56 | 256 B |
| R6 | 832.4 | 40.5 | 2.0 | 2.841 | 3.41 | 192 B |
| R7 | 805.7 | 40.5 | 2.0 | 2.700 | 3.35 | 160 B |
| arm A | 15,145.7 | — | 65.0 | 36.749 | 2.43 | 2,880 B |

Note R2: inlining `_m0` **raises** the bytecode count (5,970 → 6,037) while
lowering the time (14.508 → 14.196 µs), because it trades 18.8 call/return
sequences for repeated `LOAD_FAST` traffic.  Fewer bytecodes is not less time,
in both directions, on the same table.

---

## 6. What would actually close it, named

In descending order of measured worth on the deployment distribution:

| # | change | worth | where it belongs |
|---|---|---|---|
| 1 | **Range- and bounds-check elimination.**  The loop index is `range(1, 32)`, the declared carrier is `(0, 65535)` and the tuple width is statically 3,072; interval analysis over the frozen program's declared types proves almost every emitted `_ck`/`_ix` unreachable. | **2.834×** | both emitters |
| 2 | **Loop-invariant code motion and CSE.**  The emitter lowers one SSA statement per node in topological order and hoists nothing, so `rec[0]`, `pos // 3`, `len(obs)` and the anchor pixel's three bytes are recomputed on every scan step. | **1.844×** | both emitters |
| 3 | **Inline the guards instead of calling them.**  `tcn/compile.py` already does; `research/lazy-guard/ir.py` does not. | **1.184×** (1.457× on 3.12) | the resynthesizer's emitter |
| 4 | **Inline module bodies at the call site** where the module is called once per loop iteration. | 1.022× | both emitters |
| 5 | **A short-circuiting conjunction primitive.**  `truth_7` is a 3-input table lookup; it cannot short-circuit, so the 2nd and 3rd bytes are loaded even after the 1st differs. | 1.067× — **and this one is the synthesizer's** | the operator set / the resynthesizer |

Changes 1–4 are compiler passes on a frozen, fully-typed SSA program with no
aliasing and no dynamic dispatch — the easiest setting such passes ever get.
Together they are **5.666×**, and applying them mechanically to arm B produced
a program **6 % faster than the hand-written reference**.

**Change 5 is the only one on the synthesizer's side of the line, and it is
worth 1.067×.**

---

## 7. Falsification, condition by condition

| pre-registered condition | outcome |
|---|---|
| **B ÷ C = 6.208× does not reproduce on this host** | **DID NOT FIRE.**  6.169× / 5.951× envelope-matched over two runs, 6.364× / 6.304× bare, 6.509× out of process on a different interpreter. |
| **structural rows dominate** ⇒ the resynthesis programme has a ceiling near 2× | **DID NOT FIRE as a majority**, and **fired as a bound**.  Structural is 3.5 % (36.6 % under the strict reading); but because the structural part is the *only* part the synthesizer owns, the ceiling result stands anyway and is stated in §0: **the synthesizer's remaining headroom is 1.067×**. |
| **representational rows dominate** ⇒ ordinary engineering, name the change | **FIRED.**  93.7 % (60.7 % strict).  Named in §6, five changes, each with its measured worth. |
| **the named rows do not sum to 6.2×** | They sum by construction; the unaccounted rung is reported at its own value: **+2.8 %** on deployment, +0.5 % uniform, +3.8 % worst, **−7.2 %** out of process.  **The unexplained fraction is under 8 % in log terms and its sign is not stable.** |
| **a rung fails the equivalence gate** | **FIRED for R5a**, pre-registered as the rung at risk.  Reported in §2.1, not timed, not attributed. |
| **noise exceeds an effect** | Did not fire: no rung CI contains 1.0 on any distribution, and the instrument's own noise floor is 0.5 % (the same function timed twice as `bare:B`/`R0` and `bare:C`/`R7` differs by 0.45 % and 0.48 %). |

---

## 8. Amendments to the pre-registration

To the standard `research/lazy-latency/RESULTS.md` §9 set: each with the number
it replaced, and with its direction relative to this track's own hypothesis.

**AMENDMENT 1 — the structural/representational decision rule is not exhaustive,
and both readings are reported rather than the convenient one.**
PREREGISTRATION §2 defines *structural* as "removing it requires the synthesizer
to emit a program that performs fewer elementary operations" and
*representational* as "achievable by a semantics-preserving source-to-source
transform that leaves the executed elementary-operation count unchanged".
**Rung R3 (loop-invariant code motion) satisfies neither**: it needs no
synthesizer change, but it *does* reduce the executed operation count, from 12
subscripts per scan step to 3.  The primary reading adopted is the first clause
— *does closing it require a different synthesized program, or does a mechanical
compiler pass suffice?* — under which R3 is representational.  **Both readings
are reported on every table**, and the strict reading is the one that moves
against this track: it raises structural from 3.5 % to **36.6 %**.  The verdict
is unchanged only because representational still dominates under it (60.7 %).

**AMENDMENT 2 — rung R6 drops a branch, and it is recorded rather than
elided.**  R6 replaces the `for … else: f8 = 1` form with arm C's `while` form,
which has no `else` arm.  `research/lazy-guard` §6.5 and FINDINGS §51 both
record that `else` as unreachable on the declared domain (`Lit(1)`, never
taken); the exhaustive gate over 2,883 + 54 + 1 records is what proves it here.
**No measurement in this document exercises that branch, exactly as none in §51
did**, and R6 would differ from R0 at `x = 0`, which is not an interior
position.

**AMENDMENT 3 — R5 also removes the dead temporaries its own guards were
keeping alive.**  Deleting `if not 0 <= t19 <= 65535` leaves `t19 = v15 + 1`
with a single use, so it is copy-propagated into `obs[v15 + 1]`.  This is a
consequence of guard elimination rather than a separate pass, but it is part of
R5's measured 2.834× and is not separately attributed.  It moves *for* the
hypothesis, and is recorded for that reason.

**AMENDMENT 4 — no `k = 36` bin.**  `fit.py`'s pre-registered bin list includes
`w + h = 36`; the held-out set contains no record with that value, so eleven
bins were fitted rather than twelve.  Bins at `k = 42` and `k = 47` hold one
record each and are padded to 1,024 calls exactly as
`research/lazy-latency/crossover.py`'s AMENDMENT 3 requires.

**AMENDMENT 5 — the interpreter oracle was strengthened, not weakened.**
PREREGISTRATION §4 gates the rungs against arm B and inherits §51's interpreter
link.  `interp.py` re-runs the link independently on **41** records (39 strata
plus corners plus worst) against §51's 32, and extends it to all eight rungs.
Cost: 12.5 s, because a stratified sample of 41 is affordable where 2,883 is not.

---

## 9. What this does NOT establish

At the same volume as the results.

1. **It is one subroutine.**  Everything here is the shipped visual parser's S2
   `rect_scaffold` on a 32×32 flat-fill screen.  The *shape* of the answer —
   guards, hoisting, call frames — is generic, but the 5.666× is not a constant
   of the repository.
2. **It does not implement any of the five changes.**  §6 names them and prices
   them by measuring a hand-applied transform of one program.  A general pass in
   `tcn/compile.py` or `research/lazy-guard/ir.py` must be sound on programs
   this track never saw, and R5a is the standing proof that a transform which
   looks obviously safe can be wrong on 8.4 % of the domain.
3. **Guard elimination changes the error contract in general.**  R5 is gated
   bit-identical *on this domain*; a general range-elimination pass must
   *prove* the check unreachable, not observe it.  §48's `tests/test_compile.py`
   asserts the exact exception type at the exact edge for eleven error classes,
   and any such pass is answerable to it.
4. **R6's sign is interpreter-dependent** (−10.8 % on 3.13, +17.5 % on 3.12) and
   must not be quoted as a constant.  Neither must R1's size (1.18× against
   1.46×).
5. **Nothing here touches quality.**  Every arm computes the identical function;
   a fast wrong answer is measured nowhere.
6. **The worst case is not rescued.**  §51's result that arm B is 1.09× *slower*
   than the specification at `w + h` = 47 reproduces here at 0.919×, and no rung
   in this ladder changes that: R0's worst case is 37.7 µs against arm A's 34.7.
   The compiler passes named in §6 would fix it — R5 at worst is 4.52 µs — but
   until they exist the crossover §51 measured at `w + h` ≈ 42 stands.

**Difficulty achieved, stated rather than requested.**  This is an attribution
track.  The hard part was making the attribution *un-fudgeable*: a cumulative
chain whose log-deltas sum to the whole gap by construction, so that the
unaccounted term has to appear as a row rather than as a rounding.  The trap
avoided was assuming, from the shape of the generated source, that the clamp was
dead code — it is load-bearing on 243 of 2,883 records, and only the exhaustive
gate found it.  The trap not fully avoided is Amendment 1: the pre-registered
decision rule did not anticipate a transform that is both a compiler pass and an
operation-count reduction, and both readings are reported because choosing one
silently would have been the more comfortable option.

---

## 10. Reproduction

```bash
.venv/bin/python research/residual-gap/rungs.py       # build the ladder, ~5 s
.venv/bin/python research/residual-gap/gate.py        # PREREG section 4, ~20 s
.venv/bin/python research/residual-gap/interp.py      # the interpreter oracle, ~13 s
taskset -c 19 .venv/bin/python research/residual-gap/timing.py run1 21   # ~50 s
taskset -c 19 .venv/bin/python research/residual-gap/timing.py run2 21   # ~50 s
taskset -c 19 .venv/bin/python research/residual-gap/fit.py   # boundary/body, ~3 min
.venv/bin/python research/residual-gap/counts.py      # bytecodes, allocs, ~4 min
.venv/bin/python research/residual-gap/outproc.py     # 3.12.3 replication, ~10 s
.venv/bin/python research/residual-gap/report.py      # every number above
```

| file | what |
|---|---|
| `PREREGISTRATION.md` | the protocol, committed at `6d875b8` before any arm |
| `rungs.py` | the eight rungs; R1/R2 mechanical, R3–R6 written out, R7 arm C |
| `gate.py` | PREREGISTRATION §4 — bit-identical output, all rungs, all records |
| `interp.py` | the typed-interpreter oracle, 41 stratified records |
| `timing.py` | PREREGISTRATION §5, on `research/lazy-latency/latency.py`'s protocol |
| `fit.py` | candidate 2 — intercept and slope against `w + h` |
| `counts.py` | candidates 1, 3, 4, 5 — bytecodes per code object and per opcode, calls, allocation |
| `outproc.py` | the 3.12.3 replication |
| `report.py` | prints every number above from `out/*.json` |
| `out/R*.py` | every rung's source, committed |

`research/compiled-runtime/harness.py` is imported for its gc discipline,
`research/lazy-guard/cost.py` for its `sys.monitoring` bytecode meter (extended
with a bucket key, not re-implemented), and `research/lazy-latency/arms.py`,
`latency.py` and `crossover.py` for the arms, the timing protocol and the bin
padding.  **No third harness was built.**

### Repository state

* **`.venv/bin/python -m pytest -q` → 336 passed, 1 failed of 337.**  The
  failure is `tests/test_panel_interface.py::test_panel_episode_replays_and_restores`,
  the known worktree-only replay divergence (FINDINGS §41,
  `research/MERGE-QUEUE.md`).  **Verified by removing this directory**: with
  `research/residual-gap/` moved out of the tree the same single test fails.
  The thirteen-failure mode the brief warned about is the *missing*
  `generators/computer/engine/node_modules` symlink, not its presence; with the
  symlink restored the count is 336/1.  This track adds no tests and imports
  nothing from any tested module.
* **Shipped fixture reproduces exactly**: `.venv/bin/python -m tcn train
  --episodes 160` gives `initial_prediction_loss 0.248835613951087`,
  `final_prediction_loss 0.0022308224288281053`, `fully_frozen true`,
  `evaluation_mean_return 4.0`, `frozen_evaluation_mean_return 4.0` —
  **0.248836 → 0.002231 at 4/4 frozen.**
* Generated inputs (`out/records_deploy.json`) and `__pycache__` are gitignored;
  every rung source and every report JSON is committed.
* `research/second-family/` was not touched.
