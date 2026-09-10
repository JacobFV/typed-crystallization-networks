# Lazy latency: does the expected-work gain survive to wall clock?

Branch `worktree-agent-a9beb89f33c873803`, from `main` at `e79a660`.  Nothing
under `tcn/`, `generators/`, `research/lazy-guard/` or `research/semantic-library/`
is modified.  This directory is a pure addition.

This is the **measured-latency arm** of `research/algorithm-resynthesis/DESIGN.md`
§11 — the arm FINDINGS §49 named as its open question:

> whether the expected-work gain survives translation to wall clock given the
> bytecode regression.  That needs the measured-latency arm of DESIGN §11, which
> this track did not run.

`PREREGISTRATION.md` was written and **committed (`95dec33`) before any timing
arm ran**.  Every number below is printed by `report.py` from `out/*.json` and
can be re-checked against those files without re-running anything.

---

## 0. Headline

**On the deployment distribution the resynthesized algorithm is faster in wall
clock — 2.079× (95 % CI [2.055, 2.095]) — and the pre-registered falsification
did not fire.**  That was the outcome the brief and §49's bytecode numbers both
predicted *against*.

Three facts belong with it and none may be dropped:

| | stage 3, the shipped visual parser's S2 subroutine |
|---|---|
| **deployment distribution** (54 corner records, 1.87 %) | **A ÷ B = 2.079× — B faster** |
| uniform held-out (2,883 interior records) | A ÷ B = 3.596× — B faster |
| **worst case** (`w + h = 47` loop iterations) | **A ÷ B = 0.905× — B is 1.105× slower** |
| **the hand-written reference on the same distribution** | **A ÷ C = 12.908×** |

So on the deployed distribution **laziness is worth 2.08× of a 12.91× gap.  The
remaining 6.21× is not laziness**, and no amount of early exit reaches it.  That
is the finding that redirects the programme, and it is a negative wearing a
positive's clothes.

The wall-clock penalty in the worst case is **real but smaller than the bytecode
regression predicted**: 1.105× where the bytecode ratio is 1.20×, 1.250× where
it is 1.50×, 1.361× where it is 1.58×.  Branch tests and jumps are cheap
bytecodes; what early exit removes are expensive ones.

**Crossover.**  Arm A is constant-work and arm B is linear in loop iterations, so
laziness stops paying at a measurable point: **`w + h` ≈ 42 of a possible 62**
on the parser subroutine, and a predicate **hit rate of 0.798** on the guard
miniature.  Deployment sits far below both — mean `w + h` = 18.8, hit rate
0.021 — which is *why* B wins there.

---

## 1. The gate: equivalence before any timing

`equivalence.py`, raw data `out/equivalence.json`.  PREREGISTRATION §4.  **No
timing number in this document was recorded before all of this passed.**

| stage | interpreter == A | A == B (== C), bit-identical |
|---|---|---|
| 1 `sparse_guard` | 1,024 / 1,024 | **65,536** uniform + **65,536** deploy + worst |
| 2 `find_first` | 1,024 / 1,024 | **65,536** uniform + worst |
| 3 shipped S2 | **32 / 32** stratified | **2,883** uniform + **54** deploy + worst |

`all_gates_passed: true`.  A SHA-256 digest over each arm's canonical outputs is
recorded per distribution and the digests are equal, so the claim is checkable
without re-running: stage 3's held-out digest is `f9b53b8292bc7815…` for all
three arms, its corner digest `b4148088c7ddabd8…`.

Stage 3's interpreter link is a **stratified 32-record sample, not the full
2,883**, and is reported as weaker.  The reason is measured, not assumed: the
typed interpreter costs **0.454 s per record** on the 3,072-byte carrier, i.e.
**22 minutes** for the held-out set.  This is exactly the oracle chain
`research/lazy-guard` §6.1 declared and FINDINGS §48 certified, and the sample
covers every achieved-extent stratum including every corner stratum.

**Arm C is bit-identical to the shipped parser's S2 module on all 2,883 held-out
records and all 54 corner records.**  That is worth stating on its own: the
hand-written reference in `research/compiled-runtime/fixtures.py` and the frozen
585-node learned module compute the same function, position by position.

---

## 2. The arms and the calling convention

| arm | what | entry |
|---|---|---|
| **A** | the compiled specification | `tcn.compile.compile_program(spec, registry).module().run({port: rec}, validate=False)` |
| **B** | the resynthesized standalone algorithm | `research/lazy-guard/out/stage{1,2,3}_algorithm.py :: run` — imported verbatim, **not one character edited** |
| **C** | the hand-written reference | the extent loops of `research/compiled-runtime/fixtures.py :: visual().reference`, restricted to one interior position (stage 3 only) |

`out/*_deploy.py` is gitignored — it is only a subprocess driver; the code it
imports, `out/*_algorithm.py`, **is** committed and is arm B.

`tcn/compile.py` emits exactly one entry, `run(inputs, state=None, validate=True)`,
taking a dict and returning `({outputs}, {state})`.  B and C are bare functions.
The primary comparison is therefore **envelope-matched** — B and C wrapped to the
same dict shapes — which charges the envelope **to B and C**, the side under
test.  Bare figures are reported alongside: the envelope measures at +0.03 to
+0.16 µs and on stage 3's worst-case row it comes out at −0.07 µs, i.e. inside
the noise.  It changes no verdict on any row.

There is **no arm C for stages 1 and 2**: `fixtures.py` contains no hand-written
reference for either miniature, and writing one here would be authoring the
comparison's own answer.

---

## 3. The three-distribution × three-arm latency table

`latency.py`, raw data `out/latency.json` (= `out/latency_run1.json`).
Python 3.13.15, pinned to CPU 19 under `taskset`, gc collected then disabled
around each timed block, one discarded warm-up sweep per arm per distribution,
**21 repeats with the arms interleaved round-robin inside each repeat**, medians
with nonparametric 95 % CIs from order statistics and bootstrap CIs (10,000
resamples) on every ratio.  `/proc/loadavg` **4.61 → 5.85** on a 20-core host
shared with another agent.

Every figure is **microseconds per record, batch-one, warm.**

### 3.1 Stage 3 — the shipped visual parser's S2 subroutine

585 frozen nodes, digest `c79e712699a5aa1716f3f65c`; the module FINDINGS §48
blamed for 11.1× of the parse's 27.6× residual.

| distribution | records | **A** compiled spec | **B** resynthesized | **C** hand-written | **A ÷ B** | A ÷ C |
|---|---|---|---|---|---|---|
| **deployment (corner)** | **54** | 39.043 [38.905, 39.154] | **18.782** [18.658, 18.970] | 3.025 [2.989, 3.107] | **2.079× [2.055, 2.095]** | 12.908× [12.564, 13.096] |
| uniform held out | 2,883 | 38.031 [37.968, 38.088] | 10.576 [10.521, 10.783] | 1.532 [1.525, 1.539] | 3.596× [3.526, 3.615] | 24.820× [24.679, 24.937] |
| **worst case** (`w+h` = 47) | 1 | 37.025 [37.007, 37.052] | **40.892** [40.877, 40.921] | 5.749 [5.744, 5.758] | **0.905× [0.905, 0.906]** | 6.440× [6.430, 6.448] |

IQR: A 0.317 / 0.131 / 0.063 µs; B 0.578 / 0.362 / 0.059 µs; C 0.140 / 0.021 /
0.019 µs.  **No CI contains 1.0**, so the effect-versus-noise rule does not fire
on any row.

### 3.2 Stage 1 — the guard miniature

| distribution | inputs | **A** | **B** | **A ÷ B** |
|---|---|---|---|---|
| deployment analogue, hit rate 1/48 | 65,536 | 3.171 [3.166, 3.175] | 0.266 [0.265, 0.267] | **11.916× [11.893, 11.954]** |
| uniform, the **exhausted** 16⁴ domain | 65,536 | 3.137 [3.133, 3.146] | 0.423 [0.421, 0.424] | 7.422× [7.385, 7.454] |
| **worst case** (all four positions hit) | 1 | 3.194 [3.186, 3.210] | **3.994** [3.987, 4.002] | **0.800× [0.797, 0.803]** — B 1.250× slower |

### 3.3 Stage 2 — the find-first miniature

| distribution | inputs | **A** | **B** | **A ÷ B** |
|---|---|---|---|---|
| uniform, the **exhausted** 4⁸ domain | 65,536 | 2.315 [2.308, 2.318] | 0.859 [0.854, 0.862] | 2.696× [2.686, 2.708] |
| **worst case** (no position satisfies) | 1 | 2.284 [2.277, 2.287] | **3.109** [3.107, 3.115] | **0.735× [0.732, 0.736]** — B 1.361× slower |

Stage 2 has no deployment distribution: nothing ships it.  Reported as absent
rather than substituted for.

---

## 4. Modelled work versus measured time — the actual question

§49's worry was that the modelled gain would not survive.  It survives, and it
**attenuates in a specific, explicable way**.

| | §49 modelled, expected primitives | this track, measured wall clock |
|---|---|---|
| stage 1 uniform | 5.09× | **7.42×** |
| stage 2 uniform | 4.02× | **2.70×** |
| stage 3 uniform held out | 4.42× | **3.60×** |
| **stage 3 corner (deployment)** | **2.48×** | **2.08×** |
| stage 1 worst | 1.00× | **0.80×** |
| stage 2 worst | 1.00× | **0.74×** |
| stage 3 worst | 1.015× | **0.905×** |

The expected-case rows land within ~1.5× of the modelled ratio in both
directions; the worst-case rows are all **below 1.0**, which the primitive model
predicted as exactly 1.0.  That is the branch-overhead cost the primitive count
cannot see, and it is charged in full.

### 4.1 The bytecode regression overstates the wall-clock regression

`recheck_bytecodes.py`, raw data `out/bytecode_recheck.json`.  §49's bytecode
figures **reproduce**, counted over all 65,536 inputs rather than probes:

| | §49 recorded | reproduced here | ratio |
|---|---|---|---|
| stage 1 spec, expected / worst | 1003.25 / 1007 | **1000.25 / 1004** | |
| stage 1 resynthesized, expected / worst | 157.00 / 1507 | **155.00 / 1505** | worst **0.667×** vs §49's 0.67× ✅ |
| stage 2 spec, expected / worst | 821.00 / 821 | **821.00 / 825** | |

(arm A is counted through its dict envelope, which is the only entry `tcn/compile.py`
emits; arm B is counted bare, as §49 counted it.)
| stage 2 resynthesized, expected / worst | 338.76 / 1303 | **336.76 / 1301** | worst **0.634×** vs §49's 0.63× ✅ |

The two-to-three-count offsets are the wrapper framing of the counter and are
constant.  One small correction to §49: exhausting the domain gives the stage-2
specification a worst case of **825** bytecodes, not 821 — §49 read 821 from
probe inputs and flagged the ±1 itself; the true maximum is 4 higher.

Now the point.  Put the reproduced bytecode ratios beside the measured time:

| | bytecode penalty at worst | **measured wall-clock penalty at worst** | ns per bytecode, A → B |
|---|---|---|---|
| stage 1 | 1.499× | **1.250×** (bare B: 1.241×) | 3.181 → 2.634, **17.2 % cheaper** |
| stage 2 | 1.577× | **1.361×** (bare B: 1.350×) | 2.768 → 2.369, **14.4 % cheaper** |
| stage 3 | 1.198× | **1.105×** (bare B: 1.106×) | 2.422 → 2.237, **7.7 % cheaper** |

**Arm B's bytecodes are systematically cheaper than arm A's** — 7.7 % to 17.2 % —
because the ones laziness adds are branch tests and jumps while the ones it
removes are calls, subscripts and bound checks.  So counting bytecodes, as §48's
decomposition of the visual gap did, **overstates the cost of early exit**.  That
is a reusable correction to how this repository has been reasoning about cost.

The inversion runs the other way for the hand-written reference: arm C on the
uniform distribution costs **3.213 ns per bytecode** against arm A's 2.518, i.e. it
does more work per instruction — the same shape §48 measured for the language
fixture (2.37 ns against 2.63 ns).

---

## 5. Crossover — where laziness stops paying

`crossover.py`, raw data `out/crossover.json`.  15 repeats per bin, every bin
padded to 2,048 calls, arms interleaved.

### 5.1 Stage 3, by arm B's executed loop iterations `w + h`

Arm A is a fixed 31-term formulation per direction — constant work.  Arm B runs
one loop per direction, so it is linear in `w + h`, ceiling 62.  Measured fit:

```
B(k) = 1.887 us + 0.787 us per iteration       A = 35.84 us, flat
C(k) = 0.330 us + 0.1095 us per iteration
```

| `w+h` | 2 | 10 | 20 | 31 | 40 | **42** | 43 | 46 | 47 |
|---|---|---|---|---|---|---|---|---|---|
| A µs | 36.51 | 36.38 | 36.01 | 35.39 | 34.71 | 34.72 | 34.69 | 34.78 | 34.71 |
| B µs | 2.86 | 9.70 | 17.93 | 26.65 | 32.21 | **34.49** | 35.20 | 37.50 | 38.28 |
| A÷B | 12.77 | 3.75 | 2.01 | 1.33 | 1.08 | **1.007** | **0.986** | 0.928 | 0.907 |

**k\* = 42.3 iterations** by interpolation, **43.2** by the linear fit; the first
measured bin where B is slower is **43**.  The ceiling is 62, so **laziness pays
across roughly the lower two-thirds of the achievable range** on this subroutine.

Arm C's fitted crossover is at **324 iterations** — five times the ceiling — so
the hand-written reference never loses.  The specification's constant-work form
is not competitive with it at any extent.

**Where deployment sits.**  Mean `w + h` on the corner distribution is **18.81**
(uniform held out: 10.26).  Only **3 of 54 corner records — 5.6 % — are at or
above k\***.  That is precisely why B wins on deployment while losing at worst.

A caveat from the raw data: arm A drifts from 36.51 µs to 34.71 µs across the
sweep (−4.9 %), bins being timed in increasing order over about two minutes on a
shared host.  k* is read from locally-adjacent bins, so the drift moves it by
well under one iteration, but it is drift and not signal and is recorded as such.

### 5.2 Stage 1, by predicate hit rate

| hit rate (achieved) | 0.0000 | 0.0208 | 0.0625 | 0.1239 | 0.2489 | 0.4998 | 0.7510 | **0.8743** | 1.0000 |
|---|---|---|---|---|---|---|---|---|---|
| A µs | 3.059 | 3.062 | 3.075 | 3.081 | 3.096 | 3.101 | 3.090 | 3.092 | 3.083 |
| B µs | 0.181 | 0.267 | 0.439 | 0.679 | 1.137 | 2.052 | 2.956 | **3.394** | 3.863 |
| A÷B | 16.94 | 11.45 | 7.01 | 4.54 | 2.72 | 1.51 | 1.045 | **0.911** | 0.798 |

**p\* = 0.798.**  Laziness pays until the predicate hits about four times in
five.

For scale: the shipped parse's corner rate is **20/961 = 0.0208**, `lazy-guard`'s
corner rate is **54/2883 = 0.0187**, and the miniature's own uniform rate is
**1/16 = 0.0625**.  All are more than an order of magnitude below p*.  **The
deployed sparsity is nowhere near the crossover**, which is the substantive
reason the wall-clock win is safe here and the reason a shallower predicate would
be the thing to worry about, not a denser one.

---

## 6. Deployment: bytes, cold start, peak RSS

`deployment.py`, raw data `out/deployment.json`.  Each arm runs in a fresh
`/usr/bin/python3 -I` (**3.12.3**, not the venv) with `PATH` only; 11 runs.
`leaked_modules` is `[]` for every arm: no `tcn`, no `torch`, no `numpy`.

| stage | arm | bytes on disk | cold start (ms) | peak RSS (MB) |
|---|---|---|---|---|
| 3 | A compiled spec | **132,257** | 0.715 | 12.12 |
| 3 | **B resynthesized** | **3,291** | **0.185** | 11.73 |
| 3 | C hand-written | 510 | 0.136 | 11.71 |
| 1 | A | 10,830 | 0.190 | 23.08 |
| 1 | B | 2,147 | 0.142 | 23.05 |
| 2 | A | 8,404 | 0.187 | 27.12 |
| 2 | B | 1,165 | 0.154 | 27.09 |

Bare-interpreter baseline is **9.31 MB**, so on stage 3 the arms themselves cost
2.81 / 2.42 / 2.40 MB above an empty process — the fixture dominates and the
arms differ by ~0.4 MB.  Cold start is import plus compile plus one call; the
fixture load is excluded because it is identical for every arm and dominates it.

**On the shipped subroutine B is 40.2× smaller on disk and starts 3.9× faster**
than the compiled specification.  That is a second, independent benefit of
resynthesis that the expected-work model does not describe at all.

**A measurement trap, recorded because it silently produced a wrong number.**
`resource.getrusage(RUSAGE_SELF).ru_maxrss` in a child is **inherited across
`fork`** from the launching process.  Measured from this track's own parent —
which holds torch and the visual fixture — every arm reported an identical
**375.58 MB**.  The same child launched from a shell reports 11.7 MB.  The table
above uses `/proc/self/status: VmHWM`, the high-water mark of the process's own
`mm`, which `execve` replaces.  Both are kept in the JSON
(`peak_rss_rusage_mb_median` = 356.27 for all three stage-3 arms), because a
number that is identical across arms is the signature of this bug and not of a
tie.

### 6.1 An out-of-process replication of the headline

The same children, sweeping the same distributions in **Python 3.12.3** with no
gc discipline, no CPU pinning, no venv and a different interpreter build — a
completely separate instrument from §3:

| | in-process (§3) | out-of-process |
|---|---|---|
| **stage 3 deployment, A ÷ B** | **2.079×** | **1.853×** |
| stage 3 uniform, A ÷ B | 3.596× | 3.219× |
| stage 3 deployment, A ÷ C | 12.908× | 11.974× |
| stage 1 deployment, A ÷ B | 11.916× | 11.970× |
| stage 1 uniform, A ÷ B | 7.422× | 6.867× |
| stage 2 uniform, A ÷ B | 2.696× | 2.257× |

Same sign, same order, 3–16 % apart.  **The headline is not an artefact of the
in-process harness.**

---

## 7. Reproducibility and noise

The protocol was run **twice** end-to-end.

| | run 1 | run 2 | agreement |
|---|---|---|---|
| load at start → end | 4.61 → 5.85 | 3.21 → **30.21** | |
| stage 3 deployment A ÷ B | 2.079 [2.055, 2.095] | 2.090 [2.046, 2.110] | **+0.56 %** |
| stage 3 uniform A ÷ B | 3.596 [3.526, 3.615] | 3.491 [**2.257, 4.814**] | −2.92 % |
| stage 3 worst A ÷ B | 0.905 [0.905, 0.906] | 0.865 [**0.680, 0.980**] | −4.49 % |
| stage 1 deployment A ÷ B | 11.916 [11.893, 11.954] | 11.904 [11.892, 11.969] | −0.10 % |
| stage 1 worst A ÷ B | 0.800 [0.797, 0.803] | 0.802 [0.798, 0.808] | +0.33 % |

**Run 1 is quoted** because its measured load stayed at 4.6–5.9; run 2 met a
load spike to **30.21** on the shared host midway through.  Every point estimate
agrees within 4.5 %, and — the part that matters — **run 2's intervals widened
by an order of magnitude exactly where the noise arrived**, on stage 3's uniform
and worst rows.  The pre-registered effect-versus-noise rule works: had the
verdict rested on those rows the honest report would have been an interval, not
a point.  It does not; the deployment row is stable in both runs to 0.6 %.

Both runs are kept: `out/latency_run1.json`, `out/latency_run2.json`.

A third run was not taken.  The host's load rose to **43.0** while this document
was being written and stayed there; a third run under that load would have been
strictly worse than run 2 and would have added nothing that run 2's widened
intervals do not already show.  Recorded rather than quietly omitted.

---

## 8. Falsification, condition by condition

| pre-registered condition | outcome |
|---|---|
| **B is not faster than A on the deployment distribution** ⇒ the gain does not survive; §49 stands without a performance claim | **DID NOT FIRE.**  B is **2.079×** faster, CI [2.055, 2.095], excluding 1.0, replicated at 1.853× out of process and at +0.56 % in run 2. |
| B faster on uniform but not on the corner distribution ⇒ the win is off-deployment only | **DID NOT FIRE**, but the caution it encodes is live: the win is **2.08× on deployment against 3.60× on uniform**.  Quoting 3.60× as the deployment number would overstate it by 73 %. |
| B slower in the worst case by roughly the bytecode ratio ⇒ branch overhead dominates | **PARTIALLY FIRED.**  B is slower in **every** worst case — 1.105×, 1.250×, 1.361× — but **by less than the bytecode ratio** (1.198×, 1.499×, 1.578×).  Branch overhead is real and sub-proportional. |
| noise exceeds the effect ⇒ report the interval | **DID NOT FIRE in run 1**; no CI contains 1.0.  It **did** fire in run 2 under a load spike, on stage 3's uniform and worst rows, and is reported as an interval there. |

**Nothing was tuned until B won.**  B won on the first timing run of the
deployment distribution, before any amendment below was made, and the amendments
all move numbers **against** B or leave them unchanged.

---

## 9. Amendments to the pre-registration

Recorded as PREREGISTRATION §7 requires, with the number each replaced.

**AMENDMENT 1 — the miniatures' uniform sweep is the full domain, not a
stride-16 subsample.**  `full[::16]` over `itertools.product` order **pins the
last input digit at 0**: measured per-position hit rates on stage 1 come out
`1/16, 1/16, 1/16, 0`, so position 3 can never satisfy the predicate.  Under it,
stage 1's expected bytecodes for arm B read **93** against §49's exact **157**.
The full 65,536-input domain costs ~0.2 s per sweep, so the concession bought
nothing.  Replaced A ÷ B = 8.591× with **7.422×** on stage 1 uniform and 2.697×
with 2.696× on stage 2 — i.e. it made B **worse**.

**AMENDMENT 2 — the crossover parameter is `w + h`, not `max(w, h)`.**  Arm B
runs one loop per direction, so its cost is linear in `w + h`; a record with
`max(w, h) = 30` may execute anywhere from 31 to 61 iterations.  Binning by
`max(w, h)` smears B's cost within a bin and **finds no crossover at all** (the
`max(w, h)` sweep is retained in `out/crossover.json` as
`stage3_by_max_extent`, and its highest bin, 31, still has B ahead at 1.29×).
`w + h` is the same quantity the pre-registered worst case is chosen by.  This
amendment is what *reveals* B losing, so it too runs against B.

The same correction applies to the **worst-case record**: the first
implementation picked it by `max(w, h)` = 31, which the pre-registration's own
words ("maximizing arm B's executed loop iterations") do not permit.  Corrected
to `w + h` = 47.  It replaced stage 3 worst A ÷ B = **1.282×** (B faster) with
**0.905×** (B slower) — the single largest correction in this document, and it
reverses a sign in B's favour.

**AMENDMENT 3 — every crossover bin is timed, padded to 2,048 calls.**  The
pre-registered `MIN_BIN` of 20 discards every bin above `w + h = 32`, which is
exactly where the crossover lies (bins of one and two records).  Padding cycles a
bin's records to a common sweep length, is applied identically to all three arms,
and is what the pre-registered `D_worst` arm already does.  Without it k* is
**unmeasurable**; with it, k* = 42.3.

**AMENDMENT 4 — peak RSS is `VmHWM`, not `ru_maxrss`.**  §6 above; `ru_maxrss`
reported the launching process's 375.58 MB for every arm.

**AMENDMENT 5 — bytecode sampling is a seeded uniform draw, not a stride,** for
the same reason as amendment 1.

---

## 10. What this establishes, and what it does not

**It establishes** that the expected-work gain of `research/lazy-guard` **does**
translate into wall clock on the distribution the shipped parse actually feeds:
2.079×, on a certified bit-identical algorithm, replicated out of process, with
the worst-case penalty measured and charged.  §49's architectural result may now
be cited **with** a performance claim, provided the claim is the corner number
and not the uniform one.

**It establishes a bound that matters more.**  On that same distribution the
hand-written reference is **12.91×** faster than the compiled specification, and
laziness recovers **2.08×** of it.  **The residual — about 6.2× — is not
addressable by early exit.**  §48 attributed 11.1× of the parse's 27.6× gap to
S2's fixed-depth formulation and said the lever was "a search that finds shorter
programs, not a faster backend"; this track measures early exit at **28.6 % of that lever in log
terms** (log 2.079 / log 12.908) and locates the rest elsewhere.  Any plan that
proposes to close the visual gap by making the resynthesizer lazier is now
bounded above by these numbers.

**It does not establish** that laziness is free.  It costs 1.105× in the worst
case on the shipped subroutine, 1.250× and 1.361× on the miniatures, and the
crossover is a real place: `w + h` ≈ 42 of 62, hit rate ≈ 0.80.  A subroutine
whose loops run long, or a predicate that usually holds, would lose.  Deployment
happens to sit at `w + h` = 18.8 and hit rate 0.021.

**It does not establish** anything about stage 3's `else` branch.
`research/lazy-guard` §6.5 recorded that both loops' `else` came out as
`Lit(1)`, unreachable on the declared domain.  It is unreachable on every record
timed here too, so **no arm in this document exercises it**, and no measurement
here can notice if it is wrong.

**Difficulty achieved, stated rather than requested.**  This is a measurement
track, not a discovery one: the hard parts were the equivalence gate, choosing
the work parameter correctly, and not being fooled by three instrumentation
traps (the stride bias, `ru_maxrss` inheritance, and a `MIN_BIN` that hides the
answer).  Two of the three would each have produced a confidently wrong number
in the same direction as the hypothesis, which is the direction that should have
been hardest to believe.

---

## 11. Reproduction

```bash
.venv/bin/python research/lazy-latency/equivalence.py        # the gate, ~2 min
taskset -c 19 .venv/bin/python research/lazy-latency/latency.py 1 2 3   # ~6 min
taskset -c 19 .venv/bin/python research/lazy-latency/crossover.py       # ~5 min
taskset -c 19 .venv/bin/python research/lazy-latency/deployment.py      # ~2 min
.venv/bin/python research/lazy-latency/recheck_bytecodes.py  # ~8 min
.venv/bin/python research/lazy-latency/report.py             # every number above
```

| file | what |
|---|---|
| `PREREGISTRATION.md` | the protocol, committed at `95dec33` before any timing |
| `arms.py` | the three arms, the three distributions, the hand-written reference |
| `equivalence.py` | PREREGISTRATION §4, the gate |
| `latency.py` | PREREGISTRATION §3, §5 — the timing protocol and its statistics |
| `crossover.py` | PREREGISTRATION §6 — k\* and p\* |
| `deployment.py` | bytes, cold start, peak RSS, and the out-of-process replication |
| `recheck_bytecodes.py` | does FINDINGS §49's bytecode table reproduce? |
| `report.py` | prints every number above from `out/*.json` |

`research/compiled-runtime/harness.py` is imported for its gc discipline and
`rss_mb`, and `research/lazy-guard/cost.py` for its `sys.monitoring` bytecode
meter.  Neither is re-implemented.

### Repository state

* **`.venv/bin/python -m pytest -q` → 336 passed, 1 failed of 337**, both before
  this directory existed and after.  The failure is
  `tests/test_panel_interface.py::test_panel_episode_replays_and_restores`, the
  known worktree-only replay divergence (FINDINGS §41,
  `research/MERGE-QUEUE.md`).  The twelve further failures the brief warned about
  appear only without a `node_modules` symlink from the main checkout; with it,
  the count is 336/1.  This track adds no tests and imports nothing from any
  tested module, so removing `research/lazy-latency/` changes neither number.
* **Shipped fixture reproduces exactly**: `.venv/bin/python -m tcn train
  --episodes 160` gives `initial_prediction_loss 0.248835613951087`,
  `final_prediction_loss 0.0022308224288281053`, `fully_frozen true`,
  `evaluation_mean_return 4.0`, `frozen_evaluation_mean_return 4.0` —
  **0.248836 → 0.002231 at 4/4 frozen.**
* Large generated artifacts (`out/*_inputs.json`, `out/*_compiled.py`,
  `out/child_*.py`) are gitignored; the reports are not.
* `research/semantic-library/` was not touched.
