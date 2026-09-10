# Pre-registration: does the lazy-guard expected-work gain survive to wall clock?

**Fixed and committed before any timing arm was run.**  Branch
`worktree-agent-a9beb89f33c873803`, from `main` at `e79a660`.  Nothing under
`tcn/`, `generators/` or `research/semantic-library/` is modified; this directory
is a pure addition.

---

## 0. The question, in FINDINGS §49's own words

> whether the expected-work gain survives translation to wall clock given the
> bytecode regression.  That needs the measured-latency arm of DESIGN §11, which
> this track did not run.

§49 established three measured facts that point *against* a wall-clock win:

1. expected primitive operations improve — 5.09× / 4.02× / 4.42×;
2. worst-case primitive operations do not improve at all — 1.00× / 1.00× / 1.015×;
3. executed CPython bytecodes get **worse** — 0.67× on stage 1, 0.63× on stage 2.

§48's decomposition of the residual 27.6× visual gap was in **bytecodes**, not
primitives.  So a lazy algorithm may reduce modelled work while being slower in
real Python.  This track measures which.

---

## 1. Arms

| arm | what | entry |
|---|---|---|
| **A** | the compiled specification | `tcn.compile.compile_program(spec, registry).module().run({port: rec}, validate=False)` |
| **B** | the resynthesized standalone algorithm | `research/lazy-guard/out/stage{1,2,3}_algorithm.py :: run(rec)` |
| **C** | the hand-written reference | stage 3 only — the extent loops of `research/compiled-runtime/fixtures.py :: visual().reference`, restricted to one interior position |

`out/*_deploy.py` is gitignored (it is only a subprocess driver); the code it
imports, `out/*_algorithm.py`, is committed and is arm B verbatim.  No character
of it is edited here.

There is no arm C for stages 1 and 2: `fixtures.py` contains no hand-written
reference for either miniature, and inventing one here would be writing the
comparison's own answer.

**Calling convention, declared in advance.**  `tcn/compile.py` emits exactly one
entry point, `run(inputs, state=None, validate=True)`, which takes a dict and
returns `({outputs}, {state})`.  B and C are bare functions of the record.  The
**primary** head-to-head is therefore **envelope-matched**: B and C are wrapped
so they accept and return the same dict shapes as A.  Bare B and bare C are
reported in a second column so the envelope's cost is visible rather than
silently charged to one side.  The envelope is charged *to B and C*, i.e. the
primary comparison is the one that is unfavourable to the hypothesis under test.

---

## 2. Distributions — reported separately, never as one scalar

DESIGN §11's rule: static worst case, expected-over-distribution, and
measured-realized are three numbers and no single scalar may stand in for them.

### Stage 3 — the shipped visual parser's S2 subroutine

| id | what | records |
|---|---|---|
| **D_deploy** | **corner positions only** — the positions the assembly's `filter` actually passes to the rect module | **54** |
| D_uniform | every interior position of the held-out episodes (seeds 200, 201, 202) | 2,883 |
| D_worst | the single held-out record maximizing arm B's executed loop iterations | 1 |

`D_deploy` is the deployment distribution and is 1.87% of `D_uniform`
(§49's corner row).  **The verdict is read off `D_deploy`.**  Quoting
`D_uniform` as the deployment number is the false-win failure mode criterion 6
of `research/lazy-guard` exists to prevent, and it is forbidden here.

### Stages 1 and 2 — the miniatures

| id | what | inputs |
|---|---|---|
| D_uniform | a deterministic stride-16 subsample of the declared domain (16⁴ resp. 4⁸) | 4,096 |
| D_worst | the input maximizing arm B's work — stage 1 `(15,15,15,15)`, stage 2 the input on which no position satisfies the predicate | 1 |
| D_deploy | stage 1 only: a sparsity-matched analogue at hit rate 1/48, the parse's own 20-of-961 | 4,096 |

Equivalence (§4) is checked on the **full** 65,536 domain; only the timing sweep
is subsampled, because 21 repeats × 3 arms × 65,536 calls is 4M calls per stage
and the stride-16 subsample is drawn once, deterministically, and shared by
every arm.

---

## 3. What is measured, per (stage, distribution, arm)

* **batch-one warm latency**, wall clock, µs per record;
* **peak RSS**, MB, in a child process after one full sweep of `D_uniform`;
* **bytes on disk**, the arm's source in bytes;
* **cold start**, ms, `/usr/bin/python3 -I` from process start to first result.

"Batch-one" means one record per call; the timer brackets a full sweep of the
distribution and divides by the record count, because a single µs-scale call is
below `perf_counter_ns`'s useful resolution on this host.  For `D_worst` the
same record is called `len(D_uniform)` times so the two are measured by the same
instrument.

---

## 4. Equivalence, asserted before any timing is reported

No timing number is written for a stage unless all three hold:

1. **interpreter == A.**  `tcn.graph.Program.run` (typed, the shipped
   interpreter) equals arm A. Stage 3: on a stratified sample of 32 held-out
   records covering every corner-extent stratum, because the interpreter costs
   **0.454 s per record** on the 3,072-byte carrier (measured; 22 minutes for
   the full set) — the same oracle chain `research/lazy-guard` §6.1 declared and
   §48 certified.  Stages 1 and 2: on 1,024 inputs.
2. **A == B == C, bit-identical, on every record of every distribution used** —
   stage 3: all 2,883 plus all 54; stages 1 and 2: all 65,536.
3. A SHA-256 digest over the concatenated canonical outputs of each arm is
   recorded in `out/equivalence.json` and the three digests are equal.

If any assertion fails the stage is reported as **not measured**, with the
counterexample.

---

## 5. Noise control — this is a latency measurement on a shared host

The host has 20 cores and is **shared with another agent working under
`research/semantic-library/`**; `/proc/loadavg` read while writing this document
is `4.77 4.34 4.88`.  That is not a quiet machine and the protocol is built for
it rather than pretending otherwise.

* **Pinned** to a single CPU with `os.sched_setaffinity` (core 19) and launched
  under `taskset -c 19`.  One measurement process at a time; no arm is run
  concurrently with another.
* `gc.collect()` then `gc.disable()` around every timed block, as
  `research/compiled-runtime/harness.py :: timed` does.  That harness is reused
  rather than replaced.
* **One warm-up sweep per arm per distribution**, discarded, before any kept
  sample.  Modules are imported and compiled once, outside every timer.
* **R = 21 repeats** per (stage, distribution, arm).
* **Arms interleaved round-robin inside each repeat** — A, B, C, A, B, C, … —
  so thermal and load drift is shared by all arms instead of being assigned to
  whichever ran last.
* Reported: **median**, **min**, **IQR**, and a **nonparametric 95 % CI of the
  median** from the order statistics of the 21 repeats.
* A **ratio** (B ÷ A) is reported with a **bootstrap 95 % CI**, 10,000
  resamples over paired repeats.
* `/proc/loadavg` and the wall time are recorded before and after every stage
  and printed in the results.

**Effect-versus-noise rule, fixed in advance:** if the 95 % CI of the B ÷ A
ratio **contains 1.0**, the row is reported as *"no measurable difference"* with
the interval, and **not** as a point estimate in either direction.

---

## 6. Crossover — the reusable finding, if B wins anywhere

* **Stage 3.**  Bin the 2,883 held-out records by the achieved extent
  `max(w, h) ∈ [1, 31]` and time A and B per bin.  The crossover **k\*** is the
  smallest extent at which B's median per-record latency exceeds A's.  Then
  report where `D_deploy` sits relative to `k*`, from its measured extent
  histogram.  A is constant-work (a fixed 31-term formulation) and B is linear
  in the extent, so a crossover must exist inside [1, 31] or B wins everywhere.
* **Stage 1.**  Sweep the predicate hit rate
  `p ∈ {0, 1/48, 1/16, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0}`,
  4,096 inputs per point from `random.Random(20260909)`, and report **p\***,
  the hit rate at which B's median crosses A's, by linear interpolation between
  the bracketing points.

Achieved bin counts are reported, never requested ones.

---

## 7. Falsification, stated and to be honoured

| condition | reading, fixed in advance |
|---|---|
| **B is not faster than A on `D_deploy`** | the expected-work gain **does not survive to wall clock**; §49's architectural result stands **without** a performance claim. Reported plainly. This is the outcome the bytecode numbers predict. |
| B faster on `D_uniform` but not on `D_deploy` | the win exists **only off-deployment** and must be stated that way; the uniform number may not be quoted alone. |
| B slower in the worst case by roughly the bytecode ratio (≈1.5×) | confirms branch overhead dominates when the predicate always hits, and bounds when laziness is worth it. |
| the 95 % CI of B ÷ A contains 1.0 | **noise exceeds the effect**; report the interval, not a point estimate. |

**A negative here is a first-class deliverable.**  If B does not win on
`D_deploy` it means the residual 27.6× visual gap of §48 is not addressable by
laziness alone, which redirects the resynthesis programme.  **The measurement
will not be tuned until B wins.**  No arm, distribution, repeat count or
calling convention declared above will be changed after seeing a result; any
change that becomes necessary will be recorded as an amendment in
`RESULTS.md` with the number it replaced.

---

## 8. Repository constraints carried from the brief

* All 337 tests: `.venv/bin/python -m pytest -q` → the baseline recorded before
  this directory existed is **336 passed, 1 failed**, the known worktree-only
  `tests/test_panel_interface.py::test_panel_episode_replays_and_restores`
  replay divergence (FINDINGS §41, `research/MERGE-QUEUE.md`).  The four
  `generators/computer` tests need a gitignored `node_modules` symlink from the
  main checkout, which is in place.
* The shipped fixture must reproduce `0.248836 → 0.002231` at 4/4 frozen.
* Large artifacts stay out of git.
* Probes are supervision, never model inputs.  Nothing here feeds a model.
