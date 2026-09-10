# Pre-registration: attributing the residual 6.2×

Written and committed **before any arm of this track was run**.  Branch
`worktree-agent-ae2285740a6b89153`, from `main` at `2b9cb2b`.  Nothing under
`tcn/`, `generators/`, `research/lazy-guard/`, `research/lazy-latency/`,
`research/compiled-runtime/` or `research/second-family/` will be modified;
this directory is a pure addition.

---

## 1. The number under test

FINDINGS §51 / `research/lazy-latency/RESULTS.md` §3.1 measured, on the shipped
visual parser's S2 subroutine over its **deployment** distribution (54 corner
records), batch-one warm, envelope-matched:

| arm | µs / record |
|---|---|
| **A** compiled specification (`tcn/compile.py`) | 39.043 |
| **B** resynthesized lazy algorithm (`research/lazy-guard/out/stage3_algorithm.py`) | 18.781 |
| **C** hand-written reference (`research/compiled-runtime/fixtures.py`) | 3.025 |

A ÷ B = 2.079×, A ÷ C = 12.908×, and therefore **B ÷ C = 6.208×** is the
residual this track must attribute.  §51's verdict — "whatever closes the
remaining gap is not a lazy conditional" — is the starting point, not a
conclusion to be re-argued.

**Step 0, and it gates everything else: does B ÷ C = 6.208× reproduce on this
host?**  If it does not, that is the finding and it is reported as such
(house rule).  The reproduction is run before any attribution arm.

---

## 2. The two hypotheses, stated so they can be told apart

* **H-structural.**  The residual is dominated by the resynthesized program
  computing a *different algorithm* — more elementary operations per output
  element than the reference performs.  If so, no compiler pass and no
  representation change reaches it, the resynthesis programme has a ceiling
  near the 2× laziness already buys, and the project should stop investing in
  the direction.
* **H-representational.**  The residual is dominated by *how* the same
  algorithm is expressed — guard helper calls, module frames, unhoisted
  loop invariants, typed range checks, boundary/envelope cost.  If so it is
  ordinary engineering and this track must name the specific change.

These are distinguished by a **decision rule fixed here**: a cause counts as
*structural* if removing it requires the synthesizer to emit a program that
performs **fewer elementary operations**, and *representational* if it is
achievable by a semantics-preserving source-to-source transform that leaves the
executed elementary-operation count unchanged.  The one cause that straddles the
line (short-circuiting a conjunction) is reported in its own row and counted
under **structural**, because it does change the operation count.

---

## 3. The instrument: a cumulative ladder of semantics-preserving transforms

Rather than argue about which candidate matters, this track builds a **chain of
source-level rungs from arm B to arm C**, each rung a single named
transformation applied on top of the previous one, and times every rung.  The
log-time deltas between adjacent rungs then **sum exactly to log 6.208** by
construction, so the attribution cannot be fudged and whatever is left over is
visible as a named residual row.

Pre-registered rungs (each cumulative on the last):

| rung | transformation | class |
|---|---|---|
| **R0** | arm B verbatim, imported from `research/lazy-guard/out/stage3_algorithm.py` | baseline |
| **R1** | **guard-call inlining** — `_ck`, `_ix`, `_dz`, `_dzi` become inline expressions; every check they perform is still performed | representational |
| **R2** | **module-call inlining** — the `_m0` module body is inlined into both loops | representational |
| **R3** | **loop-invariant code motion + CSE** — `rec[0]`, `rec[1]`, `pos // 3`, `x`, `y` and the anchor pixel's three bytes are computed once outside the loops | representational |
| **R4** | **short-circuiting the conjunction** — the learned `truth_7` table lookups become `and`, which also stops loading the 2nd and 3rd bytes once the 1st differs | **structural** (changes executed operation count) |
| **R5** | **typed-guard elimination** — the remaining integer range checks and the `min(·, 3069)` index clamp are removed | representational (typed-safety tax) |
| **R6** | **loop-bound strength reduction** — the `x + i < 32` test moves out of the body into the loop form, giving the reference's `while` shape | representational |
| **R7** | arm C verbatim — the hand-written reference | target |

The **R6 → R7** delta is the honest *unaccounted* term: whatever separates a
mechanically-transformed arm B from the human's code after every named cause
has been removed.  It will be reported as unaccounted, with its sign, and will
not be attributed to anything.

**No rung may be timed until it passes the gate in §4.**

## 4. The gate — bit-identical output, before any timing

Exactly as `research/lazy-latency` §1 did.  Every rung R0…R7 must produce
**bit-identical** output to arm B on:

* all **2,883** held-out interior records (uniform distribution),
* all **54** corner records (deployment distribution),
* the pre-registered worst-case record (`w + h` = 47).

A SHA-256 digest over each rung's canonical outputs is recorded per
distribution and all digests must be equal.  A rung that fails the gate is
reported as failing and **is not timed**; the ladder is then reported truncated
at that point.  R5's clamp removal is the rung most at risk: if removing
`min(·, 3069)` changes any output the rung fails and is reported failed.

## 5. Timing protocol — inherited, not re-invented

`research/lazy-latency/latency.py`'s protocol verbatim, and
`research/compiled-runtime/harness.py` for gc discipline: CPU-pinned under
`taskset -c 19`, gc collected then disabled around each timed sweep, one
discarded warm-up sweep per rung per distribution, **21 repeats with the rungs
interleaved round-robin inside each repeat**, medians with non-parametric 95%
CIs from order statistics, bootstrap CIs (10,000 resamples) on every ratio.
All three distributions are reported — **deployment, uniform and worst case** —
because §51 established that they behave differently and that the worst case
moved a sign.

Arms A, B and C are re-timed in the same interleave so the reproduction of
§51's table is on the same instrument as the ladder.

Every figure is microseconds per record, batch-one, warm.  Both the
envelope-matched entry (dict in, `({outputs}, {state})` out) and the bare
function entry are measured, because candidate 2 is about exactly that
difference.

## 6. Candidate causes, and how each is falsified

The brief names five.  Each gets a measurement that can come out against it.

1. **Data representation** (`frozenset`/`tuple` materialisation vs native ints).
   Measured by `tracemalloc` peak and by counting object allocations per record
   for arms B and C separately.  **Falsified if arm C allocates as much as or
   more than arm B** — which would mean allocation cannot explain a gap in
   which B is the slower arm.
2. **Boundary cost.**  Measured three ways: (a) envelope-matched minus bare
   entry for B and C; (b) the intercept of a linear fit of per-record time
   against executed loop iterations `k`, which is the non-loop straight-line
   prologue and epilogue; (c) the same for arm C.  **Falsified if the
   intercept ratio and the envelope delta together account for less than 20% in
   log terms** of the 6.208×.
3. **Algorithmic difference beyond early exit.**  Measured by counting
   **executed loop iterations per record** for B and C on the same records, and
   **executed elementary operations per output element**.  **Falsified if B and
   C execute the same number of loop iterations on every record** — that would
   mean both are run scans and the difference is per-iteration constant factor,
   not algorithm.
4. **Interpreter overhead surviving compilation.**  Measured by counting
   executed CPython bytecodes **bucketed per code object** with
   `sys.monitoring` (extending `research/lazy-guard/cost.py`'s meter, not
   re-implementing it), plus CALL-event counts.  The share of B's bytecodes
   executed inside `_ck` / `_ix` / `_dz` / `_dzi` / `_m0` — helper frames that
   have no counterpart in C — is the direct measurement.  **Falsified if that
   share is below 25%.**
5. **Constant factors in the emitted Python** — repeated `rec[0]` subscripts,
   tuple indexing, call frames.  This is what rungs R1–R3 remove; its size is
   their measured log-delta.

## 7. Three cost measures, reported separately

Per §51's caution, **primitive operations, executed bytecodes and wall clock
disagree in both directions** and no claim here may silently mix them.  Every
rung is reported with all three, and any single-number claim states which
measure it is about.

## 8. Falsification conditions for the track as a whole

| condition | consequence |
|---|---|
| **B ÷ C = 6.208× does not reproduce on this host** | that is the headline finding; report it and re-derive the residual from the measured number |
| **structural rows (R4 plus any measured operation-count difference) dominate** — more than half the residual in log terms | H-structural: the resynthesis programme has a ceiling near 2×; report it as the primary deliverable |
| **representational rows dominate** | H-representational: name the specific compiler passes and their measured worth |
| **the named rows do not sum to 6.208×** | report the unaccounted fraction explicitly, with sign; do not redistribute it |
| **a rung fails the equivalence gate** | report it failed, truncate the ladder, attribute only what was gated |
| **noise exceeds an effect on any row** | report that row as an interval, not a point |

## 9. Amendments

Any deviation from this document is recorded in RESULTS.md §"Amendments" **with
the number it replaced** and with a statement of whether it moves the result
for or against the track's own hypothesis, to the standard set by
`research/lazy-latency/RESULTS.md` §9.

## 10. Repository constraints

`.venv/bin/python -m pytest -q` must give the same count before and after this
directory exists, and the shipped fixture must reproduce
0.248836 → 0.002231 at 4/4 frozen.  Large generated artifacts stay out of git.
