# Consolidated findings — 2026-09-08

All eight tracks complete. Every measurement below was produced
against `tcn/` and `generators/` **unchanged from the initial commit**; the only
non-`research/` edits so far are corrections to `docs/VALIDATION.md` and
`docs/LESSONS.md`.

Per-track detail and raw data live under `research/<track>/`.

## 1. The single most important result

**Progressive crystallization does not earn its complexity, and hierarchical
supervision does.**

Three tracks converged on this independently, which is why it is stated first:

- Track 1 measured the crystallization scheduler contributing nothing at working
  budgets and hurting at tight ones.
- Track 3 measured dense intermediate probes taking free-wiring synthesis from
  19-38% to 88-94%, flat at every depth from 3 to 16.
- Track 6 measured that the flagship joint result comes from dense privileged
  probe supervision, and that pure policy-gradient learning fails at chance for
  the typed program *and* for a matched neural baseline.

`ARCHITECTURE.md` section 5 (crystallization) is the mechanism the project
treats as central. Section 7 (supervision interfaces) is the mechanism the
measurements support. That is a smaller claim than the architecture makes and a
much better evidenced one.

## 2. What survived contact with measurement

| Claim | Verdict | Evidence |
|---|---|---|
| Exact typed programs generalize where a fitted network does not | **Holds, decisively** | Track 6: 1.8e-8 interpolating and 3.1e-8 extrapolating, against a tuned MLP's 5.2e-3 and 0.232. The MLP fails the repo's own 1e-6 assertion by three orders of magnitude. |
| Hierarchical supervision makes hard synthesis tractable | **Holds** | Track 3: free wiring 19-38% -> 88-94% at 2,120 candidates/node, no depth degradation. |
| Search can discover structure rather than be handed it | **Holds, once legality allows it** | Track 4: the interpreter candidate is selected in 8/8 seeds unprompted, survives crystallization, and the exact frozen agent scores 4.00 (sd 0) on truth tables never trained on. |
| Module cost accounting is implemented as specified | **Holds** | Track 5: definition counted once regardless of call sites, execution charged per use, transitive through nesting. |
| Type legality is structural, not a loss penalty | **Holds** | Verified by construction in `graph.py`; no track found an escape hatch. |
| Framework replay, export and exactness | **Holds** | 86 tests pass; every published number reproduced exactly on this host (track 6). |

## 3. What did not survive

| Claim | Verdict | Evidence |
|---|---|---|
| Progressive crystallization is necessary | **Refuted** | Track 1: 11 arms bit-identical at shipped budgets; an arm with no crystallizer and zero extra evaluations reaches the same frozen program. At 30 steps the scheduler conforms 3/16 where budget-matched argmax conforms 16/16 (p = 3.2e-06), discarding 94.8% of its optimizer steps. |
| Outside-in ordering, degradation tolerance, rollback | **No positive effect anywhere** | Track 1 component scorecard. Ordering is inert (A identical to C); tolerance is inert (E bit-identical to A in all five configs). |
| The gradient-connectivity guard protects interior learning | **Does not detect what it claims** | Track 1: severed choice logits still receive exactly-zero-but-present gradients from the entropy term, so `grad is None` tests graph reachability, not learning signal. The regularizer defeats the guard for exactly the parameters section 5 exists to protect. |
| Recursive abstraction helps | **No measurable benefit** | Track 5: module on the output path in 0 of 20 runs, including 0 of 10 successes. 1.4x description bits, 1.7x latency; execution-cost crossover never occurs. |
| Tiny description size | **Refuted** | Track 6: joint ships 68,768 bits to learn 8 bits of content, losing 2,150x to a 32-bit lookup table. `description_bits` measures JSON verbosity. |
| Tiny inference cost | **Refuted as stated** | Track 6: the "four-operation program" runs 450x slower than those four operations in plain Python and no faster than a 625-parameter MLP. Interpreter overhead dominates the `cost = 4` proxy by ~2.5 orders of magnitude. |
| 4/4 demonstrates structural generalization | **Refuted** | Track 4: the recorded scaffold measures 2.09/4 on unseen truth tables against a chance ceiling of 2.0, and cannot do better because one frozen program computes exactly one relation. |
| A generic scaffold cannot learn the joint task | **Refuted** | Track 2: the run was stopped before its transition (episodes ~750/800/2600). At 5120 episodes, 4.00/4 on both seeds tested. |

## 4. Instrumentation faults found

These are bugs in how the system measures itself, and they distorted the record.

- **F-conf** `SoftProgram.export()` argmaxes every node, so the conformance test
  for freezing one node demands the whole program already be correct. 57-74% of
  rejections mismatched only on untouched still-soft nodes. Removing the check
  goes 3/16 -> 16/16 at 71 instead of 721 optimizer steps. (Track 1)
- **F-soft** Soft loss and exact conformance disagree systematically: on some
  targets 17/17 failures reach zero soft BCE with a wrong argmax, and 92% passed
  through a correct argmax mid-training and left it. Reported loss is not
  evidence of a correct program. (Track 3)
- **F-bench** The `logic` generator's `depth` is not a difficulty knob. At depth
  8, 40% of targets are constant functions; random sampling finds the target in
  50-200 draws. The 97%-at-every-depth result describes the benchmark, not the
  method. This is the benchmark the curriculum and both flagship demos use. (Track 3)
- **F-budget** `--episodes 160` consumes 331 environment episodes: 170
  undisclosed `split='validation'` rollouts inside the crystallizer's loss
  closure. Sample-efficiency comparisons against that number were wrong. (Track 6)
- **F-init** `examples/joint.py` initializes the policy decoder constants to the
  exact solution (`logit0 = -2z+1`, `logit1 = 2z-1`). The code comment disclaims
  supplying a correct *operator*, which is true, but the readout is handed the
  answer. (Track 6)
- **F-dead** Exports retain dead scaffold nodes (32% description overstatement)
  and crystallized modules retain dead gates, charged transitively at every call
  site. (Track 5)

## 5. Core changes proposed, none applied

Ordered by measured payoff. Diffs are in the per-track reports.

1. **Replace entropy-based selection with perturbation-based selection.** The
   removal rule ties the scheduler on joint and beats it at mixed-30 in a third
   of the steps (track 1). DARTS-PT (ICLR 2021) falsified magnitude-based
   selection; TCN already computes the right measurement in `try_freeze` as an
   acceptance test and only needs to promote it to the selection rule.
2. **Fix or drop the conformance callback** (F-conf). Either compare only the
   frozen prefix, or remove it and keep the degradation guard.
3. **Fix the connectivity guard** to test learning signal rather than graph
   reachability — exclude the entropy regularizer from the probe.
4. **Memoize `Program.validate`** in `exact_tensor`, which re-runs it per batch
   row and makes module candidates 16-98x costlier than primitives. This is a
   prerequisite for testing recursive abstraction at all, since non-collapsing
   targets need >= 6 Boolean inputs. (Track 5)
5. **Collapse unit-arity module output types**: a one-output module resolves to
   `product(BOOL)`, so every call site pays a `project` node, making abstraction
   strictly costlier than inlining for all N. (Track 5)
6. **Make the logic generator's input width configurable** and add an opt-in
   non-degenerate table pool, so depth becomes a real difficulty axis. (Track 3)
7. **Wire the MDL term into `synthesis.fit`**, which currently has no cost term,
   so nothing prefers reuse. (Track 5)
8. **Report description size in learned content bits**, not serialized JSON
   length. (Track 6)
9. **Add a discrete baseline to every synthesis experiment.** `scaling.py`'s
   enumerator and the self-contained CDCL solver in `sat.py` cost milliseconds
   on these tasks and settle both flagship results outright; no gradient
   synthesis number should be reported without one alongside. (Track 8)
10. **Give `synthesis.fit` a learning-rate schedule.** A single trainable
   constant misses the 1e-3 conformance tolerance by 3.9e-2 purely because Adam
   at a fixed lr .05 never settles; 10x the steps fixes it, a schedule fixes it
   cheaply. (Track 8)

## 6. Open questions the tracks raised

- **Loss-gated annealing.** Both the entropy term and the crystallizer's 0.8x
  temperature decay commit on a fixed clock regardless of progress, and track 3's
  dominant failure mode is premature entropy collapse into a function-space
  attractor set by the target rather than the seed. Concentrating only while the
  loss is still falling is the cheapest untried experiment and needs no core
  change.
- **Node death, not wire chains.** Live-node fraction falls 1.00 -> 0.38 from
  depth 1 to 16. The DLGN pass-through pathology (88.4% in published deep
  circuits) is absent here (0-21%), but free and supplied wiring are
  indistinguishable on the metric, so typed binding is not isolated as the cure,
  and >= 16-node graphs cannot speak to 48k-1.8M-gate behavior.
- **Depth generalization needs a different encoding.** `program`'s width is
  3 x depth, so a fixed-width typed program cannot accept an episode of unseen
  depth. This requires a recurrent or set-shaped gate encoding. (Track 4)
- **Policy learning is the weak half.** Track 2 measures actor gradient SNR at
  0.221 and shows the policy readout, not the relation, is the slow part; track 6
  shows pure REINFORCE at chance for both TCN and the baseline. Episode batching
  was tried and reverted: it speeds prediction and starves the readout.

## 7. Track 8 — differentiable search is not earning its keep

Full detail in `research/enumerative-baseline/RESULTS.md`. Enumerative, random
and CDCL-SAT baselines over the identical candidate space `tcn/learning.py`
searches.

- **The flagship joint result is a 256-way choice that brute force settles in
  0.081 ms** (143 ms if every candidate is executed through the repo's own
  `Program.execute`). One gradient run takes 10-36 s. The solution is unique, so
  enumeration also returns a completeness certificate. `examples/mixed.py` is 96
  programs, swept exhaustively in 41 ms against a gradient median of 1.5 s.
- **As difficulty rises the gradient method breaks first, not last.** On targets
  that depend on all four input bits: at depth 3 it succeeds 0.50 of the time
  (~41 s/run), at depth 4 it succeeds 0.17. Exhaustive enumeration holds to
  depth 4 (~1 s) and first fails at depth 5. A hand-written CDCL solver never
  failed through depth 6 and was 4-64x faster than enumeration on every instance
  both finished. **TerpreT's finding reproduces on this repo's own tasks.**
- **`generators/logic`'s `depth` is not a difficulty dial.** Over 200 draws each
  at depths 4, 5 and 6, *zero* required as many gates as the generator used, and
  measured solution density *rises* with depth (7.8e-3 at depth 1 to 9.7e-2 at
  depth 6). This corroborates track 3's recommendation 6 and means any
  depth-vs-accuracy curve on this generator varies candidate-set size while
  target complexity stays flat.
- **Partial credit under noise is not a relaxation advantage.** Enumeration with
  a minimum-Hamming objective recovers the uncorrupted function 4/4, 3/4 and 2/4
  at 1, 2 and 3 flipped truth-table rows, against the gradient path's 0.50, 0.12
  and 0.12, roughly 1000x faster. (A pure SAT *decision* encoding does fail here:
  it correctly returns UNSAT. MaxSAT is the standard fix and was unavailable.)
- **The one clean win for the differentiable path is environment sample
  efficiency.** On the joint RL task at the gradient run's own budget of 704
  environment steps, random search over the same 256 candidates succeeds 0/20
  while the gradient path succeeds 5/5; a full enumerative sweep needs 16,384
  env steps. Where rollouts rather than CPU are scarce, the ordering flips.
- Two incidental findings about `examples/joint.py`: its declared constants
  already implement the correct decision rule before any training
  (`policy_correct_at_initialization = true`), so only the 256-way discrete
  choice is learned; and the reward admits **two** optimal programs, `(6,6)` and
  `(9,9)` — only the probe objective identifies the reference one.

The defensible reframing is NEAR-shaped, and matches track 3's finding about
supervision: relaxation as a heuristic guiding discrete search, or as the
mechanism for the continuous and environment-coupled parts of a program — not as
the search itself.

## 8. Recommended framing

The defensible claim is **typed candidate libraries with exact export and dense
hierarchical supervision**: when the library contains the answer, the method
recovers it exactly and generalizes where a fitted network does not, and dense
intermediate probes make that search tractable at depth.

Track 8 sharpens this. Relaxation is not currently earning its place *as the
search*: on both flagship tasks brute force settles the discrete content in
milliseconds against 10-36 seconds of gradient descent, and on hard targets the
gradient path degrades before enumeration does. The one place the ordering flips
is where it should: when environment rollouts rather than CPU are the scarce
resource, the gradient path succeeds 5/5 at a budget where random search over
the same candidates succeeds 0/20.

That draws the line cleanly. **Relaxation belongs where the search is coupled to
an environment, to continuous parameters, or to noisy partial credit — not where
the space is small, discrete and exactly checkable.** A discrete solver should
be the default backend for pure synthesis, with the differentiable path reserved
for the environment-coupled and continuous parts. That is a NEAR-shaped
architecture and it matches track 3's supervision result rather than competing
with it.

The crystallization schedule, the recursive abstraction machinery, and the
description-size and latency advantages are not currently supported by
measurement. Sections 4 and 5 of `ARCHITECTURE.md` should be revised to match
what is measured, section 2's candidate inventory should acknowledge a discrete
backend, and the corresponding claims in `docs/VALIDATION.md` corrected or
removed.

## 9. What to do next, in order

1. Fix the instrumentation faults in section 4 first. Until F-bench and F-soft
   are fixed, every future synthesis number on this repo is uninterpretable.
2. Add the discrete baseline (recommendation 9) so no synthesis claim ships
   without one.
3. Apply the cheap core fixes: recommendations 2, 3, 4, 5, 10.
4. Re-run the flagship experiments on a non-degenerate benchmark, with the
   `program` observation exposed and the interpreter candidate legal (track 4's
   scaffold), and report against enumeration.
5. Only then revisit crystallization, with perturbation-based selection
   (recommendation 1) and loss-gated annealing, on tasks large enough for the
   question to be meaningful.

## 10. Fixes applied (2026-09-08, after the tracks reported)

These are the first changes to `tcn/` and `generators/` since the initial commit.
Each one was measured before and after; the numbers are reproducible with the
scripts named. All 86 tests pass.

### F-bench — `depth` is now a real difficulty axis

`generators/logic` gains `inputs` (1-16), `nondegenerate` (restrict the table
pool to the ten tables that depend on both arguments), `tables` (an explicit
pool), and `min_relevant_inputs` with `max_attempts` (rejection-sample the
circuit until its final value depends on at least that many inputs, measured by
exact sensitivity). The default configuration is unchanged and was verified
bit-identical to the previous sampler across 192 seed/config combinations, so
recorded episodes and replays are unaffected.

| depth | configuration | constant targets | 4-ary targets |
|---|---|---|---|
| 8 | default | 34.0% | 0.0% |
| 8 | `nondegenerate` | 18.0% | 8.0% |
| 8 | `min_relevant_inputs=4` | 0.0% | **100%** |

The default draw was re-verified bit-identical across the same 192 seed/config
combinations *after* the later fix that made an explicit `table` override a
restricted pool, since that fix touched the table-selection branch.

High-arity targets are rare at shallow depth by construction — a function of all
w inputs needs at least w-1 two-input gates — so the rejection sampler reports
the best arity it reached rather than failing opaquely.

### F-soft — the relaxed loss can no longer be mistaken for success

`synthesis.fit` now reports `exact_max_error` (the largest disagreement between
the exported program and the targets) alongside `relaxed_loss`, plus the
`tolerance` they are judged against. On the mixed fixture these read 1.007e-06
and 0.0 respectively: the relaxed number is nonzero while the exported program
is exact, which is the disagreement in miniature.

### Constant fitting — diagnosis corrected, then fixed

Recommendation 10 proposed a learning-rate schedule. **That is the wrong fix and
was measured worse**: on track 8's own failing case a cosine decay took the error
from 3.9e-02 to 8.9e-02, and raising the constants' rate 2-20x was worse still.
The constant converges monotonically but slowly (k = 1.739 at 600 steps, 1.707 at
2000, 1.6999 at 6000), because its gradient is blurred by the candidate mixture
rather than oscillating.

The fix is to hold the selected structure hard and refine the continuous
parameters alone, then restore the choices so crystallization proceeds normally
(`polish`, default 200 steps). This is the discrete-structure/continuous-parameter
split that track 8's own analysis recommends.

| configuration | exact conformance | median exact error |
|---|---|---|
| 600 steps, no polish (as before) | **0/10** | 3.945e-02 |
| 600 steps, polish 200 | **10/10** | 1.271e-04 |

### F-conf — conformance is checked when it means something

`SoftProgram.export()` argmaxes every node, so testing exported conformance
mid-freeze reported the state of untrained nodes rather than the validity of the
freeze. The check now applies only when a freeze completes the program.

Measured on the mixed fixture with `polish=0`, against the previous
`tcn/crystallize.py` run in the same process. The fixture is deterministic
across seeds, so each row is one outcome rather than a distribution:

| steps | before | after |
|---|---|---|
| 30 | 0/16 conform, 0% frozen, **1,264** rollbacks | 0/16 conform, **75%** frozen, **336** rollbacks |
| 50 | 16/16, 100% frozen, 0 rollbacks | unchanged |
| 100 | 16/16, 100% frozen, 0 rollbacks | unchanged |
| 300 | 16/16, 100% frozen, 0 rollbacks | unchanged |

So the fix removes about three quarters of the wasted freeze trials at the tight
budget and lets hardening make partial progress, while changing nothing at
budgets that already worked. It does **not** make the 30-step budget succeed:
the mixed program is genuinely not learned in 30 steps (exact error 0.826).

Two harness differences to keep in mind when comparing against track 1, which
measured 3/16 conformance and 18.8% frozen for arm A at this budget where the
table above measures 0/16 and 0%. Track 1 added initialization noise to obtain
seed variation, whereas this fixture is deterministic as shipped; and its arm H
removed the conformance callback outright rather than gating it, which reports
`fully_frozen` without requiring the frozen program to be exact. The direction
of both measurements agrees — the callback wastes most of its trials — but the
counts are not directly comparable and no causal attribution is made here.

### The connectivity guard is still wrong — a fix was tried and reverted

Track 1's finding stands: `grad is None` tests reachability in the autograd
graph, not the presence of learning signal, and a discreteness or entropy
regularizer keeps every logit reachable.

Treating an all-zero gradient as disconnected was implemented and **reverted
after measurement**. It also flags nodes whose choice has legitimately
concentrated — entropy gradient vanishes at a one-hot distribution — so it
blocked the joint fixture from crystallizing at all: 286 deferrals against 4,
`fully_frozen` false, and no exported program, while prediction loss and return
were unchanged. An apparent improvement on the mixed fixture at 30 steps
(0/16 to 16/16) was not the guard working; it was the extra retraining the
deferrals bought, the same confound as above.

A correct guard needs the task objective separated from its regularizers at the
viability probe, which means changing what callers pass to `Crystallizer.run`.
That is a real interface change and is left open rather than guessed at.

### F2/F1 — module calls are affordable, and abstraction can now pay

`Program.execute` re-validated the program on every call, and a module operator
executes a whole sub-program per batch row. Validation is now memoized on the
immutable program instance. A single-output module also resolves to that
output's type directly instead of a one-field product, removing the `project`
node every call site was paying.

| measurement | before | after |
|---|---|---|
| 5-node module call | 113.8 us (27.4x a primitive) | **16.4 us (4.0x)** |
| execution cost vs inlining | 1.40-1.62x | **parity** |
| description-size crossover (3-gate body) | 4 call sites | **2 call sites** |
| 8-gate body at 8 call sites | never crosses over | **0.28x inlined** |

Track 5's verdict that abstraction is "a net cost at every scale the system can
search" was substantially an artifact of these two faults. Its central finding —
that the module was on the output path in 0 of 20 runs — is untouched and still
needs an answer; what has changed is that the economics now admit a regime where
reuse pays, and the search is fast enough to reach targets where it might.

### Step 2 — the discrete reference now ships with every synthesis claim

`tcn/search.py` enumerates the same candidate space `SoftProgram` relaxes, using
only `Program.execute` over the declared node candidates: no separate encoding,
no extra operators, no domain knowledge. It reports how much of the space it
covered, and when it exhausts the space it certifies whether the solution is
unique -- something a gradient run cannot establish. It applies to the discrete
choice only, and names a program's trainable constants as outside its reach
rather than silently searching a subspace.

`tcn/cli.py:mixed` now runs it alongside training whenever the space is small
enough, and records `space_size`, the baseline result, and whether the two
methods agree.

On the mixed scaffold (96 programs): enumeration solves it exhaustively in
**3.0 ms** against the gradient path's 2,751 ms, certifies the solution unique,
and selects the identical program. Track 8 measured 41 ms for this sweep; the
validation memoization above accounts for the rest.

Five tests in `tests/test_search.py` cover the uniqueness certificate, the
budget cap, the stop-at-first tradeoff, continuous-parameter reporting, and
illegal numeric domains being unusable candidates rather than errors.

## 11. Perception ladder and the address-relaxation wall (2026-09-08, overnight)

Full detail in `research/perception-ladder/RESULTS.md`. Verified independently
from the supervising session where noted.

**Highest rung that works: foreground/background segmentation of raw `geometry`
pixels.** 12/12 seeds, exact on training and zero error on 48 held-out episodes
at 2x2 and 3x3. The program recovers the background colour from raw bytes
searched over the full 256-value alphabet, with `tcn.search.enumerate_fit`
certifying the solution unique among 65,536 programs. Nothing pre-digested: the
input is the raw byte tuple and the supervision is an equivalence class of the
generator's own `depth` probe. Rung 1a (sinusoid frequency from raw samples)
also works, to about 150 candidate programs.

**The wall is relaxing an input BINDING, not an operator.** A mixture of
candidate operators is a blend of functions at a valid input; a mixture of
candidate addresses is a blend of unrelated pixel values and denotes nothing.
Measured at initialization, the steepest-descent direction picks the reference
candidate 0.81 of the time against a 0.004 chance rate for a constant choice
(208x), but 0.25 against a 0.29 chance rate for an address choice -- worse than
chance. The same task is 12/12 with addresses pinned and 1/12 at 576 programs,
0/12 at 46,656 with them free, while enumeration solves all of them.

This is the sharpest statement yet of where relaxation earns its place, and it
agrees with track 8: search the structure discretely, relax the parameters
inside it.

**Three faults found, all verified here independently:**

- **P2 (most consequential).** `SoftProgram` zero-initializes every choice
  logit, so `torch.manual_seed` does not perturb synthesis at all. Confirmed:
  four seeds produce one identical all-zero initialization. **Every per-seed
  synthesis number in this repository is therefore one outcome repeated N
  times, unless that harness added explicit initialization noise** -- track 1
  and the perturbation-selection track did and said so; the shipped fixtures do
  not. This is the mechanism behind the determinism already noted in section 10.
- **The `eq` surrogate underflows on byte data.** `exp(-(a-b)^2/tau)` at tau=1
  reaches exactly 0.0 in float32 at **|a-b| >= 11** (the report says 12; the
  measured threshold is 11, with 1.6e-28 still representable at 8). On 0-255
  data that kills the gradient for all but near-equal bytes.
- **The image boundary is representational, not a budget.** `role="byte"`
  excludes pixels from `Type.numeric` (confirmed), so `sum`, `mean`,
  `reduce_max`, all arithmetic and all ordering comparisons are type-illegal on
  images; `pack` is the only route to arithmetic and declares
  `gradient="none"`. On rung 4 enumeration exhausted the 49,152-program `eq`
  sub-algebra and certified no solution exists, while the solution that does
  exist is found by enumeration in 203 s and by gradient descent in 0/8 runs.

**Track 3's dense-probe result replicates, with a condition.** On entangled
outputs it is decisive (geometry 12/12 vs 0/12; relations 8/8 vs 1/8). On a
decomposable per-element output dense and output-only are identical, because
`probe_loss`'s elementwise BCE already is the mean of the per-element losses.
Dense probes re-separate a loss the output entangles; they do not cross either
wall (0/12 in all eight free-address arms).

**`relations.closure` is unattachable**: `join` inflates set capacity 64 to
4096, no conversion narrows it, and `union` requires identical types, so no
program can have the closure's type as output.

## 12. Recursive abstraction re-tested: track 5's verdict does not survive

Full detail in `research/recursive-abstraction-retest/RESULTS.md`. Track 5
concluded abstraction delivers no benefit and is a net cost; with F1 and F2
fixed, that conclusion is overturned, and the cause was the accounting rather
than the search.

**The decisive measurement flips.** Track 5: module on the output path in 0 of
20 arm-B runs, 0 of 10 successes. Re-test: **27 of 27 successes are the
abstracted program**, on both scaffolds.

| arm | wide scaffold, 8 seeds | tight scaffold, 24 seeds |
|---|---|---|
| A, flat | 0/8 | 0/24 |
| B, module available | **8/8** | **19/24** |
| C, same-size distractor module | 0/8 | 0/24 |

p = 1.6e-4 and 7.4e-9. Enumeration certifies this is not a search artifact:
arm A's 230,400-program space is exhausted with no solution, arm C's 2,709,504
likewise, and arm B's contains 144 solutions, **all** of which use the module.
The target's flat minimum is proved >= 7 gates against a 9-gate circuit, while
the abstracted route is 3 nodes.

**A correction to section 10's cost claim, resolved by measurement.** The
re-test reports a module candidate still costing 105x a primitive at batch 64,
against the 4.0x recorded here. Both are right and they measure different
comparisons, confirmed by running the re-test's own `costs_detail.py`:

- against an **exact** primitive (what section 10 measured, both sides through
  `exact_tensor`): its D1 reports 1.7x, and an independent sweep from the
  supervising session gives 1.4x-5.3x across 3, 5 and 9-node bodies at batches
  1 to 256. Section 10's figure stands for that comparison.
- against a **relaxed** primitive, which is pure tensor arithmetic with no
  Python loop: roughly two orders of magnitude at batch 64.

The second is the number that matters for search cost, because most candidates
in a scaffold are relaxed operators. Section 10 should be read as "a module call
is now comparable to an exact primitive", not as a claim about search cost. The
O(batch) Python loop in `exact_tensor` is untouched by the memoization; the
re-test's R1 proposes evaluating each distinct row once, measured at 7.45x on
its arm B with bit-identical output.

**What still does not pay is preference, not capability.** F3 remains unfixed
and F1 made it decisive: `SoftProgram.complexity()` is cost-weighted, and cost
is now at exact parity, so it measures 20.43 for both routes and could not
prefer abstraction even if enabled. Description bits differ (19,496 vs 20,800)
but have no differentiable surrogate, and `enumerate_fit` ranks by declaration
order rather than cost. One arm-B success calls the module three times where two
would do, and is accepted because it conforms.

Two further findings worth keeping: arm C reached relaxed loss 0.002 in a space
**certified to contain no solution**, which is the sharpest demonstration yet of
the relaxed-loss gap in section 4; and the remaining difficulty is binding
dilution rather than the gradient boundary, since a module over program inputs
receives exact 0/1 arguments and is therefore exact during soft search.
