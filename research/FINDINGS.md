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
| 4/4 demonstrates structural generalization | **Refuted for the recorded scaffold; achieved by a corrected one** | Track 4, and step 4 on a verified-fair benchmark: the recorded scaffold measures 1.95-2.03 against a best constant of 2.13-2.56 and cannot do better, since one frozen program computes one relation. A table-conditioned scaffold reaches 3.38-4.00 on gate families never trained on, in both directions, with the interpreter candidate selected 8/8 seeds unprompted. See section 13. |
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
- **P1, verified here.** `relaxed` implements `tuple` as a bare
  `torch.cat(xs, dim=-1)`, which does not broadcast: mixing a batched value
  with an unbatched constant raises `RuntimeError: Tensors must have same
  number of dimensions: got 2 and 1`, while a binary arithmetic operator on the
  same pair broadcasts to `(8, 1)` correctly. A scaffold that packs a trainable
  constant alongside a batched intermediate hits this.
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
and F1 made it decisive. *Corrected by the abstraction-preference track and
verified here:* the re-test's "20.43 for both" is `complexity()` evaluated at
the **diffuse mixture**, not at the discrete selections. At one-hot it charges
each node its selected candidate's cost, a module call is charged its whole
body cost, and dead scaffold nodes are charged too -- measured directly, a
2-node flat scaffold reads 2.0 while a single call to a 5-node module reads
5.0. So enabling the only MDL term the repo had would have taught the search to
**avoid** modules, which is worse than the neutrality originally recorded. Description bits differ (19,496 vs 20,800)
but have no differentiable surrogate, and `enumerate_fit` ranks by declaration
order rather than cost. One arm-B success calls the module three times where two
would do, and is accepted because it conforms.

Two further findings worth keeping: arm C reached relaxed loss 0.002 in a space
**certified to contain no solution**, which is the sharpest demonstration yet of
the relaxed-loss gap in section 4; and the remaining difficulty is binding
dilution rather than the gradient boundary, since a module over program inputs
receives exact 0/1 arguments and is therefore exact during soft search.


## 13. Step 4 — structural generalization, on a benchmark that earns the claim

Full detail in `research/nondegenerate-generalization/RESULTS.md`.

Track 4's original pools were degenerate: its `AFFINE` training set contains all
six tables whose output ignores an input. The replacement pools are disjoint,
entirely non-degenerate and balanced, with an analytic ceiling of exactly 2.00/4
verified by exhaustive search over all 16 gates and both objectives.

**The recorded scaffold is at chance** (1.95-2.03 against a best constant of
2.13-2.56) and structurally cannot exceed it. **The table-conditioned scaffold
reaches 3.38-4.00 on gate families it never trained on**, in both directions and
including when wiring varies, with the interpreter candidate **selected in 8/8
seeds in every condition** — made legal, not supplied.

That is the first structural-generalization result in this repository that its
benchmark actually supports.

**Enumeration solves it too.** The scaffold's own 272-program space, scored by
return on eight training episodes, exhausts in 130 s and reaches 4.00 held-out —
selecting the same interpreter candidate. Consistent with section 8: the
gradient path's advantage on this family is not solution quality, and this
comparison does not isolate the one advantage it does have, since a fair
sample-efficiency test holds rollouts fixed rather than wall clock.

**A discarded run worth recording.** The first execution reported `seen` and
`unseen` identical to two decimals everywhere. That was a fault introduced with
the pools: an explicit `table` was silently ignored when a pool was active, so
both schedules drew from the same distribution and "unseen" was never unseen.
Fixed with two regression tests; the invalid log is kept, since the identical
columns are the diagnostic. This is the second time in this pass that a result
too clean to be true turned out to be an instrumentation fault.

## 14. Discrete perception: rung 3.5, and where each method actually wins

Full detail in `research/discrete-perception/RESULTS.md`.

**Highest rung reached: a learned two-position spatial operator over raw
pixels** — an edge detector `fg(i) != fg(i+k)` with the neighbour offset
searched. Staged: freeze the rung-3 foreground module, then exhaust 48 programs
in 4.6 s returning exactly one, held-out max error 0.0, applied at every
position. Undecomposed the same target is 4.9e10 programs, projected 7.6 years.
Staging is what makes it reachable.

**Positional application scales as claimed.** R=8 through R=48 (a 6,912-byte
observation, 2,304 positions) with three caller nodes and max error 0.0 at every
width, from a single R=8 search. The space stays 32,000 programs at every
resolution because addresses are computed rather than chosen, against the
ladder's (3R^2)^2.

**A correction to section 8's framing.** Gradient descent wins prominently in
two places, on identical spaces and data:

- rung 3 at R=4 and R=8, 4/4 and 6/6 seeds exact on held-out where enumeration's
  *returned* program was not, and faster than the certifying sweep;
- the full 256-value byte alphabet, 4.3e9 programs, 6/6 with zero held-out
  error, where brute force projects to 107 days.

It loses exactly where the choice sits behind a `gradient="none"` boundary. So
the boundary is sharper than "discrete search wins": relaxation is strong on
*value* choices at a fixed address and useless on *address* choices, which is
section 11's result restated from the other side.

**Above rung 3.5 the wall is informational, not algorithmic.** For `object_ids`
and `depth` the best possible per-pixel predictor — an RGB lookup table, an
upper bound on the whole family — fits training pixels perfectly and scores
exactly the majority baseline on held-out episodes, advantage 0.000. Objects are
not one colour (mean 3.31 distinct RGBs, 34.6% single-coloured). Four candidate
families were exhausted with no solution. Those are completeness certificates,
not budget failures: the supervision does not determine the target from a single
pixel, so no per-pixel program can exist.

**Non-uniqueness is the norm and the tie-break was wrong.** Every rung-3 arm has
2,464-2,608 of 32,000 conforming, about 5% of which disagree with the renderer
on fresh episodes, and the count barely moves as supervision grows eightfold.
`enumerate_fit`'s lexicographic pick was measurably wrong on fresh episodes at
both R=4 and R=8. Requiring exactness at every position of a validation split is
what fixed it.

**Two faults, both verified here.** `tcn/search.py:evaluate` did not catch
`IndexError`, so one address running off a tuple end aborted the whole sweep —
the normal case for a window operator at an image border, and it killed 3 of 4
seeds in the free-offset arm. Fixed with a regression test. And enumeration
re-executes whole programs; a prefix-reusing walk returns the identical
conforming set 17.8x faster at R=8 and is flat in observation width where the
current loop is linear.

## 15. Depth generalization, achieved with no addition to the algebra

Full detail in `research/depth-generalization/RESULTS.md`. Verified from the raw
JSON by regenerating its tables, not from the summary.

Track 4's blocker was representational: the `program` observation is `3 * depth`
wide, so a fixed-width typed program cannot accept an episode of unseen depth.
Confirmed here as an executable check — the step-4 scaffold accepts depth 1 and
type-errors at 2, 3, 4, 6 and 8.

**The fix is a second typed view of the same state, not a new primitive.**
`generators/logic` gains a `gates` channel, emitted only when `gate_capacity` is
configured: `set[(index, wire_a, wire_b, table)]` at a declared capacity. The
index field is load-bearing, since two gates can be identical and a set is
duplicate-free. Measured: `gates` is **width 40 at every depth** while `program`
runs 3, 6, 9, 12, 18, 24. ARCHITECTURE section 1 already calls sequences indexed
values and relations sets of tuples; this is that, and nothing else.

**Sequential evaluation is expressible through `Program.state`.** `map` is
parallel and carries no fold, but section 4's explicit recurrence is the fold:
one gate per tick, dispatched with `insert`/`pair`/`filter`, wires read by `mux`
over `index`/`member`, the result committed with `insert`. Verified exact
against the generator's own wire values on 40 episodes at each of depths 1-8,
settling at exactly tick d-1.

**One fixed graph, trained at depths 1-2 only, on held-out episodes:**

| scaffold | space | d1-d2 (seen) | d3 | d4 | d6 | d8 |
|---|---|---|---|---|---|---|
| record | 256 | 2.12 | 1.62 | 1.50 | 2.06 | 2.44 |
| interpreter, no settle mux (ablation) | 1088 | 2.20 | 1.75 | 1.66 | 2.13 | 2.48 |
| interpreter, wire choice free | 1088 | 3.69-3.77 | 3.66 | 3.70 | 3.66 | 3.69 |
| **interpreter, wire lookup pinned** | 272 | **4.00** | **4.00** | **4.00** | **4.00** | **4.00** |
| best constant | — | 2.12 | 2.38 | 2.50 | 2.06 | 2.44 |

sd 0.00 across 8 seeds in the pinned row. Enumeration over the same 272-program
space exhausts in 25 s, certifies the optimum **unique**, and picks the same
program.

Note the exported **exact** program scores 4.00 where the soft model scores
3.20-3.31: hardening improves this program rather than degrading it, which is
the relaxation gap of section 4 running in the useful direction for once.

**Costs, all measured and all consistent with earlier findings.** A two-node
`mux` settle gate is what makes it learnable; without it three nodes have
`grad is None` under the task objective, and the 1.0e-5 seen under the
regularized objective is the entropy term — section 3's guard defeat reproduced
independently. Searching the wire binding rather than declaring it costs 2 of 8
seeds, with binding gradients of 4.7e-08 and 3.1e-05 against 8.7e-03 for the
operator choice: **the third independent instance of the address wall, now on a
task with no images in it.** Depth is generalized up to a declared capacity, not
unboundedly. `tcn/search.py` cannot score a recurrent program at all.

The default observation stream was verified bit-identical over 192
seed/config/split combinations, and independently here: the default observation
set is still exactly `bits`, `goal`, `program`.

## 16. CORRECTION: the address wall was misattributed

Full detail in `research/address-wall/RESULTS.md`. This supersedes the framing in
sections 11 and 14, which I stated three times and briefed several agents on.

**What I recorded:** relaxing an input address is worse than chance (0.25 against
a 0.29 chance rate) while relaxing a value at a fixed address is 208x better, so
addresses must be computed and never relaxed.

**What is actually true:** the failing arm never had an address-relaxation
problem. It had a dead surrogate. `eq`'s relaxation is `exp(-(a-b)^2/tau)` at
tau=1, which in float32 is **exactly 0.0** for `|a-b| >= 11`. The uniform mixture
over 12 raw `geometry` bytes sits 25.6 from the constant it is compared against,
where the surrogate reads 3.2e-31 — verified independently here as exactly 0.0.
With the colours pinned so addresses are the only free choice, 16 of 16 address
gradients are exactly zero. The recorded 0.25-against-0.29 figure came from
`pointing.py` silently dropping those dead nodes and averaging the survivors.

**The fix is one line and is derived, not tuned.** Scale the surrogate by the
declared carrier width, tau = 2^bits, which for `int[8]` is 256. At the failing
distance that lifts the surrogate from 0.0 to 7.7e-02.

| `rung3:centre_free_address` | R=2, 576 programs | R=4, 9,216 programs |
|---|---|---|
| shipped | 0/12 exact, 0/12 addresses | 0/12 exact, 0/12 addresses |
| with the surrogate scaled | **7/12 exact, 12/12 addresses** | **12/12 exact, 12/12 addresses** |

**The stated explanation was falsified too.** The claim was that mixing unrelated
values denotes nothing. The decisive control is a smooth array against the same
array permuted — identical values, marginals and spectrum. Arrangement has **no
effect** on the discrete route (11/12 and 12/12 either way) and a 7x effect only
on `index` descent. A 36-cell sweep shows correlation makes the discrete pick
*worse*, and mean spread is what destroys it — while the real `geometry` bytes
have a mean-spread/sd ratio of only 0.24, so that mechanism is not the ladder's
disease either.

**Three separate mechanisms, now distinguished.** M1, a mean confound in the
linear read, where the mean term beats the identifying covariance term 5,010 to
0.166 at byte scale; the landscape is nevertheless convex with the reference as
the unique global optimum, and Adam recovers in 35 of 36 sweep cells. M2, the
surrogate saturation above. M3, `index` kernel locality — at tau=1 the kernel
keeps only 0.564 of its mass on the addressed element, giving 5 to 25 local
minima with basins 1.3 to 2.5 addresses wide.

**What survives of the original finding.** Depth generalization measured wire
binding gradients of 4.7e-08 and 3.1e-05 against 8.7e-03 for an operator choice,
on a task with no images and no bytes, so `eq` saturation cannot explain it; that
instance is plausibly M3. So relaxed addressing is harder than relaxed values and
carries no uniqueness certificate, but it is **not** worse than chance and **not**
fundamentally broken. The construction rule stands in a weakened form: compute
addresses where you can, because it is cheaper and certifiable — but if you must
relax one, check that the downstream relaxation is valid *at the mixture* before
blaming the address.

**A coupling worth fixing (D2), verified here.** `SoftProgram` uses one
temperature per node for both the candidate softmax and the operator relaxation,
so a surrogate cannot be widened without simultaneously flattening that node's
choice distribution. Any fix along these lines needs them separated first.

**Method note.** This is the fourth instrumentation fault to produce a confident
wrong conclusion in this pass, and the second where the tell was a number that
looked too clean. The others were a crystallizer "improvement" that was extra
compute, a table pool that silently ignored an explicit config, and a
lexicographic tie-break presented as a solution. In every case the fault was in
the measurement, not the method under test.

## 17. External environments fit the contract, at a stated cost

Full detail in `research/external-environments/RESULTS.md`. `generators/control/`
adds MuJoCo pendulum swing-up and reacher as ordinary siblings, with locally
owned physics rather than an import from `world_3d`.

**Replay is verified, not assumed**, three ways: `Host.replay()` from recorded
inputs; snapshot, restore and continue with the raw MuJoCo integration vector
compared element-by-element over ten steps at varying `dt`; and the strongest,
a save reloaded in a **fresh interpreter**, continued, and compared
byte-for-byte at `max |delta| = 0.0`. Named streams are separated, with the
counterfactual measured — one shared stream moves the initial pose when the goal
draw changes. `data.time` is advanced only by the host's `dt`; no host clock is
read. The task is genuinely under-actuated: a scripted energy-pumping controller
reaches upright 0.9994 where the zero-torque arm never exceeds -0.99.

**What it costs.** The typing is authored, not derived: gym gives
`Box(low, high, shape, dtype)`, and every unit, frame, encoding and bound was
written by hand, where a wrong one is silently wrong. **77% of a step is contract
rather than physics** — 0.394 ms of MuJoCo inside a 1.68 ms typed step. And the
contract is only as strong as the engine: MuJoCo works because
`mjSTATE_INTEGRATION` exists, while Box2D through gym does not serialize and no
wrapper fixes that.

**Encoding decisions, interrogated against `Registry.resolve` rather than
assumed.** `role="byte"` admits only `eq` and `pack` — the section 14 trap,
avoided deliberately. Angles are declared dimensionless because radians are, and
because `unit="rad"` would make `sin`/`cos` type-illegal. Rates carry
`unit="rad/s"`, which correctly kills `sin(velocity)`, at the cost that no
conversion operator can change a unit, so the only legal exit is
`div(rate, rate)`. `floating(32)` over `fixed(16, 4096)` because swing-up reaches
|qvel| = 40.49 rad/s against fixed-point's +/-8.0, which would raise
`OverflowError` inside the observation channel mid-rollout. Bounds live on the
action type, where `policy.numeric_bounds` consumes them, and not on
observations, where a refinement would make `add` partial at execution.

**NES is specified, not built.** `stable-retro`'s libretro core rather than
`nes-py`, because `get_state`/`set_state` serialize CPU, PPU, APU and mapper
while `nes-py`'s single-slot backup cannot express an arbitrary snapshot. One
savestate at episode start plus recorded inputs. `dt` counted in frames at
60.0988 Hz, with non-integral `dt` rejected rather than rounded. Framebuffer as
the only observation; nametable, attributes and scroll as latents; and **OAM as
a probe**, `set[(index, tile, x, y, attributes)]` at capacity 64, the index field
load-bearing exactly as in the `gates` channel of section 15. ROMs user-supplied
by path and digest-checked, never distributed.

The reason NES suits this substrate is measured rather than aesthetic: flat
palettized colour with no anti-aliasing is the regime where `eq` is informative,
and OAM supplies exactly the object *names* that the object-identity work found
structurally unrecoverable from `geometry`'s renderer.

**Provenance, flagged for an explicit decision.** The synthetic line was already
crossed for *mechanism* before this track — `world_3d` imports `mujoco`,
`computer` shells out to Node. What `control` adds is an environment whose
external mechanism is the point, and NES is a further tier again: a fixed
third-party artifact that cannot be regenerated, held out by construction, or
redistributed. The report proposes a three-tier provenance grading with exact
replacement text for `AGENTS.md` line 42 and `ARCHITECTURE.md` sections 6 and 9.
**Those two documents were not edited**; this is the project lead's call.

**Fingerprint granularity, verified here.** `source_fingerprint()` hashes all of
`tcn/` and `generators/`, so any change to either invalidates every recorded
episode. Confirmed directly: all three `artifacts/system/*/episode.json.gz`
now fail `Host.restore` with "episode source revision mismatch". This is the
documented intent — `docs/VALIDATION.md` says fingerprints pin replay to their
code revision so stale artifacts are not silently resumed — and tonight's core
commits are what invalidated them. The finding is that the granularity is coarse:
adding a generator invalidates episodes from every other generator.

## 14. Preference: the objective can now see program size, and it costs search

Full detail in `research/abstraction-preference/RESULTS.md`. Branch
`abstraction-preference`, **not merged** — see `MERGE-QUEUE.md`.

Section 12 closed with "reuse is a capability, not an objective". Four changes
make it one, and the measurement is a genuine trade rather than a free win.

**`complexity()` could not have done it, and enabling it would have hurt.**
Section 12 records it measuring 20.43 for both routes; that is the *diffuse
mixture*. At the discrete selections realising each route it is **9.0 flat
against 15.0 abstracted**, because it charges every scaffold node including the
six that go dead on the abstracted route. It penalizes abstraction for exactly
the saving that makes it worth having.

**`SoftProgram.description_cost()` is the term that can.** Expected description
length in bits of the *pruned hardened* program under the current choice
distribution, module definitions charged once and call sites charged
individually, node liveness and module use taken as mean-field expectations. At
any one-hot distribution it equals `export().pruned().description_bits(registry)`
exactly. It penalizes each call site and rewards only shortness, so a
one-call-site module is dispreferred and abstraction wins only from two sites.
Exposed as `synthesis.fit(mdl_weight=...)` and `TrainConfig.description_weight`,
both defaulting to zero.

| wide scaffold, arm B, 8 seeds | conformant | median steps | 3-node (2-call) programs |
|---|---|---|---|
| `mdl_weight = 0` | 8/8 | 85 | **7/8** |
| `mdl_weight = 1e-5` | 8/8 | 160 | **8/8** |

Run to a full budget instead of stopping at the first success, the untermed
search **drifts into a larger exactly-conformant program in 3 of 8 runs after
already finding the small one**; with the term, 0 of 8. Section 12's redundant
three-call success was not an unlucky stopping point.

**What preference costs.** On the tight scaffold, where every node must live and
all 144 solutions have identical size, the same weight takes conformance from
**19/24 to 2/24** (p = 1.1e-6), and 1e-4 gives 0/24 and 0/8 on both scaffolds.
The pressure is "make fewer nodes live", which is aligned with the target on an
over-provisioned scaffold and opposed to it on an exactly-sized one.

**Three smaller results.** `exact_tensor` now evaluates each distinct argument
row once: **4.5-4.9x** on a whole soft forward pass, **6-8%** slower where rows
never repeat, bit-identical throughout — against the 7.45x section 12 quoted from
the re-test's projection for the module candidates alone. `Program.pruned()` is
applied by `Registry.register_module` and `runtime.save_program`, preserving
semantics over 3,875 row comparisons on every fixture; it removes a 62%
description overstatement from an abstracted export, but **does not move the
section 10 crossover** (still 2 call sites) and fixes only 1 of 5 learned
modules, because most of the "5 gates where 4 suffice" surcharge is redundant
*live* structure. And `enumerate_fit` takes `rank` in `order`/`description`/`cost`
and reports how many conforming programs it found, so declaration order no longer
decides between equivalent candidates; on a space holding both routes it returns
the 3-node 19,496-bit program where `order` returns the 5-node 23,944-bit one.

## 18. A GUI generator, rung one with a certificate, and a severing bug

Full detail in `research/gui-hierarchy/RESULTS.md`.

`generators/gui/` renders a widget tree to raw pixels and emits the tree as a
probe: `hierarchy : set[(id, parent, kind, x, y, w, h)]` at declared capacity —
the same depth-independent relation that solved depth generalization, one type
at 2 widgets and at 24. Replay, snapshot/restore and save/load all reproduce the
digest, including after `focus` and `press` actions.

**Recoverability was bounded before searching, and two exceptions are
certified.** Widget edges are exactly determined by a two-pixel neighbourhood in
the flat configuration (oracle 1.0000 against a 0.8261 majority), and **not**
determined with borders on, where the bound over the three `eq` bits the algebra
can write equals the majority baseline **exactly** — advantage 0.0000, so no
two-pixel program can beat a constant there. The search agrees: zero conforming
programs at exactly those settings. Widget `kind` is not determined by colour
(0.4620 against 0.2982) — the `geometry` trap, deliberately available and
measured. The parent relation is exactly determined by geometry alone, 1.0000 at
32, 155 and 368 widgets with zero ties.

**Rung one learned with a certificate.** Three arms over the true ownership
boundary, all exhausted (1,280 / 13,056 / 81,920 programs), each returning one
distinct Boolean function with the correct offset and held-out max error 0.0,
against a random control of 0/400. Both hand-supplied priors ablate away — the
wider spaces reach the identical function — which is exactly the enumeration
certificate the hand-initialization rule now asks for. The module is hardened,
registered, applied at every position by three caller nodes at max error 0.0,
and then **chosen** by a second program over a same-shaped distractor.

**The agent caught two of its own dial faults**, reproducing the F-bench trap:
requesting 12 widgets at `min_size 6` achieves 3, the same as requesting 6; and
a palette prefix that varied only the last channel changed the rung-1 result
when fixed, so the first run was discarded. `palette_levels` is reported
honestly as a gradient dial rather than a difficulty dial.

**A severing bug in `SoftProgram`, verified independently here.** Line 103 marks
every node with `selected` set as frozen, and the forward pass then evaluates it
through `exact_tensor(...).detach()`. So a node hand-wired with `selected=0` —
the natural way to express fixed plumbing whose operator is already decided —
**severs the gradient to everything upstream of it**. Confirmed directly: a
trainable constant feeding a `mul` through an identity node receives
`tensor([1.])` when the node is left unselected and `None` when it carries
`selected=0`.

This is a correctness bug, not a design choice. A frozen *module* boundary
should stop gradients, per section 4. A single-candidate node inside a scaffold
under training should not: its choice is fixed, but its value path should remain
differentiable whenever the operator declares a relaxation. It plausibly
explains gradient-arm failures in several tracks, and it is the highest-priority
core fix outstanding. The shipped `examples/` do not set `selected`, so their
recorded numbers are unaffected.
