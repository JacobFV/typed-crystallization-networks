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

## 19. Language: a real task learned from raw bytes, and a certificate for the rest

Full detail in `research/language-capability/RESULTS.md`. This is the first
training ever run on the language generator.

**What was learned.** On `context_free_language` ("is this string balanced?"),
staged on the privileged `construction` latent, a typed program reaches **1.000
on 724 held-out episodes of unseen lengths with 0% string overlap**, against a
0.548 majority constant, 0.5 random, and a 0.648 best fitted-feature baseline
whose training-perfect features collapse to the constant off-distribution.
Trained only on lengths {2,4,6} — twelve distinct strings — it is exact at
lengths 8 through 16 and at a nesting depth never seen.

**The lexical unit was discovered, not given.** `role="byte"` gives the agent no
notion of a symbol, so stage A learns `open(text,i) = eq(index(bytes, base+i), c)`
with `c` searched over the whole 0-255 alphabet and `base` over 0-40.
Enumeration exhausts 10,496 programs in 3.5 s with **conforming = 1 and
uniqueness certified**: `base = 14`, byte 40, which is `(`.

**What it is not, stated by the track itself.** The lesson is not context-free as
sampled: negatives are single-character flips, which always break the bracket
count, so counting and balancedness agree on 20,000 of 20,000 seeds. An
off-distribution certificate confirms the learned program agrees with counting
14/14 and with Dyck membership only 6/14 — it answers yes for `)(`. This
demonstrates counting and agreement, **not recursion**.

**Why the rest of the catalogue is out of reach, certified.** `answer` is
determined by `text` in all 179 lessons but by `construction` in only 140, so
dense staging is not uniformly available. And only **6 of 179 prompts are
byte-predictable at a fixed offset** (22 at >= 0.75), because the grammar engine
moves content with the vocabulary while `role="byte"` leaves no aggregation and
no tuple-to-set conversion, hence no scan. The proposed fix is the `gates`-shaped
one: a `set[(index, byte)]` second view of text.

**Two core defects are now blocking rather than noted.** Gradient descent
conforms in 0 of 44 runs on the identical spaces, and the cause is measured:
`lt`'s surrogate gradient is **exactly 0.0 at delta >= 17** — verified
independently here, 1.97e-01 at delta 1, 1.19e-07 at 16, exactly zero from 17 —
while this task operates at delta ~48. Choice logits still had nonzero gradients
throughout, so the signal was uninformative rather than absent.

And the derived fix from section 16 **cannot be applied as-is**: scaling `eq`'s
temperature by the carrier width also flattens that node's 256-way choice
softmax, because `SoftProgram` uses one temperature for both. Gradients fall
9e-2 to 2e-5 and every seed collapses. **Separating the surrogate temperature
from the choice temperature is a prerequisite for the whole family** — it covers
`eq`, `lt`/`le`/`gt`/`ge` and `index`, one cause behind three defects.

**Staging is what made it reachable**: 3.5 s plus 363 s, against a projected
50.7 days for the undecomposed 476M-program space.

## 20. A runnable demo, a corrected record, and two verification outcomes

`scripts/demo.sh` and a `tcn demo` subcommand run ten demonstrations against the
current tree, write artifacts under `artifacts/demo/`, and print every measured
number beside its baseline. Reported 10/10 passing in 298 s. `docs/VALIDATION.md`
is rewritten with twelve corrections stated plainly, and `STATUS.md` is new.

Two headline claims were checked independently from the supervising session.

**Confirmed, and fixed.** `.venv/bin/tcn` was broken: its shebang named
`/home/brandonin/Documents/differentiable-agentic-software/.venv/bin/python3`,
the repository's pre-rename path, so every `.venv/bin/tcn` command in
`README.md` failed with "cannot execute: required file not found". Repaired by
correcting the shebang and adding a path entry to the venv's site-packages, and
verified to run from an unrelated working directory. Note this was local venv
staleness from the rename rather than a repository defect — `scripts/setup.sh`
runs `uv sync --locked`, so a fresh environment would not have had it — but it
means every CLI instruction in the README has been failing on this machine for
the life of the project.

**Did not reproduce.** The demo track reported that a constant answer scores
3.00/4 on the joint result's own 16 test episodes, and used that to argue the
record's baseline was understated. Measured directly on those same 16 episodes
with the same objective cycling: **always-True 2.00/4 and always-False 2.00/4**,
so the best constant is 2.00/4, not 3.00. The correction the demo makes by
scoring against a 64-episode extension is still reasonable practice, but the
specific claim that motivated it is not supported and is recorded here as
unreproduced rather than propagated.

**One blocker statement in that report is ahead of its evidence.** It states
that reward-only REINFORCE now reaches 4.000 on 8/8 seeds, superseding the
finding that policy learning has never worked here. That number was read from
the policy-learning track's in-progress files while it was still running; that
track has not reported, has no `RESULTS.md`, and had experiments outstanding.
Treat it as preliminary until the track reports and its headline is checked.

## 21. Object identity: a structural impossibility, and the relation that survives it

Full detail in `research/object-identity/RESULTS.md`.

**Two structural certificates, not statistics.** Permuting the generator's object
list leaves the rendered image bit-identical while changing every foreground
pixel's `object_ids`; re-drawing every object's colour leaves `object_ids`
identical while changing the image. So **`object_ids` is not a function of the
image at any context size**, up to all 6,912 bytes. Section 14's per-pixel
certificate is a special case of this. Verified independently from the
supervising session: permuting the object list left the image identical in 8 of
8 episodes while the labels changed, and one counterexample suffices for a
"not a function" proof.

The 15-context by 12-target ceiling table agrees — advantage exactly 0.0000 at
every context for `object_ids`, `raster_rank` and `is_object_0`, and negative
when restricted to foreground. Key recurrence **falls** from 0.469 at one pixel
to 0.264 at 3x3: a wider window transfers less, not more.

**What survives is the permutation-invariant residue — the same-object
relation — and only through one predicate.** On adjacent foreground pairs, every
equality-based context is at advantage 0.0000: pixel, pair, 4-neighbourhood,
3x3, equality patterns, equality-plus-background. Adding **collinearity** takes
it to 1.0000 held-out against a 0.8842 baseline. The reason is in the renderer:
it paints `clip(base * shade)`, so one object's pixels share a colour ray. Max
integer cross product over same-object pairs is 140; the minimum over
different-object pairs is 231.

`pack` on a one-field tuple strips `role="byte"` legally, `encode` widens, and
collinearity is ordinary `mul`/`sub`/`abs`/`le`. A 393,216-program space was
exhausted by a prefix-reusing walk in 196 s (against `enumerate_fit`'s projected
1,217 s) yielding 910 conforming, 858 validation-exact, and a returned program at
**held-out max error 0.0**, applied at R=8 through 32 by three caller nodes with
**one wrong slot in 5,520**.

**The merged `operator_parameters` fix pays off immediately**: those three caller
nodes are now *discovered* rather than supplied — 8 programs, exhausted, unique,
0.1 s — closing the positional-reuse track's D4.

**A refinement that changes the fix now in flight.** The track reproduced the
dead-surrogate fault in `le` (`sigmoid(d/tau)` exactly 0.0 in value and
derivative at gap >= 89, with operating gaps of median 192), and then measured
the landscape rather than only the gradient. The shipped `tau=1` is dead **but
puts its minimum on a correct threshold** — a plateau with cliffs, not a
misleading slope. Scaling to the carrier restores the gradient and **moves the
minimum onto a wrong threshold**, collapsing the loss spread to 5.96e-08. Only a
temperature matched to the **decision margin** (32, not the carrier's 256) has
both a correct minimum and a live derivative, and it was the only arm that
worked: 1/4 conforming at the pool's own ceiling against 0/4 for shipped,
operand-scaled, carrier-scaled and offset-pinned.

So the section 16 rule is operator-dependent: **`eq` on bytes wants the carrier
width; the ordering comparisons on products of bytes want the decision margin,
and the carrier flattens them.** The core-fix track has been told not to ship a
single derived constant across the comparison family, and to report where the
loss minimum sits rather than only whether a gradient is nonzero.

**It generalizes.** `world_2d` and `world_3d`'s `agent_0/visible_ids` is
undetermined for the same reason, and the collinearity module transfers to both
unchanged at held-out 1.000000 with no re-search — it is a property of the
shared renderer rather than of one generator.

## 22. CORRECTION: a policy CAN be learned from reward here

Full detail in `research/policy-learning/RESULTS.md`. This retracts a claim I
made repeatedly, including in the briefs of four other agents.

**What I said:** pure REINFORCE fails at chance for the typed program *and* for a
budget-matched MLP, so policy learning has never worked here and is the blocker
to any closed-loop result.

**What is true:** track 6 never ran REINFORCE on the typed program. Verified
independently — `research/baselines/joint_baseline.py` imports only
`tcn.generation` and `tcn.types`, never `SoftProgram` or `JointTrainer`, so its
`reinforce` mode is a pure-PyTorch MLP. Only the MLP was measured at chance. The
F-init finding stands; the inference I drew from it does not.

With `examples/joint.py`'s hand-supplied policy constants zeroed and **all**
supervision removed, plain REINFORCE on the typed program reaches **4.00/4 on
8/8 seeds** over 64 held-out episodes, against always-false 2.13, always-true
1.88, uniform 2.05 and an oracle 4.00. It needs 400 training episodes: chance at
100, 7/8 at 200. A budget-matched MLP stays at chance across a 12-configuration
grid at every budget to 4,000 episodes.

**Environment episodes to reach 8/8 seeds at 4.00**, which is the ranking that
matters:

| approach | episodes |
|---|---|
| frozen exact model + enumeration, **no policy at all** | **27** (25 probe + 2 reward) |
| staged: probe, crystallize, then reward | 100 |
| pinned choices | 100 |
| shipped fixture (hand-initialized decoder) | 160 |
| reward only | 400 |
| probe and reward simultaneously | 400 |
| matched MLP | never |

**The project lead's world-model-first hypothesis holds in its staged form and
not its additive form.** Supervision *alongside* reward buys nothing — 400
either way. *Sequencing* it buys 4x. Of the 9 bits of learned content, 8 are
supervision-driven and 1, the readout sign, is reward-driven; neither stage
alone beats chance. And the model-based arm — enumerate action sequences against
a frozen exact world model, learning no policy — is the cheapest by an order of
magnitude at 27 episodes.

**The binding cause is the discrete-choice bottleneck, and three suspects are
refuted.** Score-function variance, the value baseline and the relaxed readout
are all indistinguishable from controls (an exact zero-variance estimator, no
baseline, a state-dependent head, and a plain `nn.Linear` readout all behave the
same). Pinning the choices raises actor SNR from 0.121 to 0.405 and cuts
episodes-to-average from 177 to 6.3. Underneath it, verified here: **a uniform
mixture over the complete 16-table `truth_*` family is the constant 0.5 with an
exactly zero Jacobian** (measured: 0.5, gradient -3.7e-09), so at zero
initialization the actor gradient to the choices is identically zero.

**Horizon does not break, and that is the diagnostic.** Everything holds to
horizon 32 under both dense and terminal-only reward — because the `logic`
generator's state is constant, no action changes it, and reward is per-step. A
horizon-4 episode is four independent draws of one 32-context bandit. **No result
in this repository has ever measured credit assignment**, and none on this
generator can. The `computer` generator at its keyboard interface is the sibling
that poses it properly.

Two corrections to track 2: batching was rejected at a fixed learning rate
(batch 32 fails at .04 and succeeds 8/8 at .1), and "logits saturate to +/-2.5
within 30 episodes" does not transfer — 0.39 at episode 30, with 0/8 seeds
reaching 2.5 by 300.

**A blocking defect for reproduction:** `TrainConfig`'s `> 0` weight checks make
every arm in this report inexpressible in the shipped trainer.

## 23. Computer use: a typed program acts in a live OS, closed loop

Full detail in `research/computer-capability/RESULTS.md`. First training ever run
on the computer generator.

The task: `/home/agent/task.txt` holds `<name> = <digit>`, and the objective is
satisfied when the file contains `str(digit+1)`. At tick 0 the terminal shows
only the setup write's JSON, carrying the file's *length* rather than its
content, so the program must issue a read, then write a value computed from what
it saw, then stop. The exact frozen `Agent` of `tcn/agent.py` drives the shipped
generator against the shipped reward:

| arm | episodes | mean return / 2 |
|---|---|---|
| **held-out documents** | 10 | **2.00 (10/10 solved)** |
| always write "5" | 10 | 0.30 |
| read, then write the modal digit | 10 | 0.20 |
| uniform random verb and digit | 10 | 0.60 |
| six random points of its own search space | 30 | 0.10 |

The learned content is `pos = sub(length, 1)` — a **computed** address chosen over
fifteen constant ones, with enumeration certifying the conforming address unique
in a 136-candidate space — plus `shift = add(value, 1)` and
`brand = eq(terminal[0], '{')` as the perceptual predicate deciding read from
write.

**It generalizes outside training structure**, which matters here because the
episode address was measured not to reach the task at all, so held-out seeds
would mean nothing. Unseen documents 2.00/2; unseen reading commands `cat`,
`head`, `tail`, `grep`, `sed`, `awk` all 2.00/2, with `wc -c` correctly at 0.00
since its stdout is a byte count; unseen document formats `count=7`, bare `7`,
`value: 7`, `  total = 7`, `answer -> 7` all 2.00/2.

**No dense supervision existed, and the fix was the `gates`-shaped one.** Over
eight documents, `state_counts` and `event_count` take one distinct series while
`terminal` takes eight — they count the agent's own actions. A gated probe
channel was added: capacity-declared `filesystem` and `processes` relations,
declared file contents, and `goal_reached`, as probes and latents only. Verified
independently here: the default observation set is `['pixels','terminal']` with
probes `['state_counts']` across 27 configurations, the channel is absent unless
`probe` is configured, and enabling it leaves observations unchanged. The agent
reports the default stream bit-identical over 192 seed/config/action/objective/
split combinations, and the channel is load-bearing — re-deriving every target
from `StepRecord.probes` alone reproduces the identical program.

**Gradient descent contributed nothing, for a fourth distinct reason.**
Enumeration exhausted 7,480 programs in 168 s with 2 conforming at held-out max
error 0.0; the relaxed arm returned exact error 52.0 and held-out accuracy 0.0.
The address choice gets `grad is None` because `unpack`, the only declared exit
from `role="byte"`, is `gradient="none"` — a **hard boundary no temperature fix
reaches**, distinct from the three numerical mechanisms in section 16. The `eq`
surrogate at the operating distance is 7.0e-251, exactly 0.0 in float32, against
1.05e-01 under section 16's carrier scaling.

**Scalar reward cannot start here, and the reason is precise.** `write.text` is a
1026-parameter typed text scored by exact string equality, so the probability
that one neutral write is rewardable is 5.65e-06 — about 176,942 write actions,
roughly 27 days. This does not contradict section 22: that correction concerns a
256-way categorical choice, while this is reward sparsity from a
high-dimensional exact-match argument. **Giving the computer generator a
narrower typed action argument is the single change most likely to open the
reward-only route.**

Independently confirms the section 18 severing bug for a third time: pinning
plumbing with a one-candidate node detached it and made `backward()` raise.
Every gradient number in that track is from a corrected scaffold.

## 24. The language curriculum was substantially exploitable; 14 lessons re-drawn

Full detail in `research/lesson-audit/RESULTS.md`.

A battery of 16 cheap exploits, each fitted on 600 training seeds and scored on
400 disjoint test seeds, with a **null control** — every answer replaced by a
uniform draw from its own choice set — bounding the selection bias of taking a
max over 16 predictors at 0.060. About 750,000 episodes in under 11 minutes.

**63 of 179 lessons are exploitable**, and **14 are solved at >= 0.80, seven at
exactly 1.000**. The mechanism split confirms the project lead's suspicion
directly: **copying 31**, surface and character statistics 18, implausible
distractors 9, memorisation of a tiny episode space 5.

Named cases: `ellipsis` appends the antecedent last, so the answer is always the
option mentioned last; `tree_to_sequence`'s "first leaf" of an in-order
rendering *is* the first token; `underspecification_reasoning`'s answer is the
number of printed `fits` lines, because the `spare` distractor list is empty on
every seed; `context_free_language` is the known bracket-count case, where both
`char_count_tree` and 1-NN reach 1.000. `parse_depth` sits at 0.998 because 85%
of prompts recur.

**14 were re-drawn**: mean best exploit 0.934 to 0.394, none above 0.63, zero
lessons at >= 0.80, and the oracle still 1.000 for every one. Exploitable count
63 to 54; the count with excess >= 0.30 fell 24 to 14.

Verified independently from the supervising session using the track's own
battery rather than a hand-rolled probe — my first attempt used a loose
heuristic that reported 1.000 in both regimes and measured nothing. On four
sampled lessons the legacy stream scores 1.000 for every one, winning through
`nearest_neighbour`, `choice_in_observation`, `prompt_length` and
`choice_ranker` respectively, while the hardened default scores 0.580, 0.205,
0.230 and 0.300 with the oracle unchanged at 1.0. Exploits collapse; the lessons
stay answerable.

**The default distribution changed deliberately**, which is the right call: an
off-by-default fix fixes nothing. The old stream is preserved and verified as
`hardening="none"`, bit-identical to the pre-audit generator over 179 lessons by
12 configurations by 8 seeds — 17,184 episodes — against a digest taken from an
independent pre-audit copy of the sources. Under the new default exactly 16
lessons change, the 14 plus two composers that draw sub-episodes; the other 163
are identical including instance ids. This also lands the language track's D3,
since `difficulty` was previously unreachable.

Three fixes change what a lesson asks and two cost a difficulty axis, all
recorded. `presupposition` is incomplete at 0.868 to 0.575, and the residual is
structural: the query's predicate type partitions the four labels, giving a hard
0.500 floor measured at 0.501 over 4,000 seeds.

**This retroactively qualifies a claim in `docs/VALIDATION.md`**, which cited
"179 executable lessons" as evidence. That was only ever a sampling check, and
63 of those lessons could be beaten by a cheap heuristic.

## 25. The substrate fixes are merged

Three defects that were blocking gradient learning across four tracks are fixed
and on main. Full detail in `research/core-gradient-fixes/RESULTS.md` and
`research/search-selection/RESULTS.md`.

**D1, severing.** `SoftProgram` now distinguishes three fixed-choice states
where it previously conflated them. A *declared* `Node.selected` fixes the
choice but not the value path, so hand-wired plumbing no longer cuts the
gradient to everything upstream — verified on merged main, a trainable constant
behind a `selected=0` node now receives `tensor([1.])` where it received `None`,
identical to the unselected case. A *frozen module* still stops gradients,
because that boundary belongs to the operator contract where section 4 puts it.
A *crystallized* node stays an exact detached boundary, since the scheduler's
connectivity guard exists to check precisely that commitment. This defect was
confirmed independently four times before it was fixed.

**D2, temperatures.** `SoftProgram.surrogate_scale` multiplies only the
relaxation path, so a surrogate widens without flattening that node's candidate
softmax. Implemented once on the selector branch and adopted verbatim by the
core-fix branch, so the merge conflicts were documentation only.

**D3, tuple broadcasting.** `relaxed`'s `tuple` branch broadcasts before
concatenating. Verified: a batched value packed with an unbatched constant now
yields shape (8,2) where it raised `RuntimeError`, with the constant still on
the gradient path.

**Also merged: automatic search-mode selection.** `tcn/select.py` with
per-example liveness, cost projection and `select_backend`, plus `mode="auto"`
on `synthesis.fit`, defaulting to the shipped path.

**Verified on merged main: 235 tests pass**, and the shipped fixture reproduces
at 0.24884 to 0.00223 with 4/4 deterministic, fully frozen and 4/4 from the
exact frozen agent.

**The costs, recorded rather than elided.** The carrier width ships for `eq`
only and opt-in. On `le` over bytes the shipped temperature is numerically dead
past |d| >= 17 **yet keeps its loss minimum on the correct threshold**, while
the carrier restores the derivative and moves that minimum onto a wrong one,
collapsing the loss spread by two to three orders of magnitude. A margin is a
property of the decision being learned rather than of the declared type, so no
constant was invented for that family. `eq`'s own width biases a
constant-selection minimum by two bytes while buying 0/12 to 12/12 on a
free-address benchmark — a trade, not a free win.

**The language track is not unblocked: still 0 of 44.** Section 19 called the
temperature separation a prerequisite. It is one, and it is not the remedy.

Liveness, measured per candidate per example: **92 of 256 candidates were live
on zero of twelve examples before the fix and none after**, while the pooled
reading reports "differentiable" in both columns. And unchanged by any of this:
`truth_0` and `truth_15` are constants with zero input gradient, so relaxation
reaches 0.875 of the mixed fixture's space and 0.766 of joint's.

## 26. The byte boundary: convolution is expressible, learnable and crystallizable

Full detail in `research/byte-numeric/RESULTS.md`. This closes the last of the
four substrate demands.

**The rule.** A byte is not a category and not a magnitude — it is a carrier
whose interpretation has not been declared, and declaring it is a graph
operation. The exclusion was right for the wrong reason: section 1 forbids
reinterpreting a *category ID* as a scalar, but a pixel channel was never a
category and never an intensity either. `role="byte"` is the **absence of a
declaration**, correctly restricted to what holds under either reading —
equality, indexing, structure — and correctly admitting a declaration.

Three role classes decide the algebra: uncommitted (`byte`), nominal
(`category`, `symbol`) and magnitude (`intensity`, and the bare role). One
operator, `interpret`, moves between them **in one direction only**, preserving
the carrier bit-for-bit so the error contract is exactly zero error.

**Verified independently from the supervising session** — the forbidden
reinterpretation stays forbidden:

| declaration | verdict |
|---|---|
| uncommitted -> intensity | legal, `gradient="exact"` |
| uncommitted -> category | legal, `gradient="none"` |
| **category -> intensity** | **illegal** |
| **symbol -> intensity** | **illegal** |
| intensity -> category | illegal |

and arithmetic opens only where it should: illegal on `byte`, illegal on
`category`, legal on `intensity`.

**The gradient is derived, not chosen.** It falls out of the lift each class
already declares in `Type.flat`: to a magnitude the lift is the identity, so the
relaxation is the identity and the derivative is 1; to a nominal ID it is a bit
decomposition, and no derivative from a magnitude into unordered bits is valid,
so the boundary is explicit. The existing relaxation contract forced that
asymmetry.

**The result.** A 3x3 Sobel-x over raw geometry pixels, nine trainable weights,
learned straight from bytes: 4/4 seeds recover the exact reference kernel at
held-out max error 0.0 after integer rounding, against 2.18e4 for the best
constant, and applied at every position by the shipped three-node caller at max
error 0.0 for R=8 and R=16.

**The isolated control is what makes it credible.** The identical 729-program
space, with the byte exiting through `pack` — the object-identity loophole —
gives the address-choice logits **exactly 0.0** gradient at every temperature
and 0/4 conforming; through `interpret` it reaches 1.3e+03 and 3/4. That is
section 23's "hard boundary no temperature fix reaches", crossed. The path
contains no `eq` and no ordering comparison, so the dead-surrogate family is
bypassed entirely rather than worked around.

**And relaxation wins outright for the first time**: it solves the strictly
larger continuous class 150-240x faster than enumeration solves the coarser
discrete analogue, whose 1,953,125 programs project to 1,211 s exhaustive.

**The cost, quantified and opt-in.** On the rung-3 foreground fixture the space
widens 216x (32,000 to 6,912,000) and the projected conforming set 63x, so
enumeration pays while relaxation does not — and with 63x more conforming
programs `enumerate_fit`'s tie-break has far more ways to be wrong, which
section 14 already measured going wrong on this exact fixture. Nothing
previously solvable became unsolvable, and the conforming set improved
qualitatively: held-out-exact fraction 0.84 to 0.91, because an ordering
predicate on an intensity generalizes across shades where equality against a
specific byte does not. Bytes remain non-numeric; arithmetic opens only at a
node whose declared output is a magnitude, downstream of a commitment the
program paid for.

**One new defect, reported not fixed.** The `index` temperature that makes the
relaxed gather exact (blur 28.07 to 0) is the same one that kills the
address-choice gradient (3/4 to 0/4) — one number at one node doing two jobs, a
fourth instance of the D2 coupling, now at `index` rather than `eq`.

243 tests pass on merged main and the fixtures are byte-for-byte identical.

## 27. The discrete backend now scores recurrence and live environment return

Full detail in `research/discrete-backend/RESULTS.md`. All core changes are in
`tcn/search.py` alone.

**Recurrence.** `enumerate_recurrent` runs a program with `Program.state` over a
**declared tick budget with a declared trailing settle window**, chosen over
run-until-stable and the reasoning is worth keeping: a stability detector would
sit invisibly inside the scorer (a program wrong the same way at every tick is
perfectly stable), one evaluation's cost would become unbounded, and candidates
settling at different ticks would be incomparable on a single budget — which is
exactly what a certificate needs them to be. `settle_window=1` is "read the
final value"; `k>1` is "arrived and stayed".

Verified independently by re-running the track's own script: on the depth
interpreter the feed-forward mode returns **0 conforming** while the recurrent
mode returns **1, certificate unique**, on the identical scaffold and data, at
both 272 and 1,088 programs. It was not slow at the recurrence — it could not
see it. The returned program is probe-exact and scores 4.00/4 on 40/40 held-out
episodes at depths 1, 2, 3, 4, 6 and 8, and recovers the general `mux` wire
lookup that the gradient run lost on 2 of 8 seeds.

**Live environment return.** `enumerate_environment` hardens a candidate, hands
it to the shipped `Agent`, rolls out and sums reward components, with an
`EpisodeLedger` counting episodes and steps as first-class returned costs. The
27-episode staged result of section 22 reproduces through the backend and now
carries a certificate the bespoke loop could not produce: **supervision proves 2
of 512 programs conform — probes determine everything except one bit — and
reward proves 1 of those 2.** Unstaged, the same answer costs 512 episodes and
still leaves 256 conforming, or 8,192 episodes for 8. So staging is a **300x
episode ratio with the better certificate on the cheaper side**.

**Prefix reuse.** `enumerate_prefix` keeps the certificate, since a rejected
subtree is *decided* rather than skipped: 6.4 s against the feed-forward
baseline's 490.8 s on the same 32,000-program space, identical conforming set of
2,608. `SearchResult` gains an explicit `certificate` field
(`unique`/`complete`/`none`) so forfeiting one cannot be silent.

**The beam is a negative, and the analysis is the useful part.** At width 1 it
found **nothing** in a space where 8.2% of programs conform. The reason: a beam
prunes by the score of a partial prefix, and a prefix has a score only where a
decided node is supervised. With dense probes, width 4 discards nothing and
stays exhaustive, so the beam is unnecessary; with output-only supervision —
which that fixture is — every prefix ties until the last node and the beam
degenerates to "keep the first k in enumeration order", returning exactly k
programs at width k. At width 1,024 it took 10.5 s against the exhaustive walk's
6.4 s and certified nothing. **No regime measured here makes the beam the right
answer**, and the recommendation is the exhaustive prefix walk as the default
fast path with the beam reserved for spaces above about 1e8 where nothing else
exists.

That is the same shape as every other result here: the tool works where
supervision is dense and degenerates where it is not.

264 tests pass on merged main.

## 28. `mode="auto"` now routes through the discrete backend

The discrete-backend track left routing exposed but not wired, because
`tcn/select.py` was not on main when it started. Both are now, so
`synthesis.fit`'s auto path dispatches through `search.route`/`solve` instead of
assuming the feed-forward scorer, with `ticks` and `settle_window` plumbed
through. That completes the hybrid-search requirement: a recurrent or
environment-coupled problem reaches the mode that can score it.

**A correction to my own demonstration, recorded because the first version
overclaimed.** I built a single-tick stateful program to show the fix mattering
and it did not: `enumerate_fit` solves that one too, because a one-tick program
with zero-initialised state happens to be scorable feed-forward. The measured
case where feed-forward genuinely returns **0 conforming** is the *multi-tick*
depth interpreter of section 27, reproduced independently earlier. So this is a
principled-routing improvement rather than a rescue of the case I first reached
for, and the toy proves nothing on its own.

264 tests pass.

## 29. A task that poses real credit assignment, and two things that solve it

Full detail in `research/credit-assignment/RESULTS.md`. This closes the last of
the four demands.

**The task.** A gated `interface='panel'` configuration of `generators/computer`:
four slot files, one holding a key and digit; actions `wait`, `look(slot)`,
`dial(value)`, `commit`. `commit` writes the register to disk and ends the
episode, and reward arrives only on that step, read back off the filesystem.

**It poses credit assignment, proved by exact computation rather than asserted.**
Belief-state dynamic programming over the task's own distribution gives
**V\*(6) = 1.0000 against a myopic optimum of 0.0625** — a 16x gap. The optimal
first action becomes `look` at horizon 3, `look` is never itself rewarded, and
its credit arrives two to five steps later. The optimal policy flips to myopic
below a discount of 0.394.

**The bandit-decomposition detector was run against the track's own task and it
clears**: V\*(H) is not H times V\*(1) — 1.0000 against 0.375 at H=6 — and the
myopic policy is not optimal at any H>1. It does not decompose the way `logic`
does, which is exactly the failure the detector existed to catch.

**Exploration is now feasible**: the probability that one neutral `commit` is
rewarded is 6.24e-02 against `write.text`'s 5.65e-06 — an **11,043x**
improvement, 28.9 episodes per reward at about 45 seconds, against roughly 27
days.

**Two things solve it, at very different prices.** Model-based — enumerate action
sequences against a frozen exact model, perception found by exhaustive
enumeration against probes, no policy learned at all — reaches **1.00/1 on 64/64
held-out in 89 environment episodes**, cheapest by an order of magnitude, as in
section 22. Reward-only REINFORCE learns the **delayed** part: 6/6 seeds recover
`look -> dial -> commit`, none of which is rewarded on the step taken.

**The controls are what make it credible.** The gamma=0 control, differing only
in `discount`, falls into the trap on 4/4 seeds, committing first at 0.0664
against the exactly computed myopic value of 0.0625; gamma=0.5 sits between, as
the 0.394 threshold predicts. `flat` scores 0.026, `probe_only` scores 0.000
reproducing F-init, and `reward_percept` scores 0.0625, re-measuring the
`unpack` gradient boundary of section 23.

**What reward-only does not learn is the argument sub-action.** It widens its
slot sampler instead of selecting the sweep (0.594 sampling, 0.240
deterministic); supplied the sub-action, it reaches 1.0000 on 3/3 seeds by
episode 200. So the hierarchy claim is half-measured: the composite-action arm
is written and smoke-tested but never run, and remains an argument rather than a
measurement.

**Verified independently here**: the default action menu is unchanged at
`('wait','command','type','key','read','write')` with probes `['state_counts']`,
while `interface='panel'` gives `('wait','look','dial','commit')`. The track
reports 384/384 identical `StepRecord`s against a pre-change copy. `observe` now
returns a literal shell menu rather than `tuple(self.action_schema)`, and panel
verbs are refused outside the panel interface.

**A required core diff, not applied**: `TrainConfig.__post_init__`'s `> 0` weight
checks make reward-only, supervision-only and gamma=0 arms inexpressible, which
is why the track re-implements `JointTrainer.episode` rather than using it.

## 30. Module selections transfer across widths; a hardened module does not

Surfaced by the object-identity track while its positional-application loop
crashed, and worth recording because it is a real constraint on the composition
story rather than a bug in that script.

Registering a module hardened at one observation width and calling it at another
raises `map: operator signature mismatch`. The reason is structural: a module's
input type **names the observation**, so a module hardened against an 8x8 raster
is typed for that raster and is not the same operator as one hardened against
16x16. Nothing in the type algebra makes them interchangeable, and nothing
should.

What transfers is the **selections**, not the hardened artifact. `apply.py`
rebuilds the scaffold at each width and reuses the chosen candidate indices,
which is what produced the reported one wrong slot in 5,520 across R=8, 16, 24
and 32 with three caller nodes, and what makes
`tcn.scaffold.positional_scaffold` agree bit-identically with a copied caller.

The practical consequence for the module library: a stored module is
width-specific. A library entry is reusable at the width it was learned at, and
generalising across widths means storing the selections and rebuilding, or
declaring an observation type that does not fix the width. Neither is
implemented. This sits alongside the interface-exactness fragility of section
27 — a stage-1 module wrong at 1 of 384 positions took stage 2 from a unique
solution to zero conforming — as the two known limits on chaining.

## 31. The composite action closes the hierarchy gap

Section 29 left one claim unmeasured: reward-only learning solves the *delay* but
does not select the argument sub-action, widening its slot sampler instead
(0.594 sampling, 0.240 deterministic). Offering a crystallized module as a
**composite action** — recursive abstraction applied to actions rather than
perception — closes it.

Six runs, 400 episodes and 528 environment episodes each, about four minutes:

| arm | deterministic eval | chose `sweep` |
|---|---|---|
| **`sweep+stare`, composite action offered** | **0.9648** (sd 0.061; 3 of 4 seeds at 1.0000, 64/64) | **4/4** |
| `stare`-only control | 0.2969 | 0/2 |
| reward only, 12-candidate pool (section 29) | 0.2396 | 0/6 |
| reward only, sub-action supplied (the ceiling) | 1.0000 | — |

Baselines: `always_wait`, `look_only` and `uniform_random` all 0.0000; myopic
`commit_now` 0.0208 measured against 0.0625 exact; `neutral_typed` 0.0833;
`fixed_slot_plan` 0.2500; oracle 1.0000. Exact ceilings: sweeping 1.0000,
i.i.d. 0.7141, constant slot 0.3333 — **and the control sits on that constant-slot
ceiling at 0.2969**, which is what makes the contrast a real effect rather than
a lucky argmax.

It is also not a tie broken at the end: two seeds hold `stare` through episode
200 and switch by 300, with the reward rate rising as they switch. Stochastic
evaluation is lower at 0.6250, because the wide slot sampler perturbs the
learned sweep — reported, not hidden.

**A verification note worth keeping.** Recomputing the summary from every
artifact gave 0.7719 over five seeds rather than 0.9648 over four, because a
fifth file scored 0.0000. That file is `macro_sweep_stare_s90`, a **10-episode**
run against the others' 400, excluded by a pre-existing `seed >= 90` filter for
short-budget runs that the report documents. The exclusion is legitimate and the
four-seed figure stands; recording the check because a headline that improves
when a zero is dropped is exactly the shape that has been wrong before in this
pass, and this time it was not.

## 32. A screenshot parses to a hierarchy

`research/visual-ladder/RESULTS.md`. The three searches drafted at the end of the
previous session were executed. Verified by recomputing every figure from
`out/rung3_parse.json` in the supervising session.

**On 12 held-out flat screens (seeds 200-211): 215 of 215 rectangles exactly
right on 12 of 12 screens, zero spurious, and 173 of 215 parent links correct
(0.805) against a `parent = root` baseline of 0.140.** Three of twelve trees are
exactly correct.

Two concessions are in the headline rather than buried: the comparison is
rectangle-not-id, and the root is supplied as the screen, so 12 of 227 widgets
are never predicted.

Every search exhausted. S0: 256/256, 2 conforming, 1 distinct function, held-out
error 0.0 — and a 16,384-program `--free` ablation returns the *same single
function*. S1: 400/400, 2 conforming (a literal swapped pair, one predicate),
with offsets `[3, 96]` discovered rather than supplied. **S2: 25/25, 1
conforming, certificate `unique`.** Random controls scored 2/400, 0/400, 1/400
and 4/50. No gradient arm was needed.

**Running it found three faults in the draft, and two were the same fault.** Fill
colour is *not* injective at `palette 32` — `bounds.json`'s own
`colour_injective` field says so, 1 of 12 episodes — and the draft leant on
injectivity twice without reading its own bound. S2's extent was a masked count
rather than a run, overshooting on 8 of 57 widgets; fixed with a prefix
conjunction at fixed depth. S0 supervised colour-equality at arbitrary pairs and
returned **0 conforming of 256, exhausted** — 1.41% of arbitrary pairs
contradict, against 0.00% of 46,528 four-neighbour pairs — and the unsatisfiable
variant is kept as a negative control. The three faults the handoff predicted
(the `min`/`last` clamp, the `lt` mask, `sum` overflow) were all fine.

**The residual is one cause, and it is measurable.** All 42 wrong links are
colour-key collisions: on the 3 collision-free screens the parse is **46/46**,
and on the rest 127/169. So the parent rule is not approximate — it is exact
wherever the key is unique, and the ceiling is a property of the key rather than
of the program.

Still missing: the root, an induced variant for S1 and S2, and all of rung two.

## 33. The parse is complete: a screenshot to an exact hierarchy

`research/visual-ladder/RESULTS.md` sections 6-9. Verified by recomputing from
`out/rung3_root.json` here: **12 of 12 held-out screens with every rectangle
exact, 227 of 227 parent links correct, 12 of 12 trees exactly correct, zero key
collisions.** Both concessions of section 32 are gone.

**The colour ceiling was a key, not a limit, and the fix needed no new program.**
The rectangle module already emits `(x, y)`, so re-keying the parent lookup on
position rather than fill colour is a different resolution rule over the *same
cached rows*: colour 173/215 (0.805) becomes position 215/215 (1.000) with 12 of
12 exact trees, corner-key collisions 0 against colour's 25. The palette was not
raised, not even as a diagnostic — that would have tuned the benchmark instead
of fixing the method.

**The root is recoverable as a program, not supplied.** A clamped corner rule
gives 227 true positives, 0 false positives and 0 false negatives over 12,288
held-out positions; S1' exhausts 400/400 with 2 conforming and S2' 25/25 unique.
The program marks its own root by `parent_key == own_key`, which is a property of
the parse rather than an annotation. Parse over 1,024 positions per screen:
227/227 rectangles, 12/12 roots, 227/227 links, 12/12 trees, against a
parent-is-root baseline of 0.132.

**Rung two ran and is not clean, reported as such.** The structural bound
reproduces (oracle 0.9462, collisions exactly `f`/`l` and `i`/`j`). T0 exhausts
256/256 but returns **64 conforming across 22 distinct functions** — the three
zero-channel tests agree at all 16,384 pixels, so the supervision does not
separate them. T1 exhausts 5/5 with **3 conforming, not unique**. The best anchor
is the widget text origin at 1.000 on 80 balanced held-out pairs against a 0.500
baseline, but that sample contains no colliding pair; over all 8,385 held-out
pairs it is 0.9957 with all 36 errors on the two known collision classes. One
unexplained observation is recorded rather than smoothed over: 10 `('i','i')`
false negatives in training.

So the visual line now reaches an exact hierarchy from raw pixels, and stops at
character identity, where the ceiling is structural and measured.
