# Track 1 — Does progressive crystallization earn its complexity?

**Status: negative result.** Run 2026-09-08 in this workspace. All numbers below
come from `research/crystallization-ablation/results/`. Nothing in `tcn/` or
`generators/` was modified; every arm is a subclass of `tcn.crystallize.Crystallizer`
or a local re-implementation of `tcn.synthesis.fit` with identical control flow.

---

## Verdict

Progressive crystallization, as implemented in `tcn/crystallize.py`, **does not
earn its complexity on either task at any budget tested.**

1. **At the shipped configurations it is a no-op.** On `mixed` at 300 steps and on
   `joint` at 160 episodes, all ten arms — including "train the soft graph and
   argmax every logit" — produce bit-identical final losses/returns and freeze
   100% of nodes. Arm A's 64 freeze trials on mixed-300 produce **zero** rollbacks;
   its 136 trials on joint produce 32 rollbacks that change no outcome.

2. **At reduced budgets it is actively harmful.** On mixed at 30 base steps the
   full crystallizer conforms 3/16 and freezes 18.8% of nodes while consuming 721
   optimizer steps; budget-matched naive argmax conforms **16/16** with the same
   721 steps (Fisher p = 3.2e-06). 94.8% of arm A's optimizer steps are computed
   and then thrown away by rollback restoring the optimizer state.

3. **Rollback fires for essentially one reason per task, and neither reason is
   what the architecture assumes.** On `mixed`, 1,074 of arm A's 1,261 rollbacks
   (85%) are `runtime conformance` and the other 187 are `degradation`; a targeted
   diagnosis shows **57–74% of the conformance rejections are checker artifacts** — the trial was rejected because a *different, still-soft* node was
   wrong. On `joint`, 32 of 32 rollbacks are `disconnected remaining region`, all
   of them deferrals that are accepted later, and disabling the guard entirely
   (arm G) changes nothing. `nonfinite loss` and `invalid trial` never fired once
   in 9,183 freeze trials, and `degradation` never changed an outcome: setting
   `tolerance=inf` (arm E) is bit-identical to arm A on **every** measured quantity
   in all five configurations, because the conformance check rejects the same
   trials instead.

4. **The conformance check is the binding constraint on freezing, not the
   scheduler design.** Arm H (arm A with the conformance callback removed, all
   other guards intact) freezes 16/16 programs fully at mixed-30 versus arm A's
   3/16 (Fisher p = 3.2e-06), using **70 optimizer steps instead of 721** and
   reaching a median loss of 0.10 versus 0.87. This is the headline finding.

5. **Against the strongest realistic competitor it ties.** The literal DARTS-PT
   perturbation-by-removal selection rule (arm FR) matches arm A exactly on joint
   (4/4 return, 8/8 seeds) and beats it on mixed-30 (13/16 vs 3/16). Outside-in
   entropy-ordered readiness is not the thing making arm A work where it works.

The only component that ever measurably helps is **residual retraining during a
trial** (arm D, `retrain_steps=0`, is worse than A at mixed-30 and mixed-50) —
and it is dominated by simply spending those same steps on ordinary training.

---

## Setup

| | mixed | joint |
|---|---|---|
| fixture | `examples/mixed.py` via a local copy of `tcn.synthesis.fit` | `examples/joint.py` + the crystallization block from `tcn.cli.joint` |
| nodes | 4 (3 with a real operator choice) | 13 (2 with a real operator choice) |
| base training | 30 / 50 / 100 / **300** steps, Adam lr 0.05 | **160** episodes, Adam lr 0.04 |
| crystallizer settings | tolerance 0.005, entropy_limit 0.9, rounds 24, retrain_steps 10, conformance callback supplied | tolerance 0.05, entropy_limit 0.9, rounds 24, retrain_steps 2, **no** conformance callback |
| primary metric | exact conformance of `model.export()` on all 16 examples; final task loss | deterministic mean return over 16 held-out test episodes (indices 10000–10015, split `test`), plus the exact frozen-program return over 16 further test episodes (30000–30015) |
| seeds | 16 | 8 |

**Seeding the mixed task.** `tcn.synthesis.fit` on `examples/mixed.py` is
bit-for-bit deterministic — choice logits start at zero, training is full-batch,
and there is no sampling anywhere. Four seeds produce four identical runs. To make
"8 seeds" mean anything, each mixed run adds seeded Gaussian noise
(std 0.5) to the choice logits at initialisation and changes nothing else. This is
recorded in every result row as `init_noise`.

### Arms

| arm | description |
|---|---|
| **A** | full `Crystallizer` as shipped (status quo) |
| **B** | naive argmax: train the soft graph for the equalized budget, then `freeze(name, argmax)` on every node. No trials, no retraining, no rollback |
| B0 | control: naive argmax with **no** budget padding (base steps only) |
| **C** | `Crystallizer` with the ready-candidate list shuffled instead of distance-ordered |
| **D** | `Crystallizer` with `retrain_steps=0` |
| **E** | `Crystallizer` with `tolerance=inf` |
| **F** | perturbation-based selection, *force* variant: score each candidate by the objective under forced hard execution; order nodes by the gap between best and second-best; decision nodes before single-candidate plumbing |
| **FR** | perturbation-based selection, *removal* variant — the literal DARTS-PT rule: score each candidate by how much the objective degrades when that candidate is deleted from the mixture |
| F2 | as F but single-candidate plumbing ordered first (isolates the ordering effect) |
| G | accept-all: every trial committed, no snapshot, no restore |
| **H** | diagnostic: arm A with the conformance callback not supplied. Degradation tolerance and gradient-connectivity guard still active. (mixed only — `tcn.cli.joint` already passes no conformance callback, so H ≡ A there by construction) |

Arms C, D, E, F, FR, F2, G, H all retain residual retraining, the degradation
tolerance and transactional rollback unless the arm name says otherwise.

---

## How the compute budget was equalized

**Counted quantity: `optimizer.step()` invocations.** `optimizer.step` is shadowed
by a counting wrapper (`arms.count_steps`) before any training begins, so the count
includes retraining steps taken *inside* freeze trials — including trials that are
later rolled back.

**Procedure, per (task, configuration, seed):**

1. Run arm A first. Record `total_steps` = base training steps + all trial
   retraining steps.
2. Run arm B with the same base training loop, then **`total_steps − base_steps`
   additional optimizer steps on exactly the same objective the crystallizer
   retrains on** (`loss_fn` for mixed; the mean loss of validation episodes
   20000–20001 for joint), then argmax-freeze. Arm B therefore ends with the
   *identical* total optimizer-step count as arm A for that seed.
3. Arms C/D/E/F/FR/F2/G/H are not padded; their own realised step counts are
   reported so the reader can see them.
4. Arm B0 is the opposite-direction control: base steps only, strictly *less*
   compute than arm A.

Realised budgets (mixed, arm A vs arm B, mean over 16 seeds):

| base steps | arm A total | arm B total | identical per seed? |
|---|---|---|---|
| 30 | 720.6 | 720.6 | yes |
| 50 | 190.0 | 190.0 | yes |
| 100 | 140 | 140 | yes |
| 300 | 340 | 340 | yes |

**This equalization is generous to arm B and I want that on the record.** Arm A's
rolled-back trial steps are real FLOPs but their *effect* is discarded — the
snapshot restore reverts both weights and optimizer state. Split three ways:

| mixed base steps | arm A total steps | steps whose effect survived | steps discarded by rollback |
|---|---|---|---|
| 30 | 720.6 | 37.5 | **683.1 (94.8%)** |
| 50 | 190.0 | 85.0 | 105.0 (55.3%) |
| 100 | 140.0 | 140.0 | 0 |
| 300 | 340.0 | 340.0 | 0 |

So at mixed-30 the honest reading is *both* of these, and both are reported:
against equal **work done** (arm B, 721 steps) arm A loses 3/16 to 16/16; against
equal **progress retained** (arm B0, 30 steps ≈ arm A's 37.5) arm A is 3/16 vs
1/16, p = 0.60 — indistinguishable. There is no budget accounting under which arm A
wins.

A second, deterministic compute proxy is reported in every table: **objective
evaluations** (calls to `loss_fn` / the validation-episode objective, including the
before/after measurements, the gradient-availability probe, the conformance calls
and the perturbation sweeps). Wall-clock on this shared 20-core machine varied by
±50% between identical repeat runs, so wall-clock is reported but should not carry
any argument.

---

## Results

### mixed synthesis — shipped configuration (300 base steps, 16 seeds)

| arm | exact conformance | final task loss | frozen nodes | optimizer steps | objective evals | wall (s) |
|---|---|---|---|---|---|---|
| **A** full crystallizer | 16/16 | 1.01e-06 | 1 | 340 | 353 | 1.84 ± 1.32 |
| **H** conformance check disabled | 16/16 | 1.01e-06 | 1 | 340 | 353 | 4.85 ± 4.50 |
| **B** naive argmax (budget-matched) | 16/16 | 1.01e-06 | 1 | 340 | 341 | 1.75 ± 1.66 |
| B0 naive argmax (unpadded) | 16/16 | 1.01e-06 | 1 | 300 | 301 | 1.15 ± 0.73 |
| **C** shuffled freeze order | 16/16 | 1.01e-06 | 1 | 340 | 353 | 1.21 ± 0.78 |
| **D** retrain_steps=0 | 16/16 | 1.01e-06 | 1 | 300 | 313 | 1.28 ± 0.97 |
| **E** tolerance=inf | 16/16 | 1.01e-06 | 1 | 340 | 353 | 1.42 ± 1.03 |
| **F** perturbation (force) | 16/16 | 1.01e-06 | 1 | 340 | 451 | 1.44 ± 0.69 |
| **FR** perturbation-by-removal (DARTS-PT) | 16/16 | 1.01e-06 | 1 | 340 | 477 | 4.29 ± 3.21 |
| F2 perturbation (plumbing first) | 16/16 | 1.01e-06 | 1 | 340 | 493 | 2.10 ± 0.77 |
| G accept-all | 16/16 | 1.01e-06 | 1 | 340 | 349 | 1.29 ± 0.59 |

The loss values are not merely close, they are the same float. Every arm recovers
the same program. **Zero rollbacks in 64 arm-A trials.** The 100-step table is
identical in every cell and is omitted here (see `results/mixed-100.jsonl`).

### mixed synthesis — reduced budgets, where the arms separate

**50 base steps, 16 seeds**

| arm | exact conformance | final task loss | frozen nodes | optimizer steps | objective evals |
|---|---|---|---|---|---|
| **A** full crystallizer | 14/16 | 0.0127 ± 0.0347 | 0.875 ± 0.342 | 190 ± 273 | 233 ± 355 |
| **H** conformance check disabled | 14/16 | 0.0127 ± 0.0347 | **1** | **90** | 103 |
| **B** naive argmax (budget-matched) | **16/16** | 1.01e-06 | 1 | 190 ± 273 | 191 ± 273 |
| B0 naive argmax (unpadded) | 11/16 | 0.534 ± 1.40 | 1 | 50 | 51 |
| **C** shuffled freeze order | 14/16 | 0.0127 ± 0.0347 | 0.875 ± 0.342 | 190 ± 273 | 233 ± 355 |
| **D** retrain_steps=0 | 11/16 | 0.0784 ± 0.164 | 0.688 ± 0.479 | 50 | 131 ± 105 |
| **E** tolerance=inf | 14/16 | 0.0127 ± 0.0347 | 0.875 ± 0.342 | 190 ± 273 | 233 ± 355 |
| **F** perturbation (force) | **16/16** | 1.01e-06 | 1 | 92.5 ± 6.8 | 228 ± 74 |
| **FR** perturbation-by-removal | **16/16** | 1.01e-06 | 1 | 93.8 ± 8.1 | 261 ± 76 |
| F2 perturbation (plumbing first) | **16/16** | 1.01e-06 | 1 | 94.4 ± 10.3 | 272 ± 78 |
| G accept-all | 14/16 | 0.0127 ± 0.0347 | 1 | 90 | 99 |

**30 base steps, 16 seeds**

| arm | exact conformance | fully frozen | median loss | frozen nodes | optimizer steps | objective evals |
|---|---|---|---|---|---|---|
| **A** full crystallizer | 3/16 | 3/16 | 0.869 | 0.188 ± 0.403 | 720.6 ± 323 | 917 ± 415 |
| **H** conformance check disabled | 6/16 | **16/16** | **0.102** | **1** | **70.6 ± 2.5** | 83.8 ± 3.0 |
| **B** naive argmax (budget-matched) | **16/16** | 16/16 | 1.01e-06 | 1 | 720.6 ± 323 | 722 ± 323 |
| B0 naive argmax (unpadded) | 1/16 | 16/16 | 4.15 | 1 | 30 | 31 |
| **C** shuffled freeze order | 3/16 | 3/16 | 0.869 | 0.188 ± 0.403 | 720.6 ± 323 | 917 ± 415 |
| **D** retrain_steps=0 | 1/16 | 1/16 | 1.25 | 0.0625 ± 0.25 | 30 | 244 ± 57 |
| **E** tolerance=inf | 3/16 | 3/16 | 0.869 | 0.188 ± 0.403 | 720.6 ± 323 | 929 ± 420 |
| **F** perturbation (force) | 14/16 | 14/16 | 1.01e-06 | 0.875 ± 0.342 | 201 ± 309 | 1102 ± 2042 |
| **FR** perturbation-by-removal | 13/16 | 13/16 | 1.01e-06 | 0.812 ± 0.403 | 261 ± 362 | 1516 ± 2374 |
| F2 perturbation (plumbing first) | 14/16 | 14/16 | 1.01e-06 | 0.875 ± 0.342 | 209 ± 305 | 1150 ± 2023 |
| G accept-all | 5/16 | 16/16 | 0.102 | 1 | 70 | 79 |

Fisher exact, mixed-30 conformance: A vs B **p = 3.2e-06**; A vs F **p = 2.4e-04**;
A vs B0 p = 0.60 (n.s.). Fully-frozen rate A vs H **p = 3.2e-06**.

**A ≡ C ≡ E in every cell of every mixed table.** Shuffling the freeze order and
disabling the degradation tolerance change nothing at all.

### joint logic task (160 episodes, 8 seeds, 16 held-out test episodes, max return 4)

| arm | fully frozen + exportable | det. mean return | frozen-program return | frozen nodes | optimizer steps | objective evals |
|---|---|---|---|---|---|---|
| **A** full crystallizer | 8/8 | **4.0** | 4.0 | 1 | 194 | 85 |
| **B** naive argmax (budget-matched) | 8/8 | **4.0** | 4.0 | 1 | 194 | 34 |
| B0 naive argmax (unpadded) | 8/8 | **4.0** | 4.0 | 1 | **160** | **0** |
| **C** shuffled freeze order | 8/8 | **4.0** | 4.0 | 1 | 190.8 ± 1.5 | 76.9 ± 3.7 |
| **D** retrain_steps=0 | 8/8 | **4.0** | 4.0 | 1 | 160 | 51 |
| **E** tolerance=inf | 8/8 | **4.0** | 4.0 | 1 | 194 | 85 |
| **F** perturbation (force) | 8/8 | 2.469 ± 0.713 | 2.719 ± 0.674 | 1 | 186 | 225 |
| **FR** perturbation-by-removal (DARTS-PT) | 8/8 | **4.0** | 4.0 | 1 | 186 | 225 |
| F2 perturbation (plumbing first) | 8/8 | 2.469 ± 0.713 | 2.719 ± 0.674 | 1 | 186 | 929 |
| G accept-all | 8/8 | **4.0** | 4.0 | 1 | 186 | 52 |

Zero variance on returns for every arm except F/F2 (per-seed F returns:
2.25, 3.5, 1.75, 2.25, 2.75, 3.5, 1.75, 2.0). **Arm B0 — 160 plain training
episodes followed by `argmax`, with no crystallizer, no trials, no rollback and
zero extra objective evaluations — reaches the same 4/4 deterministic return and
the same fully-exportable frozen program as the full scheduler.** This is the
result `docs/VALIDATION.md` currently attributes to progressive crystallization.

---

## Rollback-reason histogram

The `FreezeEvent.reason` field, pooled per task and per arm across all runs. This
is the most novel measurement in the track: no prior work reverts a discretization
decision, so what the guard actually catches matters.

### mixed, 30 base steps (16 seeds)

| arm | runs | freeze trials | accepted | degradation | disconnected remaining region | runtime conformance | nonfinite loss | invalid trial |
|---|---|---|---|---|---|---|---|---|
| **A** | 16 | 1105 | 12 | 187 | **0** | **906** | 0 | 0 |
| **H** (conformance off) | 16 | 65 | **64** | 1 | 0 | — | 0 | 0 |
| **C** | 16 | 1105 | 12 | 187 | 0 | 906 | 0 | 0 |
| **D** | 16 | 1268 | 4 | 402 | 0 | 862 | 0 | 0 |
| **E** (tolerance=inf) | 16 | 1105 | 12 | 0 | 0 | 1093 | 0 | 0 |
| **F** | 16 | 274 | 56 | 2 | 0 | 216 | 0 | 0 |
| **FR** | 16 | 370 | 52 | 74 | 0 | 244 | 0 | 0 |
| F2 | 16 | 287 | 56 | 2 | 0 | 229 | 0 | 0 |
| G (accept-all) | 16 | 64 | 64 | 0 | 0 | 0 | 0 | 0 |

### mixed, 50 base steps (16 seeds)

| arm | runs | freeze trials | accepted | degradation | disconnected | runtime conformance | nonfinite | invalid |
|---|---|---|---|---|---|---|---|---|
| **A** | 16 | 224 | 56 | 0 | **0** | **168** | 0 | 0 |
| **H** | 16 | 64 | **64** | 0 | 0 | — | 0 | 0 |
| **C** | 16 | 224 | 56 | 0 | 0 | 168 | 0 | 0 |
| **D** | 16 | 432 | 44 | 19 | 0 | 369 | 0 | 0 |
| **E** | 16 | 224 | 56 | 0 | 0 | 168 | 0 | 0 |
| **F** / **FR** / F2 | 16 | 68 / 70 / 71 | 64 | 0 | 0 | 4 / 6 / 7 | 0 | 0 |

### mixed, 100 and 300 base steps (16 seeds each)

Every arm: 64 trials, **64 accepted, zero rollbacks of any kind.**

### joint, 160 episodes (8 seeds)

| arm | runs | freeze trials | accepted | degradation | disconnected remaining region | runtime conformance | nonfinite | invalid |
|---|---|---|---|---|---|---|---|---|
| **A** | 8 | 136 | 104 | 0 | **32** | 0 (not supplied) | 0 | 0 |
| **C** | 8 | 123 | 104 | 0 | 19 | 0 | 0 | 0 |
| **D** | 8 | 136 | 104 | 0 | 32 | 0 | 0 | 0 |
| **E** | 8 | 136 | 104 | 0 | 32 | 0 | 0 | 0 |
| **F** / **FR** / F2 | 8 | 104 | 104 | 0 | 0 | 0 | 0 | 0 |
| G | 8 | 104 | 104 | 0 | 0 | 0 | 0 | 0 |

Arm A deferral analysis on joint: 8/8 runs had at least one rollback; **24 nodes
were rolled back and then frozen successfully later; 0 were rolled back and never
frozen.** Every rollback was a deferral, and arm G, which forces all 32 of those
trials through, ends at the same 4/4 return with the same fully-frozen program.

**Summary of the histogram.** 9,183 freeze trials were run across all arms and
all five configurations: 2,848 accepted, 6,335 rolled back. Pooled rollback
reasons over every arm: `runtime conformance` 5,346, `degradation` 874,
`disconnected remaining region` 115, `nonfinite loss` **0**, `invalid trial` **0**.

For arm A alone — the status quo — there were 1,261 rollbacks on mixed (1,074
`runtime conformance`, 187 `degradation`, all at the 30- and 50-step budgets;
zero at 100 and 300) and 32 on joint (all `disconnected remaining region`).

`nonfinite loss` and `invalid trial` **never fired once in 9,183 trials**.
`degradation` never changed an outcome: setting `tolerance=inf` (arm E) is
bit-identical to arm A in all five configurations, because those same trials are
then rejected by the conformance check instead (mixed-30: A = 187 degradation +
906 conformance, E = 0 + 1,093). The tolerance parameter is redundant with the
conformance callback where one exists, and inert where one does not.

---

## Why `runtime conformance` fires (targeted diagnosis)

Track 5 reported 974/~1,100 rejections with this reason. The same domination
appears here, and it is largely a **checker artifact**.

`tcn/synthesis.py:fit` passes `conform`, which calls `model.export()` — and
`SoftProgram.export()` hardens **every** node by argmax, frozen or not — then
requires every declared probe signal to match its target on every training example.
The acceptance test for freezing node *X* is therefore "the entire program,
including nodes still being trained, must already be exactly correct". Freezing can
only be accepted after plain argmax has already solved the whole task, so the
scheduler is structurally incapable of contributing anything argmax has not already
found.

`diagnose_conformance.py` records, for each rejection, which probe signal
disagreed and whether that signal's node was the node under trial, an earlier
committed freeze, or an untouched still-soft node (6 seeds each):

| base steps | conformance failures | mismatch on the trial node only (legitimate) | mismatch on trial node **and** untouched soft nodes | mismatch **only** on untouched still-soft nodes (**artifact**) | mismatch on an earlier committed freeze |
|---|---|---|---|---|---|
| 30 | 415 | 110 (27%) | 67 (16%) | **238 (57%)** | 0 |
| 50 | 84 | 22 (26%) | 0 | **62 (74%)** | 0 |

A representative artifact rejection (seed 0, steps 30): the trial froze `algebra`,
whose own exact output was correct; the rejection came from `analytic`, an
untouched soft node still selecting `identity` instead of `sin`, producing 1.80
against target sin(1.80) = 0.974.

**Arm H settles the causal question.** Removing only the conformance callback,
keeping the degradation tolerance and connectivity guard, moves mixed-30 from
3/16 fully frozen at 721 optimizer steps to **16/16 fully frozen at 71 optimizer
steps**, with median loss 0.87 → 0.10. The conformance check, not the scheduler
design, is the binding constraint. Note H still loses to budget-matched argmax
(6/16 vs 16/16 conformance) — fixing the checker makes crystallization cheap and
harmless, not better than the trivial alternative.

---

## What the gradient-connectivity guard actually detects

On joint it is the *only* reason that ever fires, so it is worth being precise
about what it tests. `try_freeze` rejects when
`torch.autograd.grad(loss, active, allow_unused=True)` returns any `None`.

Reproduced directly (train joint 160 episodes, seed 0, freeze `policy`, inspect
gradients):

```
params with grad None  (guard trips):        constants.w0, constants.w1,
                                             constants.bias0, constants.bias1
params with grad exactly 0 (guard does NOT trip):
                                             choices.0, choices.1, choices.4, choices.5,
                                             choices.6, choices.7, choices.8, choices.9,
                                             choices.11, choices.12
```

The guard fires on the **trainable constants** whose only gradient path ran through
the frozen node. It does **not** fire for choice logits whose task gradient is
completely severed, because `TrainConfig.crystal_weight * model.entropy()` keeps
every unfrozen choice parameter attached to the loss graph with an exactly-zero
gradient. `grad is None` is a proxy for "no path in the autograd graph", not for
"no learning signal", and the regularizer defeats the proxy for exactly the
parameters ARCHITECTURE.md §5 is worried about ("outside-in freezing must never
silently disconnect all learning signal to the interior").

So the guard is real — it fires 32/32 times on joint and every one of those is a
genuine severed gradient path — but (a) it protects constants rather than
architecture choices, and (b) forcing all 32 rejections through (arm G) costs
nothing: 4/4 return, 8/8 fully frozen. On this task it is a correct guard on a
harmless event.

---

## Perturbation-based selection (arm F / FR)

Implemented per the literature track's request, with residual retraining, the
degradation tolerance, the connectivity guard, conformance and rollback all intact;
only the entropy+stability readiness ordering and the argmax selection rule are
replaced.

Two variants, because they behave very differently:

* **FR — removal (the literal DARTS-PT rule).** Score candidate *i* by the
  objective when *i* is deleted from the mixture (logit set to −1e9); the winner is
  the one whose removal hurts most. **Matches arm A everywhere and beats it at
  mixed-30 (13/16 vs 3/16) using a third of the optimizer steps.** On joint it
  selects the same truth tables as argmax (6, 6 on every seed) and gets 4/4.
* **F — forcing.** Score candidate *i* by the objective when *i* alone is executed.
  This **loses badly on joint: 2.47 ± 0.71 vs 4.0.** It picks wrong truth tables
  (e.g. seed 0: relation 2, goal_relation 4 instead of 6, 6). The cause is
  specific to TCN: freezing detaches gradients, so the weight fine-tuning between
  edge decisions that DARTS-PT depends on is structurally unavailable, and forcing
  one coupled node while its partner is still a soft mixture measures the wrong
  thing. I report this as a limitation of the forcing variant, **not** as evidence
  against DARTS-PT; the faithful removal rule (FR) does not have the problem.

Arm F2 (identical to F but hardening single-candidate plumbing before decision
nodes) is bit-identical to F on both tasks at 4× the objective evaluations on
joint. Freeze *ordering* among non-decision nodes has no effect on outcome at all,
which is the same conclusion arm C reaches from the other direction.

**Bottom line for the coordinator's question:** arm A does not beat FR anywhere,
and loses to it at mixed-30. Since arm A also ties or loses to plain argmax, the
comparison to FR does not rescue it.

---

## Component-by-component scorecard

| mechanism | arms that isolate it | measured effect |
|---|---|---|
| outside-in distance ordering | A vs C, F vs F2 | **none anywhere.** Identical in every cell of every table |
| residual retraining in trials | A vs D | the only positive effect found: mixed-30 3/16 vs 1/16, mixed-50 14/16 vs 11/16. No effect at 100/300 or on joint. Dominated by spending the same steps on ordinary training |
| degradation tolerance | A vs E | **none anywhere.** `tolerance=inf` is bit-identical to A in all five configurations |
| transactional rollback (whole) | A vs G | mixed-30: A 3/16 conformance vs G 5/16 — rollback *loses*. mixed-50: 14/16 both. joint: 4/4 both. Never positive |
| gradient-connectivity guard | joint A vs G | fires 32/32 times, correctly; changes no outcome |
| runtime-conformance guard | A vs H | strongly **negative**: −10× optimizer steps and 3/16 → 16/16 fully frozen when removed |
| entropy/stability readiness rule | A vs FR | ties at high budget, loses at mixed-30 |
| the whole scheduler | A vs B | ties at the shipped budgets, loses decisively at reduced budgets |

---

## Proposed changes (NOT applied — other agents are working in `tcn/`)

These follow from the measurements above. They are written as intent plus the
exact edit; none has been made.

**1. Scope the conformance callback to what the trial actually decided.**
`tcn/synthesis.py:22-28` checks the fully-argmaxed program against targets, so a
freeze is rejected for the state of nodes it did not touch. Either (a) only invoke
the callback when `len(model.frozen) == len(program.nodes)` — i.e. as a final gate,
not a per-trial acceptance test — or (b) restrict the comparison to signals whose
source node is frozen:

```python
# tcn/synthesis.py, inside fit()
     def conform(exact):
         for ex in examples:
             _,_,trace=exact.execute(ex['inputs'],registry=model.registry)
             for signal in signals:
+                if signal.source not in model.frozen: continue   # a still-soft node is not evidence about this freeze
                 a=torch.tensor(trace[signal.source].flat());b=torch.tensor(ex['targets'][signal.target].flat())
                 if float((a-b).abs().max())>tolerance:return False
         return True
```

**2. Make the connectivity guard test for signal, not for graph reachability.**
`tcn/crystallize.py:61-62` uses `g is None`. Regularizer-only attachment passes.

```python
-                grads=torch.autograd.grad(loss,active,allow_unused=True) if active and loss.requires_grad else [None]*len(active)
-                if any(g is None for g in grads): reason="disconnected remaining region"
+                grads=torch.autograd.grad(loss,active,allow_unused=True) if active and loss.requires_grad else [None]*len(active)
+                if any(g is None or not torch.any(g!=0) for g in grads): reason="disconnected remaining region"
```
and the guard should be evaluated against the *task* objective with any entropy or
complexity regularizer excluded, or the check is decided by the regularizer.

**3. Decide whether the degradation tolerance is load-bearing.** It changed nothing
in any of the five configurations. Either give it a test that fails when it is
removed, or drop `tolerance` and the `before`/`after` bookkeeping from
`try_freeze`.

**4. Correct the claim in `docs/VALIDATION.md`.** "Progressive crystallization
accepted every node after deferring four attempts that would have disconnected
remaining gradients" is accurate as a description of the events, but the
surrounding text reads as evidence that the mechanism produced the 4/4 result. Arm
B0 gets the same 4/4 with no crystallizer at all. The sentence should say that the
deferrals occurred and that removing them (arm G) reproduces the same result.

---

## Limitations

* Two tasks, both tiny: 4 nodes with 3 real choices, and 13 nodes with 2 real
  choices (16 candidates each). "No measurable difference between arms" on a
  4-node graph is weak evidence about a 400-node graph. What generalises better is
  the mechanism-level diagnosis (§ conformance, § connectivity guard), which is
  about code paths, not about scale.
* The joint task uses `{'depth':1,'table':6,'fixed_inputs':True}`, so every episode
  instantiates the same Boolean function; held-out episodes vary the input bits and
  the objective bit only (see `research/AGENDA.md`). Zero variance across arms
  partly reflects how easy this task is.
* Mixed-task seeds come from choice-logit initialisation noise, a seeding scheme I
  introduced. Without it the fixture is deterministic and "N seeds" is meaningless;
  with it, the reduced-budget regime may be more sensitive to initialisation than
  the shipped one.
* Wall-clock was measured on a shared 20-core machine with up to 9 concurrent runs
  and varied ±50% between identical repeats; it carries no argument here.
* Arm B's extra optimizer steps are taken on the crystallizer's own retraining
  objective (bare `loss_fn` / the two validation episodes), not on a fresh training
  distribution. That is the closest match to what arm A spends its steps on, but on
  joint it means arm B takes 34 extra gradient steps on 2 validation episodes,
  which is mild overfitting pressure. Arm B0 has no such issue and reaches 4/4.
* Arm F's forcing variant is a weaker-than-intended DARTS-PT analogue; FR is the
  faithful one. Neither uses DARTS-PT's between-decision weight fine-tuning, which
  TCN's detach-on-freeze semantics do not permit.

---

## Reproduction

```bash
cd /home/brandonin/Documents/typed-crystallization-networks

# mixed synthesis: 16 seeds x 11 arms at four base-training budgets.
# Arm A runs first per seed; arm B is then padded to arm A's realised
# optimizer-step count for that seed. ~3 min per budget on 8 workers.
for s in 30 50 100 300; do
  .venv/bin/python research/crystallization-ablation/drive.py mixed \
      --seeds 16 --steps $s --workers 8
done

# joint logic task: 8 seeds x 10 arms, 160 training episodes. ~25 min.
.venv/bin/python research/crystallization-ablation/drive.py joint \
    --seeds 8 --episodes 160 --workers 8

# why runtime-conformance rejections happen
.venv/bin/python research/crystallization-ablation/diagnose_conformance.py \
    --steps 30 --seeds 6 --out research/crystallization-ablation/results/conformance-diagnosis-30.json
.venv/bin/python research/crystallization-ablation/diagnose_conformance.py \
    --steps 50 --seeds 6 --out research/crystallization-ablation/results/conformance-diagnosis-50.json

# tables
.venv/bin/python research/crystallization-ablation/summarize.py \
    research/crystallization-ablation/results/mixed-{30,50,100,300}.jsonl \
    research/crystallization-ablation/results/joint-160.jsonl

# single run of one arm (writes one JSON record)
.venv/bin/python research/crystallization-ablation/run_mixed.py --single --arm H --seed 0 --steps 30 --out /tmp/h.json
.venv/bin/python research/crystallization-ablation/run_joint.py --single --arm FR --seed 0 --out /tmp/fr.json
```

Files:

| path | role |
|---|---|
| `arms.py` | every arm as a `Crystallizer` subclass; the optimizer-step counter |
| `run_mixed.py` | mixed task; local re-implementation of `tcn.synthesis.fit` with instrumentation |
| `run_joint.py` | joint task; mirrors the crystallization block of `tcn.cli.joint` |
| `drive.py` | per-run subprocesses, two-phase budget equalization |
| `diagnose_conformance.py` | attribution of `runtime conformance` rejections |
| `summarize.py`, `merge.py` | aggregation |
| `results/*.jsonl` | one JSON record per (arm, seed), including every `FreezeEvent` |
| `results/summary.txt` | generated tables |
| `results/conformance-diagnosis-{30,50}.json` | rejection attribution, with the disagreeing values |
