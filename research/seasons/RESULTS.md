# Seasonal crystallization: reversibility repairs the scheduler and still loses to argmax

**Run 2026-09-09 on branch `seasons`, built on `0b538ec`.** All numbers come from
`research/seasons/results/`. Every arm is the scheduler in `tcn/crystallize.py`
on this branch; nothing here re-implements it. Defaults are unchanged: `seasons`,
`prune` and `close_block` are all off, and the shipped fixtures reproduce
(see section 7).

`research/loss-gated-eligibility/RESULTS.md` ended with the sentence this track
exists to test:

> the interval in which the perturbation measurement beats argmax is exactly the
> interval in which committing is a mistake. For DARTS-PT-style selection to pay
> here, the freeze would have to be reversible.

It now is. This is the answer.

---

## Verdict

1. **Reversibility does not rescue measurement-based selection.** At equalized
   optimizer steps, seasons never beat plain argmax on either fixture at any
   budget tested. This is the **fourth** independent confirmation that the
   crystallization scheduler does not earn its complexity: track 1 (progressive
   crystallization), `perturbation-selection` (DARTS-PT selection),
   `loss-gated-eligibility` (commitment timing), and now reversible commitment.

2. **It does repair the scheduler, and by more than the gate did.** On `mixed` at
   the two tight budgets where the incumbent fails, seasons take conformance from
   5/8 to 8/8 (at 5 steps) and 7/8 to 8/8 (at 10 steps); on `joint` at 40
   episodes they take frozen return from 2.281 to 2.594. Argmax handed the same
   optimizer-step total conforms 8/8 on `mixed` with 20-30% fewer forward passes,
   and scores **3.625** on `joint` -- a full point above the seasonal arm, on
   9.3x fewer environment episodes. The repair is real and it is not a win.

3. **The population does not help either, and the reason is measurable.** With
   four members on a two-objective front, selection culls something in most
   generations, but the population's conformance is identical to the same four
   members run with **no selection at all**, and to a single argmax run given the
   population's summed step budget. What a population buys here is restarts.

4. **One thing seasons found that the incumbent could not: a live defect in the
   connectivity guard.** Releasing a commitment late can strand a residual set in
   which freezing any single member disconnects the others; the per-node guard
   then refuses all of them forever. Measured on `joint` at 40 episodes, seed 0:
   91 refusals, the run ending 10/13 frozen and unable to export. The fix is a
   block trial, and it is not a weakening of the guard. That defect is latent in
   the shipped scheduler too — it is reachable whenever freeze order puts such a
   set last — and nothing before this track had a way to hit it.

---

## 1. The thaw policy, stated

Per ARCHITECTURE.md section 5, every quantity below is experiment configuration.
These are the values every number here was taken at; none was tuned per fixture.
The policy is implemented in `Crystallizer.thaw_candidates` and `Crystallizer.summer`.

**What thaws.** Only choices *this scheduler* committed. A declared
`Node.selected` is a hand-supplied prior, not a commitment the run made;
`SoftProgram.thaw` raises on one, and AGENTS.md is the reason. A node committed
by pruning is dead and never a candidate.

**On what evidence.** Two rules, in order.

- **Regression.** A node whose accepted freeze left the objective *worse* than
  before it (`after > before`, admitted only by the degradation tolerance) is
  released regardless of rank. That is direct evidence the commitment cost
  something.
- **Least decisive first.** The rest are ranked by the perturbation margin
  recorded at the moment of commitment — the gap between the best and second-best
  candidate-removal score, which is the same quantity that ordered the freeze —
  and the least decisive `thaw_fraction` are released. Ties go to the more recent
  commitment, which was measured on the more concentrated distribution.

**What a summer does.** Releases those choices (`thaw`), resets each released
node's temperature to `warm_temperature = 1.0` and drops its quantization
pressure, then spends `summer_steps = 10 * retrain_steps` optimizer steps
retraining the whole soft graph. The reset is load-bearing: a node released at
the winter's temperature floor is one-hot in all but name and re-freezes on the
same measurement.

**What prevents an infinite cycle.** Four independent rules, any one of which
terminates the run.

1. `seasons = 3` caps the number of summers.
2. `thaw_limit = 1` caps releases per node; after that its commitment is final.
3. `thaw_fraction = 0.5` decays by `thaw_decay = 0.5` each summer, so the
   released set shrinks to empty, and an empty set ends the run.
4. A node that re-freezes onto the **same** candidate it was released from is
   marked *settled* and never thaws again: the release confirmed the evidence
   instead of overturning it.

`tests/test_crystallization_seasons.py` asserts each of these, including a run
configured with 50 summers that stops early on rule 2.

**Closing.** A run always ends in a winter, and the closing winter is not capped
by the season length — a program released in the last summer must be able to
crystallize again. If it still cannot, `close_block=True` makes one transactional
block trial over the whole residual set. See verdict 4 and section 6.

**Pruning.** Within a winter, any node no live node reads is committed at its
argmax with no trial. Liveness propagates backwards from the outputs and state
updates, counting every candidate's sources for a soft node and only the selected
candidate's sources for a frozen one, so freezing shrinks the live set as the run
proceeds. `Program.pruned()` drops these from the export, so the commitment is
free and it removes the node from every later perturbation sweep.

## 2. The selection rule, stated

Implemented in `population.py` (`dominated`) and `population_joint.py`.

Each of `P = 4` members is an independent `SoftProgram` over the same scaffold,
differing at initialisation by seeded Gaussian noise on the choice logits, with
its own optimizer. The budget is spent in `G = 4` generations. At the end of each
generation every member is scored on two numbers:

- `L`, the probe loss it is being trained on (on `joint`, the unregularized
  validation loss);
- `D`, `SoftProgram.description_cost()` — the expected description length in bits
  of the *pruned hardened* program, which at any one-hot distribution is exactly
  `export().pruned().description_bits(registry)`.

Member *i* **dominates** member *j* when `L_i <= L_j` and `D_i <= D_j` with at
least one strict. The non-dominated front survives untouched. Every dominated
member is replaced by a copy of a uniformly chosen front member with fresh
Gaussian noise (`sigma = 0.5`) on its choice logits and a fresh optimizer —
Adam's moments belong to the culled trajectory.

**No weight is chosen between the two objectives, and neither is ever traded
away.** The lightest member survives even if it is the worst performer, and the
best performer survives even if it is the heaviest. That is the point:
`research/abstraction-preference/RESULTS.md` measured a single description-cost
weight taking conformance from 19/24 to 2/24 on a scaffold where every node must
live, while the same weight takes minimal-program rate from 7/8 to 8/8 on an
over-provisioned one. A population does not have to pick.

**The population's answer** is the conforming member with the fewest pruned
description bits; if none conform, the lowest-loss member. On `joint` it is the
member with the highest frozen-agent return, ties to the lightest.

**The control that matters.** `--select none` runs the same `P` members with no
selection: `P` independent restarts. With `--arm argmax` that is
`argmax-restarts`. A population is charged the **sum** of its members' compute,
so a population of 4 costs 4x, and the honest baselines are (a) that same total
handed to one argmax run and (b) best-of-P restarts at the same total. A
population that beats a single argmax but not best-of-P restarts has bought
parallel restarts, not selection.

## 3. `mixed`, at equalized compute

Supervised synthesis, `examples/mixed.py`, 8 seeds, tolerance 5e-3. The fixture
is bit-for-bit deterministic as shipped -- `SoftProgram` zero-initialises its
choice logits, training is full-batch, nothing samples -- so `torch.manual_seed`
alone does not vary synthesis. **Every run adds seeded Gaussian noise (std 0.5,
track 1's value) to the choice logits at initialisation and changes nothing
else.** `--noise 0` reproduces the shipped single deterministic outcome.

`argmax@X` is plain argmax with no crystallizer, padded with extra optimizer
steps on the crystallizer's own objective until its total matches arm `X`'s
realised total -- steps inside rolled-back trials and every summer's retraining
included. `argmax0` is unpadded.

**5 base training steps** (`results/mixed-5.jsonl`)

| arm | conformant | fully frozen | opt steps | forwards |
|---|---|---|---|---|
| `shipped` | 5/8 | 5/8 | 132 | 256 |
| `shipped-cap` | 5/8 | 5/8 | 222 | 418 |
| `prune` | 5/8 | 5/8 | 132 | 256 |
| **`seasons`** | **8/8** | 8/8 | 436 | 554 |
| `seasons-noblock` | 8/8 | 8/8 | 436 | 554 |
| `population/seasons/pareto` | 8/8 | 8/8 | 1650 | 2200 |
| `population/seasons/none` | 8/8 | 8/8 | 1604 | 2020 |
| `population/argmax/pareto` | 1/8 | 8/8 | 16 | 36 |
| `argmax-restarts` | 0/8 | 8/8 | 16 | 36 |
| `argmax@shipped` | 5/8 | 8/8 | 132 | 134 |
| **`argmax@seasons`** | **8/8** | 8/8 | 436 | **437** |
| `argmax@population/seasons/pareto` | 8/8 | 8/8 | 1650 | 1651 |
| `argmax0` | 0/8 | 8/8 | 5 | 6 |

**10 base training steps** (`results/mixed-10.jsonl`)

| arm | conformant | fully frozen | opt steps | forwards |
|---|---|---|---|---|
| `shipped` | 7/8 | 7/8 | 80 | 186 |
| `shipped-cap` | 7/8 | 7/8 | 110 | 270 |
| `prune` | 7/8 | 7/8 | 80 | 186 |
| **`seasons`** | **8/8** | 8/8 | 401 | 507 |
| `seasons-noblock` | 8/8 | 8/8 | 401 | 507 |
| `population/seasons/pareto` | 8/8 | 8/8 | 1637 | 2138 |
| `population/seasons/none` | 8/8 | 8/8 | 1591 | 1987 |
| `population/argmax/pareto` | 2/8 | 8/8 | 32 | 52 |
| `argmax-restarts` | 0/8 | 8/8 | 32 | 52 |
| `argmax@shipped` | 6/8 | 8/8 | 80 | 81 |
| **`argmax@seasons`** | **8/8** | 8/8 | 401 | **402** |
| `argmax@population/seasons/pareto` | 8/8 | 8/8 | 1637 | 1638 |
| `argmax0` | 0/8 | 8/8 | 10 | 11 |

**30 and 100 base steps** (`results/mixed-30.jsonl`, `results/mixed-100.jsonl`)
are at ceiling: every arm except the unpadded restart controls conforms 8/8, so
they discriminate nothing and are reported only for completeness. At 100 steps
even `argmax0` conforms 8/8 -- FINDINGS section 3's "inert at shipped budgets",
again.

**Reading these.** Seasons repair the incumbent: 5/8 to 8/8 and 7/8 to 8/8,
which is a larger repair than the loss gate managed on this fixture. They repair
it by spending 3.3x to 5x the optimizer steps, and handing those same steps to
plain argmax reaches 8/8 too, on **20-30% fewer forward passes** because argmax
runs no perturbation sweep. `seasons` versus `argmax@seasons` is 8/8 versus 8/8
at both budgets (Fisher p = 1). There is no budget on this fixture at which
reversible commitment beats argmax.

**The `prune` control is exactly inert** here: `prune` is bit-identical to
`shipped` at every budget (same conformance, same steps, same forwards), because
the mixed scaffold has four nodes and none of them ever goes dead. Pruning is not
what seasons buy.

**The population buys restarts, not selection.** Three comparisons, all at
`mixed-5` and `mixed-10`:

- `population/seasons/pareto` vs `population/seasons/none` (the same four members
  with selection switched off): 8/8 vs 8/8 at both budgets, and the same mean
  pruned description bits (17,728, the minimum, for every member of both). And
  selection is *active*: across the eight seeds at `mixed-10` it culled and
  reseeded a member at **18** of the 24 selection points, against 0 for the
  no-selection control. It changes nothing.
- `population/seasons/pareto` vs `argmax@population/seasons/pareto` (one argmax
  run given the population's **summed** 1,637-1,650 steps): 8/8 vs 8/8.
- `population/argmax/pareto` vs `argmax-restarts` (selection on and off, members
  frozen at argmax): 1/8 vs 0/8 at mixed-5, 2/8 vs 0/8 at mixed-10. Fisher
  p = 1 and 0.47. This is the only place selection moves the number at all, and
  it does not survive.

The mechanism is visible in the front. With four members on two objectives the
non-dominated front is usually large: at `mixed-10` seed 0 the front is
`{1,2,3}` twice and then **all four** members, so the last selection point culls
nothing. Description cost varies by ~20 bits
across members of a four-node scaffold where every program has the same shape,
so `D` almost never breaks a tie that `L` has not already decided. The population
is doing what four restarts do.

## 4. `joint`, at equalized compute

Environment-coupled prediction + policy, `examples/joint.py`, 8 seeds, tolerance
0.05, `retrain_steps = 2`, mirroring `tcn.cli.joint`. Score is the mean
deterministic return of the **exact frozen program** over 16 held-out test
episodes (maximum 4.0); a run that does not fully freeze cannot be exported and
scores 0. `episodes` is the honest environment total: training episodes plus two
validation rollouts per objective evaluation, which is FINDINGS F-budget's
undisclosed cost, counted.

**40 training episodes** (`results/joint-40.jsonl`)

| arm | frozen return | fully frozen | opt steps | forwards | episodes |
|---|---|---|---|---|---|
| `shipped` | 2.281 +- 1.299 | 7/8 | 119 | 6,400 | 768 |
| `seasons` | 2.594 +- 1.356 | 7/8 | 360 | 14,930 | 1,834 |
| `population/seasons/pareto` | 3.281 +- 0.901 | 8/8 | 1,279 | 51,898 | 6,487 |
| **`argmax@shipped`** | **3.625 +- 0.694** | 8/8 | 119 | 1,836 | **198** |
| **`argmax@seasons`** | **3.625 +- 0.694** | 8/8 | 360 | 5,704 | 681 |
| `argmax@population/seasons/pareto` | 3.812 +- 0.530 | 8/8 | 1,279 | 20,404 | 2,518 |
| `argmax-restarts` | 2.469 +- 0.674 | 8/8 | 160 | 1,600 | 200 |
| `argmax0` | 1.938 +- 0.513 | 8/8 | 40 | 576 | 40 |

This is not close. Argmax padded to the shipped crystallizer's own 119 optimizer
steps scores **3.625 against seasons' 2.594**, using **3.4x fewer optimizer
steps and 9.3x fewer environment episodes** than the seasonal arm. Seasons do
improve on the incumbent (2.594 against 2.281) and cost 3x the steps and 2.4x
the episodes to do it. Every crystallizing arm loses to the step-matched argmax
partner; the population loses to a single argmax run given its summed budget
(3.281 against 3.812) while spending 2.6x that run's episodes.

`argmax-restarts` at 2.469 is the reading that decides verdict 3 in the other
direction too: four independent argmax runs at 40 episodes each, best-of-four,
score *worse* than one argmax run given the same 160 steps in a single
trajectory (`argmax@shipped`, 3.625, and it uses fewer episodes still). On this
fixture, restarts are a worse use of compute than continuing to train -- which
is also why the population's 3.281 is not evidence for selection.

**160 training episodes -- the shipped budget** (`results/joint-160.jsonl`)

| arm | frozen return | fully frozen | opt steps | forwards | episodes |
|---|---|---|---|---|---|
| `shipped` | 4.000 +- 0.000 | 8/8 | 192 | 3,776 | 440 |
| `seasons` | 4.000 +- 0.000 | 8/8 | 444 | 12,416 | 1,520 |
| `population/seasons/pareto` | 4.000 +- 0.000 | 8/8 | 1,922 | 57,086 | 7,136 |
| `argmax@shipped` | 4.000 +- 0.000 | 8/8 | 192 | 2,048 | 224 |
| `argmax@seasons` | 4.000 +- 0.000 | 8/8 | 444 | 6,080 | 728 |
| `argmax-restarts` | 4.000 +- 0.000 | 8/8 | 640 | 5,440 | 680 |
| **`argmax0`** | **4.000 +- 0.000** | 8/8 | **160** | **1,536** | **160** |

At the shipped budget every arm is at the maximum return, including the arm with
no crystallizer and no padding at all. This is FINDINGS section 3's "inert at
shipped budgets" reproduced exactly, now with seasons and populations added to
the list of things that are inert there. The only separation is cost: seasons
spend 9.5x `argmax0`'s environment episodes and the population 44.6x, for the
same 4.000.

## 5. What reversibility found that monotone freezing could not

The connectivity guard in `try_freeze` refuses a commitment that would leave some
still-trainable parameter with no path from the task objective. It is a per-node
test, and it has an ordering hazard: a *set* of nodes can be jointly frozen-able
while no single member of it is, because freezing any one of them detaches the
others. Monotone freezing rarely reaches such a set, because it commits in
decisiveness order from a fully soft graph. A summer creates one deliberately --
it puts back exactly the nodes whose commitments were least decisive, which on
`joint` are the plumbing nodes near the policy readout.

Measured on `joint` at 40 episodes, seed 0, with `close_block=False`: after the
third summer the residual set `{bit_1, world, mul0}` was refused on every
remaining round, **91 refusals with reason `disconnected remaining region`**, and
the run ended **10 of 13 frozen** with no exportable program and therefore a
score of 0. Across 8 seeds (`results/joint-40-noblock.jsonl`) the arm fully freezes **1 of
8** times and scores **0.313 +- 0.884**, against 7 of 8 and 2.594 +- 1.356 for
the identical arm with the closing block trial enabled -- the same optimizer
steps (359 against 360) and the same episodes (1,826 against 1,834) to within a
percent. Seven of the eight runs stop at 10 or 7 of 13 nodes frozen with 88-172
`disconnected remaining region` refusals each. Without the block, seasonal
crystallization on this fixture simply does not terminate in a program

The remedy is one transactional block trial over the residual set, which section
5 of ARCHITECTURE.md already contemplates ("temporarily harden a candidate
node/**block**"). It is not a weakening of the guard: when the block completes the
program, there is no remaining trainable region for the guard to protect, and
degradation, conformance and rollback still apply to the block as a whole.

**This hazard is latent in the shipped scheduler.** Nothing about it requires
seasons -- it needs only a freeze order that leaves such a set last, which the
perturbation ordering can produce on its own. Reversibility is what made it
reachable often enough to see. That is the most useful thing this track produced,
and it is a bug report, not a win for seasons.

## 6. The population's failure mode, measured rather than asserted

Two readings explain why selection is inert here, and both are properties of
these fixtures rather than of the rule.

1. **The front is nearly everything.** With `P = 4` members and two objectives,
   non-domination is cheap. On `mixed-10` the front at seed 0 is `{1,2,3}`,
   `{1,2,3}`, then all four -- one member culled at two of three selection points,
   nothing at the third. Description cost across members of a four-node scaffold
   spans about 20 bits out of 17,750, so `D` almost never breaks a tie that `L`
   has not already decided, and the rule degenerates toward "keep everyone".
2. **On these scaffolds, restarts are worse than training.** `argmax-restarts`
   (best of four 40-episode runs) scores 2.469 where a single run given the same
   160 optimizer steps scores 3.625. Splitting a budget across members costs more
   than the diversity buys, so even a *perfect* selector over four members would
   be starting from a worse place than one longer run.

The rule itself is not refuted by this -- the scaffold where it should pay is the
one in `research/abstraction-preference/RESULTS.md`, where a light program and a
conforming program genuinely differ and a scalar weight has to choose between
them. That experiment is not run here, and section 8 says so.

## 7. The ARCHITECTURE.md section 5 amendment

Applied on this branch: the paragraph beginning "Discrete program selection does
not require one-bit numeric values" in section 5 is replaced by the following.
It is written as a description of what the scheduler *may* do, with the measured
default stated in it, because the measurement says these mechanisms should not be
on by default.

> Commitment need not be monotone. The loop above freezes and never releases: the
> only reversal is the transactional rollback of a *rejected* trial, so an
> accepted freeze is final for the rest of the run. A scheduler may instead run in
> **seasons**, alternating *winters* -- rounds 1-6 above, which freeze and may
> additionally commit dead nodes for free -- with *summers*, which release some
> already-accepted commitments back to trainable, reset those nodes' temperature
> and precision pressure to their warm values, and retrain the whole soft graph.
> The frozen fraction is then not monotone and carries no fixed ceiling: it rises
> through a winter and falls at each summer. A release is the exact inverse of a
> freeze, restoring the choice logit's gradient and the node's relaxed value path;
> it is not an inverse of a *declared* selection, which is a hand-supplied prior
> rather than a commitment the run made, and never thaws.
>
> A seasonal scheduler must state its thaw policy and must be shown to terminate.
> The policy names which commitments are released and on what evidence -- a
> recorded per-node measurement, not a clock. Termination is a property of the
> policy, not of the budget: a release limit per node, a decaying release
> fraction, and marking as settled any node that re-freezes onto the candidate it
> was released from are each sufficient, and an empty release set ends the run. A
> seasonal run always ends in a winter.
>
> Releasing late can strand a residual set in which freezing any single member
> disconnects the others, so the per-node connectivity guard refuses all of them
> and the program cannot close. A block trial -- hardening the whole residual set
> in one transaction -- is the remedy, and is not a weakening of the guard: when
> the block completes the program there is no remaining trainable region for it to
> protect. Degradation tolerance, conformance and rollback apply to the block
> exactly as to a node.
>
> Nothing requires one program per problem. A scheduler may carry a **population**
> of candidate programs over the same scaffold and select among them, provided the
> selection rule is stated. Where the two pressures are performance and program
> size, a non-dominated rule -- keep every candidate that no other beats on both
> the objective and the description cost -- is preferred to a scalarized one,
> because a single description-cost weight is aligned with the target on an
> over-provisioned scaffold and opposed to it on an exactly-sized one, and a
> population does not have to choose between them. A population's compute is the
> sum over its members and must be reported that way; its baseline is best-of-P
> independent restarts at the same total, not a single run.
>
> Seasons, pruning, block closing and populations are all **off by default**, and
> that default is a measurement rather than caution: at equalized optimizer steps
> none of them beat freezing every node at its argmax after training, on either
> shipped fixture at any budget tested. See `research/seasons/RESULTS.md`.
>
> Discrete program selection does not require one-bit numeric values. Fixed
> floating or scaled-integer arithmetic can remain in the export. All learned
> choices/parameters eventually freeze; evolving episode memory and explicit
> runtime randomness remain live. Temperature, thresholds, stability windows,
> block size, precision, season length, thaw policy, population size and selection
> rule are experiment configuration, not settled constants.

## 8. What this does not measure

- **One scaffold shape per fixture.** Both fixtures have small, exactly-sized
  scaffolds. Pruning is provably inert on `mixed` (no node ever dies) and the
  population's two objectives barely separate. The scaffold where the population
  argument was made -- `research/abstraction-preference`'s wide arm B, where a
  3-node abstracted program and a 5-node flat one both conform -- is **not** run
  here. That is the experiment that could still make a population pay, and it is
  the obvious next one.
- **`P = 4`, `G = 4`, one selection rule.** No sweep over population size,
  generation count or reseed noise. A larger population makes the front a
  smaller fraction of the members and might make selection bite.
- **`thaw_limit = 1`.** Each node gets one second chance. Nothing here tests
  whether repeated release helps; the termination argument allows a larger limit.
- **Single-fixture significance.** `joint` returns are 8 seeds with SDs near 1.0,
  so differences within the crystallizing arms (2.281 against 2.594) are not
  separable; the difference that matters -- every crystallizing arm against its
  step-matched argmax -- is large and consistent in sign across both budgets.
- **No enumerative baseline is added here.** `research/enumerative-baseline`
  already settles both fixtures in milliseconds; nothing in this track changes
  that comparison, and it should be read alongside.

## 9. Reproduction

```
python research/seasons/drive.py mixed --seeds 8 --budget 5     # and 10, 30, 100
python research/seasons/drive.py joint --seeds 8 --budget 40    # and 160
python research/seasons/summarize.py mixed research/seasons/results/mixed-5.jsonl
python research/seasons/summarize.py joint research/seasons/results/joint-40.jsonl
```

**Shipped fixtures, verified on this branch.** `python -m tcn train --episodes
160` gives prediction loss 0.248835613951087 to 0.0022308224288281053, 4/4
frozen, evaluation and frozen-agent mean return 4.0. `python -m tcn synthesize
--mode enumerate` gives `space_size` 96, `conforming` 1, `certificate` "unique",
`exact_conformance` true. **295 of 296 tests pass**, including the 8 new ones in
`tests/test_crystallization_seasons.py`.
`tests/test_panel_interface.py::test_panel_episode_replays_and_restores` fails,
and it fails identically on a pristine copy of `0b538ec` with none of this
branch's changes applied -- a pre-existing replay-determinism failure in the
panel generator, unrelated to the scheduler.

**Defaults.** `Crystallizer(seasons=0, prune=False, close_block=False)` is the
constructor default and `run` takes the same path it took before this branch;
`SoftProgram.thaw` has no caller outside seasons.
`tests/test_crystallization_seasons.py::test_seasons_are_off_by_default` asserts it.
