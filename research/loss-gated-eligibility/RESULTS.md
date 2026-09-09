# Loss-gated commitment: the gate does what it was designed to do, and argmax still wins

**Run 2026-09-09 on branch `loss-gated-eligibility`, built on `perturbation-selection`
(commit 90dd08a).** All numbers come from `research/loss-gated-eligibility/results/`.

`research/perturbation-selection/RESULTS.md` section 6 left one diagnosis
untested. The DARTS-PT removal *measurement* is better than argmax where the
learned distribution is under-trained -- it picks the reference truth table in
6/16 under-trained decisions where argmax picks it 0/16 -- but the *scheduler*
built on it loses to plain argmax, because it makes every node eligible
immediately and commits irreversibly on a round-one measurement while argmax
defers every commitment to the end. The entropy rule was wrong about *which*
candidate matters but incidentally right about *when* to commit; removing the
wrong criterion also removed the delay. `research/FINDINGS.md` section 6 raises
the same complaint from the annealing side: the crystallizer's 0.8x-per-round
temperature decay commits on a fixed clock regardless of progress.

This track puts both on the task loss and measures them separately.

---

## Verdict

1. **Loss-gated eligibility does not beat plain argmax at equal compute, on
   either fixture.** This is the third independent track to conclude that the
   crystallization scheduler does not earn its complexity.

2. **It does repair the scheduler, substantially.** On the environment-coupled
   fixture the fully gated arm takes the perturbation scheduler from
   2.531 +- 0.818 to 3.500 +- 0.720 return at 40 training episodes, and from
   selecting the reference program in 1/8 seeds to 5/8; at 80 episodes from
   3.156 +- 0.626 to 3.562 +- 0.609 and from 3/8 to 6/8. Budget-matched argmax
   scores 3.625 (7/8) and 3.844 (8/8) at the *identical* optimizer-step counts and
   **2.4-2.5x fewer environment rollouts**. The gate closes most of the gap it was
   diagnosed to open. It does not cross it.

3. **The gate works mechanically, and that is exactly why it cannot win.** A
   direct probe (section 4) shows that under the incumbent schedule the
   commitment for each real decision is taken while argmax is still uninformative
   -- argmax names the reference table in **0/16** decisions, the perturbation
   measurement in 5/16, reproducing the branch. Under the gate the same
   commitments are taken after the objective has settled, and there argmax names
   it in **11/16** and perturbation in **11/16**: they have converged on each
   other. Waiting until the measurement is trustworthy is waiting until argmax is
   trustworthy too.

4. **The apparent win on the supervised fixture is an accounting artifact, and it
   dies under the second check.** Not annealing at all conforms 48/48 against
   step-matched argmax's 42/48 over the three tight budgets (Fisher exact
   **p = 0.0265**; ~0.16 after correcting for the six arm-versus-argmax
   comparisons this track makes). But that arm spends about 2x argmax's forward
   passes on perturbation scoring, and plain argmax handed those forward passes
   **as optimizer steps** -- strictly generous, since each also carries a backward
   pass -- conforms **32/32**. Nothing beats argmax once both compute axes are
   equalized.

5. **The two changes carry the effect on different fixtures, and the anneal
   "gate" is not a gate.** On `mixed`, `anneal="plateau"` and `anneal="never"` are
   indistinguishable (48/48 and 48/48; step and forward counts equal to within one
   pass) because the gate essentially never opens there: what was measured is the
   *removal* of the temperature schedule, not its deferral. On `joint` it is the
   eligibility gate that carries everything, and removing the anneal entirely on
   top of it changes nothing (3.500 against 3.406 at 40 episodes, within noise).

6. **Gating eligibility while leaving the temperature on the fixed clock is
   actively broken.** The waiting rounds are counted by the anneal, so the
   temperature reaches its 0.05 floor long before the objective settles. At 40
   joint episodes that arm hits the 200-round cap, leaves 2/8 seeds unfrozen, and
   spends 1,639 environment rollouts against the fully gated arm's 655 for a
   *lower* return. The two clocks have to come off together.

---

## 1. The gate rule, stated

Two independent controls are added to `tcn.crystallize.Crystallizer`. Both
default to the fixed clock, so the shipped scheduler is behaviourally unchanged
and branch 90dd08a's numbers still mean what they meant.

### 1.1 The progress signal

Once per round, before anything else happens, the scheduler evaluates the
**unregularized task objective** -- `Objective.task`, the same closure the
connectivity guard probes, not the regularized `total` that residual retraining
descends. Reading `total` would be reading the discreteness term's own progress
towards a one-hot distribution, which is the fixed clock wearing a disguise.

```text
progress <- progress + [ task() ]

plateaued <=> len(progress) >= W + 1
              and (progress[-(W+1)] - min(progress[-W:])) / max(|progress[-(W+1)]|, 1e-12) < delta

open      <=> plateaued  or  rounds_waited >= P
```

The gate opens when the best of the last `W` readings has not improved on the
reading that preceded them by more than a relative `delta`. A *worsening*
objective also opens it: the question is whether more training is still buying
anything, not whether the loss moved. Opening **re-arms** the detector --
`progress` is truncated to its last reading -- so each commitment is measured
against progress made since the previous one rather than against the whole run.

| parameter | symbol | value used for every number below |
|---|---|---|
| `plateau_window` | W | 3 rounds |
| `plateau_tolerance` | delta | 1e-3 relative |
| `plateau_patience` | P | 12 closed rounds |

ARCHITECTURE.md section 5: "Temperature, thresholds, stability windows, block
size, and precision are experiment configuration, not settled constants." These
are experiment configuration. They were fixed once, before any arm was run, and
were **not** tuned per fixture or per budget. `P` is a safety valve rather than
part of the rule: without it an objective that never settles never crystallizes.

### 1.2 Loss-gated eligibility (`eligibility="plateau"`)

* On a **closed** round no node is eligible. The round is spent taking
  `retrain_steps` optimizer steps on `Objective.total` -- the residual retraining
  the scheduler already performs inside a freeze trial, only kept instead of
  rolled back. This is ARCHITECTURE.md section 5 step 1 run to a stopping
  condition instead of once.
* On an **open** round the perturbation sweep runs exactly as before and **at
  most one freeze is accepted**. The gate then closes, and the residual graph
  trains until it settles around the node just frozen before the next node is
  measured.

A deferral does not consume the opening: a rejected trial rolls back exactly, so
the round continues down the candidate list until something is accepted or the
list is exhausted.

### 1.3 Loss-gated annealing (`anneal="plateau"`)

The `max(.05, T * 0.8)` temperature decay and the `+0.1` quantization pressure
advance only on open rounds instead of every round.

`anneal="never"` holds both still for the whole run. It is an ablation, and it is
here because "concentrate later" and "never concentrate" have to be separated
before the gate can be credited with anything -- see section 5.2.

### 1.4 What did not change

Selection, residual retraining, the degradation tolerance, the connectivity
guard, the conformance gate and transactional rollback are untouched. The gate
adds one objective evaluation per round, counted separately as
`gate_evaluations`, and -- when eligibility is gated -- `retrain_steps` optimizer
steps per closed round, counted in the ordinary step total.

Eight tests were added in `tests/test_crystallization_gate.py`: the detector's
arithmetic including the rising-loss and unusable-reading cases, the deferral of
the first freeze with its optimizer-step count, one-acceptance-per-opening plus
re-arming, the held temperature against the fixed clock's `0.8^3`, that the gate
reads `task` while the closed rounds descend `total`, the patience valve on an
objective that never settles, the `never` ablation, and the rejection of unknown
rules. The suite is **104 tests and all pass** (96 on branch 90dd08a plus these eight).

---

## 2. The arms, and how compute was equalized

Every crystallizing arm is the shipped `Crystallizer` with
`selection="perturbation"`. They differ only in what is on a clock.

| arm | eligibility | anneal | round cap |
|---|---|---|---|
| `perturbation` | immediate | round | 24 (shipped) |
| `perturbation-cap` | immediate | round | 200 |
| `gate` | **plateau** | round | 200 |
| `anneal` | immediate | **plateau** | 200 |
| `noanneal` | immediate | **never** | 200 |
| `gate+anneal` | **plateau** | **plateau** | 200 |
| `gate+noanneal` | **plateau** | **never** | 200 |

`perturbation` is the incumbent scheduler exactly as branch 90dd08a ships it, and
every one of its cells below reproduces that branch's reported value to the digit
-- mixed 14/16, 16/16, 16/16 at 80.6, 60.6, 70.0 optimizer steps and 186.8,
113.8, 122.0 forwards for budgets 10/20/30; joint 2.531 +- 0.818 (728.0
rollouts), 3.156 +- 0.626 (282.2), 4.000 +- 0 (280.0) at 40/80/160 episodes.
That is both the harness check and the evidence that the default code path is
behaviourally unchanged: with `eligibility="immediate"` and `anneal="round"` the
gate is never evaluated and `run` takes the same branches it took before.
`perturbation-cap` controls for the raised round cap the gated arms need, so
"more rounds" cannot be mistaken for "the gate".

Compute is counted four ways, all exact, extending track 1's standard:

* **Optimizer steps.** `optimizer.step` is shadowed by a counting wrapper before
  any training begins, so the count includes retraining inside freeze trials --
  discarded ones included -- and the residual training the gate spends on closed
  rounds.
* **Forward passes.** `SoftProgram.forward` is shadowed the same way. This is the
  honest price of perturbation scoring and of the gate's own readings.
* **Objective evaluations.** On `joint` one objective evaluation is **two
  environment rollouts**, so `2 x loss evaluations` is the environment budget and
  is reported directly as "env rollouts".
* **Gate evaluations** separately from **sweep evaluations**, so the gate's own
  price is visible instead of folded into the sweep's.

Procedure per (fixture, budget, seed): run the crystallizing arms, record each
one's realised optimizer-step total, then run plain argmax once per arm, padded
with extra optimizer steps on the same objective the crystallizer retrains on, up
to that arm's exact total. `argmax@X` therefore ends with the **identical**
optimizer-step count as arm X on that seed. `argmax0` is the opposite control:
base steps only, strictly less compute than any arm. Section 5.3 adds a
**forward-matched** argmax, handed a scheduler arm's forward-pass count as
optimizer steps -- strictly generous to argmax, since each of those steps also
carries a backward pass the sweep never paid for.

### Seeding, and what a seed means here

**`SoftProgram` zero-initializes every choice logit, so `torch.manual_seed` does
not perturb synthesis.** The mixed fixture as shipped is bit-for-bit
deterministic -- zero-initialised logits, full-batch training, no sampling
anywhere -- and N seeds of it are one outcome repeated N times. Every mixed run
in section 5 therefore adds seeded Gaussian noise (std 0.5, track 1's
value, the scheme branch 90dd08a used) to the choice logits at initialisation and
changes nothing else. **Section 5.4 reports the shipped zero-noise fixture
separately, as single outcomes rather than statistics.** The joint fixture varies
genuinely across seeds -- generator seed and sampled rollouts -- so its 8 seeds
are real.

---

## 3. Joint prediction and policy, 8 seeds per budget

Deterministic mean return over 16 held-out test episodes (indices 10000-10015);
frozen-program return over 16 further test episodes (30000-30015) through the
exact exported agent. Maximum return 4. "Reference program" counts seeds
selecting `relation = goal_relation = truth table 6`, the program the probe
objective identifies.

**40 training episodes**

| arm | fully frozen | det. mean return | frozen-program return | reference | optimizer steps | forward passes | env rollouts | gate evals | deferrals |
|---|---|---|---|---|---|---|---|---|---|
| `perturbation` (branch 90dd08a) | 7/8 | 2.531 ± 0.818 | 2.607 | 1/8 | 118.8 | 6400 | 728.0 | 0 | 26.9 |
| `gate` | 6/8 | 3.250 ± 0.856 | 3.333 | 5/8 | 297.0 | 13692 | 1639.5 | 96.9 | 44.1 |
| `anneal` | 7/8 | 2.469 ± 0.452 | 2.500 | 1/8 | 282.8 | 18300 | 2215.5 | 32.8 | 108.8 |
| `noanneal` | 5/8 | 2.156 ± 0.566 | 2.650 | 1/8 | 916.0 | 55952 | 6922.0 | 0 | 427.1 |
| **`gate+anneal`** | **8/8** | **3.500 ± 0.720** | 3.531 | **5/8** | 174.0 | 5816 | 655.0 | 64.1 | **2.9** |
| `gate+noanneal` | 8/8 | 3.406 ± 0.834 | 3.344 | 5/8 | 193.2 | 6376 | 725.0 | 72.1 | 4.6 |
| **`argmax@perturbation`** | 8/8 | **3.625 ± 0.707** | 3.625 | **7/8** | 118.8 | 1836 | **157.5** | 0 | 0 |
| **`argmax@gate`** | 8/8 | **3.625 ± 0.707** | 3.625 | **7/8** | 297.0 | 4688 | 514.0 | 0 | 0 |
| **`argmax@gate+anneal`** | 8/8 | **3.625 ± 0.707** | 3.625 | **7/8** | 174.0 | 2720 | **268.0** | 0 | 0 |
| `argmax@gate+noanneal` | 8/8 | 3.625 ± 0.707 | 3.625 | 7/8 | 193.2 | 3028 | 306.5 | 0 | 0 |
| `argmax0` | 8/8 | 2.094 ± 0.499 | 1.938 | 0/8 | 40.0 | 576 | 0 | 0 | 0 |

**80 training episodes**

| arm | fully frozen | det. mean return | frozen-program return | reference | optimizer steps | forward passes | env rollouts | gate evals | deferrals |
|---|---|---|---|---|---|---|---|---|---|
| `perturbation` (branch 90dd08a) | 8/8 | 3.156 ± 0.626 | 3.406 | 3/8 | 114.5 | 3154 | 282.2 | 0 | 4.2 |
| `gate` | 8/8 | 3.594 ± 0.566 | 3.500 | 6/8 | 204.5 | 5864 | 621.0 | 59.9 | 2.4 |
| `anneal` | 8/8 | 3.188 ± 0.691 | 3.469 | 4/8 | 141.2 | 4762 | 483.2 | 3.4 | 17.6 |
| **`gate+anneal`** | 8/8 | **3.562 ± 0.609** | 3.656 | **6/8** | 207.0 | 5928 | 629.0 | 61.0 | 2.5 |
| **`argmax@perturbation`** | 8/8 | **3.844 ± 0.442** | 3.812 | **8/8** | 114.5 | 1448 | **69.0** | 0 | 0 |
| **`argmax@gate`** | 8/8 | **3.844 ± 0.442** | 3.812 | **8/8** | 204.5 | 2888 | 249.0 | 0 | 0 |
| **`argmax@gate+anneal`** | 8/8 | **3.844 ± 0.442** | 3.812 | **8/8** | 207.0 | 2928 | **254.0** | 0 | 0 |
| `argmax0` | 8/8 | 3.594 ± 0.778 | 3.562 | 7/8 | 80.0 | 896 | 0 | 0 | 0 |

The anneal-gate controls at 80 episodes, run separately (same 8 seeds):

| arm | fully frozen | det. mean return | frozen-program return | reference | optimizer steps | forward passes | env rollouts | deferrals |
|---|---|---|---|---|---|---|---|---|
| `noanneal` | 5/8 | 2.906 ± 0.981 | 3.550 | 7/8 | 1602.2 | 88732 | 10979.5 | 751.9 |
| `gate+noanneal` | 8/8 | 3.562 ± 0.609 | 3.656 | 6/8 | 208.8 | 5974 | 634.8 | 2.6 |
| `argmax@gate+noanneal` | 8/8 | 3.844 ± 0.442 | 3.812 | 8/8 | 208.8 | 2956 | 257.5 | 0 |

`noanneal` is the pathological arm on this fixture: without any concentration
pressure the freeze trials degrade the objective and are rolled back 752 times per
seed, so it burns 1,602 optimizer steps and 10,980 environment rollouts to freeze
5/8 seeds. Its 7/8 reference count with only 5/8 frozen says the *soft* graph
still points at the right program; it is the hardening that fails.

**160 training episodes (the shipped configuration)**

| arm | fully frozen | det. mean return | reference | optimizer steps | forward passes | env rollouts | gate evals |
|---|---|---|---|---|---|---|---|
| `perturbation` (branch 90dd08a) | 8/8 | 4.000 ± 0 | 8/8 | 192.0 | 3776 | 280.0 | 0 |
| `gate` | 8/8 | 4.000 ± 0 | 8/8 | 275.0 | 6272 | 592.0 | 55.2 |
| `anneal` | 8/8 | 4.000 ± 0 | 8/8 | 192.0 | 3744 | 276.0 | 2.0 |
| `gate+anneal` | 8/8 | 4.000 ± 0 | 8/8 | 275.0 | 6272 | 592.0 | 55.2 |
| `argmax@perturbation` | 8/8 | 4.000 ± 0 | 8/8 | 192.0 | 2048 | **64.0** | 0 |
| `argmax@gate` | 8/8 | 4.000 ± 0 | 8/8 | 275.0 | 3376 | 230.0 | 0 |
| `argmax0` | 8/8 | 4.000 ± 0 | 8/8 | 160.0 | 1536 | **0** | 0 |

Read three ways.

* **The gate is a large improvement to the scheduler and no improvement over
  argmax.** At 40 episodes it moves the perturbation arm 2.531 -> 3.500 and its
  reference-program count 1/8 -> 5/8, while cutting the rollout bill from 728 to
  655; at 80 episodes, 3.156 -> 3.562 and 3/8 -> 6/8 at 2.2x the rollouts.
  Step-matched argmax is above every scheduler arm at every budget below
  saturation, and does it on 2.4-2.5x fewer environment episodes.
* **The eligibility gate is what carries this fixture.** `anneal` alone is 2.469
  and 3.188 -- no better than the incumbent -- and once eligibility is gated the
  anneal rule barely matters (`gate+anneal` 3.500 against `gate+noanneal` 3.406,
  8 seeds, sd 0.7-0.8).
* **The fixed-clock anneal must come off with it.** `gate` (gated eligibility,
  fixed-clock anneal) is the worst-behaved configuration measured: 6/8 fully
  frozen, 44.1 deferrals per seed, 1,639 rollouts, and it hits the 200-round cap.
  Its temperature reaches the 0.05 floor during the waiting rounds, so by the
  time the objective settles every mixture is already effectively hard and the
  freeze trials thrash. `noanneal` without the gate is worse still -- 5/8 frozen,
  427 deferrals, 6,922 rollouts -- so the anneal is not simply harmful here; it is
  the *coupling to the round clock* that is.
* At the shipped 160-episode budget every arm reaches 4.000 and the only
  difference is cost, reproducing track 1 and the perturbation branch.

The 8-seed return differences at 40 and 80 episodes have sd 0.45-0.86, so the
`gate+anneal`-versus-`argmax` gaps (3.500 against 3.625; 3.562 against 3.844) are
well inside noise on their own; the reference-program counts (5/8 against 7/8,
6/8 against 8/8) are the sharper statement, and they run the same way.

---

## 4. The measurement moves to where argmax is already right

`gate_probe.py` records, at each gate opening and before any freeze is attempted,
what argmax says and what the perturbation sweep says for the two real decision
nodes -- so the two rules are read at *the scheduler's own moment of commitment*
rather than at one fixed instant. 8 seeds x 2 nodes = 16 first commitments per
schedule, 40 training episodes.

| schedule | argmax = table 6 | perturbation = table 6 | the two agree | final program = table 6 |
|---|---|---|---|---|
| immediate / round (branch 90dd08a) | **0/16** | 5/16 | 4/16 | 1/8 |
| plateau / plateau (`gate+anneal`) | **11/16** | 11/16 | 9/16 | 5/8 |

The first row reproduces the perturbation branch's section 6 exactly: under the
incumbent schedule every commitment is taken while the learned distribution is
still uninformative, which is the regime DARTS-PT is about.

The second row is this track's result, and it cuts both ways. The gate does
precisely what it was designed to do -- it moves the commitment out of that
regime, and the reference program follows it from 1/8 to 5/8. But in the regime
it moves the commitment *into*, argmax has caught up: 11/16 against 11/16, and
the two rules now agree on 9 of 16 decisions. **The interval in which the
perturbation measurement is better than argmax is exactly the interval in which
committing is a mistake.** That is the mechanism behind verdict 1, and it is a
statement about the selection rule rather than about these two fixtures.

---

## 5. Mixed synthesis

Exact conformance is the exported program matching every declared probe signal on
all 16 examples to 5e-3.

### 5.1 Per budget, 16 seeds

**10 base training steps**

| arm | exact conformance | fully frozen | optimizer steps | forward passes | sweep evals | gate evals | rounds |
|---|---|---|---|---|---|---|---|
| `perturbation` (branch 90dd08a) | 14/16 | 14/16 | 80.6 | 186.8 | 87.2 | 0 | -- |
| `perturbation-cap` | 14/16 | 14/16 | 300.6 | 802.8 | 439.2 | 0 | -- |
| `gate` | 16/16 | 16/16 | 224.4 | 298.2 | 39.4 | 21.4 | 21.4 |
| **`anneal`** | **16/16** | 16/16 | **53.1** | 112.1 | 44.0 | 1.3 | 1.3 |
| **`noanneal`** | **16/16** | 16/16 | **53.1** | 110.8 | 44.0 | 0 | -- |
| `gate+anneal` | 16/16 | 16/16 | 434.4 | 530.8 | 41.0 | 42.4 | 42.4 |
| `argmax@perturbation` | 13/16 | 16/16 | 80.6 | 81.6 | 0 | 0 | -- |
| `argmax@perturbation-cap` | 13/16 | 16/16 | 300.6 | 301.6 | 0 | 0 | -- |
| `argmax@gate` | 16/16 | 16/16 | 224.4 | 225.4 | 0 | 0 | -- |
| `argmax@anneal` | 12/16 | 16/16 | 53.1 | 54.1 | 0 | 0 | -- |
| `argmax@noanneal` | 12/16 | 16/16 | 53.1 | 54.1 | 0 | 0 | -- |
| `argmax@gate+anneal` | 16/16 | 16/16 | 434.4 | 435.4 | 0 | 0 | -- |
| **`argmax=fwd@noanneal`** | **16/16** | 16/16 | 110.0 | 111.0 | 0 | 0 | -- |
| `argmax0` | 0/16 | 16/16 | 10.0 | 11.0 | 0 | 0 | -- |

**20 base training steps**

| arm | exact conformance | fully frozen | optimizer steps | forward passes | sweep evals | gate evals | rounds |
|---|---|---|---|---|---|---|---|
| `perturbation` (branch 90dd08a) | 16/16 | 16/16 | 60.6 | 113.8 | 40.0 | 0 | -- |
| `perturbation-cap` | 16/16 | 16/16 | 60.6 | 113.8 | 40.0 | 0 | -- |
| `gate` | 16/16 | 16/16 | 230.6 | 303.2 | 38.6 | 21.1 | 21.1 |
| `anneal` | 16/16 | 16/16 | 60.6 | 114.8 | 40.0 | 1.1 | 1.1 |
| `noanneal` | 16/16 | 16/16 | 60.6 | 113.8 | 40.0 | 0 | -- |
| `gate+anneal` | 16/16 | 16/16 | 444.4 | 540.8 | 41.0 | 42.4 | 42.4 |
| `argmax@perturbation` | 14/16 | 16/16 | 60.6 | 61.6 | 0 | 0 | -- |
| `argmax@gate` | 16/16 | 16/16 | 230.6 | 231.6 | 0 | 0 | -- |
| `argmax@anneal` / `argmax@noanneal` | 14/16 | 16/16 | 60.6 | 61.6 | 0 | 0 | -- |
| `argmax@gate+anneal` | 16/16 | 16/16 | 444.4 | 445.4 | 0 | 0 | -- |
| **`argmax=fwd@noanneal`** | **16/16** | 16/16 | 113.0 | 114.0 | 0 | 0 | -- |
| `argmax0` | 0/16 | 16/16 | 20.0 | 21.0 | 0 | 0 | -- |

**30 base training steps** -- every arm 16/16 exact conformance and 16/16 fully
frozen. Costs: `perturbation` 70.0 steps / 122.0 forwards, `anneal` 70.0 / 123.0,
`noanneal` 70.0 / 122.0, `gate` 238.8 / 311.2, `gate+anneal` 454.4 / 550.8, and
each `argmax@X` matches its arm's steps at one extra forward.

**50, 100 and 300 base training steps** -- every arm 16/16 on both counts,
including `argmax0`. The only difference is cost: at 50 steps `perturbation`
spends 90.0 steps / 142.0 forwards against `gate`'s 252.5 / 325.1 and
`gate+anneal`'s 470.6 / 566.7; at 300 steps, 340.0 / 392.0 against 479.4 / 551.3
and 639.4 / 727.3.

### 5.2 Pooled over the three tight budgets (48 runs per arm)

| arm | exact conformance | its step-matched argmax | Fisher exact |
|---|---|---|---|
| `perturbation` (branch 90dd08a) | 46/48 | 43/48 | p = 0.435 |
| `perturbation-cap` | 46/48 | 43/48 | p = 0.435 |
| `gate` | 48/48 | 48/48 | p = 1 |
| `anneal` | 48/48 | 42/48 | **p = 0.0265** |
| `noanneal` | 48/48 | 42/48 | **p = 0.0265** |
| `gate+anneal` | 48/48 | 48/48 | p = 1 |
| `argmax0` | 1/48 | -- | -- |

The `perturbation` row reproduces branch 90dd08a's headline (46/48 against 43/48,
p = 0.44) exactly, which is the harness check.

Two things follow.

* **The anneal gate is not a gate on this fixture.** `anneal` and `noanneal` are
  the same arm to within one forward pass per run: 48/48 each, 53.1 steps each at
  budget 10, gate opening a mean of 1.3 times per run. What the 48/48 measures is
  the *removal* of the temperature schedule, not its deferral. Against the
  incumbent it is 48/48 against 46/48, p = 0.495 -- not a difference on its own.
* **The eligibility gate costs 3-8x the optimizer steps and buys nothing argmax
  cannot get for the same steps.** `gate` and `gate+anneal` reach 48/48, and so
  does argmax at their step counts.

### 5.3 The second check, which kills the one significant cell

`anneal`/`noanneal` reach 48/48 at 53.1 optimizer steps where argmax at 53.1
steps reaches 42/48. They also spend **110.8 forward passes against argmax's
54.1**: the perturbation sweep is bought in forwards. Handing plain argmax that
forward budget as extra optimizer steps -- generous, because each of those steps
also carries a backward pass the sweep never paid for -- gives
`argmax=fwd@noanneal`, which conforms **32/32** across budgets 10 and 20 (16/16
at each) at 110 and 113 steps.

So the only nominally significant cell in this track disappears when the second
compute axis is equalized. There is no budget at which a crystallizing arm
conforms and forward-matched argmax does not.

### 5.4 The shipped deterministic fixture

No initialization noise, one run per cell -- **single outcomes, not statistics.**

| base steps | arm | conforms | frozen | optimizer steps | forward passes |
|---|---|---|---|---|---|
| 20 | `perturbation` | yes | 100% | 60 | 112 |
| 20 | `gate` | yes | 100% | 230 | 305 |
| 20 | `anneal` | yes | 100% | 60 | 113 |
| 20 | `gate+anneal` | yes | 100% | 450 | 547 |
| 20 | `argmax@perturbation` | yes | 100% | 60 | 61 |
| 20 | `argmax0` | no | 100% | 20 | 21 |
| 30 | `perturbation` | yes | 100% | 70 | 122 |
| 30 | `gate` | yes | 100% | 240 | 315 |
| 30 | `anneal` | yes | 100% | 70 | 123 |
| 30 | `gate+anneal` | yes | 100% | 460 | 557 |
| 30 | `argmax@perturbation` | yes | 100% | 70 | 71 |
| 30 | `argmax0` | no | 100% | 30 | 31 |
| 50 | every arm including `argmax0` | yes | 100% | -- | -- |

The gate changes no outcome on the shipped fixture at any budget; it changes only
the price, by 4-7x in optimizer steps.

---

## 6. What this leaves

* **On the scheduler.** Track 1 measured it contributing nothing at working
  budgets and hurting at tight ones. The perturbation branch fixed the selection
  rule and reached parity with argmax on `mixed` while still losing on `joint`.
  This track fixes the commitment schedule, recovers most of the `joint` loss, and
  still does not beat argmax at equal compute on either fixture. Three tracks, one
  answer.
* **On the selection rule itself.** Section 4 is the more general finding, and it
  is not about these two fixtures: the window in which a perturbation measurement
  beats a learned distribution's argmax is the window in which the distribution
  has not converged, and committing there is what the gate exists to prevent.
  Any scheduler that waits for a trustworthy measurement is waiting for argmax to
  become trustworthy. For DARTS-PT-style selection to pay, the commitment has to
  be *reversible* -- so that an early, better-informed choice can be revised --
  and every freeze here is irreversible by construction.
* **On the fixed clock.** The temperature anneal is coupled to the round counter,
  and the moment rounds stop being units of training progress that coupling breaks
  the scheduler outright (section 3, arm `gate`). Whatever else is decided, a
  scheduler that can wait must not anneal while it waits.

---

## 7. Limitations

* Two fixtures, both tiny: 4 nodes with 3 real choices, and 13 nodes with 2 real
  choices. A negative result on a 4-node graph is weak evidence about a 400-node
  graph. What generalises better is section 4, which is about what a measurement
  can and cannot see.
* The joint return differences are 8 seeds with sd 0.45-0.86 and no significance
  test is claimed for them; the reference-program counts are the sharper
  statement and agree in direction.
* The p = 0.0265 cell in section 5.2 is uncorrected. Six arm-versus-argmax
  comparisons are made per pooled table; the Bonferroni-corrected value is ~0.16.
  It is reported because section 5.3 refutes it on its own terms, not because it
  survives correction.
* The gate's three parameters were fixed before any arm ran and never tuned. A
  tuned window might do better; a tuned window measured on the same runs would
  not mean anything.
* `noanneal` and `gate+noanneal` were measured on `joint` at 40 and 80 episodes
  and on `mixed` with initialization noise only, not on the shipped deterministic
  mixed fixture and not at 160 joint episodes.
* Wall-clock is used for no argument here: this is a shared 20-core machine that
  was running at a load average near 80 throughout, with several other agents'
  jobs on it.
* Only the removal variant of perturbation scoring is used, as on the branch.

---

## 8. Reproduction

```bash
cd <worktree>
V=<repo>/.venv/bin/python

# mixed: 16 seeds x 11 arms at six base-training budgets
for s in 10 20 30 50 100 300; do
  $V research/loss-gated-eligibility/drive.py mixed --seeds 16 --budget $s --workers 3
done

# the anneal-gate control (immediate eligibility, no anneal at all)
for s in 10 20 30 50; do
  $V research/loss-gated-eligibility/drive.py mixed --seeds 16 --budget $s --workers 2 \
      --arms noanneal --padded noanneal --no-argmax0 \
      --out research/loss-gated-eligibility/results/mixed-$s-noanneal.jsonl
done

# the shipped deterministic fixture, one run per cell
for s in 20 30 50; do
  $V research/loss-gated-eligibility/drive.py mixed --seeds 1 --budget $s --noise 0 --workers 1 \
      --out research/loss-gated-eligibility/results/mixed-shipped-$s.jsonl
done

# plain argmax handed the scheduler's FORWARD budget as optimizer steps
$V research/loss-gated-eligibility/forward_matched.py --budget 10 --extra 100 \
    --label "argmax=fwd@noanneal" --out research/loss-gated-eligibility/results/mixed-10-fwdmatched.jsonl
$V research/loss-gated-eligibility/forward_matched.py --budget 20 --extra 93 \
    --label "argmax=fwd@noanneal" --out research/loss-gated-eligibility/results/mixed-20-fwdmatched.jsonl

# joint: 8 seeds x 8 arms at three budgets (hours -- the gated arms are paid in rollouts)
for e in 40 80 160; do
  $V research/loss-gated-eligibility/drive.py joint --seeds 8 --budget $e --workers 4
done
for e in 40 80; do
  $V research/loss-gated-eligibility/drive.py joint --seeds 8 --budget $e --workers 2 \
      --arms gate+noanneal,noanneal --padded gate+noanneal --no-argmax0 \
      --out research/loss-gated-eligibility/results/joint-$e-noanneal.jsonl
done

# what each rule says at the scheduler's own moment of commitment
$V research/loss-gated-eligibility/gate_probe.py --episodes 40 --seeds 8

# tables
$V research/loss-gated-eligibility/summarize.py --pool --merge \
    research/loss-gated-eligibility/results/mixed-{10,20,30}*.jsonl

# the gate's unit cases
$V -m pytest tests/test_crystallization_gate.py -q
```

| path | role |
|---|---|
| `arms.py` | arm construction, the gate's parameters, and the four compute counters |
| `run_mixed.py` | mixed fixture; a local copy of `tcn.synthesis.fit` with instrumentation |
| `run_joint.py` | joint fixture; mirrors the crystallization block of `tcn.cli.joint` |
| `drive.py` | per-run subprocesses, two-phase budget equalization |
| `forward_matched.py` | plain argmax matched to a scheduler arm's forward passes |
| `gate_probe.py` | what argmax and the sweep say at each gate opening |
| `summarize.py` | tables and Fisher exact tests |
| `results/*.jsonl` | one JSON record per (arm, seed), including every `FreezeEvent` and the full gate log |
| `results/gate-probe.json` | the probe's raw output |
