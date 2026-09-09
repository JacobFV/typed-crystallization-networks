# Perturbation-based selection, and a connectivity guard that works

**Run 2026-09-08. Two core changes to `tcn/`, each measured against the
implementation it replaces at equalized compute.** All numbers come from
`research/perturbation-selection/results/`.

Track 1 (`research/crystallization-ablation/RESULTS.md`) measured the
crystallization scheduler contributing nothing at working budgets and hurting at
tight ones, and diagnosed two causes. The literature track named the first:
DARTS-PT (Wang et al., ICLR 2021, *Rethinking Architecture Selection in
Differentiable NAS*) showed that the magnitude of a learned architecture
distribution does not indicate an operation's contribution, and
`Crystallizer.candidates()` selected on softmax entropy plus argmax stability --
exactly that quantity. Track 1 named the second: `grad is None` tests
reachability in the autograd graph rather than the presence of learning signal,
so a discreteness regularizer keeps a severed interior reachable and the guard
passes it.

Both are now fixed in `tcn/crystallize.py`.

---

## Verdict

1. **Perturbation selection beats the rule it replaces, decisively and cheaply.**
   On the mixed fixture, pooled over the three tight budgets (10, 20, 30 base
   steps, 16 seeds each): **46/48 exact conformance against 14/48**, Fisher exact
   **p = 3.5e-12**, using **fewer optimizer steps and fewer total forward passes**
   than the rule it beats. At converged budgets (100, 300) they tie at 16/16 and
   the new rule costs 39 extra forward passes per run.

2. **It does not beat plain argmax at equal compute.** Against budget-matched
   argmax over those same three budgets it is 46/48 against 43/48, **p = 0.44** --
   indistinguishable. Track 1's central verdict survives the corrected rule. What
   changed is that the scheduler no longer *loses* to the trivial alternative; it
   still does not win.

3. **On the environment-coupled fixture it is expensive and slightly worse.** At
   80 training episodes the perturbation arm returns 3.156 ± 0.626 against
   argmax's and the entropy rule's 3.844 ± 0.442, having spent 282 validation
   rollouts against argmax's 69. At 160 episodes every arm reaches 4.0.

4. **The measurement is better than argmax; the schedule around it is not.** A
   direct probe (§6) shows the perturbation score picks the reference truth table
   in 6/16 under-trained decisions where argmax picks it in **0/16**. The
   scheduler nevertheless does worse, because it commits irreversibly on a
   round-one measurement while training is still moving, whereas argmax defers
   every commitment to the end. This is the sharpest finding here and it points at
   the *commitment schedule*, not the selection rule, as what is left to fix.

5. **The connectivity guard now detects what ARCHITECTURE.md section 5 says it
   detects.** With the task objective separated from its regularizers, six of the
   thirteen joint nodes are correctly reported as severing an interior region when
   frozen. The shipped guard missed three of those six entirely, and on the other
   three it named only the trainable constants -- **never a severed choice logit,
   for any node**, because the discreteness term keeps every one of them reachable.
   The new guard does **not** fire on a merely concentrated choice, which is what
   sank the previously reverted attempt, and crystallization still completes: 8/8
   fully frozen, 4.0 return.

---

## 1. What changed in `tcn/`

### 1.1 Selection (`tcn/crystallize.py`)

`Crystallizer.candidates()` no longer gates readiness on entropy and stability.
Every unfrozen node is scored by measurement:

```text
score(node, i) = objective with candidate i removed from node's mixture
selected(node) = argmax_i score(node, i)          # removal hurts most => matters most
order(node)    = best score - second-best score   # how decisive the measurement is
```

This is the literal DARTS-PT removal rule -- track 1's arm FR, the variant that
tied the shipped scheduler on `joint` and beat it at mixed-30. Track 1's *forcing*
variant (arm F), which scores a candidate by the objective under forced hard
execution, is deliberately **not** what shipped: it loses badly on `joint`
(2.47 against 4.0) because freezing detaches gradients, so forcing one coupled
node while its partner is still a soft mixture measures the wrong thing.

Details that matter:

* A removal that leaves the relaxation invalid (empty numeric domain, nonfinite
  value) scores as uninformative rather than as maximally important; a node all of
  whose removals are invalid is skipped for that round.
* A single-candidate node carries no architecture decision and is ordered
  **last**: hardening plumbing early detaches gradient paths before any real
  choice has been measured. Track 1 measured freeze ordering as inert on these two
  fixtures, so this is the defensible default rather than a measured improvement.
  It does have one measured consequence, in §7.4.
* Scores are re-measured for a node only after some freeze has been **accepted**.
  A rolled-back trial restores weights and optimizer state exactly, so its scores
  are still valid. The shipped loop recomputed the whole readiness list once per
  node; doing that with perturbation scoring would multiply the sweep cost by the
  node count for no information.
* Residual retraining, the degradation tolerance, the connectivity guard, the
  conformance gate and transactional rollback are untouched and apply to every
  trial exactly as before.

### 1.2 The objective interface

```python
@dataclass(frozen=True)
class Objective:
    total: Callable[[], Tensor]   # regularized: retraining descends it, degradation measured on it
    task:  Callable[[], Tensor]   # unregularized: the connectivity guard probes it
```

`Crystallizer.run` and `Crystallizer.try_freeze` accept either an `Objective` or a
bare callable (`Objective.of` wraps the latter with `task = total`; that caller is
then asserting the objective carries no architecture regularizer, and the guard is
only as strong as the assertion).

`JointTrainer.episode` gained `regularized=True`. With `regularized=False` it drops
`mdl_weight * complexity()` and `crystal_weight * entropy()` -- the two terms that
attach to the choice logits *directly* rather than through the graph. The policy
entropy bonus is retained, because it flows through the program's own `policy`
output and so is a genuine gradient path through the graph.

`tcn/cli.py:joint` now passes `Objective(validation, lambda: validation(False))`.
`tcn/synthesis.py:fit` passes `Objective(loss_fn, loss_fn)` explicitly: its
supervised objective genuinely carries no regularizer (the entropy term is added
in the training loop, not during residual retraining), and stating that at the call
site keeps the guard's contract visible instead of resting on a default.

**The guard change costs nothing.** The probe was already one objective evaluation
per accepted trial; it is now an evaluation of `task` instead of `total`.

### 1.3 Interface changes, stated explicitly

| symbol | change | callers updated |
|---|---|---|
| `Crystallizer.__init__` | new keyword `selection="perturbation"`; `"entropy"` restores the superseded rule for ablation | `tcn/synthesis.py`, `tcn/cli.py`, `research/crystallization-ablation/arms.py` |
| `Crystallizer.candidates` | optional `loss` argument; perturbation ordering when an objective is available, entropy ordering otherwise | internal |
| `Crystallizer.try_freeze` | new trailing keyword `index=None` (the perturbation-selected candidate; falls back to argmax) | internal |
| `Crystallizer.run` / `.try_freeze` | accept an `Objective` as well as a callable | `tcn/synthesis.py`, `tcn/cli.py` |
| `Crystallizer.distances`, `.entropy_candidates`, `.measure`, `.perturbation_scores`, `.score_node` | the old `candidates()` body split out, plus the new scoring helpers | -- |
| `JointTrainer.episode` | new trailing keyword `regularized=True` | `tcn/cli.py` |
| `tcn/synthesis.py`, `tcn/cli.py` | stopped passing `entropy_limit=.9`, which no longer applies to the default rule | -- |

`research/crystallization-ablation/arms.py` now passes `selection="entropy"`
explicitly for every arm, so track 1's numbers still mean what they meant when they
were produced.

`docs/IMPLEMENTATION.md` and `docs/VALIDATION.md` were corrected to match.

Five tests were added in `tests/test_crystallization_selection.py`; the suite is
**96 tests and all pass**.

---

## 2. How compute was equalized

Track 1's standard, extended with a second counter.

* **Optimizer steps.** `optimizer.step` is shadowed by a counting wrapper before
  any training begins, so the count includes retraining inside freeze trials --
  including trials later discarded by rollback.
* **Forward passes.** `SoftProgram.forward` is shadowed the same way. This is the
  honest price of perturbation scoring: the sweep buys its selection with forward
  passes and no gradients, and a rule that wins only by spending more is not a win.
* **Objective evaluations** are reported separately. On `joint` one objective
  evaluation is **two environment rollouts**, so the environment budget is
  `2 x loss evaluations` -- reported in §5.

Procedure per (fixture, budget, seed): run the crystallizing arms and record each
one's realised optimizer-step total; then run plain argmax twice, padded with extra
optimizer steps on the same objective the crystallizer retrains on, once to each
scheduler arm's total. `argmax@perturbation` therefore ends with the *identical*
optimizer-step count as the perturbation arm on that seed. `argmax0` is the
opposite-direction control: base steps only, strictly less compute than any arm.

This is generous to argmax in one direction and to the schedulers in the other: the
schedulers' rolled-back steps are real FLOPs whose effect is discarded, while every
padded argmax step counts. Both readings are reported.

**Seeding the mixed fixture.** As shipped it is bit-for-bit deterministic --
zero-initialised logits, full-batch training, no sampling -- so N seeds would be N
identical runs. Every mixed run in §3 therefore adds seeded Gaussian noise
(std 0.5, track 1's value) to the choice logits at initialisation and changes
nothing else. **§4 reports the shipped zero-noise fixture separately, as single
outcomes rather than statistics.** The joint fixture varies genuinely across seeds
(generator seed and sampled rollouts), so its 8 seeds are real.

---

## 3. Mixed synthesis, 16 seeds per budget

Exact conformance is the exported program matching every declared probe signal on
all 16 examples to 5e-3.

**10 base training steps**

| arm | exact conformance | fully frozen | median loss | frozen nodes | optimizer steps | forward passes | sweep evals |
|---|---|---|---|---|---|---|---|
| entropy+stability (shipped rule) | 4/16 | 4/16 | 0.102 | 0.812 | 170.6 | 218.2 | 0 |
| **perturbation (new default)** | **14/16** | 14/16 | 1.01e-06 | 0.969 | **80.6** | **186.8** | 87.2 |
| argmax @ entropy's steps | 14/16 | 16/16 | 1.01e-06 | 1 | 170.6 | 171.6 | 0 |
| argmax @ perturbation's steps | 13/16 | 16/16 | 1.01e-06 | 1 | 80.6 | 81.6 | 0 |
| argmax0 (less compute than any arm) | 0/16 | 16/16 | 4.23 | 1 | 10.0 | 11.0 | 0 |

**20 base training steps**

| arm | exact conformance | fully frozen | median loss | frozen nodes | optimizer steps | forward passes | sweep evals |
|---|---|---|---|---|---|---|---|
| entropy+stability | 4/16 | 4/16 | 0.102 | 0.812 | 185.6 | 236.3 | 0 |
| **perturbation** | **16/16** | 16/16 | 1.01e-06 | 1 | **60.6** | **113.8** | 40.0 |
| argmax @ entropy's steps | 15/16 | 16/16 | 1.01e-06 | 1 | 185.6 | 186.6 | 0 |
| argmax @ perturbation's steps | 14/16 | 16/16 | 1.01e-06 | 1 | 60.6 | 61.6 | 0 |
| argmax0 | 0/16 | 16/16 | 4.15 | 1 | 20.0 | 21.0 | 0 |

**30 base training steps**

| arm | exact conformance | fully frozen | median loss | frozen nodes | optimizer steps | forward passes | sweep evals |
|---|---|---|---|---|---|---|---|
| entropy+stability | 6/16 | 6/16 | 0.102 | 0.844 | 183.1 | 230.0 | 0 |
| **perturbation** | **16/16** | 16/16 | 1.01e-06 | 1 | **70.0** | **122.0** | 39.0 |
| argmax @ entropy's steps | 16/16 | 16/16 | 1.01e-06 | 1 | 183.1 | 184.1 | 0 |
| argmax @ perturbation's steps | 16/16 | 16/16 | 1.01e-06 | 1 | 70.0 | 71.0 | 0 |
| argmax0 | 1/16 | 16/16 | 4.15 | 1 | 30.0 | 31.0 | 0 |

**50 base training steps**

| arm | exact conformance | fully frozen | median loss | frozen nodes | optimizer steps | forward passes | sweep evals |
|---|---|---|---|---|---|---|---|
| entropy+stability | 14/16 | 14/16 | 1.01e-06 | 0.969 | 115.0 | 135.5 | 0 |
| **perturbation** | **16/16** | 16/16 | 1.01e-06 | 1 | 90.0 | 142.0 | 39.0 |
| argmax @ entropy's steps | 16/16 | 16/16 | 1.01e-06 | 1 | 115.0 | 116.0 | 0 |
| argmax @ perturbation's steps | 16/16 | 16/16 | 1.01e-06 | 1 | 90.0 | 91.0 | 0 |
| argmax0 | 11/16 | 16/16 | 1.01e-06 | 1 | 50.0 | 51.0 | 0 |

**100 and 300 base training steps** -- every arm 16/16 exact conformance, 16/16
fully frozen, median loss 1.01e-06, zero rollbacks of any kind. The only difference
is cost: at 100 steps the entropy arm spends 153.0 forward passes, the perturbation
arm 192.0, budget-matched argmax 141.0; at 300 steps, 353.0 / 392.0 / 341.0.

### Significance and cost, pooled over the three tight budgets (48 runs each)

| comparison | conformance | Fisher exact |
|---|---|---|
| perturbation vs entropy+stability | 46/48 vs 14/48 | **p = 3.5e-12** |
| perturbation vs argmax at its own step count | 46/48 vs 43/48 | p = 0.44 (n.s.) |
| entropy+stability vs argmax at its own step count | 14/48 vs 45/48 | p = 3.3e-11 |

**Forward-pass accounting.** Perturbation scoring is not paid for by extra compute
on this fixture -- it *saves* compute, because the entropy rule's freeze trials
thrash:

| base steps | entropy fwd | perturbation fwd | argmax@perturbation fwd | perturbation sweep evals |
|---|---|---|---|---|
| 10 | 218.2 | **186.8** | 81.6 | 87.2 |
| 20 | 236.3 | **113.8** | 61.6 | 40.0 |
| 30 | 230.0 | **122.0** | 71.0 | 39.0 |
| 50 | 135.5 | 142.0 | 91.0 | 39.0 |
| 100 | 153.0 | 192.0 | 141.0 | 39.0 |
| 300 | 353.0 | 392.0 | 341.0 | 39.0 |

The sweep costs 39 forward passes per run on this 4-node graph (16 + 1 + 3 + 2
candidates, one sweep, plus re-scores after acceptances). That is the whole extra
price, and below 50 base steps it is more than repaid by the rollbacks it avoids.
Plain argmax is still the cheapest arm at every budget.

### Freeze-trial reasons

| budget | arm | trials | accepted | degradation | disconnected | runtime conformance |
|---|---|---|---|---|---|---|
| 10 | entropy | 257 | 52 | 25 | 0 | 180 |
| 10 | perturbation | 113 | 62 | 51 | 0 | **0** |
| 20 | entropy | 265 | 52 | 0 | 0 | 213 |
| 20 | perturbation | 65 | 64 | 1 | 0 | **0** |
| 30 | entropy | 245 | 54 | 1 | 0 | 190 |
| 30 | perturbation | 64 | 64 | 0 | 0 | **0** |

`nonfinite loss` and `invalid trial` fired zero times, as in track 1. The
runtime-conformance rejections vanish under perturbation selection because they
only fire once the program is complete (the fix already on main), and the
perturbation rule reaches a complete, correct program instead of a complete,
incorrect one.

---

## 4. Mixed synthesis, the shipped deterministic fixture

No initialization noise, one run per cell -- single outcomes, not statistics.

| base steps | arm | exact conformance | frozen | loss | optimizer steps | forward passes |
|---|---|---|---|---|---|---|
| 20 | entropy | no | 75% | 0.102 | 260 | 333 |
| 20 | **perturbation** | **yes** | **100%** | 1.01e-06 | **60** | **112** |
| 20 | argmax @ entropy's steps | yes | 100% | 1.01e-06 | 260 | 261 |
| 20 | argmax @ perturbation's steps | yes | 100% | 1.01e-06 | 60 | 61 |
| 20 | argmax0 | no | 100% | 0.102 | 20 | 21 |
| 30 | entropy | no | 75% | 0.102 | 270 | 343 |
| 30 | **perturbation** | **yes** | **100%** | 1.01e-06 | **70** | **122** |
| 30 | argmax @ entropy's steps | yes | 100% | 1.01e-06 | 270 | 271 |
| 30 | argmax @ perturbation's steps | yes | 100% | 1.01e-06 | 70 | 71 |
| 30 | argmax0 | no | 100% | 0.102 | 30 | 31 |
| 50, 100, 300 | every arm | yes | 100% | 1.01e-06 | -- | -- |

The 30-step row is directly comparable to `research/FINDINGS.md` section 10, which
recorded the shipped rule after the conformance fix as "0/16 conform, 75% frozen".
Perturbation selection takes that to exact conformance and a fully frozen program
at a quarter of the optimizer steps. Budget-matched argmax also conforms, at fewer
forward passes still.

---

## 5. Joint prediction and policy, 8 seeds per budget

Deterministic mean return over 16 held-out test episodes (indices 10000-10015);
frozen-program return over 16 further test episodes (30000-30015) through the exact
exported agent. Maximum return 4. `perturbation-total-guard` is the perturbation
arm with the guard probing the regularized total, i.e. the pre-change guard
behaviour, and isolates the guard from the selection rule.

**40 training episodes**

| arm | fully frozen | det. mean return | frozen-program return | optimizer steps | forward passes | validation rollouts | reference program |
|---|---|---|---|---|---|---|---|
| entropy+stability | 7/8 | 2.594 ± 0.906 | 2.643 | 146.5 | 4374 | 474.8 | 3/8 |
| perturbation | 7/8 | 2.531 ± 0.818 | 2.607 | 118.8 | 6400 | 728.0 | 1/8 |
| perturbation, total-probe guard | 7/8 | 3.031 ± 0.949 | 3.250 | 120.5 | 5592 | 627.0 | 4/8 |
| **argmax @ entropy's steps** | 8/8 | **3.625 ± 0.707** | 3.625 | 146.5 | 2280 | 213.0 | 7/8 |
| **argmax @ perturbation's steps** | 8/8 | **3.625 ± 0.707** | 3.625 | 118.8 | 1836 | 157.5 | 7/8 |
| argmax0 | 8/8 | 2.094 ± 0.499 | 1.938 | 40.0 | 576 | 0 | 0/8 |

**80 training episodes**

| arm | fully frozen | det. mean return | frozen-program return | optimizer steps | forward passes | validation rollouts | reference program |
|---|---|---|---|---|---|---|---|
| **entropy+stability** | 8/8 | **3.844 ± 0.442** | 3.812 | 121.0 | 2536 | 205.0 | 8/8 |
| perturbation | 8/8 | 3.156 ± 0.626 | 3.406 | 114.5 | 3154 | 282.2 | 3/8 |
| perturbation, total-probe guard | 8/8 | 3.000 ± 0.720 | 3.188 | 109.0 | 2832 | 242.0 | 3/8 |
| **argmax @ entropy's steps** | 8/8 | **3.844 ± 0.442** | 3.812 | 121.0 | 1552 | 82.0 | 8/8 |
| **argmax @ perturbation's steps** | 8/8 | **3.844 ± 0.442** | 3.812 | 114.5 | 1448 | 69.0 | 8/8 |
| argmax0 | 8/8 | 3.594 ± 0.778 | 3.562 | 80.0 | 896 | 0 | 7/8 |

**160 training episodes (the shipped configuration)**

| arm | fully frozen | det. mean return | frozen-program return | optimizer steps | forward passes | validation rollouts | reference program |
|---|---|---|---|---|---|---|---|
| entropy+stability | 8/8 | 4.000 ± 0 | 4.000 | 200.0 | 3136 | 200.0 | 8/8 |
| perturbation | 8/8 | 4.000 ± 0 | 4.000 | 192.0 | 3776 | 280.0 | 8/8 |
| perturbation, total-probe guard | 8/8 | 4.000 ± 0 | 4.000 | 186.0 | 3344 | 226.0 | 8/8 |
| argmax @ entropy's steps | 8/8 | 4.000 ± 0 | 4.000 | 200.0 | 2176 | 80.0 | 8/8 |
| argmax @ perturbation's steps | 8/8 | 4.000 ± 0 | 4.000 | 192.0 | 2048 | 64.0 | 8/8 |
| argmax0 | 8/8 | 4.000 ± 0 | 4.000 | 160.0 | 1536 | 0 | 8/8 |

At the shipped budget everything ties at 4.0, reproducing track 1. Below it, plain
argmax wins and the perturbation arm is the most expensive arm on the axis that
matters most here: environment rollouts. **Perturbation scoring on an
environment-coupled objective costs 3.5-4.4x argmax's rollouts** (280 against 64 at
160 episodes; 728 against 157 at 40).

The "reference program" column counts seeds selecting `relation = goal_relation =
truth table 6`, the program the probe objective identifies. Argmax finds it 7/8 and
8/8; the perturbation *scheduler* finds it 1/8 and 3/8 at 40 and 80 episodes. §6
explains why, and shows the measurement itself is not the problem.

One incidental measurement: the shipped CLI path (`tcn train --episodes 160`)
reproduces 0.248836 -> 0.002231 prediction loss, 4/4 deterministic return, 4/4 from
the exact frozen agent, and a fully frozen program. The disconnection deferrals go
from four to three, which is why `docs/VALIDATION.md` was edited.

---

## 6. The measurement is right; the commitment schedule is wrong

`selection_probe.py` isolates *what the perturbation measurement says* from *when
the scheduler acts on it*. It trains the joint fixture, then -- with nothing frozen
and nothing retrained -- reports for each of the two real decision nodes the argmax
candidate and the perturbation-selected candidate. 8 seeds x 2 nodes = 16 decisions
per budget.

| training episodes | argmax and perturbation agree | argmax picks table 6 | perturbation picks table 6 |
|---|---|---|---|
| 40 | 1/16 | **0/16** | **6/16** |
| 80 | 12/16 | 15/16 | 13/16 |
| 160 | 16/16 | 16/16 | 16/16 |

This is DARTS-PT's claim, reproduced on this repo's own task: **where the learned
distribution is under-trained, its argmax is uninformative (0/16) and the
perturbation measurement is substantially better (6/16).** Where the distribution
has converged, the two agree exactly.

And yet the scheduler built on the better measurement returns *less*. The reason is
visible in the two together: the perturbation rule makes every node eligible
immediately, so it freezes on a round-one measurement and the freeze is
irreversible, while argmax defers every commitment until training has finished. The
entropy+stability rule was wrong about *which* candidate matters but incidentally
right about *when* to commit -- it waited for concentration. Removing the wrong
criterion also removed the delay.

That is the open question this track leaves: **loss-gated or progress-gated
eligibility on top of perturbation scoring**, so the measurement is taken when it is
worth taking. It needs no further core change (`Crystallizer.run` already anneals
per round and already re-scores on demand) and it is the cheapest remaining
experiment. `research/FINDINGS.md` section 6 raises the same point from the
annealing side.

---

## 7. The connectivity guard

### 7.1 Why the previous fix had to be reverted, and why this one does not

At a one-hot distribution the softmax Jacobian is zero, so *every* gradient through
that logit vanishes -- the task's as much as the regularizer's. An all-zero
gradient therefore cannot distinguish "this choice has converged" from "this choice
has been severed", which is why treating all-zero as disconnected blocked the joint
fixture from crystallizing at all (286 deferrals, `fully_frozen` false).

Reachability separates them exactly, **once the regularizer is out of the probe**:

* a severed logit has **no path** to the task objective, and
  `torch.autograd.grad(..., allow_unused=True)` returns `None`;
* a concentrated but connected logit still has a path and returns a tensor, which
  may be exactly zero.

The discreteness regularizer destroys this distinction because it attaches every
unfrozen logit to `total` directly. Probing `task` restores it. No thresholding,
no magnitude test, no new failure mode.

### 7.2 Positive case: measured on the joint fixture

`guard_probe.py` trains `examples/joint.py` (160 episodes, seed 0) and then, for
each node in turn, freezes it and records for every still-trainable parameter
whether the gradient is absent (`None`) or present-but-zero, under both objectives.
Each freeze is rolled back, so every row is measured against the same model.

| node | candidates | reachability(total) -- shipped guard | reachability(task) -- new guard | nonzero(task) -- the reverted rule | severed parameters the shipped guard missed |
|---|---|---|---|---|---|
| bit_0 | 1 | no | no | **yes** | |
| bit_1 | 1 | no | no | **yes** | |
| relation | 16 | no | **yes** | yes | `choices.0` (bit_0), `choices.1` (bit_1) |
| goal_relation | 16 | no | no | **yes** | |
| z | 1 | no | **yes** | yes | `choices.3` (goal_relation) |
| world | 1 | no | no | **yes** | |
| mul0 | 1 | no | no | **yes** | |
| logit0 | 1 | yes | yes | yes | `choices.6` (mul0) |
| mul1 | 1 | no | no | **yes** | |
| logit1 | 1 | yes | yes | yes | `choices.8` (mul1) |
| policy | 1 | yes | yes | yes | `choices.6/.7/.8/.9` (mul0, logit0, mul1, logit1) |
| prediction | 1 | no | **yes** | yes | `choices.5` (world) |
| value | 1 | no | no | **yes** | |

Read three ways:

* **The shipped guard fires on 3 of 13 nodes, and only ever on the trainable
  constants** whose path ran through the frozen node -- track 1's finding, confirmed
  exactly. For choice logits it fires **never**, because
  `crystal_weight * model.entropy()` keeps all of them reachable.
* **The new guard fires on 6 of 13**, adding `relation`, `z` and `prediction`, and
  on the three shared nodes it additionally names the severed *choice logits*. Every
  one of these detections is checkable against the graph: `bit_0`/`bit_1` feed only
  `relation`; `goal_relation` feeds only `z`; `world` feeds only `prediction`;
  `mul0`/`mul1`/`logit0`/`logit1` feed only `policy`. These are exactly the silent
  interior disconnections ARCHITECTURE.md section 5 exists to prevent.
* **The reverted rule fires on 13 of 13** -- every node, including
  `goal_relation`, which is severed by nothing. That is the false-positive mode
  that blocked crystallization.

### 7.3 Negative case: a merely concentrated choice

`test_connectivity_guard_ignores_a_merely_concentrated_choice` builds a two-node
program `hidden -> out`, drives `out`'s distribution to one-hot (logits
`[200., 0.]`), and freezes `hidden`. The test asserts that the task gradient with
respect to `out`'s logits is **present and exactly all-zero** -- so the reverted
rule would have flagged it -- and that the guard does not fire and the freeze is
accepted.

`test_connectivity_guard_catches_a_severed_interior_only_with_the_task_objective`
is the positive unit case: a 4-candidate `hidden` node whose only path to the
objective runs through `out`. Freezing `out` with a bare regularized callable is
**accepted** (the shipped behaviour, and the test asserts the regularizer keeps
`hidden`'s gradient non-`None`); freezing `out` with `Objective(total, task)` is
**rejected with reason `disconnected remaining region`** and rolled back. Same
freeze, same retraining objective, opposite outcome -- the interface change is the
whole difference.

### 7.4 The guard does not block crystallization

| budget | arm | trials | accepted | disconnected deferrals | fully frozen | return |
|---|---|---|---|---|---|---|
| 160 | perturbation, task probe | 128 | 104 | **24** | 8/8 | 4.000 |
| 160 | perturbation, total probe | 104 | 104 | 0 | 8/8 | 4.000 |
| 80 | perturbation, task probe | 138 | 104 | **25** | 8/8 | 3.156 |
| 80 | perturbation, total probe | 116 | 104 | 0 | 8/8 | 3.000 |
| 40 | perturbation, task probe | 315 | 100 | **48** | 7/8 | 2.531 |
| 40 | perturbation, total probe | 322 | 100 | 0 | 7/8 | 3.031 |

Every deferral is a deferral: the node is frozen successfully in a later round, all
104 nodes across 8 seeds end frozen, and the return is unchanged at the shipped
budget. Compare the reverted rule's 286 deferrals against 4 with no exported
program. The guard's cost is 24 extra freeze trials per 8 seeds at 160 episodes.

Under the entropy ordering the corrected guard fires more still (56 deferrals at
160 episodes against track 1's 32 for the same ordering with the total probe),
because the entropy rule can freeze a boundary node before its interior. That
0-versus-24 difference between the two perturbation arms is the clean isolation:
identical ordering, identical selection, only the probe objective differs.

---

## 8. Limitations

* Two fixtures, both tiny: 4 nodes with 3 real choices, and 13 nodes with 2 real
  choices. "Indistinguishable from argmax" on a 4-node graph is weak evidence about
  a 400-node graph. What generalises better is the mechanism-level result in §6 and
  §7, which is about what a measurement can and cannot see.
* Mixed-fixture seeds come from choice-logit initialisation noise, a scheme
  inherited from track 1. §4 reports the shipped deterministic fixture separately
  and it agrees.
* The joint fixture instantiates the same Boolean function every episode
  (`depth 1, table 6, fixed_inputs`); held-out episodes vary the input bits and the
  objective bit only. Its zero variance at 160 episodes partly reflects how easy it
  is.
* The joint return differences at 40 and 80 episodes are 8 seeds with sd 0.4-0.9;
  the 80-episode gap (3.844 against 3.156) is about 1.2 pooled sd and I have not
  run a significance test on it. The reference-program counts (8/8 against 3/8) are
  the sharper statement.
* `perturbation-total-guard` at 40 episodes returns *more* than the task-probe arm
  (3.031 against 2.531). With 8 seeds and one node's difference in freeze order
  that is within noise, and it runs the other way at 80 episodes (3.000 against
  3.156). No claim either way; the guard's job is correctness of the deferral, and
  §7.2 measures that directly.
* Wall-clock was not used for any argument: this is a shared 20-core machine and up
  to 16 runs were concurrent.
* Only the removal variant of perturbation scoring was measured here. Track 1
  measured the forcing variant and it is worse on `joint`; that result is not
  re-derived.

---

## 9. Reproduction

```bash
cd <worktree>

# mixed: 16 seeds x 5 arms at six base-training budgets (~15 min at 8 workers)
for s in 10 20 30 50 100 300; do
  .venv/bin/python research/perturbation-selection/drive.py mixed \
      --seeds 16 --budget $s --workers 8
done

# the shipped deterministic fixture, one run per cell
for s in 20 30 50 100 300; do
  .venv/bin/python research/perturbation-selection/drive.py mixed \
      --seeds 1 --budget $s --noise 0 --workers 1 \
      --out research/perturbation-selection/results/mixed-shipped-$s.jsonl
done

# joint: 8 seeds x 6 arms at three budgets (slow -- hours at 8 workers,
# because perturbation scoring on this fixture is paid in environment rollouts)
for e in 40 80 160; do
  .venv/bin/python research/perturbation-selection/drive.py joint \
      --seeds 8 --budget $e --workers 8
done

# what the connectivity guard sees, parameter by parameter
.venv/bin/python research/perturbation-selection/guard_probe.py --episodes 160 --seed 0

# what the perturbation measurement says, before any freezing
.venv/bin/python research/perturbation-selection/selection_probe.py \
    --episodes 40 80 160 --seeds 8

# tables
.venv/bin/python research/perturbation-selection/summarize.py \
    research/perturbation-selection/results/mixed-{10,20,30,50,100,300}.jsonl \
    research/perturbation-selection/results/joint-{40,80,160}.jsonl

# the guard's unit cases
.venv/bin/python -m pytest tests/test_crystallization_selection.py -q
```

| path | role |
|---|---|
| `arms.py` | arm construction; the optimizer-step and forward-pass counters |
| `run_mixed.py` | mixed fixture; a local copy of `tcn.synthesis.fit` with instrumentation |
| `run_joint.py` | joint fixture; mirrors the crystallization block of `tcn.cli.joint` |
| `drive.py` | per-run subprocesses, two-phase budget equalization |
| `guard_probe.py` | what each guard rule sees, per node and per parameter |
| `selection_probe.py` | the perturbation measurement in isolation from the schedule |
| `summarize.py` | tables and Fisher exact tests |
| `results/*.jsonl` | one JSON record per (arm, seed), including every `FreezeEvent` |
| `results/guard-probe.json`, `results/selection-probe.json` | the two probes' raw output |
