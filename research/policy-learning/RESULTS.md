# Can a policy be learned from reward in this substrate?

Research track: policy learning. Everything below was produced in this workspace
with `.venv/bin/python` (Python 3.13.15, torch 2.14.0+cpu, `torch.set_num_threads(1)`),
against `tcn/` and `generators/` **unmodified** — nothing outside
`research/policy-learning/` was touched. Scripts and raw JSON are indexed in
`README.md`; every table below is regenerated from `out/*.json` by `aggregate.py`.

Environment episodes are counted by `pl.Counter`, which wraps every `Host.create`
and every `Host.step`, so no budget figure here is taken from a configuration
number (F-budget).

---

## Verdict, first

**Yes — a policy can be learned here from reward alone, at horizon 4, and the
premise that it cannot is wrong.** With the hand-supplied policy constants
removed (all four initialised to 0.0) and *all* supervision removed
(`probe_weight = prediction_weight = 0`), plain REINFORCE with the shipped value
baseline reaches **4.00/4 on 8/8 seeds** on 64 held-out episodes, against
always-false 2.13, always-true 1.88, uniform 2.05 and the exact oracle 4.00.

The cost is the budget, and it is smaller than expected: **400 training episodes**
(464 environment episodes including evaluation), against 100 where it is at
chance and 200 where it reaches 7/8. Under the identical estimator, observation
vector and budget, a 32-wide tanh MLP is at chance at **every** budget up to
4,000 episodes. Track 6's MLP result reproduces exactly; its extension to the
typed program does not hold.

Adding dense probe supervision *alongside* reward buys almost nothing:
probe+reward reaches 8/8 at the same 400 episodes as reward alone. What does buy
something is **staging** them: 50 supervised episodes, crystallize, then 50
reward episodes — **100 training episodes, 8/8 seeds at 4.00**, where both
simultaneous arms are at 2.0-2.5.

And the sharpest result is that no policy is needed at all: **25 supervised
episodes crystallize an exactly-correct model, after which 2 reward episodes fix
the one remaining bit and enumeration against that model returns the oracle 4.00
at horizon 4 and 32.00 at horizon 32 — 27 environment episodes in total**,
15x fewer than reward-only REINFORCE's 400 and 6x fewer than the shipped
fixture's 160.

Three corrections to the record follow from this, and one warning:

- **Track 6's "pure REINFORCE fails for the typed program as well as for the
  MLP" was never measured on the typed program.** `research/baselines/joint_baseline.py`
  runs its `reinforce` mode on MLPs only; the TCN side was always run with
  `probe_weight=1`. F-init stands exactly as written — the shipped constants
  *are* the answer — but the inference drawn from it does not. Measured here,
  the typed program learns this task from reward alone at 400 episodes on 8/8
  seeds while a budget-matched 32-wide tanh MLP under the identical estimator and
  observation vector stays at chance (1.98-2.06) at every budget to 4,000
  episodes.
- **The horizon in this fixture is not a credit-assignment horizon.** The
  `logic` generator's state is constant within an episode, no action changes it,
  and `answer` is rewarded on the step it is taken. Horizon 4 is four
  independent repetitions of one contextual bandit over 32 contexts. Nothing in
  this repository, including this track, has measured credit assignment over a
  horizon, and no result on this generator can.
- **Track 2's rejection of episode batching was confounded by the learning
  rate.** At a fixed episode budget `batch=32, lr=.04` fails (2.28/4, 1/8 seeds);
  the same batch at `lr=.1` succeeds 8/8. Batching trades episodes for optimizer
  steps and needs the step size raised to match.
- **Warning, and the mechanism behind most of the difficulty:** a uniform
  mixture over the complete 16-table `truth_*` candidate set is the constant
  0.5 **and its Jacobian is exactly zero**. So at `SoftProgram`'s zero
  initialization no gradient of any kind — reward, probe or otherwise — passes
  *through* such a node to anything below it. `examples/joint.py`'s `p[12]=2.`
  is what breaks that, and its comment says so; what has not been recorded is
  that this is a property of the operator family, not of that fixture.

---

## 0. What the task actually is

Measured with `probe_env.py`, before anything was trained.

| property | measured |
|---|---|
| observation | `bits` (4 bools), `goal` (1 bool); `program` published but not read |
| target | `(bits[0] xor bits[1]) xor goal` — depth 1, `table=6` (XOR), `fixed_inputs` |
| distinct contexts | **32** over 256 training episodes |
| state within an episode | **constant** — `bits`, `goal` and the target never change |
| effect of an action on state | **none** — `advance` writes `state['answer']` and nothing else reads it |
| reward | **per step**, `1.0` iff the answer taken on that step is correct |
| horizon 4 return | 4 identical, independent binary decisions |

So the fixture is a 32-context, 2-action contextual bandit sampled four times per
episode. `answer` and `wait` are both available; the shipped action library
contains only the two `answer` templates.

**Discrete reference** (`space.py`), over the same 256-program choice space
`SoftProgram` relaxes:

| quantity | count |
|---|---|
| programs in the space (`relation` x `goal_relation`) | 256 |
| probe-optimal (`z == target`) | **2** — (6,6) and (9,9) |
| reward-optimal up to the readout sign | **4** — (6,6), (6,9), (9,6), (9,9) |
| programs whose `z` is constant, so no readout can beat chance | 44 |

Reward-only therefore searches a space with **twice as many solutions** as the
probe-supervised objective. Whatever makes it slower, it is not a larger space.

**Against `AGENTS.md`'s hand-initialization rule** (added during this track, in
commit 49838a7, and prompted by the very fault this track re-measures), the three
obligations are discharged as follows. *State it*: section 1 names every
initialization used in every arm — the four policy constants at 0.0, at
`N(0, 0.1)`, or at the shipped `(-2, 2, 1, -1)`, and the choice logits at zero,
at the shipped `p[12]=2.`, or with `N(0, 0.5)` noise. *Ablate it*: every arm is
run from the neutral start beside the initialized one, and both numbers are
reported. *Certify it*: `space.py` exhausts the same 256-program space this
scaffold searches and enumerates its solutions exactly, so the answer is proved
to lie in the space with no initialization at all — the four reward-optimal
programs above are exactly the four the reward-only arm's eight seeds land on
(section 1). By the rule's own standard the shipped `(-2, 2, 1, -1)` is a prior
that shortened the path, not the answer — but only because this track ran the
neutral arm; nothing before it did.

**Trivial references**, on the same 64 held-out episodes (`refs.py`, indices
10000-10063, `split='test'`, objectives alternating):

| horizon | always-false | always-true | uniform | oracle |
|---|---|---|---|---|
| 4 | 2.125 | 1.875 | 2.047 | **4.000** |
| 8 | 4.250 | 3.750 | 3.781 | **8.000** |
| 16 | 8.500 | 7.500 | 7.594 | **16.000** |
| 32 | 17.000 | 15.000 | 16.063 | **32.000** |

Normalised, every reference is 0.47-0.53 and the oracle is 1.00 at every horizon.

## 0b. Harness fidelity

`pl.py` re-implements `JointTrainer.episode` rather than calling it, because
`TrainConfig.__post_init__` rejects `prediction_weight <= 0` and
`policy_weight <= 0` and so cannot express either a reward-only or a
supervision-only arm at all. The re-implementation was checked against the
shipped fixture first (`parity.py`): shipped constants, `probe_weight=1`,
`p[12]=2.`, 160 episodes, seed 0 gives deterministic held-out return **4.00**,
selections **(relation 6, goal_relation 6)**, and the policy constants scaling
from the declared `(-2, 2, 1, -1)` to `(-2.64, 2.64, 1.08, -1.08)` — the same
behaviour `research/baselines` records for the real trainer (`-2.859, 2.864`),
which also rescales rather than reshapes them.

Differences from the stock trainer, all deliberate and all stated: no
crystallizer (so no undisclosed validation rollouts); one forward pass per step
rather than two; and `prediction` (horizon 1) and `probe` (horizon 0) collapsed
into one term, which is exact here because the episode state is constant, so the
two targets are literally the same values.

`tcn/` was modified by other agents during this track (an `exact_tensor`
memoization, `TrainConfig.description_weight`, `Program.pruned`). None of them is
on any path this harness uses except the memoization, which is documented as
bit-identical; `parity.py` was re-run after the change and reproduces.

---

## 1. The clean reproduction

The hand-supplied policy decoder is removed: `w0 = w1 = bias0 = bias1 = 0.0`
(stated, as required — an arm with `N(0, 0.1)` noise on the same four constants
is reported beside it, and a third arm keeps the shipped `p[12]=2.` choice-logit
bias so that exactly one thing changes from the fixture). Probe and prediction
weights are 0. Everything else is `examples/joint.py`'s configuration:
`lr=.04`, `discount=.95`, `value_weight=.5`, `entropy_weight=.01`, horizon 4,
objectives alternating, Adam, grad-clip 5.

2,000 training episodes, 8 seeds, deterministic evaluation on 64 held-out
episodes. `env ep` includes the evaluation rollouts. 2,000 is a deliberately
generous budget for the reproduction; section 3 measures where each arm actually
crosses.

| arm | what is learned from what | eval return | sd | seeds at 4.00 | env ep |
|---|---|---|---|---|---|
| A0 shipped fixture, 160 episodes | 8 bits from probes, readout supplied | **4.000** | 0.000 | 8/8 | 224 |
| A0b shipped fixture, 2000 episodes | same | 4.000 | 0.000 | 8/8 | 2,064 |
| **A1 reward only, constants at 0** | everything from reward | **4.000** | 0.000 | **8/8** | 2,256 |
| A1c reward only, constants 0, `p[12]=2.` | everything from reward | 4.000 | 0.000 | 8/8 | 2,256 |
| A1d reward only, constants 0, choice noise 0.5 | everything from reward | 4.000 | 0.000 | 8/8 | 2,256 |
| A2 reward only, constants ~ N(0,0.1) | everything from reward | 4.000 | 0.000 | 8/8 | 2,256 |
| A3 reward only, shipped constants | 8 bits from reward | 4.000 | 0.000 | 8/8 | 2,256 |
| A4 reward only, shipped constants, constants frozen | 8 bits from reward | 4.000 | 0.000 | 8/8 | 2,160 |
| **A5 probes only, constants at 0, no actor term** | 8 bits from probes, readout untrained | **2.070** | 0.222 | **0/8** | 2,064 |
| A6 probes + reward, constants at 0 | 8 bits probes, readout reward | 4.000 | 0.000 | 8/8 | 2,064 |
| **A7 32-wide tanh MLP, REINFORCE, same observations** | everything from reward | **2.023** | 0.232 | **0/8** | 2,160 |

Two things to read here.

**A5 is the F-init control and it is decisive.** Dense privileged probe
supervision, with the hand-supplied constants removed and no reward term, scores
**2.07/4 — chance** — while selecting the probe-optimal `(6,6)` in 8/8 seeds. The
representation is perfect and the return is at chance, because supervision
never touches the readout. So the shipped 4/4 is *probe supervision plus a
supplied decoder*, exactly as track 6 said; what track 6 did not measure is that
reward can supply that decoder itself.

**A1's selections are the evidence that the reward result is real.** Writing the
program as `(relation, goal_relation)`, the reward-only arm lands over 8 seeds on
`(9,6)` four times, `(6,6)` twice, and `(6,9)` and `(9,9)` once each — all four
of the four reward-optimal programs enumerated in section 0, including the two
that require a *negative* readout and that no supervised objective would pick.
The probe-supervised arms (A0, A5, A6) select `(6,6)` in 8/8, because only the
probe objective distinguishes it.

The control that closes this is A3 and A4, which keep the shipped constants (a
fixed positive readout `+(2z-1)`) and learn only the discrete choice from reward:
those select **only** `(6,6)` and `(9,9)` — 3/8 and 5/8, and 4/8 and 4/8 — the
two sign-positive solutions, never the two that need the sign flipped. The
selection distribution tracks the readout's freedom exactly as the enumeration
predicts.

### How it fails when it fails: the gradient measurements

`grad_snr` computes, over 64 independent single episodes, `SNR = ||mean g|| /
mean ||g||` per weighted term against the trainable parameters, and
`1/SNR^2` = episodes that must be averaged before the mean direction dominates.
`logit_trace` reports policy logit magnitude and max action probability.

| arm | actor SNR at init | episodes to average | actor SNR trained | ‖g_actor‖ trained | max‖logit‖ init → trained | p(max action) trained |
|---|---|---|---|---|---|---|
| A1 reward only | 0.121 | **177** | 0.707 | 0.0072 | 0.00 → 3.01 | 0.997 |
| A2 reward only, noisy constants | 0.143 | 325 | 0.625 | 0.0045 | 0.10 → 3.08 | 0.997 |
| A3 reward only, shipped constants | 0.123 | 157 | 0.860 | 0.0054 | 0.00 → 3.10 | 0.997 |
| **B6 reward only, discrete choices pinned** | **0.405** | **6.3** | 0.739 | 0.0036 | 0.00 → 3.04 | 0.997 |
| B8 reward only, pinned, external linear readout | 0.345 | 8.7 | — | — | 0.00 → — | — |
| A7 MLP REINFORCE | 0.108 | 127 | — | — | 0.12 → — | — |

Track 2 measured actor SNR 0.221 on the arithmetic scaffold and 0.238 on
`examples/joint.py`, and logits saturating to ±2.5 within 30 episodes.
**Both reproduce in direction and roughly in magnitude.** Actor SNR here is
0.12-0.14 with the discrete search live (track 2's 0.238 was measured with the
shipped constants *and* the `p[12]` bias, i.e. an easier starting point), and
trained logits settle at |logit| ~3.0, p(action) 0.997 — saturated, as track 2
described. What is new is the decomposition: **pinning the two discrete choices
raises actor SNR 3.3x and cuts the episodes-to-average 28x, from 177 to 6.3.**
The variance the actor term suffers from is mostly not action-sampling variance;
it is the candidate mixture underneath it moving.

Note also `max|logit| at init = 0.00` for the arm with the *shipped* constants
(A3). At `SoftProgram`'s zero initialisation the truth-table mixture makes
`z = 0.5` exactly (next section), so `logit0 = -2(0.5)+1 = 0` and
`logit1 = 2(0.5)-1 = 0`. The hand-supplied decoder is only correct once `z` is a
hard 0/1; it is a uniform policy at step 0.

---

## 2. Separating the causes

Four candidate causes, one ablation each, plus one measurement that turned out to
matter more than any of them.

### 2a. The truth-table mixture is exactly derivative-blocking (`cancellation.py`)

`relaxed` implements `truth_k(a,b) = Σ_i v_i·((k>>i)&1)` with
`v = ((1-a)(1-b), (1-a)b, a(1-b), ab)`. Averaged over all 16 tables each `v_i`
gets coefficient 1/2, so the uniform mixture is `Σ_i v_i/2 = 1/2` for every input
and its Jacobian is identically zero. Measured:

| (a, b) | uniform mixture value | ∂/∂a | ∂/∂b |
|---|---|---|---|
| (0, 0) | 0.5 | 0.0 | 0.0 |
| (1, 0) | 0.5 | 0.0 | 0.0 |
| (0.3, 0.7) | 0.5 | -3.7e-09 | -3.7e-09 |
| (1, 1) | 0.5 | 0.0 | 0.0 |
| (0.3, 0.7) with `p[12]=2.` | 0.443 | **0.285** | -3.7e-09 |

This is a property of the complete `truth_*` family, not of the fixture. Two
consequences the record does not state: at zero initialisation **nothing below a
full truth-table node receives any gradient at all**, and the fixture's chosen
break, `truth_12`, is the projection onto `a`, so it restores a gradient in `a`
and leaves `b` at zero.

### 2b. Where each term's descent direction points (`e2_direction.py`)

Gradients of the actor and probe terms with respect to the two 16-way choice
logit vectors, accumulated over n episodes and 8 seeds; `rank` is the position of
the better of {6, 9} in the descent direction. Chance for "optimal is top-1" is
2/16 = 0.125; chance mean rank is 4.67.

| init | n | ‖d actor/d relation‖ | actor: optimal rank | ‖d actor/d goal_relation‖ | actor rank | ‖d probe/d relation‖ | probe: top-1 rate |
|---|---|---|---|---|---|---|---|
| constants 0 | 1 | **0.0** | — | **0.0** | — | 6.25e-2 | 0.00 |
| constants 0 | 8 | **0.0** | — | **0.0** | — | 3.38e-2 | **1.00** |
| constants 0 | 64 | **0.0** | — | **0.0** | — | 3.17e-2 | **1.00** |
| constants 0 | 256 | **0.0** | — | **0.0** | — | 3.15e-2 | **1.00** |
| shipped | 1 | 0.0 | — | 8.31e-2 | 7.00 | 6.25e-2 | 0.00 |
| shipped | 8 | 6.9e-11 | 3.38 | 3.45e-2 | 7.00 | 3.38e-2 | **1.00** |
| shipped | 64 | 4.6e-11 | 0.38 | 1.44e-2 | 7.00 | 3.17e-2 | **1.00** |
| shipped | 256 | 4.3e-11 | **0.00** | 7.66e-3 | 7.00 | 3.15e-2 | **1.00** |

Read the first four rows. **With the hand-supplied constants removed, the actor
gradient to both discrete choices is exactly 0.0** — not small, zero — because
`∂logit/∂z = (w0, w1) = (0, 0)`. The reward term literally cannot see the program
at step 0; only `w`, `bias` and the value constant move. It is a saddle, and A1
shows it is escaped (once `w0 ≠ w1`, `z`'s gradient reopens), but that escape is
what the first ~200 of the 400 episodes buy — see section 3's budget curve, where
the same arm is at chance at 100 and at 7/8 by 200.

With the shipped constants the actor gradient to `relation` is **4.3e-11 against
the probe's 3.1e-2 — nine orders of magnitude** — because it has to pass through
`goal_relation`'s uniform mixture, which by 2a has zero Jacobian. Its direction
is nevertheless right: averaged over 256 episodes it ranks a reward-optimal
candidate first in 8/8 seeds. The probe gradient reaches `relation` at 3.1e-2
only because `relation` also feeds `world → prediction`, supervised by
`probes['gate']` — a second, shallower path that reward does not have.

Neither term identifies `goal_relation` at initialisation (rank 6.9-7.0, worse
than the chance 4.67), for the same reason in reverse: its input is the constant
0.5 until `relation` concentrates. Layer order is forced by the relaxation, not
chosen.

### 2c. The four named causes, ablated

2,000 episodes, 8 seeds, reward only, constants at 0 unless stated.

| cause | ablation | eval return | seeds at 4.00 | verdict |
|---|---|---|---|---|
| score-function variance | R12 exact all-actions gradient (privileged: uses the counterfactual reward, zero action-sampling variance) | 4.000 | 8/8 | **not the binding constraint** — no better than REINFORCE at this budget |
| | R13 exact gradient + choices pinned | 4.000 | 8/8 | |
| value baseline | B2 no baseline at all (advantage = return) | 4.000 | 8/8 | **the constant baseline is not the problem** |
| | B3 state-dependent learned head (`a·z + b·world + c·goal + d`) | 4.000 | 8/8 | no measurable gain |
| | B4 advantage normalisation, batch 16 | 4.000 | 8/8 | |
| | B5 state head + advantage normalisation, batch 16 | 3.766 | 7/8 | slightly worse |
| **discrete-choice bottleneck** | **B6 choices pinned to (6,6), readout only** | **4.000** | **8/8** | **this is the cause** — actor SNR 0.405 vs 0.121, and the pinned arm reaches 4.00 8/8 at **100 training episodes**, where the free arm is at chance |
| | B7/P3 pinned, 200 and 100 episodes | 4.000 | 8/8 | |
| relaxed readout as conditioner | B8 pinned choices, external `nn.Linear` on (z, world) | 4.000 | 8/8 | **refuted** — identical to the typed readout |
| | B9 free choices, external `nn.Linear` | 3.859 | 7/8 | marginally *worse* than the typed readout (A1: 8/8) |

The relaxation is not a poor conditioner for the policy readout: replacing the
typed `mul`/`add`/`tuple` readout with a plain linear layer on the same two
features changes nothing when the choices are pinned and is slightly worse when
they are free. The value baseline is not the problem either — removing it
entirely is as good as a state-dependent head. Score-function variance is real
but is not what is binding: an estimator with the action-sampling variance
removed by construction does not converge faster.

What is binding is that reward has to propagate through a candidate mixture whose
Jacobian is zero at initialisation and noisy afterwards. **Pinning the two
choices takes the reward-only budget from 400 training episodes to under 100
(4.00 on 8/8 at 100, section 3's budget table) and raises actor SNR 3.3x.**

---

## 3. Remedies, cheapest first

2,000 episodes, 8 seeds, reward only, constants at 0, unless the row says
otherwise. Baselines beside every number: always-false 2.13, always-true 1.88,
uniform 2.05, oracle 4.00.

| remedy | eval return | sd | seeds at 4.00 | what it buys |
|---|---|---|---|---|
| R0 nothing (plain REINFORCE, constant baseline) | 4.000 | 0.000 | 8/8 | the reference |
| R1 batch 8 | 4.000 | 0.000 | 8/8 | nothing measurable |
| **R2 batch 32 at `lr=.04`** | **2.281** | 0.686 | **1/8** | **harmful** |
| **R3 batch 32 at `lr=.1`** | **4.000** | 0.000 | **8/8** | recovers R2 — the loss was step count, not averaging |
| R4 entropy schedule 0.2 → 0.001 | 4.000 | 0.000 | 8/8 | nothing measurable |
| R5 entropy 0.2 flat | 4.000 | 0.000 | 8/8 | nothing measurable |
| R6 advantage normalisation, batch 32 | 2.008 | 0.233 | 0/8 | **harmful** (inherits R2's step-count loss) |
| R7 state value head + normalisation, batch 32 | 2.008 | 0.233 | 0/8 | **harmful** |
| R8 `lr=.005` | 4.000 | 0.000 | 8/8 | nothing measurable |
| R9 `lr=.2` | 3.906 | 0.248 | 7/8 | mildly harmful |
| R10 8,000 episodes | 4.000 | 0.000 | 8/8 | nothing left to buy |
| R11 everything at once (8,000 ep, batch 32, norm, entropy schedule) | 4.000 | 0.000 | 8/8 | nothing over R0 |
| R12 exact all-actions gradient (privileged) | 4.000 | 0.000 | 8/8 | nothing at this budget |
| **supervised → reward handoff (section 4)** | **4.000** | 0.000 | **8/8 at 100 training episodes** | **4x fewer environment episodes (100 vs 400)** |
| **frozen model + enumeration, no policy (section 5)** | **4.000** | 0.000 | **8/8 at 27 training episodes** | **15x fewer environment episodes** |

**Track 2's batching result, corrected.** Track 2 implemented `TrainConfig.batch`,
measured `batch=8` worse at equal episodes, and reverted it, concluding batching
"starves the readout". The same effect reproduces here (R2: 2.28/4 at batch 32)
and the stated mechanism is right — at fixed episodes, batch b divides the
optimizer steps by b. But it is a step-size question, not a reason to reject
batching: **R3, the same batch 32 with `lr` raised from .04 to .1, is 4.00/4 on
8/8 seeds.** Track 2's arm held `lr` fixed at .04 across batch sizes.

**The budget curve is the whole story.** Four arms, 8 seeds each, varying only
the number of training episodes; `env ep` includes the 64 evaluation rollouts.
Baselines: always-false 2.13, always-true 1.88, uniform 2.05, oracle 4.00.

| training ep | env ep | reward only | probe + reward | reward only, choices pinned | 32-wide MLP REINFORCE |
|---|---|---|---|---|---|
| 100 | 164 | 2.023 (0/8) | 2.547 (2/8) | **4.000 (8/8)** | 2.062 (0/8) |
| 200 | 264 | 3.750 (7/8) | 3.805 (7/8) | 4.000 (8/8) | 2.023 (0/8) |
| **400** | 464 | **4.000 (8/8)** | **4.000 (8/8)** | 4.000 (8/8) | 2.023 (0/8) |
| 800 | 864 | 4.000 (8/8) | 4.000 (8/8) | 4.000 (8/8) | 1.977 (0/8) |
| 1,200 | 1,264 | 4.000 (8/8) | 4.000 (8/8) | 4.000 (8/8) | 1.977 (0/8) |
| 1,600 | 1,664 | 4.000 (8/8) | 4.000 (8/8) | 4.000 (8/8) | 1.977 (0/8) |
| 2,000 | 2,064 | 4.000 (8/8) | 4.000 (8/8) | 4.000 (8/8) | 1.977 (0/8) |
| 4,000 | 4,064 | 4.000 (8/8) | 4.000 (8/8) | 4.000 (8/8) | 1.977 (0/8) |

Three readings.

**Reward-only is solved at 400 training episodes**, not "at chance". The failure
the record describes is real only below ~200.

**Dense probe supervision, added alongside reward, buys almost nothing here.**
The probe+reward column is within seed noise of the reward-only column at every
budget (its only lead is 2.55 vs 2.02 at 100 episodes, 2/8 vs 0/8). The strong
form of "high-dimensional supervision works in this substrate and scalar reward
does not" is not what this measures; what supervision buys is bought by
*sequencing* it, not by adding it (section 4).

**The MLP never learns it**, at any budget, under the same estimator and the same
8-float observation. To make sure that is not under-tuning, a separate grid
(`e8_mlp.py`, 12 configurations x 4 seeds, 2,000 episodes each) gives it width
8 and 32 (162 and 1,410 parameters), `lr` 0.01/0.04/0.1, and both the program's
fixed baseline at batch 1 and a proper batch baseline (advantage normalisation
over 32 episodes):

| width | lr | batch 1, fixed baseline | batch 32 + advantage normalisation |
|---|---|---|---|
| 8 | 0.01 | 2.125 (0/4) | 1.961 (0/4) |
| 8 | 0.04 | 2.125 (0/4) | 2.109 (0/4) |
| 8 | 0.10 | 2.125 (0/4) | 2.000 (0/4) |
| 32 | 0.01 | 2.000 (0/4) | 2.000 (0/4) |
| 32 | 0.04 | 2.000 (0/4) | 2.125 (0/4) |
| 32 | 0.10 | 2.000 (0/4) | 2.172 (0/4) |

Every cell is chance (best constant 2.13). The target is a 3-bit parity, which a
tanh MLP represents easily — track 6's supervised-plus-replay MLP reaches 4.00
with 153 parameters — so this is the reinforcement schedule failing, not the
model class. That is a genuine advantage of the typed candidate program on this
task and it is the one place a matched baseline separates the two: the typed
program's reward-driven content is a 256-way categorical choice plus four
scalars, and categorical evidence accumulates across noisy single episodes where
1,410 continuous weights do not — the same mechanism track 2 identified for the
prediction term, now measured on the reward term.

The same curve with the shipped `p[12]=2.` bias kept (arms R16-R18) reads
1.977 (0/8) at 160, 3.570 (6/8) at 400, 3.766 (7/8) at 800 and 4.000 (8/8) at
2,000 — slightly *worse* than the unbiased arm, because `truth_12` is not one of
the four reward-optimal tables.

---

## 4. Staged world model, then policy — the primary arm

ARCHITECTURE section 8 specifies prediction and reinforcement as **simultaneously
active** during integrated training (`L = λ_pred·L_future_latents + … +
λ_policy·L_RL`), and `TrainConfig.__post_init__` enforces it by rejecting a zero
weight on either. The staged schedule below is therefore a **deviation from the
constitutional specification**, stated as such.

Stage 0: probe supervision only (`w_probe=1`, `w_actor=0`, `w_value=0`).
Stage 1: crystallize both choice nodes at their argmax via `SoftProgram.freeze`
(no internal gradients past that point), reset the optimizer, and train only the
four readout constants from reward, on a disjoint block of episode indices.

| stage-0 episodes | return after stage 0 | selections after stage 0 | stage-1 reward episodes | final eval | seeds at 4.00 | total env ep |
|---|---|---|---|---|---|---|
| 25 | 2.070 (chance) | (6,6) 8/8 | 800 | 4.000 | 8/8 | 1,145 |
| 50 | 2.070 | (6,6) 8/8 | 800 | 4.000 | 8/8 | 1,170 |
| 100 | 2.070 | (6,6) 8/8 | 800 | 4.000 | 8/8 | 1,220 |
| 200 | 2.070 | (6,6) 8/8 | 800 | 4.000 | 8/8 | 1,320 |
| 400 | 2.070 | (6,6) 8/8 | 800 | 4.000 | 8/8 | 1,520 |
| 200, **no freeze** | 2.070 | (6,6) 8/8 | 800 | 4.000 | 8/8 | 1,256 |
| 200, exact all-actions gradient | 2.070 | (6,6) 8/8 | 800 | 4.000 | 8/8 | 1,320 |
| 200, state-dependent value head | 2.070 | (6,6) 8/8 | 800 | 4.000 | 8/8 | 1,320 |
| 200, external linear readout | 2.070 | (6,6) 8/8 | 800 | 4.000 | 8/8 | 1,320 |
| simultaneous probe + reward, 1,000 ep | — | (6,6) 8/8 | — | 4.000 | 8/8 | 1,064 |

**25 supervised episodes are enough to fix the model** — every seed selects the
probe-optimal (6,6) — and the return after stage 0 is exactly chance (2.07),
which is the F-init control again: a perfect representation with an untrained
readout is worth nothing.

How little reward stage 1 needs, with the model frozen:

| stage-0 episodes | stage-1 reward episodes | final eval | seeds at 4.00 | **total training env episodes** |
|---|---|---|---|---|
| 50 | 1 | 1.977 | 0/8 | 51 |
| 50 | 5 | 2.664 | 3/8 | 55 |
| 50 | 10 | 2.914 | 4/8 | 60 |
| 50 | 25 | 2.648 | 3/8 | 75 |
| **50** | **50** | **4.000** | **8/8** | **100** |
| 50 | 100 | 4.000 | 8/8 | 150 |
| 200 | 50 | 3.734 | 7/8 | 250 |
| 200 | 100 | 4.000 | 8/8 | 300 |

**The attribution, stated exactly.** The final program contains 8 bits of
discrete content (two 4-bit selections) plus a readout. Supervision supplies the
8 bits, in 25-50 episodes. Reward supplies the readout, which is functionally
**one bit** — whether `z` or `1-z` is the correct answer, since the four
constants only ever settle into `±(2z-1)` — and that one bit costs 50 episodes of
REINFORCE. So of the 9 bits of learned content, **8 are supervision-driven and 1
is reward-driven**; of the *return*, none is attributable to either alone, since
stage 0 by itself scores 2.07 (chance) and 50 reward episodes by themselves score
2.02 (chance).

**On the section-8 deviation.** At 100 training episodes the staged schedule
scores 4.00 on 8/8 seeds where simultaneous probe+reward scores 2.55 (2/8) and
reward alone 2.02 (0/8). Matching the staged arm's 4.00 on 8/8 costs 400 episodes
either way simultaneously — a **4x** environment saving for the schedule. That
is a real argument for relaxing section 8's simultaneity requirement to
*"prediction and reinforcement are both active over the course of integrated
training, under a declared schedule"*, and for relaxing `TrainConfig`'s `> 0`
checks to `>= 0` with at least one positive — which is also what makes a
supervision-only or reward-only arm expressible at all. Two caveats keep it
honest: it is one fixture, and **freezing is not what bought the saving** (the
no-freeze row is identical at 4.00, 8/8), so the claim is about the schedule, not
about crystallization.

---

## 5. Model-based control with no learned policy — the second primary arm

Stage 0 as above (probe supervision only), then `SoftProgram.freeze` on both
choice nodes and `export()` to an exact discrete program. No policy is trained.
At each step the frozen program supplies `z` exactly; action sequences of length
`plan_horizon` are enumerated against it and the first action of the best
sequence is taken. Reward supplies exactly one thing: the sign of the reward
model `r(a, z) = [a == (z xor sign)]`, identified by running `k` episodes under
each sign and keeping the better, at a cost of `2k` environment episodes.

| stage-0 episodes | model target accuracy on 64 held-out episodes | model gate accuracy | selections |
|---|---|---|---|
| 25 | **1.000** | **1.000** | (6,6) 8/8 |
| 50 | 1.000 | 1.000 | (6,6) 8/8 |
| 100 | 1.000 | 1.000 | (6,6) 8/8 |
| 200 | 1.000 | 1.000 | (6,6) 8/8 |

The model is not approximately right, it is exactly right — it is a discrete
program and its prediction is checked as an integer.

| sign-identification budget | reward env episodes | eval return (horizon 4) | seeds at 4.00 |
|---|---|---|---|
| k = 1 | **2** | **4.000** | **8/8** |
| k = 2 | 4 | 4.000 | 8/8 |
| k = 4 | 8 | 4.000 | 8/8 |
| k = 8 | 16 | 4.000 | 8/8 |

**Total: 25 supervised + 2 reward = 27 environment episodes to the oracle, on
8/8 seeds.** Against 400 for reward-only REINFORCE, 100 for the staged handoff,
160 for the shipped probe-plus-supplied-decoder fixture, and 331 for that
fixture's true budget once the crystallizer's validation rollouts are counted.

Enumeration against the frozen model, at every horizon:

| horizon | plan horizon | sequences enumerated per step | mean return | normalised | oracle | uniform | wall for 16 episodes |
|---|---|---|---|---|---|---|---|
| 4 | 4 | 16 | **4.000** | 1.000 | 4.000 | 2.047 | 0.03-0.09 s |
| 8 | 8 | 256 | **8.000** | 1.000 | 8.000 | 3.781 | 0.07-0.21 s |
| 16 | 16 | 65,536 | **16.000** | 1.000 | 16.000 | 7.594 | 12.7-15.3 s |
| 32 | 12 (capped) | 4,096 | **32.000** | 1.000 | 32.000 | 16.063 | 8.2-10.8 s |

**And the honest reading of that table, which matters more than the numbers.**
Enumeration recovers the oracle at every horizon, but it buys *nothing* over a
one-step greedy decision. Measured directly (`e6b_plan1.py`, 4 seeds, 16 held-out
episodes each), the same frozen model at `plan_horizon` 1, 2 and 4 returns
4.000, 8.000, 16.000 and 32.000 at horizons 4, 8, 16 and 32 — identical at every
plan depth, so 65,536 enumerated sequences per step and 2 buy the same thing.
The reason is section 0: this environment has no dynamics. The frozen model's `z` does not depend on the action, so the value
of a sequence is separable and the arg-max sequence is the greedy action
repeated. `2^h` grows as expected — at h=16 a full-sequence plan already costs
~0.9 s per episode and h=32 has to be capped — but that cost is pure overhead
here.

So arm B's result is real and its generalisation is not tested. What this fixture
supports: **a crystallized program is an exact `(observation) → latent` function,
and once you have it the reward-driven content collapses to a single bit that two
episodes identify — 27 environment episodes to the oracle, against 400 for
REINFORCE and 160 for the shipped fixture.** What it does not support: any claim
about planning depth, because there is nothing to plan through, and no claim
about a *learned transition* model, because `z_next = z` here and the thing
supervision actually fits is a perception model, not a dynamics model.

Testing the model-based route properly needs a generator whose state responds to
actions. `logic` is not one, but several siblings are (section 6).

### Every method at equal environment episodes

Training environment episodes to reach 4.00/4 on 8/8 seeds (evaluation rollouts
excluded, so the columns are comparable), horizon 4, against the same references.

| method | what supplies the answer | training env episodes to 8/8 at 4.00 |
|---|---|---|
| exact oracle (hand-written) | the experimenter | 0 |
| **frozen model + enumeration, no policy** | 25 probe + 2 reward | **27** |
| **staged: probe, crystallize, then reward** | 50 probe + 50 reward | **100** |
| reward only, discrete choices pinned | the experimenter pins (6,6); reward does the rest | 100 |
| shipped `examples/joint.py` | probes + a supplied decoder | 160 (331 through `tcn train`) |
| **reward only, nothing supplied** | reward | **400** |
| probe + reward, simultaneous | both | 400 |
| 32-wide tanh MLP, REINFORCE | reward | **never** (chance at 4,000) |
| always-false / always-true / uniform | — | 2.13 / 1.88 / 2.05, never 4.00 |

---

## 6. Horizon — and why this fixture cannot answer the question

The `logic` generator's `horizon` is configurable, so the sweep is cheap to run
and was run: 4, 8, 16 and 32, 1,000 training episodes (200 + 800 for the staged
arms), 4 seeds, deterministic evaluation on 64 held-out episodes.

Two reward regimes. **Dense** is the generator's own per-step reward (max return
`h`). **Terminal-only** masks every reward but the last inside the harness — the
environment is untouched, only the return aggregation changes — so the max return
is 1.0 and chance is ~0.5, and `h-1` of the `h` log-probabilities in each
gradient are attached to actions that cannot affect it.

| arm | h=4 | h=8 | h=16 | h=32 | env steps at h=32 |
|---|---|---|---|---|---|
| reward only, dense (max `h`) | 4.000 (4/4) | 8.000 (4/4) | 16.000 (4/4) | **32.000 (4/4)** | 34,048 |
| staged, dense (max `h`) | 4.000 (4/4) | 8.000 (4/4) | 16.000 (4/4) | **32.000 (4/4)** | 40,192 |
| reward only, terminal (max 1.0) | 1.000 (4/4) | 1.000 (4/4) | 0.930 (3/4) | **1.000 (4/4)** | 34,048 |
| staged, terminal (max 1.0) | 1.000 (4/4) | 1.000 (4/4) | 1.000 (4/4) | 0.891 (3/4) | 40,192 |
| staged, terminal, batch 32 + adv. norm (max 1.0) | 0.859 (3/4) | 1.000 (4/4) | 1.000 (4/4) | 0.867 (3/4) | 40,192 |
| MPC against the frozen model (section 5) | 4.000 | 8.000 | 16.000 | 32.000 | — |
| uniform reference, normalised | 0.512 | 0.473 | 0.475 | 0.502 | — |
| oracle, normalised | 1.000 | 1.000 | 1.000 | 1.000 | — |

**Credit assignment does not break anywhere between horizon 4 and horizon 32, in
either reward regime — and that is not a positive result.** It is the diagnostic
that the question is not being asked. Section 0 measured why: the episode state
is constant, no action changes it, and the four (or thirty-two) decisions in an
episode are independent draws of the same contextual bandit. Extending the
horizon adds `h-1` irrelevant log-probabilities to each terminal-reward gradient
— pure variance, no depth — and REINFORCE absorbs it. The one non-monotonicity
in the table (0.930 at h=16 and 1.000 at h=32 in the same row) is seed noise
across 4 seeds, not a horizon effect.

The two departures from 4/4 at h=32 (0.891 and 0.867, both 3/4 seeds) are the
only trace of horizon cost anywhere in the sweep, and they are one seed each.

**So the honest answer to "at what horizon does credit assignment break" is:
this repository cannot say, and no experiment on `generators/logic` can.** The
question needs a generator whose state responds to actions and whose reward
depends on more than the last decision. Reading `advance` across
`generators/*/generator.py`, `logic`, `arithmetic`, `relations`, `language` and
`raster_text` are all single-decision-per-step tasks with no action-dependent
state — and those are the five the curriculum's early stages and both flagship
demos use. Five siblings are not:

| generator | state the action changes | reward |
|---|---|---|
| `computer` | `buffer`, key state, file and table contents — all persistent across steps | `goal`: a named path holding the objective's content. At the keyboard interface the only verbs are `type`, `key` and `wait`, so satisfying it takes a *sequence* — characters, then Enter — and the reward is re-evaluated every step against persistent state |
| `control` | full physics state under `torque` | per-step task reward from `physics` |
| `gui` | widget state | `goal`: the target widget being pressed |
| `geometry` | camera pose and object poses | — |
| `world_3d` | full 3-D physics, multi-agent | per-agent rewards |

`computer` at its keyboard interface is the sharpest of these: its reward is
defined on persistent state that a multi-step action sequence has to build, which
is exactly the credit-assignment structure `logic` lacks. (`control` and
`world_3d` have the dynamics but their rewards are dense per-step, so they test
long-horizon control rather than sparse credit assignment.) **That, not policy learning, is the next blocker on the
path to a closed-loop result** — and this track's finding is that when it is
posed, the substrate has two routes to try that are already known to work on the
bandit case: staged supervision-then-reward at a 4x environment saving, and a
crystallized exact model with enumeration at a 15x saving.

---

## 7. Proposed changes to `tcn/`, none applied

Three agents are working concurrently and a branch with core changes is queued,
so nothing outside `research/policy-learning/` was modified. These are the diffs
this track's measurements support, in order of how well they are evidenced.

### 7.1 `TrainConfig` cannot express the schedule that works (blocking)

`prediction_weight <= 0` and `policy_weight <= 0` are both rejected, so neither a
supervision-only stage, nor a reward-only stage, nor the ablations in section 2
can be written against the shipped trainer at all. That is why this track had to
re-implement the episode loop. The measured reason to relax it is section 4: at
100 training episodes the staged schedule scores 4.00 on 8/8 seeds where the
simultaneous schedule scores 2.55 (2/8), and matching 8/8 simultaneously costs
400 episodes.

```diff
--- a/tcn/training.py
+++ b/tcn/training.py
@@ class TrainConfig:
     def __post_init__(self):
         if not self.objectives or not self.action_templates:raise ValueError('goals and actions must be nonempty')
-        if self.prediction_weight<=0 or self.policy_weight<=0:raise ValueError('integrated training requires prediction and policy objectives')
+        # Integrated training must optimize both objectives over the course of a
+        # curriculum stage; it need not weight both on every step. A staged
+        # schedule -- prediction to convergence, then reward on the readout --
+        # reaches the same return on the joint fixture at a quarter of the
+        # environment episodes (research/policy-learning, section 4). Requiring
+        # a positive weight on each *step* also makes a supervision-only or
+        # reward-only ablation inexpressible, so nothing could measure the
+        # simultaneity requirement it enforces.
+        if self.prediction_weight<0 or self.policy_weight<0:raise ValueError('objective weights must be nonnegative')
+        if self.prediction_weight<=0 and self.policy_weight<=0:raise ValueError('integrated training requires at least one active objective')
```

This is a constitutional question as well as an interface one: ARCHITECTURE
section 8 says the two are simultaneously active. The measurement argues for
"both active across a declared schedule" rather than "both active on every
step", and section 8 should be revised to say which it means.

### 7.2 `examples/joint.py` should not ship a supplied decoder (evidenced)

F-init stands. The fix that this track measured is not "delete the constants and
train longer" (that needs 400 episodes instead of 160) but "delete the constants
and stage the schedule" (100 training episodes, 8/8 seeds at 4.00) — or drop the
learned policy entirely and enumerate against the frozen model (27 episodes,
section 5).
Because 7.1 blocks the staged schedule in the shipped trainer, this diff is
contingent on it, and is written as the honest-initialisation half only:

```diff
--- a/examples/joint.py
+++ b/examples/joint.py
-    constants=(('w0',Value.of(F,-2.)),('w1',Value.of(F,2.)),('bias0',Value.of(F,1.)),('bias1',Value.of(F,-1.)),('baseline',Value.of(F,1.)))
+    # The decoder is learned, not supplied. `(-2, 2, 1, -1)` implements
+    # `argmax(logits) == goal_relation` before any training, which is the whole
+    # decision rule; with it in place the run learns 8 bits of truth-table
+    # selection and nothing else. Zero-initialised, the same scaffold reaches
+    # 4.00/4 on 8/8 seeds from reward alone in ~400 episodes, or in 100 under
+    # a staged probe-then-reward schedule (research/policy-learning).
+    constants=(('w0',Value.of(F,0.)),('w1',Value.of(F,0.)),('bias0',Value.of(F,0.)),('bias1',Value.of(F,0.)),('baseline',Value.of(F,1.)))
```

**Do not apply this one on its own.** At the shipped 160-episode budget it takes
the fixture from 4.00 to chance, because probe supervision alone never touches
the readout (arm A5: 2.07/4 with a perfect representation). It is only correct
together with either 7.1 plus a staged schedule, or a raised episode budget.

### 7.3 Batching should be reconsidered with a matched step size (evidenced)

Track 2 implemented `TrainConfig.batch`, measured it worse at equal episodes and
reverted it. The effect reproduces (batch 32 at `lr=.04`: 2.28/4, 1/8 seeds) but
the cause is step count, not averaging: the same batch at `lr=.1` is 4.00/4 on
8/8 seeds. If the batch diff is revisited, the arm to run is `(batch, lr)`
jointly, not `batch` alone.

### 7.4 The uniform truth-table mixture should be documented, or broken by default (weaker)

Section 2a is a property of the `truth_*` family: at `SoftProgram`'s zero
initialisation the mixture is the constant 0.5 with an exactly zero Jacobian, so
no gradient reaches anything below such a node. Together with P2 (zero-init means
`torch.manual_seed` does not vary synthesis) this makes a whole class of scaffold
silently un-trainable at step 0, and the only reason the shipped fixture works is
a hand-written `p[12]=2.` whose necessity is stated in a comment in one example.
The two defensible options are to document it in `docs/IMPLEMENTATION.md` beside
P2, or to give `SoftProgram` an explicit, declared, seeded initialisation noise
so that "zero-initialised" is a choice rather than an accident. This track does
not have the evidence to pick, and a change to `SoftProgram.__init__` would
alter every recorded synthesis number in the repository, so it is raised rather
than proposed.

---

## 8. Limitations

- **One fixture, one generator, one task family.** Everything is `logic` at
  `depth=1, table=6, fixed_inputs`, which section 0 shows is a 32-context,
  2-action contextual bandit. Nothing here transfers to a task with dynamics
  without being re-measured.
- **The horizon results are not credit-assignment results.** See section 6. No
  action in this environment changes any state any later reward depends on.
- **8 seeds per arm (4 for the horizon sweep and the MLP grid), and the seeds
  vary the environment, not the synthesis.** Per the P2 warning, `SoftProgram`
  zero-initialises every choice logit, so `torch.manual_seed` does not perturb
  the program. Seed variation here comes from the episode addresses and from
  action sampling, except in the two arms that add explicit noise
  (`A1d`, `choice_noise=0.5`, and `A2`, `N(0, 0.1)` on the policy constants),
  which say so.
- **The harness is a re-implementation, not the shipped trainer.** It is checked
  against the fixture (section 0b) and reproduces its behaviour, but a
  measurement made through `JointTrainer` itself could differ in ways the parity
  check does not cover — in particular the crystallizer is absent, so none of
  these numbers include its undisclosed validation rollouts.
- **`tcn/` changed under this track** while it ran (another agent's
  `exact_tensor` memoization, `TrainConfig.description_weight`,
  `Program.pruned`). `parity.py` was re-run after the change and is
  digit-identical, but the earlier arms were computed against the earlier tree.
- **The exact all-actions estimator is privileged.** It reads the counterfactual
  reward of the action not taken, which the environment does not offer an agent.
  It is a diagnostic upper bound on what removing action-sampling variance can
  buy, nothing more.
- **The reward model in arm B is one bit by construction**, because the task's
  reward is `[answer == target]`. On a task with a richer reward, "learn the
  model with supervision, learn the reward with reward" would not collapse to a
  single bit, and the arm would have to be re-run.
- **The neural baseline has no learned value head.** It gets the program's fixed
  constant baseline at batch 1 and an advantage-normalisation batch baseline at
  batch 32; `research/baselines`' MLPs had their own value output. The grid is
  12 configurations, which is comparable to track 6's 12-18, and every cell is at
  chance, so the conclusion is not fragile — but a learned critic was not tried
  here.

### Tuning disclosure

**TCN side.** Every arm inherits `examples/joint.py`'s shipped hyperparameters
(`lr=.04`, `discount=.95`, `value_weight=.5`, `entropy_weight=.01`, Adam,
grad-clip 5, horizon 4, alternating objectives) unmodified. The only tuning
performed on the TCN side is the remedy table itself (section 3), which is
reported in full including every configuration that made things worse, and it
reports `lr` 0.005/0.04/0.2, batch 1/8/32, and two entropy schedules. The
headline reward-only result (A1, P1) uses the shipped values with **no** tuning.

**Baseline side.** The MLP received the 12-configuration grid above, selected on
nothing — every cell is reported. The oracle, the two constants and the uniform
policy received no tuning. The discrete enumeration in `space.py` is exhaustive.

### Provenance

`tcn.generation.source_fingerprint()` at the time of the final parity check:
`66f9c0795a92184d4c2182637bd08744f602ef8b2e35d458a2dae1a5a9091b5f`. Machine load
averaged 30-80 on a 20-core host throughout (other agents active), so no wall
clock in this document is a performance measurement; the `2^h` enumeration timings
in section 5 are reported only to show the growth, not the cost.
