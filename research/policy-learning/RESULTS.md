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

The cost is the budget. Reward-only needs about **2,000 environment episodes**
where the probe-supervised fixture needs 160, and at 160 it is exactly at chance.
That budget gap — not an impossibility — is the real finding, and it is closed
almost entirely by a staged schedule: **50 supervised episodes then 50 reward
episodes** reaches 4.00/4 on 8/8 seeds, a 20x saving.

And the sharpest result is that no policy is needed at all: **25 supervised
episodes crystallize an exactly-correct model, after which 2 reward episodes fix
the one remaining bit and enumeration against that model returns the oracle 4.00
at horizon 4 and 32.00 at horizon 32 — 27 environment episodes in total.**

Three corrections to the record follow from this, and one warning:

- **Track 6's "pure REINFORCE fails for the typed program as well as for the
  MLP" was never measured on the typed program.** `research/baselines/joint_baseline.py`
  runs its `reinforce` mode on MLPs only; the TCN side was always run with
  `probe_weight=1`. F-init stands exactly as written — the shipped constants
  *are* the answer — but the inference drawn from it does not. Measured here,
  the typed program learns this task from reward alone at 2,000 episodes on 8/8
  seeds while a budget-matched 32-wide tanh MLP under the identical estimator and
  observation vector stays at chance (2.02) at every budget to 4,000 episodes.
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
episodes. `env ep` includes the evaluation rollouts.

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

**A1's selections are the evidence that the reward result is real.** Over 8
seeds the reward-only arm lands on `(6,9)` 4 times, `(6,6)` twice, `(9,6)` and
`(9,9)` once each — that is 4 of the 4 reward-optimal programs enumerated in
section 0 and never the probe-optimal pair alone. Reward is selecting for
return, not recovering a supervised solution by another route. The
probe-supervised arms (A0, A5, A6) select `(6,6)` in 8/8, because only the probe
objective distinguishes it.

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
what the 2,000-episode budget buys.

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
| **discrete-choice bottleneck** | **B6 choices pinned to (6,6), readout only** | **4.000** | **8/8** | **this is the cause** — actor SNR 0.405 vs 0.121, and B7 reaches 4.00 8/8 in **200 training episodes** instead of 2,000 |
| | B7 pinned, 200 episodes | 4.000 | 8/8 | |
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
choices takes the reward-only budget from ~2,000 episodes to under 200 and raises
actor SNR 3.3x.**

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
| **supervised → reward handoff (section 4)** | **4.000** | 0.000 | **8/8 at 100 training episodes** | **20x fewer environment episodes** |

**Track 2's batching result, corrected.** Track 2 implemented `TrainConfig.batch`,
measured `batch=8` worse at equal episodes, and reverted it, concluding batching
"starves the readout". The same effect reproduces here (R2: 2.28/4 at batch 32)
and the stated mechanism is right — at fixed episodes, batch b divides the
optimizer steps by b. But it is a step-size question, not a reason to reject
batching: **R3, the same batch 32 with `lr` raised from .04 to .1, is 4.00/4 on
8/8 seeds.** Track 2's arm held `lr` fixed at .04 across batch sizes.

**The budget curve is the whole story for reward-only.** Same arm, varying only
the number of training episodes:

| training episodes | env episodes | reward only, eval | seeds at 4.00 |
|---|---|---|---|
| 160 (the shipped budget, `p[12]` bias kept) | 224 | **1.977 — chance** | 0/8 |
| 400 | 464 | 3.570 | 6/8 |
| 800 | 864 | 3.766 | 7/8 |
| 2,000 | 2,064 | **4.000** | 8/8 |

So the correct statement of the failure is not "reward-only is at chance" but
"reward-only is at chance at 160 episodes and solved at 2,000".

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
REINFORCE. Reward-driven content: 1 bit of 9. Supervision-driven: 8 of 9. The
return, however, is 100% attributable to having both: stage 0 alone scores 2.07
and reward alone at the same budget scores 1.98.

**On the section-8 deviation.** The staged schedule reaches the same 4.00 as the
simultaneous schedule at **100 training episodes against 1,000**, a 10x saving,
and against reward-only's 2,000, a 20x saving. That is a real argument for
relaxing section 8's simultaneity requirement to *"prediction and reinforcement
are both active over the course of integrated training, with a declared
schedule"*, and for relaxing `TrainConfig`'s `> 0` checks to `>= 0` with the
requirement that at least one is positive — which is also what makes a
supervision-only or reward-only arm expressible in the first place. It is one
fixture, and freezing was **not** what bought the saving (the no-freeze row is
identical), so the honest claim is about the schedule, not about crystallization.

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
8/8 seeds.** Against 2,000 for reward-only REINFORCE, 160 for the shipped
probe-plus-supplied-decoder fixture, and 331 for the shipped fixture's true
budget once the crystallizer's validation rollouts are counted.

Enumeration against the frozen model, at every horizon:

| horizon | plan horizon | sequences enumerated per step | mean return | normalised | oracle | uniform | wall for 16 episodes |
|---|---|---|---|---|---|---|---|
| 4 | 4 | 16 | **4.000** | 1.000 | 4.000 | 2.047 | 0.03-0.09 s |
| 8 | 8 | 256 | **8.000** | 1.000 | 8.000 | 3.781 | 0.07-0.21 s |
| 16 | 16 | 65,536 | **16.000** | 1.000 | 16.000 | 7.594 | 12.7-15.3 s |
| 32 | 12 (capped) | 4,096 | **32.000** | 1.000 | 32.000 | 16.063 | 8.2-10.8 s |

**And the honest reading of that table, which matters more than the numbers.**
Enumeration recovers the oracle at every horizon, but it buys *nothing* over a
one-step greedy decision, and the same 27 episodes give the same 32.00 at
horizon 32 with `plan_horizon = 1`. The reason is section 0: this environment has
no dynamics. The frozen model's `z` does not depend on the action, so the value
of a sequence is separable and the arg-max sequence is the greedy action
repeated. `2^h` grows as expected — at h=16 a full-sequence plan already costs
~0.9 s per episode and h=32 has to be capped — but that cost is pure overhead
here.

So arm B's result is real and its generalisation is not tested. What this fixture
supports: **a crystallized program is an exact `(observation) → latent` function,
and once you have it the reward-driven content collapses to a single bit that two
episodes identify.** What it does not support: any claim about planning depth,
because there is nothing to plan through. Testing that needs a generator whose
state responds to actions; `logic` is not one, and neither is anything currently
under `generators/` that this scaffold can reach.

---
