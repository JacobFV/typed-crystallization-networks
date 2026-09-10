# Validation record — 2026-09-09

These results come from executions in this workspace. They establish reference
implementation behavior and small learned programs, not a trained general agent.

**The standing rule.** Every number in this file carries the reference it has to
beat. A return without a constant-policy baseline, a synthesis result without an
enumeration comparison, or a search result without the size of the space it
settled is uninterpretable, and most of the claims this file made before
2026-09-08 failed exactly there. `scripts/demo.sh` re-runs the capabilities
below on the current tree and prints each number next to its baseline;
`STATUS.md` states what does not work.

Sections 1 and 2 are what reproduced. Section 3 is the corrections, each with
its evidence — the original numbers are kept, not deleted, because several of
them are correct and only their interpretation was wrong. Section 4 is what has
been established since the corrections landed. Sections 5 and 6 are the
unsuccessful and the unestablished.

---

## 1. Framework checks

| Check | Result | Evidence |
|---|---|---|
| Python behavioral tests | **179 passed, 145.79 s** (was 86 on 2026-09-08) | `tests/`; `.venv/bin/python -m pytest -q` |
| Local computer engine tests | **370 of 372 passed** | `npm --prefix generators/computer/engine test` |
| Computer TypeScript build | Passed | `npm --prefix generators/computer/engine run typecheck` |
| Locked environment installation | Passed | `uv sync --locked --extra test` |
| Wheel build | Passed; includes local TS bridge/kernel/lock, excludes node_modules | `uv build --wheel --no-build-isolation` |
| Full curriculum | All 14 stages passed, two process workers | `artifacts/validated-system/curriculum.json` |
| Language catalog | 179 executable lessons × two seeds generated nonempty prompt/answer | `tests/test_integration_complete.py` |
| Frozen agent CLI | 4 steps, return 4/4, replay identical | `artifacts/validated-agent.json.gz` |
| Embodied raster export | Two actor streams and stable focus streams exported | `artifacts/validated-world-frames/` |

The two failing engine tests are wall-clock thresholds, not behavior: `sandbox
execution budget > does not spend the bundle budget on kernel syscall service
time` asserts `durationMs < 80` and measured 81 on a machine running a dozen
other experiments. They pass on a quiet host. Recorded here rather than omitted,
because a timing assertion that fails under load is a fact about the test.

The Python checks include signature rejection, exact/relaxed behavior, set order
invariance, recurrence, library reuse without internal gradients, rollback after
gradient disconnection, standalone execution with isolated Python, typed continuous
action conditioning, probabilistic losses, frozen checkpoint restoration,
privileged-data separation, timing, process isolation, computer causal effects,
physical interaction, occlusion, and episode replay. The bit-parallel backend is
compared against all 16 exact truth tables across 1,024 lanes and recurrent state.

**One entry point does not run as documented.** `.venv/bin/tcn` is a console
script whose shebang names the interpreter the wheel was first built against
(`/home/brandonin/Documents/differentiable-agentic-software/.venv/bin/python3`),
which does not exist on this host. Every `.venv/bin/tcn ...` command in
`README.md` fails with "No such file or directory". Use `.venv/bin/python -m
tcn ...`, which is what `scripts/demo.sh` does.

---

## 2. Reproduced learning results, with the references they lacked

Both flagship fixtures were re-run on the current tree on 2026-09-09, after the
overnight core changes landed, and both reproduce the recorded numbers.

### 2.1 Learned mixed program

`.venv/bin/python -m tcn synthesize` — or `scripts/demo.sh --only synthesis`.

The synthesis fixture learns Boolean XOR, an explicit numeric conversion,
arithmetic, and sine within a supplied coarse graph. All nodes crystallized;
exact conformance passed. The final relaxed probe loss is `1.0066e-6` and the
exported program's exact error against the same targets is `0.0` — both are now
reported, because they disagree systematically and only the second is evidence
(fault F-soft, section 3.7).

| measurement | this repository | reference | source of the reference |
|---|---|---|---|
| error on 16 fitted examples | max **1.7e-8** | matched MLP 2.5e-7 MSE | `research/baselines/RESULTS.md` §3 |
| interpolation, 146 unseen `x` | RMSE **1.8e-8**, max 6.2e-8 | matched MLP **5.2e-3 / 1.8e-2** | same |
| extrapolation, \|x\| ≤ 2 | RMSE **3.1e-8** | matched MLP **0.232 / 0.887** | same |
| discrete content | unique program of **96** | enumeration exhausts the space in **7.2 ms** against 2.7 s of gradient descent, and selects the identical program | `tcn/search.py`; `research/enumerative-baseline/RESULTS.md` §2 |
| serialized description | 17,728 bits | matched MLP 20,000 bits of float32 weights — **a tie, not a win** | `research/baselines/RESULTS.md` §7 |
| batch-one latency | 0.0487 ms recorded, **0.1045 ms** re-measured under load | the same four operations in plain Python: **0.00013 ms** | same |

The exactness row is the one large, structural advantage a matched baseline
cannot touch: the MLP fails this repository's own `abs=1e-6` test assertion by
more than three orders of magnitude, and training it longer makes it worse.
Hardware/software details and the source fingerprint are in
`artifacts/validation-environment.json`.

### 2.2 Joint prediction and reinforcement

`.venv/bin/python -m tcn train --episodes 160` — or `scripts/demo.sh --only joint`.

A typed truth-table scaffold learns a latent relation and its goal-conditioned
answer under two objectives. It receives the task bits and the objective bit;
privileged generator targets supply training losses.

Re-run 2026-09-09: average prediction loss over the first/final eight episodes
fell from **0.248836 to 0.002231**, deterministic evaluation reached **4/4** mean
return over 16 held-out episodes, the program crystallized fully, and the exact
frozen agent reached **4/4** on another 16 test episodes. Every figure matches
the record to five digits.

| reference on the same episodes | return |
|---|---|
| always-True / always-False, 64 episodes from index 30000 | 2.38 / 1.63 |
| uniform random, 64 episodes | 1.91 |
| **always-True on the 16 episodes the record actually scores** | **3.00** |
| exact oracle `(bits[0] xor bits[1]) xor goal` | 4.00 |
| budget-matched 153-parameter MLP with replay (331 episodes, 194 steps) | **4.00** on 3/3 seeds |
| 32-entry lookup table over the five observed bits | **4.00** |

So 4/4 is above chance, and it is also reachable by a 153-parameter network and
by a 32-bit table. The sixteen episodes the recorded protocol scores are not
balanced: a constant answer already reaches 3.00/4 on them, which is why
`scripts/demo.sh` takes its verdict against a 64-episode extension instead.

Five things about this result that the previous version of this file did not
state, all of them measured:

1. The policy decoder is **initialized to a correct solution**. `examples/joint.py`
   declares `w0=-2, w1=2, bias0=1, bias1=-1`, so `logit0 = -2z+1` and
   `logit1 = 2z-1`, and `argmax(logits) == goal_relation` before any training.
   Training only rescales those constants. The learned content is two 16-way
   truth-table choices: **8 bits**. (`research/baselines/RESULTS.md` §2, fault F-init.)
2. `--episodes 160` consumes **331 environment episodes**: 161 training rollouts
   plus **170 undisclosed `split='validation'` rollouts** inside the crystallizer's
   loss closure, and 194 Adam steps. (Fault F-budget.)
3. **Crystallization is not what produced it.** An argmax of the trained soft
   graph, with no crystallizer and zero extra objective evaluations, reaches the
   same 4/4 and the same frozen program.
   (`research/crystallization-ablation/RESULTS.md` arm B0; independently in
   `research/perturbation-selection/` and `research/loss-gated-eligibility/`.)
4. **Enumeration settles it in 0.081 ms.** The discrete content is a 256-way
   choice; brute force over the identical candidate space returns the identical
   program and a uniqueness certificate, against 10–36 s of gradient descent.
   Under the *reward* objective the space holds two optimal assignments, `(6,6)`
   and `(9,9)`; only the probe objective identifies the reference one.
   (`research/enumerative-baseline/RESULTS.md` §0.)
5. It is a **single fixed Boolean function** with held-out episode addresses.
   With `{'depth':1,'table':6,'fixed_inputs':True}` every episode instantiates
   the same relation. See section 3.2.

Reports, checkpoint, frozen JSON, standalone `.pyz`, and agent configuration are
under `artifacts/validated-system/joint_prediction_policy/`.

---

## 3. Corrections

Each entry states what this file previously claimed, what is now measured, and
where the measurement lives. The original numbers are preserved; what changed is
what they license.

### 3.1 Progressive crystallization was credited for a result an argmax reaches

**Previously:** "Progressive crystallization accepted every node after deferring
three attempts that would have disconnected remaining gradients."

**Measured, in three independent tracks.** At the shipped budgets the scheduler
is inert: eleven arms are bit-identical, and an arm with no crystallizer and zero
extra objective evaluations reaches the same frozen program on 160 plain episodes.
At tight budgets it is actively harmful — on the mixed fixture at 30 steps the
scheduler conforms 3/16 where budget-matched argmax conforms 16/16
(Fisher p = 3.2e-06), discarding 94.8% of its optimizer steps to rollback.
Outside-in ordering is inert (arm A identical to arm C) and degradation tolerance
is inert (arm E bit-identical to arm A in all five configurations).
A fourth track, running the sharper step-matched comparison on the joint task at
budgets below saturation, finds step-matched argmax above every scheduler arm at
2.4–2.5× fewer environment rollouts.

Evidence: `research/crystallization-ablation/RESULTS.md`,
`research/perturbation-selection/RESULTS.md`,
`research/loss-gated-eligibility/RESULTS.md`.

The mechanism that survives is smaller and better evidenced: **the interval in
which a perturbation measurement beats argmax is exactly the interval in which
committing is a mistake.** For DARTS-PT-style selection to pay here, the freeze
would have to be reversible.

### 3.2 The 4/4 joint result is not structural generalization

**Previously:** "the 4/4 joint result holds out episode addresses on a single
fixed Boolean function rather than generating structures, and measures at chance
on unseen truth tables." That correction was right and is retained. What has
changed is that the positive claim is now available, on a benchmark that earns it.

**The recorded scaffold is at chance and structurally cannot exceed it.** It never
observes which Boolean function the episode instantiates, and its gate choice is a
global parameter, so one frozen program computes one relation. On disjoint,
non-degenerate, balanced gate pools it measures **1.95–2.03** against a best
constant of **2.13–2.56**, with an analytic ceiling of exactly 2.00 verified by
exhaustive search over all 16 gates.

**A table-conditioned scaffold does generalize.** Exposing the generator's
`program` observation and adding one `index`-over-truth-table candidate — made
legal, not supplied — takes held-out gate families to **3.38–4.00**, in both
split directions and including when wiring varies, with the interpreter candidate
**selected in 8/8 seeds in every condition**. Enumeration over the same
272-program space also reaches 4.00 held-out and picks the same candidate.

Evidence: `research/nondegenerate-generalization/RESULTS.md`,
`research/structure-generalization/RESULTS.md`. Reproduce:
`scripts/demo.sh --only structure`.

**Depth generalization was also recorded as chance, and is now achieved.** The
blocker was representational: the `program` observation is `3 × depth` wide, so a
fixed-width typed program type-errors on an episode of unseen depth. A second
typed view of the same state — a `gates` channel of
`set[(index, wire_a, wire_b, table)]` at a declared capacity, width 40 at every
depth — makes it expressible with no addition to the algebra. One fixed graph
trained at depths 1–2 scores **4.00 at depths 1, 2, 3, 4, 6 and 8**, sd 0.00 over
8 seeds, against best constants of 2.06–2.50. Enumeration over the same
272-program space exhausts in 25 s, certifies the optimum unique, and picks the
same program.

Evidence: `research/depth-generalization/RESULTS.md`. Reproduce:
`scripts/demo.sh --only depth`.

### 3.3 The description-size advantage does not exist

**Previously:** "Serialized description size was 17,728 bits."

The number is correct and reproduces. What it does not support is an advantage.
`description_bits` measures serialized JSON length, not learned content. Mixed:
17,728 bits against 20,000 bits of MLP weights — a tie. Joint: **68,768 bits to
express 8 bits of learned content**, against 4,896 bits for an equally-scoring
153-parameter MLP and **32 bits** for an equally-scoring lookup table, a factor
of 2,150. Exports also retained dead scaffold nodes, a 32% overstatement; that
part is now fixed (`Program.pruned()` is applied at module registration and at
`save_program`), which removes a 62% overstatement from an abstracted export but
does not change the comparison above.

Evidence: `research/baselines/RESULTS.md` §7,
`research/abstraction-preference/RESULTS.md` §3.

### 3.4 The inference-cost advantage does not exist as stated

**Previously:** "median 0.0487 ms and p95 0.0591 ms ... It is a four-operation
program, not an LLM throughput result."

The timing is correct and reproduces (0.1045 ms p50 today, on a loaded host). The
framing defended the wrong comparison. Against four operations written in plain
Python (0.00013 ms) the exported program is **450× slower**, and it is about the
same as a 625-parameter torch MLP and 17× a numpy MLP. The joint program is
0.247 ms against 0.038 ms for an equally-scoring MLP and 0.00045 ms for a lookup
table. The `estimated_operator_cost = 4` proxy and the measured wall time are
about 2.5 orders of magnitude apart; the graph interpreter dominates.

Evidence: `research/baselines/RESULTS.md` §5, §7.

### 3.5 The sample-efficiency comparison did not count 170 rollouts

**Previously:** "After 160 episodes ... Deterministic evaluation achieved 4/4."

`joint(episodes=160)` spends **331** environment episodes and 194 optimizer
steps; `mixed(steps=300)` spends **340** full-batch Adam steps. The 170
`split='validation'` rollouts are real environment interaction taken inside the
crystallizer's loss closure and were reported nowhere. Given the same 331
episodes and the same 194 optimizer steps, and allowed to replay what it already
collected, a **153-parameter** MLP also reaches 4.00/4.00 on 3/3 seeds. What
survives is narrower: under the one-gradient-step-per-episode schedule
`tcn/training.py` uses, every MLP tried fails at every budget up to 2,560
episodes, and the typed program does not — because it only has to move 2 × 16
softmax logits.

Evidence: `research/baselines/RESULTS.md` §2, §4 (fault F-budget).

### 3.6 The `logic` generator's `depth` was not a difficulty axis

At depth 8, 40% of default targets were constant functions and random sampling
found the target in 50–200 draws; measured solution density *rises* with depth.
Any depth-versus-accuracy curve on the old sampler varied candidate-set size while
target complexity stayed flat — and that is the benchmark the curriculum and both
flagship demos use. **Fixed:** `generators/logic` gains `inputs`, `nondegenerate`,
`tables`, and `min_relevant_inputs` with `max_attempts`. At depth 8 with
`min_relevant_inputs=4`, 0% of targets are constant and 100% depend on all four
inputs. The default draw was verified bit-identical across 192 seed/config
combinations, so recorded episodes are unaffected.

Evidence: `research/search-scaling/RESULTS.md`, `research/FINDINGS.md` §10
(fault F-bench).

### 3.7 A relaxed loss was reported as evidence of a correct program

On some targets 17/17 failures reached zero soft BCE with a wrong argmax, and 92%
of runs passed through a correct argmax mid-training and left it. **Fixed:**
`synthesis.fit` now reports `exact_max_error` — the largest disagreement between
the *exported* program and the targets — alongside `relaxed_loss` and the
tolerance they are judged against. On the mixed fixture these read 1.007e-06 and
0.0: the relaxed number is nonzero while the exported program is exact, which is
the disagreement in miniature.

Evidence: `research/search-scaling/RESULTS.md` (fault F-soft).

### 3.8 The conformance callback rejected valid freezes

`SoftProgram.export()` argmaxes every node, so testing exported conformance
mid-freeze reported the state of untrained nodes rather than the validity of the
freeze; 57–74% of rejections mismatched only on untouched still-soft nodes.
**Fixed:** the check now applies only when a freeze completes the program. On the
mixed fixture at 30 steps this takes frozen nodes from 0% to 75% and rollbacks
from 1,264 to 336, and changes nothing at budgets that already worked.

Evidence: `research/crystallization-ablation/RESULTS.md` (fault F-conf).

### 3.9 The gradient-connectivity guard still does not test what it claims

`grad is None` tests reachability in the autograd graph, not the presence of
learning signal: a discreteness or entropy regularizer keeps every choice logit
attached. Treating an all-zero gradient as disconnected was implemented and
**reverted after measurement** — it also flags nodes whose choice has legitimately
concentrated, and blocked the joint fixture from crystallizing at all (286
deferrals against 4). What did land is the objective split: `Crystallizer.run`
now takes `Objective(total, task)` so the guard probes the unregularized task
loss. The joint numbers are unchanged; the disconnection deferrals go from four
to three. A fully correct guard needs an interface change that is left open
rather than guessed at.

Evidence: `research/crystallization-ablation/RESULTS.md`,
`research/perturbation-selection/RESULTS.md`.

### 3.10 Recursive abstraction: track 5's verdict was an accounting artifact

**Previously recorded:** the module was on the output path in 0 of 20 runs, at
1.4× description bits and 1.7× latency, with no execution-cost crossover.

Two faults caused it. `Program.execute` re-validated the program on every call,
making module candidates 16–98× costlier than primitives, and a one-output module
resolved to a one-field product so every call site paid a `project` node. Both
are fixed. Re-tested: **27 of 27 successes are the abstracted program.** Arm B
(module available) reaches 8/8 wide and 19/24 tight where arm A (flat) and arm C
(same-size distractor) both reach 0 (Fisher p = 1.6e-4 and 7.4e-9). Enumeration
certifies this is not a search artifact: arm A's 230,400-program space is
exhausted with no solution, and arm B's contains 144 solutions, **all** of which
use the module.

Evidence: `research/recursive-abstraction-retest/RESULTS.md`,
`research/recursive-abstraction/RESULTS.md` (the superseded track). Reproduce:
`scripts/demo.sh --only abstraction`.

What is still true is narrower: reuse was a *capability*, not an objective.
`complexity()` could not have expressed the preference — at the discrete
selections it reads 9.0 flat against 15.0 abstracted, penalizing abstraction for
exactly the saving that makes it worth having. `SoftProgram.description_cost()`,
exposed as `synthesis.fit(mdl_weight=)` and `TrainConfig.description_weight` and
defaulting to zero, is the term that can. It is a real trade: at 1e-5 it takes
the wide scaffold from 7/8 to 8/8 minimal programs and stops the search drifting
into larger conformant ones (3/8 to 0/8), and it takes the tight scaffold from
19/24 to 2/24 conformance (p = 1.1e-6), because the pressure is "make fewer nodes
live". (`research/abstraction-preference/RESULTS.md`.)

### 3.11 The "address wall" was misattributed, then corrected

An earlier reading of the perception tracks recorded that relaxing an input
*address* is worse than chance (0.25 against a 0.29 chance rate) while relaxing a
*value* at a fixed address is 208× better, and concluded that addresses must be
computed and never relaxed. That was stated three times before it was checked.

**It was a dead surrogate, not a relaxation limit.** `eq`'s relaxation is
`exp(-(a-b)²/τ)` at τ=1, which in float32 is exactly 0.0 for `|a-b| ≥ 11`. The
uniform mixture over 12 raw `geometry` bytes sits 25.6 away from the constant it
is compared against, where the surrogate reads 0.0; with colours pinned, 16 of 16
address gradients are exactly zero. The recorded 0.25 figure came from the
instrument silently dropping the dead nodes and averaging the survivors. Scaling
the surrogate by the carrier width takes the failing arm from 0/12 to 7/12 at
576 programs and 0/12 to 12/12 at 9,216.

The construction rule survives in weakened form: compute addresses where you can,
because it is cheaper and certifiable — but if you must relax one, check that the
downstream relaxation is valid *at the mixture* before blaming the address. The
one instance that survives independently is depth generalization's wire-binding
gradients (4.7e-08 and 3.1e-05 against 8.7e-03 for an operator choice) on a task
with no images and no bytes.

Evidence: `research/address-wall/RESULTS.md`, superseding the framing in
`research/perception-ladder/RESULTS.md` and `research/discrete-perception/RESULTS.md`.
**The fix is not applied to `tcn/`** — see `STATUS.md`.

### 3.12 Per-seed numbers in the shipped fixtures are one outcome repeated

`SoftProgram` zero-initializes every choice logit, so `torch.manual_seed` does
not perturb synthesis at all: four seeds produce one identical all-zero
initialization. Every per-seed synthesis number in this repository is therefore
one outcome repeated N times, unless that harness added explicit initialization
noise. The research tracks did and said so; the shipped fixtures do not. This is
the mechanism behind the determinism noted throughout.

Evidence: `research/perception-ladder/RESULTS.md` (fault P2).

---

## 4. Established after the corrections, each with its baseline

All of these reproduce on the current tree through `scripts/demo.sh`; artifacts
land under `artifacts/demo/`.

| capability | measured | baseline printed beside it |
|---|---|---|
| **typed synthesis, exact export** | held-out max error ≤ 1e-7 on 584 unfitted points | unique among 96 programs, enumeration in ~7 ms against ~3 s of gradient descent, same program; matched MLP 1.8e-2 |
| **structural generalization** | 3.38–4.00 on gate families never trained on; interpreter candidate selected 8/8 unprompted | best constant 2.13–2.56; the recorded scaffold measures 1.95–2.03 and cannot exceed 2.00 |
| **depth generalization** | 4.00 at depths 3, 4, 6 and 8 from one graph trained at depths 1–2 | best constant 2.06–2.50 per depth; 272-program space, optimum unique |
| **recursive abstraction** | module on the output path in 27/27 successes | flat arm and distractor arm both 0; the 230,400-program flat space is exhausted with no solution; 144 of 144 solutions use the module |
| **positional reuse** | one frozen module at 1,024 positions of a 3,072-value observation with 3 caller nodes (2,304 over 6,912 values under `--full`) | 17 structural symbols shared against 3N+13 per-position and 8N+1 inlined, at 0.93× execution cost |
| **segmentation from raw pixels** | held-out max error 0.0 on 48 unseen episodes, background colour recovered over the full 0–255 alphabet | unique among 65,536 programs; constant predictor 0.854 |
| **two-position edge detector** | held-out max error 0.0, accuracy 1.000, offset searched | unique among 48 staged programs; undecomposed 4.9e10 programs, 7.6 years projected; constant predictor 0.844 |
| **a language task from raw prompt bytes** | the lexical unit discovered over the full 0–255 alphabet, unique among 10,496 programs; **on the post-audit stream the generator ships**, the balancedness rule **1.000** on **859** held-out episodes at lengths 16–22 never trained on (running `min` beside the running `add`; 680,625 exhausted, 110 conforming, `complete`). **On the pre-audit stream** the grammaticality rule **0.9986187845303868** on 724 (one error, at length 16) — a lesson §24 found exploitable — see FINDINGS §39, §43, §45, §47 | post-audit majority constant 0.5262, best fitted feature 0.4738, training-string lookup 0.4738, random 0.500; pre-audit majority constant 0.548, best fitted feature 0.648; gradient descent conforms 0 of 64 runs post-audit and 0 of 44 pre-audit |
| **external simulator replay** | replay, snapshot/restore and cross-process reload all bit-identical, max \|Δ\| 0.0 | a scripted energy-pumping controller reaches upright 0.9994 where the zero-torque arm never exceeds −0.99 |

Three further results with no demo, because they are certificates rather than runs:

- **Dense hierarchical supervision makes hard synthesis tractable.** Free-wiring
  synthesis goes from 19–38% to 88–94% at 2,120 candidates per node, flat at
  every depth from 3 to 16. On entangled outputs it is decisive (geometry 12/12
  against 0/12; relations 8/8 against 1/8); on a decomposable per-element output
  it is identical to output-only supervision, because `probe_loss`'s elementwise
  BCE already is the mean of the per-element losses.
  (`research/search-scaling/RESULTS.md`, `research/perception-ladder/RESULTS.md`.)
- **The same staged perception result reproduces on a second domain.**
  `generators/gui` is a new peer generator that renders a widget tree to raw
  pixels and emits it as `set[(id, parent, kind, x, y, w, h)]` at a declared
  capacity — invariant from 2 widgets to 24 and nesting 1 to 6, the same shape as
  `logic`'s `gates` channel. Rung one exhausts spaces of 1,280, 13,056 and 81,920
  programs, each returning exactly one distinct Boolean function at held-out max
  error 0.0, against a majority baseline of 0.6700 and a uniform random program
  conforming 0 times in 400 draws; the selected program is registered and applied
  at every position by three caller nodes. Two recoverability ceilings came back
  negative *before* any search — with borders drawn the two-pixel equality bound
  is exactly the majority baseline, advantage 0.0000 — and the search returns 0
  conforming programs at exactly those settings.
  (`research/gui-hierarchy/RESULTS.md`.)
- **The per-pixel perception wall above segmentation is informational, not
  algorithmic.** For `object_ids` and `depth` the best possible per-pixel
  predictor — an RGB lookup table, an upper bound on the whole family — fits
  training pixels perfectly and scores exactly the majority baseline on held-out
  episodes, advantage **0.000**, at pixel, pixel+position, aggregate and 3×3
  window contexts. Two structural certificates support it: permuting object ids
  leaves the image identical, and recolouring leaves the ids identical. Four
  candidate families were exhausted with no solution. These are completeness
  certificates, not budget failures.
  (`research/object-identity/out/bounds.json`,
  `research/discrete-perception/out/rung4_objects.json`.)

---

## 5. Unsuccessful and unestablished results

An earlier generic arithmetic/sine scaffold trained for 640 episodes stayed near
chance: mean return **2.03125/4**, with final prediction loss around **0.2806**.
Its report remains at `artifacts/joint-long/report.json`.

That 640-episode result does not show what the scaffold can learn. It was
reproduced and audited in `research/scaffold-autopsy/`: the run was stopped before
its learning transition. The same scaffold, same trainer, same learning rate and
same task, run to 5120 episodes, reaches **4/4** deterministic return on held-out
episodes with held-out prediction loss **4.4e-4** on both seeds tested; the
transition occurs near episode 750, 800 and 2600 for seeds 0, 2 and 1. By 2048
episodes the relation itself is learned on all three seeds (prediction loss
1.1e-4 to 9.4e-4); the policy readout is the slower half. Under plain
full-batch supervision the identical program fits the same targets to MSE `1e-5`
at 100% accuracy in about 120 optimizer steps, across twelve width/rate/seed
settings, so neither capacity nor operator search was the obstacle. Its
per-episode score-function gradient has a measured signal-to-noise ratio of 0.22,
its policy readout must be found by that noisy term rather than being wired to
the latent, and its prediction and policy gradients transiently conflict (cosine
down to −0.68) while it sits on the plateau. **The recorded episode budget, not
the scaffold, produced the near-chance number.**

**Policy learning from reward, restated.** An earlier version of this file, and
of `research/FINDINGS.md`, recorded that pure policy-gradient learning fails at
chance for the typed program and for a matched neural baseline. The second half
holds; the first does not. The correction is recent enough that the track has a
`README.md` file index and **no `RESULTS.md`**, so what follows cites its raw
JSON directly. References on the same 256 test episodes
(`research/policy-learning/out/refs.json`): always-False **2.125**, always-True
1.875, uniform **2.047**, oracle **4.000**.

Reward-only REINFORCE on the typed joint program with a zero-initialized policy
readout reaches **4.000 on 8/8 seeds**, and the budget curve puts saturation at
**400 training episodes / 464 environment episodes** — against the shipped run's
331, which additionally has a pre-solved decoder and dense privileged probes
(`out/e1.json` arm `A1_rewardonly_zeroinit`, `out/e7.log`). Staging is cheaper
again: 50 probe episodes then 50 reward episodes is **356 environment episodes**
to 4.000 on 8/8 with the reference program 8/8 (`out/e4b.log`). A matched
32-hidden MLP under REINFORCE on the same raw observations sits at **1.977–2.063
at every budget up to 2,000** training episodes — below always-False — 4.3× past
the point the typed program has solved it (`out/e7.log`).

Four qualifications, all measured, and each of them limits the result more than
the headline does. **Terminal-only reward fails at every horizon tested**: dense
per-step reward solves horizons 4, 8 and 16 outright (4.000/4, 8.000/8,
16.000/16) while terminal-only gives 1.000, 1.000 and 0.859 (`out/e5.log`, still
running). **Advantage normalization destroys it**: three arms that add it land at
2.008 on 0/8 seeds, below always-False, and more episodes do not repair them;
batch 32 alone also breaks it unless the learning rate is raised tenfold
(`out/e3.log`, 19 arms × 8 seeds). **Reward does not identify the program**: it
admits four optimal assignments of the 256 and the typed arm selects the
reference one in only 2/8 seeds (`out/space.json`). And the mechanism is that at
zero initialization the actor gradient with respect to the choice logits has
**norm exactly 0.0** at every averaging level from 1 to 256 episodes, with a
uniform 16-way truth-table mixture shown to have an exactly zero Jacobian
independently; even with oracle constants and 256 averaged episodes the actor
direction never ranks `goal_relation`'s optimum above chance, while the probe
gradient identifies `relation` from 8 averaged episodes (`out/e2_direction.json`,
`out/cancellation.json`). This remains a 256-way discrete choice on a horizon-4
task with dense reward; nothing larger and nothing sparser has been learned from
reward. `out/e1.json`'s exact-policy-gradient arms return 0.000 on every seed
where `out/e3.log`'s return 4.000 on 8/8, and nothing reconciles that, so cite
neither.

**One language lesson has now been learned end to end, and the same run bounds
the rest of the catalogue.** `research/language-capability/RESULTS.md`: a typed
program discovers its own lexical unit from raw prompt bytes — which byte opens a
bracket, over the full 0–255 alphabet, and where the symbol field starts — unique
among 10,496 programs, exhausted in 3.5 s, held-out position error 0.0 at lengths
never trained on. Staged on that frozen module, a grammaticality rule reaches
**0.9986187845303868** on 724 held-out episodes at unseen string lengths — one
error in 724, at length 16, **on the pre-audit stream** (`hardening='none'`) and
on a lesson FINDINGS §24 found exploitable — against a majority constant of
0.5483425414364641, a best-fitted-feature baseline of 0.648 and random 0.500;
gradient descent on the identical spaces conforms **0 of 44 runs**. **On the
post-audit stream the generator ships**, that rule is not the capability: the
counting program lands exactly on the majority, and balancedness needs a running
`min` beside the running `add`, which reaches **1.000 on 859** held-out episodes
at unseen lengths 16–22 against a **0.5262** majority
(`research/language-post-audit/RESULTS.md`, FINDINGS §45 and §47). **The demo
prints both, and names the stream on each.** Three limits
are certified alongside it: the lesson does not exercise a stack as sampled (its
negatives always break the bracket count, 20,000 of 20,000 seeds, and the
exported program agrees with `#( == #)` 1.000 and with Dyck membership 0.429);
only **6 of 179** lessons have prompts byte-predictable at a fixed offset; and
`construction` fails to determine `answer` in **39 of 179**, so dense staging is
structurally unavailable for those. Reproduce with
`scripts/demo.sh --only language`.

The current system curriculum samples and checks replay of image, language,
computer, and physical generators. Those stages do **not** mean the agent has
learned reading, inverse rendering, computer use, robotics, or their composition.
The only two learning stages in `curricula/system.json` are `typed_synthesis`
(the mixed fixture) and `joint_prediction_policy` (the logic fixture); every
other stage is `sample` plus a replay gate.

The implementation guide records approximation boundaries: exact but
nondifferentiable set transformations; bounded action encodings; conservative
occlusion; simple mesh rendering; terminal computer affordances; CPU-tested
training; per-event computer reconstruction; and platform-local numerical replay.
No natural data has been used for the initial training experiments.

---

## 6. Standing rules

- **A number without its baseline does not ship.** Constant and random policies
  for returns, enumeration over the identical candidate space for synthesis, a
  conforming count and a uniqueness verdict for any search result. `tcn/search.py`
  exists for the third, `tcn/cli.py:mixed` runs it automatically whenever the
  space is small enough, and `scripts/demo.sh` prints all three.
- **A relaxed loss is not evidence.** Report `exact_max_error` of the exported
  program; they disagree systematically.
- **A per-seed spread is not a spread unless the harness perturbed the
  initialization.** `SoftProgram` zero-initializes its logits.
- **Relaxation belongs where the search is coupled to an environment, to
  continuous parameters, or to noisy partial credit — not where the space is
  small, discrete and exactly checkable.** On both flagship tasks brute force
  settles the discrete content in milliseconds against seconds to tens of seconds
  of gradient descent, and on hard targets the gradient path degrades before
  enumeration does. The one place the ordering flips is where it should: at the
  gradient run's own budget of 704 environment steps, random search over the same
  256 candidates succeeds 0/20 while the gradient path succeeds 5/5.
- **Where the gradient path does win, say so precisely.** It wins on *value*
  choices at a fixed address — 6/6 seeds exact over a 4.3e9-program alphabet where
  brute force projects to 107 days — and loses where the choice sits behind a
  declared `gradient="none"` boundary.
- **A gradient arm must report its surrogate's value at its own operating
  distance.** A `0/n` printed next to a surrogate of exactly 0.0 says nothing
  about the method under test, and a `0/n` next to a loss spread of 5.96e-08 says
  nothing either. `eq` is exactly 0.0 past |a−b| ≥ 11, `lt` past 17, and `index`
  keeps 0.5641 of its kernel mass on the byte it was asked for — all at the
  shipped temperature of 1 against carriers whose declared range is 2⁸ or 2³².
- **Staging is the cheapest capability multiplier measured here.** Freezing a
  learned module and searching a second scaffold that calls it took the edge
  detector from 4.9e10 programs to 48, and the language task from a projected
  50.7 days to 3.5 s plus 363 s. Report the undecomposed space alongside the
  staged one, so the saving is visible rather than assumed.

Generated experiment artifacts are intentionally gitignored. Run the documented
commands to recreate them. Source fingerprints pin replay/checkpoints to their
code revision, so artifacts made before subsequent code changes should not be
silently resumed as though generated by the current implementation — and as of
2026-09-09 **all eleven** `artifacts/system/*/episode.json.gz` fail to reload for
exactly that reason. The granularity is coarse enough to be a hazard rather than
a safeguard: the fingerprint hashes all of `tcn/` and `generators/` together, so
adding one generator invalidates episodes recorded from every other, and it moved
three times during a single working session. `research/FINDINGS.md` remains the consolidated measurement
record and supersedes this file wherever the two disagree.
