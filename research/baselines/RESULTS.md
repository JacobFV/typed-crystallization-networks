# Matched-information baselines — research track 6

ARCHITECTURE.md §9 requires "matched-information baselines" and "complete-path
cost/memory/batch-one latency". `docs/VALIDATION.md` reports neither: every number
there is TCN-versus-nothing. This document establishes the reference points.

Everything below was executed in this workspace on 2026-09-08 with
`.venv/bin/python` (Python 3.13.15, torch 2.14.0+cpu, `torch.set_num_threads(1)`,
Linux aarch64, 20 cores, other agents active on the machine — see the latency
caveat). Nothing under `tcn/` or `generators/` was modified. Code and raw JSON:
`research/baselines/*.py`, `research/baselines/out/*.json`.

The TCN side was **re-run here**, not copied from `docs/VALIDATION.md`, so both
sides are timed on the same host. The reproduction matches the record: mixed
relaxed probe loss `1.0066e-6` (doc: `1.01e-6`), p50 `0.0507 ms` (doc: `0.0487 ms`),
`description_bits` `17,728` (doc: same); joint prediction loss `0.24884 → 0.00223`
(doc: same to 5 digits), evaluation and frozen mean return `4.0` (doc: same).

---

## 1. What each method observed

### Mixed synthesis task (`examples/mixed.py`)

Target: `answer = sin(xor(a,b) + x)`. Fitting set: 4 `(a,b)` combinations × 4 values
of `x ∈ {-0.7, -0.2, 0.3, 0.8}` = **16 examples**, identical for every method.

| method | inputs seen | supervision seen |
|---|---|---|
| TCN | `a:bool`, `b:bool`, `x:float` | 4 declared `Signal`s: `logic` (BCE), `conversion`, `algebra`, `answer` — plus the candidate library that contains `truth_6`/`encode`/`add`/`sin` |
| MLP-answer | `[float(a), float(b), x]` — the same three values | `answer` only (strictly **less** than TCN gets) |
| MLP-signals | `[float(a), float(b), x]` — the same three values | the same four signals on four output heads (**information-matched**) |
| Python oracle | same three values | none (hand-written) |

No baseline saw any `x` outside the four fitted values, and none saw the held-out
test points before evaluation.

### Joint logic task (`examples/joint.py`)

Environment: the real `generators/logic` via `tcn.generation.Host`, configuration
`{'depth':1,'table':6,'fixed_inputs':True,'horizon':4}`, objectives alternating
`{'invert':False}` / `{'invert':True}` by episode index, seed 0. Training episodes
`0…`, `split='train'`; deterministic evaluation on episodes `10000…`, `split='test'`
(the indices `tcn/cli.py:joint` uses), plus an extended `n=256` set on the same family.

Per-step observation vector, **identical for TCN and every MLP baseline** (8 floats):

```
bits          4   observations['bits'].flat()        the 4 boolean task bits
goal          1   observations['goal'].flat()        the objective bit
prev action   2   one-hot over the two answer actions (the TCN Program declares an
                  'action' input of type tuple[F,F]; no node consumes it)
dt            1   the constant 1.0 the TCN 'dt' input receives
```

Never given to any method: `observations['program']` (the gate table — published by
the generator, read by neither side), `latent_states['values']`, episode seeds,
indices, or the action mask beyond `available_actions`, which both sides use identically.

Privileged channels: `probes['target']` and `probes['gate']`. The TCN consumes both
through `prediction_weight=1` (horizon 1) and `probe_weight=1` (horizon 0). The
baselines are run **both without them** (`reinforce`) and **with exactly the same two
channels** (`aux`, `supervised`). The `supervised` variants use `probes['target']`
as a per-step BCE label — the same bits, through the same channel, at the same times.

---

## 2. The joint scaffold is mostly hand-wired — read this before the tables

Decoding the frozen program (`research/baselines/out/tcn_joint/program.json`) shows
`examples/joint.py` supplies a **correct policy decoder at initialisation**:

```
z = encode(goal_relation);  logit0 = -2·z + 1;  logit1 = 2·z − 1
```

so `argmax(logits) == goal_relation` for `z ∈ {0,1}` before any training. Training
only rescales those constants (`-2 → -2.859`, `2 → 2.864`, `1 → 1.047`, `-1 → -1.048`).
The only content actually learned is two 16-way truth-table selections, both settling
on `truth_6` = XOR: **8 bits**, driven by the dense privileged probe loss.

The REINFORCE term is therefore not what produces 4/4. That matters for choosing a
fair baseline, and it is why §4 reports a supervised baseline alongside the
policy-gradient one.

Instrumenting `tcn/cli.py` for the real budget (`research/baselines/out/`, reproduced
by the snippet in `matched_budget.py`'s docstring):

| entry point | reported budget | **actual** budget |
|---|---|---|
| `mixed(steps=300)` | 300 fit steps | **340** full-batch Adam steps |
| `joint(episodes=160)` | 160 episodes | **331** environment episodes (161 train + 170 `split='validation'` rolled by the crystallizer's loss closure) and **194** Adam steps |

The 170 validation episodes are real environment interaction and are not reported
anywhere in `docs/VALIDATION.md`. All budget-matched baselines below are given 331
episodes and 194 optimizer steps, not 160/160.

---

## 3. Comparison table — mixed synthesis task

Quality is measured on three disjoint sets: the 16 fitted examples; the two held-out
points the repo's own test asserts (`a=True, b=False, x ∈ {0.13, −0.43}`); a dense
interpolation grid (146 unseen `x` in `[−0.7, 0.8]` × 4 `(a,b)`); and an extrapolation
set (`x` outside the fitted range, `|x| ≤ 2`). Latency is best-of-3 warmed p50
(see §5). Description size is given two ways: `float32_bits` = 32 × parameters (what
you need to run it) and `json_bits` = 8 × `len(json.dumps(...))` (the measure TCN's
`description_bits` uses).

| method | train MSE | held-out max-abs (repo's 2 points) | interp RMSE / max | extrap RMSE / max | params | description bits | batch-1 p50 | train wall |
|---|---|---|---|---|---|---|---|---|
| **TCN frozen program** | max err 1.7e-8 | **2.6e-8** | **1.8e-8 / 6.2e-8** | **3.1e-8 / 1.0e-7** | 0 trainable | 17,728 (`description_bits`) | 0.0588 ms | 2.86 s |
| MLP-answer (16w × 3, lr .03, 540 steps) | 2.5e-7 | 3.3e-3 | 5.2e-3 / 1.8e-2 | 0.232 / 0.887 | 625 | 20,000 f32 / 107,480 json | 0.0514 ms (torch) · 0.0034 ms (numpy) | 1.46 s |
| MLP-signals (32w × 3, lr .03, 540 steps) | 1.7e-6 | 4.9e-3 | 5.1e-3 / 1.2e-2 | 0.226 / 0.784 | 2,372 | 75,904 f32 / 405,888 json | 0.0638 ms (torch) · 0.0033 ms (numpy) | 2.16 s |
| MLP-answer, 5000 steps (over budget) | 2.8e-4 | 3.1e-2 | 3.0e-2 / 6.7e-2 | 0.306 / 0.797 | 625 | as above | as above | 15.2 s |
| Hand-written Python oracle | 0 | 0 | 0 | 0 | 0 | ~200 bits of source | **0.00013 ms** | 0 |

Budget parity check (exactly 340 steps, the TCN's real count, 3 seeds each — the 540-step
rows above happened to land on a good seed):

| method | interp RMSE across seeds 0/1/2 | held-out max-abs across seeds |
|---|---|---|
| MLP-answer @340 | 0.098 / 0.0080 / 0.0065 | 0.105 / 0.0097 / 0.0086 |
| MLP-signals @340 | 0.0062 / 0.042 / 0.014 | 0.00081 / 0.026 / 0.0050 |

**Reading.** An MLP with the same three inputs fits the 16 examples as well as TCN
does (2.5e-7 MSE) using fewer bits than TCN's serialized program (20,000 vs 17,728 —
roughly a tie) and comparable or lower batch-one latency. It generalizes to unseen
`x` **five orders of magnitude worse** (5e-3 vs 2e-8), fails the repo's own
`abs=1e-6` assertion by more than three orders of magnitude, and collapses entirely
outside the fitted range (RMSE 0.23 against a signal of amplitude 1). More steps make
it worse, not better. This is the one place where a real, large TCN advantage shows up.

---

## 4. Comparison table — joint logic task

Maximum return is 4. `n16` is the repo's own protocol (16 episodes × horizon 4 — but
the environment state is constant within an episode, so it is only **16 independent
binary decisions**). `n256` extends the same index family to 256 episodes.

### Trivial references (these were missing entirely)

| reference | mean return n=16 | mean return n=256 |
|---|---|---|
| always-False | 2.00 | 2.016 |
| always-True | 2.00 | 1.984 |
| uniform random | 1.75 | 1.992 |
| **exact oracle** (`(bits[0] xor bits[1]) xor goal`, computed from the observation vector alone) | **4.00** | **4.00** |

So `4/4` is genuinely above chance (chance = 2.0) and the curriculum gate of 3.8 is
meaningful — but the ceiling is trivially reachable from the observations, because
the oracle is a 3-input parity of bits the agent is handed directly.

### Learned methods

| method | env episodes | opt steps | return n16 | return n256 | params / description bits | batch-1 p50 | train wall |
|---|---|---|---|---|---|---|---|
| **TCN (soft model, eval)** | 331 | 194 | **4.00** | — | — | — | 17.1 s (whole `tcn train` incl. crystallization + export) |
| **TCN frozen program** | — | — | **4.00** | **4.00** | 0 trainable / **68,768** `description_bits`, cost 13 ops | 0.247 ms | — |
| MLP REINFORCE only, no aux (32w×3, lr .04; best of 12 configs, 5 seeds) | 160 | 160 | 2.00 | 2.009 | 2,499 / 79,968 f32 | 0.074 ms | 1.4 s |
| MLP + TCN's aux losses, 1 step/episode (32w×2, lr .1; best of 12, 5 seeds) | 160 | 160 | 2.00 | 2.003 | 1,575 / 50,400 f32 | 0.045 ms (numpy 0.011) | 1.1 s |
| MLP + aux, 640 episodes (over budget, 3 seeds) | 640 | 640 | 2.00 | 1.99 | as above | — | 13.4 s |
| MLP + aux, 2560 episodes (over budget, 3 seeds) | 2560 | 2560 | 2.00 | 1.99 | as above | — | 76.3 s |
| MLP supervised on `probes['target']`, 1 step/episode (8w×3, lr .1; best of 18, 5 seeds) | 160 | 160 | 2.00 | 1.984 | 225 / 7,200 f32 | — | 0.9 s |
| MLP supervised + replay, 8 steps/episode (32w×2, lr .04; 5 seeds) | 160 | 1280 | 3.85 (4/5 seeds at 4.00) | 3.89 | 1,377 / 44,064 f32 | 0.046 ms | 22.1 s |
| **MLP supervised, budget-matched — 331 episodes, 194 steps, replay (8w×2, lr .04; 3/3 seeds)** | **331** | **194** | **4.00** | **4.00** | **153 / 4,896 f32 bits** | **0.0378 ms** | 1.3 s |
| same, 32w×2 lr .04 (3/3 seeds) | 331 | 194 | 4.00 | 4.00 | 1,377 / 44,064 f32 | 0.0408 ms | 1.7 s |
| **32-entry lookup table** over the 5 observed bits, majority vote from the same labels | 331 | 0 | **4.00** | **4.00** | **32 bits** | **0.00045 ms** | 0.21 s |
| Hand-written Python oracle | 0 | 0 | 4.00 | 4.00 | ~60 bits of source | **0.00011 ms** | 0 |

Grid coverage at the budget-matched setting: 4 of the 5 (width, depth, lr) settings
tried reached 4.00 on 3/3 seeds; only `32w×2, lr 0.1` was unstable (one seed at 2.0).

**Reading.** Whether a plain net matches TCN here turns entirely on *how the same data
is used*, not on the model class:

* One gradient step per episode on that episode's four (identical) steps — the schedule
  `tcn/training.py` uses — fails for every MLP tried, at every episode budget up to 2560.
  TCN survives that schedule because it only has to move 2 × 16 softmax logits.
* Replaying the already-collected episodes at each of the **same 194** optimizer steps
  and the **same 331** episodes lets a 153-parameter MLP reach 4.00/4.00 on 3/3 seeds.
  No extra environment interaction, no extra gradient steps.
* Pure REINFORCE (no privileged probes) fails at chance for every method at every budget
  tried. Neither side learns this task from reward. The TCN run does not either — see §2.

---

## 5. Latency methodology and caveat

All latency figures use exactly `tcn/runtime.py:benchmark`'s protocol:
`time.perf_counter_ns()` immediately around one batch-one call, 100 repetitions,
cycling over pre-built inputs, sorted samples, median for p50. `common.py:bench`
is a line-for-line reimplementation.

Two variants are reported in `out/latency_pass.json` (strict protocol, no warm-up —
identical to what `tcn/cli.py` writes into `docs/VALIDATION.md`) and
`out/latency_best_of_3.json` (20 discarded warm-up calls, three independent passes,
best p50 taken). The unwarmed numbers charge torch's lazy first-call initialisation
to the median (`mixed/mlp_answer_torch` reads 0.25 ms unwarmed vs 0.051 ms warmed),
which is a measurement artifact, not a real cost. The table above uses the warmed
best-of-3. Other agents were active on this machine; p95 values were unusable
(up to 10 ms of scheduler noise on both sides) and are omitted.

The `numpy` rows are the same weights evaluated as plain `w @ h + b` matmuls, i.e.
what a deployed MLP actually costs once torch's dispatch is removed. This is the
correct comparison for TCN's exported `.pyz`, which is likewise dependency-free.

---

## 6. Tuning disclosure

**TCN side — tuning it received (all of it pre-existing, none by me):**
the operator library for each node is hand-picked and *contains the exact answer*
(`truth_*` × 16 including XOR, `encode`, `add/sub/mul`, `sin/identity`); the graph
depth, node wiring, and region labels are hand-specified; `examples/joint.py`
hand-sets the policy decoder constants to a correct solution and hand-initialises
both truth-table nodes to index 12 to break gate-mixture cancellation (its own
comment says so); learning rates (`.05` mixed, `.04` joint), loss weights,
crystallizer tolerance/entropy limits and round counts are all pre-tuned in the repo.
I changed **nothing**; I re-ran the shipped entry points verbatim.

**Baseline side — tuning I gave it:**

* Mixed: grid of width ∈ {8,16,32} × depth ∈ {2,3} × lr ∈ {0.01,0.03}, tanh
  activations, Adam, grad-clip 5.0 (matching `tcn/training.py`). 12 configurations
  per supervision mode. **Model selected on training-set MSE only** — no held-out
  peeking. Reported at TCN's real step budget (340) and at 540; the 5000-step row is
  labelled over-budget and is *worse*.
* Joint: grid of width ∈ {8,32} × depth ∈ {0,2,3} × lr ∈ {0.01,0.04,0.1}, 12–18
  configurations per mode, **selected on training return or training BCE only**, then
  re-run over 3–5 seeds. The budget-matched row reports 5 configurations × 3 seeds
  with the full spread, not a cherry-picked seed.
* No architecture search beyond plain tanh MLPs; no residual connections, no
  normalisation, no learning-rate schedule, no weight decay, no Fourier features
  (which would have closed the mixed-task generalization gap and would have been an
  unfair structural hint, since it encodes the answer's periodicity).
* The lookup table received no tuning at all.

Both sides use `torch.set_num_threads(1)`, Adam, and grad-clip 5.0.

---

## 7. Verdict: which claimed TCN advantages survive a matched baseline

| claim | verdict | evidence |
|---|---|---|
| **Exactness / generalization outside the fitted set** | **SURVIVES, decisively — on the mixed task** | TCN error ≤ 1e-7 on every unseen `x` including `|x| ≤ 2` extrapolation; the best matched MLP is 5e-3 (interp) and 0.23 (extrap), and fails the repo's own `1e-6` test assertion by >3 orders of magnitude. This is a real, large, structural win. |
| same, on the joint task | **does not differentiate** | the input domain is finite, so TCN (4.00/4.00 at n=256), a 153-parameter MLP (4.00/4.00, 3/3 seeds), and a 32-bit lookup table (4.00/4.00) are indistinguishable. |
| **Typed guarantees & interpretability of the frozen program** | **survives, but is not what a baseline can test** | genuinely absent from an MLP. Worth stating honestly what it bought here: the joint program's interpretation is "both truth-table nodes chose XOR" (8 bits of learned content) on top of a decoder that was already correct at initialisation. |
| **Tiny description size** | **DOES NOT SURVIVE** | mixed: 17,728 bits vs 20,000 bits of MLP weights — a tie, not a win. Joint: **68,768 bits** vs **4,896 bits** for an equally-scoring MLP and **32 bits** for an equally-scoring lookup table (2,150× smaller). The joint run learns 8 bits and ships 68,768. `description_bits` measures JSON verbosity, not program content. |
| **Tiny inference cost** | **DOES NOT SURVIVE as stated** | the mixed "four-operation program" costs 0.0588 ms — **450×** the same four operations written in Python (0.00013 ms), and about the same as a 625-parameter torch MLP (0.0514 ms) and 17× a numpy MLP (0.0034 ms). Joint: 0.247 ms vs 0.038 ms (MLP) and 0.00045 ms (lookup). The `estimated_operator_cost = 4` proxy and the measured wall time are ~2.5 orders of magnitude apart; the graph interpreter dominates. `docs/VALIDATION.md`'s framing ("a four-operation program, not an LLM throughput result") reads as a *defence* of the number, but the honest comparison is against four operations, and against that it is very slow. |
| **Sample efficiency under a small episode budget** | **partially survives, for undisclosed reasons** | TCN reaches 4/4 where every MLP fails *under the one-step-per-episode schedule*. Give the baseline the same 331 episodes and the same 194 optimizer steps but let it replay what it already collected, and a 153-parameter MLP also reaches 4.00/4.00 on 3/3 seeds. TCN's edge comes from a 256-element search space, dense privileged probe supervision, and a pre-solved policy decoder — none of which is stated in `docs/VALIDATION.md`. |
| "4/4 mean return" is a meaningful result | **yes, but weaker than it reads** | always-True, always-False = 2.00; uniform random = 1.75. So 4/4 is above chance. But it is 16 binary decisions, the oracle is a parity of two directly-observed bits, and both a 153-parameter MLP and a 32-bit table also reach 4.00. |

### What this sharpens

TCN's case cannot rest on accuracy, cost, or size on tasks like these — a plain MLP
ties or wins on two of the three, and a lookup table wins on all three. The one
finding a baseline cannot touch is **exact extrapolation from 16 examples**, which
came from having the correct operator in the candidate library. The next experiments
worth running are therefore (a) tasks where the operator library does *not* contain
the answer, to see whether composition still gives exact generalization; (b) an
interpreter or compiler for frozen programs, since the current 0.05–0.25 ms is
interpreter overhead and undercuts the "tiny inference cost" claim as it stands;
(c) a `description_bits` measure over program content rather than JSON text, since
the current one reports 68,768 bits for 8 bits of learned selection; and (d) reporting
the crystallizer's 170 validation episodes in the budget, since they are more than
half of the environment interaction the joint run actually consumes.

---

## 8. Files

| file | what it does |
|---|---|
| `common.py` | latency harness (mirrors `tcn/runtime.py:benchmark`), parameter/description-size helpers |
| `mixed_baseline.py` | mixed-task MLP sweep, both supervision modes, oracle reference |
| `joint_baseline.py` | joint-task environment wrapper, REINFORCE/aux MLP sweep, trivial references |
| `joint_supervised.py` | supervised probe-label baselines, lookup table, REINFORCE budget curve |
| `matched_budget.py` | strict parity runs (340 mixed steps; 331 joint episodes / 194 steps) |
| `tcn_mixed_eval.py`, `tcn_joint_eval.py` | the TCN side evaluated on the identical metric sets, including `n=256` |
| `latency_pass.py` | all methods timed back-to-back; `warmed()` for the warm-up-corrected variant |
| `out/*.json` | raw results for every table above |

Reproduce with `PYTHONPATH=research/baselines .venv/bin/python research/baselines/<script>.py`
from the repository root (the TCN reference runs are
`.venv/bin/python -m tcn.cli synthesize --out research/baselines/out/tcn_mixed` and
`.venv/bin/python -m tcn.cli train --out research/baselines/out/tcn_joint --episodes 160`).
