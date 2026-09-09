# Where typed synthesis breaks: a difficulty sweep on the `logic` generator

Research track 3. All code and data under `research/search-scaling/`.
Nothing in `tcn/` or `generators/` was modified.

---

## 0. Executive summary

1. **The `logic` generator's `depth` knob is not a difficulty knob.** Random gate tables
   make circuits collapse: at depth 8, 40% of sampled targets are *constant* functions and
   only 13% depend on more than two of the four input bits. Random uniform sampling of
   discrete programs from the free-wiring candidate space hits the exact target function
   with probability ~1e-2 (≈50–200 draws). On this distribution TCN synthesis succeeds
   **97% of the time at every depth from 1 to 16, even with fully free wiring**
   (249/256 runs) — because the problem is trivial, not because the method is strong.

2. **The real wall is target arity, not program depth.** Conditioning on targets that
   genuinely depend on all four input bits drops random-draw solution density below 1e-5
   and produces a cliff:

   | target depends on | n | free-wiring success |
   |---|---|---|
   | 0 bits (constant) | 29 | 100% |
   | 1 bit | 33 | 100% |
   | 2 bits | 50 | 100% |
   | 3 bits | 13 | 92% |
   | **4 bits** | **99** | **26%** |

3. **Free wiring costs ~4x, it does not fail outright.** On 4-relevant-bit targets,
   supplied wiring reaches 81–100% at depth ≤ 6 and 44–62% at depth 8–16; free wiring sits
   at **19–38% at every depth from 3 to 16**. It is a roughly constant multiplicative
   penalty, flat in depth, not a depth-dependent collapse.

4. **Restarts do not fix it, and more steps do not fix it.** With a *fixed* target and 24
   different initialisations, **7 of 12 hard targets are solved by zero of 24 restarts**.
   Raising the optimiser budget 32x (250 → 8000 steps) changes success by ≤ 6 pp and is
   non-monotone.

5. **Dense hierarchical supervision does fix it.** Probing every intermediate node (the
   `examples/mixed.py` supervision pattern, applied to a *free-wiring* scaffold) raises
   success from 19–38% to **88–94% at every depth 3…16**, at 172 bits of search space with
   2120 candidates on the widest node. This is the largest single effect measured here.

6. **The failure is neither "many local optima you can restart out of" nor "vanishing
   gradients".** It is *premature entropy collapse into a function-space attractor*, plus a
   distinct *relaxation/rounding gap*. Diagnostics in §6.

7. **No DLGN depth pathology.** Pass-through (wire) selection stays at **0–21% of live
   nodes at every depth 1…16**, against the published 88.4% reference and a 12.5% chance
   rate, with no upward trend in depth and no difference that favours supplied over free
   wiring. TCN's deep scaffolds degenerate a different way: nodes fall *out of the output
   cone* entirely (live-node fraction 1.00 → 0.38 from depth 1 to 16). Details in §7.

---

## 1. Method

**Task.** Sample a real episode from `generators/logic` (`Host.create('logic',
configuration={'depth': D})`), read its `gates` list, and evaluate the circuit on all 16
assignments of the 4 input bits to get the target's exact truth table. Synthesis must
recover a program computing that function.

**Scaffold.** Inputs `b0..b3 : bool` at depth 0; `N` nodes `g0..g_{N-1} : bool` at depths
1…N; output = `g_{N-1}`. Node candidates are `Candidate(resolve('truth_t', (BOOL,BOOL)),
(s1, s2))`, i.e. **operator x input-binding pairs**, exactly as `Program.validate` and
`legal_candidates` require.

- `supplied` wiring: `(s1,s2)` fixed to the target gate's actual sources (the
  `examples/joint.py` setting). Candidates per node = |table pool|.
- `free` wiring: `(s1,s2)` ranges over every legal predecessor pair. Candidates per node =
  |table pool| x |pool|². Node 15 has 2120 candidates; the depth-16 free scaffold is
  **172 bits** of discrete search space.

**Supervision.** `sparse` = one `Signal` (bce) on the output node only. `dense` = an
additional bce `Signal` on every intermediate node against the corresponding target gate's
value.

**Training.** Mirrors `tcn.synthesis.fit`: `SoftProgram`, Adam(lr=.05), loss =
`probe_loss` + `.001*(step/steps)*entropy()`, temperature annealed 1.0 → 0.1 geometrically
(the `Crystallizer.run` schedule). Success = the **argmax export** (`SoftProgram.export()`
= `Program.harden(selections())`) computes the target function exactly on all 16 rows.
`harness.py` runs this through the real `SoftProgram`; a spot check with
`verify=True` confirms `Program.execute` on the exported program agrees with the fast
truth-table evaluator on every run tested.

**Restricted targets.** Pool/window/arity restrictions are applied by *rejection sampling
real generator episodes*, so every target is a genuine `logic` circuit.

**Speed.** The literal `SoftProgram` loop evaluates each candidate as a separate Python
call: a depth-8 free run costs ~1900 s. `fast.py` evaluates all candidates of a node as one
batched tensor op. `check_fast.py` verifies step-for-step equivalence against the real
`SoftProgram` (max |loss difference| **1.8e-6** over 40 steps, identical argmax selections,
6 configurations x 3 seeds). The large sweeps (`F*` files) use the fast path; the `A_`,
`B_`, `C_`, `F_slack` files are the same protocol run through unmodified `SoftProgram` and
agree with it.

**Budget.** 1,392 fast runs + 324 `SoftProgram` runs, 16 seeds per cell (24 for the
fixed-target sweep, 12 for the `SoftProgram` sweeps).

---

## 2. The generator's difficulty dial is broken

`generators/logic/generator.py` picks each gate's truth table uniformly from 0…15. Tables
0 and 15 are constants, so a random deep circuit is an absorbing random walk toward
constants and low-arity functions.

Distribution of the number of input bits the target actually depends on (60 samples/depth):

| generator depth | 0 (constant) | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 1 | 13 | 22 | 25 | 0 | 0 |
| 2 | 12 | 16 | 25 | 7 | 0 |
| 3 | 17 | 10 | 31 | 2 | 0 |
| 4 | 14 | 16 | 24 | 4 | 2 |
| 6 | 19 | 11 | 20 | 9 | 1 |
| 8 | **24** | 14 | 14 | 6 | 2 |

Depth makes targets *simpler*, not harder.

Solution density of the free-wiring discrete space (fraction of uniformly random legal
programs computing the target; 100k draws x 8 targets per row):

| depth | default targets: median density | expected random draws | 4-relevant-bit targets: median density | expected random draws |
|---|---|---|---|---|
| 1 | 3.9e-02 | 25 | — (impossible at depth 1) | — |
| 2 | 7.0e-03 | 143 | — (impossible at depth 2) | — |
| 3 | 6.3e-03 | 160 | < 1e-05 | > 100,000 |
| 4 | 2.0e-02 | 50 | < 1e-05 | > 100,000 |
| 6 | 5.4e-03 | 185 | < 1e-05 | > 100,000 |
| 8 | 1.8e-02 | 57 | 5.0e-06 | 200,000 |

**Three orders of magnitude** separate the two target distributions. Every result below is
reported for both.

---

## 3. Axis 1 x Axis 3 — depth x wiring, default target distribution

`FA_depth_wiring.json`, 16 seeds/cell, 500 steps. "ever hit" = the argmax program was
correct at some checkpoint during training.

| depth | wiring | n | exact success | log2 space | cand/node | med steps to hit | ever hit | zero-loss but wrong | med final loss | med rows wrong/16 | med final H_norm | med sec | distinct wrong programs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | supplied | 16 | 16/16 (100%) | 4.0 | 16 | 5 | 100% | -- | 0.0000 | 0.0 | 0.000 | 1.7 | -- |
| 1 | free | 16 | 16/16 (100%) | 8.0 | 256 | 25 | 100% | -- | 0.0000 | 0.0 | 0.000 | 0.3 | -- |
| 2 | supplied | 16 | 16/16 (100%) | 8.0 | 16 | 18 | 100% | -- | 0.0000 | 0.0 | 0.000 | 0.9 | -- |
| 2 | free | 16 | 15/16 (94%) | 16.6 | 328 | 35 | 94% | 0/1 | 0.0000 | 0.0 | 0.000 | 1.4 | 1/1 sel, 1/1 fn |
| 3 | supplied | 16 | 15/16 (94%) | 12.0 | 16 | 30 | 94% | 0/1 | 0.0000 | 0.0 | 0.000 | 1.7 | 1/1 sel, 1/1 fn |
| 3 | free | 16 | 16/16 (100%) | 25.8 | 400 | 42 | 100% | -- | 0.0000 | 0.0 | 0.000 | 2.2 | -- |
| 4 | supplied | 16 | 15/16 (94%) | 16.0 | 16 | 30 | 94% | 0/1 | 0.0000 | 0.0 | 0.000 | 1.7 | 1/1 sel, 1/1 fn |
| 4 | free | 16 | 15/16 (94%) | 35.4 | 488 | 25 | 94% | 0/1 | 0.0000 | 0.0 | 0.000 | 2.5 | 1/1 sel, 1/1 fn |
| 6 | supplied | 16 | 16/16 (100%) | 24.0 | 16 | 22 | 100% | -- | 0.0000 | 0.0 | 0.000 | 2.1 | -- |
| 6 | free | 16 | 16/16 (100%) | 55.8 | 680 | 28 | 100% | -- | 0.0000 | 0.0 | 0.000 | 6.0 | -- |
| 8 | supplied | 16 | 16/16 (100%) | 32.0 | 16 | 5 | 100% | -- | 0.0000 | 0.0 | 0.000 | 3.5 | -- |
| 8 | free | 16 | 16/16 (100%) | 77.3 | 904 | 22 | 100% | -- | 0.0000 | 0.0 | 0.000 | 13.9 | -- |
| 12 | supplied | 16 | 15/16 (94%) | 48.0 | 16 | 20 | 94% | 0/1 | 0.0000 | 0.0 | 0.000 | 6.0 | 1/1 sel, 1/1 fn |
| 12 | free | 16 | 16/16 (100%) | 123.3 | 1448 | 40 | 100% | -- | 0.0000 | 0.0 | 0.000 | 35.2 | -- |
| 16 | supplied | 16 | 15/16 (94%) | 64.0 | 16 | 20 | 94% | 0/1 | 0.0000 | 0.0 | 0.000 | 16.2 | 1/1 sel, 1/1 fn |
| 16 | free | 16 | 15/16 (94%) | 172.3 | 2120 | 15 | 100% | 1/1 | 0.0000 | 0.0 | 0.000 | 45.4 | 1/1 sel, 1/1 fn |

Independently reproduced through unmodified `SoftProgram` (`A_depth_wiring.json`, 12 seeds,
141/144 success, 3610 s wall).

**Reading:** success never drops below 50% on this distribution, at any depth, under either
wiring. Median steps to convergence stays at 15–42 and does not grow with depth. Wall-clock
does grow — 0.3 s to 45 s in the fast path, 3 s to 1930 s through `SoftProgram` — because
the candidate count per node grows quadratically in the predecessor pool.

---

## 4. Axis 1 x Axis 3 — depth x wiring, targets that depend on all four bits

`FH_hard.json`, 16 seeds/cell, 500 steps, targets rejection-sampled to `relevant_inputs == 4`.

| depth | wiring | n | exact success | log2 space | cand/node | med steps to hit | ever hit | zero-loss but wrong | med final loss | med rows wrong/16 | med final H_norm | med sec | distinct wrong programs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | supplied | 16 | 13/16 (81%) | 12.0 | 16 | 45 | 81% | 0/3 | 0.0000 | 0.0 | 0.000 | 6.0 | 3/3 sel, 3/3 fn |
| 3 | free | 16 | **4/16 (25%)** | 25.8 | 400 | 85 | 31% | 0/12 | 0.1065 | 1.0 | 0.055 | 4.6 | 12/12 sel, 12/12 fn |
| 4 | supplied | 16 | 16/16 (100%) | 16.0 | 16 | 70 | 100% | -- | 0.0000 | 0.0 | 0.000 | 3.1 | -- |
| 4 | free | 16 | **6/16 (38%)** | 35.4 | 488 | 75 | 38% | 1/10 | 0.0760 | 1.0 | 0.035 | 7.5 | 10/10 sel, 10/10 fn |
| 6 | supplied | 16 | 13/16 (81%) | 24.0 | 16 | 85 | 81% | 0/3 | 0.0000 | 0.0 | 0.000 | 6.1 | 3/3 sel, 3/3 fn |
| 6 | free | 16 | **3/16 (19%)** | 55.8 | 680 | 95 | 19% | 1/13 | 0.1369 | 1.5 | 0.026 | 11.0 | 13/13 sel, 13/13 fn |
| 8 | supplied | 16 | **7/16 (44%)** | 32.0 | 16 | 80 | 44% | 0/9 | 0.0793 | 1.0 | 0.026 | 5.7 | 9/9 sel, 9/9 fn |
| 8 | free | 16 | **5/16 (31%)** | 77.3 | 904 | 125 | 38% | 1/11 | 0.0760 | 1.0 | 0.016 | 19.7 | 11/11 sel, 11/11 fn |
| 12 | supplied | 16 | 10/16 (62%) | 48.0 | 16 | 85 | 62% | 0/6 | 0.0000 | 0.0 | 0.000 | 7.3 | 6/6 sel, 6/6 fn |
| 12 | free | 16 | **4/16 (25%)** | 123.3 | 1448 | 92 | 44% | 4/12 | 0.0388 | 1.0 | 0.011 | 48.3 | 12/12 sel, 12/12 fn |
| 16 | supplied | 16 | 10/16 (62%) | 64.0 | 16 | 68 | 62% | 0/6 | 0.0000 | 0.0 | 0.000 | 18.8 | 6/6 sel, 4/6 fn |
| 16 | free | 16 | **3/16 (19%)** | 172.3 | 2120 | 85 | 25% | 4/13 | 0.0560 | 1.0 | 0.008 | 70.5 | 13/13 sel, 13/13 fn |

**Reading:**

- **Free wiring is below 50% at the smallest depth tested (3 gates, 25%).** There is no
  depth at which free wiring on a genuinely 4-ary target is reliable.
- Free wiring is **flat in depth** (19–38%), not decaying. The penalty is a constant factor
  of roughly 3–4x relative to supplied wiring at depth ≤ 6.
- Supplied wiring is the setting `examples/joint.py` uses. It stays ≥ 81% up to depth 6 and
  first drops below 50% at **depth 8 (44%)**.
- Median failure is **1 row wrong out of 16** — near misses, not garbage.
- Every failed run in a cell produced a **distinct selection vector** (e.g. 13/13),
  so it is not one deterministic wrong answer at the program level.

---

## 5. Axis 2 — candidate pool size is not the binding constraint

### 5a. Operator pool (depth 4, targets rejection-sampled to the pool), `FB_pool.json`

`T2 = {xor, and}`, `T4 = {nor, xor, and, or}`, `T8`, `T16 =` all 16 tables.

| pool | wiring | n | exact success | log2 space | cand/node | med steps to hit | med sec |
|---|---|---|---|---|---|---|---|
| T2 | supplied | 16 | 16/16 (100%) | 4.0 | 2 | 5 | 2.4 |
| T2 | free | 16 | 12/16 (75%) | 23.4 | 61 | 15 | 4.1 |
| T4 | supplied | 16 | 16/16 (100%) | 8.0 | 4 | 10 | 0.9 |
| T4 | free | 16 | 13/16 (81%) | 27.4 | 122 | 30 | 0.6 |
| T8 | supplied | 16 | 15/16 (94%) | 12.0 | 8 | 20 | 0.6 |
| T8 | free | 16 | 14/16 (88%) | 31.4 | 244 | 32 | 1.2 |
| T16 | supplied | 16 | 15/16 (94%) | 16.0 | 16 | 30 | 0.3 |
| T16 | free | 16 | 15/16 (94%) | 35.4 | 488 | 25 | 1.0 |

Shrinking the operator pool 8x **lowers** free-wiring success (94% → 75%). The pool size is
confounded with target hardness in the right direction: `T2 = {xor, and}` contains no
constant table, so `T2` circuits do not collapse and their targets are harder. This is
independent evidence for §2: **functional complexity of the target, not the size of the
candidate set, determines difficulty.**

### 5b. Predecessor window (hard targets, free wiring), `FC_window.json`

A node may bind only to the `w` most recent ports.

| depth | window | cand/node | n | exact success |
|---|---|---|---|---|
| 6 | 4 | 256 | 14 | 1/14 (7%) |
| 6 | 5 | 400 | 16 | 0/16 (0%) |
| 6 | 6 | 576 | 16 | 4/16 (25%) |
| 6 | 8 | 680 | 16 | 4/16 (25%) |
| 6 | none | 680 | 16 | 3/16 (19%) |
| 8 | 5 | 400 | 14 | 4/14 (29%) |
| 8 | 6 | 576 | 16 | 5/16 (31%) |
| 8 | 8 | 904 | 16 | 4/16 (25%) |
| 8 | none | 904 | 16 | 5/16 (31%) |

Cutting the binding pool by 3.5x buys nothing (and the narrowest windows are worse, because
a narrow window forces a longer chain to reach `b0`). Cells with n < 16 are ones where
rejection sampling could not find enough 4-ary targets inside the window; those are recorded
as capped, not extrapolated.

### 5c. Scaffold slack (depth-4 hard target, free wiring), `FS_slack.json`

| scaffold nodes | n | exact success | log2 space |
|---|---|---|---|
| 4 (exact) | 16 | 6/16 (38%) | 35.4 |
| 6 | 16 | 2/16 (12%) | 55.8 |
| 8 | 16 | 4/16 (25%) | 77.3 |
| 12 | 16 | 6/16 (38%) | 123.3 |

Over-provisioning the scaffold does not help.

---

## 6. What kind of failure is it?

### 6a. It is not step-limited (`FE_budget.json`)

| depth | 250 steps | 500 steps | 2000 steps | 8000 steps |
|---|---|---|---|---|
| 4 | 38% | 38% | 44% | 38% |
| 6 | 19% | 19% | 25% | 25% |
| 8 | 19% | 31% | 19% | 12% |

32x the optimiser budget moves success by ≤ 6 pp and non-monotonically. Wall-clock at depth
8 goes 1.1 s → 73.9 s for *worse* results.

### 6b. It is not an annealing artifact (`FT_anneal.json`)

| depth | no annealing (tau=1) | tau→0.1 | tau→0.02 | tau→0.005 |
|---|---|---|---|---|
| 4 | 38% | 38% | 38% | 38% |
| 6 | 19% | 19% | 19% | 19% |
| 8 | 12% | 31% | 25% | 19% |

The `Crystallizer` temperature schedule is not the cause and a harsher schedule is not the
cure.

### 6c. It is not seed luck: restarts do not help (`FG_restarts.json`)

**Fixed target, 24 different `torch.manual_seed` initialisations each.**

| depth | target | solved/24 | distinct wrong **functions** | modal wrong fn share | distinct wrong **selection vectors** | soft loss ≈ 0 among failures |
|---|---|---|---|---|---|---|
| 4 | t0 | **0/24** | 3 | 46% | 24 | 0/24 |
| 4 | t1 | 7/24 | 5 | 35% | 17 | **16/17** |
| 4 | t2 | **0/24** | 3 | 50% | 24 | 0/24 |
| 4 | t3 | **0/24** | 5 | 33% | 24 | 0/24 |
| 6 | t0 | **0/24** | 7 | 21% | 24 | 0/24 |
| 6 | t1 | **0/24** | 5 | 33% | 24 | 0/24 |
| 6 | t2 | 1/24 | 3 | 61% | 23 | 0/23 |
| 6 | t3 | 2/24 | 3 | 41% | 22 | 0/22 |
| 8 | t0 | 7/24 | 7 | 29% | 17 | **17/17** |
| 8 | t1 | **0/24** | 3 | 58% | 24 | 0/24 |
| 8 | t2 | 12/24 | 5 | 42% | 12 | 0/12 |
| 8 | t3 | **0/24** | 8 | 42% | 24 | 0/24 |

Targets solved by at least one of 24 restarts: **5/12**. Per-restart success: 29/288 = 10%.

This is the decisive table. **7 of 12 hard targets are unreachable by any of 24 random
restarts.** Restart-based search is not a fix. And note the shape of the failures: every
failing run picks a *different program* (17–24 distinct selection vectors) but they land on
only **3–8 distinct functions**, with one modal wrong function absorbing 21–61% of them.

**Combinatorially diverse in program space, dynamically convergent in function space.**
That is neither of the two hypotheses in the brief. The optimiser is not sampling many
local optima at random (which restarts would fix); it is being pulled to a small set of
attractor functions determined by the target and the scaffold, not by the seed.

### 6d. Choice-entropy trajectories (median over runs in a cell)

Normalised entropy `H / log|K_v|`, averaged over nodes. Hard targets, free wiring, depth 6:

```
step        0    20    40    60    80   100   120   140   180   240   300   400   499
SOLVED  (n=3)
 H_norm  0.999 0.942 0.817 0.710 0.477 0.334 0.218 0.146 0.068 0.022 0.002 0.000 0.000
 loss    0.692 0.433 0.300 0.231 0.143 0.026 0.001 0.000 0.000 0.000 0.000 0.000 0.000
 |grad|  0.011 0.007 0.004 0.010 0.008 0.008 0.001 0.000 0.001 0.001 0.000 0.000 0.000
FAILED  (n=13)
 H_norm  0.999 0.934 0.795 0.638 0.503 0.372 0.258 0.198 0.097 0.043 0.033 0.026 0.027
 loss    0.693 0.516 0.376 0.282 0.226 0.175 0.154 0.152 0.152 0.152 0.152 0.152 0.152
 |grad|  0.008 0.006 0.006 0.008 0.004 0.003 0.001 0.001 0.001 0.000 0.000 0.012 0.013
```

Failed runs are **not** stuck at high entropy and are **not** oscillating. They commit
(H_norm 0.026 = near one-hot) by step ~200 and then the loss is *exactly* flat at a positive
value for the remaining 300 steps while the gradient norm sits at 1e-4. The correct
description is **premature entropy collapse into a flat, confidently-wrong basin**: choice
concentration outruns loss reduction, and once concentrated the softmax saturates and there
is no gradient left to escape with. That is why more steps (§6a) do nothing.

### 6e. A second, distinct failure: the relaxation/rounding gap

Depth 8, fixed target t0 — solved and failed runs have **identical loss and entropy
curves**:

```
step        0    20    40    60    80   100   120   140   180   240   400   499
SOLVED  (n=7)
 H_norm  0.999 0.940 0.806 0.714 0.559 0.410 0.276 0.148 0.035 0.000 0.000 0.000
 loss    0.693 0.391 0.200 0.137 0.070 0.012 0.001 0.000 0.000 0.000 0.000 0.000
FAILED  (n=17)
 H_norm  0.999 0.938 0.803 0.689 0.538 0.412 0.285 0.157 0.045 0.025 0.025 0.019
 loss    0.693 0.390 0.202 0.129 0.057 0.010 0.001 0.000 0.000 0.000 0.000 0.000
```

**All 17 failures reach zero BCE loss** with a near-one-hot mixture, and 92% of them passed
through a correct argmax program at some checkpoint before ending on a wrong one. The
continuous relaxation attains the target exactly with a *mixture* whose argmax is a
different function. `SoftProgram.export()` then rounds to that wrong program.

This is the failure mode that the `Crystallizer` conformance gate is designed to catch
(`try_freeze` rejects on `runtime conformance`), and it is real and common:
`zero-loss but wrong` accounts for 4/12 and 4/13 of failures at depth 12 and 16 free wiring
(§4), and 16/17 and 17/17 for two of the fixed targets. **A low soft loss is not evidence
of a correct program**, and any pipeline that reports the soft loss as the result is
reporting the wrong number.

### 6f. What does fix it: dense hierarchical supervision (`FD_supervision.json`)

Free wiring, 4-relevant-bit targets, one bce probe on every intermediate node:

| depth | free + output-only (§4) | free + dense probes | log2 space | cand/node |
|---|---|---|---|---|
| 3 | 25% | **94%** | 25.8 | 400 |
| 4 | 38% | **94%** | 35.4 | 488 |
| 6 | 19% | **94%** | 55.8 | 680 |
| 8 | 31% | **94%** | 77.3 | 904 |
| 12 | 25% | **88%** | 123.3 | 1448 |
| 16 | 19% | **94%** | 172.3 | 2120 |

Dense supervision is **flat in depth at ~94%** across a 6.5x range of search-space size,
while output-only supervision is flat at ~25%. Its entropy trace shows why:

```
depth 12, dense, solved (n=14)
 step     0    20    40    60    80   100   140   200   260   320   400   499
 H_norm 0.999 0.934 0.779 0.631 0.445 0.355 0.298 0.191 0.123 0.029 0.004 0.000
 loss   8.320 5.928 3.649 1.834 0.542 0.055 0.003 0.000 0.000 0.000 0.000 0.000
```

Entropy *plateaus* around 0.30–0.36 between steps 100 and 200 instead of collapsing
monotonically — the intermediate probes hold the choices soft while the per-node targets
are still being satisfied, and only then does it concentrate. Output-only supervision
collapses on a straight line and commits before the output is right.

**Interpretation.** The joint and mixed demonstrations in this repo are not small because
free wiring is impossible; they are small because output-only supervision is the weak
setting. `examples/mixed.py` supervises all four nodes; that is why it works. The
architecture's §7 hierarchical-supervision interface is doing the load-bearing work, and it
scales to free wiring at 172 bits.

---

## 7. Pass-through diagnostic (DLGN depth-pathology comparison)

Reference point from the literature track: deep differentiable logic gate networks select
**identity/pass-through in 88.4% of gates**, diagnosed as topology-induced credit
degradation from *random* inter-layer wiring. TCN's bindings are typed and drawn from
bounded predecessor pools, so the question is whether structured binding removes the
pathology.

Taxonomy over the 16 tables (index i = 2a+b): **wire** = `truth_12` (=A) and `truth_10`
(=B) — the direct DLGN pass-through analogue, chance rate 2/16 = **12.5%**; **constant** =
`truth_0`, `truth_15`; **unary** = `truth_3`, `truth_5`; **non-computing** = any of those,
or a node whose two sources are the same port. Measured on the argmax-selected program,
restricted to nodes in the transitive cone of the output (`live`). `passthrough.py`
reconstructs each scaffold deterministically from (config, seed); no retraining.

| sweep | depth | wiring | live-node fraction | **wire (pass-through)** | constant | unary | non-computing | target circuit's own wire rate |
|---|---|---|---|---|---|---|---|---|
| default targets | 1 | free | 1.00 | 18.8% | 6.2% | 31.2% | 75.0% | 18.8% |
| | 1 | supplied | 1.00 | 25.0% | 12.5% | 18.8% | 75.0% | 18.8% |
| | 2 | free | 0.97 | 0.0% | 6.2% | 9.4% | 34.4% | 12.5% |
| | 2 | supplied | 0.72 | 9.4% | 3.1% | 0.0% | 34.4% | 12.5% |
| | 3 | free | 0.75 | 4.2% | 0.0% | 16.7% | 49.0% | 10.4% |
| | 3 | supplied | 0.54 | 15.6% | 0.0% | 3.1% | 42.7% | 10.4% |
| | 4 | free | 0.77 | 12.0% | 7.8% | 5.2% | 42.2% | 10.9% |
| | 4 | supplied | 0.48 | 2.1% | 14.6% | 6.2% | 41.7% | 10.9% |
| | 6 | free | 0.59 | 17.7% | 9.6% | 11.9% | 51.6% | 10.4% |
| | 6 | supplied | 0.45 | 10.9% | 14.1% | 12.0% | 46.9% | 10.4% |
| | 8 | free | 0.54 | 18.3% | 8.9% | 14.6% | 59.1% | 8.6% |
| | 8 | supplied | 0.45 | 10.9% | 25.6% | 4.9% | 52.5% | 8.6% |
| | 12 | free | 0.42 | 21.1% | 5.7% | 7.6% | 45.4% | 8.9% |
| | 12 | supplied | 0.38 | 12.8% | 17.0% | 8.4% | 45.5% | 8.9% |
| | 16 | free | 0.38 | 10.4% | 5.4% | 7.3% | 46.6% | 10.5% |
| | 16 | supplied | 0.41 | 10.4% | 18.8% | 12.1% | 50.6% | 10.5% |
| hard targets | 3 | free | 0.85 | 9.4% | 0.0% | 2.1% | 18.8% | 0.0% |
| | 3 | supplied | 1.00 | 0.0% | 4.2% | 2.1% | 6.2% | 0.0% |
| | 4 | free | 0.81 | 3.6% | 0.0% | 0.0% | 11.5% | 3.1% |
| | 4 | supplied | 0.81 | 0.0% | 0.0% | 1.6% | 3.1% | 3.1% |
| | 6 | free | 0.61 | 9.6% | 0.0% | 13.0% | 30.5% | 6.2% |
| | 6 | supplied | 0.67 | 7.2% | 6.2% | 5.6% | 20.3% | 6.2% |
| | 8 | free | 0.63 | 2.1% | 0.0% | 8.5% | 26.7% | 5.5% |
| | 8 | supplied | 0.51 | 8.4% | 5.9% | 3.3% | 19.0% | 5.5% |
| | 12 | free | 0.40 | 8.8% | 0.0% | 4.7% | 26.0% | 10.9% |
| | 12 | supplied | 0.43 | 6.3% | 8.5% | 6.6% | 22.9% | 10.9% |
| | 16 | free | 0.43 | 1.0% | 0.0% | 4.3% | 27.5% | 12.1% |
| | 16 | supplied | 0.39 | 6.1% | 4.8% | 5.3% | 22.1% | 12.1% |
| hard + dense probes | 3 | free | 0.96 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | 4 | free | 0.83 | 3.1% | 0.0% | 4.7% | 7.8% | 3.1% |
| | 6 | free | 0.58 | 4.1% | 0.0% | 2.8% | 6.9% | 6.2% |
| | 8 | free | 0.45 | 6.2% | 0.0% | 5.9% | 13.8% | 5.5% |
| | 12 | free | 0.38 | 10.0% | 0.0% | 5.3% | 19.7% | 10.9% |
| | 16 | free | 0.32 | 5.7% | 2.8% | 8.8% | 20.6% | 12.1% |

**Findings.**

1. **The 88.4% pass-through pathology does not appear.** Pass-through selection stays in
   the **0–21%** band at every depth from 1 to 16, in both wiring arms and both target
   distributions — i.e. at or below the 12.5% chance rate for most cells, and always within
   a factor of ~2 of it. There is no monotone rise with depth. On hard targets under free
   wiring the trend is if anything *downward* (9.4% at depth 3 → 1.0% at depth 16).

2. **Learned wire rate tracks the ground-truth circuit's own wire rate.** The last column
   (the fraction of the *target's* gates that are `A`/`B` tables) moves in the same 0–19%
   band and the learned rate follows it. The scaffolds are selecting pass-through about as
   often as the correct answer requires, not as an escape hatch.

3. **Free wiring does not show the pathology and supplied wiring does not protect against
   it** — the arms are indistinguishable. So this data does **not** isolate typed structured
   binding as the mechanism that removes DLGN's pathology: at this scale the pathology is
   simply absent everywhere. That is a negative result on the mechanism question, and it
   should be labelled as such rather than claimed as a win for typed binding.

4. **TCN degenerates a different way: node death, not wire chains.** The live-node fraction
   falls from 1.00 at depth 1 to **0.38–0.43 at depth 16** in every arm. Deep scaffolds do
   not fill up with identity gates — they route around most of their own nodes, which fall
   out of the output's transitive cone entirely. The unused capacity shows up as dead nodes,
   not as pass-through gates. In MDL terms this is arguably healthier (`execution_cost`
   still charges for them, but `description_bits` compresses better), and it is the more
   accurate statement of "wasted depth" for this architecture.

5. **Pass-through weakly correlates with failure, at nothing like 88%.** In the free-wiring
   hard-target arm, failed runs select wire more often than solved runs (12.5% vs 0.0% at
   depth 3; 11.8% vs 0.0% at 6; 10.6% vs 3.6% at 12). Dense supervision suppresses
   non-computing selections across the board (non-computing 0.0–20.6% vs 6.2–30.5% for
   sparse).

**Caveat on scale.** These scaffolds have at most 16 nodes. The DLGN result concerns
48k–1.8M gates. This is a genuine measurement of the same quantity under structured typed
binding, but at three to five orders of magnitude smaller. It is evidence that the pathology
is not intrinsic to soft-choice logic synthesis at small depth; it is not evidence that it
would not appear at DLGN scale. A scaled version of this measurement is the obvious
follow-up, and it needs a generator with more than 4 input bits (§9).

---

## 8. Direct answers to the brief

**At what depth does success drop below 50%?**

- On the `logic` generator's default target distribution: **never**, in either wiring arm,
  up to depth 16 and 172 bits of search space (97% overall, 249/256 runs). The generator's
  depth parameter does not produce hard targets.
- On targets that depend on all four input bits: **free wiring is already below 50% at
  depth 3** (25%) and stays at 19–38% through depth 16 — the crossing point is at the
  smallest depth that admits a 4-ary function. **Supplied wiring first drops below 50% at
  depth 8** (44%).
- Restated with the variable that actually predicts success — the arity of the target
  function: 100% at 0–2 relevant bits, 92% at 3, **26% at 4**.

**Does free wiring cost an order of magnitude in seeds/steps, or does it fail outright?**

Neither, exactly. It costs a **constant factor of ~3–4x in success rate** (81–100% → 19–38%
at matched depth), flat in depth, and that factor is **not buyable back with seeds or
steps**: 24 restarts leave 7/12 hard targets at 0/24, and 32x the step budget moves success
by ≤ 6 pp. So the honest statement is: free wiring costs roughly 4x more restarts *on the
targets it can solve at all*, and fails outright on a little over half of hard targets no
matter how many restarts you spend. It becomes as good as supplied wiring (88–94%) the
moment you add intermediate probes.

**Is the failure combinatorial or optimisation-dynamical?**

**Dynamical, and specifically two dynamical failures — not a many-local-optima story.**

- *Not combinatorial in the useful sense*: restarts don't help. 7/12 fixed targets are
  0/24. If the landscape were a rough surface with many basins and the solution in one of
  them, 24 restarts would find several.
- *Not vanishing gradients in the usual sense either*: gradients are healthy for the first
  ~120 steps and the loss falls from 0.69 to its plateau in that window.
- **Failure mode 1 — premature entropy collapse into a function-space attractor.**
  Choice entropy falls to near-one-hot (H_norm ≈ 0.02) by step ~200 while the loss is still
  positive; from then on the loss is *bit-for-bit flat* for 300 steps and |grad| ≈ 1e-4. The
  runs commit before they are correct and then cannot move. Across 24 seeds the *programs*
  chosen are all different (17–24 distinct selection vectors) but the *functions* computed
  collapse to 3–8, with a modal wrong function taking 21–61%. The attractor is in function
  space and is set by the target and scaffold, not the seed.
- **Failure mode 2 — relaxation/rounding gap.** For some targets every failure reaches
  **zero** soft BCE loss with a near-one-hot mixture whose argmax is a different function
  (17/17 failures on one target; 4/13 at depth 16). The soft optimum is not the discrete
  optimum, and 92% of those runs visited a correct argmax program mid-training and left it.

Both are distinct from "many local optima, seed-dependent" and from "gradients vanish". The
`Crystallizer`'s conformance gate is the existing machinery that catches mode 2; nothing in
the current codebase addresses mode 1.

---

## 9. Limitations, and proposed core changes (not applied)

**Limitations.**
- `generators/logic` hardcodes **4 input bits**, so the target space is only 2^16 functions
  and functional difficulty saturates at "depends on all 4 bits". This is why success is
  flat in depth rather than decaying. Every claim about depth here is a claim about
  *scaffold* size at fixed target arity.
- Success is measured on the argmax export, not through the full `Crystallizer` freeze
  schedule. The `A_`/`B_`/`C_` files are the only ones exercising real `SoftProgram`
  forwards end to end; the `F*` files use the equivalence-checked fast forward.
- No noise, no held-out inputs: all 16 rows are supervised, so this measures search, not
  generalisation.
- Pass-through numbers are at ≤ 16 nodes; see §7 caveat.

**Proposed diff 1 — `generators/logic/generator.py`: configurable input width.**
The single change that would make this generator a real difficulty dial.

```python
     def initialize(self,address,configuration):
-        rng=address.rng(); count=int(configuration.get('depth',3))
+        rng=address.rng(); count=int(configuration.get('depth',3))
+        width=int(configuration.get('inputs',4))
+        if not 1<=width<=16: raise ValueError('inputs must be 1..16')
         if not 1<=count<=64: raise ValueError('depth must be 1..64')
-        bits=[bool(rng.randrange(2)) for _ in range(4)]; gates=[]; values=bits[:]
+        bits=[bool(rng.randrange(2)) for _ in range(width)]; gates=[]; values=bits[:]
```

**Proposed diff 2 — same file: a non-degenerate table distribution.**
`table` is currently drawn uniformly from 0..15, which includes the two constants and makes
deep circuits collapse (§2). An opt-in setting would let a curriculum request circuits whose
difficulty actually grows with depth:

```python
-            table=int(configuration.get('table',rng.randrange(16)))
+            pool=configuration.get('tables')
+            table=int(configuration.get('table',
+                rng.choice(tuple(pool)) if pool else rng.randrange(16)))
```

with `{'tables': [1,6,7,8,9,14]}` (nor/xor/nand/and/xnor/or) as the natural
non-degenerate default for a synthesis curriculum.

**Proposed diff 3 — `tcn/synthesis.py`: report the discrete result, not the soft loss.**
`fit` already computes `exact_conformance`, but it also returns `'loss'` from the soft
graph, and §6e shows those disagree systematically. Recommend that any caller treats
`exact_conformance` as the result and never reports `loss` as evidence of success. No code
change strictly needed; this is a reporting discipline point for `docs/VALIDATION.md`.

**Suggested next experiment (does not need a core change).** Mode 1 is an entropy-schedule
problem: the `.001*(step/steps)*entropy()` term and the `Crystallizer`'s 0.8x-per-round
temperature decay both push toward commitment on a fixed clock, independent of whether the
loss has stopped falling. A loss-gated schedule — only anneal/concentrate while the loss is
still decreasing — is a one-line change in an experiment script and is the obvious first
thing to try against the 7/12 unreachable targets.

---

## 10. Files

| file | contents |
|---|---|
| `harness.py` | scaffold construction, target sampling from the real generator, the `SoftProgram` training loop, fast truth-table evaluator |
| `fast.py` | vectorised forward, semantically identical to `tcn.learning.relaxed(truth_k)` + softmax mixing |
| `check_fast.py` | equivalence proof vs. real `SoftProgram` (max |loss diff| 1.8e-6, identical argmax) |
| `fastrun.py`, `fast_sweeps.py` | fast sweep driver (`F*.json`) |
| `run_sweeps.py` | `SoftProgram` sweep driver (`A_`, `B_`, `C_`, `F_slack`) |
| `baseline.py` | random-program solution density, target-arity histograms |
| `passthrough.py` | pass-through / degeneracy diagnostic (§7) |
| `analyze.py`, `make_report.py` | table generation |
| `*.json` | per-run records: config, seed, success, first-hit step, final loss, entropy trace, gradient-norm trace, selections, target and achieved truth tables, wall-clock |
