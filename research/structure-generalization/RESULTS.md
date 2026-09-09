# Track 4 — Does the joint agent generalize to unseen gate families?

Run 2026-09-08 in this workspace with `.venv/bin/python`. All code, logs and raw
JSON are in this directory; nothing under `tcn/` or `generators/` was modified.

**Verdict up front.** The scaffold in `examples/joint.py` — the one the
validation record's 4/4 comes from — exhibits **no structural generalization,
and cannot**. It never observes which Boolean function the episode instantiates,
and its gate choice is a global parameter that cannot depend on the episode, so
one frozen program computes exactly one relation. Measured over 8 seeds it sits
at chance (2.09/4 on unseen gate families, 2.12/4 on the *seen* ones) against a
2.0/4 analytic ceiling, and at chance for every unseen depth.

Structural generalization **is** achievable in this substrate, but only after a
scaffold change: expose the generator's `program` observation and add one
`index`-over-truth-table candidate. That variant selects the table-conditioned
candidate in 8/8 seeds and, after progressive crystallization, its **exact frozen
program scores 4.00/4 on eight gate families it never trained on, in 8/8 seeds.**

---

## 1. Is the gate family visible to the agent? (answered before running)

`generators/logic/generator.py:observe()` emits three observation channels:

```
obs = {'bits':   product(BOOL x 4),                       # the four input bits
       'program': vector_value([a, b, table] per gate),   # wiring + truth table
       'goal':    BOOL}                                   # the objective bit
```

Measured, at `index=3`:

| configuration | `program` | `bits` | `target` probe |
|---|---|---|---|
| `{'depth':1,'table':6,'fixed_inputs':True}` | `(0.0, 1.0, 6.0)` | `(T,F,F,T)` | `True` |
| `{'depth':1,'table':11,'fixed_inputs':False}` | `(2.0, 1.0, 11.0)` | `(T,F,F,T)` | `True` |
| `{'depth':3,'fixed_inputs':False}` | `(2.0,1.0,6.0, 1.0,1.0,14.0, 3.0,1.0,7.0)` | `(T,F,F,T)` | `True` |

So the truth table **is** present in the observation channel as a scalar field —
the task is an interpretation task in principle. But `examples/joint.py` declares

```python
names = ('bits','goal')     # 'program' is NOT an input to the model
```

so **the record's scaffold does not receive it.** Two consequences, both
established before any training was run:

1. **Held-out tables are unlearnable in principle for that scaffold.** Nothing in
   its input distinguishes an XOR episode from an AND episode.
2. Worse, the *training* tables are unlearnable too. `relation` is a 16-way
   candidate choice over a global logit vector (`SoftProgram.choices`), not a
   function of the input; a frozen program computes one fixed `g(bit_0,bit_1)`.

### Analytic ceiling (`ceiling.py`, `out/ceiling.json`)

For a set of gate families `T`, the best any input-independent `g` can do is
`max_g mean_{t in T} mean_i [g_i = t_i]`. Both table sets used below are exactly
balanced at every input position (fraction-true `[0.5,0.5,0.5,0.5]`), so:

| table set | best fixed table | best accuracy | max return (horizon 4) |
|---|---|---|---|
| the record's single table 6 (XOR) | 6 | 1.00 | **4.00** |
| 8 affine tables `{0,3,5,6,9,10,12,15}` | any | 0.50 | **2.00** |
| 8 non-affine tables `{1,2,4,7,8,11,13,14}` | any | 0.50 | **2.00** |
| all 16 tables | any | 0.50 | **2.00** |

Chance is 2.00/4, so **the record scaffold's ceiling on any multi-table
distribution is exactly chance.** That is a property of the graph, not of
training.

### Why this table split

The 16 two-input tables split cleanly into the 8 **affine** functions
`f = c0 ⊕ c1·a ⊕ c2·b` (tables `0,3,5,6,9,10,12,15`: constants, projections,
negations, XOR, XNOR — the record's table 6 lives here) and the 8 **non-affine**
ones (`1,2,4,7,8,11,13,14`: AND/OR/NAND/NOR and variants). It is a real
structural boundary, and both halves are perfectly balanced, so the constant
baselines are 2.00/4 on both by construction. Both directions were run.

---

## 2. Method

- Generator: `logic`, horizon 4, two objectives `invert ∈ {False, True}`
  alternating by episode index, exactly as `examples/joint.py`.
- `research/structure-generalization/common.py` rebuilds the record scaffold
  from the public API and adds the variants. `ScheduledTrainer` subclasses
  `tcn.training.JointTrainer` so `generator_config` can be a *function of the
  episode* — `TrainConfig.generator_config` is one dict for a whole run, which
  is structurally why the record could only instantiate one Boolean function.
- Training: 160 episodes for the fixed-XOR reproduction (the record's budget),
  320 for every multi-table / multi-depth condition. Table cycles every two
  episodes so each table meets both objectives.
- Evaluation: **64 held-out episode addresses** (`index 10000..10063`,
  `split='test'`), deterministic policy (`train=False`, argmax), same call the
  record used. **8 seeds** for every number; sd and range across seeds reported.
- Baselines recomputed per eval set by actually rolling the generator with a
  constant or uniform-random `answer` action.

### Scaffold variants

| name | observations | `relation` node |
|---|---|---|
| **record** | `bits`, `goal` | 16 candidates `truth_i(bit_0, bit_1)` — the record's graph, unchanged |
| **lookup** | `bits`, `goal`, `program` | 17 candidates: the same 16, **plus** `index(⟨truth_0..truth_15(bit_0,bit_1)⟩, encode(program[2]))` |
| **interpreter** | `bits`, `goal`, `program` | 1 candidate: only the `index` lookup (upper bound / no search) |
| **+wiring** | as above | gate inputs are `index(bits, encode(program[0..1]))` instead of hard-wired `bits[0], bits[1]`, so wiring varies too |

All variants share the record's symmetry-breaking init (`p[12] = 2.`, "copy input
one") and its `z/world/policy/prediction/value` tail verbatim. No correct
operator is supplied to any 16- or 17-way choice.

---

## 3. Results — train condition × eval condition × mean return

Every row is 8 seeds × 64 held-out episodes. Maximum return is 4. **Chance is
2.0**; the constant baselines are measured on the same 64 episodes (they wobble
around 2.0 from finite sampling).

| train scaffold | train condition | eval condition | mean return | sd (seeds) | range | always-True | always-False | uniform random | best constant |
|---|---|---|---|---|---|---|---|---|---|
| record (`bits`,`goal`) | fixed table 6 (XOR), fixed wiring | fixed XOR, held-out episodes (16 — record budget) | **4.00** | 0.00 | 4.00–4.00 | 2.00 | 2.00 | 2.19 | 2.00 |
| record | fixed table 6 (XOR), fixed wiring | fixed XOR, held-out episodes (64) | **4.00** | 0.00 | 4.00–4.00 | 1.88 | 2.12 | 1.91 | 2.12 |
| record | 8 affine tables | affine tables (**seen**) | **2.12** | 0.19 | 1.69–2.25 | 2.12 | 1.88 | 2.22 | 2.12 |
| record | 8 affine tables | **non-affine tables (unseen)** | **2.09** | 0.20 | 1.88–2.44 | 2.25 | 1.75 | 2.03 | 2.25 |
| record | 8 non-affine tables | non-affine tables (**seen**) | **1.85** | 0.15 | 1.56–2.06 | 2.25 | 1.75 | 2.03 | 2.25 |
| record | 8 non-affine tables | **affine tables (unseen)** | **1.88** | 0.19 | 1.69–2.19 | 2.12 | 1.88 | 2.22 | 2.12 |
| lookup (17 cand.) | 8 affine tables | affine tables (seen) | **3.80** | 0.24 | 3.44–4.00 | 2.12 | 1.88 | 2.22 | 2.12 |
| lookup (17 cand.) | 8 affine tables | **non-affine tables (unseen)** | **3.68** | 0.32 | 3.31–4.00 | 2.25 | 1.75 | 2.03 | 2.25 |
| lookup (17 cand.) | 8 non-affine tables | non-affine tables (seen) | **3.77** | 0.32 | 3.25–4.00 | 2.25 | 1.75 | 2.03 | 2.25 |
| lookup (17 cand.) | 8 non-affine tables | **affine tables (unseen)** | **3.80** | 0.28 | 3.44–4.00 | 2.12 | 1.88 | 2.22 | 2.12 |
| interpreter (1 cand.) | 8 affine tables | affine tables (seen) | **3.80** | 0.24 | 3.44–4.00 | 2.12 | 1.88 | 2.22 | 2.12 |
| interpreter (1 cand.) | 8 affine tables | **non-affine tables (unseen)** | **3.70** | 0.33 | 3.31–4.00 | 2.25 | 1.75 | 2.03 | 2.25 |
| interpreter (1 cand.) | 8 non-affine tables | non-affine tables (seen) | **3.77** | 0.32 | 3.25–4.00 | 2.25 | 1.75 | 2.03 | 2.25 |
| interpreter (1 cand.) | 8 non-affine tables | **affine tables (unseen)** | **3.80** | 0.28 | 3.44–4.00 | 2.12 | 1.88 | 2.22 | 2.12 |
| lookup + wiring | 8 affine tables, random wiring | affine tables + wiring (seen) | **3.35** | 0.29 | 2.75–3.69 | 2.31 | 1.69 | 1.88 | 2.31 |
| lookup + wiring | 8 affine tables, random wiring | **non-affine + wiring (unseen)** | **3.34** | 0.48 | 2.44–3.94 | 1.94 | 2.06 | 1.84 | 2.06 |
| interpreter + wiring | 8 affine tables, random wiring | affine + wiring (seen) | **3.37** | 0.28 | 2.75–3.62 | 2.31 | 1.69 | 1.88 | 2.31 |
| interpreter + wiring | 8 affine tables, random wiring | **non-affine + wiring (unseen)** | **3.34** | 0.47 | 2.44–3.81 | 1.94 | 2.06 | 1.84 | 2.06 |
| record | depth 1–2, random wiring, random tables | depth 1 (seen) | **1.98** | 0.31 | 1.75–2.50 | 1.81 | 2.19 | 2.16 | 2.19 |
| record | depth 1–2, random wiring, random tables | depth 2 (seen) | **2.00** | 0.21 | 1.62–2.31 | 2.19 | 1.81 | 1.97 | 2.19 |
| record | depth 1–2, random wiring, random tables | **depth 3 (unseen)** | **2.16** | 0.20 | 1.94–2.56 | 2.12 | 1.88 | 2.06 | 2.12 |
| record | depth 1–2, random wiring, random tables | **depth 4 (unseen)** | **2.01** | 0.22 | 1.75–2.38 | 1.75 | 2.25 | 2.09 | 2.25 |

### Crystallized exact frozen agents (`crystal.py`, 8 seeds)

Progressive crystallization (`tcn.crystallize.Crystallizer`, tolerance .05,
entropy limit .9, 24 rounds) then `tcn.agent.Agent` deterministic rollout of the
exported exact program — the same pipeline `tcn/cli.py:joint` uses for the
record's "exact frozen agent … 4/4".

| scaffold | trained on | fully frozen | seen affine tables | **unseen non-affine tables** | best constant |
|---|---|---|---|---|---|
| record | 8 affine tables | 6/8 seeds | **1.91** (sd 0.23) | **2.01** (sd 0.14) | 2.13 / 2.25 |
| lookup (17 cand.) | 8 affine tables | 8/8 seeds | **4.00** (sd 0.00) | **4.00** (sd 0.00) | 2.13 / 2.25 |

### What got selected

| condition | `relation` selection per seed |
|---|---|
| record, fixed XOR | `truth_6` (XOR) in 8/8 — correct, hence 4/4 |
| record, 8 affine tables | `truth_12`×7, `truth_11`×1 — a single fixed gate, i.e. "copy `bit_0`" |
| record, 8 non-affine tables | `truth_12`×6, `truth_11`, `truth_3` |
| record, depth 1–2 | `truth_12` in 8/8 |
| **lookup, all four table conditions** | **candidate 16 — the table-conditioned `index` lookup — in 8/8 seeds** |

`goal_relation` converges to `truth_6` (XOR with the objective bit) in 8/8 seeds
in every table-conditioned condition, which is the correct goal conditioning.

---

## 4. Reading the numbers

**1. The reproduction holds, and it is exactly as narrow as it looks.** Record
scaffold, fixed table 6: 4.00/4 on 8/8 seeds, at both the record's 16-episode
budget and 64 episodes. The record's number is real. It measures learning
`truth_6` and `truth_6`, two 16-way choices, from a distribution whose only
variation is four input bits and one objective bit.

**2. On multiple gate families the record scaffold is at chance — on both the
unseen and the *seen* families.** 2.12 seen / 2.09 unseen (affine training),
1.85 seen / 1.88 unseen (non-affine training), against an analytic ceiling of
2.00 and constant baselines of 1.75–2.25. There is no train/test gap here
because there is no learning to lose: nothing above chance is representable.
The frozen agents land in the same place (1.91 / 2.01), and 2/8 seeds could not
even be fully crystallized.

**3. Depth generalization is chance at every depth, including the trained
ones.** 1.98 / 2.00 at trained depths 1–2, 2.16 / 2.01 at unseen depths 3–4, all
inside the baseline band. With `fixed_inputs=False` the gate wiring is random too,
so even the two bits the scaffold projects out are usually the wrong ones. There
is a second, harder obstruction: the `program` observation has width `3 × depth`
(3 at depth 1, 9 at depth 3 — measured above), so a *fixed-width typed program
cannot even accept episodes of a different depth*. Depth generalization in this
substrate needs a recurrent or set-shaped gate encoding, not just a wider input.

**4. The table-conditioned scaffold does generalize, and the search finds it.**
Adding one candidate — `index` over the 16 gate outcomes, addressed by the
observed truth-table field — takes held-out gate families from 2.09 to 3.68/3.80
(soft model) and to **4.00/4 exactly after crystallization, 8/8 seeds**. The
17-candidate version is statistically indistinguishable from the hand-built
1-candidate interpreter (3.68 vs 3.70, 3.80 vs 3.80), and picks the lookup
candidate in 8/8 seeds, so the choice is genuinely *learned* rather than
supplied. Seen and unseen families score the same, in both split directions,
which is what structural generalization is supposed to look like. The soft
model's shortfall below 4.00 is the smooth `index` relaxation
(`softmax(-(x-k)²/τ)` blurs neighbouring table indices); the exact frozen program
has no such blur and is perfect.

**5. Generalizing over wiring as well as table also works, at a cost.** Routing
the gate inputs through `index(bits, program[0..1])` holds up on unseen families
(3.34 unseen vs 3.35 seen) but loses ~0.45 return overall — two more soft
`index` relaxations in the path. Crystallized numbers were not collected for this
variant; expect them to be better.

**Caveats.** All of section 3 is one generator, horizon 4, two objectives, and
a two-input gate universe of 16 functions. The `index` candidate makes the
correct program *representable at depth 1*; the finding is that gradient search
plus crystallization reliably finds it and that it transfers to unseen gate
families, not that this scaffold family scales. Nothing here says anything about
depth, composition, or non-Boolean domains.

---

## 5. Proposed correction to `docs/VALIDATION.md`

The current text says:

> Deterministic evaluation achieved **4/4** mean return in 16 held-out episodes,
> across both objectives.

and later:

> This is a small fixed-structure task with held-out episode addresses. It does
> not test unseen relation families […]

The closing caveat is honest but the headline is not qualified where it is read.
ARCHITECTURE.md §9 requires holding out "generating structures … as well as
seeds", and this experiment holds out neither structure nor generating family.
Suggested replacement for the evaluation sentences:

> Deterministic evaluation achieved **4/4** mean return on 16 held-out episode
> addresses, across both objectives. Every episode — training and held-out —
> instantiates the *same* Boolean function (`{'depth':1,'table':6,
> 'fixed_inputs':True}`, i.e. XOR over two fixed inputs); the held-out episodes
> vary only the four input bits and the objective bit. This is a held-out-seed
> result, **not** the held-out-generating-structure test ARCHITECTURE.md §9
> requires. Measured directly (`research/structure-generalization/`), the same
> scaffold trained over eight gate families scores 2.09/4 on eight unseen
> families and 2.12/4 on the *seen* ones, against a 2.0/4 chance baseline and a
> 2.0/4 analytic ceiling: it observes only `bits` and `goal`, so its single
> global gate choice cannot depend on the episode's truth table. Exposing the
> generator's `program` observation and adding one table-conditioned `index`
> candidate lifts held-out gate families to 4.00/4 for the exact frozen agent in
> 8/8 seeds.

The "Unsuccessful and unestablished results" section should also gain a line:

> Structural generalization of the joint logic agent was tested in
> `research/structure-generalization/` and **failed for the recorded scaffold**;
> it succeeded only after a scaffold change that makes the gate identity an
> input. The recorded 4/4 should not be cited as evidence of generalization
> across relation families or depths.

---

## 6. Files

| file | what |
|---|---|
| `common.py` | scaffold variants, per-episode `ScheduledTrainer`, baselines |
| `run.py` | one condition × 8 seeds → `out/<condition>.json` |
| `crystal.py` | train → crystallize → exact frozen `Agent` eval → `out/crystal_*.json` |
| `ceiling.py` | analytic ceiling for an input-independent gate choice |
| `report.py` | renders `out/*.json` into the tables above |
| `out/*.json`, `out/logs/*.log` | raw per-seed returns, selections, baselines, stdout |

Reproduce: `for c in record_fixed_xor record_multitable_affine
record_multitable_nonaffine lookup_affine lookup_nonaffine interpreter_affine
interpreter_nonaffine lookup_wired_affine interpreter_wired_affine
record_depth12; do .venv/bin/python research/structure-generalization/run.py $c;
done` then `.venv/bin/python research/structure-generalization/crystal.py lookup`
and `... crystal.py record`.
