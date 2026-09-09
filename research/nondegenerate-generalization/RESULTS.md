# Step 4 — structural generalization on a non-degenerate benchmark

Re-runs track 4's experiment now that `generators/logic` has a real difficulty
axis. Raw data in `out/results.json`, log in `run.log`.

## Benchmark, verified rather than asserted

`pools.py` checks both properties by exhaustive search:

- Track 4's `AFFINE` training pool `(0,3,5,6,9,10,12,15)` contains **all six**
  tables whose output ignores an input, so six of its eight members are
  constants or projections and a learner could score without reading the table.
- The replacement pools `A = (1,2,13,14)` and `B = (4,6,9,11)` are disjoint,
  drawn only from the ten tables that depend on both arguments, and balanced:
  no fixed input-independent gate exceeds 0.500 accuracy, so the analytic
  ceiling is exactly 2.00/4.

Stated limitation: only two non-degenerate tables are affine (6 and 9), and no
balanced disjoint 4-subset split places one in each pool, so pool B holds both.
Both directions are run so the confound is symmetric.

## A run was discarded

The first execution reported `seen` and `unseen` identical to two decimals in
all six conditions across all eight seeds. That was a fault in the pool
implementation: an explicit `table` was silently ignored when a pool was
active, so both schedules evaluated on the same random draw and "unseen" was
never unseen. Fixed, with two regression tests. The invalid log is kept as
`run-invalid-pool-override.log`, since those identical columns are the
diagnostic that exposed it.

## Results

Eight seeds, 64 held-out episodes per condition, deterministic evaluation.
Empirical baselines on the same evaluation episodes, which differ slightly from
the analytic ceiling because the sampled input assignments are not perfectly
balanced:

| pool | always true | always false | random | best constant |
|---|---|---|---|---|
| A | 2.03 | 1.97 | 2.16 | 2.13 |
| B | 1.44 | 2.56 | 1.91 | 2.56 |

| condition | seen | unseen | interpreter candidate chosen |
|---|---|---|---|
| record A->B | 2.03 | 1.95 | n/a |
| record B->A | 1.99 | 1.98 | n/a |
| lookup A->B | 3.73 | **3.74** | 8/8 |
| lookup B->A | 4.00 | **4.00** | 8/8 |
| lookup wired A->B | 3.46 | **3.38** | 8/8 |
| lookup wired B->A | 3.40 | **3.53** | 8/8 |

## Reading

**The recorded scaffold is at chance on a fair benchmark**, 1.95-2.03 against a
best constant of 2.13-2.56, and it cannot do better: its `relation` node is a
choice over a global logit vector, so one frozen program computes exactly one
Boolean relation regardless of the episode's table.

**The table-conditioned scaffold generalizes to gate families it never trained
on**, 3.38-4.00 against the same baselines, in both directions, including when
gate wiring also varies. The interpreter candidate was **selected in 8/8 seeds
in every lookup condition** — it was made legal, not supplied, so this is search
discovering the structure rather than being handed it.

`seen` and `unseen` are close throughout the lookup rows, which is the expected
signature here rather than a defect: a program that reads the table performs the
same on any table. The earlier run's identical columns were a different thing —
there the two evaluations were literally the same episodes.

## Discrete reference

Enumerating the lookup scaffold's own 272-program space, scored by actual return
on eight training episodes: exhausted in 130 s, selecting `relation = 16` (the
interpreter candidate) and `goal_relation = 6`, reaching **4.00 training and
4.00 held-out** return.

So enumeration solves this task too, and generalizes exactly as well. Consistent
with section 8 of `research/FINDINGS.md`: the gradient path's advantage here is
not solution quality. Its measured advantage on this family of tasks is
environment sample efficiency, which this comparison does not isolate — the
enumerator spends 272 x 8 rollouts to score candidates, and a fair
sample-efficiency comparison would hold rollouts rather than wall clock fixed.

## Limits

Eight seeds per condition, one gate depth, two-input circuits, and a single
generator. Depth generalization is untouched and remains blocked: the `program`
observation is 3 x depth wide, so a fixed-width typed program cannot accept an
episode of unseen depth without a recurrent or set-shaped gate encoding.
