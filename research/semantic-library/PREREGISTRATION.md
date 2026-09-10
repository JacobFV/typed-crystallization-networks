# Pre-registration — written and committed before any arm of this track was run

Written 2026-09-09, branch `research/semantic-library`.

## What had already been run in this directory when this was written

Nothing that bears on any outcome. Exactly three things:

1. `_paths.py` was written and `build_variant("C-minall")` / `build_variant("C-trace")`
   were called once, to confirm the two upstream tracks import cleanly and that
   §46's `out/corpora.json` is on disk (130 and 296 entries respectively).
2. The **class balance** of the six earlier tasks and of the two declared
   off-family tasks was computed from their truth tables alone — this fixes the
   constant baselines below and nothing else.
3. The truth table of the §44 distractor `D134` was printed to confirm it is the
   function `later.MINIMAL_BODIES["distractor"]` computes.

**No corpus was mined here, no module was published, no arm of any evaluation
task was executed, and no ranked table produced by this track was read.**
Everything I know about the ranked tables comes from §46's already-published
`research/premin-abstraction/RESULTS.md` §10, which is cited in the brief.

## Where this comes from

FINDINGS §46 recorded, **as an exploratory measurement outside its own
pre-registration**, that pooling mined fragments by `(arity, truth table)`
instead of by `Program.digest` moves `MAJ3` to rank 1 and lifts §44's later task
from 0 conforming to 144 — the hand-authored ceiling — with 18/24 tight and 8/8
wide gradient conformance. §46 explicitly declined to treat this as a
capability, requiring "a pre-registered replication with its own wrong-module
control".

FINDINGS §50 is the cautionary precedent: §47 Q3's equally exciting exploratory
finding was explained *entirely* by its control once one was run. **The default
assumption of this track is that semantic pooling is another §50**, and the arms
below are designed to make that visible if it is true.

## The hypothesis

> Pooling mined fragments by `(arity, truth table)` rather than by
> `Program.digest` **selects** a reusable abstraction — not merely *some*
> abstraction of the right size — and the selected abstraction helps a later
> task the rule did not mine from.

Two claims, tested separately. The second is the one that distinguishes an
*abstraction* result from a *compression* result, and it is the decisive one.

## What is fixed in advance

### Imported unchanged, not restated

* The later task, both scaffolds, the gradient loop, the accuracy and
  conformance functions and the hand-authored modules: `research/earned-abstraction/later.py`.
* The mining rule R1/R2/rewriting/R3/R4/R5: `research/earned-abstraction/mine.py`.
* The multi-entry variant of the rule (R3 counts base tasks, R4 sums over corpus
  entries): `research/premin-abstraction/mine_multi.py`.
* The semantic-identity variant — the rule under test:
  `research/premin-abstraction/mine_semantic.py`. **Not one line of it is
  modified by this track.**
* The corpus bands: `research/premin-abstraction/out/corpora.json`, built and
  verified by §46. Not rebuilt.
* `tcn.search.enumerate_fit` with tolerance `1e-3`, `rank="order"`,
  `max_programs` set above the space size so every enumeration is a **full
  sweep** and the certificate survives.

### The pooling rule under test

`mine_semantic.propose`, unchanged. Occurrences are pooled by
`(len(canon.inputs), truth_table(canon))`; each pool elects the representative
with fewest nodes, ties by digest; every occurrence in the pool is rewritten to
a call to that representative and **every rewrite is executed against the task's
rows before it is scored**. Parameters `MAX_NODES = 5`, `MAX_HOLES = 4`,
`MIN_TASKS = 2`, as §44 and §46.

### The primary mining corpus

**`C-minall`** — every pruned program at the certified minimum length `k_t`, per
task, capped at 32 by §46's deterministic sha256 subsample; 130 entries over six
tasks. Declared primary in advance for two reasons stated before any arm:
§46 measured it as the *cheap* half of the corpus change (cheaper than §44's own
corpus build) and as sufficient on its own to put `MAJ3` at rank 1 under
semantic pooling; and the leave-one-out protocol below must rebuild the corpus
six times, so the cheap build is the feasible one.

**`C-trace`** (296 entries) is declared as a **secondary replication** of §46's
exact published configuration, so that the module digest `165bc290d9c82b70a8ea3cc2`
can be checked for reproduction. It is reported whatever it shows.

### The off-family corpus `F′` — the source of the pooled wrong module

Six 4-input tasks built by taking `research/earned-abstraction/corpus.py`'s six
task *shapes* and replacing `maj` by **`D134`**, the §44 distractor
(`xor(xor(a,b), and(c, or(a,b)))`, truth table 134, same arity, same gate count,
verified by the retest to shorten `MAJ3` by nothing):

| task | function |
|---|---|
| `s1` | `D134(a,b,c) xor d` |
| `s2` | `D134(a,b,c) and d` |
| `s3` | `D134(b,c,d) or a` |
| `s4` | `D134(a,c,d) xor b` |
| `s5` | `D134(a,b,d) or c` |
| `s6` | `D134(a,b,c) xor D134(b,c,d)` |

`F′`'s `C-minall` is built by §46's own `enumerate_programs`/`corpora` machinery
with the same cap, the same subsample and the same in-`tcn` verification, and
its certified minimum lengths are reported (not assumed equal to the majority
family's). This is a family with *shared structure that is not majority*, so
the identical pooled rule run on it yields a module that is plausible, earned,
semantically pooled, and known-wrong for the later task.

### The evaluation tasks

**L1 — the later task, unchanged from §44 and §46.** `maj(a,b,c) xor maj(d,e,f)`
over six Boolean inputs; the tight (3-node) and wide (9-node) scaffolds of
`later.py`. Constant baseline 0.5, uniform-random baseline 0.5, both exact.
L1 is in no mining corpus.

**H — the held-out tasks, and the compact scaffold.** A held-out task is one of
the six *earlier* tasks, removed **entirely** from the mining corpus; the rule
mines from the remaining five and the induced library is then offered to the
held-out task. The evaluation scaffold is the **compact scaffold**: two nodes
over the four Boolean inputs `a,b,c,d`, built by the same candidate
construction as `later.py` —

* `n1` reads `{a,b,c,d}`: 56 flat candidates, plus `4**arity` module calls;
* `y` reads `{a,b,c,d,n1}`: 85 flat candidates, plus `5**arity` module calls;
* declared space with a 3-ary module: `120 × 210 = 25 200`; flat: `56 × 85 = 4 760`.

Every one of `t1`–`t5` has certified minimum flat length 5 (§46's
`out/corpora.json`), so **no flat program fits two nodes** and arm 1 must
exhaust at 0 conforming — the same structural property that makes §44's tight
scaffold a fair test. `t6` has certified minimum length 3 and is therefore
**excluded from the held-out protocol and declared excluded here**, before any
arm: it is solvable flat in the compact scaffold, so it cannot separate the
arms. `t6` stays in every mining corpus.

Held-out tasks: **`t1`, `t2`, `t3`, `t4`, `t5`** — five leave-one-out runs.

Two further evaluation tasks, in **no** majority-family corpus:

* **`H_par` = `a xor b xor c xor d`** — a negative control. No 3-ary window
  structure of any kind. If `MAJ3` "helps" this, the compact scaffold is doing
  the work, not the module.
* **`H_d134` = `D134(a,b,c) xor d`** — a positive control in the opposite
  direction, and it is `s1` of `F′`, i.e. **mined-from for the off-family
  library and held out from the majority library**, which is stated here rather
  than discovered later. The hand-authored `MAJ3` should fail it and the pooled
  off-family module should solve it. If neither happens the compact scaffold is
  not measuring what it is meant to.

Constant (majority-class) baselines, computed from the truth tables before any
arm: `t1` 0.5000, `t2` 0.7500, `t3` 0.7500, `t4` 0.5000, `t5` 0.7500,
`H_par` 0.5000, `H_d134` 0.5000. Uniform-random baseline 0.5 throughout.

### The arms

The **only** thing that differs between arms is the library. Same scaffolds,
same examples, same signals, same objective, same seeds.

| arm | library |
|---|---|
| `arm1_none` | none — the flat space |
| `arm2_syntactic` | rank-1 of `mine_multi.propose` (identity = `Program.digest`) on the corpus |
| **`arm2s_semantic`** | **rank-1 of `mine_semantic.propose` (identity = `(arity, truth table)`) on the corpus — the claim under test** |
| `arm2s_trace` | the same rule on `C-trace`; §46's exact configuration, secondary |
| `arm3_authored` | the hand-authored `MAJ3` — the ceiling |
| `arm4_wrong_authored` | the hand-authored `D134`, same size and arity |
| `arm4b_wrong_mined` | the syntactic rule's runner-up, same machinery (§44's arm 4b) |
| **`arm4s_runnerup`** | **the highest-ranked eligible arity-3 pooled class that does *not* compute majority, same corpus, same rule** |
| **`arm4s_offfamily`** | **the rank-1 eligible arity-3 pooled class the identical rule mines from `F′`** |

`arm4s_runnerup` and `arm4s_offfamily` together are **the control §46 lacked**.
They are defined here **by rule, not by identity**, so nothing can be chosen
after the ranked table is read. If the two resolve to the same class that is
reported as such and only one enumeration is run.

**A third, stronger form of the same control, declared now:** every eligible
arity-3 pooled class on the primary corpus is enumerated on L1's tight scaffold,
and the number that reach 144 conforming is reported. If many do, pooling's
*rank* is not selecting anything; if only the majority class does, the rank is
load-bearing.

**Declared budget constraint, not a post-hoc filter.** The arm-4s controls and
the sweep are drawn from **arity-3** classes only. A 4-ary module makes the
tight space 64 161 792 programs, which §46 measured at 8 184 wall-seconds for a
single arm (`arm2p_plus1one`). Arity-3 keeps every arm at the 2 709 504-program
space §44 and §46 used, which is what makes the numbers directly comparable.
Any arity-4 class that outranks the reported arity-3 one is named in the results
even though it is not enumerated.

### The protocol per arm

* **Enumeration.** `enumerate_fit`, full sweep. Reported: `space_size`,
  `evaluated`, `exhausted`, `certificate` (`unique` / `complete` / `none`),
  `conforming`. "No solution exists in this family" is distinguished from
  "search failed" by the certificate, never inferred.
* **Gradient.** `later.gradient_search`, unchanged: `SoftProgram` + Adam,
  `lr = 0.05`, 400 steps, init noise 0.5, conformance of the argmax export
  checked every 10 steps. **24 seeds** on the tight and compact scaffolds,
  **8 seeds** on the wide scaffold — §44's and §46's seed counts exactly.
  Reported: seeds conforming, median steps to first conforming export, median
  exact accuracy of the export, with the constant and random baselines beside it.
* **Cost.** Wall seconds and DFS nodes to build every corpus; wall seconds to
  mine it under each identity relation; the pooled rule's cost against the
  syntactic rule's on the same corpus. If pooling costs more than it saves that
  is reported as the finding.

### Reported regardless

* The rule's three parameters (`MAX_NODES`, `MAX_HOLES`, `MIN_TASKS`) swept once
  under semantic identity, exactly as §44 and §46 swept them. **The rule is not
  re-tuned until a number improves.**
* Every corpus built, including any abandoned, with the reason.
* The full ranked pooled table, not only the winner.
* Whether the published module digest reproduces §46's `165bc290d9c82b70a8ea3cc2`.

## What would falsify the claim

Stated as the brief states them, and honoured.

* **F1 — pooling any fragment works, so semantic *selection* contributes
  nothing.** If `arm2s_semantic` **ties** `arm4s_runnerup` or `arm4s_offfamily`
  — same conforming count in the same exhausted space and comparable gradient
  conformance. **This is §50's failure mode repeating and it is the one to check
  hardest.** It also fires if a large fraction of the eligible arity-3 pooled
  classes reach 144 in the sweep.
* **F2 — compression, not reusable abstraction.** If `arm2s_semantic` reaches
  the ceiling on L1 but does **not** beat `arm1_none` and the wrong-module arms
  on the held-out tasks. A library that only helps tasks it was mined from is a
  compression result and will be reported as one.
* **F3 — frequency artifact.** If the rank-1 pooled class is trivially the most
  frequent pool regardless of semantics. Operationalised two ways, both
  reported: (a) whether the identical rule on `F′` proposes `F′`'s *own* shared
  structure or the *same* fragment as the majority family — the same fragment
  from both families means the ranking is reading the fragment enumerator, not
  the corpus; (b) whether replacing R4's MDL saving by raw pooled occurrence
  count leaves the rank-1 class unchanged.
* **F4 — the cost exceeds the saving.** Reported honestly whichever way it
  comes out, as §46 reported 7.6× DFS nodes and 12.9× CPU while noting that the
  half that mattered was cheaper.
* **F5 — the scaffold is doing the work.** If `arm1_none` solves any held-out
  task, or if `MAJ3` solves `H_par`, the compact scaffold does not separate the
  arms and the held-out result is void.

Any of these is a legitimate deliverable and is reported as it comes out.

## Discipline

* Enumeration with its certificate beside every synthesis number; `exhausted`
  reported separately from `evaluated`.
* Constant and random baselines beside every accuracy.
* The achieved configuration is reported, never the requested one.
* Probes are supervision, never model inputs.
* If a recorded §44 or §46 number does not reproduce here, that is the finding,
  not something to work around.
* Negative results are preserved.
* If only one held-out task can be constructed, the result is marked
  anecdote-strength. Five leave-one-out tasks plus two controls is what is
  planned; whatever is achieved is what is reported.
* **`tcn/` is not modified.** If library integration turns out to warrant a core
  hook, it is *designed* in the results document and not implemented unless the
  result survives every criterion above.
