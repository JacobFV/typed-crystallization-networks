# A ranking objective that selects the reusable abstraction — and what it does not fix

`PREREGISTRATION.md` was committed at **`69da5f8`**, before any objective was
scored, any module published, or any enumeration run. Every number below comes
from a file in `out/`; `summarize.py`, `show_folds.py` and `show_off.py`
regenerate every table directly from those files.

`tcn/`, `generators/`, `research/semantic-library/`,
`research/premin-abstraction/`, `research/earned-abstraction/`,
`research/lazy-guard/` and `research/depth-encoding/` are **not modified**. The
rule under test is imported, not restated.

---

## VERDICT

**The ranking layer is fixable, and two objectives fix it.** Weighting the MDL
saving by the number of distinct tasks a class occurs in (**O2**), and scoring a
class by how much it saves on tasks withheld from its own derivation (**O4**),
both rank the majority class first on **all six** corpora — the full one and all
five leave-one-outs — and their rank-1 module then helps **5 of 5** held-out
tasks, equal to the hand-authored ceiling, against the incumbent's 2 of 5. Every
enumeration exhausts with certificate `complete`. §52's framing was not
optimistic. The counterpoint, which belongs in the headline: on the **two** tasks
where the incumbent's module does apply, it is the *better* module — 24 of 24
gradient seeds at accuracy 1.0000 against the winners' 4 of 24 at 0.8750,
because a 4-ary module absorbs the whole task body (§3.1).

**The frequency baseline is not the result, and the data says so sharply.** Both
trivial frequency counts help **0 of 5**. The majority class is *never* the most
frequent class in any corpus — under a pure task count it is **rank 4**, behind
three classes that occur in strictly more tasks, so no tie-break could rescue a
frequency count. And sweeping the breadth exponent `α` in `|T|^α · Σ s_e − D`
shows **both endpoints failing**: `α = 0` **is** the incumbent and picks the
wrong narrow class; at `α = 10` breadth swamps the saving and it picks the wrong
broad one — the same class B1 picks. The right class holds the interior,
`α ∈ [0.08, 3]` on every corpus. **Neither factor alone selects it; the product
does.** F2 does not fire.

**An exploratory replication on a second family agrees, with one caveat that
must travel with it.** On `F′` — §52's six task shapes with the majority window
replaced by the §44 distractor — O2 and O4 select `F′`'s **own** window class
(different digest *and* different truth table) at rank 1 on all five
leave-one-outs and match the hand-authored ceiling; both frequency baselines
stay at 0 of 5. But on `F′` the **incumbent also reaches 5 of 5**, so that
family corroborates the *selection* result and does **not** reproduce the
transfer failure. It was added after the primary table was complete
(Amendment 1) and counts toward no criterion.

**The win is half a fix, and the other half belongs to a different objective.**
On §41's ten-program language family, O2 and O4 are *degenerate* — with one task
the breadth weight is a constant multiplier and the cross-validation has nothing
to hold out — so they still select the **bytecode-maximal** program, exactly as
`description_bits` does. The only objective that fixes §41 is **O5**, the
measured execution-cost one, which picks the bytecode-minimal program and helps
**0 of 5** held-out tasks. **Transfer and §41 are two defects, and no single
objective tested here repairs both.** F6 fires, and it is reported as the
headline's second half rather than as a footnote.

**A third result, and the one that most surprised me.** O5's measured instrument
finds that **0 of 40 pooled classes reduce executed primitive operations at all**
— rewriting a call site does not remove the leaves it calls — and **0 of 40
reduce executed bytecodes**; every one *increases* them. So §41's "the modelled
cost carries no signal" survives replacement of the model by a measurement: the
measured cost is not constant here, it is **monotonically hostile to
abstraction**.

---

## 1. What was fixed in advance, and the integrity checks

Seven objectives, the leave-one-out protocol, the floor, the ceiling, the two
negative controls, six falsification criteria and three integrity checks were
all fixed in `PREREGISTRATION.md` before anything ran. §0 of that file discloses
that one leave-one-out ranked table (`∖ t1`) had been read beforehand, so the
predictions in its §6 are not blind for that corpus.

Every objective is an aggregation of **one** instrumented pooling pass: the same
fragments, the same `(arity, truth table)` identity, the same representative
election, the same rewriting, the same in-`tcn` verification of every rewrite,
the same `MAX_NODES = 5, MAX_HOLES = 4, MIN_TASKS = 2`. Only the score and the
ordering differ. The tie-break is `mine_semantic.propose`'s own with the score
swapped, so any change in rank 1 is attributable to the score — §52 observed
that promoting `tasks` above `saving` **in the tie-break** would fix the
ordering, and that is deliberately not what any arm here does.

| corpus | entries | eligible classes | IC1 `Σ s_e − D == saving_bits` | IC2 O1 rank-1 == the rule's rank-1 |
|---|---|---|---|---|
| `full` | 130 | 40 | PASS | PASS |
| `wo_t1` | 103 | 28 | PASS | PASS |
| `wo_t2` | 109 | 40 | PASS | PASS |
| `wo_t3` | 103 | 31 | PASS | PASS |
| `wo_t4` | 103 | 28 | PASS | PASS |
| `wo_t5` | 103 | 31 | PASS | PASS |

IC1 is exact integer equality, per class, on every corpus — 198 classes checked
against `mine_semantic.propose`'s own `saving_bits`, `digest`, `tasks`,
`entries`, `occurrences`, `definition_bits` and `nodes`, with **zero**
mismatches. IC2 confirms the re-derivation reproduces the incumbent: O1's rank-1
is the rule's rank-1 on all six corpora. IC3 — `exhausted: true` and certificate
`complete` — holds on **every one of the 134 enumerations** in this track, and
`evaluated == space_size` in every row.

**§52 reproduces exactly and independently.** O1's rank-1 modules on the five
leave-one-out corpora are `08735e50` (`∖ t1`, `∖ t4`) and `0ba4287a` (`∖ t2`,
`∖ t3`, `∖ t5`) — character for character §52's `arm2s_semantic`, with the same
held-out conforming counts (0, 0, 12, 0, 12), the same 221 520 and 25 200 space
sizes, and the same 0.85 % / 5.57 % margins. This pipeline was rebuilt from the
rule upward and lands on §52's numbers.

## 2. The seven objectives

For a class `c` on a corpus with entries `E` and tasks `T`: `s_e(c)` is the
per-entry description-bit saving, `D(c)` the definition charged once, `T(c)` the
tasks `c` occurs in.

| id | score | in one line |
|---|---|---|
| **O1** | `Σ_{e∈E} s_e − D` | the incumbent, known to fail |
| **O2** | `\|T(c)\| · Σ_{e∈E} s_e − D` | breadth-weighted |
| **O3** | `(1/\|T\|) Σ_{t∈T} mean_{e∈E_t} s_e − D` | per-task mean of the per-entry saving |
| **O4** | `(1/\|T\|) Σ_{u∈T} v_u(c)` | scored on tasks withheld from its own derivation |
| **O5** | `Σ_E [X(flat) − X(rewritten)] − X(def)` | measured executed operations, not `execution_cost` |
| **B1** | `\|T(c)\|` | trivial frequency count — distinct tasks |
| **B2** | occurrences | trivial frequency count — call sites |

O4's `v_u(c)` re-mines the pool on the corpus **minus `u`**, matches `c` into it
by pooling key, and scores that pool's own elected representative on `u`'s
entries — which were withheld from its derivation *and* from its scoring. A key
that does not survive the reduced pool scores 0 for that fold. O5's `X` counts
executed primitive `tcn` leaf operations exhaustively over the complete 16-row
domain, with `research/lazy-guard/cost.py`'s meter — §51's instrument — rather
than `Program.execution_cost`, which §41 measured as constant.

## 3. The decisive table — objective × leave-one-out, with certificates

A held-out task contributes **no program at any length** to the corpus its
module was mined from. "Helps" means `conforming > 0` on the compact scaffold
with `exhausted: true` and certificate `complete`.

| objective | t1 | t2 | t3 | t4 | t5 | **helps** | `H_par` | `H_d134` |
|---|---|---|---|---|---|---|---|---|
| **O1** incumbent, `description_bits` summed over entries | 0 | 0 | **12** | 0 | **12** | **2 of 5** | 0 | 0 |
| **O2** breadth-weighted | **12** | **12** | **12** | **12** | **12** | **5 of 5** | 0 | 0 |
| **O3** per-task mean | 0 | 0 | 0 | 0 | 0 | **0 of 5** | 0 | 0 |
| **O4** in-corpus leave-one-out cross-validated | **12** | **12** | **12** | **12** | **12** | **5 of 5** | 0 | 0 |
| **O5** execution-cost-aware, measured | 0 | 0 | 0 | 0 | 0 | **0 of 5** | 0 | 0 |
| **B1** frequency baseline — distinct tasks | 0 | 0 | 0 | 0 | 0 | **0 of 5** | 0 | 0 |
| **B2** frequency baseline — occurrences | 0 | 0 | 0 | 0 | 0 | **0 of 5** | 0 | 0 |
| *ceiling*: hand-authored `MAJ3` | **12** | **12** | **12** | **12** | **12** | **5 of 5** | 0 | 0 |
| *floor*: no library | 0 | 0 | 0 | 0 | 0 | **0 of 5** | 0 | 0 |
| *wrong module*: hand-authored `D134` | 0 | 0 | 0 | 0 | 0 | **0 of 5** | 0 | **4** |

Every row: `exhausted: true`, certificate `complete`, `evaluated == space_size`.
Spaces are 4 760 flat, 7 920 with a 2-ary module, 25 200 with a 3-ary and
221 520 with a 4-ary; **4 006 520 programs** evaluated in total on this family.

**The floor is live.** `arm1_none` exhausts at 0 conforming on all seven compact
tasks, so the scaffold has no flat solution and cannot be doing the work.

**Both negative controls hold, in both directions.** No majority-family module
scores above 0 on `H_par` or on `H_d134` — including the ceiling — while the
hand-authored `D134` scores **4** on `H_d134` and 0 on every majority task. The
controls are not merely silent; they are demonstrably able to fire. **F5 does
not fire.**

**What each objective selected**, and where the class it should have selected
sat in its table:

| objective | rank-1 on the five leave-one-outs | is it the window class | rank of the window class | rank 1 decided by |
|---|---|---|---|---|
| O1 | `08735e50` ×2, `0ba4287a` ×3 (4-ary, 5 nodes, **2 tasks**) | no, 0 of 5 | 2, 3, 2, 2, 2 | score |
| **O2** | `165bc290` ×5 (3-ary, 4 nodes, 4 tasks) | **yes, 5 of 5** | 1 | score |
| O3 | `5e50f170` ×5 (2-ary, 1 node) | no | 11–16 | score |
| **O4** | `165bc290` ×5 | **yes, 5 of 5** | 1 | score |
| O5 | `9e8092e7` ×4, `85aa887f` ×1 (2-ary, 1 node) | no | 14–16 | **tie-break** |
| B1 | `82b93e4b` ×5 (3-ary, 2 nodes, 5 tasks) | no | **4** | **tie-break** |
| B2 | `9e8092e7` ×4, `85aa887f` ×1 | no | 6–7 | score |

`165bc290d9c82b70a8ea3cc2` is the digest §46 published, §52 reproduced, and §52
could only reach with an *oracle* selector that knew the target function. **O2
and O4 reach it with no oracle**: they read the mining corpus and nothing else.

### 3.1 Gradient beside the enumeration — and one place the incumbent wins

`later.gradient_search` unchanged, 24 seeds, 400 steps, the same rows.
Seeds conforming / median exact accuracy:

| objective | t1 | t2 | t3 | t4 | t5 | `H_par` | `H_d134` |
|---|---|---|---|---|---|---|---|
| **O1** | 0/24 · .7500 | 0/24 · .8750 | **24/24 · 1.0000** | 0/24 · .7500 | **24/24 · 1.0000** | 0/24 · .5000 | 0/24 · .6250 |
| **O2** | **4/24** · .7500 | **7/24** · .8750 | **4/24** · .8750 | **7/24** · .7500 | **3/24** · .8750 | 0/24 · .5000 | 0/24 · .6250 |
| **O3** | 0/24 · .7500 | 0/24 · .8750 | 0/24 · .8750 | 0/24 · .7500 | 0/24 · .8750 | 0/24 · .5000 | 0/24 · .6250 |
| **O4** | **4/24** · .7500 | **7/24** · .8750 | **4/24** · .8750 | **7/24** · .7500 | **3/24** · .8750 | 0/24 · .5000 | 0/24 · .6250 |
| **O5** | 0/24 · .7500 | 0/24 · .8750 | 0/24 · .8750 | 0/24 · .7500 | 0/24 · .8750 | 0/24 · .5000 | 0/24 · .6250 |
| **B1** | 0/24 · .7500 | 0/24 · .8750 | 0/24 · .8750 | 0/24 · .7500 | 0/24 · .8750 | 0/24 · .5000 | 0/24 · .6250 |
| **B2** | 0/24 · .7500 | 0/24 · .8750 | 0/24 · .8750 | 0/24 · .7500 | 0/24 · .8750 | 0/24 · .5000 | 0/24 · .6250 |
| *ceiling* `MAJ3` | 4/24 · .7500 | 7/24 · .8750 | 4/24 · .8750 | 7/24 · .7500 | 3/24 · .8750 | 0/24 · .5000 | 0/24 · .6250 |
| *floor*, none | 0/24 · .7500 | 0/24 · .8750 | 0/24 · .8750 | 0/24 · .7500 | 0/24 · .8750 | 0/24 · .5000 | 0/24 · .6250 |
| *wrong* `D134` | 0/24 · .7500 | 0/24 · .8750 | 0/24 · .8750 | 0/24 · .7500 | 0/24 · .8750 | 0/24 · .8750 | **15/24 · 1.0000** |
| *constant baseline* | .5000 | .7500 | .7500 | .5000 | .7500 | .5000 | .5000 |
| *random baseline* | .5000 | .5000 | .5000 | .5000 | .5000 | .5000 | .5000 |

Three things, and the second is a point against the winners.

**O2 and O4 match the hand-authored ceiling row for row** — 4, 7, 4, 7, 3 of 24
seeds — because they select the same class, and every one of their conforming
seeds puts the module on the output path. Every §52 row this overlaps reproduces
exactly.

**Where the incumbent does solve a task, it solves it far more easily.** On `t3`
and `t5` O1's 4-ary module reaches **24/24 at median accuracy 1.0000** against
the winners' 4/24 and 3/24 at 0.8750, because a 4-ary module absorbs the whole
task body and leaves the gradient almost nothing to search. §52 recorded the
same effect. So the honest reading of the two tables together is: **the
incumbent's module is the better module on the two tasks where it happens to
apply, and applies to two tasks of five; the winners' module applies to all
five and is harder to learn on each.** The enumeration certificate is the
decisive measurement because it is the one that separates "no solution exists in
this space" from "gradient search failed" — but it is not the only fact.

**The gradient controls fire in both directions.** `D134` reaches 15/24 at
1.0000 on `H_d134` and 0/24 everywhere else; no majority-family arm exceeds its
constant baseline on either control.

## 4. Why the incumbent fails and what the two winners change

### 4.1 The margin, reproduced

| corpus | O1's rank-1 over the window class | O2's rank-1 over its runner-up | O4's rank-1 over its runner-up |
|---|---|---|---|
| `full` | *is the window class* | +1 268 456 (149.0 %) | +38 640 (173.8 %) |
| `wo_t1` | **+2 824 (0.85 %) the wrong way** | +673 720 (99.2 %) | +33 053 (132.3 %) |
| `wo_t2` | **+18 432 (5.57 %) the wrong way** | +642 480 (90.4 %) | +34 992 (151.9 %) |
| `wo_t3` | **+18 432 (5.57 %) the wrong way** | +642 480 (90.4 %) | +40 378 (228.7 %) |
| `wo_t4` | **+2 824 (0.85 %) the wrong way** | +673 720 (99.2 %) | +33 053 (132.3 %) |
| `wo_t5` | **+18 432 (5.57 %) the wrong way** | +642 480 (90.4 %) | +40 378 (228.7 %) |

§52's 0.85 % and 5.57 % reproduce exactly. The replacements do not decide on a
knife edge: their margins are two orders of magnitude larger.

### 4.2 O4's fold decomposition — the mechanism, measured

On `wo_t1`, the saving each class earns on each fold's **withheld** task:

| class | tasks | O1's saving | t2 | t3 | t4 | t5 | t6 | **O4** |
|---|---|---|---|---|---|---|---|---|
| `165bc290` the window `MAJ3` | 4 | 330 672 | 75 072 | 75 072 | 75 072 | 75 072 | −10 128 | **58 032** |
| `ca785ec9` pooled runner-up | 4 | 149 088 | 18 192 | 52 128 | 8 496 | 52 128 | −6 048 | **24 979** |
| `82b93e4b` B1's pick, the most task-frequent | 5 | 102 752 | 18 304 | 18 304 | 28 048 | 18 304 | −4 432 | **15 706** |
| `08735e50` **O1's rank-1** | 2 | **333 496** | −12 360 | 0 | −12 360 | 0 | −12 360 | **−7 416** |

This is the whole finding in one table. The class the incumbent ranks first has
the **largest raw saving** — and is worth **negative** bits on every task
withheld from it, because it occurs in only two tasks: on three folds it cannot
be reconstructed at all, and on the folds where it exists its definition is
never repaid. The class the incumbent ranks second earns **+75 072 on every
majority fold**. §52 diagnosed that summing over entries "penalises broad
fragments and leaves narrow ones untouched"; this measures the inversion
directly.

(`t6` is negative for every class: its certified minimum is 3 gates and it needs
**two** module calls, so a single rewrite never pays there. §52 recorded the
same fact from the other side.)

### 4.3 O3 and O5 fail, and both failures are informative

**O3 — per-task mean — helps 0 of 5, and it fails for a reason that is worth
recording.** Dividing the saving by the entry count inside each task brings the
per-task figure down to the order of a few thousand bits, while the definition
`D(c)` is charged undivided at 3 800–12 384 bits. **All 40 scores go
negative**, and the objective degenerates into "prefer the smallest
definition": its rank-1 is the same 1-node, 2-ary class on all six corpora, with
the window class at rank 11–16. Normalising the *benefit* without normalising
the *cost* inverts the objective. That is a defect of this particular
formulation, not evidence against per-task normalisation in general, and it is
reported as such rather than repaired — the pre-registration forbids re-tuning.

**O5 — execution-cost-aware, measured — helps 0 of 5, and it carries literally no
reuse signal.** With `research/lazy-guard/cost.py`'s meter, exhaustive over the
16-row domain:

| | flat corpus | any rewritten corpus | classes that improve it |
|---|---|---|---|
| executed primitive operations | 10 368 | **10 368**, for all 40 classes | **0 of 40** |
| executed CPython bytecodes | 1 763 270 | 1 773 510 – 2 297 606 | **0 of 40** |

Rewriting a call site does not remove the leaves it calls, so the operation
count is *identical* for every class; the bytecode count strictly *increases*,
by 0.58 % to 30.3 %, because a module call is interpreter overhead. O5's score is
therefore `−X(definition)` for every class, its ranking is decided entirely by
definition size, and rank 1 is settled by the **tie-break** on all six corpora.

This is §41's finding recurring under a measured instrument, and in a stronger
form. §41 found `execution_cost` **constant** (148.0) across a family and
concluded it could not discriminate. Here the measurement is not constant — the
bytecode range spans 1.78 M to 2.30 M — and it still carries no *reuse* signal,
because it is monotonically hostile to abstraction. Replacing the modelled cost
with a measured one does not turn it into a reuse objective. §51's measured
ns-per-bytecode constants (2.24–3.21 ns) are recorded but not applied: a
constant factor cannot change a ranking.

## 5. The frequency-count baselines — F2, the criterion most likely to fire

It does not fire, and three independent measurements say so.

**(a) Both baselines help 0 of 5.** B1 (distinct tasks) and B2 (occurrences) are
in the decisive table above at 0 of 5, against O2's and O4's 5 of 5.

**(b) The right class is never the most frequent, so no frequency count *could*
select it.** Under a pure task count the window class is **rank 4** on every
corpus, behind three classes that occur in *strictly more* tasks (6 vs 5 on the
full corpus, 5 vs 4 on each leave-one-out) — the tie-break is irrelevant, it is
strictly dominated. Under a pure occurrence count it is rank 6–7.

**(c) The exponent sweep shows both endpoints failing.** Generalise O2 to
`|T|^α · Σ s_e − D`. `α = 0` **is** the incumbent; `α → ∞` **is** the frequency
count.

| corpus | rank 1 at `α = 0` | rank 1 for `α ∈ [0.08, 3]` | rank 1 at `α = 10` |
|---|---|---|---|
| `full` | `165bc290` **window** | `165bc290` **window** | `82b93e4b` |
| `wo_t1` | `08735e50` | `165bc290` **window** | `82b93e4b` |
| `wo_t2` | `0ba4287a` | `165bc290` **window** | `82b93e4b` |
| `wo_t3` | `0ba4287a` | `165bc290` **window** | `82b93e4b` |
| `wo_t4` | `08735e50` | `165bc290` **window** | `82b93e4b` |
| `wo_t5` | `0ba4287a` | `165bc290` **window** | `82b93e4b` |

Swept `α ∈ {0, .02, .05, .08, .1, .25, .5, .75, 1, 1.5, 2, 3, 5, 10}`. The
window class holds a wide interior band on every corpus and **neither endpoint**.
The MDL term and the breadth term are each necessary and neither is sufficient.
`α = 1`, the pre-registered O2, sits in the middle of the band on all six.

## 6. Sensitivity to the rule's own parameters — reported, not selected from

`MAX_NODES × MAX_HOLES × MIN_TASKS` over `{3,4,5} × {2,3,4} × {2,3,4}` — §52 §8's
grid — on all six corpora, 162 cells. **Every declared arm of this track uses
the defaults 5 / 4 / 2; nothing was tuned.** The window class is constructible
in 72 of the 162 cells (it is 3-ary with a 4-node representative, so
`MAX_HOLES = 2` or `MAX_NODES = 3` excludes it, exactly as §52 measured).

| objective | rank 1 is the window class, of the 72 cells where it is constructible |
|---|---|
| O1 | 67 of 72 |
| **O2** | **72 of 72** |
| O3 | 0 of 72 |
| **O4** | **52 of 72** |
| B1 | 0 of 72 |
| B2 | 0 of 72 |

Two things to read off this, and the second is uncomfortable for the objective
the brief expected to work.

**O1's failure is narrower than it looked.** All five of its failures are at the
default `5 / 4 / 2`, on the five leave-one-out corpora, and nowhere else. The
narrow 4-ary competitor that beats the majority class needs 5 nodes and 4 holes
to exist at all; at any smaller budget the incumbent is fine. So the incumbent
is not globally broken — it breaks precisely when the search budget is large
enough to admit a deep, narrow fragment. That is a sharper statement of §52's
mechanism than §52 made.

**O4 is the more brittle of the two winners, and it breaks in a way intrinsic to
cross-validation.** All 20 of its failures are the cells with `MIN_TASKS = 4` on
a leave-one-out corpus. The window class occurs in 4 tasks there; removing one
for a fold drops it to 3, below the reuse gate, so it is absent from **every**
fold's pool and scores 0 on all of them. A cross-validated objective must not
apply a reuse gate that the fold itself can violate. O2 has no such coupling and
is rank-1-correct in 72 of 72.

**On this evidence O2 is the objective to prefer**, notwithstanding that the
brief expected O4 to be the cleanest. Both reach 5 of 5 at the declared
parameters; O2 is robust across the grid and across the exponent, and it is far
cheaper (§9).

## 7. EXPLORATORY: the same protocol on a second family — Amendment 1

**Amendment 1** adds to `PREREGISTRATION.md` §6; it replaces no clause and can
change no primary verdict. It was written and run **after** the majority-family
table was complete, so it is exploratory in the strict sense. §52 disclosed as a
limitation that "the family count is one"; the off-family corpus `F′` — §52's
six task shapes with the majority window replaced by the §44 distractor `D134`,
117 entries over six tasks, every entry verified in `tcn` — is already on disk,
so the identical protocol costs five mines and 2 743 360 further evaluations.

IC1 passes on all five `F′` corpora. Every row exhausts, certificate `complete`.

| objective | s1 | s2 | s3 | s4 | s5 | helps | rank-1 is `F′`'s window | controls: t1 / t3 / `H_par` |
|---|---|---|---|---|---|---|---|---|
| O1 | **4** | **4** | **4** | **4** | **4** | **5 of 5** | 2 of 5 | 0 / 0 / 0 |
| **O2** | **4** | **8** | **4** | **4** | **4** | **5 of 5** | **5 of 5** | 0 / 0 / 0 |
| O3 | 0 | 0 | 0 | 0 | 0 | 0 of 5 | 0 of 5 | 0 / 0 / 0 |
| **O4** | **4** | **8** | **4** | **4** | **4** | **5 of 5** | **5 of 5** | 0 / 0 / 0 |
| O5 | 0 | 0 | 0 | 0 | 0 | 0 of 5 | 0 of 5 | 0 / 0 / 0 |
| B1 | 0 | 0 | 0 | 0 | 0 | 0 of 5 | 0 of 5 | 0 / 0 / 0 |
| B2 | 0 | 0 | 0 | 0 | 0 | 0 of 5 | 0 of 5 | 0 / 0 / 0 |
| *ceiling*: hand-authored `D134` | **4** | **8** | **4** | **4** | **4** | 5 of 5 | — | 0 / 0 / 0 |
| *floor*: no library | 0 | 0 | 0 | 0 | 0 | 0 of 5 | — | 0 / 0 / 0 |

**What this corroborates.** O2 and O4 select `F′`'s **own** window class
(`0b7ea9a5a2d4` — a different digest *and a different truth table* from the
majority family's) at rank 1 on all five leave-one-outs, match the hand-authored
`D134` ceiling row for row, and score 0 on the cross-family majority tasks and
on parity. Both frequency baselines remain at 0 of 5. The mechanism is not
specific to the majority family.

**What it does not corroborate, stated plainly.** On `F′` the **incumbent also
reaches 5 of 5**, even though its rank-1 is the window class in only 2 of 5: its
4-ary alternative happens to solve the held-out tasks too. So this family
replicates the *selection* result and does **not** replicate the *transfer*
failure that motivates the track. It is corroboration of O2 and O4, not a second
independent demonstration that O1 is inadequate.

## 8. The §41 cross-check — F6 fires

`research/program-length/out/language_family.json` is *read*; nothing is
re-enumerated. It holds ten conforming programs of one language family (45 375
evaluated, `exhausted: true`), all at unseen accuracy **1.000** against a
**0.5661** majority constant and a 0.5 random baseline, with
`description_bits ∈ {4 043 552, 4 043 560, 4 043 568}` and measured
`bytecodes_total ∈ {41 417, 41 441, 41 465}`.

| objective | degenerate on a one-task family | description bits of its pick | measured bytecodes | bytecode-maximal? |
|---|---|---|---|---|
| O1 | no | 4 043 552 | 41 465 | **YES** |
| **O2** | **yes** — `\|T\| = 1` is a constant multiplier | 4 043 552 | 41 465 | **YES** |
| O3 | **yes** — one task, the mean is the value | 4 043 552 | 41 465 | **YES** |
| **O4** | **yes** — holding out the only task empties the corpus; every candidate scores 0 | *no pick* | — | — |
| **O5** | no | 4 043 560 | **41 417** | **no — bytecode-minimal** |
| B1 / B2 | **yes** — every candidate covers the one task once | *no pick* | — | — |

**F6 fires for both winners.** O2 and O4 do not fix §41. Their entire mechanism
is cross-*task* structure, and §41's family has one task; O2 collapses to the
incumbent's ordering and O4 collapses to no ordering at all. The only objective
that selects the bytecode-minimal program is **O5**, which helps 0 of 5 held-out
tasks and, as §4.3 shows, carries no reuse signal whatever.

**The honest conclusion is that these are two different defects.** §52's failure
is a *ranking across tasks* problem and a breadth term fixes it. §41's failure is
a *cost model* problem and only a measured cost term fixes it. The two terms are
orthogonal, they can be summed, and this track did not test the sum — that would
be a new arm and the pre-registration does not license inventing one after
reading the results. It is recorded as the obvious next pre-registered
experiment, not as a result.

## 9. Cost

| stage | cost |
|---|---|
| one instrumented pool (`C-minall`, 130 entries), measured alone | 0.55 s |
| pool + IC1 + O4's five folds + O5's measurement, per corpus | 7.0 – 11.6 s (`out/ranked.json`) |
| all six corpora × seven objectives, scored and published | 51 s total |
| held-out enumeration, majority family | 70 runs, **4 006 520** programs, 64 s wall at 12 workers |
| held-out enumeration, `F′` (exploratory) | 64 runs, **2 743 360** programs |
| §52 §8's parameter grid, 162 cells | 450 s, single-threaded |
| gradient, 7 tasks × 10 modules × 24 seeds × 400 steps | 1 680 runs, ≈ 36 000 CPU-s — **the dominant cost of the track**, and it decides nothing the enumeration has not already certified |
| §41 cross-check | a file read; nothing re-enumerated |

**O2 is essentially free**: it is one multiplication per class on numbers the
incumbent already computes. **O4 costs `|T|` extra mines per corpus** — five
times the mining work — plus a rewriting pass over the held-out entries of each
fold. On this corpus that is seconds; on a corpus where mining is the budget it
is a real multiplier, and combined with §6's brittleness it is the weaker of the
two recommendations.

The machine was shared with other agents' jobs throughout, so wall seconds are
indicative. Program counts, space sizes, conforming counts, DFS-free evaluation
counts and every bit figure are not.

## 10. Verdict against each pre-registered falsification criterion

| id | fires? | evidence |
|---|---|---|
| **F1** — no objective beats 2 of 5 | **NO** | O2 and O4 both reach **5 of 5**, equal to the hand-authored ceiling, every enumeration exhausted with certificate `complete`. |
| **F2** — a 5-of-5 objective is really a frequency count | **NO** | B1 and B2 both help **0 of 5**. The window class is **rank 4** under a pure task count, strictly dominated by three classes, so no frequency count can select it under any tie-break. The exponent sweep fails at **both** endpoints — `α = 0` (the incumbent) and `α = 10` (pure frequency) — and succeeds only in the interior. |
| **F3** — wins in-sample, loses out-of-sample | **NO** | O2 and O4 rank the window class first on the **full** corpus *and* on all five leave-one-outs, and the module then transfers 5 of 5. In-sample and out-of-sample agree. |
| **F4** — the §52 windowing was doing the work | **NO** | O2 and O4 are non-oracle by construction: they read the mining corpus and never the held-out task's truth table, examples or scaffold. They reach 5 of 5 **without** windowing, on the same module §52 could only reach with an oracle. |
| **F5** — a negative control breaches | **NO** | No majority-family arm exceeds 0 on `H_par` or `H_d134`, ceiling included; the hand-authored `D134` scores 4 on `H_d134` and 0 on every majority task, so the controls are live. `arm1_none` exhausts at 0 on all seven compact tasks. |
| **F6** — the winner still picks §41's bytecode-maximal program | **YES** | O2 picks it (degenerately); O4 makes no pick at all. Only O5 picks the bytecode-minimal program, and O5 helps 0 of 5. The fix is half a fix and is reported as such. |

## 11. Disclosures

* **The pre-registration is not fully blind, and says so.** Its §0 records that
  the top six rows of §52's `∖ t1` ranked table were read before it was written,
  so the prediction that B1 would fail was informed for that corpus. The
  prediction was recorded anyway and then checked against four further corpora,
  seven objectives and a second family.
* **The winning module is not new.** `165bc290d9c82b70a8ea3cc2` is the digest §46
  published and §52 reproduced. What is new is that a **non-oracle** objective
  selects it. This track moves the selection rule, not the artifact, and it
  should not be read as a new abstraction-discovery result.
* **O2 is one of a band, not a unique point.** §5(c) shows the whole interval
  `α ∈ [0.08, 3]` selecting correctly. `α = 1` was pre-registered because it is
  the natural form, not because it was found by search — but it is not
  distinguished within the band, and nothing here says the linear form is
  privileged.
* **O4 is brittle in a way the brief did not anticipate and I did not predict.**
  It fails in every `MIN_TASKS = 4` cell of the parameter grid, because a class
  in exactly `MIN_TASKS` tasks cannot survive its own folds. The brief expected
  O4 to be the cleanest result; it is the *less* robust of the two winners.
* **O3's failure may be an artifact of my formulation.** Dividing the benefit by
  the entry count while charging the definition undivided drives every score
  negative. A per-task-mean objective with a correspondingly amortised definition
  term was not tested. It is not re-tuned here, and no claim is made that
  per-task normalisation cannot work — only that this formulation does not.
* **The second family corroborates the selection and not the failure.** On `F′`
  the incumbent also reaches 5 of 5 (§7). The transfer failure this track exists
  to fix has been demonstrated on **one** family.
* **Seven tasks, two families, one corpus band.** This is small. The
  leave-one-out result is unanimous across five corpora and replicates on a
  second family, which is more than anecdote-strength, but it is not a broad
  benchmark and should not be quoted as one.
* **`t6` is in the mining corpus and is never a held-out task.** Its certified
  flat minimum is 3 gates and it needs two module calls where the compact
  scaffold has one free node — §52's correction, which this track inherits and
  which is visible in §4.2 as `t6`'s negative fold for every class.
* **The wall-clock caveat.** The host was shared with other agents' jobs
  throughout.
* **An import shadowing bug was found and fixed mid-track.** `_paths.py`
  originally left `research/semantic-library` ahead of this directory on
  `sys.path`, so `run_grad.py` imported §52's `run_heldout` instead of this
  track's and crashed immediately. No result was affected — the crash was total
  and no other module name collides — and `run_rank.py` / `run_heldout.py` were
  re-run after the fix and reproduced identical output. Recorded because a
  silent version of the same bug would have been invisible.

## 12. Amendments to the pre-registration

| # | clause | change |
|---|---|---|
| **1** | **adds to §6** (replaces nothing) | The off-family `F′` replication of §7. Written and run **after** the majority-family table was complete, therefore **exploratory**, labelled as such throughout, and counting toward no falsification criterion. Added because §52 disclosed "the family count is one" and the corpus was already on disk. |

No clause of the pre-registration was replaced or weakened. The parameters
`MAX_NODES / MAX_HOLES / MIN_TASKS` were exposed as arguments in `pool.build`
solely so §6's grid could be *reported*; every declared arm uses the defaults
`5 / 4 / 2`.

## 13. Reproduction and provenance

* **Nothing in `tcn/` or `generators/` was modified.** `git status` on this
  branch touches only `research/reuse-ranking/`.
* **`PREREGISTRATION.md` was committed at `69da5f8`** — with `_paths.py` and
  nothing else — before any objective was scored, module published or
  enumeration run.
* **The rule under test is imported unchanged.** `mine.py`, `mine_multi.py`,
  `mine_semantic.py`, `corpus.py`, `corpora.py`, `retention.py`, `later.py`,
  `enumerate_programs.py`, `minimal.py`, `evaltasks.py`, `family.py` and
  `cost.py` are imported, not restated. IC1 proves the instrumented pass agrees
  with `mine_semantic.propose` to the bit on all 198 classes over six corpora.
* **Every enumeration is a full sweep**: `max_programs = 2^24` above every space
  size, `exhausted: true`, certificate `complete`, `evaluated` reported
  separately from `space_size` and equal to it in all 134 runs.
* **Every published module is a real `tcn.library` entry** with a full
  truth-table fixture, loaded under `policy="strict"`. `module_on_output_path`
  is true in every conforming enumeration of the winners' module and in all 25
  of its conforming gradient seeds.
* **Both primary runs were repeated after the `sys.path` fix and reproduce.**
  `run_rank.py` re-run gives a `ranked.json` identical to the first field for
  field once wall-clock timings are dropped; `run_heldout.py` re-run gives
  identical rows on every field except timings — same conforming counts, same
  space sizes, same certificates.
* **Every accuracy carries its constant and random baselines** (§3 controls,
  §8's 0.5661 majority, the gradient table's per-task constants).
* **Test suite: 337 tests, and the count that passes depends only on a
  gitignored symlink.** Both configurations were measured **in this worktree**,
  as §52 did, and both match §52's figures exactly.
  * With `generators/computer/engine/node_modules` symlinked from the main
    checkout: **336 passed, 1 failed** —
    `tests/test_panel_interface.py::test_panel_episode_replays_and_restores`,
    the documented worktree-only failure on main's own code.
  * With the symlink removed (the state this branch is committed in):
    **324 passed, 13 failed**, all in `tests/test_panel_interface.py`, all
    environmental.

  This track cannot affect either: `pyproject.toml` sets `testpaths = ["tests"]`,
  so nothing under `research/` is collected, and this track adds no file under
  `tests/` or `tcn/`.
* **The shipped fixture reproduces**: `python -m tcn train --episodes 160` gives
  `initial_prediction_loss` **0.248835613951087** → `final_prediction_loss`
  **0.0022308224288281053**, `fully_frozen: true`,
  `frozen_evaluation_mean_return` **4.0**.
* **Large artifacts stay out of git**: `out/` holds JSON tables and two short
  logs; no checkpoints, no enumeration dumps.
* Every figure in this document comes from a file in `out/`; `summarize.py`,
  `show_folds.py` and `show_off.py` regenerate the tables from those files.
