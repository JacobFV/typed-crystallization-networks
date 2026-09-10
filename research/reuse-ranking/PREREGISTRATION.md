# Pre-registration — written and committed before any arm of this track was run

Written 2026-09-09, branch `research/reuse-ranking`, from `main` at `3b0fcb6`
(which contains `research/semantic-library` merged, i.e. FINDINGS §52).

House standard since §44: this file is committed before any objective is scored,
any module is published, or any enumeration is run.

---

## 0. What had already been run or read in this directory when this was written

Nothing was computed here. Three things were **read**, and one of them bears on
an outcome, so it is disclosed rather than buried:

1. `research/FINDINGS.md` §41, §46, §52 and
   `research/semantic-library/RESULTS.md` in full, including its §5.1 table of
   the rank flip and its §5 "observation, explicitly not adopted" that
   `mine.propose`'s tie-break would put the majority class first if `tasks`
   preceded `saving`. All of this is already published in the repository.
2. The source of `mine_semantic.py`, `mine_multi.py`, `poolmine.py`,
   `armlib.py`, `evaltasks.py`, `family.py`, `run_heldout.py` — the harness this
   track reuses and does not rebuild.
3. **Outcome-relevant, disclosed:** the top six rows of
   `research/semantic-library/out/mined_maj_minall_sem_wo_t1.json` were printed
   while reading the harness. So before writing this file I knew that on the
   `∖ t1` corpus the majority class (`165bc290`, 4 tasks, 60 entries,
   +330 672) is rank 2 behind a 4-ary class (`08735e50`, 2 tasks, 46 entries,
   +333 496), and that a *different* non-majority class (`82b93e4b`) occurs in
   **5** tasks — more tasks than the majority class's 4.

Item 3 means the predictions in §6 below are not blind for the `∖ t1` corpus.
They are recorded anyway, because a prediction that is stated and then checked
against four further corpora and seven objectives is still worth more than none.
No score was computed, no module published, and no enumeration run before this
file was committed.

---

## 1. Where this comes from

FINDINGS §52 established two things and left one gap.

**Established.** Semantic pooling by `(arity, truth table)` solves the
*identity* problem: 144 conforming equal to the hand-authored ceiling, both
pooled wrong-module controls at 0 in the same exhausted 2 709 504-program space,
and of 13 eligible pooled arity-3 classes exactly one yields any conforming
program at all — the one the rule ranks first.

**The gap.** The *ranking* fails out of sample. Across five leave-one-out
corpora the published MDL rank-1 module helps **2 of 5** held-out tasks; the
same pooled class selected by an oracle window helps **5 of 5**, matching the
ceiling. The class is in the pool every time, at rank 2 or 3, bit-identical
(`165bc290d9c82b70a8ea3cc2`) in all five. The measured mechanism (§52 §5.1): the
MDL score sums saving over corpus **entries**, so removing a task subtracts
saving from broad fragments and leaves narrow ones numerically untouched. Rank 1
flips by **0.85 %** in two of five.

This converges with §41, which certified that description-bit ranking selects
the **bytecode-maximal** program of a ten-program family, and where
`Program.execution_cost` was measured **constant** (148.0) across programs
differing by 48 executed bytecodes. Two independent tracks now show
`description_bits` selecting wrongly when used to *rank*.

## 2. The hypothesis

> There exists a ranking objective over the same pooled eligible set which
> selects the reusable abstraction, and which does so **out of sample** — its
> rank-1 class, mined from five of six tasks, helps the sixth.

The decisive measurement is out-of-sample, not in-sample. An objective that
ranks the majority class first on the full corpus but not on the
leave-one-outs has reproduced the incumbent's exact failure under a new name.

## 3. What is fixed in advance, and imported unchanged

* **The pool.** `mine_semantic.propose`'s pooling — identity
  `(len(canon.inputs), truth_table(canon))`, representative elected by fewest
  nodes then digest, every occurrence rewritten to a call to that
  representative, every rewrite executed against the task's rows before it is
  scored. `MAX_NODES = 5`, `MAX_HOLES = 4`, `MIN_TASKS = 2`, exactly as §44,
  §46 and §52. **These three parameters will not be tuned.**
* **The corpus.** `C-minall`, 130 entries over six tasks, loaded through
  `research/semantic-library/family.corpus_for` from §46's
  `research/premin-abstraction/out/corpora.json`. Not rebuilt.
* **The leave-one-out protocol.** `family.corpus_for(..., exclude=(t,))` removes
  a task's programs **entirely, at every length**, as §52.
* **The evaluation.** `research/semantic-library/evaltasks.compact_scaffold`,
  `evaltasks.examples_for`, `evaltasks.SIGNALS`; `tcn.search.enumerate_fit` with
  `tolerance=1e-3`, `rank="order"`, `max_programs = 1 << 24` so **every
  enumeration is a full sweep** and the certificate survives.
* **Publication.** Each selected class is published into a real
  `tcn.library.Library` on disk with a full-truth-table fixture and loaded under
  `policy="strict"`, via `poolmine.rebuild` / `poolmine.publish` unchanged.
* **`tcn/` is not modified.** Neither is `research/semantic-library/`,
  `research/premin-abstraction/`, `research/earned-abstraction/` or
  `research/depth-encoding/`.

This track adds exactly one thing to the rule: an **instrumented pooling pass**
that records the per-entry saving `s_e(c)` for every class, so that every
objective below is a different aggregation of the *same* numbers. It is a
re-derivation, not a re-implementation, and it carries its own integrity check
(§5, IC1): for every class on every corpus,
`Σ_e s_e(c) − D(c)` must equal `mine_semantic`'s own `saving_bits` **exactly**.
If IC1 fails anywhere the track stops and reports the discrepancy.

## 4. Notation

For a pooled class `c` on a mining corpus with entry set `E` and task set `T`:

* `b_e` — `description_bits` of the flat corpus program for entry `e`.
* `r_e(c)` — `description_bits` of `e` rewritten with `c` (`= b_e` if `c` does
  not occur in `e`).
* `s_e(c) = b_e − r_e(c)` — the per-entry saving.
* `D(c)` — `description_bits` of `c`'s elected representative: the definition,
  charged **once**.
* `E_t` — the entries of task `t`. `T(c)` — the tasks in which `c` occurs.

## 5. The objectives to be compared

All seven read only the mining corpus. **None of them may read the held-out
task's truth table, its examples, or its scaffold.** Any objective that did
would be the oracle window of §52, which is exactly what this track is trying to
replace.

| id | name | score |
|---|---|---|
| **O1** | incumbent, `description_bits` summed over entries | `Σ_{e∈E} s_e(c) − D(c)` |
| **O2** | breadth-weighted | `|T(c)| · Σ_{e∈E} s_e(c) − D(c)` |
| **O3** | per-task mean | `(1/|T|) · Σ_{t∈T} mean_{e∈E_t} s_e(c) − D(c)` |
| **O4** | leave-one-out cross-validated, **inside** the mining corpus | `(1/|T|) · Σ_{u∈T} v_u(c)` — see below |
| **O5** | execution-cost-aware, measured | `Σ_{e∈E} [X(flat_e) − X(rw_e(c))] − X(c)` |
| **B1** | frequency baseline — distinct tasks | `|T(c)|` |
| **B2** | frequency baseline — occurrences | occurrence count |

**O3** averages over *all* tasks of the corpus, not only `T(c)`: a class absent
from a task contributes 0 for that task. This is what makes it "a fragment in 5
tasks is not beaten by one appearing twice in 2" rather than its opposite, and
the per-entry mean inside a task removes the entry-count imbalance that O1 is
sensitive to.

**O4** is the honest in-corpus analogue of the out-of-sample test, and the one
the brief expects to work; it is therefore the one to test hardest. For each
task `u` of the mining corpus, the pool is re-mined on the corpus **minus `u`**
under the identical rule. A class `c` is matched into that reduced pool by its
`(arity, truth table)` key. Its validation score on `u` is

    v_u(c) = Σ_{e ∈ E_u} s_e(c′) − D(c′)

where `c′` is the reduced pool's own class for that key, with its own elected
representative, and `s_e` is obtained by rewriting `u`'s entries — which were
withheld from `c′`'s derivation and from its scoring — with `c′`. If the key
does not survive the reduced pool (fails `MIN_TASKS`, fails rewriting, or fails
verification), `v_u(c) = 0`. O4 is a mean over the `|T|` folds.

**O5** uses the *measured* cost model of §51 and not `Program.execution_cost`,
which §41 measured as carrying no signal. `X(p)` is the number of executed
primitive `tcn` leaf operations of `p` over its **complete** 16-row Boolean
domain — exhaustive, so an exact total and not a sample — counted with
`research/lazy-guard/cost.py`'s `Meter`, the instrument §51 used. The
definition is charged once; a call site pays for the leaves inside it.
Executed CPython bytecodes are recorded alongside as a cross-check. §51's
measured ns-per-bytecode constants (2.24–3.21 ns) are recorded but **not**
applied, because a constant factor cannot change a ranking; if the ops ranking
and the bytecode ranking disagree, both are reported.

**Tie-break.** O1–O5 all use `(−score, −|T(c)|, −occurrences, −nodes, digest)`,
which is `mine_semantic.propose`'s own tie-break with only the score swapped, so
that any change in rank 1 is attributable to the **score** and not to the
tie-break. §52 observed that promoting `tasks` above `saving` in the tie-break
alone would fix the ordering; that is deliberately *not* what any arm here does.
Whether a rank-1 decision was ever settled by a tie-break rather than by the
score will be reported per corpus. B1 and B2 use `(−score, −nodes, digest)` and
carry no MDL term at all.

**Integrity checks, reported whether or not they pass.**
* **IC1** — `Σ_e s_e(c) − D(c) == saving_bits` from `mine_semantic.propose`, for
  every class on every corpus. Exact integer equality.
* **IC2** — O1's rank-1 class must equal `mine_semantic.propose`'s rank-1 class
  on all six corpora (full + five LOO). If O1 is not the incumbent, the
  re-derivation is wrong.
* **IC3** — every enumeration reports `exhausted: true` and certificate
  `complete`. A truncated sweep is not a certificate and is not reportable.

## 6. The arms and the decisive measurement

For each objective `O` and each of the five leave-one-out corpora:

1. Compute the full ranked table under `O`.
2. Take its **rank-1** class, publish it as a real library module, load it
   `strict`.
3. Enumerate the compact scaffold for the **held-out** task exhaustively.
4. **"Helps"** := `conforming > 0` with `exhausted: true` and certificate
   `complete`. Report **helps N of 5** with the certificate beside every count,
   exactly as §52's table does.

Beside every objective:

* **Ceiling** — `arm3_authored`, the hand-authored `MAJ3`. §52 measured 5 of 5.
  Re-run here, not cited.
* **Floor** — `arm1_none`, the flat compact scaffold. §52 measured 0 of 5
  (0 conforming on all seven tasks). Re-run here.
* **Incumbent** — O1, which must reproduce §52's 2 of 5.
* **Negative controls** — every objective's rank-1 module, on every corpus, is
  also enumerated on `H_par` and `H_d134`. These must stay at **0** for every
  arm including the ceiling. A majority-family module scoring above 0 on either
  invalidates that arm.

Cost is bounded in advance: the compact scaffold is 4 760 programs flat, 25 200
with a 3-ary module, 221 520 with a 4-ary module. 7 objectives × 5 corpora × 3
tasks, deduplicated by published digest, is a few million programs — the same
order as one of §52's L1 arms.

**Predictions, recorded now** (see §0 item 3 — not blind for `∖ t1`):

* O1 helps 2 of 5, reproducing §52.
* O2 helps 5 of 5. Its rank-1 will be the majority class.
* B1 does **not** help 5 of 5: a non-majority class occurs in 5 tasks where the
  majority class occurs in 4, so a pure task-count baseline should pick the
  wrong one. This is the prediction that decides whether the objective is doing
  anything a frequency count could not.
* O4 helps 5 of 5, and I expect it to be the cleanest result.
* O5 helps 0 of 5 and carries almost no signal, because rewriting a Boolean
  circuit into a module call does not remove leaf operations — the module still
  computes them. If so, that is §41's `execution_cost` finding recurring under a
  measured instrument, and it is a result, not a failure of the arm.
* O3 is genuinely uncertain to me.

## 7. Falsification — stated in advance, and to be honoured

| id | fires when | consequence |
|---|---|---|
| **F1** | No objective helps more than the incumbent's **2 of 5**. | Ranking is **not** the fixable layer, §52's framing was optimistic, and the library-induction programme is bounded here. Report plainly as the headline. |
| **F2** | Some objective reaches 5 of 5 **and** B1 or B2 — a trivial frequency count with no MDL term — reaches the same. | **The frequency baseline is the result.** The MDL framing adds nothing, and this is §50's failure mode. Report the baseline as the finding. |
| **F3** | An objective's rank-1 is the majority class on the full six-task corpus but not on the leave-one-outs, i.e. it wins in-sample and loses out-of-sample. | The incumbent's exact failure repeating under a new name. That objective is not a fix. |
| **F4** | No objective reaches 5 of 5 without the oracle window selector of §52. | The windowing was doing the work, not the pooling or the score. (All seven objectives here are non-oracle by construction — §5 — so this fires only if every one of them fails.) |
| **F5** | Any majority-family arm scores `conforming > 0` on `H_par` or on `H_d134`. | The scaffold or the module is doing something other than what is claimed; that arm is void. (`arm4s_offfamily`'s 4 on `H_d134` is the declared exception and the proof the control is live.) |
| **F6** | The winning objective, applied to §41's ten-program language family, still selects the **bytecode-maximal** program. | The fix is **half a fix**: it repairs transfer and leaves §41's defect standing. Report both tracks against the same objective either way. |

**A negative is a first-class deliverable.** If F1 fires, that bounds the
library-induction programme and is worth as much as a win. It will be reported
as the headline, not as a footnote.

## 8. The §41 cross-check

§41's `research/program-length/out/language_family.json` holds ten conforming
programs of one language family, exhaustively enumerated (45 375 evaluated,
`exhausted: true`), all at accuracy 1.000 against a 0.5661 majority constant,
with `description_bits ∈ {4 043 552, 4 043 560, 4 043 568}` and
`bytecodes_total ∈ {41 417, 41 441, 41 465}`. The description-minimal program is
the bytecode-**maximal** one.

The winning objective of §6 will be applied to those ten rows and the selection
reported. This is a cross-check, not a new experiment: the file is read, nothing
is re-enumerated. Where an objective is undefined on a single-task family
(O2, O3, O4 all reduce to a constant reweighting of O1 when `|T| = 1`), that
degeneracy will be stated as such rather than papered over.

## 9. Constraints, and what "done" means

* Repo `.venv` (`/home/brandonin/Documents/typed-crystallization-networks/.venv`,
  Python 3.13.15). The worktree has no `.venv` of its own; this is recorded so
  the runs are reproducible.
* All 337 tests, measured in **both** symlink configurations, as §52 did:
  `generators/computer/engine/node_modules` present → 336 passed / 1 failed
  (`test_panel_interface::test_panel_episode_replays_and_restores`, the
  documented worktree-only failure); absent → 324 passed / 13 failed, all
  environmental, all on main's own code.
* The shipped fixture must reproduce `0.248836 → 0.002231` at 4/4 frozen.
* `out/` holds JSON tables only. No checkpoints, no enumeration dumps, nothing
  large in git.
* Every synthesis number carries an enumeration certificate; every accuracy
  carries its constant and random baselines.
* "No solution exists in this family" and "search failed" are distinguished by
  the certificate and never conflated.
* Achieved difficulty is reported, never the requested configuration.
* Any amendment to this file is recorded in `RESULTS.md` **with the number of
  the clause it replaces**, as §51 did.
