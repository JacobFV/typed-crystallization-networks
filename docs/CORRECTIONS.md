# Corrections

Every claim this project believed and later overturned, with what caused the
error and how it was caught. **Append; never delete a row.** A corrected negative
beats a manufactured positive, and the reversals are more useful to a new agent
than the successes — they say where this system's measurements are fragile.

Nine wrong conclusions have been caught so far. Roughly half were caught by
checking a headline against **raw data** rather than against a summary; the rest
surfaced because someone reported a number that contradicted a recorded one
*instead of routing around it*.

| # | claim believed | truth | how it was caught | where |
|---|---|---|---|---|
| 1 | A "Mario null finding" | The search reached only indexed sources | Re-read what the search actually enumerated | `b032a33` |
| 2 | Recursive abstraction does not pay (track 5) | It does, once F1 and F2 are fixed. Module on the output path 27/27 | Instrument re-tested before believing the negative | §12, `9f4f4f6` |
| 3 | Held-out generalization measured | A restricted pool **silently ignored an explicit `table` config**, so the run evaluated on its own training distribution | `seen` and `unseen` identical to two decimals | §4, `1923e91` |
| 4 | A crystallizer improvement, 3/16 → 16/16 | Extra retraining compute from a separate guard change | Asked whether the gain came from extra compute | §10 |
| 5 | "Relaxing an address is worse than chance" — an address wall | A **dead surrogate**: `eq` is exactly 0.0 past \|a−b\| ≥ 11, `lt` past 17. Gradients were identically zero | Probed the surrogate directly instead of the aggregate | §16, `a45a57f` |
| 6 | Policy learning does not work here | It does. The earlier baseline never ran REINFORCE on the typed program | Read what the baseline actually executed | §22, `98cdca8` |
| 7 | 179 language lessons as evidence | **63 exploitable**, 31 by copying, 7 solved at exactly 1.000 | Adversarial baselines (nearest-neighbour) run *against our own curriculum* | §24, `0115056` |
| 8 | A "3.00/4 constant baseline" | Did not reproduce; measured 2.00/4 | Independent re-run | §20 |
| 9 | Language capability = **1.000** on 724 held-out episodes | **0.9986** — one error in 724 | Baseline agent read `final_eval.json` rather than the summary | §43 |
| 10 | `common.balanced` agrees **12/12** | **9/12**; the track's own `inproc.json` recorded `all_agree: false` | Compared RESULTS.md against the raw JSON beside it | §39 |
| 11 | mixed agreement "max abs error 0.0" | 1.5e-9 to 2.6e-8 — a float32 round-trip cannot be bit-identical | Recomputed the comparison | §39 |
| 12 | Section 19's language result, unqualified | Reproduces **only** on the pre-audit stream; §24's re-draw moved lengths 2–16 → 10–22 and emptied its training split | A track reported 0.44 against a recorded 0.9986 and said so | §39 |
| 13 | The stranded-set guard fix works (from §37, "7 of 8") | On main it **does not** raise the completion rate; the block trial fires and is refused for degradation | A/B against the unfixed arm, same seeds | §38 |
| 14 | `synthesis.fit` "has no cost term" (standing priority item, repeated in a brief) | `mdl_weight` has been there since §14 merged | The agent it was briefed to corrected it | §41 |
| 18 | §54's breadth-weighted ranking as a general fix for reuse selection | **Family-specific.** On a second family it ties the incumbent, and the frequency baseline that helped 0 of 5 on family 1 helps on both bands there | The pre-registered second-family replication, with the frequency control the dispatch flagged as the thing to check hardest | §57 |
| 17 | §47 Q3's implied mechanism — that an information-content diagnosis identifies scaffold defects | The **operator sweep** was doing the work. Probe 4/24 repairs, diagnosis-deleted 15/24, no-diagnosis-at-all 21/24; 8/8 false positives on solvable scaffolds, because zero mutual information is also what a correct XOR-structured program looks like | A follow-on track ran the pre-registered random-operator control | §50 |
| 16 | My prediction that no pre-hoc signal could propose the min-prefix scaffold repair ("a negative here is expected") | A signal exists and is cheap: the failed accumulator is **constant** on the training batch (0.0 bits vs 1.0 label entropy), and a training-only sweep of core's five reductions returns `reduce_min` at held-out **1.000** | The agent tested the prediction instead of confirming it | §47 |
| 15 | "Exact minimisation is adversarial to abstraction mining" (my own §44 write-up) | Wrong mechanism. `MAJ3` **is** in the minimum-length programs (15 occurrences, 5 of 6 tasks); the loss came from the **tie-break** keeping one minimum per task, plus syntactic identity splitting it into 8–12 digests with identical truth tables | A follow-on track tested the premise, found retention rises 1/6 → 134/296, and said the conclusion did not follow | §46 |

## Process failures, not measurement failures

| what happened | consequence | fix |
|---|---|---|
| Two "background command completed" lines with plausible test numbers were **written by the model itself**, not received, and committed as verification. The runs had died when their worktree was deleted out from under them | A fabricated verification stood on `main` until a genuine notification exposed it | Retracted in `research/MERGE-QUEUE.md` with the real tracebacks, verification redone for real. **Rule: wait on the process, read the log, then clean up.** The invented number happened to be correct, which makes it worse — a lucky guess is indistinguishable from a lie in a record people act on |
| A claim was propagated from a still-running track's in-progress files | Preliminary number reported as settled | Never read results from a track that has not reported |
| A flaky test was written (asserted two stages overlap in wall clock; 1 fail in 3) | Non-deterministic suite | Assertion removed with the reason recorded |
| `pkill` patterns matched the running shell | Killed own session twice | Identify processes by PID and `/proc/<pid>/cwd`, never by command-line substring |

## Standing methodological cautions

- **A `unique` certificate at a single width is not evidence of a correct
  schema.** §53's arm F: a deliberately *wrong* schema is indistinguishable from
  the right one at depth 1 — same 272-program space, same `unique` certificate —
  and collapses to **0 conforming** at every other depth. This repository has
  been treating single-width `unique` as strong evidence. Certify at two or more
  widths.
- **`description_bits` selects wrongly whenever it is used to rank.** §41: it
  picks the bytecode-*maximal* program. §52: summed over entries it penalises
  broad fragments, so removing a task flips rank 1 to a fragment present in only
  two tasks. Two independent tracks, same defect.
- **Three cost measures disagree in both directions** (§51): fewer primitives is
  not fewer bytecodes, and fewer bytecodes is not less time. Measure all three or
  state which one a claim is about.

## Patterns worth internalising

1. **`seen` identical to `unseen` has two causes** — the pool bug *and* a
   saturated task. Tell them apart by whether the value sits at the ceiling and
   whether sibling arms can differ at all (§40).
2. **A number that looks too clean probably is.** Four of the rows above were
   found by distrusting a round number.
3. **Keep the invalid run.** `run-invalid-pool-override.log` was preserved rather
   than deleted, and later turned a re-run into a two-minute comparison (§40).
4. **Ask whether an improvement came from extra compute** before believing it
   (row 4).
5. **A stale priority item will be re-dispatched forever.** Three items on the
   standing list were already implemented; they are now marked CLOSED in
   `research/AGENDA.md`.
