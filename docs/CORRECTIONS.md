# Corrections

Every claim this project believed and later overturned, with what caused the
error and how it was caught. **Append; never delete a row.** A corrected negative
beats a manufactured positive, and the reversals are more useful to a new agent
than the successes — they say where this system's measurements are fragile.

~~Nine wrong conclusions have been caught so far.~~ The table has 34 rows as of
2026-09-10; rows 15–21 were appended out of numeric order, so a row's number, not
its position, is its identifier. Roughly half were caught by
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
| 8 | ~~A "3.00/4 constant baseline" did not reproduce; measured 2.00/4~~ **THIS ROW WAS ITSELF WRONG** | **3.00/4 reproduces.** Recomputed directly through `_joint_reference` on the recorded protocol (`split='test'`, indices 30000–30015, objectives cycled): always-True **3.00**, always-False **1.00**, uniform-random 1.6875. No protocol yields the 2.00/2.00 that was recorded. `tcn/cli.py:174` and `docs/VALIDATION.md` both always said 3.00 | The §62 audit re-derived it from `Host`; confirmed independently here | §20, §62 |
| 9 | Language capability = **1.000** on 724 held-out episodes | **0.9986** — one error in 724 | Baseline agent read `final_eval.json` rather than the summary | §43 |
| 10 | `common.balanced` agrees **12/12** | **7/12** — five episodes disagree. *(This row itself first said 9/12; that was wrong. The original check printed only the first three episodes and generalised from one disagreement. Recounted from `inproc.json`: `[T,T,F,F,T,T,F,T,F,F,T,T]`.)* | Compared RESULTS.md against the raw JSON; the undercount was caught by the §62 audit | §39, §62 |
| 11 | mixed agreement "max abs error 0.0" | 1.5e-9 to 2.6e-8 — a float32 round-trip cannot be bit-identical | Recomputed the comparison | §39 |
| 12 | Section 19's language result, unqualified | Reproduces **only** on the pre-audit stream; §24's re-draw moved lengths 2–16 → 10–22 and emptied its training split | A track reported 0.44 against a recorded 0.9986 and said so | §39 |
| 13 | The stranded-set guard fix works (from §37, "7 of 8") | On main it **does not** raise the completion rate; the block trial fires and is refused for degradation | A/B against the unfixed arm, same seeds | §38 |
| 14 | `synthesis.fit` "has no cost term" (standing priority item, repeated in a brief) | `mdl_weight` has been there since §14b merged | The agent it was briefed to corrected it | §41 |
| 21 | The language demo's shipped `_LANGUAGE_RULE` as "§19's result" | **A different program.** `stage_b.json` records `{symbols:99, plus:1, minus:3, answer:12}`; the code shipped `{symbols:101, plus:0, minus:4, answer:6}`, the 1.000-scoring member of a **ten-way tie** training accuracy cannot break. The 1.000 §43/§62 overturned was **baked into shipped code**, not just prose | The demo-fix track compared the shipped constant against the recorded selection | §63 |
| 20 | §59's (and my) naming of `(0, 3069)` as the missing refinement bound | **Unsound as a type bound.** `_m1` reads `obs[a+2]`, which carries `a`'s type and reaches **3071**. The clamp's runtime value is not the type's bound. Sound bound is `(0, 3071)`, and only after rewriting `min(p+d,L)` as `p+min(d,L−p)` | A follow-on track tried to declare it and proved the range exhaustively first | §61 |
| 19 | §58's prose that the 3-node motif recurs in **all three** real artifacts (my write-up) | It recurs in **two** — language and visual. The computer artifact's 3-node instance has a constant address at arity 3. The 2-node form spans all three. §58's structural conclusions are unaffected | A follow-on track checked the example before building on it | §60 |
| 18 | §54's breadth-weighted ranking as a general fix for reuse selection | **Family-specific.** On a second family it ties the incumbent, and the frequency baseline that helped 0 of 5 on family 1 helps on both bands there | The pre-registered second-family replication, with the frequency control the dispatch flagged as the thing to check hardest | §57 |
| 17 | §47 Q3's implied mechanism — that an information-content diagnosis identifies scaffold defects | The **operator sweep** was doing the work. Probe 4/24 repairs, diagnosis-deleted 15/24, no-diagnosis-at-all 21/24; 8/8 false positives on solvable scaffolds, because zero mutual information is also what a correct XOR-structured program looks like | A follow-on track ran the pre-registered random-operator control | §50 |
| 16 | My prediction that no pre-hoc signal could propose the min-prefix scaffold repair ("a negative here is expected") | A signal exists and is cheap: the failed accumulator is **constant** on the training batch (0.0 bits vs 1.0 label entropy), and a training-only sweep of core's five reductions returns `reduce_min` at held-out **1.000** | The agent tested the prediction instead of confirming it | §47 |
| 15 | "Exact minimisation is adversarial to abstraction mining" (my own §44 write-up) | Wrong mechanism. `MAJ3` **is** in the minimum-length programs (15 occurrences, 5 of 6 tasks); the loss came from the **tie-break** keeping one minimum per task, plus syntactic identity splitting it into 8–12 digests with identical truth tables | A follow-on track tested the premise, found retention rises 1/6 → 134/296, and said the conclusion did not follow | §46 |
| 22 | §14a: "requiring exactness at every position of a validation split is what fixed" the tie-break (repeated in `STATUS.md` B9) | **Inverted.** `discrete-perception/out/tiebreak.json`: `validation_filtered` returns the identical program as `lexicographic`, 3 of 384 slots wrong; only the gradient runs are exact. The track's own RESULTS said the filter "does not fix it". Nothing fixed the tie-break | The §62 audit read the file the section cited; applied by the record-audit gate | §14a, §62 |
| 23 | §21: ceiling advantage "exactly 0.0000 at every context" for `object_ids`, `raster_rank`, `is_object_0` | True only at **raw-byte** contexts. `bounds.json` is 15 × 14, and the derived contexts are non-zero in 8, 8 and 2 of 15, up to +0.2018. The permutation certificate, which carries the section, is unaffected | §62 audit; the track's report carried the qualifier FINDINGS dropped | §21, §62 |
| 24 | §19: the language program is "exact at lengths 8 through 16" | Exact at 8, 10, 12, 14; **0.5 at 16** — the single error in 724 (`final_eval.json` `per_length`). Also: stage B was `unique: false` with 10 conforming | §62 audit; §45 had already located the error at length 16 | §19, §45 |
| 25 | §36/§43/`README.md`: the typed side wins deployment footprint "across the board", 19–60 MB | **A verdict, not only a figure.** The range omitted the fourth artifact: `visual.pyz` is 377.5 MB RSS and 5,179 ms cold start, above the 270.8 MB torch baseline. Only the compiled zipapp (§48, 30.2 MB) wins on visual. §43's "1.7–7.2 s" torch import is 0.83–9.59 s in `latency.json` | §62 audit swept every `pyz_*.json` rather than the three quoted | §36, §43 |
| 26 | The visual parse's 27.6× "decomposes as 11.1× × 2.25×" (§41, §48, `README.md`, `STATUS.md`) | The factors multiply to **24.9×**, `attribute_visual.json`'s own total; 27.6× is `bench_visual.json`, a different timing of the hand-written reference | §62 audit multiplied the factors | §48, §41 |
| 27 | §41 "certifies `none exists` at spans 4 through 29" | Only spans 4, 8, 12, 16, 20, 24, 28, 29 were enumerated — 8 of 26. Plausible by monotonicity; not certified | §62 audit listed the spans in `span_sweep*.json` | §41 |
| 28 | §47 Q2: both scaffolds pick "the identical `c` at every seed", and `symbols`' first-step gradient is "exactly 0.0" | Identity holds in 3 of 4 arm pairs; at `tau_lt=128, steps=3000` the scaffolds diverge. The gradient is 0.0 only at `tau_lt=0`. The conclusion survives on the identical medians | §62 audit re-read `q2_summary.json` | §47 |
| 29 | §43/`README.md`: "20/20 rectangles, 20/20 parent links … on every held-out screen" | Screens carry 15–20 widgets; the totals are **227/227** and 12/12 trees. "20/20 on every screen" is true of 8 of 12. The verdict (perfect everywhere) stands | §62 audit | §43 |
| 30 | `README.md`: §25 holds "the persistent module library and curriculum artifact flow"; crystallization refuted in "sections 7, 12, 37"; the compiled result is "section 42"; the language 0.9986 quoted without its stream | §25 contains neither (`STATUS.md` B1 says no registry outlives a script); the refutations are §1/§3, §10, §37; **§42 never existed** — the result is §48; the 0.9986 is pre-audit-stream only | §62 audit cross-document pass; the gate now fails on a citation of a missing section | `README.md`, §62 |
| 31 | `STATUS.md`: policy-learning has "no `RESULTS.md`"; object-identity has "unrendered placeholders"; B7 "the entry point does not run"; "179 Python tests", "370 of 372" engine tests | All stale — both tracks reported, B7 was closed in §20 (the file's own header said so), and the counts are 337 and 372/372 | §62 audit; the gate re-checks all four | `STATUS.md` |
| 32 | Twenty-three further medium and low figures (§11 "3x3", §12/§3 "0 of 10 successes", §14a "4/4 and 6/6", §16's minima band, §17's 77% / 0.394 ms / "`eq` and `pack` only", §24 "bit-identical", §25 "two to three orders" and "0 of 44", §26 "2.18e4", §36's 99.68% scope and "consistent with 206×", §44's order-dependence, §46 "11th to 18th", §51 "3.9×", §53 "4 of 24", §54 "two digests", §55 "widths 5, 7, 12", §56 "−3.1%", and others) | Each corrected in place to its artifact with the original struck through; the evidence per item is in `research/record-audit/RESULTS.md` §1.2–1.3 and "Gate, 2026-09-10" | §62 audit, applied by the record-audit gate | many |
| 33 | §2 and `STATUS.md`: dense supervision reaches 88–94% "at 2,120 candidates per node"; §7 "9.7e-2 at depth 6" | **Found by the gate, not the audit.** No committed `n_candidates` list contains 2,120 — the widest node is 5,776 in `FD_supervision.json` — and the depth-6 density is 9.6e-2 in the committed JSON. Both figures came from track prose that disagrees with the track's own raw files | The gate's per-section coverage pass, spot-checked independently | §2, §7 |
| 34 | The §62 audit itself: "§53's arm F and §61 both established" that a single-width `unique` certificate is not evidence | The section that established it is **§60** — a deliberately wrong schema earns the same `unique` at width 128 — and §61 is the refinement bound | The gate re-derived the audit's citations before applying them | §62 |
| 35 | §62 audit's strike of README's "persistent module library" claim as not holding up | **The strike was itself an overcorrection.** §25 was the wrong citation and the library has no FINDINGS section, but it holds within a domain: `research/module-library/RESULTS.md` measured stage 2 given stage 1 at 48 programs, exhausted and `unique`, held-out error 0.0, against 4.9e10 programs without; `tcn/library.py` is on main (`60616f7`) and tested. The strike relied on a `STATUS.md` B1 line written before that merge | Supervising-session review of the audit branch before merge, checking the library's ancestry on main and B1's edit timestamp | README, STATUS B1, §62 |

## Process failures, not measurement failures

| what happened | consequence | fix |
|---|---|---|
| Two "background command completed" lines with plausible test numbers were **written by the model itself**, not received, and committed as verification. The runs had died when their worktree was deleted out from under them | A fabricated verification stood on `main` until a genuine notification exposed it | Retracted in `research/MERGE-QUEUE.md` with the real tracebacks, verification redone for real. **Rule: wait on the process, read the log, then clean up.** The invented number happened to be correct, which makes it worse — a lucky guess is indistinguishable from a lie in a record people act on |
| A claim was propagated from a still-running track's in-progress files | Preliminary number reported as settled | Never read results from a track that has not reported |
| A flaky test was written (asserted two stages overlap in wall clock; 1 fail in 3) | Non-deterministic suite | Assertion removed with the reason recorded |
| The first `research/record-audit/verify.py` asserted the record's *believed* values against the artifacts ("the count must be 9", "the best constant must be 2.00") | Its checks could only fail, including after the record was corrected — 18 permanent failures that told no one anything | Rewritten so every check reads the claim *from the document* and compares it with the artifact; a claim that disappears fails. It runs as `tests/test_record_gate.py` |
| Two corrections (rows 8 and 10) were accepted on a summary check rather than the raw file | A catalogue of overturned claims carried two wrong rows | A correction is a claim: the gate re-derives it from the artifact before it is applied |
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
