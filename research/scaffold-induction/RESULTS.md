# Scaffold induction as a cross-domain outer loop — results

`PREREGISTRATION.md` was committed before the first arm. `tcn/` and
`generators/` are untouched. Every figure below sits inside a
`<!-- BEGIN:x -->` block rendered from `out/` by `report.py`; `verify.py`
re-renders each block, re-derives every headline from raw JSON without importing
the run or report scripts, and writes `out/verify.json` and `out/headline.json`.

**[Single-configuration evidence.]** One seed set per domain, one base scaffold
per domain, one defect generator, one edit enumerator, one estimator family, one
train/held-out split per domain. The domain count is small enough that the
cross-domain result is anecdote-strength and is labelled so wherever it is
stated — the same disclosure §50 made about its own task count.

---

## Amendments to the pre-registration

Every change to `PREREGISTRATION.md` after it was committed, in order, with what
it replaces. **All of them were made during corpus construction, before any arm
ordering or arm cost was computed**, and none of them moves a number that had
already been measured — the arms had not run. They are listed anyway, because a
pre-registration that is quietly re-read is not one.

**A1 — `rel` is a three-step scaffold over three entities, replacing §2.1's
"two-step reachability relation of a 4-entity directed graph".** The generator's
target is membership in the full transitive closure. A two-step scaffold cannot
express it: the base scaffold itself was decided as having **no** conforming
member over all episodes, exhausted, certificate `complete`, so no edit of any
defect of it could have been a repair and the domain would have contributed
nothing. Three edge steps is what the closure needs at three entities (a simple
path between distinct vertices is at most two edges, a cycle at most three). The
base's certificate on all episodes is recorded in `out/cases_rel.json` and
checked by `verify.py`.

**A2 — `rel` uses more episodes than the pre-registered seed ranges, and `bool`
trains on every even row rather than a subset.** With the smaller splits, almost
every defect still had a member conforming on the training episodes and the
domain admitted one case. The admission rule (§2.3) is unchanged; only the
episode counts moved. Both counts are rendered in the corpus block.

**A3 — `keep_prefix`'s `k` values are {1, 2, 3, 4, 6, 8}, and defects are
enumerated over every node of the base scaffold.** §2.2 fixed neither. Stated
here so the defect enumeration is reproducible from the document.

**A4 — the `lang` domain is dropped**, under §2.1's own drop clause. The reason
is in the corpus section below.

**A5 — the training-only decision is skipped when the all-episode decision
returns a witness.** A member conforming on every episode conforms on the
training ones, so the second decision would be redundant; the field records that
it was implied by the witness rather than measured. No metric changes.

**A6 — §3.4's edit-space sizes were probe figures and `rel`'s has moved** with
A1. The measured sizes are in the corpus block; nothing in §3.4 is a criterion.

**A7 — §6.1's expectation formula was written down wrong and is corrected.** The
document said `(s − k + 1)/(k + 1)` for a uniform tier; the expected number of
draws to the first of `k` good items among `s`, without replacement, is
`(s + 1)/(k + 1)` — the formula §6.1 also named as "the flagship's". The typo
was in the pre-registration and in the first draft of both `analyse.py` and
`verify.py`, which is exactly the failure mode a verifier that shares a
mis-derivation cannot catch; it was found by checking the degenerate case
(`s = k = 1` must cost one draw, not a half) before any arm was costed. No
measured number changes, because none had been produced.

---

## The pre-registered verdict

<!-- BEGIN:criteria -->
<!-- END:criteria -->

<!-- BEGIN:costs -->
<!-- END:costs -->

<!-- BEGIN:ratios -->
<!-- END:ratios -->

---

## The corpus

Defects are typed narrowings of a base scaffold; a defective scaffold is
admitted only when it has **no** member conforming on the training episodes,
exhausted, certificate `complete` — a proof, not a timeout — and its typed edit
space contains at least one repair.

<!-- BEGIN:corpus -->
<!-- END:corpus -->

**The `lang` domain was dropped, and why.** §19/§45's bracket-grammaticality
family was pre-registered as a fourth domain and does not survive contact with
the generic enumerator. The post-audit stream emits only long strings, so the
scaffold needs its full width; at that width the program carries a large
constant pool, and `legal_candidates` exceeds its enumeration budget on almost
every operator, so the typed edit space collapses to a handful of edits and
takes longer to *enumerate* than a whole corpus takes to decide in the other
domains. §2.1's drop clause is exercised: the domain is dropped, disclosed here,
and its partial numbers are in `out/memory_floor.log` and the probe scripts. The
cross-domain claim therefore rests on the remaining domains, which is the
pre-registered minimum and no more.

<!-- BEGIN:percase -->
<!-- END:percase -->

---

## What the inherited prior learned

<!-- BEGIN:estimator -->
<!-- END:estimator -->

<!-- BEGIN:families -->
<!-- END:families -->

---

## The deployed protocol (M2), and the §65 trap

M1 above is the pre-registered criterion: expected edits to the first edit whose
scaffold conforms on the **held-out** episodes. M2 is what a system that cannot
see held-out data would actually do — enumerate in the arm's order and stop at
the first edit with a **training** conformer — and it is reported beside M1, not
substituted for it.

<!-- BEGIN:deployed -->
<!-- END:deployed -->

---

## Validity checks, read before the criteria

<!-- BEGIN:validity -->
<!-- END:validity -->

**V2 — the decider.** Every (case, edit) pair is decided by
`tcn.search.enumerate_prefix`, core machinery rather than a track simulator, so
there is nothing to validate against `Program.execute` in the way §45 and §47
had to validate theirs. It is checked anyway, two ways: a deterministic sample
of edits per case is re-decided by the flat `enumerate_fit` walk over the same
space, and **every** recorded witness is re-executed through `Program.execute` on
**every** episode of its domain. `validate_decider.py` also asserts that the edit
enumeration is reproducible — that re-applying the defect and re-enumerating
yields exactly the committed key set — because an ordering claim over an
irreproducible list would mean nothing.

<!-- BEGIN:decider -->
<!-- END:decider -->

---

## S1 — re-scoring §50's own corpus on held-out conformance

<!-- BEGIN:s50 -->
<!-- END:s50 -->

---

## Resources

**A process-management near-miss, recorded rather than tidied away.** Trying to
keep the worker count at its cap, this track sent `SIGSTOP` to a PID selected by
`pgrep -f "run_domain.py rel"` — and the pattern matched the shell running the
command, which stopped itself. This is exactly the failure
`docs/CORRECTIONS.md` already records ("`pkill` patterns matched the running
shell; identify processes by PID and `/proc/<pid>/cwd`, never by command-line
substring"). It cost one hung command and no data. The rule is not new; it was
not followed. Nothing was paused after that, and the S1 arm was run after a
corpus job had finished rather than beside four.

Every job ran inside a capped scope (`MemoryMax=20G`, `CPUQuota=100%` per
worker, single-threaded BLAS, at most four workers), with `MemAvailable` checked
against the floor before each phase and logged to `out/memory_floor.log`. The
other project's GPU processes were never touched.

<!-- BEGIN:resources -->
<!-- END:resources -->
