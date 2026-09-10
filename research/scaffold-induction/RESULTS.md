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

---

## S1 — re-scoring §50's own corpus on held-out conformance

<!-- BEGIN:s50 -->
<!-- END:s50 -->

---

## Resources

Every job ran inside a capped scope (`MemoryMax=20G`, `CPUQuota=100%` per
worker, single-threaded BLAS, at most four workers), with `MemAvailable` checked
against the floor before each phase and logged to `out/memory_floor.log`. The
other project's GPU processes were never touched.

<!-- BEGIN:resources -->
<!-- END:resources -->
