# Pre-registration — is the §45 Dyck fix reachable by the system, or was it authored?

Written and committed **before any arm was run**, per the house standard set by
§44 and §45. Nothing below was edited after the first arm started; deviations, if
any, are recorded in `RESULTS.md` under "Deviations from pre-registration".

## Background this track takes as given (not re-litigated)

FINDINGS §45 (`research/language-post-audit/RESULTS.md`, branch
`language-post-audit` @ `090b780`, not merged) established, on the re-drawn
non-exploitable `context_free_language` stream:

- the shipped stage-B scaffold contains **no** conforming program — space 45,375,
  exhausted, 0 conforming, certificate `complete`; the cause is expressiveness,
  since every member is a thresholded affine function of a bracket **count** and
  §24's re-draw made counts identical in both classes (the honest counting program
  scores exactly the majority, 0.5262);
- post-audit, `balanced ⟺ min prefix ≥ 0`, and a minimum is not a sum;
- adding a running `min` beside the running `add` — `min` and `and`, both already
  in `tcn/operators.py` — yields a program at **1.000 on all 859 held-out
  episodes at unseen lengths 16–22**, recovered by the **unmodified** search
  inside a **hand-chosen** window `c ∈ [95,110)` (84,375 evaluated, exhausted,
  110 conforming, certificate `complete`);
- the full 680,625-program space was started and **stopped at ~52%**, so it
  carries **no** certificate.

**The solution exists and enumeration finds it in a window chosen by knowing the
answer.** This track asks whether the system could get there on its own. It adds
**no** new operator and modifies nothing under `tcn/` or `generators/`.

## Shared configuration, fixed in advance

- **Stream**: `hardening='context_free_language'` (post-audit), pinned explicitly
  at every `common.dataset` call. The vendored `common.py` takes `hardening` as a
  required-by-convention argument with no silent inheritance (the §39 defect).
  Every number in `RESULTS.md` names its stream.
- **Split**: §45's, verbatim, so numbers are directly comparable — train lengths
  {10,12,14} (n=24), held out seen lengths {10,12,14} (n=120), held out **unseen**
  lengths {16,18,20,22} (n=859). Per-length counts and the train/test
  string-overlap check are reported for the split actually drawn.
- **Baselines beside every accuracy**: majority constant (0.5262 on the unseen
  split), random 0.5, best fitted feature.
- **Certificates**: every enumeration number is reported with `evaluated`,
  `space_size`, `exhausted`, `conforming` and `certificate`. An unexhausted
  enumeration is reported as a **bounded search with no certificate** and is
  stated to support neither "no solution exists" nor "search failed".
- **Probes are supervision only** and are never packed into a program input.
- `positions=22` (length-matched to the post-audit stream) throughout, as in §45's
  Dyck arms.

---

## Q1 — Does the search find it without the hand-chosen window?

**Arm 1a.** `tcn.search.enumerate_fit`, unmodified, `rank='order'`, on the full
min-prefix scaffold space: `sub_range = range(0,121)`, i.e. 121 × 5 × 5 × 15 × 15
= **680,625** programs. Budget: the space is partitioned into **eleven disjoint
contiguous windows of 11 `c` values each** (61,875 programs per shard) run
concurrently; the union covers `[0,121)` exactly, so eleven `exhausted` shards
constitute an exhaustive decision of the full space. `sub_range` is an argument
§45's own runner already exposes; the search itself is untouched. **Stated
budget: 3 h wall clock.** If the union is not exhausted inside it, Q1 is reported
as a bounded search with no certificate.

**Arm 1b (control, and it must reproduce).** The same runner on §45's window
`c ∈ [95,110)` must return space 84,375, exhausted, **110 conforming**,
certificate `complete`. If it does not, the recorded §45 number does not
reproduce and that is itself the finding.

**Reported for each shard and for the union**: space size, evaluated, exhausted,
conforming count, certificate, wall clock, seconds/program. Plus the **global
mixed-radix enumeration index of the first conforming member**, and its fraction
through the 680,625-program order, since that determines whether any sensible
ordering finds it early. The index formula is order-preserving under the `c`
partition and is validated against §45's recorded witness index 571,822.

**Falsifies Q1 ("enumeration reaches it unaided")** if: the union of shards is
not exhausted within the 3 h budget, **or** it is exhausted with 0 conforming. In
either case the conclusion recorded is that the fix is **not** reachable by
enumeration as configured and §45's recovery depended on knowing the answer.

**Supports Q1** if the union is exhausted with conforming > 0 **and** the wall
clock and the first-conforming index are stated. A first-conforming index late in
the order (say > 50% through) is reported as a caveat even when the search
succeeds, because it means default enumeration order is not what found it.

## Q2 — Does the gradient path find it, and does it beat its own control?

§44 measured the shipped gradient path solving **1 of 6** corpus tasks, so this
is the question that matters most for the architecture.

**Arm 2a — min-prefix scaffold.** `tcn.learning.SoftProgram` on
`dyck_scaffold.stage_b_dyck(positions=22)`, the identical space arm 1a
enumerates. **8 seeds** (0–7), **600 steps**, Adam, lr 0.05, logits perturbed by
`0.01 * randn` (SoftProgram zero-initialises them). Both temperature settings are
run, because §19 measured `lt`'s surrogate gradient as **exactly 0.0 at
delta ≥ 17** while this scaffold operates at delta up to 22:
  - **default** (`tau_lt = tau_eq = 1.0`, the shipped path);
  - **widened** (`tau_lt`, `tau_eq` set as the language track's own
    `stage_b_grad_tau` arm does), reported separately and never merged with the
    default.

**Arm 2b — counting-only scaffold (the control).** The identical protocol on
`scaffolds.stage_b(positions=22)`, where §45 proves by exhaustion
(`complete`, 0 conforming) that **no** conforming program exists. This is the
control that makes 2a interpretable.

**Reported**: conforming rate over seeds (a run counts as conforming iff its
argmax export has exact train max-error ≤ 1e-6, i.e. train accuracy 1.000 on all
24 episodes); **steps to first conforming export** (argmax export checked every
10 steps); accuracy on all three splits with the **0.5262 majority and 0.5 random
beside it**; and whether the run selects the `min` readout correctly.

*Operationalising "selects the `min` node correctly", fixed now.* In this
scaffold the running-`min` chain is structural (single-candidate nodes); what is
**searched** is the readout `min_ok` over the `{eq,ge,le} × {-2..2}` grid. Since
the running minimum on this stream lies in {0,−1,−2}, a readout is
  - **informative** iff it is not constant over the 859 unseen episodes;
  - **correct** iff it equals `min_prefix(s) ≥ 0` on all 859;
  - **vacuous** otherwise — and a vacuous `min_ok` means the exported program has
    collapsed to the counting program, i.e. to arm 2b's family.
Per-seed classification of `min_ok` is reported.

**Falsifies Q2 ("the gradient path finds the structure")** if the conforming rate
on 2a is not strictly greater than on 2b, **or** if 2a conforms at a comparable
rate while its `min_ok` readouts are vacuous. Either is reported as
**noise-fitting, not success** — a path that "succeeds" equally on a scaffold
that provably contains a solution and on one that provably contains none is not
finding structure.

**Supports Q2** if 2a conforms on strictly more seeds than 2b **and** the
conforming exports carry a correct `min_ok`, with steps-to-conforming stated.

Additional expected-negative note fixed in advance: if **both** arms conform on
0 seeds, that is a clean negative for the gradient path and is reported as such —
it is not evidence for or against expressiveness, which arm 1 settles.

## Q3 — Could the scaffold change have been proposed rather than authored?

The scaffold change ("add a running `min`") was chosen by reading §45, which had
already localised the answer. Q3 asks whether anything in the system could
propose it from evidence available **before** knowing the solution: only the
failed counting scaffold, its exhaustion certificate, and the 24 training
episodes.

**Arm 3a — is there a pre-hoc "your statistic is insufficient" signal?**
Compute, from the training episodes alone, the failed family's **sufficient
statistic** — every stage-B member is a function of
`(length, #open in the first K symbols)` for the single cut `K = min(positions,
length − c)`, so the family's whole discriminative content is the vector of
prefix open-counts. Test for **label collisions**: two training episodes agreeing
on the statistic and disagreeing on the label. A collision is a proof, computable
without any search and without knowing the answer, that **no** function of that
statistic fits the training set. Reported as a count, with the colliding
episodes exhibited.

**Arm 3b — does the failed family's error structure point anywhere?** §45 notes
24 members tie at train 0.75 with held-out spread 0.474–0.703. Re-derive that tie
set and ask what it says: which episodes the tied members get wrong, whether the
residual is concentrated on a structurally identifiable subset (e.g. episodes with
`min_prefix < 0` and matching counts), and whether that residual is a signal a
mechanical procedure could read.

**Arm 3c — is the scaffold change mechanically generable from core?** The
proposal "add a second accumulator" is a scaffold template with one hole: the
binary operator folded along the prefix. Enumerate that hole over **every**
operator core already declares with an `CNT × CNT → CNT` signature (from
`tcn.operators`), build the corresponding two-accumulator scaffold for each, and
report which ones admit a conforming program. If `min` (or an equivalent, e.g.
`max` with flipped step signs) is recovered by a mechanical sweep over core's own
operator inventory, then the scaffold change **was** proposable without knowing
the answer, and the human input reduces to the template "add a second
accumulator". If it is not, scaffold design is a human input at the operator
level too. Conformance per operator is decided by a semantics simulator that is
**first validated against the real typed program** on random members (the §45
`bound.py` discipline: report the number of members checked and the number of
mismatches); any headline is confirmed on the real program, not the simulator.

**Falsifies Q3 ("the system could have proposed it")** if no pre-hoc signal
exists — no collision in 3a and no readable residual structure in 3b — or if 3c's
mechanical sweep does not reach a conforming operator. Then the conclusion
recorded, plainly and without gloss, is that **scaffold design is currently a
human input** and that is a limit of the architecture.

**A negative on Q3 is expected and is the valuable outcome.** It is recorded as a
limit, not softened.

## Verification obligations

- All non-environmental tests pass. The ~13 worktree failures (`computer` and
  `panel` tests needing gitignored `node_modules`, including
  `test_panel_interface.py::test_panel_episode_replays_and_restores`, which fails
  on main's own code in any worktree with a symlinked `node_modules`) are verified
  environmental by removing `research/dyck-learnability/` and re-running, as §45's
  track did.
- The shipped fixture reproduces **0.248836 → 0.002231 at 4/4 frozen**.
- `git diff main...HEAD -- tcn/ generators/` is empty.
- Large artifacts (episode pickles, per-program dumps) stay out of git.
- Any recorded number that does not reproduce is reported as a finding.
