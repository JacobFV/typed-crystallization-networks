# Pre-registration — does §47's pre-hoc scaffold probe generalise, or is it a constant-detector?

Written and committed **before any arm was run**, per the house standard set by
§44, §45, §46 and §47. Nothing below was edited after the first arm started;
deviations, if any, are recorded in `RESULTS.md` under "Deviations from
pre-registration".

## The claim under test

FINDINGS §47 Q3 (`research/dyck-learnability/RESULTS.md`) established, on the
post-audit `context_free_language` stream, that a **pre-hoc** signal identified
the §45 scaffold's defect using only the failed scaffold, its exhaustion
certificate, and the 24 training episodes:

- the failed scaffold's terminal accumulator `acc21` had `distinct_values: [0]`
  — literally constant across the training batch, **0.0** information bits
  against **1.0** bits of label entropy;
- a training-only sweep of core's five reductions returned `reduce_min` at
  held-out **1.000 on n=859** while `sum`, the shipped choice, capped at 0.75;
- a training-only sweep of the fold hole over the **nine** operators core
  declares at `CNT × CNT → CNT` returned exactly `min` and `max`.

§47 recorded the limit of that claim itself: *"Whether the same probe generalises
beyond a constant-node diagnosis is untested and should not be assumed."*

**Hypothesis H.** A cheap, label-light diagnostic over a *failed* scaffold —
per-node information content on the training batch, plus a swap sweep over core
operators of matching signature — predicts *which* scaffold change repairs it,
across tasks and across defects, without knowing the answer.

**Everything here is a prediction task.** The probe commits to a prediction,
written to `out/predictions.json` and **committed to git**, before that file is
joined against the known repairs. `probe.py` and `run_probe.py` never read a
case's `repair` or `solvable` field; `score.py` performs the join and is the only
file that does. This is enforced by construction, not by intention, and the
commit order is the evidence.

## Scope and honesty about what is available

The project record contains **very few** failed scaffolds with an independently
established repair. This track uses:

- **two recorded task families** — the post-audit Dyck/counting scaffold
  (§45/§47) and the tight 6-input Boolean scaffold (§44/§46);
- **two constructed tasks** on the Dyck family's own template, whose repair is
  fixed by construction rather than by knowing the probe's answer.

That is **two recorded tasks**, not two dozen. Every rate below is reported with
its denominator and with the number of *independent tasks* stated separately from
the number of *scaffolds*, and the verdict is written as an anecdote-strength
claim if the independent-task count stays at two.

## Shared configuration, fixed in advance

- **Stream**: `hardening` is pinned explicitly at every `common.dataset` call
  (the §39 defect). Language cases use `hardening='context_free_language'`
  (post-audit) unless the case says `'none'` (pre-audit), and every number names
  its stream.
- **Split**: §45's, drawn by the vendored `splits.py` — train lengths {10,12,14}
  (n=24), held-out seen lengths (n=120), held-out **unseen** lengths {16,18,20,22}
  (n=859). Re-drawn here and asserted equal to §45's per-length counts before any
  arm.
- **Baselines beside every accuracy**: constant/majority and random.
- **Certificates**: every enumeration number carries `space_size`, `evaluated`,
  `exhausted`, `conforming`, `certificate`. An unexhausted enumeration is a
  bounded search with no certificate and supports neither "no solution exists"
  nor "search failed".
- **Probes are supervision only.** The diagnostic reads training labels; it is
  never packed into a program input.
- **Simulators**: any family simulator is validated against the real typed
  program per variant with `Program.execute` (§45's `bound.py` discipline,
  §47's per-episode form), reporting members checked and mismatches; every
  headline is re-confirmed on the real program.
- **No core hook.** `git diff main...HEAD -- tcn/ generators/` must be empty.

---

## The probe, pinned in full before it is run

`probe.py`. Two **separable** stages, so that the falsification "the sweep is
doing the work, not the diagnosis" can be tested directly rather than argued.

### Stage D — degeneracy diagnosis (per-node information on the training batch)

1. **Member selection.** `m*` = the member of the failed scaffold's own space
   with the highest accuracy on the training episodes, ties broken by
   enumeration order (mixed-radix over the scaffold's candidate lists). Training
   labels only; held-out episodes are never used to select anything.
2. **Trace.** Execute the **real typed program** under `m*` on each training
   episode and record every node's value.
3. **Per node**: number of distinct values, `H(node)`, `I(node; label)` in bits,
   and `H(label)`.
4. **Fire rule (pinned).** The probe *fires* iff at least one node satisfies
   `distinct == 1` **or** (`I(node; label) == 0` and `H(label) >= 0.5` bits).
5. **Site (pinned).** Among fired nodes, the one of greatest `depth`; ties broken
   by latest position in the node list.
6. **Rule D0 (pinned).** If **no** member of the scaffold runs on the training
   batch at all, the probe fires with site = the node whose operator raises.

Stage D records `fired`, `site`, and the full per-node table for every case,
including the cases where it does not fire.

### Stage S — swap sweep over core operators of matching signature

7. **Candidate set.** For the site's producing operator, its signature σ is read
   off the resolved operator, and the candidates are every name in
   `tcn.operators.BINARY` / `UNARY` that `registry.resolve` accepts at σ.
   Nothing is hand-curated and nothing is added to core.
8. **Two repair families**, both declared here in advance:
   - **swap**: substitute each candidate for the site's operator, at the site and
     at every node of the site's homogeneous chain;
   - **fold**: §47's template — add a second accumulator folding the chain that
     feeds the site with each candidate operator, with a readout over the same
     grid the scaffold already searches. The *template* is a human input; which
     operator fills the hole is what the sweep decides.
9. **Fit.** For each candidate, re-enumerate the scaffold's free parameters and
   fit on the **training episodes only**. Record best training accuracy, the
   number of members tied at it, and the enumeration-order-first selection.
10. **Prediction (pinned).** The candidate with the highest best-train accuracy;
    ties are reported as a tie set and the prediction counts as correct only if
    the tie set is **contained in** the set of known repairs.

### Controls, all pinned

- **B-random.** A uniform draw from the signature-matching candidate set.
  Reported analytically as |correct| / |candidates| and empirically over 2,000
  draws scored on training conformance.
- **B-sweeponly.** Stage S run at a *fixed default site* — the output-adjacent
  accumulator — with Stage D deleted entirely. If B-sweeponly's prediction equals
  the probe's on every case, the information framing contributes **nothing** to
  repair selection and the write-up says so.
- **B-terminal.** Stage D replaced by "always blame the output-adjacent node".
  Compared against Stage D on site selection.

---

## Cases

`cases.py`. Each case declares: the scaffold, the episodes, whether a conforming
program exists (with its certificate), and the known repair. `probe.py` receives
only the scaffold and the episodes.

### Recorded (R) — failed scaffolds whose repair is established elsewhere

| id | scaffold | task / stream | contains a solution? | known repair |
|---|---|---|---|---|
| `R1_counting_postaudit` | `language-capability/scaffolds.stage_b`, positions=22 | Dyck grammaticality, post-audit | **no** — 0/45,375, `complete` (§45) | fold template with `min` (§45), `max` its twin (§47) |
| `R2_tight_flat` | `earned-abstraction/later.tight_scaffold(module=None)` | `maj(a,b,c) xor maj(d,e,f)` | **no** — 0/230,400, `complete` (§44) | inherit a `MAJ3` module (§44 arm 3, 144 conforming) |
| `R3_tight_wrong_module` | `later.tight_scaffold(module=distractor)` | same | **no** — 0/2,709,504, `complete` (§44) | replace the module with `MAJ3` |

### Negative controls (S) — scaffolds that DO contain a solution

The probe must not fire on these. A fire is a false positive and makes the probe
unusable as a gate.

| id | scaffold | contains a solution? |
|---|---|---|
| `S1_dyck_min` | `dyck_scaffold.stage_b_dyck` (fold `min`) | **yes** — 110, `complete` (§47) |
| `S2_tight_maj3` | `later.tight_scaffold(module=MAJ3)` | **yes** — 144, `complete` (§44) |
| `S3_counting_preaudit` | the **identical** counting scaffold of `R1`, positions=16, on the **pre-audit** stream | **yes** — §19/§45 control, 0.9986 on n=724 |

`S3` is the sharpest of the three: the same scaffold as `R1`, differing only in
the stream. If the probe fires on `R1` and not on `S3`, it is responding to the
data; if it fires on both, it is responding to the scaffold.

### Constructed (C) — the same template with the fold hole pinned to each core operator

Template: `accum_scaffold.stage_b_accum(fold=F)`, i.e. §45's two-accumulator
scaffold with the fold operator pinned. `F` ranges over **every** operator core
declares at `CNT × CNT → CNT` — `add, idiv, max, min, mod, mul, shl, shr, sub`
(nine, read off `tcn.operators.BINARY` by `registry.resolve`, not curated). Two
tasks:

- **task `bal`** — label = the stream's own `answer` probe, i.e. balancedness.
  Known repairs: `{min, max}`, established independently by §47's exhaustive
  sweep and re-derived here.
- **task `max2`** — label = `max_prefix(s) >= 2`, a **constructed** label over
  the same episodes and the same stream. Known repair: `{max}` — fixed by
  construction (`ge 2` on a running maximum with `plus=+1, minus=-1`), and the
  full nine-fold sweep is run to establish it by enumeration rather than by
  assertion. Its threshold is 2 because `rule_consts = (-2..2)` makes no other
  threshold expressible; label rates are 0.7083 train / 0.7334 held-out unseen,
  so the majority constant is reported beside every accuracy and the task is
  described as imbalanced.

For each task, `F ∈ {min, max}` (or `{max}` for `max2`) are **solvable
controls**; the rest are failed scaffolds with a known repair. This yields
7 + 8 = 15 failed scaffolds and 2 + 1 = 3 further solvable controls, **on two
tasks**. That is the honest denominator: many scaffolds, few tasks.

The point of family C is graded degeneracy. `fold=mul` makes the fold chain
literally constant (`lo_0 = m_0 * 0 = 0`); `fold=add` and `fold=sub` leave it
varying; `fold=idiv` and `fold=mod` make the whole family unusable. So the
family separates "the defect is a constant node" from "the defect is a wrong
operator that still varies", which is precisely §47's untested boundary.

---

## Falsification criteria, stated now and honoured

**F1 — narrow constant-detector.** If Stage D fires only on cases whose site has
`distinct == 1` (or where rule D0 applies), and fails to fire on at least half of
the failed cases whose defect node varies, then the probe is recorded plainly as
a **narrow constant-detector, not a scaffold diagnostic**. That is still a
useful, bounded tool and is written up as one, without gloss.

**F2 — false positives.** If Stage D fires on **any** of the solvable-scaffold
controls (`S1`, `S2`, `S3`, `C:bal:min`, `C:bal:max`, `C:max2:max`), the probe
cannot be used as a gate. The count is reported, not the rate alone.

**F3 — the sweep is doing the work.** Three numbers are reported side by side:
the probe's repair-correct rate, **B-sweeponly**'s, and **B-random**'s. If the
probe's rate equals B-sweeponly's, the conclusion recorded is that the
information-content framing **adds nothing** to repair selection and its only
contribution is site localisation. If B-sweeponly's rate is not better than
B-random's, the sweep itself is recorded as no better than chance. This is the
most likely way §47's result is less than it appears and it is tested directly.

**F4 — anecdote, not capability.** The number of *independent tasks* is reported
next to the number of scaffolds. With two recorded tasks the verdict is written
as an anecdote about a bounded tool, never as a capability.

**F5 — site localisation adds nothing.** If **B-terminal** — "always blame the
output-adjacent node" — selects the same site as Stage D on every case, then the
per-node information table is decoration for these scaffolds and the write-up
says so.

**What would support H.** Stage D fires on failed scaffolds and not on solvable
ones; its site is the defect site more often than B-terminal's; and Stage S's
prediction matches the known repair strictly more often than B-random, on at
least two independent tasks with different correct answers.

---

## Verification obligations

- All non-environmental tests pass. The ~13 worktree failures (`generators/computer`
  and `test_panel_interface.py::test_panel_episode_replays_and_restores`, which
  fail on main's own code in any worktree with a symlinked `node_modules`) are
  verified environmental by removing `research/scaffold-diagnosis/` and re-running.
- The shipped fixture reproduces **0.248836 → 0.002231 at 4/4 frozen**.
- `git diff main...HEAD -- tcn/ generators/` is empty.
- Large artifacts stay out of git.
- Any recorded number that does not reproduce is reported as a finding rather
  than worked around.
- §16 is taken as given: `eq`'s surrogate is exactly 0.0 past |a−b| ≥ 11 and
  `lt`'s past 17, and §47 measured `symbols`' first-step gradient as exactly 0.0.
  No gradient arm is run here, and no claim about search is made where a dead
  surrogate is the available explanation.
