# RESUME — schema induction from the §65 `gap 1` counterexample

Written at a deliberate pause point, 2026-09-11. **The pre-registration is
approved and unchanged. This is a pause, not a redesign — nothing about the
experiment needs re-litigating. The arms simply run from the committed state.**

* branch: `worktree-agent-a7ba0542f4ac459db`
* head at pause: `10ca012`
* pre-registration revision: `r3-2026-09-11-clean-producer-and-structural-c3`
  (`stamp.REVISION`; artifacts carry it and `verify.py` fails on a mismatch)

## 1. State in one paragraph

Phase 0 is complete, committed and verified. **No arm has run. No outer loop
has run. No mutation grammar exists yet.** The `gap 2` and `gap 3` episodes —
the blind final split — are **deliberately unbuilt**; nothing in the track has
constructed, read or named one. Everything that exists is the pre-registration,
the phase-0 validity checks, and the provenance / promise / guard machinery.

## 2. What is built

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | approved, r3. §1 claim block, §1.2 the induction invariant, §2 S0 and the counterexample, §3 the grammar and the scoped widening proof, §4 arms, §5 three splits, §6 criteria (C1–C3 primary, R1–R5 reported), §7 outcomes O0–O6, §8 six costs, §9 validity checks, §10 measured budget, §13 provenance, §14 limitations, then amendments r3 and r2 |
| `phase0.py` | V2 expressivity, V3 scoped-widening insufficiency, V8 cost pilot. Reads **only** the committed `gap 0` / `gap 1` caches |
| `stamp.py` | provenance: the owner's five standard fields, fail-closed on a dirty producer, no back-fill path |
| `promises.json` | 34 promises, each with phase + implementing `verify_*` name |
| `check_promises.py` | fails on an activated-but-unimplemented promise, a deleted promise, an unstamped `out/` artifact |
| `verify.py` | the 9 checks phase 0 activates |
| `selftest_guards.py` | 24 negative tests; every guard driven to its failure state |
| `out/phase0.json` | the phase-0 evidence, produced from clean commit `2012f1e` |
| `out/phase_phase0.done.json` | the phase marker — written by the job, never by hand |
| `out/verify.json` | the verifier's own output |

Gates at pause: `verify.py` **77 PASS / 0 FAIL** · `check_promises.py` **34
promises, 9 kept, 25 pending, 0 failures** · `selftest_guards.py` **24 PASS /
0 FAIL**.

## 3. What remains, in order

Each step's promises activate when its phase marker appears, so **a step is not
done until its `verify_*` functions exist**. `check_promises.py` enforces this.

1. **`grammar.py`** — the six families of §3.1 (`SUBST`, `WIDEN`, `REWIRE`,
   `ADD_NODE`, `COMPOSE`, `PARAMETERIZE`). Must export `GRAMMAR_VERSION` and
   `FAMILIES`; `stamp.grammar_digest()` already looks for both and will start
   recording them automatically the moment the file exists. The schema object
   must serialize to an **AST with named, typed free parameters**, because
   C3a/C3b are checked on that structure, not on outputs.
2. **`edits.py`** — copied from `research/scaffold-induction/edits.py` on branch
   `scaffold-induction` (`Edit` frozen dataclass, `enumerate_edits`,
   `apply_edit`), extended with `PARAMETERIZE` and `COMPOSE`. **Do not edit
   that track's files.** Inherit its A11 rule: truncation bounds stay off
   (`MAX_NEW = MAX_NEW_NODE = None`); an over-budget edit is recorded
   `undecided`, never silently redefined as unrepairable.
3. **V1 + V4**, phase `grammar` → writes `out/v1_copy.json`,
   `out/v4_grammar.json`, then `phase_grammar.done.json`. Implement
   `verify_v1_copy_reproduces_flagship_counts`, `verify_v4_grammar_excludes_answer`,
   `verify_f3_no_contradiction_with_s65` **in the same commit**.
4. **The fast decider + V6**, phase `outer`. V6 must pass before any criterion
   is read; if it fails, **stop** — §10 says the exact-counter fallback is ~44 h
   per arm, which exceeds what this track should spend, and the design goes back
   for re-approval rather than running at reduced coverage.
5. **The arms** (§4): `S0`, `S1`, `S1_prior`, `S_widen`, `S_blind`, `S_hand`,
   `S_flat`, `D_prior`.
6. **`final_cache.py` → `score_final.py`**, phase `final`. This is the **only**
   step that builds or reads `gap 2` / `gap 3`.
7. **`report.py --fill`**, then `verify.py`, then `RESULTS.md`.

## 4. Exact commands

Check the floor first — this host is shared and has been driven to ~2 GB
available by another project twice:

```
grep MemAvailable /proc/meminfo        # need >= 7.5 GB (20 x the 0.374 GB peak)
nvidia-smi                             # unified memory: a GPU job is a RAM job
```

Every heavy job, capped (measured peak 0.374 GB, so `MemoryMax=8G` is ~21x):

```
cd research/schema-induction
systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=200% \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  <repo>/.venv/bin/python <script> [args] > out/<name>.log 2>&1
```

`python` is not on `PATH`; use the venv interpreter explicitly
(`/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python`).

Re-run phase 0 (≈ 6 min with the pilot; ≈ 5 s without):

```
systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=200% \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  <venv>/python phase0.py --pilot --sizes 5 10 20 > out/phase0_pilot.log 2>&1
```

The three gates, in the order a reviewer should read them:

```
<venv>/python selftest_guards.py     # the guards bite
<venv>/python check_promises.py      # every activated promise is implemented
<venv>/python verify.py              # every claim re-derived from raw JSON
```

**The commit dance is mandatory, not stylistic.** `stamp.present` refuses any
artifact produced from a dirty tree, so:

```
git add <code + PREREGISTRATION.md> && git commit     # 1. sources clean
<run the job>                                         # 2. produce artifacts
git add research/schema-induction/out && git commit    # 3. commit outputs
```

`dirty` counts tracked modifications to **sources only** (everything outside
`research/schema-induction/out/`); churn inside `out/` is disclosed in
`dirty_outputs`, not counted. Untracked **sources** are refused outright, so a
new `grammar.py` must be `git add`ed before the run that uses it.

## 5. Unbuilt by design

`gap 2` and `gap 3` episodes. `out/phase0.json` records
`splits.gap2.final: null` and `splits.gap3.final: null` — **present and null is
required**: it records the splits as withheld rather than leaving them silently
unmentioned. V5's wording was corrected in r3 precisely so that the null entries
are permitted (they were forbidden by its r2 wording, which contradicted the
provenance standard). Do not build these until `final_cache.py`, and note that
`cache.build(gap)` in the flagship takes ~12 s per configuration, so there is no
efficiency reason to build them early.

## 6. Decisions a successor would otherwise re-derive

**The geometry, measured (`out/phase0.json`, `V2_expressivity`).** Offsets are
raw byte units; 3 bytes per pixel, `3W = 48` bytes is one raster row.

* `gap 0` `above`: `sub(lo, o)`, o ∈ [7,48] ∪ [55,96] ∪ [103,144]. S0 has 9, 48.
* `gap 1` `above`: `sub(lo, o)`, o ∈ [55,96] ∪ [103,144] ∪ [151,192]. **S0 has
  nothing** — its largest offset is 51. That is the whole counterexample:
  **55 > 51.**
* `left_of` / `right_of` stay expressible at both gaps with small offsets
  (3, 6, 9). Only `above` is broken, and that is a **feature** — it isolates
  the missing-structure axis (owner, approved).
* The band law, derived and **pre-registered as a prediction, not measured**:
  at `gap g`, `above` needs o ∈ ⋃ₖ [48(g+k) − 41, 48(g+k)], k = 1…h, h = the
  smallest target height (3 in both measured configurations). The right endpoint
  of every band is a multiple of 48, so `{48·i}` contains a solution at every
  gap, at i = g+1, while no constant list fixed on admission does.
* Predictions P1–P3: S0 fails `gap 2`/`gap 3`; the flat substrate (offsets
  1…128) reaches `gap 2` but **fails `gap 3`** (151 > 128); **no single constant
  serves `gap 0` and `gap 3`** ([7,144] and [151,288] are disjoint) — which is
  why a patch cannot pass the blind test.

**The scoped widening claim.** `L = {0…15, 48, 51}`, the scaffold's own declared
integer constants. `max(L ∪ S0) = 51 < 55 = min required`. Always state this as
**"widening within S0's declared source cannot"** — never "widening cannot solve
the task". Unrestricted widening *does* reach the answer; it is the `S_blind`
arm, a first-class comparator charged its full total cost and a genuine
falsifier (R1 / outcome O3). `out/phase0.json` carries `scope` and
`does_not_claim` fields and `verify.py` checks both, so the qualifier cannot
drift out of the artifact.

**The cost model, measured five times.** `seconds ≈ intercept + slope × step
candidates` for the exact 48-episode counter, with S0's non-STEP pools fixed.
Verified as a **band** — intercept ∈ [45, 85] s, slope ∈ [0.8, 1.4], largest
residual < 10% of the largest measured time — never as coefficients: five runs
gave visibly different timings. The **exact conformer counts were byte-identical
all five times** (1,376,372,736 / 2,767,343,616 / 11,224,903,680 at 20 / 40 / 80
candidates) and are checked by token equality. Two successive *absolute*
residual bounds (2 s, then 3 s) each failed on a later run while the affine
shape held every time; do not reintroduce one.

The band does **not** extrapolate to `S_flat`, whose ADDR and MATCH pools are
much larger (2,304 matchers vs 256) and feed the fixed term. §65 never exhausted
the flat arm at 48 episodes. `S_flat` is budgeted as *unknown, capped*.

**C3 is structural (r3).** Novelty of the final displacement is a **necessary
condition only** (C3c) — "the value was unseen, therefore it was re-derived from
geometry" can hold by accident. The criterion is C3a (a free geometry parameter
in the AST/dataflow, not fixed during induction) ∧ C3b (structure digest
identical across instantiation at `gap 0…3` while the emitted constant varies as
the band law predicts) ∧ C3c. Design `grammar.py`'s serialization with this in
mind — it is the one place where a late design choice would be expensive.

**No cost thresholds anywhere (r2).** The primary criterion is the existence
claim C1 ∧ C2 ∧ C3. Cost is R1–R5, reported faithfully with no pass mark. Do not
reintroduce a 2× or 10× cutoff; the owner rejected them as unjustifiable before
running.

**The headline is an outcome id, O0–O6 (§7.1), not pass/fail**, and each is
reported on its own merits. O6 (induction) neither suppresses nor is weakened by
O3–O5 (the cost outcomes): induction can happen and still be the wrong
engineering choice.

**How to state a positive result** (owner's framing, to be used verbatim): it
would **not** mean "the answer was restored"; it would mean **a missing
computational schema was induced from a counterexample and then instantiated
correctly outside the induction range**.

**Two contradictions with other records, already reported, do not re-discover
them:** scaffold-induction's A13 claims its `verify.py` enforces the blind
split's fingerprint and `final_untouched` — `git grep -n final verify.py` on
that branch returns nothing, the check was never written; and its three-split
deletion comment names `AttributeError` where a plain dict raises `KeyError`.
Both are handled in this track's own design (§5).

## 7. Dependencies and traps

* **Do not modify `tcn/` or `generators/`.** Sidecar only. `source_fingerprint()`
  hashes both, so a change there invalidates every recorded episode in every
  track.
* The track reads `research/integrated-flagship/out/episodes_gap{0,1}.json`
  **read-only**, and will copy `env.py`, `family.py`, `engine.py`, `cache.py`
  into itself (V1 asserts the copy reproduces §65's counts: 686,985,984 at
  `gap 0`, 0 at `gap 1`, 261,654,545,280 training conformers at `gap 1`).
* One heavy job at a time; at most 4 workers; re-read `MemAvailable` before each
  phase. A background waiter of this session was OOM-killed during phase 0 while
  three other projects held ~8 GB.
* `test_panel_interface.py::test_panel_episode_replays_and_restores` fails in
  any worktree with symlinked `node_modules`, on main's code. Not this branch's
  fault.

## 8. Estimated cost to complete

From §10's measured band, single worker, capped:

| step | estimate |
|---|---|
| `grammar.py` + `edits.py` + V1/V4 | build time, then ~15 min compute |
| fast decider + V6 (40 schemas, exact-counter cross-check) | ~1.8 h |
| six cheap arms (`S0`, `S1`, `S1_prior`, `S_widen`, `S_hand`, `D_prior`) | ~1.7 h total |
| `S_blind` (3,068 step candidates) | ~4 h, capped 2 h per configuration |
| `S_flat` | unknown, capped 2 h per configuration |
| `final_cache.py`, `score_final.py`, `report.py`, `verify.py` | ~1.5 h |

**Headline path ≈ 5 h** of capped compute (everything except `S_blind` and
`S_flat`, which are bounded controls reported with two-sided bounds if they do
not exhaust — F7: a sampled bound is never quoted as an exact count).
