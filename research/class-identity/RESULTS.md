# Class identity beside artifact identity: the design, the measurement, and what is not justified yet

`PREREGISTRATION.md` was committed at `9224438` **before any arm was run**. Every
number below comes from a file in `out/`; `check.py` re-resolves every
pre-registered criterion against those files and writes `out/check.json`, so no
claim here is separable from the raw data.

**`tcn/` receives no diff in this track.** The class store is a sidecar
(`classes.py`, 240 lines) beside a `tcn.library.Library` root. Whether any of it
belongs in `tcn/library.py` is the verdict below, not an assumption.

---

## VERDICT

**Q1 — a stored class does produce a measurable saving on a held-out width, and
it is large.** At depths 5, 7 and 12 — widths 15, 21 and 36, **untouched by any
prior run in this repository** — enumeration costs **272 programs / 17 408
environment episodes / 69 632 steps** and returns certificate `unique`. The same
answer, instantiated from a class certified only at widths 3 and 6, costs
**1 program / 64 episodes / 256 steps**: a **272× saving in episodes** and a
**267–269× saving in wall clock**, and the artifact the class instantiates is
**digest-identical** to the one the enumeration certifies at all three widths.

**The format rejects arm F's wrong schema, twice, for two different reasons.**
R1 (two-width rule) refuses a single-width certification whatever its
certificate; R2 (vector agreement) refuses a two-width certification whose stored
vector scores 1.75 at the second width. A deliberately unsound single-width
format publishes the wrong schema and then hands a consumer **1.625, 1.9375 and
2.125** at the three held-out widths — at or below the best constant (2.0,
2.1875, 2.125) every time — while the correct class returns 4.00/4.

**Q2 — a stored class also produces a measurable saving, in §52's own record, and
it is not the saving §52 was looking for.** Class identity finds that two of
§52's L1 arms — `arm2_syntactic` (`ec516b38…`) and the pooled wrong-module
control `arm4s_runnerup` (`ca785ec9…`) — are **two circuits of one semantic
class**, `tt/3/57`, identical in arity, types, truth table, node count, execution
cost and description bits. §52 exhausted **2 709 504 programs twice** and
recorded identical results in every column. A class-aware library would have
detected the duplicate before the second run.

**Is a core change to `tcn/library.py` justified? Not yet, and one specific part
of DESIGN §7 is refuted.** The measured saving is entirely delivered by a
240-line sidecar that touches nothing. But the sidecar also exposes a hard
format fact: **`Entry.from_dict` reads named keys, so an extra `semantic_id` in
the manifest survives loading and verification untouched and is then silently
erased by the first `_save()`.** Measured, both halves. So "additive field on
`Entry`" is *not* free — it is a real, if small, core change — and nothing
measured here needs it. **And DESIGN §7's "the class holds the schema" cannot be
implemented at all**: `tcn.library` stores `Program.to_dict()`, and a schema is a
function from width to `Program`, which has no serializable form anywhere in
`tcn/`. A class record is a **witness**, not a self-contained artifact.

**Bounding the claim, as pre-registered (F-b).** The 272× saving *equals the
space size*. It is a property of the schema's two free nodes (17 × 16), not of
the class store, and it would not survive being reported as a capability at
scale. What the store demonstrates is a **mechanism** — that the invariant 8.09
bits are enough to skip the search — not that the search it skips is hard.

---

## 1. The design, evaluated against `tcn/library.py` as it actually is

`research/algorithm-resynthesis/DESIGN.md` §7 proposes four things. Read against
the code rather than against the intent, they do not all fare the same.

| §7 clause | verdict | why, against the code |
|---|---|---|
| keep `digest` as **artifact** identity | **holds, and costs nothing** | `by_digest`, `_module_path`, `_fixture_path`, the `program.digest != entry.digest` check in `_load_module` and the `operator != "module:" + digest` check in `publish` all key on the digest. A class touches none of them. Every arm here left `strict`, `revalidate`, fixture replay and `source_fingerprint` untouched and measured that they were untouched. |
| add `semantic_id` as **class** identity | **holds, with one amendment: it must be scheme-tagged** | §52 needs an *extensional* key (`(arity, truth table)`), computable only over a small finite domain — `poolmine._table` already guards `len(canon.inputs) <= 4`. §53 needs an *intensional* key (schema + selection vector); the truth table of a depth-8 interpreter over `tuple[24]` is not enumerable. Neither relation subsumes the other, so one opaque string would make unequal classes compare equal. `classes.py` tags the scheme (`tt/3/17`, `schema/f72db2cf6eca/8958463f0dee`) and `same_class` refuses cross-scheme comparison. |
| an entry becomes **a class with several artifacts and a preferred member per cost axis** | **the grouping holds; "per cost axis" is real but negligible on this evidence** | Making `Entry` hold several digests is not additive: it changes `by_digest`, `dependents`, `_load_module`, `publish`'s "same digest → no-op", and `load`'s alias map, which binds *one* `module:<digest>` per alias. The additive form that gets the same behaviour is to keep `Entry` 1:1 with an artifact and make a class a **grouping over entries**. Measured: see §4. |
| **the class holds the schema, artifacts hold instantiations** | **cannot be implemented in `tcn.library`** | `Library` stores `Program.to_dict()`. A §53 schema is `research/depth-encoding/scaffold.py:interpreter_scaffold` — a *function from width to `Program`*. Nothing in `tcn/` has a representation of a width-parametric program. What a record can hold declaratively is a schema **reference** (module, qualified name, free-node shape, `source_fingerprint`), the selection vector, the certified widths and the per-width digests. Reuse then requires the schema code to be present and unchanged — which `source_fingerprint` already governs exactly. |

**A fact the design should have had and did not.** `mine_semantic.propose`
already does what §7 asks for, and then throws it away:
`research/premin-abstraction/mine_semantic.py:96` elects
`min(reps, key=(len(nodes), digest))` over a pool of same-class circuits and
publishes that one. For §52's rank-1 majority class the pool holds **8 distinct
circuits**; `research/semantic-library/library` stores **1**. The class identity
exists upstream and is discarded at the library boundary.

### 1.1 The one measured surprise about the on-disk format

`Entry.from_dict` reads named keys, so unknown manifest fields are **tolerated on
read and dropped on write**. Both halves measured (arm B4, `out/q2_compat.json`):

* A copy of §52's library with `semantic_id` and `class_members` stamped onto all
  22 entries behaves **identically to an unstamped control** — same 22 entries,
  same names, same `verify()` row for row, `verify_all_ok: true` on both, **42
  successful loads** (21 names × `strict` and `revalidate`), **0 errors** on both.
* One `Library._save()` and `extra_keys_survive_save: false`.

So the manifest is *forward*-compatible with a class field and cannot *carry* one
without a core change. That is the whole argument for the sidecar.

---

## 2. The format under test

`classes.py`. `class_id` is scheme-tagged; a `schema` record holds a schema
reference, a selection vector, the certified widths and per-width digests; an
`extensional` record holds members and a preferred member per cost axis.
Admission rules, all enforced in `ClassStore.admit`:

* **R1, two-width rule.** A schema class certified at fewer than **2 distinct
  widths** is refused, whatever its certificate. This is §53 arm F made
  mechanical.
* **R2, vector agreement.** The **same** vector must conform at **every**
  certified width, against that width's recorded threshold.
* **R3, instantiation check.** Re-instantiating at a certified width must
  reproduce the recorded digest.
* **R4, reuse is verified, never assumed.** Instantiating at a new width yields a
  *conformance check* at that width and **never a uniqueness certificate**. Its
  cost is charged to the with-class arm.

---

## 3. Q1 — the held-out widths

Schema, scaffold, generator config, horizon, threshold and episode indices are
**§53's, unchanged** (`research/depth-encoding/scaffold.py`, `logic`,
`{inputs: 4, nondegenerate: true, min_relevant_inputs: 2}`, horizon 4, threshold
4.0, fit `range(16)`/`train`, held-out `range(10000, 10064)`/`test`). Both arms
call the **same** `tcn.search.program_return` against the **same**
`EnvironmentTask`; one calls it 272 times and the other once, so the comparison
is not an artefact of two harnesses. `enumerate_environment` — the shipped
enumerator — was run on the same space at depth 5 and **agrees on all seven
recorded fields** (`out/crosscheck.json`), so nothing here is a private
reimplementation of a verdict.

**The class was built from widths 3 and 6 only** (depths 1 and 2, 272 programs
each on the *fit* episodes, both `unique`, both giving
`{relation: 16, goal_relation: 6}`, `vectors_agree: true`), admitted under R1–R3
as `schema/f72db2cf6eca/8958463f0dee`. Depths 5, 7 and 12 were **not enumerated
during construction**.

| depth | width | arm | programs | episodes | steps | wall s | certificate | result |
|---|---|---|---|---|---|---|---|---|
| 5 | 15 | without class | 272 | 17 408 | 69 632 | 58.1 | `unique` | best 4.00 |
| 5 | 15 | **with class** | **1** | **64** | **256** | **0.22** | conformance | **4.00** |
| 7 | 21 | without class | 272 | 17 408 | 69 632 | 79.5 | `unique` | best 4.00 |
| 7 | 21 | **with class** | **1** | **64** | **256** | **0.30** | conformance | **4.00** |
| 12 | 36 | without class | 272 | 17 408 | 69 632 | 146.9 | `unique` | best 4.00 |
| 12 | 36 | **with class** | **1** | **64** | **256** | **0.55** | conformance | **4.00** |

**Saving: 272× in episodes and steps at every width; 269×, 269×, 267× in wall
clock.** All 272 programs were usable at every width (`unusable: 0`), so the
space is genuinely 272 wide.

**The two arms return the same program, not merely the same score.** The digest
the enumeration certifies and the digest the class instantiates agree at every
width: `f6f17d4c3e9a…` (d 5, 150 nodes), `117f4c84e949…` (d 7, 204),
`fca19ad75855…` (d 12, 339). Criterion C1.

**Baselines beside the return, as the standing rule requires.**

| depth | class result | best constant | always-true | always-false | uniform random | whole-space mean of 272 | second-best of 272 |
|---|---|---|---|---|---|---|---|
| 5 | **4.0000** | 2.0000 | 2.0000 | 2.0000 | 1.9375 | 2.0000 (sd 0.4518) | 3.3125 |
| 7 | **4.0000** | 2.1875 | 2.1875 | 1.8125 | 1.9062 | 2.0000 (sd 0.2827) | 3.1250 |
| 12 | **4.0000** | 2.1250 | 2.1250 | 1.8750 | 2.0000 | 2.0000 (sd 0.3358) | 3.1875 |

**Achieved difficulty, never the requested config** (`out/difficulty.json`, 64
held-out episodes each): 64/64 distinct circuits at every depth; ten distinct
final truth tables `{1,2,4,6,7,8,9,11,13,14}` at every depth; minimum relevant
inputs **2** achieved, as requested; relevant-input histograms
`{2:36, 3:24, 4:4}` (d 5), `{2:32, 3:24, 4:8}` (d 7), `{2:31, 3:25, 4:8}` (d 12);
majority fraction 0.500, 0.547, 0.531.

**Secondary confirmation at §53's own widths** (`out/confirm.json`): depths 3, 4,
6, 8 each enumerate to `unique` at 4.00 with the same vector, and the class
instantiates `a6a183ea4f6c…`, `98f038905351…`, `ddbedead8876…`, `2ffc8dad769d…`
— **the four digests §53 recorded in its `size.json`**, reproduced here through
the class store rather than through §53's script.

### 3.1 What the saving actually costs, and what it does not buy

**The class never yields a uniqueness certificate.** The without-class arm ends
with `unique` — 271 programs proved not to conform. The with-class arm ends with
a conformance check on one program. That asymmetry is R4, it is deliberate, and
it is the honest price of the 272×.

**Reuse without verification is not on offer.** Charging the 64 episodes is what
makes arm F's failure visible; a store that skipped them would be exactly the
unsound format of §5 below.

**Size** (`out/size.json`). The class record is **1 082 bytes** of JSON (1 692 on
disk) and covers all five widths. The five instantiated artifacts serialize to
**572 840 bytes** in total and carry 169 264 → 2 323 632 description bits
(execution cost 42 → 339 nodes), with **5 distinct digests**. §53's figure
reproduces exactly: the invariant content is **8.09 bits** —
`log2(17) + log2(16)`, two entries.

---

## 4. Q2 — §52's pooled abstraction as a class

§52's library was **copied**; the original at
`research/semantic-library/library` was never written to.

### 4.1 The partition of the library as it actually stands

`out/q2_partition.json`: **22 manifest entries → 7 distinct digests → 6 distinct
classes.** `165bc290…` alone is published under **7 logical names**
(`sem_minall`, `sem_trace`, `sem_wo_t1_win` … `sem_wo_t5_win`), which
content-addressing already deduplicates as *files* but not as *entries*, and
which no query can reach by asking for the abstraction rather than the name.

| digest | class | arity | nodes | exec | description bits | published under |
|---|---|---|---|---|---|---|
| `165bc290…` | `tt/3/17` | 3 | 4 | 4.0 | 10 128 | 7 names |
| `ca785ec9…` | **`tt/3/57`** | 3 | 2 | 2.0 | 6 048 | 1 name |
| `ec516b38…` | **`tt/3/57`** | 3 | 2 | 2.0 | 6 048 | 5 names |
| `08735e50…` | `tt/4/577f` | 4 | 5 | 5.0 | 12 360 | 2 names |
| `0ba4287a…` | `tt/4/566a` | 4 | 5 | 5.0 | 12 384 | 3 names |
| `82b93e4b…` | `tt/3/14` | 3 | 2 | 2.0 | 6 056 | 3 names |
| `0b7ea9a5…` | `tt/3/29` | 3 | 4 | 4.0 | 10 128 | 1 name |

### 4.2 The one thing class identity finds that digest identity cannot

**`ec516b38…` and `ca785ec9…` are the same semantic class.** Truth table
`01010111`, arity 3, 2 nodes, execution cost 2.0, description bits 6 048 —
identical in every property a caller or an enumerator can observe. They are
different circuits, so their digests differ, and §52 ran them as two arms:

* `arm2_syntactic` — `mine_multi`'s rank-1 under digest identity;
* `arm4s_runnerup` — `mine_semantic`'s pre-registered wrong-module control, "the
  rank-1 arity-3 pooled class that does not compute the family's window
  function".

§52's L1 table records them as **identical in every column**: 0 conforming of
2 709 504 exhausted, `complete`, 0/24 tight, median accuracy 0.7812, 0/8 wide,
wide median 0.7812. Every other arm differs somewhere. Two full exhaustive
enumerations — **2 709 504 programs, 186.5 s and 178.2 s of §52's recorded wall
clock** — for one class.

**This does not disturb §52's verdict, and it should not be read as doing so.**
The other wrong-module controls are genuinely different classes: `arm4b_wrong_mined`
is `tt/3/14` and `arm4s_offfamily` is `tt/3/29`, and their recorded numbers
differ accordingly (median accuracy 0.6250, not 0.7812). §52's control structure
holds. What the class store adds is that **one of its arms was a duplicate of
another under a different name**, which was invisible to a library that indexes
by digest and would have been visible at selection time to one that indexes by
class.

### 4.3 Several artifacts, and a preferred member per cost axis

Two measurements, and they point the same way: the clause is **real but small**.

**The pool §52 discarded** (`out/q2_members.json`). Recovering the rank-1
majority class from the same corpus recovers **8 members**, matching the
`circuits_pooled: 8` §52 recorded. All 8 have **4 nodes** and **execution cost
4.0**; their description bits are **{10 120, 10 128, 10 136}**. The miner elects
by `(nodes, digest)` — a tie on nodes across all 8, broken by digest — and
publishes `165bc290…` at **10 128** bits while the class contains a member at
**10 120**. So the elected representative is not preferred on the description
axis, and the spread is **16 bits in ~10 128, 0.16 %**.

**Inheritance across two independently-obtained artifacts** (`out/q2_inherit.json`).
Publishing the hand-authored `MAJ3` (§52's `arm3_authored` ceiling) into the copy
beside the mined circuit gives two entries whose **digests separate**
(`165bc290…` vs `8ceedf7b…`) and whose **class unifies** (`tt/3/17` both), with
different bodies — `and,xor,and,or` against `and,or,or,and`. §52 measured both at
**144 conforming, 18/24 tight, 8/8 wide, median accuracy 1.0000**: the class's
members are interchangeable in measured capability. Their costs are 4 nodes and
4.0 execution cost each, and **10 128 against 10 056 description bits**, so
`preferred` resolves to `165bc290…` on nodes and execution cost (by digest
tie-break) and to `8ceedf7b…` on description bits. `cost_axes_agree_on_preferred:
false` — the clause is exercised, by **72 bits, 0.7 %**.

`Library.verify()` returns `ok` on every entry after the publish
(`library_verify_all_ok: true`).

### 4.4 What Q2 does *not* show

**Class identity does not fix §52's ranking failure and no claim is made that it
does.** §52's F2 fires at the *ranking* layer — `mine.propose` sums saving over
corpus entries, so removing one task penalises broad fragments and leaves narrow
ones untouched, and rank 1 flips by 0.85 % in two of five leave-one-out corpora.
That is upstream of storage. A class-aware library publishes the same rank-1
class the same rule proposes. **Ranking is another track's subject and is not
touched here.**

---

## 5. The soundness test — §53's arm F, run against the format

The wrong schema is §53's `first_gate_scaffold`: identical to the right one but
for one edge, `relation`'s lookup reading `gate_0` rather than `gate_{d-1}`. At
depth 1 those are the same node.

**F1 — one certified width.** The depth-1 fit gives exactly what §53 said it
would: 272 evaluated, exhausted, **1 conforming, certificate `unique`, best
4.00**, vector `{relation: 16, goal_relation: 6}` — *indistinguishable from the
correct schema's fit*. The format refuses:

> `R1 two-width rule: schema/f72db2cf6eca/d24b464be2ed is certified at 1 width(s) [1]; a schema class needs at least 2 distinct widths. A `unique` certificate at one width is not evidence of a correct schema (FINDINGS section 53, arm F)`

**F2 — two certified widths.** At width 6 (depth 2) the stored vector scores
**1.75**, and the wrong schema's whole 272-program space exhausts at **0
conforming, `complete`, best 3.00**. The format refuses again, on a different
rule:

> `R2 vector agreement: … at width 2 has conforming=0 mean_return=1.75 against threshold 4.0; the stored selection vector must conform at every certified width`

**F3 — the unsound format, on purpose.** Bypassing R1 and publishing the
single-width record, a later consumer instantiating it at the held-out widths
receives:

| depth | width | class result | best constant | verdict |
|---|---|---|---|---|
| 5 | 15 | **1.6250** | 2.0000 | below the constant |
| 7 | 21 | **1.9375** | 2.1875 | below the constant |
| 12 | 36 | **2.1250** | 2.1250 | **equal to** the constant |

Never above 4.0's threshold, and never better than answering the same way every
time. Reported as measured: at depth 12 it *ties* the best constant rather than
falling below it.

**F4 — the wrong schema's whole space, so a collapse is not confused with "no
solution exists at this width".** 272 programs exhausted at each held-out width:
**0 conforming, `complete`**, best 2.6250 / 2.3125 / 2.5625, whole-space mean
2.0000 at all three. The *correct* schema reaches 4.00 in the same 272-program
space at the same widths, so the collapse belongs to the schema.

**The format is sound against exactly the attack §53 identified**, and it is
sound for two independent reasons rather than one.

---

## 6. Every pre-registered criterion, resolved

`out/check.json`.

| criterion | verdict | evidence |
|---|---|---|
| **C1** A2 and A3 return the same program, ≥ 4.0 | **PASS** | digests agree at d 5, 7, 12; 4.00 each |
| **C2** A3's episode cost < 1/100 of A2's | **PASS** | 64 / 17 408 = 1/272 at every width |
| **C3** beats constant, random and whole-space mean | **PASS** | 4.00 against ≤ 2.1875 |
| **C4** F1 and F2 both refused, naming the rule | **PASS** | R1, then R2 |
| **C5** the unsound format is wrong everywhere; F4 shows 0 conforming | **PASS** | 1.625 / 1.9375 / 2.125; 0 conforming, `complete` |
| **C6** the extra field disturbs nothing | **PASS** | identical to control on all six probes, 0 errors |
| **C7** no new test failure; shipped fixture reproduces | **PASS** | §8 |
| *(protocol)* shipped enumerator agrees on the same space | **PASS** | `out/crosscheck.json`, all seven fields |
| *(protocol)* class admitted at 2 widths, vectors agree | **PASS** | `out/construct.json` |

**Falsification criteria, and which fired.**

* **F-a** (no saving → premature) — **does not fire.** 272× in episodes, same
  program.
* **F-b** (the saving is only enumeration being cheap) — **fires, and is
  honoured.** The ratio *equals* the space size, 272 = 17 × 16, which is a
  property of the schema's two free nodes. On a search space where enumeration
  is not affordable, the class store would not be *more* valuable in this
  measurement's terms — the measurement simply could not have been run at all,
  because the without-class arm would not terminate. **This bounds the claim to
  a demonstrated mechanism, not a demonstrated capability at scale.**
* **F-c** (a type check relaxed) — **does not fire.** No type check was relaxed,
  no width erased; the schema is rebuilt exactly typed at each width and
  `Program.validate` runs unchanged. §53's finding that polymorphism is reachable
  without weakening typing carries through the store intact.
* **F-d** (the format admits a wrong schema) — **does not fire.** R1 and R2 both
  refuse.
* **F-e** (Q2 shows only that storage is possible) — **fires partially.** Q2
  produces one real saving (§4.2, a duplicated 2 709 504-program enumeration) and
  no improvement whatever on §52's held-out transfer, which fails at ranking.
* **F-f** (preferred-per-axis is degenerate) — **fires in the weak form.** The
  axes genuinely disagree, twice, but by 0.16 % within the pool and 0.7 %
  between the two independently-obtained members, and node count and execution
  cost are *constant* across all 8 pool members. The clause is exercised, not
  vindicated.

---

## 7. What is and is not proposed

**Proposed: nothing in `tcn/`, yet.** The measured saving is delivered by a
sidecar that changes no core file, breaks no policy and needs no new field on
`Entry`. The house rule is to build the smallest thing a measurement requires;
the measurement required a sidecar.

**If a core change is later wanted, this is the smallest shape it can take,** and
it is stated so a future track does not have to rediscover it:

1. `Entry` gains `semantic_id: str = ""` with `d.get("semantic_id", "")` in
   `from_dict` and the key in `to_dict`. Backward compatible (old manifests read
   as `""`), forward compatible (measured: old code ignores it), and **necessary
   only because `_save()` erases what it does not know**.
2. A class is a **grouping over entries sharing a `semantic_id`**, not an entry
   holding several digests. `Library` gains one query
   (`entries_with(semantic_id)`) and computes `preferred` on demand
   (`min(..., key=axis)`). Nothing in `by_digest`, `dependents`, `_load_module`,
   `publish` or `load` changes, so `strict`, `revalidate`, fixture replay and
   `source_fingerprint` are untouched.
3. `semantic_id` **must** carry its scheme. `tt/…` and `schema/…` are different
   equivalence relations and must never compare equal.
4. A schema class **must** record its certified widths, and admission **must**
   require two or more. This is the only part of the design that is
   load-bearing for soundness rather than for convenience.

**Not proposed, and refuted:** DESIGN §7's "the class holds the schema".
`tcn.library` stores programs, and a schema is code. A class record can hold a
schema *reference* and is a witness whose validity is bounded by
`source_fingerprint` — which is the guard that already exists and is exactly the
right one.

---

## 8. Costs, environment and reproduction

**Environmental baseline, recorded before any change on this worktree, on main's
own code: 324 passed, 13 failed.** The 13 are `test_panel_interface` and the
`generators/computer` contract tests, which need the gitignored `node_modules`
symlink; it is absent in this worktree and in the main checkout. **After this
track: 324 passed, 13 failed — the same 13.** `tcn/` has no diff, so this is
expected and was verified rather than assumed.

`.venv/bin/python -m tcn train --episodes 160` reproduces the shipped fixture
exactly: `initial_prediction_loss 0.248835613951087`,
`final_prediction_loss 0.0022308224288281053`, `evaluation_mean_return 4.0`,
`frozen_evaluation_mean_return 4.0` — **0.248836 → 0.002231 at 4/4 frozen**.

**Compute spent.** Q1: 3 held-out enumerations (284.5 s), 4 confirmation
enumerations (226.5 s), 2 construction enumerations (12.9 s), 7 class
instantiations (1.93 s total). Arm F: 5 enumerations (290.1 s). Q2: one fragment
re-mining of §52's `C-minall` corpus, plus library copies. Wall seconds are
indicative — this machine was shared with other agents' jobs throughout — and
every primary metric is the deterministic episode and step count instead.

**Large artifacts stay out of git.** `out/library_q2/`, `out/library_compat/` and
`out/library_control/` are working copies of §52's library and are gitignored;
the originals are at `research/semantic-library/library`. What is committed is
the class store (`out/library/classes.json`, 1 692 bytes) and the JSON results.

**Reproduce.**

```bash
.venv/bin/python research/class-identity/run_q1.py construct without with \
                                                   baselines difficulty crosscheck armf confirm
.venv/bin/python research/class-identity/run_q2.py partition members inherit compat
.venv/bin/python research/class-identity/size.py
.venv/bin/python research/class-identity/check.py     # re-resolves every criterion
```

---

## 9. Preserved negatives and disclosures

* **The class buys no certificate.** 272× fewer episodes and a strictly weaker
  guarantee. Stated in the pre-registration as R4 and not softened here.
* **The 272× is the space size.** F-b fires; the claim is a mechanism.
* **Q2's saving is retrospective.** It is a duplicated enumeration in a
  *completed* experiment, found by re-reading §52's artifacts through a class
  key. No future §52-shaped run has yet been made cheaper by it.
* **"Preferred member per cost axis" is exercised at 0.16 % and 0.7 %.** On this
  evidence it is not worth a core field on its own.
* **At depth 12 the unsound format ties the best constant rather than falling
  below it** (2.125 vs 2.125). C5 as pre-registered asks only that it fail the
  4.0 threshold, which it does at every width; the tie is reported because it is
  what the data says.
* **`arm4s_runnerup` being a duplicate does not weaken §52.** Its other two
  wrong-module controls are genuinely distinct classes and reach 0 as well; the
  strong form of §52's control — one of thirteen eligible arity-3 pooled classes
  yields any conforming program — is a sweep over *classes* and is unaffected.
* **This track re-mined §52's corpus to recover the discarded pool.** The
  recovered count (8) matches §52's recorded `circuits_pooled` (8) and the elected
  digest matches what §52 published, so the recovery is a replication and not a
  new rule.
