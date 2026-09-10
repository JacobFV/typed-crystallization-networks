# Semantic pooling: the pre-registered replication, its wrong-module control, and the held-out test

`PREREGISTRATION.md` was committed at `30904e2` **before any arm was mined,
published or enumerated**. Every number below comes from a file in `out/`;
`tables.py` regenerates the tables from those files.

---

## VERDICT

**Semantic pooling is a capability on the task it was mined for, and it is not
§50 repeating.** §46's exploratory result replicates exactly — same module
digest, same 144 conforming, same 18/24 and 8/8 — and the control §46 lacked
**does not** explain it. Two independent wrong-module controls, both
semantically pooled by the identical rule, both reach **0 conforming in the same
exhausted 2 709 504-program space**. A sweep of *every* eligible arity-3 pooled
class finds **exactly one of thirteen** that yields a single conforming program,
and it is the rank-1 class. Every pre-registered falsification criterion for the
first claim fails to fire.

**The second claim — that the induced library helps a task the rule did not mine
from — fails, and it fails in a specific and measurable way.** On five
leave-one-out corpora the pooled rule still *contains* the majority class, at
rank 2 or 3, and the class it contains is bit-identical to the one it publishes
on the full corpus (`165bc290d9c82b70a8ea3cc2` in all five). But its **MDL rank-1
proposal is no longer that class**, and the module it actually publishes solves
only **2 of 5** held-out tasks against the hand-authored ceiling's 5 of 5. The
margin that decides this is **0.85 %** in two of the five corpora.

So: **pooling makes the reusable abstraction visible; the MDL score then fails to
select it as soon as one task of six is removed.** That is criterion F2 firing —
a compression result at the ranking layer sitting on top of a real selection
result at the identity layer — and it is reported as such.

---

## 1. What was fixed in advance, and what reproduced

Nine arms, the held-out protocol, five falsification criteria and the arity-3
budget constraint were all fixed in `PREREGISTRATION.md` before anything ran.
Two things reproduced exactly and are worth stating first, because they are what
makes the rest a measurement rather than a re-derivation:

* **§46's published module digest reproduces.** The pooled rule's rank-1
  proposal on `C-minall` **and** on `C-trace` publishes
  `module:165bc290d9c82b70a8ea3cc2` — §46's digest, character for character —
  and it is **not** the hand-authored `module:8ceedf7b792a7116514f90ab`. Its
  body is `and(x0,x1); xor(x0,x1); and(x2,n1); or(n0,n2)`; the hand-authored one
  is a different circuit with the same truth table. It remains *discovery, not
  rediscovery*.
* **§44's numbers reproduce.** `arm1_none` 0 of 230 400; the syntactic rank-1
  is `module:ec516b3808ddef7e7d0e7d22`, §44's digest, 0 of 2 709 504; the
  hand-authored `MAJ3` 144 of 2 709 504. All exhausted, certificate `complete`.

`C-trace`'s ranked table also reproduces §46 §10 line for line: majority rank 1
at **+860 960** with **58 circuits pooled**, §44's `M` at rank 4 with +323 272.

---

## 2. The corpora

| corpus | family | entries | tasks | build | verified in `tcn` |
|---|---|---|---|---|---|
| `C-minall` | majority | 130 | 6 | §46's, reused (11 536 818 DFS nodes) | yes |
| `C-trace` | majority | 296 | 6 | §46's, reused (844 111 950 DFS nodes) | yes |
| `C-minall` | off-family `F′` | 117 | 6 | built here, 13 843 086 DFS nodes, 75 s | yes, all 117 |
| leave-one-out ×5 | majority | 103–109 | 5 | subsets of `C-minall` | yes |

`F′` is `corpus.py`'s six task shapes with `maj` replaced by `D134`, the §44
distractor. Its certified minimum flat length is **5 gates for all six tasks**
(the majority family is 5 for `t1`–`t5` and 3 for `t6`), so `F′` is at least as
hard, not a softer family.

**One corpus was abandoned and it is reported here rather than in a footnote.**
`F′`'s `C-plus1` band was started and killed. Exhausting length 6 over four
inputs costs ~166.5 M DFS nodes per task (§46's measured figure) and **no
declared arm reads it** — `C-minall` is this track's declared primary corpus and
the only `F′` corpus any arm uses. Roughly 18 minutes of compute were spent
before the abandonment.

---

## 3. What each corpus proposed

| corpus | identity | entries | eligible | rank-1 is the family's own window? | best window rank | rank-1 arity/nodes | rank-1 digest |
|---|---|---|---|---|---|---|---|
| `maj` `C-minall` | digest | 130 | 94 | no | 11 | 3/2 | `ec516b3808dd` |
| **`maj` `C-minall`** | **(arity, tt)** | 130 | **40** | **YES** | **1** | 3/4 | `165bc290d9c8` |
| `maj` `C-trace` | (arity, tt) | 296 | 112 | **YES** | **1** | 3/4 | `165bc290d9c8` |
| `off` `C-minall` | digest | 117 | 104 | no | 29 | 4/3 | `d08809475b9e` |
| **`off` `C-minall`** | **(arity, tt)** | 117 | **41** | **YES** | **1** | 3/4 | `0b7ea9a5a2d4` |
| `maj` LOO ∖ `t1` | (arity, tt) | 103 | 28 | **no** | **2** | 4/5 | `08735e504400` |
| `maj` LOO ∖ `t2` | (arity, tt) | 109 | 40 | **no** | **3** | 4/5 | `0ba4287a0409` |
| `maj` LOO ∖ `t3` | (arity, tt) | 103 | 31 | **no** | **2** | 4/5 | `0ba4287a0409` |
| `maj` LOO ∖ `t4` | (arity, tt) | 103 | 28 | **no** | **2** | 4/5 | `08735e504400` |
| `maj` LOO ∖ `t5` | (arity, tt) | 103 | 31 | **no** | **2** | 4/5 | `0ba4287a0409` |

Two things to read off this table.

**Pooling replicates across two independent families.** Under digest identity
the shared window is rank 11 in the majority family and rank 29 in `F′`; under
`(arity, truth table)` it is **rank 1 in both**. The mechanism §46 identified —
one function realised as many circuits competes at a fraction of its own
frequency — is not specific to majority.

**Pooling collapses the eligible set by more than half** — 94 → 40 on the
majority corpus, 104 → 41 on `F′` — which is the same fact seen from the other
side: pooling is merging fragments, not inventing them.

---

## 4. The arms on L1 — `maj(a,b,c) xor maj(d,e,f)`

Tight scaffold, `enumerate_fit` full sweep, `max_programs = 2^27` so every run
exhausts. Gradient: `later.gradient_search` unchanged, 24 seeds tight, 8 wide.
Constant baseline 0.5000, uniform-random baseline 0.5000, both exact.

| arm | module | space | evaluated | exhausted | certificate | conforming | tight | median steps | median acc | wide | wide median acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `arm1_none` | — | 230 400 | 230 400 | true | `complete` | **0** | 0/24 | — | 0.6250 | 0/8 | 0.6875 |
| `arm2_syntactic` | `ec516b38` | 2 709 504 | 2 709 504 | true | `complete` | **0** | 0/24 | — | 0.7812 | 0/8 | 0.7812 |
| **`arm2s_semantic`** | `165bc290` | 2 709 504 | 2 709 504 | true | `complete` | **144** | **18/24** | **50** | **1.0000** | **8/8** | **1.0000** |
| `arm2s_trace` | `165bc290` | 2 709 504 | 2 709 504 | true | `complete` | **144** | 18/24 | 50 | 1.0000 | 8/8 | 1.0000 |
| `arm3_authored` | `8ceedf7b` | 2 709 504 | 2 709 504 | true | `complete` | **144** | 18/24 | 50 | 1.0000 | 8/8 | 1.0000 |
| `arm4_wrong_authored` | `dc59f82e` | 2 709 504 | 2 709 504 | true | `complete` | 0 | 0/24 | — | 0.6250 | 0/8 | 0.6875 |
| `arm4b_wrong_mined` | `82b93e4b` | 2 709 504 | 2 709 504 | true | `complete` | 0 | 0/24 | — | 0.6250 | 0/8 | 0.6250 |
| **`arm4s_runnerup`** | `ca785ec9` | 2 709 504 | 2 709 504 | true | `complete` | **0** | 0/24 | — | 0.7812 | 0/8 | 0.7812 |
| **`arm4s_offfamily`** | `0b7ea9a5` | 2 709 504 | 2 709 504 | true | `complete` | **0** | 0/24 | — | 0.6250 | 0/8 | 0.6875 |
| `arm4s_matched` | — | — | — | — | — | **no candidate exists** | — | — | — | — | — |

`module_on_output_path` is true for exactly the three arms that solve it, and the
hardened program has 3 live nodes in each.

**`arm4s_matched` has no candidate, and that is a measurement.** The rule was
"the highest-ranked arity-3 pooled class that does not compute majority and whose
representative has the same node count as rank 1 (4 nodes)". The eligible set
holds **thirteen** arity-3 classes; the majority class is the **only** one with
4 nodes. Every other arity-3 class has 2 or 3. The control is therefore reported
as *not constructible*, not as a tie.

**Wall seconds** (this machine was shared with other agents' jobs throughout, so
these are indicative): `arm1` 4.7, `arm4` 151.6, `arm3` 176.5,
`arm4s_runnerup` 178.2, `arm2` 186.5, `arm4b` 197.5, `arm4s_offfamily` 214.6,
`arm2s_semantic` 276.8, `arm2s_trace` 277.5.

### 4.1 The strong form of the control: every eligible arity-3 pooled class

Each of the 13 eligible arity-3 classes was registered as a module and
enumerated on L1's tight scaffold — **35 223 552 programs evaluated in total**,
every run exhausted with certificate `complete`.

| rank | window? | nodes | conforming |
|---|---|---|---|
| **1** | **yes** | 4 | **144** |
| 4 | no | 2 | 0 |
| 5 | no | 2 | 0 |
| 8 | no | 2 | 0 |
| 12 | no | 2 | 0 |
| 15 | no | 3 | 0 |
| 16 | no | 2 | 0 |
| 17 | no | 2 | 0 |
| 26 | no | 3 | 0 |
| 27 | no | 3 | 0 |
| 28 | no | 3 | 0 |
| 31 | no | 2 | 0 |
| 32 | no | 2 | 0 |

**One of thirteen reaches 144. One of thirteen reaches *any* conforming program
at all, and it is the one the rule ranks first.** Pooling's rank is load-bearing
on this corpus; it is not "any plausible fragment works".

---

## 5. The held-out test — the decisive one

A held-out task is one of the six earlier tasks removed **entirely** from the
mining corpus, at every length. The rule mines the remaining five and the
induced library is offered to the removed task on the **compact scaffold**: two
nodes over `a,b,c,d`, 4 760 programs flat, 25 200 with a 3-ary module,
221 520 with the 4-ary module the LOO rule actually proposes. Every enumeration
below is a full sweep, `exhausted: true`, certificate `complete`.

`arm1_none` exhausts at **0 conforming on all seven tasks**, so the scaffold has
no flat solution and cannot itself be doing the work (F5's first clause).

| task | arm1 none | arm2 syntactic | **arm2s semantic** | *arm2s window* | arm3 authored | arm4 wrong authored | arm4s offfamily | constant |
|---|---|---|---|---|---|---|---|---|
| `t1` | 0 | 0 | **0** | *12* | 12 | 0 | 0 | 0.5000 |
| `t2` | 0 | 0 | **0** | *12* | 12 | 0 | 0 | 0.7500 |
| `t3` | 0 | 0 | **12** | *12* | 12 | 0 | 0 | 0.7500 |
| `t4` | 0 | 0 | **0** | *12* | 12 | 0 | 0 | 0.5000 |
| `t5` | 0 | 0 | **12** | *12* | 12 | 0 | 0 | 0.7500 |
| `H_par` | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0.5000 |
| `H_d134` | 0 | 0 | 0 | 0 | **0** | **4** | **4** | 0.5000 |

Gradient, 24 seeds each, seeds conforming / median exact accuracy:

| task | arm1 none | arm2 syntactic | **arm2s semantic** | *arm2s window* | arm3 authored | arm4 wrong | arm4s offfamily |
|---|---|---|---|---|---|---|---|
| `t1` | 0/24 · .7500 | 0/24 · .7500 | **0/24 · .7500** | *4/24 · .7500* | 4/24 · .7500 | 0/24 | 0/24 |
| `t2` | 0/24 · .8750 | 0/24 · .8750 | **0/24 · .8750** | *7/24 · .8750* | 7/24 · .8750 | 0/24 | 0/24 |
| `t3` | 0/24 · .8750 | 0/24 · .8750 | **24/24 · 1.0000** | *4/24 · .8750* | 4/24 · .8750 | 0/24 | 0/24 |
| `t4` | 0/24 · .7500 | 0/24 · .7500 | **0/24 · .7500** | *7/24 · .7500* | 7/24 · .7500 | 0/24 | 0/24 |
| `t5` | 0/24 · .8750 | 0/24 · .8750 | **24/24 · 1.0000** | *3/24 · .8750* | 3/24 · .8750 | 0/24 | 0/24 |
| `H_par` | 0/24 · .5000 | 0/24 · .5000 | 0/24 · .5000 | 0/24 · .5000 | **0/24 · .5000** | 0/24 · .8750 | 0/24 · .8750 |
| `H_d134` | 0/24 · .6250 | 0/24 · .6250 | 0/24 · .6250 | 0/24 · .6250 | **0/24 · .6250** | **15/24 · 1.0000** | **15/24 · 1.0000** |

*`arm2s_window` is EXPLORATORY and was not pre-registered.* It is the
highest-ranked **majority** class in the same leave-one-out table, whatever its
rank — the same device §46 used to separate "the corpus does not contain it"
from "the ranking does not pick it". It is labelled as exploratory everywhere
and counts toward no criterion.

**Four things this table settles.**

1. **The pre-registered `arm2s_semantic` helps 2 of 5 held-out tasks.** On
   `t3` and `t5` it not only solves the task but *beats the hand-authored
   ceiling on the gradient* — 24/24 at median accuracy 1.0000 against 4/24 and
   3/24 — because its 4-ary module absorbs the whole task body. On `t1`, `t2`,
   `t4` it reaches 0. This is a real, non-trivial partial transfer, and it is
   not the majority abstraction.
2. **The majority class is present in every leave-one-out table, and it is the
   same artifact every time.** All five publish
   `module:165bc290d9c82b70a8ea3cc2` — the identical digest the full corpus
   publishes. Semantic pooling recovers the abstraction from *any* five of the
   six tasks. It is the **MDL rank** that loses it, not the pooling.
3. **The scaffold discriminates, in both directions.** `MAJ3` is worthless on
   `H_par` and on `H_d134` (0 everywhere); the `D134` module — hand-authored and
   semantically pooled alike — solves `H_d134` (4 conforming, 15/24 at 1.0000)
   and is worthless on all five majority tasks. F5 does not fire.
4. **The pooled off-family module is a functioning module, not a broken one.**
   `arm4s_offfamily` solving `H_d134` at the same 4 / 15-of-24 as the
   hand-authored `D134` proves that its 0 on L1 and on `t1`–`t5` is a *correct
   negative*, not a load failure.

### 5.1 Why the rank flips, measured

`mine_multi`/`mine_semantic`'s R4 sums the saving over corpus **entries**, so
removing a task removes saving from every fragment that occurs in it and leaves
fragments that do not occur in it **untouched**:

| corpus | rank-1 (4-ary, 5 nodes, 2 tasks) | majority (3-ary, 4 nodes) | margin |
|---|---|---|---|
| full `C-minall` | +349 104 (rank 2) | **+415 872 (rank 1)** | majority ahead by 66 768 (**19.1 %**) |
| ∖ `t1` | **+333 496** | +330 672 (rank 2) | **2 824 (0.85 %)** |
| ∖ `t2` | **+349 104** | +330 672 (rank 3) | 18 432 (5.57 %) |
| ∖ `t3` | **+349 104** | +330 672 (rank 2) | 18 432 (5.57 %) |
| ∖ `t4` | **+333 496** | +330 672 (rank 2) | **2 824 (0.85 %)** |
| ∖ `t5` | **+349 104** | +330 672 (rank 2) | 18 432 (5.57 %) |

Majority's saving falls from 415 872 to 330 672 (it loses 15 of the removed
task's entries, ~5 680 bits each); the 4-ary competitor occurs in only 2 tasks
and its saving is **numerically unchanged** across corpora. The rank-1 decision
in two of five leave-one-outs turns on **0.85 %**.

**An observation, explicitly not adopted.** `mine.propose` sorts by
`(-saving, -tasks, -occurrences, -nodes, digest)`. The majority class occurs in
**4 or 5 tasks**; the competitor in **2**. If tasks preceded saving in the
tie-break, majority would be rank 1 in all six corpora. **The rule was not
re-tuned** — the pre-registration forbids it and this is recorded as an
observation for a future pre-registered experiment, not as a result.

---

## 6. Verdict against each pre-registered falsification criterion

| criterion | fires? | evidence |
|---|---|---|
| **F1 — pooling any fragment works** | **NO** | `arm2s_semantic` 144 vs `arm4s_runnerup` **0** and `arm4s_offfamily` **0**, same exhausted 2 709 504 space, same certificate; 18/24 and 8/8 vs 0/24 and 0/8. The sweep: **1 of 13** eligible arity-3 pooled classes reaches any conforming program. |
| **F2 — compression, not reusable abstraction** | **YES, at the ranking layer** | `arm2s_semantic` reaches the ceiling on L1 but helps only **2 of 5** held-out tasks. The class that would help all five *is in the pooled table every time*, at rank 2–3. Reported as compression at the score, selection at the identity. |
| **F3 — frequency artifact** | **NO** | (a) The identical rule on `F′` proposes `F′`'s **own** window — different digest **and different truth table** from the majority family's. It reads the corpus, not the fragment enumerator. (b) Re-ranking the eligible set by raw pooled occurrence count changes the rank-1 class in **15 of 15** corpora, so the MDL score is not a frequency proxy. |
| **F4 — cost exceeds saving** | **NO** | §7. |
| **F5 — the scaffold is doing the work** | **NO** | `arm1_none` exhausts at 0 conforming on all seven compact tasks; `MAJ3` scores 0 on `H_par` and `H_d134`; `D134` scores 0 on all five majority tasks. |

---

## 7. Cost, reported honestly

**Pooling itself is nearly free.** On the same corpus, under the same rule,
switching identity from `Program.digest` to `(arity, truth table)`:

| corpus | syntactic mine | semantic mine | ratio | eligible (syn → sem) |
|---|---|---|---|---|
| `maj` `C-minall` | 0.507 s | 0.547 s | **1.08×** | 94 → 40 |
| `off` `C-minall` | 0.400 s | 0.476 s | **1.19×** | 104 → 41 |

The extra work is one exhaustive truth-table evaluation per canonical fragment,
memoised by digest; it costs 8–19 % more mining time and yields a *smaller*
eligible set. This is not where the money goes.

**The corpus is where the money goes, and §46's cheap half is enough.** This
track's primary corpus is `C-minall` (11 536 818 DFS nodes, reused from §46).
`C-trace` costs **844 111 950** DFS nodes — **73×** more — and buys nothing here:
`C-minall` and `C-trace` publish the *same digest* and give the *same* 144 /
18-of-24 / 8-of-8. **The expensive corpus is unnecessary.** Building `F′` from
scratch cost 13 843 086 DFS nodes and 75 wall-seconds for 117 verified entries.

**The enumeration is the real budget.** Nine L1 arms plus a 13-class sweep:
about **60 million programs** evaluated on the tight scaffold, ~2 000 CPU-seconds
of enumeration, plus ~5 300 CPU-seconds of gradient. The arity-3 restriction
declared in advance kept every arm in the 2 709 504-program space; §46 measured
a single 4-ary arm at 64 161 792 programs and 8 184 seconds.

**Verdict on F4: pooling costs 1.08–1.19× the mining and saves the difference
between 0 and 144 conforming programs.** It pays, easily, *on the task it was
mined for*.

---

## 8. Rule sensitivity, reported regardless

`mine_semantic.propose` swept once over `MAX_NODES ∈ {3,4,5} × MAX_HOLES ∈
{2,3,4} × MIN_TASKS ∈ {2,3,4}` on `C-minall`. Rank-1 is the majority class in
**12 of 27** settings. The 15 failures are **exactly** the settings where the
class is not constructible at all: `MAX_NODES = 3` (the representative has 4
nodes) or `MAX_HOLES = 2` (it is 3-ary). **Within the 12 settings where the
class can exist, it is rank 1 in 12 of 12** — including every `MIN_TASKS`. The
rule was not re-tuned.

---

## 9. Disclosures

* **A defect in this track's own control task, caught by its own baseline.**
  `evaltasks._parity` was first written `bool(a) != bool(b) != bool(c) !=
  bool(d)`, which Python evaluates as the **chained comparison** `(a≠b) ∧ (b≠c) ∧
  (c≠d)` — the alternating-sequence predicate, not the XOR the pre-registration
  declared. It was caught because the class balance came out 14/16 where the
  pre-registration recorded 8/16. It is fixed and the `H_par` rows were re-run;
  every other row in the held-out tables is from the original run. **Both
  versions give the same verdict** — all seven arms 0 conforming, exhausted,
  `complete` — so no conclusion changes, but the published number is the correct
  one.
* **`arm4b_wrong_mined` needed a reading of "runner-up".** The pre-registration
  says "the syntactic rule's runner-up, same machinery (§44's arm 4b)", which is
  rank 2. The `runnerup` *selection function* written later ("top arity-3
  non-majority") collapses onto rank 1 for the syntactic table, because that
  table's rank 1 is already non-majority. `arm4b` therefore uses rank 2
  (`82b93e4b`, §44's `M′`), which is what §44's arm 4b was; `arm4s_runnerup`
  uses the top-non-majority rule as written. Both readings are in
  `poolmine.pick` and named.
* **The family sharing is declared, not hidden.** `corpus.py` states openly that
  the six earlier tasks share a latent majority window. The held-out test shows
  transfer *within* a family with declared shared structure; it does **not** show
  cross-family transfer, and `arm4s_offfamily` scoring 0 on all five majority
  tasks is the measurement of that boundary.
* **Five held-out tasks plus two controls.** `t6` was excluded in advance
  because its certified flat minimum is 3 gates. Measured here, `t6` is in fact
  0 conforming flat in the compact scaffold too, so the pre-registered reason was
  not the operative one; the operative reason is that `t6` needs **two** module
  calls and the compact scaffold has one free node. Recording the correction.
* **Seven tasks is small.** The held-out conclusion is stronger than
  anecdote-strength (five leave-one-out runs, unanimous on the rank flip) but the
  family count is one, plus one off-family corpus.
* **Wall-clock caveat.** The machine was shared with other agents' training and
  test jobs throughout. Wall seconds are indicative; DFS nodes, evaluated counts,
  space sizes and conforming counts are not.
* **A 12-minute gradient run was lost** to `arm4s_matched` having no module; the
  arm list is now filtered by `armlib.runnable_l1_arms()` and the run repeated.

---

## 10. If a semantic class were integrated into `tcn.library` — design only

**This is a design, not an implementation. `tcn/` is not modified by this track
and no core hook was needed for anything above.** The result that would justify
implementing it — the rank-1 proposal transferring to held-out tasks — did
**not** hold, so the recommendation is *not to implement yet*. Recorded because
DESIGN.md §7 asked for it and because the measurements above sharpen it.

Following DESIGN.md §7, additively:

* `digest` stays **artifact** identity. Everything `strict` and `revalidate`
  verify today is untouched; the fixture still re-executes exactly.
* `semantic_id` becomes **class** identity: for a finite carrier, the canonical
  hash of the exhaustive input→output table together with the arity — precisely
  the key `mine_semantic` pools on, which this track measured collapsing
  8 circuits (`C-minall`) and 58 (`C-trace`) into one class.
* An entry becomes a class holding several artifacts with a declared
  **preferred** member. §46 measured that `Program.digest` is not canonical up
  to graph isomorphism — two topological orders of the same four-gate majority
  give two digests — so a class will routinely hold artifacts that are the *same
  circuit*, and the preferred-member rule must be deterministic (fewest nodes,
  then digest, as `mine_semantic` already elects).
* §30's constraint is preserved: a hardened module is width-specific, so the
  **class holds the schema and artifacts hold instantiations**.

**Two things this track measured that the design must absorb.** First, the
class-vs-artifact split is exactly the right shape — `165bc290` and `8ceedf7b`
are two artifacts of one class and perform *identically* on every axis measured
here (144/144, 18/24, 8/8, median 1.0000). Second, and less comfortably: making
the class the identity is **not sufficient** to make the *library selection*
robust. §5.1 shows the winning class changing under a 0.85 % MDL margin when one
task of six is removed. A class-aware library would need a selection rule with a
stability criterion — not merely a class-aware identity relation — before it
would earn its place.

---

## 11. Reproduction and provenance

* **Nothing in `tcn/` or `generators/` was modified.** `git status` for this
  branch touches only `research/semantic-library/`.
* **`PREREGISTRATION.md` was committed at `30904e2` before any arm ran**,
  together with `_paths.py` and nothing else.
* **`mine_semantic.py`, `mine_multi.py`, `mine.py`, `later.py`, `corpus.py`,
  `corpora.py`, `enumerate_programs.py`, `minimal.py` and `retention.py` are
  imported unchanged.** Not one line of the rule under test was modified.
* **Every enumeration is a full sweep**, `exhausted: true`, certificate
  `complete`, `evaluated` reported separately from `space_size`. Nine L1 arms,
  a 13-class sweep and 49 held-out enumerations.
* **Every corpus program is executed through `tcn`** against its full 16-row
  truth table before entering a corpus (`verified_in_tcn: true` for all six `F′`
  bands; §46 verified the majority bands).
* **Test suite: 337 tests, and the count that passes depends only on a gitignored
  symlink.** Both configurations were measured here.
  * With `generators/computer/engine/node_modules` symlinked from the main
    checkout: **336 passed, 1 failed** —
    `tests/test_panel_interface.py::test_panel_episode_replays_and_restores`.
  * With the symlink removed (the state this branch is committed in):
    **324 passed, 13 failed**, all in `tests/test_panel_interface.py`.

  Both are the documented environmental failures on main's own code. This track
  cannot affect either: `pyproject.toml` sets `testpaths = ["tests"]`, so nothing
  under `research/` is ever collected, and this track adds no file under `tests/`
  or `tcn/`. The symlink was removed before committing.
* **The shipped fixture reproduces**: `python -m tcn train --episodes 160` gives
  `initial_prediction_loss` 0.248835613951087 → `final_prediction_loss`
  0.0022308224288281053, `fully_frozen: true`,
  `frozen_evaluation_mean_return` 4.0.
* **Large artifacts stay out of git**: `out/` holds JSON tables only; no
  checkpoints, no `.pt`, no enumeration dumps.
* Every figure in this document comes from a file in `out/`; `tables.py`
  regenerates the tables directly from those files.
