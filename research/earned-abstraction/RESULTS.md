# Earned abstraction — a selection rule for what to crystallize, and whether inheriting it pays

The library, the publish/inherit flow and recursive abstraction all work and are
measured (FINDINGS §12, §25, §30). In every existing demonstration a **human**
chose which subprogram became the module. This track closes that gap: a rule
that reads solved programs from earlier tasks and decides, from the typed
program graph alone, which subprogram to crystallize; then a four-arm
measurement of whether inheriting what it proposed pays on a later task.

Nothing in `tcn/` or `generators/` is modified — no core hook was needed.
[`PREREGISTRATION.md`](PREREGISTRATION.md) was written before the arms were run.
Reproduce with `PYTHONPATH=<repo>:. <repo>/.venv/bin/python <script>` from this
directory.

---

## VERDICT

**The rule works, proposes a real abstraction, and the abstraction does not pay
on this later task. The reason is measured rather than guessed, and it is not
the rule: the corpus the rule was given does not contain the abstraction the
later task needs, because exact minimisation fused that primitive into its
wrappers.**

1. **The rule proposed `M(x0,x1,x2) = x2 ∨ (x0 ∧ x1)`** — two nodes, arity 3,
   occurring six times across four of the six earlier tasks, worth **+3 648
   description bits** over the corpus by the repo's own `description_bits`
   accounting. It is *not* a degenerate answer: the single-gate fragments
   `and`, `or`, `xor` occur far more often (11, 10 and 7 sites) and the MDL
   score correctly rejects all three at **−6 272, −6 120 and −5 376 bits**,
   because one call site cannot pay for a definition. The proposal is identical
   under **30 of 40** parameter settings swept, and under every setting that
   admits a 3-ary abstraction at all.

2. **What it proposed is a genuine part of the later task's primitive, not
   noise.** `MAJ3(a,b,c) = and(or(a,b), M(a,b,c))`, verified by execution. With
   the earned module inherited, the later task's shortest verified route falls
   from **9 nodes to 7**. That is a real reduction earned from earlier tasks.

3. **It is not enough.** The hand-authored MAJ3 takes the same task to **3
   nodes**. On the only scaffold small enough to certify — three nodes, where
   the flat minimum is proved ≥ 7 — the exhaustive sweep gives
   **0 conforming out of 2 709 504 for the earned module**, identical to both
   wrong-module controls and to no library at all, against **144 for the
   hand-authored module**. All five sweeps are `exhausted`, certificate
   `complete`.

4. **Against the pre-registered criteria, on the certified comparison: the
   earned module ties `arm1_none`, ties both wrong-module controls, and loses to
   `arm3_authored`.** All three payoff falsification conditions fire.

5. **The cause is in the corpus, and it is measurable.** Only **1 of the 6**
   solved earlier programs still contains a MAJ3 body; in three of the others
   the exact solver fused or distributed the wrapper gate into the majority, in
   a fourth it used an xor realisation with no majority-valued node, and the
   sixth collapses to 3 gates (track 5's collapse, reproduced independently).
   Dropping the reuse requirement to
   `MIN_TASKS = 1` does not rescue it: the MAJ3 fragment then ranks **14th with
   a saving of −4 448 bits**, because a definition charged once against a single
   call site is a loss. The rule is giving the right answer for the corpus it
   was given. **You can only mine an abstraction the solver left intact**, and
   minimisation is exactly the process that does not leave it intact.

6. **A second, unplanned negative, and it is the sharper one for the project.**
   The shipped gradient path **cannot solve the earlier tasks at all**: 0 of 8
   seeds on five of six tasks at 5 nodes, 0 of 8 at 6 nodes, and 0 of 16 at
   3 000 steps. Only the collapsing task is solved. The corpus exists only
   because it was solved by exhaustive straight-line search instead. A closed
   abstraction loop needs earlier tasks that are *solved*; on 4-input Boolean
   synthesis at five gates the shipped synthesizer does not get there.

---

## 1. The setting

**Earlier tasks (the corpus).** Six Boolean functions over four inputs
`a,b,c,d`, in `corpus.py`. The family has declared latent shared structure —
each is a 3-of-4 majority window over some triple, combined with the remaining
input by a two-input gate — because library learning is only meaningful over a
family that shares structure. What is *not* supplied anywhere is which
subprogram realises that structure, its arity, where it sits in a solved
program, or that it exists at all.

| task | function |
|---|---|
| `t1_maj_abc_xor_d` | `maj(a,b,c) xor d` |
| `t2_maj_abc_and_d` | `maj(a,b,c) and d` |
| `t3_maj_bcd_or_a` | `maj(b,c,d) or a` |
| `t4_maj_acd_xor_b` | `maj(a,c,d) xor b` |
| `t5_maj_abd_or_c` | `maj(a,b,d) or c` |
| `t6_maj_abc_xor_maj_bcd` | `maj(a,b,c) xor maj(b,c,d)` — overlapping windows, included deliberately as a decoy |

**Later task.** `maj(a,b,c) xor maj(d,e,f)` over six inputs, on the scaffolds of
`research/recursive-abstraction-retest` — the current state of the art for this
question, which is why arm 3 is literally that track's hand-authored module in
the same scaffold. The later task was fixed before the rule ran and **the rule
never sees it**: its only inputs are the six solved 4-input programs.

**The trap in §30 is designed around, not weakened.** A stored module's input
type names the observation, so a module hardened at one width is not callable at
another. Here every port is `BOOL`, which has no width, so a published module
transfers from 4-input tasks to a 6-input task with no change to the type
system.

---

## 2. Solving the earlier tasks

### 2.1 The shipped gradient path does not solve them

`run_corpus.py` runs the repo's gradient synthesis (`SoftProgram` + Adam +
argmax export, the retest's loop) on a chain scaffold where each node sees every
input and every earlier node.

| scaffold | space | t1 | t2 | t3 | t4 | t5 | t6 |
|---|---|---|---|---|---|---|---|
| 5 nodes, 8 seeds, 900 steps | 56·85·120·161·208 = 1.94·10¹⁰ | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | **5/8** |
| 6 nodes, 8 seeds, 900 steps | 6.2·10¹² | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | **3/8** |
| 6 nodes, 16 seeds, 3 000 steps | 6.2·10¹² | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | **10/16** |

Only `t6` is solved, and `t6` is the task that **collapses**: its certified
minimum is 3 gates, not 9 — track 5's collapse finding, reproduced here with
independent machinery. Enumeration is not an option either: the 5-node space is
1.94·10¹⁰ programs, four orders of magnitude past what `enumerate_fit` can
exhaust here.

This is a negative result about the substrate that the abstraction question
depends on: **a closed loop needs earlier tasks that get solved**, and at five
gates over four Boolean inputs the shipped synthesizer does not solve them.

### 2.2 So the corpus is solved exactly, with a certificate

`minimal.py` does iterative-deepening exhaustive search over straight-line
programs in the scaffold's own basis (`and`, `or`, `xor`, `not`) — the same
instrument `research/recursive-abstraction-retest/min_program.py` uses to prove
its flat minima, reimplemented here so node counts and exhaustion can be
reported. Every length below the answer is exhausted, so the reported length is
the **certified minimum in that basis**; within that length the first program in
enumeration order is returned, exactly as `enumerate_fit(rank='order')` does.
Every circuit is rebuilt as a `tcn.graph.Program` and executed through `tcn`
against the full 16-row table before it enters the corpus.

**Cross-check against a recorded number.** Run on MAJ3, this reproduces
`min_program.json` exactly: minimum 4 gates and the identical program
`and(a,b); or(a,b); or(c,g0); and(g1,g2)`. That recorded figure reproduces.

| task | min gates | lengths exhausted | DFS nodes expanded | wall (s) | verified in `tcn` |
|---|---|---|---|---|---|
| `t1_maj_abc_xor_d` | 5 | 0–4 | 4 267 741 | 5.2 | yes |
| `t2_maj_abc_and_d` | 5 | 0–4 | 4 266 864 | 6.9 | yes |
| `t3_maj_bcd_or_a` | 5 | 0–4 | 79 689 596 | 97.0 | yes |
| `t4_maj_acd_xor_b` | 5 | 0–4 | 18 238 057 | 23.0 | yes |
| `t5_maj_abd_or_c` | 5 | 0–4 | 4 266 865 | 4.1 | yes |
| `t6_maj_abc_xor_maj_bcd` | **3** | 0–2 | 15 679 | 0.0 | yes |

The six solved programs:

```
t1  g0=and(a,b) g1=or(a,b)  g2=or(c,g0)  g3=and(g1,g2)  g4=xor(d,g3)
t2  g0=and(a,b) g1=or(a,b)  g2=or(c,g0)  g3=and(d,g1)   g4=and(g2,g3)
t3  g0=and(b,c) g1=or(a,g0) g2=or(b,c)   g3=and(d,g2)   g4=or(g1,g3)
t4  g0=xor(a,b) g1=xor(a,c) g2=xor(a,d)  g3=and(g1,g2)  g4=xor(g0,g3)
t5  g0=and(a,b) g1=or(a,b)  g2=or(c,g0)  g3=and(d,g1)   g4=or(g2,g3)
t6  g0=xor(a,d) g1=xor(b,c) g2=and(g0,g1)
```

Read `t1` against `t2`: both compute the same majority, but only `t1` leaves a
node whose value *is* `maj(a,b,c)` (`g3`). `t2` fuses the `and d` into one of the
majority's conjuncts — `g4 = (c ∨ ab) ∧ (d ∧ (a ∨ b))` — and `t3` and `t5`
distribute their `or` across the majority's two terms, so in all three the
majority exists in the algebra but not at any node. `t4` realises the majority in
an entirely different xor basis, and `t6` collapses. **One of six solved programs
contains a MAJ3 body.** That single fact determines everything below.

---

## 3. The rule, stated precisely enough to reimplement

`mine.py`. It reads node sets, edges, operator contracts and
`Program.description_bits`, and nothing else — no operator meanings, no task
semantics, nothing domain-specific.

> **R1 — Fragments.** For each solved, pruned, frozen program `P` and each node
> `r` of `P`, a node set `S ∋ r` is a *fragment rooted at r* when
> (a) every node of `S` has a directed path to `r` inside `S`;
> (b) `|S| ≤ MAX_NODES`;
> (c) `S` is *single-exit*: no node of `S \ {r}` is read from outside `S`, and
> none of them is a program output or a state update.
> (a) makes it a sub-DAG with one root; (c) makes replacing it by one call
> semantics-preserving.
>
> **R2 — Abstraction.** Every port a node of `S` reads that is not itself in `S`
> becomes a hole, **one hole per distinct external port**, so sharing inside the
> fragment is preserved — a port read twice stays one argument. The fragment
> becomes a `Program`: inputs `x0..x{k-1}` in order of first reference under the
> topological node order, nodes `n0..n{m-1}` in that order, output the root. Its
> identity is `Program.digest`, so two occurrences are the same abstraction
> exactly when the content-addressed store would give them one file.
>
> **R3 — Reuse.** Occurrences within one program are reduced to a maximal
> node-disjoint subset (greedy in node order), since only disjoint occurrences
> can be rewritten together. An abstraction is eligible only if it occurs in at
> least `MIN_TASKS` distinct earlier tasks — reuse *across* tasks, not
> repetition inside one.
>
> **R4 — Score.** The corpus description-length saving in the repo's own units,
> the definition charged once and every call site individually:
>
> `saving(F) = Σᵢ bits(Pᵢ) − Σᵢ bits(rewriteᵢ(F)) − bits(F)`
>
> where `bits` is `Program.description_bits()` with no registry, so each
> rewritten program is charged for its call sites and `F`'s body is charged
> exactly once — which is what `Library` storage does.
>
> **R5 — Proposal.** Eligible abstractions are ranked by saving, ties broken by
> (tasks, occurrences, nodes, digest); the top one is proposed. Every rewrite is
> **executed against its task's examples before it is scored**; a rewrite that
> changes any output disqualifies the abstraction.

Parameters: `MAX_NODES = 5`, `MAX_HOLES = 4`, `MIN_TASKS = 2`. `MAX_HOLES`
exists because candidate enumeration at a call site is `O(ports^arity)`, so an
abstraction the search cannot afford to offer is not a useful abstraction.

`test_mine.py` checks the machinery on hand-built programs: canonical identity
across differently-named ports, single-exit enforcement, semantics-preserving
rewrites, and that on a corpus actually built from MAJ3 the rule does propose
MAJ3.

---

## 4. What the rule proposed

`run_mine.py`, corpus of §2.2, 13 eligible abstractions.

| rank | body | arity | tasks | disjoint sites | saving (bits) |
|---|---|---|---|---|---|
| **1** | `n0 = and(x0,x1); n1 = or(x2,n0)` | 3 | 4 | 6 | **+3 648** |
| 2 | `n0 = or(x0,x1); n1 = and(x2,n0)` | 3 | 4 | 4 | +416 |
| 3–5 | three 3-node, 4-ary fragments | 4 | 2 | 2 | −1 376 |
| 6–10 | five 2-node fragments | 3 | 2 | 2 | −2 808 … −2 824 |
| 11 | `xor(x0,x1)` | 2 | 3 | 7 | −5 376 |
| 12 | `or(x0,x1)` | 2 | 4 | 10 | −6 120 |
| 13 | `and(x0,x1)` | 2 | 6 | 11 | −6 272 |

The bottom three rows are the check that the rule is not counting frequency:
`and` occurs in **all six** tasks at **11** sites and is ranked last, because a
one-gate definition saves nothing at a call site and still has to be stored.
Only two abstractions in the whole corpus have a positive MDL saving, and both
are two-node fragments of the majority.

**Published through the shipped path.** Rank 1 is published to a real
`tcn.library.Library` as `earned@1`
(`module:ec516b3808ddef7e7d0e7d22`, 2 nodes, execution cost 2.0, 6 048
description bits, 8 recorded fixture cases) and rank 2 as `earned_runner_up@1`
(`module:b22e2ef14a32a68f1224304f`, same shape). The later task loads them with
`policy="strict"`, so the module crosses from one task to another through the
manifest, not through a Python variable.

**What rank 1 is.** `M(x0,x1,x2) = x2 ∨ (x0 ∧ x1)`, truth table `01010111`. It
is the and-or half of the majority: `MAJ3(a,b,c) = and(or(a,b), M(a,b,c))`,
verified inside the 64-row execution of the arm-2 route in §5. Rank 2 is the
other half,
`M'(x0,x1,x2) = x2 ∧ (x0 ∨ x1)`, and `MAJ3(a,b,c) = M'(a, b, or(c, and(a,b)))`.
Neither is the primitive; each is exactly half of it.

---

## 5. Expressive economy: what each library makes reachable

Search-free and verified. Each route is an explicit program in that arm's basis,
built as a `tcn.graph.Program` and executed against all 64 rows of the later
task (`run_economy.py`).

| arm | library | verified route | nodes | description bits | execution cost |
|---|---|---|---|---|---|
| `arm1_none` | — | 9-gate flat circuit | **9** | 20 784 | 9.0 |
| `arm2_earned` | `M(x0,x1,x2)=x2∨(x0∧x1)` | `or; M; and` twice, then `xor` | **7** | 23 600 | 9.0 |
| `arm3_authored` | MAJ3 | `MAJ3; MAJ3; xor` | **3** | 19 464 | 9.0 |
| `arm4_wrong_authored` | truth table 134 | the flat circuit — the module is unused | **9** | 20 784 | 9.0 |
| `arm4b_wrong_mined` | `M'(x0,x1,x2)=x2∧(x0∨x1)` | `and; or; M'` twice, then `xor` | **7** | 23 600 | 9.0 |

These are **verified upper bounds**, not proved minima: each is an explicit
program executed against all 64 rows. The one lower bound available is the flat
one, **≥ 7 nodes**, by gate elimination over the full binary basis B2 (the
retest's `min_program.py`); every `and`/`or`/`xor`/`not` node is at most one B2
gate, so it bounds node count in the scaffold basis too. A module arm's scaffold
still offers every flat candidate, which is why `arm4_wrong_authored`'s shortest
route is the flat one: its module is pure cost.

Two things to read off this table. The earned module **does** buy something
real — 9 nodes to 7 — but it buys it by supplying half a primitive, so the
caller still has to rebuild the other half at every call site. And it is
**bigger in bits**, not smaller: 23 600 against the flat 20 784, because the
definition is charged and the saving at two call sites does not cover it. The
hand-authored module is the only one that is smaller in bits as well as in
nodes.

Note also that `arm2_earned` and `arm4b_wrong_mined` have **identical** node
counts, description bits and execution cost. The rank-1 and rank-2 abstractions
are exactly equally useful to this later task, which is the first sign that the
*selection* between them cannot be contributing anything here.

---

## 6. The four arms — exhaustive enumeration, with certificates

Tight scaffold: three `BOOL` nodes, six inputs, the retest's construction. The
flat minimum is proved ≥ 7, so no flat program fits; the space is small enough
to exhaust. `tcn.search.enumerate_fit`, tolerance 1e-3, `rank='order'`,
`max_programs = 2^24`, full sweep (no early exit), so the certificate survives.

| arm | library | candidates/node | search space | evaluated | exhausted | conforming | certificate | wall (s) |
|---|---|---|---|---|---|---|---|---|
| `arm1_none` | none | 120/120/16 | 230 400 | 230 400 | yes | **0** | `complete` | 17 |
| `arm2_earned` | **earned** `M = x2∨(x0∧x1)` | 336/336/24 | 2 709 504 | 2 709 504 | yes | **0** | `complete` | 635 |
| `arm3_authored` | hand-authored MAJ3 | 336/336/24 | 2 709 504 | 2 709 504 | yes | **144** | `complete` | 658 |
| `arm4_wrong_authored` | hand-authored table 134 | 336/336/24 | 2 709 504 | 2 709 504 | yes | **0** | `complete` | 531 |
| `arm4b_wrong_mined` | **earned runner-up** `M' = x2∧(x0∨x1)` | 336/336/24 | 2 709 504 | 2 709 504 | yes | **0** | `complete` | 694 |

Every sweep is exhausted and every certificate is `complete`, so these are
statements about the whole space and not about a search that gave up. Task
quality: a conforming program is exact on all 64 rows; the best constant and a
uniform random predictor both score 0.5.

`arm1_none` and `arm3_authored` reproduce the retest exactly: 230 400 exhausted
with no solution, and 2 709 504 exhausted with **144** conforming. The first
conforming program in enumeration order is bit-for-bit the one that track
recorded — `module(a,b,c) xor module(d,e,f)`, 3 live nodes, **19 496**
description bits with the library charged and **9 440** without, execution cost
9.0, every figure matching its `enumeration.json`. That track's further claim —
that **all** 144 use the module — follows from this table rather than needing a
second count. The flat sub-space of the arm-3 scaffold is exactly `arm1_none`'s
230 400 programs, and it is exhausted with none conforming; a conforming program
must therefore select a module somewhere, and if that node were dead the pruned
program would be flat and would lie in that same exhausted sub-space. So every
one of the 144 has the module live on the output path.

**The earned module contributes nothing here.** It is 0 conforming in a space of
2 709 504 — the same as no library at all, the same as a hand-authored wrong
module, and the same as its own runner-up. It has to be: its shortest verified
route is seven nodes and the scaffold has three.

---

## 7. Sample complexity — the same task learned instead of enumerated

Identical scaffold, examples, signals, objective, schedule and seeds in every
arm; the library is the only difference. `SoftProgram` + Adam, lr 0.05, 400
steps, conformance of the argmax export checked every 10 steps, first
conforming step recorded. Best constant and uniform random both score 0.5 exact
accuracy.

### 7.1 Tight scaffold (3 nodes), 24 seeds

| arm | solved | median steps to conform | mean exact accuracy | best | module on output path |
|---|---|---|---|---|---|
| `arm1_none` | 0/24 | — | 0.6250 | 0.6250 | 0 |
| `arm2_earned` | **0/24** | — | 0.6875 | 0.7812 | 0 |
| `arm3_authored` | **18/24** | **50** | 0.8750 | 1.0000 | 18 |
| `arm4_wrong_authored` | 0/24 | — | 0.6146 | 0.6250 | 0 |
| `arm4b_wrong_mined` | 0/24 | — | 0.7161 | 0.7812 | 0 |

### 7.2 Wide scaffold (9 nodes), 8 seeds — both routes fit

| arm | solved | median steps to conform | mean exact accuracy | best | module on output path |
|---|---|---|---|---|---|
| `arm1_none` | 0/8 | — | 0.6875 | 0.6875 | 0 |
| `arm2_earned` | **0/8** | — | 0.7930 | 0.8750 | 0 |
| `arm3_authored` | **8/8** | **90** | 1.0000 | 1.0000 | 8 |
| `arm4_wrong_authored` | 0/8 | — | 0.6641 | 0.6875 | 0 |
| `arm4b_wrong_mined` | **0/8** | — | **0.7930** | **0.8750** | 0 |

`arm1_none` 0/8 and `arm3_authored` 8/8 reproduce §12's wide-scaffold table
exactly, including that every arm-3 success puts the module on the output path.

Worth noting in passing: `arm4_wrong_authored` is *below* `arm1_none` on mean
accuracy in both tables (0.6146 vs 0.6250 tight, 0.6641 vs 0.6875 wide). A wrong
module is not neutral — it is a wider candidate list with nothing in it, and it
costs a little.

**The sharpest single number in this track is the last row.** On the wide
scaffold `arm2_earned` and `arm4b_wrong_mined` agree to four decimal places on
mean accuracy (0.7930) and on best accuracy (0.8750), and the economy table has
them identical in nodes (7), description bits (23 600) and execution cost (9.0).
The earned module and its runner-up are interchangeable for this task. Whatever
the small accuracy edge over `arm1_none` is (0.7930 against 0.6875), it is an
effect of *having a 3-ary module candidate at every node*, not of *which* module
the rule selected — which is precisely the pre-registered "the selection
contributed nothing" condition.

Wall clock, for completeness and with the caveat that these runs shared a loaded
20-core machine with each other: 2 391 s for the 120 tight runs, 1 471 s for the
40 wide runs; 0.03–0.36 s per step tight, 0.59–1.65 s per step wide.

---

## 8. Why the rule did not propose MAJ3 — measured, not guessed

Three independent measurements, all pointing at the corpus rather than the rule.

**8.1 The primitive is not in the corpus.** Only **1 of the 6** solved programs
(`t1`) contains a node whose value is `maj(·,·,·)`. `t2` fuses its `and d` into
one of the majority's conjuncts and `t3` and `t5` distribute their `or` across
the majority's two terms, so all three compute the majority without ever holding
it; `t4` realises the majority in an xor basis with no majority-valued node at
all; `t6` collapses to 3 gates.

**8.2 Relaxing the reuse requirement does not rescue it.** With
`MIN_TASKS = 1`, so that a fragment occurring in a single task is eligible, the
MAJ3 fragment does appear — at **rank 14 of 36, with a saving of −4 448 bits**.
A definition charged once against exactly one call site is a loss under any
honest MDL, and the rule is right to reject it. The obstacle is not the
threshold; it is that a four-node body occurs once.

**8.3 It is not an artifact of the solver's tie-break.** Each task has many
equally minimal programs and the exact solver returns the first in *its*
enumeration order, exactly as `enumerate_fit(rank='order')` does. Sweeping all
six permutations of the binary operator order re-solves the whole corpus and
re-runs the rule (`run_order_robustness.py`):

| operator order | min gates (t1..t6) | tasks whose program contains a MAJ3 body | rank-1 proposal | MAJ3's rank |
|---|---|---|---|---|
| `and, or, xor` (the declared order) | 5,5,5,5,5,3 | 1 | 2-node, 3-ary, +3 648 | not eligible |
| `and, xor, or` | 5,5,5,5,5,3 | 1 | 2-node, 3-ary, +5 312 | not eligible |
| `or, and, xor` | 5,5,5,5,5,3 | 2 | 3-node, 4-ary, +5 536 | **5th, +1 224** |
| `or, xor, and` | 5,5,5,5,5,3 | 2 | 3-node, 4-ary, +5 536 | **5th, +1 224** |
| `xor, and, or` | 5,5,5,5,5,3 | 1 | 2-node, 3-ary, +5 312 | not eligible |
| `xor, or, and` | 5,5,5,5,5,3 | 1 | 2-node, 3-ary, +5 312 | not eligible |

The certified minima are identical under every order, as they must be. **Under
no order does the rule propose MAJ3**, and under no order do more than two of
six solved programs retain a MAJ3 body. Under two orders MAJ3 becomes eligible
with a *positive* saving and still ranks fifth. The negative is robust to the
one arbitrary choice in the pipeline.

**The general statement this supports.** A library-learning rule can only mine
an abstraction that the solver left intact, and *minimisation is exactly the
process that does not leave it intact*. Sharing a subprogram and minimising gate
count pull in opposite directions: a minimal circuit fuses a primitive's last
gate into whatever consumes it whenever that saves one gate. This is the
counterpart, on the acquisition side, of the preference problem §12 recorded on
the reuse side — there, nothing in the objective prefers a module; here, nothing
in the *solver* preserves one.

---

## 9. Sensitivity of the rule to its own parameters

`run_sensitivity.py`, 40 settings — `MAX_NODES ∈ {2,3,4,5,6}`,
`MAX_HOLES ∈ {2,3,4,5}`, `MIN_TASKS ∈ {2,3}`.

* **30 of 40 settings give exactly the same rank-1 proposal**,
  `ec516b3808dd`, at the same +3 648 bits.
* The 10 that do not are all `MAX_HOLES = 2`, which admits only single-gate
  abstractions; there the best available saving is **−5 376 bits** and the rule
  is reporting, correctly, that nothing in reach is worth abstracting.
* `MAX_NODES` never changes the answer between 2 and 6.

The rule is not sitting on a knife edge, and the negative in §6–7 is not a
parameter choice.

---

## 10. Verdict against the pre-registered falsification criteria

| pre-registered condition | fires? | evidence |
|---|---|---|
| "the rule found nothing worth abstracting" — earned does not beat no-library | **yes** | 0 vs 0 conforming in an exhausted 2 709 504-program space; 0/24 vs 0/24 tight; 0/8 vs 0/8 wide. The only difference is sub-threshold accuracy (0.7930 vs 0.6875 wide), with no conforming program in either arm. |
| "the selection contributed nothing" — earned ties a comparable wrong module | **yes** | Ties `arm4_wrong_authored` (0 vs 0, exhausted) and ties `arm4b_wrong_mined` to four decimals on wide accuracy (0.7930 / 0.8750 both), with identical nodes (7), bits (23 600) and cost (9.0). |
| "the rule is worse than a human choosing" | **yes** | 0 vs 144 conforming; 0/24 vs 18/24; 0/8 vs 8/8; 7 nodes vs 3; 23 600 bits vs 19 464. |
| "the rule is degenerate" — proposes a 1-node fragment, or nothing clears the reuse gate | **no** | 13 eligible abstractions; the three single-gate fragments are ranked last with negative savings despite being the most frequent. |

**All three payoff criteria fire. The degeneracy criterion does not.** The rule
is well-behaved and its proposal is a real, verified half of the later task's
primitive that genuinely shortens the required program from 9 nodes to 7 — and
that is not enough to beat a wrong module of the same size, because seven nodes
does not fit a three-node scaffold and does not make a nine-node scaffold
searchable.

**Nothing was re-tuned after seeing this.** The parameter sweep in §9 and the
tie-break sweep in §8.3 were both planned as reported-regardless robustness
checks, and both are reported in full.

---

## 11. What this closes, and what it does not

**Closed.** The loop now has no human in it: a rule reads solved programs,
proposes a subprogram from the typed program graph alone, publishes it to a real
`tcn.library.Library` with a conformance fixture and provenance, and a later
task loads it under `policy="strict"` and offers it as an ordinary typed
candidate. Every step of experience → program → proposal → crystallize →
publish → inherit → select runs end to end without anyone naming the module.
The rule is generic: node sets, edges, operator contracts and
`Program.description_bits`, no pixels, no language, no computer.

**Not closed.** The last link — *demonstrably reduces search or sample
complexity* — does not hold on this later task. Two things have to change
before it can, and both are now measurable rather than speculative:

1. **The solver has to leave abstractions intact.** A corpus of minimal circuits
   is close to the worst case for library learning. What is needed is either a
   solver that keeps modular structure, or a rule that mines *semantic* reuse
   (fragments computing the same function under different circuits) rather than
   the *structural* reuse mined here. `t2`, `t3` and `t5` all compute
   `maj(a,b,c)` somewhere in their algebra; none of them has a node holding it.
2. **The acquisition path has to work at all.** The shipped gradient synthesizer
   solved 1 of 6 four-input, five-gate earlier tasks across 8, 8 and 16 seeds
   and 900–3 000 steps, and the one it solved is the one that collapses to 3
   gates. `enumerate_fit` cannot reach these tasks either — 1.94·10¹⁰ programs
   at five nodes. Until earlier tasks get solved by the system, a closed loop has
   nothing to mine.

A third, smaller point worth recording: on this family the two positive-saving
abstractions are the two halves of the same primitive, and they are exactly
equally useful downstream. A rule that ranks by corpus MDL has no way to prefer
"half a primitive that composes into something the next task needs" over "the
other half". Corpus MDL is a statement about the past; usefulness is a statement
about the future, and nothing in the current objective connects them — the same
gap §12 found on the preference side.

---

## 12. Reproduction and provenance

* **Nothing in `tcn/` or `generators/` was modified.** No core hook was needed.
  `git status` for this branch touches only `research/earned-abstraction/`.
* **Test suite: 287 passed, 1 failed.** The failure is
  `tests/test_panel_interface.py::test_panel_episode_replays_and_restores`, the
  documented environmental failure in a worktree with `node_modules` symlinked
  from the main checkout. It is on main's own code and is not this track's.
* **The shipped fixture reproduces.** `python -m tcn train --episodes 160` gives
  initial 0.248836 → final 0.002231, `fully_frozen: true`, frozen evaluation
  mean return 4.0 (`out_fixture.log`).
* **Recorded numbers checked against their sources, all reproducing:**
  MAJ3's minimum of 4 gates and its exact minimal program from
  `recursive-abstraction-retest/min_program.json`; the tight-scaffold space of
  230 400 exhausted with no solution and 2 709 504 exhausted with 144 conforming
  from that track's `enumeration.json`; and its wide-scaffold 0/8 flat against
  8/8 with the module. Nothing failed to reproduce.
* Every figure in this document comes from a file in `out/`; `tables.py`
  regenerates the tables in §6 and §7 directly from those files.
