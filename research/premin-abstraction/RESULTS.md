# Pre-minimisation abstraction — mining from a corpus that was not (only) minimised

FINDINGS §44 (`research/earned-abstraction`) closed the abstraction loop
mechanically, measured that it does not pay, and located the cause: **exact
minimisation is adversarial to abstraction mining.** Each earlier task's
minimum-gate program factors differently, so `MAJ3` — the fragment that would
have paid — is fused into its wrapper in five of six solved programs, and the
frequency statistics the rule reads never see it.

This track tests the follow-on hypothesis: *if the corpus is not minimised — or
not only minimised — the reusable abstraction survives in enough programs for a
frequency rule to find it.*

[`PREREGISTRATION.md`](PREREGISTRATION.md) was written and committed **before**
any arm was run. Nothing in `tcn/` or `generators/` is modified — no core hook
was needed. The later task, both scaffolds, the arms and the rule's R1/R2/
rewriting are imported from `research/earned-abstraction`, not restated, so
every number below is directly comparable to §44's.

Reproduce from this directory with
`PYTHONPATH=<repo>:.:../earned-abstraction <repo>/.venv/bin/python <script>`.

---

## VERDICT

**The premise of the hypothesis is right and its conclusion is wrong.** A
corpus that is not only minimised does retain the abstraction — retention rises
from **1 of 6 programs and 1 of 6 tasks** to **134 of 296 programs and 5 of 6
tasks** — and yet the rank-1 proposal does not move by one bit. It is the same
two-node fragment, with the same content digest `ec516b3808dd…`, that §44
mined from the minimised corpus.

1. **The rank-1 proposal is unchanged.** `C-minall`, `C-plus1` and `C-trace`
   all propose `M(x0,x1,x2) = x2 ∨ (x0 ∧ x1)` — byte-identical to §44's
   published module, digest `ec516b3808ddef7e7d0e7d22`. `arm2p_trace` and
   `arm2_earned` therefore load *the same module through two different
   libraries*, and the enumerations agree exactly: **0 conforming out of
   2 709 504, exhausted, certificate `complete`**, both.

2. **All three pre-registered payoff criteria fire, as they did in §44.**
   `arm2p_trace` ties `arm2_earned`; ties `arm4_wrong_authored` and
   `arm4b_wrong_mined`; and loses to `arm3_authored` 0 vs 144 conforming.

3. **The corpus is no longer the binding constraint. The ranking is.** This is
   the new result and it is measured, not argued: an **exploratory** arm
   (declared as unregistered) inheriting `C-trace`'s **rank-18** proposal — a
   four-node majority abstraction the rule did mine, from five of six tasks —
   finds **144 conforming out of 2 709 504, exhausted, `complete`**: *exactly
   the hand-authored ceiling*. The abstraction that pays is now in the corpus,
   is legal under the rule's own R1/R2, and is ranked 18th of 182.

4. **Why the ranking still misses it, measured two ways.** First, corpus MDL
   rewards breadth over depth: the majority abstraction is worth **5 048 bits
   per corpus program** against the winner's **1 965**, but appears in **16**
   programs against **140**, so it loses on the product. Second, and this is
   new, the majority's mass is **shattered across canonical identities**: 134
   corpus programs offer a legal majority fragment, but they realise it as
   **58 distinct `Program.digest`s**, the largest carrying 16 programs. A
   frequency rule ranks digests, not functions, so a function realised 58 ways
   competes at one fifty-eighth of its own frequency.

5. **Non-minimality specifically made things worse, not better.** The `C-plus1`
   band — every conforming program at one gate above the minimum — raises the
   *number* of programs holding a majority-valued node but drops the largest
   single majority digest from **15 programs** (`C-minall`) to **2**, across
   **54** digests. A spare gate is not spent on a cleaner majority; it is spent
   on a *different* majority circuit, so each extra program brings a new digest
   instead of reinforcing an existing one. §44's headline should be
   sharpened: it is not minimisation that hides the abstraction, it is
   **committing to one program per task**; and going *past* the minimum
   fragments the abstraction's identity instead of consolidating it.

6. **The richer corpus cost 7.6× the search and 12.9× the CPU for no change in
   the proposal** — and the part that actually helped, `C-minall`, cost **less
   than §44's own corpus build** (11.5 M DFS nodes and 21 CPU-s against
   110.7 M and 136 CPU-s). The expensive half is the half that hurt.

7. **Exploratory, outside the pre-registration, and the most actionable thing
   here (§10).** Change *only* the identity relation — pool occurrences by
   (arity, truth table) instead of by `Program.digest`, everything else in the
   rule imported unchanged — and on `C-trace` majority becomes **rank 1 of
   112** at **+860 960 bits**, ahead of `M`'s +323 272; its arm gives **144
   conforming, 18/24 tight, 8/8 wide**, equal to the hand-authored ceiling on
   every measured axis, with no human choosing the module. **It needs both
   changes**: on §44's own one-program-per-task corpus, semantic identity
   changes nothing, because majority occurs in one task and never clears the
   reuse gate. This was run after the pre-registered arms and is counted toward
   no criterion; it should be re-run pre-registered on another family before it
   is claimed as more than a measurement of this one.

---

## 1. What was fixed in advance, and what reproduced

Everything in [`PREREGISTRATION.md`](PREREGISTRATION.md) was honoured. Nothing
was re-tuned; the one parameter sweep was declared in advance as
reported-regardless and is in §7.

Two §44 numbers were nominated as reproduction checks. Both reproduce here, on
this branch, with this track's own machinery:

| check | §44 | here |
|---|---|---|
| `arm1_none`, tight scaffold | 230 400 exhausted, 0 conforming, `complete` | **230 400 exhausted, 0 conforming, `complete`** |
| `arm2_earned`, tight scaffold | 2 709 504 exhausted, 0 conforming, `complete` | **2 709 504 exhausted, 0 conforming, `complete`** |
| `arm3_authored`, tight scaffold | 2 709 504 exhausted, **144** conforming, `complete` | **2 709 504 exhausted, 144 conforming, `complete`** |
| certified minima `t1..t6` | 5, 5, 5, 5, 5, 3 | **5, 5, 5, 5, 5, 3** (re-derived independently, `out/min_certificate.json`) |
| `C-min` ranked table | 13 eligible, rank 1 = `M`, +3 648 bits | **13 eligible, rank 1 = `M`, +3 648 bits** |
| FINDINGS §44 median accuracies | 0.6250 / 0.7812 / 1.0000 / 0.6250 / 0.7812 | **identical**, re-derived from that track's raw records |
| §44's retention check on its own corpus | `MAJ3` in **1 of 6**, `M` in **4 of 6** | **1 of 6 and 4 of 6**, from §44's stored programs with this track's instrument |
| §44's tight/wide gradient tables | 0/24, 0/24, 18/24, 0/24, 0/24 and 0/8, 0/8, 8/8, 0/8, 0/8 | **identical on every arm and every column** |

Nothing failed to reproduce. The reproduction of `arm2_earned` at 0 conforming
is the gate the brief set for continuing, and it passed.

**Corpus definitions tried.** All five were declared in the pre-registration
before any was built, and all five are reported above. **None was changed,
added or abandoned after the fact.** The declared sampler fallback was never
needed — every band exhausted inside the budget — and the declared cap of 32
per task per band bound only the two `k_t + 1` bands.

---

## 2. The corpora

All five variants were declared before any was built. Every program is
*pruned* (every gate but the last is read by a later gate, and the last gate is
the output) and is rebuilt as a `tcn.graph.Program` and executed through `tcn`
against the full 16-row truth table before it enters a corpus.

| corpus | definition | entries | build |
|---|---|---|---|
| `C-min` | §44's: one program per task, first minimum in that track's enumeration order | 6 | loaded from §44's artifact |
| `C-minall` | **every** pruned program at the certified minimum length `k_t` | 130 | exhaustive |
| `C-plus1` | **every** pruned program at length `k_t + 1` | 166 | exhaustive |
| `C-trace` | `C-minall ∪ C-plus1` — **the primary non-minimised corpus, feeds `arm2p_trace`** | 296 | exhaustive |
| `C-plus1-one` | one program per task, first of `C-plus1` in enumeration order | 6 | exhaustive |

**Every band was exhausted; the declared sampler was never needed.** The
30-minute-per-task fallback in the pre-registration did not fire — the slowest
band took 365 s.

Uncapped counts, and the declared cap of 32 per task per band:

| task | `k_t` | minima found | at `k_t+1` found | kept (min / plus1) |
|---|---|---|---|---|
| `t1_maj_abc_xor_d` | 5 | 27 | 1 119 | 27 / 32 |
| `t2_maj_abc_and_d` | 5 | 21 | 1 233 | 21 / 32 |
| `t3_maj_bcd_or_a` | 5 | 27 | 1 581 | 27 / 32 |
| `t4_maj_acd_xor_b` | 5 | 27 | 1 119 | 27 / 32 |
| `t5_maj_abd_or_c` | 5 | 27 | 1 581 | 27 / 32 |
| `t6_maj_abc_xor_maj_bcd` | 3 | 1 | 6 | 1 / 6 |

Where the enumeration exceeded the cap, the retained subset is the declared
deterministic subsample (sort by the sha256 of the canonical key, take 32),
which is independent of enumeration order. `C-min`'s own program is present in
its task's full minimum-length set in **all six** cases, as it must be.

**The first thing this table says on its own:** §44's corpus was one draw from
a set of 21–27 equally minimal programs per task. The tie-break, not
minimisation, is doing most of the work in §44's `1 of 6`.

---

## 3. Retention — the mechanism

Computed the fair way, over **all 24 ordered 3-subsets** of the four inputs.
The naive check against `(a,b,c)` only is unfair to `t3_maj_bcd`,
`t4_maj_acd` and `t5_maj_abd`, each of which takes a majority of a different
triple; done fairly, every task's own triple is the one that fires.

### 3.1 Value retention — does a node hold the majority at all?

Over the **full** enumerated sets, not the capped ones:

| task | minima with a `MAJ3` body | at `k_t+1` | minima with an `M` body | at `k_t+1` |
|---|---|---|---|---|
| `t1_maj_abc_xor_d` | 15 / 27 (0.556) | 441 / 1 119 (0.394) | 3 / 27 (0.111) | 108 / 1 119 (0.097) |
| `t2_maj_abc_and_d` | 15 / 21 (0.714) | 450 / 1 233 (0.365) | 9 / 21 (0.429) | 345 / 1 233 (0.280) |
| `t3_maj_bcd_or_a` | 15 / 27 (0.556) | 513 / 1 581 (0.324) | 9 / 27 (0.333) | 543 / 1 581 (0.343) |
| `t4_maj_acd_xor_b` | 15 / 27 (0.556) | 441 / 1 119 (0.394) | 3 / 27 (0.111) | 108 / 1 119 (0.097) |
| `t5_maj_abd_or_c` | 15 / 27 (0.556) | 513 / 1 581 (0.324) | 9 / 27 (0.333) | 543 / 1 581 (0.343) |
| `t6_maj_abc_xor_maj_bcd` | 0 / 1 | 0 / 6 | 0 / 1 | 0 / 6 |

Rolled up to the corpus, over the programs that were actually mined (the capped
sets) and over the full enumerations:

| corpus | programs mined | `MAJ3` | `M` | `M'` | | programs enumerated | `MAJ3` | `M` | `M'` |
|---|---|---|---|---|---|---|---|---|---|
| `C-min` | 6 | **1** (0.167) | 4 (0.667) | 3 (0.500) | | 6 | 1 | 4 | 3 |
| `C-minall` | 130 | **75** (0.577) | 33 (0.254) | 30 (0.231) | | 130 | 75 (0.577) | 33 (0.254) | 30 (0.231) |
| `C-plus1` | 166 | 61 (0.367) | 43 (0.259) | 35 (0.211) | | 6 639 | 2 358 (0.355) | 1 647 (0.248) | 1 596 (0.240) |
| `C-trace` | 296 | **136** (0.459) | 76 (0.257) | 65 (0.220) | | 6 769 | 2 433 (0.359) | 1 680 (0.248) | 1 626 (0.240) |
| `C-plus1-one` | 6 | 1 (0.167) | 3 (0.500) | 1 (0.167) | | 6 | 1 | 3 | 1 |

`C-min`'s row — **1 of 6 for `MAJ3`, 4 of 6 for `M`** — is §44's independently
verified figure, reproduced here from that track's stored programs with this
track's own instrument. The capped and uncapped fractions agree to within 0.012
on `C-plus1`, which is the check that the declared subsample is not biased.

**A limitation of this measurement, stated because §44 has it too.** The value
check asks whether a node's value equals the fragment over an ordered 3-subset
of the four *inputs*. A fragment's holes need not be inputs — they can be
internal ports — so this **undercounts**: the rule finds its rank-1 fragment `M`
at 140 of `C-trace`'s 296 entries while the value check sees it at 76, the
difference being occurrences whose arguments are other gates. The two
measurements are therefore not nested in either direction, and §3.2's
structural count is the one that governs the outcome.

**§44's `1 of 6` is a property of the tie-break, not of minimisation.** Every
one of the five non-collapsing tasks has a *minimum-length* program that holds
the majority at a node — 15 of them, in each case — and §44's enumeration order
happened to return one for `t1` and not for the others. `t6` retains nothing
because it collapses to three gates, exactly as §44 recorded.

### 3.2 Structural retention — can the rule actually *rank* it?

A frequency rule ranks `Program.digest`s, not functions. This is the number
that decides the outcome (`out/structural_retention.json`):

| corpus | entries | entries holding `maj` at a node | entries offering a **legal** `maj` fragment | tasks | **distinct `maj` digests** | largest digest: entries / tasks |
|---|---|---|---|---|---|---|
| `C-min` | 6 | 1 | 1 | 1 | 1 | 1 / 1 |
| `C-minall` | 130 | 75 | 75 | **5** | **8** | **15 / 5** |
| `C-plus1` | 166 | 61 | 59 | 5 | **54** | **2 / 2** |
| `C-trace` | 296 | 136 | 134 | **5** | **58** | **16 / 5** |
| `C-plus1-one` | 6 | 1 | 1 | 1 | 1 | 1 / 1 |

Three things to read off this table.

* **The abstraction is now in the corpus.** From 1 program and 1 task to 134
  programs and 5 tasks, and every one of those 134 offers it as a legal,
  single-exit, ≤5-node, ≤4-hole fragment — something the rule enumerates and
  scores. This is exactly what the hypothesis predicted.
* **Its identity is shattered.** 134 programs, **58** different canonical
  digests. The rule's own R2 says two occurrences are the same abstraction
  exactly when the content-addressed store would give them one file; 58 files
  is 58 competitors, each with a fraction of the frequency.
* **Non-minimality is the shattering half.** `C-minall` splits the majority
  across 8 digests with a largest of 15 entries; adding the `k_t+1` band takes
  it to 58 digests, and the `k_t+1` band *on its own* is the worst of all —
  **54 digests, largest 2**. A spare gate does not get spent on a cleaner
  majority; it gets spent on a *different* majority circuit, so the extra
  programs each carry a new digest instead of reinforcing an existing one.
  (Single-exit breakage is a second, much smaller effect: 2 of the 61
  majority-holding programs in `C-plus1`, and 2 of 136 in `C-trace`, hold the
  value at a node whose sub-DAG is read from outside and so cannot be replaced
  by one call.)

---

## 4. What the rule proposed, per corpus

`mine_multi.propose`, `MAX_NODES = 5`, `MAX_HOLES = 4`, `MIN_TASKS = 2`. R1,
R2 and the rewriting are §44's code, imported. The two declared changes are
R3 counting base tasks and R4 summing over corpus entries.

| corpus | entries | eligible | rank-1 body | arity | tasks | entries | sites | saving (bits) | rank-1 = `MAJ3`? | best `MAJ3` rank |
|---|---|---|---|---|---|---|---|---|---|---|
| `C-min` | 6 | 13 | `n0=and(x0,x1); n1=or(x2,n0)` | 3 | 4 | 4 | 6 | +3 648 | no | — (ineligible) |
| `C-minall` | 130 | 94 | **the same fragment** | 3 | 5 | 75 | 89 | +137 776 | no | **11 of 94** |
| `C-plus1` | 166 | 137 | **the same fragment** | 3 | 5 | 65 | 85 | +131 312 | no | **65 of 137** |
| `C-trace` | 296 | 182 | **the same fragment** | 3 | 5 | 140 | 174 | +275 136 | no | **18 of 182** |
| `C-plus1-one` | 6 | 13 | `n0=or(x0,x1); n1=and(n0,x2); n2=or(x3,n1)` | 4 | 3 | 3 | 3 | +2 080 | no | — (ineligible) |

`C-min` reproduces §44's ranked table exactly, which is the check that the two
declared rule changes are inert on a one-program-per-task corpus.

**Three corpora, one digest.** `C-minall`, `C-plus1` and `C-trace` all publish
`module:ec516b3808ddef7e7d0e7d22` — bit-for-bit §44's `earned@1`. `arm2p_trace`
is not merely *equivalent* to `arm2_earned`; it is the same content-addressed
module reached through a second library. The full module identity map, printed
from the arms themselves by `check_arms.py`:

| arm | module operator | nodes |
|---|---|---|
| `arm1_none` | — | — |
| `arm2_earned` | `module:ec516b3808ddef7e7d0e7d22` | 2 |
| `arm2p_trace`, `arm2p_minall`, `arm2p_plus1` | `module:ec516b3808ddef7e7d0e7d22` | 2 |
| `arm3_authored` | `module:8ceedf7b792a7116514f90ab` | 4 |
| `arm4_wrong_authored` | `module:dc59f82e89cd0ad1797ef375` | 4 |
| `arm4b_wrong_mined` | `module:b22e2ef14a32a68f1224304f` | 2 |
| `arm2p_plus1one` | `module:26f930da7b619fd563401a8c` | 3 |
| *`arm3p_mined_maj3`*, *`arm2s_semantic_trace`* | `module:165bc290d9c82b70a8ea3cc2` | 4 |

Every mined module is loaded from a real `tcn.library.Library` on disk under
`policy="strict"`, with a manifest, a module file and a recorded conformance
fixture — the module crosses from the earlier tasks to the later one through the
store, not through a Python variable.

### 4.1 The top of `C-trace`, with every majority row it contains

| rank | body | arity | tasks | entries | saving | per entry | `MAJ3`? |
|---|---|---|---|---|---|---|---|
| 1 | `and(x0,x1); or(x2,n0)` | 3 | 5 | 140 | +275 136 | 1 965 | |
| 2 | `xor(x0,x1); and(x2,n0)` | 3 | 6 | 148 | +255 408 | 1 726 | |
| 3 | `and(x0,x1); xor(x2,n0)` | 3 | 6 | 116 | +195 320 | 1 684 | |
| 4 | `xor(x0,x1); and(x2,n0); xor(x3,n1)` | 4 | 5 | 57 | +189 600 | 3 326 | |
| 5 | `or(x0,x1); and(x2,n0); or(x3,n1)` | 4 | 5 | 57 | +188 704 | 3 311 | |
| … | | | | | | | |
| **18** | `and(x0,x1); xor(x0,x1); and(x2,n1); or(n0,n2)` | 3 | **5** | 16 | **+80 760** | **5 048** | **YES** |
| 19 | `and(x0,x1); or(x0,x1); and(x2,n1); or(n0,n2)` | 3 | 5 | 16 | +80 640 | 5 040 | YES |
| 22 | `and(x0,x1); xor(x0,x1); and(x2,n1); xor(n0,n2)` | 3 | 5 | 15 | +75 184 | 5 012 | YES |
| 31 | `and(x0,x1); or(x0,x1); or(x2,n0); and(n1,n2)` | 3 | 5 | 11 | +52 280 | 4 753 | YES |
| 74 | `and(x0,x1); or(x2,n0); or(x0,x1); and(n1,n2)` | 3 | 5 | 5 | +18 240 | 3 648 | YES |

Row 74 is §44's hand-authored `MAJ3` body, mined. It is in the corpus, eligible,
positive, and ranked 74th.

**The failure is legible in one comparison.** Rank 18 is worth **2.6× more per
corpus program** than rank 1 and appears in **8.75× fewer** of them. Corpus MDL
multiplies the two, and breadth wins. The **12** majority digests that clear the
reuse gate sum to 87 entries between them — still fewer than rank 1's 140, and
split twelve ways. The other **46** of the 58 majority digests in the corpus
occur in a single task and are never even eligible.

---

## 5. The arms — exhaustive enumeration, with certificates

Tight scaffold: three `BOOL` nodes over six inputs, §44's construction
unchanged. The flat minimum for the target is proved ≥ 7, so no flat program
fits. `tcn.search.enumerate_fit`, tolerance 1e-3, `rank='order'`, full sweep so
the certificate survives. A conforming program is exact on all 64 rows; the
best constant and a uniform random predictor both score 0.5.

| arm | library | candidates/node | search space | evaluated | exhausted | conforming | certificate | live nodes | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| `arm1_none` | none | 120/120/16 | 230 400 | 230 400 | yes | **0** | `complete` | — | 4 |
| `arm2_earned` | §44's module, mined from **`C-min`** | 336/336/24 | 2 709 504 | 2 709 504 | yes | **0** | `complete` | — | 204 |
| **`arm2p_trace`** | **the module mined from `C-trace`** | 336/336/24 | 2 709 504 | 2 709 504 | yes | **0** | `complete` | — | 188 |
| `arm3_authored` | hand-authored `MAJ3` | 336/336/24 | 2 709 504 | 2 709 504 | yes | **144** | `complete` | 3 | 190 |
| `arm4_wrong_authored` | truth table 134, hand-authored | 336/336/24 | 2 709 504 | 2 709 504 | yes | **0** | `complete` | — | 159 |
| `arm4b_wrong_mined` | §44's runner-up | 336/336/24 | 2 709 504 | 2 709 504 | yes | **0** | `complete` | — | 216 |
| `arm2p_plus1one` | the module mined from `C-plus1-one` (arity 4) | 1416/1416/32 | **64 161 792** | 64 161 792 | yes | **0** | `complete` | — | 8 185 |
| *`arm3p_mined_maj3`* (exploratory) | `C-trace`'s **rank-18** majority abstraction | 336/336/24 | 2 709 504 | 2 709 504 | yes | **144** | `complete` | 3 | 363 |
| *`arm2s_semantic_trace`* (exploratory, §10) | rank-**1** of the semantic-identity rule on `C-trace` — the same digest | 336/336/24 | 2 709 504 | 2 709 504 | yes | **144** | `complete` | 3 | 277 |

`arm2p_minall` and `arm2p_plus1` are **not** separate rows because they are not
separate arms: `C-minall` and `C-plus1` publish the *same content digest* as
`C-trace`, so under the pre-registration ("if the rank-1 proposal … differs
from `C-trace`'s **and** from §44's") no extra arm is owed. That agreement is
itself the result.

`arm2p_plus1one` is the one pre-registered secondary arm that *is* owed: a
different digest, and a 4-ary one, which makes its call-site candidate block
`6**4` wide and its tight space **64 161 792** programs — 24× every other arm.
It still got a full sweep, with the cap raised to `2**27` so the certificate
survives: **64 161 792 evaluated, exhausted, `complete`, 0 conforming**, at a
cost of 8 185 s against the other arms' 159–363 s. This is `MAX_HOLES` earning
its keep — an abstraction the search cannot afford to offer is not a useful
abstraction — and it is the only arm in this track that mining made *worse* on
both axes at once: 40× the search for a lower mean accuracy than the module it
replaced.

Three readings:

* **`arm1_none`, `arm2_earned` and `arm3_authored` reproduce §44 exactly**, to
  the program: arm 3's first conforming program in enumeration order is
  `module(a,b,c) xor module(d,e,f)`, 3 live nodes, 19 496 description bits with
  the library charged, execution cost 9.0 — every figure matching that track's
  `enumeration.json`.
* **`arm2p_trace` ties `arm2_earned` exactly, because it *is* `arm2_earned`.**
  Same digest, different library. The first pre-registered falsification
  condition does not merely fire, it fires by identity.
* **The exploratory arm reaches the ceiling.** `arm3p_mined_maj3` inherits a
  module the rule *mined from the corpus*, four nodes, a different circuit from
  the hand-authored one (`and(x0,x1); xor(x0,x1); and(x2,n1); or(n0,n2)` against
  the demonstration's `and(a,b); or(a,b); or(c,g0); and(g1,g2)`), and lands on
  **144 conforming, exhausted, `complete`** — the same 144, the same three-node
  route, 19 568 description bits against 19 496 and the same execution cost 9.0.
  Every one of the 144 has the module live on the output path, by §44's
  argument: the flat sub-space of this scaffold is exactly `arm1_none`'s 230 400
  programs and it is exhausted with none conforming.

  This arm was **not pre-registered**. It was added after the ranked table was
  read, it is labelled exploratory everywhere it appears, and it is not counted
  toward any falsification criterion. Its only job is to separate two
  explanations that §44 could not separate: *the corpus does not contain the
  abstraction* versus *the ranking does not pick it*. After this track it is
  the second.

## 6. Sample complexity — the same task learned instead of enumerated

Identical scaffold, examples, signals, objective, schedule and seeds in every
arm; the library is the only difference. `SoftProgram` + Adam, lr 0.05, 400
steps, conformance of the argmax export checked every 10 steps. Best constant
and uniform random both score **0.5000** exact accuracy.

### 6.1 Tight scaffold (3 nodes), 24 seeds

| arm | solved | median steps | **median accuracy** | mean | best | constant baseline | module on output path |
|---|---|---|---|---|---|---|---|
| `arm1_none` | 0/24 | — | 0.6250 | 0.6250 | 0.6250 | 0.5000 | 0 |
| `arm2_earned` | 0/24 | — | 0.7812 | 0.6875 | 0.7812 | 0.5000 | 0 |
| **`arm2p_trace`** | **0/24** | — | **0.7812** | **0.6875** | 0.7812 | 0.5000 | 0 |
| `arm3_authored` | **18/24** | **50** | **1.0000** | 0.8750 | 1.0000 | 0.5000 | 18 |
| `arm4_wrong_authored` | 0/24 | — | 0.6250 | 0.6146 | 0.6250 | 0.5000 | 0 |
| `arm4b_wrong_mined` | 0/24 | — | 0.7812 | 0.7161 | 0.7812 | 0.5000 | 0 |
| `arm2p_minall` | 0/24 | — | 0.7812 | 0.6875 | 0.7812 | 0.5000 | 0 |
| `arm2p_plus1` | 0/24 | — | 0.7812 | 0.6875 | 0.7812 | 0.5000 | 0 |
| `arm2p_plus1one` | 0/24 | — | 0.7812 | 0.6641 | 0.7812 | 0.5000 | 0 |
| *`arm3p_mined_maj3`* (exploratory) | **18/24** | **50** | **1.0000** | 0.8750 | 1.0000 | 0.5000 | 18 |
| *`arm2s_semantic_trace`* (exploratory, §10) | **18/24** | **50** | **1.0000** | 0.8750 | 1.0000 | 0.5000 | 18 |

`arm1_none`, `arm2_earned`, `arm3_authored`, `arm4_wrong_authored` and
`arm4b_wrong_mined` reproduce §44's tight table to four decimals on every
column, including the medians FINDINGS §44 quotes.

**`arm2p_trace` is identical to `arm2_earned` on every figure**, as it must be:
same module, same scaffold, same seeds. `arm2p_minall` and `arm2p_plus1` are
the same module again and give the same row a third and fourth time. The only
mined arm that moves at all is `arm2p_plus1one`, and it moves *down* — mean
0.6641 against 0.6875, in a space 24× larger.

**And the exploratory arm reaches the ceiling here too**: 18 of 24 seeds, median
50 steps to first conforming export, median accuracy 1.0000 — every figure equal
to the hand-authored arm's, from a module the rule mined.

### 6.2 Wide scaffold (9 nodes), 8 seeds — both routes fit

The tight scaffold cannot hold a flat solution at all (flat minimum proved ≥ 7
against three nodes), so it cannot separate "the module helped" from "nothing
fits". The wide scaffold is §44's non-degenerate comparison: nine nodes, wide
enough for the verified 9-gate flat program, far too large to exhaust.

| arm | solved | median steps | **median accuracy** | mean | best | constant baseline | module on output path |
|---|---|---|---|---|---|---|---|
| `arm1_none` | 0/8 | — | 0.6875 | 0.6875 | 0.6875 | 0.5000 | 0 |
| `arm2_earned` | 0/8 | — | 0.7812 | 0.7930 | 0.8750 | 0.5000 | 0 |
| **`arm2p_trace`** | **0/8** | — | **0.7812** | **0.7930** | 0.8750 | 0.5000 | 0 |
| `arm3_authored` | **8/8** | **90** | **1.0000** | 1.0000 | 1.0000 | 0.5000 | 8 |
| `arm4_wrong_authored` | 0/8 | — | 0.6875 | 0.6641 | 0.6875 | 0.5000 | 0 |
| `arm4b_wrong_mined` | 0/8 | — | 0.7812 | 0.7930 | 0.8750 | 0.5000 | 0 |
| `arm2p_minall` | 0/8 | — | 0.7812 | 0.7930 | 0.8750 | 0.5000 | 0 |
| `arm2p_plus1` | 0/8 | — | 0.7812 | 0.7930 | 0.8750 | 0.5000 | 0 |
| `arm2p_plus1one` | 0/8 | — | 0.7812 | 0.7812 | 0.7812 | 0.5000 | 0 |
| *`arm3p_mined_maj3`* (exploratory) | **8/8** | **90** | **1.0000** | 1.0000 | 1.0000 | 0.5000 | 8 |
| *`arm2s_semantic_trace`* (exploratory, §10) | **8/8** | **90** | **1.0000** | 1.0000 | 1.0000 | 0.5000 | 8 |

Every §44 row reproduces to four decimals, including the tie §44 called its
sharpest single number: `arm2_earned` and `arm4b_wrong_mined` at 0.7930 mean
and 0.8750 best. **`arm2p_trace` joins that tie exactly** — the earned module
from a corpus 49× larger is indistinguishable from a *wrong* module of the same
size, on the same four decimals. That is the second pre-registered
falsification condition, unchanged from §44.

`arm2p_plus1one` is the only mined arm with a different module, and it is the
worst of them: 0.7812 on every seed, below `arm2_earned`'s 0.7930 mean, in a
scaffold whose candidate lists it inflates by a factor of twelve.

<!--GRAD_WIDE-->

## 7. Sensitivity of the rule to its own parameters

Declared in the pre-registration as a reported-regardless robustness check, run
once, reported whatever the headline says. 40 settings per corpus —
`MAX_NODES ∈ {2,3,4,5,6}`, `MAX_HOLES ∈ {2,3,4,5}`, `MIN_TASKS ∈ {2,3}` —
over the two corpora that matter (`out/sensitivity.json`).

| corpus | rank-1 digest | settings | what it is | saving |
|---|---|---|---|---|
| `C-trace` | `ec516b3808dd` | **30 of 40** (`MAX_HOLES` 3,4,5; every `MAX_NODES`; both `MIN_TASKS`) | 2 nodes, arity 3 — §44's module | +275 136 |
| `C-trace` | `90cc9f2a0ef2` | 10 of 40 (all `MAX_HOLES = 2`) | 2 nodes, arity 2, `not(x0); xor(x1,n0)` | +4 088 |
| `C-minall` | `ec516b3808dd` | **30 of 40** (same pattern) | the same module | +137 776 |
| `C-minall` | `5e50f17066eb` | 10 of 40 (all `MAX_HOLES = 2`) | 1 node, `or(x0,x1)` | **−44 864** |

* **In none of the 80 settings does the rank-1 proposal compute majority.**
* The 10 settings per corpus that move the answer are all `MAX_HOLES = 2`,
  which admits only very small abstractions; on `C-minall` the best available
  there is *negative*, so the rule is reporting — correctly — that nothing in
  reach is worth abstracting, exactly as §44 recorded at the same setting.
* `MAX_NODES` never changes the answer between 2 and 6, and the best majority
  rank moves only between 8 and 19 on `C-trace` and 5 and 11 on `C-minall`; it
  is `None` in the 22 settings where a four-node body is out of reach by
  construction (`MAX_NODES < 4`, or `MAX_HOLES = 2`).

The negative is not a parameter choice, and it is not a knife edge.

## 8. What the richer corpus cost

Reported because the brief asks for it, and because it is the part of this
result that generalises to any system that would do this for real. A rule that
needs 100× the search to find a module that saves 10× is not a win.

| corpus | DFS nodes expanded | CPU-seconds | wall (6 tasks in parallel) | rank-1 |
|---|---|---|---|---|
| `C-min` (§44's own build) | 110 744 802 | 136.2 | 99.1 s | `ec516b3808dd` |
| `C-minall` band | **11 536 818** | **21.1** | 4.5 s | `ec516b3808dd` |
| `C-plus1` band | 832 575 132 | 1 730.0 | 365.1 s | `ec516b3808dd` |
| `C-trace` (both bands) | 844 111 950 | 1 751.2 | 369.9 s | `ec516b3808dd` |
| ratio, `C-trace` : §44 | **7.6×** | **12.9×** | 3.7× | unchanged |

Mining is free by comparison: 0.0 s for the 6-entry corpora, 0.5 s for
`C-minall`, 1.2 s for `C-trace`'s 296 entries. Re-deriving the minimality
certificates independently cost a further 11.7 M nodes and 58.6 CPU-s.

Two things worth saying plainly.

* **The half that helped was cheaper than §44's own corpus.** Getting *all* the
  minima instead of one costs **11.5 M DFS nodes and 21 CPU-s**, against
  §44's 110.7 M and 136 CPU-s for its single-program corpus — an order of
  magnitude *less*, because enumerating a fixed length with a completions
  shortcut at the last level replaces iterative deepening's generate-and-test.
  Removing the tie-break is not expensive. It is cheaper than keeping it.
* **The half that cost 79× more is the half that hurt.** The `k_t+1` band is
  72× the `C-minall` band in DFS nodes and 82× in CPU, and it *lowers* the
  largest single majority digest from 15 entries to 2. On this family, paying
  more search to go past the minimum buys a worse corpus for mining.

## 9. Verdict against the pre-registered falsification criteria

The pre-registration named four conditions. Three fire, one does not, and all
four are read off the tables above rather than argued.

| pre-registered condition | fires? | evidence |
|---|---|---|
| **"The corpus was not the problem; the rule is"** — `arm2p_trace` ties `arm2_earned` | **yes, by identity** | Both load `module:ec516b3808ddef7e7d0e7d22`. 0 vs 0 conforming in the same exhausted 2 709 504-program space, certificate `complete` both; 0/24 vs 0/24 tight and 0/8 vs 0/8 wide; identical median, mean and best accuracy on both scaffolds. `C-minall` and `C-plus1` publish the same digest, so this holds for every non-minimised corpus tried. |
| **"Any module of that size helps and the selection contributes nothing"** — `arm2p_trace` ties a wrong module | **yes** | Ties `arm4_wrong_authored` 0 vs 0 conforming, exhausted. Ties `arm4b_wrong_mined` to four decimals on the wide scaffold (median 0.7812, mean 0.7930, best 0.8750 both) and on the tight scaffold (0.7812 / 0.6875 / 0.7812 vs 0.7812 / 0.7161 / 0.7812 — same median and best). |
| **"Mining does not reach what a human picks, even from a richer corpus"** — `arm2p_trace` loses badly to `arm3_authored` | **yes** | 0 vs 144 conforming in an exhausted space; 0/24 vs 18/24 tight; 0/8 vs 8/8 wide; median accuracy 0.7812 vs 1.0000. |
| **"The rule is degenerate"** — rank-1 is a 1-node fragment, or nothing clears the reuse gate | **no** | 94–182 eligible abstractions on the non-minimised corpora; rank-1 is a 2-node, 3-ary fragment at the declared parameters. On `C-trace` the four single-gate fragments are ranked **179, 180, 181 and 182 of 182** — dead last — despite `xor` occurring at 614 sites and `and` at 534, because a one-gate call site *costs* 224–232 bits more than the inline gate no matter how many sites amortise the definition. The one place a 1-node fragment wins is `C-minall` at `MAX_HOLES = 2`, with a **negative** saving of −44 864: the rule correctly reporting that nothing in reach is worth abstracting, the same behaviour §44 recorded at that setting. |

**All three payoff criteria fire, exactly as in §44, on a corpus 49× larger
that demonstrably contains the abstraction.** The degeneracy criterion does not
fire, which matters: the rule is not broken and is not gaming its own score. It
is answering the question it was asked — *which subprogram is most reused, by
content digest, across this corpus* — and the answer to that question is not
the answer to *which subprogram will make the next task easy*.

Nothing was re-tuned after seeing this. The parameter sweep in §7 was declared
in advance as reported-regardless and is reported in full; the one arm added
after the fact (`arm3p_mined_maj3`) is labelled exploratory everywhere and is
counted toward no criterion.

<!--CRITERIA-->

## 10. Exploratory: the identity relation, not the corpus

**Not pre-registered.** Everything in this section was run after the tables
above were read; it is counted toward no falsification criterion and it does not
change any verdict. It is here because it converts §44's closing speculation
into a measurement.

§44 closed by asking for "a rule that mines *semantic* reuse (fragments
computing the same function under different circuits) rather than the structural
reuse mined here". `mine_semantic.py` is that rule and **only** that rule: R1,
R2's construction, the rewriting, R3's reuse gate and R4's MDL score are
imported unchanged. One thing differs — occurrences are pooled by
**(arity, truth table)** instead of by `Program.digest`, each pool elects a
representative circuit (fewest nodes, then digest), and every occurrence in the
pool is rewritten to a call to that representative. This is semantics-preserving
because a fragment's truth table is taken in its own hole order, and every
rewrite is still executed against the task's rows before it is scored.

| corpus | entries | eligible | rank-1 computes `MAJ3`? | rank-1 saving | circuits pooled at rank 1 | best `MAJ3` rank |
|---|---|---|---|---|---|---|
| `C-min` | 6 | 13 | **no** (still `M`) | +3 648 | 1 | — (ineligible) |
| `C-minall` | 130 | 40 | **YES** | +415 872 | 8 | **1** |
| `C-plus1` | 166 | 107 | **YES** | +434 960 | 54 | **1** |
| `C-trace` | 296 | 112 | **YES** | +860 960 | **58** | **1** |
| `C-plus1-one` | 6 | 9 | **no** | +4 064 | 3 | — (ineligible) |

The top of `C-trace` under semantic identity, with the structural winner for
comparison:

| rank | representative body | arity | circuits pooled | tasks | entries | saving | per entry |
|---|---|---|---|---|---|---|---|
| **1** | `and(x0,x1); xor(x0,x1); and(x2,n1); or(n0,n2)` — **`MAJ3`** | 3 | **58** | 5 | **134** | **+860 960** | 6 425 |
| 2 | a 5-node, 4-ary fragment | 4 | 13 | 2 | 53 | +386 152 | 7 286 |
| 3 | a 5-node, 4-ary fragment | 4 | 16 | 2 | 48 | +349 104 | 7 273 |
| **4** | `and(x0,x1); or(n0,x2)` — **§44's `M`** | 3 | 4 | 5 | 144 | +323 272 | 2 245 |

**It takes both changes, and neither alone.** On `C-min` — one minimal program
per task — semantic identity does *not* help: majority occurs in a single task,
so it never clears the reuse gate, and the rule proposes `M` at +3 648 exactly
as §44 did. On `C-trace` with structural identity the rule proposes `M` at
+275 136 with majority 18th. With **both** a tie-break-free corpus **and**
semantic identity, majority is rank 1 of 112 at +860 960, ahead of `M`'s
+323 272 — a 2.7× margin, from a fragment that was 3.4× behind before.

Two details worth recording.

* **The 58-way split is real, and part of it is not even about different
  circuits.** Of the 12 *eligible* majority digests on `C-trace`, ten are
  distinct circuits — and two of them (ranks 31 and 74, carrying 11 and 5
  entries) are **the same circuit written in two different topological orders**,
  which is §44's own hand-authored `MAJ3` body. R2's canonical form numbers
  nodes by the host program's node order and keeps each operator's argument
  order, so it is not canonical up to graph isomorphism (`check_digest_split.py`
  verifies this by grouping the digests under an order-free key). Ranks 72 and
  73 are a second such pair.
* **The arm confirms it, measured rather than inferred.** The semantic rule's
  rank-1 proposal publishes as `module:165bc290d9c82b70a8ea3cc2` — the *same
  content digest* as the exploratory `arm3p_mined_maj3` module. Run as its own
  arm anyway:

| arm | conforming / space | exhausted | certificate | tight | wide |
|---|---|---|---|---|---|
| `arm3_authored` (the ceiling) | 144 / 2 709 504 | yes | `complete` | 18/24, median 50 steps, median acc 1.0000 | 8/8, median 90, 1.0000 |
| *`arm2s_semantic_trace`* (exploratory) | **144 / 2 709 504** | yes | `complete` | **18/24, median 50 steps, median acc 1.0000** | **8/8, median 90, 1.0000** |

A rule that reads only solved programs, with **no human choosing the module**,
reaches the hand-authored ceiling on every measured axis — once its corpus keeps
every minimum instead of one and its identity relation is semantic instead of
structural. That is a positive result, and it is explicitly **outside** the
pre-registration: it was not among the arms declared in advance, it followed
from reading the ranked table, and it should be re-run as a pre-registered
experiment (on a family whose shared structure is not majority, and with the
later task fixed in advance again) before it is claimed as more than a
measurement of this family.

## 11. What this changes about §44, and what it does not

**§44's headline needs one word changed, and it is load-bearing.** §44 wrote
that exact minimisation is adversarial to abstraction mining because each
task's minimum-gate program factors differently. Measured over all 21–27
minimum-gate programs per task rather than one, that is not what happened: a
majority body survives in **15 of 27** minimum-gate programs of `t1`, **15 of
21** of `t2`, **15 of 27** of `t3`, `t4` and `t5`, and in 5 of the 6 tasks. What
§44 measured as "minimisation fuses the primitive" was in most part
**committing to a single minimal program per task** — the solver's enumeration
tie-break. §44's own §8.3 came close: sweeping six operator orders moved the
count from 1 to 2 tasks. Sweeping *all* minima moves it to 5.

**But §44's conclusion survives intact, for a different reason.** The
abstraction is now in the corpus, legal, eligible, positive, mined from five of
six tasks — and the rule still does not propose it, on any of the three
non-minimised corpora and under 80 parameter settings. Two mechanisms, both
measured here:

1. **Corpus MDL rewards breadth over depth.** The majority is worth 2.6× more
   per corpus program and appears in 8.75× fewer of them.
2. **Canonical identity shatters.** A function realised as 58 different
   circuits competes at a fraction of its own frequency, because R2's identity
   is the content digest — which is the right identity for a *store* and the
   wrong one for a *statistic*.

Mechanism 2 is new, and it is the one that would not go away by re-weighting
the score. It says a structural frequency rule of this shape has a ceiling that
has nothing to do with how the corpus was solved: **the more ways there are to
compute a useful function, the less any one of them looks reused.** §44's
closing paragraph asked for "a rule that mines *semantic* reuse (fragments
computing the same function under different circuits) rather than the
structural reuse mined here". §10 measures how much that is worth on this
family: **58 digests collapse to 1**, majority goes from rank 18 to **rank 1**
at +860 960 bits against `M`'s +323 272, and the arm goes from 0 conforming to
**144** — with 18/24 and 8/8 on the two gradient scaffolds, equal to the
hand-authored ceiling on every measured axis. That is exploratory and outside
the pre-registration, and it is the single most actionable thing in this track.

**Not closed.** Everything §44 listed as open stays open, and this adds one and
narrows another:

* the acquisition path still needs exhaustive search — §44's finding that the
  shipped gradient synthesiser solves 1 of 6 earlier tasks is untouched here
  and was not re-run;
* **new**: `Program.digest` is not canonical up to graph isomorphism (two
  topological orders of the same four-gate majority give two digests, §10),
  which is fine for a content-addressed store and quietly wrong for anything
  that counts occurrences;
* **narrowed**: the selection obstacle is no longer "the solver does not
  preserve modular structure". It is the pair *(one program per task,
  structural identity)*, and §10 shows that fixing both — and only both — lifts
  the arm to the ceiling on this family.

## 12. Reproduction and provenance

* **Nothing in `tcn/` or `generators/` was modified.** No core hook was needed.
  `git diff` for this branch touches only `research/premin-abstraction/`.
* **`PREREGISTRATION.md` was committed before any arm was run** (commit
  `89307ee`), together with the corpus machinery and before `run_corpora.py`,
  `run_mine_premin.py` or any enumeration was executed.
* **Every corpus program is executed through `tcn`** against its full 16-row
  truth table before entering a corpus (`verified_in_tcn: true` for all twelve
  bands).
* **Every enumeration is a full sweep**, `exhausted: true`, certificate
  `complete`, reported separately from `evaluated`. Nine arms, **83 358 720**
  programs evaluated in total.
* **What was and was not pre-registered.** Arms 1, 2, 2′, 3, 4, 4b and the
  secondary `arm2p_plus1one` were declared in `PREREGISTRATION.md` before any
  arm ran. `arm3p_mined_maj3`, `arm2s_semantic_trace` and the whole of §10 were
  **not**; they were added after the ranked tables were read, are labelled
  exploratory wherever they appear, and are counted toward no falsification
  criterion.
* **Test suite: 287 passed, 1 failed.** The failure is
  `tests/test_panel_interface.py::test_panel_episode_replays_and_restores`, the
  documented environmental failure in a worktree with `node_modules` symlinked
  from the main checkout. It is on main's own code and is not this track's; the
  symlink was removed before committing.
* **The shipped fixture reproduces**: `python -m tcn train --episodes 160` gives
  `initial_prediction_loss` 0.248835613951087 → `final_prediction_loss`
  0.0022308224288281053, `fully_frozen: true`, `frozen_evaluation_mean_return`
  4.0.
* **Wall-clock caveat.** This machine was shared with other agents' training
  jobs throughout. Wall figures are indicative; DFS node counts, evaluated
  counts and space sizes are not.
* Every figure in this document comes from a file in `out/`; `tables_premin.py`
  and the `show_*.py` scripts regenerate the tables directly from those files.
