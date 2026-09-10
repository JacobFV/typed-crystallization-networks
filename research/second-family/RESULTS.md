# A second task family: §52 replicates, §54 does not

`PREREGISTRATION.md` committed at `9737072` **before any arm ran**; Amendment 1
at `b81f656`, after the primary tables and labelled exploratory. `tcn/` and
`generators/` are untouched by this track and nothing under
`research/residual-gap/` was read or written.

Every number below is re-derivable from `out/*.json` by the scripts in this
directory; the joins that produce the tables are in `tables.py`, `show_b1.py`,
`show_b1_tie.py`, `show_cost.py` and `show_baselines.py`.

---

## 0. The verdict in four lines

* **Semantic pooling (§52) replicates.** On the corpus band with §52's geometry
  it reaches the hand-authored ceiling exactly — **48 of 1,005,360, exhausted,
  `complete`** — while **every** wrong-module control scores **0**, and it does
  it with a **different digest from the hand-authored module**. §46's
  "discovery, not rediscovery" holds on a second family.
* **Breadth-weighted ranking (§54) does not replicate.** O2 ties the incumbent
  O1 on both bands — **5 of 5 vs 5 of 5**, and **1 of 5 vs 1 of 5** — because
  the incumbent never fails here. There is nothing for the breadth term to fix.
* **§54's central control flips.** The frequency baseline B1, which helped **0
  of 5** on family 1, helps **5 of 5** on both bands here — **beating O2** on
  one of them. And the breadth exponent is inert: the window is rank 1 at
  **every** α from 0 to 10, against family 1's α ∈ [0.08, 3].
* **And a new caution, which is the most transferable thing here.** Semantic
  pooling *inverts* on the thin corpus band: it scores **0** where plain digest
  identity scores **48**. Pooling helps exactly when the abstraction is
  syntactically fragmented and hurts when its competitors are more fragmented
  than it is. That was invisible in family 1, where the abstraction was always
  the fragmented one.

---

## 1. The family, and why it is a fair test rather than a rerun

The shared abstraction is

    W4(w, x, y, z) = (w XOR x) AND (y XOR z)          "both pairs differ"

over five Boolean inputs `a..e`. §54's own second-family check was the same six
task *shapes* with the window swapped — a relabelling. This one differs in
every dimension that could plausibly break the method:

| | family 1 (§44 / §52 / §54) | family 2 (this track) |
|---|---|---|
| window | `MAJ3(a,b,c)` | `W4(w,x,y,z) = (w⊕x)∧(y⊕z)` |
| arity | 3 | **4** — exactly `mine.MAX_HOLES` |
| algebraic character | symmetric, monotone, threshold | **non-monotone, non-threshold, a conjunction of two parities** |
| hole-symmetry group | all of S3 (6 of 6 orderings) | **8 of 24 orderings** |
| window size in the basis | 4 gates | **3 gates** |
| corpus inputs | 4 | **5** |
| task minimum length | 5 | **4** |
| pooling key width | 8 rows | **16 rows** |
| module call-site cost | `ports³` | **`ports⁴`** |
| evaluation space with the module | 25,200 | **1,005,360** |

The symmetry group is the point. `MAJ3` is invariant under **every** ordering
of its holes, so §52's pooling key could not split it however a task realised
it. `W4` is invariant under only 8 of the 24, so hole ordering is not free —
a live way for semantic pooling to fail here that did not exist in family 1.
(It did not fire: all three syntactic realisations of the window landed in one
pooled class. That is a measurement, not an assumption.)

**Seven in-family tasks, six pairings, one never mined.**

| task | definition | pairing | role |
|---|---|---|---|
| `u1_w_abcd_xor_e` | `W4(a,b,c,d) ⊕ e` | `{ab\|cd}` | corpus + held out |
| `u2_w_acbd_and_e` | `W4(a,c,b,d) ∧ e` | `{ac\|bd}` | corpus + held out |
| `u3_w_bcde_or_a` | `W4(b,c,d,e) ∨ a` | `{bc\|de}` | corpus + held out |
| `u4_w_acde_xor_b` | `W4(a,c,d,e) ⊕ b` | `{ac\|de}` | corpus + held out |
| `u5_w_abde_or_c` | `W4(a,b,d,e) ∨ c` | `{ab\|de}` | corpus + held out |
| `u6_w_abce_and_d` | `W4(a,b,c,e) ∧ d` | `{ab\|ce}` | corpus, always |
| **`L1_w4_bdae_xor_c`** | `W4(b,d,a,e) ⊕ c` | **`{bd\|ae}`** | **arm table; in no corpus, and its pairing is used by no corpus task** |

Off-family twin `F''`: the same six shapes with `X4(w,x,y,z) = (w∧x) ⊕ (y∨z)`
— same arity, same 3-gate size, a different table.

Two control tasks: `H_par5 = a⊕b⊕c⊕d⊕e` and `H_x4 = X4(a,b,c,d) ⊕ e`.

**Every space is exhaustible, and the tightness is structural, not hoped for.**
A *pruned* straight-line program of 3 gates over five inputs reaches at most
four distinct inputs; every task depends essentially on all five. So every
minimum length is ≥ 4 by construction — certified at exactly **4** by
`minimal.scaffold_min`, which exhausted lengths 0, 1, 2 and 3 on all twelve
tasks — and the two-node evaluation scaffold cannot express any task without a
module, which is why `arm1_none` exhausts at zero rather than merely failing.

---

## 2. The corpora, with certificates

§46's `enumerate_programs.exhaustive`, unchanged, `N` 4→5 the only difference;
the same cap of 32 per task per band and the same deterministic
sha256-of-canonical-key subsample; every retained program rebuilt as a
`tcn.graph.Program` and executed against its full 32-row truth table before
entering the corpus (`verified_in_tcn: true` on all 24 bands, IC3).

| band | length | per-task found (capped at 32) | entries | exhausted | DFS nodes / task |
|---|---|---|---|---|---|
| `C-minall` (`W4`) | 4 | 1, 3, 1, 1, 1, 3 | **10** | true | 133,136 |
| `C-trace` (`W4`) | 4 ∪ 5 | + 10, 37→**32**, 21, 10, 21, 37→**32** | **136** | true | 10,676,386 |
| `C-minall` (`X4`) | 4 | 3, 1, 1, 3, 1, 1 | **10** | true | 133,136 |
| `C-trace` (`X4`) | 4 ∪ 5 | + 6, 11, 16, 6, 16, 11 | **76** | true | 10,676,386 |

Two of the twelve `W4` bands hit the cap (37 found, 32 retained by the
deterministic subsample); the other twenty-two did not, and the uncapped counts
are in `out/bands_*.json` beside the retained ones.

**Both bands are declared primary and both are reported**, as
`PREREGISTRATION.md` §3 required. At five inputs the family's tasks have only
1 to 3 minimum-length programs each, so `C-minall` is a 10-entry corpus, while
`C-trace` at **136 entries over 6 tasks** is the entry-count match to §52's
primary corpus (130 entries over 6 tasks). Declaring both in advance is what
keeps the band from being a post-hoc choice — and, as it turned out, the two
bands disagree, so the disagreement is a result rather than a selection.

Leave-one-out corpora (`C-trace`): 125, 101, 114, 125, 114 entries;
(`C-minall`): 9, 7, 9, 9, 9.

---

## 3. Identity — §52's arm table, replicated

Task `L1_w4_bdae_xor_c`, two-node compact scaffold, full sweep,
`max_programs = 2²⁴` against a space of 1,005,360. Constant baseline 0.5000,
random 0.5000 on this task.

### Band `C-trace` (the §52 geometry match)

| arm | module | space | evaluated | exhausted | certificate | conforming |
|---|---|---|---|---|---|---|
| `arm1_none` | — | 10,200 | 10,200 | true | `complete` | **0** |
| `arm2_syntactic` | `82b93e4bc7b2` (arity 3) | 70,560 | 70,560 | true | `complete` | **0** |
| **`arm2s_semantic`** | **`544e46660ae8`** | 1,005,360 | 1,005,360 | true | `complete` | **48** |
| `arm3_authored` (ceiling) | `609cedb54934` | 1,005,360 | 1,005,360 | true | `complete` | **48** |
| `arm4_wrong_authored` | `3140bbf72e88` (`X4`) | 1,005,360 | 1,005,360 | true | `complete` | **0** |
| `arm4b_wrong_mined` | `c8a9fa9083c9` | 1,005,360 | 1,005,360 | true | `complete` | **0** |
| `arm4s_runnerup` | `07e7d820aa71` | 1,005,360 | 1,005,360 | true | `complete` | **0** |
| `arm4s_matched` | `1e8b0e63e5f6` | 1,005,360 | 1,005,360 | true | `complete` | **0** |
| `arm4s_offfamily` | `8dd1a248244e` | 1,005,360 | 1,005,360 | true | `complete` | **0** |

**F1 does not fire.** Semantic pooling equals the ceiling and every wrong-module
control is 0 — the §52 result, on a family that shares nothing with §52's but
the harness.

**It is discovery, not rediscovery, again.** The pooled class publishes
`module:544e46660ae8ba6140998d95`; the hand-authored module is
`module:609cedb5493485c4f0f6cd43`. Different digests, byte-identical truth
table, identical ceiling — §46's finding on a second family.

**§55's caution, honoured and sharpened.** §52's arm count overstated its
distinct controls by one. Here the nine arms carry **six distinct mined
digests** (`tables.py` prints them) — but two of them, `c8a9fa9083c9` and
`1e8b0e63e5f6`, have the **same 16-row truth table**, and `8dd1a248244e` has
the same table as the hand-authored `X4`. So the honest count is:

* **5 distinct wrong-module digests**, collapsing to **3 distinct semantic
  wrong-module classes** (`X4`; truth table `…0001…0001…`; truth table
  `…0001……1…`), plus the floor.
* Reporting arm *names* would have claimed five controls. There are three.

### Band `C-minall` — the inversion

| arm | module | space | exhausted | certificate | conforming |
|---|---|---|---|---|---|
| `arm1_none` | — | 10,200 | true | `complete` | 0 |
| **`arm2_syntactic`** | **`544e46660ae8`** | 1,005,360 | true | `complete` | **48** |
| **`arm2s_semantic`** | `82b93e4bc7b2` (arity 3) | 70,560 | true | `complete` | **0** |
| `arm3_authored` | hand `W4` | 1,005,360 | true | `complete` | 48 |
| `arm4_wrong_authored` | hand `X4` | 1,005,360 | true | `complete` | 0 |
| `arm4b_wrong_mined` | `c8a9fa9083c9` | 1,005,360 | true | `complete` | 0 |
| `arm4s_runnerup` | `1e8b0e63e5f6` | 1,005,360 | true | `complete` | 0 |
| `arm4s_matched` | `1e8b0e63e5f6` | 1,005,360 | true | `complete` | 0 |
| `arm4s_offfamily` | `36a7373d0d76` | 1,005,360 | true | `complete` | 0 |

**On the thin band, semantic pooling loses to the identity it was introduced to
replace**, and **F7 fires**: `arm4s_runnerup` and `arm4s_matched` resolve to the
same class, so this band has **5 distinct classes across 6 mined arm names** —
§55's collision, reproduced independently.

### Gradient panel — the achieved configuration, not §52's

§52 ran 24 seeds × 400 steps with a 3-ary module. A 4-ary module makes this
scaffold's output node **1,416 candidates** wide over 32 rows and one 400-step
seed costs **~240 wall-seconds** here, so the panel is **8 seeds × 200 steps**
on `C-trace` and `L1_w4_bdae_xor_c`. That is the configuration achieved, and
the enumeration above — not this panel — carries the claims.

| arm | solved / 8 | median acc | mean acc | best acc | constant | random |
|---|---|---|---|---|---|---|
| `arm1_none` | 0 | 0.7500 | 0.7500 | 0.7500 | 0.5000 | 0.5000 |
| `arm2_syntactic` | 0 | 0.7500 | 0.7500 | 0.7500 | 0.5000 | 0.5000 |
| **`arm2s_semantic`** | **2** | 0.7500 | 0.8125 | **1.0000** | 0.5000 | 0.5000 |
| `arm3_authored` | **2** | 0.7500 | 0.8125 | **1.0000** | 0.5000 | 0.5000 |
| `arm4_wrong_authored` | 0 | 0.7500 | 0.7344 | 0.7500 | 0.5000 | 0.5000 |
| `arm4s_runnerup` | 0 | 0.7500 | 0.7500 | 0.7500 | 0.5000 | 0.5000 |

Pooling ties the ceiling on gradient as well, and no wrong module ever reaches 1.

---

## 4. Transfer — §54's leave-one-out table, which does **not** replicate

A module selected on `band ∖ t` is offered to `t`, which contributed no program
at any length to that corpus. Every cell exhausted, `evaluated == space_size`,
certificate `complete`; **0 IC4 violations across all 105 held-out and 54 arm
enumerations**.

### Band `C-trace`

| objective | u1 | u2 | u3 | u4 | u5 | helps, of 5 | window is rank 1 |
|---|---|---|---|---|---|---|---|
| **O1** `description_bits` summed (incumbent) | 48 | 16 | 16 | 48 | 16 | **5 of 5** | 5 of 5 |
| **O2** breadth-weighted (§54's winner) | 48 | 16 | 16 | 48 | 16 | **5 of 5** | 5 of 5 |
| O3 per-task mean | 0 | 0 | 0 | 0 | 0 | 0 of 5 | 0 of 5 |
| O4 in-corpus leave-one-out CV | 48 | 16 | 16 | 48 | 16 | 5 of 5 | 5 of 5 |
| **B1** frequency count (§54's control) | 48 | 16 | 16 | 48 | 16 | **5 of 5** | 5 of 5 |
| B2 occurrence count | 0 | 0 | 0 | 0 | 0 | 0 of 5 | 0 of 5 |
| *exploratory:* best window class, any rank | 48 | 16 | 16 | 48 | 16 | 5 of 5 | — |
| floor `arm1_none` | 0 | 0 | 0 | 0 | 0 | **0 of 5** | — |
| ceiling hand-`W4` | 48 | 16 | 16 | 48 | 16 | **5 of 5** | — |
| wrong hand-`X4` | 0 | 0 | 0 | 0 | 0 | **0 of 5** | — |

Constant baselines on these tasks: u1 0.5000, u2 0.8750, u3 0.6250, u4 0.5000,
u5 0.6250; random 0.5000 throughout. The floor's 0 is a *certified absence* —
the space was exhausted — not a search failure.

### Band `C-minall`

| objective | u1 | u2 | u3 | u4 | u5 | helps, of 5 |
|---|---|---|---|---|---|---|
| O1 | 0 | 16 | 0 | 0 | 0 | **1 of 5** |
| O2 | 0 | 16 | 0 | 0 | 0 | **1 of 5** |
| O3 | 0 | 0 | 0 | 0 | 0 | 0 of 5 |
| O4 | 0 | 0 | 0 | 0 | 0 | 0 of 5 |
| **B1** | 48 | 16 | 16 | 48 | 16 | **5 of 5** |
| B2 | 0 | 0 | 0 | 0 | 0 | 0 of 5 |
| floor / ceiling / wrong hand-`X4` | | | | | | 0 / **5** / 0 of 5 |

**F2 fires on both bands.** O2 does not beat O1: 5 = 5, then 1 = 1.
**F3 fires on both bands.** B1 matches O2 on `C-trace` and **beats it 5 to 1**
on `C-minall`.

---

## 5. F3, checked hardest — three independent probes

F3 was the criterion the brief said to check hardest, and it survives two of
three probes.

**(a) The breadth exponent is inert.** §54 swept `|T(c)|^α · Σ s_e − D` and
found the right class held only for **α ∈ [0.08, 3]**, with α = 0 (the
incumbent) and α = 10 (the frequency class) both failing — its evidence that
*neither term alone suffices*. Here, on `C-trace`, the window is rank 1 at
**every α tested, 0.0 through 10.0, on all six corpora**. **α = 0 is the
incumbent and it works.** On `C-minall` the window is rank 1 at **no α** on
five of six corpora. In neither direction does the breadth term do any work.

**(b) B1's rank 1 is never decided by its score.** On all **twelve** corpora
`objectives.rank1_decided_by` returns `tie_break`. The frequency score ties at
the top on 6 classes (`C-trace`) or 4 (`C-minall`) every single time; what
selects is the tie-break `(-nodes, digest)`. So B1's "5 of 5" is not frequency
selecting — it is *"prefer the largest class among the most widely shared"*,
which is a different rule.

**(c) Amendment 1 — and this is the one that qualifies the headline.** On
`C-trace` the tie-break's `-nodes` term still leaves **two** three-node classes:
the window `544e46660ae8` and a non-window arity-3 class `a20f9a11fd36`. The
window wins because `5` sorts before `a` in hex. Running the other one under
the identical protocol:

| | u1 | u2 | u3 | u4 | u5 | helps, of 5 |
|---|---|---|---|---|---|---|
| B1's pick `544e46660ae8` | 48 | 16 | 16 | 48 | 16 | **5 of 5** |
| the class the digest comparison rejected, `a20f9a11fd36` | 0 | 8 | 0 | 0 | 0 | **1 of 5** |

All exhausted, certificate `complete`. **Had the hex digest gone the other way,
B1 would have helped 1 of 5, not 5 of 5, on this band.** On `C-minall` there is
no such coin flip — the window is the unique three-node class in the tie group,
so `-nodes` alone decides.

So the precise, defensible statement is: **B1 matches or beats O2 on both bands
of this family, but its selection rests on the tie-break rather than on the
frequency score, and on one band on a two-way lexicographic coin flip.** §54's
"frequency helps 0 of 5" is nonetheless family-specific: there the window sat at
**rank 4** under a task count, strictly dominated; here it is inside the top tie
group on every corpus, because it occurs in all six tasks.

---

## 6. Why the two sections come apart here — the mechanism, measured

**O1 does not flip on this family, so O2 has nothing to fix.** §54's diagnosis
was that summing savings over entries penalises broad fragments when a task is
removed, and rank 1 flipped by **0.85 %** in two of five corpora. The same
margin here:

| corpus | entries | O1 rank-1 margin over rank 2 | O2 rank-1 margin |
|---|---|---|---|
| `C-trace` full | 136 | 27.05 % | 27.05 % |
| `C-trace` ∖ u1 | 125 | 20.39 % | 20.54 % |
| `C-trace` ∖ u2 | 101 | 43.26 % | 42.86 % |
| `C-trace` ∖ u3 | 114 | **6.47 %** | 16.27 % |
| `C-trace` ∖ u4 | 125 | 20.39 % | 20.54 % |
| `C-trace` ∖ u5 | 114 | **6.47 %** | 16.27 % |

The tightest margin on this family is **7.6× §54's flip margin**, and the
incumbent never loses rank 1. §54's defect is real but it is a property of a
family in which the window's MDL advantage is thin; here a 4-ary, 3-node class
saves so much more per site than its 2- and 3-ary competitors that removing a
task cannot close the gap. **O2 is not wrong here — it is redundant here.**

**Why pooling inverts on the thin band.** Pooling helps a class in proportion to
how many *digests* it was split across:

| band | window digests under syntactic identity | syntactic rank | pooled rank | pooling cost | eligible set |
|---|---|---|---|---|---|
| `C-trace` | **3** | 4 | **1** | 0.97× | 87 → 32 |
| `C-minall` | **1** | 1 | **2** | 0.91× | 16 → 11 |

On `C-trace` the window is realised by three circuits, pooling unifies them and
lifts it from rank 4 to rank 1. On `C-minall` the window has exactly **one**
circuit, so pooling gives it nothing — while its nearest competitor is an
arity-3 function split across **two** digests (`82b93e4b` and `a5924e78`, both
10,184 bits, 10 occurrences each). Pooling merges those two into one class at
**16,680 bits and 14 occurrences**, which overtakes the window's unchanged
12,528. **Semantic pooling is not a free improvement: it redistributes rank
toward whichever class was most fragmented, and that need not be the one you
want.** Family 1 could not show this, because there the abstraction *was* the
fragmented one (8 to 12 digests, §46). This is the finding of this track that
most deserves to be carried forward.

For symmetry with §5's criticism of B1: `arm2_syntactic`'s win on `C-minall` is
**not** a coin flip. Its rank 1 (`544e4666`, 12,528 bits) ties rank 2
(`c8a9fa90`, also 12,528) on score, but the tie-break's *first* term separates
them — 6 tasks against 2 — so a substantive criterion decides it, not the digest.

**Cost, reported honestly.** Pooling is **0.97× and 0.91×** the wall clock of
syntactic mining — *cheaper*, against §52's 1.08–1.19× — and shrinks the
eligible set (87 → 32, 16 → 11) exactly as §52 found (94 → 40).

---

## 7. The negative controls

| | `H_par5` | `H_x4` | verdict |
|---|---|---|---|
| every `W4`-family arm, ceiling included, both bands | **0** | **0** | the right module does not help tasks it should not |
| hand-authored `X4` | 0 | **40** | the control is **live** |
| `arm4s_offfamily`, `C-trace` (`8dd1…`, the `X4` table recovered by mining `F''`) | 0 | **40** | mining the off-family corpus recovers the off-family window |
| `arm4s_offfamily`, `C-minall` (`36a7…`) | **360** | **104** | **disclosed, not hidden** — see below |

`H_par5` and `H_x4` both have constant baseline 0.5000 and random baseline
0.5000, so a 0 there is a certified absence in an exhausted space, not a
near-miss.

**One control is not clean and it is reported rather than dropped.** The
`C-minall` off-family arm's rank-1 arity-4 class `36a7373d0d76` is
`(x0 ∧ x1) ⊕ x2 ⊕ x3`, and it solves `H_par5` (360 conforming) and `H_x4` (104).
The reason is exact: bind `x0` and `x1` to the same port and the class
degenerates to three-way parity, so two nodes reach five-input parity.
**F4 does not fire** — F4 is scoped to `W4`-family arms and this module is
mined from `F''` — but the arm is a poor negative control on that band and
should not be cited as one. It changes no row of the `W4` tables: every
`W4`-family arm is still 0 on both controls.

---

## 8. Falsification criteria — what fired

| | criterion | fired? |
|---|---|---|
| **F1** | identity is family-specific | **no on `C-trace`** (pooling = ceiling, all controls 0); **yes on `C-minall`** (pooling 0, digest identity 48) |
| **F2** | O2 does not beat O1 | **YES, both bands** (5 = 5; 1 = 1) |
| **F3** | B1 matches O2 | **YES, both bands** (5 = 5; 5 > 1) — with the tie-break qualification of §5 |
| **F4** | the controls are dead | no — every `W4` arm 0 on both, hand-`X4` solves `H_x4` at 40 |
| **F5** | not exhaustible | no — every band exhaustive, every enumeration `exhausted: true`, **0 IC4 violations in 159 enumerations** |
| **F6** | the scaffold is not tight | no — `arm1_none` 0 on every family task, exhausted |
| **F7** | the arms are not distinct | **yes on `C-minall`** (runner-up = node-matched); on `C-trace` 6 distinct digests but only **3 distinct semantic wrong-module classes** |

Against §7 of the pre-registration ("what counts as replicates"): **criterion 1
holds on `C-trace` and fails on `C-minall`; criteria 2 and 3 fail on both
bands; criterion 4 holds.** So §52 replicates and §54 does not.

---

## 9. Sensitivity

The rule's parameters, swept as §52 §8 swept them. `MAX_HOLES` is not swept
below 4: the window has arity 4, so a lower ceiling removes it by construction.

| band | `MIN_TASKS` | `MAX_NODES` | eligible | O1 | O2 | B1 |
|---|---|---|---|---|---|---|
| `C-trace` | 2 | 4 | 32 | window r1 | window r1 | window r1 |
| `C-trace` | 2 | 5 | 32 | window r1 | window r1 | window r1 |
| `C-trace` | 3 | 4 | 8 | window r1 | window r1 | window r1 |
| `C-trace` | 3 | 5 | 8 | window r1 | window r1 | window r1 |
| `C-minall` | 2 | 4 | 11 | r2 | r2 | window r1 |
| `C-minall` | 2 | 5 | 11 | r2 | r2 | window r1 |
| `C-minall` | 3 | 4 | 4 | r2 | r2 | window r1 |
| `C-minall` | 3 | 5 | 4 | r2 | r2 | window r1 |

Every declared arm uses the defaults `MAX_NODES=5, MAX_HOLES=4, MIN_TASKS=2`;
nothing is tuned. The band difference is **not** a parameter artifact: it holds
at all four cells of each band. This is the two-configuration check §53 and §55
require — a result certified at one configuration is not evidence, and here the
two configurations disagree in exactly the way that matters.

---

## 10. Integrity

* **IC1** — `pool.build`'s re-derivation agreed with `mine_semantic.propose`
  exactly on **all 12 corpora** (`ic1_pass: true` in both `ranked_*.json`).
* **IC2** — O1's rank 1 equalled the rule's own rank 1 on **all 12 corpora**.
* **IC3** — `verified_in_tcn: true` on all 24 corpus bands.
* **IC4** — `evaluated == space_size` and `exhausted: true` on **all 159**
  reported enumerations (54 arm + 105 held-out), plus 5 in Amendment 1.
* Upstream code was **not edited**. `pool.py`'s one family-specific function
  (`is_window`, which maps a family name to a truth table) is re-pointed at
  runtime by `patches.py`, delegating to the original for §52's two family
  names; `family.py` and `evaltasks.py` shadow §54's by module name. Every
  scoring, pooling, rewriting and verification path is §46's and §54's own.

**Constraints.** Repo `.venv`. Test baseline taken on this worktree *before any
file of this track existed*: **336 passed, 1 failed** —
`tests/test_panel_interface.py::test_panel_episode_replays_and_restores`, the
documented environmental failure, with `generators/computer/engine/node_modules`
symlinked from the main checkout (without the symlink that one passes and 13
`generators/computer` tests fail instead). This track adds no test and touches
no tested module.

---

## 11. What this changes about §52 and §54

**§52 stands, with a new boundary condition.** Semantic pooling reaching the
ceiling while every wrong-module control stays at 0 is now shown on two families
that share only the harness — a 3-ary symmetric monotone threshold over four
inputs, and a 4-ary non-monotone conjunction of parities over five. Its
mechanism is also now pinned: it buys rank exactly in proportion to how badly
the abstraction was fragmented by digest identity, and on a corpus where the
abstraction is *not* fragmented it can hand rank 1 to a competitor that is.
**"Use semantic pooling" is not yet safe advice; "use semantic pooling when the
target abstraction has several circuits in the corpus" is.**

**§54 does not stand as a general result.** Its headline — breadth-weighted
ranking transfers where the incumbent does not, with frequency helping 0 of 5 —
reproduces none of its three parts here. The incumbent transfers, the breadth
weighting adds nothing (α = 0 works, and so does α = 10), and frequency helps
5 of 5. §54's own caveat, *"one family, seven tasks — anecdote-strength on
breadth"*, was the correct reading, and the present track converts it from a
caveat into a measurement: **§54 described a defect of a particular family's MDL
geometry — a 0.85 % rank-1 margin — not a defect of `description_bits` as a
ranking objective.** §41's separate certified negative about `description_bits`
is untouched by this and still stands.

**What the claim now rests on.** Two families; **14 distinct in-family tasks**
(7 + 7) and 4 off-family control tasks; 12 mining corpora here plus §52's 6;
**3 distinct semantic wrong-module classes** on this family's primary band
(5 digests) and 3 on §52's. Two families is two, not many — this track has
doubled the evidence for §52 and removed the evidence for §54, and neither is
yet a demonstration at breadth.

---

## 12. Reproducing

    .venv/bin/python research/second-family/family.py        # via build_bands
    .venv/bin/python research/second-family/run_mine.py
    .venv/bin/python research/second-family/run_arms.py enum
    .venv/bin/python research/second-family/run_rank.py
    TCN_WORKERS=16 .venv/bin/python research/second-family/run_heldout.py enum
    .venv/bin/python research/second-family/run_sensitivity.py
    .venv/bin/python research/second-family/run_b1alt.py      # Amendment 1
    .venv/bin/python research/second-family/run_grad.py
    .venv/bin/python research/second-family/tables.py

Wall clock: corpus bands ~5 min, mining ~2 s, 159 enumerations ~12 min on 16
workers, gradient panel ~15 min on 12 workers.
