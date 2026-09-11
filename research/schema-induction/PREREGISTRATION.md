# Pre-registration — structural schema induction from the §65 `gap 1` counterexample

Committed before any arm runs. `tcn/` and `generators/` are not touched; this
track is a sidecar. Every figure in `RESULTS.md` will be rendered from `out/`
by `report.py`, and `verify.py` will re-derive each headline from raw JSON
without importing the run or report scripts, writing `out/verify.json` and
`out/headline.json` for the merge gate.

## 0. What was run before this document, disclosed

`phase0.py` ran before this document was written and its output
(`out/phase0.json`) is quoted throughout §2 and §3. It reads **only** the
`gap 0` and `gap 1` episode caches already committed under
`research/integrated-flagship/out/`. It does not construct, read or name a
`gap 2` or `gap 3` episode. Its three checks (V2, V3, V8) are validity checks,
not criteria; they are reported whatever they say, and V3 is a stop condition
(F2) rather than a result.

Nothing else has run. No arm, no induction, no final-split episode.

## 1. The claim block, the question and the criterion

```
claim:             can an outer learner repair an expressivity failure by inducing
                   a geometry-parametric schema that generalizes to unseen gaps?
starting evidence: gap 0 works; gap 1 is a CEGIS/admission counterexample
selection data:    train + admission/gap-1 only
blind final:       gap 2 + gap 3
comparators:       S0; blind unrestricted widening; induced S1;
                   hand-written parametric schema; patch/frozen-index schema
success:           S1 passes gap 2/3, with total induction + downstream-search cost
                   reported against the blindly widened alternative
failure modes reported separately:
                   no repair found; repair is a frozen patch; repair works gap 1
                   but not gap 2/3; blind widening is cheaper; hand-written schema
                   dominates; outer-loop cost erases downstream savings
```

Each failure mode is a **distinct reported outcome**, not a single "fail". §7
gives them ids (O0–O6) and §6 gives the criteria that decide between them.

### 1.1 The question

§65 refuted the inherited STEP schema at its second configuration: at `gap 1`
the schema space holds 261,654,545,280 programs conforming on the 12 training
episodes and **0** conforming across all 48 episodes, exhausted, certificate
`complete` (`research/integrated-flagship/out/generalize_gap1.json`). No prior
over that pool can generalize, because the pool excludes the answer. That is a
failure of **expressiveness**, not of selection.

**The question.** Given S0, a training split, admission counterexamples from
`gap 1`, and a generic typed structural mutation grammar — and *not* given the
correct offset, the fact that "above means two rows", the repaired schema, or
any `gap 2` / `gap 3` example — can an outer learner induce a schema S1 that
(a) is admitted at `gap 0` and `gap 1`, and (b) generalizes to the blind
`gap 2` and `gap 3` configurations?

### 1.2 The criterion is structural, and it is an invariant, not a mechanism

> **bad:** S1 contains the `gap 1` numerical answer.
> **good:** S1 contains a rule whose concrete displacement is **re-derived from
> the new instance's geometry**.

That is the whole criterion, and it is deliberately mechanism-free. **Any
equivalent general solution counts** — a row-stride rule, an affine
displacement, a relative-coordinate operator, a derived spatial-step function,
an indexed generator. §3.1's `PARAMETERIZE` is *one* way to reach it, not a
definition of it, and a repair that reaches the invariant by some other family
is scored as induction just the same.

This is §64's lesson applied one level up: **a concrete instantiation is
evidence about a schema, never the schema itself.** A schema that carries the
answer as a constant has recorded the evidence; a schema that re-derives the
answer from the instance has kept the conjecture.

* **Patching** — the repair carries the `gap 1` displacement (or a special case
  producing it) as a value fixed during induction. It fits admission and does
  not extrapolate. A *frozen index* — a generator whose range was fitted on
  admission — is patching, however parametric it looks.
* **Induction** — at `gap 2` and `gap 3` the concrete displacement is computed
  from that configuration's geometry, and is a value that was in no pool S1
  instantiated during induction.

**The declared null.** The outer learner produces a patch. This is the default
reading of any result that fits admission; induction has to be shown, with the
blind split, against the hand-written parametric baseline.

## 2. S0, and the `gap 1` counterexample, exactly

### 2.1 S0

S0 is `research/integrated-flagship/family.py`'s `pools("schema", 16)` —
§65's inherited arm, unmodified. The coarse graph (hand-initialisation H1) and
every non-STEP pool are shared by every arm of this track:

| slot kind | slots | pool |
|---|---|---|
| ADDR | `cpos`, `ra` | `legal_candidates` over `("add","sub","mul","min","max")`, arity 2 |
| LIT | `lc1..lc3`, `lr1`, `lr2` | the 26 letters |
| GROUND | `K1..K4` | the 8 palette colours |
| MATCH | `M` | `(1, 2, rg, same)` over the 16×16 truth tables |
| **STEP** | `X1`, `X2`, `X3` | `{(op, base, o)}`, op ∈ {`add`,`sub`}, base ∈ {`lo`,`hi`}, **o ∈ (6, 51, 3, 48, 9)** — 20 candidates |

The STEP offsets are `rung3_widgets.offset_pool(16)` = `(6, 3W+3, 3, 3W, 9)`
with `W = 16`, in raw byte units (3 bytes per pixel, `3W = 48` bytes = one
raster row). The click is `q = (base ± o) // 3`; a hit is
`owner[q] == target`.

### 2.2 The counterexample, as two integers

`phase0.py` computes, for each relation and each admission configuration, the
exact set of single-constant step specifications that put the click inside the
target on **every** episode of that relation (16 episodes each). Measured
(`out/phase0.json`, `V2_expressivity`):

| configuration | relation | conforming steps (offset ranges, raw bytes) | S0's own offsets that work |
|---|---|---|---|
| `gap 0` | `above` | `sub(lo)`: [7,48] ∪ [55,96] ∪ [103,144] | 9, 48 |
| `gap 0` | `left_of` | `sub(lo)`: [1,12]; `add(lo)`: [36,47] ∪ … | 3, 6, 9 |
| `gap 0` | `right_of` | `add(hi)`: [3,11]; `sub(hi)`: [37,45] ∪ … | 3, 6, 9 |
| `gap 1` | `above` | `sub(lo)`: **[55,96] ∪ [103,144] ∪ [151,192]** | **none** |
| `gap 1` | `left_of` | `sub(lo)`: [4,12]; `add(lo)`: [36,44] ∪ … | 6, 9 |
| `gap 1` | `right_of` | `add(hi)`: [6,14]; `sub(hi)`: [34,42] ∪ … | 6, 9 |

The whole counterexample is:

> **min required offset for `above` at `gap 1` = 55. max offset in S0 = 51.**

S0 solves all three relations at `gap 0` and exactly one relation —
`above` — is unsayable at `gap 1`. This reproduces §65's exact count (0
conformers over 48 episodes) from an independent derivation, and V1 will
assert the two agree.

### 2.3 Why this is a real test, and not the previous track's collapse

The scaffold-induction track (branch `scaffold-induction`, its amendments A15
and A16) found that `ADD_NODE` and `ADD_PATH` repaired **nothing**: its defect
generator only *narrowed* existing scaffolds, so `SUBST` and `WIDEN` were
always the inverse of the damage, and structure-inventing edits answered a
question the corpus never asked. Its measured generalisation, supplied by that
track's agent: of 32 `bool` defects rejected as already solvable, **20 (62%)
left the base scaffold's own conforming member completely intact** — a
narrowing applied at a random site mostly misses, because a conforming member
uses a small fraction of the candidates available to it.

The two defect shapes that cannot miss are (i) narrowing *along the solution*
and (ii) removing *expressiveness*. **The `gap 1` counterexample is the second
kind by construction**: no re-selection inside S0's pools can succeed, because
the required displacement is not sayable at all. §2.2's two integers are the
mechanical statement of that, and §65's exhausted 0-count is the independent
one. This is the property that makes the present design a test rather than a
restoration exercise, and A16 of the scaffold-induction pre-registration named
this experiment in advance, predicting that structure-inventing edits become
the only family that repairs anything. That prediction is inherited here as
**P4** (§7).

### 2.4 The geometry, stated as a prediction — not measured

From `env.neighbour` and §2.2's band structure: at `gap g`, `above` requires
`sub(lo, o)` with `o` in ⋃ₖ [48·(g+k) − 41, 48·(g+k)] for k = 1…h, where h is
the smallest target height in that configuration's episodes (h = 3 in both
measured configurations — exactly three bands appear). The right endpoint of
every band is a multiple of 48 = 3W, so **{48·i : i ≥ 1} contains a solution at
every gap, at i = g+1**, while no finite constant list fixed on admission does.

Three consequences are pre-registered as predictions, to be checked when the
final split is read and reported whether or not they hold:

* **P1.** S0 has zero conforming programs at `gap 2` and `gap 3` (max offset 51
  < 103 and < 151).
* **P2.** The **flat substrate** (§65's `N`, offsets 1…128) has a conforming
  program at `gap 2` (103 ≤ 128) and **zero at `gap 3`** (151 > 128). The flat
  substrate is expressively limited too, one configuration later.
* **P3.** **No single constant offset serves `gap 0` and `gap 3` together** for
  `above`: `gap 0`'s solution set is contained in [7,144] and `gap 3`'s in
  [151,288], and they are disjoint. A patch therefore *cannot* pass the blind
  final test, however good its `gap 1` numbers are.

P3 is the load-bearing one: it is why this experiment can separate patching
from induction at all.

## 3. The mutation grammar

### 3.1 Deliberately small, and typed

Six families, on the sidecar schema object only (no change to the spec IR, no
generic schema language — HANDOFF priority 7):

| family | what it does | can it invent structure? |
|---|---|---|
| `SUBST` | replace one candidate specification at one slot | no |
| `WIDEN` | add candidates at one slot, with constants drawn from the **declared literal source** L (§3.2) | no |
| `REWIRE` | re-point one node port to another value of the same type | no |
| `ADD_NODE` | insert one typed node computing an existing operator over values already in scope | yes |
| `COMPOSE` | fold one existing registered module into the step path | yes |
| **`PARAMETERIZE`** | replace a slot's **constant pool** with an **indexed generator** `{f(s, t, i) : i ∈ 1…n}`, where `s`, `t` are values already present in that slot's pool and `f` ranges over the declared shapes `s·i`, `s·i + t`, `s + t·i`; **`n` is a free schema parameter, not fixed during induction** | yes |

`PARAMETERIZE` is the family most likely to reach §1.2's invariant, and it is
**one mechanism, not the definition**. C3 is stated on the invariant, so
`ADD_NODE` or `COMPOSE` reaching a relative-coordinate operator, an affine
displacement or a derived spatial-step function scores as induction just the
same, and a `PARAMETERIZE` whose range was fitted on admission does not.

The enumerator, the `Edit` frozen dataclass and the `apply_edit` /
`enumerate_edits` signatures are **copied** from
`research/scaffold-induction/edits.py` into this track's `edits.py` and
extended with `PARAMETERIZE` and `COMPOSE`. That track's files are not edited.
Its A11 lesson is inherited: **candidate-truncation bounds stay off**
(`MAX_NEW = MAX_NEW_NODE = None`); an over-budget edit is recorded
**undecided**, never silently redefined as unrepairable.

### 3.2 The declared literal source L, and why widening *within it* is insufficient

`WIDEN` may only introduce integer constants from

    L = the integer constants the scaffold `family.build` already declares
      = {0,…,15}  (the text-address constants k0…k15)
      ∪ {1, 2, 3} (the scoring head's, and `three`, the raster stride)
      ∪ {3, 6, 9, 48, 51}  (S0's own offsets)
      = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,48,51}

No element of L was chosen with the answer in view; each is already a declared
constant of §65's scaffold. Measured (`out/phase0.json`,
`V3_widen_insufficiency`):

    max(L ∪ S0 offsets) = 51   <   min required offset at `gap 1` = 55
    |{reachable offsets} ∩ {required offsets}| = 0
    holds = true

So **no finite sequence of `WIDEN` edits over L can reach the `gap 1` repair**,
and the grammar's structure-generating families are *required relative to S0's
starting representation*, not merely available. `verify.py` re-derives this
from `out/phase0.json` by token equality on the two integers.

**The qualifier is load-bearing and belongs in every statement of this result.**
Never "widening cannot solve the task" — always "widening **within S0's
declared source** cannot". Unrestricted widening *does* reach the answer; it is
the `S_blind` arm, a first-class comparator charged its full total cost, and a
genuine falsifier of this track: **if brute widening turns out cheaper than
induced structure, the induction did not earn its complexity** (R1, outcome
O3). The meaningful claim here is about the starting representation, not about
widening as such.

### 3.3 The grammar does not contain the repaired schema verbatim

Three checks, all mechanical, all run in phase 0 (V4):

1. **No literal.** No `*.py` of this track contains an integer literal in the
   `gap 1` requirement set [55,96] ∪ [103,144] ∪ [151,192]. Checked by a regex
   over every integer token in the track's `*.py`; `range(a, b)` is checked as
   the two literals `a` and `b`, so `S_blind`'s `range(1, 768)` passes while a
   hard-coded 96 would not. `out/*.json` measurements and this document's own
   quotation of the measured ranges are exempt, being outputs rather than
   grammar. Verified for `phase0.py` at the time of writing: no match.
2. **No production.** The `PARAMETERIZE` shapes are `s·i`, `s·i + t`,
   `s + t·i` with `s`, `t` drawn from the *slot's existing pool*. The repaired
   family `{48·i}` arises only as one of the enumerated `(s = 48, shape = s·i)`
   instances, alongside `{3·i}`, `{6·i}`, `{9·i}`, `{51·i}` and every `s·i + t`
   combination. The shapes are declared here, before running; adding a shape
   later is an amendment, written before the arm that uses it.
3. **No hidden ordering.** The enumeration order is `SUBST`, `WIDEN`,
   `REWIRE`, `ADD_NODE`, `COMPOSE`, `PARAMETERIZE` — the structure-inventing
   families **last**, so a uniform-order learner is not handed the answer by
   position. (This mirrors H3 of `rung3_widgets`: the correct candidate is
   never at index 0.)

### 3.4 The decision predicate, and a fast decider that must be validated

An edit is *decided* by asking whether the edited schema contains a program
conforming on all episodes of `train ∪ admission` at both admission
configurations. The exact instrument is `engine.Counter` (copied from the
flagship): K > 0 with certificate `complete`.

Because the exact counter costs ~86 s per 48-episode configuration at S0's pool
size and scales linearly in the step pool (§10), the outer loop uses a **fast
decider** that exploits the fact that edits in this grammar touch only the STEP
path: the schema conforms iff a valid routing exists **and** each relation has
a step candidate hitting all of its episodes (§2.2's test). This is declared as
an instrument, not assumed:

**V6 (required before any criterion is read).** `fast_decide` must agree with
`engine.Counter(K > 0)` on ≥ 40 edited schemas spanning all six families and
both admission configurations, including ≥ 10 that the fast decider calls
non-conforming. Any disagreement voids the fast decider and the outer loop is
re-run with the exact counter, at the cost in §10. Every figure reported in
`RESULTS.md` is produced by the exact counter regardless.

## 4. The arms

Every arm is scored at all four configurations. Arms differ only in how the
STEP pool is obtained; the coarse graph and every other pool are identical.

| arm | what it is | role |
|---|---|---|
| `S0` | §65's inherited schema, unrepaired | the refuted starting point |
| `S1` | induced by the outer learner, **uniform** edit order | the object under test |
| `S1_prior` | induced with a **learned edit prior** fitted on the scaffold-induction corpus' repair history | "does inherited outer knowledge help?" |
| `S_widen` | the grammar restricted to `WIDEN` over L | the mechanical proof of §3.2, run as an arm |
| `S_blind` | **unrestricted widening**: offsets 1…767, ops {add,sub}, bases {lo,hi} (3,068 step candidates) | **first-class comparator and genuine falsifier** — it reaches the answer; charged its full total cost including the downstream search its larger pool forces (R1, O3) |
| `S_hand` | **hand-written parametric schema**: offsets `{3·i} ∪ {48·i}`, `i` a free schema parameter | **mandatory** hand baseline |
| `S_flat` | §65's flat substrate: 5 ops × 2 bases × offsets 1…128 | no schema at all |
| `D_prior` | **the distractor**: the same six-family grammar (so its space *contains* the parametric repair) with an edit prior fitted on the scaffold-induction `bool` + `rel` corpus, where `WIDEN`/`SUBST` repair every case and `ADD_NODE`/`ADD_PATH` repair none | wrong *experience*, same expressivity |

`D_prior` is the distractor the HANDOFF demands after §65's vacuous C2: it
**can** succeed — the parametric repair is in its space and reachable by its
grammar — but its inherited experience genuinely teaches "prefer `WIDEN` over
structural edits", which is true of that corpus and false of this one. If
`D_prior` matches `S1_prior`, the prior carried nothing.

`S_hand` is a *parametric* baseline, not a constant one, because §65's lesson
is that the obvious hand prior is often a one-line rule. Here the one-line rule
is "displacements are multiples of the pixel stride or of the row stride", and
it must be beaten, not merely matched.

## 5. The three splits, mechanically enforced

| split | configurations | episodes | what may read it |
|---|---|---|---|
| `train` | `gap 0`, `gap 1` | 12 + 12 = 24 (flagship `split="train"`, indices 0–11, seed 0) | edit proposal, prior fitting, ordering |
| `admission` / CEGIS | `gap 0`, `gap 1` | 36 + 36 = 72 (flagship `split="test"`, indices 0–35, seed 0) | admissibility; supplies counterexamples. **A counterexample, once consumed, is training information** and is counted in cost measure 4 |
| `final` (blind) | **`gap 2`, `gap 3`** | 48 + 48 = 96 (seed 0, indices 0–47, `split="test"`) | **nothing until the headline is scored** |

`gap 0` and `gap 1` episodes are the byte-identical caches already committed at
`research/integrated-flagship/out/episodes_gap{0,1}.json` (seed 0, 12 train +
36 test each, ~12 s to rebuild); this track re-reads them rather than
regenerating, and `verify.py` asserts the SHA-256 of each file matches the
committed one.

**Enforcement**, adapted from `scaffold-induction/run_domain.py` and
`score_final.py` and copied into this track:

1. `final_cache.py` builds `gap 2` and `gap 3` into `out/final/` and writes
   `out/final_digest.json` — a SHA-256 over `repr(value.flat())` of each
   episode's inputs then targets, keys sorted, episodes in order. Nothing else
   in the track opens `out/final/`.
2. The induction-time corpus object carries `final_digest`, `n_final` and
   `final_untouched: true`, and the live object has its `final` key **deleted**.
   The scaffold-induction track deleted it from a plain dict, so an accidental
   read raised `KeyError` while its comment claimed `AttributeError`; here the
   key is replaced by a `_Blind` sentinel whose `__getitem__`, `__iter__` and
   `__len__ ` all raise `RuntimeError("the final split is blind during
   induction")`, so an accidental read raises loudly and with the right
   message.
3. `score_final.py` — the only file that reads `out/final/` — refuses to score
   if the corpus' `final_digest` differs from the digest it recomputes.
4. **`verify.py` asserts** (a) every corpus artifact carries
   `final_untouched: true`, (b) all shards withheld the same digest, (c) the
   blindness condition below, and (d) the digest recorded before induction
   equals the one at scoring time.

**(c), in the owner's wording, after a reconciliation (r3, 2026-09-11):**

> no induction-phase artifact contains **non-null** `gap 2` / `gap 3` episode
> data, derived statistics, or values read from those splits; **null / unopened
> provenance entries are required and permitted.**

The earlier wording — "no induction-phase artifact contains any `gap 2` /
`gap 3` field" — **directly contradicted §13's Rule 0**, which *requires*
`splits.gap2.final` and `splits.gap3.final` to be present-and-null so that the
blind splits are recorded as withheld rather than silently unmentioned.
`out/phase0.json` already carries both, so the old V5 would have failed the
artifact that best demonstrates the standard. Two guards promising incompatible
things is exactly the failure mode §13's registry exists to catch, so the
reconciliation is **recorded here and in `promises.json` (V5, amended in
place)** rather than edited silently. It was fixed **before V5 activated** — its
phase (`final`) has not run.

Point 4 also closes a hole: scaffold-induction's amendment A13 states that its
`verify.py` performs these checks, and `git grep -n final verify.py` on that
branch returns **nothing** — the check was never implemented. **This
contradicts that track's own record and is reported here rather than routed
around**; it is also why this track's V5 is a validity check with an artifact,
not a claim in prose.

## 6. Criteria — an existence claim, with cost reported beside it

All reported **per configuration**, never pooled. Each is decided by an exact
count with a certificate; a bound is never quoted as an exact count.

**The primary criterion is an existence claim, and it carries no cost
threshold.** This track sets no 2× or 10× cutoff, because no cutoff can be
justified before running (owner, 2026-09-11). Cost is measured exactly and
**reported faithfully beside** the existence result, where the reader applies
their own judgement. If blind widening wins by 20×, that is a devastating
practical result on its own and needs no pre-declared cutoff to be worth
recording.

**PRIMARY — did parametric induction happen?** All three of:

* **C1 — admission repair.** S1 has K > 0, certificate `complete`, on all 48
  `gap 0` episodes **and** all 48 `gap 1` episodes.
* **C2 — blind generalization.** S1 has K > 0, certificate `complete`, at
  `gap 2` **and** at `gap 3`.
* **C3 — the invariant of §1.2, established *structurally*.** "The final
  displacement was not in an induction pool, **therefore** it was re-derived
  from geometry" is not logically sufficient: the implication can hold by
  accident (owner, 2026-09-11). So C3 is a conjunction of three checks, the
  first two structural and the third the old novelty test kept as a **necessary
  condition, not the criterion**:

  * **C3a — a free geometry parameter exists in the object.** S1's AST /
    dataflow contains at least one node that is a declared geometry parameter —
    a stride, a gap, a relative coordinate — whose value is **not fixed during
    induction** and is bound per configuration. Recorded as the serialized AST
    with its free-parameter set named and typed.
  * **C3b — structure constant, emission varying.** Re-instantiating **the same
    schema** at `gap 0`, `gap 1`, `gap 2` and `gap 3` leaves the structure
    unchanged while the emitted concrete displacement changes as geometry
    predicts. Mechanically: `digest(structure(S1 @ gap g))` is **identical** for
    all four g, the emitted constants are **pairwise different** where geometry
    says they must be, and each matches the prediction of §2.4's law. This is
    §64's schema/vector distinction in a form a verifier can check: the schema
    is what stays fixed, the vector is what moves.
  * **C3c — the unseen-value condition (necessary, not sufficient).** At
    `gap 3` the conforming displacement is **not** an element of any pool S1
    instantiated during induction. Exact sets on both sides, integer token
    equality.

  Any rule shape can satisfy this — row stride, affine displacement,
  relative-coordinate operator, derived spatial-step function, indexed
  generator. A generator whose range was fitted on admission fails C3a (its
  parameter is not free) even if it happens to reach `gap 2`, and a schema that
  passes C3c by luck fails C3b (its structure would have to change to emit the
  new constant).

C1 ∧ C2 ∧ C3 is the claim. Nothing about cost enters it.

**REPORTED BESIDE IT — measured exactly, with no threshold.** Each is a number
in `out/headline.json` and a row in `RESULTS.md`, stated per configuration:

* **R1 — total cost against blind widening.** S1's induction cost (§8 measures
  1–5) plus downstream search cost (measure 6) against `S_blind`'s, which is
  charged its **full** total including the downstream search its larger pool
  forces. `S_blind` is a first-class comparator and a genuine falsifier: if it
  is cheaper, the induced structure did not earn its complexity.
* **R2 — total cost against the hand-written parametric schema.** §65's
  standing rule is that the obvious hand prior often reproduces the learned
  effect more cheaply; the comparison is reported, and its reading is stated in
  prose, without a pass mark.
* **R3 — did inherited outer knowledge change the outer search?** `S1_prior`'s
  outer cost against `S1`'s, and `D_prior`'s against both. The distractor can
  succeed by construction, so a null result here is informative.
* **R4 — did the outer loop erase the downstream saving?** Measure 3 against
  measure 6, per arm. Moving combinatorics up a level is not a win.
* **R5 — the flat substrate.** Whether `S_flat` reaches `gap 2` and `gap 3` at
  all (P2), and at what cost.

## 7. Outcomes and falsification — declared now, honoured whatever happens

### 7.1 The seven outcomes, each reported separately

The headline of `RESULTS.md` is **one of these ids**, not "pass" or "fail".
Several can hold at once — O3–O6 are about cost and are reported alongside
whichever of O0–O2 obtained. Each is decided mechanically from
`out/headline.json`.

| id | outcome | decided by |
|---|---|---|
| **O0** | **no repair found** — the outer loop produced no schema passing C1 | ¬C1 |
| **O1** | **repair is a frozen patch** — passes C1, carries the `gap 1` displacement as a fixed value | C1 ∧ ¬C3 |
| **O2** | **works at `gap 1`, not at `gap 2`/`gap 3`** — the blind split refutes it | C1 ∧ ¬C2 |
| **O3** | **blind widening is cheaper** — `S_blind` reaches `gap 2`/`gap 3` at lower total cost than S1 | R1 |
| **O4** | **the hand-written schema dominates** — `S_hand` reaches the same configurations at lower total cost | R2 |
| **O5** | **the outer loop erases the downstream saving** — measure 3 exceeds the measure-6 saving | R4 |
| **O6** | **induction** — C1 ∧ C2 ∧ C3, with R1–R5 reported beside it | C1 ∧ C2 ∧ C3 |

**O1 and O2 are the patching verdicts and they stand however good the `gap 1`
numbers are.** A frozen index — a generator whose range was fitted on admission
— is O1, however parametric it looks. No metric is swapped afterwards; any
corrected measurement is labelled exploratory.

**O6 is not weakened by O3, O4 or O5, and does not suppress them.** Induction
can happen and still be the wrong engineering choice; that is exactly the pair
of facts the record should carry, which is why the existence claim and the cost
comparison are separate (§6).

### 7.2 Stop conditions and standing commitments

* **F2 — design collapse (a stop condition, already checked).** If `WIDEN`
  **within S0's declared source L** repairs `gap 1`, the experiment has
  collapsed into "restore a candidate" and must be redesigned before any arm is
  read. **Checked in phase 0: it does not** (51 < 55). This says nothing about
  unrestricted widening, which does reach the answer and is the `S_blind` arm.
* **F3 — contradiction with §65.** If S0 shows a non-zero conformer count at
  `gap 1` over all 48 episodes, this track's premise is wrong and §65 is wrong.
  Halt and report the contradiction rather than proceeding.
* **F4 — no schema needed.** If `S_flat` reaches `gap 2` **and** `gap 3` at a
  lower total cost than S1, schema induction is refuted here: the flat
  substrate needed no repair. Keep as the headline beside the outcome id.
* **F5 — the prior carries nothing.** `D_prior`'s outer cost against
  `S1_prior`'s is reported as a ratio with no pass mark (R3). The distractor
  can succeed by construction, so a null result is informative rather than a
  failure of the experiment.
* **F6 — the grammar's effective width.** If `PARAMETERIZE`, `ADD_NODE` and
  `COMPOSE` produce zero repairs, the grammar is reported as having been
  effectively three families wide, and every arm as an ordering over that
  narrower space. (Inherited commitment: scaffold-induction A15.)
* **F7 — exhaustion, not sampling.** If an arm's space cannot be exhausted
  inside its declared wall-clock cap (§10), its certificate is `incomplete` and
  a **two-sided** bound is reported. §65 could not separate N″ from flat N
  because a one-sided sampled bound sat below the exact cost; that must not
  recur.
* **P4 (inherited prediction, scaffold-induction A16).** Structure-inventing
  edits become the only family that repairs anything here, making that corpus
  and this one exact complements. Reported whether or not it holds.

## 8. The six costs, reported separately

Per arm, per configuration, written to `out/cost_<arm>_gap<g>.json`:

1. `outer_edits_proposed` — edits the grammar enumerated.
2. `outer_edits_decided` — edited schemas actually conformance-decided
   (`proposed − undecided`); `undecided` is reported, never folded in.
3. `inner_programs_evaluated` — programs decided inside those decisions. The
   scaffold-induction corpus could only report an *upper bound* here because
   `decide()` never recorded the figure; **this track's `decide()` records the
   exact evaluated count from its first shard**, and the upper bound
   (`space × episodes`) is reported beside it.
4. `counterexamples_consumed` — admission episodes that produced a refutation
   and were fed back into proposal.
5. `wall_clock_seconds` and `peak_rss_gb`.
6. `downstream_cost` — expected programs to the first conforming program under
   the repaired schema at that configuration, exact `(S+1)/(K+1)` per tier, as
   a `Fraction`.

`total_cost = 3 + 6`. A learner that spends 10⁸ inner evaluations to save 10⁴
later has not won, and C4 is written so the record shows it.

## 9. Validity checks, run and reported before any criterion is read

* **V1 — the copy is faithful.** This track's copied `engine.py` reproduces
  §65's exact counts byte-for-byte: 686,985,984 (schema, `gap 0`, 48 episodes)
  and 0 (schema, `gap 1`), both `complete`; and 261,654,545,280 training
  conformers at `gap 1`. Any difference halts the track.
* **V2 — expressivity.** §2.2's table, recomputed. **Done; passes.**
* **V3 — widening *within S0's declared source* is insufficient.** §3.2's two
  integers. **Done; passes** (51 < 55, empty intersection). It is not, and is
  never to be stated as, a claim that widening cannot solve the task.
* **V4 — the grammar does not contain the answer.** §3.3's three checks.
* **V5 — blindness.** §5's four `verify.py` assertions, plus the digest
  recorded before induction.
* **V6 — the fast decider.** §3.4's ≥ 40-schema agreement check.
* **V7 — evaluator equivalence at the final configurations.** The fast
  evaluator matches `Program.execute` on a declared sample of programs at
  `gap 2` and `gap 3`, as §65's V1/V2 did at `gap 0`. Run at scoring time only.
* **V8 — the cost model.** §10's wall clock measured against step-pool size, so
  the budget below is measured rather than guessed. The *band*, not a point
  estimate: wall clock on this shared host is not reproducible and is not
  verified as though it were.
* **V9 — provenance.** §13's rule: every artifact under `out/` carries a stamp
  that agrees with the inputs, parameters and sources on disk now. An **absent**
  stamp fails exactly as a disagreeing one does.
* **S1 — the owner's permanent provenance standard.** §13's Rule 0: every
  artifact records the base scaffold digest, the split digests (including the
  blind ones it did not read), the mutation-grammar version digest, the
  producing commit and the pre-registration revision.
* **V11 — the guards bite.** `selftest_guards.py` drives every provenance guard
  to its failure state and each one must raise.
* **V10 — promises are checks.** §13's registry: `check_promises.py` exits
  non-zero if any promise whose phase has run lacks an implementing `verify_*`
  function, or if a committed promise was deleted.

Verifiers compare **typed fields**, not prose; numeric checks use token
equality; any process selection uses PID plus `/proc/<pid>/cwd`. Substring
matching is banned in this track — it has produced three separate false-pass
bugs in this project.

## 10. Compute budget, estimated before running

**The cost model is measured, not guessed** (V8, `out/phase0.json`,
`V8_cost_pilot`). One exact 48-episode `engine.Counter` run at `gap 0`, with
S0's non-STEP pools and a STEP pool of n candidates:

| n step candidates | conformers (exact, 48 episodes) | seconds | peak RSS |
|---|---|---|---|
| 20 (S0) | 1,376,372,736 | 83.40 | 0.371 GB |
| 40 | 2,767,343,616 | 103.44 | 0.371 GB |
| 80 | 11,224,903,680 | 151.44 | 0.371 GB |

Least squares over those three points gives `seconds ≈ 59.4 + 1.14 × candidates`
with residuals under 1.7 s. **The fit is not a reproducible claim and is not
verified as one.** The pilot was run twice on this shared host; the exact
conformer counts were identical both times, and the wall clock was not
(the earlier run fitted `66.8 + 1.00 × candidates`). §10's verified claims are
therefore the *band* and the *shape*, not the coefficients:

* intercept in [45, 85] s — the route/class/matcher-table enumeration, which no
  arm in this track changes;
* slope in [0.8, 1.4] s per step candidate — `Counter.route_products` in direct
  mode;
* the cost is **affine, not superlinear**: the 80-candidate point sits within
  10% of the extrapolation from 20 and 40;
* peak RSS < 0.5 GB, hence `kit.check_floor`'s requirement (20× peak) under the
  declared `MemoryMax=8G`;
* the three exact conformer counts above, by token equality — the reproducible
  quantity the pilot also produces.

An independent anchor sits inside the band: 85.58 s at n = 20 in
`integrated-flagship/out/generalize_gap0.json`. Every estimate below carries a
±20% wall-clock tolerance and is rounded up.

**Per arm** (4 configurations each; the outer loop runs only at the two
admission configurations):

| arm | step candidates | exact counts | outer loop | total |
|---|---|---|---|---|
| `S0` | 20 | 4 × ~85 s ≈ 6 min | — | **≤ 8 min** |
| `S_widen` | ≤ 92 | 4 × ~165 s ≈ 11 min | ~500 edits × 2 × fast decide ≈ 4 min | **≤ 18 min** |
| `S_hand` | ≤ 96 | 4 × ~170 s ≈ 12 min | — | **≤ 14 min** |
| `S1` | ≤ 96 expected | ~12 min | ~4 min | **≤ 20 min** |
| `S1_prior` | ≤ 96 expected | ~12 min | ~4 min | **≤ 20 min** |
| `D_prior` | ≤ 96 expected | ~12 min | ~4 min | **≤ 20 min** |
| `S_blind` | 3,068 | 4 × ~3,600 s ≈ **4 h** | — | **≤ 5 h**, capped 2 h/configuration |
| `S_flat` | 1,280 | band says 4 × ~1,600 s ≈ 1.8 h, **but the band does not apply** (see below) | — | capped **2 h/configuration** |

Plus: final-split cache ~25 s (12 s per configuration, measured); V6 (40
schemas × fast decide + 40 exact counts at ≤ 96 candidates) ≈ 1.8 h; V1, V4,
V5, V7, scoring, `report.py`, `verify.py` ≈ 1 h.

**Headline path** — `S0`, `S1`, `S1_prior`, `S_widen`, `S_hand`, `D_prior`,
plus V6 and the verifiers — **≤ 5 h** of capped single-worker compute, peak RSS
well under 0.5 GB.

**Two honest caveats on the controls.**

* The affine band is measured with S0's ADDR/LIT/MATCH pools held fixed. It
  applies to `S_blind` (which changes only STEP) and **not** to `S_flat`, whose
  ADDR pool admits arity-1 operators and whose MATCH pool is 2,304 rather than
  256 — both feed the fixed term. §65 never exhausted the flat arm's
  48-episode count and reported a sampled bound instead. `S_flat` is therefore
  budgeted as *unknown, capped at 2 h per configuration*, and is expected to be
  reported `incomplete` with two-sided bounds (F7).
* The outer-loop estimate assumes V6 validates the fast decider. If it does
  not, each arm's outer loop becomes ~500 edits × 2 configurations × ~160 s ≈
  **44 h per arm**, which exceeds any budget this track should spend: in that
  case the track stops and the design is brought back for re-approval rather
  than run at reduced coverage.

## 11. Evidence standard and resources

* Every figure in `RESULTS.md` is rendered from `out/` by `report.py --fill`
  into `<!-- BEGIN:x -->` blocks; `verify.py` re-renders and compares, and
  fails on any number in prose outside the structural allow-list.
* `verify.py` writes `out/verify.json` and `out/headline.json` for the merge
  gate and exits non-zero on any FAIL.
* Negative results and invalid logs are preserved, not deleted.
* Every heavy job runs as
  `systemd-run --user --scope -q -p MemoryMax=<20× measured peak> -p CPUQuota=400% env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 <cmd>`.
* `kit.check_floor` / `kit.check_workers` (copied from
  `scaffold-induction/kit.py`) gate every phase: MemAvailable ≥ max(2 GB,
  20 × measured peak RSS) with ≥ 1 GB headroom, at most 4 workers, workers
  identified by PID + `/proc/<pid>/cwd`. Every start and refusal is logged to
  `out/resource_gate.log` and reported.
* The host is shared. `MemAvailable` was 20.6 GB when phase 0 ran, with 13 GB
  of swap in use; available memory, not the 121 GB total, is the budget, and it
  is re-read before every phase. One heavy phase at a time.

## 12. What is inherited, and from where

| from | what | how |
|---|---|---|
| `research/integrated-flagship` | `env.py`, `family.py`, `engine.py`, `cache.py` — the task, S0, the exact counter and evaluator | copied into this track, V1 asserts the copy reproduces §65's counts |
| `research/scaffold-induction` (branch `scaffold-induction`) | `edits.py` (the `Edit` dataclass, `enumerate_edits`/`apply_edit`, A11's no-truncation rule), `kit.py` (`check_floor`, `check_workers`, `measured_peak`), the three-split deletion + digest pattern, `score_final.py`'s digest refusal, `report.py`'s outer-loop cost block | copied; **that track's files are not edited** |
| `research/visual-ladder` | `rung3_widgets.offset_pool` | read, not modified |

No change is proposed to `tcn/` or `generators/`. A construct is promoted to
core only when several domains need it; nothing here qualifies yet.

## 13. Provenance, and promises that are checks

Two failures from the scaffold-induction track — both found *inside* the
machinery built to prevent this class of error — set the rules here. Assume
they do not exhaust the class.

**The incident.** A validation JSON describing a superseded corpus reappeared
in `out/` *after* the directory was cleaned, because the superseded job was
still running and wrote afterwards. Cleaning a directory cannot remove
artifacts that have not been produced yet. It was nearly reported as current,
and the verifier would have accepted it: its checks asserted "a validation ran
and reported zero mismatches", never "it validated **this** corpus".

**Rule 0 — the owner's permanent provenance standard (2026-09-11).** Every
evidence artifact carries, as named fields that must be *present*:

| field | what it pins |
|---|---|
| `base_digest` | the base scaffold / program the artifact is about |
| `splits_digest` | the split / corpus digests, **including splits deliberately not read** — the blind final split's digest is recorded here, unopened |
| `grammar_digest` | the mutation grammar's version digest (and its family list) |
| `commit` | the producing commit |
| `revision` | the pre-registration / amendment revision, plus the digest of `PREREGISTRATION.md` itself |

An **explicit absence with a reason** is acceptable where an omission is not:
`out/phase0.json` records `grammar_digest.digest = null` with
`absent_because: "no mutation grammar existed when this artifact was produced"`,
because a missing field reads as "not recorded" and would let a later artifact
inherit the silence. `verify_s1_provenance_standard` checks all five on every
artifact.

**Rule 1 — every derived artifact carries a fingerprint of what it was derived
from.** `stamp.py` embeds a `provenance` block in every artifact this track
writes: the five standard fields above, plus the digest of every input file,
the digest of each *split* computed from the fields a decision could depend on,
the declared parameters (pool contents and sizes, gaps, seeds, configuration)
and their digest, the digest of every source file whose behaviour produced it,
and the git HEAD with its dirty flag. `stamp.require` fails when the block
**disagrees** with disk and equally
when it is **absent** — otherwise every artifact predating this rule silently
counts as current. It also fails when called with nothing to compare: an
assertion that checks nothing must fail, not pass.

**Rule 2 — re-run to obtain a stamp; never back-fill one.** There is no
function in `stamp.py` that adds a block to an existing artifact, and
`stamp.write` refuses a payload that already carries one. A hand-added stamp
asserts exactly what it cannot check. The `out/phase0.json` committed with this
document was **re-run, not annotated**, after this rule was adopted.

**Rule 3 — a promise is a check, shipped in the same commit.** Amendment A13 of
the scaffold-induction track promises that its `verify.py` enforces the blind
split's fingerprint, `final_untouched`, and train-only ordering; that file
contains no such check, and nothing failed because of it. So this track keeps
`promises.json`: every mechanical check this document promises, with the phase
that activates it and the name of the `verify_*` function that implements it.
`check_promises.py` fails when

1. a promise whose phase has **run** has no implementing function in
   `verify.py`;
2. a promise committed earlier has been **deleted** — promises are amended,
   never removed;
3. a promise's artifact exists but carries **no** provenance stamp, or one that
   disagrees with disk;
4. the registry is malformed, or a promise names no declared phase.

A phase counts as run when its marker exists under `out/`, and markers are
written by the phase's own job — `check_promises.mark_phase` — never by hand.

**Rule 4 — a guard nobody has seen fail is a guard nobody has tested.**
`selftest_guards.py` drives each provenance guard to its failure state — absent
block, unknown format, missing field, empty inputs, parameters edited after
writing, an input digest disagreeing with disk, an unnamed input, a changed
source, the wrong kind, a comparison with nothing to compare, an overwrite of
an existing block, a missing input file — and asserts each one raises. 13 cases,
all refusing. `verify_v11_guards_bite` runs it, so the guards cannot rot into
no-ops.

**Shipped with this document**, because `phase0` has run: `verify_v2_expressivity`,
`verify_v3_widen_insufficiency`, `verify_f2_no_design_collapse`,
`verify_v8_cost_model`, `verify_v9_provenance`, `verify_v10_promises_kept` and
`verify_v11_guards_bite`. The remaining 20 promises are `pending` against the
`grammar`, `outer` and `final` phases, and each becomes a hard failure the
moment its phase's marker appears. `verify.py` records two narrow, named exemptions from the provenance
scan — its own `verify.json` and `headline.json`, which the run in progress
overwrites — and reports the exemption as a claim so it cannot become a silent
hole.

**V9** and **V10** are added to §9's validity checks and run before any
criterion is read.

## 14. Limitations, declared before the result

* **One slot, one relation.** Measured in §2.2: `left_of` and `right_of` remain
  expressible in S0 at `gap 1`, and are predicted to stay expressible at
  `gap 2` and `gap 3` with offsets of the form 3·i. The entire experiment
  therefore turns on a single relation (`above`) and a single slot family
  (STEP). Whatever it shows is an existence claim about one defect of one
  schema, not a general result about schema induction. It will be reported that
  way, with the per-configuration, per-relation table, never a pooled mean.
* **The discovery is small.** "Displacements are multiples of the row stride"
  is a one-line statement. That is deliberate — §65's lesson is that inherited
  objects lose to one-line hand rules — but it means a pass on C1–C3 shows the
  outer loop can find a small parametric family, not that it can find a large
  one.
* **One episode source.** One seed set, one palette, one relation vocabulary,
  one resolution (16), 12 training and 36 held-out episodes per configuration,
  four configurations. The same single-configuration caveat §65 carries.
* **The band-structure prediction (§2.4) is derived from `gap 0` and `gap 1`
  plus `env.neighbour`.** If the `gap 2` / `gap 3` draws produce a smaller
  minimum target height than 3, P1–P3 may shift; they are stated as falsifiable
  predictions and will be reported as measured against.

# Amendments, written before any arm ran

## r3 — 2026-09-11, three fixes required before the arms are released

* **B1 — fail closed on a dirty producer.** `r2`'s `out/phase0.json` named
  `3e86052` with `dirty: true`. Source hashes mean the evidence is not lost,
  but a dirty tree is not an exact snapshot and the commit field then pins
  "some tree state" rather than a real one. `stamp.present` now **refuses** any
  artifact whose `commit.dirty` is not `false`, and refuses any artifact
  produced by an **untracked source** (which is in no commit at all). Required
  workflow, and the one used to produce this revision's artifacts: commit the
  code and the pre-registration → re-run phase 0 from the clean commit →
  commit the outputs. *Definition:* `dirty` counts **tracked** modifications
  only, because a run necessarily creates untracked files — its own outputs —
  and counting those would make a clean run impossible; the hole that opens is
  closed by the untracked-source refusal.
* **B2 — V5 and the provenance standard contradicted each other, reconciled in
  the open.** V5 promised that no induction-phase artifact contains a `gap 2` /
  `gap 3` field; Rule 0 *requires* `splits.gap2.final` and `splits.gap3.final`
  to be present-and-null. `out/phase0.json` already carried both, so the old V5
  would have failed the artifact that best demonstrates the standard. The
  owner's wording replaces it (§5 point 4c), and the reconciliation is recorded
  in `promises.json` (V5, amended in place) rather than edited silently. Fixed
  **before V5 activated** — its phase has not run. Two guards promising
  incompatible things is precisely the failure mode the registry exists to
  catch, and this is the registry catching one.
* **B3 — C3 is now structural, not an inference from novelty.** "The value was
  unseen, therefore it was re-derived from geometry" can hold by accident. C3
  becomes C3a ∧ C3b ∧ C3c: a **free geometry parameter** in the AST/dataflow;
  **structure digest constant across instantiation at `gap 0…3` while the
  emitted constant varies as geometry predicts**; and the unseen-value test
  kept as a **necessary condition** beside them. This is §64's schema/vector
  distinction in the only form a verifier can actually check — the schema is
  what stays fixed, the vector is what moves.

**What a positive result would and would not mean.** It would **not** mean "the
answer was restored". It would mean **a missing computational schema was
induced from a counterexample and then instantiated correctly outside the
induction range.** `RESULTS.md` states it in exactly those terms, and reports
each of O0–O6 on its own merits.

## r2 — 2026-09-11, the owner's rulings

Approved: unrestricted widening stays a first-class comparator; the scope
ruling (one slot, one relation family, one resolution, an existence result)
stands as written, with `left_of` / `right_of` staying expressible treated as a
*feature*, because `above` isolates the missing-structure axis. Replication is
the next experiment, not a prerequisite for this one. Four changes were
required and are made here. **No arm had run; nothing is retrofitted.**

* **A1 — the widening claim is always scoped.** Never "widening cannot solve
  the task"; always "widening **within S0's declared source** cannot". The
  meaningful claim is that a structure-generating move is required *relative to
  that starting representation*. `S_blind` is therefore a genuine falsifier,
  charged its full total cost: if brute widening is cheaper than induced
  structure, the induction did not earn its complexity (§3.2, §4, R1/O3). The
  qualifier is now enforced in the artifact, not only the prose —
  `out/phase0.json` carries `scope` and `does_not_claim` fields and `verify.py`
  checks both.
* **A2 — the induction criterion is an invariant, not a mechanism** (§1.2).
  *bad:* S1 contains the `gap 1` numerical answer. *good:* S1 contains a rule
  whose concrete displacement is re-derived from the new instance's geometry.
  Any equivalent general solution counts — row stride, affine displacement,
  relative-coordinate operator, derived spatial-step function, indexed
  generator. `PARAMETERIZE` is demoted from definition to one mechanism.
  Frozen-index failure remains patching, and `gap 2` / `gap 3` still decide.
  This is §64's lesson one level up: a concrete instantiation is evidence about
  a schema, never the schema itself.
* **A3 — every invented cost threshold is removed** (§6). The 10× (C4), 2×
  (C5) and "strictly below" (C6) cutoffs could not be justified before running
  and are gone. The primary criterion is now the existence claim C1 ∧ C2 ∧ C3;
  cost is measured exactly and reported faithfully beside it as R1–R5, with no
  pass mark. The four affected promises are **amended in place, not deleted**,
  and record what changed.
* **A4 — the claim block is stated explicitly** (§1), and the failure modes
  become seven distinct reported outcomes **O0–O6** (§7.1) rather than a single
  "fail". The headline of `RESULTS.md` is an outcome id. O6 (induction) neither
  suppresses nor is weakened by O3–O5 (the cost outcomes): induction can happen
  and still be the wrong engineering choice, and the record should carry both.
* **Also adopted: the owner's permanent provenance standard** (§13 Rule 0) —
  base scaffold digest, split/corpus digests including the blind ones, mutation
  grammar version digest, producing commit, pre-registration revision. This
  document's revision is `r2-2026-09-11-owner-rulings`; artifacts carry it and
  `verify.py` fails when an artifact's revision differs from the code's.
