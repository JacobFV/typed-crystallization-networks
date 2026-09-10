# Does §47's pre-hoc scaffold probe generalise? No — the sweep was doing the work

FINDINGS §47 Q3 found, against the supervising session's expectation, that a
cheap pre-hoc signal identified the §45 scaffold's defect from the failed
scaffold, its exhaustion certificate and 24 training episodes alone: the
terminal accumulator `acc21` was **literally constant** on the training batch
(`distinct_values: [0]`, **0.0** bits against **1.0** bits of label entropy), and
a training-only sweep returned `reduce_min`/`min` at held-out **1.000 on n=859**.
§47 stated its own limit: *"Whether the same probe generalises beyond a
constant-node diagnosis is untested and should not be assumed."*

This track tests that. `PREREGISTRATION.md` was committed at `0439bfb`
**before any arm ran**; `out/predictions.json` was committed at `66ad7fd`
**before `score.py` existed**; the addendum was committed at `4c29b85` **before
the arms it governs**. Nothing under `tcn/` or `generators/` is touched —
`git diff main...HEAD -- tcn/ generators/` is **empty** — and no operator is
added to core.

---

## Verdict, up front

**The signal does not generalise, and the falsification the brief named as most
likely is the one that fired.** On 32 scaffolds — 24 failed, 8 that contain a
solution — across three distinct correct answers:

| method | repairs found (of 24 failed scaffolds) | declined | confidently wrong | proposed a change on a scaffold that already contains a solution |
|---|---|---|---|---|
| **the probe** (information diagnosis → sweep at the flagged site) | **4** | 20 | 0 | **8 of 8** |
| the same probe restricted to input-dependent nodes | **0** | 24 | 0 | 8 of 8 |
| **B-random** — uniform draw from the same candidate set | **5.17 expected** | — | — | — |
| **B-sweeponly** — the sweep with the diagnosis **deleted** | **15** | 9 | 0 | 8 of 8 |
| **B-allholes** (exploratory) — sweep every hole, no diagnosis at all | **21** | 2 | 1 | 8 of 8 |

**The probe does not beat picking an operator at random** (4 against an expected
5.17, and the random null is *generous* — see §4). **Deleting the diagnosis
entirely and sweeping raises the score to 15; deleting localisation as well and
sweeping every hole raises it to 21.** All four of the probe's successes are
cases where no member of the scaffold runs at all, so the "diagnosis" is a
crash, not an information measurement.

**It is worse than a narrow constant-detector: as a gate it is unusable.** Stage
D fired on **8 of 8** scaffolds that provably contain a conforming program,
including §44's arm 3 (144 conforming, certificate `complete`) — and for a
*structural* reason, not by accident: in that solved program the two operands of
the final `xor` carry **exactly 0.0 bits** of label information each, while the
output carries **1.000**. Zero mutual information with the label is what a
correct XOR-structured program *looks like*. §47's `acc21` had 0 bits and was a
defect; §44's `n1`, `n2` have 0 bits and are the solution. Nothing local to a
node distinguishes them.

**§47's own numbers reproduce exactly** — they were verified here before being
bounded (§5). What §47 did not record, and what this track measures, is that
the constant-node observation is **not selection-invariant**: `acc21` is
constant at §45's *semantically honest* counting program (`c=101`), and that is
**not** the best-on-training member (`c=110`, train 0.75). A probe that must
choose its member by the only label-light rule available — training accuracy —
does not see the constant node at all.

**Honest denominator.** Two recorded tasks carry an independently established
repair (§45/§47's Dyck scaffold and §44/§46's Boolean scaffold). The other 27
scaffolds are constructed on the Dyck template with three distinct answers
(`{min,max}`, `{add,sub}`, and the Boolean `{MAJ3}`). **This is an anecdote
about a bounded tool, not a capability measurement**, and the negative is the
part that carries weight: a probe that fires on every solvable scaffold in the
set cannot be used as a gate no matter how many more cases were added.

---

## 0. Setup, and what reproduces before anything is claimed

**Stream.** `hardening` is pinned explicitly at every `common.dataset` call
(§39's defect). Language cases use `hardening='context_free_language'`
(post-audit) except `S3`, which uses `hardening='none'` (pre-audit) and says so
in every table. Boolean cases use the complete 64-row truth table — no sampling
and no stream.

**Provenance.** Seven modules are vendored byte-identical from
`research/dyck-learnability` so the numbers are comparable to §47's by
construction: `common.py` (`c8f856f3…`), `splits.py` (`c04acb42…`),
`prepare.py`, `family.py` (`a6e20426…`), `accum_scaffold.py`,
`dyck_scaffold.py`, `run_stage_b.py` (`ad06fa06…`). The Boolean cases import
`research/earned-abstraction/{later,arms}.py` unmodified and build their arms
through `arms.build`.

**Split, re-drawn here.** Train {10,12,14} n=**24** (majority 0.5000); held-out
seen lengths n=**120** (0.5333); held-out **unseen** lengths {16,18,20,22}
n=**859** (majority **0.5262**, random 0.5). Per-length counts identical to
§45's and §47's; `counts_match_rate` **1.000** on every split, which is the fact
that makes a bracket count useless post-audit.

**Scaffold identity, asserted not assumed.** This track's two-hole builder
`scaffolds2.stage_b_gen` reproduces the shipped scaffolds by `Program.digest`:

| scaffold | recorded builder | this track | identical | space |
|---|---|---|---|---|
| counting, positions 16 | `language-capability/scaffolds.stage_b` | `28ebd9d6210d29df` | **yes** | 45,375 |
| counting, positions 22 | same | `682e10e573f704b0` | **yes** | 45,375 |
| two-accumulator, `fold=min` | `dyck-learnability/accum_scaffold` | **`a7aa228bddbeb1cd`** | **yes** | 680,625 |

`a7aa228bddbeb1cd` is the digest §47 itself asserted.

**Simulators, validated against the real typed program.** §45's `bound.py`
discipline in §47's per-episode form: `sim2.validate` compares the simulator to
`Program.execute` episode by episode, on both the answer and the raise/no-raise
verdict. **81 language variants** (every `(acc_fold, second_fold)` pair touched,
at both position counts) × 40 random members × 24 episodes: **0 mismatches,
every variant.** The Boolean bitmask simulator: 150 random members × 64 examples
per case: **0 mismatches**.

**Recorded numbers re-derived here before use.**

| recorded | source | here |
|---|---|---|
| counting family: space 45,375, exhausted, **0 conforming**, `complete` | §45 | space 45,375, usable 42,000, **0 conforming**, `complete` |
| best train **0.7500**, **24** members tied | §45 / §47 | **0.7500**, **24** tied |
| min-prefix family: **110 conforming**, `complete` | §47 Q1 | **110**, `complete` |
| first conforming at index **571,746 of 680,625 = 84.00%** | §47 Q1 | **571,746 / 680,625 = 84.00%** |
| that member: `c=101, plus=+1, minus=−1`, 1.000 / 1.000 / **1.000 on n=859** | §45 / §47 | identical, on the **real** program via `Program.execute`, per-length 1.000 at 16/18/20/22, majority 0.5262 |
| per-node information at §45's counting program: `acc2` 0.3154, `acc4` 0.2749, `acc8` 0.2217, `acc13…acc21` **0.0000** | §47 `q3_prehoc.json` | `acc2` **0.3154**, `acc4` **0.2749**, `acc8` **0.2217**, `acc13…acc21` **0.0000** |
| tight Boolean: arm 1 **0/230,400** `complete`; arm 3 **144/2,709,504** `complete`; arm 4 **0/2,709,504** `complete` | §44 | **0/230,400**; **144/2,709,504**; **0/2,709,504**, all `complete` |

Everything §45, §47 and §44 recorded that this track touches reproduces. The
disagreement below is not with their numbers; it is with what those numbers
support.

---

## 1. The probe, and the two stages held apart

`probe.py`, pinned in `PREREGISTRATION.md` before running.

**Stage D — diagnosis.** Select the failed scaffold's best member on the 24
training episodes (ties by enumeration order; training labels only). Execute the
**real typed program** under it and record, per node, `distinct`, `H(node)`,
`I(node; label)` and `H(label)`. **Fire** iff some node is constant, or carries
zero label information while the label carries ≥ 0.5 bits. **Site** = the fired
node of greatest depth.

**Stage S — sweep.** Read the site's signature off the registry, enumerate every
core operator that `registry.resolve` accepts at it (nothing curated, nothing
added), substitute each in the site's chain, refit the scaffold's free
parameters on **training only**, rank by best training accuracy. Prediction =
the tie set at the top; correct only if that set is contained in the known
repair set.

Two repair families, both declared in advance: **swap** (change the operator in
the site's accumulator chain) and **fold** (§47's template — add a second
accumulator over the chain feeding the site). The *template* is a human input;
which operator fills the hole is what the sweep decides.

**The controls exist so that "the sweep is doing the work" is a measurement.**
B-sweeponly is Stage S at a fixed site (the output-adjacent accumulator) with
Stage D deleted. B-terminal is "always blame the output-adjacent node".
B-random is a uniform draw from the same candidate set. B-allholes (exploratory,
added after the pre-registered arm) deletes localisation too and sweeps every
hole.

---

## 2. The case table, predictions made before checking

`out/predictions.json` (committed at `66ad7fd`) and `out/predictions_c2.json`
(committed at `5d32902`) were both written by `run_probe.py`, which imports
`cases` and never `answers`. `score.py` is the only file that imports the
answers and it ran afterwards.

`solves@` is the best **training** accuracy the method's proposal reaches;
"known" is the repair established independently (recorded) or by exhaustive
enumeration here.

| case | contains a solution? | known repair | Stage D fired | Stage D site | probe's proposal | B-sweeponly | B-allholes | probe correct |
|---|---|---|---|---|---|---|---|---|
| `R1_counting_postaudit` (§45) | no, `complete` 0/45,375 | `min`,`max` | yes | `m21` | — declined | `min`,`max` @1.000 | `min`,`max` @1.000 | **no** |
| `R2_tight_flat` (§44 arm 1) | no, `complete` 0/230,400 | `MAJ3` | yes | `n2` | all 7 BOOL ops @0.625 | 5 ops @0.625 | 7 ops @0.625 | **no** |
| `R3_tight_wrong_module` (§44 arm 4) | no, `complete` 0/2,709,504 | `MAJ3` | **no** | — | — | `xor` @0.656 | `xor` @0.656 | **no** |
| `S1`≡`C_bal_min` (§47) | **yes**, 110 `complete` | — | **yes (FP)** | `total_ok` | — | `min`,`max` @1.000 | `max`,`sub` @1.000 | n/a |
| `S2_tight_maj3` (§44 arm 3) | **yes**, 144 `complete` | — | **yes (FP)** | `n2` | `and`,`or` @0.750 | `xor` @1.000 | `xor` @1.000 | n/a |
| `S3_counting_preaudit` (§19) | **yes** | — | **yes (FP)** | `m15` | — | `add`,`sub` @1.000 | 5 folds @1.000 | n/a |
| `C_bal_{add,sub,mul,shl,shr}` | no | `min`,`max` | yes | `total_ok`/`min_ok` | — declined | `min`,`max` @1.000 | `min`,`max` @1.000 | **no** ×5 |
| `C_bal_{idiv,mod}` | no (0 usable members) | `min`,`max` | yes (**rule D0**) | `lo0` | `min`,`max` @1.000 | `min`,`max` @1.000 | `min`,`max` @1.000 | **yes** ×2 |
| `C_bal_{min,max}` (`C_bal_min` is `S1`) | **yes**, 110 each | — | **yes (FP)** ×2 | `total_ok` | — | `min`,`max` @1.000 | @1.000 | n/a |
| `C_max2_{add,sub,mul,shl,shr}` | no | `min`,`max` | yes | `total_ok`/`min_ok`/`answer` | — declined | `min`,`max` @1.000 | see §6 | **no** ×5 |
| `C_max2_{idiv,mod}` | no (0 usable) | `min`,`max` | yes (**D0**) | `lo0` | `min`,`max` @1.000 | `min`,`max` @1.000 | `min`,`max` @1.000 | **yes** ×2 |
| `C_max2_{min,max}` | **yes**, 527 each | — | **yes (FP)** ×2 | `total_ok` | — | @1.000 | @1.000 | n/a |
| `C2_bal_{idiv,max,min,mod,mul,shl,shr}` | no | **`add`,`sub`** | yes | `total_ok`/`min_ok` | — declined | 0.583–0.833, **declines** | **`add`,`sub`** @1.000 | **no** ×7 |
| `C2_bal_{add,sub}` | **yes**, 110 / 29 | — | **yes (FP)** ×2 | `total_ok` | — | `min`,`max` @1.000 | @1.000 | n/a |

Totals: **32 scaffolds, 24 failed, 8 solvable controls, three distinct correct
answers** (`{min,max}`, `{add,sub}`, `{MAJ3}`).

---

## 3. Stage D: what the information table actually says

**It fires on everything.** Across the 28 language scaffolds, Stage D flags
between **42 and 124** of the scaffold's **84–138** nodes. On the three-node
Boolean scaffolds it flags 2 of 3 (`R2`, `S2`) or 0 of 3 (`R3`). A rule that
selects roughly half the graph is not a localisation.

**Why it fires so widely, measured.** Three separate mechanisms, all
label-independent:

1. *Chosen constants.* `plus` and `minus` are `identity` over a searched
   constant. They are constant on any batch, by construction. The exploratory
   "input-dependent nodes only" variant (`probe.input_dependent`) removes these
   mechanically — and changes **no** verdict: it still fires on 8 of 8 solvable
   controls, and its repair score falls from 4/24 to **0/24**.
2. *Padding.* Training strings are 10–14 symbols; the scaffold materialises 22
   positions. `in15…in21` and `m15…m21` are therefore constant on the training
   batch for every member. On `R1` this is exactly what the pinned "deepest fired
   node" rule selects: **`m21`**, a `mux` at depth 4, whose signature admits no
   core binary operator at all — so Stage S has nothing to sweep and the probe
   declines.
3. *Vacuous readouts.* `total_ok` reads the total accumulator, which is 0 on
   every post-audit episode (`counts_match_rate` 1.000), so most readouts over
   it are constant-true. That is a real degeneracy — it is §47's signal one node
   downstream — and it is present identically in the scaffolds that **do**
   contain a solution.

**The false positive is structural, not incidental.** §44 arm 3 (`S2`) contains
144 conforming programs, certificate `complete`. Its own best-on-training member
*is* one of them — train 1.000 — and its per-node table reads:

| node | operator | distinct | `I(node; label)` |
|---|---|---|---|
| `n1` | `module:8ceedf7b…` (MAJ3) | 2 | **0.0000** |
| `n2` | `module:8ceedf7b…` (MAJ3) | 2 | **0.0000** |
| `y` | `xor` | 2 | **1.0000** |

`maj(a,b,c)` and `maj(d,e,f)` are each independent of their `xor`. **A correct
program's internal nodes carry zero label information here.** Any rule that
reads "0.0 bits" as "this node cannot be the discriminator" flags a solved
scaffold. This is the single sharpest reason the probe cannot be a gate, and no
amount of extra cases changes it.

**And it misses a real failure.** `R3` (§44 arm 4, the wrong inherited module,
**0** conforming of 2,709,504, `complete`) is the one failed scaffold where
Stage D does **not** fire: `I(n1;label) = I(n2;label) = 0.0121`, non-zero, so
nothing flags. A failed scaffold and a solved scaffold differ here by 0.0121
bits, in the direction that makes the *failed* one look healthier.

**F5 — localisation adds nothing, and is worse than the trivial baseline.**
Stage D's site equals B-terminal's "output-adjacent accumulator" in **0 of 32**
cases. B-terminal's site is the site at which B-sweeponly scores 15/24; Stage
D's site is where the probe scores 4/24. The information table does move the
site — it moves it to a worse one.

---

## 4. F3 — the sweep is doing the work, measured three ways

This is the falsification the brief flagged as most likely, and it fires
unambiguously.

| method | what it uses | repairs found of 24 |
|---|---|---|
| **B-random** | nothing; uniform draw from the signature-matching candidate set | **5.17 expected** |
| **the probe** | per-node information → sweep at the flagged site | **4** |
| **B-sweeponly** | *no diagnosis*; sweep at a fixed site | **15** |
| **B-allholes** | *no diagnosis and no localisation*; sweep every hole | **21** |

The random null is `|known repairs| / |candidates|` per case: 2/9 on 22 language
scaffolds, 1/7 on the two Boolean ones — **generous**, because `MAJ3` is not in
the Boolean candidate set at all, so the strictly correct Boolean term is 0 and
the honest expectation is **4.88**. Either way the probe does not beat it.

**All four of the probe's successes are `rule D0`** — `C_bal_idiv`, `C_bal_mod`,
`C_max2_idiv`, `C_max2_mod`, where the pinned fold divides by the second
accumulator's seed `0` and **no member of the 680,625-program space runs on the
training batch** (`usable_members_on_train` = 0). There the "diagnosis" is a
crash traced to `lo0` by bisection, and any sweep from that site finds
`min`/`max`. **On every failed scaffold where the program actually runs, the
probe's information diagnosis contributes zero repairs.**

**What the diagnosis was supposed to buy is localisation, and B-allholes shows
localisation is not scarce.** Sweeping *every* hole means at most **17–18** extra
scaffolds for a language case and **21** for a Boolean one — each decided
exhaustively in well under a second by a validated simulator. Brute force over
holes is cheaper than the diagnosis it would replace.

---

## 5. §47's number reproduces — and is not selection-invariant

Arm A3 runs the identical Stage D at §47's own member, §45's semantically honest
counting program `c=101, plus=+1, minus=−1`:

| | §47 `q3_prehoc.json` | here |
|---|---|---|
| `acc21` distinct values | `[0]` | **`[0]`** |
| `acc21` information | **0.0** bits | **0.0** bits |
| label entropy | 1.0 bits | **1.0** bits |
| `acc2` / `acc4` / `acc8` | 0.3154 / 0.2749 / 0.2217 | **0.3154 / 0.2749 / 0.2217** |
| `acc13…acc21` | 0.0000 | **0.0000** |

**The recorded number is exactly right.** What it does not survive is being
made mechanical:

- **Member selection.** §47's table is computed at `c=101`. The **best-on-training**
  member of that same family is `c=110, plus=−2, minus=+1, rule le −2`, train
  0.7500 — where `acc21` takes **7** distinct values and carries **0.370** bits,
  the exact figure §47 itself recorded as `a1_best_terminal_accumulator_information`.
  §47 reports both numbers; it does not say which member a mechanical probe
  should use, and the only label-light rule available (best on training) gives
  the one where the signal is absent.
- **Node selection.** Even at `c=101`, **59 of 114** nodes satisfy the fire rule
  and **nine of them** — `acc13` through `acc21` — are tied at exactly 0.0 bits.
  The information table cannot pick `acc21` out of its own accumulator chain,
  let alone out of the scaffold. §47's table was over the accumulator chain,
  chosen by a human who had read §45; the pinned "deepest fired node" rule
  selects `answer` instead.

So the honest reading of §47 Q3 is narrower than §47 wrote it: **the evidence
was there, in a table a human had already narrowed to the right nine nodes.**
The part that is mechanical and does the work is the sweep — which §47 also
reported, and which this track confirms is where the whole result lives.

---

## 6. Two disclosures this track makes about itself

**The constructed `max2` task's declared answer was wrong, and enumeration said
so.** `PREREGISTRATION.md` declared `max2`'s repair as `{max}` "fixed by
construction" and committed to letting the nine-fold sweep establish it. The
sweep returned `{min, max}`: sign-flipping the step values turns a running
minimum of the negated prefix sums into a running maximum of the prefix sums,
and `min` conforms on **527** training members exactly as `max` does. Addendum
A1 records the correction, `C_max2_min` was reclassified from failed to
solvable, and the effect is to make the case set **weaker** — `bal` and `max2`
now share one answer, which is why family **C2** (`acc_fold` hole, answer
`{add, sub}` established by enumeration in `out/c2_answer.json`) was added.

**Training conformance is not repair, and the one wrong B-allholes answer proves
it.** On `C_max2_add`, B-allholes returns `{min, max, sub}` at train 1.000; the
extra `sub` sits at the `acc_fold` hole and conforms on all 24 training episodes
with **24** members. Held-out (n=859, majority **0.7334**, random 0.5) it scores
**0.7485–0.7567** — noise-fitting, barely above the constant. The `min` and
`max` repairs on the same task score **0.8265–1.0000**. That is the single
`wrong_not_declined` in the whole table and it is a training overfit that
held-out data exposes.

**Held-out spread across every conforming set** (addendum A4), so that
"conforms on 24 training episodes" is never reported as "solves the task":

| task | repair | conforming on train | held-out unseen spread | first in enumeration order | majority |
|---|---|---|---|---|---|
| `bal` | `second_fold=min` | 110 | **0.9953 – 1.0000** | 1.0000 | 0.5262 |
| `bal` | `second_fold=max` | 110 | **0.9953 – 1.0000** | 1.0000 | 0.5262 |
| `max2` | `second_fold=min` | 527 | **0.8265 – 1.0000** | 1.0000 | 0.7334 |
| `max2` | `second_fold=max` | 527 | **0.8265 – 1.0000** | **0.8265** | 0.7334 |
| `C2 bal` | `acc_fold=add` | 110 | **0.9953 – 1.0000** | 1.0000 | 0.5262 |
| `C2 bal` | `acc_fold=sub` | 29 | **0.9884 – 1.0000** | 1.0000 | 0.5262 |

§47's 0.9953–1.000 spread for the `bal` repair reproduces. The `max2` rows show
the same sweep on a different label returning a tie set training cannot resolve:
527 members, and the enumeration-order-first `max` member generalises to 0.8265.

---

## 7. Against the pre-registered falsification criteria

| criterion | outcome |
|---|---|
| **F1** — fires only on literally-constant nodes → a narrow constant-detector | **Fires far more widely than that**, on 42–124 nodes per scaffold, including chosen constants, unreached padding positions and vacuous readouts. 25 of 32 selected sites are constant nodes; **none** is a defect site. It is not even a constant-detector — it is a *constancy enumerator*. |
| **F2** — flags scaffolds that already contain a solution → unusable as a gate | **Fires on 8 of 8 solvable controls**, including §44 arm 3 with 144 conforming programs and certificate `complete`, for the structural reason in §3. **Fires.** |
| **F3** — the sweep, not the diagnosis, is doing the work | **4** (probe) vs **5.17** (random) vs **15** (sweep, no diagnosis) vs **21** (sweep every hole). **Fires, decisively.** |
| **F4** — only one or two tasks → anecdote | **Two recorded tasks** with independently established repairs; 30 further scaffolds constructed on one template with three distinct answers. Reported as anecdote. |
| **F5** — site localisation adds nothing | Stage D's site differs from "blame the output-adjacent accumulator" in **32 of 32** cases, and is **worse** at every one. **Fires.** |

**What would have supported the hypothesis** — firing on failures and not on
solvable scaffolds, localising better than the terminal-node heuristic, and
beating B-random on ≥ 2 tasks with different answers — fails on all three
counts.

---

## 8. What survives, stated as narrowly as it should be

1. **§47's Q3 numbers are correct and reproduce here to the digit** — the
   constant `acc21`, the per-node bits, the 110 conforming members, the
   84.00%-through enumeration index, the 1.000 on n=859 against a 0.5262
   majority, confirmed on the **real** typed program.
2. **What was doing the work in §47 was the training-only operator sweep**, not
   the information content. The sweep is genuinely cheap, genuinely label-light,
   and genuinely mechanical: `registry.resolve` over `tcn.operators` at the
   hole's signature, exhaustive, with a certificate. It found the repair on
   15 of 24 failed scaffolds from a fixed site and **21 of 24** when allowed to
   try every hole — **with the information diagnosis deleted entirely**.
3. **"Per-node information content" is not a scaffold diagnostic.** Zero mutual
   information with the label is a property of correct XOR-structured programs
   as readily as of dead accumulators; constancy on a training batch is a
   property of chosen constants and of unreached padding positions as readily as
   of a defect. Both fire on solved scaffolds. The right conclusion is not "it
   is narrow" but "it is not measuring what it was read as measuring".
4. **A scaffold-repair mechanism worth building here is the sweep, gated on the
   exhaustion certificate the search already produces** — *this scaffold is
   exhausted with 0 conforming, so try every one-hole substitution over core's
   own inventory and rank on training* — with the tie set's **held-out spread**
   reported, because §6 shows the sweep can return a training-conforming
   proposal that is noise (0.7485 against a 0.7334 majority). No node-level
   information measurement is needed, and adding one made every score worse.
5. **§47's own caution was right and should be strengthened.** Its sentence
   *"Whether the same probe generalises beyond a constant-node diagnosis is
   untested"* can now be replaced with: it does not, and the constant-node
   diagnosis itself is an artifact of a member and a node subset chosen by a
   human who had read §45.

---

## Deviations from pre-registration

- **The probe's repair-family choice.** Step 10 pins "the candidate with the
  highest best-train accuracy"; where both declared families (`swap`, `fold`)
  apply, the prediction is the argmax across both. Recorded here as the reading
  taken; it never changed a verdict, because the probe declined on 20 of 24.
- **Addendum A1** corrected the constructed `max2` task's declared answer from
  `{max}` to `{min, max}`, on the authority of the exhaustive sweep. Recorded in
  `PREREGISTRATION.md` and §6, and it weakens the case set.
- **Addendum A2** added family C2 after the pre-registered arm was committed, to
  restore answer diversity that A1 removed. Its answer was established by
  enumeration and written to `out/c2_answer.json` **before** `score.py` read it.
- **B-allholes is exploratory**, not pre-registered, and is labelled as such
  everywhere. It strengthens the *negative* (it beats the probe), which is the
  direction in which an unregistered control is least dangerous — but it is not
  established and would need its own pre-registered replication before being
  treated as a method.
- **The "input-dependent nodes only" variant** of Stage D is likewise
  exploratory. It changed no verdict and made the repair score worse (4 → 0).
- **No gradient arm was run.** §16 and §47 measured `eq`'s surrogate as exactly
  0.0 past |a−b| ≥ 11, `lt`'s past 17, and `symbols`' first-step gradient as
  exactly 0.0 on this scaffold. Nothing here is attributed to search where a
  dead surrogate is the available explanation.

## Verification

- **Tests**: 324 pass, 13 fail. Identical — 324 pass, 13 fail — with
  `research/scaffold-diagnosis/` **removed**, so all 13 are the known
  environmental failures (`generators/computer` and `tests/test_panel_interface.py`
  need the gitignored `node_modules`/`tsx` symlink, absent in a worktree).
  324 + 13 = the 337 main carries.
- **Shipped fixture**: `python -m tcn train --episodes 160` →
  `initial_prediction_loss` **0.248835613951087**, `final_prediction_loss`
  **0.0022308224288281053**, `evaluation_mean_return` **4.0**,
  `fully_frozen: true`, `frozen_evaluation_mean_return` **4.0** — reproduces
  **0.248836 → 0.002231 at 4/4 frozen**.
- **`git diff main...HEAD -- tcn/ generators/` is empty.** No core hook was
  needed and none was added; the probe reads `Program.execute` traces and the
  operator registry through their public surfaces only.
- **Blindness**: `run_probe.py` imports `cases`, never `answers`;
  `out/predictions.json` was committed at `66ad7fd` and
  `out/predictions_c2.json` at `5d32902`, both before `score.py` existed or ran.
- **Large artifacts**: `out/splits.pkl` (0.9 MB) is gitignored; the JSON
  evidence files are kept because they are the record.

## Files

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | criteria and probe, committed before any arm; addendum committed before the arms it governs |
| `common.py` `splits.py` `prepare.py` `family.py` `accum_scaffold.py` `dyck_scaffold.py` `run_stage_b.py` | vendored byte-identical from `research/dyck-learnability` |
| `scaffolds2.py` | one builder with both holes open; digest-identical to the shipped scaffolds |
| `sim2.py` | the family simulator, generalised by one parameter, with its validation |
| `langfam.py` `boolfam.py` | exhaustive family decision, bitmasked, for the two adapters |
| `cases.py` | the 32 cases the probe sees |
| `answers.py` | the known repairs; imported by `score.py` only |
| `probe.py` | Stage D and Stage S |
| `run_probe.py` → `out/predictions.json`, `out/predictions_c2.json` | the blind arm |
| `allholes.py` → `out/allholes.json` | the exploratory sweep-every-hole control |
| `addendum.py` → `out/c2_answer.json`, `out/addendum.json` | A1–A4 |
| `verify.py` → `out/verify.json` | real-program confirmation and the overfit check |
| `score.py` → `out/scored.json` | the join, and the only file that reads the answers |

## Reproduction

```
cd research/scaffold-diagnosis
export PYTHONPATH=<repo>:.
<repo>/.venv/bin/python prepare.py          # draw the post-audit split once
<repo>/.venv/bin/python run_probe.py        # the 23 pre-registered cases, blind
<repo>/.venv/bin/python addendum.py --c2    # establish family C2's answer
<repo>/.venv/bin/python run_probe.py --only C2_bal_add,...,C2_bal_sub \
       --out predictions_c2.json
<repo>/.venv/bin/python allholes.py
<repo>/.venv/bin/python addendum.py --member --spread
<repo>/.venv/bin/python verify.py
<repo>/.venv/bin/python score.py
```

Total wall clock: about **15 minutes**, single-threaded, dominated by validating
the simulators against `Program.execute`. The 81 family decisions themselves
take under a second each.
