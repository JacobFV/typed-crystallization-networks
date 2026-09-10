# Was the §45 Dyck fix reachable by the system, or was it authored?

FINDINGS §45 closed one question and opened a sharper one. On the re-drawn
non-exploitable `context_free_language` stream the shipped stage-B scaffold
contains **no** conforming program — space 45,375, exhausted, 0 conforming,
certificate `complete` — because every member is a thresholded affine function of
a bracket **count**, and §24's re-draw made counts identical in both classes.
Adding a running `min` beside the running `add`, using `min` and `and` which core
already has, gives a program at **1.000 on all 859 held-out episodes at unseen
lengths 16–22**, recovered by the **unmodified** search inside a **hand-chosen**
window `c ∈ [95,110)`.

So the solution exists and enumeration finds it once someone who knows the answer
narrows the window. This track asks whether the system can get there on its own,
and answers three questions in order. Criteria were written to
`PREREGISTRATION.md` and committed before any arm ran (`98b061f`, addendum
`fb86892`, both before the arm they govern).

**Nothing under `tcn/` or `generators/` is touched, and no operator is added.**
`git diff main...HEAD -- tcn/ generators/` is empty. Every episode draw pins
`hardening` explicitly; every number below comes from the post-audit
`hardening='context_free_language'` stream unless it says otherwise.

---

## Verdict, up front

*(filled in below once every arm has a number)*

---

## 0. Setup: the same split, the same code, verified identical

**Stream.** `hardening='context_free_language'`, pinned at every
`common.dataset` call. The vendored `common.py` takes `hardening` as an argument
with no silent inheritance of the generator default — the §39 defect. The
pre-audit stream is not used anywhere in this track.

**Provenance.** Four modules are vendored byte-identical from branch
`language-post-audit` @ `090b780` so this track's numbers are comparable to §45's
by construction rather than by resemblance:

| file | sha256 | source |
|---|---|---|
| `common.py` | `c8f856f3…` | `research/language-post-audit/common.py` |
| `splits.py` | `c04acb42…` | `research/language-post-audit/splits.py` |
| `dyck_scaffold.py` | `81f3f54f…` | `research/language-post-audit/dyck_scaffold.py` |
| `run_stage_b.py` | `ad06fa06…` | `research/language-post-audit/run_stage_b.py` |

`accum_scaffold.stage_b_accum(fold='min')` builds a program whose digest is
**equal** to `dyck_scaffold.stage_b_dyck`'s (`a7aa228bddbeb1cd`), asserted rather
than assumed, so the operator sweep in §3 varies exactly one thing.

**The split, re-drawn here and identical to §45's.** `prepare.py` →
`splits.json`. Train on the three shortest lengths the stream emits, hold out the
four longest.

| split | n | per-length | majority constant | share of strings seen in training |
|---|---|---|---|---|
| `train` | 24 | 10:3, 12:11, 14:10 | 0.5000 | 1.000 (by definition) |
| `heldout_seen_lengths` | 120 | 10:34, 12:42, 14:44 | 0.5333 | 0.125 |
| `heldout_unseen_lengths` | **859** | 16:221, 18:209, 20:215, 22:214 | **0.5262** | **0.000** |

**Train/test string overlap is 0.0%** on the unseen-length split, and every
per-length count matches §45's table exactly. Depths 1–4 occur in all three
splits. `counts_match_rate` is 1.000 everywhere, which is the fact that makes a
bracket count useless here.

**Pipeline check before any arm.** One shard of the min-prefix space,
`c ∈ [101,102)`: space 5,625, evaluated 5,625, exhausted, **35 conforming**,
certificate `complete`, 156 s. Its first-in-order member scores **1.000 / 1.000 /
1.000** on train / seen / unseen (per-length 1.000 at 16, 18, 20 and 22) against
the 0.5262 majority. §45's headline reproduces.

---

## 1. Q1 — the unwindowed enumeration

§45's recovery used `c ∈ [95,110)`, a window chosen by hand from knowledge of the
answer, and its attempt at the full space was stopped at ~52% and therefore
carried no certificate. This runs the whole space.

**Method, and what is and is not modified.** The search is
`tcn.search.enumerate_fit`, unmodified, with `rank='order'` so it returns the
*first* conforming member in enumeration order — which is what Q1 asks about. The
scaffold is `dyck_scaffold.stage_b_dyck`, unmodified. The one thing this track
does is **partition** the `c` axis into eleven disjoint contiguous windows of 11
values, run concurrently; `sub_range` is an argument §45's own runner already
exposes, the windows are disjoint and cover `[0,121)` exactly, and eleven
`exhausted` shards therefore constitute an exhaustive decision of the full space.
Wall clock and total CPU are reported separately, because only the second is the
cost a single process would pay.

The partition is **order-preserving**: `enumerate_fit` walks `itertools.product`
in node order and `symbols` (which carries `c`) is the most significant
multi-candidate node, so a contiguous window of `c` is a contiguous block of the
global mixed-radix order,

```
index = (((c*5 + plus)*5 + minus)*15 + total_ok)*15 + min_ok
```

which reproduces §45's own recorded witness index, **571,822**, exactly
(`index_arithmetic_agrees: true`).

### The answer: yes, and with a certificate

`q1_launch.sh` → `out/q1_c*.json` → `q1_aggregate.py` → `out/q1_union.json`.

| | value |
|---|---|
| space size | **680,625** |
| evaluated | **680,625** |
| **exhausted** | **true** |
| **conforming** | **110** |
| **certificate** | **`complete`** |
| wall clock, 11 concurrent shards | **2,069.9 s** (34.5 min) |
| **total CPU** | **20,990 s = 5.83 h** |
| seconds per program | 0.0308 |
| partition exact (121 `c` values, 0 overlaps) | **true** |

Every one of the eleven shards is `exhausted` with certificate `complete`, so the
union decides the whole space. **This is the certificate §45 could not obtain** —
its full-space attempt was stopped at ~52% and reported, correctly, as
establishing nothing.

### Where the conforming members are, and where the first one sits

| shard (`c`) | conforming | seconds |
|---|---|---|
| [0,11) … [88,99) — nine shards | **0** each | 1,953 – 2,070 |
| **[99,110)** | **110** | 1,974 |
| [110,121) | 0 | 1,002 (most members raise: `length − c` underflows unsigned `POS`) |

**All 110 conforming programs live in `c ∈ [99,110)`.** §45's hand-chosen window
was `c ∈ [95,110)`, and the 110 conforming members it reported inside that window
are **the entire conforming set of the full space** — the same count, now known to
be exhaustive rather than window-scoped. So the window cost §45 nothing in
completeness. What it bought was time: 84,375 programs instead of 680,625, an 8×
saving.

The **first conforming member in enumeration order** sits at global index
**571,746 of 680,625 — 84.00% of the way through**. At this run's measured
0.0308 s/program that is a projected **4.90 hours of single-threaded work before
the first success**. Decoded, it is `c = 101` (so `symbols = prompt_bytes − 101`,
exactly the string's own length), `plus = +1`, `minus = −1`,
`total_ok = ge(acc21, −2)`, `min_ok = eq(lo21, 0)`, and by exact execution:

| readout | n | accuracy | majority | random |
|---|---|---|---|---|
| train | 24 | **1.0000** | 0.5000 | 0.5 |
| held-out, seen lengths | 120 | **1.0000** | 0.5333 | 0.5 |
| held-out, **unseen** lengths 16/18/20/22 | **859** | **1.0000** | **0.5262** | 0.5 |

Per-length on the unseen split: 16: 1.000, 18: 1.000, 20: 1.000, 22: 1.000.

*A note on which readout is doing the work, since the selected member is not the
textbook one.* On this stream bracket counts always match, so with
`plus = +1, minus = −1` and `c = 101` the total accumulator `acc21` is **≡ 0** on
every episode and `ge(acc21, −2)` is vacuously true. The entire 1.000 is carried
by `min_ok = eq(lo21, 0)`, i.e. by the running minimum. That is the §3a finding
falling out of the enumeration on its own: the sum contributes nothing and the
minimum contributes everything.

### Q1 verdict

**Enumeration finds it without the window.** The pre-registered falsification —
"if the unwindowed search cannot reach a conforming member within the stated
budget, the fix is not reachable by enumeration as configured" — **did not fire**:
the union exhausted inside the 3 h stated budget (34.5 min wall) with 110
conforming and certificate `complete`.

So §45's recovery did **not** depend on knowing the answer for *existence*. It
depended on it for *cost*, and the cost is the part worth recording: **the first
success is 84% of the way through the default enumeration order, ~4.9
single-thread hours in.** Any budget-capped run that stops earlier — as §45's did,
at ~52% — sees nothing and can conclude nothing. The pre-registered caveat
applies and is recorded: *a first-conforming index past 50% is reported as a
caveat even when the search succeeds, because default enumeration order is not
what found it.* Here it is past 84%.

The reason is structural rather than accidental: the conforming members all need
`c ≈ 101`, the constant that makes the program read exactly the string's own
symbols, and `c` is the **most significant** digit of the enumeration order. Any
ordering that tried larger `c` earlier — or that ranked by description bits, or
that noticed `symbols` should equal the string length — would reach it
immediately. The default order is the worst case for this scaffold, and the
scaffold's own structure says so before the search is run.

---

## 2. Q2 — the gradient path, against its own control

This is the question that matters most for the architecture, because §44 measured
the shipped gradient path solving **1 of 6** corpus tasks and §19 measured it
conforming in **0 of 44** runs on the pre-audit language spaces.

**The two arms, and why the second is the control.** `tcn.learning.SoftProgram`,
unmodified, on:

- **min-prefix** (`dyck_scaffold.stage_b_dyck`, 680,625 programs) — §45 proves a
  solution **exists** here, and §1 above locates 110 of them;
- **counting-only** (`language-capability/scaffolds.stage_b`, 45,375 programs) —
  §45 proves **none** exists, exhausted, certificate `complete`.

A path that conforms at the same rate on both is not selecting the running
minimum; it is fitting noise. That is the comparison, stated in advance.

**Protocol**, following the language track's own `run_stage_b_grad.py`: Adam,
lr 0.05, logits perturbed by `0.01 · randn` (SoftProgram zero-initialises them),
8 seeds, at two budgets (600 and 3,000 steps) and two temperature settings. The
second temperature setting exists because §19 measured `lt`'s surrogate gradient
as **exactly 0.0 at delta ≥ 17** and this scaffold operates at delta up to 22;
`tau_lt = 128.0` is that track's own widened value, and the `in{i}` nodes have a
single candidate, so raising their temperature widens the surrogate without
flattening any choice distribution.

**Conformance is the exact-execution test**, not the relaxed loss: an argmax
export conforms iff `tcn.search.evaluate` gives max error ≤ 1e-6 on all 24
training episodes, checked every 10 steps so "steps to first conforming export"
is measured rather than read off the end of the run.

**"Selects the `min` node correctly" is operationalised**, as pre-registered. The
running-`min` chain is structural (single-candidate nodes); what is *searched* is
the readout `min_ok` over the `{eq,ge,le} × {−2..2}` grid. Since the running
minimum lies in {0,−1,−2} on this stream, a readout is **correct** iff it equals
`min_prefix ≥ 0` on all 859 unseen episodes, **informative** iff it is at least
non-constant there, and **vacuous** otherwise — and a vacuous `min_ok` means the
export has collapsed to the counting program, i.e. into the control's family.

*(numbers pending)*

---

## 3. Q3 — could the scaffold change have been proposed rather than authored?

The scaffold change was chosen by reading §45, which had already localised the
answer. Q3 asks whether anything could propose it from evidence available
**before** knowing the solution: only the failed counting scaffold, its
exhaustion certificate, and the 24 training episodes. Held-out episodes are used
**only to score**, never to select; every selection below is made on the 24
training episodes alone.

**The answer is yes, the signal is there, and it is cheap — and no component of
the shipped system computes it.** Both halves matter and neither is softened.

### 3a-1 The failed accumulator's output is a constant, and that needs no labels

`q3_prehoc.py` → `q3_prehoc.json`. Mutual information, in bits, between each node
the failed scaffold **already materialises** and the label, over the 24 training
episodes. Label entropy is 1.000 bits (12 yes / 12 no).

| node | information (bits) |
|---|---|
| `acc2` | 0.3154 (highest of any prefix node) |
| `acc4` | 0.2749 |
| `acc8` | 0.2217 |
| … | … |
| `acc13` … `acc20` | **0.0000** |
| **`acc21` — the node the answer reads** | **0.0000** |

`acc21` takes **exactly one distinct value, 0, on all 24 training episodes**. The
scaffold's output accumulator is a **constant**. That is detectable by looking at
one node's values across the training batch, with **no search, no certificate,
and no labels at all** — a constant node is a purely unsupervised diagnostic. It
is a strictly sharper and cheaper signal than the 45,375-program exhaustion
certificate that §45 spent 624 s to obtain, and it says the same thing.

Asked of the whole family rather than one member: the most label information any
member's terminal accumulator carries is **0.370 bits** of the 1.000 available
(at `c=110, plus=−2, minus=−1`). The running **minimum** carries **1.000 bits** —
the full label entropy — at `c=101, plus=+1, minus=−1`. On the training episodes
the running minimum is a **sufficient statistic** for the label and the running
sum is very nearly an ancillary one.

### 3a-2 A core reduction over the nodes the scaffold already has separates the labels

The prefix sums are already nodes. Sweeping the reductions core declares over
that existing node set, thresholding each with the same `{eq,ge,le} × {−2..2}`
grid stage B searches, **fitted on the 24 training episodes only** (ties broken by
enumeration order, never by held-out accuracy):

| reduction | best train | members tied at best train | held-out **unseen** (majority 0.5262) | held-out spread across the tie set |
|---|---|---|---|---|
| **`reduce_min`** | **1.0000** | 14 | **1.0000** | **0.9953 – 1.0000** |
| **`reduce_max`** | **1.0000** | 14 | **1.0000** | **0.9953 – 1.0000** |
| `sum` (what the scaffold has) | 0.7500 | 38 | 0.6577 | 0.5600 – 0.6950 |
| `mean` | 0.7917 | 2 | 0.5832 | 0.5832 – 0.5832 |
| `count` | 0.5000 | 42,000 | 0.4738 | 0.4738 – 0.5262 |

`reduce_min`'s training-selected member is `c=101, plus=+1, minus=−1, eq 0` —
**the Dyck reduction**, arrived at by fitting five candidate reductions on 24
training episodes. `reduce_max` is its sign-flipped twin (`plus=−2, minus=+2`).

The tie-set column is the part that decides whether training can *act* on the
signal. In the failed counting family the 24 members tied at the best training
accuracy spread **0.4738 – 0.7031** on held-out data — 23 accuracy points, so
training cannot choose among them (§3b). Under `reduce_min` the 14 tied members
spread **0.9953 – 1.0000**: half a point. Training identifies the answer to
within noise.

### 3b The failed family's error structure, re-derived

`q3_prehoc.json: b_counting_family`. The counting family, decided by the
validated simulator over all 121 × 5 × 5 × 15 = 45,375 members (42,000 of them
usable — the rest raise on unsigned `sub` underflow):

| | value | §45 records |
|---|---|---|
| members conforming on train | **0** | 0, certificate `complete` |
| best train accuracy | **0.7500** | 0.7500 |
| members tied at best train | **24** | 24 |
| held-out unseen spread across the tie set | **0.4738 – 0.7031** (mean 0.6224) | 0.474 – 0.703 |

**§45's recorded numbers reproduce exactly.** The residual is readable but much
weaker than the constancy signal: across the 24 tied members, 144 training errors
fall **126 on episodes with `min_prefix < 0`** and **18 on episodes with
`min_prefix ≥ 0`**, a 1.75× concentration on the class whose only distinguishing
property is a negative prefix minimum (the two classes are 12 and 12). On the
training set `label ⟺ min_prefix ≥ 0` holds for all 24 episodes and
`counts_match` holds for all 24 — the two facts that jointly force the conclusion.

### 3c The scaffold change is mechanically generable from core's own operators

`q3_operators.py` → `q3_operators.json`. "Add a second accumulator beside the
running sum" is a template with **one hole**: the binary operator folded along the
prefix sums. The candidates for that hole are read straight out of
`tcn.operators.BINARY` by asking the registry to resolve each name at
`CNT × CNT → CNT` — nothing hand-curated, nothing added to core. That yields nine
candidates. Each one's full 680,625-member family is decided on the 24 training
episodes.

| fold | usable members | conforming on train | held-out **unseen** of the first conforming member (real program) |
|---|---|---|---|
| **`min`** | 630,000 | **110** | **1.0000** |
| **`max`** | 630,000 | **110** | **1.0000** |
| `add` | 630,000 | 0 | — |
| `sub` | 630,000 | 0 | — |
| `mul` | 630,000 | 0 | — |
| `shl` | 32,175 | 0 | — |
| `shr` | 92,700 | 0 | — |
| `idiv` | **0** | 0 | — |
| `mod` | **0** | 0 | — |

**Two of nine candidates solve the task, and they are the two that compute a
running extremum.** So the operator §45 chose by knowing the answer is also
what a mechanical sweep over core's own inventory returns — the human input
reduces from "know that balancedness is a running minimum" to "when your
accumulator is constant, try folding the prefix with each core binary operator".

Two honest notes. `idiv` and `mod` have **zero** usable members not because they
are bad accumulators but because the template seeds the second accumulator with
the constant `0`, so their first step divides by zero; a template that seeded it
differently would give them a chance, and this is a property of the template, not
of core. And every one of the 110 conforming `min` members scores **≥ 0.9953** on
the 859 unseen episodes, 70 of them exactly **1.000** — the family does not
contain a conforming member that fails to generalise.

The first conforming member of each solving fold sits late in enumeration order —
`min` at global index **571,746 (84.00% through)**, `max` at **569,046 (83.61%)** —
which is the §1 finding again, arrived at independently.

**Simulator validation.** Every count in this section comes from a semantics
simulator that is checked against the **real typed program per fold**, 150 random
members × 24 episodes = **3,600 episode evaluations each**, comparing both the
answer and whether the member runs at all. **All nine folds: 0 mismatches.** Every
headline is then re-confirmed by executing the real program — the `min` and `max`
rows above are real-program numbers (train 1.000, held-out seen 1.000, held-out
unseen 1.000), not simulator numbers.

**Two defects found by that validation, one in the harness and one real, both
recorded rather than quietly fixed.** The first version of the check compared
through `tcn.search.evaluate`, which abandons an example once the tolerance is
exceeded, so a member whose *later* example would raise comes back with a finite
error. That harness reported **8 mismatches in 150 for `shl` and 31 for `shr`**
and 0 elsewhere, and the short-circuit was the obvious suspect. Rebuilding the
check as a per-episode `Program.execute` comparison **did not clear them** — it
reported 8 and **34** — which showed they were not a harness artifact at all but
a genuine simulator bug: core raises `shift outside bit width` unless
`0 ≤ b < bits`, not merely on a negative shift, and the simulator modelled only
the negative case. With the width bound added, every fold reports **0
mismatches** — the numbers in the table above. Both superseded runs are kept
(`out/q3_operators_evaluate_harness.json`, `out/q3_operators_shift_bug.json`)
rather than deleted. No conclusion changes — `shl` and `shr` conform on nothing
under either semantics — but the plausible explanation was the wrong one, and a
simulator whose disagreements had been waved through as a harness artifact would
have been an unvalidated simulator. That is the failure mode §45's
validate-against-the-real-program discipline exists to prevent, and here it
caught something.

### Q3 verdict

**A pre-hoc signal exists, it is cheap, and it is specific.** Three independent
pieces of evidence, all computable from the failed scaffold and the training
episodes with no knowledge of the answer:

1. the scaffold's output accumulator is **constant** across the training batch —
   one distinct value, zero label information — which needs no labels and no
   search at all;
2. the running minimum of the quantity it already accumulates carries **1.000
   bits**, the entire label entropy, while the best terminal sum in the whole
   family carries **0.370**;
3. a sweep of the **five reductions core already declares**, over the nodes the
   failed scaffold already materialises, fitted on the 24 training episodes only,
   returns `reduce_min` at train 1.000 and **held-out 1.000**, with a tie set that
   training can actually resolve (0.9953–1.000, against 0.4738–0.7031 for the
   counting family's tie set).

Independently, the scaffold change viewed as a template with one operator hole is
**mechanically generable**: enumerating that hole over the nine operators core
declares at `CNT × CNT → CNT` returns exactly the two that compute a running
extremum, `min` and `max`, both at 1.000 on the real program.

**And nothing in the shipped system computes any of it.** There is no component
that watches a node for constancy, no component that measures label information
per node, no component that proposes a scaffold edit from a failed run, and no
component that sweeps an operator hole in a template. Every one of those was
written in this track. So the pre-registered negative — *"if no pre-hoc signal
exists, state that scaffold design is a human input"* — is not the outcome; the
outcome is narrower and, for the architecture, more actionable:

> **Scaffold design is a human input in the shipped system today, but the
> evidence needed to automate this particular class of scaffold repair is
> already present in a failed run and costs seconds to extract.** The missing
> piece is a mechanism, not a signal.

What is *not* claimed: that this generalises to arbitrary scaffold failures. The
signal here is strong because the failure mode is degenerate in an especially
visible way — the output node is literally constant. A scaffold that fails while
still varying would not announce itself this loudly, and this track measured one
task.

---

## Verification

| obligation | result |
|---|---|
| `git diff main...HEAD -- tcn/ generators/` | **empty** |
| new operators added to core | **none** — `min`, `max` and `and` were already in `tcn/operators.py` |
| test suite | **275 passed, 13 failed** |
| the 13 failures are environmental | verified: the identical 13 fail with `research/dyck-learnability/` removed, on main's own code — they are the `computer` and `panel` tests, which spawn `node --import tsx` and need the gitignored `node_modules` tree (`test_panel_interface.py::test_panel_episode_replays_and_restores` among them) |
| shipped fixture | `python -m tcn train --episodes 160` → initial prediction loss **0.248835613951087**, final **0.0022308224288281053**, evaluation mean return **4.0**, frozen evaluation mean return **4.0** — reproduces **0.248836 → 0.002231 at 4/4** |
| large artifacts | the episode pickle and run logs are gitignored; every JSON result is committed |
| probes | supervision only; no probe is ever packed into a program input |
| stream | `hardening='context_free_language'` pinned at every draw, stated on every number |

---

## Files

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | the three questions, the arms and the falsification criteria, committed before the arms |
| `common.py`, `splits.py`, `dyck_scaffold.py`, `run_stage_b.py` | vendored byte-identical from `language-post-audit` @ `090b780` |
| `prepare.py` / `splits.json` | the split, drawn once and pickled so every arm sees identical episodes |
| `q1_enum.py` | one shard of the min-prefix space through the unmodified `enumerate_fit` |
| `q1_launch.sh` | the eleven shards plus arm 1b, concurrently |
| `q1_aggregate.py` / `out/q1_union.json` | the shards unioned into one statement about the full space |
| `q2_gradient.py` | the differentiable path on either scaffold |
| `q2_launch.sh`, `q2_aggregate.py` / `out/q2_summary.json` | the eight arms, and the paired comparison |
| `q2_null.py` / `out/q2_null.json` | the uniform-random-selection null |
| `family.py` | exact semantics of the two-accumulator family, and its per-episode validation |
| `accum_scaffold.py` | the scaffold with the fold operator left as a hole |
| `q3_prehoc.py` / `out/q3_prehoc.json` | node label information, the core-reduction sweep, the failed family's error structure |
| `q3_operators.py` / `out/q3_operators.json` | the mechanical sweep over core's `CNT × CNT → CNT` operators |
| `out/q3_operators_evaluate_harness.json` | the superseded run with the short-circuiting validation harness, kept |
| `reproduce` | every command, in order |

## Reproduction

```
bash research/dyck-learnability/reproduce
```

Wall clock: Q1 is the long pole — eleven concurrent shards on a 20-core machine,
or ~5 h of CPU as a single process (the union reports both). Q2 is ~1 h as eight
concurrent arms. Q3 is ~15 min. Episode pickles stay out of git
(`out/*.pkl` is ignored); every JSON artifact in `out/` is committed.
