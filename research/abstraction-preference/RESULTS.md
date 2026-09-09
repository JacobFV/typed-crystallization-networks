# Preference — making the system prefer the cheaper program

The recursive-abstraction re-test
([`../recursive-abstraction-retest/RESULTS.md`](../recursive-abstraction-retest/RESULTS.md))
established that reuse **works**: the module is on the output path in 27 of 27
successes, arm B reaches 8/8 and 19/24 where the flat arm and a same-size
distractor both reach 0, and enumeration certifies that arm A's space contains no
solution while arm B's contains 144, all of which use the module. Its closing
finding was that none of this is *preferred*:

> Reuse is currently a capability, not an objective.

This track implements the diffs that re-test proposed — R1 row deduplication, R2
a description-cost term, R3 dead-node pruning, plus cost-ranked enumeration — and
measures what preference buys and what it costs.

Reproduce with `PYTHONPATH=<repo>:. .venv/bin/python <script>` from this
directory. Unlike the re-test, this track **does** modify `tcn/`; the diff is
`tcn/{learning,graph,operators,runtime,search,synthesis,training}.py` plus spec
notes in `ARCHITECTURE.md` section 4/8 and `docs/IMPLEMENTATION.md`.

---

## VERDICT

**The system now prefers the cheaper program. On the scaffold where a cheaper
program exists it finds it every time instead of most of the time, and it stops
drifting into larger ones. What preference costs is search: roughly double the
steps where it helps, and most of the success rate on a scaffold where there is
nothing cheaper to find.**

1. **`complexity()` was never the term that could express this, and enabling it
   would have made things worse.** The re-test measured it at 20.43 for both
   routes on the diffuse mixture. At the discrete selections that realise the two
   routes it is **9.0 for the flat program and 15.0 for the abstracted one** —
   it charges every scaffold node, dead ones included, so it penalizes the
   abstraction precisely for the saving that makes it good (§1.1).

2. **The semantics chosen is expected description length of the *pruned* program**
   — the serialized program under the current choice distribution, each module
   definition charged once and each call site charged individually, with node
   liveness and module use taken as mean-field expectations. **At any one-hot
   distribution it equals `export().pruned().description_bits(registry)`
   exactly** (0.0 bits of difference on three scaffolds, 0.004 bits of float32
   rounding on the fourth), so it is a relaxation of a real quantity rather than
   a bonus for modules. It *penalizes* every module call site; a one-call-site
   module is correctly dispreferred, and abstraction wins only from two sites,
   where it actually collapses nodes (§1.2, §1.3).

3. **It finds the 2-call solution instead of the 3-call one.** At `mdl_weight =
   1e-5` on the wide scaffold, **8 of 8 seeds return the 3-node two-call program
   against 7 of 8 without it, with success unchanged at 8/8** — including the
   exact seed whose three-call program the re-test flagged. Given a full budget
   rather than stop-at-first, the untermed search **drifts into a larger program
   in 3 of 8 runs after already having found the small one**; with the term, 0 of
   8 (§5.1, §5.2).

4. **Preference costs success where nothing cheaper exists.** On the tight
   scaffold, where all 144 solutions have identical size and every node must
   live, the same weight takes conformance from **19/24 to 2/24**
   (p = 1.1·10⁻⁶), and 1e-4 takes it to zero on both scaffolds. The pressure is
   "make fewer nodes live"; that is aligned with the target on an
   over-provisioned scaffold and directly opposed to it on an exactly-sized one.
   This is a real trade, not a tuning artifact (§5.3).

5. **Enumeration now answers "what is the best program".** `rank='description'`
   or `'cost'` makes declaration order stop deciding between equivalent
   candidates, and on a space containing both routes it returns the 3-node
   19 496-bit program where `rank='order'` returns the 5-node 23 944-bit one. On
   the tight arm-B space it changes nothing, because all **144** conforming
   programs — a count that reproduces the re-test's independent one exactly — are
   argument-symmetry variants of each other (§4).

6. **R1 and R3 land, with smaller numbers than projected.** Row deduplication is
   **4.5–4.9×** on a whole soft forward pass (the re-test projected 7.45× for the
   module candidates alone) at a **6–8%** worst case, bit-identical throughout.
   Pruning preserves semantics on every fixture over 3 875 row comparisons and
   removes a 62% description overstatement from an abstracted export — but it
   **does not move the FINDINGS section 10 crossover**, which stays at 2 call
   sites, and it fixes only **1 of 5** learned modules, because four fifths of
   the "5 gates where 4 suffice" surcharge is redundant *live* structure that
   only the preference term can remove (§2, §3).

---

## 1. R2 — what description cost should mean

### 1.1 The problem the semantics has to solve

`SoftProgram.complexity()` is a softmax-weighted sum of `operator.cost`. After
F1, a module call's `operator.cost` **is** its body's execution cost, so calling
and inlining cost the same. Re-measured here (`probe_cost.json`), on the wide
scaffold, at the exact selections that realise each route:

| route on the wide scaffold | `complexity()` | pruned execution cost | pruned description bits | live nodes |
|---|---|---|---|---|
| 9-gate flat | **9.0** | 9.0 | 20 784 | 9 |
| 2 module calls + `xor` | **15.0** | 9.0 | **19 496** | 3 |
| 3 module calls + `or` + `xor` (the redundant success) | **18.0** | 14.0 | 23 960 | 5 |

Two corrections to the re-test's reading, both in the same direction. The
re-test measured `complexity()` at the *diffuse* mixture and found 20.43 for both
routes; at a *discrete* selection the two are not equal — `complexity()` charges
every node of the scaffold, dead ones included, so the abstracted program scores
**worse** (15.0 against 9.0) precisely because its saving is that six scaffold
nodes go dead. So `complexity()` cannot prefer abstraction, and where it does
distinguish the routes it prefers the wrong one. It is not the term that can
express preference, and turning it on would have taught the search to avoid
modules.

### 1.2 The semantics chosen

**`SoftProgram.description_cost()` is the expected description length, in bits,
of the *pruned hardened* program under the current factorized choice
distribution** — that is, of the program the search would export and register if
it stopped now, not of the scaffold it is searching.

It measures exactly what `Program.description_bits(registry)` measures, which is
what ARCHITECTURE section 4 already specifies for sharing: the serialized
program, plus each distinct frozen module definition charged **once**, with every
call site charged individually. Two expectations make it differentiable:

* **Liveness.** A node is charged only when it can reach an output or a state
  update. Under the factorized distribution, in reverse topological order,
  `P(live(n)) = 1 - Π_{m after n} (1 - P(live(m)) · q_m(n))`, where `q_m(n)` is
  the softmax mass at node `m` on candidates that read `n`; sinks are live.
* **Definitions.** A module definition is charged once, when *some* live node
  calls it: `P(used(d)) = 1 - Π_n (1 - P(live(n)) · q_n(d))`, over the transitive
  closure of module references, mirroring the `seen` set in `description_bits`.

Both are mean-field: they ignore correlations between node choices while the
distribution is diffuse. **At any one-hot distribution the approximation is
exact**, which is the property that makes it a relaxation of a real quantity
rather than a heuristic. Checked on four scaffolds × six random near-one-hot
initializations (`check_exact.py`, `exactness.json`):

| scaffold | max abs. difference from `export().pruned().description_bits(registry)` |
|---|---|
| retest wide, arm A (no module, 3 060 candidates) | **0.0 bits** |
| retest wide, arm B (module, 5 004 candidates) | 0.0039 bits (float32 rounding at 2·10⁴) |
| retest tight, arm B | **0.0 bits** |
| 5-gate sub-scaffold | **0.0 bits** |

The decomposition that makes this possible is that a serialized program is a
selection-independent header plus one serialized node per live node (plus its two
JSON separator characters), so the per-candidate contributions are static and can
be tabulated once.

### 1.3 Why this is not a bonus for modules

The requirement was a real cost of the program, not a hand-tuned reward for
reuse. This term **penalizes** module use per call site and rewards only
shortness, which is why it gets the direction right for the wrong-looking reason:

* a module-call node serializes **larger** than an `and`/`or`/`xor` node, because
  it carries the digest name and three typed inputs;
* a module definition is a fixed charge that a flat program never pays;
* on track 5's own crossover construction the abstracted form is **1.42× the size
  of the inlined form at one call site** (14 480 against 10 176 bits) and only
  passes it at two (17 072 against 18 520). See `crossover.json`.

So abstraction wins under this term only where it actually collapses many nodes
into few, and a shorter flat program would win against a module for the same
reason. A one-call-site module is correctly dispreferred.

### 1.4 Wiring and cost

`synthesis.fit` gains `mdl_weight` (default `0.`), `TrainConfig` gains
`description_weight` (default `0.`) — kept distinct from the existing
`mdl_weight`, which weights `complexity()` and measures execution. Both default
off, so every shipped fixture is unchanged; `tcn train --episodes 160` still
reports `0.248836 -> 0.002231`, `fully_frozen: true` and 4.0/4.0 return
(`tripwire_joint.log`). Since the term is in bits against a probe loss of order
1, weights around 1e-6 to 1e-4 are the competitive range.

Per-step overhead (`probe_cost.json`): a one-time table build of 1–19 ms, then
0.07–0.87 ms per evaluation, which is **0.14%–1.6%** of one optimizer step on the
four scaffolds used here.

---

## 2. R1 — evaluate each distinct row once

`exact_tensor` re-ran the whole module body per batch row. Exact execution is a
pure function of the row, and a typed row over finite carriers repeats, so the
call now memoizes within itself. Measured before/after **in the same process**,
with the pre-change implementation reproduced in `dedup.py` so both sides run
against the same registry and tensors:

| case | distinct rows | bit-identical | before | after | ratio |
|---|---|---|---|---|---|
| MAJ3 module over `(a,b,c)`, 64-row truth table | 8 / 64 | yes | 679 µs | 145 µs | **4.69× faster** |
| MAJ3 module over `(a,a,b)` | 4 / 64 | yes | 685 µs | 101 µs | **6.79× faster** |
| one soft forward pass, tight arm-B scaffold | — | — | 306 ms | 62 ms | **4.93× faster** |
| one soft forward pass, wide arm-B scaffold | — | — | 1 380 ms | 308 ms | **4.48× faster** |
| `int[8]` `add` module, 64 random rows (worst case) | 64 / 64 | yes | 573 µs | 619 µs | 1.08× **slower** |
| `int[8]` `add` module, 512 random rows | 510 / 512 | yes | 4 591 µs | 4 888 µs | 1.06× slower |
| float `atan2`, 64 random rows (worst case) | 64 / 64 | yes | 499 µs | 533 µs | 1.07× slower |

**Bit-identity holds in every case**, checked with `torch.equal` on dtype, shape
and contents, and covered by two regression tests on the Boolean and float paths.

Two honest deltas from the re-test's projection. It reported **7.45×** for "every
module candidate at every node of arm B's scaffold"; measured here over a
complete soft forward pass — which also contains the relaxed primitives that
dedup does not touch — the gain is **4.5–4.9×**. And its worst case was 4–13%;
here it is **6–8%**. Both directions of its claim reproduce; the magnitudes are
smaller.

The one correctness subtlety is signed zero: `-0.0 == 0.0` and they hash alike,
but `atan2` distinguishes them, so a call whose arguments contain a negative zero
skips the cache entirely rather than risk a wrong reuse. NaN keys simply never
match, which is safe. Both are tested.

In the arms this shows as wide arm B costing **0.41–0.60 s per optimizer step**
against the re-test's 5.11 s on the same scaffold. Some of that is a quieter
machine; the forward-pass measurement above isolates the part that is R1.

---

## 3. R3 — prune dead nodes before export and before registration

`Program.pruned()` drops nodes that cannot reach an output or a state update.
Reachability is taken over the selected candidate where a node is hardened and
over *every* candidate where it is not, so it is safe on a soft scaffold too.
`Registry.register_module` and `runtime.save_program` now apply it.

### 3.1 Semantic preservation

Every fixture, hardened at 25 random selections each and executed row by row
against its unpruned self (`prune_fixtures.py`):

| program | rows compared | disagreements |
|---|---|---|
| `examples/mixed` | 400 | **0** |
| `examples/joint` | 75 | **0** |
| retest wide scaffold, arm A | 1 600 | **0** |
| retest wide scaffold, arm B | 1 600 | **0** |
| retest tight scaffold, arm B | 1 600 | **0** |
| retest sub-scaffold | 200 | **0** |

3 875 row comparisons, zero disagreements. The 93 shipped tests still pass, and
`tcn train --episodes 160` is unchanged end to end.

### 3.2 What pruning is worth, and what it is not

The overstatement is large. On the wide scaffold the abstracted program's
hardened export is **31 640** description bits against the pruned **19 496** — a
**62%** overstatement, larger than the 32% the re-test measured on its track-5
scaffold, because six of nine scaffold nodes go dead on the abstracted route.

On the module side it is smaller than the re-test implies, and the reason is a
distinction its wording blurs. Learning MAJ3 on the 5-node sub-scaffold, taking
the raw export straight to `register_module` with no experiment-side pruning
(`acquire.py`):

| | measured over 8 acquisition seeds |
|---|---|
| acquired in one attempt | 5 / 8 |
| registered with 5 live nodes where 4 suffice | 4 / 5 |
| registered with a genuinely **dead** node that pruning removes | **1 / 5** |
| that one: nodes 5 → 4, call cost 5.0 → **4.0**, definition 12 080 → **10 056** bits |  |
| pruned module still exactly MAJ3 on all 8 rows | 5 / 5 |

So the "20% permanent surcharge on every future call" is real, but four fifths of
it is a **redundant live** circuit, which pruning cannot touch — only a
preference term can (§5.3). Pruning fixes the remaining fifth completely.

### 3.3 It does not move the crossover

FINDINGS section 10 records the description-size crossover against inlining
moving from 4 call sites to 2 after F1. Re-measured on track 5's own construction
with pruning on and off, and with a learned body's dead gate present
(`crossover.py`):

| registered body | module nodes | definition bits | call cost | crossover | ratio at 8 sites |
|---|---|---|---|---|---|
| minimal, pruning off | 4 | 10 056 | 4.0 | **2 sites** | 0.476 |
| minimal, pruning on | 4 | 10 056 | 4.0 | **2 sites** | 0.476 |
| learned (1 dead gate), pruning off | 5 | 12 104 | 5.0 | **2 sites** | 0.407 |
| learned (1 dead gate), pruning on | 4 | 10 056 | 4.0 | **2 sites** | 0.476 |

**Pruning does not move the crossover point.** A dead gate inflates the
definition *and* every inlined copy, so the ratio moves in abstraction's apparent
favour (0.407 against 0.476) while the crossover stays at 2. What pruning
actually buys is that the call stops costing 5.0 execution units where 4.0 is
correct, and the definition stops carrying 2 048 bits of dead scaffold — a charge
paid at every call site, in every program, forever. The FINDINGS section 10 row
"description-size crossover (3-gate body): 4 call sites → 2" is unchanged by this
work.

---

## 4. Cost-ranked enumeration

`enumerate_fit` collected every conforming selection and returned `found[0]`,
which ranks by declaration order. It now takes `rank` in `{'order',
'description', 'cost'}`, reports `conforming` (how many it found), and reports
the `description_bits` and `execution_cost` of what it returned — all measured
after pruning, since a hardened scaffold's size and cost describe the search
space. Ranking needs the whole conforming set, so it is rejected together with
`stop_at_first` rather than silently ignoring it.

### 4.1 Declaration order no longer decides

The re-test's demonstration: a registered `and` module and the primitive `and`
compute the same function and have the same `operator.cost`
(`enumerate_rank.json`).

| declaration order | `rank='order'` | `rank='description'` | `rank='cost'` |
|---|---|---|---|
| primitive first | `and`, 3 760 bits | `and`, 3 760 bits | `and`, 3 760 bits |
| **module first** | **`module:c6f94b361fab5`, 7 744 bits** | **`and`, 3 760 bits** | **`and`, 3 760 bits** |

### 4.2 On the tight arm-B scaffold, ranking changes nothing — and that is informative

Exhausting all 2 709 504 programs (`enumerate_rank.json`):

| ranking | exhausted | conforming | returned | bits | cost | seconds |
|---|---|---|---|---|---|---|
| order | (stopped at first) | — | `xor(MAJ3(a,b,c), MAJ3(d,e,f))` | 19 496 | 9.0 | 48 |
| description | yes | **144** | the same program | 19 496 | 9.0 | 178 |
| cost | yes | **144** | the same program | 19 496 | 9.0 | 176 |

The conforming count reproduces the re-test's independent packed-truth-table
count of **144** exactly. All 144 are argument-symmetry variants of one program,
so they have identical size and cost and ranking cannot separate them. Ranking is
free to add and answers a question this scaffold does not pose.

### 4.3 Where the space does pose it, ranking picks the cheaper program

A 1 024-program scaffold admitting **both** the intended two-call program and the
redundant three-call program the gradient search actually accepted in the re-test
(wide, seed 1), with the redundant one first in enumeration order
(`enumerate_redundant.py`):

| ranking | returned | live nodes | description bits | execution cost |
|---|---|---|---|---|
| `order` | `n3 = MAJ3(d,e,f); m = or(n2,n3); y = xor(n1,m)` | **5** | **23 944** | **14.0** |
| `description` | `y = xor(MAJ3(a,b,c), MAJ3(d,e,f))` | **3** | **19 496** | **9.0** |
| `cost` | the same | **3** | 19 496 | 9.0 |

168 of the 1 024 programs conform, exhausted in 0.4 s. The discrete reference now
answers "what is the best program" and not merely "what is a program".

---

## 5. What preference buys, and what it costs

The re-test's arms re-run with the description term switchable and nothing else
changed (`armlib.py`, `experiment.py`, `tables.py`). The `mdl_weight = 0` column
reproduces the re-test: 8/8 on the wide scaffold at a median of **85** steps,
19/24 on the tight one, both exactly as it reported.

### 5.1 The wide scaffold — the term finds the cheaper program at no cost in success

Nine nodes where three suffice, 8 seeds, stop at the first conformant checkpoint
(the re-test's criterion):

| `mdl_weight` | conformant | median steps | **3-node (2-call) programs** | redundant programs found | Fisher vs 0 |
|---|---|---|---|---|---|
| **0** | **8 / 8** | 85 | **7** | **1** — 5 live nodes, **3 calls**, 23 960 bits, cost 14.0 | — |
| 1e-6 | 8 / 8 | 80 | 7 | 1 — 4 live nodes, 21 528 bits | 1.00 |
| **1e-5** | **8 / 8** | **160** | **8** | **0** | 1.00 |
| 1e-4 | **0 / 8** | — | — | — | 1.6·10⁻⁴ |
| 1e-3 | **0 / 8** | — | — | — | 1.6·10⁻⁴ |

**This is the direct answer to the re-test's complaint.** Its wide seed 1 —
`w0 = MAJ3(d,e,f); w1 = MAJ3(b,c,a); w4 = MAJ3(e,d,f); w7 = or(w0,w4); y = xor(w1,w7)`
— reproduces exactly at `mdl_weight = 0` and is accepted because it conforms. At
1e-5 that seed returns the 3-node program instead, and so does every other seed:
**8 of 8 against 7 of 8, with success unchanged at 8/8**. The price is search
time: the median first-conformant step roughly doubles, 85 → 160.

### 5.2 Given a full budget, the term is what *holds* the search at the small program

The same scaffold and seeds run for the whole 300-step budget rather than
stopping at the first success, which is the only way to see a preference that
acts after conformance:

| `mdl_weight` | conformant | live nodes of the program the run **ends** on, all 8 seeds | still exact at the end |
|---|---|---|---|
| **0** | 8 / 8 | **3, 3, 3, 3, 3, 4, 4, 5** | 8 / 8 |
| **1e-5** | 8 / 8 | **3, 3, 3, 3, 3, 3, 3, 3** | 8 / 8 |

Without the term the search reaches a three-node program and then **wanders into
a larger one in 3 of 8 runs** — it stays exactly conformant while growing,
because nothing in the objective distinguishes the two. With the term, none of
the eight drifts. That drift is the re-test's finding in its general form: the
redundant program it caught was not an unlucky stopping point, it is where an
unweighted search ends up.

### 5.3 The tight scaffold — preference costs success where there is nothing cheaper to find

Three nodes, all of them necessary; all 144 conforming programs have identical
size and cost (§4.2). 24 seeds:

| `mdl_weight` | conformant | median steps | median live nodes | Fisher vs 0 |
|---|---|---|---|---|
| **0** | **19 / 24** | 50 | 3 | — |
| 1e-6 | 13 / 24 | 40 | 3 | 0.125 |
| 1e-5 | **2 / 24** | 45 | 3 | **1.1·10⁻⁶** |
| 1e-4 | **0 / 24** | — | — | 7.4·10⁻⁹ |
| 1e-3 | **0 / 24** | — | — | 7.4·10⁻⁹ |

**This is the real trade, and it is not a tuning accident.** The term's pressure
is "make fewer nodes live". On the wide scaffold that pressure is aligned with
the target, because six of nine nodes *should* die. On the tight scaffold every
node must live, so the same pressure pushes directly against the only solution
that exists, and it wins: at 1e-5, the weight that makes the wide scaffold find
the minimum every time, the tight scaffold falls from 19/24 to 2/24. Every run
that *did* conform still produced the intended 3-node program — the term does not
corrupt what it finds, it stops the search finding it.

The failure at 1e-4 and above is the same mechanism at its limit: the run ends on
a one- or two-node program that is exactly conformant with nothing, which is the
cheapest program in the scaffold.

### 5.4 Acquisition — the term does what pruning cannot, at a cost it cannot yet pay

Pruning removes dead nodes; the learned MAJ3's problem is mostly a *live* fifth
gate (§3.2), which only a preference term can address. Twelve seeds on the 5-node
sub-scaffold, 600 steps, no restarts, run to the full budget (`acquire_mdl.py`):

| `mdl_weight` | exactly conformant at the end | median live nodes | **minimal (≤ 4 gates)** | median definition bits |
|---|---|---|---|---|
| **0** | **4 / 12** | 5 | **1 of 4** | 12 088 |
| 1e-6 | 3 / 12 | 5 | 1 of 3 | 12 088 |
| **1e-5** | **2 / 12** | **4** | **2 of 2** | **10 056** |
| 1e-4 | 0 / 12 | — | — | — |

Both modules acquired with the term are the **minimum 4-gate circuit**; without
it, three of four are the 5-gate one that will overcharge every future call site
by 20% in execution cost and 2 032 bits in size, forever. But acquisition rate
halves, and neither contrast is significant at these counts (p = 0.64 for the
rate, p = 0.40 for minimality). This is the weakest result in the track and is
reported as suggestive only.

---

## 6. Limitations

* **Seed variation is injected, not intrinsic.** `SoftProgram` zero-initializes
  every choice logit, so `torch.manual_seed` alone does **not** vary synthesis:
  every run of a given scaffold would be bit-identical. Every arm here adds
  N(0, 0.5) noise to the logits before training (`armlib.run`, inherited from the
  re-test's `common.search`), which is what makes the seed distributions
  meaningful. Any figure quoted from a script that does not add that noise is a
  single outcome, not a distribution.
* **Which cost comparison a number refers to.** A module call is ~1.7× an *exact*
  primitive and ~100× a *relaxed* primitive at batch 64 (FINDINGS section 12).
  Every ratio in section 2 is against the same route through `exact_tensor`
  before and after R1 — it is a speedup of the module path against itself, not a
  claim about parity with primitives. Section 1's execution costs are
  `operator.cost` units, not seconds.
* **Boolean domain, complete truth tables, supervised synthesis, argmax export.**
  No crystallization phase in the composite arms and no RL path, following the
  re-test's design so the `mdl_weight = 0` column stays comparable to it.
* **Sample sizes** are 8 seeds on the wide scaffold, 24 on the tight one, and 12
  per weight on acquisition. Only large effects are detectable, and the
  acquisition contrast in particular rests on single-digit counts.
* **The weight is a free parameter and the useful range is narrow.** The term is
  in bits; the sweep covers 1e-6 to 1e-3 and both ends are visible in the
  results. Nothing here normalizes it automatically.
* **The liveness and definition expectations are mean-field.** They are exact at
  any one-hot distribution and approximate in between, and the approximation
  systematically ignores the correlation between "this node is live" and "that
  node feeds it".
* **The machine was loaded throughout** (load average 15–18 on 20 cores, several
  agents in the same tree). Wall-clock figures are ratios measured in one
  process; step counts are the load-independent metric.

---

## 7. Files

| file | what it does | output |
|---|---|---|
| `check_exact.py` | the surrogate against the real quantity at one-hot, four scaffolds | `exactness.json` |
| `probe_cost.py` | what `complexity()` and `description_cost()` read on each route; per-step overhead | `probe_cost.json` |
| `dedup.py` | R1 speedup and worst case, with the pre-change implementation reproduced in-process | `dedup.json` |
| `prune_fixtures.py` | R3 semantic preservation on every fixture, row by row | `prune_fixtures.json` |
| `crossover.py` | the FINDINGS section 10 crossover with pruning on and off, minimal and learned bodies | `crossover.json` |
| `acquire.py` | what registration charges for a learned module, before and after pruning | `acquire.json` |
| `armlib.py` | the re-test's training loop with the description term switchable | — |
| `experiment.py` | the arms, per scaffold and weight, stop-at-first or full budget | `runs_*.json`, `*_B*.log` |
| `acquire_mdl.py` | module acquisition with the term on and off | `acquire_mdl.json` |
| `enumerate_rank.py` | ranked enumeration: equivalent candidates, and the tight arm-B space exhausted | `enumerate_rank.json` |
| `enumerate_redundant.py` | a space where ranking changes the answer | `enumerate_redundant.json` |
| `tables.py` | arm summaries and two-sided Fisher exact | `tables.json` |
| (shipped) | `tcn train --episodes 160` tripwire | `tripwire_joint.log` |
