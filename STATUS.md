# STATUS — 2026-09-09

What works, what is measured not to work, what has never been tested, and what
is in the way. Every claim below cites either a command you can run or a file
under `research/`. Anything that could not be traced to one has been cut.

Read `docs/VALIDATION.md` for the corrected record and `research/FINDINGS.md`
for the consolidated measurement pass. This file is the short, hostile version.

## How to check any of it

```bash
scripts/demo.sh                 # every demonstrated capability, ~5 minutes
scripts/demo.sh --only depth    # one of them
scripts/demo.sh --full          # more seeds, wider inputs
scripts/demo.sh --list
```

Artifacts land in `artifacts/demo/<name>/result.json`, with the table in
`artifacts/demo/summary.md`. Exit status is non-zero if anything fails to
reproduce. `.venv/bin/tcn` was broken for the life of the project by a stale
shebang from the repository rename and is fixed (blocker B7 closed);
`.venv/bin/python -m tcn` always worked and remains equivalent.

---

## 1. What works, with evidence

Ten capabilities reproduce on the current tree. Each row's measured number is
printed next to the reference it has to beat, because on this project the
missing reference has four times been the difference between a result and a
mistake: a crystallizer "improvement" that was extra compute, a table pool that
silently ignored an explicit config, a lexicographic tie-break presented as a
solution, and a dead surrogate reported as an address wall. In every case the
fault was in the measurement, not the method under test.
(`research/FINDINGS.md` §16, method note.)

| # | capability | measured | baseline | reproduce |
|---|---|---|---|---|
| 1 | Typed synthesis with exact export | held-out max error ≤ 1e-7 on 584 unfitted points; exported `.pyz` runs on stdlib Python | matched MLP 1.8e-2 max / 0.232 extrapolation RMSE (`research/baselines/RESULTS.md` §3) | `scripts/demo.sh --only synthesis` |
| 2 | Enumeration cross-check on that synthesis | unique program of 96, found in ~7 ms, identical to the gradient path's | gradient path ~3 s, no uniqueness certificate (`research/enumerative-baseline/RESULTS.md` §2) | same |
| 3 | Structural generalization to unseen Boolean gate families | 3.38–4.00 on families never trained on, both split directions; interpreter candidate selected 8/8 unprompted | best constant 2.13–2.56; the recorded scaffold measures 1.95–2.03 and provably cannot exceed 2.00 (`research/nondegenerate-generalization/RESULTS.md`) | `scripts/demo.sh --only structure` |
| 4 | Depth generalization from one fixed graph | trained at depths 1–2 only, scores 4.00 at depths 1, 2, 3, 4, 6 and 8, sd 0.00 over 8 seeds | best constant 2.06–2.50 per depth; 272-program space, optimum certified unique (`research/depth-generalization/RESULTS.md`) | `scripts/demo.sh --only depth` |
| 5 | Recursive abstraction | module on the output path in 27/27 successes; arm B 8/8 wide and 19/24 tight | flat arm 0/8 and 0/24, same-size distractor 0/8 and 0/24; the 230,400-program flat space exhausted with no solution; 144 of 144 solutions use the module (`research/recursive-abstraction-retest/RESULTS.md`) | `scripts/demo.sh --only abstraction` |
| 6 | Positional reuse | one frozen module applied at **1,024** positions of a 3,072-value observation with **three** caller nodes, exact; `--full` does 2,304 positions over 6,912 values | 17 structural symbols shared against 3N+13 per-position and 8N+1 inlined, at 0.93× execution cost (`research/positional-reuse/RESULTS.md`) | `scripts/demo.sh --only positional` |
| 7 | Foreground/background segmentation from raw pixels | held-out max error 0.0 on 48 unseen episodes; background colour recovered over the full 0–255 byte alphabet | unique among 65,536 programs; constant predictor 0.854 (`research/perception-ladder/RESULTS.md` §4) | `scripts/demo.sh --only segmentation` |
| 8 | Two-position edge detector, offset searched | held-out max error 0.0, accuracy 1.000, applied at every position | unique among 48 staged programs; undecomposed the same target is 4.9e10 programs, 7.6 years projected; constant predictor 0.844 (`research/discrete-perception/RESULTS.md` §5) | `scripts/demo.sh --only edge` |
| 9 | A language task learned from raw prompt bytes | stage A discovers its own lexical unit — which byte opens a bracket, over the full 0–255 alphabet, and where the symbol field starts — unique among 10,496 programs; stage B reaches **1.000** on string lengths never trained on | majority constant 0.548, best fitted feature 0.648, random 0.500; gradient descent on the identical spaces conforms **0 of 44 runs** (`research/language-capability/RESULTS.md`) | `scripts/demo.sh --only language` |
| 10 | MuJoCo behind the generator contract | replay, snapshot/restore and cross-process reload all bit-identical, max \|Δ\| 0.0 | a scripted energy-pumping controller reaches upright 0.9994 where the zero-torque arm never exceeds −0.99 (`research/external-environments/RESULTS.md` §3) | `scripts/demo.sh --only control` |

Three more results that are certificates rather than runs, so they have no demo:

- **Dense hierarchical supervision is what makes hard synthesis tractable.** Free
  wiring goes 19–38% → 88–94% at 2,120 candidates per node, flat from depth 3 to
  16; on entangled outputs geometry is 12/12 against 0/12 and relations 8/8
  against 1/8. On a decomposable per-element output it is exactly equal to
  output-only supervision, because `probe_loss`'s elementwise BCE already is the
  mean of the per-element losses.
  (`research/search-scaling/RESULTS.md`, `research/perception-ladder/RESULTS.md` §11.)
- **The same staged perception result reproduces on a second domain.**
  `generators/gui` renders a widget tree to raw pixels and emits it as
  `set[(id, parent, kind, x, y, w, h)]` at a declared capacity — width invariant
  from 2 widgets to 24 and nesting 1 to 6, the same shape as `logic`'s `gates`
  channel. Rung one (a widget boundary from a two-pixel neighbourhood) exhausts
  spaces of 1,280, 13,056 and 81,920 programs, each returning exactly **one
  distinct Boolean function**, held-out max error **0.0**, against a majority of
  0.6700 on that screen and a uniform random program conforming 0 times in 400
  draws. The
  selected program is registered and applied at every position by three caller
  nodes. Two recoverability ceilings came back **negative before any search** —
  with borders drawn the two-pixel equality bound is *exactly* the majority
  baseline, advantage 0.0000 — and the search returns 0 conforming programs at
  exactly those settings. (`research/gui-hierarchy/RESULTS.md`; reproduce with
  `.venv/bin/python research/gui-hierarchy/rung1_edges.py --arms narrow --tag scratch`,
  about 11 s — pass `--tag` so it does not overwrite the recorded JSON.)
- **Type legality is structural, not a loss penalty.** Verified by construction
  in `tcn/graph.py`; no track found an escape hatch. It is also load-bearing in
  the wrong direction — see B3.

Framework health, re-run twice today: **179 Python tests pass** (145.79 s and
157.45 s);
**370 of 372** computer-engine tests pass, the two failures being wall-clock
thresholds (`durationMs < 80`, measured 81) on a machine running a dozen other
experiments.

---

## 2. What is measured not to work

Not "untried". Measured, with the arm that beat it.

| claim | verdict | evidence |
|---|---|---|
| Progressive crystallization earns its complexity | **Refuted, four times independently.** Inert at shipped budgets — eleven arms bit-identical, and an arm with no crystallizer and zero extra objective evaluations reaches the same frozen program. Harmful at tight ones: 3/16 conformance against budget-matched argmax's 16/16 (p = 3.2e-06), discarding 94.8% of its optimizer steps. Step-matched argmax beats every scheduler arm on the joint task below saturation, at 2.4–2.5× fewer environment rollouts. The reversible form the third track asked for was then built and also loses: seasons 2.938 against step-matched argmax 3.625 and 3.781 for argmax at the population's summed budget, the latter using 9.3× fewer environment episodes. | `research/crystallization-ablation/RESULTS.md`, `research/perturbation-selection/RESULTS.md`, `research/loss-gated-eligibility/RESULTS.md`, `research/seasons/RESULTS.md` (branch `seasons`) |
| Outside-in ordering, degradation tolerance, rollback | **No positive effect anywhere.** Ordering is inert (arm A identical to arm C); tolerance is inert (arm E bit-identical to arm A in all five configurations). | `research/crystallization-ablation/RESULTS.md` |
| The gradient-connectivity guard protects interior learning | **Does not detect what it claims.** `grad is None` tests autograd reachability, not learning signal; the entropy regularizer keeps every severed logit reachable. A fix was implemented and reverted after measurement: it also flags legitimately concentrated choices and blocked the joint fixture from crystallizing at all (286 deferrals against 4). | `research/crystallization-ablation/RESULTS.md`; `docs/VALIDATION.md` §3.9 |
| Tiny description size | **Refuted.** Joint ships 68,768 bits to express 8 bits of learned content, against 4,896 for an equally-scoring 153-parameter MLP and 32 for an equally-scoring lookup table. Mixed is a tie (17,728 against 20,000). `description_bits` measures JSON verbosity. | `research/baselines/RESULTS.md` §7 |
| Tiny inference cost | **Refuted as stated.** The "four-operation program" runs 450× slower than those four operations in plain Python, and no faster than a 625-parameter torch MLP. Interpreter overhead dominates the `cost = 4` proxy by ~2.5 orders of magnitude. | `research/baselines/RESULTS.md` §5, §7 |
| Differentiable relaxation is the right way to search a typed operator space | **Not as the search itself.** Brute force settles the joint result's 256-way choice in 0.081 ms against 10–36 s of gradient descent, and the mixed fixture's 96 programs in 7 ms against 2.7 s. As difficulty rises the gradient path breaks *first*: at depth 4 it succeeds 0.17 where exhaustive enumeration holds to depth 4 and a hand-written CDCL solver never failed through depth 6. TerpreT's finding reproduces on this repo's own tasks. | `research/enumerative-baseline/RESULTS.md` |
| Relaxing an input address is worse than chance | **Retracted — it was a dead surrogate.** `eq`'s relaxation is exactly 0.0 in float32 past \|a−b\| ≥ 11, so 16 of 16 address gradients were exactly zero and the instrument averaged the survivors. Scaling the surrogate by the carrier width takes the failing arm from 0/12 to 12/12 — which is the right rule for `eq` on bytes and the wrong one for `le` on byte products, where it flattens a correct landscape (loss spread 5.96e-08). | `research/address-wall/RESULTS.md`, superseding `research/perception-ladder/RESULTS.md` §11; `research/object-identity/RESULTS.md` §4.4 |
| A per-pixel program can recover object identity or depth | **Refuted by completeness certificate, not by budget.** The best possible per-pixel predictor — an RGB lookup table, an upper bound on the whole family — fits training pixels perfectly and scores *exactly* the majority baseline on held-out episodes, advantage 0.000, at pixel, pixel+position, aggregate and 3×3 window contexts. Permuting object ids leaves the image identical; recolouring leaves the ids identical. The supervision does not determine the target. | `research/object-identity/out/bounds.json`, `research/discrete-perception/out/rung4_objects.json` |
| A matched neural baseline can learn this task from reward | **Refuted.** A 32-hidden MLP under REINFORCE on the same raw observations sits at 1.977–2.063 at every budget up to 2,000 training episodes — below always-False (2.125) — where the typed program has solved it by 400. With auxiliary losses it is still at chance to 2,560 episodes. | `research/policy-learning/out/e7.log`, `out/e1.json`, `research/baselines/RESULTS.md` §4 |
| Terminal-only reward can train this substrate | **Refuted at every horizon tested.** Dense per-step reward solves horizons 4, 8 and 16 outright; terminal-only reward scores 1.000, 1.000 and 0.859 against oracles of 4, 8 and 16. | `research/policy-learning/out/e5.log` (horizon 32 still running) |
| Advantage normalization is a safe default here | **Refuted.** Three arms that add it land at 2.008 on 0/8 seeds — below always-False — and more episodes do not repair them. | `research/policy-learning/out/e3.log` |
| `relations.closure` is usable | **Unattachable.** `join` inflates set capacity 64 → 4096, no conversion narrows it, and `union` requires identical types, so no program can have the closure's type as output. | `research/perception-ladder/RESULTS.md` §11 |

---

## 3. What is untested

Declared, exercised by sampling and replay, never learned from.

- **Language, beyond one lesson.** As of 2026-09-09 one lesson
  (`context_free_language`) *has* been learned end to end — see section 1 row 9 —
  and that run also bounds the rest of the catalogue, which is the more useful
  result. Only **6 of 179** lessons have prompts byte-predictable at a fixed
  offset, so a constant-address program cannot follow the grammar engine's
  varying realization anywhere else; and `construction` fails to determine
  `answer` in **39 of 179** lessons, so dense staging is structurally unavailable
  for those. The learned program also implements *counting*, not recursion: the
  lesson's negatives always break the bracket count (agreement on 20,000 of
  20,000 seeds), and interrogating the exported program on strings the generator
  cannot emit gives 1.000 agreement with `#( == #)` and 0.429 with Dyck
  membership. (`research/language-capability/RESULTS.md` §1, §2, §3, §6.)
  The curriculum still trains on none of it: the only learning stages in
  `curricula/system.json` are `typed_synthesis` and `joint_prediction_policy`.
- **Computer use.** `generators/computer` has a real Node kernel with 370
  passing engine tests and genuine causal shell/file/app transitions. No program
  has ever been trained on it (`curricula/system.json`, stage `computer`, is
  `sample` plus a replay gate).
- **Raster text, world_2d, world_3d, embodied_world, composition.** Sampled and
  replayed only, same file, same gate.
- **GUI hierarchy above rung one.** The generator, the bounds and rung one are
  done (section 1). Nothing above rung one — grouping, containment, the tree
  itself — has been searched, and two of the dial settings are certified
  unreachable by any program over a two-pixel neighbourhood.
  (`research/gui-hierarchy/RESULTS.md` §0.)
- **Policy learning at any scale beyond 256 programs.** See B2.
- **NES / stable-retro.** Specified in detail in
  `research/external-environments/RESULTS.md` §6, not built.
- **Anything about composition between the ten capabilities in section 1.**
  See B1.
- **Deep circuits.** Live-node fraction falls 1.00 → 0.38 from depth 1 to 16, but
  free and supplied wiring are indistinguishable on that metric, so typed binding
  is not isolated as the cure, and ≥16-node graphs cannot speak to the
  48k–1.8M-gate behavior the DLGN literature reports.
  (`research/search-scaling/RESULTS.md`.)

Two tracks are **in flight right now** and should not be cited as settled.
`research/policy-learning/` has a `README.md` file index and **no `RESULTS.md`**
— `aggregate.py` and the README both refer to tables in a file nobody has
written — with E5 (horizon) and E7 (the 4000-episode budget point) still
executing. `research/object-identity/RESULTS.md` still carries three unrendered
`{{ }}` placeholders in its temperature-policy table, with two `surrogate_fix.py`
processes live. Everything B2 and B3 quote from them comes from raw JSON and
logs, named individually.

---

## 4. Blocking constraints, in priority order

### B1 — Nothing composes across domains. Ten capabilities, ten hand-built scaffolds.

State the good half first, because it is the most effective technique measured
anywhere in this repository. **Staging works, inside a domain.** Freeze a learned
module, register it, and search a second scaffold that calls it, and the cost
collapses: the edge detector is 48 programs in 4.6 s where the undecomposed
target is 4.9e10 programs and 7.6 years projected; the language task is
3.5 s + 363 s where the undecomposed search is 476,256,000 programs and 50.7 days
projected — a factor of about 1.2e4. `positional_scaffold` then applies one such
module at any number of positions with three caller nodes. Three separate tracks
found the same thing independently.

**What does not exist is composition across domains, or any shared library.**
Every row of section 1 is a separately authored graph with its own node names,
its own supervision, and its own evaluation harness under `research/<track>/`.
The segmentation module and the edge detector compose because one agent wrote
both halves of the staging by hand. Nothing connects the edge detector to the
control task, the language module to anything, or any of them to the joint
policy. No registry outlives a single script. There is no curriculum stage that
consumes a module another stage produced — `curricula/system.json` has exactly
two learning stages and neither shares an artifact with the other.

The honest statement of the project's position is: **the substrate supports ten
demonstrations, and no demonstration of the substrate.** Everything below is
downstream of this.

*Check:* `scripts/demo.sh` runs ten demos and shares no learned artifact between
any two of them; `grep -n '"operation"' curricula/system.json`.

### B2 — Reward learning works only under dense per-step reward, on a 256-way choice.

The brief for this task said policy learning from reward has never worked here.
That is now **out of date in both directions**, and the correction matters, so
state it precisely. All numbers are 8 seeds against the same references
(`research/policy-learning/out/refs.json`, horizon 4, 256 test episodes):
always-False **2.125**, always-True 1.875, uniform **2.047**, oracle **4.000**.

**What works.** Reward-only REINFORCE on the typed joint program with a
**zero-initialized** policy readout reaches **4.000 on 8/8 seeds**, and the
budget curve says it saturates at **400 training episodes / 464 environment
episodes** — against the shipped run's 331, which also has a pre-solved decoder
and dense privileged probes. So reward is not seven times more expensive than
supervision here; it is about 1.4×. Staging is cheaper still: 50 probe episodes
then 50 reward episodes is **356 environment episodes** to 4.000 on 8/8 with the
reference program 8/8, and a pinned readout solves it at the smallest budget
tested, 164 environment episodes.
(`out/e1.json` arm `A1_rewardonly_zeroinit`, `out/e7.log`, `out/e4b.log`.)

**What does not.** Four separate walls, each measured:

1. **Terminal-only reward fails at every horizon.** Dense per-step reward solves
   horizon 4, 8 and 16 outright (4.000/4, 8.000/8, 16.000/16); terminal-only
   reward gives 1.000, 1.000 and 0.859. Every real task this substrate is aimed
   at has sparse reward. (`out/e5.log` — **still running**, horizon 32 not
   started.)
2. **Advantage normalization kills it outright.** Three arms that add it
   (`R6_advnorm_batch32`, `R7_statevalue_advnorm_b32`, `R15_rewardonly_bias12_b32`)
   all land at **2.008**, which is *below* always-False, on 0/8 seeds, and more
   episodes do not repair them. Batch 32 alone also breaks it (1/8) unless the
   learning rate is raised tenfold, which repairs it (8/8). The result is real
   but it is not robust to the first three things anyone would try.
   (`out/e3.log`, 19 arms × 8 seeds, complete.)
3. **The matched MLP never learns.** A 32-hidden network under REINFORCE on the
   same raw observations sits at 1.977–2.063 at **every** budget up to 2,000
   training episodes, 4.3× past the point the typed program has solved the task.
   That is below always-False. (`out/e7.log`, `out/e1.json` arm
   `A7_mlp_reinforce_raw`.)
4. **Reward does not identify the program.** The reward objective admits four
   optimal assignments of the 256, and the typed arm selects the *reference* one
   in only 2/8 seeds. Only the probe objective identifies it.
   (`out/space.json`, `out/e1.json`, `research/enumerative-baseline/RESULTS.md` §0.)

**The mechanism, measured rather than inferred.** At zero initialization the
actor gradient with respect to the 16-way choice logits has **norm exactly 0.0**
at every averaging level from 1 to 256 episodes — the logits are `w·z + b` with
`w = b = 0` — and a uniform 16-way truth-table mixture has an exactly zero
Jacobian independently. Even with oracle constants and 256 averaged episodes the
actor direction never ranks `goal_relation`'s optimum above chance (rank 6–7 of
16), while the probe gradient identifies `relation` from **8** averaged episodes.
(`out/e2_direction.json`, `out/cancellation.json`.)

**Treat this track as raw data, not a report.** It has a `README.md` file index
and **no `RESULTS.md`**; `aggregate.py` and the README both refer to tables in a
file that has not been written. Two of its experiments are still executing
(E5 horizon, E7 at budget 4000), and E1's `B1_exactPG_*` arms return **0.000 on
every seed** where E3's `R12/R13/R14` exact-PG arms return 4.000 on 8/8 — nothing
reconciles that, so cite neither as an exact-policy-gradient result.

*Check:* `research/policy-learning/out/{refs,space,e1,e2_direction,cancellation}.json`;
`out/{e3,e4,e4b,e5,e6,e7}.log`.

### B3 — Perception is walled at the type level, and the wall is deliberate.

`role="byte"` makes `Type.numeric` false. Enumerated against `tcn.operators.Registry`
on the current tree, a byte carrier admits `eq`, `identity`, `mux`, `tuple`,
`project`, `index`, `count`, `member`, `insert`/`remove`, `union`/`intersection`,
`pair`/`join`, `delay`, `pack`, `unpack`, and role-preserving `decode`/`quantize`.
It rejects **all** arithmetic (`add`, `sub`, `mul`, `div`, …), all ordering
(`lt`, `le`, `gt`, `ge`), all analytic operators, all reductions (`sum`, `mean`,
`reduce_min`, `reduce_max`), and `fft`.

So the only comparison expressible on a pixel is equality, and the only route
into arithmetic is `pack`, which declares `gradient="none"`. That boundary is
what stopped rung 4: enumeration exhausted the 49,152-program `eq` sub-algebra
and **certified no solution exists**, while a solution that does exist was found
by enumeration in 203 s and by gradient descent in 0/8 runs.

Compounding it: **every surrogate that reads an integer carrier is dead at that
carrier's operating distances**, and this is unfixed on the current tree.
`tcn/learning.py` still reads
`torch.exp(-((a-b)**2).sum(-1,keepdim=True)/temperature)` with
`self.temperatures = {n.name: 1. for n in program.nodes}`. Measured: `eq` is
exactly 0.0 past \|a−b\| ≥ 11 and dead for **235 of the 256** candidate byte
constants on the language task; `lt`'s sigmoid gradient is exactly 0.0 past
distance 17 and therefore exactly 0.0 at that task's operating distance of 48;
and `index`'s kernel keeps only **0.5641** of its mass on the byte it was asked
for. Three defects, one cause: a temperature of 1 against carriers whose declared
range is 2⁸ or 2³². It is also the mechanism behind the retracted "address wall"
(§2).

**One derived rule fixes all three — scale the surrogate temperature by the
carrier's declared range — and one coupling blocks it.** `SoftProgram` uses a
single temperature per node for *both* the candidate softmax and the operator
relaxation, so widening the surrogate simultaneously flattens that node's choice
distribution. Measured on the language stage-A search: raising `eq`'s τ to the
derived 256 drops choice gradients from 9e-2 to 2e-5 and collapses all 8 seeds
onto the same wrong byte. **Separating the two temperatures is a prerequisite for
the fix, not an independent cleanup.** Independently,
`research/object-identity/RESULTS.md` §4.4 measures that the carrier rule is the
wrong one for `le` on byte products, where it flattens a correct landscape — the
right scale there is the decision margin. Nobody has committed to a rule.

Two consequences worth stating as rules, both now measured. **A live surrogate
is necessary but not sufficient**: on the object-identity rung every
temperature-corrected arm is still 0/4, because the neighbour offset is
`grad = None` under *every* policy — `pack` declares `gradient="none"` and no
scaling reaches behind that. And **a gradient arm must report the surrogate's
value at its own operating distance**: a `0/n` printed next to a surrogate of
0.0 says nothing about the method, and a `0/n` next to a loss spread of 5.96e-08
says nothing either.

A second, simpler fault in the same file blocks perception search on its own:
**`SoftProgram` treats any node with `selected is not None` as frozen and
evaluates it through `exact_tensor(...).detach()`**, so a single-candidate node
sitting between a choice and the loss makes that choice's logit report
`grad is None`. Measured directly on the GUI rung-1 scaffold, and independent of
the data. That is one of three separately measured reasons the gradient arm there
scores 1/4 against an exhaustive, certified discrete arm.

*Check:* `grep -n 'temperature' tcn/learning.py`;
`research/perception-ladder/RESULTS.md` §11; `research/address-wall/RESULTS.md`
§7; `research/object-identity/RESULTS.md` §4.4;
`research/language-capability/RESULTS.md` §5 and `research/language-capability/surrogate.json`;
`research/gui-hierarchy/RESULTS.md` §7.

### B4 — Relaxation is not earning its place as the search.

On both flagship tasks brute force settles the discrete content in milliseconds
against seconds to tens of seconds of gradient descent, and returns a
completeness certificate the gradient path cannot. As targets get harder the
gradient path degrades *before* enumeration does. The defensible architecture is
NEAR-shaped: a discrete solver as the default backend for pure synthesis, with
relaxation reserved for the environment-coupled and continuous parts.

The single place the ordering flips is where it should: at the gradient run's own
budget of 704 environment steps, random search over the same 256 candidates
succeeds 0/20 while the gradient path succeeds 5/5; a full enumerative sweep in
the environment needs 16,384 environment steps. And on *value* choices at a fixed
address the gradient path wins outright — 6/6 seeds exact over a 4.3e9-program
byte alphabet where brute force projects to 107 days.

`tcn/search.py` now ships and `tcn/cli.py:mixed` runs it automatically whenever
the space is small enough, which is the right default. What has not happened is
the corresponding revision of `ARCHITECTURE.md` sections 4 and 5.

*Check:* `research/enumerative-baseline/RESULTS.md`;
`research/discrete-perception/RESULTS.md` §"where each method actually wins".

### B5 — `source_fingerprint` invalidates every recorded episode on any core change.

`tcn/generation.py:source_fingerprint()` hashes every `.py`/`.ts`/`.json` under
`tcn/` **and** `generators/`, and `Host.restore` refuses any mismatch. Adding a
generator therefore invalidates episodes recorded from every other generator.
Confirmed today: **all eleven** `artifacts/system/*/episode.json.gz` fail to
reload with `ValueError: episode source revision mismatch`.
`artifacts/validation-environment.json` records `2fc0600d…`, and the live
fingerprint moved at least three times during this session alone — `66f9c079…`
after the overnight merges, `e9d08f91…` once the GUI generator and a
`generators/computer` edit landed. On a repository with six agents committing,
this invalidates recorded episodes roughly hourly.

Those particular files have a second problem worth separating: they carry **no
`source` key at all**, so the comparison fails trivially rather than because the
revision moved. Freshly recorded episodes do round-trip. The proposed narrowing
(`source_fingerprint(name)` over `tcn` plus one generator) is written up but not
applied, and `generators/control/manifest.json` still has no `engines` field
pinning the MuJoCo version an episode was recorded against.

*Check:* `.venv/bin/python -c "from tcn.generation import Host; Host.load('artifacts/system/logic/episode.json.gz')"`;
`research/external-environments/RESULTS.md` §7.3.

### B6 — Seeds do not do anything in the shipped fixtures.

`SoftProgram.__init__` zero-initializes every choice logit
(`torch.zeros(len(n.candidates))`), so `torch.manual_seed` does not perturb
synthesis at all: four seeds produce one identical initialization. Every per-seed
number produced by a shipped entry point is one outcome repeated N times. Every
research track that reports a spread added explicit initialization noise and said
so; `examples/` and `curricula/` do not. Any future "N of M seeds" claim from the
shipped path is meaningless until this changes.

*Check:* `grep -n 'torch.zeros' tcn/learning.py`;
`research/perception-ladder/RESULTS.md` fault P2.

### B7 — The documented entry point does not run.

`.venv/bin/tcn` has the shebang
`#!/home/brandonin/Documents/differentiable-agentic-software/.venv/bin/python3`,
which does not exist. Every `.venv/bin/tcn ...` command in `README.md` fails with
"No such file or directory". This is cosmetic to fix and expensive to leave: it
is the first thing a reader tries. `scripts/demo.sh` uses `python -m tcn`.

*Check:* `head -1 .venv/bin/tcn`.

### B8 — `tcn/search.py` cannot score a recurrent program.

The discrete backend that B4 says should be the default only handles feed-forward
scoring against declared `Signal`s. The depth-generalization track had to write
its own enumerator to score a recurrent program by settled return over live
episodes. So the two capabilities that most need a discrete reference —
recurrence and environment-coupled search — are exactly the two it cannot serve.

It is also slower than it needs to be: `enumerate_fit` re-executes whole programs
per candidate, where a prefix-reusing walk returns the identical conforming set
**17.8× faster** at R=8 and is flat in observation width where the current loop
is linear. Two tracks had to import that walk from `research/discrete-perception/`
to finish their sweeps at all — the object-identity 393,216-program search is
196.1 s with it against a projected 1,217 s without. The improvement is written
up and unapplied.

*Check:* `research/depth-generalization/RESULTS.md` §5.4;
`research/discrete-perception/RESULTS.md` (fault D6);
`research/object-identity/RESULTS.md` §9; `tcn/search.py`.

### B9 — Two smaller faults that will bite.


- **`relaxed('tuple')` does not broadcast.** It is a bare
  `torch.cat(xs, dim=-1)`, so mixing a batched value with an unbatched constant
  raises `RuntimeError: Tensors must have same number of dimensions`, while a
  binary arithmetic operator on the same pair broadcasts correctly. Any scaffold
  that packs a trainable constant alongside a batched intermediate hits it.
  (`research/perception-ladder/RESULTS.md`, fault P1.)
- **Non-uniqueness is the norm, the tie-break is wrong, and the known fix does
  not always work.** Every rung-3 arm has 2,464–2,608 of 32,000 programs
  conforming, about 5% of which disagree with the renderer on fresh episodes, and
  the count barely moves as supervision grows eightfold. `enumerate_fit`'s
  lexicographic pick was measurably wrong on fresh episodes there; it was wrong
  again on language, where 10 of 45,375 conform and declaration order returns one
  that fails at the longest unseen length. `rank='description'`/`'cost'` now
  exists, and requiring exactness on a validation split fixed both — but on the
  GUI generator at 2 widgets that same filter leaves 4 survivors denoting 2
  functions with a survivor held-out max error of **1.0000**, because a
  validation split drawn from equally sparse screens does not separate the
  candidates either. What fixed it there was denser supervision, which is a
  statement about the probe, not about the ranking rule. Nothing in `tcn/`
  enforces any of this.
  (`research/discrete-perception/RESULTS.md`;
  `research/language-capability/RESULTS.md` §4;
  `research/gui-hierarchy/RESULTS.md` §0 result 5.)

---

## 5. What the project should be told, in one paragraph

The defensible claim is **typed candidate libraries with exact export, dense
hierarchical supervision, and staged reuse**: when the library contains the
answer this system recovers it exactly and generalizes where a fitted network
does not; dense intermediate probes make that search tractable at depth; and
freezing a learned module so a second search can call it collapses targets that
are otherwise years of brute force into seconds. Three claims the architecture
led with have since been settled, and `ARCHITECTURE.md` sections 5 and 8.1 have
been rewritten accordingly rather than softened. **Progressive crystallization**
is refuted outright, four times, including in the reversible form its third
refutation asked for. **Tiny description size** is refuted *as shipped* — the
visual artifact is 117.7 MB of which 99.68% is repeated type declarations —
while the canonical program is 41 KB and the learned content 21.3 bits; all
three numbers are real and none may stand in for another. **Tiny inference
cost** is likewise refuted as currently implemented, at 146× to 90,400× slower
than the same function in plain Python, with 97.7% attributed to
decode/encode/validate marshalling against 0.56% for operator semantics plus the
graph walk. Whether that third one is a property of the *method* or only of the
*interpreter* is not yet known and is the project's first open question; it is
being decided by a compiled-runtime experiment with a four-arm A/B harness, not
by argument. A discrete backend belongs in section 2's candidate inventory. Two capabilities the measurement pass wrote off, recursive
abstraction and structural generalization, turned out to be instrumentation
faults and now work, which is the strongest argument in the file for fixing the
instrument before believing any negative result. The gap between here and a
system is not any single mechanism: it is that ten working demonstrations share
no learned artifact, that reward has taught this substrate one 256-way choice
under dense per-step reward and nothing under sparse, and that every surrogate
reading an integer carrier is numerically dead at that carrier's own operating
distances.
