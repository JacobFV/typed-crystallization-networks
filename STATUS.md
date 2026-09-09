# STATUS — 2026-09-09

What works, what is measured not to work, what has never been tested, and what
is in the way. Every claim below cites either a command you can run or a file
under `research/`. Anything that could not be traced to one has been cut.

Read `docs/VALIDATION.md` for the corrected record and `research/FINDINGS.md`
for the consolidated measurement pass. This file is the short, hostile version.

## How to check any of it

```bash
scripts/demo.sh                 # every demonstrated capability, ~4 minutes
scripts/demo.sh --only depth    # one of them
scripts/demo.sh --full          # more seeds, wider inputs
scripts/demo.sh --list
```

Artifacts land in `artifacts/demo/<name>/result.json`, with the table in
`artifacts/demo/summary.md`. Exit status is non-zero if anything fails to
reproduce. Note that `.venv/bin/tcn` does not run at all (blocker B7); use
`.venv/bin/python -m tcn`.

---

## 1. What works, with evidence

Nine capabilities reproduce on the current tree. Each row's measured number is
printed next to the reference it beats, because on this project that has twice
been the difference between a result and a mistake.

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
| 9 | MuJoCo behind the generator contract | replay, snapshot/restore and cross-process reload all bit-identical, max \|Δ\| 0.0 | a scripted energy-pumping controller reaches upright 0.9994 where the zero-torque arm never exceeds −0.99 (`research/external-environments/RESULTS.md` §3) | `scripts/demo.sh --only control` |

Two more results that are certificates rather than runs, so they have no demo:

- **Dense hierarchical supervision is what makes hard synthesis tractable.** Free
  wiring goes 19–38% → 88–94% at 2,120 candidates per node, flat from depth 3 to
  16; on entangled outputs geometry is 12/12 against 0/12 and relations 8/8
  against 1/8. On a decomposable per-element output it is exactly equal to
  output-only supervision, because `probe_loss`'s elementwise BCE already is the
  mean of the per-element losses.
  (`research/search-scaling/RESULTS.md`, `research/perception-ladder/RESULTS.md` §11.)
- **Type legality is structural, not a loss penalty.** Verified by construction
  in `tcn/graph.py`; no track found an escape hatch. It is also load-bearing in
  the wrong direction — see B3.

Framework health, re-run today: **179 Python tests pass in 145.79 s**;
**370 of 372** computer-engine tests pass, the two failures being wall-clock
thresholds (`durationMs < 80`, measured 81) on a machine running a dozen other
experiments.

---

## 2. What is measured not to work

Not "untried". Measured, with the arm that beat it.

| claim | verdict | evidence |
|---|---|---|
| Progressive crystallization earns its complexity | **Refuted, three times independently.** Inert at shipped budgets — eleven arms bit-identical, and an arm with no crystallizer and zero extra objective evaluations reaches the same frozen program. Harmful at tight ones: 3/16 conformance against budget-matched argmax's 16/16 (p = 3.2e-06), discarding 94.8% of its optimizer steps. Step-matched argmax beats every scheduler arm on the joint task below saturation, at 2.4–2.5× fewer environment rollouts. | `research/crystallization-ablation/RESULTS.md`, `research/perturbation-selection/RESULTS.md`, `research/loss-gated-eligibility/RESULTS.md` |
| Outside-in ordering, degradation tolerance, rollback | **No positive effect anywhere.** Ordering is inert (arm A identical to arm C); tolerance is inert (arm E bit-identical to arm A in all five configurations). | `research/crystallization-ablation/RESULTS.md` |
| The gradient-connectivity guard protects interior learning | **Does not detect what it claims.** `grad is None` tests autograd reachability, not learning signal; the entropy regularizer keeps every severed logit reachable. A fix was implemented and reverted after measurement: it also flags legitimately concentrated choices and blocked the joint fixture from crystallizing at all (286 deferrals against 4). | `research/crystallization-ablation/RESULTS.md`; `docs/VALIDATION.md` §3.9 |
| Tiny description size | **Refuted.** Joint ships 68,768 bits to express 8 bits of learned content, against 4,896 for an equally-scoring 153-parameter MLP and 32 for an equally-scoring lookup table. Mixed is a tie (17,728 against 20,000). `description_bits` measures JSON verbosity. | `research/baselines/RESULTS.md` §7 |
| Tiny inference cost | **Refuted as stated.** The "four-operation program" runs 450× slower than those four operations in plain Python, and no faster than a 625-parameter torch MLP. Interpreter overhead dominates the `cost = 4` proxy by ~2.5 orders of magnitude. | `research/baselines/RESULTS.md` §5, §7 |
| Differentiable relaxation is the right way to search a typed operator space | **Not as the search itself.** Brute force settles the joint result's 256-way choice in 0.081 ms against 10–36 s of gradient descent, and the mixed fixture's 96 programs in 7 ms against 2.7 s. As difficulty rises the gradient path breaks *first*: at depth 4 it succeeds 0.17 where exhaustive enumeration holds to depth 4 and a hand-written CDCL solver never failed through depth 6. TerpreT's finding reproduces on this repo's own tasks. | `research/enumerative-baseline/RESULTS.md` |
| Relaxing an input address is worse than chance | **Retracted — it was a dead surrogate.** `eq`'s relaxation is exactly 0.0 in float32 past \|a−b\| ≥ 11, so 16 of 16 address gradients were exactly zero and the instrument averaged the survivors. Scaling the surrogate takes the failing arm from 0/12 to 12/12. | `research/address-wall/RESULTS.md`, superseding `research/perception-ladder/RESULTS.md` §11 |
| A per-pixel program can recover object identity or depth | **Refuted by completeness certificate, not by budget.** The best possible per-pixel predictor — an RGB lookup table, an upper bound on the whole family — fits training pixels perfectly and scores *exactly* the majority baseline on held-out episodes, advantage 0.000, at pixel, pixel+position, aggregate and 3×3 window contexts. Permuting object ids leaves the image identical; recolouring leaves the ids identical. The supervision does not determine the target. | `research/object-identity/out/bounds.json`, `research/discrete-perception/out/rung4_objects.json` |
| A matched neural baseline cannot learn from reward here | **Confirmed.** A 32-hidden MLP under REINFORCE on the same raw observations stays at 2.023 (chance ≈ 2.0) over 2,160 environment episodes, and at every budget up to 2,560 with auxiliary losses. | `research/policy-learning/out/e1.json`, `research/baselines/RESULTS.md` §4 |
| `relations.closure` is usable | **Unattachable.** `join` inflates set capacity 64 → 4096, no conversion narrows it, and `union` requires identical types, so no program can have the closure's type as output. | `research/perception-ladder/RESULTS.md` §11 |

---

## 3. What is untested

Declared, exercised by sampling and replay, never learned from.

- **Language.** `generators/language` produces 179 executable lessons × two
  seeds, all checked non-empty. No model has ever been trained on it. The only
  learning stages in `curricula/system.json` are `typed_synthesis` and
  `joint_prediction_policy`; every other stage is `sample` plus a replay gate.
- **Computer use.** `generators/computer` has a real Node kernel with 370
  passing engine tests and genuine causal shell/file/app transitions. Never
  trained on. Same evidence.
- **Raster text, world_2d, world_3d, embodied_world, composition.** Sampled and
  replayed only, same file.
- **GUI hierarchy.** `generators/gui` is in the tree with 13 passing tests and a
  running measurement track, but `research/gui-hierarchy/` has only
  `RESULTS.template.md` — no rendered `RESULTS.md`, and `out/rung1_free.log` has
  no matching JSON. Treat its numbers as in flight.
- **Policy learning at any scale beyond 256 programs.** See B2.
- **NES / stable-retro.** Specified in detail in
  `research/external-environments/RESULTS.md` §6, not built.
- **Anything about composition between the nine capabilities in section 1.**
  See B1.
- **Deep circuits.** Live-node fraction falls 1.00 → 0.38 from depth 1 to 16, but
  free and supplied wiring are indistinguishable on that metric, so typed binding
  is not isolated as the cure, and ≥16-node graphs cannot speak to the
  48k–1.8M-gate behavior the DLGN literature reports.
  (`research/search-scaling/RESULTS.md`.)

---

## 4. Blocking constraints, in priority order

### B1 — Nothing composes. Nine capabilities, nine hand-built scaffolds.

This is the blocker, and it is not on anyone's list because no track measured it.
Every row of section 1 is a separately authored graph with its own node names,
its own supervision, and its own evaluation harness under `research/<track>/`.
There is no measured path from the segmentation module to the edge detector
except a hand-written staging step, none from the edge detector to the control
task, and none from any of them to the joint policy. The one composition
mechanism that does work — `positional_scaffold`, three caller nodes, any width
— composes a module *with itself*, not two different capabilities.

The honest statement of the project's position is: **the substrate supports nine
demonstrations, and no demonstration of the substrate.** Everything below is
downstream of this.

*Check:* `scripts/demo.sh` runs nine demos and shares no learned artifact between
any two of them.

### B2 — Policy learning works only on a 256-way choice, and not the way the record says.

The brief for this task said policy learning from reward has never worked here.
That is now **out of date and the correction matters**, so state it precisely.

- Reward-only REINFORCE on the typed joint program with a **zero-initialized**
  policy readout reaches **4.00/4 on 8/8 seeds** — at 2,000 training episodes /
  2,256 environment episodes, against the shipped run's 331.
  (`research/policy-learning/out/e1.json`, arm `A1_rewardonly_zeroinit`.)
- It selects the *reference* program only **2/8** times; the rest land on the
  other reward-optimal assignment. Only the probe objective identifies the
  reference one. (Same file; `research/enumerative-baseline/RESULTS.md` §0.)
- The matched MLP does stay at chance: **2.023** over 2,160 environment episodes.
  (Arm `A7_mlp_reinforce_raw`.)
- The mechanism: with zero-initialized policy constants the actor gradient with
  respect to the 16-way choice logits is **exactly zero** — the logits are
  `w·z + b` with `w = b = 0`. With oracle constants and 256 averaged episodes the
  actor direction finds `relation` but **never** ranks `goal_relation`'s optimum
  above chance (rank 7 of 16), while the probe gradient finds `relation` from 8
  averaged episodes. (`research/policy-learning/out/e2_direction.json`.)
- The cheapest reliable route measured anywhere: **50 probe episodes then 50
  reward episodes = 356 environment episodes** to 4.00 on 8/8 with the reference
  program 8/8. (`research/policy-learning/out/e4b.json`.)

So the blocker is narrower and sharper than "reward does not work": **reward has
learned exactly one thing here, a 256-way discrete choice on a horizon-4 task,
at seven times the sample cost of supervision, and it does not identify the
program.** The flagship result's 4/4 still comes from a policy decoder
initialized to the exact solution plus dense privileged probes, not from reward.
The track has no `RESULTS.md` yet and `e3_remedies.py` is still running; do not
treat its remedy table as final.

*Check:* `research/policy-learning/out/e1.json`, `e2_direction.json`, `e4b.json`.
`e5_horizon.py` crashes on a duplicate keyword argument and `e7_budget.py` has
never been run, so horizon scaling and the return-versus-budget curve do not
exist.

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

Compounding it: **`eq`'s surrogate is exactly 0.0 in float32 past \|a−b\| ≥ 11**,
and this is unfixed on the current tree. `tcn/learning.py` still reads
`torch.exp(-((a-b)**2).sum(-1,keepdim=True)/temperature)` with
`self.temperatures = {n.name: 1. for n in program.nodes}`. On 0–255 data that
kills the gradient for all but near-equal bytes, and it is the mechanism behind
the retracted "address wall" (§2). The one-line fix — scale τ by the carrier
width — is derived rather than tuned and takes a failing arm from 0/12 to 12/12,
but `research/object-identity/RESULTS.md` §4.4 measures that the *same* rule is
wrong for `le` on byte products, where it flattens a correct landscape. The
right rule is the decision margin, not the carrier, and nobody has committed to
one. A related coupling has to be fixed first: `SoftProgram` uses **one
temperature per node** for both the candidate softmax and the operator
relaxation, so a surrogate cannot be widened without simultaneously flattening
that node's choice distribution.

*Check:* `grep -n 'temperature' tcn/learning.py`;
`research/perception-ladder/RESULTS.md` §11; `research/address-wall/RESULTS.md`
§7; `research/object-identity/RESULTS.md` §4.4.

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
reload with `ValueError: episode source revision mismatch`. The current
fingerprint is `66f9c079…`; `artifacts/validation-environment.json` records
`2fc0600d…`.

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

*Check:* `research/depth-generalization/RESULTS.md` §5.4; `tcn/search.py`.

### B9 — Two smaller faults that will bite.

- **`relaxed('tuple')` does not broadcast.** It is a bare
  `torch.cat(xs, dim=-1)`, so mixing a batched value with an unbatched constant
  raises `RuntimeError: Tensors must have same number of dimensions`, while a
  binary arithmetic operator on the same pair broadcasts correctly. Any scaffold
  that packs a trainable constant alongside a batched intermediate hits it.
  (`research/perception-ladder/RESULTS.md`, fault P1.)
- **Non-uniqueness is the norm and the tie-break was wrong.** Every rung-3 arm
  has 2,464–2,608 of 32,000 programs conforming, about 5% of which disagree with
  the renderer on fresh episodes, and the count barely moves as supervision grows
  eightfold. `enumerate_fit`'s lexicographic pick was measurably wrong on fresh
  episodes. `rank='description'`/`'cost'` now exists, but requiring exactness on
  a validation split is what actually fixed it, and nothing enforces that.
  (`research/discrete-perception/RESULTS.md`.)

---

## 5. What the project should be told, in one paragraph

The defensible claim is **typed candidate libraries with exact export and dense
hierarchical supervision**: when the library contains the answer, this system
recovers it exactly, generalizes where a fitted network does not, and dense
intermediate probes make that search tractable at depth. Three claims the
architecture leads with — progressive crystallization, tiny description size,
tiny inference cost — are refuted by measurement and should be removed from
`ARCHITECTURE.md` sections 4 and 5 rather than softened. Two capabilities the
measurement pass wrote off, recursive abstraction and structural generalization,
turned out to be instrumentation faults and now work. The gap between here and a
system is not any single mechanism; it is that nine working demonstrations share
no learned artifact, and that reward has taught this substrate exactly one
256-way choice.
