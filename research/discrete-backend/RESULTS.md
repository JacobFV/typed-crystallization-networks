# Three new modes for the discrete backend

`tcn/search.py` could score one thing: a feed-forward program run once per
supervised example and compared against probe targets. That excluded the two
settings the project cares about most. The depth-generalization track built a
circuit interpreter that evaluates one gate per tick through `Program.state`,
exact at depths 1 through 8, and recorded plainly that "`tcn/search.py` cannot
score a recurrent program at all". The policy-learning track found that
enumerating against a frozen exact world model reaches the oracle in **27
environment episodes**, an order of magnitude cheaper than anything else
measured — with a hand-rolled loop, not through the backend. Both capabilities
lived outside the search infrastructure that was supposed to own them.

They do not any more. This report states the semantics chosen for each mode, why,
the acceptance measurements, and where each stops.

**Answer to the two questions this track was set.** The discrete backend now
handles recurrence and live environment return: the depth interpreter is
certified unique in 0.61 s where the feed-forward mode finds nothing at all, and
the 27-episode result reproduces exactly through the backend with a certificate
the original did not have. The beam gives up the certificate, and on the one real
task measured here it gives up more than that: at width 1 it fails to find a
solution that 8.2 % of the space satisfies.

---

## 0. What changed

| file | change |
|---|---|
| `tcn/search.py` | the only core file touched: `SearchResult` gains an explicit `certificate` and the cost fields the new modes need; `enumerate_recurrent`, `enumerate_environment`, `enumerate_prefix` (with `beam`), `EpisodeLedger`, `EnvironmentTask`, `probe_examples`, `DiscreteProblem`/`route`/`solve`/`viability` are new |
| `tests/test_search_modes.py` | new; 21 tests over the three modes, the certificate contract and the router |
| `research/discrete-backend/` | new; the three acceptance scripts, `report.py`, raw JSON in `out/` |

Nothing else under `tcn/` is modified. `tcn/learning.py` and `tcn/select.py` are
untouched, as the two queued merges require. The full suite is 200 passed
(179 before, +21).

`tcn/select.py` **is not on main**, so task 4 is exposed rather than performed:
`DiscreteProblem`, `route` and `solve` are the interfaces a selector calls, and
`test_routing_sends_each_problem_shape_to_the_mode_that_can_score_it` pins the
routing table. Wiring `mode="auto"` to them is a three-line change in the
selector when it lands.

## 1. The certificate is now a field, not a convention

Every mode returns the same `SearchResult`, and every `SearchResult` now carries
`certificate`:

| value | meaning |
|---|---|
| `'unique'` | the space was exhausted and exactly one program conforms |
| `'complete'` | the space was exhausted and the reported conforming set is the whole of it — zero members or many |
| `'none'` | a budget, an early exit, or a beam left part of the space unexamined; nothing is claimed about what was not looked at |

This exists because the uniqueness certificate is the backend's main advantage
over a gradient run, and the beam is a mode whose entire purpose is to trade it
away. Making the trade a returned value rather than a property of the call site
means it cannot be forfeited silently. `exhausted`, `unique` and `certificate`
are kept mutually consistent in all four modes, and `discarded` counts exactly
the programs a beam dropped without deciding — nonzero exactly when the beam,
rather than a budget, is what cost the certificate.

---

## 2. Recurrence

### Semantics, chosen and stated

A recurrent candidate is scored on a **declared tick budget with a declared
trailing settle window**. It is not run until its state stops changing.

Both readings were on the table and the brief asked for a decision rather than a
guess. "Run until stable" was rejected for three reasons.

1. It needs a stability detector, and that detector is a modelling choice that
   would sit inside the scorer where no experiment can see it. A program that is
   wrong at every tick in the same way is perfectly stable.
2. It makes one evaluation's cost data-dependent and in principle unbounded, so a
   sweep can no longer state the work it will do before doing it.
3. It makes two candidates that settle at different ticks incomparable on one
   budget — which is precisely what a certificate needs them to be.

A declared horizon is instead an ordinary configuration record of the kind
ARCHITECTURE section 7 already requires for time alignment; it fixes the cost at
`ticks × examples × programs`; and it is what the depth track measured against
(horizon 12, trailing window 4).

The trailing window subsumes the alternative rather than discarding it.
`settle_window=1` is exactly "run to a fixed horizon and read the final value".
`settle_window=k` additionally requires the scored value to be correct at each of
the last k ticks — a settling criterion, "arrived and stayed", at a bounded cost.
`tick_targets` covers the third shape, an explicit per-tick target sequence. An
example may carry both.

Prefix reuse is **not** offered here, and that is a real limit rather than an
omission: a node inside tick t>0 depends on state committed at tick t−1, which
depends on every node, so its value is not a function of the choice prefix and
the walk of `enumerate_prefix` would be unsound. `enumerate_prefix` refuses a
program with state rather than silently producing a wrong answer.

### Acceptance: the depth-generalization scaffold

`research/discrete-backend/recurrent_depth.py`. The interpreter of
`research/depth-generalization/interpreter.py`, unchanged: one fixed graph with
no per-gate node and no per-depth constant, evaluating one gate per tick.
Training episodes are depths 1-2 only; evaluation is depths 3, 4, 6 and 8 on
held-out addresses. Probes are the generator's own `gate` and `target` channels,
both on the probe channel, nothing read off the observation stream.

| arm | space | free nodes | train episodes | fit conforming | fit s | recurrent conforming | certificate | recurrent s | node evaluations |
|---|---|---|---|---|---|---|---|---|---|
| pinned | 272 | relation:17 goal_relation:16 | 16 | 0 | 0.11 | 1 | unique | 0.61 | 3,133,440 |
| free | 1088 | va:2 vb:2 relation:17 goal_relation:16 | 16 | 0 | 0.30 | 1 | unique | 2.40 | 12,533,760 |

| arm | selection | depth | episodes | probe max error | mean settled return | exact |
|---|---|---|---|---|---|---|
| pinned | relation=16 goal_relation=6 | 1 | 40 | 0.0 | 4.00 | 40/40 |
| pinned | relation=16 goal_relation=6 | 2 | 40 | 0.0 | 4.00 | 40/40 |
| pinned | relation=16 goal_relation=6 | 3 | 40 | 0.0 | 4.00 | 40/40 |
| pinned | relation=16 goal_relation=6 | 4 | 40 | 0.0 | 4.00 | 40/40 |
| pinned | relation=16 goal_relation=6 | 6 | 40 | 0.0 | 4.00 | 40/40 |
| pinned | relation=16 goal_relation=6 | 8 | 40 | 0.0 | 4.00 | 40/40 |
| pinned | | *total* | 496 episodes, 2880 steps | | | |
| free | va=0 vb=0 relation=16 goal_relation=6 | 1 | 40 | 0.0 | 4.00 | 40/40 |
| free | va=0 vb=0 relation=16 goal_relation=6 | 2 | 40 | 0.0 | 4.00 | 40/40 |
| free | va=0 vb=0 relation=16 goal_relation=6 | 3 | 40 | 0.0 | 4.00 | 40/40 |
| free | va=0 vb=0 relation=16 goal_relation=6 | 4 | 40 | 0.0 | 4.00 | 40/40 |
| free | va=0 vb=0 relation=16 goal_relation=6 | 6 | 40 | 0.0 | 4.00 | 40/40 |
| free | va=0 vb=0 relation=16 goal_relation=6 | 8 | 40 | 0.0 | 4.00 | 40/40 |
| free | | *total* | 496 episodes, 2880 steps | | | |

**What this says.**

* The feed-forward mode finds **0 conforming programs** in both arms. It is not
  slow at the recurrence; it cannot see it. Running the identical program on the
  identical data, `enumerate_fit` executes tick 0 only, before the fold has run
  and before the answer wire exists. That is the exact claim FINDINGS section 15
  recorded, reproduced here as a measurement rather than an assertion.
* The recurrent mode returns **exactly one** conforming program in both arms and
  certifies it `unique` — the 272-program pinned space and the 1,088-program free
  space alike. The selection is `relation=16` (the table-conditioned `index`
  lookup) and `goal_relation=6` (xor), which is `check_exact.py`'s reference
  program; in the free arm it also recovers `va=vb=0`, the general `mux` wire
  lookup.
* That last point is worth stating separately. FINDINGS section 15 reports that
  searching the wire binding rather than declaring it **costs 2 of 8 gradient
  seeds**. Exhaustive search over the same 1,088-program space loses nothing and
  certifies the answer unique in 2.40 s. This is the certificate obligation of
  AGENTS.md discharged for that scaffold: the answer provably lies in the space,
  so the depth track's pinning was a prior and not the answer.
* Held-out generalization is exact at every depth: probe max error 0.0 and mean
  settled return 4.00/4 with 40/40 episodes perfect at depths 1, 2, 3, 4, 6 and 8
  — the depth track's headline row, now produced by the backend.
* The whole sweep, both arms, all held-out evaluation, is 8.7 s wall. The depth
  track's bespoke return-scoring loop took 25 s for the 272-program arm alone.
  The difference is the tolerance short-circuit: `evaluate_recurrent` abandons a
  candidate at the first scored tick that misses, and most candidates miss at the
  first scored tick.

### Where the recurrent mode stops

Throughput measured on this scaffold is **≈87,000 program-ticks per second**
(52,224 program-ticks in 0.61 s pinned; 208,896 in 2.40 s free — 4× the space at
3.9× the time, so linear as expected). At 16 examples and 12 ticks that is 192
program-ticks per candidate, so one hour of CPU exhausts roughly **1.6 × 10⁶
programs** at this shape.

Two things make that optimistic and one makes it pessimistic. Optimistic: the
short-circuit means most candidates die at the first scored tick, and a scaffold
where candidates survive further costs proportionally more; and a wider settle
window or per-tick targets raise the per-candidate cost toward the full
`ticks × examples`. Pessimistic: `viability()` projects at a flat rate and
ignores the short-circuit entirely (it projected 2.6 s against 0.61 s measured
and 10.4 s against 2.40 s, so it is conservative by about 4× here).

The hard limit is that **prefix reuse is unavailable**, so the recurrent mode
cannot borrow the 45× that section 4 gets on feed-forward scaffolds. A recurrent
space above roughly 10⁷ candidates is out of reach for exhaustive scoring, and
there is no in-mode remedy — the remedy is staging, exactly as FINDINGS
section 14 found for perception.

---

## 3. Live environment return

### Semantics, chosen and stated

A candidate is scored by the return it actually earns. Harden the selection, drop
the trainable-constant flag, hand the frozen program to the shipped
`tcn.agent.Agent`, roll it in the generator the task names, and sum the
`reward_components` of the step records over a declared trailing window. That is
the same signal `Agent` already drives and the same quantity `JointTrainer`
reinforces; only the search over it moves inside the backend.

Conformance has two readings and both are supported because they answer different
questions.

* With a `threshold`, a program conforms when its mean return reaches it.
  Exhausting the space then certifies **how many programs in the whole space
  reach the objective**, and `'unique'` means exactly one does.
* Without one, conformance means "attains the best return observed", which is
  only meaningful on an exhausted sweep. This is what the depth track's bespoke
  loop computed.

`exact_max_error` keeps its meaning as the shortfall `max(0, threshold − best)`,
zero exactly when the sweep solved.

**Environment episodes and steps are first-class returned costs.** FINDINGS
section 8 established that environment rollouts are the one axis on which the
relaxed path genuinely wins, so a search that spends them silently is not
comparable with one that does not. Every `Host` the module creates goes through
an `EpisodeLedger`, and the ledger can be shared across stages so a staged search
reports **one honest total**. `probe_examples` harvests supervised examples from
real episodes through the same ledger, which is what makes the staged number
below a single figure rather than two that have to be added by hand.

Prefix reuse and a beam are **not** offered here: return is only defined at a
complete program — there is nothing to roll out with half the nodes chosen — so
the choice tree has no partial score. The way to spend fewer episodes is to shrink
the candidate set with a supervised sweep first, which is what the acceptance
below measures.

### Acceptance: 27 environment episodes, through the backend

`research/discrete-backend/environment_policy.py`. The `examples/joint.py`
scaffold with the hand-supplied policy decoder removed, as
`research/policy-learning/pl.py` requires. The one change of shape: the readout
sign is a two-candidate node rather than four trainable constants, because a
discrete backend cannot search a continuous parameter. That is a representation
change, not an easier problem — the same one bit still has to come from reward.

| stage | mode | episodes | space | evaluated | conforming | certificate | s |
|---|---|---|---|---|---|---|---|
| 1 supervision | fit | 25 | 512 | 512 | 2 | complete | 0.03 |
| 2 reward | environment | 2 | 2 | 2 | 1 | unique | 0.00 |
| **total** | | **27** | | | | | |

selection {'relation': 6, 'goal_relation': 6, 'policy': 0}, 8 environment steps
held out 4.00/4 over 64 episodes, 64/64 perfect; always_false 2.12, always_true 1.88, uniform 2.05, oracle 4.00

| unstaged, episodes per program | episodes | programs | conforming | certificate | s |
|---|---|---|---|---|---|
| 1 | 512 | 512/512 | 256 | complete | 1.1 |
| 4 | 2048 | 512/512 | 32 | complete | 1.8 |
| 16 | 8192 | 512/512 | 8 | complete | 9.0 |

**What this says.**

* **27 environment episodes**, 25 supervised and 2 reward, reproducing FINDINGS
  section 22's cheapest row exactly, through `tcn.search` with both stages
  charged to one ledger. Held-out return **4.00/4 over 64 episodes, 64/64
  perfect**, against always-false 2.12, always-true 1.88, uniform 2.05 and
  oracle 4.00 — the same four references section 22 reports, measured on the same
  episodes and reproducing its numbers.
* The staged sweep produces something the original could not: a **certificate at
  both stages**. Supervision exhausts all 512 programs and returns `conforming=2`
  with a `'complete'` certificate — that is the discrete backend *proving* that
  probes determine everything except exactly one bit, which is section 22's "of
  the 9 bits of learned content, 8 are supervision-driven and 1 is
  reward-driven", certified rather than inferred. The reward stage then exhausts
  the 2-program residual and returns `'unique'`.
* The unstaged arm is where the mode stops, and the number is stark. Enumerating
  the whole 512-program space by live return costs **512 episodes at one episode
  per program and still leaves 256 conforming**; 2,048 episodes leaves 32; 8,192
  episodes leaves 8. Reward never identifies the program uniquely here at any
  budget measured, because many distinct programs induce the same action map on
  the observed distribution. Supervision pins it to 2 for 25 episodes.
* So staging is not an optimization on this task. **It is the difference between
  27 episodes with a certificate and 8,192 episodes without one** — a 300× ratio
  on the resource that matters, with the better outcome on the cheaper side.

### Where the environment mode stops

The cost is `space_size × episodes_per_program` episodes, exactly, and it is
reported. `logic` is a trivially cheap generator, so 8,192 episodes is 9.0 s of
wall clock; that is not the constraint anywhere it matters. On the `computer`
generator an episode is a live OS interaction of seconds to minutes, so the same
512-program sweep is hours to days, and FINDINGS section 23's 7,480-program
enumeration would be unreachable by return even though it was reachable by
supervision in 168 s.

The rule this makes explicit: **the environment mode is viable only on a residual
space already reduced by supervision.** It is the last stage of a staged search,
never the first. `viability(program, episodes=n)` reports
`environment_episodes = space_size × n` for exactly this decision, and
`max_episodes` bounds the resource directly and forfeits the certificate when it
bites — the same way `max_programs` does.

A second limit worth recording: the mode searches the discrete choice only.
`SearchResult.continuous` names the trainable constants held at their declared
values, and `frozen_selection` drops the trainable flag rather than optimizing
it. The readout-sign node above exists precisely because `examples/joint.py` put
that bit in a constant where no discrete backend could reach it. **Content in a
trainable constant is content this backend cannot search**, and re-expressing it
as a choice node is a scaffold decision, not something the backend can do for a
caller.

---

## 4. Prefix reuse and the beam

### Semantics, chosen and stated

A `Program` is validated to have no forward reference, so node i's value depends
only on the choices at nodes before it. Walking the choice tree depth-first and
computing each node once per **prefix** rather than once per **leaf** therefore
returns the identical conforming set. Rejecting a subtree is sound for the same
reason: conformance requires every probed node to match within tolerance, and a
probed node that already misses cannot be repaired by a later choice, so the
whole subtree is **decided**, not skipped.

That accounting is what preserves the contract. `evaluated` counts programs
decided — a rejection at node i decides every leaf under it — so
`evaluated == space_size` remains the exhaustion test and
`enumerate_prefix(beam=None)` is exhaustive and certifies uniqueness exactly as
`enumerate_fit` does.

`beam=k` is the part that gives something up. After each node the surviving
prefixes are ordered by the worst probe error decided so far and only the best k
are kept; the rest are **discarded without being decided**. `discarded` counts
them, `exhausted` is False, `unique` is None and `certificate` is `'none'`.
Nothing about the unexamined part of the space is claimed.

One deliberate refinement over "a beam always forfeits the certificate": if a
beam provably discarded nothing — the surviving front never exceeded the width —
the sweep *was* exhaustive and says so. This is strictly more honest than a
blanket forfeit and it can never over-claim, because `discarded == 0` is checked
rather than assumed. A test pins both directions.

### Acceptance A: the rung-3 foreground module, R=8

`research/discrete-backend/beam_search.py`, arm A. The discrete-perception
track's scaffold, byte pool and example sampler unchanged: 32,000 programs over
384 supervised pixels of a 192-byte observation, one probe, at the output.

| mode | evaluated | conforming | certificate | node evaluations | s |
|---|---|---|---|---|---|
| enumerate_fit | 32,000 | 2608 | complete | 159,744,000 | 490.8 |
| enumerate_prefix | 32,000 | 2608 | complete | 2,957,584 | 6.4 |

identical result: True; speedup 76.2x; viability {'space_size': 32000, 'executions': 12288000, 'projected_seconds': 614.4, 'environment_episodes': 0, 'exhaustible': True}

| beam | solved | conforming | same pick | discarded | space examined | certificate | s |
|---|---|---|---|---|---|---|---|
| 1 | False | 0 | False | 31,984 | 0.0005 | none | 0.2 |
| 4 | True | 4 | True | 31,940 | 0.0019 | none | 0.2 |
| 16 | True | 16 | True | 31,760 | 0.0075 | none | 0.4 |
| 64 | True | 64 | True | 31,072 | 0.0290 | none | 2.0 |
| 256 | True | 256 | True | 28,256 | 0.1170 | none | 3.1 |
| 1024 | True | 1024 | True | 16,208 | 0.4935 | none | 10.5 |

**What this says.**

* The prefix walk returns the **identical** result — same conforming count
  (2,608, inside FINDINGS section 14's reported 2,464-2,608 band), same returned
  program, same certificate, same `unique` — in 6.4 s against 490.8 s,
  **76.2×**, against the 17.8× the `incremental.py` prototype measured. The
  load-independent figure is the one to quote: **2,957,584 node evaluations
  against 159,744,000**, a 54× reduction that is identical across runs. Wall
  clock on this machine varied with load — an earlier run of the same script
  measured 373.0 s and 8.2 s, 45.2× — so the shipped JSON reports 76.2× and the
  honest range is 45-76×.
* The mechanism is precise and it bounds the technique. The rung-3 scaffold has
  eight fixed nodes above five choice nodes, and the fixed nodes are the
  expensive ones: three `index` reads into a 192-byte tuple per example. The flat
  sweep pays them 384 × 32,000 times; the walk pays them 384 × 1. So **the saving
  is the fixed work above the first choice**, and a scaffold whose choices sit at
  the top gets nothing. This is why the speedup is flat in observation width
  where the flat sweep is linear in it.
* The beam at width 1 **fails**: 0 conforming, on a task where 2,608 of 32,000
  programs (8.2 %) conform. Width 4 succeeds and returns the same program the
  exhaustive sweep does. The reason is arm B.
* At every width the beam reports `certificate: none` and a nonzero `discarded`,
  and the fraction of the space actually examined runs from 0.05 % at width 1 to
  49 % at width 1,024. At width 1,024 it examines half the space, certifies
  nothing, and takes **10.5 s against the exhaustive walk's 6.4 s** — it is
  strictly worse on both axes. The beam is not competitive with the prefix walk
  at any width on this task.

### Acceptance B: what a beam needs to be a search

Arm B holds the program and the data fixed and changes only the supervision: a
four-node chain of truth tables, 65,536 programs, probed at every node or probed
at the output alone.

| supervision | mode | conforming | certificate | node evaluations | s | reference conforms |
|---|---|---|---|---|---|---|
| dense | enumerate_fit | 4 | complete | 1,048,576 | 1.03 | True |
| dense | enumerate_prefix | 4 | complete | 162 | 0.00 | True |
| output_only | enumerate_fit | 9536 | complete | 1,048,576 | 1.20 | True |
| output_only | enumerate_prefix | 9536 | complete | 152,640 | 0.31 | True |

| supervision | beam | solved | conforming | recovers reference | discarded | space examined | certificate |
|---|---|---|---|---|---|---|---|
| dense | 1 | True | 1 | False | 17 | 0.9997 | none |
| dense | 4 | True | 4 | False | 0 | 1.0000 | complete |
| dense | 16 | True | 4 | False | 0 | 1.0000 | complete |
| dense | 64 | True | 4 | False | 0 | 1.0000 | complete |
| output_only | 1 | True | 1 | False | 65,523 | 0.0002 | none |
| output_only | 4 | True | 4 | False | 65,478 | 0.0009 | none |
| output_only | 16 | True | 16 | False | 65,304 | 0.0035 | none |
| output_only | 64 | True | 64 | False | 64,600 | 0.0143 | none |

**What this says, and it is the finding of this section.** A beam prunes by the
score of a partial prefix, and a partial prefix has a score only where a node
that has been decided is supervised.

* With **dense** probes hard rejection at each probed node keeps the live front
  under 4, so a beam of width 4 **discards nothing, stays exhaustive and keeps
  its `'complete'` certificate** — the refinement above, demonstrated. Width 1
  does discard (17 programs) and correctly drops to `'none'`. The prefix walk
  does 162 node evaluations against the flat sweep's 1,048,576 nominal, and finds
  all 4 conforming programs, the reference among them. A beam buys nothing here.
* With **output-only** supervision the same 65,536-program space has **9,536
  conforming programs** — the output alone does not identify the chain — no
  prefix has a score until the last node, every prefix ties, and the beam
  degenerates to "keep the first k in enumeration order". It then returns exactly
  k conforming programs at width k — 1, 4, 16, 64 — which is the signature of a
  budget rather than a search: it found what its width let it carry, and its
  width told it nothing about what to carry. Prefix reuse also collapses from
  6,473× to 6.9× fewer node evaluations, for the same reason: with no
  intermediate probe there is nothing to reject early.

That is why width 1 fails on arm A: rung 3 is an output-only scaffold, so at
width 1 the beam is following enumeration order, and enumeration order does not
happen to lead to a conforming program.

**The beam is a supervision-shaped tool.** Where dense intermediate probes exist
— the staging FINDINGS section 14 found is what makes rung 3.5 reachable at all
— the exhaustive prefix walk is already cheap and the beam is unnecessary. Where
they do not, the beam is available but is a budget, not a search. There is no
regime measured here in which the beam is the right answer; the honest
recommendation is the exhaustive prefix walk as the default fast path and the
beam as an explicit, certificate-forfeiting last resort above roughly 10⁸
programs, where nothing else is available.

### Where the prefix walk stops

`enumerate_prefix` saves the fixed work above the choice nodes and nothing below
them, so its speedup is a property of the scaffold, not of the space size. On the
full-alphabet arm of FINDINGS section 14 — 256 byte values per channel, 4.3 × 10⁹
programs — the two deepest choice nodes (`rg`, `foreground`, 16 candidates each)
sit under 256³ prefixes, so the walk still pays on the order of 10⁹ leaf
evaluations and remains out of reach. Prefix reuse moves the wall; it does not
remove it. Spaces above roughly 10⁹ stay unreachable exhaustively by any mode
here, which is what the brief already recorded and what these measurements
confirm rather than overturn.

---

## 5. Summary: what each mode gives and what it costs

| mode | acceptance task | space | programs decided | environment episodes | wall clock | exhausted | conforming | certificate |
|---|---|---|---|---|---|---|---|---|
| recurrent | depth interpreter, pinned wire lookup | 272 | 272 | 0 (16 supervised episodes harvested) | 0.61 s | yes | 1 | unique |
| recurrent | depth interpreter, wire choice free | 1,088 | 1,088 | 0 (16 supervised episodes harvested) | 2.40 s | yes | 1 | unique |
| environment | joint scaffold, staged: supervision then reward | 512 then 2 | 512 then 2 | **27** | 0.03 s | yes | 2 then 1 | complete then unique |
| environment | joint scaffold, unstaged, 16 episodes each | 512 | 512 | 8,192 | 9.0 s | yes | 8 | complete |
| prefix | rung 3, R=8 | 32,000 | 32,000 | 0 | 6.4 s | yes | 2,608 | complete |
| beam (w=1024) | rung 3, R=8 | 32,000 | 15,792 | 0 | 10.5 s | **no** | 1,024 | **none** |
| beam (w=4) | rung 3, R=8 | 32,000 | 60 | 0 | 0.2 s | **no** | 4 | **none** |
| beam (w=1) | rung 3, R=8 | 32,000 | 16 | 0 | 0.2 s | **no** | **0 — failed** | **none** |
| fit (baseline) | rung 3, R=8 | 32,000 | 32,000 | 0 | 490.8 s | yes | 2,608 | complete |
| prefix | chain, dense probes | 65,536 | 65,536 | 0 | 0.00 s | yes | 4 | complete |
| prefix | chain, output probe only | 65,536 | 65,536 | 0 | 0.31 s | yes | 9,536 | complete |
| fit (baseline) | depth interpreter, either arm | 272 / 1,088 | 272 / 1,088 | 0 | 0.11 / 0.30 s | yes | **0 — cannot see the recurrence** | complete |

Where each stops:

| mode | limit | number |
|---|---|---|
| recurrent | CPU, linear in `programs × examples × ticks`; no prefix reuse available | ≈1.6 × 10⁶ programs per CPU-hour at 16 examples × 12 ticks |
| environment | environment episodes, exactly `space_size × episodes_per_program` | viable only on a residual already reduced by supervision; 512 programs is days on a live-OS generator |
| environment | cannot search trainable constants at all | content in a constant must be re-expressed as a choice node by the scaffold author |
| prefix | saves only the fixed work above the first choice node, and early rejection only at probed nodes | 54× fewer node evaluations on rung 3 (45-76× wall clock), 6,473× on a densely probed chain, 6.9× on the same chain with output supervision only; ~10⁹ programs stays unreachable |
| beam | forfeits the certificate, and is informative only under dense intermediate supervision | at width 1 it missed a solution 8.2 % of the space satisfies |

## 6. Reproduction

    .venv/bin/python research/discrete-backend/recurrent_depth.py
    .venv/bin/python research/discrete-backend/environment_policy.py
    .venv/bin/python research/discrete-backend/beam_search.py
    .venv/bin/python research/discrete-backend/report.py     # regenerates every table above
    .venv/bin/python -m pytest tests -q                      # 200 passed

Every table in this file is printed by `report.py` from the JSON in `out/`.
Nothing is transcribed by hand.

**Reproduction checked.** `recurrent_depth.py` and `environment_policy.py` were
re-run against the shipped fixtures: every measured field is bit-identical —
conforming counts, certificates, `unique`, `evaluated`, episodes, steps, returns,
probe errors, selections and digests — and the only diff is the `seconds` fields,
which are wall clock. `beam_search.py` is the same apart from wall clock, which
is why section 4 quotes node evaluations as the load-independent measure and
gives the wall-clock speedup as a range.
