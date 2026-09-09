# Research agenda

Started 2026-09-08. Each track answers one falsifiable question with measured
numbers. Negative results are results and stay in the record; see the standing
rule in `docs/VALIDATION.md` that separates tested mechanics from demonstrated
learning from unestablished capability.

## Standing observations that motivate the tracks

`examples/joint.py` is the only successful joint experiment. In it:

- Exactly two nodes carry real operator choice (16 truth tables each). Every
  other node is single-candidate, so wiring is supplied, not learned.
- The value function is a trainable constant, not a state-dependent estimate.
- With `{'depth':1,'table':6,'fixed_inputs':True}` every episode instantiates
  the *same* Boolean function. Held-out "episodes" vary the four input bits and
  the objective bit, not the structure. Generalization across gate families is
  therefore untested.

`tcn/scaffold.py:arithmetic_scaffold` is the scaffold that stayed near chance
(`artifacts/joint-long/report.json`: 2.03/4 after 640 episodes). It has almost
no operator choice — only `sin` vs `identity` at activations — so it is a small
sin-activated network with typed plumbing. A network of that capacity failing a
depth-1 truth table is more likely a training-dynamics fault than evidence about
crystallization.

## Tracks

| # | Question | Directory |
|---|---|---|
| 1 | Does progressive crystallization beat naive argmax rounding? | `research/crystallization-ablation/` |
| 2 | Why did the arithmetic scaffold stay at chance? | `research/scaffold-autopsy/` |
| 3 | Where does candidate search break as depth and pool size grow? | `research/search-scaling/` |
| 4 | Does the joint agent generalize to unseen gate families? | `research/structure-generalization/` |
| 5 | Does crystallized-module reuse measurably help? | `research/recursive-abstraction/` |
| 6 | How does TCN compare to a matched-information baseline? | `research/baselines/` |
| 7 | What does the literature actually establish about DLGNs? | `research/literature/` |

Track 1 is the load-bearing one. Progressive hardening with residual retraining
and transactional rollback is the architecture's central mechanism. If argmax
rounding at the end of training matches it, the mechanism is unnecessary.

## Track 5 result (2026-09-08)

Answered in `research/recursive-abstraction/RESULTS.md`. **No measurable
benefit.** Two matched experiments (half adder -> full adder; MAJ3 -> sliding
majority), 8-12 seeds per arm, one shared scaffold whose only difference is
whether the crystallized module is offered as a candidate:

- E1 3/12 flat vs 2/12 with module (Fisher p = 1.00); E2 5/8 vs 7/8 (p = 0.57).
- In 0 of 20 arm-B runs was the module on the output path of the discovered
  program, including 0 of 10 successes.
- The sec. 4 accounting claim is correctly implemented (definition once, call
  sites each, cost per use, transitive) but the abstracted program is 1.4x
  larger, 1.4-1.6x costlier and 1.7x slower at batch one; description-size
  crossover needs 4 call sites and execution cost never crosses over.
- Proposed core changes in the report: F1 collapse unit-arity module output
  types, F2 memoize `Program.validate` (module candidates are 16-98x costlier
  than primitives), F3 wire the MDL term into `synthesis.fit`, F4 prune dead
  nodes before export/registration.
