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
