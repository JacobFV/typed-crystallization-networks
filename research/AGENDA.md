# Research agenda

Started 2026-09-08. Each track answers one falsifiable question with measured
numbers. Negative results are results and stay in the record; see the standing
rule in `docs/VALIDATION.md` that separates tested mechanics from demonstrated
learning from unestablished capability.

## CLOSED — do not re-dispatch these

Two long-standing items were verified on 2026-09-09 to be **already implemented**,
after a brief repeated one of them verbatim and an agent had to correct it. A
priority item that a merged change already closed will otherwise be re-dispatched
indefinitely.

- **"Wire the MDL term into `synthesis.fit`, which has no cost term" (track 5
  F3).** `fit` has accepted `mdl_weight` since §14 merged, at
  `tcn/synthesis.py:32,131`, scaling ARCHITECTURE §8's `L_program_description`.
  Beyond being done, §41 then measured it to be **useless on the artifacts we
  have**: the description-minimal program is the bytecode-maximal one, and
  `execution_cost` is constant across a family that differs by 48 bytecodes.
- **"Prune dead nodes before export and module registration" (track 5 F4, the
  32% overstatement).** Both sites prune today: `tcn/runtime.py:17` on
  save/export and `tcn/operators.py:33` on `register_module`, each with the
  reason in a comment.

Also closed by measurement rather than by implementation: **loss-gated
eligibility** (§7, §12, §37 — it is one of the four independent refutations of
the progressive scheduler, not an open experiment).

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

| # | Question | Directory | Verdict |
|---|---|---|---|
| 1 | Does progressive crystallization beat naive argmax rounding? | `crystallization-ablation/` | **No.** Inert at shipped budgets, harmful at tight ones. |
| 2 | Why did the arithmetic scaffold stay at chance? | `scaffold-autopsy/` | **Stopped too early.** Transition at ~750-2600 episodes; 4.00/4 at 5120. |
| 3 | Where does candidate search break as depth and pool size grow? | `search-scaling/` | **Target arity, not depth.** Free wiring <50% at depth 3; dense probes fix it (88-94%). |
| 4 | Does the joint agent generalize to unseen gate families? | `structure-generalization/` | **Not as recorded** (2.09/4 = chance). Fixed by an interpreter candidate: 4.00 frozen, unseen tables. |
| 5 | Does crystallized-module reuse measurably help? | `recursive-abstraction/` | **No.** Module on the output path in 0 of 20 runs. |
| 6 | How does TCN compare to a matched-information baseline? | `baselines/` | **Exactness holds** (1e-8 vs 0.232). Size and latency claims do not. |
| 7 | What does the literature actually establish about DLGNs? | `literature/` | Selection rule falsified by DARTS-PT; depth known not to pay; rollback unclaimed. |
| 8 | Does differentiable search beat enumeration/SAT on the same space? | `enumerative-baseline/` | **No.** Brute force settles the flagship result in 0.081 ms. Gradient wins only on env samples. |

All eight tracks are complete. **The synthesis is in
[`research/FINDINGS.md`](FINDINGS.md)** — what survived measurement, what did
not, the instrumentation faults, the proposed core changes ordered by payoff,
and the recommended next steps. Read that first; the per-track `RESULTS.md`
files hold the detail and raw data.

Track 1 was the load-bearing one and it came back negative, so the headline is
not the one the agenda anticipated: across tracks 1, 3 and 6, progressive
crystallization does not earn its complexity and hierarchical supervision does.

`tcn/` and `generators/` are unchanged from the initial commit. Every proposed
core change is recorded as a diff in a report rather than applied, pending
review.
