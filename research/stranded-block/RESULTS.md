# The viability guard is incomplete over sets

**Status: hazard confirmed on `main`; fix implemented, measured, and it does not
raise the completion rate.** Read the verdict before using this branch.

`research/FINDINGS.md` §37 reported, as a side finding of the refuted seasons
scheduler, that "releasing late strands a node set that the per-node
connectivity guard can never close ... That structure exists on main today,
independent of seasons." This track isolates that claim from the seasons
machinery, tests it, and fixes what can be fixed.

## 1. The hazard reproduces on `main`, with no seasons machinery

`reproduce_on_main.py` runs the shipped `Crystallizer` on the shipped
`examples/joint` fixture with exactly the settings `tcn.cli.joint` uses: 40
training episodes, `rounds=24`, `retrain_steps=2`, `tolerance=0.05`. Eight seeds.

```bash
.venv/bin/python research/stranded-block/reproduce_on_main.py 8
```

| seed | frozen | fully frozen | "disconnected remaining region" refusals |
|---|---|---|---|
| 0 | 13/13 | yes | 3 |
| 1 | 13/13 | yes | 3 |
| 2 | 13/13 | yes | 0 |
| 3 | 13/13 | yes | 1 |
| 4 | 13/13 | yes | 3 |
| 5 | 13/13 | yes | 3 |
| 6 | 13/13 | yes | 6 |
| **7** | **9/13** | **no** | **29** |

Raw records: `results/reproduce-main-8seeds.jsonl`.

Seed 7 ends with `{goal_relation, prediction, relation, z}` still soft. Every
remaining round proposes a member of that set and the viability guard refuses it
for the same reason: freezing that member severs the others from the task
objective. The guard is **correct per node and incomplete over sets** — it can
report a state that no single-node trial can leave.

**It is not a rounds shortage.** Doubling the budget does not help:

| seed 7 | rounds | frozen | node trials | disconnected refusals |
|---|---|---|---|---|
| shipped | 24 | 9/13 | 124 | 29 |
| shipped | 48 | 9/13 | 220 | **77** |

The extra 24 rounds bought 96 more refused trials and zero progress. The residual
is stranded, not merely short of budget.

## 2. The fix, and the A/B that has to accompany it

`try_freeze_block` hardens a residual set in one transactional trial. It carries
the same degradation tolerance, the same conformance check, and the same
all-or-nothing rollback as a single-node freeze, so it is a completeness fix for
the guard and not a bypass of the freeze contract. It is not a weakening of the
guard either: when the block completes the program there is no remaining
trainable region left to disconnect, which is precisely the condition the guard
tests for.

It fires only on evidence of the pathology — the final round must have refused a
freeze for "disconnected remaining region". A run that merely ran out of rounds,
or whose eligibility gate deliberately deferred every commitment, produces no
such refusal and gets no trial. Two pre-existing gate tests caught an earlier
version that fired unconditionally; that is what narrowed the trigger.

**Arm A is `close_block=False`, which is `main` exactly. Arm B is the fix. Same
seeds, same episodes, same rounds.**

```bash
.venv/bin/python research/stranded-block/run_ab.py > results/ab.jsonl
```

| seed | A frozen | B frozen | committed selections identical | block trial |
|---|---|---|---|---|
| 0 | 13/13 | 13/13 | **yes** | not fired |
| 1 | 13/13 | 13/13 | **yes** | not fired |
| 2 | 13/13 | 13/13 | **yes** | not fired |
| 3 | 13/13 | 13/13 | **yes** | not fired |
| 4 | 13/13 | 13/13 | **yes** | not fired |
| 5 | 13/13 | 13/13 | **yes** | not fired |
| 6 | 13/13 | 13/13 | **yes** | not fired |
| 7 | 9/13 | 9/13 | **yes** | **fired, refused** |

Identical-output assertion: the full `selections()` map of the committed program
is bit-identical between arms on all eight seeds, and the node-trial count on
seed 7 is 124 in both arms. Where the guard was already sufficient the fix is
inert, which is the property it had to have.

## 3. The verdict, which is negative

**The block trial does not rescue seed 7.** It fires, and it is refused:

```json
{"node": "relation+goal_relation+z+prediction", "accepted": false,
 "before": 1.4636, "after": 2.4682, "reason": "block degradation"}
```

Committing the stranded set together raises the objective from 1.4636 to 2.4682
against a tolerance of 0.05, so the trial rolls back and the run ends 9/13, the
same as `main`. **The guard's incompleteness was not the only thing standing in
the way of that run.** The residual set is both unfreezable one node at a time
*and* genuinely damaging to commit jointly at the point the schedule reaches it.

Loosening the tolerance would convert this into a pass and would be dishonest;
the tolerance is the contract that makes a freeze mean something.

So the measured value of this change is narrower than §37 implied:

- **What it buys.** A silent non-terminating strand becomes a recorded,
  correctly-refused `block degradation` event with its before/after loss. The
  failure is now diagnosable from the event log instead of appearing as an
  unexplained partial freeze.
- **What it does not buy.** Any improvement in completion rate on this fixture.
  1 of 8 seeds fails before the change and 1 of 8 fails after it.

§37's "7 of 8 with a block trial" was measured **under seasons**, where the set
had been thawed and rewarmed before the trial. That is consistent with what is
measured here and does not transfer to `main`: the block trial is necessary for
closing a strand but not sufficient, and on `main` there is no thaw to make the
joint commitment cheap.

## 4. What is still open

The real defect on seed 7 is upstream of the guard: the schedule reaches a state
where `{goal_relation, prediction, relation, z}` cannot be committed at all —
neither singly nor jointly — at a loss the tolerance will accept. Diagnosing
*why* the schedule arrives there is a separate question and is not answered here.
Given that four independent tracks now agree the progressive scheduler should not
be used at all (FINDINGS §7, §12, §37), the cheaper answer may be that this
failure mode is only reachable from a schedule nobody should run.

## 5. Verification record

- 295 tests pass, including 7 new in `tests/test_stranded_block.py` covering
  whole-block rollback, all-at-once commitment, tolerance, conformance,
  inertness on a run that already closes, the off switch, and the deferred-run
  trigger.
- The shipped fixture reproduces exactly with the fix enabled:
  `python -m tcn train --episodes 160` gives 0.24884 → 0.00223, fully frozen,
  4.0 evaluation return, 4.0 from the exact frozen agent, and **zero** block
  events — the fix never fires on the shipped path.
- Both tables above were produced by the scripts in this directory against the
  final code, not against an intermediate version.
