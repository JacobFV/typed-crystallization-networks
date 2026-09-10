# Semantic pooling — the pre-registered replication §46 asked for

FINDINGS §46 recorded, **outside its own pre-registration**, that pooling mined
fragments by `(arity, truth table)` instead of by `Program.digest` moves `MAJ3`
to rank 1 and lifts §44's later task from 0 conforming to the hand-authored
ceiling of 144. It declined to call that a capability, requiring "a
pre-registered replication with its own wrong-module control".

This is that replication. It adds the control §46 lacked — a *semantically
pooled* wrong module, in two forms — and the test §46 never ran: whether the
induced library helps a task the rule **did not mine from**.

Read [`RESULTS.md`](RESULTS.md). [`PREREGISTRATION.md`](PREREGISTRATION.md) was
committed before any arm was run.

Nothing in `tcn/` or `generators/` is modified. §44's later task, both
scaffolds, the gradient loop and the hand-authored modules are **imported** from
`research/earned-abstraction`; §46's corpus bands, `mine_multi` and
`mine_semantic` are **imported unchanged** from `research/premin-abstraction`.

## Files

| file | what it is |
|---|---|
| `_paths.py` | puts the two upstream tracks on the import path; no PYTHONPATH needed |
| `family.py` | the two families (majority, and the off-family `F'` built on the §44 distractor) and the leave-one-out corpora |
| `evaltasks.py` | the compact scaffold and the held-out / control tasks |
| `poolmine.py` | drives both identity relations and resolves the pre-registered selection rules |
| `armlib.py` | the arms; the only thing that differs between them is the library |
| `run_mine.py` | build `F'`, mine every corpus, publish to a real `tcn.library.Library` |
| `run_enum.py`, `run_enum_all.py` | exhaustive enumeration of L1 with certificates |
| `run_grad.py` | L1 gradient runs, 24 seeds tight / 8 wide |
| `run_heldout.py` | the held-out protocol: enumeration and gradient on the compact scaffold |
| `run_sweep.py` | every eligible arity-3 pooled class, enumerated on L1 |
| `run_rest.py`, `run_rest2.py` | drivers that chain the stages after the corpora exist |
| `rerun_hpar.py` | re-runs only the `H_par` rows after the chained-comparison defect (see RESULTS §9) |
| `run_checks.py` | falsification criterion F3 (frequency artifact) and the cost accounting |
| `run_sensitivity.py` | the rule's three parameters swept once |
| `tables.py` | regenerates every table in `RESULTS.md` from `out/` |

Run with the repo `.venv`, from the repository root:

```
.venv/bin/python research/semantic-library/run_mine.py off
.venv/bin/python research/semantic-library/run_mine.py mine
.venv/bin/python research/semantic-library/run_mine.py publish
.venv/bin/python research/semantic-library/run_enum_all.py <arms...>
.venv/bin/python research/semantic-library/run_grad.py tight 24
.venv/bin/python research/semantic-library/run_grad.py wide 8
.venv/bin/python research/semantic-library/run_heldout.py
.venv/bin/python research/semantic-library/run_sweep.py maj_minall_sem
.venv/bin/python research/semantic-library/run_checks.py
.venv/bin/python research/semantic-library/run_sensitivity.py
.venv/bin/python research/semantic-library/tables.py
```
