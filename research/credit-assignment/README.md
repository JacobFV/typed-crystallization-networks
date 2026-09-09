# research/credit-assignment

The task this project needed and did not have: **delayed credit assignment, reachable
by exploration, without dense probes**. `RESULTS.md` is the report.

Everything ran in the repository `.venv` against `tcn/` **unmodified**. The only
non-research change is a gated `interface='panel'` configuration of
`generators/computer/`, plus `generators/computer/engine/session.ts`, a long-lived
sibling of `bridge.ts`; both are verified byte-identical to the pre-change behaviour
on every default configuration (`equivalence.py`, `session_equivalence.py`).

| file | what it is |
|---|---|
| `panel.py` | the task's action library, reference policies, and the environment-episode counter |
| `analysis.py` | the exact belief-state DP: optimal vs myopic value at each horizon, the discount threshold, the exploration probability, the bandit-decomposition detector |
| `program.py` | the typed scaffold: perception, the argument encoders, the situation-conditioned policy |
| `perception.py` | perception by exhaustive enumeration against the probe channel |
| `run.py` | the training harness (a re-implementation of `JointTrainer.episode` with configurable weights and discount) |
| `arms.py` | one learning arm, one seed |
| `run_all.sh` | every arm, in waves |
| `mpc.py` | the model-based arm: enumerate action sequences against a frozen exact model, no policy |
| `macro.py` | composite actions as crystallized modules, and the test that they are not an oracle menu |
| `refs.py` | every baseline, measured on the real generator |
| `equivalence.py` | the default observation set and sampling stream of `generators/computer` are unchanged |
| `session_equivalence.py` | `session.ts` and `bridge.ts` produce identical episodes |
| `aggregate.py` | regenerates every table in `RESULTS.md` from `out/*.json` |
| `generator_before.py` | the pre-change copy of `generators/computer/generator.py`, for `equivalence.py` |

Reproduce, in order:

```sh
.venv/bin/python research/credit-assignment/analysis.py
.venv/bin/python research/credit-assignment/refs.py 64
.venv/bin/python research/credit-assignment/perception.py 25
.venv/bin/python research/credit-assignment/mpc.py 25 64
bash research/credit-assignment/run_all.sh
.venv/bin/python research/credit-assignment/macro.py 400
for i in 0 1 2 3 4 5; do .venv/bin/python research/credit-assignment/equivalence.py $i 6 & done; wait
.venv/bin/python research/credit-assignment/equivalence.py merge 6
.venv/bin/python research/credit-assignment/session_equivalence.py
.venv/bin/python research/credit-assignment/aggregate.py
```
