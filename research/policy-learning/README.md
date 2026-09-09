# research/policy-learning — files

| file | what it does |
|---|---|
| `pl.py` | the harness: the `examples/joint.py` scaffold rebuilt with the hand-supplied policy constants, the value head, the discrete choices and the readout all made configurable; a re-implementation of `JointTrainer.episode` with every loss weight, the baseline, the batch size, the entropy schedule and the reward shaping exposed; explicit environment-episode counting; gradient SNR/direction instrumentation; `freeze_choices` (the crystallization step used by the staged arms); `mpc_rollout` (enumerating action sequences against a frozen exact program). |
| `parity.py` | checks the harness against the shipped fixture before anything else is trusted |
| `probe_env.py` | establishes the task's actual structure (contexts, reward timing, state constancy) |
| `space.py` | discrete reference: enumerates the 256-program choice space and counts probe-optimal and reward-optimal programs |
| `cancellation.py` | numerical check that a uniform 16-way truth-table mixture has an exactly zero Jacobian |
| `run_arms.py` | generic multi-seed arm runner; writes raw JSON, one record per (arm, seed) |
| `e1_reproduce.py` | E1/E2: the clean reproduction and the per-cause ablations |
| `e2_direction.py` | where the reward-optimal candidates rank in each term's descent direction |
| `e3_remedies.py` | E3: the remedy table |
| `e4_staged.py`, `e4b_handoff_budget.py` | E4: staged world model then policy, and how little reward the handoff needs |
| `e5_horizon.py` | E5: horizon 4/8/16/32, dense and terminal-only |
| `e6_mpc.py`, `e6b_plan1.py` | E6: model-predictive control against the crystallized program, no learned policy; and whether enumerating deeper action sequences buys anything |
| `e7_budget.py` | E7: return against environment-episode budget |
| `e8_mlp.py` | E8: a tuning grid for the neural baseline, so "the MLP never learns it" is not under-tuning |
| `refs.py` | trivial references (always-false, always-true, uniform, oracle) at every horizon |
| `aggregate.py` | turns raw JSON into the tables in `RESULTS.md` |
| `out/*.json`, `out/*.log` | raw results and logs |

Reproduce with `PYTHONPATH=.:research/policy-learning .venv/bin/python research/policy-learning/<script>.py`
from the repository root.
