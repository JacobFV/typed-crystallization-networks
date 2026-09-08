# Executable system blueprint

The implementation fills these boxes with the same public contracts. The
constitutional architecture remains authoritative.

| Box | Owning implementation | Acceptance path |
|---|---|---|
| Types and representations | `tcn/types.py` | Round trips, invalid carriers, set invariance |
| Universal operator algebra | `tcn/operators.py` | Exact domains and relaxed gradients |
| Typed recurrent programs | `tcn/graph.py` | Type legality, recurrence, immutable exports |
| Differentiable synthesis | `tcn/learning.py` | Learned choices, wiring, probe objectives |
| Progressive crystallization | `tcn/crystallize.py` | Trial, residual retraining, rollback, no-grad reuse |
| Generator host and records | `tcn/generation.py` | Visibility, replay, composition, schema checks |
| All synthetic domains | `generators/*/` | Same host lifecycle for every manifest |
| Curriculum orchestration | `tcn/curriculum.py` | DAG scheduling, resume, prerequisite gates |
| Joint predictive control | `tcn/training.py` | Prediction and policy gradients in one cycle |
| Typed action distributions and frozen agents | `tcn/policy.py`, `tcn/agent.py` | Continuous/discrete arguments, closed-loop exact control |
| Bit-parallel execution | `tcn/packed.py` | All truth tables versus exact execution, recurrent state |
| Export and measurements | `tcn/runtime.py` | Standalone exact execution and latency |
| User entry points | `tcn/cli.py`, `examples/` | Reproducible commands and saved evidence |

A completed software box is distinct from a trained general capability. Tests
and run manifests record which behavior has actually been demonstrated.
