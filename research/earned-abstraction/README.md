# Earned abstraction — which subprogram is worth crystallizing?

The library, the publish/inherit flow and recursive abstraction all work and are
measured (FINDINGS §12, §25, §30). In every existing demonstration a **human**
chose which subprogram becomes the module. This track builds a selection rule
that makes that choice from the typed program graph alone, and measures whether
inheriting what it proposes pays.

Read [`RESULTS.md`](RESULTS.md). [`PREREGISTRATION.md`](PREREGISTRATION.md) was
written before the arms were run.

Nothing in `tcn/` or `generators/` is modified.

## Files

| file | what it is |
|---|---|
| `corpus.py` | the earlier-task family and the shipped gradient solver |
| `minimal.py` | exact straight-line synthesis in the scaffold basis, with a minimality certificate |
| `mine.py` | **the rule** — fragment enumeration, canonical identity, rewriting, MDL score |
| `later.py` | the later task and its two scaffolds (the retest's, rebuilt for arbitrary module arity) |
| `arms.py` | the five arms; the only thing that differs between them is the library |
| `run_corpus.py` | solve the earlier tasks with the shipped gradient path (it mostly fails) |
| `probe_gradient.py` | the same, with a much larger budget, to show the failure is not budget |
| `run_corpus_exact.py` | solve them exactly instead, with certificates |
| `run_mine.py` | run the rule, write the ranked table, publish to a real `tcn.library.Library` |
| `run_sensitivity.py` | the rule's three parameters swept |
| `run_order_robustness.py` | how much the corpus depends on the solver's tie-break |
| `run_enumerate.py` | exhaustive enumeration of one arm's tight scaffold via `tcn.search.enumerate_fit` |
| `run_gradient.py` | sample complexity: the same task learned, per arm, across seeds |
| `run_economy.py` | verified minimum-node route for the later task under each library |
| `test_mine.py` | machinery checks for the rule, on hand-built programs |
| `library/` | the published library the later task inherits from |
| `out/` | every artifact every script wrote |

Reproduce from this directory with
`PYTHONPATH=<repo>:. <repo>/.venv/bin/python <script>`.
