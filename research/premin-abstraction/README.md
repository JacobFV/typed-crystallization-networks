# Pre-minimisation abstraction — does a richer corpus change what gets mined?

FINDINGS §44 (`research/earned-abstraction`) closed the abstraction loop
mechanically and measured that it does not pay, and located the cause: **exact
minimisation is adversarial to abstraction mining.** Each earlier task's
minimum-gate program factors differently, so `MAJ3` is fused into its wrapper
and the frequency statistics the rule reads never see it.

This track tests the follow-on hypothesis: *if the corpus is not minimised — or
not only minimised — the reusable abstraction survives in enough programs for a
frequency rule to find it.*

Short answer: the abstraction does survive — retention goes from 1 of 6 programs
to 134 of 296 — and the rank-1 proposal does not change by one bit. What
changes it is the *identity relation*, and only in combination with the richer
corpus (§10 of `RESULTS.md`, exploratory).

Read [`RESULTS.md`](RESULTS.md). [`PREREGISTRATION.md`](PREREGISTRATION.md) was
written and committed before any arm was run.

Nothing in `tcn/` or `generators/` is modified. The later task, both scaffolds,
the arms and the mining rule's R1/R2/rewriting are **imported** from
`research/earned-abstraction`, not restated, so every number is directly
comparable to §44's.

## Files

| file | what it is |
|---|---|
| `enumerate_programs.py` | exhaustive and sampled enumeration of *pruned* straight-line programs, at any length |
| `corpora.py` | the five corpus variants, the cap and the deterministic subsample |
| `mine_multi.py` | §44's rule over a corpus that may hold several programs per task (two declared changes) |
| `retention.py` | value retention over **all ordered 3-subsets** of the inputs |
| `arms_premin.py` | the arms; the only thing that differs between them is the library |
| `run_corpora.py` | build and verify every corpus, and measure retention |
| `run_mine_premin.py` | run the rule per corpus, write the ranked tables, publish to a real `tcn.library.Library` |
| `run_structural_retention.py` | value retention vs *structural* retention, and how the majority's mass splits across canonical digests |
| `mine_semantic.py`, `run_semantic.py` | **exploratory, not pre-registered**: the same rule with occurrences pooled by (arity, truth table) instead of by `Program.digest` |
| `check_digest_split.py` | whether two mined majority digests are the same circuit in two topological orders |
| `run_sensitivity_premin.py` | the rule's three parameters swept |
| `run_enum_premin.py`, `run_enum_all.py` | exhaustive enumeration of each arm's tight scaffold |
| `run_grad_premin.py` | the same task learned instead, per arm, across seeds |
| `show_*.py`, `tables_premin.py` | render every reported table from `out/` |
| `library/` | the published library the later task inherits from |
| `out/` | every artifact every script wrote |

Reproduce from this directory with

```
PYTHONPATH=<repo>:.:../earned-abstraction <repo>/.venv/bin/python <script>
```
