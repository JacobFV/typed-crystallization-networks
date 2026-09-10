# `research/reuse-ranking` — a ranking objective that selects a reusable abstraction

FINDINGS §52 left one gap: semantic pooling by `(arity, truth table)` makes the
reusable abstraction *visible* — it is in the pooled table of every leave-one-out
corpus, bit-identical — but the MDL score does not *select* it, so the published
module helps 2 of 5 held-out tasks where an oracle selecting the same pooled
class helps 5 of 5.

This track compares seven ranking objectives over that same pooled set and
measures each one out of sample.

* `PREREGISTRATION.md` — committed at `69da5f8`, **before any arm**.
* `RESULTS.md` — the verdict, the objective × leave-one-out table with
  certificates, the frequency-count baselines, and the §41 cross-check.

## What runs what

| script | writes | what it does |
|---|---|---|
| `pool.py` | — | one **instrumented** pooling pass: `mine_semantic`'s rule with the per-entry saving kept. `check()` is integrity check IC1. |
| `objectives.py` | — | the seven scores and the shared tie-break. |
| `context.py` | — | O4's in-corpus cross-validation folds; O5's measured cost, via `research/lazy-guard/cost.py` (§51's instrument). |
| `run_rank.py` | `out/ranked.json` | every objective on every corpus; publishes each rank-1 class to `library/`. |
| `run_heldout.py` | `out/heldout.json` | the decisive out-of-sample enumerations, plus floor, ceiling and the two negative controls. |
| `run_grad.py` | `out/heldout_gradient.json` | `later.gradient_search`, 24 seeds, beside the enumeration. |
| `run_s41.py` | `out/s41_crosscheck.json` | reads §41's `language_family.json`; re-enumerates nothing. |
| `run_sensitivity.py` | `out/sensitivity.json` | the breadth exponent, and §52 §8's parameter grid. |
| `run_offfamily.py` | `out/offfamily.json` | **exploratory** (Amendment 1): the same protocol on the second family `F′`. |
| `show_folds.py` | `out/o4_folds.json` | O4's per-fold decomposition. |
| `summarize.py` | — | regenerates every table in `RESULTS.md` from `out/*.json`. |

Nothing under `tcn/`, `generators/`, `research/semantic-library/`,
`research/premin-abstraction/`, `research/earned-abstraction/` or
`research/lazy-guard/` is modified. The rule under test is imported, not
restated.

Run with the repository `.venv`, from this directory:

```
../../.venv/bin/python run_rank.py && ../../.venv/bin/python run_heldout.py
../../.venv/bin/python summarize.py
```
