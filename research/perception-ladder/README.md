# research/perception-ladder

Track: the first attempt at a non-toy learned program in this system. Read
[RESULTS.md](RESULTS.md).

| file | what it is |
|---|---|
| `common.py` | dataset helpers, `local_fit` (a verified copy of `tcn.synthesis.fit` with seed variation), enumeration/random references, accuracy |
| `check_local_fit.py` | proves `local_fit(init_noise=0)` is bit-identical to `tcn.synthesis.fit` |
| `audit.py` | expressibility and information audits (image operators, surrogate ranges, relaxation bugs, generator statistics) |
| `pointing.py` | does the initial gradient point at the reference program? the mechanism measurement |
| `excursion.py` | rejected hypothesis: do relaxed mixtures leave the achievable range? (they do not) |
| `rung1_signal.py`, `run_rung1.py` | rung 1: frequency and `future` from raw samples |
| `rung2_relations.py`, `run_rung2.py` | rung 2: the set-operator gradient boundary and the reachable `target` task |
| `rung3_geometry.py`, `run_rung3.py` | rung 3: background segmentation from pixels, width sweep, dense vs output-only |
| `rung4_raster.py`, `run_rung4.py` | rung 4: pixels to symbol, `eq` route vs `pack` route |
| `tables.py` | renders every table in RESULTS.md from `out/*.json` |
| `RESULTS.template.md`, `render.py` | RESULTS.md is generated: edit the template, then `python render.py` |
| `weights.py`, `phase_check.py` | two small checks cited in RESULTS.md |
| `out/` | raw JSON and run logs |

Reproduce (from the repository root, with the repo `.venv`):

```
PYTHONPATH=. .venv/bin/python research/perception-ladder/check_local_fit.py
PYTHONPATH=. .venv/bin/python research/perception-ladder/audit.py
PYTHONPATH=. .venv/bin/python research/perception-ladder/pointing.py
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung1.py ladder --seeds 16
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung1.py gap
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung1.py future --seeds 12
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung2.py wall
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung2.py gradients
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung2.py arms --seeds 8 --ntrain 44
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung3.py surrogate
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung3.py width --seeds 12
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung3.py colour --seeds 12
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung3.py dense --seeds 12
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung3.py dense --seeds 12 --free-address 0 --pool 24,30,43,15 --tag pinned
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung4.py eq --seeds 8
PYTHONPATH=. .venv/bin/python research/perception-ladder/run_rung4.py threshold --seeds 8
PYTHONPATH=. .venv/bin/python research/perception-ladder/tables.py
```
