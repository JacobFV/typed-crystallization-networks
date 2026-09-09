# gui-hierarchy

The foundation for hierarchical screen parsing: a `gui` generator that renders a
synthetic widget tree to raw pixels and emits the true component hierarchy as a
probe, the recoverability bounds for the rungs above it, and rung one -- widget
edges -- learned.

Everything here was produced with the repository `.venv`. **Nothing under `tcn/`
is modified**, and no existing generator is modified; `generators/gui/` is a new
peer under the same contract.

| file | what it does |
|---|---|
| `common.py` | Episode/probe access, hand-wiring helpers, the discrete and gradient references. Names the three things copied from `research/discrete-perception`. |
| `bounds.py` -> `out/bounds.json` | The recoverability bound for each intended rung, computed **before** any search. |
| `dial.py` -> `out/dial.json` | Whether each declared difficulty dial moves structure, appearance, the ceiling, and the search. |
| `rung1_edges.py` -> `out/rung1.json`, `out/rung1.log` | Rung one: three arms over the same target, with the hand-initialisation ablations. |
| `report.py` | Renders every table in `RESULTS.md` from `out/*.json`. |

Reproduce:

```sh
.venv/bin/python research/gui-hierarchy/bounds.py
.venv/bin/python research/gui-hierarchy/dial.py
.venv/bin/python research/gui-hierarchy/rung1_edges.py
.venv/bin/python research/gui-hierarchy/report.py
```

Generator tests: `.venv/bin/python -m pytest tests/test_gui_generator.py`.
