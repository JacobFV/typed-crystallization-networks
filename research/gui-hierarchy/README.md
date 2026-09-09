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
| `rung1_edges.py` -> `out/rung1_<arm>.json`, `out/rung1_<arm>.log` | Rung one: three arms over the same target, with the hand-initialisation ablations, plus stage B and stage C. Run one arm per process with `--arms`. |
| `tiebreak.py` -> `out/tiebreak.json` | Whether `enumerate_fit`'s lexicographic pick is safe, per dial setting, and whether the validation filter repairs it. |
| `surrogate.py` -> `out/surrogate.json` | `eq`'s surrogate value and slope against the palette, the distances actually compared, `index`'s relaxed blur, and the measured gradient at every choice node. |
| `report.py` | Renders every table in `RESULTS.md` from `out/*.json`. |

Reproduce:

```sh
.venv/bin/python research/gui-hierarchy/bounds.py
.venv/bin/python research/gui-hierarchy/dial.py
for a in narrow wide free; do
  .venv/bin/python research/gui-hierarchy/rung1_edges.py --arms $a --tag rung1_$a &
done; wait
.venv/bin/python research/gui-hierarchy/surrogate.py
.venv/bin/python research/gui-hierarchy/tiebreak.py
.venv/bin/python research/gui-hierarchy/report.py render
```

Generator tests: `.venv/bin/python -m pytest tests/test_gui_generator.py`.
