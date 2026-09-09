# module-library — a persistent library, artifact flow, and a three-stage chain

The gap this track addresses: the composition machinery works and nothing wires
it together. No learned artifact outlives the script that produced it, no
registry persists, and the shipped curriculum's stages pass nothing to each
other.

## What is here

| file | what it is |
|---|---|
| `chain.py` | the three stages' scaffolds and data; stages 1 and 2 import `research/discrete-perception` rather than reimplementing it |
| `runner.py` | the curriculum runner: each stage searches with and without the prior stage's module, publishes, and reports both arms |
| `chain.json` | the curriculum — `requires` for ordering, `inherits` for artifacts, `publishes` for what a stage owes the next one |
| `run_chain.py` | runs `chain.json` through the shipped `tcn.curriculum.Curriculum` |
| `identify.py` | how much validation it takes to identify stage 1's module out of its conforming set |
| `accounting.py` | description size, execution cost and the abstraction crossover, composed against flat |
| `conformers.py` | every conforming stage-3 program, so "the module was selected" is enumerated rather than inferred |
| `library/` | the shipped library the chain produces: manifest, modules, conformance fixtures |
| `out/` | raw results |
| `RESULTS.md` | the report |

## Reproducing

```sh
.venv/bin/python research/module-library/run_chain.py --fresh   # builds library/ and out/chain
.venv/bin/python research/module-library/identify.py            # caches out/conforming_r8.json
.venv/bin/python research/module-library/accounting.py          # out/accounting.json
.venv/bin/tcn library list   --root research/module-library/library
.venv/bin/tcn library verify --root research/module-library/library
```

Core changes are in `tcn/library.py` (new), `tcn/curriculum.py` and `tcn/cli.py`;
`tests/test_module_library.py` covers them.
