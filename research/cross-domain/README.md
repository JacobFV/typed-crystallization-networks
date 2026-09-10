# `research/cross-domain`

**Question.** Does semantic abstraction find anything shared across the visual,
language and computer artifacts — or only within synthetic Boolean families?

**Answer.** Only within Boolean families. See `RESULTS.md`.
`PREREGISTRATION.md` was committed before any arm (`5d9ce5c`).

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | the pre-registration, plus amendments A1–A3 with what each replaced |
| `RESULTS.md` | **the deliverable** |
| `artifacts.py` | the eight frozen artifact programs, rebuilt and digest-checked |
| `frag.py` | `mine.py`'s R1/R2 enumerated by connected growth so it scales to 586 nodes |
| `identity.py` | the three exact identity relations D, S, S\*, and the triviality rule |
| `run_check.py` | the 862-check bit-identity gate against `mine.py`; the inventory is void without it |
| `run_inventory.py` | the fragment inventory and the identity of every class |
| `run_shared.py` | which classes span a domain boundary, and are any non-trivial |
| `run_bounds.py` | B1 the type-signature bound, B2 the operator-shape relation, B3 the carrier inventory |
| `run_control.py` | P1 the Boolean positive control, P2 within-domain sharing |
| `run_regime.py` | which §57 fragmentation regime the artifacts are in |
| `show_shared.py` | reader for `out/shared_*.json` |
| `out/` | every number in `RESULTS.md`, plus `pytest.log` and `fixture.log` |

`tcn/` and `generators/` are not modified. Nothing here trains or searches; it
reads published, frozen programs.
