# Discrete backend: recurrence, live return, and a bounded walk

`tcn/search.py` could score exactly one thing: a feed-forward program against
probe targets. This track adds the two settings that were reachable only through
bespoke scripts -- recurrence and live environment return -- and a prefix-reusing
walk with an optional beam, and measures all three.

    .venv/bin/python research/discrete-backend/recurrent_depth.py     #  ~9 s
    .venv/bin/python research/discrete-backend/environment_policy.py  # ~15 s
    .venv/bin/python research/discrete-backend/beam_search.py         # ~7 min
    .venv/bin/python research/discrete-backend/report.py              # regenerates every table

Raw JSON lands in `out/`. `report.py` regenerates every table in `RESULTS.md`
from it; nothing there is transcribed by hand.

Files changed under `tcn/`: `tcn/search.py` only. New tests:
`tests/test_search_modes.py`.
