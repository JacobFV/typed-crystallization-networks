"""Put the two upstream tracks on the import path.

This track *imports* §44's harness (`research/earned-abstraction`) and §46's
corpus machinery and semantic rule (`research/premin-abstraction`) rather than
restating them, so every number is directly comparable.  Importing this module
is the only setup any script here needs; no PYTHONPATH is required.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EA = ROOT / "research" / "earned-abstraction"
PM = ROOT / "research" / "premin-abstraction"

for _p in (ROOT, EA, PM, HERE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
