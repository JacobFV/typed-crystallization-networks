"""Put the three upstream tracks on the import path.

This track *imports* §52's harness (`research/semantic-library`), which itself
imports §44's (`research/earned-abstraction`) and §46's
(`research/premin-abstraction`).  It also imports §51's cost instrument
(`research/lazy-guard/cost.py`) for the execution-cost-aware objective.  Nothing
upstream is modified; importing this module is the only setup any script here
needs.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EA = ROOT / "research" / "earned-abstraction"
PM = ROOT / "research" / "premin-abstraction"
SL = ROOT / "research" / "semantic-library"
LG = ROOT / "research" / "lazy-guard"

for _p in (ROOT, EA, PM, SL, LG):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# This directory must come *first*: §52's `research/semantic-library` contains
# modules with the same names as some of this track's (`run_heldout`, `tables`),
# and a script run from here already has this directory on `sys.path`, so the
# inserts above would otherwise shadow it.
_here = str(HERE)
while _here in sys.path:
    sys.path.remove(_here)
sys.path.insert(0, _here)
