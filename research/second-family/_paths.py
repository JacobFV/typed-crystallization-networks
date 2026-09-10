"""Put the upstream tracks on the import path.

This track *imports* the harnesses of §46 (`research/premin-abstraction`), §52
(`research/semantic-library`) and §54 (`research/reuse-ranking`) rather than
restating them, so every number is directly comparable to theirs.  Only the
task family is new.

This directory is inserted **first**, so the modules this track overrides
(`family`, `evaltasks`) shadow the same-named modules of the upstream tracks
while the ones it reuses verbatim (`pool`, `objectives`, `context`, `mine`,
`mine_semantic`, `mine_multi`, `enumerate_programs`, `minimal`) are picked up
from upstream unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EA = ROOT / "research" / "earned-abstraction"
PM = ROOT / "research" / "premin-abstraction"
SL = ROOT / "research" / "semantic-library"
RR = ROOT / "research" / "reuse-ranking"
LG = ROOT / "research" / "lazy-guard"

for _p in (ROOT, EA, PM, SL, RR, LG):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

_here = str(HERE)
while _here in sys.path:
    sys.path.remove(_here)
sys.path.insert(0, _here)
