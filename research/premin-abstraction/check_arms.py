"""Every arm builds, loads its module under policy='strict', and reports it."""
from __future__ import annotations

import arms_premin as arms
import later

for arm in arms.ARMS:
    r, module = arms.build(arm)
    program = later.tight_scaffold(r, module)
    print(f"{arm:24s} module={module} nodes={len(program.nodes)}")
