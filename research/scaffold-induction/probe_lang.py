"""Schedulability probe ONLY for the language domain (PREREGISTRATION section 0)."""
from __future__ import annotations
import sys
import time

import kit  # noqa: F401
import domains
import edits as E
from tcn.search import enumerate_prefix, space_size

pos = int(sys.argv[1]) if len(sys.argv) > 1 else 12
stream = sys.argv[2] if len(sys.argv) > 2 else "none"
t = time.perf_counter()
d = domains.build_lang(positions=pos, stream=stream)
p, r = d["program"], d["registry"]
print(f"build {time.perf_counter() - t:.1f}s nodes {len(p.nodes)} space {space_size(p)} "
      f"train {len(d['train'])} heldout {len(d['heldout'])}")
if not d["train"]:
    raise SystemExit("empty split")
t = time.perf_counter()
res = enumerate_prefix(p, d["train"], d["signals"], r, tolerance=1e-6,
                       max_programs=1 << 26, stop_at_first=True)
print(f"train decide {time.perf_counter() - t:.1f}s conforming {res.conforming} "
      f"cert {res.certificate}")
t = time.perf_counter()
es = E.enumerate_edits(p, r)
print(f"edits {len(es)} in {time.perf_counter() - t:.1f}s")
