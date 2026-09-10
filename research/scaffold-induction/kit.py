"""Shared paths, the capped-resource helper, and JSON dump/load.

Nothing under `tcn/` or `generators/` is imported for modification; this track
reads the core and the shipped generators only.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "out"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def dump(name, obj):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    return str(path)


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def mem_available_gb():
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemAvailable not found")


def check_floor(phase, floor_gb=25.0, log="memory_floor.log"):
    """Refuse to start a phase below the available-memory floor (HANDOFF trap)."""
    avail = mem_available_gb()
    ok = avail >= floor_gb
    OUT.mkdir(exist_ok=True)
    with open(OUT / log, "a") as fh:
        fh.write(f"{phase}\tMemAvailable={avail:.2f}GB\tfloor={floor_gb}GB\t"
                 f"{'START' if ok else 'REFUSED'}\n")
    if not ok:
        raise SystemExit(f"refusing {phase}: MemAvailable {avail:.2f} GB < {floor_gb} GB")
    return avail


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
