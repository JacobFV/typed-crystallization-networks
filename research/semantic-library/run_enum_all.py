"""Enumerate several L1 arms in parallel, one process each."""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _paths  # noqa: F401

import run_enum

OUT = Path(__file__).resolve().parent / "out"


def _one(arm):
    return run_enum.enumerate_arm(arm, "tight")


if __name__ == "__main__":
    arms = sys.argv[1:]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("TCN_WORKERS", "10"))) as pool:
        rows = list(pool.map(_one, arms))
    print(json.dumps([{k: r[k] for k in ("arm", "conforming", "evaluated", "space_size",
                                         "exhausted", "certificate", "wall_seconds")}
                      for r in rows], indent=2, sort_keys=True))
