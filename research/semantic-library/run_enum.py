"""Exhaustive enumeration of one arm on the later task L1, with the certificate.

`python run_enum.py <arm> [tight|wide]` -- one arm per process; the tight space
is 2 709 504 programs and takes minutes.  `max_programs` is set above the space
size so every run is a full sweep and `exhausted` / `certificate` mean what they
say.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import _paths  # noqa: F401

from tcn.search import enumerate_fit, space_size, candidate_counts

import later

import armlib

OUT = Path(__file__).resolve().parent / "out"


def enumerate_arm(arm, scaffold="tight"):
    r, module_name = armlib.build(armlib.l1_spec(arm))
    program = (later.tight_scaffold(r, module_name) if scaffold == "tight"
               else later.wide_scaffold(r, module_name))
    examples = later.later_examples()
    counts = candidate_counts(program)
    total = space_size(program)
    print(arm, scaffold, "candidates", counts, "space", total, flush=True)
    t0 = time.perf_counter()
    res = enumerate_fit(program, examples, later.SIGNALS, registry=r,
                        tolerance=.001, max_programs=1 << 27, rank="order")
    row = res.to_dict()
    row.update(arm=arm, scaffold=scaffold, description=armlib.DESCRIPTION.get(arm, ""),
               module=module_name, candidates_per_node=list(counts),
               declared_space_size=total, wall_seconds=time.perf_counter() - t0)
    if res.selections:
        hardened = program.harden(res.selections)
        row["module_on_output_path"] = later.module_on_output_path(hardened)
        row["live_nodes"] = len(hardened.pruned().nodes)
    OUT.mkdir(exist_ok=True)
    (OUT / f"enum_{scaffold}_{arm}.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    print(json.dumps({k: row[k] for k in
                      ("arm", "solved", "conforming", "evaluated", "space_size",
                       "exhausted", "certificate", "wall_seconds")}, sort_keys=True),
          flush=True)
    return row


if __name__ == "__main__":
    enumerate_arm(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "tight")
