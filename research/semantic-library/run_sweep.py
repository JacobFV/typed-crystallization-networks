"""The strong form of the wrong-module control: *every* eligible arity-3 pooled
class, enumerated on L1's tight scaffold.

If a large fraction of the eligible pooled classes reach 144 conforming, then
pooling's **rank** is selecting nothing and the ceiling is a property of the
scaffold, not of the abstraction.  If only the majority class does, the rank is
load-bearing.  Each class is registered directly as a module operator (the same
`Registry.register_module` a `Library` load ends up calling), so the sweep costs
one enumeration per class and nothing else.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _paths  # noqa: F401

from tcn.operators import Registry
from tcn.search import enumerate_fit, space_size

import later

import poolmine

OUT = Path(__file__).resolve().parent / "out"


def _one(job):
    name, digest, rank, is_window, tt, nodes, arity, scaffold = job
    payload = json.loads((OUT / f"mined_{name}.json").read_text())
    canon = poolmine.rebuild(payload, digest)
    r = Registry()
    module_name = r.register_module(canon)
    program = (later.tight_scaffold(r, module_name) if scaffold == "tight"
               else later.wide_scaffold(r, module_name))
    t0 = time.perf_counter()
    res = enumerate_fit(program, later.later_examples(), later.SIGNALS, registry=r,
                        tolerance=.001, max_programs=1 << 27, rank="order")
    row = res.to_dict()
    row.update(source=name, digest=digest, rank=rank, is_window=is_window,
               truth_table=tt, nodes=nodes, arity=arity, module=module_name,
               declared_space_size=space_size(program),
               wall_seconds=time.perf_counter() - t0)
    return row


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "maj_minall_sem"
    payload = json.loads((OUT / f"mined_{name}.json").read_text())
    classes = [r for r in payload["ranked"] if r["arity"] == 3]
    jobs = [(name, r["digest"], r["rank"], r["is_window"], r["truth_table"],
             r["nodes"], r["arity"], "tight") for r in classes]
    print("sweeping", len(jobs), "arity-3 classes of", payload["eligible"],
          "eligible", flush=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("TCN_WORKERS", "10"))) as pool:
        rows = list(pool.map(_one, jobs))
    rows.sort(key=lambda r: r["rank"])
    reached = [r for r in rows if r["conforming"] >= 144]
    payload_out = {"source": name, "classes_swept": len(rows),
                   "eligible_total": payload["eligible"],
                   "reached_ceiling": len(reached),
                   "reached_ceiling_ranks": [r["rank"] for r in reached],
                   "any_conforming": sum(1 for r in rows if r["conforming"] > 0),
                   "rows": rows}
    (OUT / f"sweep_{name}.json").write_text(json.dumps(payload_out, indent=2, sort_keys=True))
    for r in rows:
        print(r["rank"], "window" if r["is_window"] else "-", "nodes", r["nodes"],
              "conforming", r["conforming"], r["certificate"], "exhausted",
              r["exhausted"], flush=True)
    print("reached 144:", len(reached), "of", len(rows), "arity-3 classes")


if __name__ == "__main__":
    main()
