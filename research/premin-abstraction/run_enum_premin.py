"""Exhaustive enumeration of one arm's scaffold, with the certificate.

`python run_enum_premin.py <arm> [scaffold] [max_programs_log2]` -- one arm per
process.  §44's `run_enumerate.py`, unchanged except for the arm table it reads
and an explicit `max_programs` so a larger module arity can still be exhausted.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from tcn.search import enumerate_fit, space_size, candidate_counts

import arms_premin as arms
import later

OUT = Path(__file__).parent / "out"


def node_op(program, node_name, index):
    node = next(n for n in program.nodes if n.name == node_name)
    return node.candidates[index].operator.name


def node_src(program, node_name, index):
    node = next(n for n in program.nodes if n.name == node_name)
    return list(node.candidates[index].sources)


def main():
    arm = sys.argv[1]
    scaffold = sys.argv[2] if len(sys.argv) > 2 else "tight"
    cap = 1 << (int(sys.argv[3]) if len(sys.argv) > 3 else 24)
    r, module_name = arms.build(arm)
    program = (later.tight_scaffold(r, module_name) if scaffold == "tight"
               else later.wide_scaffold(r, module_name))
    examples = later.later_examples()
    counts = candidate_counts(program)
    total = space_size(program)
    print(arm, scaffold, "candidates", counts, "space", total, "cap", cap, flush=True)
    t0 = time.perf_counter()
    res = enumerate_fit(program, examples, later.SIGNALS, registry=r,
                        tolerance=.001, max_programs=cap, rank="order")
    row = res.to_dict()
    row.update(arm=arm, scaffold=scaffold, description=arms.DESCRIPTION[arm],
               module=module_name, candidates_per_node=list(counts),
               declared_space_size=total, max_programs=cap,
               wall_seconds=time.perf_counter() - t0)
    if res.selections:
        hardened = program.harden(res.selections).pruned()
        row["chosen"] = {k: {"operator": node_op(program, k, v),
                             "sources": node_src(program, k, v)}
                         for k, v in res.selections.items()}
        row["module_on_output_path"] = later.module_on_output_path(
            program.harden(res.selections))
        row["live_nodes"] = len(hardened.nodes)
        row["description_bits_pruned"] = hardened.description_bits(r)
        row["description_bits_call_sites_only"] = hardened.description_bits()
        row["execution_cost_pruned"] = hardened.execution_cost(r)
    OUT.mkdir(exist_ok=True)
    (OUT / f"enum_{scaffold}_{arm}.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    print(json.dumps({k: row[k] for k in
                      ("arm", "solved", "conforming", "evaluated", "space_size",
                       "exhausted", "certificate", "wall_seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
