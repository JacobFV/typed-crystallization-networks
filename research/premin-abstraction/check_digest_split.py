"""Are two mined majority abstractions the *same circuit* under different digests?

R2's canonical identity is `Program.digest`, and the canonical form numbers
nodes by the host program's topological order and keeps each operator's
argument order.  Neither is canonical up to graph isomorphism, so one circuit
can occupy several digests.  This checks that directly: it rebuilds each mined
majority abstraction, computes the order-free identity used by
`enumerate_programs.key_of` (the set of gate definitions keyed by truth-table
value, with commutative arguments sorted), and groups the digests by it.  It
also compares them against §44's hand-authored MAJ3 body.
"""
from __future__ import annotations

import json
from pathlib import Path

import enumerate_programs as E
import later

OUT = Path(__file__).parent / "out"
N = 3
RENAME = {"a": "x0", "b": "x1", "c": "x2"}


def key_from_body(body):
    """Order-free identity of a 3-input abstraction given its node list."""
    base = [E.var_table(N, j) for j in range(N)]
    vals = {f"x{j}": base[j] for j in range(N)}
    m = E.mask(N)
    defs = set()
    for op, name, args in body:
        u = vals[args[0]]
        v = vals[args[1]] if len(args) > 1 else u
        w = E.apply(op, u, v, m)
        vals[name] = w
        defs.add((w, op, u, u) if op == "not" else (w, op, min(u, v), max(u, v)))
    return frozenset(defs)


def authored_body():
    return [[op, name, [RENAME.get(u, u), RENAME.get(v, v)]]
            for op, name, u, v in later.MINIMAL_BODIES["maj"]]


def main():
    d = json.loads((OUT / "proposal_C-trace.json").read_text())
    groups = {}
    for r in d["maj3_proposals"]:
        groups.setdefault(key_from_body(r["body"]), []).append(r)
    authored_key = key_from_body(authored_body())

    print("eligible majority digests:", len(d["maj3_proposals"]))
    print("distinct majority *circuits* among them:", len(groups))
    for key, rows in sorted(groups.items(), key=lambda kv: min(r["rank"] for r in kv[1])):
        print(f"  ranks {[r['rank'] for r in rows]}: {len(rows)} digest(s), "
              f"{sum(r['entries'] for r in rows)} entries"
              f"{'   <-- §44 hand-authored body' if key == authored_key else ''}")
    print("hand-authored body present among mined digests:", authored_key in groups)


if __name__ == "__main__":
    main()
