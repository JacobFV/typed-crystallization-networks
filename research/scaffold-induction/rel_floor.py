"""Is `rel`'s one-case ceiling a property of the scaffold or of the episode count?

`rel` admits exactly one defect at its committed split: of 103 defects, 62 are
invalid (a narrowing that empties a single-candidate node is not a program) and
40 leave the training episodes still solvable.  The second number is the one that
could move: with more training episodes, a defect that currently keeps a spurious
conforming member would stop keeping one, and would become admissible.

This measures the admissible count as a function of the training-episode budget.
No edit is decided and no repair label is computed; it is the admission rule
only, which is cheap.

    python rel_floor.py 96 192 384
"""
from __future__ import annotations

import sys
import time

import kit
import domains
import run_domain as R


def scan(n_train):
    t0 = time.perf_counter()
    d = domains.build_rel(n_train=n_train, n_heldout=96)
    base, r, sig = d["program"], d["registry"], d["signals"]
    defects = R.enumerate_defects(base, r, [nd.name for nd in base.nodes])
    inv = solv = adm = 0
    names = []
    for kind, site, arg in defects:
        f = R.APPLY[kind](base, r, site, arg)
        if f is None:
            inv += 1
            continue
        tr = R.decide(f, d["train"], sig, r)
        if tr["decided"] and not tr["conforming"] and tr["exhausted"]:
            adm += 1
            names.append(f"{kind}:{site}:{arg}")
        else:
            solv += 1
    row = {"n_train": n_train, "defects": len(defects), "invalid": inv,
           "solvable_on_train": solv, "admissible": adm, "names": names,
           "seconds": round(time.perf_counter() - t0, 1)}
    print(f"  n_train={n_train}: admissible {adm} "
          f"(invalid {inv}, still solvable {solv})  [{row['seconds']}s]", flush=True)
    return row


if __name__ == "__main__":
    budgets = [int(x) for x in sys.argv[1:]] or [96, 192, 384]
    rows = [scan(n) for n in budgets]
    print(kit.dump("rel_floor", {"rows": rows,
                                 "note": "admission rule only; no edit decided"}))
