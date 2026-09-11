"""How many defects a domain can admit at all, enumerated exhaustively.

This answers a question the corpus runs answer only slowly: the pre-registration
asked for 12-24 admitted cases per domain, and whether that is reachable is a
property of the defect generator and the scaffold, not of the time budget.  The
admission test alone (`0 conformers on the training episodes, exhausted`) is
cheap; deciding a case's edits is what is expensive.  So this enumerates every
defect, applies the admission rule, and reports the ceiling.

It does **not** decide any edit, so an admitted defect here may still turn out to
contain no repair and be rejected by `run_domain.py` -- the number it prints is
an upper bound on the corpus size, and it says so.

    python admissible.py bool arith rel
"""
from __future__ import annotations

import sys
import time

import kit
import domains
import run_domain as R
from tcn.search import space_size


def scan(name):
    kit.check_workers(1)
    t0 = time.perf_counter()
    d = domains.BUILDERS[name]()
    base, r, sig = d["program"], d["registry"], d["signals"]
    defects = R.enumerate_defects(base, r, [nd.name for nd in base.nodes])
    rows, counts = [], {"invalid": 0, "solvable_on_train": 0, "admissible": 0}
    for kind, site, arg in defects:
        failed = R.APPLY[kind](base, r, site, arg)
        if failed is None:
            counts["invalid"] += 1
            continue
        tr = R.decide(failed, d["train"], sig, r)
        ok = tr["decided"] and not tr["conforming"] and tr["exhausted"]
        counts["admissible" if ok else "solvable_on_train"] += 1
        if ok:
            rows.append({"defect": f"{kind}:{site}:{arg}",
                         "failed_space": tr["space"],
                         "certificate": tr["certificate"]})
    out = {"domain": name, "defects_enumerated": len(defects), **counts,
           "admissible_defects": rows,
           "note": "an upper bound on corpus size: a defect admitted here is "
                   "still rejected by run_domain.py if its edit space contains "
                   "no repair",
           "base_space": space_size(base), "seconds": time.perf_counter() - t0}
    print(f"{name}: {len(defects)} defects -> invalid {counts['invalid']}, "
          f"solvable on train {counts['solvable_on_train']}, "
          f"ADMISSIBLE {counts['admissible']}  [{out['seconds']:.0f}s]", flush=True)
    kit.dump(f"admissible_{name}", out)
    return out


if __name__ == "__main__":
    for n in sys.argv[1:] or ["bool", "rel", "arith"]:
        scan(n)
