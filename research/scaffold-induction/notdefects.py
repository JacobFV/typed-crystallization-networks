"""How many rejected defects were not defects at all?

A defect is rejected when the narrowed scaffold still contains a conforming
member.  That can happen two ways, and they mean different things for whoever
builds the next defect generator:

  * the narrowing removed candidates the solution never used, so a conforming
    member the BASE scaffold already had survives untouched — the "defect" did
    not damage the program at all; or
  * it removed the base's member but the scaffold could reach another one, so the
    narrowing did damage and the scaffold routed around it.

The first is a generator producing non-defects.  This measures how many of each.

    python notdefects.py <domain>
"""
from __future__ import annotations

import sys

import kit
import domains
import run_domain as R


def candidates_of(program, selections):
    """The Candidate objects a selection vector picks, by node name."""
    return {nd.name: nd.candidates[selections[nd.name]] for nd in program.nodes
            if nd.name in selections}


def survives(narrowed, picked):
    """Is every candidate the base's member used still present after narrowing?"""
    by = {nd.name: nd.candidates for nd in narrowed.nodes}
    for name, cand in picked.items():
        if name not in by or cand not in by[name]:
            return False
    return True


def main():
    kit.check_workers(1)
    domain = sys.argv[1] if len(sys.argv) > 1 else "bool"
    d = domains.BUILDERS[domain]()
    base, r, sig = d["program"], d["registry"], d["signals"]
    allep = d["train"] + d["admission"]

    got = R.decide(base, allep, sig, r)
    if not got["conforming"]:
        raise SystemExit(f"{domain}: the base scaffold has no conforming member; "
                         f"the question does not arise")
    picked = candidates_of(base, got["witness"])

    rows, intact, damaged = [], 0, 0
    for kind, site, arg in R.enumerate_defects(base, r, [n.name for n in base.nodes]):
        failed = R.APPLY[kind](base, r, site, arg)
        if failed is None:
            continue
        tr = R.decide(failed, allep, sig, r)
        if not tr["decided"] or not tr["conforming"]:
            continue                      # admissible, or undecidable: not our case
        ok = survives(failed, picked)
        intact += ok
        damaged += not ok
        rows.append({"defect": f"{kind}:{site}:{arg}",
                     "base_member_survives": bool(ok)})
    out = {"domain": domain, "rejected_solvable": len(rows),
           "base_member_survived": intact,
           "scaffold_routed_around_it": damaged,
           "fraction_not_really_defects": round(intact / len(rows), 4) if rows else None,
           "rows": rows,
           "note": "`base_member_survives` means the narrowing removed only "
                   "candidates the base scaffold's own conforming member never "
                   "used, so the 'defect' did not damage the program"}
    print(kit.dump(f"notdefects_{domain}", out))
    print(f"{domain}: of {len(rows)} defects rejected as still solvable, "
          f"{intact} left the base's own conforming member intact "
          f"({100.0 * intact / len(rows):.0f}%), {damaged} were routed around")


if __name__ == "__main__":
    main()
