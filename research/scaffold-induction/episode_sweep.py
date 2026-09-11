"""Fix each domain's training-episode budget by a STATED RULE, not by its yield.

The rule, adopted before any budget is chosen and applied identically to every
domain (PREREGISTRATION amendment A8):

    The training budget is the SMALLEST B on the doubling ladder whose
    **admissible defect set** -- the set itself, not its size -- is identical at
    B, 2B and 4B.

Choosing a budget by which value admits more cases would be tuning a corpus
parameter on the quantity the corpus is meant to measure, and a reader could not
tell the two apart.  Stability across two consecutive doublings is a property of
the domain: it says the admission decision has stopped depending on how many
episodes were drawn, so the remaining `solvable on train` defects are solvable
because a conforming program really exists, not because too few episodes were
drawn to rule one out.

Only the admission rule is evaluated here -- no edit is decided, no repair label
computed, no arm run.  The full sweep is committed to `out/episode_sweep.json`
whatever it says, including the budgets the rule rejects.

    python episode_sweep.py bool rel arith
"""
from __future__ import annotations

import sys
import time

import kit
import domains
import run_domain as R

# Doubling ladders.  `bool`'s episodes are rows of a complete 64-row truth table,
# so its ladder is bounded by the table: past 32 the held-out half is smaller
# than the training half, and at 64 there is no held-out set at all.
LADDERS = {"bool": [8, 16, 32], "rel": [96, 192, 384, 768, 1536],
           "arith": [16, 32, 64, 128, 256]}


def build(domain, budget):
    if domain == "bool":
        return domains.build_bool(n_train=budget)
    if domain == "rel":
        return domains.build_rel(n_train=budget, n_eval=96)
    if domain == "arith":
        return domains.build_arith(n_train=budget, n_eval=32)
    raise ValueError(domain)


def admissible_set(domain, budget):
    t0 = time.perf_counter()
    d = build(domain, budget)
    base, r, sig = d["program"], d["registry"], d["signals"]
    defects = R.enumerate_defects(base, r, [nd.name for nd in base.nodes])
    inv, solv, names = 0, 0, []
    for kind, site, arg in defects:
        f = R.APPLY[kind](base, r, site, arg)
        if f is None:
            inv += 1
            continue
        tr = R.decide(f, d["train"] + d["admission"], sig, r)
        if tr["decided"] and not tr["conforming"] and tr["exhausted"]:
            names.append(f"{kind}:{site}:{arg}")
        else:
            solv += 1
    row = {"budget": budget, "n_train": len(d["train"]),
           "n_admission": len(d["admission"]), "n_final": len(d["final"]),
           "defects": len(defects),
           "invalid": inv, "solvable_on_train": solv,
           "admissible": len(names), "admissible_set": sorted(names),
           "seconds": round(time.perf_counter() - t0, 1)}
    print(f"  {domain} budget={budget} (train={row['n_train']}): "
          f"admissible {row['admissible']}  [{row['seconds']}s]", flush=True)
    return row


def choose(rows):
    """The smallest budget whose admissible SET equals that at 2B and 4B."""
    by = {r["budget"]: r for r in rows}
    for r in rows:
        b = r["budget"]
        two, four = by.get(b * 2), by.get(b * 4)
        if two is None or four is None:
            continue
        if r["admissible_set"] == two["admissible_set"] == four["admissible_set"]:
            return b, "stable at B, 2B and 4B"
    return None, ("no budget on the ladder is stable across two doublings; "
                  "the largest swept budget is used and the instability is "
                  "reported as a limitation")


def main():
    kit.check_workers(1)
    kit.check_floor("episode_sweep")
    out = {"rule": ("the smallest budget whose admissible defect SET is "
                    "identical at B, 2B and 4B"),
           "ladders": LADDERS, "domains": {}}
    for domain in sys.argv[1:] or ["bool", "rel", "arith"]:
        print(f"== {domain}")
        rows = [admissible_set(domain, b) for b in LADDERS[domain]]
        chosen, why = choose(rows)
        stable = chosen is not None
        if chosen is None:
            chosen = LADDERS[domain][-1]
        # `stable` is recorded explicitly so a verifier never has to substring
        # match the prose: "no budget on the ladder is stable" contains "stable".
        out["domains"][domain] = {"rows": rows, "chosen_budget": chosen,
                                  "reason": why, "stable": stable}
        print(f"  -> {domain}: chosen budget {chosen} ({why})", flush=True)
    print(kit.dump("episode_sweep", out))


if __name__ == "__main__":
    main()
