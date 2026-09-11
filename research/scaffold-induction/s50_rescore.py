"""Secondary arm S1: re-score section 50's own corpus on **held-out** conformance.

Section 50's `allholes` arm repaired **21 of 24** failed scaffolds.  That count
is scored on *training* accuracy 1.000 against a known-repair set.  This track's
criterion is held-out conformance (PREREGISTRATION section 5), and section 65
recorded that a training-conforming first hit can generalize 0/400.  So the same
corpus is re-decided here with section 50's own validated deciders
(`research/scaffold-diagnosis/langfam.py`, `boolfam.py`) and its own cases -- no
new scaffold, no new task, nothing re-derived.

Pre-registered prediction (F8): the count **falls**.

For each failed case, the same one-hole variants section 50 enumerated are
decided twice: on the 24 training episodes, and on those plus 40 held-out
episodes at unseen lengths.  Two numbers come out:

  repair_exists      a variant conforms on all 64 episodes
  training_stop_ok   the variant a training-only protocol stops at also
                     conforms on all 64

The Boolean cases have no held-out split -- their episodes are the complete
64-row truth table -- so they are reported separately and unchanged.
"""
from __future__ import annotations

import sys
import time

import kit
sys.path.insert(0, str(kit.ROOT / "research" / "scaffold-diagnosis"))
sys.path.insert(0, str(kit.ROOT / "research" / "earned-abstraction"))

import cases as case_mod                                          # noqa: E402
import langfam                                                    # noqa: E402
import boolfam                                                    # noqa: E402
import probe                                                      # noqa: E402
from answers import TRUTH                                         # noqa: E402

N_HELD = 40


def lang_variants(case):
    out = [(case["acc_fold"], f) for f in case_mod.FOLDS]
    out += [(f, case["second_fold"]) for f in case_mod.FOLDS]
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def run_lang(cid, case):
    train, labels = case["train"], case["labels"]
    held_eps, held_lab = case["heldout"]["heldout_unseen_lengths"]
    held_eps, held_lab = held_eps[:N_HELD], held_lab[:N_HELD]
    allep, alllab = train + held_eps, labels + held_lab
    o_tr = langfam.opens_cache(train, case["positions"])
    o_all = langfam.opens_cache(allep, case["positions"])
    rows = []
    for acc, sec in lang_variants(case):
        if (acc, sec) == (case["acc_fold"], case["second_fold"]):
            continue
        a = langfam.sweep(train, labels, case["positions"], o_tr, acc, sec)
        b = langfam.sweep(allep, alllab, case["positions"], o_all, acc, sec)
        rows.append({"variant": f"{acc}/{sec}",
                     "best_train_accuracy": a["best_train_accuracy"],
                     "conforming_on_train": a["conforming_on_train"],
                     # `langfam.sweep` names its count `conforming_on_train`
                     # whatever episodes it is given; `b` was given all 64.
                     "conforming_on_all": b["conforming_on_train"]})
    return rows, len(allep)


def run_bool(cid, case):
    rows = []
    for nd in case["program"].nodes:
        for name in probe.core_candidates(case, nd.name):
            ext = boolfam.extend_site(case["program"], case["registry"], nd.name, name)
            if not ext:
                continue
            r = boolfam.sweep(case["program"], case["registry"], case["train"],
                              override={nd.name: ext})
            rows.append({"variant": f"{nd.name}:{name}",
                         "best_train_accuracy": r["best_train_accuracy"],
                         "conforming_on_train": r["conforming_on_train"],
                         "conforming_on_all": r["conforming_on_train"]})
    return rows, len(case["train"])


def main():
    kit.check_workers(1)
    kit.check_floor("s50_rescore")
    t0 = time.perf_counter()
    out = {"n_held_out_used": N_HELD, "cases": [],
           "note": "section 50's own cases and deciders; the Boolean cases have "
                   "no held-out split (complete 64-row truth table) and are "
                   "reported unchanged"}
    for cid in case_mod.CASES:
        truth = TRUTH.get(cid)
        if truth is None or truth["contains_solution"]:
            continue                       # solvable controls are not failed scaffolds
        case = case_mod.CASES[cid]()
        rows, n_ep = (run_lang(cid, case) if case["adapter"] == "lang"
                      else run_bool(cid, case))
        usable = [r for r in rows if r["best_train_accuracy"] is not None]
        best = max((r["best_train_accuracy"] for r in usable), default=None)
        winners = [r for r in usable if r["best_train_accuracy"] == best]
        out["cases"].append({
            "case_id": cid, "adapter": case["adapter"], "episodes": n_ep,
            "n_variants": len(rows),
            "s50_best_train_accuracy": best,
            "s50_solves_on_train": [r["variant"] for r in rows
                                    if r["conforming_on_train"]],
            "repair_exists_heldout": any(r["conforming_on_all"] for r in rows),
            "training_stop_variant": winners[0]["variant"] if winners else None,
            "training_stop_conforms_heldout":
                bool(winners and winners[0]["conforming_on_all"]),
            "rows": rows})
        print(f"  {cid}: repair_exists={out['cases'][-1]['repair_exists_heldout']} "
              f"stop_ok={out['cases'][-1]['training_stop_conforms_heldout']}", flush=True)

    lang = [c for c in out["cases"] if c["adapter"] == "lang"]
    boolc = [c for c in out["cases"] if c["adapter"] == "bool"]
    out["summary"] = {
        "failed_cases": len(out["cases"]),
        "lang_cases": len(lang), "bool_cases": len(boolc),
        "s50_solved_on_train": sum(1 for c in out["cases"] if c["s50_solves_on_train"]),
        "repair_exists_heldout": sum(1 for c in out["cases"] if c["repair_exists_heldout"]),
        "training_stop_conforms_heldout":
            sum(1 for c in out["cases"] if c["training_stop_conforms_heldout"]),
        "lang_training_stop_conforms_heldout":
            sum(1 for c in lang if c["training_stop_conforms_heldout"]),
        "lang_repair_exists_heldout":
            sum(1 for c in lang if c["repair_exists_heldout"]),
    }
    out["seconds"] = time.perf_counter() - t0
    print("wrote", kit.dump("s50_rescore", out))
    print(out["summary"])


if __name__ == "__main__":
    main()
