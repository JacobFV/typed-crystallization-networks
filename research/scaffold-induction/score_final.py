"""A13: score the arms against the `final` split — the only step that reads it.

Nothing upstream touches `final`.  `run_domain.py` fingerprints it and deletes it
from the domain dictionary before building a corpus; this script rebuilds it,
records the same fingerprint, and refuses to proceed if the two disagree — so a
corpus and its final scoring provably refer to the same blind split.

What it scores, for every arm and every case:

  M1  the first **repair** in the arm's order (repair = conforms on
      `train ∪ admission`).  Does the witness that made it a repair also conform
      on `final`?
  M2  the first edit with a **training** conformer, which is what a system
      without an admission split would accept.  Same question.

An arm that finds repairs cheaply but whose repairs do not survive `final` has
not solved anything, and that is the number this split exists to expose.

    python score_final.py
"""
from __future__ import annotations

import hashlib
import time

import kit
import domains
import edits as E
import run_domain as R


def digest(episodes):
    h = hashlib.sha256()
    for ex in episodes:
        for k in sorted(ex["inputs"]):
            h.update(repr(ex["inputs"][k].flat()).encode())
        for k in sorted(ex["targets"]):
            h.update(repr(ex["targets"][k].flat()).encode())
    return {"n": len(episodes), "sha256": h.hexdigest()[:32]}


def conforms(program, registry, selections, episodes, signals, tol=R.TOL):
    """Does this exact member conform on every episode of the split?"""
    for ex in episodes:
        try:
            _, _, trace = program.execute(ex["inputs"], registry=registry,
                                          selections=selections)
        except Exception:                                     # noqa: BLE001
            return False
        for s in signals:
            a = trace[s.source].flat()
            b = ex["targets"][s.target].flat()
            if max((abs(x - y) for x, y in zip(a, b)), default=0.) > tol:
                return False
    return True


def main():
    kit.check_workers(1)
    kit.check_floor("score_final", peak_gb=kit.measured_peak("arith"))
    t0 = time.perf_counter()
    a = kit.load("analysis")
    out = {"note": "the only step that reads the `final` split", "domains": {},
           "cases": [], "arms": a["arms"]}

    for dom in a["domains"]:
        d = domains.BUILDERS[dom]()
        base, r, sig = d["program"], d["registry"], d["signals"]
        fin = d["final"]
        fp = digest(fin)
        corpus = kit.load(f"cases_{dom}")
        if corpus.get("final_digest") != fp["sha256"]:
            raise SystemExit(
                f"{dom}: the corpus was built against a different `final` split "
                f"({corpus.get('final_digest')} vs {fp['sha256']}); refusing to "
                f"score, because the blind split would not be the one withheld")
        out["domains"][dom] = {"final_digest": fp["sha256"], "n_final": fp["n"],
                               "corpus_final_untouched": corpus.get("final_untouched"),
                               "provenance": kit.provenance(
                                   base_digest=base.digest,
                                   split_digest=fp["sha256"])}

        by_case = {c["case_id"]: c for c in corpus["cases"] if c.get("admitted")}
        for row in [x for x in a["cases"] if x["domain"] == dom]:
            case = by_case[row["case_id"]]
            df = case["defect"]
            failed = R.APPLY[df["kind"]](base, r, df["site"], df["arg"])
            progs = {e.key: p for e, p in E.enumerate_edits(failed, r)}
            store = {e["key"]: e for e in case["edits"]}
            res = {"domain": dom, "case_id": row["case_id"], "arms": {}}
            for arm in a["arms"]:
                sel = row.get("selected", {}).get(arm) or {}
                entry = {}
                for metric in ("M1", "M2"):
                    key = sel.get(metric)
                    if not key:
                        entry[metric] = {"edit": None, "generalizes": None}
                        continue
                    e = store[key]
                    ok = (bool(e["witness"]) and
                          conforms(progs[key], r, e["witness"], fin, sig))
                    entry[metric] = {"edit": key, "generalizes": bool(ok),
                                     "was_repair": bool(e["repair"])}
                res["arms"][arm] = entry
            out["cases"].append(res)
            print(f"  {row['case_id']}", flush=True)

    tally = {}
    for arm in a["arms"]:
        for metric in ("M1", "M2"):
            vals = [c["arms"][arm][metric]["generalizes"] for c in out["cases"]
                    if c["arms"][arm][metric]["generalizes"] is not None]
            tally.setdefault(arm, {})[metric] = {
                "generalizes": sum(1 for v in vals if v), "of": len(vals)}
    out["tally"] = tally
    out["seconds"] = time.perf_counter() - t0
    print("wrote", kit.dump("final_scores", out))
    print(tally)


if __name__ == "__main__":
    main()
