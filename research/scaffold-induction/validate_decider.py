"""V2 — the decider, checked two ways (PREREGISTRATION section 9).

`enumerate_prefix` with `stop_at_first` is what decides every (case, edit) pair.
It is core machinery rather than a track simulator, so there is no simulator to
validate; it is nevertheless checked here, because a decision that is wrong in
one direction would silently create or destroy repairs:

  * **against `enumerate_fit`** — the flat `itertools.product` walk over the same
    space, run to exhaustion, on a deterministic sample of edits per case.  The
    two must agree on whether a conforming member exists.
  * **against `Program.execute`** — every recorded witness is re-executed, node
    by node, on **every** episode of its domain, and its output compared with the
    target.  A witness that does not reproduce is a fabricated repair.

    python validate_decider.py <domain> [--per-case 40]
"""
from __future__ import annotations

import argparse
import time

import kit
import domains
import edits as E
import run_domain as R
from tcn.search import enumerate_fit, space_size

FIT_CAP = 60_000          # `enumerate_fit` has no prefix reuse; keep it bounded


def rebuild(domain):
    d = domains.BUILDERS[domain]()
    return d


def check(domain, per_case):
    kit.check_workers(1)
    kit.check_floor(f"validate_decider:{domain}",
                    peak_gb=kit.measured_peak(domain))
    t0 = time.perf_counter()
    d = rebuild(domain)
    base, r, sig = d["program"], d["registry"], d["signals"]
    allep = d["train"] + d["admission"]
    corpus = kit.load(f"cases_{domain}")
    out = {"domain": domain, "per_case": per_case,
           "fit_checked": 0, "fit_mismatches": [],
           "witnesses_checked": 0, "witness_mismatches": [],
           "episode_evaluations": 0}
    for case in corpus["cases"]:
        if not case.get("admitted"):
            continue
        df = case["defect"]
        failed = R.APPLY[df["kind"]](base, r, df["site"], df["arg"])
        assert failed is not None, case["case_id"]
        es = {e.key: p for e, p in E.enumerate_edits(failed, r)}
        rows = {e["key"]: e for e in case["edits"]}
        assert set(es) == set(rows), (
            f"{case['case_id']}: the edit enumeration is not reproducible")

        # (a) a deterministic sample, smallest spaces first so the flat walk ends
        sample = sorted((e for e in case["edits"] if e["decided"]),
                        key=lambda e: (e["space"], e["key"]))[:per_case]
        for e in sample:
            prog = es[e["key"]]
            if space_size(prog) > FIT_CAP:
                continue
            res = enumerate_fit(prog, allep, sig, r, tolerance=R.TOL,
                                max_programs=1 << 24, stop_at_first=True)
            got = res.conforming > 0
            out["fit_checked"] += 1
            if got != bool(e["repair"]):
                out["fit_mismatches"].append(
                    {"case": case["case_id"], "edit": e["key"],
                     "prefix": e["repair"], "fit": got})

        # (b) every recorded witness, re-executed
        for e in case["edits"]:
            if not e["repair"] or not e["witness"]:
                continue
            prog = es[e["key"]]
            ok = True
            for ex in allep:
                try:
                    _, _, trace = prog.execute(ex["inputs"], registry=r,
                                               selections=e["witness"])
                except Exception:                            # noqa: BLE001
                    ok = False
                    break
                out["episode_evaluations"] += 1
                for s in sig:
                    a = trace[s.source].flat()
                    b = ex["targets"][s.target].flat()
                    if max((abs(x - y) for x, y in zip(a, b)), default=0.) > R.TOL:
                        ok = False
                        break
                if not ok:
                    break
            out["witnesses_checked"] += 1
            if not ok:
                out["witness_mismatches"].append(
                    {"case": case["case_id"], "edit": e["key"]})
        print(f"  {case['case_id']}: fit {out['fit_checked']} witnesses "
              f"{out['witnesses_checked']} mismatches "
              f"{len(out['fit_mismatches'])}/{len(out['witness_mismatches'])}",
              flush=True)
    out["seconds"] = time.perf_counter() - t0
    out["peak_rss_gb"] = round(kit.peak_rss_gb(), 3)
    print("wrote", kit.dump(f"validate_{domain}", out))
    print({k: v for k, v in out.items() if not isinstance(v, list)},
          "mismatches", len(out["fit_mismatches"]), len(out["witness_mismatches"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--per-case", type=int, default=40)
    a = ap.parse_args()
    check(a.domain, a.per_case)
