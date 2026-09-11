"""Peak memory of one case's edit enumeration, measured before arith is launched.

The rate probe died under a 12 GB cap without writing, which is evidence about
`arith` rather than a one-off.  The suspect is `enumerate_edits`, which
materialises **every** edited program at once: without the A11 truncation each
`ADD_NODE`/`WIDEN` edit can carry thousands of candidates, and `arith` produces
~465 edits per case, so the whole list is alive simultaneously.

This measures peak RSS at three points, so the cap can be sized to a number
rather than a guess:

  1. after building the domain (episodes + base scaffold),
  2. after materialising the full edit list (the suspect),
  3. after deciding a handful of edits one at a time.

    python mem_probe.py <domain> [--decide 6]
"""
from __future__ import annotations

import argparse
import gc
import resource

import kit
import domains
import edits as E
import run_domain as R
from tcn.search import space_size


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--decide", type=int, default=6)
    a = ap.parse_args()

    marks = {"start": rss_gb()}
    d = domains.BUILDERS[a.domain]()
    base, r, sig = d["program"], d["registry"], d["signals"]
    allep = d["train"] + d["admission"]
    marks["after_domain"] = rss_gb()

    failed = None
    for kind, site, arg in R.enumerate_defects(base, r, [n.name for n in base.nodes]):
        f = R.APPLY[kind](base, r, site, arg)
        if f is None:
            continue
        tr = R.decide(f, d["train"], sig, r)
        if tr["decided"] and not tr["conforming"] and tr["exhausted"]:
            failed, defect = f, f"{kind}:{site}:{arg}"
            break
    marks["after_admission_scan"] = rss_gb()

    es = E.enumerate_edits(failed, r)
    marks["after_full_edit_list"] = rss_gb()
    cands = sum(len(n.candidates) for _, p in es for n in p.nodes)
    sizes = sorted(space_size(p) for _, p in es)

    for i, (edit, p) in enumerate(es[:a.decide]):
        R.decide(p, allep, sig, r)
    marks["after_decisions"] = rss_gb()

    del es
    gc.collect()
    marks["after_release"] = rss_gb()

    out = {"domain": a.domain, "defect": defect,
           "n_edits": len(sizes),
           "total_candidates_held": cands,
           "median_space": sizes[len(sizes) // 2], "max_space": sizes[-1],
           "episodes_all": len(allep), "episodes_train": len(d["train"]),
           "peak_rss_gb": {k: round(v, 3) for k, v in marks.items()},
           "note": "ru_maxrss is a high-water mark, so later marks include "
                   "earlier peaks; the jump at after_full_edit_list is the cost "
                   "of materialising every edited program at once"}
    print(kit.dump(f"mem_{a.domain}", out))
    for k, v in marks.items():
        print(f"  {k:26s} {v:7.3f} GB")
    print(f"  edits {len(sizes)}  candidates held {cands:,}  "
          f"median space {sizes[len(sizes) // 2]:,}  max {sizes[-1]:,}")


if __name__ == "__main__":
    main()
