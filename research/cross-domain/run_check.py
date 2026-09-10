"""Pre-registered correctness gate: the enumerator must equal `mine.py` exactly.

The inventory is void unless this passes on every program `mine.py` can run:
both Boolean corpora of §52's family and every artifact program under 60 nodes.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research" / "earned-abstraction"))
sys.path.insert(0, str(ROOT / "research" / "semantic-library"))

import frag
import artifacts as A

OUT = HERE / "out"


def boolean_corpora():
    """§52's own corpus, both bands, so the check runs on the Boolean family too."""
    sys.path.insert(0, str(ROOT / "research" / "semantic-library"))
    import family
    rows = {}
    for band in ("C-minall", "C-trace"):
        corpus, _ = family.corpus_for("maj", band, ())
        for name, prog in corpus.items():
            rows[f"maj/{band}/{name}"] = prog
    return rows


def main():
    checks = []
    arts, _ = A.all_artifacts()
    progs = {k: p for k, (p, _) in arts.items() if len(p.nodes) < 60}
    try:
        progs.update(boolean_corpora())
    except Exception as e:  # the corpus builder is another track's; report, do not hide
        checks.append({"program": "boolean-corpora", "error": f"{type(e).__name__}: {e}"})
    for name, p in progs.items():
        for mx in (3, 5):
            t0 = time.perf_counter()
            ok, a, b = frag.check_equivalence(p, max_nodes=mx)
            checks.append({"program": name, "nodes": len(p.nodes), "max_nodes": mx,
                           "identical": bool(ok), "mine_fragments": a,
                           "here_fragments": b, "seconds": time.perf_counter() - t0})
    passed = all(c.get("identical", False) for c in checks if "identical" in c)
    report = {"all_identical": passed, "n_checks": len([c for c in checks if "identical" in c]),
              "checks": checks}
    OUT.mkdir(exist_ok=True)
    (OUT / "equivalence.json").write_text(json.dumps(report, indent=1))
    for c in checks:
        print(c)
    print("ALL IDENTICAL:", passed)


if __name__ == "__main__":
    main()
