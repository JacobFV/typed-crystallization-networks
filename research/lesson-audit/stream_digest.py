"""A digest of the whole sampled stream, for equivalence checking.

The precedent is FINDINGS section 10: a fix to a generator is not landed until
the default draw has been shown bit-identical across a grid of configurations.
Here the grid is every implemented lesson x language x difficulty x seed, and
the artefact is one digest per (lesson, configuration) cell.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "generators", "language"))
import engine as lc                                            # noqa: E402

LANGS = ("english", "spanish", "turkish")
DIFFS = (None, 0.0, 0.5, 1.0)
SEEDS = range(8)


def cell(job):
    lid, kwargs = job
    L = lc.get(lid)
    h = hashlib.blake2b(digest_size=16)
    for lang in LANGS:
        for d in DIFFS:
            for s in SEEDS:
                try:
                    ex = L.example(seed=s, language=lang, difficulty=d, **kwargs)
                    h.update(json.dumps(ex.to_dict(), sort_keys=True,
                                        ensure_ascii=False).encode("utf8"))
                except Exception as exc:
                    h.update(f"!{type(exc).__name__}".encode())
    return lid, h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--hardening", default="")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    kwargs = {"hardening": a.hardening} if a.hardening else {}
    ids = [i for i in lc.lesson_ids() if lc.get(i).status == "implemented"]
    out = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for lid, h in pool.map(cell, [(i, kwargs) for i in ids]):
            out[lid] = h
    n = len(ids) * len(LANGS) * len(DIFFS) * len(SEEDS)
    json.dump({"cells": len(ids), "configurations": len(LANGS) * len(DIFFS),
               "seeds": len(SEEDS), "episodes": n,
               "hardening": a.hardening or "(default)",
               "digests": out}, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"{len(ids)} lessons x {len(LANGS)*len(DIFFS)} configurations x "
          f"{len(SEEDS)} seeds = {n} episodes -> {a.out}")


if __name__ == "__main__":
    main()
