"""Run the cheap-exploit battery over the whole language catalogue.

Usage:  python audit.py [--train 600] [--test 400] [--out audit.json]
        [--lessons a,b,c] [--workers 8] [--difficulty X]

Episodes are drawn by seed.  The train and test seed ranges are disjoint, so
every exploit is fitted on one set of worlds and scored on another.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "generators", "language"))
sys.path.insert(0, HERE)

import engine as lc                                            # noqa: E402
from battery import Episode, answer_stats, run_battery         # noqa: E402


def draw(lesson_id, seeds, difficulty=None, null=False):
    """Episodes for a seed range.

    ``null`` replaces every answer with a uniformly random member of the
    episode's own choice set.  Nothing is then predictable from the text, so the
    battery's best score on a null lesson is exactly the selection bias of
    taking a maximum over sixteen fitted predictors -- the band a real lift has
    to clear.
    """
    import random as _r
    L = lc.get(lesson_id)
    rng = _r.Random(0xA0D17 ^ hash(lesson_id) & 0xFFFFFFFF)
    out, bad = [], 0
    for s in seeds:
        try:
            ex = L.example(seed=s, difficulty=difficulty)
        except Exception:
            bad += 1
            continue
        ans = ex.answer
        if null and ex.choices:
            ans = rng.choice(list(ex.choices))
        out.append(Episode(ex.observation, ex.prompt, ans, ex.choices))
    return out, bad


def oracle_bound(eps):
    """Best possible accuracy for anything reading only the observation."""
    d = defaultdict(list)
    for e in eps:
        d[e.obs].append(e.a)
    from collections import Counter
    good = sum(Counter(v).most_common(1)[0][1] for v in d.values())
    return round(good / max(1, len(eps)), 4)


def audit_lesson(args):
    lesson_id, n_tr, n_te, difficulty, null = args
    t0 = time.time()
    tr, bad_a = draw(lesson_id, range(0, n_tr), difficulty, null)
    te, bad_b = draw(lesson_id, range(1_000_000, 1_000_000 + n_te), difficulty, null)
    if not tr or not te:
        return lesson_id, {"error": "no episodes", "failed_seeds": bad_a + bad_b}
    row = {"stats": answer_stats(te), "train_n": len(tr), "test_n": len(te),
           "failed_seeds": bad_a + bad_b,
           "oracle": oracle_bound(list(tr) + list(te))}
    row["exploits"] = run_battery(tr, te)
    ex = {k: v for k, v in row["exploits"].items() if not k.endswith("_error")}
    best = max(ex.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else -1))
    row["best_exploit"], row["best_score"] = best[0], best[1]
    row["gap"] = round(row["oracle"] - best[1], 4)
    row["seconds"] = round(time.time() - t0, 2)
    return lesson_id, row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=600)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=os.path.join(HERE, "audit.json"))
    p.add_argument("--lessons", default="")
    p.add_argument("--difficulty", type=float, default=None)
    p.add_argument("--null", action="store_true",
                   help="randomise every answer within its own choice set")
    a = p.parse_args()

    ids = ([x for x in a.lessons.split(",") if x] or
           [i for i in lc.lesson_ids() if lc.get(i).status == "implemented"])
    jobs = [(i, a.train, a.test, a.difficulty, a.null) for i in ids]
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for k, (lid, row) in enumerate(pool.map(audit_lesson, jobs), 1):
            res[lid] = row
            if k % 20 == 0:
                print(f"  {k}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    wall = time.time() - t0
    payload = {"meta": {"lessons": len(ids), "train_seeds": a.train,
                        "test_seeds": a.test, "difficulty": a.difficulty, "null": a.null,
                        "episodes": len(ids) * (a.train + a.test),
                        "wall_seconds": round(wall, 1),
                        "workers": a.workers},
               "lessons": res}
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    print(f"{len(ids)} lessons, {len(ids)*(a.train+a.test)} episodes, {wall:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
