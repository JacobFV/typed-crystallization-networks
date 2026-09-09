"""How much validation identifies stage 1's foreground module?

Conformance on a supervised sample is not identification: at R=8 the rung-3
space has 32,000 programs and thousands of them fit the training records. This
sweeps the validation budget and reports, for each budget, how many conforming
programs survive and how accurate the surviving pick is at *every* position of
fresh test episodes. The sweep is run once and cached.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "discrete-perception"))

import chain as C  # noqa: E402
from common import all_conforming  # noqa: E402
from tcn.operators import Registry  # noqa: E402
from tcn.search import evaluate  # noqa: E402

R = 8
TOL = 1e-6
CACHE = HERE / "out" / "conforming_r8.json"


def main():
    registry = Registry()
    scaffold = C.module_scaffold(registry, R, C.POOL)
    signals = C.fg_signals()
    train = C.pixel_examples(tuple(range(6)), R, "train", 48, seed=1)
    if CACHE.exists():
        conforming = json.loads(CACHE.read_text())["conforming"]
        seconds = json.loads(CACHE.read_text())["seconds"]
    else:
        sweep = all_conforming(scaffold, train, signals, registry, tolerance=TOL)
        conforming, seconds = sweep["conforming"], sweep["seconds"]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"conforming": conforming, "seconds": seconds,
                                     "count": len(conforming)}))
    print(f"conforming on {len(train)} supervised records: {len(conforming)} of 32000 "
          f"({seconds:.1f} s)")

    every_test = C.pixel_examples(tuple(range(100, 106)), R, "test", None, seed=12)
    rows = []
    for episodes in (1, 2, 4, 8, 16):
        validation = C.pixel_examples(tuple(range(300, 300 + episodes)), R, "validation",
                                      None, seed=7)
        t0 = time.perf_counter()
        survivors = [s for s in conforming
                     if (lambda e: e is not None and e <= TOL)(
                         evaluate(scaffold, s, validation, signals, registry, TOL))]
        pick = survivors[0] if survivors else None
        error = C.exact_error(scaffold, every_test, signals, registry, selections=pick) if pick else None
        hits = sum(1 for ex in every_test
                   if evaluate(scaffold, pick, [ex], signals, registry, TOL) == 0.) if pick else 0
        rows.append({"validation_episodes": episodes, "validation_records": len(validation),
                     "survivors": len(survivors), "seconds": time.perf_counter() - t0,
                     "pick_test_max_error": error,
                     "pick_test_accuracy_every_position": hits / len(every_test)})
        print(json.dumps(rows[-1]))
    (HERE / "out" / "identification.json").write_text(
        json.dumps({"conforming": len(conforming), "sweep_seconds": seconds, "rows": rows},
                   indent=2))


if __name__ == "__main__":
    main()
