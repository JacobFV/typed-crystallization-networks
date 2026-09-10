"""Record the training and held-out episodes once, live, into `out/`.

Each record is read off one live episode of the integrated generator (GUI draw,
language realization, kernel setup) and holds the actor's observations plus the
probes used only for scoring and equivalence checks. The evaluator reads these
records; V3 (`live.py`) re-runs episodes live to show the recorded rewards are
the environment's.

    python cache.py            # both configurations
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env  # noqa: E402

OUT = HERE / "out"
SEED = 0
N_TRAIN = 12
N_TEST = 36


def build(gap):
    started = time.perf_counter()
    rows = []
    for split, n in (("train", N_TRAIN), ("test", N_TEST)):
        for index in range(n):
            rows.append(env.episode_record(SEED, index, split, gap, session=f"cache{gap}"))
    payload = {"gap": gap, "seed": SEED, "n_train": N_TRAIN, "n_test": N_TEST,
               "configuration": dict(env.GUI_CONFIGURATION, gap=gap),
               "colours": list(env.COLOURS), "relations": list(env.RELATIONS),
               "episodes": rows, "seconds": time.perf_counter() - started}
    OUT.mkdir(exist_ok=True)
    (OUT / f"episodes_gap{gap}.json").write_text(json.dumps(payload))
    return payload


def load(gap):
    return json.loads((OUT / f"episodes_gap{gap}.json").read_text())


if __name__ == "__main__":
    for gap in (0, 1):
        p = build(gap)
        print(f"gap {gap}: {len(p['episodes'])} episodes in {p['seconds']:.1f}s")
