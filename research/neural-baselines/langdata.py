"""Episodes for the language arm, on the stream the §19 artifact was produced on.

WHY THIS FILE EXISTS AND IS NOT AN IMPORT OF THE TRACK'S OWN `common.py`.

`research/language-capability/common.py` calls the generator with the *default*
configuration. FINDINGS §24 re-drew 14 lessons after §19 was recorded, and
`context_free_language` is one of them, so on the current tree that default
stream emits string lengths {10..22} and **the splits §19 reports -- train on
lengths {2,4,6}, hold out {8..16} -- do not exist in it at all**. The pre-audit
stream is preserved exactly as `hardening="none"` (FINDINGS §24: "bit-identical
to the pre-audit generator over 179 lessons"), and that is the stream the frozen
program was searched on.

So every §19 number is reproduced here under `hardening="none"`, and the current
default stream is reported *as a second held-out set* that both methods face on
identical terms. Both are labelled everywhere they appear.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcn.generation import Host, read_text
from tcn.types import Value

LESSON = "context_free_language"
CAPACITY = 128
LEGACY = "none"        # the pre-audit stream the artifact was searched on
CURRENT = None         # the post-audit default


def episode(seed: int, split: str = "train", capacity: int = CAPACITY, hardening=LEGACY):
    configuration = {"lesson": LESSON, "capacity": capacity}
    if hardening is not None:
        configuration["hardening"] = hardening
    host = Host.create("language", seed=seed, split=split, configuration=configuration)
    record = host.records[-1]
    view = record.actor_view()
    assert set(view.observations) == {"text"}, view.observations
    text = record.observations["text"].value
    construction = record.latent_states["construction"]
    answer = read_text(record.probes["answer"])
    string = read_text(Value(construction.type.items[1].items[1], construction.raw[1][1]))
    depth = Value(construction.type.items[0].items[1], construction.raw[0][1]).decoded
    prompt = read_text(text)
    return {"seed": seed, "split": split, "text": text, "prompt": prompt, "string": string,
            "answer": answer, "label": answer == "yes", "length": len(string),
            "depth": depth, "offset": prompt.index(string) if string in prompt else -1,
            "hardening": hardening}


def dataset(n, seed0=0, split="train", hardening=LEGACY):
    return [episode(seed0 + i, split, hardening=hardening) for i in range(n)]


def splits(n_train=24, n_val=120, train_lengths=(2, 4, 6), pool_n=900, test_n=1500):
    """`run_stage_b.py` / `final_eval.py`'s indices, verbatim."""
    pool = dataset(pool_n, seed0=0, split="train", hardening=LEGACY)
    short = [e for e in pool if e["length"] in train_lengths]
    test_pool = dataset(test_n, seed0=100000, split="test", hardening=LEGACY)
    current = dataset(400, seed0=100000, split="test", hardening=CURRENT)
    return {"train": short[:n_train],
            "val": short[n_train:n_train + n_val],
            "test": [e for e in test_pool if e["length"] not in train_lengths],
            "over_budget_train": short,
            "current_stream_test": current}
