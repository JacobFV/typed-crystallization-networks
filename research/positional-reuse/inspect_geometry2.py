"""Find a downscaled geometry configuration whose foreground fraction is not degenerate.

The camera step is an ordinary generator action, so the episode stays replayable
and nothing is preprocessed; it just brings the objects closer.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tcn.generation import Action, Host
from tcn.generation import SCALAR
from tcn.types import Value, product

VEC3 = product(SCALAR, SCALAR, SCALAR)


def approach(delta):
    return Action("camera", arguments=(("delta", Value.of(VEC3, delta)),))


def measure(res, objects, delta, seeds=range(8)):
    total = fg = disagree = 0
    for seed in seeds:
        h = Host.create("geometry", seed=seed,
                        configuration={"resolution": res, "objects": objects, "horizon": 4})
        rec = h.step([approach(delta)]) if delta else h.records[0]
        data = rec.observations["pixels"].value.decoded[3]
        labels = rec.probes["object_ids"].decoded
        for i, lab in enumerate(labels):
            px = (data[3 * i], data[3 * i + 1], data[3 * i + 2])
            total += 1
            fg += lab >= 0
            disagree += ((px == (24, 30, 43)) != (lab < 0))
    return total, fg, disagree


if __name__ == "__main__":
    for res in (8, 12, 16):
        for objects in (3, 6):
            for delta in ((0., 0., 0.), (-2., -2., -3.5), (-2.4, -2.4, -4.2)):
                t, f, d = measure(res, objects, delta if any(delta) else None)
                print(f"res={res:3d} objects={objects} delta={delta} "
                      f"foreground {100*f/t:5.1f}%  disagreements {d}/{t}")
