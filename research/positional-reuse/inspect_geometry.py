"""What the geometry generator actually emits at a downscaled resolution.

Checks the one thing the demonstration depends on: whether the per-pixel
foreground label in the `object_ids` probe is exactly determined by the pixel
in the `pixels` observation. If it is not exact, say so and report the rate.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tcn.generation import Host

RES = 8

if __name__ == "__main__":
    disagree = 0
    total = 0
    fg = 0
    for seed in range(12):
        h = Host.create("geometry", seed=seed, configuration={"resolution": RES, "objects": 3})
        rec = h.records[0]
        px = rec.observations["pixels"].value
        ids = rec.probes["object_ids"]
        depth = rec.probes["depth"]
        if seed == 0:
            print("pixels type width:", px.type.width)
            print("pixels fields:", [t.kind for t in px.type.items], "byte field len:",
                  len(px.type.items[3].items))
            print("byte element type:", px.type.items[3].items[0].to_dict())
            print("probe object_ids width:", ids.type.width, "depth width:", depth.type.width)
            print("visible observation keys:", sorted(rec.actor_view().observations))
            print("probe keys:", sorted(rec.probes))
        h_, w_, c_, data = px.decoded
        labels = ids.decoded
        assert (h_, w_, c_) == (RES, RES, 3), (h_, w_, c_)
        assert len(labels) == RES * RES
        for i, lab in enumerate(labels):
            r, g, b = data[3 * i], data[3 * i + 1], data[3 * i + 2]
            is_background_colour = (r, g, b) == (24, 30, 43)
            label_background = lab < 0
            total += 1
            fg += 0 if label_background else 1
            if is_background_colour != label_background:
                disagree += 1
    print(f"pixels checked {total}, foreground {fg} ({100*fg/total:.1f}%), "
          f"colour/label disagreements {disagree}")
