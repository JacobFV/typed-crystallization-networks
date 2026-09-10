"""Why one widget is missed at resolution 40, seed 205 -- named, not glossed.

The parse at held-out resolution 40 recovers 119 of 120 rectangles; the missing
one is `[3, 11, 6, 9]`.  This asks whether that is a *width* failure -- an extent
the scaffold cannot represent -- or §32's known colour-collision residual, which
is a property of `palette 32` and has nothing to do with reuse.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / 'research' / 'visual-ladder'), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import FLAT, colour_at, episode         # noqa: E402


def main():
    rows = []
    for resolution, seed, rect in ((40, 205, (3, 11, 6, 9)),):
        cfg = dict(FLAT)
        cfg["resolution"] = resolution
        ep = episode(seed, "test", **cfg)
        x, y, w, h = rect
        own = colour_at(ep, x, y)
        left = colour_at(ep, x - 1, y) if x >= 1 else None
        up = colour_at(ep, x, y - 1) if y >= 1 else None
        widget = [d for d in ep["probes"]["hierarchy"] if tuple(d["rect"]) == rect]
        parent = None
        if widget and widget[0]["parent"] != 255:
            parent = [d for d in ep["probes"]["hierarchy"]
                      if d["id"] == widget[0]["parent"]]
        rows.append({
            "resolution": resolution, "seed": seed, "rect": list(rect),
            "own_colour": list(own),
            "left_neighbour_colour": list(left) if left else None,
            "upper_neighbour_colour": list(up) if up else None,
            "left_differs": left != own, "upper_differs": up != own,
            "corner_predicate_would_fire": (left != own) and (up != own),
            "parent_rect": list(parent[0]["rect"]) if parent else None,
            "extent_within_span": max(w, h) <= resolution,
            "distinct_colours_on_screen": len({colour_at(ep, i, j)
                                               for j in range(ep["height"])
                                               for i in range(ep["width"])}),
            "widgets": len(ep["probes"]["hierarchy"])})
        print(json.dumps(rows[-1]))
    (HERE / "out" / "collision.json").write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
