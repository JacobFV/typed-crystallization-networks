"""What the class costs to store against what the artifacts cost, and the
invariant content in bits -- §55's `size.py`, on the visual pipeline.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / 'research' / 'visual-ladder'),
           str(ROOT / 'research' / 'class-identity'), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run as RUN                                 # noqa: E402
import schema as S                                # noqa: E402
from tcn.operators import Registry                # noqa: E402

OUT = HERE / "out"
ALL_WIDTHS = (16, 24, 32, 40, 48)


def main():
    construct = RUN.load("construct")
    vectors = construct["vectors"]
    classes = json.loads((OUT / "library" / "classes.json").read_text())
    record_bytes = len(json.dumps(classes, sort_keys=True).encode())

    per_width, digests = {}, {}
    for width in ALL_WIDTHS:
        registry = Registry()
        p0 = S.RIGHT.build_s0(registry, width, width)
        module = S.module_of(p0, S.full(p0, vectors["s0"]), registry)
        p1 = S.RIGHT.build_s1(registry, width, width, module)
        p2 = S.RIGHT.build_s2(registry, width, width, module)
        row = {}
        for stage, program in (("s0", p0), ("s1", p1), ("s2", p2)):
            hard = S.freeze(program, S.full(program, vectors[stage]), registry)
            row[stage] = {"digest": hard.digest, "nodes": len(hard.nodes),
                          "serialized_bytes": len(json.dumps(hard.to_dict(),
                                                             sort_keys=True).encode()),
                          "description_bits": hard.description_bits(registry),
                          "execution_cost": hard.execution_cost(registry)}
            digests.setdefault(stage, set()).add(hard.digest)
        per_width[width] = row

    spaces = {s: RUN.load("enum_32")["stages"][s]["space_size"] for s in S.STAGES}
    out = {"class_store_bytes": record_bytes,
           "class_store_bytes_on_disk": (OUT / "library" / "classes.json").stat().st_size,
           "widths": list(ALL_WIDTHS),
           "per_width": per_width,
           "artifact_bytes_total": sum(r[s]["serialized_bytes"]
                                       for r in per_width.values() for s in S.STAGES),
           "distinct_digests_per_stage": {s: len(v) for s, v in digests.items()},
           "space_sizes": spaces,
           "invariant_bits_per_stage": {s: math.log2(n) for s, n in spaces.items()},
           "invariant_bits_total": sum(math.log2(n) for n in spaces.values())}
    print(json.dumps({k: v for k, v in out.items() if k != "per_width"}, indent=1))
    RUN.dump("size", out)


if __name__ == "__main__":
    main()
