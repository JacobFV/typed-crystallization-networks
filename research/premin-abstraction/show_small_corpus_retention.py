"""Value retention for the two one-program-per-task corpora, C-min and C-plus1-one."""
from __future__ import annotations

import json
from pathlib import Path

import retention

OUT = Path(__file__).parent / "out"


def main():
    d = json.loads((OUT / "corpora.json").read_text())
    sets = {
        "C-min": [[tuple(g) for g in v] for v in d["c_min"].values()],
        "C-plus1-one": [[tuple(g) for g in t["bands"]["plus1"]["first_in_order"]]
                        for t in d["tasks"]],
    }
    for name, gate_lists in sets.items():
        out = []
        for label, fn in (("MAJ3", retention.maj3), ("M", retention.frag_M),
                          ("M'", retention.frag_Mprime)):
            hit, total, per_triple = retention.value_retention(gate_lists, fn)
            out.append(f"{label} {hit}/{total} {per_triple}")
        print(name, "|", " | ".join(out))


if __name__ == "__main__":
    main()
