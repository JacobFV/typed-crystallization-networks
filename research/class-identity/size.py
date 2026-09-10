"""What the class record costs, against what the artifacts it replaces cost."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "depth-encoding"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import run as DEP                                                  # noqa: E402
import classes as C                                                # noqa: E402
from tcn.search import frozen_selection                            # noqa: E402

OUT = HERE / "out"
WIDTHS = (1, 2, 5, 7, 12)


def main():
    store = C.ClassStore(OUT / "library")
    record = store.records[0]
    rows = {}
    for d in WIDTHS:
        _, program, registry = DEP.build("interpreter", d)
        full = {n.name: 0 for n in program.nodes}
        full.update(record.selections)
        exact = frozen_selection(program, full, registry)
        rows[str(d)] = {"program_fields": 3 * d, "nodes": len(exact.nodes),
                        "description_bits": exact.description_bits(registry),
                        "execution_cost": exact.execution_cost(registry),
                        "digest": exact.digest,
                        "serialized_json_bytes": len(json.dumps(exact.to_dict()))}
    free = dict(record.schema["free_nodes"])
    report = {
        "per_width": rows,
        "distinct_digests": len({r["digest"] for r in rows.values()}),
        "selection_vector_bits": sum(math.log2(c) for c in free.values()),
        "selection_vector_entries": len(free),
        "class_record_json_bytes": len(json.dumps(record.to_dict())),
        "classes_json_bytes": len((OUT / "library" / "classes.json").read_bytes()),
        "artifact_json_bytes_total": sum(r["serialized_json_bytes"] for r in rows.values()),
    }
    print(json.dumps(report, indent=2))
    (OUT / "size.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
