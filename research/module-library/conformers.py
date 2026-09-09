"""Which stage-3 programs conform, and does a distractor ever survive?

`conforming: 3` in arm A could mean three spellings of one readout over the
right module, or it could mean a distractor also fits. That distinction decides
whether the inherited module was *selected* or merely available, so it is
enumerated explicitly rather than inferred.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "discrete-perception"))

import chain as C  # noqa: E402
from tcn.library import Library  # noqa: E402
from tcn.search import evaluate  # noqa: E402

R, TILE, TOL = 8, 2, 1e-6


def readout(index):
    return f"{C.COMPARISONS[index // len(C.THRESHOLDS)]}(total, {C.THRESHOLDS[index % len(C.THRESHOLDS)]})"


def main():
    library = Library(HERE / "library")
    registry, aliases = library.load(["perception.foreground", "perception.edge"],
                                     policy="revalidate")
    fg, edge = aliases["perception.foreground"], aliases["perception.edge"]
    keep = registry.register_module(C.keep_flag_module(registry))
    wrappers = ["inherited stage-2 edge", "distractor: offset 6, xor", "distractor: offset 3, xnor"]
    names = [registry.register_module(C.wrapper_over_edge(registry, R, edge)),
             registry.register_module(C.wrapper_over_foreground(registry, R, fg, C.offsets(R)[1], 6)),
             registry.register_module(C.wrapper_over_foreground(registry, R, fg, C.EDGE_STEP, 9))]
    scaffold = C.region_scaffold(registry, R, names, keep, TILE)
    signals = C.region_signals()
    train = C.region_examples(tuple(range(6)), R, "train", TILE, seed=5)
    validation = C.region_examples(tuple(range(300, 304)), R, "validation", TILE, seed=7)
    held = C.region_examples(tuple(range(100, 106)), R, "test", TILE, seed=6)

    rows = []
    for m in range(len(names)):
        for k in range(len(C.COMPARISONS) * len(C.THRESHOLDS)):
            selections = {"held": 0, "records": 0, "mapped": m, "kept": 0, "total": 0, "region": k}
            error = evaluate(scaffold, selections, train, signals, registry, TOL)
            if error is not None and error <= TOL:
                rows.append({"wrapper": wrappers[m], "wrapper_index": m, "readout": readout(k),
                             "readout_index": k,
                             "validation_exact": evaluate(scaffold, selections, validation,
                                                          signals, registry, TOL) == 0.,
                             "held_out_exact": evaluate(scaffold, selections, held, signals,
                                                        registry, TOL) == 0.})
    out = {"train_examples": len(train), "validation_examples": len(validation),
           "held_examples": len(held), "space": 3 * 20, "conforming": len(rows),
           "distinct_wrappers_conforming": len({r["wrapper_index"] for r in rows}), "rows": rows}
    (HERE / "out").mkdir(parents=True, exist_ok=True)
    (HERE / "out" / "conformers.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
