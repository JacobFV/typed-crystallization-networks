"""Is `enumerate_fit`'s lexicographic pick safe when the supervision is loose?

On the rung-1 screen the answer turns out to be unique as a *function*, so the
tie-break never bites there.  That is a property of that screen, not of the
method, and the discrete-perception track measured the opposite on `geometry`
(2,464-2,608 of 32,000 conforming, ~5% wrong on fresh episodes, and the returned
program wrong at both resolutions it tried).  So the question is asked here on
the settings this generator's own dial says are loose: fewer widgets, fewer
colours on screen, a colour map fixed per kind.

For each setting: how many programs conform, how many distinct Boolean functions
they denote, how many of them are wrong on held-out episodes, whether the
lexicographically first is one of the wrong ones, and whether requiring exactness
at every position of a validation split repairs it.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import all_conforming, dump, episode, exact_error, report
from rung1_edges import (SCREEN, examples, induced_function, narrow_offsets, scaffold, signals)
from tcn.operators import Registry
from tcn.search import space_size

SETTINGS = {
    "rung-1 screen (12 widgets, nesting 4, min_size 4)": SCREEN,
    "2 widgets": dict(SCREEN, widgets=2, nesting=1),
    "3 widgets, nesting 1": dict(SCREEN, widgets=3, nesting=1),
    "colour_mode=kind": dict(SCREEN, colour_mode="kind"),
    "palette 4": dict(SCREEN, palette=4),
    "palette_levels 32": dict(SCREEN, palette_levels=32),
}


def main():
    registry = Registry()
    tolerance = 1e-6
    result = {}
    for name, configuration in SETTINGS.items():
        probe = episode(0, "train", **configuration)
        offsets = narrow_offsets(probe["width"])
        program = scaffold(registry, probe, offsets, False)
        train = examples(range(8), "train", 24, seed=1, **configuration)
        validation = examples(range(50, 54), "validation", 24, seed=2, **configuration)
        held = examples(range(100, 108), "test", 24, seed=3, **configuration)
        survey = all_conforming(program, train, signals(), registry, tolerance)
        conforming = survey.pop("conforming")
        errors = [exact_error(program, held, signals(), registry, s) for s in conforming]
        wrong = [e for e in errors if e > tolerance]
        survivors = [s for s in conforming
                     if exact_error(program, validation, signals(), registry, s) <= tolerance]
        survivor_errors = [exact_error(program, held, signals(), registry, s) for s in survivors]
        row = {"configuration": configuration, "space_size": space_size(program),
               "conforming": len(conforming),
               "distinct_functions": len({induced_function(program, s, offsets)
                                          for s in conforming}),
               "wrong_on_held_out": len(wrong),
               "lexicographic_pick_held_error": errors[0] if errors else None,
               "lexicographic_pick_wrong": bool(errors) and errors[0] > tolerance,
               "validation_survivors": len(survivors),
               "survivor_functions": len({induced_function(program, s, offsets)
                                          for s in survivors}),
               "survivor_max_held_error": max(survivor_errors) if survivor_errors else None,
               "seconds": survey["seconds"]}
        result[name] = row
        report(name, f"conforming {row['conforming']:3d} functions {row['distinct_functions']:2d} "
                     f"wrong-on-held-out {row['wrong_on_held_out']:3d} "
                     f"lex-pick-wrong {row['lexicographic_pick_wrong']} "
                     f"-> survivors {row['validation_survivors']} "
                     f"functions {row['survivor_functions']} "
                     f"max held err {row['survivor_max_held_error']}")
    dump("tiebreak", result)


if __name__ == "__main__":
    main()
