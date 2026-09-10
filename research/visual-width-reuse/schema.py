"""The width-parametric schema of the shipped visual parse, and two wrong ones.

Nothing here re-implements a scaffold.  The three stage constructors are the
ones §32/§33 shipped -- `rung3_widgets.same_scaffold`,
`rung3_root.corner_scaffold_masked`, `rung3_root.rect_scaffold_clamped` -- and a
`Schema` is exactly the pair of *derivations* that turn a raster resolution into
their remaining arguments:

    offsets(width)  ->  the five candidate spatial steps
    span(W, H)      ->  how many prefix-conjunction terms `rect_scaffold` unrolls

`RIGHT` derives both from the width.  `FROZEN_OFFSETS` and `FROZEN_SPAN` freeze
one of them at the value it takes at `resolution=32`, which is the arm F payload:
at 32 they are bit-identical to `RIGHT`, and everywhere else they are wrong.
That is the mistake a single-width `unique` certificate cannot catch (§53 arm F,
enforced in §55, re-confirmed in §60).
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Callable

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / 'research' / 'visual-ladder'),
           str(ROOT / 'research' / 'class-identity')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rung3_widgets as R                       # noqa: E402
import rung3_root as RR                         # noqa: E402
from tcn.operators import Registry              # noqa: E402
from tcn.search import frozen_selection         # noqa: E402

REFERENCE_RESOLUTION = 32
STAGES = ("s0", "s1", "s2")


@dataclass(frozen=True)
class Schema:
    name: str
    offsets: Callable[[int], tuple]
    span: Callable[[int, int], int]
    note: str = ""

    # -- stage constructors, in the order the pipeline crystallizes them ----
    def build_s0(self, registry, W, H):
        return R.same_scaffold(registry, W, H)

    def build_s1(self, registry, W, H, same_module):
        return RR.corner_scaffold_masked(registry, W, H, same_module, self.offsets(W))

    def build_s2(self, registry, W, H, same_module):
        return RR.rect_scaffold_clamped(registry, W, H, same_module, self.offsets(W),
                                        span=self.span(W, H))

    def qualified(self, stage):
        return f"research.visual-width-reuse.schema:{self.name}.{stage}"


RIGHT = Schema("right", R.offset_pool, lambda W, H: max(W, H),
               "offsets (6, 3W+3, 3, 3W, 9) and span max(W, H), both derived from the width")
FROZEN_OFFSETS = Schema("frozen_offsets", lambda W: (6, 99, 3, 96, 9),
                        lambda W, H: max(W, H),
                        "arm F1: the offset pool frozen at its resolution-32 values")
FROZEN_SPAN = Schema("frozen_span", R.offset_pool, lambda W, H: REFERENCE_RESOLUTION,
                     "arm F2: span frozen at 32, the value that is correct at resolution 32")

SCHEMAS = {s.name: s for s in (RIGHT, FROZEN_OFFSETS, FROZEN_SPAN)}


# ---------------------------------------------------------------- building

def freeze(program, full_selections, registry):
    """The exact callable one selection names -- §55's `freeze` hook."""
    return frozen_selection(program, full_selections, registry)


def module_of(program, full_selections, registry):
    """Register the hardened stage as a callable module.

    `program.harden(...)` rather than `frozen_selection(...)`, because that is
    literally what `rung3_root.main` does; a different call here would make
    every module name -- and so every downstream digest -- incomparable with
    §33's.
    """
    return registry.register_module(program.harden(full_selections))


def free_nodes(program):
    return {n.name: len(n.candidates) for n in program.nodes if len(n.candidates) > 1}


def full(program, selections):
    out = {n.name: 0 for n in program.nodes}
    out.update(selections)
    return out


def stage_builder(schema, stage, vectors):
    """`builder(width) -> (names, program, registry)` for §55's `ClassStore`.

    `vectors` supplies the *already certified* selection vectors of the earlier
    stages, because S1' and S2' both call the hardened S0 module and a stage
    cannot be rebuilt without its predecessor.
    """
    def builder(width):
        registry = Registry()
        W = H = width
        p0 = schema.build_s0(registry, W, H)
        if stage == "s0":
            return tuple(n.name for n in p0.nodes), p0, registry
        module = module_of(p0, full(p0, vectors["s0"]), registry)
        p = (schema.build_s1(registry, W, H, module) if stage == "s1"
             else schema.build_s2(registry, W, H, module))
        return tuple(n.name for n in p.nodes), p, registry
    return builder
