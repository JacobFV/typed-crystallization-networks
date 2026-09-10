"""Locate the §58 motif in the eight frozen artifacts, exactly and reproducibly.

Nothing here searches or trains. The artifacts come from
`research/cross-domain/artifacts.py` (imported, not copied); the fragment rule
comes from `research/cross-domain/frag.py`, which §58 gated bit-identical to
`research/earned-abstraction/mine.py` on 862 checks. This module only *filters*
that fragment set by operator shape and reports what it finds.

Two shapes are looked for, and they are kept apart throughout:

  M2   eq(index(x0, x1), x2)                 -- 2 nodes
  M3   eq(index(x2, add(x0, x1)), x3)        -- 3 nodes

§58's prose names M3 and says it recurs in all three artifacts. §58's own §6.4
table says M3 is language + visual and M2 is all three. `run_step1.py` decides
which reading the raw programs support.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "cross-domain"))

import artifacts as ART            # noqa: E402
import frag                        # noqa: E402
from identity import tag, tshort   # noqa: E402


def body(canon):
    """(operator name, node name, sources) per node, in program order."""
    return tuple((n.candidates[0].operator.name, n.name, tuple(n.candidates[0].sources))
                 for n in canon.nodes)


def shape(canon):
    """Operator names and edges, types erased -- §58's relation T, as a tuple."""
    return body(canon)


def spell(canon):
    """A readable spelling of the canonical fragment."""
    text = {}
    for op, name, srcs in body(canon):
        text[name] = f"{op}({', '.join(text.get(s, s) for s in srcs)})"
    return text[canon.nodes[-1].name]


# The two shapes, written in the canonical naming `frag.canonicalize` assigns.
M2_SHAPE = (("index", "n0", ("x0", "x1")), ("eq", "n1", ("n0", "x2")))
M3_SHAPE = (("add", "n0", ("x0", "x1")), ("index", "n1", ("x2", "n0")),
            ("eq", "n2", ("n1", "x3")))
# The third shape the computer artifact actually carries at three nodes.
M3I_SHAPE = (("identity", "n0", ("x0",)), ("index", "n1", ("x1", "n0")),
             ("eq", "n2", ("n1", "x2")))

SHAPES = {"M2": M2_SHAPE, "M3": M3_SHAPE, "M3_identity": M3I_SHAPE}


def find(max_nodes=5, only=None):
    """Every occurrence of every shape in every artifact.

    `only` restricts the search to named artifacts. The full sweep is the
    reported inventory; `only` exists because `V_rect` is 586 nodes and mining it
    costs ~45 s, which is wasted when a caller needs one known fragment back.

    Returns a list of dicts, one per (artifact, shape, occurrence).
    """
    arts, _ = ART.all_artifacts()
    rows = []
    for aid, (program, registry) in arts.items():
        if only is not None and aid not in only:
            continue
        for root, S, canon, holes in frag.fragments(program, max_nodes=max_nodes,
                                                    registry=registry):
            sh = shape(canon)
            for label, want in SHAPES.items():
                if sh != want:
                    continue
                rows.append({
                    "artifact": aid,
                    "domain": ART.DOMAIN[aid],
                    "shape": label,
                    "root": root,
                    "nodes_in_artifact": sorted(S),
                    "holes_in_artifact": list(holes),
                    "digest": canon.digest,
                    "spell": spell(canon),
                    # `label#sha256-of-the-full-to_dict()` -- §58's own compact
                    # exact tag. A 4,096-field product serialises to ~330 kB, so
                    # storing full type dicts would put this report at ~5 MB for
                    # no gain: the sha distinguishes any two distinct types
                    # exactly, and every identity below is computed in memory
                    # from the real `Type` objects, never from the tag.
                    "input_type_tags": [tag(t) for _, t in canon.inputs],
                    "input_labels": [tshort(t.to_dict()) for _, t in canon.inputs],
                    "arity": len(canon.inputs),
                    "output_type_tag": tag(canon.nodes[-1].output),
                    "output_label": tshort(canon.nodes[-1].output.to_dict()),
                    "_canon": canon,
                    "_registry": registry,
                    "_program": program,
                })
    return arts, rows
