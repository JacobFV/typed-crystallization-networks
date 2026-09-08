"""``knowledge_update`` — assert / correct / retract / hedge.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS
from ..generators.semantics import _shuffled


def gen_knowledge_update(rng: random.Random, ctx):
    """Assertion, correction, retraction and uncertainty are different operations.

    All four arrive as utterances about a knowledge base and only two of them
    overwrite it: a retraction takes a slot back to ``unknown`` and a hedge
    leaves it exactly as it was. The query sometimes names an untouched slot, so
    "return whatever the utterance mentioned" is wrong a third of the time.
    """
    ids = _shuffled(rng, [f"k{i}" for i in range(1, ctx.at(3, 9, default=3) + 1)])
    kb = {o: (rng.choice(COLORS) if rng.random() < 0.85 else "unknown") for o in ids}
    before = dict(kb)
    tgt = rng.choice(ids)
    op = rng.choice(["assert", "correct", "retract", "hedge"])
    new = rng.choice([c for c in COLORS if c != kb[tgt]])

    if op == "assert":
        utter = Pred("assert", Ident(tgt), Ident(new))
        kb[tgt] = new
    elif op == "correct":
        old = kb[tgt] if kb[tgt] != "unknown" else rng.choice(COLORS)
        utter = Pred("correct", Ident(tgt), Ident(old), Ident(new))
        kb[tgt] = new
    elif op == "retract":
        utter = Pred("retract", Ident(tgt))
        kb[tgt] = "unknown"
    else:
        utter = Pred("hedge", Ident(tgt), Ident(new))

    q = rng.choice(ids)
    obs = Rec(kb=Lst([Pred("color", Ident(o), Ident(before[o])) for o in ids]),
              said=utter,
              operations=Lst([Ident(x) for x in ("assert", "correct", "retract", "hedge")]),
              query=Pred("color_of", Ident(q)))
    vocab = _shuffled(rng, COLORS + ["unknown"])
    return obs, vocab, kb[q], {"op": op, "target": tgt, "queried": q, "after": dict(kb)}


class KnowledgeUpdate(Lesson):
    """Assert / correct / retract / hedge."""

    id = "knowledge_update"
    level = 34
    tags = ("pragmatics", "language-as-action")
    teaches = "assert / correct / retract / hedge"
    capabilities = ('ontology_learning', 'belief_modeling')
    axes = {'discourse_horizon': 3, 'reasoning_depth': 3, 'world_complexity': 2}

    generate = staticmethod(gen_knowledge_update)
