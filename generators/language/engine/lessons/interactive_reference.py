"""``interactive_reference`` — clarify or act: the value of a question.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Tok, Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, SHAPES
from ..generators.semantics import POSITIONS, SIZES, _shuffled


def gen_interactive_reference(rng: random.Random, ctx):
    """Ask, or act? and if asking, ask about *what*.

    The instruction names one attribute value. If it already denotes uniquely
    the right move is to act; otherwise exactly one other attribute separates
    all the candidates, and every other question would come back ambiguous —
    so the value of each possible question is computed, not guessed.
    """
    attrs = {"color": COLORS, "shape": SHAPES, "size": SIZES, "position": POSITIONS}
    keys = sorted(attrs)
    n_obj = ctx.at(4, 8, default=4)              # objects in the scene
    for _ in range(200):
        label = rng.choice(keys + ["act_now"])
        ids = _shuffled(rng, [f"x{i + 1}" for i in range(n_obj)])
        objs = {o: {k: rng.choice(v) for k, v in attrs.items()} for o in ids}

        if label == "act_now":
            named = rng.choice(keys)
            val = rng.choice(attrs[named])
            objs[ids[0]][named] = val
            for o in ids[1:]:
                objs[o][named] = rng.choice([v for v in attrs[named] if v != val])
        else:
            named = rng.choice([k for k in keys if k != label])
            val = rng.choice(attrs[named])
            k = rng.randint(*ctx.span((2, 3), (3, 4)))    # candidates the instruction leaves open
            cands, rest = ids[:k], ids[k:]
            for o in cands:
                objs[o][named] = val
            for o in rest:
                objs[o][named] = rng.choice([v for v in attrs[named] if v != val])
            sep = rng.sample(attrs[label], k)            # the resolving attribute
            for o, v in zip(cands, sep):
                objs[o][label] = v
            for k2 in keys:                              # every other attribute collides
                if k2 in (label, named):
                    continue
                v = rng.choice(attrs[k2])
                for o in cands:
                    objs[o][k2] = v

        cands = [o for o in ids if objs[o][named] == val]
        useful = [k for k in keys if k != named
                  and len({objs[o][k] for o in cands}) == len(cands)]
        got = "act_now" if len(cands) == 1 else (useful[0] if len(useful) == 1 else None)
        if got == label:
            break
    else:                                                  # pragma: no cover
        label = got or "act_now"

    obs = Rec(scene=Lst([Pred("obj", Ident(o), Ident(objs[o]["color"]), Ident(objs[o]["shape"]),
                              Ident(objs[o]["size"]), Ident(objs[o]["position"]))
                         for o in ids]),
              schema=Lst([Pred("attribute", Num(i), Ident(k)) for i, k in enumerate(keys)]),
              instruction=Pred("bring_the", Ident(named), Ident(val)),
              # a question is a word, not a name: as an `Ident` it printed
              # "question" in every language that has one
              cost=Pred("cost", Tok("question"), Num(1)),
              query=Ident("what_should_i_ask"))
    return obs, _shuffled(rng, keys + ["act_now"]), label, {"named": named, "value": val,
                                                            "candidates": cands}


class InteractiveReference(Lesson):
    """Clarify or act: the value of a question."""

    id = "interactive_reference"
    level = 38
    tags = ("pragmatics", "language-as-action")
    teaches = "clarify or act: the value of a question"
    capabilities = ('metareasoning', 'multi_agent_coordination')
    axes = {'ambiguity': 4, 'reasoning_depth': 3, 'world_complexity': 3}
    answers = ['color', 'shape', 'size', 'position', 'act_now']

    generate = staticmethod(gen_interactive_reference)
