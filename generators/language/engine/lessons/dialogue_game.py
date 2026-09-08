"""``dialogue_game`` — multi-turn reference repair and shared state.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random
from typing import Mapping

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.base import COLORS, SHAPES
from ..generators.social import SIZES, _obj_facts, _shuffled


def gen_dialogue_game(rng: random.Random, ctx):
    """A two-party clarification dialogue, run to agreement, then asked about.

    ``a`` wants one object and describes it one attribute at a time; ``b`` reports
    how many objects still match and the pair keeps going until the description
    is unique. Crucially ``a`` opens with a *wrong* attribute value and repairs it
    mid-dialogue, so the referent is fixed by the dialogue **state** rather than
    by the bag of words in the transcript: an agent that intersects every
    mentioned attribute lands on a different object, or on none.
    """
    n_obj = ctx.at(4, 9, default=4)
    for _ in range(400):
        ids = _shuffled(rng, [f"o{i}" for i in range(n_obj)])
        objs = [{"id": ids[i], "color": rng.choice(COLORS), "shape": rng.choice(SHAPES),
                 "size": rng.choice(SIZES)} for i in range(n_obj)]
        tgt = objs[rng.randrange(n_obj)]
        order = _shuffled(rng, ["color", "shape", "size"])
        constraints: dict[str, str] = {}
        used: list[str] = []
        for attr in order:
            constraints[attr] = tgt[attr]
            used.append(attr)
            if sum(1 for o in objs if all(o[k] == v for k, v in constraints.items())) == 1:
                break
        else:
            continue                                  # target not uniquely describable
        if len(used) < 2:                             # a one-turn exchange is not a dialogue
            continue
        first = used[0]
        domain = {"color": COLORS, "shape": SHAPES, "size": SIZES}[first]
        wrong_pool = [v for v in domain if v != tgt[first]]
        wrong = rng.choice(wrong_pool)

        def _count(cs: Mapping[str, str]) -> int:
            return sum(1 for o in objs if all(o[k] == v for k, v in cs.items()))

        turns: list[Term] = []
        t = 0

        def _say(speaker: str, act: Term) -> None:
            nonlocal t
            turns.append(Pred("turn", Num(t), Ident(speaker), act))
            t += 1

        _say("a", Pred("request", Ident(first), Ident(wrong)))
        _say("b", Pred("matches", Num(_count({first: wrong}))))
        _say("a", Pred("revise", Ident(first), Ident(wrong), Ident(tgt[first])))
        _say("b", Pred("matches", Num(_count({first: tgt[first]}))))
        running = {first: tgt[first]}
        for attr in used[1:]:
            running[attr] = tgt[attr]
            _say("a", Pred("request", Ident(attr), Ident(tgt[attr])))
            _say("b", Pred("matches", Num(_count(running))))
        _say("b", Pred("confirm"))

        obs = Rec(scene=_obj_facts(objs), transcript=Lst(turns), query=Ident("agreed_object"))
        return (obs, _shuffled(rng, ids), tgt["id"],
                {"constraints": dict(constraints), "retracted": {first: wrong},
                 "turns": len(turns), "target": dict(tgt)})
    raise RuntimeError("dialogue_game: no admissible world")


class DialogueGame(Lesson):
    """Multi-turn reference repair and shared state."""

    id = "dialogue_game"
    level = 48
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "multi-turn reference repair and shared state"
    capabilities = ('multi_agent_coordination', 'belief_modeling')
    axes = {'discourse_horizon': 4, 'compositional_depth': 3, 'world_complexity': 2, 'ambiguity': 2}

    generate = staticmethod(gen_dialogue_game)
