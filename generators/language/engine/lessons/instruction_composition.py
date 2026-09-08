"""``instruction_composition`` — sequencing, branching, nested conditions.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Rec, Term
from ..lesson import Lesson
from ..generators.base import COLORS
from ..generators.semantics import _act_sym, _apply, _instruction_query, _rand_action, _world, _world_facts


def gen_instruction_composition(rng: random.Random, ctx):
    """Sequencing plus branching: ``if b is blue then ... else ...``, ``unless``.

    Conditions are evaluated against the state *as it stands at that step*, so a
    later branch can depend on an earlier step's effect; the interpreter that
    produces the label is the definition of the language.
    """
    ids, st = _world(rng)
    init = {o: dict(v) for o, v in st.items()}
    kinds = ["plain", rng.choice(["if", "unless"])]
    if rng.random() < 0.5:
        kinds.append(rng.choice(["plain", "if", "unless"]))
    for _ in range(ctx.at(0, 4, default=0)):      # a longer program to interpret in order
        kinds.append(rng.choice(["plain", "if", "unless"]))
    rng.shuffle(kinds)

    stmts: list[Term] = []
    trace: list[str] = []
    for i, kind in enumerate(kinds):
        if kind == "plain":
            act = _rand_action(rng, ids)
            stmts.append(_act_sym("step", i, act))
            _apply(st, act)
            trace.append(f"{i}:do")
            continue
        cobj = rng.choice(ids)
        ccolor = st[cobj]["color"] if rng.random() < 0.5 else rng.choice(COLORS)
        holds = st[cobj]["color"] == ccolor
        if kind == "if":
            then_act, else_act = _rand_action(rng, ids), _rand_action(rng, ids)
            stmts.append(_act_sym("if_step", i, then_act, Ident(cobj), Ident(ccolor)))
            stmts.append(_act_sym("else_step", i, else_act))
            _apply(st, then_act if holds else else_act)
            trace.append(f"{i}:if={holds}")
        else:
            act = _rand_action(rng, ids)
            stmts.append(_act_sym("unless_step", i, act, Ident(cobj), Ident(ccolor)))
            if not holds:
                _apply(st, act)
            trace.append(f"{i}:unless={holds}")

    query, vocab, answer, (slot, q) = _instruction_query(rng, ids, st)
    obs = Rec(world=_world_facts(ids, init), program=Lst(stmts), query=query)
    return obs, vocab, answer, {"trace": trace, "queried": [q, slot],
                                "final": {o: dict(v) for o, v in st.items()}}


class InstructionComposition(Lesson):
    """Sequencing, branching, nested conditions."""

    id = "instruction_composition"
    level = 21
    tags = ("pragmatics", "language-as-action")
    teaches = "sequencing, branching, nested conditions"
    capabilities = ('planning', 'program_synthesis')
    axes = {'compositional_depth': 4, 'reasoning_depth': 3, 'world_complexity': 3}

    generate = staticmethod(gen_instruction_composition)
