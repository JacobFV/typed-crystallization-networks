"""``planning_language`` — goal to executable plan: the first action of a shortest plan.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.causal import _plan_domain, _plan_fallback


def gen_planning_language(rng: random.Random, ctx):
    """Answer the first action of a shortest plan. The episode is only emitted
    when breadth-first search says that first action is unique, so 'correct' is
    well defined; the plan length travels in ``hidden``."""
    min_len = ctx.at(2, 6, default=2)          # shortest-plan length the episode must reach
    dom = None
    for _ in range(300):
        dom = _plan_domain(rng)
        if dom is not None and dom["length"] >= min_len:
            break
    if dom is None:                                            # pragma: no cover
        dom = _plan_fallback(rng)

    def name(a: tuple[str, str]) -> str:
        return f"{a[0]}_{a[1]}"

    facts: list[Term] = [Pred("road", Ident(a), Ident(b)) for a, b in dom["edges"]]
    if dom["locked"]:
        a, b = dom["locked"]
        facts.append(Pred("locked", Ident(a), Ident(b)))
        facts.append(Pred("unlocked_by", Ident(a), Ident(b), Ident(dom["key"])))
    facts.append(Pred("key_at", Ident(dom["key"]), Ident(dom["key_at"])))
    rng.shuffle(facts)
    vocab = [name(a) for a in dom["actions"]]
    rng.shuffle(vocab)
    obs = Rec(world=Lst(facts),
              state=Lst([Pred("at", Ident("agent"), Ident(dom["start"])),
                         Pred("holding", Ident("agent"), Ident("nothing"))]),
              goal=Pred("at", Ident("agent"), Ident(dom["goal"])),
              actions=Lst([Ident(a) for a in vocab]),
              query=Ident("first_action_of_shortest_plan"))
    hidden = {"plan_length": dom["length"], "goal": dom["goal"], "start": dom["start"],
              "locked": list(dom["locked"]) if dom["locked"] else None,
              "answer": name(dom["best"])}
    return obs, vocab, name(dom["best"]), hidden


class PlanningLanguage(Lesson):
    """Goal to executable plan: the first action of a shortest plan."""

    id = "planning_language"
    level = 44
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "goal to executable plan: the first action of a shortest plan"
    capabilities = ('planning', 'spatial_reasoning', 'proof_search')
    axes = {'reasoning_depth': 4, 'world_complexity': 3, 'lexical_novelty': 3, 'compositional_depth': 2}

    generate = staticmethod(gen_planning_language)
