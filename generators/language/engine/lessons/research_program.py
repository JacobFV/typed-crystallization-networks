"""``research_program`` — next action in a partially resolved question graph.

Open-ended epistemology.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import _labels, _rules, _shuffled


def gen_research_program(rng: random.Random, ctx):
    """Pick the next research action from a partially resolved question graph.

    An action is available only if everything it depends on is already resolved,
    and it is worth taking in proportion to how much of the remaining programme it
    unblocks. Both conditions are structural, so "what should I work on next"
    reduces to reachability plus a count.
    """
    n = ctx.at(6, 14, default=6)
    for _ in range(200):
        deps = {i: set() for i in range(n)}       # i depends on each j in deps[i], j < i
        for i in range(1, n):
            for j in range(i):
                if rng.random() < 0.35:
                    deps[i].add(j)
        resolved: set[int] = set()
        for i in range(n):
            if deps[i] <= resolved and rng.random() < 0.5:
                resolved.add(i)
        open_q = [i for i in range(n) if i not in resolved]
        ready = [i for i in open_q if deps[i] <= resolved]

        def descendants(q: int) -> set[int]:
            out, frontier = set(), {q}
            while frontier:
                cur = frontier.pop()
                for i in open_q:
                    if cur in deps[i] and i not in out:
                        out.add(i)
                        frontier.add(i)
            return out

        if len(ready) < 2:
            continue
        vals = {q: len(descendants(q)) for q in ready}
        top = max(vals.values())
        if list(vals.values()).count(top) == 1 and top > 0:
            break
    best = max(ready, key=lambda q: vals[q])

    qids = _labels(rng, "q", n)
    facts = [Pred("question", Ident(qids[i])) for i in range(n)]
    facts += [Pred("depends_on", Ident(qids[i]), Ident(qids[j])) for i in range(n) for j in deps[i]]
    facts += [Pred("resolved", Ident(qids[i])) for i in sorted(resolved)]
    obs = Rec(program=Lst(_shuffled(rng, facts)),
              rules=_rules("a_question_is_available_iff_it_is_unresolved_and_all_it_depends_on_are_resolved",
                           "the_value_of_a_question_is_the_number_of_unresolved_questions_that_depend_on_it_directly_or_indirectly",
                           "choose_the_available_question_of_greatest_value"),
              query=Ident("next_research_action"))
    return (obs, _shuffled(rng, qids), qids[best],
            {"ready": [qids[q] for q in ready], "values": {qids[q]: vals[q] for q in ready}})


class ResearchProgram(Lesson):
    """Next action in a partially resolved question graph."""

    id = "research_program"
    level = 148
    tags = ("open-ended-epistemology",)
    teaches = "next action in a partially resolved question graph"
    capabilities = ('open_ended_discovery', 'planning', 'metareasoning')
    axes = {'reasoning_depth': 4, 'discourse_horizon': 4, 'world_complexity': 3}

    generate = staticmethod(gen_research_program)
