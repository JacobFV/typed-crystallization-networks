"""``question_answering`` — interrogative form -> reasoning operation.

Language as action.
"""

from __future__ import annotations

import random
from typing import Any

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.semantics import ACTIONS, PLACES, _shuffled


def gen_question_answering(rng: random.Random, ctx):
    """who / what / where / when / how-many over an explicit event world.

    Each interrogative is a different reasoning operation over the same facts.
    The generator enumerates the questions whose answer is *unique* and samples
    among them, and for counting questions samples uniformly over the achievable
    counts so that "one" is not the default reply.
    """
    for _ in range(200):
        n = rng.randint(*ctx.span((4, 5), (7, 8)))
        actors = rng.sample(NAMES, 4)
        actions = rng.sample(ACTIONS, 4)
        places = rng.sample(PLACES, 4)
        times = rng.sample(range(0, 9), n)
        evs = [{"id": f"e{i}", "actor": rng.choice(actors), "action": rng.choice(actions),
                "place": rng.choice(places), "time": times[i]} for i in range(n)]

        opts: dict[str, list[tuple[Any, Any]]] = {}
        for act in actions:                       # who did <action>?
            m = sorted({e["actor"] for e in evs if e["action"] == act})
            if len(m) == 1:
                opts.setdefault("who", []).append((act, m[0]))
        for who in actors:                        # what did <actor> do? / where?
            m = sorted({e["action"] for e in evs if e["actor"] == who})
            if len(m) == 1:
                opts.setdefault("what", []).append((who, m[0]))
            p = sorted({e["place"] for e in evs if e["actor"] == who})
            if len(p) == 1:
                opts.setdefault("where", []).append((who, p[0]))
            t = sorted({e["time"] for e in evs if e["actor"] == who})
            if len(t) == 1:
                opts.setdefault("when", []).append((who, t[0]))
        counts: dict[int, list[str]] = {}
        for p in places:
            counts.setdefault(sum(1 for e in evs if e["place"] == p), []).append(p)
        if counts:
            c = rng.choice(sorted(counts))
            opts["how_many"] = [(rng.choice(sorted(counts[c])), c)]
        if len(opts) < 2:
            continue
        kind = rng.choice(sorted(opts))
        arg, answer = rng.choice(opts[kind])
        break
    else:                                                  # pragma: no cover
        raise RuntimeError("no unique question")

    vocab = {"who": actors, "what": actions, "where": places,
             "when": sorted(times), "how_many": list(range(0, n + 1))}[kind]
    qsym = Pred(kind, Num(arg) if isinstance(arg, int) else Ident(arg))
    obs = Rec(world=Lst(_shuffled(rng, [
                  Pred("event", Ident(e["id"]), Ident(e["actor"]), Ident(e["action"]),
                       Ident(e["place"]), Num(e["time"])) for e in evs])),
              query=qsym)
    return obs, _shuffled(rng, vocab), answer, {"kind": kind, "arg": arg}


class QuestionAnswering(Lesson):
    """Interrogative form -> reasoning operation."""

    id = "question_answering"
    level = 36
    tags = ("pragmatics", "language-as-action")
    teaches = "interrogative form -> reasoning operation"
    capabilities = ('proof_search', 'quantification')
    axes = {'world_complexity': 3, 'reasoning_depth': 3, 'compositional_depth': 2}

    generate = staticmethod(gen_question_answering)
