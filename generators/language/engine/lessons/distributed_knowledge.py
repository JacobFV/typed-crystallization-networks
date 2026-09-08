"""``distributed_knowledge`` — which single message makes the group able to answer.

Protocols, institutions, and distributed intelligence.
"""

from __future__ import annotations

import random
from typing import Sequence

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, SHAPES
from ..generators.reflective import _SIZES, _labels, _shuffled


def gen_distributed_knowledge(rng: random.Random, ctx):
    """Which single message lets the group answer the question?

    One agent must name the object matching a description but knows only part of
    the world, so two objects remain possible. Exactly one of the four candidate
    messages held by the other agents rules the impostor out; the rest are true
    but useless. The possibility set after every message is computed explicitly.
    """
    attrs = ["color", "shape", "size"]
    n_obj = ctx.at(5, 9, default=5)
    fallback = None
    for _ in range(400):
        objs = [f"o{i}" for i in range(n_obj)]
        world = {o: {"color": rng.choice(COLORS), "shape": rng.choice(SHAPES),
                     "size": rng.choice(_SIZES)} for o in objs}
        d_attrs = rng.sample(attrs, 2)
        target = rng.choice(objs)
        desc = {k: world[target][k] for k in d_attrs}
        if sum(1 for o in objs if all(world[o][k] == v for k, v in desc.items())) != 1:
            continue
        w = rng.choice([o for o in objs if o != target])
        known: list[tuple[str, str, str]] = []
        for o in objs:
            if o == target:
                for k in d_attrs:
                    if rng.random() < 0.5:
                        known.append((o, k, world[o][k]))
            elif o == w:
                for k in d_attrs:
                    if world[o][k] == desc[k]:
                        known.append((o, k, world[o][k]))
            else:                                  # already excluded, and q knows why
                bad = rng.choice([k for k in d_attrs if world[o][k] != desc[k]])
                known.append((o, bad, world[o][bad]))
        for _ in range(3):                                   # non-diagnostic extra knowledge
            o = rng.choice(objs)
            k = rng.choice([x for x in attrs if x not in d_attrs])
            known.append((o, k, world[o][k]))
        known = sorted(set(known))

        def possible(k_facts: Sequence[tuple[str, str, str]]) -> list[str]:
            kb = {(o, at): v for o, at, v in k_facts}
            return [o for o in objs
                    if all(kb.get((o, at), v) == v for at, v in desc.items())]

        base = possible(known)
        if sorted(base) != sorted([target, w]):
            continue
        wrong = [k for k in d_attrs if world[w][k] != desc[k]]
        if not wrong:
            continue
        msgs = [(w, wrong[0], world[w][wrong[0]])]
        pool: list[tuple[str, str, str]] = []
        for o in objs:
            for k in attrs:
                f = (o, k, world[o][k])
                if f not in known and f not in msgs:
                    if o == w and k in wrong:
                        continue
                    pool.append(f)
        if len(pool) < 3:
            continue
        msgs += rng.sample(pool, 3)
        ids = _labels(rng, "m", 4)
        order = _shuffled(rng, range(4))
        assign = {ids[k]: msgs[order[k]] for k in range(4)}
        solving = [i for i in ids if len(possible(list(known) + [assign[i]])) == 1]
        cand = (objs, world, desc, known, assign, ids, solving[0] if solving else ids[0], target)
        if fallback is None:
            fallback = cand
        if len(solving) == 1:
            fallback = cand
            break
    objs, world, desc, known, assign, ids, answer, target = fallback
    obs = Rec(question=Lst([Pred("requires", Ident(k), Ident(v)) for k, v in sorted(desc.items())]),
              objects=Lst([Ident(o) for o in objs]),
              knows=Lst([Pred("knows", Ident("q"), Ident(o), Ident(k), Ident(v))
                         for o, k, v in known]),
              messages=Lst(_shuffled(rng, [Pred("message", Ident(i), Ident(assign[i][0]),
                                                Ident(assign[i][1]), Ident(assign[i][2]))
                                           for i in ids])),
              query=Ident("which_message_makes_q_certain"))
    return obs, _shuffled(rng, ids), answer, {"target": target, "description": dict(desc)}


class DistributedKnowledge(Lesson):
    """Which single message makes the group able to answer."""

    id = "distributed_knowledge"
    level = 128
    tags = ("protocols", "institutions", "distributed-intelligence")
    teaches = "which single message makes the group able to answer"
    capabilities = ('multi_agent_coordination', 'belief_modeling', 'quantification')
    axes = {'reasoning_depth': 5, 'world_complexity': 4, 'ambiguity': 3}

    generate = staticmethod(gen_distributed_knowledge)
