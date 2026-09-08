"""``curriculum_invention`` — which curriculum's prerequisite closure reaches a target capability.

Civilization-scale symbolic learning.
"""

from __future__ import annotations

import random
from typing import Sequence

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.capstone import _curriculum_valid, _nonce_pool, _shuffled


def gen_curriculum_invention(rng: random.Random, ctx):
    """Given a target capability and four candidate training curricula, which
    one actually induces it?

    Each stage declares what it presupposes and what it produces; a curriculum
    induces the target iff prerequisite closure holds *in order* and the target
    is produced. The three distractors are near-misses of three different kinds
    — never produces the target, one prerequisite never produced at all, right
    stages in an order that uses a capability before it exists — and all four
    are run through the checker so that exactly one is valid.
    """
    depth = rng.randint(*ctx.span((3, 4), (7, 9)))
    pool = _nonce_pool(rng, depth + 5)
    base = pool[0]
    chain = pool[1:depth + 1]              # base -> chain[0] -> ... -> target
    extra = pool[depth + 1:]
    target = chain[-1]
    # every candidate gets its own block of same-shape nonce stage names: reusing
    # or suffixing names would make the answer readable off the labels alone
    names = _nonce_pool(rng, 4 * (depth + 1), 1)
    blocks_of = [names[c * depth:(c + 1) * depth] for c in range(4)]

    def stage_block(i: int, nm: Sequence[str]) -> tuple[str, tuple[str, ...], str]:
        req = (base,) if i == 0 else (chain[i - 1],)
        return (nm[i], req, chain[i])

    good = [stage_block(i, blocks_of[0]) for i in range(depth)]
    # (a) all prerequisites met, but the target is never produced
    trunc = [stage_block(i, blocks_of[1]) for i in range(depth - 1)]
    trunc.append((blocks_of[1][depth - 1],
                  (chain[depth - 2],) if depth >= 2 else (base,), extra[0]))
    # (b) one stage presupposes something nothing produces
    broke = [stage_block(i, blocks_of[2]) for i in range(depth)]
    j = rng.randrange(1, depth)
    broke[j] = (broke[j][0], (extra[1],), broke[j][2])
    # (c) the right stages, in an order that uses a capability before it exists
    swapped = [stage_block(i, blocks_of[3]) for i in range(depth)]
    i = rng.randrange(depth - 1)
    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]

    cands = [good, trunc, broke, swapped]
    flags = [_curriculum_valid(c, [base], target) for c in cands]
    if flags != [True, False, False, False]:               # never trust construction
        return gen_curriculum_invention(random.Random(rng.random()), ctx)

    labels = _shuffled(rng, [f"c{i}" for i in range(4)])
    order = _shuffled(rng, list(range(4)))
    blocks: list[Term] = []
    answer = labels[0]
    for slot, idx in enumerate(order):
        lab = labels[slot]
        if idx == 0:
            answer = lab
        steps = [Pred("stage", Num(k), Ident(nm),
                      Lst([Ident(r) for r in req]), Ident(prod))
                 for k, (nm, req, prod) in enumerate(cands[idx])]
        blocks.append(Rec(curriculum=Ident(lab), syllabus=Lst(steps)))
    obs = Rec(learner=Rec(has=Lst([Ident(base)])),
              proposals=Lst(blocks),
              semantics=Lst([Pred("stage_teaches", Ident("produces")),
                             Pred("stage_needs", Ident("requires")),
                             Pred("order", Pred("stages_run_in_listed_index_order"))]),
              query=Pred("which_curriculum_induces", Ident(target)))
    hidden = {"target": target, "answer": answer, "depth": depth,
              "chain": chain, "kinds": {"good": 0, "no_target": 1, "unmet_prereq": 2, "misordered": 3}}
    return obs, _shuffled(rng, labels), answer, hidden


class CurriculumInvention(Lesson):
    """Which curriculum's prerequisite closure reaches a target capability."""

    id = "curriculum_invention"
    level = 166
    tags = ("civilization-scale",)
    teaches = "which curriculum's prerequisite closure reaches a target capability"
    capabilities = ('metareasoning', 'planning', 'abstraction')
    axes = {'reasoning_depth': 5, 'compositional_depth': 4, 'world_complexity': 3, 'lexical_novelty': 4}

    generate = staticmethod(gen_curriculum_invention)
