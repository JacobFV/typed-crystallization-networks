"""``self_repair`` — the patch that restores competence.

Self-modeling and architecture adaptation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import MODULE_NAMES, SLOTS, _labels, _rules, _shuffled


def gen_self_repair(rng: random.Random, ctx):
    """One slot of the described architecture is under-powered; which patch fixes it?

    A repair names a slot and the module that would replace it. Distractor
    repairs either upgrade a slot that was never the problem or install a module
    into the broken slot that is still below requirement, so both the *locus* of
    the fault and the *sufficiency* of the patch have to be checked.
    """
    req = {s: rng.randint(2, 5) for s in SLOTS}
    broken = rng.choice(SLOTS)
    cur_level = {s: (rng.randint(0, req[s] - 1) if s == broken else rng.randint(req[s], 6))
                 for s in SLOTS}
    installed = dict(zip(SLOTS, rng.sample(MODULE_NAMES, len(SLOTS))))

    n_fix = ctx.at(4, 9, default=4)               # candidate repairs to check one by one
    ids = _labels(rng, "fix", n_fix)
    fix_i = rng.randrange(n_fix)
    spares = rng.sample([m for m in MODULE_NAMES if m not in installed.values()], n_fix)
    repairs: list[tuple[str, str, str, int]] = []
    for i in range(n_fix):
        if i == fix_i:
            slot, lvl = broken, rng.randint(req[broken], 6)
        elif rng.random() < 0.5:
            slot = rng.choice([s for s in SLOTS if s != broken])
            lvl = rng.randint(0, 6)
        else:
            slot, lvl = broken, rng.randint(0, req[broken] - 1)
        repairs.append((ids[i], slot, spares[i], lvl))

    def fixes(r: tuple[str, str, str, int]) -> bool:
        lv = dict(cur_level)
        lv[r[1]] = r[3]
        return all(lv[s] >= req[s] for s in SLOTS)

    assert sum(1 for r in repairs if fixes(r)) == 1
    obs = Rec(agent=Lst([Pred("installed", Ident(s), Ident(installed[s]), Num(cur_level[s]))
                         for s in SLOTS]),
              task=Lst([Pred("requires", Ident(s), Num(req[s])) for s in SLOTS]),
              repairs=Lst(_shuffled(rng, [Pred("repair", Ident(rid), Ident(sl), Ident(m), Num(l))
                                          for rid, sl, m, l in repairs])),
              rules=_rules("a_repair_replaces_only_the_slot_it_names",
                           "the_task_is_solved_iff_every_slot_level_is_at_least_its_requirement"),
              query=Ident("which_repair_solves_the_task"))
    return (obs, _shuffled(rng, ids), ids[fix_i],
            {"broken_slot": broken, "requirement": req[broken], "level": cur_level[broken]})


class SelfRepair(Lesson):
    """The patch that restores competence."""

    id = "self_repair"
    level = 138
    tags = ("self-modeling", "architecture")
    teaches = "the patch that restores competence"
    capabilities = ('self_modeling', 'architecture_adaptation', 'planning')
    axes = {'reasoning_depth': 3, 'world_complexity': 3, 'compositional_depth': 3}

    generate = staticmethod(gen_self_repair)
