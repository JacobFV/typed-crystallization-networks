"""``long_horizon_projects`` — the next required step of an interrupted project.

Problem formulation and hierarchical agency.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import _nonces, _shuffled


def gen_long_horizon_projects(rng: random.Random, ctx):
    """A project mid-flight: what has to happen next.

    The critical path is a chain of prerequisites toward the deliverable, the
    log records completions *and* failures (a failed task is not done, whatever
    the log said earlier), and several side tasks are ready but irrelevant. The
    answer is the unique task that is both required for the goal and unblocked,
    which cannot be read off the log's last line.
    """
    for _ in range(200):
        n_chain = rng.randint(*ctx.span((4, 5), (9, 12)))
        names = _nonces(rng, n_chain + 4, 4)
        chain = names[:n_chain]
        goal = chain[-1]
        side = names[n_chain:]
        prereqs: dict[str, list[str]] = {chain[0]: []}
        for i in range(1, n_chain):
            prereqs[chain[i]] = [chain[i - 1]]
        for i, s in enumerate(side):
            prereqs[s] = [chain[0]] if (i % 2 == 0 and rng.random() < 0.6) else []

        cut = rng.randrange(0, n_chain)         # everything before ``cut`` is complete
        status = {t: "pending" for t in names}
        for t in chain[:cut]:
            status[t] = "done"
        log = [Pred("completed", Num(i), Ident(t)) for i, t in enumerate(chain[:cut])]
        if cut > 0 and rng.random() < 0.45:     # an interruption undid earlier work
            k = rng.randrange(cut)
            status[chain[k]] = "failed"
            log.append(Pred("failed", Num(len(log)), Ident(chain[k])))
        for s in side:
            r = rng.random()
            if r < 0.35:
                status[s] = "done"
                log.append(Pred("completed", Num(len(log)), Ident(s)))
            elif r < 0.5:
                status[s] = "failed"
                log.append(Pred("failed", Num(len(log)), Ident(s)))

        done = {t for t in names if status[t] == "done"}
        required = set(chain)
        ready = [t for t in names if t not in done and all(p in done for p in prereqs[t])]
        answer_pool = sorted(t for t in ready if t in required)
        if len(answer_pool) == 1 and len(ready) >= 2:
            break
    else:                                     # pragma: no cover - construction
        answer_pool = [chain[0]]

    obs = Rec(tasks=Lst(_shuffled(rng, [Pred("task", Ident(t)) for t in names])),
              dependencies=Lst(_shuffled(rng, [Pred("requires", Ident(t), Ident(p))
                                               for t in names for p in prereqs[t]])),
              log=Lst(log),
              query=Pred("next_required_step_for", Ident(goal)))
    return (obs, _shuffled(rng, names), answer_pool[0],
            {"goal": goal, "answer": answer_pool[0], "done": sorted(done),
             "ready": sorted(ready), "chain": chain})


class LongHorizonProjects(Lesson):
    """The next required step of an interrupted project."""

    id = "long_horizon_projects"
    level = 107
    tags = ("problem-formulation", "hierarchical-agency")
    teaches = "the next required step of an interrupted project"
    capabilities = ('long_horizon_agency', 'planning', 'state_tracking')
    axes = {'planning_horizon': 4, 'discourse_horizon': 4, 'world_complexity': 4}

    generate = staticmethod(gen_long_horizon_projects)
