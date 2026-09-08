"""``historical_reconstruction`` — which history is consistent with the present state and all records.

History, narrative, perspective, and identity.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.reflective import _labels, _shuffled


def gen_historical_reconstruction(rng: random.Random, ctx):
    """Which history is consistent with the present state and every record?

    Items pass between people; the learner sees where they started, where they
    ended, and a handful of partial records (who held what at some time, what
    never happened, how many transfers there were). All four candidate histories
    are legal move sequences, so only re-simulation separates them — and the
    generator keeps the world only when exactly one candidate survives.
    """
    people = ["alice", "bob", "carol"]
    items = ["key", "cup", "book"]

    def run(init: Mapping[str, str], hist: Sequence[tuple[str, str, str]]):
        state = dict(init)
        seen = {(state[i], i) for i in items}
        traj = [dict(state)]
        for f, t, it in hist:
            if state.get(it) != f or f == t:
                return None, None, None
            state[it] = t
            seen.add((t, it))
            traj.append(dict(state))
        return state, traj, seen

    fallback = None
    for _ in range(400):
        init = {it: rng.choice(people) for it in items}
        true_hist = []
        state = dict(init)
        for _ in range(rng.randint(*ctx.span((3, 4), (7, 9)))):
            it = rng.choice(items)
            f = state[it]
            t = rng.choice([p for p in people if p != f])
            true_hist.append((f, t, it))
            state[it] = t
        present, traj, seen = run(init, true_hist)
        records: list[tuple] = []
        tt = rng.randint(1, len(true_hist))
        it = rng.choice(items)
        records.append(("at_time", tt, traj[tt][it], it))
        never = [(p, i) for p in people for i in items if (p, i) not in seen]
        if never and rng.random() < 0.7:
            p, i = rng.choice(never)
            records.append(("never_held", p, i))
        if rng.random() < 0.6:
            records.append(("n_moves", len(true_hist)))

        def consistent(hist: Sequence[tuple[str, str, str]]) -> bool:
            st, tj, sn = run(init, hist)
            if st is None or st != present:
                return False
            for r in records:
                if r[0] == "at_time":
                    if len(tj) <= r[1] or tj[r[1]][r[3]] != r[2]:
                        return False
                elif r[0] == "never_held":
                    if (r[1], r[2]) in sn:
                        return False
                elif r[0] == "n_moves" and len(hist) != r[1]:
                    return False
            return True

        alts: list[list[tuple[str, str, str]]] = []
        for _ in range(120):
            st = dict(init)
            h = []
            for _ in range(rng.randint(*ctx.span((2, 5), (5, 10)))):
                i2 = rng.choice(items)
                f = st[i2]
                t = rng.choice([p for p in people if p != f])
                h.append((f, t, i2))
                st[i2] = t
            if h != true_hist and not consistent(h) and h not in alts:
                alts.append(h)
            if len(alts) == 3:
                break
        if len(alts) < 3:
            continue
        options = [true_hist] + alts
        ids = _labels(rng, "h", 4)
        order = _shuffled(rng, range(4))
        assign = {ids[k]: options[order[k]] for k in range(4)}
        good = [i for i in ids if consistent(assign[i])]
        cand = (init, present, records, assign, ids, good[0] if good else ids[0])
        if fallback is None:
            fallback = cand
        if len(good) == 1:
            fallback = cand
            break
    init, present, records, assign, ids, answer = fallback
    rec_syms = []
    for r in records:
        if r[0] == "at_time":
            rec_syms.append(Pred("record", Pred("held_at_time"), Num(r[1]), Ident(r[2]), Ident(r[3])))
        elif r[0] == "never_held":
            rec_syms.append(Pred("record", Pred("never_held"), Ident(r[1]), Ident(r[2])))
        else:
            rec_syms.append(Pred("record", Pred("number_of_transfers"), Num(r[1])))
    obs = Rec(initial=Lst([Pred("holds", Ident(init[i]), Ident(i)) for i in items]),
              present=Lst([Pred("holds", Ident(present[i]), Ident(i)) for i in items]),
              records=Lst(_shuffled(rng, rec_syms)),
              candidates=Lst(_shuffled(rng, [
                  Pred("history", Ident(i),
                       Lst([Pred("give", Ident(f), Ident(t), Ident(it)) for f, t, it in assign[i]]))
                  for i in ids])),
              query=Ident("which_history_is_consistent"))
    return obs, _shuffled(rng, ids), answer, {"n_records": len(records),
                                              "length": len(assign[answer])}


class HistoricalReconstruction(Lesson):
    """Which history is consistent with the present state and all records."""

    id = "historical_reconstruction"
    level = 131
    tags = ("history", "narrative", "perspective", "identity")
    teaches = "which history is consistent with the present state and all records"
    capabilities = ('temporal_reasoning', 'causal_reasoning', 'planning')
    axes = {'reasoning_depth': 5, 'discourse_horizon': 5, 'world_complexity': 4}

    generate = staticmethod(gen_historical_reconstruction)
