"""``contract_reasoning`` — conditional obligations over an event timeline: active, fulfilled, breached.

Protocols, institutions, and distributed intelligence.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.reflective import _CONTRACT_STATUSES, _contract_status, _nonces, _shuffled


def gen_contract_reasoning(rng: random.Random, ctx):
    """Is the obligation inactive, active, fulfilled, breached or cancelled?

    A promise creates a future obligation the moment its trigger event occurs,
    to be discharged within a deadline, and a cancellation clause can void it in
    between. The timeline is generated to aim at a status and the status is then
    *recomputed* from the timeline, so the label is what the events actually
    imply rather than what they were built to imply.
    """
    target = rng.choice(_CONTRACT_STATUSES)
    n_ev = ctx.at(5, 11, default=5)
    fallback = None
    for _ in range(300):
        names = _nonces(rng, n_ev, 3)
        trigger, action, cancel = names[0], names[1], names[2]
        noise = names[3:]
        has_cancel = rng.random() < 0.6
        deadline = rng.randint(1, 3)
        times = sorted(rng.sample(range(1, 12), n_ev))
        events: list[tuple[int, str]] = []
        t0 = times[1]
        if target != "inactive":
            events.append((t0, trigger))
            if target == "fulfilled":
                events.append((t0 + rng.randint(0, deadline), action))
            elif target == "cancelled" and has_cancel:
                events.append((t0 + rng.randint(0, deadline), cancel))
            elif target == "breached":
                pass
        events.append((times[0], rng.choice(noise)))
        for t in times[2:]:
            if rng.random() < 0.5:
                events.append((t, rng.choice(noise + [action if target == "inactive" else noise[0]])))
        if target in ("inactive",) and rng.random() < 0.5:
            events.append((times[n_ev - 1], trigger))  # triggers only after "now"
        events = sorted({(t, n) for t, n in events})
        if target == "fulfilled":
            now = max(t for t, _ in events)
        elif target == "active":
            events = [e for e in events if e[1] not in (action, cancel)] + [(t0, trigger)]
            events = sorted(set(events))
            now = t0 + deadline
        elif target == "breached":
            now = t0 + deadline + rng.randint(1, 3)
        elif target == "cancelled":
            now = max(t for t, _ in events)
        else:
            now = times[1] - 1 if times[1] > 1 else 1
            events = [e for e in events if not (e[1] == trigger and e[0] <= now)]
        cancel_arg = cancel if has_cancel else None
        status = _contract_status(trigger, action, deadline, cancel_arg, events, now)
        cand = (trigger, action, cancel_arg, deadline, events, now, status)
        if fallback is None:
            fallback = cand
        if status == target:
            fallback = cand
            break
    trigger, action, cancel_arg, deadline, events, now, status = fallback
    clause = [Pred("on_event", Ident(trigger)), Pred("obliges", Pred("party_a"), Ident(action)),
              Pred("within", Num(deadline))]
    if cancel_arg is not None:
        clause.append(Pred("cancelled_by", Ident(cancel_arg)))
    obs = Rec(contract=Lst(clause),
              timeline=Lst([Pred("event", Num(t), Ident(n)) for t, n in sorted(events)]),
              query=Pred("status_at", Num(now)))
    return (obs, _shuffled(rng, _CONTRACT_STATUSES), status,
            {"status": status, "now": now, "deadline": deadline, "n_events": len(events)})


class ContractReasoning(Lesson):
    """Conditional obligations over an event timeline: active, fulfilled, breached."""

    id = "contract_reasoning"
    level = 125
    tags = ("protocols", "institutions", "distributed-intelligence")
    teaches = "conditional obligations over an event timeline: active, fulfilled, breached"
    capabilities = ('temporal_reasoning', 'belief_modeling', 'planning')
    axes = {'reasoning_depth': 4, 'discourse_horizon': 4, 'world_complexity': 3, 'compositional_depth': 3}
    answers = ['inactive', 'active', 'fulfilled', 'breached', 'cancelled']

    generate = staticmethod(gen_contract_reasoning)
