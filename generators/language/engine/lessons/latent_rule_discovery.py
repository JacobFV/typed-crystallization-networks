"""``latent_rule_discovery`` — infer a hidden law from a trajectory and run it.

Scientific induction and model discovery.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Rec
from ..lesson import Lesson
from ..generators.science import _CA_CELLS, _ca_fits, _ca_step, _shuffled


def gen_latent_rule_discovery(rng: random.Random, ctx):
    """A hidden law runs a world; the trajectory is shown; predict the next state.

    Half the episodes are an elementary cellular automaton on a ring, half an
    affine recurrence in a modular arithmetic system. In both, the episode is
    *rejected* unless every rule consistent with the displayed trajectory agrees
    on the next state — so the answer is entailed by what the learner sees, not
    merely by what the generator knows. The previous state is always offered as
    a distractor, so "predict no change" is available and always wrong.
    """
    ca = rng.random() < 0.5
    cells = ctx.at(_CA_CELLS, 11, default=_CA_CELLS)       # width of the ring
    moduli = ctx.among([[7, 9, 11, 13], [11, 13, 17, 19], [17, 19, 23, 29]])
    for _ in range(400):
        if ca:
            rule = rng.randrange(256)
            traj = [tuple(rng.randint(0, 1) for _ in range(cells))]
            for _ in range(5):
                traj.append(_ca_step(traj[-1], rule))
            nxt = _ca_step(traj[-1], rule)
            if len(set(traj)) < 3 or nxt == traj[-1]:
                continue                                  # degenerate / fixed point
            consistent = [r for r in range(256) if _ca_fits(traj, r)]
            if len({_ca_step(traj[-1], r) for r in consistent}) != 1:
                continue                                  # under-determined by the data
            rivals = []
            for r in range(256):
                p = _ca_step(traj[-1], r)
                if p != nxt and p != traj[-1] and p not in rivals:
                    rivals.append(p)
            distract = [traj[-1]] + _shuffled(rng, rivals)[:3]
            if len(distract) < 4:
                continue
            fmt = lambda s: "".join(str(b) for b in s)
            obs = Rec(trajectory=Lst([Lst([Num(b) for b in s]) for s in traj]),
                      dynamics=Ident("synchronous_ring_update"),
                      query=Ident("next_state"))
            answer = fmt(nxt)
            hidden = {"mode": "cellular_automaton", "rule": rule,
                      "consistent_rules": len(consistent)}
            return obs, _shuffled(rng, [answer] + [fmt(d) for d in distract]), answer, hidden

        m = rng.choice(moduli)
        a, b = rng.randrange(1, m), rng.randrange(m)
        seq = [rng.randrange(m)]
        for _ in range(5):
            seq.append((a * seq[-1] + b) % m)
        nxt = (a * seq[-1] + b) % m
        if len(set(seq)) < 3 or nxt == seq[-1]:
            continue
        consistent = [(aa, bb) for aa in range(m) for bb in range(m)
                      if all((aa * seq[i] + bb) % m == seq[i + 1] for i in range(len(seq) - 1))]
        if len({(aa * seq[-1] + bb) % m for aa, bb in consistent}) != 1:
            continue
        rivals = [v for v in range(m) if v != nxt and v != seq[-1]]
        distract = [seq[-1]] + _shuffled(rng, rivals)[:3]
        obs = Rec(trajectory=Lst([Num(v) for v in seq]), modulus=Num(m),
                  dynamics=Ident("affine_recurrence"), query=Ident("next_state"))
        hidden = {"mode": "arithmetic_system", "a": a, "b": b, "modulus": m,
                  "consistent_rules": len(consistent)}
        return obs, _shuffled(rng, [nxt] + distract), nxt, hidden
    raise RuntimeError("latent_rule_discovery: no admissible episode")


class LatentRuleDiscovery(Lesson):
    """Infer a hidden law from a trajectory and run it."""

    id = "latent_rule_discovery"
    level = 68
    tags = ("science", "induction", "model-discovery")
    teaches = "infer a hidden law from a trajectory and run it"
    capabilities = ('scientific_induction', 'program_synthesis', 'abstraction')
    axes = {'reasoning_depth': 4, 'world_complexity': 3, 'grammar_complexity': 2, 'discourse_horizon': 2}

    generate = staticmethod(gen_latent_rule_discovery)
