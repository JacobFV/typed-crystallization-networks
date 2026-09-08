"""``speaker_listener_game`` — inferring an invented communication code from its use.

Language as action.
"""

from __future__ import annotations

import random
from typing import Any

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.base import COLORS, SHAPES
from ..generators.causal import _nonce_names


def gen_speaker_listener_game(rng: random.Random, ctx):
    """A speaker's code is arbitrary: which *dimension* it speaks about, and
    which value each signal denotes, are both invented per episode and both
    recoverable only from the rounds shown.

    Each signal appears in two past rounds whose successful choices agree on the
    code dimension and *disagree* on the other one, which refutes the competing
    hypothesis exactly rather than merely making it unlikely."""
    dim = rng.choice(["color", "shape"])
    other = "shape" if dim == "color" else "color"
    dim_vals = COLORS if dim == "color" else SHAPES
    other_vals = SHAPES if dim == "color" else COLORS

    n_sig = ctx.at(2, 5, default=2)
    signals = _nonce_names(rng, n_sig)
    denote = dict(zip(signals, rng.sample(dim_vals, n_sig)))
    query_sig = rng.choice(signals)

    rounds: list[dict[str, Any]] = []
    for sig in signals:
        # two rounds per signal: the chosen objects share the code value and
        # differ on the other attribute, so only the code dimension explains both
        others = rng.sample(other_vals, 2)
        for r in range(2):
            tgt = {"color": denote[sig] if dim == "color" else others[r],
                   "shape": denote[sig] if dim == "shape" else others[r]}
            objs = [dict(tgt)]
            while len(objs) < 3:                       # distractors never match the code value
                o = {"color": rng.choice(COLORS), "shape": rng.choice(SHAPES)}
                if o[dim] != denote[sig]:
                    objs.append(o)
            rng.shuffle(objs)
            for i, o in enumerate(objs):
                o["id"] = f"o{i}"
            chosen = next(o for o in objs if o[dim] == denote[sig] and o[other] == tgt[other])
            rounds.append({"signal": sig, "objs": objs, "choice": chosen["id"]})
    rng.shuffle(rounds)

    # the current scene: exactly one object carries the queried signal's value
    tgt = {"color": denote[query_sig] if dim == "color" else rng.choice(COLORS),
           "shape": denote[query_sig] if dim == "shape" else rng.choice(SHAPES)}
    scene = [dict(tgt)]
    while len(scene) < 4:
        o = {"color": rng.choice(COLORS), "shape": rng.choice(SHAPES)}
        if o[dim] != denote[query_sig]:
            scene.append(o)
    rng.shuffle(scene)
    for i, o in enumerate(scene):
        o["id"] = f"o{i}"
    answer = next(o for o in scene if o[dim] == denote[query_sig])["id"]

    facts: list[Term] = []
    for r, rd in enumerate(rounds):
        facts.append(Pred("signal", Num(r), Ident(rd["signal"])))
        for o in rd["objs"]:
            facts.append(Pred("round_obj", Num(r), Ident(o["id"]), Ident(o["color"]), Ident(o["shape"])))
        facts.append(Pred("listener_chose", Num(r), Ident(rd["choice"])))
    obs = Rec(rounds=Lst(facts),
              scene=Lst([Pred("obj", Ident(o["id"]), Ident(o["color"]), Ident(o["shape"])) for o in scene]),
              query=Pred("act_on", Ident(query_sig)))
    hidden = {"code_dimension": dim, "code": dict(denote), "signal": query_sig, "target": answer}
    return obs, [o["id"] for o in scene], answer, hidden


class SpeakerListenerGame(Lesson):
    """Inferring an invented communication code from its use."""

    id = "speaker_listener_game"
    level = 27
    tags = ("pragmatics", "language-as-action")
    teaches = "inferring an invented communication code from its use"
    capabilities = ('multi_agent_coordination', 'lexical_grounding', 'abstraction')
    axes = {'lexical_novelty': 3, 'ambiguity': 2, 'reasoning_depth': 2, 'discourse_horizon': 2, 'world_complexity': 2}

    generate = staticmethod(gen_speaker_listener_game)
