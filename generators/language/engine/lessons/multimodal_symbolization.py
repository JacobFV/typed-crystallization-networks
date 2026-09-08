"""``multimodal_symbolization`` — perception symbols carrying uncertainty.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.social import DETECT_CLASSES, SURFACES, _inside, _shuffled


def gen_multimodal_symbolization(rng: random.Random, ctx):
    """Detector symbols — ``(label, score, cx, cy, w, h)`` — with two surfaces.

    Perception is modular here: the
    detector emits boxes, classes and scores, and the remaining problem is
    symbolic. The twist is that confidence carries information. Two detections
    share the queried surface label and only one is above the trust threshold, so
    the *region* comes from a high-confidence detection; the answer is the
    uncertain object inside it, so the *identity* comes from a low-confidence one.
    Reading either surface alone gives a specific wrong answer: the object in the
    untrusted surface, or the confident object sitting in the right one.
    """
    n_off = ctx.at(2, 5, default=2)               # uncertain detections outside every surface
    for _ in range(200):
        tau = round(rng.uniform(0.45, 0.6), 2)
        surface = rng.choice(SURFACES)
        labels = rng.sample(DETECT_CLASSES, 3 + n_off)
        answer, decoy = labels[0], labels[1]
        offs = labels[2:2 + n_off]
        confident = labels[2 + n_off]

        def _lo() -> float:
            return round(rng.uniform(0.15, tau - 0.05), 2)

        def _hi() -> float:
            return round(rng.uniform(tau + 0.05, 0.99), 2)

        trusted = {"cx": 25 + rng.randint(-3, 3), "cy": 25 + rng.randint(-3, 3), "w": 20, "h": 20}
        untrusted = {"cx": 75 + rng.randint(-3, 3), "cy": 75 + rng.randint(-3, 3), "w": 20, "h": 20}

        def _in(box: Mapping[str, float]) -> tuple[int, int]:
            return (box["cx"] + rng.randint(-6, 6), box["cy"] + rng.randint(-6, 6))

        dets: list[dict[str, Any]] = [
            {"label": surface, "score": _hi(), **trusted},
            {"label": surface, "score": _lo(), **untrusted},
        ]
        for lab, (cx, cy), sc in [(answer, _in(trusted), _lo()),
                                  (decoy, _in(untrusted), _lo()),
                                  (confident, _in(trusted), _hi())]:
            dets.append({"label": lab, "score": sc, "cx": cx, "cy": cy, "w": 8, "h": 8})
        for lab in offs:                                       # uncertain, but nowhere near
            dets.append({"label": lab, "score": _lo(), "cx": rng.randint(45, 55),
                         "cy": rng.randint(5, 15), "w": 8, "h": 8})

        # recompute the answer from the symbols, the way the learner has to; an
        # episode whose reading is not unique is thrown away rather than labelled
        anchors = [d for d in dets if d["label"] == surface and d["score"] >= tau]
        if len(anchors) != 1:
            continue
        uncertain_inside = [d for d in dets if d["score"] < tau and d["label"] != surface
                            and _inside(d, anchors[0])]
        if len(uncertain_inside) != 1:
            continue
        obs = Rec(detections=Lst([Pred("det", Ident(d["label"]), Num(d["score"]), Num(d["cx"]),
                                       Num(d["cy"]), Num(d["w"]), Num(d["h"]))
                                  for d in _shuffled(rng, dets)]),
                  trust_threshold=Num(tau),
                  query=Pred("uncertain_on", Ident(surface)))
        return (obs, _shuffled(rng, [answer, decoy, *offs, confident]),
                uncertain_inside[0]["label"],
                {"threshold": tau, "surface": surface, "decoy_answer": decoy,
                 "confident_distractor": confident})
    raise RuntimeError("multimodal_symbolization: no admissible world")


class MultimodalSymbolization(Lesson):
    """Perception symbols carrying uncertainty."""

    id = "multimodal_symbolization"
    level = 55
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "perception symbols carrying uncertainty"
    capabilities = ('spatial_reasoning', 'lexical_grounding')
    axes = {'world_complexity': 3, 'ambiguity': 3, 'compositional_depth': 3}

    generate = staticmethod(gen_multimodal_symbolization)
