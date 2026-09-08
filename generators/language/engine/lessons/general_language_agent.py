"""``general_language_agent`` — a fresh lesson family and answer alphabet every episode.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Rec, Tok
from ..lesson import Lesson
from ..generators.ontology import _OWN_IDS, _STATE, _local_episode, _shuffled


def gen_general_language_agent(rng: random.Random, ctx):
    """A fresh lesson family per episode, with its answer alphabet made explicit.

    The registry is read at *call* time (never at import time, which would be
    circular) and this module's own lessons are excluded, so the composition
    cannot recurse into itself. Other modules also contribute composing lessons,
    so a re-entrancy flag catches the *mutual* case too: drawn from inside
    another composer, this lesson falls back to a local, non-composing episode.
    A candidate family is regenerated once from the same child seed and dropped
    unless it reproduces exactly, so a nondeterministic lesson elsewhere in the
    registry cannot make this one nondeterministic.

    The legal answers travel in the observation as ``answer_options``: the
    contract the agent must read is part of the episode, not of the harness.
    """
    if _STATE["composing"]:                          # another composer drew us
        return _local_episode(rng)

    from ..registry import all_lessons              # local: the registry imports us

    # the difficulty knob is the floor on the level of the family drawn: at zero
    # the whole registry is in play, higher up only the deeper lessons are
    floor = ctx.at(0, 120, default=0)
    pool = [(k, l) for k, l in sorted(all_lessons().items())
            if k not in _OWN_IDS and l.status == "implemented" and l.level >= floor]
    picked = None
    _STATE["composing"] = True
    try:
        for _ in range(8):
            if not pool:
                break
            lid, lesson = pool[rng.randrange(len(pool))]
            seed = rng.getrandbits(63)
            try:
                sub_obs, vocab, ans, sub_hidden = lesson.invoke(random.Random(seed))
                again = lesson.invoke(random.Random(seed))
            except Exception:                        # a broken lesson must not break this one
                continue
            vocab = list(vocab)
            if str(again[0]) != str(sub_obs) or again[2] != ans:
                continue                             # nondeterministic elsewhere: skip
            if ans in vocab and len(vocab) >= 2 and sub_obs.type == "record":
                picked = (lid, sub_obs, vocab, ans, sub_hidden)
                break
    finally:
        _STATE["composing"] = False
    if picked is None:                               # pragma: no cover - registry present
        return _local_episode(rng)
    lid, sub_obs, vocab, ans, sub_hidden = picked

    fields = {str(k): v for k, v in sub_obs.value}
    if not all(k.isidentifier() for k in fields) or "query" not in fields:
        fields = {"episode": sub_obs, "query": Ident("answer")}
    fields["answer_options"] = Lst([Tok(a) for a in vocab])
    obs = Rec(**fields)
    hidden = {"family": lid, "level": lesson.level, "vocab": len(vocab),
              "sub_hidden": {k: str(v) for k, v in dict(sub_hidden).items()}}
    return obs, _shuffled(rng, vocab), ans, hidden


class GeneralLanguageAgent(Lesson):
    """A fresh lesson family and answer alphabet every episode."""

    id = "general_language_agent"
    level = 59
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "a fresh lesson family and answer alphabet every episode"
    capabilities = ('task_transfer', 'contract_inference', 'meta_generalization')
    axes = {'lexical_novelty': 4, 'grammar_complexity': 3, 'compositional_depth': 3, 'world_complexity': 3, 'reasoning_depth': 3}

    generate = staticmethod(gen_general_language_agent)
