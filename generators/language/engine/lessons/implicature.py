"""``implicature`` — scalar inference from an explicit speaker policy.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS
from ..generators.semantics import _nonce_words, _shuffled


def gen_implicature(rng: random.Random, ctx):
    """A stated speaker policy makes the scalar inference exactly determined.

    The scale is a fresh set of nonce words meaning ``at least t`` for every
    ``t`` in ``0..n``, and the policy — given explicitly in the observation — is
    that the speaker utters the *strongest true* term. The literal content of
    the utterance leaves ``n - t + 1`` counts open; the implicature ("they would
    have said the stronger word if it were true") pins the count to exactly one.
    """
    n = rng.randint(*ctx.span((3, 5), (9, 13)))
    words = _nonce_words(rng, n + 1, 4)
    scale = dict(zip(words, range(n + 1)))              # word -> threshold
    k = rng.randint(0, n)
    said = words[k]                                     # the strongest true term
    prop = rng.choice(COLORS)

    lex = _shuffled(rng, [Pred("means", Ident(w), Pred("at_least"), Num(t))
                          for w, t in scale.items()])
    obs = Rec(scale=Lst(lex),
              policy=Lst([Pred("speaker", Ident("cooperative")),
                          Pred("policy", Pred("utters_strongest_true_term")),
                          Pred("alternatives", Pred("the_whole_scale"))]),
              context=Lst([Pred("domain", Ident("boxes"), Num(n)),
                           Pred("property", Ident(prop))]),
              said=Pred("said", Ident(said), Ident("boxes"), Ident(prop)),
              query=Pred("how_many", Ident(prop)))
    return obs, _shuffled(rng, list(range(n + 1))), k, {"threshold": scale[said],
                                                        "count": k, "domain": n}


class Implicature(Lesson):
    """Scalar inference from an explicit speaker policy."""

    id = "implicature"
    level = 26
    tags = ("pragmatics", "language-as-action")
    teaches = "scalar inference from an explicit speaker policy"
    capabilities = ('belief_modeling', 'multi_agent_coordination')
    axes = {'reasoning_depth': 4, 'ambiguity': 3, 'lexical_novelty': 3}

    generate = staticmethod(gen_implicature)
