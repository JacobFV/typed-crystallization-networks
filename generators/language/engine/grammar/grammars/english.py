"""English, as a set of parameters rather than a set of format strings.

The interesting thing about this file is how little is in it, and what little
there is. English is SVO, adjectives precede nouns, determiners precede
adjectives, it is prepositional, it fronts wh-words, and it has essentially no
case on nouns and no agreement worth the name. All of that is nine lines of
:class:`WordOrder` and an :class:`Alignment` that marks nothing.

Everything that is a *list* — the closed class, the relational lexicon, the
irregular plurals, the minimal pairs the morphology lessons build from — lives
in ``data/english.json`` beside the vocabulary. A table of words is data whoever
wrote it, and keeping it in Python meant that adding a language involved editing
code rather than supplying a file.

What remains here is what is genuinely a *rule*: the phonological ``a``/``an``
alternation, the pluralization patterns, and the fact that English forms polar
questions and negation by moving or attaching to a finite auxiliary. That last
is typologically unusual enough that the linearizer's defaults — a clause-final
particle, a clause-edge negator — are wrong for it. Three overrides. That is the
target every other grammar here should be measured against.
"""

from __future__ import annotations

import re

from ..category import N, NUM, PL
from ..features import FS
from ..linearize import NO_CASE, Alignment, Concord, Typography, WordOrder
from ..morphology import StoredMorphology
from ..syntax import Node
from .vocab import VocabularyGrammar

__all__ = ["English", "EnglishSynonym"]

#: article selection is phonological, so it is fixed after the words are chosen
_A_BEFORE_VOWEL = re.compile(r"\ba (?=[aeiou])")

#: Regular pluralization: a pattern, how many characters it replaces, and what
#: it appends. These are rules and stay here; the irregulars are a *list* and
#: live in the data file.
_PLURAL_RULES: tuple[tuple[str, int, str], ...] = (
    (r"(s|x|z|ch|sh)$", 0, "es"),
    (r"[^aeiou]y$", 1, "ies"),
    (r"[^f]f$", 1, "ves"),
)

#: English attaches negation and question formation to a finite auxiliary
_AUXILIARIES = (" is ", " are ", " was ", " were ", " has ", " have ")


#: Words that make what follows a clause rather than a noun phrase. English
#: only, and a heuristic: the tree cannot say, because a query head spelled
#: out is one lexical item however many words it holds.
_FINITE = frozenset({"is", "are", "was", "were", "has", "have", "had",
                     "should", "can", "must", "will", "would", "do", "does"})


def _is_clause(text: str) -> bool:
    words = text.lower().split()
    return bool(words) and (words[0] in _FINITE or any(w in _FINITE for w in words[1:]))


class English(VocabularyGrammar):
    """English prose. The reference grammar and the regression baseline."""

    code = "english"
    name = "English"
    pack = "english"

    order = WordOrder(
        clause="SVO", adj="AN", det="DN", numeral="NumN",
        adposition="pre", possessive="NG", label="LV", conditional="CA",
        wh_fronting=True, copula_overt=True,
        negation="aux",            # *is not yellow*, never *not is yellow*
    )
    typography = Typography()
    #: English marks no case on full noun phrases at all
    alignment = Alignment(case_of=NO_CASE)
    #: and agrees nothing with anything, inside the noun phrase
    concord = Concord()

    notes = (
        "SVO, determiner then adjective then noun, prepositional",
        "no case on nouns, no adjective agreement",
        "polar questions and negation attach to a finite auxiliary — the one "
        "typological oddity, and the reason for two of the three overrides",
        "indefinite article by phonology (a/an)",
        "regular -s/-es/-ies plural, with the irregulars supplied as data",
    )

    def __init__(self) -> None:
        super().__init__()
        self._irregular = dict(self.raw.get("irregular_plurals") or {})
        self.morphology[N.name] = StoredMorphology(rule=self.pluralize)

    def pluralize(self, lemma: str, feats: FS) -> str:
        """Regular English pluralization, applied only when a plural is asked for."""
        if feats.get_atom(NUM) != PL:
            return lemma
        if lemma in self._irregular:
            return self._irregular[lemma]
        for pattern, cut, ending in _PLURAL_RULES:
            if re.search(pattern, lemma):
                return (lemma[:-cut] if cut else lemma) + ending
        return lemma + "s"

    # ---- the three genuinely English things -----------------------------
    def sentence(self, text: str, end: str | None = None) -> str:
        """Fix ``a`` before a vowel once, after the words have been chosen."""
        return _A_BEFORE_VOWEL.sub("an ", super().sentence(text, end))

    def _split_auxiliary(self, text: str) -> tuple[str, str, str] | None:
        for aux in _AUXILIARIES:
            if aux in text:
                head, _, rest = text.partition(aux)
                return head, aux.strip(), rest
        return None

    def lin_Neg(self, node: Node, ctx: FS) -> str:
        """English negation attaches to a finite auxiliary, not to the clause.

        Neither ``pre`` nor ``post`` describes *the cube is not red*: the negator
        goes inside the predicate, after the copula. Languages that put it at one
        edge or the other are the common case and the parameter covers them;
        English is the outlier and pays for it here.
        """
        inner = node.arg("inner")
        assert inner is not None
        text = self.lin(inner, ctx)
        split = self._split_auxiliary(text)
        if split is None:
            return self.join(["does not", text])
        head, aux, rest = split
        return f"{head} {aux} not {rest}"

    def lin_WhQ(self, node: Node, ctx: FS) -> str:
        """``what is the X of Y?`` — English asks for a value with a copula.

        Fronting the wh-word is only half of it: English also needs a copula and
        a determiner the abstract tree does not supply, because most languages
        do not need them. A determiner-wh over a noun phrase is the shared
        construction and goes to the base.
        """
        body = node.arg("body")
        assert body is not None
        key = node.feats.get_atom("wh", "what")
        if key in self.WH_DETERMINERS and body.fn in ("NP", "CN"):
            return super().lin_WhQ(node, ctx)
        wh = self.cw(key, "what")
        inner = self.lin(body, ctx)
        if body.fn in ("PredAttr", "PredIdent", "PredRel", "PredLoc"):
            return self.join([wh, inner])
        if _is_clause(inner):
            # A label spelled out from a query head is a `Lex` whatever its
            # words say, so the shape of the tree does not tell a phrase from
            # a clause and the words have to. "which_claim_is_causal" asks
            # "which claim is causal", and putting a copula in front of it
            # gave "which is the claim is causal" -- two verbs and an article
            # nothing needs.
            return self.join([wh, inner])
        # the body may already carry its own determiner
        article = "" if inner.split(" ", 1)[0].lower() == "the" else "the"
        return self.join([wh, "is", article, inner])

    def lin_YNQ(self, node: Node, ctx: FS) -> str:
        """English fronts an auxiliary rather than appending a particle.

        Typologically this is the marked strategy — most languages use a particle
        or intonation — so the linearizer's default is the particle and English
        is the one that overrides. Getting that polarity right is the difference
        between a grammar engine and an English engine with translations bolted
        on.
        """
        body = node.arg("body")
        assert body is not None
        inner = self.lin(body, ctx)
        split = self._split_auxiliary(inner)
        if split is None:
            return self.join(["does", inner])
        head, aux, rest = split
        return self.join([aux, head, rest])


class EnglishSynonym(English):
    """English whose *question* uses near-synonyms held out of training.

    The asymmetry is the whole test. The body of the episode keeps the words a
    model was trained on and only the question switches — ``red`` becomes
    ``crimson``, ``cube`` becomes ``block`` — so a learner has to connect a word
    it has never read to one it has. Substituting in both places would make the
    episode *easier* than the default rather than harder, which is why the scope
    is the question and nothing else.
    """

    code = "english_synonym"
    name = "English (held-out synonyms)"
    pack = "english"

    notes = English.notes + (
        "the question's content words are near-synonyms held out of the body",
    )

    def __init__(self) -> None:
        super().__init__()
        self.synonyms = dict(self.raw.get("synonyms") or {})
        self._in_query = False

    def word(self, lemma: str, pos: str = "") -> str:
        surface = super().word(lemma, pos)
        return self.synonyms.get(surface, surface) if self._in_query else surface

    def question(self, node, ctx=None):
        """Substitute only while the question itself is being linearized."""
        from ..features import EMPTY
        previous, self._in_query = self._in_query, True
        try:
            return super().question(node, EMPTY if ctx is None else ctx)
        finally:
            self._in_query = previous
