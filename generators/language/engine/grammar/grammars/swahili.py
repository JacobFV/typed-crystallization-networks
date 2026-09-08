"""Swahili: eighteen noun classes, and why that needed no new mechanism.

Spanish has two genders and the previous realizer hard-coded them — a ``gender``
field with the values ``m`` and ``f``, an ``Adjective`` with exactly four
agreement slots. That design cannot represent Swahili, and the failure is not
one of degree. Bantu concord differs from Romance agreement in three ways at
once:

* there are **eighteen classes**, not two, and they are not semantic categories
  a lexicographer can guess — ``kitabu`` "book" is class 7 and ``mti`` "tree" is
  class 3 for historical reasons;
* the **plural is a different class**, not a different number. ``kitabu`` (7)
  pluralizes to ``vitabu`` (8) by *replacing its prefix*, and class 9 nouns like
  ``nyumba`` do not change at all;
* concord reaches **everything** — the adjective, the verb's subject marker, the
  demonstrative, the possessive, the relative — not just the article and the
  adjective.

Under the feature-structure design none of that needs new machinery. The class
is the value of :data:`~langcurriculum.grammar.category.CLS`, exactly where
Spanish puts ``"f"``. Unification propagates it. The concord tables below are
data. What Spanish and Swahili share is a mechanism; what differs is a table,
which is the correct place for a difference between two languages to live.

The vowel-initial alternation
-----------------------------

Swahili concord prefixes have two shapes depending on whether the adjective stem
begins with a vowel: class 7 is ``ki-`` before a consonant (``kitabu kikubwa``)
and ``ch-`` before a vowel (``kitabu chekundu``). Class 8 is ``vi-`` and ``vy-``.
This is phonology, and it is handled as such rather than by storing both forms.

Invariant loanwords
-------------------

A large and growing part of the Swahili adjective inventory does not agree at
all: ``buluu``, ``kijani``, ``manjano``, ``zambarau`` are borrowings and take no
concord prefix. The lexicon records this per adjective, because getting it wrong
in either direction produces something no speaker would write.

Not attempted
-------------

Verb morphology beyond the subject prefix — tense/aspect infixes, object
markers, the relative construction, the many derivational extensions
(applicative, causative, passive, stative). The curriculum's flat predicates do
not carry the argument structure those need, and inventing it would assert
things the English episode does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..category import CLS, N, NUM, PL, SG
from ..features import EMPTY, FS
from ..linearize import (
    NO_CASE, Alignment, Concord, Grammar, Typography, WordOrder,
)
from ..morphology import Morphology
from ..syntax import Node
from .vocab import VocabularyGrammar

__all__ = ["Swahili", "Concordance"]


# ======================================================================
# the rules. The tables they read are in data/swahili.json
# ======================================================================
_VOWELS = "aeiou"


@dataclass(frozen=True)
class Concordance:
    """Swahili's class system, as data: pairings, prefixes, allomorphs.

    Eighteen classes' worth of prefixes is a *table* — somebody looked each one
    up — and it belongs in the data file with the vocabulary. What stays in code
    is the handful of things that are rules over that table: which allomorph a
    nasal takes before a given consonant, and when class 5 shows its ``ji-``.
    """

    class_pairs: dict
    noun_prefix: dict
    adjective_prefix: dict
    subject_prefix: dict
    nasal_class: dict

    @classmethod
    def from_data(cls, raw: dict) -> "Concordance":
        spec = raw.get("concord") or {}
        return cls(class_pairs=dict(spec.get("class_pairs") or {}),
                   noun_prefix=dict(spec.get("noun_prefix") or {}),
                   adjective_prefix={k: tuple(v) for k, v in
                                     (spec.get("adjective_prefix") or {}).items()},
                   subject_prefix=dict(spec.get("subject_prefix") or {}),
                   nasal_class=dict(spec.get("nasal_class") or {}))

    def plural_of(self, cls_: str) -> str:
        return self.class_pairs.get(cls_, cls_)

    def nasal(self, stem: str) -> str:
        """Resolve the class 9/10 nasal against the consonant that follows it.

        Before a voiceless consonant the nasal simply does not surface, which is
        why *kubwa* and *kitabu kikubwa* look as though class 9 had no concord at
        all. It has; it is null in that environment and audible in others.
        """
        first = stem[:1]
        if first in "rl":
            return "nd"
        return self.nasal_class.get(first, "")

    def noun(self, cls_: str, stem: str) -> str:
        """The noun's own class prefix, with the one class that varies.

        Class 5 takes ``ji-`` before a monosyllabic stem and nothing otherwise:
        *ji-we* "stone" and *ji-cho* "eye", but *neno* "word". Storing both forms
        per noun would hide a rule that applies to every class-5 noun the
        vocabulary will ever gain.
        """
        if cls_ == "5":
            return "ji" if _syllables(stem) <= 1 else ""
        return self.noun_prefix.get(cls_, "")

    def adjective(self, cls_: str, stem: str) -> str:
        consonantal, prevocalic = self.adjective_prefix.get(cls_, ("", ""))
        if stem[:1] in _VOWELS:
            return prevocalic
        return self.nasal(stem) if consonantal == "N" else consonantal


def _syllables(stem: str) -> int:
    """Syllable count, approximated by vowel count — enough for the ji- rule."""
    return sum(1 for ch in stem if ch in _VOWELS)


class SwahiliNoun(Morphology):
    """The plural is a prefix substitution driven by the class pairing.

    ``kitabu`` (7) → ``vitabu`` (8); ``mtu`` (1) → ``watu`` (2); ``nyumba`` (9) →
    ``nyumba`` (10), unchanged, because classes 9 and 10 have the same prefix.
    Nothing here is stored: the stem and the class are, and the forms follow.
    """

    def __init__(self, stems: dict[str, tuple[str, str]],
                 concord: "Concordance"):
        self.concord = concord
        #: keyed by the **surface** singular, because that is what the
        #: linearizer has resolved the vocabulary key to by the time inflection
        #: is asked for
        self.stems = stems

    def inflect(self, lemma: str, feats: FS = EMPTY) -> str:
        entry = self.stems.get(lemma)
        if entry is None:
            return lemma
        stem, cls = entry
        target = self.concord.plural_of(cls) if feats.get_atom(NUM) == PL else cls
        return self.concord.noun(target, stem) + stem

    def forms(self, lemma: str) -> set[str]:
        return {lemma, self.inflect(lemma), self.inflect(lemma, FS({NUM: PL}))}


class Swahili(VocabularyGrammar):
    """Swahili. Eighteen-class concord over the same unification as Spanish."""

    code = "swahili"
    name = "Swahili (Kiswahili)"
    pack = "swahili"
    iso = "swh"

    order = WordOrder(
        clause="SVO",
        adj="NA",                  # kitabu kikubwa — adjective FOLLOWS the noun
        det="ND",
        numeral="NNum",            # vitabu vitatu
        adposition="pre",
        possessive="NG",           # kitabu cha mtoto
        label="LV",
        conditional="AC",          # kama ..., basi ...
        wh_fronting=False,         # wh in situ: "unataka nini?"
        copula_overt=True,         # invariant "ni"
        numeral_forces_plural=True,
        negation="pre",
    )
    typography = Typography(label_separator=":")
    alignment = Alignment(case_of=NO_CASE)
    #: the adjective copies the noun's class — the same declaration Spanish
    #: makes, over an inventory nine times larger
    concord = Concord(adjective=(CLS, NUM), predicative=(CLS, NUM))

    notes = (
        "SVO, adjective and determiner follow the noun",
        "eighteen noun classes; the plural is a class change, not a suffix",
        "class concord on adjectives, with the vowel-initial alternation "
        "(kitabu kikubwa / kitabu chekundu)",
        "invariant borrowed adjectives take no concord (buluu, kijani)",
        "invariant copula ni",
        "NOT attempted: verb tense/aspect infixes, object markers, relatives, "
        "or the derivational extensions",
    )

    def __init__(self) -> None:
        super().__init__()
        self.concord_tables = Concordance.from_data(self.raw)
        raw_nouns = self.raw.get("nouns") or {}
        self.morphology[N.name] = SwahiliNoun(
            {v["lemma"]: (v.get("stem", v["lemma"]), str(v.get("class", "9")))
             for v in raw_nouns.values()}, self.concord_tables)
        #: which adjectives agree at all
        self._agreeing = {k for k, v in (self.raw.get("adjectives") or {}).items()
                          if v.get("concord")}

    def _build_inherent(self) -> None:
        """A noun's class is its inherent feature — exactly where Spanish gender goes."""
        super()._build_inherent()
        for key, entry in (self.raw.get("nouns") or {}).items():
            cls = str(entry.get("class", ""))
            if cls:
                self.inherent[key] = FS({CLS: cls})

    # ---- the one genuinely Bantu thing ----------------------------------
    def concord_prefix(self, cls: str, plural: bool, stem: str) -> str:
        """The agreement prefix for a class, choosing the right allomorph."""
        target = self.concord_tables.plural_of(cls) if plural else cls
        return self.concord_tables.adjective(target, stem)

    def stem_after_concord(self, prefix: str, stem: str) -> str:
        """``-refu`` surfaces as ``ndefu``: the nasal prefix changes the stem's r."""
        if prefix == "nd" and stem[:1] in "rl":
            return "d" + stem[1:]
        return stem

    def inflect(self, cat: str, lemma: str, feats: FS) -> str:
        """Adjectives take a class prefix; nouns take their class's own prefix."""
        if cat != "A":
            return super().inflect(cat, lemma, feats)
        stem = self.word(lemma)
        if lemma not in self._agreeing:
            return stem                              # a borrowing: invariant
        cls = feats.get_atom(CLS)
        if not cls:
            return stem
        prefix = self.concord_prefix(str(cls), feats.get_atom(NUM) == PL, stem)
        return prefix + self.stem_after_concord(prefix, stem)

    def subject_prefix(self, cls: str, plural: bool) -> str:
        """The marker a verb carries for the class of its subject.

        Not yet wired into clause linearization — the curriculum's predicates
        supply a relational phrase rather than a verb stem, so there is nothing
        to prefix. Exposed because it is the same table, and because the moment
        the abstract syntax carries verb stems this is what it needs.
        """
        target = self.concord_tables.plural_of(cls) if plural else cls
        return self.concord_tables.subject_prefix.get(target, "")
