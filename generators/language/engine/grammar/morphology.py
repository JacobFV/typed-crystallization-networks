"""Word formation: paradigms, affix slots, and an ordered phonological layer.

This module exists because storing inflected forms does not scale. It is a
perfectly good strategy for Spanish, where an adjective has four forms and a
noun two, and it is free for Chinese, which has one. It fails completely for the
languages this curriculum is not yet in: a Finnish noun has on the order of a
hundred and forty forms, a Turkish verb considerably more, and a Swahili verb
is a template with six agreement-sensitive slots. Those forms are not looked up.
They are *derived*.

So a morphology here is a function, and there are four kinds:

:class:`IsolatingMorphology`
    the identity. Chinese. Included not as a degenerate case but because
    "this language does not do that" has to be sayable in the same vocabulary as
    everything else, or the linearizer ends up full of special cases.

:class:`StoredMorphology`
    a table of forms with a rule for what is not in it. English and Spanish,
    where the irregulars are the interesting part and the regulars are a suffix.

:class:`ConcatenativeMorphology`
    ordered affix slots either side of a stem, each slot choosing its affix by
    feature match, with a phonological layer cleaning up the seams. Turkish,
    Swahili, Finnish, Hungarian, Japanese.

:class:`TemplaticMorphology`
    a consonantal root interleaved into a vowel pattern. Arabic and Hebrew,
    where ``k-t-b`` plus ``CaCaCa`` is ``kataba`` and no amount of concatenation
    will produce it.

Phonology
---------

Affixes are written in **archiphonemes** — segments underspecified for the
features harmony will fill in. The Turkish plural is ``lAr``, not ``lar`` and
``ler``; the accusative is ``(y)I``, four vowels and an optional buffer
consonant. :class:`Harmony` resolves them against the preceding surface string,
and it does so **cyclically**, as each affix is attached, because the vowel a
suffix harmonizes to may itself have been supplied by the suffix before it —
``ev-ler-i-mi-z-de`` resolves left to right and any other order gets it wrong.

:class:`PhonRule` handles what is left: voicing assimilation, degemination,
buffer-consonant insertion, final devoicing. Rules are **ordered**, and applied
in sequence, which is the one thing about the SPE tradition that turned out to
be straightforwardly right for engineering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .features import FS, EMPTY, subsumes

__all__ = [
    "Harmony", "PhonRule", "Phonology", "Affix", "Slot",
    "Morphology", "IsolatingMorphology", "StoredMorphology",
    "ConcatenativeMorphology", "TemplaticMorphology",
    "TURKISH_HARMONY", "TURKISH_PHONOLOGY",
]


# ======================================================================
# phonology
# ======================================================================
@dataclass(frozen=True)
class Harmony:
    """Resolve underspecified vowels against the last vowel of what precedes.

    A harmony system is described by the vowel inventory partitioned along the
    features that spread — backness always, rounding where the language has
    four-way harmony — and a table saying what each archiphoneme becomes in each
    resulting context.

    ``neutral`` names vowels that are transparent to harmony: Finnish ``i`` and
    ``e`` are front but do not make a back stem front, so they are skipped when
    looking backwards for the trigger.
    """

    #: every vowel of the language, for finding the trigger
    vowels: str
    #: the subset that is [+back]
    back: str
    #: the subset that is [+round]
    rounded: str = ""
    #: vowels skipped when looking for the harmony trigger
    neutral: str = ""
    #: archiphoneme -> (back, rounded) -> surface vowel
    table: Mapping[str, Mapping[tuple[bool, bool], str]] = field(default_factory=dict)
    #: what to assume when the stem has no vowel at all (an acronym, a symbol)
    default: tuple[bool, bool] = (True, False)

    def trigger(self, stem: str) -> tuple[bool, bool]:
        """The (back, rounded) context the next affix must harmonize to."""
        for ch in reversed(stem.lower()):
            if ch in self.neutral:
                continue
            if ch in self.vowels:
                return (ch in self.back, ch in self.rounded)
        # only neutral vowels: they still set backness in Finnish-type systems
        for ch in reversed(stem.lower()):
            if ch in self.vowels:
                return (ch in self.back, ch in self.rounded)
        return self.default

    def resolve(self, affix: str, stem: str) -> str:
        """Fill in every archiphoneme of ``affix`` from the context ``stem`` sets."""
        if not self.table:
            return affix
        ctx = self.trigger(stem)
        out = []
        for ch in affix:
            spec = self.table.get(ch)
            if spec is None:
                out.append(ch)
                continue
            v = spec.get(ctx) or spec.get((ctx[0], False)) or next(iter(spec.values()))
            out.append(v)
            # a resolved vowel becomes the context for the rest of the affix
            ctx = (v in self.back, v in self.rounded)
        return "".join(out)


@dataclass(frozen=True)
class PhonRule:
    """One ordered rewrite. ``repl`` may be a string or a match function."""

    pattern: str
    repl: Any
    name: str = ""

    def apply(self, s: str) -> str:
        return re.sub(self.pattern, self.repl, s)


@dataclass(frozen=True)
class Phonology:
    """An ordered rule sequence, applied at each morpheme boundary.

    Rules see a ``+`` marking the seam they were called about, and are expected
    to consume it; :meth:`apply` strips any that survive, so a rule set that does
    not care about boundaries needs to say nothing.
    """

    rules: tuple[PhonRule, ...] = ()
    harmony: Harmony | None = None

    def attach(self, stem: str, affix: str, *, prefix: bool = False) -> str:
        """Attach one affix and run the rules over the seam."""
        if self.harmony is not None and not prefix:
            affix = self.harmony.resolve(affix, stem)
        joined = f"{affix}+{stem}" if prefix else f"{stem}+{affix}"
        for rule in self.rules:
            joined = rule.apply(joined)
        return joined.replace("+", "")

    def apply(self, s: str) -> str:
        for rule in self.rules:
            s = rule.apply(s)
        return s.replace("+", "")


# ======================================================================
# affixes and slots
# ======================================================================
@dataclass(frozen=True)
class Affix:
    """One realization of one slot, selected when ``when`` subsumes the features.

    An empty ``form`` is a real and common answer — the Turkish nominative and
    the English singular are both marked by nothing, and saying so explicitly is
    better than leaving the slot out, because it documents that the slot was
    considered.
    """

    form: str
    when: FS = EMPTY
    #: higher wins when two affixes both match; lets a specific case override a default
    priority: int = 0


@dataclass(frozen=True)
class Slot:
    """A position in the word, filled by whichever of its affixes fits.

    Slots are ordered by ``order``, ascending outward from the stem, which is the
    order morphology actually stacks in: Turkish ``ev-ler-im-de`` is
    stem, number, possessive, case, and no language interleaves them freely.
    """

    name: str
    affixes: tuple[Affix, ...]
    order: int = 0
    prefix: bool = False
    #: when nothing matches, whether that is a defect or simply an empty slot
    required: bool = False

    def choose(self, feats: FS) -> str | None:
        best: Affix | None = None
        for a in self.affixes:
            if subsumes(a.when, feats) and (best is None or a.priority > best.priority):
                best = a
        if best is None:
            if self.required:
                raise KeyError(f"slot {self.name!r} has no affix for {feats!r}")
            return None
        return best.form


# ======================================================================
# the four kinds of morphology
# ======================================================================
class Morphology:
    """Turn a lemma plus a feature bundle into a surface form."""

    def inflect(self, lemma: str, feats: FS = EMPTY) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def forms(self, lemma: str) -> set[str]:
        """Every form this morphology can produce for a lemma.

        The test suite asserts that a word survives from the structure into the
        sentence, and it cannot do that without knowing the shapes the word may
        legitimately have taken on the way.
        """
        return {lemma}


class IsolatingMorphology(Morphology):
    """No inflection. The lemma is the word, whatever is asked of it."""

    def inflect(self, lemma: str, feats: FS = EMPTY) -> str:
        return lemma


class StoredMorphology(Morphology):
    """Forms from a table, with a rule for what the table does not list.

    ``table`` maps a lemma to a mapping from feature bundle to form. Lookup finds
    the most specific listed bundle that the requested features satisfy, so a
    table may list only the irregular cells and let the rule handle the rest —
    which is what an irregular *is*.
    """

    def __init__(self, table: Mapping[str, Mapping[FS, str]] | None = None,
                 rule: Callable[[str, FS], str] | None = None):
        self.table = dict(table or {})
        self.rule = rule or (lambda lemma, feats: lemma)

    def inflect(self, lemma: str, feats: FS = EMPTY) -> str:
        cells = self.table.get(lemma)
        if cells:
            best, best_size = None, -1
            for spec, form in cells.items():
                if subsumes(spec, feats) and len(spec) > best_size:
                    best, best_size = form, len(spec)
            if best is not None:
                return best
        return self.rule(lemma, feats)

    def forms(self, lemma: str) -> set[str]:
        out = {lemma, *(self.table.get(lemma) or {}).values()}
        return {f for f in out if f}


class ConcatenativeMorphology(Morphology):
    """Ordered affix slots around a stem, with phonology over the seams.

    This is the workhorse for agglutinative languages. Slots fire in order,
    each choosing its affix by feature match, and each attachment goes through
    the phonology so that harmony resolves against what has actually been built
    so far rather than against the bare stem.
    """

    def __init__(self, slots: Sequence[Slot], phonology: Phonology | None = None,
                 stems: Mapping[str, Mapping[FS, str]] | None = None):
        self.slots = tuple(sorted(slots, key=lambda s: s.order))
        self.phonology = phonology or Phonology()
        #: suppletive or otherwise irregular stems, keyed as in StoredMorphology
        self.stems = StoredMorphology(stems or {})

    def inflect(self, lemma: str, feats: FS = EMPTY) -> str:
        out = self.stems.inflect(lemma, feats)
        for slot in self.slots:
            form = slot.choose(feats)
            if not form:
                continue
            out = self.phonology.attach(out, form, prefix=slot.prefix)
        return out

    def forms(self, lemma: str) -> set[str]:
        """Every combination of one affix per slot — the paradigm, enumerated.

        Bounded deliberately: the point is to let a test recognize a form, not to
        materialize the hundreds of cells an agglutinative paradigm technically
        has, so a slot contributes its affixes independently rather than in every
        combination with every other.
        """
        out = {lemma, self.stems.inflect(lemma)}
        for slot in self.slots:
            for affix in slot.affixes:
                if not affix.form:
                    continue
                try:
                    out.add(self.phonology.attach(lemma, affix.form, prefix=slot.prefix))
                except Exception:                     # pragma: no cover - defensive
                    pass
        return {f for f in out if f}


class TemplaticMorphology(Morphology):
    """A consonantal root interleaved into a vowel pattern.

    ``patterns`` maps a feature bundle to a template in which ``C`` marks a root
    consonant slot: ``k-t-b`` with ``CaCaCa`` gives ``kataba``, with ``maCCuuC``
    gives ``maktuub``. Concatenation cannot express this, which is the whole
    reason the class exists — a template engine that assumes affixes glue to
    edges simply cannot produce Arabic.
    """

    def __init__(self, patterns: Mapping[FS, str],
                 roots: Mapping[str, Sequence[str]] | None = None,
                 phonology: Phonology | None = None):
        self.patterns = dict(patterns)
        self.roots = dict(roots or {})
        self.phonology = phonology or Phonology()

    def root_of(self, lemma: str) -> Sequence[str]:
        return self.roots.get(lemma) or tuple(c for c in lemma if c not in "aeiou")

    def inflect(self, lemma: str, feats: FS = EMPTY) -> str:
        best, best_size = None, -1
        for spec, pattern in self.patterns.items():
            if subsumes(spec, feats) and len(spec) > best_size:
                best, best_size = pattern, len(spec)
        if best is None:
            return lemma
        radicals = list(self.root_of(lemma))
        out, i = [], 0
        for ch in best:
            if ch == "C" and i < len(radicals):
                out.append(radicals[i])
                i += 1
            elif ch != "C":
                out.append(ch)
        return self.phonology.apply("".join(out))

    def forms(self, lemma: str) -> set[str]:
        return {lemma} | {self.inflect(lemma, spec) for spec in self.patterns}


# ======================================================================
# Turkish: the reference agglutinative system
# ======================================================================
#: Turkish four-way harmony. ``A`` is the two-way archiphoneme (a/e), ``I`` the
#: four-way one (ı/i/u/ü). Written out rather than computed so that the table is
#: readable as the paradigm it is.
TURKISH_HARMONY = Harmony(
    vowels="aeıioöuü",
    back="aıou",
    rounded="oöuü",
    table={
        # (back, rounded) -> vowel
        "A": {(True, True): "a", (True, False): "a",
              (False, True): "e", (False, False): "e"},
        "I": {(True, False): "ı", (False, False): "i",
              (True, True): "u", (False, True): "ü"},
    },
)

_VOICELESS = "fstkçşhp"
_VOICED_PAIR = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}


def _devoice(m: re.Match) -> str:
    """``D`` and ``C`` assimilate to the voicing of the segment before them."""
    prev, arch = m.group(1), m.group(2)
    voiceless = prev in _VOICELESS
    return prev + "+" + ({"D": "t", "C": "ç"}[arch] if voiceless
                         else {"D": "d", "C": "c"}[arch])


_TR_VOWELS = "aeıioöuü"


def _soften(m: re.Match) -> str:
    """Final-stop softening: ``kitap+I`` -> ``kitabı``, ``renk+I`` -> ``rengi``.

    Blocked in **monosyllables**, which is a real and unforgiving constraint:
    *at* "horse" is *atı*, not *adı*; *top* is *topu*, not *tobu*; *disk* is
    *diski*. Softening a monosyllable produces a different word or no word, so
    the rule counts the vowels before it fires.
    """
    stem = m.string[:m.start(1) + 1]
    if sum(1 for ch in stem if ch in _TR_VOWELS) <= 1:
        return m.group(1) + m.group(2)
    return _VOICED_PAIR[m.group(1)] + m.group(2)


TURKISH_PHONOLOGY = Phonology(
    harmony=TURKISH_HARMONY,
    rules=(
        # a parenthesised segment survives only where it is needed: the buffer
        # consonant of -(y)I appears after a vowel and is dropped after one
        PhonRule(r"([aeıioöuü])\+\(([a-zçğşöü])\)", r"\1+\2", "buffer kept after vowel"),
        PhonRule(r"\+\(([a-zçğşöü])\)", "+", "buffer dropped after consonant"),
        PhonRule(r"([a-zçğşöü])\+([DC])", _devoice, "voicing assimilation"),
        # ordered before the general rule: after a nasal, k softens to g rather
        # than to ğ — renk/rengi, denk/dengi, but ekmek/ekmeği
        PhonRule(r"([nm])k\+([aeıioöuü])", r"\1g+\2", "post-nasal k > g"),
        PhonRule(r"([pçtk])\+([aeıioöuü])", _soften, "intervocalic softening"),
        PhonRule(r"([aeıioöuü])\+([aeıioöuü])", r"\1y+\2", "hiatus breaking"),
    ),
)
