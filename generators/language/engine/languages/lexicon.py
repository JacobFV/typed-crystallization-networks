"""The typed vocabulary a language pack carries, and how it loads from data.

The curriculum's generators coin their constants in English — ``cube``, ``red``,
``alice``, ``requires`` — because they have to coin them in *something*. A
language pack therefore keys its vocabulary by those English strings and stores
whatever its own grammar needs alongside the translation:

* a Spanish :class:`Noun` carries gender and plural, because an article and every
  adjective in the phrase have to agree with it;
* a Chinese :class:`Noun` carries a measure word, because you cannot count or
  point at it without one;
* a Spanish :class:`Adjective` carries four agreement forms; a Chinese one
  carries whether it takes ``的`` before a noun.

What a language does *not* need, it leaves empty. The realizer asks the pack for
a phrase and the pack uses the fields its grammar actually has.

Words the pack has never heard of pass through unchanged. That is not a
fallback so much as a requirement: most of what these lessons talk about is
invented per episode — three-letter nonce forms, object ids, freshly-named
predicates — and inventing a translation for a coined word would destroy the
lesson. A nonce word is the same nonce word in every language.

Vocabularies live as JSON under ``grammar/data/packs/`` and are loaded once,
lazily.
No dependency, no download, no model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = ["Noun", "Adjective", "Verb", "Vocabulary", "load_vocabulary", "DATA_DIR"]

#: One directory for every hand-written pack. There were three: the original
#: packs kept their vocabulary here, newer ones shipped beside their grammar,
#: and English, Spanish and Chinese had a file in each that was merged at load
#: time. Same kind of content, three places, two of them for historical
#: reasons -- so a reader counting the files could not tell how many languages
#: the package has.
DATA_DIR = (Path(__file__).resolve().parent.parent
            / "grammar" / "data" / "packs")


@dataclass(frozen=True)
class Noun:
    """A noun, with whatever the language needs to put it in a phrase."""

    lemma: str
    #: ``m`` or ``f`` where the language has grammatical gender, else ``""``
    gender: str = ""
    #: the plural form, where the language forms one
    plural: str = ""
    #: the measure word, where the language requires one to count or point
    classifier: str = ""

    def form(self, *, plural: bool = False) -> str:
        return (self.plural or self.lemma) if plural else self.lemma


@dataclass(frozen=True)
class Adjective:
    """An adjective in every form its language distinguishes."""

    #: the citation form, used when the adjective stands alone
    base: str
    #: masculine/feminine x singular/plural, for languages with agreement
    ms: str = ""
    fs: str = ""
    mp: str = ""
    fp: str = ""
    #: whether the language links it to its noun with a particle (Chinese ``的``)
    linker: bool = False

    def agree(self, gender: str = "m", *, plural: bool = False) -> str:
        """The form that agrees with a noun. Languages without agreement get
        ``base`` back whatever they ask for."""
        key = ("f" if gender == "f" else "m") + ("p" if plural else "s")
        return getattr(self, key, "") or self.base


@dataclass(frozen=True)
class Verb:
    """A verb in the forms the pack's clause templates use."""

    base: str
    infinitive: str = ""
    s3: str = ""
    p3: str = ""
    past3: str = ""
    #: an aspect particle appended after the verb, where the language uses one
    aspect: str = ""

    def finite(self, *, plural: bool = False, past: bool = False) -> str:
        if past and self.past3:
            return self.past3
        form = (self.p3 if plural else self.s3) or self.base
        return form + self.aspect


@dataclass(frozen=True)
class Vocabulary:
    """Everything a pack knows about open-class words, keyed by English."""

    nouns: Mapping[str, Noun] = field(default_factory=dict)
    adjectives: Mapping[str, Adjective] = field(default_factory=dict)
    verbs: Mapping[str, Verb] = field(default_factory=dict)
    names: Mapping[str, str] = field(default_factory=dict)
    words: Mapping[str, str] = field(default_factory=dict)

    def translate(self, key: str) -> str:
        """The citation form of a word, or the word itself if it is not ours.

        Order matters only for the rare word that is in two tables; nouns first
        because that is the commonest case in this curriculum.
        """
        n = self.nouns.get(key)
        if n is not None:
            return n.lemma
        a = self.adjectives.get(key)
        if a is not None:
            return a.base
        v = self.verbs.get(key)
        if v is not None:
            return v.base
        return self.names.get(key) or self.words.get(key) or key

    def forms(self, key: str) -> set[str]:
        """Every surface form this pack can produce for one source word.

        A word does not reach the page in one shape: a noun may be plural, an
        adjective may agree four ways, a verb may be finite. Anything checking
        that a word survived into the output has to know all of them.
        """
        out: set[str] = {key}
        n = self.nouns.get(key)
        if n is not None:
            out |= {n.lemma, n.plural}
        a = self.adjectives.get(key)
        if a is not None:
            out |= {a.base, a.ms, a.fs, a.mp, a.fp}
        v = self.verbs.get(key)
        if v is not None:
            out |= {v.base, v.infinitive, v.s3, v.p3, v.past3, v.finite(),
                    v.finite(plural=True), v.finite(past=True)}
        for table in (self.names, self.words):
            if key in table:
                out.add(table[key])
        return {f for f in out if f}

    def knows(self, key: str) -> bool:
        return (key in self.nouns or key in self.adjectives or key in self.verbs
                or key in self.names or key in self.words)

    def __len__(self) -> int:
        return (len(self.nouns) + len(self.adjectives) + len(self.verbs)
                + len(self.names) + len(self.words))

    def counts(self) -> dict[str, int]:
        return {"nouns": len(self.nouns), "adjectives": len(self.adjectives),
                "verbs": len(self.verbs), "names": len(self.names),
                "words": len(self.words), "total": len(self)}


_CACHE: dict[str, tuple[Vocabulary, dict[str, Any]]] = {}


def load_vocabulary(code: str) -> tuple[Vocabulary, dict[str, Any]]:
    """Load ``data/<code>.json``. Returns the vocabulary and the raw rest of it.

    The rest — field lead-ins, predicate words, query templates — belongs to the
    :class:`~langcurriculum.languages.base.Lexicon`, which the pack assembles.
    """
    if code in _CACHE:
        return _CACHE[code]
    path = DATA_DIR / f"{code}.json"
    if not path.exists():
        raise FileNotFoundError(f"no vocabulary data for {code!r} at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    vocab = Vocabulary(
        nouns={k: Noun(lemma=v["lemma"], gender=v.get("gender", ""),
                       plural=v.get("plural", ""), classifier=v.get("classifier", ""))
               for k, v in (raw.get("nouns") or {}).items()},
        adjectives={k: Adjective(base=v.get("base") or v.get("ms") or v.get("form", ""),
                                 ms=v.get("ms", ""), fs=v.get("fs", ""),
                                 mp=v.get("mp", ""), fp=v.get("fp", ""),
                                 linker=bool(v.get("attributive_de", False)))
                    for k, v in (raw.get("adjectives") or {}).items()},
        verbs={k: Verb(base=v.get("base") or v.get("form") or v.get("inf") or v.get("s3", ""),
                       infinitive=v.get("inf", ""), s3=v.get("s3", ""),
                       p3=v.get("p3", ""), past3=v.get("past3", ""),
                       aspect=v.get("aspect", ""))
               for k, v in (raw.get("verbs") or {}).items()},
        names=dict(raw.get("names") or {}),
        words=dict(raw.get("words") or {}),
    )
    _CACHE[code] = (vocab, raw)
    return _CACHE[code]
