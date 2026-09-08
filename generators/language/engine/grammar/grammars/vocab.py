"""Wiring the existing typed vocabularies into the grammar engine.

The hand-written packs carry 360 typed open-class entries each, loaded
from JSON, with the gender, plural, classifier and agreement forms their
languages need. None of that work is invalidated by moving to a grammar — it is
exactly the lexicon a grammar wants — so this module adapts it rather than
replacing it.

The adaptation is one idea: a vocabulary entry's stored fields become
**inherent features** on the lemma. A Spanish noun's ``gender`` becomes
``FS(cls="f")``; a Chinese noun's ``classifier`` becomes ``FS(clf="本")``; a
Swahili noun's class becomes ``FS(cls="7")``. From there the concord machinery
in :mod:`~langcurriculum.grammar.linearize` handles all three identically,
which is the whole argument of the rewrite in one function.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ...languages.lexicon import Vocabulary, load_vocabulary
from ..category import CLF, CLS
from ..features import FS
from ..linearize import Grammar, Sandhi
from ..derived import _is_affix
from ..typology import instructions_for

__all__ = ["VocabularyGrammar", "load_pack"]

#: distinguishes "not looked up yet" from "looked up and absent"
_UNSET = object()


def load_pack(code: str) -> tuple[Vocabulary, dict[str, Any]]:
    """Load a hand-written pack from ``grammar/data/packs``.

    One file per pack, and one place to look. There were two loaders and three
    directories, because the original packs kept their vocabulary where it
    started and later ones shipped beside their grammar -- so English, Spanish
    and Chinese each had two files merged at load time and nobody reading the
    tree could tell what was where.
    """
    return load_vocabulary(code)


class VocabularyGrammar(Grammar):
    """A grammar backed by one of the typed JSON vocabularies.

    The curated vocabulary is small and verified; the language database is large
    and scraped. Both are useful and the order between them is the whole point:
    **curated wins, and the database fills the gaps.** Turkish ships 115 of the
    curriculum's 403 keys by hand and the database has 224 of them, so consulting
    it roughly doubles what a hand-written grammar can say without touching a
    single one of its verified entries.
    """

    #: which data file to load; defaults to the grammar's code
    pack: str = ""

    #: ISO 639-3 code, for reaching the language database. Empty means the
    #: grammar is curated-only — English, whose database rows would be
    #: English-to-English and therefore empty anyway.
    iso: str = ""

    #: an extra data file merged over the pack's own, for material that used to
    #: live in the template module: English's field lead-ins and synonyms

    def __init__(self) -> None:
        super().__init__()
        self._database: Any = _UNSET
        self.vocabulary, self.raw = load_pack(self.pack or self.code)
        self.predicate_words = dict(self.raw.get("predicate_words") or {})
        self.field_intros = dict(self.raw.get("field_intros") or {})
        self.closed = dict(self.raw.get("closed") or {})
        # A hand-written grammar keeps its boundary rules in its own pack
        # rather than in the shared ISO-keyed table: it is identified by an
        # English name, and everything else it knows already lives here.
        self.instructions = {**instructions_for(getattr(self, "iso", "")),
                             **(self.raw.get("instructions") or {})}
        sandhi = self.raw.get("sandhi") or {}
        if sandhi:
            self.sandhi = Sandhi(elide=sandhi.get("elide", {}),
                                 contract=sandhi.get("contract", {}))
        self.paradigms = {k: ([tuple(x) if isinstance(x, list) else x for x in v]
                              if isinstance(v, list) else v)
                          for k, v in (self.raw.get("paradigms") or {}).items()}
        self._build_inherent()
        self._import_gaps()

    def _build_inherent(self) -> None:
        """Turn stored lexical fields into inherent features on the lemma."""
        for key, noun in self.vocabulary.nouns.items():
            feats: dict[str, Any] = {}
            if noun.gender:
                feats[CLS] = noun.gender
            if noun.classifier:
                feats[CLF] = noun.classifier
            cls = (self.raw.get("noun_class") or {}).get(key)
            if cls:
                feats[CLS] = str(cls)
            if feats:
                self.inherent[key] = FS(feats)

    # ---- lexical access -------------------------------------------------
    def _import_gaps(self) -> None:
        """Fill the curated vocabulary's gaps from the database, unambiguously.

        Two rules, and both are about not breaking the episode.

        **Only curriculum vocabulary.** A coined identifier must never be
        translated — the lesson turns on it — and some are short enough to
        collide with a real word: Spanish rendered the nonce ``nu`` as ``ni``,
        which is also what the nonce ``ni`` rendered as. Restricting the import
        to keys the curriculum actually coins means a minted token is never
        looked up at all.

        **No collisions.** A dictionary will happily give one word for two
        concepts — Turkish *para* is both *money* and *coin*, Swahili *sanduku*
        both *crate* and *box* — and an episode naming both becomes
        unanswerable. Where an imported form would collide, with another import
        or with a curated entry, it is dropped and the English passes through.
        A visibly untranslated word is a much smaller problem than an ambiguous
        one, which is the same trade the curated packs were built on.
        """
        self._imported: dict[str, str] = {}
        db = self.database
        if db is None:
            return
        from ..compile import curriculum_vocabulary
        from ..derived import _lemmas

        taken = {self.vocabulary.translate(k) for k in
                 (set(self.vocabulary.nouns) | set(self.vocabulary.adjectives)
                  | set(self.vocabulary.verbs) | set(self.vocabulary.words)
                  | set(self.vocabulary.names))}
        taken |= set(self.predicate_words.values())
        candidates: dict[str, str] = {}
        for key in sorted(curriculum_vocabulary()):
            if self.vocabulary.knows(key) or key in self.predicate_words:
                continue
            entry = db.lookup(self.iso, key)
            if entry is None or not entry.form:
                # The same citation form the derived grammars ask for. A
                # dictionary keys on `accept`, the curriculum coins `accepted`,
                # and asking only for the coined spelling left twenty Turkish
                # and seventeen Swahili words in English that the database
                # holds. The mapping is hand-written; only the target word
                # comes from the scrape, as every imported word does.
                lemma = _lemmas().get(key)
                entry = db.lookup(self.iso, lemma) if lemma else None
            if entry is None or not entry.form:
                continue
            form = entry.form
            # the screen the derived grammars apply: a table sometimes answers a
            # one-word question with an explanation rather than a word
            # the same screen the closed class applies. An affix is not a word:
            # Turkish has a locative case rather than a preposition and offers
            # "-da" for *at*, which was printed as its own token.
            if (form.count(" ") > 1 or "'" in form or form == key
                    or _is_affix(form)):
                continue
            candidates[key] = form

        # Two curriculum words rendering alike is a collision and both are
        # dropped; the same word twice is not. `bind` and `binds` are one
        # verb, and counting them as a clash cost Turkish bağlamak the moment
        # `binds` became resolvable -- a word going back to English while the
        # change around it was adding them.
        from collections import defaultdict

        by_form: dict[str, list[str]] = defaultdict(list)
        for key, form in candidates.items():
            by_form[form].append(key)
        for form, keys in by_form.items():
            if form in taken:
                continue
            if len({_lemmas().get(k, k) for k in keys}) > 1:
                continue
            for key in keys:
                self._imported[key] = form

    @property
    def database(self):
        """The language database, or ``None`` where it has not been built.

        Looked up lazily and cached: a grammar must stay usable with no database
        at all, since the hand-written five are the ones that work out of the box.
        """
        if self._database is _UNSET:
            self._database = None
            if self.iso:
                from ..store import LanguageDB
                db = LanguageDB()
                self._database = db if db.exists() else None
        return self._database

    def knows(self, lemma: str) -> bool:
        """``lookup`` hides an entry that maps a word to itself; this does not.

        English translates *green* as "green", and the entry exists precisely
        so the word is recognised. Reporting it unknown would leave English
        unable to say it knows its own vocabulary.
        """
        return (self.vocabulary.knows(lemma) or lemma in self._imported
                or lemma in self.predicate_words)

    def lookup(self, lemma: str, pos: str = "") -> str:
        """The open-class word, then the relational lexicon, then the database.

        ``predicate_words`` is consulted second because it is exactly a lexicon
        of relational predicates — 475 entries per pack, already authored — and a
        relational verb reaching this point is precisely what it is for. The
        lemma passing through unchanged is the last resort and the commonest
        outcome, because most of what this curriculum names is coined per
        episode.
        """
        relational = self.predicate_words.get(lemma)
        if pos == "V" and relational:
            return relational
        if self.vocabulary.knows(lemma):
            surface = self.vocabulary.translate(lemma)
            # A vocabulary entry that maps a word to itself is not a
            # translation — several are present only so the word is recognised.
            # Letting one shadow the relational lexicon is how ``left_of`` comes
            # out as "left_of" instead of "to the left of".
            if surface != lemma:
                return surface
        if relational:
            return relational
        return self._imported.get(lemma, "")

    def known(self, lemma: str) -> bool:
        return self.vocabulary.knows(lemma)

    def forms(self, lemma: str) -> set[str]:
        """Every surface form this grammar could produce for one source word.

        Union of what the vocabulary stores and what the morphology derives —
        the latter matters for agglutinative languages, where the stored form is
        one cell of a paradigm with hundreds.
        """
        out = set(self.vocabulary.forms(lemma))
        surface = self.word(lemma)
        for morph in self.morphology.values():
            out |= morph.forms(surface)
        return {f for f in out if f}

    def info(self) -> dict[str, Any]:
        base = super().info()
        base["vocabulary"] = self.vocabulary.counts()
        return base
