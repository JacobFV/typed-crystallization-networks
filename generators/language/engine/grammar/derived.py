"""Grammars assembled from data rather than written by hand.

A hand-written grammar — :mod:`~langcurriculum.grammar.grammars.turkish` is the
model — states its parameters, supplies a lexicon, and overrides the handful of
things its language does that the parameters cannot express. Writing one takes a
day and knowing whether it is *right* takes a speaker.

A **derived** grammar takes the same parameters from WALS, the same lexicon from
Wiktionary, and the same morphology from UniMorph, and assembles the object
automatically. It has no overrides, because nobody has looked at it. That is the
whole difference, and it is the reason derived grammars are labelled tier 2–4 and
hand-written ones tier 1: the machinery is identical, the *verification* is not.

What a derived grammar gets right
---------------------------------

Word order, adposition side, article inventory, alignment, concord presence,
classifier requirement, and inflection for any lemma UniMorph attests — all from
sources that coded them deliberately. For a language with dense coverage this is
a genuinely usable rendering.

What it gets wrong, and says so
-------------------------------

It has no phonology, so it cannot know that a Turkish suffix harmonizes; the
inducer recovers that statistically from stem endings, which works for the
attested inventory and degrades on novel stems. It has no idiomatic lead-ins, so
a section heading is the English field name until someone supplies one. It
cannot know that Turkish drops its copula, because WALS does not code copula
omission. Each of these is visible in :meth:`DerivedGrammar.gaps`, and the tests
assert that a grammar's declared tier matches what its data actually supports.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .category import A, ADV, CASE, CLS, N, NUM, V
from .features import EMPTY, FS
from .induce import DataMorphology
from .linearize import (
    ERG_ABS, NOM_ACC, NO_CASE, PREDICATE_GLOSS, Alignment, Concord, Grammar,
    Typography, WordOrder,
)
from .store import LanguageDB
from .typology import (
    SCRIPT_PUNCTUATION, SPACE_BEFORE_PUNCT, articles_for, classifier_for,
    copula_for, field_intros_for, instructions_for, sandhi_for,
)

__all__ = ["DerivedGrammar", "CLOSED_CLASS_KEYS"]

_ALIGNMENTS = {"NOM_ACC": NOM_ACC, "ERG_ABS": ERG_ABS, "NO_CASE": NO_CASE}


#: Tag requirements for the cells the lessons need. ``_NOUN_SG`` is a marker:
#: the singular is the headword and is never listed as a cell of its own.
_OBLIQUE = frozenset({"GEN", "DAT", "ACC", "ABL", "LOC", "INS", "VOC", "ESS",
                      "PRT", "INE", "ILL", "ADE", "ALL", "ABE", "TRANS", "COM"})
_NOUN_SG = object()
_NOUN_PL = (frozenset({"N", "PL"}), (_OBLIQUE - {"ACC"}) | {"DEF"})
_NOT_FINITE = frozenset({"SBJV", "COND", "IMP", "PASS", "NFIN", "V.PTCP", "V.CVB"})
_V3SG = (frozenset({"V", "PRS", "3", "SG"}), _NOT_FINITE)
_V3PL = (frozenset({"V", "PRS", "3", "PL"}), _NOT_FINITE)
_VPAST = (frozenset({"V", "PST"}), _NOT_FINITE | {"1", "2", "PL", "FEM", "NEUT"})

#: The people the episodes are about. Their names are not words of any
#: language and their genders are a fact about them, so every pack shares one
#: table rather than inventing its own.
_NAME_GENDER = {"alice": "f", "bob": "m", "carol": "f",
                "dave": "m", "erin": "f", "frank": "m"}

_LEMMA_DATA = Path(__file__).resolve().parent / "data" / "tables" / "lemmas.json"


@lru_cache(maxsize=1)
def _lemmas() -> Mapping[str, str]:
    """Curriculum word -> the citation form a dictionary keys on."""
    raw = json.loads(_LEMMA_DATA.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def probe_form(key: str) -> str:
    """What to look ``key`` up as. Itself, unless it is an inflected form.

    ``glows`` is not a word any dictionary lists, and it had a translation in
    no language at all while ``glow`` had one in thirty-two.
    """
    return _lemmas().get(key, key)


_SEED_DATA = Path(__file__).resolve().parent / "data" / "tables" / "paradigm_seeds.json"


@lru_cache(maxsize=1)
def _paradigm_seeds() -> Mapping[str, Any]:
    raw = json.loads(_SEED_DATA.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def _curriculum_keys() -> frozenset[str]:
    """The words the lessons can coin. Imported late; compile imports us."""
    from .compile import curriculum_vocabulary
    return frozenset(curriculum_vocabulary())


def _singulars(words: str) -> list[str]:
    """Singular forms of a field name worth trying, best first.

    Only ever *offered*: the caller keeps a candidate solely if it translates,
    so a word that is not a plural at all costs nothing. That guarantee is the
    important half. Stripping the final letter and using the result whatever
    came back turned *entities* into "entitie" in every language, and made
    "corpu" of *corpus*, "calculu" of *calculus* and "boxe" of *boxes* — all
    three of which the curriculum uses as headings.
    """
    last = words.rsplit(" ", 1)[-1]
    head = words[:len(words) - len(last)]
    out: list[str] = []
    if last.endswith("ies") and len(last) > 4:
        out.append(last[:-3] + "y")
    for ending in ("ses", "xes", "zes", "ches", "shes"):
        if last.endswith(ending):
            out.append(last[:-2])
            break
    # Greek and Latin plurals the curriculum heads blocks with. *hypotheses*
    # reached neither rule above -- "ses" offers "hypothes" and "s" offers
    # "hypothese", and Russian translates neither, so the heading stayed
    # English although гипотеза was sitting in the dictionary under
    # *hypothesis*. Offering costs nothing: a candidate is kept only if it
    # translates.
    if last.endswith("es") and len(last) > 4:
        out.append(last[:-2] + "is")            # hypotheses, analyses, bases
    if last.endswith("i") and len(last) > 3:
        out.append(last[:-1] + "us")            # calculi, radii
    if last.endswith("s") and not last.endswith("ss"):
        out.append(last[:-1])
    return [head + o for o in out]


def _shared_prefix(lemma: str, form: str) -> int:
    """How much of its own lemma an inflected form keeps.

    Near zero for a suppletive paradigm, which is what a copula has.
    """
    n = 0
    for a, b in zip(lemma.lower(), form.lower()):
        if a != b:
            break
        n += 1
    return n

#: The engine's closed-class slots, and the English lemma whose Wiktionary
#: translation table supplies each. These are ordinary dictionary entries, so
#: the closed class comes from exactly the same source as the open one rather
#: than from a per-language table somebody would have to write a hundred times.
#: the character ranges each script occupies, for rejecting a translation
#: written in the wrong one — Romanian is listed with a Cyrillic *уну* beside
#: the Latin *unu*, and a Cyrillic article in a Latin-script episode is simply
#: the wrong row
_SCRIPT_RANGES = {
    "Cyrl": ("\u0400", "\u04ff"), "Grek": ("\u0370", "\u03ff"),
    "Arab": ("\u0600", "\u06ff"), "Hebr": ("\u0590", "\u05ff"),
    "Deva": ("\u0900", "\u097f"), "Armn": ("\u0530", "\u058f"),
    "Geor": ("\u10a0", "\u10ff"),
}


def _in_script(form: str, script: str) -> bool:
    """Whether a form is written in the script its language uses."""
    letters = [c for c in form if c.isalpha()]
    if not letters:
        return False
    expected = _SCRIPT_RANGES.get(script or "Latn")
    if expected is None:                       # Latin and everything unlisted
        return not any(_SCRIPT_RANGES["Cyrl"][0] <= c <= _SCRIPT_RANGES["Cyrl"][1]
                       for c in letters)
    lo, hi = expected
    return sum(lo <= c <= hi for c in letters) >= len(letters) / 2


def _diacritics(form: str) -> int:
    import unicodedata
    return sum(1 for c in unicodedata.normalize("NFD", form)
               if unicodedata.combining(c))


#: Every dash a dictionary marks an affix with. The screen knew only the
#: ASCII hyphen, so Korean's copula came through as *–당하다* -- an en dash,
#: and an affix all the same.
_DASHES = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uff0d"


def _is_affix(form: str) -> bool:
    """Whether the dictionary is offering a bound form rather than a word."""
    return bool(form) and (form[0] in _DASHES or form[-1] in _DASHES)


def _usable_copula(form: str) -> bool:
    """Whether a candidate is a word at all, before asking whether it is right.

    The scored path already refused a multi-word answer; the fallback taken
    when nothing scores did not, so a translation table's aside became the verb
    of every sentence. Chinese was copularised by "or implied", Irish by
    "bí cothrom le", Ancient Greek by "εἰμί +".
    """
    if not form or _is_affix(form) or " " in form or len(form) > 18:
        return False
    return not (set(form) & set("+/(),")) and any(c.isalpha() for c in form)


def _usable_word(form: str, english: str) -> bool:
    """Reject what a translation table offers that is not a word of the language.

    Three kinds of junk reach a closed-class slot and each produced a visible
    defect: an **affix** (Finnish gives ``-lla`` for *at*, since it has a case
    rather than a preposition, and it was being printed as a separate token), a
    **phrase** (*un certo*, *az ember*), and an **untranslated leak** — the
    English word echoed back, which is how *a cube jaune* happened.
    """
    if not form or form == english:
        return False
    if _is_affix(form) or " " in form:
        return False
    return len(form) <= 14 and not (len(form) == 1 and form.isupper())


CLOSED_CLASS_KEYS: Mapping[str, str] = {
    "the": "the", "a": "a", "not": "not", "and": "and", "or": "or",
    "of": "of", "if": "if", "then": "then", "to": "to", "at": "at",
    "what": "what", "which": "which", "who": "who", "where": "where",
    "when": "when", "why": "why", "how": "how",
    "all": "all", "some": "some", "none": "none", "most": "most",
    "every": "every", "no_quant": "no",
    "few": "few", "empty": "empty",
    "yes": "yes", "no": "no",
    # NOT "a". English "a" is, to a dictionary, the letter A and the musical
    # note, and its translation table is full of both — German came out with
    # *A* and *den*. The indefinite article is historically the numeral in most
    # languages that have one, which is what WALS 38A code 2 records, so "one"
    # is the key that actually retrieves *ein*, *un*, *uno*, *een*, *bir*.
    "a": "one",
    "step": "step", "round": "round", "case": "case", "block": "block",
    "trial": "trial", "turn": "turn", "stage": "stage", "rule": "rule",
    "is": "be", "are": "be",
}

#: Slots whose English key must be looked up as a **noun**. An ordinal row label
#: — *step 4*, *round 2*, *trial 7* — is a noun in that use, and the untyped
#: lookup returns whichever sense the dictionary lists first: German *round*
#: gives ``rund`` "circular" and *turn* gives a verb. Saying which part of
#: speech is wanted is the whole fix.
NOMINAL_SLOTS = frozenset({"step", "round", "case", "block", "trial", "turn",
                           "stage", "rule"})


class DerivedGrammar(Grammar):
    """A grammar for one language, assembled from the language database."""

    def __init__(self, db: LanguageDB, code: str):
        super().__init__()
        self.db = db
        self.code = code
        row = db.language(code)
        self.name = (row["name"] if row else code) or code
        self.tier = row["tier"] if row else 4
        self._params = db.profile(code)
        self._apply_profile(self._params)
        self._word_cache: dict[tuple[str, str], str] = {}
        self._copula: dict[bool, str] = {}
        self._warm_closed_class()
        self._ambiguous, self._ambiguous_gloss = self._find_collisions()
        self._build_articles()
        for category, pos in ((N, "N"), (A, "A"), (V, "V")):
            self.morphology[category.name] = DataMorphology(db, code, pos)
        self.paradigms = self._build_paradigms()
        self.notes = self._notes()

    # ---- parameters ------------------------------------------------------
    def _apply_profile(self, p: Mapping[str, Any]) -> None:
        self.order = WordOrder(
            clause=p.get("clause", "SVO"), adj=p.get("adj", "AN"),
            det=p.get("det", "DN"), numeral=p.get("numeral", "NumN"),
            adposition=p.get("adposition", "pre"),
            possessive=p.get("possessive", "GN"),
            label=p.get("label", "LV"),
            conditional=p.get("conditional", "CA"),
            wh_fronting=bool(p.get("wh_fronting", True)),
            copula_overt=bool(p.get("copula_overt", True)),
            numeral_forces_plural=bool(p.get("numeral_forces_plural", True)),
            negation=p.get("negation", "pre"),
        )
        self.alignment = Alignment(
            case_of=_ALIGNMENTS.get(p.get("alignment", "NO_CASE"), NO_CASE))
        shared = tuple(p.get("concord_adjective") or ())
        if not shared and self._lexicon_records_gender(p):
            shared = (CLS, NUM)
        self.concord = Concord(
            adjective=shared,
            predicative=tuple(p.get("concord_predicative") or ()) or shared)
        self.typography = Typography(
            word_joiner=p.get("word_joiner", " "),
            capitalizes=bool(p.get("capitalizes", True)),
            rtl=bool(p.get("rtl", False)),
            label_separator=":" if not p.get("word_joiner", " ") else "",
            space_before=SPACE_BEFORE_PUNCT.get(self.code, ""),
            **SCRIPT_PUNCTUATION.get(
                (self.db.language(self.code) or {"script": ""})["script"], {}),
        )
        self.sandhi = sandhi_for(self.code)
        self.instructions = instructions_for(self.code)
        # A section a speaker would introduce, introduced. Everything not
        # written down keeps the translated noun that `block_heading` builds,
        # which is what English does for the two hundred field names it does
        # not write an intro for either.
        self.field_intros = dict(field_intros_for(self.code))

    def _lexicon_records_gender(self, p: Mapping[str, Any]) -> bool:
        """Whether the dictionary says this language has gender when WALS did not.

        WALS 30A is **absent** for Italian, Portuguese, Romanian, Polish and
        Czech — not coded zero, simply not coded — and reading a missing value
        as "no genders" switched concord off for five major languages that
        plainly have it. The lexicon settles it: tens of thousands of their
        nouns carry a masculine or feminine tag, which no genderless language's
        entries would. An absent feature is a question, and this answers it from
        evidence rather than from the default.
        """
        if "30A" in (p.get("evidence") or {}):
            return False                       # WALS did code it; believe WALS
        n = self.db.conn.execute(
            "SELECT COUNT(*) FROM sense WHERE code=? AND pos='N' "
            "AND gender IN ('masculine','feminine','neuter')",
            (self.code,)).fetchone()[0]
        return n >= 100

    def _warm_closed_class(self) -> None:
        """One bulk query for the whole closed class, at construction."""
        for slot, english in CLOSED_CLASS_KEYS.items():
            pos = "N" if slot in NOMINAL_SLOTS else ""
            form = self._first_usable(english, pos)
            if form:
                self.closed[slot] = form
        # a language WALS says has no article must not acquire one from a
        # dictionary that happily translates "the" into a demonstrative
        if not self._params.get("has_definite", False):
            self.closed["the"] = ""
        if not self._params.get("has_indefinite", False):
            self.closed["a"] = ""
        elif self._params.get("indefinite_from_one"):
            self.closed["a"] = self._indefinite_article()
        else:
            # WALS 38A says the indefinite word is distinct from the numeral,
            # and nothing here knows which word it is. The numeral would be the
            # wrong one — Japanese 一つ is "one thing", not an article.
            self.closed["a"] = ""
        if not self.order.copula_overt:
            self.closed["is"] = self.closed["are"] = ""

    # ---- lexicon ---------------------------------------------------------
    def _find_collisions(self) -> tuple[frozenset[str], frozenset[str]]:
        """Curriculum words whose translations would be indistinguishable.

        French offers *donner* for both *give* and *hand*, and the taxonomy
        lesson lists both as separate rungs — so it printed the same premise
        twice and the distinction the episode turns on was gone. Turkish *para*
        is both *money* and *coin*; Swahili *sanduku* is both *crate* and *box*.

        The hand-written grammars have refused this since the import was
        written: where two keys would share a form, both are dropped and the
        English shows through, because a visibly untranslated word is a much
        smaller problem than an ambiguous one. Only the derived half was
        exempt, for no better reason than that it reads the database directly.

        Both sides go, not the loser of some tie-break. Keeping one would still
        leave a reader unable to tell whether *donner* was the word for *give*
        or the untranslated survivor of *hand*.
        """
        from .compile import curriculum_vocabulary

        keys = sorted(curriculum_vocabulary())
        # The one-word glosses too: a head is rendered by translating its
        # gloss, so *minus* is the spelling whose translation can collide, and
        # querying only curriculum keys never retrieves it.
        probes = sorted(set(keys) | {probe_form(k) for k in keys}
                        | {g for g in PREDICATE_GLOSS.values() if " " not in g})
        marks = ",".join("?" * len(probes))
        # Mirror LanguageDB.lookup exactly, including its fallback: asking for a
        # part of speech the language has no row for returns the untyped best
        # row instead. A detector that skipped that fallback disagreed with the
        # renderer and passed while Dutch still merged *behind* and *back*.
        # Exactly what `_best_form` will choose, computed once for the whole
        # probe set. Mirroring the lookup is the point: a detector that picks a
        # different row than the renderer agrees with itself and misses the
        # collision the reader sees.
        rows: dict[str, list] = {}
        for row in self.db.conn.execute(
                f"SELECT key, pos, form FROM sense "
                f"WHERE code=? AND key IN ({marks}) ORDER BY rank",
                (self.code, *probes)):
            rows.setdefault(row["key"], []).append((row["pos"] or "", row["form"]))

        def best(key: str, pos: str) -> str:
            listed = rows.get(key) or rows.get(probe_form(key)) or []
            for entry_pos, form in listed:
                if pos and entry_pos and entry_pos != pos:
                    continue
                form = self._variant(form)
                if self._speakable(form, key):
                    return form
            if not pos:
                return ""
            for _entry_pos, form in listed:
                form = self._variant(form)
                if self._speakable(form, key):
                    return form
            return ""

        def effective(key: str, pos: str) -> str:
            return (best(key, pos) or key).lower()

        # Predicate heads reach the page through their gloss, so they are
        # invisible to a pass over lexicon keys. Group them BY gloss: `imp` and
        # `implies` are one concept with two spellings, and flagging them as a
        # collision with each other -- an earlier attempt did -- forces a
        # perfectly good translation back into English for no reason.
        # Keyed by the word actually looked up, so a head whose gloss *is* a
        # curriculum word is the same concept as that word rather than a second
        # one. Keeping them apart made every language collide `claim` with
        # itself -- the key and the head `claims` both probe "claim" -- and
        # withhold a translation to resolve an ambiguity that did not exist.
        # A key that is looked up as itself is first class. One that borrows a
        # citation form is not: `paints` is only translatable at all because
        # `paint` is, and in German *Farbe* is both paint and colour -- so
        # admitting it made `color` ambiguous and cost a word that appears in
        # most scenes to gain one that appears in few. A borrowed lemma is
        # therefore taken only where it collides with nothing, and where it
        # does the borrower alone is dropped.
        concepts: dict[str, set[str]] = {k: {k} for k in keys
                                         if probe_form(k) == k}
        borrowers: dict[str, set[str]] = {}
        for key in keys:
            if probe_form(key) != key:
                borrowers.setdefault(probe_form(key), set()).add(key)
        for head, gloss in PREDICATE_GLOSS.items():
            if " " in gloss:          # a phrase will not collide with a token
                continue
            concepts.setdefault(gloss, set()).add(head)

        # Two different prohibitions, and they were one set until German
        # printed "rwzt mean grüne". *Mittel* translates device, tool and the
        # NOUN means, so the key `means` must not be looked up -- but the
        # predicate head spelled the same way is the verb, its gloss is *mean*,
        # and *bedeuten* collides with nothing. Blocking the head because its
        # noun twin is ambiguous threw away a translation the language had.
        blocked_keys: set[str] = set()
        blocked_glosses: set[str] = set()
        for pos in ("", "N", "A", "V"):
            forms: dict[str, list[str]] = {}
            for probe in concepts:
                forms.setdefault(effective(probe, pos), []).append(probe)
            for sharers in forms.values():
                if len(sharers) > 1:
                    for probe in sharers:
                        blocked_glosses.add(probe)
                        blocked_keys.update(m for m in concepts[probe] if m in keys)
            # Then the borrowers -- against the slots the first class holds and
            # against each other. Russian *пла́вать* is both float and swim, and
            # neither is a curriculum word in its own right, so checking only
            # against the first class let the two borrowers collide unseen.
            claimed: dict[str, list[str]] = {}
            for probe in borrowers:
                claimed.setdefault(effective(probe, pos), []).append(probe)
            for probe, members in borrowers.items():
                form = effective(probe, pos)
                if len(claimed[form]) > 1 or (form in forms
                                              and forms[form] != [probe]):
                    blocked_keys.update(members)
        return frozenset(blocked_keys), frozenset(blocked_glosses)

    def _spell_out(self, lemma: str, pos: str) -> str:
        """As the base does, unless the gloss collides with some other word.

        Then the English gloss goes out untranslated, for the same reason a
        colliding lexicon entry does. The two must stay distinguishable, and
        each falls back to a different English form, so they do.
        """
        if PREDICATE_GLOSS.get(lemma, "") in self._ambiguous_gloss:
            return PREDICATE_GLOSS.get(lemma, "")
        return super()._spell_out(lemma, pos)

    # ---- material for the lessons that are about morphology --------------
    def _build_paradigms(self) -> dict[str, Any]:
        """Inflected material for the seven lessons that are *about* inflection.

        Those lessons build their sentences out of real inflected words rather
        than translating at render time, so a language that supplies none of
        this is presented in English however good its grammar is.

        All or nothing across the whole set, not per table. A pack supplying
        its own nouns but falling back for its verbs would put half a sentence
        in each language, and the learner could not tell which half the
        question was about.

        Cells are taken from what UniMorph attests, selected by tag. Asking the
        morphology to inflect instead returns a near-miss when the exact cell
        is missing: Greek answered a request for the third plural with the
        first singular, and Romanian answered a plural with a genitive. Both
        are attested forms, and both would be the wrong word.
        """
        seeds = _paradigm_seeds()
        tables: dict[str, Any] = {}

        nouns = self._collect(seeds["noun_forms"], "N", _NOUN_SG, _NOUN_PL, 6)
        agree = self._collect(seeds["agreement_forms"], "V", _V3SG, _V3PL, 6)
        if not nouns or not agree:
            return {}
        tables["noun_forms"] = nouns
        tables["agreement_forms"] = agree

        for field, need in (("verbs", 6), ("intransitive_verbs", 6)):
            forms = self._single(seeds[field], "V", _VPAST, need)
            if not forms:
                return {}
            tables[field] = forms
        # "Adv" and "P" are the tags this lexicon uses. Asking untyped took the
        # first sense listed, and for *on* that is the switched-on adjective:
        # Spanish offered *encendido* and Czech *zapnuto* among its prepositions.
        for field, pos, need in (("adverbs", "Adv", 4), ("preposition_words", "P", 5)):
            forms = self._words(seeds[field], pos, need)
            if not forms:
                return {}
            tables[field] = forms

        # The pronouns are a separate question from the sentence material, and
        # conflating them cost Finnish, Turkish and Hungarian everything. All
        # three have complete tables and a single genderless third person --
        # hän, o, ő -- so requiring two distinct pronouns discarded six good
        # paradigms to protect one lesson. Their absence is recorded instead,
        # and the lesson that needs them declines on its own account.
        pronouns = {k: self.word(v) for k, v in seeds["pronouns"].items()}
        if len(set(pronouns.values())) == len(pronouns) and \
                not any(w == seeds["pronouns"][k] for k, w in pronouns.items()):
            tables["pronouns"] = pronouns
            # Names are the people the episode is about, not words of the
            # language, so every pack shares one table.
            tables["name_gender"] = dict(_NAME_GENDER)
        return tables

    def _cells(self, lemma: str) -> list[tuple[frozenset[str], str]]:
        return [(frozenset(f.split(";")), surface)
                for f, surface in self.db.paradigm(self.code, lemma)]

    @staticmethod
    def _cell(cells, need: frozenset[str], ban: frozenset[str]) -> str:
        for tags, surface in cells:
            # A dash is how the source writes "no form here", and one reached a
            # Czech sentence as a word: "carol - sledoval - dave - zase - pak -
            # on - -".
            if (need <= tags and not (tags & ban) and " " not in surface
                    and surface and not _is_affix(surface)):
                return surface
        return ""

    def _collect(self, pool: Sequence[str], pos: str, first, second,
                 need: int) -> list[tuple[str, str]]:
        """Pairs of contrasting cells, taking the first ``need`` that work."""
        out: list[tuple[str, str]] = []
        for english in pool:
            lemma = self.word(english, pos)
            if not lemma or " " in lemma or lemma == english:
                continue
            cells = self._cells(lemma)
            a = lemma if first is _NOUN_SG else self._cell(cells, *first)
            b = self._cell(cells, *second)
            # Identical members teach nothing: a head noun whose number cannot
            # be seen leaves the agreement question with no evidence in it.
            if a and b and a != b and (a, b) not in out:
                out.append((a, b))
            if len(out) == need:
                return out
        return []

    def _single(self, pool: Sequence[str], pos: str, spec,
                need: int) -> list[str]:
        out: list[str] = []
        for english in pool:
            lemma = self.word(english, pos)
            if not lemma or " " in lemma or lemma == english:
                continue
            form = self._cell(self._cells(lemma), *spec)
            if form and form not in out:
                out.append(form)
            if len(out) == need:
                return out
        return []

    def _words(self, pool: Sequence[str], pos: str, need: int) -> list[str]:
        """Uninflected material — an adverb, an adposition — straight from the
        lexicon, which is where a language without a paradigm for them keeps
        them."""
        out: list[str] = []
        for english in pool:
            form = self.lookup(english, pos) or self.cw(english)
            if form and form != english and " " not in form and form not in out:
                out.append(form)
            if len(out) == need:
                return out
        return []

    def _curriculum_coverage(self) -> int:
        """How many curriculum words this language has, in one query.

        Counted here rather than by asking :meth:`knows` four hundred times,
        because :meth:`gaps` runs during construction and every grammar pays
        for it.
        """
        keys = sorted(_curriculum_keys())
        # By the word each key is looked up as, which is not always the key:
        # `glows` is found under `glow`. Counting raw keys disagreed with
        # `knows` the moment the lemma table landed, which is what the test
        # pinning the two counts together is for.
        # Both spellings: the lookup tries the key first and only then the
        # citation form, so a key that happens to be listed itself counts even
        # when its lemma is not. Hindi lists `accepted` and not `accept`.
        probes = sorted(set(keys) | {probe_form(k) for k in keys})
        marks = ",".join("?" * len(probes))
        present = {r[0] for r in self.db.conn.execute(
            f"SELECT DISTINCT key FROM sense WHERE code=? AND key IN ({marks})",
            (self.code, *probes))}
        # Subtract only what was counted. Withholding a word that has no entry
        # in the first place is not a second loss, and taking the size of the
        # withheld set off the total counted one of them twice.
        return len({k for k in keys
                    if (k in present or probe_form(k) in present)
                    and k not in self._ambiguous})

    def _entry(self, lemma: str, pos: str = ""):
        """The dictionary row for a word, by whatever spelling it is listed as.

        One resolution, used by everything that asks the dictionary anything.
        The lookup probed the citation form and :meth:`features_of` did not, so
        a borrowed noun arrived with a translation and no gender: German wrote
        *ein grüner Scheibe* where *Scheibe* is feminine, and the concord
        machinery had nothing to agree with.
        """
        entry = self.db.lookup(self.code, lemma, pos)
        if entry is None:
            citation = probe_form(lemma)
            if citation != lemma:
                entry = self.db.lookup(self.code, citation, pos)
        return entry

    def knows(self, lemma: str) -> bool:
        """The database has a word for it, and it is one this grammar may use.

        A word withheld to keep two concepts apart is not known for this
        purpose either: offering it as an option would reintroduce exactly the
        ambiguity dropping it prevented.
        """
        return (lemma not in self._ambiguous
                and self._entry(lemma) is not None)

    def lookup(self, lemma: str, pos: str = "") -> str:
        """One word from the language database, screened for junk.

        Only the primitive. Composition around it — falling back to a phrase
        rendered token by token, then to a predicate head's English gloss —
        belongs to :meth:`Grammar.word`, which every grammar shares. This class
        used to reimplement that template, which is how it came to have a
        private near-copy of ``phrase`` that joined with a literal space and so
        put spaces into languages written without them.
        """
        if lemma in self._ambiguous:
            return ""                      # see _find_collisions
        # A predicate head whose gloss differs from its own spelling is an
        # abbreviation or an inflected form, not a word of English being used
        # as itself -- and a dictionary keyed on spelling answers anyway.
        # German gave *U-Boot* for `sub`, Spanish *submarino* and *zas* for
        # `pow`, and every language offered a small demon for `imp`. The gloss
        # is what the head means; the spelling is a coincidence.
        gloss = PREDICATE_GLOSS.get(lemma)
        if gloss is not None and gloss != lemma:
            return ""
        return self._best_form(lemma, pos)

    # ---- choosing among what the dictionary offers ------------------------
    def _variant(self, form: str) -> str:
        """One writing of a Chinese entry, not both at once.

        Wiktionary gives a Han-script headword as "Traditional /Simplified" --
        *詞典 /词典* for dictionary -- and there are sixty-two thousand such
        rows across Mandarin, Cantonese, Hakka, Wu and Min. Printed whole they
        are not a word in any of them: a derived Mandarin scene read
        "o0是黃 /黄立方體 /立方体".

        Which side depends on how the language is written, which the database
        already records.
        """
        if " /" not in form:
            return form
        left, _, right = form.partition(" /")
        script = (self.db.language(self.code) or {"script": ""})["script"]
        return left if script == "Hant" else right

    @staticmethod
    def _speakable(form: str, lemma: str) -> bool:
        """Whether a dictionary answer is a word this engine can print.

        Three ways it is not, each of which produced a visible defect. A
        **gloss** rather than a word: English *turn* came back "be one's turn",
        and a multi-word answer to a single-word question is an explanation.
        An **affix**: Finnish offered "-lla" for *at* and it was printed as its
        own token. And **nothing at all**.

        Not the English word echoed back, which looks like a fourth and is
        not. French answers *cube* with "cube" and *opaque* with "opaque"
        because those are the French words, and refusing them promoted the
        wrong sense filed behind: the answer options offered *oranger* and
        *Apfelsinenbaum*, the orange **tree**. An echo is handled where it can
        be told apart -- see :meth:`_best_form`.

        Applied to what the dictionary said, never to what composition later
        builds: a phrase assembled token by token is legitimately several
        words, and screening it by this rule threw away every translation it
        produced.
        """
        if not form or _is_affix(form):
            return False
        return not (" " not in lemma and (form.count(" ") > 1 or "'" in form))

    def _candidates(self, lemma: str) -> list:
        """Every row worth trying, the word's own first and its citation form
        after.

        Both, and in that order. Trying the citation form only where the word
        has no rows at all was not enough: French lists exactly one row for
        *links* and it is "links", the English echoed back, so the word looked
        answered while *lien* sat under *link* one entry over.
        """
        rows = list(self.db.lookup_all(self.code, lemma))
        citation = probe_form(lemma)
        if citation != lemma:
            rows += self.db.lookup_all(self.code, citation)
        return rows

    def _best_form(self, lemma: str, pos: str) -> str:
        """The first usable row, preferring the part of speech asked for.

        The first *row* is not the first usable one, and taking it and giving
        up is how French lost a word it has. Its top row for *step* as a verb
        is "faire un pas" -- three words, an explanation rather than a lexeme,
        so the screen rejected it and the lookup returned nothing, while
        *marcher* and *pas* sat in the same table untried. Sixty-five word and
        part-of-speech pairs went that way in French alone.

        The closed class has chosen this way since French put *ne ... pas*
        ahead of *pas* for `not`; the open class read one row.
        """
        own = list(self.db.lookup_all(self.code, lemma))
        citation = probe_form(lemma)
        borrowed = (list(self.db.lookup_all(self.code, citation))
                    if citation != lemma else [])

        def first(rows: list, typed: bool) -> str:
            for entry in rows:
                if typed and pos and entry.pos and entry.pos != pos:
                    continue
                form = self._variant(entry.form)
                if self._speakable(form, lemma):
                    return form
            return ""

        best = first(own, True) or first(own, False)
        # An echo -- the English word handed back -- is only overruled by the
        # *citation form's* entry, never by another sense of the same one.
        # French lists a single row for *links* and it is "links", hiding the
        # *lien* filed under *link*; it also lists "cube" for *cube*, which is
        # simply the French word, and "orange", where the alternative sense in
        # the same entry is *oranger*, the orange **tree**. Going one entry
        # over separates a leak from a cognate; picking a different sense of
        # the same entry cannot.
        if best == lemma and borrowed:
            better = first(borrowed, True) or first(borrowed, False)
            if better:
                return better
        return best or first(borrowed, True) or first(borrowed, False)
        return form

    def _first_usable(self, english: str, pos: str = "") -> str:
        """The first candidate that is a word of the language, not just the first.

        Taking only the top-ranked entry and giving up when it fails the filter
        threw away perfectly good words sitting immediately behind it. French
        lists *ne … pas* first for *not* — a discontinuous negator that cannot
        occupy a single slot — and *pas* second, so the slot came out empty and
        French rendered "every prism is yellow" and "no prism is yellow"
        identically. An episode whose two candidate glosses collapse is not
        clumsy, it is unanswerable.
        """
        candidates = self.db.lookup_all(self.code, english)
        for entry in candidates:
            if pos and entry.pos and entry.pos != pos:
                continue
            if _usable_word(entry.form, english):
                return entry.form
        # Then without the part of speech, which is what the ordinary lookup
        # has always done when a language has no row of the kind asked for.
        # Insisting here left the slot empty while a good word sat in the
        # table: Greek tags *στρογγυλός* an adjective, so `round` had no noun
        # to be found and the English word was printed a hundred times in a
        # three-seed sweep. Hindi lost its article and its genitive the same
        # way. Preferring the part of speech is still right -- German *round*
        # as an adjective is *rund*, "circular", where the noun is *Kreis* --
        # and this only decides between a wrong-class word and none at all.
        if not pos:
            return ""
        for entry in candidates:
            if _usable_word(entry.form, english):
                return entry.form
        return ""

    def _indefinite_article(self) -> str:
        """The reduced numeral, not the impersonal pronoun.

        English *one* has two senses and both are translated: the numeral, and
        the impersonal subject of *one does not simply…*. The second gives Dutch
        ``je``, Romanian ``se``, German ``man`` — all short, so choosing by
        length alone picks them. Restricting to the **primary sense** keeps the
        numeral, and choosing the shortest of those then reduces the numeral to
        the article: *eins* to *ein*, *uno* to *un*.
        """
        rows = self.db.conn.execute(
            "SELECT form FROM sense WHERE code=? AND key='one' AND rank=0 "
            "AND pos='Card'", (self.code,))
        script = (self.db.language(self.code) or {"script": "Latn"})["script"]
        forms = [r["form"] for r in rows
                 if _usable_word(r["form"], "one") and _in_script(r["form"], script)]
        if not forms:
            return ""
        # A one-letter "article" is almost always the *name* of the numeral
        # rather than the article: Catalan lists u beside un. Prefer a real
        # word where there is one, then the shortest, then the form with the
        # fewest diacritics — Dutch writes één only when stressing the numeral,
        # and een unstressed is the article.
        real = [f for f in forms if len(f) > 1] or forms
        return min(real, key=lambda f: (len(f), _diacritics(f)))

    # ---- the article, from the dictionary's own gender tags ---------------
    #: The dictionary writes the Scandinavian and Dutch common gender as
    #: "common-gender", not "common", and the map had only the latter -- so
    #: forty-six thousand rows carried a class nobody read, every Swedish noun
    #: came out classless, and every one of them took the neuter article:
    #: "ett gul kub" where *kub* is common and wants *en*.
    _GENDERS = {"masculine": "m", "feminine": "f", "neuter": "n",
                "common": "c", "common-gender": "c", "plural": "pl"}

    def _build_articles(self) -> None:
        """Assemble a gendered article paradigm out of the translation table.

        Wiktionary translates *the* into German as three entries — ``der``
        tagged masculine, ``die`` feminine, ``das`` neuter — and into Spanish,
        French, Italian and Portuguese the same way. So the paradigm that makes
        the difference between *der Buch* and *das Buch* is already in the data,
        and needs reading rather than authoring.
        """
        self._articles: dict[tuple[str, str, bool], str] = {}
        written = articles_for(self.code)
        if written is not None:
            # The translation table carries one entry per gender and almost
            # never one per number, so the plural article was missing in every
            # language but French. Where the paradigm has been written out, it
            # is used whole rather than patched.
            for slot, cells in written.items():
                for spec, form in cells.items():
                    gender, _, number = spec.partition(".")
                    plural = number == "pl"
                    for cls in (("m", "f", "n", "c") if gender == "-"
                                else (gender,)):
                        self._articles[(slot, cls, plural)] = form
            return
        for slot, key in (("def", "the"), ("indef", "one")):
            if not self._params.get(f"has_{'definite' if slot == 'def' else 'indefinite'}"):
                continue
            best: dict[tuple[str, str, bool], str] = {}
            candidates = [e for e in self.db.lookup_all(self.code, key)
                          if key != "one" or e.pos in ("Card", "Det", "A", "")]
            for entry in candidates:
                gender = self._GENDERS.get(entry.gender or "", "")
                if not gender or not _usable_word(entry.form, key):
                    continue
                slots = ([(slot, "m", True), (slot, "f", True)] if gender == "pl"
                         else [(slot, gender, False)])
                for cell in slots:
                    # Shortest wins, as for the copula and for the same reason.
                    # An article is a worn-down numeral and the numeral is still
                    # in the table beside it: German lists *eins* as well as
                    # *ein*, Italian *uno* as well as *un*. The article is the
                    # reduced form, which is to say the shorter one.
                    if cell not in best or len(entry.form) < len(best[cell]):
                        best[cell] = entry.form
            self._articles.update(best)

    def numeral_phrase(self, count: str, head: Node | None, feats: FS) -> str:
        """``三个`` — the numeral and its classifier are one constituent.

        WALS says whether the language needs one and the profile has carried
        the answer all along; a derived grammar even announced "numeral
        classifiers obligatory" among its notes. Nothing wrote one, so
        Mandarin counted 三立方體.

        The general classifier only. A language that chooses by the shape of
        the thing counted -- and Mandarin does, 本 for books and 张 for flat
        things -- is served the over-general 个, which is what a speaker
        reaches for when the specific one will not come. A wrong specific
        classifier would be worse than a general one.

        Joined with the language's own word joiner, so Chinese and Thai write
        the two together and Vietnamese apart, without either being told to.
        """
        classifier = classifier_for(self.code)
        if not classifier or self._params.get("classifiers") != "obligatory":
            return count
        return self.join([count, classifier])

    def determiner(self, kind: str, head: Node | None, feats: FS) -> str:
        """The gendered article where the data supports one, else the bare word."""
        if kind not in ("def", "indef"):
            return ""
        gender = feats.get_atom(CLS, "") or ""
        plural = feats.get_atom(NUM) == "pl"
        if gender:
            hit = (self._articles.get((kind, gender, plural))
                   or self._articles.get((kind, gender, False)))
            if hit:
                return hit
        # No class recorded for this noun -- four in a hundred and ten of
        # German's, and a sixth of Portuguese's. The paradigm still has the
        # right word: Dutch *een* is the same for every class, and giving up
        # here left "o3 is paarse dennenappel" with no article beside
        # "o1 is een paarse bol" with one. Masculine is the conventional
        # default where the classes do differ, and it beats the closed class,
        # which holds the numeral: Spanish would say *uno cono*.
        fallback = (self._articles.get((kind, "m", plural))
                    or self._articles.get((kind, "m", False))
                    or next((form for (slot, _, pl), form
                             in self._articles.items()
                             if slot == kind and pl == plural), "")
                    or next((form for (slot, _, _pl), form
                             in self._articles.items() if slot == kind), ""))
        return fallback or self.cw("the" if kind == "def" else "a")


    def copula(self, kind: str, feats: FS) -> str:
        """The finite copula, inflected, rather than the dictionary's infinitive.

        Looking up "be" gives a citation form — *sein*, *être*, *ser*, *olmak* —
        and putting an infinitive where a finite verb belongs is the single most
        conspicuous thing a derived grammar can get wrong. UniMorph has the
        third-person singular present for every language it covers, so the
        paradigm answers instead of the headword.
        """
        if not self.order.copula_overt:
            return ""
        plural = feats.get_atom(NUM) == "pl"
        cached = self._copula.get(plural)
        if cached is not None:
            return cached
        # A written-down form wins. The paradigm data gets German *ist* and
        # Greek *είναι* right on its own, but it has no entry at all for a
        # suppletive Polish *jest*, and for Arabic it offers the past tense
        # where the present has no copula to offer.
        written = copula_for(self.code)
        if written is not None:
            self._copula[plural] = written["pl" if plural else "sg"]
            return self._copula[plural]
        lemma = self._copula_lemma()
        want = FS({"pers": "3", NUM: "pl" if plural else "sg",
                   "tense": "pres", "mood": "ind"})
        morph = self.morphology.get(V.name)
        # Attested cells only. The copula is suppletive in most of the world's
        # languages — *is* is not *be* plus a suffix — so the analogical
        # inflector, which learns edge transformations, produces confident
        # nonsense on exactly this verb. Where UniMorph has no paradigm for it
        # the citation form stands: a real word in the wrong cell beats an
        # invented one.
        form = morph._attested(lemma, want) if morph is not None else None
        self._copula[plural] = form or lemma
        return self._copula[plural]

    def _copula_lemma(self) -> str:
        """Pick the copula among the candidates the dictionary offers.

        Wiktionary lists several verbs under *be* — German *werden* as well as
        *sein*, Greek *ίσον* as well as *είμαι*, Turkish *imek* as well as
        *olmak* — and the first is often the wrong one. The tie-break is
        evidence rather than a guess: the copula is the most-inflected verb in
        almost every language that has one, so the candidate with the largest
        attested paradigm is overwhelmingly the right answer. Where no candidate
        is attested at all, the first stands, and the gap is reported by
        :meth:`gaps`.
        """
        # An affix is not a copula either. Korean's table offers *–당하다* and
        # *-기다*, both bound forms, and one of them stood as the verb in every
        # Korean scene.
        candidates = [e.form for e in self.db.lookup_all(self.code, "be")
                      if e.pos in ("V", "") and _usable_copula(e.form)]
        if not candidates:
            # Nothing offered is a word. Printing no copula is a shape the
            # linearizer already supports -- plenty of languages drop it -- and
            # is better than putting "nyob nov" where a verb belongs.
            return ""
        # mood is safe to request now that a cell is rejected only where it
        # *disagrees*: an untagged cell still matches, and one tagged
        # conditional no longer does
        want = FS({"pers": "3", NUM: "sg", "tense": "pres", "mood": "ind"})
        morph = self.morphology.get(V.name)
        scored: list[tuple[int, int, int, str]] = []
        for position, form in enumerate(dict.fromkeys(candidates[:10])):
            finite = morph._attested(form, want) if morph is not None else None
            # a copula is one word; " lenne or " is a parse artefact, not a verb
            if not finite or " " in finite or len(finite) > 12 or _is_affix(finite):
                continue
            scored.append((len(finite), _shared_prefix(form, finite), position, form))
        if scored:
            # Shortest wins, and the dictionary's own sense ranking breaks
            # ties. Length is not an arbitrary proxy: the copula is the most
            # frequent verb in any language that has one, and by Zipf's law the
            # most frequent words are the shortest. Wiktionary lists *werden*
            # beside *sein* and *venire* beside *essere* with nothing to tell
            # them apart, but *ist* is shorter than *wird* and *è* than
            # *viene*. Where two are the same length — Czech *je* beside *má* —
            # suppletion decides: a copula shares almost nothing with its own
            # infinitive (*být*/*je*, *sein*/*ist*, *essere*/*è*) while an
            # ordinary verb keeps its stem (*mít*/*má*). Both are universals
            # about frequency, and between them they pick the right verb in
            # every language checked.
            return min(scored)[3]
        return candidates[0]

    def known(self, lemma: str) -> bool:
        return self.db.lookup(self.code, lemma) is not None

    def inflect(self, cat: str, lemma: str, feats: FS) -> str:
        """As the base does, unless the rule destroys the word.

        The paradigms are induced by analogy, and an analogy drawn from a bad
        row can be arbitrarily destructive: Hindi turned *प्रिज़्म* into
        *प्film*, splicing Latin into the middle of a Devanagari word, and
        Arabic reduced twenty-nine nouns to a bare hyphen. Both had a perfectly
        good citation form a step earlier.

        So the result has to still be a word of the language's script. An
        uninflected noun is a visible, ordinary gap -- most of these languages
        have unattested paradigms anyway, and :meth:`gaps` says so. A word with
        another alphabet spliced into it is not a gap, it is wreckage.
        """
        surface = self.word(lemma, pos=cat)
        inflected = super().inflect(cat, lemma, feats)
        if inflected == surface:
            return inflected
        # A translation the paradigm data has never seen is not a stem to
        # reason from. Analogy invented *bla* for Italian *blu*, which is
        # invariable, and *átlátszatlanig* for Hungarian -- a case suffix
        # meaning "until" on an adjective. Every unattested adjective it
        # touched came out wrong, and most of the nouns.
        #
        # A word that passes through untranslated is the opposite case and is
        # deliberately left alone here: a coined nonce form has no dictionary
        # entry by definition, analogy is the only thing that can inflect it,
        # and inflecting it is what a morphology lesson is for.
        if surface != lemma and not self.db.paradigm(self.code, surface):
            return surface
        # An affix-shaped result is the same wreckage in the right alphabet:
        # Arabic reduced *كَعْبَة* to *كَ-*, which the script test happily
        # accepted because a prefix of an Arabic word is still Arabic.
        if _is_affix(inflected) and not _is_affix(surface):
            return surface
        script = (self.db.language(self.code) or {"script": "Latn"})["script"]
        if _in_script(surface, script) and not _in_script(inflected, script):
            return surface
        return inflected

    def features_of(self, lemma: str) -> FS:
        """Inherent features, where the dictionary records them.

        Wiktionary tags a translation with its gender for the languages that
        have one. Where it does not, the noun simply carries no class and the
        concord machinery leaves it alone — which is the correct behaviour for a
        language without gender and an honest one for a gap in the data.
        """
        entry = self._entry(lemma)
        if entry is None or not entry.gender:
            return EMPTY
        # One table, shared with the article builder. There were two, and they
        # had drifted: this one never learned the dictionary's spelling of the
        # common gender, so a noun could have an article chosen for a class the
        # noun itself was not given.
        mapped = self._GENDERS.get(entry.gender, "")
        return FS({CLS: mapped}) if mapped and mapped != "pl" else EMPTY

    def forms(self, lemma: str) -> set[str]:
        surface = self.word(lemma)
        out = {surface, lemma}
        for morph in self.morphology.values():
            out |= morph.forms(surface)
        return {f for f in out if f}

    # ---- section headings -------------------------------------------------
    def block_heading(self, name: str) -> str:
        """``Szene:``, ``escena:``, ``場面:`` — the field name, in the language.

        Not idiomatic: a speaker would write "In der Szene:", and nothing here
        knows how to build that. But a *noun* in the right language is strictly
        better than an English one, and the field names the curriculum uses —
        scene, rules, facts, premises, goal, state — are ordinary words the
        dictionary already has. Translating the head is the honest half of the
        job; the preposition and the article are the half still missing, and
        :meth:`gaps` says so.
        """
        words = name.replace("_", " ")
        tokens = words.split()
        if len(tokens) > 1:
            # A compound is translated a word at a time, each with its own
            # singular offered. Whole, "answer options" is in no dictionary,
            # and the generic phrase composer settles for whatever it can get:
            # a third of the compound headings read "Antwort options",
            # "отве́т options", "candidat examples". Half a heading in each
            # language is worse than either language on its own, and in pieces
            # both halves are in the dictionary -- *options* only under
            # *option*, which is why the singular has to be offered per word
            # and not just to the compound as a whole.
            parts = [self._heading_word(t) for t in tokens]
            if all(part != token for part, token in zip(parts, tokens)):
                return self.punctuate(self.join(parts) + self.typography.colon)
            return self.punctuate(words + self.typography.colon)
        return self.punctuate(self._heading_word(words) + self.typography.colon)

    def _heading_word(self, words: str) -> str:
        """One heading word, in the number the English field name is in.

        A dictionary lists the singular, so a plural field was labelled with
        one: German headed a list of rules *Regel*, French *règle*, Italian
        *regola*. The singular is what has to be looked up and not what should
        be printed — the language's own plural is a step away, through the
        same morphology that inflects everything else.
        """
        translated = self.word(words, pos="N")
        if translated != words:
            return translated
        for singular in _singulars(words):
            attempt = self.word(singular, pos="N")
            if attempt == singular:
                continue
            # Only a noun is pluralised. A field name ending in -s need not be
            # a plural at all: `knows` is a verb, and putting it through a
            # noun paradigm gave Russian *зна́ем*, "we know".
            entry = self._entry(singular, "N")
            if entry is not None and entry.pos == "N":
                plural = self.inflect(N.name, singular,
                                      FS({NUM: "pl", CASE: "nom"}))
                return plural or attempt
            return attempt
        return words

    # ---- honesty ---------------------------------------------------------
    def gaps(self) -> list[str]:
        """What this grammar does not know, stated rather than inferred."""
        out: list[str] = []
        p = self._params
        if p.get("order_uncertain"):
            out.append("WALS records no dominant word order; SVO assumed")
        if not p.get("evidence"):
            out.append("no typological coding at all; every parameter is a default")
        if not self.field_intros:
            out.append("section headings are the bare translated noun "
                       "(\u201cSzene:\u201d), not an idiomatic lead-in "
                       "(\u201cIn der Szene:\u201d)")
        else:
            out.append(f"{len(self.field_intros)} sections have an idiomatic "
                       f"lead-in; the rest are the bare translated noun")
        row = self.db.language(self.code)
        if row is not None and not row["n_forms"]:
            out.append("no UniMorph data; nouns are not inflected")
        if (self._params.get("has_indefinite")
                and not self._params.get("indefinite_from_one")):
            out.append("WALS records an indefinite word distinct from the "
                       "numeral and nothing here knows which word it is, so "
                       "no indefinite article is emitted")
        if not self._articles and self._params.get("has_definite"):
            out.append("the dictionary records no gender on this language's "
                       "article, so it does not agree with the noun")
        lemma = self._copula_lemma()
        if copula_for(self.code) is not None:
            pass                       # written down; nothing to report
        elif self.order.copula_overt and not lemma:
            out.append("the dictionary offers no single word for the copula, "
                       "so none is written")
        elif self.order.copula_overt and not self.db.paradigm(self.code, lemma):
            out.append(f"no attested paradigm for the copula {lemma!r}; "
                       f"the citation form is used")
        if row is not None and row["n_senses"] < 500:
            out.append(f"small lexicon ({row['n_senses']} senses); "
                       f"unknown words pass through in English")
        if not p.get("has_definite") and "37A" not in (p.get("evidence") or {}):
            out.append("WALS does not code 37A for this language, so no "
                       "definite article is emitted even if it has one")
        known = self._curriculum_coverage()
        total = len(_curriculum_keys())
        if known < total * 0.9:
            out.append(f"has a word for {known} of the {total} words the "
                       f"lessons can coin ({100 * known // total}%); the rest "
                       f"pass through in English")
        if self._ambiguous:
            out.append(f"{len(self._ambiguous)} words are withheld because the "
                       f"dictionary gives two of them the same form, and an "
                       f"ambiguous episode is worse than an English one")
        if self.concord.adjective:
            # Only the words that *are* nouns. Asking the lexicon for the noun
            # reading of a verb succeeds -- the lookup falls back when a part
            # of speech is missing -- so counting over every curriculum key
            # reported nine tenths of German ungendered when the figure is
            # four in a hundred and ten.
            from .compile import classify
            nouns = [k for k in _curriculum_keys()
                     if classify(k) == "noun" and self.lookup(k, "N")]
            bare = [k for k in nouns if not self.features_of(k)]
            if bare:
                out.append(f"{len(bare)} of {len(nouns)} translated nouns carry "
                           f"no gender in the dictionary, so a modifier has "
                           f"nothing to agree with and takes the default")
        if not self.predicate_words:
            out.append("relational predicates are composed word by word from "
                       "their English gloss (“left of” → the "
                       "words for “left” and “of”), which "
                       "is literal rather than idiomatic")
        return out

    def _notes(self) -> tuple[str, ...]:
        p = self._params
        coded = len(p.get("evidence") or {})
        notes = [
            f"derived grammar (tier {self.tier}), not hand-verified",
            f"{self.order.clause}, adjective {self.order.adj}, "
            f"{self.order.adposition}positional, "
            f"{'wh-fronting' if self.order.wh_fronting else 'wh in situ'}",
            f"{coded} WALS/Grambank features coded",
        ]
        if self.concord.adjective:
            notes.append(f"noun class concord over {p.get('n_classes')} classes")
        if p.get("classifiers") != "absent":
            notes.append(f"numeral classifiers {p.get('classifiers')}")
        notes.append("NOT attempted: " + "; ".join(self.gaps())
                     if self.gaps() else "NOT attempted: idiomatic phrasing")
        return tuple(notes)

    def info(self) -> dict[str, Any]:
        base = super().info()
        row = self.db.language(self.code)
        base.update({
            "tier": self.tier,
            "senses": row["n_forms"] if row else 0,
            "n_senses": row["n_senses"] if row else 0,
            "n_forms": row["n_forms"] if row else 0,
            "typology_features": len(self._params.get("evidence") or {}),
            "gaps": self.gaps(),
        })
        return base
