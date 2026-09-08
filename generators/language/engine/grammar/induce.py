"""Learning morphology from attested paradigms instead of authoring it.

The Turkish grammar in this package has a hand-written affix table: the plural
is ``lAr``, the accusative ``(y)I``, and an archiphonemic phonology resolves them
against the stem. That is the right way to do one language and an impossible way
to do a hundred and seventy-four. Nobody is going to hand-write a Finnish case
system, a Hungarian one, and a Yakut one, and if they did, nobody could check it.

So for every language that is not hand-written, morphology is **induced** from
attested (lemma, form, features) triples — 56.7 million of them. Two mechanisms,
in order of trust:

**Attested lookup.** If the database has the exact cell — this lemma, these
features — that form is used. It is not a guess; it is a fact somebody recorded.
This covers the curriculum's core vocabulary in the well-resourced languages.

**Analogical inflection.** For a lemma UniMorph never listed, the inflector
applies the transformation that lemmas *ending the same way* undergo. This is
the part that matters, and the conditioning is what makes it work: Turkish
plurals are ``-lar`` after a back vowel and ``-ler`` after a front one, so a rule
learned from whole stems would be wrong half the time, while a rule indexed on
the final characters recovers vowel harmony without being told harmony exists.

The same indexing recovers Finnish consonant gradation, German umlaut plurals,
Spanish ``-z``/``-ces``, and Swahili class prefixes, because all of them are
conditioned on material at the edge of the stem. It does not recover
non-concatenative morphology — Arabic root-and-pattern needs
:class:`~langcurriculum.grammar.morphology.TemplaticMorphology` and a root
lexicon — and the inducer reports that honestly rather than emitting a mangled
concatenation.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .category import CASE, CLS, DEF, DEGREE, MOOD, NUM, PERS, PL, SG, TENSE
from .features import EMPTY, FS
from .morphology import Morphology

__all__ = ["UM_FEATURES", "parse_unimorph", "Rule", "InducedParadigm",
           "DataMorphology", "induce"]


# ======================================================================
# UniMorph feature bundles -> engine features
# ======================================================================
#: The UniMorph schema is a flat set of tags; the engine's features are named
#: slots. This is the translation, restricted to the dimensions the linearizer
#: actually consults — carrying the rest would add cells nothing can select.
UM_FEATURES: Mapping[str, tuple[str, str]] = {
    # number
    "SG": (NUM, "sg"), "PL": (NUM, "pl"), "DU": (NUM, "dual"),
    # person
    "1": (PERS, "1"), "2": (PERS, "2"), "3": (PERS, "3"),
    # case — the inventory UniMorph uses, mapped onto the engine's names
    "NOM": (CASE, "nom"), "ACC": (CASE, "acc"), "GEN": (CASE, "gen"),
    "DAT": (CASE, "dat"), "ABL": (CASE, "abl"), "INS": (CASE, "ins"),
    "LOC": (CASE, "loc"), "ERG": (CASE, "erg"), "ABS": (CASE, "abs"),
    "VOC": (CASE, "voc"), "ESS": (CASE, "ess"), "TRANS": (CASE, "tra"),
    "COM": (CASE, "com"), "PRT": (CASE, "par"), "IN+ESS": (CASE, "ine"),
    "AT+ESS": (CASE, "ade"), "ON+ESS": (CASE, "sup"),
    "IN+ALL": (CASE, "ill"), "AT+ALL": (CASE, "all"), "ON+ALL": (CASE, "sub"),
    "IN+ABL": (CASE, "ela"), "AT+ABL": (CASE, "abl"), "ON+ABL": (CASE, "del"),
    "FRML": (CASE, "frm"), "PROPR": (CASE, "prp"),
    # gender / noun class
    "MASC": (CLS, "m"), "FEM": (CLS, "f"), "NEUT": (CLS, "n"),
    # definiteness
    "DEF": (DEF, "def"), "INDF": (DEF, "indef"),
    # tense and mood, for the verb cells
    "PRS": (TENSE, "pres"), "PST": (TENSE, "past"), "FUT": (TENSE, "fut"),
    "IND": (MOOD, "ind"), "SBJV": (MOOD, "subj"), "IMP": (MOOD, "imp"),
    # unmapped, "COND" left a Hungarian conditional looking like an unmarked
    # cell, so it matched a request for the plain present and *volna* was
    # offered as the copula
    "COND": (MOOD, "cond"), "POT": (MOOD, "pot"),
    # degree of comparison: an adjective asked for plainly must not come back
    # comparative, and unmapped these three were invisible
    "CMPR": (DEGREE, "cmpr"), "SPRL": (DEGREE, "sprl"),
}

#: bundles carrying any of these are skipped: they are derivational or
#: polarity-marked cells that the curriculum never asks for, and including them
#: makes the commonest-cell statistics meaningless.
_SKIP = frozenset({"NEG", "LGSPEC1", "LGSPEC2", "LGSPEC3", "NFIN", "V.PTCP",
                   "V.CVB", "V.MSDR", "PROPN"})


def parse_unimorph(bundle: str) -> FS | None:
    """One UniMorph bundle as an engine feature structure, or ``None`` to skip."""
    tags = bundle.split(";")
    if any(t in _SKIP for t in tags):
        return None
    out: dict[str, str] = {}
    for tag in tags[1:]:                       # tags[0] is the part of speech
        mapped = UM_FEATURES.get(tag)
        if mapped is not None:
            out[mapped[0]] = mapped[1]
    return FS(out) if out else None


# ======================================================================
# the learned transformation
# ======================================================================
@dataclass(frozen=True)
class Rule:
    """Strip ``cut`` characters from the end, append ``add``.

    Deliberately the simplest transformation that covers concatenative
    morphology with stem changes: a plain suffix is ``cut=0``, Spanish
    *lápiz*/*lápices* is ``cut=1, add="ces"``, and a prefixing language is
    handled by the mirrored fields.
    """

    cut: int = 0
    add: str = ""
    prefix_cut: int = 0
    prefix_add: str = ""

    def apply(self, stem: str) -> str:
        out = stem[:len(stem) - self.cut] if self.cut else stem
        out = out + self.add
        if self.prefix_cut:
            out = out[self.prefix_cut:]
        return self.prefix_add + out


def _learn(lemma: str, surface: str) -> Rule:
    """The transformation taking one lemma to one of its forms."""
    # longest common prefix, then whatever each side has left
    i = 0
    while i < min(len(lemma), len(surface)) and lemma[i] == surface[i]:
        i += 1
    if i >= len(lemma) * 0.4 or not lemma:
        return Rule(cut=len(lemma) - i, add=surface[i:])
    # the shared material is at the end instead: a prefixing language
    j = 0
    while j < min(len(lemma), len(surface)) and lemma[-1 - j] == surface[-1 - j]:
        j += 1
    if j >= len(lemma) * 0.4:
        return Rule(prefix_cut=len(lemma) - j, prefix_add=surface[:len(surface) - j])
    return Rule(cut=len(lemma), add=surface)          # suppletive: store it whole


def unimorph_tags(bundle: str) -> frozenset[str] | None:
    """The bundle as a tag set, minus the part of speech, or ``None`` to skip.

    Cells are keyed on this rather than on the parsed feature structure, and the
    distinction matters more than it looks. :func:`parse_unimorph` is *lossy* by
    design — it keeps the handful of dimensions the linearizer can select on and
    drops the rest. Keying cells on the lossy version merges every Hungarian
    possessive into one cell, so a single "plural" cell has to account for
    *könyvek*, *könyveim*, *könyveid* and *könyvei* at once, and learns none of
    them. Keying on the full tag set keeps them apart; selection then picks the
    least specific cell that satisfies the request, which is the bare plural.
    """
    tags = bundle.split(";")
    if any(t in _SKIP for t in tags):
        return None
    return frozenset(tags[1:])


@dataclass
class InducedParadigm:
    """The transformations for one cell, indexed by what the stem ends with.

    ``by_context`` maps a stem-final string of length 1..4 to the rule that the
    training lemmas ending that way underwent. Lookup tries the longest context
    first, so a rule conditioned on ``-ük`` beats one conditioned on ``-k``,
    which is how harmony and gradation come out right without being modelled.
    """

    cell: FS
    tags: frozenset = field(default_factory=frozenset)
    by_context: dict[str, Rule] = field(default_factory=dict)
    default: Rule = field(default_factory=Rule)
    support: int = 0

    #: how many stem-final characters the index is built over
    MAX_CONTEXT = 4

    def inflect(self, stem: str) -> str:
        for k in range(self.MAX_CONTEXT, 0, -1):
            rule = self.by_context.get(stem[-k:].lower())
            if rule is not None:
                return rule.apply(stem)
        return self.default.apply(stem)


def induce(pairs: Iterable[tuple[str, str, str]], *,
           min_support: int = 2) -> dict[frozenset, InducedParadigm]:
    """Learn one paradigm per feature cell from ``(lemma, surface, bundle)``.

    A cell is kept only if it is attested at least ``min_support`` times: a
    single example is as likely to be a typo or a lexicalized oddity as a rule,
    and a wrong rule applied to every noun in a language is much worse than a
    missing one.
    """
    cells: dict[frozenset, list[tuple[str, str]]] = defaultdict(list)
    parsed: dict[frozenset, FS] = {}
    for lemma, surface, bundle in pairs:
        tags = unimorph_tags(bundle)
        feats = parse_unimorph(bundle)
        if tags is None or feats is None or not lemma or not surface:
            continue
        cells[tags].append((lemma, surface))
        parsed[tags] = feats

    out: dict[frozenset, InducedParadigm] = {}
    for tags, examples in cells.items():
        feats = parsed[tags]
        if len(examples) < min_support:
            continue
        # a rule per context length, decided by majority among the lemmas
        votes: dict[str, Counter] = defaultdict(Counter)
        overall: Counter = Counter()
        for lemma, surface in examples:
            rule = _learn(lemma, surface)
            overall[rule] += 1
            low = lemma.lower()
            for k in range(1, InducedParadigm.MAX_CONTEXT + 1):
                if len(low) >= k:
                    votes[low[-k:]][rule] += 1
        if not overall:
            continue
        default = overall.most_common(1)[0][0]
        by_context: dict[str, Rule] = {}
        for context, counter in votes.items():
            rule, n = counter.most_common(1)[0]
            # keep a context only where it is both attested and informative
            if n >= min_support and rule != default:
                by_context[context] = rule
        out[tags] = InducedParadigm(feats, tags, by_context, default, len(examples))
    return out


# ======================================================================
# the morphology object the linearizer uses
# ======================================================================
#: Case tags a paradigm may carry, and the UniMorph spelling of each case the
#: engine asks for.
#: Enough of UniMorph's case inventory to tell a cell that names a case from
#: one that does not. Finnic and Ugric paradigms reach fifteen of them, and a
#: case missing here reads as "no case named", which is how the Finnish
#: allative came to pass for a citation form.
_CASE_TAGS = frozenset({"NOM", "ACC", "GEN", "DAT", "INS", "LOC", "PRP",
                        "VOC", "ABL", "ESS", "PRT", "ILL", "INE", "ADE",
                        "ALL", "ELA", "ABE", "TRA", "COM", "TERM", "EQTV",
                        "ABS", "ERG", "SUB", "DEL", "SUP", "SBL", "LAT",
                        "PRIV", "PROL", "FRML", "VERS", "REM", "PROX"})
_CASE_TAG = {"nom": "NOM", "acc": "ACC", "gen": "GEN", "dat": "DAT",
             "ins": "INS", "loc": "LOC", "voc": "VOC", "abl": "ABL"}

#: The same, for noun class.
_CLASS_TAGS = frozenset({"MASC", "FEM", "NEUT", "COM"})
_CLASS_TAG = {"m": "MASC", "f": "FEM", "n": "NEUT", "c": "COM"}


def _unambiguous(cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop any tag set that is recorded with more than one surface.

    Such a row is not evidence. It cannot say which form the features pick out,
    and taking either is a coin flip dressed as a paradigm.

    Adjectives only, which is a restriction and not a hedge. There the
    ambiguity is a merged *degree* -- one tag set holding the comparative and
    the superlative -- and neither member is the plain form always wanted, so
    dropping the row loses nothing. In a verb paradigm the same shape means two
    lexemes merged under one headword: Swedish ``V;PRS`` holds *är* from "to
    be" beside *varar* from "to last", and French ``faire`` beside ``être``.
    Dropping those rows cost three languages their copula -- French said
    "o1 fait une sphère" -- so the verb paradigms keep every row and lean on
    the scoring in `_copula_lemma` instead.

    Dutch is the clear case. Wiktionary gives *paars* the row
    ``A;INDF;NEUT;SG`` twice, once as *paarser* and once as *paarste* -- the
    comparative and the superlative, merged because the degree was never
    tagged. Asking for the plain adjective returned *paarser*, so a Dutch scene
    read "a purpler sphere". The rows that *are* tagged for degree carry one
    surface each and survive, which leaves exactly the good half of the table.
    """
    seen: dict[str, set[str]] = {}
    for feats, surface in cells:
        seen.setdefault(feats, set()).add(surface)
    return [(f, s) for f, s in cells if len(seen[f]) == 1]


class DataMorphology(Morphology):
    """Attested forms first, learned rules second.

    Holds no paradigm in memory beyond what it has been asked for: the lookup
    goes to SQLite, and the induced rules are built once per language on first
    use. A process rendering one language pays for one language.
    """

    def __init__(self, db, code: str, pos: str = "N", *, induce_limit: int = 3000):
        self.db = db
        self.code = code
        self.pos = pos
        self._induce_limit = induce_limit
        self._rules: dict[FS, InducedParadigm] | None = None
        self._cache: dict[tuple[str, FS], str] = {}

    def cells(self, lemma: str) -> list[tuple[str, str]]:
        """One lemma's paradigm, read in a single annotation scheme.

        The choice is made **per lemma**, not per language, and both halves of
        that matter. Pooling the two schemes lets an untagged Wiktionary row
        answer for a tagged UniMorph one and Swedish *gult* comes back
        *gulare*. But choosing once for the whole language is just as wrong in
        the other direction: UniMorph systematically omits suppletive
        auxiliaries, so a language that "has UniMorph" still has no copula
        there, and preferring it globally lost *ist*, *är*, *есть* — the very
        forms the second source was harvested to supply.

        So: the preferred scheme where this lemma is attested in it, the other
        where it is not.
        """
        preferred = self.db.paradigm(self.code, lemma, "unimorph")
        cells = preferred or self.db.paradigm(self.code, lemma, "wiktionary")
        return _unambiguous(cells) if self.pos == "A" else cells

    # ---- the learned half -------------------------------------------
    @property
    def rules(self) -> dict[frozenset, InducedParadigm]:
        if self._rules is None:
            self._rules = induce(self._training())
        return self._rules

    def _training(self) -> list[tuple[str, str, str]]:
        """Training pairs from **one** source, not a blend of two.

        UniMorph and Wiktionary annotate the same languages with different
        conventions, and pooling them puts two incompatible labellings behind
        one cell key — which halved held-out accuracy on Finnish and Turkish the
        moment the second source arrived. UniMorph is preferred where it exists
        because it is a paradigm resource with a fixed schema; Wiktionary fills
        in for the languages and the closed-class items it does not cover.
        """
        for source in ("unimorph", "wiktionary"):
            cur = self.db.conn.execute(
                "SELECT lemma, surface, feats FROM wordform "
                "WHERE code=? AND pos=? AND source=? LIMIT ?",
                (self.code, self.pos, source, self._induce_limit * 20))
            rows = [(r["lemma"], r["surface"], r["feats"]) for r in cur]
            if len(rows) >= 200:
                return rows
        return rows

    def _best_cell(self, feats: FS) -> InducedParadigm | None:
        """The most specific learned cell the request satisfies.

        Specificity is the number of features matched, so a request for
        accusative plural prefers the cell that fixes both over one that fixes
        only number — and falls back to the latter when the language does not
        distinguish them.
        """
        # A request that names no case still means the unmarked one, and a
        # rule learned from nominative cells records `CASE:nom` -- so it was
        # rejected for saying something the request did not, while a rule
        # learned from datives survived whenever its case tag failed to parse
        # and left the cell looking like a plain plural. Latvian answered
        # "kokiem", Estonian invented "arvateni". The nominative is admitted
        # here and preferred below; every other case still needs asking for.
        wants_case = feats.get_atom(CASE)
        best, best_key = None, (-1, -1, -(10 ** 6))
        for tags, paradigm in self.rules.items():
            cell = paradigm.cell
            if not cell:
                continue
            unmarked = 0
            mismatch = False
            for k, v in cell.items():
                if k is CASE and wants_case is None:
                    if v != "nom":
                        mismatch = True
                    else:
                        unmarked = 1
                    continue
                if feats.get_atom(k) != v:
                    mismatch = True
            if mismatch:
                continue
            matched = len(cell)
            # extra tags are dimensions the request said nothing about — a
            # possessive, an evidential, a definiteness the caller did not ask
            # for. Fewer of them means a cell closer to what was actually asked.
            extra = len(tags) - matched
            key = (matched, unmarked, -extra)
            if key > best_key:
                best, best_key = paradigm, key
        return best

    # ---- the interface ------------------------------------------------
    def inflect(self, lemma: str, feats: FS = EMPTY) -> str:
        # Noun class is selectable for an **agreeing** word and not for a noun.
        # A noun does not decline for gender, it *has* one: asking the inflector
        # for a masculine *casa* sends it looking for a cell that cannot exist
        # and it invents *casessa*. An adjective is the opposite case — class is
        # the only thing its agreement runs on, and leaving it out left German
        # *ein gelb Kubus* undeclined.
        selectable = (NUM, CASE, PERS, DEF, TENSE, MOOD)
        if self.pos != "N":
            selectable += (CLS, DEGREE)
        wanted = FS({k: v for k, v in feats.items()
                     if k in selectable and v is not None})
        if self.pos == "A" and DEGREE not in wanted and wanted:
            # An adjective asked for without a degree wants the plain one. Left
            # unsaid it does not disagree with the comparative, and Swedish
            # *gult* came back *gulare*.
            wanted = wanted.but(**{DEGREE: "pos"})
        if not wanted:
            return lemma
        key = (lemma, wanted)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if (self.pos == "N" and self._is_unmarked(wanted)
                and self.cells(lemma)):
            # The citation form *is* the unmarked cell, and the loader omits it
            # because it equals the lemma. So an all-default request can never
            # match exactly, and whatever it does match is over-specified:
            # Finnish offered N;1;NOM;SG — *kuutioni*, "my cube" — because the
            # possessive was the closest thing to a plain nominative singular
            # left in the table. Where the request asks for nothing marked, the
            # lemma is the answer and the table has nothing to add.
            out = lemma
        else:
            out = self._attested(lemma, wanted)
            if out is None:
                out = self._analogical(lemma, wanted)
        self._cache[key] = out
        return out

    #: the value each dimension takes when nothing has marked it
    UNMARKED = {NUM: "sg", CASE: "nom", PERS: "3", DEF: "indef",
                TENSE: "pres", MOOD: "ind", DEGREE: "pos"}

    def _is_unmarked(self, wanted: FS) -> bool:
        """Whether a request asks for the citation form and nothing more.

        Consulted for **nouns** only. An adjective's whole job is to agree, so a
        request naming its class is the most marked thing it can receive, not an
        unmarked one — treating the two alike silently undid agreement in
        Italian, Portuguese, Swedish and Greek at once.
        """
        return (all(self.UNMARKED.get(k) == v for k, v in wanted.items()
                    if k in self.UNMARKED)
                and not set(wanted) - set(self.UNMARKED))

    def _attested(self, lemma: str, wanted: FS) -> str | None:
        """A recorded cell that does not contradict the request.

        The distinction matters and cost a bug. A request for the third-person
        singular present *indicative* must still match a cell tagged only
        ``V;PRS;SG;3`` — the source simply did not mention mood, which is not the
        same as the cell being non-indicative. Requiring every requested feature
        to appear made the Russian copula ``есть`` invisible although it was
        sitting in the table.

        So a cell is rejected only where it **disagrees**, and among the
        survivors the one agreeing on most of the request wins, with the fewest
        unrequested extras breaking ties.
        """
        # An adjective in a case-marking language must match the case it was
        # asked for rather than merely fail to contradict it. Russian tags its
        # long forms ``A;FEM;NOM`` and its short forms ``A;MASC`` with no case
        # at all, so a request for the masculine nominative walked past the
        # absent long form and took the short one: scenes read *жёлт куб* where
        # *жёлтый куб* belongs. There is no ``A;MASC;NOM`` row because *жёлтый*
        # is the headword, and a paradigm's silence about its own citation form
        # is evidence rather than a gap — which is what :meth:`_analogical`
        # says two methods down and what this was quietly getting wrong.
        #
        # Only where the paradigm marks case at all. Italian, Portuguese,
        # Swedish and French mark none, so demanding an explicit case there
        # would send every adjective back to its citation form and undo
        # agreement in exactly the four languages an earlier attempt broke.
        # The same holds of the class, and for the same reason. Czech lists
        # ``A;FEM;NOM;SG`` and ``A;NEUT;NOM;SG`` and no masculine, because
        # *žlutý* is the headword — but it also lists a classless ``A;NOM``
        # whose surface is the neuter, and that answered every masculine
        # request: scenes read *žluté hranol* where *žlutý hranol* belongs.
        rows = self.cells(lemma)
        wanted_case = _CASE_TAG.get(wanted.get_atom(CASE) or "", "")
        wanted_class = _CLASS_TAG.get(wanted.get_atom(CLS) or "", "")
        # The test is whether the paradigm marks the very case or class being
        # asked for, not whether it marks any at all. Dutch *blauw* carries an
        # ``A;PRT`` row -- a partitive adjective -- and nothing else about
        # case, so "marks some case" was true, an explicit ``NOM`` was demanded
        # of a table that never mentions one, and every match was rejected: a
        # scene read "een paarse bol" and "een blauw bol" in the same breath.
        marks = {t for bundle, _s in rows for t in bundle.split(";")}
        strict_case = bool(self.pos == "A" and wanted_case in marks)
        strict_class = bool(self.pos == "A" and wanted_class in marks)

        best, best_score = None, (-1, -1, 1)
        for bundle, surface in rows:
            feats = parse_unimorph(bundle)
            tags = unimorph_tags(bundle)
            if feats is None or tags is None:
                continue
            if any(k in feats and feats.get_atom(k) != v for k, v in wanted.items()):
                continue
            raw = bundle.split(";")
            if strict_case and wanted_case not in raw:
                continue
            if strict_class and wanted_class not in raw:
                continue
            matched = sum(1 for k, v in wanted.items() if feats.get_atom(k) == v)
            if not matched:
                continue
            # Specificity is counted over the **raw** tags, not the parsed
            # features. Anything this module has no mapping for disappears from
            # the parse, so a possessive cell N;SG;PSS1S looked like a bare
            # singular and beat the real one: Finnish answered *kuutioni*, "my
            # cube". Counting what the source actually said means an unmapped
            # tag costs the cell instead of being invisible.
            # Where no case was asked for, prefer the unmarked one. Every
            # plural cell matches a bare NUM:pl request equally and scores
            # the same, so row order decided it: Finnish answered "luvuille",
            # the allative, Polish "liczbach", the locative, Czech "čísel",
            # the genitive -- nine of the fourteen case-marking languages
            # sampled put nouns in an oblique case for no reason. A cell that
            # names no case is unmarked too; that is the citation form.
            unmarked = 0
            if not wanted_case:
                # An explicit nominative outranks everything, and a cell
                # naming no case outranks one naming an oblique. Testing only
                # for membership of `_CASE_TAGS` was not enough: Finnish tags
                # its allative `ALL`, which is not in that set, so the cell
                # looked caseless, tied with the nominative, and won on row
                # order -- which is the bug, not a near miss of it.
                unmarked = (2 if "NOM" in raw
                            else 1 if not set(raw) & _CASE_TAGS else 0)
            # Before specificity, not after. Finnish writes its allative
            # `N;AT+ALL;PL`, the composite tag does not parse, so the cell
            # counts as *fewer* tags than `N;NOM;PL` and looked the more
            # specific of the two. Ranked after specificity the preference
            # never got a say; the oblique had already won.
            score = (matched, unmarked, -(len(tags) - matched))
            if score > best_score:
                best, best_score = surface, score
        return best

    def _analogical(self, lemma: str, wanted: FS) -> str:
        """Apply a learned rule — but only where the lemma has no paradigm.

        A lemma that *does* have a table and still matched no cell has almost
        always matched nothing because the requested cell is the **unmarked**
        one, and the unmarked cell is the citation form: Spanish lists
        ``amarilla``, ``amarillos`` and ``amarillas`` for *amarillo* and does
        not list *amarillo* again. Analogizing there is how a singular request
        came back ``amarillos``. Where a paradigm exists, its silence is
        evidence, not a gap to be filled by inference.
        """
        if self.cells(lemma):
            return lemma
        cell = self._best_cell(wanted)
        return cell.inflect(lemma) if cell is not None else lemma

    def forms(self, lemma: str) -> set[str]:
        out = self.db.surface_forms(self.code, lemma)
        for paradigm in self.rules.values():
            try:
                out.add(paradigm.inflect(lemma))
            except Exception:                            # pragma: no cover
                pass
        return {f for f in out if f}

    def coverage(self) -> dict[str, int]:
        """What the inducer actually learned, for the honesty tests."""
        return {"cells": len(self.rules),
                "contexts": sum(len(p.by_context) for p in self.rules.values()),
                "support": sum(p.support for p in self.rules.values())}
