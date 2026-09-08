"""The linearizer: one walk over the abstract tree, parameterized by typology.

This is the module that has to earn the whole design. The claim is that word
order, agreement, case marking and question formation vary along a small number
of parameters, and that a single walk reading those parameters can produce
English, Turkish, Chinese and Swahili without any of them being a special case.
Where that claim fails, a grammar overrides a method — but the overrides should
be *few*, and each one should correspond to something genuinely idiosyncratic
about the language rather than to a gap in the parameterization.

The parameters
--------------

:class:`WordOrder` collects the ones that are pure ordering: whether the object
precedes or follows the verb, the adjective its noun, the possessor its
possessum, the adposition its complement. These are not eighteen independent
choices — Greenberg's correlations mean a verb-final language is usually
postpositional and genitive-initial — but they are recorded independently
because the correlations are tendencies and a grammar should be able to state
what its language actually does.

:class:`Alignment` is the one that is not about order at all: how a language
decides which argument gets which case. Nominative–accusative marks the agent of
a transitive clause the same as the sole argument of an intransitive one;
ergative–absolutive marks the *patient* the same as the sole argument. Both are
expressed here as a function from role and transitivity to case, so Basque and
Hindi need no new mechanism.

:class:`Concord` says which features a modifier must copy from its head. Spanish
copies class and number; Swahili copies class alone, from an inventory of
eighteen; Turkish copies nothing. The mechanism is the same unification in all
three, which is the point made at length in :mod:`~langcurriculum.grammar.features`.

What a grammar still has to write
---------------------------------

Vocabulary, closed-class words, the morphology object, and the handful of
overrides where the parameters genuinely do not reach — Chinese measure words,
Spanish ``ser``/``estar``, Turkish's question clitic. That is the bounded job
adding a language is supposed to be.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .category import (
    AGENT, ATTRIBUTE, CASE, CLS, DEF, GOAL, INDEX, LOCATION, NUM, PATIENT, PL,
    RECIPIENT, SG, SOURCE, THEME, VALUE,
)
from .features import EMPTY, FS, Var, unify
from .morphology import IsolatingMorphology, Morphology
from .syntax import Node

__all__ = ["WordOrder", "Typography", "Alignment", "Concord", "Grammar",
           "PREDICATE_GLOSS", "Sandhi",
           "NOM_ACC", "ERG_ABS", "NO_CASE"]


_PREDICATE_DATA = Path(__file__).resolve().parent / "data" / "tables" / "predicates.json"


def _load_predicate_gloss() -> Mapping[str, str]:
    raw = json.loads(_PREDICATE_DATA.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


#: predicate head -> the English words that mean it. See the data file: this is
#: a translation source, not any language's realization of the head.
PREDICATE_GLOSS: Mapping[str, str] = _load_predicate_gloss()


# ======================================================================
# parameters
# ======================================================================
@dataclass(frozen=True)
class WordOrder:
    """Where each dependent sits relative to its head."""

    #: one of SVO SOV VSO VOS OSV OVS
    clause: str = "SVO"
    #: ``AN`` adjective before noun, ``NA`` after
    adj: str = "AN"
    #: ``DN`` determiner before noun, ``ND`` after
    det: str = "DN"
    #: ``NumN`` numeral before noun, ``NNum`` after
    numeral: str = "NumN"
    #: ``pre`` prepositions, ``post`` postpositions
    adposition: str = "pre"
    #: ``GN`` possessor before possessum (Turkish, English -'s), ``NG`` after
    #: (English "of", Spanish "de")
    possessive: str = "GN"
    #: ``LV`` label before value ("weight: 3"), ``VL`` after
    label: str = "LV"
    #: ``CA`` consequent first ("a if b"), ``AC`` antecedent first ("if b, a")
    conditional: str = "CA"
    #: whether a content question fronts its wh-word, or leaves it in situ
    wh_fronting: bool = True
    #: whether the copula is pronounced at all in the present tense
    #: (Russian and Arabic drop it; Chinese uses one only for identity)
    copula_overt: bool = True
    #: whether a noun counted by a numeral goes plural. English and Spanish say
    #: yes; Turkish, Chinese, Japanese and Hungarian say no — *üç kitap*, not
    #: *üç kitaplar* — because the numeral has already expressed the plurality.
    numeral_forces_plural: bool = True
    #: where the negator sits relative to the predicate. ``pre`` for Chinese
    #: *不* and Spanish *no*, which negate the verb; ``post`` for Turkish
    #: *değil* and Japanese *-nai*; ``aux`` for English, whose negator follows
    #: the finite auxiliary rather than preceding it — *is not yellow*, not
    #: *not is yellow*. Three values because there are three behaviours, which
    #: is cheaper than the override English used to need.
    negation: str = "pre"

    @property
    def verb_final(self) -> bool:
        return self.clause in ("SOV", "OSV")

    def order_clause(self, subject: str, verb: str, obj: str) -> list[str]:
        """Arrange the three constituents according to ``clause``."""
        slot = {"S": subject, "V": verb, "O": obj}
        return [slot[c] for c in self.clause]


@dataclass(frozen=True)
class Typography:
    """How the script is written. Everything here is orthography, not grammar."""

    word_joiner: str = " "
    capitalizes: bool = True
    full_stop: str = "."
    question_mark: str = "?"
    #: an opening mark, for languages that bracket the interrogative clause
    question_open: str = ""
    list_separator: str = ", "
    clause_separator: str = "; "
    #: the enumerating separator, where it differs from the list separator
    item_separator: str = ""
    colon: str = ":"
    #: what separates a label from its value in a data row. English writes
    #: "weight 3" with nothing; Chinese and Turkish want the colon.
    label_separator: str = ""
    bullet: str = "  - "
    #: a Latin function call keeps half-width punctuation in any script
    arg_separator: str = ", "
    #: right-to-left scripts need the marks, not a reversed string
    rtl: bool = False
    #: marks this language sets off with a space *before* them. French writes
    #: "Dans la scène :" and "rwzt ?"; the cleanup that removes a stray space
    #: before punctuation is right everywhere else and wrong here, so the
    #: language says which marks it spaces rather than the rule guessing.
    space_before: str = ""
    #: what to put there. A no-break space, because the mark must not begin a
    #: line on its own.
    space_before_char: str = "\u00a0"


@dataclass(frozen=True)
class Sandhi:
    """What two words do to each other at the boundary between them.

    Two distinct processes, because languages have both and they interact:

    **Elision** — a function word loses its final vowel before a vowel-initial
    word. French *l'eau*, not *la eau*; Italian *l'albero*, not *il albero*.
    The conditioning is phonological, but the membership is lexical: only a
    listed set of words elide, and which ones differ by language.

    **Contraction** — a specific pair of words is written as one, regardless of
    what follows. Spanish *del*, not *de el*; German *im*, not *in dem*. Here
    the conditioning is not phonological at all; it is the identity of both
    words, so a vowel test would never find it.

    Contraction is checked first because it is the more specific statement: it
    names both words, where elision names one and a phonological class.

    Spanish is the useful control for elision. It has none, so its elision
    table is empty and vowel-initial words are left alone; *el agua* is an
    unrelated phenomenon — a feminine noun taking the masculine article before
    stressed /a/ — and modelling it here would wrongly catch *la avenida*.

    Known limit: French distinguishes mute *h*, which elides (*l'homme*), from
    aspirate *h*, which does not (*le héros*). The distinction is lexical, not
    predictable from spelling, and nothing in our lexicon records it. We elide
    before *h* because mute *h* is much the commoner class.
    """

    #: word -> what it becomes before a vowel, e.g. ``{"la": "l'"}``
    elide: Mapping[str, str] = field(default_factory=dict)
    #: ``"de el"`` -> ``"del"``; both words named, no phonology involved
    contract: Mapping[str, str] = field(default_factory=dict)
    #: what counts as vowel-initial in this orthography
    vowels: str = "aeiouhàáâäèéêëìíîïòóôöùúûüœæ"

    def __bool__(self) -> bool:
        return bool(self.elide or self.contract)

    def apply(self, left: str, right: str, joiner: str) -> str | None:
        """The pair written as one string if the boundary changes, else ``None``.

        ``left`` may already be several words — :meth:`Grammar.join` folds left
        to right — so the test is against its final token, and likewise the
        first token of ``right``. Whatever surrounds them is carried through
        untouched.
        """
        if not left or not right or not self:
            return None
        tail = left.rsplit(joiner, 1)[-1] if joiner else left
        head = right.split(joiner, 1)[0] if joiner else right
        before, after = left[:len(left) - len(tail)], right[len(head):]

        merged = self.contract.get(f"{tail.lower()} {head.lower()}")
        if merged is not None:
            return before + self._match_case(tail, merged) + after

        form = self.elide.get(tail.lower())
        if form is None or right[0].lower() not in self.vowels:
            return None
        return before + self._match_case(tail, form) + right

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        """Keep a sentence-initial capital that the substitution would drop."""
        if not original[:1].isupper():
            return replacement
        return replacement[:1].upper() + replacement[1:]


#: role -> case, given whether the clause is transitive
CaseRule = Callable[[str, bool], str]


def NOM_ACC(role: str, transitive: bool) -> str:
    """Nominative–accusative: the agent is unmarked, the patient is marked."""
    return {AGENT: "nom", PATIENT: "acc", THEME: "acc",
            RECIPIENT: "dat", LOCATION: "loc"}.get(role, "nom")


def ERG_ABS(role: str, transitive: bool) -> str:
    """Ergative–absolutive: the *patient* patterns with the intransitive subject."""
    if role == AGENT:
        return "erg" if transitive else "abs"
    return {PATIENT: "abs", THEME: "abs",
            RECIPIENT: "dat", LOCATION: "loc"}.get(role, "abs")


def NO_CASE(role: str, transitive: bool) -> str:
    """For languages that do not case-mark at all."""
    return ""


@dataclass(frozen=True)
class Alignment:
    """How arguments are case-marked, and whether the verb agrees with them."""

    case_of: CaseRule = NO_CASE
    #: which arguments the verb agrees with — Swahili agrees with two
    verb_agrees_with: tuple[str, ...] = ()


@dataclass(frozen=True)
class Concord:
    """Which features a dependent copies from its head."""

    #: features an attributive adjective shares with its noun
    adjective: tuple[str, ...] = ()
    #: features a determiner shares with its noun
    determiner: tuple[str, ...] = ()
    #: features a predicative adjective shares with its subject
    predicative: tuple[str, ...] = ()

    def share(self, features: Sequence[str], head: FS) -> FS:
        return FS({f: head[f] for f in features if f in head})


# ======================================================================
# the grammar
# ======================================================================
_SENTENCE_END = re.compile(r"[.!?。？！]$")
#: A space before a mark, in either width. Knowing only the ASCII marks left
#: Chinese writing "forall a exists b ；" -- the content ends in a Latin word,
#: the clause separator is full-width, and nothing joined them up.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([?.!,;:？．！，；：、。])")


class Grammar:
    """A concrete grammar: parameters, a lexicon, a morphology, and the walk.

    Subclasses set the class attributes and override only what the parameters
    cannot express. The walk itself is here and is not meant to be replaced.
    """

    code: str = ""
    name: str = ""
    order: WordOrder = WordOrder()
    typography: Typography = Typography()
    alignment: Alignment = Alignment()
    concord: Concord = Concord()
    sandhi: Sandhi = Sandhi()

    #: what the grammar claims to implement, and what it does not attempt
    notes: tuple[str, ...] = ()

    #: The directive an episode ends with, keyed ``instruction``,
    #: ``instruction_many`` and ``options_heading``. Kept apart from the closed
    #: class on purpose: these are format templates and the closed class holds
    #: words, and Turkish already had a *label* reading "instruction" that
    #: would have been formatted as one -- silently dropping the answer set,
    #: because a string with no ``{choices}`` in it formats to itself.
    instructions: Mapping[str, str] = {}

    def __init__(self) -> None:
        #: per-category morphology; a category with no entry is not inflected
        self.morphology: dict[str, Morphology] = {}
        #: closed-class words, keyed by an English-ish label
        self.closed: dict[str, str] = {}
        #: predicate head -> the words realizing it between two arguments
        self.predicate_words: dict[str, str] = {}
        #: section name -> the lead-in that introduces it
        self.field_intros: dict[str, str] = {}
        #: lemma -> inherent features (noun class, classifier, animacy)
        self.inherent: dict[str, FS] = {}
        self._word_cache: dict[tuple[str, str], str] = {}
        #: Inflectional material the morphology lessons *are about*, rather than
        #: merely use: verb paradigms, singular/plural pairs, agreement pairs,
        #: pronouns. A lesson on long-range agreement has to draw its own
        #: minimal pairs from the language it is presented in, so this is
        #: grammar data and belongs on the grammar.
        self.paradigms: dict[str, Any] = {}

    # ==================================================================
    # lexical access — overridden by a grammar that has a vocabulary
    # ==================================================================
    def word(self, lemma: str, pos: str = "") -> str:
        """The citation form of an open-class word, or the lemma untouched.

        A template method: subclasses supply :meth:`lookup`, which knows where
        their data lives, and this composes around it. The composition — falling
        back to a token-by-token rendering of a multi-word label — used to live
        in one subclass, which meant the five hand-written grammars silently
        kept every phrase in English while the derived ones translated them.

        ``pos`` matters more than it looks. A dictionary keyed only on spelling
        answers "red" with whichever sense it happened to list first, and for a
        great many languages that is the *noun* — German ``Rot``, Spanish
        ``tinto``, Korean ``적포도주`` "red wine". Passing the category the
        linearizer already knows turns a coin-flip into a lookup.

        Passing an unknown word through is a requirement, not a fallback: most
        of what this curriculum names is coined per episode, and translating a
        nonce form would destroy the lesson that turns on it.
        """
        cached = self._word_cache.get((lemma, pos))
        if cached is not None:
            return cached
        form = self.lookup(lemma, pos)
        if not form and " " in lemma:
            form = self.phrase(lemma, pos)
        if not form:
            form = self._spell_out(lemma, pos)
        out = form or lemma
        self._word_cache[(lemma, pos)] = out
        return out

    def _spell_out(self, lemma: str, pos: str) -> str:
        """Render a predicate head no dictionary keys on, via its English gloss.

        The heads this curriculum uses are inflected English verbs (``implies``),
        abbreviations (``sub``, ``pow``) and identifiers (``left_of``). A
        dictionary keys on none of them, so a grammar with no hand-written entry
        was returning the head itself and printing an internal identifier —
        underscore and all — as though it were a word of the language.

        :data:`PREDICATE_GLOSS` says what each one *means* in citation English,
        which :meth:`phrase` can then render a token at a time. If even that
        fails, English words are still better than an identifier: the gloss goes
        out unmodified, which is what the realizer this engine replaced did.

        Restricted to the listed heads. A coined nonce form must pass through
        untouched, and this is the guarantee that it does.
        """
        gloss = PREDICATE_GLOSS.get(lemma)
        if gloss is None or gloss == lemma:
            return ""
        # A one-word gloss keeps the category it was asked for. Composing it
        # through `phrase` looks up each token untyped, and untyped is how
        # `claims` came back as *Anspruch* and *réclamation* -- the noun, a
        # legal demand, in a slot the prose uses as a verb.
        if " " not in gloss:
            return self.word(gloss, pos) or gloss
        return self.phrase(gloss, pos) or gloss

    def lookup(self, lemma: str, pos: str) -> str:
        """One word from wherever this grammar keeps its lexicon. ``""`` if absent."""
        return ""

    def knows(self, lemma: str) -> bool:
        """Whether this grammar has a word of its own for ``lemma``.

        Asked of an answer option, which must be in the prompt's language or
        the episode quietly becomes "translate, then answer". The default is
        the lexical primitive; the two kinds of grammar answer it differently
        because they keep their words in different places, and a question only
        one of them could answer is how four hundred languages came to offer
        English options against translated evidence.
        """
        return bool(self.lookup(lemma, ""))

    def phrase(self, lemma: str, pos: str) -> str:
        """Render a multi-word label a token at a time.

        Labels like *value of* and sequences like *green blue green* are built
        from several ordinary words, and no dictionary has an entry for the
        whole. Each token does. A literal rendering in the right language beats
        a fluent one in the wrong language.

        Returns nothing unless a token actually translated, so a phrase made of
        words the language does not know stays intact rather than being half
        converted — which matters most for the coined sequences a few-shot
        lesson turns on.
        """
        out, translated = [], False
        for token in lemma.split():
            # A closed-class entry of "" is an answer, not a miss. Russian and
            # Turkish record `the: ""` because they have no article, and
            # `cw(...) or word(...)` read the empty string as "not found" and
            # asked the dictionary, which offered a demonstrative -- Polish
            # "ten", Finnish "se", Japanese "その" -- or the English word
            # itself. Russian rules read "правило the ценность".
            if token in self.closed:
                hit = self.closed[token]
                if hit != token:
                    translated = True
                if hit:
                    out.append(hit)
                continue
            hit = self.word(token)
            if hit and hit != token:
                translated = True
            out.append(hit or token)
        # the language's own joiner, not a literal space: a script written
        # without spaces must not acquire them by way of a composed label
        return self.join(out) if translated else ""

    def features_of(self, lemma: str) -> FS:
        """A word's inherent features — its class, its classifier, its animacy."""
        return self.inherent.get(lemma, EMPTY)

    def inflect(self, cat: str, lemma: str, feats: FS) -> str:
        morph = self.morphology.get(cat)
        surface = self.word(lemma, pos=cat)
        return morph.inflect(surface, feats) if morph else surface

    def cw(self, key: str, default: str = "") -> str:
        """A closed-class word."""
        return self.closed.get(key, default)

    # ==================================================================
    # assembly
    # ==================================================================
    def join(self, parts: Sequence[str]) -> str:
        joiner = self.typography.word_joiner
        kept = [p for p in parts if p]
        if not self.sandhi:
            return joiner.join(kept)
        # fold left to right so a boundary created by an earlier join is
        # still visible to the next one
        out = ""
        for part in kept:
            if not out:
                out = part
                continue
            out = self.sandhi.apply(out, part, joiner) or (out + joiner + part)
        return out

    def join_list(self, items: Sequence[str]) -> str:
        """Coordinate a list of noun-like items."""
        typ = self.typography
        items = [i for i in items if i]
        if len(items) <= 1:
            return items[0] if items else ""
        sep = typ.item_separator or typ.list_separator
        return self.join([sep.join(items[:-1]), self.cw("and"), items[-1]])

    def join_clauses(self, items: Sequence[str]) -> str:
        # Stripped, because the separator is written straight after the clause
        # and a clause that ends in a space puts one in front of the mark:
        # Chinese read "exists b ；" where the semicolon is full-width and
        # belongs against the character before it.
        items = [i.strip() for i in items if i and i.strip()]
        return self.typography.clause_separator.join(items)

    def capitalize(self, text: str) -> str:
        if not self.typography.capitalizes:
            return text
        for i, ch in enumerate(text):
            if ch.isalpha():
                return text[:i] + ch.upper() + text[i + 1:]
        return text

    def punctuate(self, text: str) -> str:
        """Set off the marks this language spaces, after the general cleanup.

        Run second on purpose. The cleanup normalises away any space before a
        mark, wherever it came from, and this puts back the one the language
        actually wants -- so the result does not depend on whether the caller
        happened to leave a space there.
        """
        marks = self.typography.space_before
        if not marks:
            return text
        out = []
        for i, ch in enumerate(text):
            if ch in marks and i and text[i - 1] not in " \u00a0\n":
                out.append(self.typography.space_before_char)
            out.append(ch)
        return "".join(out)

    def sentence(self, text: str, end: str | None = None) -> str:
        typ = self.typography
        text = " ".join(text.split()) if typ.word_joiner else text.strip()
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text).strip()
        if not text:
            return ""
        text = self.capitalize(text)
        if end != "":
            end = typ.full_stop if end is None else end
            if not _SENTENCE_END.search(text):
                text = text + end
        return self.punctuate(text)

    # ==================================================================
    # the walk
    # ==================================================================
    def lin(self, node: Node, ctx: FS = EMPTY) -> str:
        """Linearize a node. ``ctx`` carries features inherited from above."""
        handler = getattr(self, f"lin_{node.fn}", None)
        if handler is None:
            raise NotImplementedError(
                f"{self.code}: no linearization for construction {node.fn!r}")
        merged = unify(node.feats, ctx)
        return handler(node, merged[0] if merged else node.feats)

    # ---- leaves --------------------------------------------------------
    def lin_Sym(self, node: Node, ctx: FS) -> str:
        return node.text

    def lin_Lex(self, node: Node, ctx: FS) -> str:
        feats = unify(node.feats, ctx)
        feats = feats[0] if feats else node.feats
        # Only a **noun** carries an inherent class. An adjective does not have
        # a gender, it agrees with one, and the gender a dictionary records
        # against it is an artefact of which form happened to be tagged. Taking
        # it as inherent gave the Italian adjective a masculine of its own,
        # which then matched the masculine *plural* cell: *cubo gialli*.
        inherent = (self.features_of(node.lemma)
                    if node.cat.name == "N" else EMPTY)
        merged = unify(inherent, feats)
        return self.inflect(node.cat.name, node.lemma, merged[0] if merged else feats)

    # ---- phrases -------------------------------------------------------
    def lin_CN(self, node: Node, ctx: FS) -> str:
        """Noun plus adjectives, ordered and agreed per the parameters."""
        head = node.arg("head")
        assert head is not None
        head_feats = self._np_features(head, ctx)
        noun = self.lin(head, head_feats)
        shared = self.concord.share(self.concord.adjective, head_feats)
        if self.concord.adjective and CASE in head_feats:
            # An attributive adjective agrees in case wherever case exists —
            # near enough a universal that it is not worth a parameter. Without
            # it the request is ambiguous between the nominative and genitive
            # cells and German *gelbes Haus* came out *gelber Haus*.
            shared = shared.but(**{CASE: head_feats[CASE]})
        mods = [self.lin(m, shared) for m in node.all_args("mod")]
        if not mods:
            return noun
        return (self.join([*mods, noun]) if self.order.adj == "AN"
                else self.join([noun, *mods]))

    def _np_features(self, head: Node, ctx: FS) -> FS:
        """Combine a noun's inherent features with what the context imposes.

        Number defaults to singular where nothing has said otherwise. Leaving it
        absent is not neutrality: a paradigm lookup rejects a cell only where it
        *disagrees*, so an unspecified number does not disagree with a plural
        cell and Spanish *amarillo* came back as ``amarillos``. Singular is the
        unmarked value in every language here, and saying so is what makes the
        rejection work.
        """
        if NUM not in ctx:
            ctx = ctx.but(**{NUM: SG})
        if CASE not in ctx:
            # Nominative for the same reason as singular: an absent case does
            # not disagree with a genitive cell, so German *Haus* came back as
            # *Hauses* and Greek *κύβος* as *κύβου*. Naming the unmarked value
            # is what lets the lookup reject the marked ones.
            ctx = ctx.but(**{CASE: "nom"})
        base = self.features_of(head.lemma) if head.lemma else EMPTY
        for candidate in (unify(base, head.feats), None):
            if candidate:
                base = candidate[0]
                break
        merged = unify(base, ctx)
        return merged[0] if merged else base

    def lin_NP(self, node: Node, ctx: FS) -> str:
        """Determiner and numeral scoped over a common noun."""
        cn = node.arg("head")
        assert cn is not None
        head_lex = self._head_lemma(cn)
        count = node.arg("count")
        if count is not None and self.order.numeral_forces_plural:
            ctx = ctx.but(**{NUM: PL})
        feats = self._np_features(head_lex, ctx) if head_lex else ctx
        inner = self.lin(cn, feats)
        pieces = [inner]

        if count is not None:
            numeral = self.numeral_phrase(self.lin(count, ctx), head_lex, feats)
            pieces = ([numeral, inner] if self.order.numeral == "NumN"
                      else [inner, numeral])
        else:
            det = self.determiner(node.feats.get_atom("det", "bare"), head_lex, feats)
            if det:
                pieces = ([det, inner] if self.order.det == "DN"
                          else [inner, det])
        return self.join(pieces)

    def _head_lemma(self, node: Node | None) -> Node | None:
        """Dig down to the lexical head through however many phrase layers.

        A predicative adjective has to agree with the *noun* inside the subject
        noun phrase, which is two layers down — NP over CN over N. Stopping at
        the first layer is how a grammar silently loses concord on exactly the
        constructions where a reader would notice it most.
        """
        seen = 0
        while node is not None and not node.lemma and seen < 8:
            node = node.arg("head")
            seen += 1
        return node if node is not None and node.lemma else None

    def determiner(self, kind: str, head: Node | None, feats: FS) -> str:
        """The article. Languages without articles return the empty string."""
        if kind == "def":
            return self.cw("the")
        if kind == "indef":
            return self.cw("a")
        return ""

    def numeral_phrase(self, count: str, head: Node | None, feats: FS) -> str:
        """A numeral as it attaches to a noun. Classifier languages override."""
        return count

    def lin_AP(self, node: Node, ctx: FS) -> str:
        head = node.arg("head")
        assert head is not None
        return self.lin(head, ctx)

    # ---- predication ---------------------------------------------------
    def copula(self, kind: str, feats: FS) -> str:
        """``kind`` is ``attr``, ``ident`` or ``loc``.

        Kept as one hook with three kinds because the languages that split the
        copula do not all split it the same way: Spanish contrasts ``ser`` with
        ``estar`` on permanence, Chinese uses ``是`` for identity and nothing at
        all for a bare adjective, Russian drops it in the present throughout.
        """
        if not self.order.copula_overt:
            return ""
        plural = feats.get_atom(NUM) == PL
        return self.cw("are" if plural else "is")

    def _predicate(self, subject: str, verb: str, obj: str) -> str:
        return self.join(self.order.order_clause(subject, verb, obj))

    def _arg(self, node: Node, role: str, ctx: FS, *, transitive: bool) -> str:
        """Realize one argument, case-marked for the role it fills."""
        case = self.alignment.case_of(role, transitive)
        feats = ctx.but(**({"case": case} if case else {}))
        return self.lin(node, feats.but(role=role))

    def lin_PredAttr(self, node: Node, ctx: FS) -> str:
        subj, attr = node.arg(AGENT), node.arg(ATTRIBUTE)
        assert subj is not None and attr is not None
        s = self._arg(subj, AGENT, ctx, transitive=False)
        subj_feats = self.subject_features(subj, ctx)
        shared = self.concord.share(self.concord.predicative, subj_feats)
        return self._predicate(s, self.copula("attr", subj_feats),
                               self.lin(attr, shared))

    def subject_features(self, subject: Node, ctx: FS) -> FS:
        """Everything an agreeing predicate needs to know about the subject.

        The features live in two places and both matter: the *lexical head*
        supplies inherent ones like noun class, and the *phrase* supplies
        imposed ones like number. Reading only the head loses the plural; reading
        only the phrase loses the class. Merging them is what makes ``vitabu ni
        vikubwa`` come out with both prefixes agreeing.
        """
        head = self._head_lemma(subject)
        merged = unify(subject.feats, ctx)
        return self._np_features(head or subject, merged[0] if merged else ctx)

    def lin_PredIdent(self, node: Node, ctx: FS) -> str:
        subj, val = node.arg(AGENT), node.arg(VALUE)
        assert subj is not None and val is not None
        s = self._arg(subj, AGENT, ctx, transitive=False)
        return self.with_adjunct(s, self.copula("ident", ctx),
                                 self.lin(val, ctx), node, ctx)

    def with_adjunct(self, subject: str, verb: str, complement: str,
                     node: Node, ctx: FS) -> str:
        """Assemble a copular clause that may carry a locative adjunct.

        ``o0 is a yellow cube at (4, 8)`` is one clause with a place adjunct, not
        two clauses. Where the adjunct goes is not a free choice: a verb-final
        language puts it *after the subject and before the predicate* — Turkish
        ``o0 (4, 8)'de bir sarı küp`` — so appending it to a finished clause, as
        a head-initial language can, produces the wrong string.
        """
        place = node.arg(LOCATION)
        if place is None:
            return self._predicate(subject, verb, complement)
        marked = self.oblique(self.lin(place, ctx.but(case="loc")), LOCATION)
        if self.order.verb_final:
            return self.join([subject, marked, complement, verb])
        return self.join([self._predicate(subject, verb, complement), marked])

    def lin_PredLoc(self, node: Node, ctx: FS) -> str:
        subj, loc = node.arg(AGENT), node.arg(LOCATION)
        assert subj is not None and loc is not None
        s = self._arg(subj, AGENT, ctx, transitive=False)
        place = self.oblique(self._arg(loc, LOCATION, ctx, transitive=False), LOCATION)
        return self._predicate(s, self.copula("loc", ctx), place)

    def lin_PredRel(self, node: Node, ctx: FS) -> str:
        subj, rel, obj = node.arg(AGENT), node.arg("rel"), node.arg(PATIENT)
        assert subj is not None and rel is not None and obj is not None
        s = self._arg(subj, AGENT, ctx, transitive=True)
        o = self._arg(obj, PATIENT, ctx, transitive=True)
        return self._predicate(s, self.lin(rel, ctx), o)

    def lin_PredRel3(self, node: Node, ctx: FS) -> str:
        subj, rel = node.arg(AGENT), node.arg("rel")
        theme = node.arg(THEME)
        third = node.arg(RECIPIENT) or node.arg(GOAL) or node.arg(SOURCE)
        assert subj is not None and rel is not None and theme is not None
        s = self._arg(subj, AGENT, ctx, transitive=True)
        t = self._arg(theme, THEME, ctx, transitive=True)
        parts = [self.lin(rel, ctx), t]
        if third is not None:
            marked = self._arg(third, RECIPIENT, ctx, transitive=True)
            parts.append(self.oblique(marked, RECIPIENT))
        core = self.join(parts if not self.order.verb_final
                         else [*parts[1:], parts[0]])
        return self.join([s, core] if not self.order.verb_final else [s, core])

    def oblique(self, phrase: str, role: str) -> str:
        """Wrap an oblique argument in its adposition, on the correct side.

        A language that marks the role with a case suffix instead — Turkish
        locative ``-DA``, Finnish inessive ``-ssA`` — leaves the closed-class
        entry empty and the phrase comes back untouched, already inflected.
        """
        marker = self.cw({RECIPIENT: "to", LOCATION: "at"}.get(role, role), "")
        if not marker:
            return phrase
        return (self.join([marker, phrase]) if self.order.adposition == "pre"
                else self.join([phrase, marker]))

    # ---- packaging -----------------------------------------------------
    def clean_label(self, label: str) -> str:
        """Trim what a relational phrase leaves dangling when it labels one value.

        ``predicate_words`` are authored to sit *between* a subject and an
        object, so used as a label over a single value they can leave a particle
        or copula with nothing to attach to — Chinese ``的类型是`` in front of
        nothing, Spanish ``el estado de`` with no complement. The default trims
        nothing, because most languages have nothing to trim.
        """
        return label

    def lin_Labelled(self, node: Node, ctx: FS) -> str:
        label, value = node.arg("label"), node.arg(VALUE)
        assert label is not None and value is not None
        # clean_label is a declared strategy that the walk never invoked, so a
        # language only benefited from it by overriding this whole method —
        # which Spanish and Chinese both did, for nothing but that call.
        l = self.clean_label(self.lin(label, ctx))
        v = self.lin(value, ctx)
        l += self.typography.label_separator
        return self.join([l, v]) if self.order.label == "LV" else self.join([v, l])

    def lin_Enumerated(self, node: Node, ctx: FS) -> str:
        label = node.arg("label")
        values = self.join_list([self.lin(v, ctx) for v in node.all_args(VALUE)])
        # an empty label contributes nothing, not a bare colon
        rendered = self.lin(label, ctx).strip() if label is not None else ""
        l = (rendered + self.typography.colon) if rendered else ""
        return self.join([l, values]) if self.order.label == "LV" else self.join([values, l])

    def lin_Indexed(self, node: Node, ctx: FS) -> str:
        idx, body = node.arg(INDEX), node.arg("body")
        assert idx is not None and body is not None
        kind = node.feats.get_atom("kind", "step")
        head = self.join([self.cw(kind, kind), self.lin(idx, ctx)])
        # the joiner, not a literal space: a script written without spaces must
        # not acquire one here of all places
        return self.join([head + self.typography.colon, self.lin(body, ctx)])

    def lin_Mapping(self, node: Node, ctx: FS) -> str:
        src, goal = node.arg(SOURCE), node.arg(GOAL)
        assert src is not None and goal is not None
        return f"{self.lin(src, ctx)} → {self.lin(goal, ctx)}"

    def lin_Modified(self, node: Node, ctx: FS) -> str:
        """A head with a phrase attached. No separator — it is not a label.

        The modifier follows its head in nearly every language that has the
        construction at all, including the ones that put *adjectives* first:
        German says *das Buch auf dem Tisch*, not the reverse. It is the same
        asymmetry that puts a relative clause after its noun, so it follows the
        adposition parameter rather than the adjective one.
        """
        head = node.arg("head")
        assert head is not None
        modifiers = [self.lin(m, ctx) for m in node.all_args("mod")]
        rendered = self.lin(head, ctx)
        if not modifiers:
            return rendered
        parts = [rendered, *modifiers]
        return self.join(parts if self.order.adposition == "pre"
                         else [*modifiers, rendered])

    def lin_FnApp(self, node: Node, ctx: FS) -> str:
        """Notation stays notation, in every script.

        A *bare tuple* is not notation, though, and it arrives here with an
        empty name because that is how the compiler represents one. Its
        contents are ordinary words, and giving them half-width punctuation
        put a Latin comma and a space between two Chinese ones: 场景中 read
        "(杆, 大型)" where Chinese writes no space at all.
        """
        if not node.lemma:
            typ = self.typography
            separator = typ.item_separator or typ.list_separator
            return "(" + separator.join(
                self.lin(a, ctx) for a in node.children) + ")"
        inner = self.typography.arg_separator.join(
            self.lin(a, ctx) for a in node.children)
        return f"{node.lemma}({inner})"

    # ---- combination ---------------------------------------------------
    def lin_Coord(self, node: Node, ctx: FS) -> str:
        items = [self.lin(i, ctx) for i in node.all_args("item")]
        conj = node.feats.get_atom("conj", "and")
        if conj == "or":
            return self.disjoin(items)
        return self.join_list(items)

    def disjoin(self, items: Sequence[str]) -> str:
        items = [i for i in items if i]
        if len(items) <= 1:
            return items[0] if items else ""
        sep = self.typography.item_separator or self.typography.list_separator
        return self.join([sep.join(items[:-1]), self.cw("or"), items[-1]])

    def negator(self) -> str:
        """The word that marks negation, and never nothing.

        A missing negator does not make a clause positive — it makes two
        different claims render identically, and an episode whose candidate
        glosses collapse is unanswerable rather than merely clumsy. Lithuanian
        had no entry for *not* and rendered "some prism is yellow" for both
        polarities. Where the dedicated word is absent the negative determiner
        stands in, and where that is absent too the English word does: a
        visibly foreign negator is a small problem and a vanished one is not.
        """
        return self.cw("not") or self.cw("no_quant") or "not"

    def lin_Neg(self, node: Node, ctx: FS) -> str:
        inner = node.arg("inner")
        assert inner is not None
        text = self.lin(inner, ctx)
        neg = self.negator()
        return (self.join([text, neg]) if self.order.negation == "post"
                else self.join([neg, text]))

    def lin_Cond(self, node: Node, ctx: FS) -> str:
        cons, ante = node.arg("consequent"), node.arg("antecedent")
        assert cons is not None and ante is not None
        c, a = self.lin(cons, ctx), self.lin(ante, ctx)
        if self.order.conditional == "CA":
            return self.join([c, self.cw("if"), a])
        return self.join([self.cw("if"), a, self.cw("then"), c])

    def lin_Compare(self, node: Node, ctx: FS) -> str:
        left, right = node.arg(AGENT), node.arg(PATIENT)
        assert left is not None and right is not None
        rel = node.feats.get_atom("rel", "gt")
        # `gt` is an abbreviation, not a word. A pack with its own idiomatic
        # entry wins; otherwise the gloss composes one, which is what stopped
        # four hundred derived grammars printing "gt" between two numbers.
        return self._predicate(self.lin(left, ctx),
                               self.cw(rel) or self.word(rel, "V"),
                               self.lin(right, ctx))

    def lin_Possess(self, node: Node, ctx: FS) -> str:
        er, ed = node.arg("possessor"), node.arg("possessed")
        assert er is not None and ed is not None
        possessor = self.lin(er, ctx.but(case="gen"))
        possessed = self.lin(ed, ctx)
        if self.order.possessive == "GN":
            return self.join([possessor, possessed])
        return self.join([possessed, self.cw("of"), possessor])

    def lin_Quant(self, node: Node, ctx: FS) -> str:
        """``every prism is yellow``, ``no prism is yellow``, ``some prism is not yellow``.

        A quantifier over a restriction, predicated of a scope. With no
        restriction — the form a yes/no question about a scene takes — it falls
        back to quantifying the objects themselves.
        """
        restriction, scope = node.arg("restriction"), node.arg("scope")
        quantifier = node.feats.get_atom("q", "all")
        negated = node.feats.get_atom("pol") == "neg"
        # "no P is Q" is how English negates a universal; a language whose
        # closed class has no such word negates the scope instead
        if negated and quantifier == "all" and self.cw("no_quant"):
            word, inner_negation = self.cw("no_quant"), False
        else:
            # a universal claim is distributive — *every prism*, not *all prism*
            word = (self.cw("every") if quantifier == "all" and self.cw("every")
                    else self.cw(quantifier, quantifier))
            inner_negation = negated

        if restriction is None:
            parts = [word, self.cw("of"),
                     self.lin(scope, ctx) if scope is not None else ""]
            return self.join(parts)
        subject = self.join([word, self.lin(restriction, ctx)])
        if scope is None:
            # a quantified noun phrase, not a clause: *every agent*. Running it
            # through the copula gave "every agent is" with nothing after it.
            return subject
        complement = self.lin(scope, ctx)
        verb = self.copula("attr", ctx)
        if inner_negation:
            negator = self.negator()
            if self.order.negation == "post":
                complement = self.join([complement, negator])
            elif self.order.negation == "aux":
                verb = self.join([verb, negator])
            else:
                # a preverbal negator negates the **verb**, not the complement:
                # Spanish *no es amarillo*, not *es no amarillo*. Attaching it
                # to the complement reads as constituent negation, which is a
                # different claim.
                verb = self.join([negator, verb])
        return self._predicate(subject, verb, complement)

    # ---- questions -----------------------------------------------------
    #: wh-words that are determiners rather than arguments. These sit in the
    #: determiner slot of the phrase they question — *which book*, *hangi kitap*,
    #: *哪本书* — even in languages that otherwise leave wh in situ, because it is
    #: the noun phrase that is questioned, not the clause.
    WH_DETERMINERS = frozenset({"which", "how_many", "whose"})

    def lin_WhQ(self, node: Node, ctx: FS) -> str:
        """A content question. Fronted, in situ, or determiner-attached."""
        body = node.arg("body")
        assert body is not None
        key = node.feats.get_atom("wh", "what")
        wh = self.cw(key, "what")
        inner = self.lin(body, ctx)
        if key in self.WH_DETERMINERS and body.fn in ("NP", "CN"):
            return self.wh_identity(wh, body, ctx)
        return self.join([wh, inner]) if self.order.wh_fronting \
            else self.join([inner, wh])

    def wh_identity(self, wh: str, body: Node, ctx: FS) -> str:
        """``which object is the green disc?`` — asking which thing matches.

        The wh-word is a determiner and cannot simply be stacked on top of one
        the noun phrase already has: *which the green disc* and *哪个这个绿色的圆盘*
        are both ungrammatical for the same reason. So the wh takes the
        determiner slot of its own head — the generic word for a *thing*, which
        every language in the curriculum's vocabulary has — and the described
        phrase becomes the complement of a copula.

        This is one construction serving four typologies: the copula is silent
        in Turkish, the head follows its determiner in Spanish and precedes it
        in Swahili, and none of that needs saying here.
        """
        thing = self.word("object", pos="N")
        head = (self.join([wh, thing]) if self.order.det == "DN"
                else self.join([thing, wh]))
        return self._predicate(head, self.copula("ident", ctx),
                               self.lin(body, ctx))

    def lin_YNQ(self, node: Node, ctx: FS) -> str:
        """A polar question. The default is a clause-final particle, which is
        both the commonest strategy across languages and the one that degrades
        most gracefully; inverting grammars override."""
        body = node.arg("body")
        assert body is not None
        return self.join([self.lin(body, ctx), self.cw("q_particle")])

    def lin_AltQ(self, node: Node, ctx: FS) -> str:
        body = node.arg("body")
        options = self.disjoin([self.lin(o, ctx) for o in node.all_args("option")])
        inner = self.lin(body, ctx) if body is not None else ""
        return self.join([inner, options])

    def question(self, node: Node, ctx: FS = EMPTY) -> str:
        typ = self.typography
        text = self.lin(node, ctx)
        if typ.question_open:
            text = typ.question_open + text
        return self.sentence(text, end=typ.question_mark)

    # ---- blocks --------------------------------------------------------
    def lin_Block(self, node: Node, ctx: FS) -> str:
        """A named section of an episode, with its lead-in."""
        typ = self.typography
        name = node.feats.get_atom("name", "")
        items = [self.lin(i, ctx) for i in node.all_args("item")]
        intro = self.field_intros.get(name) or self.block_heading(name)
        if not items:
            return self.sentence(self.join([intro, self.cw("empty", "—")]))
        if len(items) > 4 or any(len(i) > 64 for i in items):
            head = intro if intro.rstrip().endswith(typ.colon) else intro.rstrip() + typ.colon
            return (self.capitalize(head) + "\n"
                    + "\n".join(f"{typ.bullet}{i}" for i in items))
        joined = self.join_clauses(items) if len(items) > 1 else items[0]
        return self.sentence(self.join([intro, joined]))

    def block_heading(self, name: str) -> str:
        return name.replace("_", " ") + self.typography.colon

    # ==================================================================
    def info(self) -> dict[str, Any]:
        return {"code": self.code, "name": self.name,
                "order": self.order.clause, "adjective": self.order.adj,
                "alignment": self.alignment.case_of.__name__,
                "notes": list(self.notes)}
