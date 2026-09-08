"""Abstract syntax: what an episode means, before any language has been chosen.

An abstract syntax tree here says *what is predicated of what*, with arguments
labelled by the role they fill, and says nothing whatever about word order,
inflection, or which words are used. It is the interlingua. A grammar reads one
and produces a sentence; two grammars reading the same tree produce two
sentences that mean the same thing, which is the property the curriculum needs
and the property a template system cannot guarantee.

The construction inventory
--------------------------

The previous realizer had four constructions — a label with one value, a label
with several, a subject-relation-object triple, and function-application
notation — and they carried 58% of everything the curriculum printed. That is
the right *idea* and far too small an inventory, and being positional rather
than role-labelled it could only ever produce English order.

What follows is the inventory that replaces it. It is deliberately small. Each
entry is a construction in the linguist's sense: a pairing of a form with a
meaning, realized differently in each language but recognizably the same thing::

    predication       PredAttr   the cube is red
                      PredIdent  the cube is a solid
                      PredLoc    the cube is at (3, 4)
                      PredRel    the compiler requires the parser
                      PredRel3   alice gives the key to bob
    packaging         Labelled   weight: 3
                      Enumerated rule: a, b and c
                      Indexed    step 4: …
                      Mapping    aba → 1
                      FnApp      t(a, b)
                      Modified   the object left of the prism
    combination       Coord      a and b
                      Neg        not a
                      Cond       a if b
                      Compare    a is greater than b
                      Possess    the value of a
                      Quant      all of the objects are red
    questions         WhQ        which object is red?
                      YNQ        is the string balanced?
                      AltQ       is it high or low?

Twenty. Adding a language means saying how *these* are realized, which is a
bounded job a linguist can finish, rather than translating 399 predicate heads,
which is not.

Roles, not positions
--------------------

Every argument arrives under a role from :mod:`~langcurriculum.grammar.category`
— agent, patient, recipient, value, index. This is what lets a grammar decide
for itself that an agent is preverbal and unmarked (Turkish), or suffixed with
``が`` (Japanese), or ergative when the clause is transitive (Basque). A
positional slot cannot express any of that, which is why the old
``relational(subject, relation, object)`` was locked to English order no matter
what a pack overrode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .category import (
    A, ADV, AGENT, AP, ATTRIBUTE, CARD, CL, CN, CONJ, DET, GOAL, INDEX,
    LOCATION, N, NP, PATIENT, PP, QCL, RECIPIENT, S, SOURCE, SYM, THEME, TEXT,
    UTT, V, VALUE, Cat,
)
from .features import EMPTY, FS

__all__ = [
    "Node", "Arg", "sym", "lex", "noun", "adj", "verb",
    "mk_cn", "mk_np", "mk_ap",
    "pred_attr", "pred_ident", "pred_loc", "pred_rel", "pred_rel3",
    "labelled", "enumerated", "indexed", "mapping", "fn_app", "modified",
    "coord", "negate", "cond", "compare", "possess", "quant",
    "wh_question", "yn_question", "alt_question",
    "text_block", "walk_nodes", "CONSTRUCTIONS",
]


@dataclass(frozen=True)
class Arg:
    """One argument of a construction, labelled by the role it fills."""

    role: str
    node: "Node"


@dataclass(frozen=True)
class Node:
    """A node of the abstract syntax tree.

    ``fn`` names the construction; the linearizer dispatches on it. ``lemma`` is
    set on lexical leaves and is a *key into the vocabulary*, in English, because
    that is what the curriculum's generators coin their constants in. ``text`` is
    set on symbols that must pass through untouched.
    """

    fn: str
    cat: Cat = SYM
    args: tuple[Arg, ...] = ()
    feats: FS = EMPTY
    lemma: str = ""
    text: str = ""

    # ---- access --------------------------------------------------------
    def arg(self, role: str) -> "Node | None":
        for a in self.args:
            if a.role == role:
                return a.node
        return None

    def all_args(self, role: str) -> list["Node"]:
        return [a.node for a in self.args if a.role == role]

    @property
    def children(self) -> tuple["Node", ...]:
        return tuple(a.node for a in self.args)

    def but(self, **kw: Any) -> "Node":
        """This node with some features overridden."""
        return Node(self.fn, self.cat, self.args, self.feats.but(**kw),
                    self.lemma, self.text)

    def __repr__(self) -> str:
        if self.fn == "Sym":
            return f"'{self.text}'"
        if self.lemma:
            return f"{self.fn}:{self.lemma}"
        inner = " ".join(f"{a.role}={a.node!r}" for a in self.args)
        return f"({self.fn} {inner})" if inner else f"({self.fn})"


def walk_nodes(n: Node) -> Iterator[Node]:
    yield n
    for c in n.children:
        yield from walk_nodes(c)


def _n(fn: str, cat: Cat, args: Sequence[Arg] = (), **feats: Any) -> Node:
    return Node(fn, cat, tuple(args), FS(feats))


# ======================================================================
# leaves
# ======================================================================
def sym(text: str) -> Node:
    """An opaque symbol: an object id, a nonce form, a number, a coordinate.

    Most of what this curriculum talks about is invented per episode, and
    inventing a translation for a coined word destroys the lesson. A symbol is
    never inflected and never translated in any grammar.
    """
    return Node("Sym", SYM, text=str(text))


def lex(cat: Cat, lemma: str, **feats: Any) -> Node:
    return Node("Lex", cat, (), FS(feats), lemma=lemma)


def noun(lemma: str, **feats: Any) -> Node:
    return lex(N, lemma, **feats)


def adj(lemma: str, **feats: Any) -> Node:
    return lex(A, lemma, **feats)


def verb(lemma: str, **feats: Any) -> Node:
    return lex(V, lemma, **feats)


# ======================================================================
# phrases
# ======================================================================
def mk_cn(head: Node, *modifiers: Node, **feats: Any) -> Node:
    """A common noun with its adjectives — the constituent a determiner scopes over.

    Separate from :func:`mk_np` because languages differ on all three of where
    the adjective goes, whether it agrees, and whether a determiner is present at
    all; keeping the layers apart is what lets one linearizer serve all three.
    """
    args = [Arg("head", head)] + [Arg("mod", m) for m in modifiers]
    return _n("CN", CN, args, **feats)


def mk_np(cn: Node, *, det: str = "", count: Any = None, **feats: Any) -> Node:
    """A noun phrase. ``det`` is ``def``/``indef``/``""``; ``count`` a numeral.

    A language with no articles ignores ``det``; a classifier language routes
    both ``det`` and ``count`` through its measure word. Neither fact is visible
    here, which is the point.
    """
    args = [Arg("head", cn)]
    if count is not None:
        args.append(Arg("count", sym(count) if not isinstance(count, Node) else count))
    return _n("NP", NP, args, det=det or "bare", **feats)


def mk_ap(a: Node, **feats: Any) -> Node:
    return _n("AP", AP, [Arg("head", a)], **feats)


# ======================================================================
# predication
# ======================================================================
def pred_attr(subject: Node, attribute: Node, **feats: Any) -> Node:
    """``X is red`` — a property predicated of an entity."""
    return _n("PredAttr", CL, [Arg(AGENT, subject), Arg(ATTRIBUTE, attribute)], **feats)


def pred_ident(subject: Node, kind: Node, **feats: Any) -> Node:
    """``X is a cube`` — classification, which many languages mark differently
    from attribution and a few (Chinese, Russian) mark differently again."""
    return _n("PredIdent", CL, [Arg(AGENT, subject), Arg(VALUE, kind)], **feats)


def pred_loc(subject: Node, place: Node, **feats: Any) -> Node:
    """``X is at (3, 4)`` — location, which Spanish marks with a different copula."""
    return _n("PredLoc", CL, [Arg(AGENT, subject), Arg(LOCATION, place)], **feats)


def pred_rel(subject: Node, relation: Node, obj: Node, **feats: Any) -> Node:
    """``X requires Y`` — the workhorse. 27% of everything the curriculum prints."""
    return _n("PredRel", CL,
              [Arg(AGENT, subject), Arg("rel", relation), Arg(PATIENT, obj)], **feats)


def pred_rel3(subject: Node, relation: Node, obj: Node, third: Node,
              third_role: str = RECIPIENT, **feats: Any) -> Node:
    """``alice gives the key to bob``."""
    return _n("PredRel3", CL, [Arg(AGENT, subject), Arg("rel", relation),
                               Arg(THEME, obj), Arg(third_role, third)], **feats)


# ======================================================================
# packaging — the shapes that are data rows rather than sentences
# ======================================================================
def labelled(label: Node, value: Node, **feats: Any) -> Node:
    """``weight: 3``. Head-initial in English, head-final in Japanese and Turkish."""
    return _n("Labelled", CL, [Arg("label", label), Arg(VALUE, value)], **feats)


def enumerated(label: Node, values: Sequence[Node], **feats: Any) -> Node:
    """``rule: a, b and c``."""
    return _n("Enumerated", CL,
              [Arg("label", label)] + [Arg(VALUE, v) for v in values], **feats)


def indexed(index: Node, body: Node, kind: str = "step", **feats: Any) -> Node:
    """``step 4: …``, ``in round 2, …`` — an ordinal that situates a clause."""
    return _n("Indexed", CL, [Arg(INDEX, index), Arg("body", body)],
              kind=kind, **feats)


def mapping(lhs: Node, rhs: Node, **feats: Any) -> Node:
    """``aba → 1`` — an input paired with its output, in an example list."""
    return _n("Mapping", CL, [Arg(SOURCE, lhs), Arg(GOAL, rhs)], **feats)


def modified(head: Node, *modifiers: Node, **feats: Any) -> Node:
    """``the purple object to the left of the prism`` — a head and a modifier phrase.

    Distinct from :func:`labelled`, which was standing in for it and is wrong:
    a label takes a separator — *weight: 3*, *重量：3* — and a modifier does
    not, so Spanish came out as *el objeto morado: a la izquierda: el prisma*.
    Distinct from :func:`mk_cn` too, whose modifiers are adjectives that agree;
    this one is a phrase that attaches whole.
    """
    return _n("Modified", NP, [Arg("head", head)]
              + [Arg("mod", m) for m in modifiers], **feats)


def fn_app(head: str, args: Sequence[Node], **feats: Any) -> Node:
    """``t(a, b)`` — notation, kept as notation.

    The escape hatch, and a principled one: where the structure is a formula
    rather than a sentence, every language shows it as a formula. Rendering
    ``mod(x, 7)`` as prose would be a translation the source never asked for.
    """
    return Node("FnApp", CL, tuple(Arg("arg", a) for a in args),
                FS(feats), lemma=head)


# ======================================================================
# combination
# ======================================================================
def coord(conj: str, items: Sequence[Node], **feats: Any) -> Node:
    """``a and b``, ``a, b or c``. ``conj`` is ``and`` or ``or``."""
    return _n("Coord", CL, [Arg("item", i) for i in items], conj=conj, **feats)


def negate(inner: Node, **feats: Any) -> Node:
    return _n("Neg", inner.cat, [Arg("inner", inner)], **feats)


def cond(consequent: Node, antecedent: Node, **feats: Any) -> Node:
    """``a if b`` / ``if b then a`` — languages differ on which clause leads."""
    return _n("Cond", CL,
              [Arg("consequent", consequent), Arg("antecedent", antecedent)], **feats)


def compare(left: Node, relation: str, right: Node, **feats: Any) -> Node:
    """``a is greater than b`` — separated from PredRel because comparatives
    inflect in some languages and take a dedicated case in others."""
    return _n("Compare", CL, [Arg(AGENT, left), Arg(PATIENT, right)],
              rel=relation, **feats)


def possess(possessor: Node, possessed: Node, **feats: Any) -> Node:
    """``the value of a`` / ``a's value`` / Turkish ``a-nın değer-i``."""
    return _n("Possess", NP,
              [Arg("possessor", possessor), Arg("possessed", possessed)], **feats)


def quant(quantifier: str, restriction: Node, scope: Node, **feats: Any) -> Node:
    """``all of the objects are red``."""
    return _n("Quant", CL, [Arg("restriction", restriction), Arg("scope", scope)],
              q=quantifier, **feats)


# ======================================================================
# questions
# ======================================================================
def wh_question(wh: str, body: Node, **feats: Any) -> Node:
    """A content question. ``wh`` names what is asked for — ``which``, ``what``,
    ``who``, ``how_many``, ``where``, ``why``.

    Nothing here says the wh-word moves. English fronts it, Chinese and Japanese
    leave it in situ, Turkish leaves it in situ and case-marks it. The grammar
    decides; the abstract tree only records that this is the questioned element.
    """
    return _n("WhQ", QCL, [Arg("body", body)], wh=wh, **feats)


def yn_question(body: Node, **feats: Any) -> Node:
    """A polar question — inversion in English, ``吗`` in Chinese, ``mI`` in
    Turkish, ``か`` in Japanese, intonation alone in Spanish."""
    return _n("YNQ", QCL, [Arg("body", body)], **feats)


def alt_question(body: Node, options: Sequence[Node], **feats: Any) -> Node:
    """``is it high or low?`` — an alternative question, which several languages
    build with a dedicated construction rather than a coordinated polar one."""
    return _n("AltQ", QCL,
              [Arg("body", body)] + [Arg("option", o) for o in options], **feats)


def text_block(name: str, items: Sequence[Node], *, is_list: bool = False,
               **feats: Any) -> Node:
    """A named section of an episode: the scene, the premises, the log.

    Carries the field name so the grammar can supply its own lead-in — English
    "In the scene:", Chinese topic-marked "场景中：", Turkish "Sahnede:" — rather
    than having one translated after the fact.
    """
    return _n("Block", TEXT, [Arg("item", i) for i in items],
              name=name, is_list=is_list, **feats)


#: every construction the linearizer must handle. A grammar that misses one gets
#: a clear failure rather than silently dropping the constituent.
CONSTRUCTIONS = frozenset({
    "Sym", "Lex", "CN", "NP", "AP",
    "PredAttr", "PredIdent", "PredLoc", "PredRel", "PredRel3",
    "Labelled", "Enumerated", "Indexed", "Mapping", "FnApp", "Modified",
    "Coord", "Neg", "Cond", "Compare", "Possess", "Quant",
    "WhQ", "YNQ", "AltQ", "Block",
})
