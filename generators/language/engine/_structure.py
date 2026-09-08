"""The internal structured representation lessons build, and how it becomes text.

This module is **private**. Nothing here crosses the public API: a caller of
:mod:`langcurriculum` only ever sees ``str``, ``dict``, ``list`` and numbers.
The structure exists because a question like "which object is left of the blue
cube?" is easier to *generate correctly* as a tree than as a string — the
generator can check uniqueness, count, and compute the exact answer over the
structure, and only then render it.

A :class:`Term` is an immutable, hashable, typed node. Composite terms hold
other terms in ``value``; records hold ``(name, term)`` pairs. Two renderings
are provided:

* :func:`sexpr` — a compact, paste-able, fully-parenthesized notation, which is
  what the ``symbols`` language pack shows an agent;
* :func:`to_json` — a plain-data mirror, for callers that want the structure
  rather than the string.

Rendering is total: every term has a text form, so a lesson can never hand a
caller something it cannot print.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

__all__ = [
    "Term",
    "Tok", "Str", "Num", "Ident", "Nil",
    "Pred", "Rel", "Tup", "Lst", "Node", "Rec", "App",
    "walk", "leaves", "size", "depth", "sexpr", "to_json",
]

PRIMITIVE_TYPES = frozenset({"token", "str", "num", "ident", "nil"})
COMPOSITE_TYPES = frozenset({"tuple", "list", "pred", "rel", "node", "app", "record"})


def _freeze(v: Any) -> Any:
    if isinstance(v, Term):
        return v
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted(((str(k), _freeze(x)) for k, x in v.items()), key=lambda kv: kv[0]))
    return v


@dataclass(frozen=True)
class Term:
    """``type`` says what kind of node it is; ``value`` is the payload."""

    type: str
    value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze(self.value))

    # ---- structure -------------------------------------------------
    @property
    def children(self) -> tuple["Term", ...]:
        """Sub-terms, including the values of named fields, so a traversal
        reaches every node in the structure."""
        if not isinstance(self.value, tuple):
            return ()
        out: list[Term] = []
        for x in self.value:
            if isinstance(x, Term):
                out.append(x)
            elif isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], Term):
                out.append(x[1])
        return tuple(out)

    @property
    def is_composite(self) -> bool:
        return bool(self.children)

    def field(self, name: str, default: Any = None) -> Any:
        """Field lookup for record/app terms (value is a tuple of (key, Term))."""
        if isinstance(self.value, tuple):
            for item in self.value:
                if isinstance(item, tuple) and len(item) == 2 and item[0] == name:
                    return item[1]
        return default

    def __iter__(self) -> Iterator["Term"]:
        return iter(self.children)

    def __len__(self) -> int:
        return len(self.children)

    def __bool__(self) -> bool:
        # a term is always a thing; without this, atoms (0 children) are falsy
        return True

    def __repr__(self) -> str:
        return sexpr(self)

    def __str__(self) -> str:
        return sexpr(self)


# --------------------------------------------------------------------------
# constructors
# --------------------------------------------------------------------------
def Tok(t: Any) -> Term:
    """A token: an atom drawn from a finite vocabulary."""
    return Term("token", t)


def Str(s: str) -> Term:
    return Term("str", s)


def Num(x: float | int) -> Term:
    return Term("num", x)


def Ident(name: str) -> Term:
    """A name, variable or constant identifier."""
    return Term("ident", name)


def Nil() -> Term:
    return Term("nil", None)


def Pred(name: str, *args: Term) -> Term:
    """Predicate application: ``on(a, b)``, ``even(x)``."""
    return Term("pred", (name, *args))


def Rel(name: str, src: Term, dst: Term) -> Term:
    """A binary relation edge."""
    return Term("rel", (name, src, dst))


def Tup(*xs: Term) -> Term:
    return Term("tuple", xs)


def Lst(xs: Iterable[Term]) -> Term:
    return Term("list", tuple(xs))


def Node(label: str, children: Sequence[Term] = ()) -> Term:
    """A tree node."""
    return Term("node", (label, *children))


def App(fn: str, **kwargs: Term) -> Term:
    """Function application with named arguments."""
    return Term("app", (fn, *sorted(kwargs.items(), key=lambda kv: kv[0])))


def Rec(**fields: Term) -> Term:
    """A record with named fields."""
    return Term("record", tuple(sorted(fields.items(), key=lambda kv: kv[0])))


# --------------------------------------------------------------------------
# traversal
# --------------------------------------------------------------------------
def walk(s: Term) -> Iterator[Term]:
    yield s
    for c in s.children:
        yield from walk(c)


def leaves(s: Term) -> Iterator[Term]:
    if not s.children:
        yield s
    for c in s.children:
        yield from leaves(c)


def size(s: Term) -> int:
    return sum(1 for _ in walk(s))


def depth(s: Term) -> int:
    return 1 + max((depth(c) for c in s.children), default=0)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _atom_repr(v: Any) -> str:
    if isinstance(v, str):
        return v if v.isidentifier() or v.isdigit() else json.dumps(v)
    return repr(v)


def sexpr(s: Any) -> str:
    """Compact, human-readable text for a term (or a plain value)."""
    if not isinstance(s, Term):
        return _atom_repr(s)
    if s.type in ("token", "str", "num", "ident"):
        return _atom_repr(s.value)
    if s.type == "nil":
        return "()"
    if s.type in ("pred", "rel", "node"):
        head, *rest = s.value
        inner = " ".join(sexpr(x) for x in rest if isinstance(x, Term))
        return f"({head}{' ' + inner if inner else ''})"
    if s.type == "tuple":
        return "(" + " ".join(sexpr(x) for x in s.children) + ")"
    if s.type == "list":
        return "[" + " ".join(sexpr(x) for x in s.children) + "]"
    if s.type == "app":
        fn, *kvs = s.value
        args = ", ".join(f"{k}={sexpr(v)}" for k, v in kvs)
        return f"{fn}({args})"
    if s.type == "record":
        return "{" + ", ".join(f"{k}: {sexpr(v)}" for k, v in s.value) + "}"
    return f"<{s.type} {s.value!r}>"


def to_json(s: Any) -> Any:
    """Plain-data mirror of a term: dicts, lists, strings, numbers."""
    if isinstance(s, Term):
        if s.type in ("token", "str", "num", "ident", "nil"):
            return {"t": s.type, "v": s.value}
        if s.type in ("pred", "rel", "node"):
            head, *rest = s.value
            return {"t": s.type, "head": head, "args": [to_json(x) for x in rest]}
        if s.type in ("tuple", "list"):
            return {"t": s.type, "items": [to_json(x) for x in s.children]}
        if s.type in ("record", "app"):
            if s.type == "app":
                fn, *kvs = s.value
                return {"t": "app", "fn": fn, "args": {k: to_json(v) for k, v in kvs}}
            return {"t": "record", "fields": {k: to_json(v) for k, v in s.value}}
        return {"t": s.type, "v": str(s.value)}
    if isinstance(s, (list, tuple)):
        return [to_json(x) for x in s]
    return s
