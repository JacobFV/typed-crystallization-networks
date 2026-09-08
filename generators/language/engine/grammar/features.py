"""Feature structures and unification: the one mechanism agreement runs on.

Every language in this curriculum makes some words depend on others. Spanish
makes an article and an adjective depend on a noun's gender. Chinese makes a
measure word depend on the noun it counts. Turkish makes every suffix depend on
the backness and rounding of the last stem vowel. Swahili makes the subject
prefix, the object prefix, the adjective, the demonstrative and the relative
marker all depend on one noun's class, of which there are eighteen.

Written by hand these look like five unrelated problems, which is why the
previous realizer solved them five unrelated ways — a ``gender`` string here, a
``classifier`` lookup there, a ``linker`` boolean somewhere else. Written as
**feature structures** they are one problem: some node carries information, some
other node needs it, and the two have to be made consistent. That operation is
unification, and it is the whole of this module.

A :class:`FS` is an immutable mapping from feature name to value, where a value
is an atom (``"sg"``, ``3``, ``True``), a nested :class:`FS`, or a
:class:`Var` standing for "whatever the other side says". Unification of two
structures succeeds when every feature they share can be made equal, and returns
the structure carrying everything both of them knew::

    >>> unify(FS(num="sg", gen="f"), FS(gen=Var("g"), case="nom"))[0]
    FS(case='nom', gen='f', num='sg')

The variable is the part that does the work. A grammar says *the adjective's
gender is the noun's gender* by giving both the same variable, and the value
flows from wherever it is known to wherever it is needed — without the grammar
having to know which of the two that will be. That is why one mechanism covers
Spanish two-gender agreement and Swahili eighteen-class concord: the grammar
changes, the mechanism does not.

Failure is a value, not an exception. :func:`unify` returns ``None`` when the
structures conflict, because a linearizer trying alternatives needs to ask
"does this fit?" far more often than it needs to crash.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping

__all__ = ["Var", "FS", "Bindings", "unify", "resolve", "subsumes", "EMPTY"]


class Var:
    """A placeholder for a value some other node will supply.

    Two occurrences of a variable with the same name are the *same* variable, so
    a grammar can tie an adjective's gender to its noun's simply by mentioning
    ``Var("g")`` in both. Variables are scoped to a single unification run; the
    caller renames them if two independently-authored structures might collide.
    """

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"?{self.name}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Var) and other.name == self.name

    def __hash__(self) -> int:
        return hash(("Var", self.name))


#: variable name -> the value or variable it is bound to
Bindings = dict[str, Any]


class FS(Mapping[str, Any]):
    """An immutable feature structure.

    Constructed from keyword arguments or a mapping. Hashable, so a feature
    bundle can key a paradigm table, which is exactly how the morphology engine
    looks up an inflected form.
    """

    __slots__ = ("_d", "_hash")

    def __init__(self, _m: Mapping[str, Any] | None = None, **kw: Any):
        d = dict(_m or {})
        d.update(kw)
        # sorted so that equal structures have equal repr and equal hash
        object.__setattr__(self, "_d", dict(sorted(d.items())))
        object.__setattr__(self, "_hash", None)

    # ---- mapping protocol ---------------------------------------------
    def __getitem__(self, k: str) -> Any:
        return self._d[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __repr__(self) -> str:
        return "FS(" + ", ".join(f"{k}={v!r}" for k, v in self._d.items()) + ")"

    def __hash__(self) -> int:
        if self._hash is None:
            object.__setattr__(self, "_hash", hash(tuple(self._d.items())))
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FS) and other._d == self._d

    # ---- construction --------------------------------------------------
    def but(self, **kw: Any) -> "FS":
        """This structure with some features overridden. The common edit."""
        return FS({**self._d, **kw})

    def without(self, *keys: str) -> "FS":
        return FS({k: v for k, v in self._d.items() if k not in keys})

    def get_atom(self, key: str, default: Any = None) -> Any:
        """The value of a feature, or ``default`` if absent *or still a variable*.

        A linearizer asking "is this plural?" wants a straight answer. An unbound
        variable is not one, and treating it as a value is how a grammar starts
        emitting the string ``?n`` into its own output.
        """
        v = self._d.get(key, default)
        return default if isinstance(v, Var) else v

    def ground(self) -> bool:
        """Whether every feature has a real value — nothing left to resolve."""
        return not any(isinstance(v, Var) or (isinstance(v, FS) and not v.ground())
                       for v in self._d.values())

    def rename(self, suffix: str) -> "FS":
        """Freshen every variable, so two structures can be unified independently.

        Authoring a grammar means writing ``Var("n")`` in a dozen unrelated
        rules. Without renaming, applying two of them in one derivation would
        accidentally tie them together — the subject's number would silently
        become the object's. Constructions are renamed as they are instantiated.
        """
        def fresh(v: Any) -> Any:
            if isinstance(v, Var):
                return Var(f"{v.name}#{suffix}")
            if isinstance(v, FS):
                return v.rename(suffix)
            return v
        return FS({k: fresh(v) for k, v in self._d.items()})


EMPTY = FS()


def resolve(value: Any, bindings: Bindings) -> Any:
    """Follow a variable through the binding chain to whatever it stands for.

    Variables may be bound to other variables — ``?a`` to ``?b`` to ``"sg"`` —
    because two rules can tie their features together before either knows the
    value. Following the chain is what makes the order rules apply in irrelevant.
    """
    seen: set[str] = set()
    while isinstance(value, Var):
        if value.name in seen:            # a cycle: ?a bound to ?a
            return value
        seen.add(value.name)
        if value.name not in bindings:
            return value
        value = bindings[value.name]
    if isinstance(value, FS):
        return FS({k: resolve(v, bindings) for k, v in value.items()})
    return value


def _bind(var: Var, value: Any, bindings: Bindings) -> Bindings | None:
    target = resolve(var, bindings)
    if isinstance(target, Var):
        if isinstance(value, Var) and resolve(value, bindings) == target:
            return bindings                       # already the same variable
        return {**bindings, target.name: value}
    return _unify_values(target, value, bindings)


def _unify_values(a: Any, b: Any, bindings: Bindings) -> Bindings | None:
    a, b = resolve(a, bindings), resolve(b, bindings)
    if isinstance(a, Var):
        return _bind(a, b, bindings)
    if isinstance(b, Var):
        return _bind(b, a, bindings)
    if isinstance(a, FS) and isinstance(b, FS):
        merged = unify(a, b, bindings)
        return None if merged is None else merged[1]
    return bindings if a == b else None


def unify(a: FS, b: FS, bindings: Bindings | None = None) -> tuple[FS, Bindings] | None:
    """Merge two feature structures, or ``None`` if they conflict.

    The result carries every feature either side had: shared features must agree,
    unshared ones come through untouched. This is the operation that makes a
    grammar rule declarative — ``the adjective agrees with the noun`` is written
    once and works whether the noun's gender is known before the adjective is
    built or after.

        >>> unify(FS(num="sg"), FS(num="pl")) is None
        True
        >>> ok, _ = unify(FS(num="sg"), FS(gen="f"))
        >>> ok
        FS(gen='f', num='sg')
    """
    bindings = dict(bindings or {})
    out = dict(a)
    for k, bv in b.items():
        if k not in out:
            out[k] = bv
            continue
        nb = _unify_values(out[k], bv, bindings)
        if nb is None:
            return None
        bindings = nb
    return FS({k: resolve(v, bindings) for k, v in out.items()}), bindings


def subsumes(general: FS, specific: FS) -> bool:
    """Whether every feature of ``general`` is satisfied by ``specific``.

    Paradigm selection needs this rather than equality: a rule keyed
    ``FS(case="acc")`` should fire for ``FS(case="acc", num="pl", pers=3)``. A
    variable in the general structure matches anything, which is how a paradigm
    says "any number, but the accusative".
    """
    for k, gv in general.items():
        if isinstance(gv, Var):
            continue
        if k not in specific:
            return False
        sv = specific[k]
        if isinstance(sv, Var):
            return False
        if isinstance(gv, FS) and isinstance(sv, FS):
            if not subsumes(gv, sv):
                return False
        elif gv != sv:
            return False
    return True
