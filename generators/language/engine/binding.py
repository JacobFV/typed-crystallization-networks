"""The one hook that lets something outside choose this curriculum's words.

Written here rather than upstream — see :data:`symbolic_ai_data._vendor.ADDED`.
It knows nothing about knowledge graphs; it is a protocol and a context
variable, and the policy that fills it in lives in :mod:`symbolic_ai_kg`.

Why a context variable
----------------------

The same shape :data:`symbolic_ai_data.generators.extra.ACTIVE_LANGUAGE` already
uses, and for the same reason: 57 lesson modules import ``COLORS``, ``SHAPES``
and ``NAMES`` by name at module scope, and threading a binder through 179
generator signatures would be a rewrite of the corpus rather than a seam into
it.

**Inert unless set.** With no binder active, :class:`SlotVocabulary` returns its
default list and :func:`bind_scene` returns ``None``, so every draw is the draw
that was always made. ``tests/test_curriculum_fingerprint.py`` hashes 179
lessons x 5 seeds x 7 languages and is the guard that says so; that test firing
means this file broke the corpus, and the correct response is to fix this file,
never to re-record the fingerprint.

The invariant that makes the comparison worth anything
------------------------------------------------------

A bound vocabulary **must be the same length as the one it replaces**. Not a
style preference: ``rng.choice`` and ``rng.sample`` consume a number of random
bits that depends on the population size, so a vocabulary of a different length
desynchronises the rest of the episode's random stream and every later draw — the
scene size, the coordinates, which object is asked about — comes out different.
The curriculum would then differ from its baseline in the symbols *and* in
everything downstream of them, and no score difference could be attributed to
either. :meth:`SlotVocabulary.bind` enforces it.

That invariant is what makes the KG arm a clean manipulation: same tree shapes,
same arities, same scene sizes, same coordinates, same question — different
symbols, drawn under constraint.
"""

from __future__ import annotations

import random
from contextvars import ContextVar
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

__all__ = ["Binder", "ACTIVE_BINDER", "SlotVocabulary", "active_binder",
           "bind_scene", "using"]


@runtime_checkable
class Binder(Protocol):
    """What a policy has to provide. Deliberately three methods.

    ``vocabulary`` answers "what are this episode's six colours", ``entity``
    answers "given that this object's kind is X, what may its other slot be",
    and ``describe`` is what lands in a run directory so a number can be
    explained later without refitting anything.
    """

    def vocabulary(self, slot: str, default: Sequence[str]) -> Sequence[str]:
        ...

    def entity(self, slot_values: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def describe(self) -> Mapping[str, Any]:
        ...


#: The active policy, or ``None``. Never read except through
#: :func:`active_binder`, so there is one place to put a breakpoint.
ACTIVE_BINDER: ContextVar[Binder | None] = ContextVar("ACTIVE_BINDER", default=None)


def active_binder() -> Binder | None:
    return ACTIVE_BINDER.get()


class using:
    """Context manager: run a block with a binder active.

    ``with using(b): ...`` rather than a bare ``set``/``reset`` pair, because the
    reset has to happen on the exception path too — the morphology lessons had
    exactly this bug with ``ACTIVE_LANGUAGE`` before it was wrapped in
    ``try/finally``.
    """

    def __init__(self, binder: Binder | None) -> None:
        self._binder = binder
        self._token = None

    def __enter__(self) -> "Binder | None":
        self._token = ACTIVE_BINDER.set(self._binder)
        return self._binder

    def __exit__(self, *exc: object) -> None:
        ACTIVE_BINDER.reset(self._token)
        return None


class SlotVocabulary(Sequence):
    """A closed word list that a binder may replace, and nothing else may.

    Behaves as the plain ``list`` it replaced in every way the 57 importing
    modules use it — indexing, slicing, iteration, ``in``, ``+``, ``==``,
    ``rng.choice``, ``rng.sample`` — and is a real
    :class:`collections.abc.Sequence` so ``random.sample``'s type check passes.
    """

    __slots__ = ("_slot", "_default")

    def __init__(self, slot: str, default: Sequence[str]) -> None:
        self._slot = slot
        self._default = list(default)

    # ---- the seam ----------------------------------------------------
    @property
    def slot(self) -> str:
        return self._slot

    @property
    def default(self) -> list[str]:
        """The list this was before anything could rebind it."""
        return list(self._default)

    def current(self) -> list[str]:
        b = ACTIVE_BINDER.get()
        if b is None:
            return self._default
        got = list(b.vocabulary(self._slot, self._default))
        if len(got) != len(self._default):
            raise ValueError(
                f"binder returned {len(got)} words for slot {self._slot!r} where "
                f"the curriculum has {len(self._default)}; a different length "
                f"desynchronises every later draw in the episode -- see "
                f"symbolic_ai_data/binding.py")
        return got

    # ---- Sequence ----------------------------------------------------
    def __len__(self) -> int:
        return len(self._default)

    def __getitem__(self, i):
        return self.current()[i]

    def __iter__(self):
        return iter(self.current())

    def __contains__(self, x: object) -> bool:
        return x in self.current()

    def __add__(self, other):
        return self.current() + list(other)

    def __radd__(self, other):
        return list(other) + self.current()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SlotVocabulary):
            return self.current() == other.current()
        if isinstance(other, (list, tuple)):
            return self.current() == list(other)
        return NotImplemented

    def __hash__(self) -> int:                       # pragma: no cover
        return hash((self._slot, tuple(self._default)))

    def index(self, value, *a):
        return self.current().index(value, *a)

    def count(self, value) -> int:
        return self.current().count(value)

    def __repr__(self) -> str:
        cur = self.current()
        tail = "" if cur == self._default else f" (bound from {self._default})"
        return f"<{self._slot} {cur}{tail}>"


def bind_scene(objs: Sequence[dict], rng: random.Random) -> list[dict] | None:
    """Give the active binder a chance to re-bind an already-drawn scene.

    The scene is drawn first, with the ordinary uniform draws, and only then
    handed over. That ordering is the point: the random stream is spent exactly
    as it always was, and the binder rewrites the *symbols* it produced under
    constraint from each other. A binder that wanted to draw its own scene could
    not keep the coordinates and the scene size identical, and then the KG arm
    and the baseline would differ in more than one thing at once.

    Returns ``None`` when no binder is active, which is the caller's signal to
    keep what it drew.
    """
    b = ACTIVE_BINDER.get()
    if b is None:
        return None
    return [dict(b.entity(o)) for o in objs]
