"""What a generator is told about the episode it is being asked for.

A generator is a pure function of a :class:`random.Random`, and for most lessons
that is the whole story. Two things it sometimes has to know are not derivable
from the seed:

* **which language the episode will be read in** — the morphology lessons draw
  their inflected material from the pack the episode is rendered in, so a lesson
  about agreement asked for in Turkish has to sample Turkish forms;
* **how hard the episode should be** — the difficulty knob that turns a lesson
  from one point into a curve, so that a curriculum can be a schedule rather than
  just an ordering.

Both arrive in a :class:`GenerationContext`. Lessons opt in by declaring a second
parameter on ``generate``; the ones that do not are called with a single argument
exactly as before, which is why adding this cost nothing across a hundred and
eighty modules.

Difficulty is deliberately *not* part of a presentation. It changes the problem,
not the surface, so it belongs with the seed — see ``INTENT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

from .languages import DEFAULT_LANGUAGE

__all__ = ["GenerationContext", "HARDENING", "DEFAULT_HARDENING",
           "NO_HARDENING", "resolve_hardening"]

T = TypeVar("T")

#: Anti-exploit draws, named by the lesson each one belongs to.
#:
#: A lesson is in this set because an audit found a cheap heuristic -- copying a
#: token, counting characters, taking the last-mentioned option -- reaching the
#: lesson's own accuracy without doing what the lesson is named after.  The
#: named draw changes *how episodes are sampled*, never what the lesson asks or
#: how its answer is computed.  See ``research/lesson-audit/RESULTS.md``.
HARDENING: frozenset[str] = frozenset({
    "center_embedding",
    "context_free_language",
    "ellipsis",
    "language_culture",
    "nesting_depth_compare",
    "next_symbol",
    "paradigm_shift",
    "parse_depth",
    "presupposition",
    "symbol_discrimination",
    "symbol_equivalence",
    "tree_to_sequence",
    "underspecification_reasoning",
    "unification",
})

#: What a caller that says nothing gets.  The hardened draw is the default
#: because an exploitable lesson is worse than a changed distribution; the
#: previous stream stays reachable, exactly, as ``hardening="none"``.
DEFAULT_HARDENING: frozenset[str] = HARDENING

#: Every pre-audit draw, bit-identical to the catalogue before the audit.
NO_HARDENING: frozenset[str] = frozenset()


def resolve_hardening(spec: Any) -> frozenset[str]:
    """Read a hardening selection from whatever a caller had to hand.

    ``None`` means the default set, ``"none"``/``"off"``/``False`` the empty
    one, ``"all"``/``True`` every named draw, and any iterable or
    comma-separated string names the draws to enable.  Unknown names are an
    error rather than a silent no-op, because a misspelled fix that quietly does
    nothing is exactly the failure this whole exercise is about.
    """
    if spec is None:
        return DEFAULT_HARDENING
    if spec is True:
        return HARDENING
    if spec is False:
        return NO_HARDENING
    if isinstance(spec, str):
        key = spec.strip().casefold()
        if key in ("", "default"):
            return DEFAULT_HARDENING
        if key in ("none", "off", "legacy"):
            return NO_HARDENING
        if key in ("all", "on"):
            return HARDENING
        names = [x.strip() for x in spec.split(",") if x.strip()]
    else:
        names = [str(x).strip() for x in spec]
    unknown = sorted(set(names) - HARDENING)
    if unknown:
        raise ValueError(f"unknown hardening name(s): {unknown}; "
                         f"known: {sorted(HARDENING)}")
    return frozenset(names)


@dataclass(frozen=True)
class GenerationContext:
    """The episode's language and difficulty, handed to generators that want them.

    ``difficulty`` is ``None`` when the caller did not ask for one, which means
    "use whatever you would normally do". A lesson must behave identically to its
    pre-difficulty self in that case, or the whole committed sample set moves
    under it.
    """

    language: str = DEFAULT_LANGUAGE
    difficulty: float | None = None
    #: which anti-exploit draws are in force for this episode
    hardening: frozenset[str] = DEFAULT_HARDENING

    def __post_init__(self) -> None:
        if self.difficulty is not None and not 0.0 <= self.difficulty <= 1.0:
            raise ValueError(f"difficulty must be in [0, 1], got {self.difficulty}")
        if not isinstance(self.hardening, frozenset):
            object.__setattr__(self, "hardening", resolve_hardening(self.hardening))

    # ---- sampling regime --------------------------------------------
    def hardens(self, name: str) -> bool:
        """Whether the named anti-exploit draw applies to this episode.

        A lesson asks this once and branches; the unhardened branch has to be
        left exactly as it was, because that is what the legacy stream check in
        ``tests/test_language_hardening.py`` compares against.
        """
        if name not in HARDENING:
            raise ValueError(f"{name!r} is not a declared hardening name")
        return name in self.hardening

    # ---- knobs -------------------------------------------------------
    def at(self, lo: int, hi: int, *, default: int | None = None) -> int:
        """An integer knob interpolated across ``[lo, hi]`` by difficulty.

        With no difficulty set, ``default`` is returned — and a lesson should
        pass whatever constant it used before, so that the unset case reproduces
        its old behaviour exactly.
        """
        if self.difficulty is None:
            return lo if default is None else default
        return int(round(lo + self.difficulty * (hi - lo)))

    def span(self, lo: tuple[int, int], hi: tuple[int, int]) -> tuple[int, int]:
        """A ``(low, high)`` pair for ``rng.randint``, interpolated end to end."""
        if self.difficulty is None:
            return lo
        return (self.at(lo[0], hi[0]), self.at(lo[1], hi[1]))

    def among(self, items: Sequence[T]) -> T:
        """The element of an ordered list that this difficulty selects."""
        if not items:
            raise ValueError("among() needs at least one item")
        if self.difficulty is None:
            return items[0]
        return items[min(len(items) - 1, int(self.difficulty * len(items)))]

    @property
    def scaled(self) -> bool:
        """Whether a difficulty was actually asked for."""
        return self.difficulty is not None
