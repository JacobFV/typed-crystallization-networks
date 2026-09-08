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
from typing import Sequence, TypeVar

from .languages import DEFAULT_LANGUAGE

__all__ = ["GenerationContext"]

T = TypeVar("T")


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

    def __post_init__(self) -> None:
        if self.difficulty is not None and not 0.0 <= self.difficulty <= 1.0:
            raise ValueError(f"difficulty must be in [0, 1], got {self.difficulty}")

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
