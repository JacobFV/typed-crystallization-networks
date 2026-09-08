"""Addressing an episode, and drawing batches from a set nobody can enumerate.

``(lesson, seed, difficulty, presentation)`` is a pure function to bytes. That is
the whole storage design: any slice of the corpus is reproducible by anyone who
has the package, so object storage is a *cache* rather than the store of record,
and there is no manifest to keep in sync and nothing to drift.

Two consequences are worth stating plainly.

**Renderer versions belong in the key.** A rendering cached under an address that
does not mention which renderer made it becomes silently wrong the moment the
renderer changes. :meth:`Address.cache_key` includes it; the plain
:meth:`Address.key` does not, because the *episode* is the same episode however
it was drawn.

**An infinite set cannot be shuffled.** You cannot list the corpus to sample it,
so :class:`Space` defines the axes and :func:`draw` maps a batch index to an
address through a keyed bijection. Batches are reproducible, non-overlapping and
unbiased, and nothing is materialized to make them. The bijection is a small
Feistel network with cycle-walking, which is the standard way to permute a range
without writing the range down.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from .presentation import DEFAULT_PRESENTATION, Presentation

__all__ = ["Address", "Space", "draw", "batch", "permute", "SPACE_KEY"]

#: Domain-separates the permutation. Changing it reshuffles every batch index, so
#: it is part of the format rather than a tuning knob.
SPACE_KEY = b"langcurriculum/address/v1"


@dataclass(frozen=True)
class Address:
    """Everything needed to reproduce one episode, and nothing else."""

    lesson: str
    seed: int = 0
    difficulty: float | None = None
    presentation: Presentation = DEFAULT_PRESENTATION

    def key(self) -> str:
        """Identifies the episode. Two renderers agreeing on this agree on the task."""
        d = "-" if self.difficulty is None else f"{self.difficulty:.6f}"
        return f"{self.lesson}/{self.seed}/{d}/{self.presentation.key()}"

    def cache_key(self, renderer_version: str = "") -> str:
        """Identifies the *bytes*. Includes whatever produced them."""
        from .surfaces import RENDERER_VERSIONS
        version = renderer_version or RENDERER_VERSIONS.get(self.presentation.surface, "?")
        return f"{self.key()}@{version}"

    def digest(self, renderer_version: str = "") -> str:
        """A short content-addressable name for a cached rendering."""
        return hashlib.blake2b(self.cache_key(renderer_version).encode(),
                               digest_size=16).hexdigest()

    def instance(self) -> str:
        """The problem's identity, independent of presentation.

        Shared by every rendering of one episode, which is what makes agreement
        across surfaces measurable without a gold label. See ``INTENT.md``.
        """
        from .lesson import instance_id
        return instance_id(self.lesson, self.seed, self.difficulty)

    def example(self):
        """Materialize the episode."""
        from .registry import get
        return get(self.lesson).example(self.seed, presentation=self.presentation,
                                        difficulty=self.difficulty)

    def content(self, **options: Any):
        """Materialize the episode and render it into its surface."""
        from .surfaces import transcode_example
        return transcode_example(self.example(), self.presentation.surface, **options)

    def to_dict(self) -> dict[str, Any]:
        return {"lesson": self.lesson, "seed": self.seed, "difficulty": self.difficulty,
                "presentation": self.presentation.key()}

    def __str__(self) -> str:
        return self.key()


@dataclass(frozen=True)
class Space:
    """The axes a batch is drawn from, as sizes rather than as contents.

    Seeds are a *range*, not a list, which is the point: the space is far larger
    than anything that will be drawn from it, and stays that way.
    """

    lessons: tuple[str, ...]
    seeds: tuple[int, int] = (0, 1_000_000)
    difficulties: tuple[float | None, ...] = (None,)
    presentations: tuple[Presentation, ...] = (DEFAULT_PRESENTATION,)

    def __post_init__(self) -> None:
        if not self.lessons:
            raise ValueError("a space needs at least one lesson")
        if self.seeds[1] <= self.seeds[0]:
            raise ValueError(f"empty seed range {self.seeds}")
        if not self.difficulties or not self.presentations:
            raise ValueError("a space needs at least one difficulty and presentation")

    @property
    def n_seeds(self) -> int:
        return self.seeds[1] - self.seeds[0]

    def __len__(self) -> int:
        """How many distinct episodes the space contains.

        Large, and deliberately never iterated. This is the number a batch index
        is permuted within.
        """
        return (len(self.lessons) * self.n_seeds
                * len(self.difficulties) * len(self.presentations))

    def at(self, index: int) -> Address:
        """The address at a raw (unpermuted) index. Mixed-radix, lowest axis last."""
        n = len(self)
        if not 0 <= index < n:
            raise IndexError(f"index {index} outside a space of {n}")
        index, p = divmod(index, len(self.presentations))
        index, d = divmod(index, len(self.difficulties))
        index, s = divmod(index, self.n_seeds)
        lesson = self.lessons[index]
        return Address(lesson=lesson, seed=self.seeds[0] + s,
                       difficulty=self.difficulties[d], presentation=self.presentations[p])

    def describe(self) -> dict[str, Any]:
        return {"lessons": len(self.lessons), "seeds": list(self.seeds),
                "difficulties": list(self.difficulties),
                "presentations": [p.key() for p in self.presentations],
                "size": len(self)}


# --------------------------------------------------------------------------
# permuting a range without writing it down
# --------------------------------------------------------------------------
def _prf(key: bytes, round_no: int, value: int, width: int) -> int:
    h = hashlib.blake2b(key + round_no.to_bytes(2, "big") + value.to_bytes(8, "big"),
                        digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << width) - 1)


def permute(index: int, size: int, *, key: bytes = SPACE_KEY, rounds: int = 4) -> int:
    """A keyed bijection of ``[0, size)`` onto itself.

    A four-round Feistel network over the smallest even bit width that covers
    ``size``, then cycle-walking until the result lands back in range. Both parts
    are standard, and together they give a permutation that can be evaluated at
    one point without constructing the rest of it — which is what lets a batch be
    drawn from a space too large to enumerate.
    """
    if not 0 <= index < size:
        raise IndexError(f"index {index} outside a space of {size}")
    if size == 1:
        return 0
    bits = max(2, (size - 1).bit_length())
    bits += bits % 2                                  # even, so the halves match
    half = bits // 2
    mask = (1 << half) - 1
    x = index
    while True:
        left, right = x >> half, x & mask
        for r in range(rounds):
            left, right = right, left ^ _prf(key, r, right, half)
        y = (left << half) | right
        if y < size:
            return y
        x = y                                          # cycle-walk


def draw(space: Space, index: int, *, key: bytes = SPACE_KEY) -> Address:
    """The address at a *shuffled* batch index."""
    return space.at(permute(index, len(space), key=key))


def batch(space: Space, start: int, n: int, *, key: bytes = SPACE_KEY) -> Iterator[Address]:
    """``n`` addresses from a batch, starting at ``start``.

    Two batches that do not overlap in index do not overlap in content, because
    the permutation is a bijection — disjointness by construction rather than by
    checking, which is the same discipline the seed splits already use.
    """
    size = len(space)
    for i in range(start, start + n):
        yield draw(space, i % size, key=key)
