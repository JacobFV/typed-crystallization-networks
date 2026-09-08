"""Emitting the curriculum as data: JSONL anyone can train on.

The generators are the asset; a file is just one crystallization of them. Export
as much as you want, from whichever seed range you want, and the only thing you
have to keep straight is that **train and evaluation must not overlap** — on any
axis you intend to draw a conclusion from.

One record per episode::

    {"lesson_id": ..., "seed": ..., "language": ..., "presentation": ...,
     "instance_id": ..., "prompt": ..., "target": ..., "answer": ...,
     "choices": [...], "observation": ..., "metadata": {...}}

``prompt``/``target`` is the pair most training pipelines want; ``target`` is the
expected open-form reply, which under a lettered answer format is not the same
string as ``answer``. ``instance_id`` is shared by every rendering of the same
episode, so records that differ only in surface can be joined — that join is the
agreement measurement, and it is the reason the field exists.

``metadata`` carries the lesson's level, tags, axes and the episode's hidden
ground truth. The hidden state is the structural probe, and it is the one field
you must not feed to a model you intend to then probe with it.

Splits
------
:func:`splits` partitions seeds, which is the classic question. But a system can
memorize a surface as easily as an instance, so :func:`held_out` partitions any
of the other axes — language, answer format, surface — and
:func:`compositional_splits` reads train/eval pairs straight off a curriculum
graph, where everything upstream of a node trains and the node itself is held
out. See ``INTENT.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .languages import DEFAULT_LANGUAGE
from .presentation import Presentation
from .registry import resolve

__all__ = ["iter_records", "write_jsonl", "read_jsonl", "splits", "export",
           "held_out", "compositional_splits", "invariance_set", "iter_structures"]


def _shape(rec: dict[str, Any], *, include_hidden: bool, include_observation: bool) -> dict[str, Any]:
    if not include_hidden:
        rec["metadata"] = {k: v for k, v in rec["metadata"].items() if k != "hidden"}
    if not include_observation:
        rec.pop("observation", None)
    return rec


def iter_records(lessons: str | Sequence[str] | None = None, *, n: int = 100,
                 seed0: int = 0, language: str = DEFAULT_LANGUAGE,
                 presentation: str | Presentation | None = None,
                 difficulty: float | None = None,
                 include_hidden: bool = True,
                 include_observation: bool = True) -> Iterator[dict[str, Any]]:
    """Generate ``n`` episodes per lesson as plain dicts.

    ``observation`` is the prompt without its answer-set block; dropping it
    roughly halves the bytes and loses nothing you cannot rebuild.
    """
    for lesson in resolve(lessons):
        if lesson.status != "implemented":
            continue
        for ex in lesson.examples(n, seed0=seed0, language=language,
                                  presentation=presentation, difficulty=difficulty):
            yield _shape(ex.to_dict(), include_hidden=include_hidden,
                         include_observation=include_observation)


def iter_structures(lessons: str | Sequence[str] | None = None, *, n: int = 100,
                    seed0: int = 0, difficulty: float | None = None) -> Iterator[dict[str, Any]]:
    """The structural form of each episode, for use as a guiding signal.

    This is the generator's own construction — the tree it built, the grammar it
    sampled, the graph it drew — rather than any rendering of it. It is
    modality-invariant by definition, which makes it the natural target for a
    system meant to recover structure rather than to pattern-match a surface, and
    the natural probe for asking whether it did.
    """
    for lesson in resolve(lessons):
        if lesson.status != "implemented":
            continue
        for i in range(n):
            yield lesson.structured(seed0 + i, difficulty=difficulty)


def invariance_set(lesson: str, seed: int, presentations: Iterable[str | Presentation],
                   *, difficulty: float | None = None) -> list[dict[str, Any]]:
    """One episode, rendered several ways, sharing an ``instance_id``.

    The set a consumer needs to ask whether a system answers the same problem the
    same way through different surfaces — a measurement that needs no gold label
    and no judge, because it is agreement rather than correctness.

    Emitting the full cross-product of every seed by every presentation would
    make a corpus of near-duplicates; this exists so that invariance sets are
    something you build deliberately, in a proportion you choose.
    """
    from .registry import get

    l = get(lesson)
    return [l.example(seed, presentation=p, difficulty=difficulty).to_dict()
            for p in presentations]


def write_jsonl(path: str | os.PathLike[str], records: Iterable[dict[str, Any]]) -> int:
    """Write records to a JSONL file, creating parent directories. Returns the count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a JSONL file back into dicts."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def export(path: str | os.PathLike[str], lessons: str | Sequence[str] | None = None, *,
           n: int = 100, seed0: int = 0, language: str = DEFAULT_LANGUAGE,
           presentation: str | Presentation | None = None,
           difficulty: float | None = None,
           include_hidden: bool = True, include_observation: bool = True,
           per_lesson: bool = False, verify_first: bool = False) -> int:
    """Export a dataset.

    With ``per_lesson=False`` (the default) everything goes into one JSONL file
    at ``path``. With ``per_lesson=True``, ``path`` is a directory and each
    lesson gets ``<lesson_id>.jsonl``, which is what the committed sample set
    under ``data/samples/`` looks like.

    ``verify_first`` runs the admission test over the selection and refuses to
    write if anything fails. Nothing grades an open-form corpus after the fact,
    so this is the only place a broken generator can still be caught.
    """
    chosen = [l for l in resolve(lessons) if l.status == "implemented"]
    if verify_first:
        from .verify import failures, verify_all
        bad = failures(verify_all([l.id for l in chosen], episodes=100))
        if bad:
            raise RuntimeError(f"refusing to export: {len(bad)} lessons fail "
                               f"verification ({', '.join(bad[:5])})")
    if not per_lesson:
        return write_jsonl(path, iter_records(lessons, n=n, seed0=seed0, language=language,
                                              presentation=presentation, difficulty=difficulty,
                                              include_hidden=include_hidden,
                                              include_observation=include_observation))
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    total = 0
    for lesson in chosen:
        recs = (_shape(ex.to_dict(), include_hidden=include_hidden,
                       include_observation=include_observation)
                for ex in lesson.examples(n, seed0=seed0, language=language,
                                          presentation=presentation, difficulty=difficulty))
        total += write_jsonl(root / f"{lesson.id}.jsonl", recs)
    return total


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def splits(train: int = 1000, eval: int = 200, *, gap: int = 1_000_000) -> dict[str, tuple[int, int]]:
    """Disjoint seed ranges, as ``{"train": (seed0, n), "eval": (seed0, n)}``.

    Disjointness here is by construction rather than by sampling: the two ranges
    cannot overlap, so an evaluation episode is a world the training set could
    not have contained.
    """
    return {"train": (0, train), "eval": (gap, eval)}


def held_out(values: Sequence[str], *, fraction: float = 0.25,
             salt: str = "") -> dict[str, list[str]]:
    """Split any presentation axis into train and eval halves, deterministically.

    Use it on languages, surfaces or answer formats. Training on text and
    rasterized episodes and evaluating on dictated ones asks whether the surface
    was incidental; training on three hundred languages and evaluating on a
    hundred unseen ones asks whether the language was.

    The assignment is a hash rather than a slice, so adding a value to the list
    does not reshuffle everything already assigned.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be strictly between 0 and 1, got {fraction}")
    cut = int(fraction * (1 << 32))
    train, ev = [], []
    for v in values:
        h = hashlib.blake2b(f"{salt}|{v}".encode(), digest_size=4).digest()
        (ev if int.from_bytes(h, "big") < cut else train).append(v)
    return {"train": train, "eval": ev}


def compositional_splits(curriculum: str | Any = "progressive") -> list[dict[str, Any]]:
    """Train/eval pairs read off a curriculum graph, one per node with ancestors.

    For each node, everything upstream is fair to train on and the node itself is
    held out. If a system has learned the structure rather than the instances,
    the held-out composition should be reachable from its parts — which is the
    question the whole resource exists to ask.
    """
    from .curricula import get as get_curriculum

    c = get_curriculum(curriculum)
    return [{"node": key, "curriculum": c.id, "train": list(train), "eval": list(ev)}
            for key, train, ev in c.splits()]
