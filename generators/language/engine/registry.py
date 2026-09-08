"""The registry: lesson id -> lesson class, and the ways of slicing it.

The registry is deliberately explicit. Every lesson class is imported by name
in :mod:`langcurriculum.lessons`, so what is in the curriculum is a fact you can
read off the source tree rather than the result of a directory scan that might
quietly skip a file that failed to import.

What the registry no longer does is order anything. It is a flat mapping, and
every question of the form "which comes first" belongs to a
:class:`~langcurriculum.curricula.Curriculum`. The slices that survive here —
by tag, by capability — are the ones a lesson answers about *itself*.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .lesson import Lesson
from .lessons import LESSON_CLASSES

__all__ = ["REGISTRY", "all_lessons", "get", "lesson_ids", "by_tag", "by_capability",
           "tags", "capabilities", "resolve"]

#: lesson id -> the class implementing it
REGISTRY: Mapping[str, type[Lesson]] = {c.id: c for c in LESSON_CLASSES}

_INSTANCES: dict[str, Lesson] = {}

if len(REGISTRY) != len(LESSON_CLASSES):  # pragma: no cover - guards a build mistake
    raise RuntimeError("duplicate lesson id in the registry")


def get(lesson_id: str) -> Lesson:
    """The lesson with this id, as a ready-to-use instance."""
    if lesson_id not in REGISTRY:
        raise KeyError(f"unknown lesson {lesson_id!r}; "
                       f"there are {len(REGISTRY)}, try lesson_ids()")
    inst = _INSTANCES.get(lesson_id)
    if inst is None:
        inst = _INSTANCES[lesson_id] = REGISTRY[lesson_id]()
    return inst


def all_lessons(*, implemented_only: bool = False) -> dict[str, Lesson]:
    """Every lesson, alphabetically, id -> instance."""
    out = {c.id: get(c.id) for c in sorted(LESSON_CLASSES, key=lambda c: c.id)}
    if implemented_only:
        out = {k: v for k, v in out.items() if v.status == "implemented"}
    return out


def lesson_ids(*, implemented_only: bool = False) -> list[str]:
    return list(all_lessons(implemented_only=implemented_only))


def by_tag() -> dict[str, list[str]]:
    """tag -> the lessons carrying it."""
    out: dict[str, list[str]] = {}
    for l in all_lessons().values():
        for t in l.tags:
            out.setdefault(t, []).append(l.id)
    return out


def by_capability() -> dict[str, list[str]]:
    """capability tag -> the lessons that exercise it."""
    out: dict[str, list[str]] = {}
    for l in all_lessons().values():
        for c in l.capabilities:
            out.setdefault(c, []).append(l.id)
    return out


def tags() -> list[str]:
    return sorted(by_tag())


def capabilities() -> list[str]:
    return sorted(by_capability())


def resolve(spec: str | Sequence[str] | None) -> list[Lesson]:
    """Turn a selector into lessons.

    ``None`` or ``"all"`` means every implemented lesson. Otherwise:

    ==========================  ===================================================
    ``core170``, ``progressive``a curriculum by name, in its own order
    ``curriculum:<name>``       the same, spelled out
    ``tag:<name>``              every lesson carrying a tag
    ``capability:<name>``       every lesson exercising a capability
    ``a,b,c``                   lesson ids
    ==========================  ===================================================
    """
    if spec is None or spec in ("all", ""):
        return list(all_lessons(implemented_only=True).values())
    if isinstance(spec, str):
        name = spec.strip()
        from .curricula import curriculum_ids, get as get_curriculum
        if name.startswith("curriculum:"):
            name = name[len("curriculum:"):]
        if name in curriculum_ids() or name.startswith(("tag:", "capability:")):
            chosen = get_curriculum(name)
            return [get(n.lesson) for n in chosen.linearize()
                    if get(n.lesson).status == "implemented"]
        names = [s.strip() for s in name.split(",") if s.strip()]
    else:
        names = list(spec)
    return [get(n) for n in names]
