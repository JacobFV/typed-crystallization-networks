"""The curricula registry: named opinions about how the lessons fit together.

Four ship, and they deliberately disagree with each other:

===============  ==========================================================
``canonical``    every lesson, in the order the package used to declare them
``core170``      the original numbered sequence, order preserved, no edges
``supplementary``the ten lessons that sat outside that sequence
``everything``   all 180, alphabetical, no edges — the null opinion
``progressive``  edges derived from the declared difficulty axes
===============  ==========================================================

plus a family built on demand: ``tag:<name>`` and ``capability:<name>`` slice the
registry by something a lesson declares about itself.

Curricula that are derived rather than written down are built lazily and cached,
for the same reason the language registry does it: computing every one of them to
serve a caller who wants one would be indefensible, and most callers want one.
"""

from __future__ import annotations

from typing import Iterator

from ..lesson import AXES
from .core170 import CANONICAL, CORE170, SUPPLEMENTARY
from .graph import Curriculum, CurriculumError, Node, derive_edges, transitive_reduction

__all__ = ["Curriculum", "Node", "CurriculumError", "derive_edges", "transitive_reduction",
           "CORE170", "SUPPLEMENTARY", "CANONICAL", "get", "curriculum_ids", "curricula",
           "everything", "progressive", "by_tag", "by_capability"]

#: the curricula written down rather than derived
_DECLARED: dict[str, Curriculum] = {c.id: c for c in (CANONICAL, CORE170, SUPPLEMENTARY)}

#: derived curricula, built once on demand. Kept apart from :data:`_DECLARED` on
#: purpose — that dict is what the package *states* it has, and a cache must not
#: change a statement.
_CACHE: dict[str, Curriculum] = {}


def _lessons():
    from ..registry import all_lessons
    return all_lessons()


def everything() -> Curriculum:
    """Every lesson, alphabetically, with no edges at all.

    The null opinion, and a useful one: it is what you want when you mean "the
    whole registry" and do not want to imply an order by saying so.
    """
    if "everything" not in _CACHE:
        _CACHE["everything"] = Curriculum(
            id="everything", title="every lesson",
            description="All lessons, alphabetically. No ordering claim is made.",
            nodes=tuple(Node(lid, order_hint=i)
                        for i, lid in enumerate(sorted(_lessons()))),
        )
    return _CACHE["everything"]


def progressive() -> Curriculum:
    """Every lesson, ordered by what the axes actually say.

    The edges are derived (see :func:`~langcurriculum.curricula.graph.derive_edges`):
    X comes before Y when Y is at least as demanding on every declared axis and
    strictly more on one, and the two are related by a shared tag or capability.

    This is the curriculum to reach for when you want compositional splits, since
    an edgeless graph affords none — ``train_eval_split`` needs ancestors to
    exist before it can hold anything out.
    """
    if "progressive" not in _CACHE:
        lessons = _lessons()
        edges = derive_edges(lessons, axes=AXES)
        _CACHE["progressive"] = Curriculum(
            id="progressive", title="ordered by declared difficulty",
            description=("Every lesson, with prerequisite edges derived from the "
                         "difficulty axes each lesson declares about itself. "
                         "Auditable, and wrong in ways someone can point at."),
            nodes=tuple(Node(lid, order_hint=i) for i, lid in enumerate(sorted(lessons))),
            edges=edges,
        ).validate(known_lessons=lessons)
    return _CACHE["progressive"]


def by_tag(tag: str) -> Curriculum:
    """The lessons carrying a tag, alphabetically."""
    lessons = _lessons()
    members = sorted(lid for lid, l in lessons.items() if tag in l.tags)
    if not members:
        raise CurriculumError(f"no lesson tagged {tag!r}; try one of {sorted(tags())}")
    return Curriculum(id=f"tag:{tag}", title=f"lessons tagged {tag}",
                      description=f"Every lesson declaring the tag {tag!r}.",
                      nodes=tuple(Node(m, order_hint=i) for i, m in enumerate(members)))


def by_capability(capability: str) -> Curriculum:
    """The lessons exercising a capability, alphabetically."""
    lessons = _lessons()
    members = sorted(lid for lid, l in lessons.items() if capability in l.capabilities)
    if not members:
        raise CurriculumError(f"no lesson exercises {capability!r}")
    return Curriculum(id=f"capability:{capability}",
                      title=f"lessons exercising {capability}",
                      description=f"Every lesson declaring the capability {capability!r}.",
                      nodes=tuple(Node(m, order_hint=i) for i, m in enumerate(members)))


def tags() -> dict[str, list[str]]:
    """tag -> the lessons carrying it."""
    out: dict[str, list[str]] = {}
    for lid, l in _lessons().items():
        for t in l.tags:
            out.setdefault(t, []).append(lid)
    return out


def curriculum_ids() -> list[str]:
    """Every curriculum available by name, declared ones first."""
    return [*_DECLARED, "everything", "progressive"]


def get(spec: str | Curriculum | None = None) -> Curriculum:
    """The curriculum for a name.

    ``None`` gives ``everything``. ``tag:x`` and ``capability:x`` are built on
    demand from what the lessons declare.
    """
    if isinstance(spec, Curriculum):
        return spec
    name = (spec or "everything").strip()
    if name in _DECLARED:
        return _DECLARED[name]
    if name == "everything":
        return everything()
    if name == "progressive":
        return progressive()
    if name.startswith("tag:"):
        return by_tag(name[4:])
    if name.startswith("capability:"):
        return by_capability(name[11:])
    raise CurriculumError(
        f"unknown curriculum {spec!r}; try one of {curriculum_ids()}, "
        f"or tag:<name> / capability:<name>")


def curricula() -> Iterator[Curriculum]:
    """Every named curriculum, built."""
    for name in curriculum_ids():
        yield get(name)
