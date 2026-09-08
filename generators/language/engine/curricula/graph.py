"""A curriculum: a directed acyclic graph over lessons, and the ways to walk it.

Lessons are flat. A curriculum is an *opinion* about them — which ones belong
together, which must come before which, how hard each should be asked at. There
may be many curricula over the same lessons, overlapping and disagreeing, and
none of them is privileged. That is why a lesson carries no number and no
section: those were one curriculum's opinion written onto the material itself.

Two things a graph gives you that an ordered list does not.

**Many flattenings.** A DAG usually admits an enormous number of valid
topological orders, and each one is a legitimate sequence in which the material
could be taught. :meth:`Curriculum.linearize` returns a canonical one — canonical
because the site and the committed samples need to be reproducible — and
:meth:`Curriculum.linearizations` enumerates alternatives.

**Compositional splits.** For any node, everything upstream of it is a training
set and the node itself is an evaluation set, so the graph *is* a generator of
compositional-generalization tests. That is the point of the whole exercise, and
:meth:`Curriculum.train_eval_split` is where it lands. See ``INTENT.md``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = ["Node", "Curriculum", "CurriculumError", "derive_edges", "transitive_reduction"]


class CurriculumError(ValueError):
    """A curriculum that does not describe a usable graph."""


@dataclass(frozen=True)
class Node:
    """One position in a curriculum.

    A node is not a lesson; it is *a lesson, asked a particular way*. The same
    lesson may appear twice in one curriculum under different keys — once in
    text and once dictated, say — and an edge between them is a claim about
    transfer that can actually be tested rather than a decorative arrow.
    """

    lesson: str
    #: unique within the curriculum; defaults to the lesson id
    key: str = ""
    label: str = ""
    #: this curriculum's opinion of the difficulty level, overriding the lesson's
    level: int | None = None
    #: pin the surface this node is taught through, as a presentation key
    presentation: str | None = None
    #: the generator difficulty this node asks for, if the lesson supports one
    difficulty: float | None = None
    #: tie-break when several nodes become available at once
    order_hint: int = 0

    def __post_init__(self) -> None:
        if not self.lesson:
            raise CurriculumError("a node needs a lesson id")
        if not self.key:
            object.__setattr__(self, "key", self.lesson)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"lesson": self.lesson, "key": self.key}
        for f in ("label", "level", "presentation", "difficulty", "order_hint"):
            v = getattr(self, f)
            if v not in ("", None, 0):
                d[f] = v
        return d


@dataclass(frozen=True)
class Curriculum:
    """An ordering opinion over lessons, as a DAG."""

    id: str
    title: str
    description: str = ""
    nodes: tuple[Node, ...] = ()
    #: ``(prerequisite_key, dependent_key)`` pairs
    edges: tuple[tuple[str, str], ...] = ()
    _index: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        idx = {}
        for i, n in enumerate(self.nodes):
            if n.key in idx:
                raise CurriculumError(f"{self.id}: duplicate node key {n.key!r}")
            idx[n.key] = i
        object.__setattr__(self, "_index", idx)

    # ---- basics ------------------------------------------------------
    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __contains__(self, key: str) -> bool:
        return key in self._index

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(n.key for n in self.nodes)

    @property
    def lessons(self) -> tuple[str, ...]:
        """The distinct lesson ids this curriculum draws on, in node order."""
        seen: dict[str, None] = {}
        for n in self.nodes:
            seen.setdefault(n.lesson, None)
        return tuple(seen)

    def node(self, key: str) -> Node:
        try:
            return self.nodes[self._index[key]]
        except KeyError:
            raise CurriculumError(f"{self.id}: no node {key!r}") from None

    # ---- structure ---------------------------------------------------
    def prerequisites(self, key: str) -> tuple[str, ...]:
        return tuple(a for a, b in self.edges if b == key)

    def dependents(self, key: str) -> tuple[str, ...]:
        return tuple(b for a, b in self.edges if a == key)

    def ancestors(self, key: str) -> frozenset[str]:
        """Everything that must be learned before ``key``, transitively."""
        out: set[str] = set()
        stack = list(self.prerequisites(key))
        while stack:
            k = stack.pop()
            if k in out:
                continue
            out.add(k)
            stack.extend(self.prerequisites(k))
        return frozenset(out)

    def descendants(self, key: str) -> frozenset[str]:
        out: set[str] = set()
        stack = list(self.dependents(key))
        while stack:
            k = stack.pop()
            if k in out:
                continue
            out.add(k)
            stack.extend(self.dependents(k))
        return frozenset(out)

    def roots(self) -> tuple[str, ...]:
        """Nodes with no prerequisites — where a learner may start."""
        has_prereq = {b for _a, b in self.edges}
        return tuple(n.key for n in self.nodes if n.key not in has_prereq)

    def leaves(self) -> tuple[str, ...]:
        has_dep = {a for a, _b in self.edges}
        return tuple(n.key for n in self.nodes if n.key not in has_dep)

    # ---- validation --------------------------------------------------
    def validate(self, *, known_lessons: Iterable[str] | None = None) -> "Curriculum":
        """Check the graph is usable, and return it so this can wrap a literal.

        Raises rather than warns. A curriculum that names a lesson which does not
        exist, or that contains a cycle, cannot be walked at all, and finding
        that out at import is much better than finding it out halfway through a
        million-episode export.
        """
        if known_lessons is not None:
            known = set(known_lessons)
            missing = sorted({n.lesson for n in self.nodes} - known)
            if missing:
                raise CurriculumError(f"{self.id}: unknown lessons {missing}")
        for a, b in self.edges:
            if a not in self._index:
                raise CurriculumError(f"{self.id}: edge from unknown node {a!r}")
            if b not in self._index:
                raise CurriculumError(f"{self.id}: edge to unknown node {b!r}")
            if a == b:
                raise CurriculumError(f"{self.id}: self-edge on {a!r}")
        self._toposort()                     # raises on a cycle
        return self

    # ---- flattening --------------------------------------------------
    def _toposort(self, priority=None) -> list[str]:
        """Kahn's algorithm with a deterministic ready-set order.

        A DAG has many topological orders and the caller needs one particular
        one, or the published site and the committed samples move under it. The
        ready set is therefore a sorted list rather than a set, keyed by
        ``priority``.
        """
        priority = priority or (lambda k: (self.node(k).order_hint, self._index[k]))
        indeg = {n.key: 0 for n in self.nodes}
        for _a, b in self.edges:
            indeg[b] += 1
        ready = sorted((k for k, d in indeg.items() if d == 0), key=priority)
        out: list[str] = []
        while ready:
            k = ready.pop(0)
            out.append(k)
            for b in self.dependents(k):
                indeg[b] -= 1
                if indeg[b] == 0:
                    ready.append(b)
                    ready.sort(key=priority)
        if len(out) != len(self.nodes):
            stuck = sorted(set(indeg) - set(out))
            raise CurriculumError(f"{self.id}: cycle among {stuck[:8]}")
        return out

    def linearize(self, strategy: str = "default") -> tuple[Node, ...]:
        """One valid teaching order, deterministically chosen.

        ``default`` follows the declared order hints; ``level`` prefers whatever
        the curriculum calls easiest next; ``breadth`` finishes a layer before
        starting the next; ``depth`` follows one thread as far as it goes.
        """
        if strategy == "default":
            keys = self._toposort()
        elif strategy == "level":
            def pri(k: str) -> tuple:
                n = self.node(k)
                return (n.level if n.level is not None else 99, n.order_hint, self._index[k])
            keys = self._toposort(pri)
        elif strategy == "breadth":
            layer = self.layers()
            keys = self._toposort(lambda k: (layer[k], self.node(k).order_hint, self._index[k]))
        elif strategy == "depth":
            keys = self._depth_first()
        else:
            raise CurriculumError(f"unknown strategy {strategy!r}; try default, "
                                  f"level, breadth or depth")
        return tuple(self.node(k) for k in keys)

    def _depth_first(self) -> list[str]:
        indeg = {n.key: 0 for n in self.nodes}
        for _a, b in self.edges:
            indeg[b] += 1
        pri = lambda k: (self.node(k).order_hint, self._index[k])
        ready = sorted((k for k, d in indeg.items() if d == 0), key=pri)
        out: list[str] = []
        while ready:
            k = ready.pop(0)                              # a stack of one thread
            out.append(k)
            freed = []
            for b in sorted(self.dependents(k), key=pri):
                indeg[b] -= 1
                if indeg[b] == 0:
                    freed.append(b)
            ready = freed + ready                         # follow the thread first
        if len(out) != len(self.nodes):                   # pragma: no cover
            raise CurriculumError(f"{self.id}: cycle")
        return out

    def linearizations(self, limit: int = 8) -> Iterator[tuple[Node, ...]]:
        """Several distinct valid orders, for showing that there is no one order.

        Enumerating them all is factorial in the width of the graph, so this
        yields up to ``limit`` distinct ones produced by rotating the ready-set
        tie-break. An edgeless curriculum has exactly one node set and many
        orders; a chain has exactly one order, and this yields just that.
        """
        seen: set[tuple[str, ...]] = set()
        strategies = ["default", "level", "breadth", "depth"]
        for s in strategies:
            order = self.linearize(s)
            sig = tuple(n.key for n in order)
            if sig not in seen:
                seen.add(sig)
                yield order
            if len(seen) >= limit:
                return
        for rot in range(1, limit):
            keys = self._toposort(
                lambda k: ((self._index[k] + rot * 7) % max(1, len(self.nodes)),
                           self.node(k).order_hint))
            sig = tuple(keys)
            if sig in seen:
                continue
            seen.add(sig)
            yield tuple(self.node(k) for k in keys)
            if len(seen) >= limit:
                return

    def layers(self) -> dict[str, int]:
        """Longest-path depth per node — the columns you would draw it in."""
        depth = {n.key: 0 for n in self.nodes}
        for k in self._toposort():
            for b in self.dependents(k):
                depth[b] = max(depth[b], depth[k] + 1)
        return depth

    # ---- teaching ----------------------------------------------------
    def frontier(self, mastered: Iterable[str] = ()) -> tuple[Node, ...]:
        """Nodes whose prerequisites are all mastered and which are not yet.

        What an adaptive runner asks for next, and the reason a graph beats a
        list: there is usually more than one right answer.
        """
        done = set(mastered)
        return tuple(n for n in self.nodes
                     if n.key not in done and all(p in done for p in self.prerequisites(n.key)))

    def train_eval_split(self, key: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """``(train, eval)`` lesson ids for a compositional-generalization test.

        Everything upstream of the node is fair to train on; the node itself is
        held out. If a system has learned the *structure* rather than the
        instances, the held-out composition should be reachable from its parts.
        Disjointness is by construction: a lesson upstream of the node cannot
        also be the node.
        """
        node = self.node(key)
        train = tuple(self.node(k).lesson for k in self._toposort()
                      if k in self.ancestors(key))
        return train, (node.lesson,)

    def splits(self) -> Iterator[tuple[str, tuple[str, ...], tuple[str, ...]]]:
        """Every compositional split this curriculum affords, node by node."""
        for n in self.nodes:
            train, ev = self.train_eval_split(n.key)
            if train:
                yield n.key, train, ev

    # ---- composition -------------------------------------------------
    def subgraph(self, keys: Iterable[str], *, id: str | None = None) -> "Curriculum":
        keep = {k for k in keys if k in self._index}
        return Curriculum(
            id=id or f"{self.id}_sub", title=self.title, description=self.description,
            nodes=tuple(n for n in self.nodes if n.key in keep),
            edges=tuple((a, b) for a, b in self.edges if a in keep and b in keep))

    def merge(self, other: "Curriculum", *, id: str, title: str = "",
              description: str = "") -> "Curriculum":
        """Union of two curricula. Nodes with the same key must agree."""
        nodes = list(self.nodes)
        have = {n.key: n for n in nodes}
        for n in other.nodes:
            if n.key in have:
                if have[n.key] != n:
                    raise CurriculumError(f"merge: {n.key!r} differs between "
                                          f"{self.id} and {other.id}")
                continue
            nodes.append(n)
        edges = list(dict.fromkeys([*self.edges, *other.edges]))
        return Curriculum(id=id, title=title or self.title, description=description,
                          nodes=tuple(nodes), edges=tuple(edges))

    def with_presentation(self, presentation: str, *, id: str, title: str = "",
                          suffix: str | None = None) -> "Curriculum":
        """The same graph, taught entirely through one surface.

        Used to build a cross-modal curriculum: take a text graph, take the same
        graph rasterized, and join them node to node. Those edges are transfer
        claims, and unlike an invented prerequisite they are testable.
        """
        tag = suffix if suffix is not None else f"@{presentation}"
        remap = {n.key: f"{n.key}{tag}" for n in self.nodes}
        return Curriculum(
            id=id, title=title or f"{self.title} ({presentation})",
            description=self.description,
            nodes=tuple(replace(n, key=remap[n.key], presentation=presentation)
                        for n in self.nodes),
            edges=tuple((remap[a], remap[b]) for a, b in self.edges))

    # ---- data --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "description": self.description,
                "nodes": [n.to_dict() for n in self.nodes],
                "edges": [list(e) for e in self.edges]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Curriculum":
        return cls(id=d["id"], title=d["title"], description=d.get("description", ""),
                   nodes=tuple(Node(**n) for n in d.get("nodes", ())),
                   edges=tuple((a, b) for a, b in d.get("edges", ())))

    def __repr__(self) -> str:
        return f"<Curriculum {self.id}: {len(self.nodes)} nodes, {len(self.edges)} edges>"


# --------------------------------------------------------------------------
# deriving edges
# --------------------------------------------------------------------------
def transitive_reduction(nodes: Sequence[str],
                         edges: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Drop edges implied by a longer path. Keeps a derived graph readable.

    Domination over axes produces a very dense relation — if A is easier than B
    on every axis and B than C, then A is easier than C, and the graph states it
    three times. Only the edges that are not implied are worth keeping.
    """
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].add(b)
    reach: dict[str, set[str]] = {}

    def reachable(n: str) -> set[str]:
        if n in reach:
            return reach[n]
        reach[n] = set()                              # guard against re-entry
        out: set[str] = set()
        for m in adj[n]:
            out.add(m)
            out |= reachable(m)
        reach[n] = out
        return out

    kept: list[tuple[str, str]] = []
    for a in nodes:
        for b in sorted(adj[a]):
            # b is redundant if some other child of a already reaches it
            if any(b in reachable(m) for m in adj[a] if m != b):
                continue
            kept.append((a, b))
    return tuple(sorted(kept))


def derive_edges(lessons: Mapping[str, Any], *, axes: Sequence[str],
                 require_shared_tag: bool = True) -> tuple[tuple[str, str], ...]:
    """Prerequisite edges implied by the declared difficulty axes.

    ``X -> Y`` when Y is at least as demanding as X on **every** axis and
    strictly more on at least one, and the two share a tag or a capability so
    that the claim is about related material rather than about two numbers that
    happened to compare.

    This is the only honest source of edges available. Hand-authoring a hundred
    and seventy prerequisites would be inventing structure to make a graph look
    complete; a derived relation is auditable, and wrong in ways someone can
    point at. The result is transitively reduced, because the dominance relation
    states every consequence of itself.
    """
    keys = sorted(lessons)

    def vec(lid: str) -> tuple[int, ...]:
        a = dict(lessons[lid].axes or {})
        return tuple(int(a.get(k, 0)) for k in axes)

    def related(x: str, y: str) -> bool:
        if not require_shared_tag:
            return True
        lx, ly = lessons[x], lessons[y]
        return bool((set(lx.tags) & set(ly.tags)) or
                    (set(lx.capabilities) & set(ly.capabilities)))

    raw: list[tuple[str, str]] = []
    for x in keys:
        vx = vec(x)
        for y in keys:
            if x == y:
                continue
            vy = vec(y)
            if all(a <= b for a, b in zip(vx, vy)) and any(a < b for a, b in zip(vx, vy)) \
                    and related(x, y):
                raw.append((x, y))
    return transitive_reduction(keys, raw)
