"""The typed structural-edit enumerator: generic over `tcn.graph.Program`.

An **edit** is a deterministic function from one validated `Program` to another.
Five families, all typed -- every produced program is put through
`Program.validate(registry)` and discarded if it does not type-check, so no edit
can widen the type system:

  SUBST(site, op)      replace the site's candidate tuple with every legal
                       wiring of `op` over the site's own source pool
  WIDEN(site, op)      union those candidates into the site's existing tuple
  REWIRE(site, port)   union in every wiring of the site's *existing* operator
                       names that uses `port`, a source it did not have
  ADD_NODE(site, op, t) insert a new node of type `t` computing `op` over the
                       pool strictly below `site`, then widen `site` to admit it
  ADD_PATH(op)         insert a new output-adjacent node combining the current
                       output node with another value in scope, and re-point the
                       program's output at it

The pool of a site is derived from the program (inputs, constants and nodes of
strictly smaller depth), never hand-listed, and the operator names come from
`tcn.operators` name sets, never from a curated per-domain list.  Nothing here
knows what any scaffold's defect is.
"""
from __future__ import annotations

import dataclasses
import itertools

from tcn.graph import Candidate, Node, Program, legal_candidates
from tcn.operators import BINARY, COMPARE, CONVERSIONS, LOGIC, STRUCTURAL, UNARY

# Every operator name the enumerator may propose, read off `tcn.operators`'s own
# name sets plus the extras `Registry.resolve` accepts but does not export.
EXTRA = ("identity", "not", "mux", "sum", "mean", "reduce_min", "reduce_max", "count")
OP_NAMES = tuple(sorted(set(BINARY) | set(UNARY) | set(COMPARE) | set(LOGIC) |
                        set(STRUCTURAL) | set(CONVERSIONS) | set(EXTRA)))

FAMILIES = ("SUBST", "WIDEN", "REWIRE", "ADD_NODE", "ADD_PATH")

# Declared bounds on how much one edit may enlarge the selection space, so that
# every edited scaffold stays exhaustible and no arm is compared on a space that
# could not be decided.  Truncation is deterministic, in `legal_candidates`
# order, identical for every arm, and the number of truncated edits is reported.
MAX_NEW = 48           # candidates one edit may add at an existing site
MAX_NEW_NODE = 12      # candidates the node an ADD_NODE inserts may carry
TRUNCATED = set()      # edit keys whose candidate list was cut by the bounds


@dataclasses.dataclass(frozen=True)
class Edit:
    family: str
    site: str
    operator: str | None
    port: str | None
    type_key: str | None
    key: str

    def to_dict(self):
        return {"family": self.family, "site": self.site, "operator": self.operator,
                "port": self.port, "type_key": self.type_key, "key": self.key}


def _port_types(program):
    """Every named value in the program and its type."""
    types = dict(program.inputs)
    for name, value in program.constants:
        types[name] = value.type
    for nd in program.nodes:
        types[nd.name] = nd.output
    return types


def _depths(program):
    d = {k: 0 for k, _ in program.inputs}
    d.update({k: 0 for k, _ in program.constants})
    for nd in program.nodes:
        d[nd.name] = nd.depth
    return d


def pool_below(program, depth):
    """Names usable as a source at `depth`: everything strictly shallower."""
    types = _port_types(program)
    depths = _depths(program)
    return {k: types[k] for k in types if depths[k] < depth}


def site_pool(program, site):
    """The source pool of one node, exactly as the depth scaffold allows."""
    nd = next(x for x in program.nodes if x.name == site)
    return pool_below(program, nd.depth)


def _replace_node(program, name, new_node):
    return dataclasses.replace(
        program,
        nodes=tuple(new_node if nd.name == name else nd for nd in program.nodes))


def _cands(registry, op, pool, output, limit=4096):
    try:
        return legal_candidates(registry, (op,), pool, output, arities=(1, 2, 3),
                                limit=limit)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return ()


def _cut(key, cands, limit):
    if len(cands) > limit:
        TRUNCATED.add(key)
        return tuple(cands[:limit])
    return tuple(cands)


def _validated(program, registry):
    try:
        return program.validate(registry)
    except Exception:                                            # noqa: BLE001
        return None


# ------------------------------------------------------------------- appliers

def apply_edit(program, registry, edit):
    """The edited program, or None if the edit does not type-check."""
    if edit.family in ("SUBST", "WIDEN"):
        nd = next((x for x in program.nodes if x.name == edit.site), None)
        if nd is None:
            return None
        new = _cands(registry, edit.operator, site_pool(program, edit.site), nd.output)
        if not new:
            return None
        new = _cut(edit.key, new, MAX_NEW)
        keep = () if edit.family == "SUBST" else nd.candidates
        cands = tuple(dict.fromkeys(keep + new))
        if cands == nd.candidates:
            return None
        return _validated(_replace_node(program, nd.name,
                                        dataclasses.replace(nd, candidates=cands)), registry)

    if edit.family == "REWIRE":
        nd = next((x for x in program.nodes if x.name == edit.site), None)
        if nd is None:
            return None
        pool = site_pool(program, edit.site)
        if edit.port not in pool:
            return None
        names = tuple(dict.fromkeys(c.operator.name for c in nd.candidates))
        add = []
        for op in names:
            for c in _cands(registry, op, pool, nd.output):
                if edit.port in c.sources:
                    add.append(c)
        if not add:
            return None
        cands = tuple(dict.fromkeys(nd.candidates + _cut(edit.key, tuple(add), MAX_NEW)))
        if cands == nd.candidates:
            return None
        return _validated(_replace_node(program, nd.name,
                                        dataclasses.replace(nd, candidates=cands)), registry)

    if edit.family == "ADD_NODE":
        nd = next((x for x in program.nodes if x.name == edit.site), None)
        if nd is None:
            return None
        pool = pool_below(program, nd.depth)
        newtype = TYPE_KEYS.get(edit.type_key)
        if newtype is None:
            return None
        new = _cands(registry, edit.operator, pool, newtype)
        if not new:
            return None
        new = _cut(edit.key, new, MAX_NEW_NODE)
        # The new node takes the site's own depth; the site and everything at or
        # below it shift down by one, so the depth scaffold still orders every
        # edge and no existing wiring changes meaning.
        fresh = f"ind_{edit.site}_{edit.operator}_{edit.type_key}"
        node = Node(fresh, newtype, tuple(new), nd.region, nd.depth)
        nodes = []
        for x in program.nodes:
            if x.name == nd.name:
                nodes.append(node)
            nodes.append(x if x.depth < nd.depth
                         else dataclasses.replace(x, depth=x.depth + 1))
        widened = dataclasses.replace(program, nodes=tuple(nodes))
        nd = next(x for x in widened.nodes if x.name == edit.site)
        pool2 = dict(site_pool(widened, nd.name))
        pool2[fresh] = newtype
        names = tuple(dict.fromkeys(c.operator.name for c in nd.candidates))
        add = []
        for op in names:
            for c in _cands(registry, op, pool2, nd.output):
                if fresh in c.sources:
                    add.append(c)
        if not add:
            return None
        cands = tuple(dict.fromkeys(nd.candidates + _cut(edit.key, tuple(add), MAX_NEW)))
        return _validated(_replace_node(widened, nd.name,
                                        dataclasses.replace(nd, candidates=cands)), registry)

    if edit.family == "ADD_PATH":
        out_port, out_node = program.outputs[0]
        nd = next(x for x in program.nodes if x.name == out_node)
        pool = dict(pool_below(program, nd.depth + 1))
        pool[nd.name] = nd.output
        new = _cands(registry, edit.operator, pool, nd.output)
        new = tuple(c for c in new if nd.name in c.sources)
        if not new:
            return None
        new = _cut(edit.key, new, MAX_NEW)
        fresh = f"path_{edit.operator}"
        node = Node(fresh, nd.output, new, nd.region, nd.depth + 1)
        edited = dataclasses.replace(program, nodes=program.nodes + (node,),
                                     outputs=((out_port, fresh),))
        return _validated(edited, registry)

    raise ValueError(f"unknown edit family {edit.family}")


# --------------------------------------------------------------- enumeration

TYPE_KEYS = {}


def register_types(program):
    """Name every type the program already carries, so ADD_NODE stays typed."""
    for name, t in _port_types(program).items():
        TYPE_KEYS.setdefault(_type_key(t), t)


def _type_key(t):
    return f"{t.kind}{t.bits}:{t.encoding.kind}:{t.role}:{len(t.items)}"


def enumerate_edits(program, registry, op_names=OP_NAMES):
    """Every typed edit of `program`, in a deterministic order.

    Order is by (family, site, operator/port/type), all sorted -- an order fixed
    by the enumerator, never by any knowledge of the repair.  The returned list
    contains only edits that produced a validated program.
    """
    register_types(program)
    out = []
    sites = [nd.name for nd in program.nodes]
    types_here = sorted({_type_key(nd.output) for nd in program.nodes} |
                        {_type_key(t) for _, t in program.inputs})

    def add(edit):
        p = apply_edit(program, registry, edit)
        if p is not None:
            out.append((edit, p))

    for family in ("SUBST", "WIDEN"):
        for site in sorted(sites):
            for op in op_names:
                add(Edit(family, site, op, None, None, f"{family}|{site}|{op}"))
    for site in sorted(sites):
        for port in sorted(site_pool(program, site)):
            add(Edit("REWIRE", site, None, port, None, f"REWIRE|{site}|{port}"))
    for site in sorted(sites):
        for op in op_names:
            for tk in types_here:
                add(Edit("ADD_NODE", site, op, None, tk,
                         f"ADD_NODE|{site}|{op}|{tk}"))
    out_node = program.outputs[0][1]
    for op in op_names:
        add(Edit("ADD_PATH", out_node, op, None, None, f"ADD_PATH|{op}"))
    return out


def edit_space_size(program, registry, op_names=OP_NAMES):
    return len(enumerate_edits(program, registry, op_names))


# --------------------------------------------------------------- defects

def drop_operator(program, registry, site, op_name):
    nd = next(x for x in program.nodes if x.name == site)
    cands = tuple(c for c in nd.candidates if c.operator.name != op_name)
    if not cands or cands == nd.candidates:
        return None
    return _validated(_replace_node(program, site,
                                    dataclasses.replace(nd, candidates=cands)), registry)


def drop_source(program, registry, site, port):
    nd = next(x for x in program.nodes if x.name == site)
    cands = tuple(c for c in nd.candidates if port not in c.sources)
    if not cands or cands == nd.candidates:
        return None
    return _validated(_replace_node(program, site,
                                    dataclasses.replace(nd, candidates=cands)), registry)


def keep_prefix(program, registry, site, k):
    nd = next(x for x in program.nodes if x.name == site)
    if len(nd.candidates) <= k or k < 1:
        return None
    return _validated(_replace_node(program, site,
                                    dataclasses.replace(nd, candidates=nd.candidates[:k])),
                      registry)


def delete_node(program, registry, site):
    """Remove a node and re-point its consumers at one of its own sources."""
    nd = next((x for x in program.nodes if x.name == site), None)
    if nd is None or site == program.outputs[0][1]:
        return None
    types = _port_types(program)
    subs = [s for c in nd.candidates for s in c.sources if types.get(s) == nd.output]
    if not subs:
        return None
    sub = sorted(set(subs))[0]
    nodes = []
    for x in program.nodes:
        if x.name == site:
            continue
        cands = tuple(Candidate(c.operator, tuple(sub if s == site else s for s in c.sources))
                      for c in x.candidates)
        nodes.append(dataclasses.replace(x, candidates=tuple(dict.fromkeys(cands))))
    return _validated(dataclasses.replace(program, nodes=tuple(nodes)), registry)


def is_inverse(defect, edit):
    """Whether an edit is the syntactic inverse of the defect that made the case.

    Reported as a corpus-validity disclosure: if most first repairs are inverses,
    the corpus is easy and every arm's number must be read that way.
    """
    kind, site, arg = defect
    if kind in ("drop_operator",) and edit.family in ("SUBST", "WIDEN"):
        return edit.site == site and edit.operator == arg
    if kind == "drop_source" and edit.family == "REWIRE":
        return edit.site == site and edit.port == arg
    if kind == "keep_prefix" and edit.family in ("SUBST", "WIDEN"):
        return edit.site == site
    if kind == "delete_node" and edit.family in ("ADD_NODE", "ADD_PATH"):
        return True
    return False


def combinations(seq, k):
    return list(itertools.combinations(seq, k))
