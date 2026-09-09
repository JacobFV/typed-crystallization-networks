"""The selection rule: which subprogram of the solved corpus is worth crystallizing.

The rule is stated over the generic typed program graph. It never looks at an
operator's meaning, at a task's semantics, or at anything domain-specific; it
reads node sets, edges, operator contracts and the repo's own
`Program.description_bits` accounting, and nothing else.

    R1  FRAGMENTS.  For each solved, pruned, frozen program P and each node r
        of P, every node set S with r in S is a *fragment rooted at r* when
        (a) every node of S has a directed path to r inside S,
        (b) |S| <= MAX_NODES,
        (c) S is *single-exit*: no node of S \\ {r} is read from outside S and
            none of them is a program output or a state update.
        (a) makes the fragment a sub-DAG with one root, (c) makes replacing it
        by one call semantics-preserving.

    R2  ABSTRACTION.  Every port a node of S reads that is not itself in S
        becomes a hole. Holes are one per *distinct* external port, so sharing
        inside the fragment is preserved (a port read twice stays one
        argument). The fragment becomes a `Program`: inputs x0..x{k-1} in order
        of first reference under the topological node order, nodes n0..n{m-1}
        in that same order, output the root. Its canonical identity is
        `Program.digest`, so two occurrences are "the same abstraction" exactly
        when the content-addressed store would give them one file.

    R3  REUSE.  Occurrences of one abstraction are counted per program after
        taking a maximal node-disjoint subset (greedy in node order), because
        only disjoint occurrences can be rewritten together. An abstraction is
        eligible only if it occurs in at least MIN_TASKS distinct earlier tasks
        -- reuse across tasks, not repetition inside one.

    R4  SCORE.  The score is the corpus description-length saving in the repo's
        own units, the definition charged once and every call site
        individually:

            saving(F) = sum_i bits(P_i) - sum_i bits(rewrite_i(F)) - bits(F)

        where `bits` is `Program.description_bits()` with no registry, so the
        rewritten programs are charged for their call sites and F's body is
        charged exactly once, which is what `Library` storage does.

    R5  PROPOSAL.  Eligible abstractions are ranked by saving, ties broken by
        (tasks, occurrences, nodes, digest). The top one is proposed. Every
        rewrite is executed against its task's examples before it is scored; a
        rewrite that changes any output disqualifies the abstraction.

MAX_NODES, MAX_HOLES and MIN_TASKS are the rule's only parameters. MAX_HOLES
exists because candidate enumeration at a call site is O(ports ** arity), so an
abstraction the search cannot afford to offer is not a useful abstraction.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from tcn.types import BOOL
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate

MAX_NODES = 5
MAX_HOLES = 4
MIN_TASKS = 2


# --------------------------------------------------------------------- R1, R2

def _selected(node):
    return node.candidates[node.selected if node.selected is not None else 0]


def fragments(program, max_nodes=MAX_NODES, max_holes=MAX_HOLES):
    """Every legal fragment of a frozen program, as (root, nodes, canonical, holes)."""
    nodes = {n.name: n for n in program.nodes}
    order = [n.name for n in program.nodes]
    index = {k: i for i, k in enumerate(order)}
    srcs = {k: tuple(_selected(n).sources) for k, n in nodes.items()}
    readers = {k: set() for k in order}
    for k in order:
        for s in srcs[k]:
            if s in readers:
                readers[s].add(k)
    exits = {v for _, v in program.outputs} | {u for _, _, u in program.state}

    out = []
    for root in order:
        ancestors = set()
        stack = [root]
        while stack:
            k = stack.pop()
            for s in srcs[k]:
                if s in nodes and s not in ancestors:
                    ancestors.add(s)
                    stack.append(s)
        pool = sorted(ancestors, key=lambda k: index[k])
        for size in range(0, min(max_nodes, len(pool) + 1)):
            for extra in itertools.combinations(pool, size):
                S = set(extra) | {root}
                # (a) every node reaches the root inside S
                reach = {root}
                changed = True
                while changed:
                    changed = False
                    for k in S:
                        if k not in reach and any(t in reach for t in readers[k] if t in S):
                            reach.add(k)
                            changed = True
                if reach != S:
                    continue
                # (c) single exit
                if any(readers[k] - S or k in exits for k in S if k != root):
                    continue
                canon, holes = canonicalize(program, S, root)
                if canon is None or not 1 <= len(holes) <= max_holes:
                    continue
                out.append((root, frozenset(S), canon, holes))
    return out


def canonicalize(program, S, root):
    """The fragment as a standalone `Program`, plus the external ports in order."""
    nodes = {n.name: n for n in program.nodes}
    order = [n.name for n in program.nodes if n.name in S]
    holes = []
    for k in order:
        for s in _selected(nodes[k]).sources:
            if s not in S and s not in holes:
                holes.append(s)
    rename = {k: f"n{i}" for i, k in enumerate(order)}
    hole_name = {h: f"x{i}" for i, h in enumerate(holes)}
    depth = {hole_name[h]: 0 for h in holes}
    new_nodes = []
    for k in order:
        cand = _selected(nodes[k])
        sources = tuple(rename[s] if s in S else hole_name[s] for s in cand.sources)
        d = max(depth[s] for s in sources) + 1
        depth[rename[k]] = d
        new_nodes.append(Node(rename[k], nodes[k].output,
                              (Candidate(cand.operator, sources),), "core", d, 0))
    types = program.port_types()
    inputs = tuple((hole_name[h], types[h]) for h in holes)
    try:
        canon = Program(inputs, tuple(new_nodes), (("out", rename[root]),)).validate(Registry())
    except (TypeError, ValueError):
        return None, ()
    return canon, tuple(holes)


# ------------------------------------------------------------------------- R3

def disjoint_occurrences(occurrences):
    """A maximal node-disjoint subset, greedy in root order."""
    chosen, used = [], set()
    for occ in occurrences:
        if occ["nodes"] & used:
            continue
        chosen.append(occ)
        used |= occ["nodes"]
    return chosen


# ------------------------------------------------------------------- rewriting

def rewrite(program, occurrences, operator_name, registry):
    """Replace each disjoint occurrence with one module call; return the program."""
    drop, calls = set(), {}
    for occ in occurrences:
        drop |= (occ["nodes"] - {occ["root"]})
        calls[occ["root"]] = occ["holes"]
    types = program.port_types()
    new_nodes = []
    for n in program.nodes:
        if n.name in drop:
            continue
        if n.name in calls:
            args = calls[n.name]
            op = registry.resolve(operator_name, tuple(types[a] for a in args))
            new_nodes.append(Node(n.name, n.output, (Candidate(op, tuple(args)),),
                                  n.region, n.depth, 0))
        else:
            new_nodes.append(n)
    return Program(program.inputs, tuple(new_nodes), program.outputs, program.constants,
                   program.state, program.input_depths, program.version + 1).validate(registry)


# ------------------------------------------------------------------------ rule

@dataclass
class Proposal:
    digest: str
    canonical: Program
    nodes: int
    holes: int
    tasks: tuple
    occurrences: int
    saving_bits: int
    definition_bits: int
    baseline_bits: int
    rewritten_bits: int
    execution_cost: float
    sites: tuple = ()
    rank: int = 0

    def row(self):
        return {"digest": self.digest, "nodes": self.nodes, "holes": self.holes,
                "tasks": list(self.tasks), "occurrences": self.occurrences,
                "saving_bits": self.saving_bits, "definition_bits": self.definition_bits,
                "baseline_bits": self.baseline_bits, "rewritten_bits": self.rewritten_bits,
                "execution_cost": self.execution_cost, "sites": list(self.sites),
                "rank": self.rank}


def propose(corpus, examples_by_task, max_nodes=MAX_NODES, max_holes=MAX_HOLES,
            min_tasks=MIN_TASKS, verify=True):
    """Rank every eligible abstraction over a corpus of solved programs.

    `corpus` is a mapping task name -> frozen pruned `Program`.
    `examples_by_task` supplies the rows each rewrite is verified against.
    """
    by_digest = {}
    for task, program in corpus.items():
        for root, S, canon, holes in fragments(program, max_nodes, max_holes):
            slot = by_digest.setdefault(canon.digest, {"canonical": canon, "per_task": {}})
            slot["per_task"].setdefault(task, []).append(
                {"root": root, "nodes": set(S), "holes": holes})

    baseline = {t: p.description_bits() for t, p in corpus.items()}
    total_baseline = sum(baseline.values())

    proposals = []
    for digest, slot in by_digest.items():
        canon = slot["canonical"]
        tasks = tuple(sorted(t for t in slot["per_task"]))
        if len(tasks) < min_tasks:
            continue
        registry = Registry()
        name = registry.register_module(canon)
        if name != "module:" + canon.digest:
            continue
        rewritten_total, occurrences, sites, ok = 0, 0, [], True
        for task, program in corpus.items():
            occs = slot["per_task"].get(task)
            if not occs:
                rewritten_total += baseline[task]
                continue
            chosen = disjoint_occurrences(sorted(occs, key=lambda o: o["root"]))
            try:
                rw = rewrite(program, chosen, name, registry)
            except (TypeError, ValueError, KeyError):
                ok = False
                break
            if verify and not _agrees(rw, program, examples_by_task[task], registry):
                ok = False
                break
            rewritten_total += rw.description_bits()
            occurrences += len(chosen)
            sites += [f"{task}:{o['root']}" for o in chosen]
        if not ok or occurrences < min_tasks:
            continue
        definition = canon.description_bits()
        proposals.append(Proposal(
            digest=digest, canonical=canon, nodes=len(canon.nodes),
            holes=len(canon.inputs), tasks=tasks, occurrences=occurrences,
            saving_bits=total_baseline - rewritten_total - definition,
            definition_bits=definition, baseline_bits=total_baseline,
            rewritten_bits=rewritten_total,
            execution_cost=canon.execution_cost(registry), sites=tuple(sites)))

    proposals.sort(key=lambda p: (-p.saving_bits, -len(p.tasks), -p.occurrences,
                                  -p.nodes, p.digest))
    for i, p in enumerate(proposals):
        p.rank = i + 1
    return proposals


def _agrees(rewritten, original, examples, registry):
    for ex in examples:
        try:
            a, _, _ = rewritten.execute(ex["inputs"], registry=registry)
            b, _, _ = original.execute(ex["inputs"], registry=Registry())
        except (ValueError, TypeError, KeyError, IndexError, OverflowError):
            return False
        if {k: v.to_dict() for k, v in a.items()} != {k: v.to_dict() for k, v in b.items()}:
            return False
    return True


def truth_table(program, registry=None):
    """The full input/output table of a Boolean fragment, for reporting only."""
    from tcn.types import Value
    r = registry or Registry()
    arity = len(program.inputs)
    rows = []
    for bits in itertools.product((False, True), repeat=arity):
        out, _ = program.run({k: Value.of(BOOL, v) for (k, _), v in zip(program.inputs, bits)},
                             registry=r)
        rows.append((bits, bool(next(iter(out.values())).decoded)))
    return tuple(rows)
