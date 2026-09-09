"""A persistent, versioned library of crystallized modules.

`Registry.register_module` already turns a crystallized program into one typed
candidate operator, but the registry lives and dies with the process, so no
learned artifact has ever outlived the script that produced it. This module is
the missing storage: a directory, a manifest, versions, and a `Registry` that
can be populated from it by a later, unrelated run.

Layout, all plain JSON so an artifact is readable without this package::

    <root>/manifest.json          index: every published version, in order
    <root>/modules/<digest>.json  the pruned frozen program, content-addressed
    <root>/fixtures/<digest>.json recorded conformance cases for that program

ARCHITECTURE section 4 is the contract this implements:

* **A definition is counted once.** Storage is content-addressed, so two logical
  names that crystallize to the same program share one file and one
  `module:<digest>` operator; `description_bits` then charges that definition
  once and every call site individually, transitively through nesting, exactly
  as it already does for an in-process registry.
* **Execution is charged per use.** Nothing here changes that: a loaded module
  resolves to the same `Operator` with the same `cost` as a freshly registered
  one.
* **Description cost is transitive.** An entry records the operator names of
  every module it calls, and loading one loads its dependencies first, so a
  caller's description bits include its callees' internals.
* **Relearning creates a new version and revalidates dependents.** Publishing a
  different program under an existing name allocates the next version and marks
  every entry that calls the superseded digest `stale`; `revalidate` re-executes
  the dependent's recorded fixture against the new head before clearing it.

**Source-revision policy.** `generation.source_fingerprint()` hashes all of
`tcn/` and `generators/`, so *any* edit to either invalidates every stored
module's provenance -- adding one generator invalidates artifacts from all the
others. A stored module therefore records the fingerprint it was built against
and the loader refuses to guess:

* ``policy="strict"`` (the default) raises `SourceRevisionMismatch` naming both
  fingerprints. A module never crosses a code revision silently.
* ``policy="revalidate"`` is the opt-in escape, and it is not a bypass: the
  module is loaded only after its recorded conformance fixture re-executes and
  agrees **exactly** under the current source. One disagreement raises
  `FixtureMismatch`. A success is written back to the manifest as an explicit
  `revalidated` stamp, so the crossing is recorded rather than forgotten.

There is deliberately no third policy that just ignores the mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import json

from .generation import source_fingerprint
from .graph import Program
from .operators import Registry
from .types import Value

FORMAT = "tcn.library/1"


class LibraryError(Exception):
    """Any refusal to load or publish; always names what disagreed."""


class SourceRevisionMismatch(LibraryError):
    pass


class FixtureMismatch(LibraryError):
    pass


class MissingDependency(LibraryError):
    pass


def module_dependencies(program):
    """Operator names of every frozen module this program calls directly.

    A module reaches a program two ways: as the node operator itself
    (``module:<digest>``) and as the ``module`` parameter of `map`/`filter`.
    `Program.description_bits` charges both, so both are dependencies.
    """
    found = []
    for node in program.nodes:
        for candidate in node.candidates:
            names = [candidate.operator.name,
                     dict(candidate.operator.parameters).get("module", "")]
            for name in names:
                if name.startswith("module:") and name not in found:
                    found.append(name)
    return tuple(found)


@dataclass(frozen=True)
class Entry:
    """One published version of one logical module name."""
    name: str
    version: int
    digest: str
    operator: str
    inputs: tuple
    outputs: tuple
    requires: tuple
    source: str
    nodes: int
    execution_cost: float
    description_bits: int
    provenance: dict = field(default_factory=dict)
    fixture_cases: int = 0
    stale: tuple = ()
    revalidated: tuple = ()

    def to_dict(self):
        return {"name": self.name, "version": self.version, "digest": self.digest,
                "operator": self.operator, "inputs": [list(x) for x in self.inputs],
                "outputs": [list(x) for x in self.outputs], "requires": list(self.requires),
                "source": self.source, "nodes": self.nodes,
                "execution_cost": self.execution_cost, "description_bits": self.description_bits,
                "provenance": self.provenance, "fixture_cases": self.fixture_cases,
                "stale": list(self.stale), "revalidated": list(self.revalidated)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["version"], d["digest"], d["operator"],
                   tuple(tuple(x) for x in d["inputs"]), tuple(tuple(x) for x in d["outputs"]),
                   tuple(d["requires"]), d["source"], d["nodes"], d["execution_cost"],
                   d["description_bits"], d.get("provenance", {}), d.get("fixture_cases", 0),
                   tuple(d.get("stale", ())), tuple(d.get("revalidated", ())))

    @property
    def reference(self):
        return f"{self.name}@{self.version}"


class Library:
    """A directory of crystallized modules, addressable by name and version."""

    def __init__(self, root):
        self.root = Path(root)
        self.entries = []
        manifest = self.root / "manifest.json"
        if manifest.exists():
            d = json.loads(manifest.read_text())
            if d.get("format") != FORMAT:
                raise LibraryError(f"library format mismatch: {d.get('format')!r}")
            self.entries = [Entry.from_dict(x) for x in d["entries"]]

    # ---------------------------------------------------------------- storage

    def _module_path(self, digest):
        return self.root / "modules" / f"{digest}.json"

    def _fixture_path(self, digest):
        return self.root / "fixtures" / f"{digest}.json"

    def _save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"format": FORMAT, "entries": [e.to_dict() for e in self.entries]}
        tmp = self.root / "manifest.tmp"
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.root / "manifest.json")

    # ----------------------------------------------------------------- lookup

    def names(self):
        out = []
        for e in self.entries:
            if e.name not in out:
                out.append(e.name)
        return tuple(out)

    def versions(self, name):
        return tuple(e for e in self.entries if e.name == name)

    def head(self, name):
        versions = self.versions(name)
        if not versions:
            raise KeyError(f"no module named {name!r} in {self.root}")
        return versions[-1]

    def entry(self, reference):
        """`name` for the head, or `name@version` for one exactly."""
        if "@" in reference:
            name, version = reference.rsplit("@", 1)
            for e in self.versions(name):
                if e.version == int(version):
                    return e
            raise KeyError(f"no version {version} of {name!r}")
        return self.head(reference)

    def by_digest(self, digest):
        for e in self.entries:
            if e.digest == digest:
                return e
        raise KeyError(f"no module with digest {digest}")

    def dependents(self, digest):
        """Entries whose stored program calls the module with this digest."""
        return tuple(e for e in self.entries if f"module:{digest}" in e.requires)

    # ---------------------------------------------------------------- publish

    def publish(self, name, program, registry, fixture=(), provenance=None):
        """Store a crystallized program under a logical name; return its `Entry`.

        Republishing the identical program is a no-op that returns the existing
        entry, so a rerun of the same stage does not manufacture versions.
        A *different* program under an existing name is a relearning: it takes
        the next version, and every entry that calls the superseded digest is
        marked stale until `revalidate` clears it.
        """
        staging = Registry()
        for dep in module_dependencies(program):
            try:
                entry = self.by_digest(dep.split(":", 1)[1])
            except KeyError:
                raise MissingDependency(
                    f"{name} calls {dep}, which is not in {self.root}; publish the module it "
                    "depends on first so the description cost stays transitive") from None
            self._load_module(entry, staging, "strict", set())
        operator = staging.register_module(program)
        stored = staging.modules[operator]
        digest = stored.digest
        if operator != "module:" + digest:
            raise LibraryError("registered operator name disagrees with the stored digest")

        cases = [{"inputs": {k: v.to_dict() for k, v in case.items()},
                  "outputs": {k: v.to_dict() for k, v in
                              stored.run(case, registry=staging)[0].items()}}
                 for case in fixture]

        existing = self.versions(name)
        if existing and existing[-1].digest == digest:
            return existing[-1]

        entry = Entry(name=name, version=(existing[-1].version + 1 if existing else 1),
                      digest=digest, operator=operator,
                      inputs=tuple((k, t.to_dict()) for k, t in stored.inputs),
                      outputs=tuple((k, stored.port_types()[v].to_dict()) for k, v in stored.outputs),
                      requires=module_dependencies(stored), source=source_fingerprint(),
                      nodes=len(stored.nodes), execution_cost=stored.execution_cost(staging),
                      description_bits=stored.description_bits(staging),
                      provenance=dict(provenance or {}), fixture_cases=len(cases))

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "modules").mkdir(exist_ok=True)
        (self.root / "fixtures").mkdir(exist_ok=True)
        self._module_path(digest).write_text(json.dumps(stored.to_dict(), indent=2, sort_keys=True))
        self._fixture_path(digest).write_text(json.dumps({"cases": cases}, indent=2, sort_keys=True))

        superseded = existing[-1].digest if existing else None
        self.entries.append(entry)
        if superseded is not None:
            # Relearning revalidates dependents: anything built on the previous
            # version is not known to hold against the new one until checked.
            for i, e in enumerate(self.entries):
                if f"module:{superseded}" in e.requires and entry.reference not in e.stale:
                    self.entries[i] = replace(e, stale=e.stale + (entry.reference,))
        self._save()
        return entry

    # ------------------------------------------------------------------- load

    def _load_module(self, entry, registry, policy, seen):
        if entry.operator in registry.modules:
            return entry.operator
        if entry.digest in seen:
            raise LibraryError(f"module dependency cycle at {entry.digest}")
        seen = seen | {entry.digest}
        for dep in entry.requires:
            self._load_module(self.by_digest(dep.split(":", 1)[1]), registry, policy, seen)
        spec = json.loads(self._module_path(entry.digest).read_text())
        program = Program.from_dict(spec, registry)
        if program.digest != entry.digest:
            raise LibraryError(f"stored program digest {program.digest} != manifest {entry.digest}")
        if not program.is_frozen:
            raise LibraryError(f"{entry.reference} is not crystallized")
        name = registry.register_module(program)
        if name != entry.operator:
            raise LibraryError(f"registered {name} for manifest operator {entry.operator}")
        return name

    def _check_source(self, entry, policy, registry):
        current = source_fingerprint()
        if entry.source == current:
            return "current"
        if policy == "strict":
            raise SourceRevisionMismatch(
                f"{entry.reference} was built against source {entry.source[:12]} and this tree is "
                f"{current[:12]}; load with policy='revalidate' to re-execute its recorded fixture, "
                "or relearn it")
        if policy != "revalidate":
            raise LibraryError(f"unknown source policy {policy!r}")
        self._replay_fixture(entry, registry)
        if current not in entry.revalidated:
            i = self.entries.index(entry)
            self.entries[i] = replace(entry, revalidated=entry.revalidated + (current,))
            self._save()
        return "revalidated"

    def _replay_fixture(self, entry, registry):
        path = self._fixture_path(entry.digest)
        cases = json.loads(path.read_text())["cases"] if path.exists() else []
        if not cases:
            raise FixtureMismatch(
                f"{entry.reference} has no recorded conformance fixture, so its behaviour under a "
                "different source revision cannot be checked; relearn it")
        module = registry.modules[entry.operator]
        for case in cases:
            inputs = {k: Value.from_dict(v) for k, v in case["inputs"].items()}
            got, _ = module.run(inputs, registry=registry)
            want = {k: Value.from_dict(v) for k, v in case["outputs"].items()}
            if {k: v.to_dict() for k, v in got.items()} != {k: v.to_dict() for k, v in want.items()}:
                raise FixtureMismatch(f"{entry.reference} no longer reproduces its recorded fixture")
        return len(cases)

    def load(self, references, registry=None, policy="strict"):
        """Populate a `Registry` from the library; return it and the alias map.

        `references` is a sequence of `name` / `name@version` strings, or a
        mapping of alias -> reference. The returned mapping takes an alias to
        the `module:<digest>` operator name a scaffold binds as a candidate.
        """
        registry = registry or Registry()
        items = (references.items() if isinstance(references, dict)
                 else [(r, r) for r in references])
        aliases = {}
        for alias, reference in items:
            entry = self.entry(reference)
            self._load_module(entry, registry, policy, set())
            self._check_source(entry, policy, registry)
            if entry.stale:
                raise LibraryError(
                    f"{entry.reference} is stale against {', '.join(entry.stale)}; "
                    "revalidate it before it is used as a candidate operator")
            aliases[alias] = entry.operator
        return registry, aliases

    # ------------------------------------------------------------- validation

    def revalidate(self, reference, policy="revalidate"):
        """Clear a stale mark by re-executing the dependent's own fixture."""
        entry = self.entry(reference)
        registry = Registry()
        self._load_module(entry, registry, policy, set())
        cases = self._replay_fixture(entry, registry)
        i = self.entries.index(entry)
        self.entries[i] = replace(entry, stale=())
        self._save()
        return {"reference": entry.reference, "cases": cases, "cleared": list(entry.stale)}

    def verify(self, reference=None):
        """Check every stored entry: digest, dependencies, source, fixture."""
        current = source_fingerprint()
        rows = []
        for entry in (self.entries if reference is None else [self.entry(reference)]):
            row = {"reference": entry.reference, "digest": entry.digest,
                   "source_current": entry.source == current, "stale": list(entry.stale)}
            try:
                registry = Registry()
                self._load_module(entry, registry, "revalidate", set())
                row["loads"] = True
                row["dependencies"] = list(entry.requires)
                row["fixture_cases"] = self._replay_fixture(entry, registry)
                row["fixture_reproduces"] = True
                module = registry.modules[entry.operator]
                row["execution_cost"] = module.execution_cost(registry)
                row["description_bits"] = module.description_bits(registry)
                row["ok"] = not entry.stale
            except Exception as e:                                 # reported, never swallowed
                row["loads"] = row.get("loads", False)
                row["fixture_reproduces"] = False
                row["ok"] = False
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
        return rows
