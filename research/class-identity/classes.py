"""A class store beside a `tcn.library.Library` — the smallest thing the
measurement in `PREREGISTRATION.md` needs, and no more.

`tcn/library.py` is **not modified by this file or by any arm in this track.**
The store is a sidecar `classes.json` written next to a library root; whether its
fields belong in `tcn/library.py` is the verdict of `RESULTS.md`, not an
assumption of this module.

What the store holds, and what it deliberately cannot hold
----------------------------------------------------------
`digest` stays artifact identity. `class_id` is class identity. A class holds
several artifacts and a preferred member per cost axis, exactly as
`research/algorithm-resynthesis/DESIGN.md` §7 proposes.

Where §7 does not survive contact with the code: **the class cannot hold the
schema.** `tcn.library` stores `Program.to_dict()`, and a §53 schema is not a
`Program` — it is a function from width to `Program`
(`research/depth-encoding/scaffold.py:interpreter_scaffold`). Nothing in `tcn/`
has a serializable representation of a width-parametric program. So a `schema`
record holds a *reference* — module, qualified name, the free-node shape, and the
`source_fingerprint` it was built against — plus the selection vector, the
certified widths and the per-width artifact digests. It is a **witness**, not a
self-contained artifact: reusing it requires the schema code to be present and
unchanged, which is precisely what `source_fingerprint` already governs.

Class identity is scheme-tagged
-------------------------------
§52 wants an extensional key, `(arity, truth table)`, computable only over a
small finite input domain. §53 wants an intensional key, schema plus selection
vector; the truth table of a depth-8 interpreter over `tuple[24]` is not
enumerable. Neither relation subsumes the other, so `class_id` carries its
scheme and **two ids of different schemes are never equal** (`same_class`).

Admission rules R1-R4 are `PREREGISTRATION.md` §2.2, enforced here rather than
by convention.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from tcn.generation import source_fingerprint

FORMAT = "tcn.classes/0"

MIN_CERTIFIED_WIDTHS = 2                       # R1


class ClassError(Exception):
    """Any refusal to admit or instantiate a class; always names the rule."""


# --------------------------------------------------------------- identity keys

def schema_class_id(fingerprint, qualified_name, selections):
    """Intensional key: what §53 needs. Schema plus the selection vector."""
    body = json.dumps({"schema": qualified_name,
                       "selections": dict(sorted(selections.items()))},
                      sort_keys=True).encode()
    return f"schema/{fingerprint[:12]}/{hashlib.sha256(body).hexdigest()[:12]}"


def extensional_class_id(arity, table):
    """Extensional key: what §52 pools by. `(arity, truth table)`."""
    bits = "".join("1" if v else "0" for v in table)
    packed = format(int(bits, 2), "x") if bits else "0"
    return f"tt/{arity}/{packed}"


def same_class(a, b):
    """Equality that refuses to compare across schemes (PREREGISTRATION §2.1)."""
    if a.split("/", 1)[0] != b.split("/", 1)[0]:
        return False
    return a == b


# ------------------------------------------------------------------- the record

@dataclass(frozen=True)
class Certification:
    """One width at which the class's vector was certified, and how."""
    width: int
    space_size: int
    evaluated: int
    exhausted: bool
    conforming: int
    certificate: str
    mean_return: float
    threshold: float
    artifact_digest: str

    def to_dict(self):
        return dict(width=self.width, space_size=self.space_size,
                    evaluated=self.evaluated, exhausted=self.exhausted,
                    conforming=self.conforming, certificate=self.certificate,
                    mean_return=self.mean_return, threshold=self.threshold,
                    artifact_digest=self.artifact_digest)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in
                      ("width", "space_size", "evaluated", "exhausted", "conforming",
                       "certificate", "mean_return", "threshold", "artifact_digest")})


@dataclass(frozen=True)
class ClassRecord:
    class_id: str
    kind: str                                   # "schema" | "extensional"
    schema: dict = field(default_factory=dict)
    selections: dict = field(default_factory=dict)
    certified: tuple = ()
    members: tuple = ()
    preferred: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    @property
    def widths(self):
        return tuple(sorted({c.width for c in self.certified}))

    def to_dict(self):
        return {"class_id": self.class_id, "kind": self.kind, "schema": self.schema,
                "selections": self.selections,
                "certified": [c.to_dict() for c in self.certified],
                "members": list(self.members), "preferred": self.preferred,
                "provenance": self.provenance}

    @classmethod
    def from_dict(cls, d):
        return cls(d["class_id"], d["kind"], d.get("schema", {}), d.get("selections", {}),
                   tuple(Certification.from_dict(x) for x in d.get("certified", ())),
                   tuple(d.get("members", ())), d.get("preferred", {}),
                   d.get("provenance", {}))


# -------------------------------------------------------------------- the store

class ClassStore:
    """`classes.json` beside a library root. Never touches the manifest."""

    def __init__(self, root):
        self.root = Path(root)
        self.records = []
        path = self.root / "classes.json"
        if path.exists():
            d = json.loads(path.read_text())
            if d.get("format") != FORMAT:
                raise ClassError(f"class store format mismatch: {d.get('format')!r}")
            self.records = [ClassRecord.from_dict(x) for x in d["records"]]

    def _save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"format": FORMAT, "records": [r.to_dict() for r in self.records]}
        tmp = self.root / "classes.tmp"
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.root / "classes.json")

    def get(self, class_id):
        for r in self.records:
            if same_class(r.class_id, class_id):
                return r
        raise KeyError(f"no class {class_id!r} in {self.root}")

    def classes_of(self, digest):
        return tuple(r for r in self.records if digest in r.members)

    # ------------------------------------------------------------- admission

    def admit(self, record, threshold=None):
        """R1-R3. Raises `ClassError` naming the rule that refused."""
        if record.kind not in {"schema", "extensional"}:
            raise ClassError(f"unknown class kind {record.kind!r}")
        if record.kind == "schema":
            widths = record.widths
            # R1 -- the two-width rule. FINDINGS section 53 arm F: a deliberately
            # wrong schema earns the same `unique` certificate as the right one at
            # a single width. One width is not evidence.
            if len(widths) < MIN_CERTIFIED_WIDTHS:
                raise ClassError(
                    f"R1 two-width rule: {record.class_id} is certified at {len(widths)} "
                    f"width(s) {list(widths)}; a schema class needs at least "
                    f"{MIN_CERTIFIED_WIDTHS} distinct widths. A `unique` certificate at "
                    "one width is not evidence of a correct schema (FINDINGS section 53, arm F)")
            # R2 -- the *same* vector must conform at *every* certified width.
            for c in record.certified:
                limit = threshold if threshold is not None else c.threshold
                if c.conforming < 1 or c.mean_return < limit:
                    raise ClassError(
                        f"R2 vector agreement: {record.class_id} at width {c.width} has "
                        f"conforming={c.conforming} mean_return={c.mean_return} against "
                        f"threshold {limit}; the stored selection vector must conform at "
                        "every certified width")
            if not record.selections:
                raise ClassError(f"R2 vector agreement: {record.class_id} stores no selections")
            fp = record.schema.get("source_fingerprint")
            if fp != source_fingerprint():
                raise ClassError(
                    f"source revision: {record.class_id} was certified against "
                    f"{str(fp)[:12]} and this tree is {source_fingerprint()[:12]}; "
                    "a schema record is a witness, not a self-contained artifact")
        self.records = [r for r in self.records if r.class_id != record.class_id]
        self.records.append(record)
        self._save()
        return record

    # --------------------------------------------------------- instantiation

    def instantiate(self, class_id, width, builder, freeze):
        """Rebuild the schema at `width` and apply the stored vector.

        `builder(width) -> (names, program, registry)` and
        `freeze(program, full_selection, registry) -> Program` are supplied by the
        caller, because the store cannot execute a schema it can only reference.
        R3 is checked here for any width the record already certifies.
        """
        record = self.get(class_id)
        if record.kind != "schema":
            raise ClassError(f"{class_id} is not a schema class")
        names, program, registry = builder(width)
        free = {n.name: len(n.candidates) for n in program.nodes if len(n.candidates) > 1}
        declared = {k: v for k, v in record.schema.get("free_nodes", ())}
        if free != dict(declared):
            raise ClassError(
                f"schema shape at width {width} is {free}, but {class_id} declares "
                f"{dict(declared)}; the shape of the choice is not width-invariant here")
        for k, v in record.selections.items():
            if k not in free:
                raise ClassError(f"selection {k!r} is not a free node at width {width}")
            if not 0 <= v < free[k]:
                raise ClassError(f"selection {k}={v} outside {free[k]} candidates at width {width}")
        full = {n.name: 0 for n in program.nodes}
        full.update(record.selections)
        exact = freeze(program, full, registry)
        for c in record.certified:                                   # R3
            if c.width == width and c.artifact_digest != exact.digest:
                raise ClassError(
                    f"R3 instantiation check: {class_id} records digest "
                    f"{c.artifact_digest} at width {width}, rebuilt as {exact.digest}")
        return names, exact, registry, record
