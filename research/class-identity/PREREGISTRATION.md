# Pre-registration — class identity beside artifact identity

Committed **before any arm was run**. House standard since §44; §51 and §53 are
the models for how an amendment is recorded (numbered, dated, replacing a stated
criterion, committed before the arms it governs).

Nothing in this file is a result. Every number below is a threshold or a
protocol, not a measurement.

---

## 0. What is being asked, and why it is being asked now

Three tracks have independently arrived at the same missing representation.

* **§30** — a hardened module is width-specific; its input type *names* the
  observation, so `map` raises a signature mismatch at another width. What
  transfers is the **selections**.
* **§53** — measured end to end. One selection vector `{relation: 16,
  goal_relation: 6}` is certified `unique` at six depths (widths 3–24), scores
  4.00/4 held out, with no type check relaxed and no core change needed. The
  hardened artifact still refuses to cross: six distinct digests, and the
  invariant content is **8.09 bits**. Its closing line: *a schema and its
  certified widths have nowhere to live.*
* **§52** — semantic pooling by `(arity, truth table)` solves abstraction
  *identity* (144 conforming, equal to the hand-authored ceiling; both pooled
  wrong-module controls at 0). But the library has no place for a semantic
  class, only for artifacts by digest.

`research/algorithm-resynthesis/DESIGN.md` §7 proposes a shape. This track
**evaluates** that shape against `tcn/library.py` as it actually is, and then
asks whether any of it is justified by a measurement.

**The owner's standing rule governs the verdict:** do not claim an abstraction is
reusable until it helps a later held-out task. If nothing measurably improves,
the deliverable is "three tracks want this and nothing measurably improves yet",
and no machinery ships.

---

## 1. The two questions, fixed in advance

**Q1 — held-out width.** Does a stored class let a later task at an **unseen
width** reuse a learned result *without re-searching*? §53 exhausted 272 programs
per depth. If a stored schema reference plus a selection vector lets a new width
be instantiated without enumeration, that is a measurable saving. Report search
cost **with and without**, and the certificate **either way**.

**Q2 — semantic class.** Does a stored semantic class let §52's pooled
abstraction be registered and inherited, where §52 could only publish one
artifact per logical name?

---

## 2. The format under test (`ClassRecord`)

A class record is a JSON object stored in a sidecar `classes.json` beside a
`tcn.library.Library` root. **`tcn/` is not modified by any arm in this track.**
The sidecar is the smallest thing that permits the measurement; whether the
fields belong in `tcn/library.py` is a *conclusion* of this track, not an
assumption of it.

```
{
  "class_id":        scheme-tagged string, see §2.1
  "kind":            "schema" | "extensional"
  "schema":          {module, qualified_name, free_nodes: [[name, count], ...],
                      source_fingerprint}          # kind == "schema" only
  "selections":      {node: index, ...}            # kind == "schema" only
  "certified":       [{width, space_size, evaluated, exhausted, conforming,
                       certificate, mean_return, threshold, artifact_digest}, ...]
  "members":         [digest, ...]                 # artifacts of this class
  "preferred":       {cost_axis: digest, ...}
}
```

### 2.1 `class_id` is scheme-tagged and never compared across schemes

§52 wants an **extensional** key — `(arity, truth table)` — computable only when
the input domain is small and finite. §53 wants an **intensional** key — a schema
plus a selection vector; the truth table of a depth-8 interpreter over
`tuple[24]` is not enumerable. These are different equivalence relations and
neither subsumes the other. A single opaque `semantic_id` string that silently
mixes them would make two classes compare equal that are not.

Fixed in advance: `class_id` is `"<scheme>/<payload>"`, e.g. `"tt/3/e8"` and
`"schema/<source_fingerprint[:12]>/<selection-hash>"`. **Two class ids with
different schemes are never equal and are never merged.** Any arm that relies on
cross-scheme comparison is void.

### 2.2 Admission rules, fixed before arm F is run

**R1 — two-width rule.** A `kind == "schema"` record is admissible only if
`certified` names **≥ 2 distinct widths**. A record certified at one width is
**refused**, whatever its certificate. This is §53 arm F's lesson made
mechanical: a `unique` certificate at one width is not evidence of a correct
schema.

**R2 — vector agreement.** The **same** selection vector must be conforming at
**every** certified width. "Something conforms at each width" is not enough.

**R3 — instantiation check.** Each certified width records the digest the schema
freezes to under that vector. Publishing recomputes it; a disagreement refuses.

**R4 — reuse is verified, never assumed.** Instantiating a class at a new width
yields a *conformance check* at that width, never a uniqueness certificate.
The cost of that check is charged to the with-class arm.

---

## 3. Arms

### Q1 arms — the depth-encoding schema (§53)

The schema, the scaffold, the generator config, the horizon, the threshold and
the held-out episode indices are **§53's, unchanged**:
`research/depth-encoding/scaffold.py`, `logic` generator,
`{inputs: 4, nondegenerate: true, min_relevant_inputs: 2}`, horizon 4,
threshold 4.0, fit indices `range(16)` split `train`, held-out indices
`range(10000, 10064)` split `test`.

**Truly novel widths.** §53 measured depths `(1, 2, 3, 4, 6, 8)`. This track
declares **depths 5, 7 and 12** — widths 15, 21 and 36 — as the held-out set.
**No prior run in this repository has touched them.** Depths 3, 4, 6, 8 are
reported as a secondary confirmation against §53's recorded numbers, but the
primary claim rests on 5, 7, 12.

| arm | what it does |
|---|---|
| **A1 · construct** | Certify the class at depths **1 and 2 only** (§53's fit protocol, enumerating 272 programs at each on the *fit* episodes). Publish the record under R1–R3. Depths 5, 7, 12 are not enumerated in this arm. |
| **A2 · without class** | At each of depths 5, 7, 12: build the scaffold and **enumerate all 272 programs** on the 64 held-out episodes. Record programs evaluated, episodes, steps, seconds, conforming, certificate, best return. |
| **A3 · with class** | At each of depths 5, 7, 12: load the record, instantiate the schema at that width, apply the stored vector, freeze, and run the 64 held-out episodes **once**. Record the same cost fields and the resulting mean return. |
| **A4 · baselines** | At each of depths 5, 7, 12: always-true, always-false, uniform-random, best constant, whole-space mean and second-best-of-272 (the last two from A2). |
| **A5 · difficulty** | At each of depths 5, 7, 12: achieved difficulty over the held-out episodes — distinct circuits, distinct final tables, relevant-input histogram, majority fraction. Achieved, never requested. |

### Q1 arm F — the soundness test, run exactly as §53 defined it

`first_gate_scaffold` — §53's deliberately wrong schema, identical to the right
one but for one edge (`relation`'s lookup reads `gate_0` rather than
`gate_{d-1}`). At depth 1 the two nodes coincide, so a depth-1 fit *cannot* tell
the schemas apart.

| arm | what it does |
|---|---|
| **F1** | Certify the wrong schema at **depth 1 only** and attempt to publish. |
| **F2** | Certify the wrong schema at **depths 1 and 2** and attempt to publish. |
| **F3** | Bypass R1 (a deliberately unsound format), publish the depth-1-only wrong record, and instantiate it at depths 5, 7, 12. Report what a consumer would have got. |
| **F4** | Enumerate all 272 programs of the **wrong** schema at depths 5, 7, 12, so a collapse is distinguished from "no solution exists in this family at this width". |

### Q2 arms — §52's shipped library

`research/semantic-library/library` is copied read-only into this track's `out/`;
**§52's library is never written to.**

| arm | what it does |
|---|---|
| **B1 · partition** | Compute the extensional class id `(arity, truth table)` of every distinct digest in §52's manifest. Report entries, distinct digests, distinct classes, and every class with ≥ 2 members. |
| **B2 · pooled membership** | For §52's rank-1 majority class, recover the full pool of same-class circuits its miner computed and discarded (`mine_semantic.propose` elects `min(reps, key=(len(nodes), digest))`). Report the members' node counts, execution costs and description bits — i.e. whether "preferred member per cost axis" is a real choice or degenerate. |
| **B3 · inheritance** | Publish the hand-authored `MAJ3` (§52 arm3) into the copied library beside the mined `165bc290`, form the class, and check that the two distinct digests carry the identical `class_id` while `digest` separates them. Cite §52's recorded capability for each. |
| **B4 · additive-field compatibility** | Write a manifest carrying an extra `semantic_id` key and confirm the **unmodified** `tcn.library.Library` loads it, `verify()` passes on every entry, and `load` succeeds under `policy="strict"` and `policy="revalidate"`. This tests forward compatibility of the on-disk format with zero core diff. |

---

## 4. Metrics, fixed in advance

Primary cost metrics are **deterministic and machine-independent**: programs
evaluated, episodes rolled out, environment steps. Wall seconds are reported as
*indicative only* — this machine is shared with other agents' jobs.

Saving is reported as a **ratio of episodes**, and separately as a ratio of wall
seconds, with the certificate obtained in each arm stated beside it.

---

## 5. Pre-registered success criteria (all must hold for "the machinery is justified")

* **C1.** A2 and A3 agree on the answer at every held-out width: the program A3
  instantiates from the class is the same program A2's enumeration certifies as
  conforming (same digest), and scores ≥ 4.0 on the 64 held-out episodes.
* **C2.** A3's episode cost is **< 1/100** of A2's at every held-out width.
* **C3.** A3's return beats best constant, uniform random and whole-space mean at
  every held-out width.
* **C4.** F1 and F2 are both **refused** by the format, and the refusal names
  which rule fired.
* **C5.** F3 — the unsound single-width format — produces a **wrong** result at
  every held-out width (mean return < 4.0), and F4 shows 0 conforming in the
  wrong schema's own 272-program space at those widths, so the collapse is the
  schema's and not the width's.
* **C6.** B4 passes: the extra field does not disturb `strict`, `revalidate`,
  fixture replay or `source_fingerprint`.
* **C7.** The full test suite shows **no failure that is not present on the same
  worktree before any change**, and the shipped fixture reproduces
  0.248836 → 0.002231 at 4/4 frozen.

---

## 6. Pre-registered falsification — stated so it can fire

* **F-a.** If A3's episode cost is not measurably below A2's, or A3 does not
  reproduce A2's answer, **the machinery is premature**: report it and stop. No
  core change is proposed.
* **F-b.** If the saving exists **only because enumeration is cheap here**
  (272 programs), say so explicitly. The saving ratio equals the space size,
  which is a property of the *schema*, not of the class store; a store that
  saves 272× on a 272-program space has demonstrated a mechanism, not a
  capability at scale. This bounds the claim and is reported in the verdict, not
  in a footnote.
* **F-c.** If storing a class requires **relaxing any type check or erasing any
  width**, that is a **forbidden solution**. Report it as a failed approach and
  propose nothing.
* **F-d.** If the format admits a wrong schema that single-width certification
  accepted (C4 fails), the storage format is **unsound** and is not proposed.
* **F-e.** If Q2 shows only that the format *can hold* a class, with no measured
  improvement over §52's published numbers, the honest verdict is **"storage
  feasible, no measured saving"**. §52's failure is at the *ranking* layer, which
  is upstream of storage; class identity is not expected to fix it and no claim
  that it does will be made. Ranking is another track's subject and is not
  touched here.
* **F-f.** If B2 finds the pooled members cost-identical, "a declared preferred
  member per cost axis" is **degenerate on this evidence** and is reported as
  unsupported rather than as designed-in.

---

## 7. Constraints acknowledged

* **The type system is not weakened.** No relaxed type check, no width erasure.
  §53 showed polymorphism is reachable without either.
* **Existing library verification keeps working exactly.** `strict`,
  `revalidate`, fixture re-execution and `source_fingerprint` are untouched;
  `tcn/` receives **no diff** in any arm. Whether it should is the verdict.
* **Two or more widths, always.** No claim in this track rests on a single-width
  `unique` certificate.
* **Probes are supervision, never model inputs.** §53's scaffold is reused
  verbatim; `probes["target"]` is read only by the baseline and difficulty arms.

## 8. Known environmental baseline

Recorded before any change, on this worktree, on main's own code:
**324 passed, 13 failed** — the `generators/computer` and `test_panel_interface`
failures that need the gitignored `node_modules` symlink, which is absent here.
`.venv/bin/python -m tcn train --episodes 160` gives
`initial_prediction_loss 0.248835613951087`,
`final_prediction_loss 0.0022308224288281053`, `evaluation_mean_return 4.0`,
`frozen_evaluation_mean_return 4.0`. Any arm's test run is compared against this
baseline, not against 337.
