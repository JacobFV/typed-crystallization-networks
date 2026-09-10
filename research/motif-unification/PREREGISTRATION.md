# Motif unification — pre-registration

Branch `worktree-agent-a4a82f2a201419e0d`, working directory
`research/motif-unification/`. Committed **before any arm is run**. Amendments
are appended at the bottom, each naming the number it replaces.

`tcn/` and `generators/` are **not to be modified**. `research/refinement-bounds/`
is another agent's directory and is not touched.

---

## 0. What is being tested, and what was already known when this was written

FINDINGS §58 (`research/cross-domain/RESULTS.md`) enumerated 1,928 fragments and
217 canonical classes over the eight frozen artifacts of the visual, language and
computer tracks, and found **2 cross-domain semantic classes, 0 of them
non-trivial**. Its stated cause is the type system: only 2 of 131 type
signatures cross a domain boundary, so the result is bounded independently of
search. Its named example is the motif

```
eq(index(buffer, add(base, offset)), literal)
```

which §58's prose says "genuinely recurs in all three artifacts" and is three
semantic classes because the buffers are declared `(128×u8[byte])`,
`(3072×u8[byte])` and `(4096×u8[byte])`.

FINDINGS §53 measured that width-polymorphism is reachable as *schema plus
instantiation* with no type check relaxed, and that **a `unique` certificate at
one width is not evidence** (arm F). FINDINGS §55 built a class format
(`research/class-identity/classes.py`) enforcing a two-width rule (R1), vector
agreement (R2) and an instantiation digest check (R3), and rejected §53's unsound
schema twice.

**The question.** Can §55's class format unify this motif across the three real
domains, where type-exact identity splits it into three?

**Prior observation disclosed.** Before writing this document I read §58's
committed raw output `research/cross-domain/out/bounds_5.json` and printed the
node lists of `V_same`, `L_stage_a` and `C_agent` by running
`research/cross-domain/artifacts.py`. Nothing was searched, fitted or scored.
That reading is what sets the F1 threshold below, and it is recorded here rather
than presented later as a result of an arm.

---

## 1. Definitions fixed in advance

* **Artifact fragment.** The canonical single-exit sub-DAG produced by
  `research/cross-domain/frag.py` (gated bit-identical to
  `research/earned-abstraction/mine.py` on 862 checks by §58). Node names `n0…`,
  hole names `x0…`, exactly as `frag.canonicalize` assigns them.
* **Bit-identical.** Equality of `tcn.graph.Program.digest` between an
  instantiated schema and the artifact fragment. Nothing weaker (not "same
  operators", not "same behaviour on a sample") counts as bit-identical.
* **Schema.** A Python function `f(width, address_type, …) -> (Program, Registry)`
  building a scaffold with free nodes, in the sense of
  `research/depth-encoding/scaffold.py`. A **class** is that schema reference plus
  a selection vector plus its certified widths, in the sense of
  `research/class-identity/classes.py` (imported, not copied, not modified).
* **Certificate.** `complete` when the scaffold space was exhausted, `unique`
  when exhausted with exactly one conforming program, `none` when a stored class
  was instantiated instead of enumerating. Reported beside every synthesis count,
  with `space_size` and `evaluated`.
* **Relaxing a type check.** Any of: constructing a `Type` that differs from the
  artifact's declared type at the corresponding port; passing `registry=None` or
  skipping `Program.validate`; catching and ignoring a `TypeError` from
  `Registry.resolve`; widening or dropping `bounds`; editing `tcn/`.

---

## 2. Arms

### Step 1 — the motif and its instances (confirmation, independent)

**Q1.** Locate every fragment of the eight artifacts whose *operator shape*
(operator names and edges, types erased — §58's relation T) is
`eq(index(x, y), z)` (2 nodes) or `eq(index(x, add(y, z)), w)` (3 nodes). Report
artifact, node names, canonical digest, declared hole and output types.

**Q2.** Are they the same computation? Three checks, all exact:

* **C1** identical operator shape and edge structure;
* **C2** distinct `Program.digest` (three D-classes) and distinct
  `(input types, output type)` signature (three S-classes), re-derived here and
  not read from §58's JSON;
* **C3** a behavioural characterisation of each instance, exhaustive over the
  address carrier restricted to the buffer's own index range: for every
  `a ∈ [0, N)` the fragment returns `buffer[a] == other`, and its value is
  unchanged when any single cell `b ≠ a` is perturbed. The quantifier is stated
  in RESULTS.md exactly as run; a sampled agreement is never called a match.

**Pre-registered falsification F1.** If the 3-node motif is **absent** from any
one of the three domains, or if the third argument of `eq` plays a materially
different role in one domain (a literal in one, a second buffer read in another),
then §58's sentence "genuinely recurs in all three artifacts" is an overstatement
of §58's own §6.4 table, and RESULTS.md must say so in the headline, with the
correction, before any unification claim is made. **Check this first and
hardest.**

### Step 2 — one schema, three instantiations

**Q3.** Express the motif as one width-parametric schema and instantiate it at
`(N, address type)` = `(128, u32≤128)`, `(3072, u16)`, `(4096, u32≤4096)`.

**Gate G1 (bit-identity), pre-registered as pass/fail.** For each instantiation,
`schema(N, A).harden(vector).pruned().digest` must equal the artifact fragment's
canonical digest. A gate that fails is reported as a failure, never repaired by
loosening the comparison.

**Gate G2 (no relaxation).** `git diff --stat` over `tcn/` and `generators/` must
be empty; every instantiated program must pass `Program.validate(registry)` with
the artifact's declared types; every declared `bounds` and `role` must be
reproduced exactly. Recorded as a checklist in RESULTS.md.

**Pre-registered falsification F2.** A schema that unifies the instances
nominally but cannot be instantiated bit-identically at every width it claims →
the unification is nominal, not semantic, and is reported as such.

**Pre-registered falsification F5 (forbidden approach).** If bit-identity is
reachable only by relaxing a type check or erasing width per §1, the approach is
**abandoned** and reported as a failed approach. The owner's standing constraint
is that this is not to be solved by weakening the type system.

### Step 3 — soundness (§55 R1–R3, §53 arm F)

**Q4.** Certify the class at two or more widths by real enumeration over the
scaffold space at each width, scored on episodes drawn from that domain's own
generator, and admit it to a `ClassStore`. Report `space_size`, `evaluated`,
`exhausted`, `conforming`, `certificate`, `mean_return`, best constant and
uniform-random baselines at every width.

**Q5, arm F.** Construct **deliberately wrong** schemas that agree at one width:

* **F-a** a schema that hard-codes the buffer width of the first certified width;
* **F-b** a schema whose candidate ordering is width-dependent, so the *same*
  stored selection vector means a different operator at another width — §53 arm
  F's exact shape.

**Pre-registered pass condition.** Each wrong schema must earn the same
certificate as the right one at its agreeing width, and the format must refuse it
— `ClassStore.admit` raising `ClassError` naming R1, R2 or R3, or `instantiate`
raising on R3. If the format admits a wrong schema, that is a defect of the
format and is reported as one.

### Step 4 — transfer to a domain the schema was not derived from

**Q6.** Derive the class from **language (128) + visual (3072)** only. The
**computer** domain (4096) is held out and contributes nothing to the schema, the
vector or the certification.

Held-out task `T_C`, fixed here before any arm is run: from the real computer
track's raw terminal observation and the previous action, decide **whether the
agent should write** — §23's `visible ∧ ¬wrote`. Episodes come from
`research/computer-capability/task.py`'s own host, on that track's own
`TRAIN_DOCUMENTS` / `TEST_DOCUMENTS` split; the held-out documents use names and
digits never seen in training. Supervision is the track's own `reference` action
index. Only `record.actor_view().observations` reaches a program input.

The scaffold fixes the observation head (`project` the buffer, `project` the
previous-action field, `gt` against the threshold constant) and leaves **two**
free nodes: a **detector** over `(buffer, address, literal)` and a **combiner**
over two Booleans (16 `truth_j`). The detector's node budget is **one node**.
That budget is the abstraction pressure and is stated as such: the measurement is
whether a module earned in other domains makes a one-node detector expressible.

Arms, §52/§57 structure:

| arm | content |
|---|---|
| **A1** no library — core operators only, one detector node |
| **A1b** no library, detector budget raised to two nodes — the difficulty control |
| **A2** the unified class, instantiated at 4096 and registered as a module |
| **A3** a **wrong** module of comparable size (same node count, same arity, same signature) |
| **A4** a hand-authored equivalent module written directly for the computer domain |
| **B0** best constant | **B1** uniform random over the space | **B2** whole-space mean |

Every arm reports `space_size`, `evaluated`, `exhausted`, `conforming`,
`certificate`, train and held-out mean return. `description_bits` is **not** used
to rank anything (§41, §52, §54, §57). No pooling is performed, so §57's
inversion cannot arise.

**Pre-registered falsification F3.** If A2's held-out score is not strictly
greater than **max(A1, B0)**, the unified abstraction helps no cross-domain task.
That is the §57 outcome — identity without transfer — and is a first-class
result, reported in the headline.

**Pre-registered falsification F4 (bookkeeping).** If A1 already solves `T_C` at
the same held-out score, or if A2's saving equals exactly the space size it
removes from enumeration (§55's F-b), then A2 is bookkeeping rather than
abstraction, and RESULTS.md says so. **Achieved difficulty** — the fraction of
A1's space that conforms — is reported for every arm.

**Pre-registered falsification F6 (wrong-module control dead).** If A3 scores as
well as A2, the arm structure is not discriminating and no claim is made from it.

---

## 3. Reporting discipline

* Enumeration certificates beside every synthesis number; constant and random
  baselines beside every accuracy.
* "No solution exists in this family" is distinguished from "search failed" by
  `exhausted`.
* Surprising headlines are re-derived from raw JSON by a separate verifier script
  that does not import the analysis modules.
* Negative results are preserved. A negative here is a first-class deliverable.
* The language artifacts are static programs; no generator is re-run for them. If
  a language stream were regenerated it would be pinned to `hardening='none'`
  (§39, §45) and the stream named.
* Repo constraints: repo `.venv`; the test tally is reported with the known
  environmental `node_modules`/`tsx` failures identified by removing this
  directory and re-running; the shipped fixture must reproduce
  0.248836 → 0.002231 at 4/4 frozen. Large artifacts stay out of git.

---

## Amendments

*(none at time of commit)*
