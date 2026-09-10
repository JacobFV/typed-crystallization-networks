# Motif unification — can §55's class format unify §58's cross-domain motif?

Branch `worktree-agent-a4a82f2a201419e0d`, working directory
`research/motif-unification/`. `tcn/` and `generators/` are **not modified**;
`git diff main -- tcn/ generators/` is empty and that emptiness is a computed
field of `out/step2.json`, not a sentence here. `research/refinement-bounds/` is
another agent's directory and is untouched.

`PREREGISTRATION.md` committed at `c761882`, **before any arm**. Amendments are
in §10 with the number each replaces.

Every number below is recomputed from `out/*.json` by
`verify_claims.py`, which imports none of the analysis modules: **82 checks, all
pass**. Reproduction commands are in §9.

---

## 1. The headline

**The motif is genuinely shared, one schema instantiates it bit-identically at
all three widths, the format rejects a wrong schema twice — and the unified
abstraction helps no task in the domain it was not derived from.**

Four findings, in the order they were tested:

1. **§58's illuminating example is weaker than its prose says, and its own §6.4
   table already said so.** The three-node motif
   `eq(index(buffer, add(base, offset)), literal)` occurs in **two** of the three
   real artifacts, not three. The computer artifact's three-node instance is
   `eq(index(x1, identity(x0)), x2)` — a **constant** address, arity 3, not a
   sum. What does recur in all three is the **two-node** motif
   `eq(index(buffer, address), other)`. And the compared value is a literal in
   language (`40`, `(`) and computer (`123`, `{`) but a **second buffer read** in
   visual. Pre-registered **F1 fires on both clauses.**
2. **One schema does instantiate bit-identically at every width where the
   fragment exists** — 5 of 5 gated instantiations, **zero differing keys** in
   `Program.to_dict()`, with no type check relaxed and no width erased.
3. **The format's soundness rule holds on this schema.** Two deliberately wrong
   schemas are *bit-identical to the right one at width 128* and earn the
   identical `unique` certificate there; both are refused — by **R1** at one
   width and by **R2** at two.
4. **Transfer fails, and the failure is precise.** On the one held-out computer
   task whose address cannot be a constant, the certified class scores **0
   conforming, certificate `complete`** while no-library at equal node count
   scores **1 conforming, `unique`, 1.00 held out**. Re-selecting the schema's
   vector on the new domain *does* solve it — and enumerates **exactly the same
   750-program space** as no-library, so it buys nothing at all. Pre-registered
   **F3 fires**; **F4 fires** on the other task.

**The one-line verdict.** What crosses a domain boundary here is the **schema** —
a function from width to a scaffold. What does not cross is the **certified
selection vector**, which is the entire content of a §55 class beyond the schema
reference. §58's blocker is real and this removes it; removing it buys no
capability.

---

## 2. Step 1 — the motif and its instances

`motif.py`, `run_step1.py` → `out/step1.json`. The eight artifacts are
`research/cross-domain/artifacts.py`, imported unmodified and reproducing every
recorded digest (including §45's Dyck witness `4d0ce927662af9fe54799de0`). The
fragment rule is `research/cross-domain/frag.py`, which §58 gated bit-identical
to `research/earned-abstraction/mine.py` on 862 checks. This track only
**filters** that fragment set by operator shape; it re-implements no rule.

### 2.1 Every occurrence, at `MAX_NODES = 5`

| shape | artifact | domain | root | canonical fragment | digest | declared holes |
|---|---|---|---|---|---|---|
| **M2** | `V_same` | visual | `cmp_r`, `cmp_g`, `cmp_b` | `eq(index(x0, x1), x2)` | `fa7725dfc8e7` | `(3072×u8[byte])`, `u16`, `u8[byte]` |
| **M2** | `L_stage_a` | language | `open` | `eq(index(x0, x1), x2)` | `c9522d87ebb5` | `(128×u8[byte])`, `u32≤128`, `u8[byte]` |
| **M2** | `C_agent` | computer | `brand` | `eq(index(x0, x1), x2)` | `a69fc2da0cb7` | `(4096×u8[byte])`, `u32≤4096`, `u8[byte]` |
| **M3** | `V_same` | visual | `cmp_g`, `cmp_b` | `eq(index(x2, add(x0, x1)), x3)` | `e38275a68420` | `u16`, `u16`, `(3072×u8[byte])`, `u8[byte]` |
| **M3** | `L_stage_a` | language | `open` | `eq(index(x2, add(x0, x1)), x3)` | `db59f9b09bb7` | `u32≤128`, `u32≤128`, `(128×u8[byte])`, `u8[byte]` |
| **M3ᵢ** | `C_agent` | computer | `brand` | `eq(index(x1, identity(x0)), x2)` | `9a62ee528c69` | `u32≤4096`, `(4096×u8[byte])`, `u8[byte]` |

**M2: 5 occurrences, 3 domains, 3 digests, 3 type signatures, widths 128 / 3,072
/ 4,096.
M3: 3 occurrences, 2 domains, 2 digests.
M3ᵢ: 1 occurrence, 1 domain.**

### 2.2 F1, checked first and hardest — it fires, twice

**Clause one: the three-node motif is not in all three artifacts.** §58's prose
says the motif `eq(index(buffer, add(base, offset)), literal)` "genuinely recurs
in all three artifacts". §58's own §6.4 table lists that three-node shape as
`language + visual` and the *two*-node shape `eq(index(x0,x1), x2)` as all three.
The raw programs support the table. `C_agent` computes
`ppos = identity(q0)` with `q0 = 0` — its address is a program constant, so its
canonical fragment has **three** holes where M3 has four, and **no selection of a
four-hole schema can produce a three-hole program**. The three buffer widths §58
quotes (128 / 3,072 / 4,096) are the widths of the *two*-node instances.

**Clause two: the third argument is not a literal in all three.** Measured
binding of the last hole in the artifact:

| artifact | binds to |
|---|---|
| `V_same` / `cmp_r`, `cmp_g`, `cmp_b` | **a node** — `index(obs, b)`, `index(obs, b_g)`, `index(obs, b_b)` |
| `L_stage_a` / `open` | constant `40` (`(`) |
| `C_agent` / `brand` | constant `123` (`{`) |

So in vision the motif compares **two computed buffer reads to each other**; in
language and computer it compares one read **to a literal**. As a *fragment*
those are the same function of four holes — canonicalisation makes a constant a
hole exactly as it makes a node one — but the claim "the byte at a computed
address equals a literal recurs in all three artifacts" is not what the artifacts
say.

**The correction, stated plainly.** §58's structural claim survives in the form
its own table gives it. Its prose overstates it. Everything §58 concludes from
the motif — that it is three semantic classes, that width typing is the blocker,
that raising the enumeration cap cannot help — is **unaffected**, because those
follow from the type signatures, which this track re-derives independently and
confirms.

### 2.3 Are they the same computation? Yes, and the check is exhaustive

`run_step1.characterise`. For **every** address `a` in `[0, N)` of **every**
instance, against a seeded buffer `R`:

1. `f(R, a, R[a])` is `True`;
2. `f(R, a, R[a]+1 mod 256)` is `False`;
3. `f(P_a, a, R[a])` is `False`, where `P_a` is `R` with cell `a` bumped;
4. `f(P_a, a', R[a'])` is `True` for `a' = a+1 mod N` — no other cell matters;
5. for M3 only, a *different* decomposition `(a−3, 3)` of the same sum agrees.

| instance | addresses checked | exhaustive over addresses | checks each | failures |
|---|---|---|---|---|
| `V_same` M2 / M3 | 3,072 | yes | 4 / 5 | **0** |
| `L_stage_a` M2 / M3 | 128 | yes | 4 / 5 | **0** |
| `C_agent` M2 / M3ᵢ | 4,096 | yes | 4 | **0** |

**Quantifier, stated exactly.** This is exhaustive over the address carrier
restricted to the buffer's index range and over the perturbation of every buffer
cell, at one seeded buffer per instance. It is **not** exhaustive over the
buffer's `256^N` inhabitants, which is why §58 could not compute an S identity
here at all. It establishes that each instance reads exactly `buffer[base+offset]`
and compares it for equality — which is what "the same computation" means for
this motif.

---

## 3. Step 2 — one schema, instantiated bit-identically

`schema.py`, `run_step2.py` → `out/step2.json`.

The schema takes **two** parameters, not one: `width` (the buffer's field count)
and `address` (the declared carrier of the index). They are independent — the
language and computer buffers refine `u32` to `(0, N)` for their own `N`, but the
visual buffer's addresses are a plain `u16` over 3,072 fields. That is exactly
what `research/cross-domain/RESULTS.md` §10 said would be needed: "a class
identity that abstracts over *product width and integer carrier together*".

### 3.1 Gate G1 — bit-identity

| schema | domain | width | address | free nodes | built digest | artifact digest | bit-identical |
|---|---|---|---|---|---|---|---|
| M2 | language | 128 | `u32≤128` | **{}** | `c9522d87ebb5` | `c9522d87ebb5` | **yes** |
| M2 | visual | 3,072 | `u16` | **{}** | `fa7725dfc8e7` | `fa7725dfc8e7` | **yes** |
| M2 | computer | 4,096 | `u32≤4096` | **{}** | `a69fc2da0cb7` | `a69fc2da0cb7` | **yes** |
| M3 | language | 128 | `u32≤128` | `{n0: 5}` | `db59f9b09bb7` | `db59f9b09bb7` | **yes** |
| M3 | visual | 3,072 | `u16` | `{n0: 5}` | `e38275a68420` | `e38275a68420` | **yes** |
| M3 | computer | 4,096 | `u32≤4096` | `{n0: 5}` | `538ff495bdc9` | *(no such fragment)* | — |

**5 of 5 gated instantiations bit-identical**, and the comparison is stronger
than a digest: the field-by-field diff of `Program.to_dict()` is **empty** in
every case. Every declared input type and the output type match the artifact's
own `Type` objects, compared object-to-object and never through a tag.

The last row is the honest negative and it is not a gate failure: the schema
*builds* a valid four-hole program at 4,096, but `C_agent` contains no four-hole
fragment for it to equal. §58's motif reaches two domains at three nodes.

### 3.2 Gate G2 — nothing relaxed, and it is computed rather than asserted

| check | value |
|---|---|
| `git diff main -- tcn/` | empty |
| `git diff main -- generators/` | empty |
| any `except` in `schema.py` | **false** — no `TypeError` is caught anywhere |
| declared `bounds` reproduced | `(0,128)`→`(0,128)`, `None`→`None`, `(0,4096)`→`(0,4096)` |
| every instantiation `Program.validate(registry)` | passes |
| fields normalised by `freeze` | **`version` only** — and the empty `to_dict` diff shows even that ends up equal, because `freeze` normalises it back to the fragment's own value |

`version` is a monotone provenance counter that `Program.harden` increments; it
is not part of the graph, and normalising it is the only normalisation applied.
**F5 does not fire: nothing was solved by weakening the type system.**

### 3.3 The measurement that decides M2's fate, and it is the type system's

`schema.compare_pool` reads `Type.numeric` rather than guessing. `tcn/types.py:27`
declares `byte` an **UNCOMMITTED** role, so `tcn/operators.py:67` admits **only
`eq`** on two `u8[byte]` values — `lt`, `le`, `gt`, `ge` all raise. And `index` is
the only operator that reads a product at a computed offset. Therefore:

> **M2 has zero free nodes at every width.** The one motif that spans all three
> real domains is too small to carry a selection vector.

That is not a design choice made here. It is measured, and §5.3 shows what it
costs.

---

## 4. Step 3 — certification and §53's arm F

`run_step3.py` → `out/step3.json`. The class format is
`research/class-identity/classes.py`, **imported unmodified**; every refusal below
is that file raising `ClassError` and naming its own rule. Episodes are each
domain's own generator — `research/language-capability`'s `context_free_language`
lesson at capacity 128, and `research/visual-ladder`'s FLAT screens. No generator
setting is changed and **no artifact is re-hardened, so no `hardening` choice is
exercised anywhere in this track**; the language artifacts are static programs
(§39/§45).

### 4.1 The class, certified at two widths

| domain | width | space | evaluated | exhausted | conforming | certificate | stored vector | best constant | uniform random over the space | held out |
|---|---|---|---|---|---|---|---|---|---|---|
| language | 128 | 5 | 5 | true | **1** | **`unique`** | **1.000** | 0.531 | 0.485 | **1.000** |
| visual | 3,072 | 5 | 5 | true | **1** | **`unique`** | **1.000** | 0.521 | 0.660 | **1.000** |

96 training and 96 held-out episodes per width, labels balanced by construction
so the best constant sits at ~0.5. Per-candidate accuracy, which is what makes
`conforming = 1` readable rather than inferred:

| candidate | language | visual |
|---|---|---|
| **0 `add`** | **1.000** | **1.000** |
| 1 `sub` | 0.146 | 0.646 |
| 2 `mul` | 0.094 | 0.448 |
| 3 `min` | 0.552 | 0.615 |
| 4 `max` | 0.635 | 0.594 |

`ClassStore.admit` **admits** the record: class `schema/f72db2cf6eca/a491efadcab4`,
kind `schema`, selections `{n0: 0}`, certified at widths **128 and 3,072**,
members `db59f9b09bb7` and `e38275a68420`, provenance `language/128, visual/3072`.

### 4.2 The refusals

| record | outcome | rule |
|---|---|---|
| M3, certified at 128 **and** 3,072 | **admitted** | — |
| M3, certified at 128 **only** | refused | **R1** two-width rule |
| **M2**, bit-identical at 128 / 3,072 / 4,096, empty vector | **refused** | **R2** vector agreement — "stores no selections" |

The M2 row is the sharpest thing the format says here. M2 is the *only* shape
that spans all three real domains; it instantiates bit-identically at all three
widths; and §55's format cannot hold it, because a two-node fragment carries no
choice and R2 presupposes one. **The format is built for schemas that can be
wrong about a vector, and the one motif that crosses all three domains cannot
be.** Its three digests are still three distinct artifacts, so nothing about
storage is solved by observing this.

### 4.3 Arm F — a wrong schema is invisible at one width, and both are refused

Two wrong schemas, both **bit-identical to the right one at width 128** and both
earning the identical `unique` certificate there:

| arm | 128 | 3,072 | refused at two widths | refused at one width |
|---|---|---|---|---|
| **F-a** width hard-coded | conforming **1**, `unique`, vector **1.000** | conforming **0**, `complete`, vector **0.000** | **R2** vector agreement | **R1** two-width rule |
| **F-b** width-dependent candidate order | conforming **1**, `unique`, vector **1.000** | conforming **1**, `unique`, vector **0.594** | **R2** vector agreement | **R1** two-width rule |

F-a reproduces §53 arm F exactly — the wrong schema collapses to **0 conforming**
at the other width. F-b is the subtler failure the two-width rule was written
for: the space still *contains* the right program at 3,072, and it is still
`unique` there, but the **stored index 0 now names `max` instead of `add`** and
scores 0.594. Both are refused, and each refusal names its rule. **§53's caution
is enforceable on this motif, not merely stated.**

---

## 5. Step 4 — does it help a domain it was not derived from?

The class is derived from **language (128) and visual (3,072)** only. The
**computer** domain contributes nothing to the schema, the vector or the
certification — `class_certified_widths` is `[128, 3072]`, verified. Two held-out
tasks in the computer domain, both on `research/computer-capability`'s own host
and its own document splits.

### 5.1 `T_C1` — "should the agent write?" (§23's own decision)

Target: the track's reference action is `write`. Inputs: the raw terminal
observation and the previous action. 15 training and 30 held-out examples over
the track's `TRAIN_DOCUMENTS` / `TEST_DOCUMENTS`; best constant **0.667** on both.

| arm | detector nodes | space | evaluated | exhausted | conforming | certificate | held out | P(random pick conforms) |
|---|---|---|---|---|---|---|---|---|
| **A1** no library | 1 | 1,680 | 1,680 | true | **0** | `complete` | — | 0 |
| **A1b** no library | 2 | 320 | 320 | true | 6 | `complete` | **1.000** | 0.0188 |
| **A1c** no library | 3 | 6,400 | 6,400 | true | 106 | `complete` | **1.000** | 0.0166 |
| **A2** the transferred class | 1 (module) | 1,280 | 1,280 | true | 14 | `complete` | **1.000** | 0.0109 |
| **A3** wrong module, comparable size | 1 (module) | 256 | 256 | true | **0** | `complete` | — | 0 |
| **A4** hand-authored equivalent | 1 (module) | 1,280 | 1,280 | true | 14 | `complete` | **1.000** | 0.0109 |

**A2 equals the hand-authored ceiling exactly** — same conforming count, same
held-out score — and the two modules are the **same graph**: the digests differ
only in node names and `version`, and the structural comparison is identical.
That is the positive half.

**But F4 fires.** A1b solves the task with **two** core nodes — fewer than the
module's own three-node body — at held-out 1.000. So the module's advantage over
no-library exists **only at a one-node budget**, and at equal node count it
vanishes. The reason is in the data: `T_C1`'s discriminating byte sits at a
**constant** address, so `index(data, p2)` suffices and the address computation
the module supplies is dead weight. **A2 beats A1 only because A1 was denied a
node.** That is bookkeeping, and it is why `T_C2` exists.

Controls that are live: A3, the *other* real motif (`V_same`'s "are the bytes at
two computed addresses equal", 3 nodes, instantiated at 4,096) reaches **0
conforming**. And §30's control — the class instantiated at the **language**
width and offered to the computer scaffold — cannot be constructed at all:
`TypeError: module:db59f9b09bb7…: operator signature mismatch`. A hardened
artifact still does not cross a width; only the schema does.

### 5.2 `T_C2` — the task whose address cannot be a constant

§23 states the causal structure of its own environment: *"`name` varies in
length, so no constant byte address finds the digit."* `T_C2` takes it at its
word. From the terminal showing the file's own content (`counter = 5`), decide
whether the digit is `5`. The digit sits at `length − 1`; name lengths run 1–11.
20 training and 20 held-out documents from the same generator, disjoint names,
half positive, so **the best constant is 0.500 on both**.

| arm | space | evaluated | exhausted | conforming | certificate | held out | P(random pick conforms) |
|---|---|---|---|---|---|---|---|
| **A1b** no library, constant addresses only | 30 | 30 | true | **0** | `complete` | — | 0 |
| **A1c** no library, 3 nodes | 750 | 750 | true | **1** | **`unique`** | **1.000** | 0.00133 |
| **A2 the certified class** (`n0 = add`) | 150 | 150 | true | **0** | `complete` | — | **0** |
| **A2b the same schema, vector re-selected here** | 750 | 750 | true | **1** | **`unique`** | **1.000** | 0.00133 |
| **A3** wrong module, comparable size | 25 | 25 | true | **0** | `complete` | — | 0 |
| **A4** hand-authored `add` equivalent | 150 | 150 | true | **0** | `complete` | — | 0 |

Read the table in this order:

* **A1b's `complete` at 0 is a proof, not a failure to search**: no constant
  address solves `T_C2`. The task really does need a computed address.
* **A2 scores zero.** The class certified `unique` at two widths, with a
  bit-identical instantiation at 4,096 in hand, contributes **nothing**. Its
  stored vector is `add`; the computer domain needs `sub`.
* **A4 scores zero too**, which is the control that matters: the failure is the
  **vector**, not the transfer machinery. A hand-authored `add` detector fails
  identically.
* **A2b succeeds** — the schema instantiated at 4,096 and its vector re-selected
  on computer episodes picks `module:0cf38f0599e2…(blen, p1, data, b53)`, the
  **`sub`** member of the same five-program family, `unique`, 1.000 held out.
* **And A2b enumerates 750 programs — exactly what A1c enumerates.** The schema
  buys **zero** search saving here. §55's F-b in its sharpest form: when the
  vector must be re-found, "not enumerating" saves nothing, because you enumerate.

**F3 fires.** A2's held-out score is not merely equal to `max(A1, best constant)`;
there is no held-out score, because nothing in the class's space conforms.

### 5.3 What this says about §55's format, precisely

A §55 schema class is a **schema reference plus a selection vector plus certified
widths**. This track separates those two halves against a real cross-domain
boundary and finds them to have opposite fates:

| half of the class | crosses a domain boundary? | evidence |
|---|---|---|
| the **schema** (width → scaffold) | **yes** | bit-identical instantiation at 128, 3,072 **and** 4,096 (M2); at 128 and 3,072 (M3); a valid 4,096 build with the right free-node shape |
| the **certified vector** | **no** | `n0 = add`, `unique` at two widths, **0 conforming** on `T_C2` |
| a **hardened artifact** | **no**, as §30 said | `TypeError: operator signature mismatch` at another width |

And the two motifs pull in opposite directions. **M2 spans all three domains but
carries no vector**, so the format refuses it (R2). **M3 carries a vector but
spans two domains**, and its vector does not transfer to the third. There is no
configuration of this motif in which §55's format both applies and helps.

---

## 6. Falsifications, as registered

| | outcome |
|---|---|
| **F1** the three instances are not the same computation once checked | **FIRES**, on both clauses — the three-node motif is in two domains, not three; the compared value is a literal in two domains and a buffer read in the third. The two-node motif *is* in all three and *is* the same computation, exhaustively over every address. §58's structural conclusions are unaffected. |
| **F2** unifies nominally but cannot instantiate bit-identically | **does not fire** — 5 of 5 gated instantiations bit-identical, zero differing `to_dict` keys |
| **F3** unifies and instantiates but helps no cross-domain task | **FIRES** — `T_C2`: 0 conforming, `complete`, against no-library's 1 conforming, `unique`, 1.000 held out at equal node count |
| **F4** bookkeeping — no-library already solves it, or the saving equals the enumeration removed | **FIRES** — `T_C1`: A1b solves it with two core nodes at 1.000 held out; `T_C2`: A2b's space is 750, identical to A1c's |
| **F5** the unification requires relaxing a type check or erasing width | **does not fire** — `tcn/` and `generators/` diffs empty, no `except` in `schema.py`, every bound reproduced, `version` the only normalised field |
| **F6** the wrong-module control is not discriminating | **does not fire** — A3 reaches 0 conforming on both tasks; the wrong-width module cannot be constructed at all |

---

## 7. Achieved difficulty, stated rather than implied

* `T_C1`: the whole no-library space at 3 nodes is 6,400 programs of which 106
  conform — a random pick conforms with probability **0.0166**. Best constant
  0.667 on both splits; every conforming program scores 1.000 held out.
* `T_C2`: 750 programs, **1** conforms — probability **0.00133**. Best constant
  **0.500** on both splits. This is the harder task and the only one that
  discriminates.
* Certification: the schema's own space is **5 programs** at each width. That is
  a small space, and `unique` in a space of five is a weak certificate on its
  own — which is exactly why the two-width rule and arm F carry the weight here
  rather than the certificate.

---

## 8. What this track discloses about itself

* **The M2/M3 split was visible in §58's committed `bounds_5.json` before this
  track's pre-registration was written.** That reading is what set F1's
  threshold, and `PREREGISTRATION.md` §0 records it as prior observation rather
  than presenting it later as an arm's result. Nothing was searched, fitted or
  scored before the pre-registration commit.
* **`T_C1` was the wrong task and is kept rather than deleted.** It was designed
  before its data was inspected, and its discriminating byte turned out to sit at
  a constant address, which makes it unable to separate a transferred address
  computation from a constant lookup. `T_C2` replaces it as the deciding arm
  (amendment A1). `T_C1`'s numbers are reported in full because they are what
  showed the flaw, and because A2 = A4 on it is the one place the transferred
  artifact does reach the hand-authored ceiling.
* **`T_C2` is 20 training and 20 held-out documents.** That is small. The claim
  it carries is structural rather than statistical — `A1b`'s `complete` at 0
  conforming proves no constant address can work, and `A2`'s `complete` at 0
  proves no call site of the `add` module can — but the held-out 1.000 figures
  rest on 20 examples and should be read as such.
* **The certification episodes balance the label deliberately.** The compared
  byte is the buffer's own byte half the time, so the best constant sits at ~0.5
  rather than at the artifact's own skew. That is stated in `episodes.py` and is
  why the constant baselines read 0.531 and 0.521.
* **The behavioural characterisation is not exhaustive over buffers.** §2.3 gives
  its quantifier exactly. No sampled agreement is called a match anywhere.
* **`description_bits` is used to rank nothing** (§41, §52, §54, §57), and no
  pooling is performed, so §57's fragmentation inversion cannot arise here.
* **The class store is a sidecar.** `research/class-identity/classes.py` is
  imported unmodified; `tcn/library.py` is not touched and no core change is
  proposed. §55's verdict on that stands unchanged.

---

## 9. Verification

**Reproduce**, repo `.venv`, from the worktree root, in order:

```bash
.venv/bin/python research/cross-domain/artifacts.py     # the 8 programs + digest checks
.venv/bin/python research/motif-unification/run_step1.py      # ~2m45s
.venv/bin/python research/motif-unification/run_step2.py
.venv/bin/python research/motif-unification/run_step3.py
.venv/bin/python research/motif-unification/run_transfer.py   # T_C1
.venv/bin/python research/motif-unification/run_transfer2.py  # T_C2
.venv/bin/python research/motif-unification/verify_claims.py  # 82 checks
```

**Every headline re-derived independently.** `verify_claims.py` re-reads
`out/*.json` and recomputes each claim without importing `motif.py`, `schema.py`
or `episodes.py`, so a bug in a run script cannot make its own claim true:
**82 checks, all pass** — the occurrence counts and domains, the arity difference,
both F1 clauses, the behavioural failure counts, all five bit-identity gates and
the empty `to_dict` diffs, the G2 checklist, both certifications with their
baselines, all three admission outcomes with the rule each refusal names, both
arm-F arms at both widths, and every cell of both transfer tables.

**Repo constraints.**

* Tests: **324 passed, 13 failed** (`out/pytest.log`). All 13 are the known
  environmental class — `generators/computer/engine/bridge.ts` launched with
  `node --import tsx`, failing `ERR_MODULE_NOT_FOUND: Cannot find package 'tsx'`
  because the gitignored `node_modules` symlink is absent from this worktree.
  Verified per the standing instruction by **moving
  `research/motif-unification/` out of the tree entirely** and re-running
  `tests/test_generators.py tests/test_panel_interface.py`: **the same 13 fail,
  17 pass**. 324 + 13 = **337**.
* Shipped fixture (`out/fixture.log`, `.venv/bin/python -m tcn train --episodes 160`):
  `initial_prediction_loss` **0.248835613951087** → `final_prediction_loss`
  **0.0022308224288281053**, `evaluation_mean_return` **4.0**, `fully_frozen`
  **true**, `frozen_evaluation_mean_return` **4.0** — 0.248836 → 0.002231 at 4/4
  frozen, as required.
* `tcn/` and `generators/` untouched; `research/refinement-bounds/` untouched;
  `research/class-identity/`, `research/cross-domain/` and
  `research/depth-encoding/` imported, never edited.
* Largest artifact written: `out/step1.json` at 16 kB. Full `Type` dictionaries
  are replaced in the stored records by §58's own `label#sha256` tag — a
  4,096-field product serialises to ~330 kB and the first version of this report
  was 4.9 MB. The tag is exact, and every identity is computed in memory from the
  real `Type` objects, never from the tag.

---

## 10. Amendments

### A1 — replaces `PREREGISTRATION.md` §2 step 4's single held-out task

Recorded **after `T_C1` was scored and before `T_C2` was built**. The
pre-registration fixed one held-out computer task, `T_C1` ("should the agent
write?"). Its arms ran as registered and are reported in full in §5.1. Inspecting
its data then showed that its discriminating byte sits at a **constant** address,
so no arm of `T_C1` can distinguish "the module supplied a transferred address
computation" from "a constant lookup sufficed". `T_C2` is added as a second
held-out task in the same domain, on the same generator, with the same arm
structure and the same registered falsifications, chosen so that the address
**cannot** be a constant — the property §23 states about its own environment.
`T_C1` is **not** withdrawn and its numbers are not reinterpreted; `T_C2` is the
arm the verdict rests on, and F3 is evaluated against it.

### A2 — replaces `PREREGISTRATION.md` §2 step 4's arm A3

Recorded **before any transfer arm was scored**. The pre-registration says A3 is
"a wrong module of comparable size (same node count, same arity, same
signature)". The obvious candidate — the same schema with `n0 = sub` — is *not* a
valid control on `T_C1`, because the available address constants let `sub` reach
the same address as `add`. A3 is instead the **other real motif**: `V_same`'s
"are the bytes at two computed addresses equal", three nodes, instantiated at
4,096. Same size, drawn from a real artifact rather than invented, and wrong for
the task rather than broken. Arity is 3 rather than 4 as a consequence; the
signature match required by the original wording is therefore **not** met, and
that is recorded here rather than glossed. A second control, **A3b**, restores
§30's version of the same question: the class instantiated at the *language*
width, offered to the computer scaffold, which cannot be constructed at all.

### A3 — records the one field `freeze` normalises

Recorded before step 2 was gated. `PREREGISTRATION.md` §1 defines "bit-identical"
as equality of `Program.digest` with "nothing weaker". `Program.harden`
increments `version`, a monotone provenance counter that is not part of the
graph, while `frag.canonicalize` builds a fresh program at `version = 1`.
`schema.freeze` normalises `version` back, and **only** `version`. §3.1 reports
the field-by-field `to_dict()` diff so the reader can confirm the normalisation
is not doing work: the diff is empty on every gated row.
