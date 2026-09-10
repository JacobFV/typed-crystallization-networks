# Pre-registration — does §55's schema-plus-instantiation mechanism carry to the shipped visual parser?

Written before any arm is run. House standard since §44. Any change after the
first arm is recorded as a numbered amendment naming the criterion it replaces.

Track directory: `research/visual-width-reuse/`. Nothing under `tcn/` is
expected to change; if a core hook turns out to be required it will be minimal,
off by default, and justified from a measurement recorded here.

---

## 0. The gap this addresses

§53 measured that a depth-parametric schema's selection vector is invariant
across six widths, and §55 measured a **272× saving in episodes** by
instantiating that vector at three unseen widths instead of enumerating.
**Both numbers are on the synthetic `program` generator**, whose observation
width is `3 × depth` on a toy channel. §58 then showed width typing is what
blocks cross-domain abstraction, and §60 showed the schema crosses domains while
the certified vector does not — so *within-domain, across-width* reuse is the
strongest surviving form of the reuse claim, and **it has never been
demonstrated on a real artifact**.

§55 also flagged its own falsification: its saving **equalled the space size**
(17 × 16 = 272), because "not enumerating" trivially saves exactly the
enumeration. It called itself *"a demonstrated mechanism, not a capability at
scale."* Whether that caveat is an artifact of a toy space or a property of the
mechanism is the question this track is for.

---

## 1. Step 1 — the width axis, justified from code

The shipped visual parse (§32, §33; `research/visual-ladder/rung3_widgets.py`
and `rung3_root.py`) has two candidate width-like parameters. They are not the
same kind of thing and only one of them is a width in the typing sense.

### 1.1 Raster resolution — a real width, and the one the *types* depend on

`generators/gui/generator.py:119-121` reads `resolution` from the generator
configuration and derives `width` and `height` from it. The track's own type
constructors are then functions of that:

* `research/visual-ladder/common.py:81-82`
  `bytes_type(width, height, channels=3) = product(*(BYTE for _ in range(3*W*H)))`
  — the observation port's type is a product with **`3·W·H` components**.
* `common.py:85-86` `record_type(W, H) = product(IDX, bytes_type(W, H))` — every
  stage's input port.

So the `obs`/`rec` port type of S0, S1' and S2' literally *is* the resolution:
3,072 components at `resolution=32`, 6,912 at 48. This is exactly §30's and
§53's situation — a fixed-width typed program cannot accept an unseen
resolution — on a real artifact instead of a synthetic channel. It is a
**strictly stronger analogue** of §53's `3 × depth`, because here the width also
drives the *constants*:

* `rung3_widgets.py:172-181` `offset_pool(width) = (6, 3W+3, 3, 3W, 9)` — two of
  the five candidate spatial steps are width-derived. The correct vertical step
  is `3W`, whose *value* changes with resolution while its *index* does not.
* `rung3_root.py:59` `stride = 3W`; `rung3_widgets.py:243` /
  `rung3_root.py:107` `last = 3·W·H − 3`, and `width`, `height` as declared
  constants.

**Prediction (not a criterion):** the selection vector is index-valued and
therefore width-invariant, while every program digest is width-specific.

### 1.2 Scaffold span — width-derived, but not a width in the type system

`rect_scaffold(..., span)` (`rung3_widgets.py:233-243`, `rung3_root.py:105-110`)
unrolls `span − 1` prefix-conjunction terms per axis, nine nodes apiece. §41
swept it with every space exhausted and certified **`none exists` at spans 4–29
and `unique` at 30** on `resolution=32`.

But `span` appears in **no type**: it changes the node count and nothing else.
It is a *derived schema parameter*, defaulted to `span = max(W, H)`
(`rung3_widgets.py:234`). Its sufficiency is structural: a widget with corner
`x ≥ 1` and width `w` needs `x + w ≤ W`, so `w ≤ W − 1 ≤ span`, and the extent
the scaffold can represent is exactly `span`.

**Decision.** The width axis is **raster resolution**. `span` is carried as a
width-derived parameter of the schema (`span = max(W, H)`), not as an
independent axis — because the parse's *types* depend on resolution and not on
span, and §55's mechanism is about types. §41's sweep is used as the reason the
derivation must scale rather than be frozen, and it supplies arm F's payload.

### 1.3 The widths

Every other key of `common.py:57`'s `FLAT`
(`widgets=20, nesting=5, min_size=4, palette=32, horizon=2`) is held fixed;
only `resolution` moves. Achieved difficulty (achieved widget count, achieved
extents) is reported per width, never the requested config.

| role | resolutions |
|---|---|
| **certified (training)** | **24, 32** |
| **held out** | **16, 40, 48** |

At least two certified widths, per §55's R1. `32` is the shipped configuration,
so certifying there doubles as the §33 replication; `24` is a second, smaller
certified width. The held-out set deliberately contains one width **below** both
certified widths (16) and two **above** (40, 48) — §55 held out only widths
above its certified pair.

`resolution=12` and `20` were seen by `probe.py` (a feasibility probe run before
this document, reported in `out/probe.json`, consulted for nothing but
schedulability) and are excluded to bound compute; 12 achieves only 3 widgets
per screen, which is a different task.

---

## 2. What is stored, and what the arms are

The class store is **reused from §55 unchanged**:
`research/class-identity/classes.py` (`ClassStore`, `ClassRecord`,
`Certification`, rules R1–R3). `tcn/library.py` is not modified. The schema is a
**reference** to the width-parametric constructors already in the repository —
`same_scaffold`, `corner_scaffold_masked`, `rect_scaffold_clamped` — exactly as
§55 established a class can hold a schema reference but not a schema.

Known limitation, stated in advance: `tcn.generation.source_fingerprint()`
(`tcn/generation.py:33-42`) walks only `tcn/` and `generators/`, so schema code
living under `research/` is **outside the fingerprint's scope**. This was
equally true of §55 (whose schema lived in `research/depth-encoding/run.py`) and
is recorded, not fixed here.

| arm | what it does |
|---|---|
| **A — certify** | Full staged pipeline (S0 → S1' → S2') enumerated exhaustively at resolutions 24 and 32. Space size, evaluated, exhausted, conforming, certificate per stage. Admit the class. |
| **B — without** | The same full pipeline enumerated exhaustively at 16, 40, 48. This is the honest cost of not having the class. |
| **C — with** | Instantiate the stored vector at 16, 40, 48 from the schema. Zero programs searched; one conformance check per stage. Compare digest against arm B's enumerated program. |
| **F — the wrong schema** | §53's arm F on this artifact: a schema that agrees at exactly one width. |
| **N — null control** | A timing null control for the §59 instrument floor (~0.6% wall clock on this host). |

### 2.1 Arm F, stated in advance

The wrong schema **freezes the offset pool at its resolution-32 values**:
`offset_pool_wrong(width) = (6, 99, 3, 96, 9)` instead of
`(6, 3W+3, 3, 3W, 9)`. At `resolution=32` this is *identical* to the right
schema — same space, same nodes, same certificate, bit-identical program. At any
other resolution the vertical step `96` is wrong (it must be `3W`).

This is the faithful analogue of §53's arm F and it is the mistake that
single-width certification cannot possibly catch: it is literally "write down
the constants you measured at the width you tested".

A second wrong schema, **F2**, freezes `span = 32`, testing the §1.2 derived
parameter: correct at resolution 32, unrepresentable extents above it.

### 2.2 Cost currencies

Reported for every arm, because the §55 falsification lives in exactly this
choice:

1. **programs evaluated** — §55's headline currency.
2. **row-evaluations** — programs × supervision rows; the analogue of §55's
   "episodes".
3. **node evaluations** — `SearchResult.node_evaluations`, machine-independent.
4. **wall clock** — with the null control, per §59.

Both arms are measured through the *same* code path (an exhaustive
`all_conforming` walk versus a single `evaluate` on the same rows) so the ratio
is not an artifact of two different interpreters. The shipped
`enumerate_prefix` search wall clock is reported beside it as the realistic
search cost.

The **joint** space (`256 × 400 × 25 = 2,560,000`) is reported as context, but
the headline "without" arm is the **staged** pipeline `256 + 400 + 25 = 681`,
because staging is what `rung3_root.py` actually runs and crediting the class
with the staging decomposition would inflate the result.

---

## 3. Criteria, resolved by `check.py` against raw JSON

* **C1 — reproduction.** At every held-out resolution and for every stage, the
  digest of the instantiated artifact equals the digest of the program the
  exhaustive enumeration selects at that resolution.
* **C2 — the parse still parses.** At every held-out resolution the instantiated
  parse's rect recall, link accuracy and exact-tree count equal arm B's,
  component for component.
* **C3 — beats the baselines.** At every held-out resolution the instantiated
  vector's held-out accuracy exceeds the best constant predictor, uniform random
  over the selection space, and the whole-space mean, at every stage.
* **C4 — arm F is rejected.** The class format refuses the wrong schema by R1
  when certified at one width and by R2 when offered two; and the wrong schema
  yields **0 conforming, exhausted** at every resolution other than 32.
* **C5 — §33 does not move.** At resolution 32: S0 space 256 / 2 conforming /
  1 distinct function / `complete`; S1' 400 / 2 / `complete`; S2' 25 / 1 /
  `unique`; parse 227/227 rectangles, 227/227 links, 12/12 exact trees, 0 corner
  key collisions.
* **C6 — the honest saving.** State the saving on all four currencies of §2.2
  and say plainly whether it exceeds the largest per-stage space size.

**C6 carries a pre-registered prediction, so that a null result cannot be
re-narrated after the fact:** I expect the saving in programs and in node
evaluations to be **bounded by the per-stage space sizes**, because replacing an
exhaustive enumeration with one evaluation saves exactly the enumeration —
which would make §55's caveat a property of the mechanism rather than of its toy
space. If that is what the data says, it is the finding.

---

## 4. Falsification — declared now, honoured whatever happens

* **F-a — no width to hold out.** If the parse's types do not in fact vary with
  resolution, the analogue does not exist and the mechanism cannot be tested
  here. (§1.1 is a code reading, not a measurement; the arms must confirm that a
  resolution-32 artifact is *refused* at resolution 40.)
* **F-b — it does not reproduce.** If instantiation at a held-out resolution
  does not reproduce the enumerated program, **the mechanism does not carry to a
  real artifact**. This is the most consequential possible outcome and it is
  reported as the headline if it fires.
* **F-c — the saving is again the space size.** Then §55's caveat stands
  unchanged and scale buys nothing in ratio. Reported as such, not softened.
* **F-d — arm F is not rejected.** Then the two-width soundness rule is
  insufficient on real artifacts, which is a serious finding about §53/§55/§60's
  shared assumption.
* **F-e — a §33 certificate moves.** That is a regression, not a win, and is
  reported as one.
* **F-f — the derived span is not width-safe.** If `span = max(W, H)` yields
  0 conforming at some resolution, the schema's derived parameter is wrong and
  §1.2's structural argument is refuted.

---

## 5. Discipline

Enumeration with certificates beside every synthesis number; constant and random
baselines beside every accuracy; "no solution exists in this family" is
distinguished from "search failed" by exhaustion; achieved difficulty is
reported, never the requested config; every surprising headline is verified
against raw JSON in `out/`; negative results are preserved. Large artifacts stay
out of git.

Constraints honoured: the repository `.venv`; all 337 tests pass; the shipped
fixture reproduces 0.248836 → 0.002231 at 4/4 frozen.
