# Within-domain width reuse on the shipped visual parser

`PREREGISTRATION.md` was committed at `94c7cf4`, before any arm. `tcn/` and
`generators/` have no diff on this branch.

**The evidence rule this document follows.** Every figure-bearing table and
sentence sits between `<!-- BEGIN:… -->` / `<!-- END:… -->` markers and is
written by `report.py` from files under `out/`. `verify.py` re-renders every
block, recomputes each headline claim independently from the raw JSON (rebuilding
the schema to re-derive the digests), and fails on any number in the prose that
does not trace to a block or to a cited file. Run `capped.sh verify verify.py`.

**Scope of the evidence, stated once and repeated where it matters.** Every
result below rests on **one seed set** (train seeds `0–5`, validation `50–52`,
held-out `100–105`, parse screens from `200`), **one palette** (`palette 32`),
**square screens only**, and **one span rule** (`span = max(W, H)`). Nothing here
varies those.

---

## VERDICT

<!-- BEGIN:headline -->
<!-- END:headline -->

*(Section 7 gives the verdict in full, including the arm F result.)*

---

## 1. The width axis, justified from code

The shipped visual parse has two parameters that look like widths.
**Only one of them is a width in the type system**, and that is the one this
track holds out.

### 1.1 Raster resolution is a width: the port type *is* the resolution

`generators/gui/generator.py:119-121` derives `width` and `height` from the
`resolution` key. Every stage's input port is built from functions of it:

* `research/visual-ladder/common.py:81-82` —
  `bytes_type(W, H, channels=3) = product(*(BYTE for _ in range(3*W*H)))`
* `research/visual-ladder/common.py:85-86` —
  `record_type(W, H) = product(IDX, bytes_type(W, H))`

Measured by building S2' at seven `(resolution, span)` pairs and counting the
leaves of its `rec` port type (`out/types_probe.json`):

<!-- BEGIN:types_table -->
<!-- END:types_table -->

The leaf count is `3·W·H + 1`, exactly — §53's `3 × depth`, on a real artifact.
The width also drives the *constants*, which §53's channel did not:
`rung3_widgets.py:172-181` `offset_pool(W) = (6, 3W+3, 3, 3W, 9)`, so the
correct vertical step `3W` changes its *value* with resolution but not its
*index*; `rung3_root.py:59` `stride = 3W`; `rung3_root.py:107`
`last = 3·W·H − 3`.

### 1.2 `rect_scaffold`'s span is *not* a width — it appears in no type

The three rows at resolution 32 above are the measurement: three spans give
three different programs (three digests, three node counts) and one identical
port type. §41 swept this parameter and certified `none exists` at spans 4–29
and `unique` at 30 (on resolution 32 only). So `span` is a derived schema
parameter, defaulted to `span = max(W, H)` (`rung3_widgets.py:234`). That
derivation is structurally sufficient: a widget with corner `x ≥ 1` and width `w`
needs `x + w ≤ W`, and the scaffold represents extents up to exactly `span`.

**Decision:** the axis is raster resolution; `span` is carried inside the schema,
and freezing it is arm F's second payload.

### 1.3 The widths, and the difficulty actually achieved at each

Only `resolution` moves; every other key of `FLAT` is held fixed. Achieved, not
requested (`achieved` in each `out/enum_*.json`, 12 screens per resolution):

<!-- BEGIN:widths_table -->
<!-- END:widths_table -->

The achieved widget count **falls below the request at the two smallest
resolutions** — the layout cannot place twenty widgets there — so those are an
easier task, and that is stated rather than smoothed over.

---

## 2. What is measured, and how the arms are made comparable

The pipeline is §33's, staged as `rung3_root.main` stages it: **S0**
`same(a, b, obs)` is exhausted and hardened into a module; **S1'** `corner(rec)`
calls that module twice at searched offsets; **S2'** `rect(rec)` calls it
`2·(span−1)` times. Supervision reaches a program only through
`record.actor_view().observations`.

**The class store is §55's, unchanged** — `research/class-identity/classes.py`,
rules R1 (two widths), R2 (vector agreement), R3 (instantiation digest). The
schema is a *reference* to the width-parametric constructors already in the
repository, exactly as §55 found a class can hold.

**Both arms run through one code path.** The without arm walks the whole space
calling `tcn.search.evaluate` once per program; the with arm calls the same
`evaluate` once, on the same rows. The shipped `enumerate_prefix` search is timed
beside it because that is what the repository pays.

**One mechanical fact decides how to read the wall-clock column.**
`tcn.search.evaluate` (`tcn/search.py:111-133`) returns as soon as one row
exceeds the tolerance. A wrong program therefore usually costs one row; the right
program costs every row. So an exhaustive walk is much cheaper than
`space × (one full check)`, and "row-evaluations" and "node units" — which count
`programs × rows` — are **nominal upper bounds** on the walk's work, not
measurements of it. Only wall clock through the same path measures executed work.

**The staged space is the honest "without".** The joint space is the product of
the three stage spaces, but staging is what `rung3_root.py` runs, so the without
arm is charged their *sum*. Crediting the class with the staging would inflate
the saving by orders of magnitude, and it is not credited.

---

## 3. Certificates at the training widths

Exhaustive walk, every program in each space (`out/enum_24.json`,
`out/enum_32.json`; one seed set):

<!-- BEGIN:cert_table -->
<!-- END:cert_table -->

The shipped `enumerate_prefix` search agrees on every count:

<!-- BEGIN:cert_prefix_table -->
<!-- END:cert_prefix_table -->

S0 and S1' carry `complete`, not `unique`, and that is §33's own record, not a
weakening: S0's two conforming programs are one function, and validation
filtering leaves the same selected vector. Only S2' is `unique`.

### 3.1 §33 does not move

Resolution 32 is both a certified width and the §33 configuration, so the
replication is compared against §33's own files, not against constants typed
here:

<!-- BEGIN:s33_table -->
<!-- END:s33_table -->

### 3.2 The same vector at every resolution, and the class

The exhaustive search, run independently at all five resolutions, selects:

<!-- BEGIN:vector_table -->
<!-- END:vector_table -->

The vector is index-valued, and the byte offsets it names track `3W` exactly.
The class admitted from the two certified widths (`out/construct.json`):

<!-- BEGIN:class_table -->
<!-- END:class_table -->

---

## 4. With the class against without it, at the held-out widths

### 4.1 Per stage

<!-- BEGIN:ww_table -->
<!-- END:ww_table -->

**The class yields a conformance check, never a uniqueness certificate** — the
same price §55 stated. Every without-arm row ends with an exhaustive certificate
(`unique` for S2', `complete` for S0 and S1'); every with-arm row ends with
`none`. What the with arm knows is that the one program it built passes the
training rows; it does not know that nothing else would.

Accuracy against the baselines. The whole-space columns score **every** program
in the space on the same held-out rows, so "uniform random over the space" is an
exact expectation, not a sample:

<!-- BEGIN:acc_table -->
<!-- END:acc_table -->

### 4.2 The whole pipeline, and whether the saving exceeds the space size

<!-- BEGIN:pipe_table -->
<!-- END:pipe_table -->

The parse, both arms, component for component:

<!-- BEGIN:parse_table -->
<!-- END:parse_table -->

<!-- BEGIN:collision_line -->
<!-- END:collision_line -->
It is a property of `palette 32`, not of width: no corner rule over colour can
see that widget, and both arms miss it identically because they are the same
program. §33's result at resolution 32 is unaffected.

### 4.3 §55 beside this track

<!-- BEGIN:s55_table -->
<!-- END:s55_table -->

---

## 5. Arm F — the soundness test on a real artifact

Two wrong schemas, each **bit-identical to the right one at resolution 32**.
`frozen_offsets` freezes the offset pool at its resolution-32 bytes;
`frozen_span` freezes `span` at its resolution-32 value. The first is wrong on
both sides of 32. The second is only wrong *above* 32: a larger span than needed
is masked off by the scaffold's own in-bounds test.

<!-- BEGIN:armf_table -->
<!-- END:armf_table -->

Offered to §55's store:

<!-- BEGIN:armf_format_table -->
<!-- END:armf_format_table -->

### 5.1 F-a: the width barrier is real

<!-- BEGIN:refusal_table -->
<!-- END:refusal_table -->

---

## 6. Every pre-registered criterion, resolved

`check.py` against raw JSON (`out/check.json`):

<!-- BEGIN:criteria_table -->
<!-- END:criteria_table -->

---

## 7. Verdict

*(written after arm F and resolution 48 report)*

---

## 8. Costs, environment, reproduction

<!-- BEGIN:size_line -->
<!-- END:size_line -->

<!-- BEGIN:null_line -->
<!-- END:null_line -->

Every job after the host crash ran under `capped.sh` — a `systemd-run --user`
scope with a memory cap, `CPUQuota=800%`, single-threaded BLAS and torch, and at
most four workers at once. Peak memory, from `/usr/bin/time -v`:

<!-- BEGIN:resource_table -->
<!-- END:resource_table -->

Reproduce, in order, each line under `capped.sh`: `run.py enumerate` at each
resolution, `run.py construct`, `run.py instantiate` at each held-out
resolution, `run.py armf`, `run.py refusal`, `run.py null`, `size.py`,
`types_probe.py`, `collision.py`, `check.py`, `report.py`, then `verify.py`.

## 9. Preserved negatives and disclosures
