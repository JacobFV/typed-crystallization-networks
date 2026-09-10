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
- **Instantiation reproduces the enumerated program:** 9 of 9 stage artifacts at held-out resolutions 16, 40, 48 are digest-identical to what exhaustive enumeration selects there, and 9 of 9 pass the conformance check. The selection vector is identical at all 5 resolutions. (One seed set, one palette, square screens only.)
- **The saving, on four currencies, per held-out resolution:** programs 227.0× / 227.0× / 227.0×; row-evaluations 343.8× / 323.7× / 323.7×; wall clock through the same code path 4.4× / 2.2× / 1.8×; against the shipped prefix search 14.8× / 20.1× / 21.6×. The largest per-stage space is 400.
- **Does the saving exceed the space size?** On programs and row-evaluations, no: both are bounded by the largest stage space by construction. On same-path wall clock, no — and at S2' alone the same-path ratio is 1.5× / 1.2× / 1.2× against a space of 25, because a wrong program fails on an early row and the right one must run every row. Even against the shipped `enumerate_prefix` search the ratio stays below 400.
- **Arm F, frozen offsets:** refused at one width, refused at two; 0 conforming at 8 of 8 off-reference stage spaces. §53's result carries.
- **Arm F, frozen span:** refused at one width and **ADMITTED at two** (24, 32), because it is correct at and below the value it froze. It fails the conformance check at 40, 48, where it would ship held-out accuracy 0.9639 / 0.9639 against a best constant of 0.1236 / 0.1250: plausible, and wrong.
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
| resolution | span | S2' nodes | `rec` leaf components | S2' digest (vector `{step_w: 2, step_h: 3}`) |
|---|---|---|---|---|
| 16 | 16 | 298 | 769 | `c999cee93879b10f5e6a791c` |
| 24 | 24 | 442 | 1,729 | `33124a0ea737bdf447bb5fee` |
| 32 | 30 | 550 | 3,073 | `aaa20fe4984fb0c6c1f570fb` |
| 32 | 32 | 586 | 3,073 | `009b0bc80630eb8fcf02493e` |
| 32 | 48 | 874 | 3,073 | `970614c55b6e8f00421b982b` |
| 40 | 40 | 730 | 4,801 | `2d7f7a90e5bae6a50c9aa0f5` |
| 48 | 48 | 874 | 6,913 | `ea79d7a5415e530155d99cd4` |
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
| resolution | role | observation components | achieved widgets/screen (12 screens) | max widget width | max widget height | span | span ≥ every extent |
|---|---|---|---|---|---|---|---|
| 16 | held out | 768 | 7–10 (mean 8.00) | 16 | 16 | 16 | yes |
| 24 | **certified** | 1,728 | 8–15 (mean 11.75) | 24 | 24 | 24 | yes |
| 32 | **certified** | 3,072 | 16–20 (mean 18.83) | 32 | 32 | 32 | yes |
| 40 | held out | 4,800 | 20–20 (mean 20.00) | 40 | 40 | 40 | yes |
| 48 | held out | 6,912 | 20–20 (mean 20.00) | 48 | 48 | 48 | yes |
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
| res | stage | space | evaluated | exhausted | conforming | certificate | distinct functions | validation survivors | selected vector | artifact digest | held-out accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 24 | S0 `same` | 256 | 256 | True | 2 | `complete` | 1 | 2 | `{"rg": 7, "same": 2}` | `0db07f4b574b61296a4a762a` | 1.0000 (288 rows) |
| 24 | S1' `corner` | 400 | 400 | True | 2 | `complete` | — | 2 | `{"back_a": 2, "back_b": 3, "corner": 1}` | `52877e45c7f280abb5908df0` | 1.0000 (725 rows) |
| 24 | S2' `rect` | 25 | 25 | True | 1 | `unique` | — | 1 | `{"step_h": 3, "step_w": 2}` | `33124a0ea737bdf447bb5fee` | 1.0000 (65 rows) |
| 32 | S0 `same` | 256 | 256 | True | 2 | `complete` | 1 | 2 | `{"rg": 7, "same": 2}` | `b409b71ea0e41d77abb7cb9c` | 1.0000 (288 rows) |
| 32 | S1' `corner` | 400 | 400 | True | 2 | `complete` | — | 2 | `{"back_a": 2, "back_b": 3, "corner": 1}` | `6804983bda88826441cba10d` | 1.0000 (725 rows) |
| 32 | S2' `rect` | 25 | 25 | True | 1 | `unique` | — | 1 | `{"step_h": 3, "step_w": 2}` | `009b0bc80630eb8fcf02493e` | 1.0000 (109 rows) |
<!-- END:cert_table -->

The shipped `enumerate_prefix` search agrees on every count:

<!-- BEGIN:cert_prefix_table -->
| res | stage | `enumerate_prefix` evaluated | exhausted | conforming | certificate | node evaluations | seconds |
|---|---|---|---|---|---|---|---|
| 24 | S0 `same` | 256 | True | 2 | `complete` | 10,326 | 0.4 |
| 24 | S1' `corner` | 400 | True | 2 | `complete` | 205,630 | 33.7 |
| 24 | S2' `rect` | 25 | True | 1 | `unique` | 827,363 | 133.3 |
| 32 | S0 `same` | 256 | True | 2 | `complete` | 10,064 | 0.7 |
| 32 | S1' `corner` | 400 | True | 2 | `complete` | 200,692 | 58.9 |
| 32 | S2' `rect` | 25 | True | 1 | `unique` | 1,694,889 | 481.6 |
<!-- END:cert_prefix_table -->

S0 and S1' carry `complete`, not `unique`, and that is §33's own record, not a
weakening: S0's two conforming programs are one function, and validation
filtering leaves the same selected vector. Only S2' is `unique`.

### 3.1 §33 does not move

Resolution 32 is both a certified width and the §33 configuration, so the
replication is compared against §33's own files, not against constants typed
here:

<!-- BEGIN:s33_table -->
| quantity | §33 (`visual-ladder/out/rung3*.json`) | this run, resolution 32 | verdict |
|---|---|---|---|
| S0 space / conforming / distinct functions | 256 / 2 / 1 | 256 / 2 / 1 | same |
| S0 certificate | `complete` | `complete` | same |
| S0 module digest | `module:b409b71ea0e41d77abb7cb9c` | `module:b409b71ea0e41d77abb7cb9c` | same |
| S1' space / conforming / certificate | 400 / 2 / `complete` | 400 / 2 / `complete` | same |
| S1' vector | `{"back_a": 2, "back_b": 3, "corner": 1}` | `{"back_a": 2, "back_b": 3, "corner": 1}` | same |
| S2' space / conforming / certificate | 25 / 1 / `unique` | 25 / 1 / `unique` | same |
| S2' vector | `{"step_h": 3, "step_w": 2}` | `{"step_h": 3, "step_w": 2}` | same |
| parse `widgets_in_probe` | 227 | 227 | same |
| parse `rects_predicted` | 227 | 227 | same |
| parse `screens_rects_exact` | 12 | 12 | same |
| parse `roots_predicted` | 12 | 12 | same |
| parse `parent_links_correct` | 227 | 227 | same |
| parse `parent_links_wrong` | 0 | 0 | same |
| parse `trees_exact` | 12 | 12 | same |
| parse `corner_key_collisions` | 0 | 0 | same |
<!-- END:s33_table -->

### 3.2 The same vector at every resolution, and the class

The exhaustive search, run independently at all five resolutions, selects:

<!-- BEGIN:vector_table -->
| stage | free nodes (candidates) | res 16 | res 24 | res 32 | res 40 | res 48 |
|---|---|---|---|---|---|---|
| S0 `same` | `{"rg": 16, "same": 16}` | `{"rg": 7, "same": 2}` | `{"rg": 7, "same": 2}` | `{"rg": 7, "same": 2}` | `{"rg": 7, "same": 2}` | `{"rg": 7, "same": 2}` |
| S1' `corner` | `{"back_a": 5, "back_b": 5, "corner": 16}` | `{"back_a": 2, "back_b": 3, "corner": 1}` → bytes [3, 48] | `{"back_a": 2, "back_b": 3, "corner": 1}` → bytes [3, 72] | `{"back_a": 2, "back_b": 3, "corner": 1}` → bytes [3, 96] | `{"back_a": 2, "back_b": 3, "corner": 1}` → bytes [3, 120] | `{"back_a": 2, "back_b": 3, "corner": 1}` → bytes [3, 144] |
| S2' `rect` | `{"step_h": 5, "step_w": 5}` | `{"step_h": 3, "step_w": 2}` → bytes [3, 48] | `{"step_h": 3, "step_w": 2}` → bytes [3, 72] | `{"step_h": 3, "step_w": 2}` → bytes [3, 96] | `{"step_h": 3, "step_w": 2}` → bytes [3, 120] | `{"step_h": 3, "step_w": 2}` → bytes [3, 144] |
<!-- END:vector_table -->

The vector is index-valued, and the byte offsets it names track `3W` exactly.
The class admitted from the two certified widths (`out/construct.json`):

<!-- BEGIN:class_table -->
| stage | class id | admitted | certified widths | stored vector | member digests (24, 32) |
|---|---|---|---|---|---|
| S0 `same` | `schema/f72db2cf6eca/e50a32f8f00a` | True | [24, 32] | `{"rg": 7, "same": 2}` | `0db07f4b574b61296a4a762a`, `b409b71ea0e41d77abb7cb9c` |
| S1' `corner` | `schema/f72db2cf6eca/854934728ea4` | True | [24, 32] | `{"back_a": 2, "back_b": 3, "corner": 1}` | `52877e45c7f280abb5908df0`, `6804983bda88826441cba10d` |
| S2' `rect` | `schema/f72db2cf6eca/b551ea069e50` | True | [24, 32] | `{"step_h": 3, "step_w": 2}` | `33124a0ea737bdf447bb5fee`, `009b0bc80630eb8fcf02493e` |
<!-- END:class_table -->

---

## 4. With the class against without it, at the held-out widths

### 4.1 Per stage

<!-- BEGIN:ww_table -->
| res | stage | space | without: programs | row-evals | walk s | prefix-search s | certificate | with: programs | row-evals | check s | certificate | same program | digest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | S0 `same` | 256 | 256 | 73,728 | 0.8 | 0.2 | `complete` | 0 + 1 check | 288 | 0.19 | `none` | **yes** | `727a9d5f2fa56a03dc08d2bc` |
| 16 | S1' `corner` | 400 | 400 | 288,800 | 9.7 | 15.6 | `complete` | 0 + 1 check | 722 | 1.47 | `none` | **yes** | `0274a30b6438b543c9062086` |
| 16 | S2' `rect` | 25 | 25 | 1,200 | 1.8 | 26.0 | `unique` | 0 + 1 check | 48 | 1.16 | `none` | **yes** | `c999cee93879b10f5e6a791c` |
| 40 | S0 `same` | 256 | 256 | 73,728 | 6.5 | 1.1 | `complete` | 0 + 1 check | 288 | 1.09 | `none` | **yes** | `6636dab9a907e52defb3d0c7` |
| 40 | S1' `corner` | 400 | 400 | 290,400 | 57.9 | 91.1 | `complete` | 0 + 1 check | 726 | 8.96 | `none` | **yes** | `0672f2cc49eadbad341bc6cc` |
| 40 | S2' `rect` | 25 | 25 | 3,000 | 49.0 | 964.1 | `unique` | 0 + 1 check | 120 | 42.41 | `none` | **yes** | `2d7f7a90e5bae6a50c9aa0f5` |
| 48 | S0 `same` | 256 | 256 | 73,728 | 11.7 | 1.6 | `complete` | 0 + 1 check | 288 | 1.58 | `none` | **yes** | `0c0e85da1eb5ce067e395984` |
| 48 | S1' `corner` | 400 | 400 | 290,400 | 59.9 | 140.0 | `complete` | 0 + 1 check | 726 | 12.76 | `none` | **yes** | `878a367422894babff869ae0` |
| 48 | S2' `rect` | 25 | 25 | 3,000 | 86.5 | 1748.7 | `unique` | 0 + 1 check | 120 | 73.22 | `none` | **yes** | `ea79d7a5415e530155d99cd4` |
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
| res | stage | with-class accuracy (held rows) | best constant (same rows) | whole-space rows | chosen, on those rows | best constant, on those rows | uniform random over the space (exact mean) | second-best of the space | worst of the space |
|---|---|---|---|---|---|---|---|---|---|
| 16 | S0 `same` | 1.0000 (288) | 0.6181 | 150 | 1.0000 | 0.6333 | 0.5000 | 1.0000 | 0.0000 |
| 16 | S1' `corner` | 1.0000 (722) | 0.9668 | 150 | 1.0000 | 0.9800 | 0.5000 | 1.0000 | 0.0000 |
| 16 | S2' `rect` | 1.0000 (48) | 0.1840 | 48 | 1.0000 | 0.1840 | 0.7792 | 0.8958 | 0.6806 |
| 40 | S0 `same` | 1.0000 (288) | 0.6424 | 150 | 1.0000 | 0.5667 | 0.5000 | 1.0000 | 0.0000 |
| 40 | S1' `corner` | 1.0000 (725) | 0.9821 | 150 | 1.0000 | 0.9733 | 0.5000 | 1.0000 | 0.0000 |
| 40 | S2' `rect` | 1.0000 (120) | 0.1236 | 120 | 1.0000 | 0.1236 | 0.7644 | 0.8875 | 0.6722 |
| 48 | S0 `same` | 1.0000 (288) | 0.7431 | 150 | 1.0000 | 0.7667 | 0.5000 | 1.0000 | 0.0000 |
| 48 | S1' `corner` | 1.0000 (726) | 0.9890 | 150 | 1.0000 | 0.9733 | 0.5000 | 1.0000 | 0.0000 |
| 48 | S2' `rect` | 1.0000 (120) | 0.1250 | 120 | 1.0000 | 0.1250 | 0.7642 | 0.8889 | 0.6764 |
<!-- END:acc_table -->

### 4.2 The whole pipeline, and whether the saving exceeds the space size

<!-- BEGIN:pipe_table -->
| res | staged space | joint space | largest stage space | programs | ratio | row-evaluations | ratio | nominal node-unit ratio | walk s → check s | ratio | shipped prefix search s | prefix ÷ check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 681 | 2,560,000 | 400 | 681 → 3 | 227.0× | 363,728 → 1,058 | 343.8× | 226.5× | 12.2 → 2.8 | 4.4× | 41.7 | 14.8× |
| 40 | 681 | 2,560,000 | 400 | 681 → 3 | 227.0× | 367,128 → 1,134 | 323.7× | 89.8× | 113.5 → 52.5 | 2.2× | 1056.3 | 20.1× |
| 48 | 681 | 2,560,000 | 400 | 681 → 3 | 227.0× | 367,128 → 1,134 | 323.7× | 80.8× | 158.1 → 87.6 | 1.8× | 1890.3 | 21.6× |
<!-- END:pipe_table -->

The parse, both arms, component for component:

<!-- BEGIN:parse_table -->
| res | arm | screens | widgets | rects predicted | screens exact | roots | links right / wrong | trees exact | key collisions |
|---|---|---|---|---|---|---|---|---|---|
| 16 | without (enumerated) | 6 | 51 | 51 | 6 | 6 | 51 / 0 | 6 | 0 |
| 16 | with (instantiated) | 6 | 51 | 51 | 6 | 6 | 51 / 0 | 6 | 0 |
| 40 | without (enumerated) | 6 | 120 | 119 | 5 | 6 | 119 / 0 | 5 | 0 |
| 40 | with (instantiated) | 6 | 120 | 119 | 5 | 6 | 119 / 0 | 5 | 0 |
| 48 | without (enumerated) | 6 | 120 | 120 | 6 | 6 | 120 / 0 | 6 | 0 |
| 48 | with (instantiated) | 6 | 120 | 120 | 6 | 6 | 120 / 0 | 6 | 0 |
<!-- END:parse_table -->

<!-- BEGIN:collision_line -->
The one miss is resolution 40, seed 205, rectangle `[3, 11, 6, 9]` (`out/collision.json`, one screen): its fill `[30, 30, 170]` equals its left neighbour `[30, 30, 170]` and its upper neighbour `[30, 30, 170]`, both pixels of its parent `[2, 2, 8, 19]`; the screen shows 16 distinct colours for 20 widgets. Its extent is within the span (`True`).
<!-- END:collision_line -->
It is a property of `palette 32`, not of width: no corner rule over colour can
see that widget, and both arms miss it identically because they are the same
program. §33's result at resolution 32 is unaffected.

### 4.3 §55 beside this track

<!-- BEGIN:s55_table -->
| track | held-out width | space | without: programs | without: episodes / row-evals | without s | with s | episode / row ratio | wall-clock ratio (same code path) |
|---|---|---|---|---|---|---|---|---|
| §55 `program` | depth 5 | 272 | 272 | 17,408 | 58.1 | 0.22 | 272.0× | 268.8× |
| §55 `program` | depth 7 | 272 | 272 | 17,408 | 79.5 | 0.30 | 272.0× | 269.5× |
| §55 `program` | depth 12 | 272 | 272 | 17,408 | 146.9 | 0.55 | 272.0× | 266.8× |
| this track, visual parse | resolution 16 | 681 | 681 | 363,728 | 12.2 | 2.81 | 343.8× | 4.4× |
| this track, visual parse | resolution 40 | 681 | 681 | 367,128 | 113.5 | 52.47 | 323.7× | 2.2× |
| this track, visual parse | resolution 48 | 681 | 681 | 367,128 | 158.1 | 87.56 | 323.7× | 1.8× |
<!-- END:s55_table -->

---

## 5. Arm F — the soundness test on a real artifact

Two wrong schemas, each **bit-identical to the right one at resolution 32**.
`frozen_offsets` freezes the offset pool at its resolution-32 bytes;
`frozen_span` freezes `span` at its resolution-32 value. The first is wrong on
both sides of 32. The second is only wrong *above* 32: a larger span than needed
is masked off by the scaffold's own in-bounds test.

<!-- BEGIN:armf_table -->
| wrong schema | res | stage | space | evaluated | exhausted | conforming | certificate | stored vector conforms | stored vector held accuracy | best constant | bit-identical to right schema |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `frozen_offsets` | 16 | S1' `corner` | 400 | 400 | True | 0 | `none exists` | False | 0.9557 | 0.9668 | no |
| `frozen_offsets` | 16 | S2' `rect` | 25 | 25 | True | 0 | `none exists` | False | 0.8542 | 0.1840 | no |
| `frozen_offsets` | 24 | S1' `corner` | 400 | 400 | True | 0 | `none exists` | False | 0.6979 | 0.9793 | no |
| `frozen_offsets` | 24 | S2' `rect` | 25 | 25 | True | 0 | `none exists` | False | 0.8333 | 0.1564 | no |
| `frozen_offsets` | 32 | S1' `corner` | 400 | 400 | True | 2 | `complete` | True | 1.0000 | 0.9738 | yes |
| `frozen_offsets` | 32 | S2' `rect` | 25 | 25 | True | 1 | `unique` | True | 1.0000 | 0.1147 | yes |
| `frozen_offsets` | 40 | S1' `corner` | 400 | 400 | True | 0 | `none exists` | False | 0.7007 | 0.9821 | no |
| `frozen_offsets` | 40 | S2' `rect` | 25 | 25 | True | 0 | `none exists` | False | 0.8333 | 0.1236 | no |
| `frozen_offsets` | 48 | S1' `corner` | 400 | 400 | True | 0 | `none exists` | False | 0.7259 | 0.9890 | — |
| `frozen_offsets` | 48 | S2' `rect` | 25 | 25 | True | 0 | `none exists` | False | 0.8333 | 0.1250 | — |
| `frozen_span` | 16 | S1' `corner` | 400 | 400 | True | 2 | `complete` | True | 1.0000 | 0.9668 | yes |
| `frozen_span` | 16 | S2' `rect` | 25 | 25 | True | 1 | `unique` | True | 1.0000 | 0.1840 | no |
| `frozen_span` | 24 | S1' `corner` | 400 | 400 | True | 2 | `complete` | True | 1.0000 | 0.9793 | yes |
| `frozen_span` | 24 | S2' `rect` | 25 | 25 | True | 1 | `unique` | True | 1.0000 | 0.1564 | no |
| `frozen_span` | 32 | S1' `corner` | 400 | 400 | True | 2 | `complete` | True | 1.0000 | 0.9738 | yes |
| `frozen_span` | 32 | S2' `rect` | 25 | 25 | True | 1 | `unique` | True | 1.0000 | 0.1147 | yes |
| `frozen_span` | 40 | S1' `corner` | 400 | 400 | True | 2 | `complete` | True | 1.0000 | 0.9821 | yes |
| `frozen_span` | 40 | S2' `rect` | 25 | 25 | True | 0 | `none exists` | False | 0.9639 | 0.1236 | no |
| `frozen_span` | 48 | S1' `corner` | 400 | 400 | True | 2 | `complete` | True | 1.0000 | 0.9890 | — |
| `frozen_span` | 48 | S2' `rect` | 25 | 25 | True | 0 | `none exists` | False | 0.9639 | 0.1250 | — |
<!-- END:armf_table -->

Offered to §55's store:

<!-- BEGIN:armf_format_table -->
| record offered to §55's store (stage S2') | outcome | rule |
|---|---|---|
| `frozen_offsets/F1_one_width` | refused | `R1 two-width rule` |
| `frozen_offsets/F2_two_widths` | refused | `R2 vector agreement` |
| `frozen_span/F1_one_width` | refused | `R1 two-width rule` |
| `frozen_span/F2_two_widths` | **admitted** | — |
<!-- END:armf_format_table -->

### 5.1 F-a: the width barrier is real

<!-- BEGIN:refusal_table -->
| artifact built at | offered a screen at | accepted | error |
|---|---|---|---|
| 32 | 16 | False | `TypeError: input representation mismatch` |
| 32 | 32 | True | — |
| 32 | 40 | False | `TypeError: input representation mismatch` |
| 32 | 48 | False | `TypeError: input representation mismatch` |
<!-- END:refusal_table -->

---

## 6. Every pre-registered criterion, resolved

`check.py` against raw JSON (`out/check.json`):

<!-- BEGIN:criteria_table -->
| verdict | criterion (`check.py`) |
|---|---|
| PASS | (info) second-best ties the chosen program exactly where two programs conform |
| PASS | C1 instantiation reproduces the enumerated program |
| PASS | C2 the parse is component-for-component identical |
| PASS | C3 beats constant / random / whole-space mean |
| PASS | C4a frozen offsets: 0 conforming at every resolution but 32 |
| PASS | C4a frozen offsets: refused by the format at one and at two widths |
| PASS | C4b frozen span: 0 conforming above the frozen value (40, 48) |
| **FAIL** | C4b frozen span: refused by the format at one and at two widths |
| PASS | C4c arm F is bit-identical to the right schema at 32 |
| PASS | C4d the instantiation-time conformance check rejects frozen span at 40 and 48 |
| PASS | C5 section 33 does not move |
| **FAIL** | C6 saving exceeds the largest stage space (prefix wall clock) |
| **FAIL** | C6 saving exceeds the largest stage space (row evaluations) |
| PASS | F-a a resolution-32 artifact is refused elsewhere |
| PASS | F-f the derived span is width-safe |
| PASS | class admitted at 2 widths, vectors agree |
| PASS | the selection vector is identical at all five resolutions |
<!-- END:criteria_table -->

---

## 7. Verdict

**In the owner's vocabulary — schema plus specialization — within one domain,
across widths, the certified vector transfers, not only the schema.** §60 found
the reverse across domains: the schema crossed and the frozen vector did not.
Here the exhaustive search, run independently at every one of the five
resolutions, returns the *same* index vector for all three stages (§3.2), and
the vector certified at two widths rebuilds, at three unseen widths, the exact
artifact enumeration selects there — digest for digest, with the parse equal
component for component (§4). The certificate behind "the vector transfers" is
therefore the one the without arm earns at each held-out width: `unique` for S2'
and `complete` for S0 and S1', each exhaustive, each containing the stored
vector. The with arm itself earns only a conformance check. This rests on one
seed set, one palette and square screens.

**§55's mechanism carries to a real artifact.** Falsification F-b does not fire.

**The saving does not exceed the space size — §55's caveat stands, and it is
now shown to be structural rather than an artifact of a toy space.** On
programs and row-evaluations the ratio is bounded by the largest stage space by
construction: replacing one exhaustive enumeration per stage with one evaluation
saves that enumeration and nothing more. Measured as executed work through one
code path it is far smaller, because `evaluate` abandons a wrong program on an
early failing row while the right program must run every row: at S2' the walk
over the whole space costs little more than checking the one correct program
(§4.1), and over the whole pipeline the same-path wall-clock saving is a small
single-digit factor. Even the shipped `enumerate_prefix` search, which pays
roughly one full evaluation per program, stays far below the largest stage
space. **F-c fires, and more strongly than §55 stated it.** §55's same-path
ratio matched its space because every one of its programs had to be rolled out
over every episode to score a mean return; here exact-match supervision kills a
wrong program early, so the enumeration the class avoids was never expensive.
The real artifact makes the saving *smaller*, not larger (§4.3). Scale buys
nothing in ratio.

**Arm F splits, and the split is the new finding.** The wrong schema §53
anticipated — constants frozen at the tested width — is refused twice, exactly
as in §55 and §60. But a wrong schema that is only *insufficient in one
direction* passes the two-width rule: `frozen_span` is correct at and below the
width it froze, so certifying at 24 and 32 cannot see it, and §55's store admits
it. **F-d fires for this payload: R1 + R2 are insufficient on a real artifact
when the certified widths do not bracket the failure.** What stops it is the
conformance check §55 charged at every instantiation — it fails at 40 and 48 —
so the format *as used* stays sound, but only because that check is mandatory.
A store that skipped the check to claim a larger saving would ship an artifact
that scores well above the best constant and is wrong.

**§33 does not move** (F-e does not fire). The width barrier is real (F-a does
not fire). The derived span is width-safe at every resolution tested (F-f does
not fire).

**What would change these conclusions.** A certified pair that brackets the
held-out widths, or a rule requiring one certified width above every
instantiation, would close the `frozen_span` gap at the cost of more
certification; that is proposed, not tested. The flat saving would change only
with a search whose per-program cost grows faster than one full check, which
nothing here exhibits.

---

## 8. Costs, environment, reproduction

<!-- BEGIN:size_line -->
The class store is 5,003 bytes on disk and covers every width; the 15 per-width stage artifacts it stands for serialize to 227,412,631 bytes with {'s0': 5, 's1': 5, 's2': 5} distinct digests per stage. The invariant content is 21.29 bits — log2(256) + log2(400) + log2(25).
<!-- END:size_line -->

<!-- BEGIN:null_line -->
Null control (`out/null.json`): one conformance check of S1' at resolution 32 timed 7 times with nothing changed — median 5.763 s, min 5.732 s, max 6.145 s, spread 7.2% of the median.
<!-- END:null_line -->

Every job after the host crash ran under `capped.sh` — a `systemd-run --user`
scope with a memory cap, `CPUQuota=800%`, single-threaded BLAS and torch, and at
most four workers at once. Peak memory, from `/usr/bin/time -v`:

<!-- BEGIN:resource_table -->
| capped job | peak RSS (GB) | wall clock | exit |
|---|---|---|---|
| `armf` | 0.62 | 15:27.33 | 0 |
| `enum_48` | 0.64 | 1:37:21 | 0 |
| `memprobe_16` | 0.26 | 1:04.06 | 0 |
| `memprobe_24` | 0.31 | 4:03.04 | 0 |
| `null` | 0.25 | 0:44.37 | 0 |
| `refusal` | 0.34 | 0:04.39 | 0 |
| `size` | 0.63 | 0:35.75 | 0 |
| `with_16` | 0.25 | 0:16.88 | 0 |
| `with_40` | 0.44 | 6:41.78 | 0 |
| `with_48` | 0.59 | 12:13.42 | 0 |
<!-- END:resource_table -->

**Constraints honoured.** The full suite, capped and run with no enumeration in
flight, gives 336 passed and 1 failed out of 337: the failure is
`test_panel_interface.py::test_panel_episode_replays_and_restores`, the
documented worktree-only replay failure (`research/MERGE-QUEUE.md`). It was
confirmed environmental by moving this track's directory out of the tree and
re-running that single test — it still fails — then restoring the directory.
The shipped fixture, `python -m tcn train --episodes 160`, reproduces
0.248836 → 0.002231 with evaluation and frozen evaluation both at 4/4.

Reproduce, in order, each line under `capped.sh`: `run.py enumerate` at each
resolution, `run.py construct`, `run.py instantiate` at each held-out
resolution, `run.py armf`, `run.py refusal`, `run.py null`, `size.py`,
`types_probe.py`, `collision.py`, `check.py`, `report.py`, then `verify.py`.

## 9. Preserved negatives and disclosures

* **Pre-registered C4 fails as written, for `frozen_span`.** C4 required the
  format to refuse the wrong schema at one width *and* at two. It refuses
  `frozen_span` at one and admits it at two. The criterion is resolved as
  written in §6, not reworded; §7 is what it means.
* **`check.py` first resolved C3 more strictly than it was pre-registered**, also
  requiring the chosen program to beat the second-best program of its space. That
  fails by construction at S0 and S1', whose spaces each hold two conforming
  programs (S0's two are one function), and it produced a FAIL on the first
  resolution. The check was corrected to the pre-registered text — best constant,
  uniform random, whole-space mean — and the second-best column stays in the
  accuracy table, where it ties at S0 and S1' and falls below at S2'.
* **The parse is not exact at held-out resolution 40** — one rectangle on one
  screen, a parent-colour collision (§4.2). Both arms miss it identically. It is
  reported, and `palette` was not raised to make it go away.
* **"Row-evaluations" and "node units" are nominal**, not measured work (§2).
  The pre-registration listed node units as a cost currency without knowing
  `evaluate` short-circuits; the column is kept and labelled as an upper bound
  rather than dropped.
* **The whole-space baseline is scored on a fixed prefix of the held-out rows**,
  identical at every resolution, because scoring every program on every row is
  quadratic in the supervision. The chosen program is re-scored on exactly those
  rows so the comparison is like for like; held-out accuracy in every other
  column uses all held-out rows. The cap was added after a smoke run at
  resolution 16 showed the uncapped S1' baseline dominating the arm
  (`out/smoke_16.json`), and before any reported arm ran.
* **Timing was not taken on a quiet host.** Resolutions 16, 24, 32 and 40 of the
  without arm ran alone, before the host crash. The resolution-48 re-run, the
  three with-arm runs and arm F ran after it, up to four capped workers at once,
  beside an unrelated GPU process holding most of the unified memory (next
  item). Wall-clock ratios at resolution 48 therefore mix two load conditions,
  and the null control (§8) bounds only the spread of repeated identical work
  under the load it saw. Programs and row-evaluations are load-independent.
* **An unrelated process held most of the host's memory during the capped
  runs**: a GPU training job from another project directory, not started by this
  track. It was left running — it is not this track's to stop — and is flagged
  in the track's final report. Every job here peaked well under the cap (§8).
* **`enum_48` was re-run from scratch after the crash.** The pre-crash file
  stopped after S1'; nothing from it is used.
* **The schema lives outside `source_fingerprint()`'s scope**
  (`tcn/generation.py:33-42` walks only `tcn/` and `generators/`), exactly as
  §55's did. A class record here is a witness, not a self-contained artifact.
* **Staging is credited to neither arm.** Charging the without arm the joint
  space instead of the staged sum would have produced a saving in the thousands
  and a headline that "exceeds the space size". It is shown in §4.2 and not
  used.
