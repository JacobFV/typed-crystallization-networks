# Within-domain width reuse on the shipped visual parser

*(draft — arms in flight. Sections marked FILLED carry numbers recomputed from
`out/` by `check.py`; nothing appears here that a file in `out/` does not carry.)*

`PREREGISTRATION.md` was committed at `94c7cf4`, before any arm.
`tcn/` and `generators/` have no diff on this branch.

---

## 1. The width axis, justified from code — FILLED (`out/types_probe.json`, `out/probe.json`)

The shipped visual parse has two parameters that look like widths.
**Only one of them is a width in the type system**, and that is the one this
track holds out.

### 1.1 Raster resolution is a width: the port type *is* the resolution

`generators/gui/generator.py:119-121` derives `width` and `height` from the
`resolution` configuration key. The track's type constructors are then functions
of that, and every stage's input port is built from them:

* `research/visual-ladder/common.py:81-82` —
  `bytes_type(W, H, channels=3) = product(*(BYTE for _ in range(3*W*H)))`
* `research/visual-ladder/common.py:85-86` —
  `record_type(W, H) = product(IDX, bytes_type(W, H))`

Measured directly, by building S2' at seven `(resolution, span)` pairs and
counting the leaves of its `rec` port type:

| resolution | span | S2' nodes | `rec` leaf components | S2' digest |
|---|---|---|---|---|
| 16 | 16 | 298 | **769** | `c999cee93879b10f5e6a791c` |
| 24 | 24 | 442 | **1 729** | `33124a0ea737bdf447bb5fee` |
| 32 | **30** | 550 | **3 073** | `aaa20fe4984fb0c6c1f570fb` |
| 32 | **32** | 586 | **3 073** | `009b0bc80630eb8fcf02493e` |
| 32 | **48** | 874 | **3 073** | `970614c55b6e8f00421b982b` |
| 40 | 40 | 730 | **4 801** | `2d7f7a90e5bae6a50c9aa0f5` |
| 48 | 48 | 874 | **6 913** | `ea79d7a5415e530155d99cd4` |

`3·W·H + 1` leaves, exactly. This is §53's `3 × depth` on a real artifact: a
program crystallized at one resolution cannot accept another, and §33's own
parse is a fixed-resolution artifact for that reason.

The width also drives the *constants*, which §53's channel did not:
`rung3_widgets.py:172-181` `offset_pool(W) = (6, 3W+3, 3, 3W, 9)` — two of the
five candidate spatial steps are width-derived, and the *correct* vertical step
is `3W`, whose value changes with resolution while its index does not.
`rung3_root.py:59` `stride = 3W`; `rung3_root.py:107` `last = 3·W·H − 3`.

### 1.2 `rect_scaffold`'s span is *not* a width — it appears in no type

The three rows at `resolution=32` above are the measurement: spans 30, 32 and 48
give **three different programs** (three digests, 550 / 586 / 874 nodes) and
**one identical port type** (3 073 leaves). §41 swept exactly this parameter and
certified `none exists` at spans 4–29 and `unique` at 30 on `resolution=32`.

So `span` is a *derived schema parameter*, not an axis: the schema defaults it to
`span = max(W, H)` (`rung3_widgets.py:234`), and that derivation is structurally
sufficient — a widget with corner `x ≥ 1` and width `w` needs `x + w ≤ W`, so
`w ≤ W − 1 ≤ span`, and the count-plus-one the scaffold computes represents
extents up to exactly `span`.

Because §55's mechanism is about *types*, the axis is raster resolution. `span`
is carried inside the schema — and it supplies arm F's second payload, because
freezing it is precisely the mistake a single-width certificate cannot catch.

### 1.3 The widths, and the achieved difficulty at each

Only `resolution` moves; every other key of `FLAT`
(`widgets=20, nesting=5, min_size=4, palette=32, horizon=2`) is held fixed.
Achieved, not requested (`out/probe.json`, and `achieved` in each `enum_*.json`):

| resolution | role | observation components | achieved widgets/screen | max widget extent | span |
|---|---|---|---|---|---|
| 16 | held out | 768 | 7–10 | *(filled)* | 16 |
| 24 | **certified** | 1 728 | 10–15 | *(filled)* | 24 |
| 32 | **certified** | 3 072 | 20 | *(filled)* | 32 |
| 40 | held out | 4 800 | 20 | *(filled)* | 40 |
| 48 | held out | 6 912 | 20 | *(filled)* | 48 |

The achieved widget count **falls below the request at 16 and 24** — the layout
cannot place 20 widgets on a 16×16 screen — so those two resolutions are an
easier task, and that is stated rather than smoothed over. `resolution=12`
achieves 3 widgets and was excluded before the pre-registration was written.

---

---

## 2. What is measured, and how the two arms are made comparable

The pipeline is the one §33 shipped, staged exactly as `rung3_root.main` stages
it: **S0** `same(a, b, obs)` is exhausted and hardened into a module; **S1'**
`corner(rec)` calls that module twice at searched offsets, masked so an
off-screen neighbour reads as different; **S2'** `rect(rec)` calls it
`2·(span−1)` times and clamps its parent pixel at the origin. Supervision comes
only from `record.actor_view().observations`; the `hierarchy` and `owner` probes
build targets and nothing else.

**The class store is §55's, unchanged** — `research/class-identity/classes.py`,
rules R1 (two widths), R2 (vector agreement), R3 (instantiation digest).
`tcn/library.py` is untouched. The schema is a *reference* to the width-
parametric constructors already in the repository; §55 established that a class
can hold a schema reference but not a schema, and that remains true here.

**Both arms go through the same code path.** The without arm is an exhaustive
`itertools.product` walk calling `tcn.search.evaluate` once per program; the with
arm is *one* call to the same `tcn.search.evaluate` on the same rows. So a ratio
of wall clock is a ratio of identical work rather than of two interpreters. The
shipped `enumerate_prefix` search is timed separately and reported beside it,
because that is what the repository actually pays.

**Four currencies**, because §55's falsification lives in the choice:
programs evaluated, row-evaluations (programs × supervision rows — the analogue
of §55's episodes), node units (programs × rows × nodes), and wall clock with a
null control for the §59 instrument floor.

**The staged space is the honest "without".** The joint space is
`256 × 400 × 25 = 2 560 000`, but the staged decomposition is what
`rung3_root.py` runs, so the without arm is charged `256 + 400 + 25 = 681`
programs. Crediting the class with the staging would inflate the result by
almost four orders of magnitude and it is not credited.

---

## 3. Certificates at the training widths — pending

## 4. The with/without table at the held-out widths — pending

## 5. Arm F, the soundness test — pending

## 6. Every pre-registered criterion, resolved — pending

## 7. Verdict — pending
