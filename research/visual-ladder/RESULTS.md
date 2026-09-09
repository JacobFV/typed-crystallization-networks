# Visual ladder, rungs two and three — the first screenshot-to-hierarchy parse

Research track `visual-ladder`, continuing `research/gui-hierarchy` directly.

**STATUS: rung three is now run.**  Section 1 is the earlier session's bounds and
is unchanged; every number in it is **inherited** from `out/bounds.json`.
Sections 2, 3 and 3.1 are **newly measured** in this session from
`out/rung3.json` and `out/rung3_parse.json`, and every table there is marked
NEW.  Rung two was not attempted and no number for it is claimed.

Produced with
`/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python`.
**Nothing under `tcn/` is modified. Nothing under `generators/` is modified.**
No generator extension was needed or made.

---

## 0. Verdict, stated plainly

**A screenshot is parsed into a widget hierarchy.**  On 12 held-out flat screens
the four-node caller returns **215 rectangles for 215 non-root widgets, every one
exactly right, on every screen, with no spurious rectangle**, and **173 of 215
parent links correct (0.805)** against a `parent = root` baseline of **0.140**.
Three of the 12 screens are recovered as an *exactly* correct tree.

**Two concessions, in the headline sentence and not in a footnote.**  (i) The
comparison is on **rectangles, not ids** — a permutation of the generator's
widget list leaves the raster bit-identical, so no program over the screen can
produce the generator's numbering, and `score_tree` matches a predicted widget to
the probe by its `(x, y, w, h)`.  (ii) The **root is supplied**, as "the widget
whose rectangle is the whole screen": the corner predicate reads a left and an
upper neighbour, so position (0,0) is outside the position set by construction
(H6), and 12 of the 227 widgets in the probe are therefore never predicted.

**All three searches exhausted and all three carry a certificate.**  S0 256/256,
S1 400/400, S2 25/25 — the last returning **exactly one** conforming program,
certificate `unique`.  Every stage's validation-filtered survivors are at
**held-out max error 0.0**.

**The handoff's two named risks were both real, and both were the same fault.**
Fill colour is *not* injective per widget at `palette 32` — `bounds.json`'s own
`colour_injective` field says 1 of 12 episodes, 192 distinct colours over 226
widgets — and the draft leant on injectivity twice without consulting it.  S2's
extent summed same-colour bits over the whole row and overcounted; S0 supervised
a colour predicate with `owner`-equality at *arbitrary* address pairs and was
unsatisfiable.  Both are fixed, both fixes are measured, and the unsatisfiable
variant is kept as a reported negative control (0 conforming of 256, exhausted).
The third named risk — `sum` over 31 `int[16]` terms against `overflow='error'`
— did **not** materialise.

**What is still missing.**  The parent link is resolved through the widget's
*colour* key, and colour collisions are exactly where it fails: on the 3 of 12
screens where the predicted keys are distinct the links are **46/46**; on the 9
where two widgets share a fill colour they are **127/169**.  That is an
information limit of the two-pixel parent route at this palette, not a search
failure, and section 5 says what to do about it.

---

## 1. Bounds, computed before any search

`bounds.py` -> `out/bounds.json`. 12 training and 12 held-out episodes per
setting. Two kinds of bound are reported and they answer different questions:

* an **informational ceiling** (`lookup_bound`, the object-identity method) — a
  table keyed by exactly the context is an upper bound on every function of that
  context, so a ceiling at the majority baseline is a certificate that no program
  over that context can beat a constant;
* an **exactness check on a stated rule** — does *this* rule reproduce the target
  at every instance. Rung three's candidate rules are not statistical, so this is
  the form its bound has to take.

### 1.1 Rung two — the glyph code from an anchored ink window

Screen `TEXT`: `resolution 64, widgets 24, nesting 6, min_size 6, palette 48,
labels, label_size 8`. Anchor = the glyph's own bounding-box top-left (taken
from the probe *for the bound only*; whether the anchor is findable is bounded
separately in 1.2). Majority baseline 0.0449 throughout.

| window | reduction | oracle ceiling | transfer | unseen keys |
|---|---|---|---|---|
| 6x5 | ink | 0.9625 | 0.5618 | 0.378 |
| 6x5 | raw bytes | 1.0000 | 0.0562 | 0.966 |
| 5x5 | ink | 0.9588 | 0.7154 | 0.217 |
| 5x5 | raw bytes | 0.9925 | 0.0824 | 0.936 |
| 4x5 | ink | 0.9588 | **0.8801** | 0.052 |
| 4x5 | raw bytes | 0.9925 | 0.1011 | 0.914 |
| 8x6 | ink | 0.9700 | 0.4607 | 0.498 |
| 3x5 | ink | 0.8801 | 0.7978 | 0.022 |

**The ink reduction is confirmed as the transfer path**, exactly as
`research/gui-hierarchy` R2 measured: raw bytes reach a *higher* oracle (the fill
colour behind the glyph makes the key nearly unique) and transfer at 0.06-0.10
with 91-97% of keys never seen, while the ink-binarised window transfers at up to
0.8801 with 5% unseen. The oracle/transfer split is the same geometry trap the
earlier track named.

**A new number the earlier track did not have: the window is contaminated by the
next character.** Transfer falls monotonically as the window widens (0.8801 at
4x5, 0.5618 at 6x5, 0.4607 at 8x6) while the oracle rises, because a wider window
reaches into the following glyph of the same four-character label and the key
becomes label-specific rather than character-specific. Setting `label_length=1`
(an existing generator dial, no extension needed) removes it: 305 train / 206
held-out glyphs, 36 codes, **oracle = transfer = 0.9223** at a 6x5 window with
**0.000 unseen keys**.

Anchoring the window at the *widget's* text origin `(rect.x+1, rect.y+1)` instead
of at the glyph's own bounding box — which is what rung three can supply and the
probe does not have to — gives, at `label_length=1`:

| screen | window | oracle | transfer | unseen | distinct patterns | colliding |
|---|---|---|---|---|---|---|
| TEXT, label_length 1 | 8x8 | 0.9417 | 0.9417 | 0.000 | 38 | 2 |
| TEXT, min_size 8, label_length 1 | 8x8 | 0.9713 | 0.9330 | 0.000 | 34 | 2 |
| TEXT, min_size 10, widgets 16, label_length 1 | 8x8 | **0.9730** | 0.9568 | 0.000 | 34 | 2 |
| TEXT, label_length 1 | 7x7 | 0.8544 | 0.8252 | 0.000 | 29 | 6 |

**The ceiling is 0.973 and the residue is structural, not statistical.** The
colliding classes at `label_size 8` are codes `{8, 9}` and `{5, 11}` — `i`/`j`
and `f`/`l` — which render to bit-identical masks after the bounding box is
trimmed. Two characters that put the same bits on the screen cannot be separated
by any program over the screen, at any context size. That is the same shape of
statement as `research/object-identity`'s permutation certificate, obtained a
different way, and it should be re-checked at a larger `label_size` before it is
reported as final.

### 1.2 Rung two — is the anchor findable?

| context | majority | oracle | transfer |
|---|---|---|---|
| 3x3 ink window, is this an ink pixel a glyph's bbox top-left | 0.9240 | 0.9599 | 0.9590 |
| 5x5 ink window, same | 0.9240 | 0.9820 | 0.9608 |

**Not exactly.** A local ink window does not determine the glyph anchor, so a
rung two built on a *self-detected* glyph anchor is capped at 0.98 before any
program is written. This is why the intended staging inverts the brief's order:
the anchor should come from rung **three**'s widget corner, which section 1.4
measures as exact, and the glyph window should be taken at the widget's text
origin. Rung two is therefore downstream of rung three, not upstream of it.

### 1.3 Rung two — two program families excluded before searching

**The continuous family is excluded.** With `interpret` available, the natural
rung-two program is `code = sum_k w_k * decode(interpret(ink_k))` — the byte-
numeric track's convolution shape with the ink bits as taps. Solving that family
exactly outside the substrate (least squares over the distinct patterns; the
recovered weights are used nowhere) gives:

| screen / window | distinct patterns | features | matrix rank | max abs residual | exactly linear |
|---|---|---|---|---|---|
| TEXT (label_length 4), glyph-anchored 6x5 | 138 | 31 | 31 | 20.67 | no |
| TEXT label_length 1, glyph-anchored 6x5 | 33 | 31 | 23 | 14.2 | no |
| TEXT label_length 1, widget-anchored 8x8 | 38 | 65 | 26 | 7.34 | no |
| TEXT min_size 10, label_length 1, widget-anchored 8x8 | 34 | 65 | 23 | 7.03 | no |

The ink matrix is heavily rank-deficient — 23 of 65 columns are independent
against 34 distinct patterns — so **no weight vector reproduces the code**, and
the residual of 7.03 is far above the 0.5 that would round to the right integer.
A gradient arm over this family cannot succeed, and running one would measure the
optimiser rather than the question. That is a bound, and it is the reason no
gradient arm was run.

**The small Boolean families are excluded.** The code has six bits; the algebra
can enumerate one ink bit, or a two-input truth table over two ink bits, per bit.
Best achievable accuracy over the *whole family*, on the default TEXT screen:

| code bit | majority | best 1-address | best 2-address |
|---|---|---|---|
| 0 | 0.5175 | 0.7393 | 0.8016 |
| 1 | 0.5642 | 0.6615 | 0.7315 |
| 2 | 0.6109 | 0.7510 | 0.7860 |
| 3 | 0.5642 | 0.6770 | 0.7860 |
| 4 | 0.5992 | 0.7743 | 0.8132 |
| 5 | 0.8872 | 0.8366 | 0.9416 |

Every entry is below 1.0, so both families are excluded exhaustively rather than
by budget. Note bit 5: the best single-address predictor (0.8366) is **worse than
the majority baseline** (0.8872), which is the clearest available warning that a
conforming-count result on this target would be reporting a constant.

**What that leaves, and it is the recommendation.** The code is an arbitrary
index into `ALPHABET`, and the algebra has no lookup operator over a 30-to-64-bit
key. What *is* expressible and exact is the object-identity survivor: the packed
ink window is an **episode-independent character key** (`pack` on a tuple of ink
bits), and `same_character(a, b)` is `eq` on two such keys — a conjunction of ink
equalities, well inside the algebra. Rung two should be re-aimed at that, with
the 0.973 ceiling and the `i`/`j`, `f`/`l` collisions reported as the certificate
of what the raster does not carry. **This was not run.**

### 1.4 Rung three — the rules, checked at every instance

Screen `FLAT`: `resolution 32, widgets 20, nesting 5, min_size 4, palette 32`,
achieved mean 18.8 widgets, 3072 observation bytes. 12 episodes.

| screen | corner rule tp / fp / fn | extent exact | parent-pixel exact | smallest-container exact |
|---|---|---|---|---|
| **flat** | **226 / 0 / 0** | **226/226** | **214/214** | 214/214 |
| borders | 12 / 250 / 214 | 12/226 | 214/214 | 214/214 |
| labels | 226 / 16 / 0 | 226/226 | 214/214 | 214/214 |
| palette 8 | 174 / 0 / 52 | 174/226 | 214/214 | 214/214 |
| TEXT screen | 288 / 984 / 0 | 288/288 | 276/276 | 276/276 |
| TEXT screen, labels off | 288 / 0 / 0 | 288/288 | 276/276 | 276/276 |

The three rules are:

* **corner** — a position is its widget's top-left corner **iff** the pixel to
  its left and the pixel above it both differ in colour. Exact on the flat
  screen: 226 corners found, no false positive, no false negative. This is two
  calls to a rung-one-style same-colour module at two offsets.
* **extent** — at a corner, the count of same-colour pixels along the row is the
  widget's width and along the column its height. Exact at 226/226. It holds
  because `layout` insets every child by `margin`, so a widget's own top row and
  left column are never covered, and because colour is injective per widget, so
  no other pixel of that row or column carries its colour. **It needs no
  sequential run-length accumulator** — a masked count suffices, which is what
  makes it expressible as a fixed-depth graph.
* **parent** — the pixel immediately left of a widget's top-left corner belongs
  to its parent. Exact at 214/214 non-root widgets. This is a *two-pixel* route
  to the parent relation and it is cheaper than `research/gui-hierarchy`'s R4
  smallest-containing-rectangle rule, which is re-confirmed here at 214/214 with
  zero ties on the same screens.

**The failures are the certificates.** With `borders` the corner rule collapses
to 12 true positives against 250 false ones, exactly as R1b predicted — a
widget's own border-to-fill transition presents the same evidence as an ownership
boundary. At `palette 8` the corner rule loses 52 of 226 corners, because colour
is no longer injective. With `labels` it gains 16 false positives on the flat
screen and 984 on the text screen, because glyph ink creates spurious colour
corners. **So rung two and rung three cannot share a screen as written**, and the
repair is stated but unmeasured: a corner predicate conjoined with "this pixel is
not ink" should recover the labelled case, since ink is `(0,0,0)` and no palette
entry is.

**CORRECTION, newly measured this session.**  Section 1.4's justification for the
extent rule says "no other pixel of that row or column carries its colour".  That
is false, and `bounds.json` already contained the refutation: `colour_injective`
on the flat screen is **1 of 12 episodes**, 192 distinct colours over 226
widgets.  `extent_rule` checks the *contiguous run* and is exact at 226/226; the
scaffold as drafted summed a masked same-colour *count* over the whole row, which
is a different rule and overcounts whenever a same-coloured sibling sits further
along the row.  Measured on 12 flat episodes: `owner`-equality and colour-equality
disagree on **1.41%** of 48,000 arbitrary address pairs and **0.87%** of 48,000
same-row pairs, but on **0.00%** of 46,528 four-neighbour pairs — which is the
same fact `corner_colour` states as 226/0/0.  Both faults below follow from this
one line.

### 1.5 Rung three — widget `kind`, re-confirmed as blocked

| context | majority | oracle | transfer |
|---|---|---|---|
| fill colour | 0.3836 | 0.4483 | 0.2629 |
| size (w, h) | 0.3836 | 0.7716 | 0.5733 |
| fill + size | 0.3836 | 0.9828 | 0.4267 |
| position + size | 0.3836 | 0.9914 | 0.4353 |

Colour does not determine kind (0.4483 against a 0.3836 majority), reproducing
the earlier track's 0.4620/0.2982 on a different screen. The high-oracle rows are
memorisation: `fill + size` reaches 0.9828 fitted on the evaluation set and
transfers at 0.4267, because that key is nearly unique per widget. **`kind` was
therefore not chosen as the rung-three predicate**; extent and containment were,
and section 1.4 is why.

---

## 2. What was run — NEW, all of this section

`rung3_widgets.py --train 6 --validation 3 --held 6 --pairs 48
--corner-per-image 120 --parse 6 --free` -> `out/rung3.json`;
`parse_report.py --screens 12` -> `out/rung3_parse.json`.  Screen `FLAT`
(`resolution 32, widgets 20, nesting 5, min_size 4, palette 32`), achieved mean
19.5 widgets, 3,072 observation bytes, 961 interior positions.  The flat
configuration was used deliberately: section 1.4 already measures that borders
take `extent` to 12 of 226 and that a small palette collapses `corner_colour`, so
a first result does not fight them.

### 2.1 Three faults found by running it

| # | fault | evidence | fix |
|---|---|---|---|
| F1 | S2's extent was a masked **count**, not a run | 8 of 57 widgets over 3 training episodes overshot by 1-2 (e.g. got `w,h = 7,5` where the probe says `5,4`) | each term conjoined with every earlier one — a prefix conjunction, still a fixed-depth feedforward graph, no accumulator and no recurrence |
| F2 | S0 supervised a colour predicate by `owner` at **arbitrary** pairs | **0 conforming of 256, exhausted** — the target is not a function of the context; 1.41% of arbitrary pairs disagree | pairs drawn at the four spatial neighbours, where the disagreement is 0.00% over 46,528 draws |
| F3 | a stage with 0 conforming crashed with `KeyError: 'chosen'` | S0's first run | explicit `SystemExit` naming the stage |

The handoff named three likely faults in S2.  The `min`/`last` address clamp and
the `lt` in-bounds mask were both **correct as written**, and `sum` over 31
`int[16]` terms against `overflow='error'` never tripped.  The fault was the one
place the draft asserted a rendering property instead of reading the bound it had
already computed.

### 2.2 The searches — NEW

Every sweep is `enumerate_prefix` through `common.sweep`, and the conforming set
is walked a second time by `all_conforming` as a cross-check; the two agree on
the count in all four arms.

| stage | space | evaluated | exhausted | conforming | distinct functions | certificate | validation survivors | held-out max error | held-out accuracy | random control | sweep s | node evals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 `same(a,b,obs)` | 256 | 256 | yes | **2** | **1** | complete | 2 | **0.0** | 1.000 | 2/400 | 0.7 | 10,064 |
| S0 ablation `--free` (H2) | 16,384 | 16,384 | yes | **2** | **1** | complete | 2 | **0.0** | 1.000 | 0/400 | 1.2 | 361,862 |
| S0 negative control (draft supervision) | 256 | 256 | yes | **0** | — | complete | — | — | — | 0/400 | 0.7 | 9,120 |
| S1 `corner(rec)` | 400 | 400 | yes | **2** | not induced | complete | 2 | **0.0** | 1.000 | 1/400 | 97.6 | 64,866 |
| S2 `rect(rec)` | 25 | 25 | yes | **1** | not induced | **unique** | 1 | **0.0** | 1.000 | 4/50 | 472.2 | 1,605,195 |

**Non-uniqueness is the norm and it is reported as such.**  S0's two conforming
programs are two spellings of one Boolean function — `induced_same` collapses
them and returns **1 distinct function**, `r AND g AND b`, reached as
`NOT NAND(r,g) AND b` by the lexicographic pick.  S1's two survivors are literally the
swapped pair — `{back_a: 2, back_b: 3, corner: 1}` and
`{back_a: 3, back_b: 2, corner: 1}` — the same predicate `NOT same_a AND
NOT same_b` with its two arguments exchanged; `induced` was not written for S1,
so "one predicate" is read off those selections rather than computed, and it is
the one uniqueness claim here that is an argument and not a number.  **S2 is
unique outright**: 1 of 25, certificate `unique`.

**The requested offsets were discovered, not supplied.**  The pool is
`(6, 99, 3, 96, 9)` and the answers are at indices 2 and 3, not 0.  S1 chose
`[3, 96]` — the left and the upper neighbour — and S2 chose `[3, 96]` for the row
step and the column step.  The `--free` ablation additionally searches S0's
operand binding against three constant-byte distractors, with the correct binding
at index 2 of 4.  Both survivors select `cmp_r = cmp_g = cmp_b = 2` — the
neighbour comparison, not a distractor constant — and denote the identical single
function: **H2 ablates away.**

**Supervision densities**, so the conforming counts can be read against them:
S0's neighbour pairs are 64.2% positive (the draft's arbitrary pairs were 35.4%),
S1's positions are **1.94%** positive — 720 sampled positions per split, about 14
true corners — and S2 has one example per non-root widget, 111 training and 103
held out.

---

## 3. The parse — NEW

`assembly` is four caller nodes: `insert`, `pair`, `filter`, `map`.  The `filter`
runs the 35-cost corner module at all 961 interior positions; the `map` runs the
1,453-cost rectangle module only at the positions it kept.  Wall clock 13.8-15.8 s
per screen.

12 held-out screens, seeds 200-211, split `test`, `out/rung3_parse.json`:

| quantity | program | baseline |
|---|---|---|
| non-root widgets in the probe | 215 | — |
| rectangles predicted | **215** | 0 (empty parse) |
| rectangles exactly right | **215 / 215** | 0 |
| screens with the rectangle set exactly right (no miss, no spurious) | **12 / 12** | 0 |
| parent links correct | **173 / 215 = 0.805** | `parent = root`: 30 / 215 = **0.140** |
| screens with an exactly correct tree | **3 / 12** | 0 |

For the corner predicate the majority baseline is "no position is a corner", at
0.980 accuracy and **zero** widgets recovered; the program is at 1.000 on the
held-out corner split, which is the number that matters because the parse needs
every corner and no other.

**Where the 42 wrong links are.**  All of them are colour-key collisions.  The
program emits `parent_key` as the packed RGB of the pixel to the left of the
corner — exact at 214/214 as a *relation* (section 1.4, inherited) — and
`score_tree` resolves that key against the widgets it parsed.  When two widgets
on a screen share a fill colour the key is ambiguous:

| screens | key collisions | parent links |
|---|---|---|
| 3 of 12 (seeds 202, 205, 209) | 0 | **46 / 46 = 1.000** |
| 9 of 12 | 25 in total | 127 / 169 = 0.751 |

So the parse is exact wherever colour is injective, and the residue is the
`palette 32` collision rate, not a search failure.  Section 5.1 says what fixes
it.

---

## 3.1 How far up the ladder — NEW

**Rung three is now the highest rung with a measured result, and the ladder
reaches a parse.**  Rung one (`research/gui-hierarchy`) is a per-position
boundary predicate; rung three is a whole-screen relation of the probe's own
type, produced by four caller nodes over three frozen modules, at held-out
rectangle recall 1.000.  Rung two was not attempted this session and section 1's
bounds for it stand unchanged and unsearched.

The two staging conclusions the earlier session drew survive: rung two's target
should be the character key rather than the code index, and rung two is
downstream of rung three because the glyph anchor is not findable from a local
window (0.982 ceiling, inherited) while the widget corner that supplies it is now
not merely exact as a rule but **learned, exhausted and applied**.

---

## 4. Files

Everything new is under `research/visual-ladder/`:

| file | state |
|---|---|
| `common.py` | unchanged this session |
| `bounds.py` -> `out/bounds.json` | unchanged this session; section 1 is inherited from it |
| `rung3_widgets.py` -> `out/rung3.json` | **run**; three fixes (2.1) and one added negative control |
| `parse_report.py` -> `out/rung3_parse.json` | **new**, **run**; re-runs S3 from the frozen selections in `out/rung3.json` on 12 held-out screens with the baselines beside it, so section 3 is regenerated rather than transcribed |
| rung two script | still **does not exist** |
| `report.py` | still **does not exist**; sections 2 and 3 are transcribed from the two JSONs by hand and section 1 from `bounds.json` |

**Nothing outside `research/visual-ladder/` was changed.**  `tcn/` is untouched
and `generators/gui/` is untouched.  There is therefore no default-stream
verification to report.  `enumerate_prefix`, `positional_scaffold`,
`register_module` and `filter`/`map`/`pair`/`insert` were all used as shipped; no
core change is proposed.

---

## 5. What to do next, in priority order

1. **Break the parent-key collision.**  The parent is currently identified by the
   parent pixel's *colour*, which is ambiguous at `palette 32` and costs all 42
   wrong links.  Two routes are already measured as exact in section 1.4 and both
   are cheap: emit the parent's **corner position** rather than its colour (the
   left pixel's colour identifies the parent *rectangle* once the rectangle set
   is known, which the parse already has), or fall back to
   `research/gui-hierarchy`'s R4 smallest-containing-rectangle rule, 214/214 with
   zero ties.  Either should take the tree from 3 of 12 to 12 of 12; neither has
   been run.
2. **Recover the root.**  H6 excludes position (0,0) by construction, so 12 of 227
   widgets are never predicted and the root is supplied to the scorer.  A corner
   predicate that treats an off-screen neighbour as "different" would include it;
   the cost is one clamp and it was not attempted.
3. **Write an `induced` for S1 and S2** so the distinct-function count is measured
   rather than argued.  S1's "2 conforming, 1 predicate" is the only claim in
   section 2.2 that rests on reading the selections.
4. **Write rung two against the character key**, staged on S1's corner so the
   glyph window is anchored at the widget text origin, with `label_length=1`, and
   report the 0.973 ceiling and the `i`/`j`, `f`/`l` collisions as the
   certificate.  Unchanged from the earlier handoff and still unrun.
5. **Measure the corner-plus-not-ink predicate** on a labelled screen, so rung two
   and rung three can share one.  Unchanged and still unrun.
6. **Add `report.py`.**  Two JSONs now exist and section 3 is one function away
   from being generated.
7. Do **not** run a gradient arm on the glyph code (section 1.3), and note that no
   gradient arm was needed anywhere in rung three: all four discrete searches
   exhausted, so there was nothing for an optimiser to be measured against.

**Limitations.**  Section 1's bounds are on 12 training and 12 held-out episodes
of one screen configuration.  Section 2's searches are on 6 training, 3
validation and 6 held-out episodes; the held-out max errors are 0.0 but a
third-decimal claim is not supported by that sample.  Section 3 is 12 held-out
screens and 215 widgets.  The exactness statements are exact **on these samples**
and are not proofs about the generator.  Timings are single-run wall clock on a
shared host and are upper bounds; `node_evaluations` is the load-independent
cost.
