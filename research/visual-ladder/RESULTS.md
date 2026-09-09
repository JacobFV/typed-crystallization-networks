# Visual ladder, rungs two and three — INCOMPLETE HANDOFF

Research track `visual-ladder`, continuing `research/gui-hierarchy` directly.

**STATUS: incomplete. The session ended during setup.** Everything in section 1
is measured and reproducible from `out/bounds.json`. Everything in section 2 is
*written but never executed* — no search was run, so **no rung above rung one is
claimed here and no conforming count, uniqueness statement or held-out number
exists for rung two or rung three.** Section 3 says what the bounds imply and
section 5 says what to do first.

Produced with
`/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python`.
**Nothing under `tcn/` is modified. Nothing under `generators/` is modified.**
No generator extension was needed or made, so the question of the default
observation stream changing does not arise (see section 4).

---

## 0. Verdict, stated plainly

* **Rung two (glyphs) was not reached.** The bounds for it were computed and
  they are the useful output: they **exclude two of the three program families
  that could have produced a character code**, before any search was run. That
  is the "say so instead of searching" the brief asks for, and it redirects the
  rung rather than merely failing it.
* **Rung three (widgets) was not reached as a search**, but its *rules* were
  checked exactly and they hold at every instance on the flat screen. The
  scaffold that would learn them is written and unrun.
* **A screenshot-to-hierarchy parse is in reach and the blocking work is now
  small** — three exhaustible searches and one four-node caller, all drafted in
  `rung3_widgets.py`. The reason for the confidence is section 1.4: the three
  rules the parse needs are exact at 226/226, 226/226 and 214/214 on the
  measured screen, with the failures isolated to configurations that are already
  known to destroy the information (`borders`, small `palette`).

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

## 2. What is written and unrun

`rung3_widgets.py` implements the four stages below. **It has never been
executed.** Its scaffolds type-check in isolation (the module composition in
particular was prototyped: a frozen module registered with
`Registry.register_module` is callable as an ordinary three-source operator node,
verified end to end) but no search has been run, so **every conforming count,
uniqueness statement and held-out number for rung three is absent.**

| stage | what it searches | declared space | certificate expected |
|---|---|---|---|
| S0 `same(a, b, obs) -> bool` | two Boolean combinators over three channel comparisons | 256 | exhaustible |
| S0 ablation (`--free`) | plus each comparison's operand binding, against three constant-byte distractors | 16,384 | exhaustible |
| S1 `corner(rec) -> bool` | two offsets from a five-step pool, one truth table | 400 | exhaustible |
| S2 `rect(rec) -> (x,y,w,h,key,parent_key)` | the row step and the column step, from the same pool | 25 | exhaustible |
| S3 the parse | nothing; four caller nodes | 1 | n/a |

S0 externalises rung one's offset, which is the change that makes the module
reusable: `research/gui-hierarchy` baked offset 3 into the module, so it could
only ever answer about the right-hand neighbour. With two free address arguments
the same 256-program space serves the corner test, both extent counts and the
parent lookup. It is supervised at *arbitrary* raster pairs by
`owner(a) == owner(b)`, which is denser than the neighbour-only rung and was
expected to collapse the two-spelling tie the earlier track reported.

S3 is `insert` / `pair` / `filter` / `map` — `tcn.scaffold.positional_scaffold`
plus one `filter` node, so the expensive rectangle module runs at the ~20 corners
rather than at all 961 interior positions. Its output type is
`set[(x, y, w, h, key, parent_key)]`, which is the `hierarchy` probe's relation
up to the episode's colour relabelling. `score_tree` compares on **rectangles**,
not ids, because a permutation of the widget list leaves the raster identical —
the object-identity result applies here unchanged.

---

## 3. How far up the ladder, and is a parse in reach

**How far up: rung one remains the highest rung with a measured result.** This
track added no searched rung. What it added is bounds, and two of them change
what the next rungs should be:

1. rung two's target should be the **character key and the same-character
   relation**, not the generator's code index, because the code needs a lookup
   the algebra cannot write and both expressible families are excluded above;
2. rung two is **downstream of rung three**, not upstream, because the glyph
   anchor is not findable from a local window (0.982 ceiling) while the widget
   corner that supplies it is exact.

**Is a screenshot-to-hierarchy parse in reach: yes, and the remaining work is
three exhaustible searches.** The three rules the parse is made of are exact at
226/226, 226/226 and 214/214 on the flat screen — not "high accuracy", exact at
every instance — and each is expressible as a fixed-depth graph over operators
already on main. The parse it yields would be up to colour relabelling and would
exclude the root, which the corner predicate cannot see. **That claim is not
made here; it is a prediction from the bounds, and nothing was searched.**

---

## 4. Files

Everything new is under `research/visual-ladder/`:

| file | state |
|---|---|
| `common.py` | written, exercised by `bounds.py` |
| `bounds.py` -> `out/bounds.json` | **run**; every number in section 1 comes from it, except the `label_length=1` and widget-anchored rows of 1.1/1.3, which were measured ad hoc from `bounds.py`'s helpers and are **not** yet in a committed JSON |
| `rung3_widgets.py` | written, **never run** |
| rung two script | **does not exist** |
| `report.py` | **does not exist** — this document is written by hand, which is a departure from `research/gui-hierarchy`'s practice and should be repaired |

**Nothing outside `research/visual-ladder/` was changed.** `tcn/` is untouched
and `generators/gui/` is untouched — no generator extension was made, because
`label_length` and `min_size` are existing dials and they were sufficient. There
is therefore no default-stream verification to report, and none was run.

No core changes are proposed. `enumerate_prefix`, `interpret`, the severing fix
and `positional_scaffold` were all used as shipped.

---

## 5. What to do first, in priority order

1. **Run `rung3_widgets.py`.** `--train 6 --held 6 --parse 3`, then `--free` for
   the S0 ablation. It is the only thing between this track and a first
   screenshot-to-hierarchy parse. Expect to debug S2 first: it is by far the
   largest scaffold (roughly 450 nodes at `resolution 32`) and it is the one
   piece never executed even once. Check the `min`/`last` address clamp and the
   `lt` in-bounds mask before anything else, and confirm `sum` over 31 `int[16]`
   terms does not trip the `overflow='error'` encoding.
2. **Report S3 against the probe honestly.** `score_tree` compares rectangles and
   resolves parents through the program's own colour keys; the root is supplied
   as "the widget whose rectangle is the whole screen". Both of those are
   concessions and both need to be stated in the headline sentence, not in a
   footnote.
3. **Write rung two against the character key**, not the code. Stage it on S1's
   corner so the window is anchored at the widget text origin, use
   `label_length=1`, and report the 0.973 ceiling and the `i`/`j`, `f`/`l`
   collisions as the certificate. Re-check the collisions at a larger
   `label_size` before calling them structural.
4. **Measure the corner-plus-not-ink predicate** on a labelled screen. Rung two
   and rung three currently cannot share a screen (984 false corners on the text
   screen); if that predicate is exact, they can, and the ladder becomes one
   pipeline rather than two.
5. **Add `report.py`** and regenerate this document from `out/*.json`, so no
   number in it is transcribed by hand. Several currently are, and they are
   flagged in section 4.
6. Do **not** run a gradient arm on the glyph code. Section 1.3 excludes the
   family before the optimiser is involved; a seed count there would measure
   nothing.

**Limitations of what is here.** All bounds are on 12 training and 12 held-out
episodes, enough to separate 1.0000 from 0.85 but not to resolve a third decimal.
The exactness checks in 1.4 are on 226 corners and 214 non-root widgets across 12
episodes of one screen configuration; they are exact on that sample and are not a
proof about the generator. No timing is reported because no search was run.
