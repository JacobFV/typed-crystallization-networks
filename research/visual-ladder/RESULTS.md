# Visual ladder, rungs two and three — INCOMPLETE

**Status: cut short by the session ending.** The track computed its
recoverability bounds and began rung three; no rung was searched or learned.
This file was written by the supervising session from `out/bounds.json` rather
than by the track itself, so it reports measurements only and draws no
conclusions the data does not support. `bounds.py`, `common.py` and
`rung3_widgets.py` are its code.

Nothing under `tcn/` or `generators/` was modified.

## What the bounds say

Method is the one established by `research/object-identity` and
`research/gui-hierarchy`: a lookup table keyed by exactly the context in
question upper-bounds *every* function of that context, so a ceiling at the
majority baseline certifies that no program over that context can beat a
constant. `oracle_advantage` is that ceiling above majority; `transfer_advantage`
is what survives on held-out screens.

### Rung three rules are exact, not statistical — and that is the headline

Measured on 226 widgets over 12,288 positions (flat configuration):

| rule | result |
|---|---|
| `corner_owner` | **226 true positives, 0 false positives, 0 false negatives**, 12,062 true negatives |
| `smallest_container` (the parent relation) | **214 of 214 exact, 0 ties, 0 wrong** |
| `parent_pixel` | **214 of 214 exact, 0 wrong** |
| `extent` | 226 of 226 exact |

So the hierarchy rules this track set out to test are *exactly* true, not
approximately. That reproduces `research/gui-hierarchy`'s parent-relation bound
independently and extends it to corner ownership and extent.

**Three of them break under specific configurations**, which is the useful part:

- with **borders on**, `extent` falls to 12 of 226 and `corner_colour` collapses
  (214 false negatives, 250 false positives);
- with **labels on**, `corner_colour` gains 16 false positives;
- at **palette 8**, `corner_colour` loses 52 corners and `extent` 52 widgets;
- on the larger **text screen**, `corner_colour` gains 984 false positives while
  `corner_owner`, `parent_pixel` and `smallest_container` stay exact at 288 and
  276.

`corner_owner` and the parent relation are the robust pair; `corner_colour` and
`extent` are configuration-dependent and should not be relied on.

### Glyphs: the ink reduction is what makes it transfer, and the window size matters more than expected

267 held glyphs, 36 distinct codes, majority baseline 0.0449:

| window | reduction | oracle | **transfer** | unseen keys |
|---|---|---|---|---|
| 6x5 | raw | 1.0000 | **0.0562** | 96.6% |
| 6x5 | ink | 0.9625 | 0.5618 | 37.8% |
| 5x5 | ink | 0.9588 | 0.7154 | 21.7% |
| **4x5** | **ink** | 0.9588 | **0.8801** | **5.2%** |
| 8x6 | raw | 1.0000 | 0.0337 | 99.3% |

A raw window reaches a perfect oracle and transfers at chance — it is
memorisation, and the unseen-key fraction says so directly. The ink reduction
plus the *smaller* window is what transfers, at 0.8801. This sharpens
`research/gui-hierarchy`'s 0.0875-to-0.9250 finding: the reduction is necessary
but not sufficient, and a wider window is actively worse.

A linear rule over 31 features does **not** reproduce glyph codes exactly
(`exactly_linear: false`, max absolute residual 20.67 where below 0.5 would
round correctly), with 2 colliding patterns among 138.

### Widget kind: colour is worse than useless, and the strong contexts are memorisation

232 records, majority baseline 0.3836:

| context | oracle | oracle advantage | **transfer advantage** | unseen keys |
|---|---|---|---|---|
| fill colour | 0.4483 | 0.0647 | **-0.1207** | 0% |
| size (w,h) | 0.7716 | 0.3879 | **+0.1897** | 27.6% |
| fill + size | 0.9828 | 0.5991 | +0.0431 | 91.4% |
| position + size | 0.9914 | 0.6078 | +0.0517 | 84.5% |

Colour transfers *below the majority baseline* — it is not merely uninformative,
fitting it actively hurts, which confirms `research/gui-hierarchy`'s 0.4620
figure and explains why. Size is the only context with real transfer. The two
high-oracle combinations have 84-91% unseen keys on held-out screens, so their
ceilings are memorisation and should not be quoted as achievable.

## What to do next

1. **Search rung three first, not rung two.** Its rules are exact and its robust
   subset (`corner_owner`, `parent_pixel`, `smallest_container`) is already
   identified, so it is a small exhaustible search rather than an open one — and
   it is the rung that assembles a hierarchy.
2. Use the **flat configuration**; borders and small palettes break `extent` and
   `corner_colour`, so a first result should not fight them.
3. For glyphs, use the **4x5 ink window**. Do not use a raw window at any size.
4. For widget kind, use **size**, not colour, and treat the fill+size ceiling as
   unreachable.
5. The generator emits a `hierarchy` probe of shape
   `set[(id, parent, kind, x, y, w, h)]`, so a parse can be scored directly
   against it once corner ownership and the parent relation are learned.
