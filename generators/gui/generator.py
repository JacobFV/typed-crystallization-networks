"""Synthetic GUI screens: raw pixels to the agent, the component tree as a probe.

Contract.  An ordinary peer generator under `ARCHITECTURE.md` section 6 --
`initialize` / `advance` / `observe`, seeded from the episode `Address`, no host
clock, all content synthetic, complete state in the snapshot, replayable.  The
trainer does not need to know it exists.

Observation (what the agent may see)
    pixels        the raster, as `image_value`: (height, width, channels, bytes)

Probes (supervision only, never a model input)
    hierarchy     set[(id, parent, kind, x, y, w, h)] at a declared capacity
    owner         per-pixel id of the deepest widget covering that pixel
    glyphs        set[(id, slot, code, x, y, w, h)], emitted only when `labels`
                  is configured; one element per fully drawn character

The hierarchy probe's *type* does not depend on how many widgets an episode
draws, exactly as `generators/logic`'s `gates` channel does not depend on depth:
`ARCHITECTURE.md` section 1 already says "relations are sets of tuples", and this
is that and nothing else.  Sets are duplicate-free, so the `id` field is
load-bearing twice over -- it is what lets `parent` name another element, and it
is what keeps two widgets of the same kind and geometry from collapsing into one.
Cardinality moves with the screen; the type does not.

Difficulty dials.  `generators/logic`'s `depth` looked like a difficulty axis and
was not (FINDINGS F-bench), so every dial here is one whose effect is measured in
`research/gui-hierarchy/RESULTS.md` rather than asserted:

    resolution / width / height   pixels per screen
    widgets                       requested widget count (achieved count reported)
    nesting                       requested maximum tree depth (achieved reported)
    palette                       distinct colours available; the assignment is
                                  injective iff palette >= 2 * widgets, and below
                                  that two widgets share a colour
    palette_levels                channel quantisation, 2..32: colour separation
                                  is 210/(levels-1), which is what decides
                                  whether `eq`'s training surrogate carries any
                                  gradient between two different colours
    borders                       1px ink outline per widget: a colour change
                                  that is *not* an ownership change
    labels                        glyphs drawn inside widgets: likewise
    colour_mode                   'random' redraws the widget->colour map every
                                  episode (owner recoverable, kind not);
                                  'kind' fixes colour per kind (kind recoverable,
                                  owner not)

Why the rendering is flat and sharp, and why that is a contract.

`role="byte"` is an *uncommitted* carrier and is excluded from `Type.numeric`, so
the legal algebra over a raw pixel is `eq(byte, byte) -> bool`, `pack` (gradient
`none`), `index`, and the structural operators; every arithmetic and ordering
operator is signature-illegal until a program pays for an explicit `interpret`
node committing the octet to a magnitude.  So the cheapest perception program
that can exist over these pixels is Boolean and relational, and `eq` is the one
predicate that needs no commitment.  This generator is therefore rendered
so that `eq` is sufficient:

* every widget is a filled axis-aligned rectangle at integer coordinates with one
  flat colour, so a pixel carries a colour and not a blend;
* the palette is by default a four-level-per-channel grid, so two distinct
  palette colours differ by at least 70 in some channel.  That is past the point
  where `eq`'s `exp(-(a-b)^2/tau)` surrogate underflows to exactly 0.0 in float32
  at the shipped `tau = 1` (|a-b| >= 11), which makes the comparison an exactly
  hard indicator -- excellent for the discrete path and a dead gradient for any
  choice upstream of it.  `palette_levels` moves that separation, so it is a
  declared dial and the trade is measured rather than assumed;
* glyphs are thresholded to a hard mask before painting, because an antialiased
  edge would put a blend of two palette entries on the screen.

None of this is a numeric view of the pixels handed to the agent.  There is no
pre-converted channel here and there must not be: the observation is the raw
byte tuple `image_value` produces, exactly as `geometry` emits it.

Actions.  `wait`; `focus` selects a widget; `press` toggles a pressable widget
between its two colours, which is the only way an action changes the raster.  The
objective may name a target widget, and `goal` rewards that widget being pressed.
"""
from __future__ import annotations

from tcn.generation import Generator, Value, image_value, vector_value
from tcn.types import product, setof, integer
from .render import (KINDS, PRESSABLE, assign_colours, assign_kinds, assign_labels, layout,
                     palette_colours, render)

FIELD = integer(8, signed=False)
ELEMENT = product(FIELD, FIELD, FIELD, FIELD, FIELD, FIELD, FIELD)
NO_PARENT = 255
LIMIT = 255


def hierarchy_type(capacity):
    """Depth- and count-independent type of the component relation."""
    return setof(ELEMENT, int(capacity))


def hierarchy_value(widgets, capacity):
    """`widgets` as {(id, parent, kind, x, y, w, h)}; lossless for count <= capacity."""
    rows = tuple((w["id"], w["parent"], w["kind"], *w["rect"]) for w in widgets)
    return Value.of(hierarchy_type(capacity), rows)


def glyphs_from_set(value):
    """Glyph rows in drawing order; (id, slot) is the key the set cannot lose."""
    return [{"id": i, "slot": s, "code": c, "rect": [x, y, w, h]}
            for i, s, c, x, y, w, h in sorted(value.decoded)]


def widgets_from_set(value):
    """Inverse of `hierarchy_value`: the id field restores the pre-order exactly."""
    return [{"id": i, "parent": p, "kind": k, "rect": [x, y, w, h]}
            for i, p, k, x, y, w, h in sorted(value.decoded)]


class Implementation(Generator):
    action_schema = {"wait": {}, "focus": {"id": FIELD}, "press": {"id": FIELD}}

    def initialize(self, address, configuration):
        rng = address.rng()
        resolution = int(configuration.get("resolution", 32))
        width = int(configuration.get("width", resolution))
        height = int(configuration.get("height", resolution))
        if not 8 <= width <= LIMIT or not 8 <= height <= LIMIT:
            raise ValueError("screen size must be 8..255 so every coordinate fits int[8]")
        capacity = int(configuration.get("hierarchy_capacity", 32))
        if not 1 <= capacity <= 64:
            raise ValueError("hierarchy_capacity must be 1..64")
        target = int(configuration.get("widgets", 6))
        if not 1 <= target <= capacity:
            raise ValueError("widget count outside the declared hierarchy capacity")
        if target > NO_PARENT:
            raise ValueError("widget id must fit the declared int[8] field")
        nesting = int(configuration.get("nesting", 2))
        if not 0 <= nesting <= 8:
            raise ValueError("nesting must be 0..8")
        levels = int(configuration.get("palette_levels", 4))
        colours = palette_colours(levels)
        palette = int(configuration.get("palette", 32))
        if not 2 <= palette <= len(colours):
            raise ValueError(f"palette must be 2..{len(colours)} at palette_levels={levels}")
        min_size = int(configuration.get("min_size", 6))
        if not 2 <= min_size <= min(width, height):
            raise ValueError("min_size outside the screen")
        widgets = layout(rng, width, height, target, nesting, min_size,
                         int(configuration.get("margin", 1)), int(configuration.get("gap", 1)))
        assign_kinds(widgets, rng)
        assign_colours(widgets, rng, palette, configuration.get("colour_mode", "random"), levels)
        assign_labels(widgets, rng, int(configuration.get("label_length", 4)))
        for w in widgets:
            w["pressed"] = False
        glyph_capacity = int(configuration.get("glyph_capacity", 64))
        if not 1 <= glyph_capacity <= 256:
            raise ValueError("glyph_capacity must be 1..256")
        return {"time": 0., "tick": 0, "widgets": widgets, "width": width, "height": height,
                "capacity": capacity, "borders": bool(configuration.get("borders", False)),
                "labels": bool(configuration.get("labels", False)),
                "label_size": int(configuration.get("label_size", 8)),
                "glyph_capacity": glyph_capacity,
                "focus": 0, "horizon": int(configuration.get("horizon", 4)),
                "objective": configuration.get("objective", {}) or {}}

    def advance(self, state, actions, dt, rng):
        changed = 0
        for a in actions:
            if a.verb == "wait":
                continue
            index = int(a.arg("id"))
            if not 0 <= index < len(state["widgets"]):
                raise ValueError("unknown widget id")
            widget = state["widgets"][index]
            if a.verb == "focus":
                state["focus"] = index
            if a.verb == "press":
                if KINDS[widget["kind"]] not in PRESSABLE:
                    raise ValueError("widget kind does not accept a press")
                widget["pressed"] = not widget["pressed"]
                changed += 1
        target = state["objective"].get("target")
        reward = 0. if target is None else float(
            0 <= int(target) < len(state["widgets"]) and state["widgets"][int(target)]["pressed"])
        state["done"] = state["tick"] + 1 >= state["horizon"]
        return state, {"pressed": Value.of(integer(32, signed=False), changed)}, {"goal": reward}

    def observe(self, s):
        image, owner, glyphs = render(s["widgets"], s["width"], s["height"], s["borders"],
                                      s["labels"], s["label_size"])
        observations = {"pixels": image_value(image)}
        latents = {"focus": Value.of(FIELD, s["focus"])}
        probes = {"hierarchy": hierarchy_value(s["widgets"], s["capacity"]),
                  "owner": vector_value(owner.flatten())}
        if s["labels"]:
            if len(glyphs) > s["glyph_capacity"]:
                raise ValueError("drawn glyphs exceed the declared glyph_capacity")
            probes["glyphs"] = Value.of(setof(ELEMENT, s["glyph_capacity"]),
                                        tuple((g["id"], g["slot"], g["code"], *g["rect"])
                                              for g in glyphs))
        menu = tuple(v for v in self.action_schema
                     if v != "press" or any(KINDS[w["kind"]] in PRESSABLE for w in s["widgets"]))
        return observations, latents, probes, {"agent_0": menu}
