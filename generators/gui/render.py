"""Layout and rasterization for synthetic GUI screens.

Generator-side ground truth, never agent preprocessing.  Everything here is
axis-aligned, integer-coordinate and flat-coloured on purpose: the point of this
generator is that the widget hierarchy is *recoverable in principle* from the
pixels, so the renderer must not destroy the information the probe claims to
describe.  Where it does destroy it -- shared border ink, glyph pixels, a palette
too small to give every widget its own colour -- that is an explicit
configuration dial and the loss is measured rather than assumed.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def palette_levels(levels):
    """`levels` evenly spaced channel values in [30, 240]; separation 210/(levels-1)."""
    if not 2 <= levels <= 32:
        raise ValueError("palette_levels must be 2..32")
    return tuple(round(30 + i * 210 / (levels - 1)) for i in range(levels))


def palette_colours(levels=4):
    """The full colour grid at that separation, ordered so a prefix stays spread.

    The ordering is load-bearing, because `palette` takes a *prefix* of this
    list.  A plain lexicographic product would make the first `palette` entries
    vary only the last channel or two -- at `palette_levels 32, palette 32` every
    widget would share a red and a green value, and a comparison on those
    channels would be constant.  Sorting by the largest channel index first fills
    a growing cube corner instead, so any prefix varies all three channels.

    Separation is a declared dial, not a cosmetic one.  `eq`'s training surrogate
    is `exp(-(a-b)^2/tau)`; at the shipped `tau = 1` it is exactly 0.0 in float32
    for `|a-b| >= 11`, so how far apart two palette entries sit decides whether a
    comparison between them carries any gradient at all.  The exact path is
    unaffected either way -- distinct is distinct.
    """
    values = palette_levels(levels)
    grid = sorted(((i, j, k) for i in range(levels) for j in range(levels)
                   for k in range(levels)), key=lambda t: (max(t), t))
    return tuple((values[i], values[j], values[k]) for i, j, k in grid)


# Four levels per channel: 64 colours, minimum per-channel separation 70, none of
# them the ink colour.  A flat, widely separated palette keeps "two pixels have
# the same colour" an exact test on raw bytes rather than a threshold.
PALETTE = palette_colours(4)
INK = (0, 0, 0)
KINDS = ("window", "panel", "button", "label", "field")
PRESSABLE = ("button", "field")
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def layout(rng, width, height, target, nesting, min_size=6, margin=1, gap=1):
    """Recursive axis-aligned split.  Returns widgets in pre-order (parents first).

    `target` is the requested widget count and `nesting` the requested maximum
    tree depth; both are upper bounds, because a container that cannot be split
    into parts of at least `min_size` is a leaf whatever the request says.  The
    achieved count and depth are reported by the generator so a difficulty dial
    is never assumed to have moved.
    """
    widgets = [{"id": 0, "parent": 255, "depth": 0, "rect": [0, 0, width, height],
                "children": []}]
    openable = [0]
    while len(widgets) < target and openable:
        openable.sort(key=lambda i: (-widgets[i]["rect"][2] * widgets[i]["rect"][3], i))
        index = openable.pop(0)
        parent = widgets[index]
        if parent["depth"] >= nesting:
            continue
        x, y, w, h = parent["rect"]
        ix, iy, iw, ih = x + margin, y + margin, w - 2 * margin, h - 2 * margin
        vertical = ih > iw if ih != iw else rng.random() < .5
        span = ih if vertical else iw
        parts = 3 if rng.random() < .4 else 2
        while parts > 1 and span - (parts - 1) * gap < parts * min_size:
            parts -= 1
        if parts < 2:
            continue
        room = span - (parts - 1) * gap
        lengths = [min_size] * parts
        for _ in range(room - parts * min_size):
            lengths[rng.randrange(parts)] += 1
        offset = 0
        for length in lengths:
            if vertical:
                rect = [ix, iy + offset, iw, length]
            else:
                rect = [ix + offset, iy, length, ih]
            offset += length + gap
            child = {"id": len(widgets), "parent": parent["id"], "depth": parent["depth"] + 1,
                     "rect": rect, "children": []}
            parent["children"].append(child["id"])
            widgets.append(child)
            openable.append(child["id"])
            if len(widgets) >= target:
                break
    return widgets


def assign_kinds(widgets, rng, leaf_kinds=("button", "label", "field")):
    """Root is a window, an interior node is a panel, a leaf draws a leaf kind."""
    for w in widgets:
        if w["id"] == 0:
            name = "window"
        elif w["children"]:
            name = "panel"
        else:
            name = leaf_kinds[rng.randrange(len(leaf_kinds))]
        w["kind"] = KINDS.index(name)
    return widgets


def assign_colours(widgets, rng, palette, mode, levels=4):
    """Two colours per widget: resting and pressed.

    `mode='random'` draws a fresh per-episode assignment, so colour identifies a
    widget *within* an episode and carries no cross-episode meaning; when
    `palette >= 2 * len(widgets)` the assignment is injective and colour
    determines the owner exactly.  `mode='kind'` makes colour a fixed function of
    the widget kind, which is the opposite trade: kind becomes recoverable from
    appearance and owner does not.
    """
    colours = palette_colours(levels)[:palette]
    if mode == "kind":
        for w in widgets:
            w["colour"] = list(colours[(2 * w["kind"]) % palette])
            w["pressed_colour"] = list(colours[(2 * w["kind"] + 1) % palette])
        return widgets
    if mode != "random":
        raise ValueError("colour_mode must be 'random' or 'kind'")
    order = list(range(palette))
    rng.shuffle(order)
    for i, w in enumerate(widgets):
        w["colour"] = list(colours[order[(2 * i) % palette]])
        w["pressed_colour"] = list(colours[order[(2 * i + 1) % palette]])
    return widgets


def assign_labels(widgets, rng, length=4):
    for w in widgets:
        w["text"] = "".join(ALPHABET[rng.randrange(len(ALPHABET))] for _ in range(length)) \
            if KINDS[w["kind"]] in ("button", "label", "field") else ""
    return widgets


def render(widgets, width, height, borders=False, labels=False, size=8):
    """Raster plus the exact per-pixel owner map.

    Fills are painted in pre-order, so a child covers its parent and the owner of
    a pixel is the deepest widget containing it.  Borders and glyphs are painted
    afterwards and never change ownership -- they change only the *colour* of a
    pixel, which is exactly why turning them on lowers the recoverability of an
    owner boundary from a local neighbourhood.
    """
    image = Image.new("RGB", (width, height), INK)
    draw = ImageDraw.Draw(image)
    owner = np.zeros((height, width), dtype=np.int16)
    for w in widgets:
        x, y, ww, hh = w["rect"]
        fill = tuple(w["pressed_colour"] if w.get("pressed") else w["colour"])
        draw.rectangle([x, y, x + ww - 1, y + hh - 1], fill=fill)
        owner[y:y + hh, x:x + ww] = w["id"]
    if borders:
        for w in widgets:
            x, y, ww, hh = w["rect"]
            draw.rectangle([x, y, x + ww - 1, y + hh - 1], outline=INK)
    glyphs = []
    if labels:
        font = ImageFont.load_default(size=size)
        for w in widgets:
            text = w.get("text", "")
            x, y, ww, hh = w["rect"]
            if not text or ww < 4 or hh < 4:
                continue
            ix, iy, iw, ih = x + 1, y + 1, ww - 2, hh - 2
            offset = 0
            for slot, character in enumerate(text):
                x0, y0, x1, y1 = font.getbbox(character)
                # A character is drawn only when its whole box fits the widget
                # interior, so the emitted box always holds exactly that glyph's
                # ink and the question "is a glyph determined by its bounding
                # box" is asked of complete glyphs.
                if offset + x1 > iw or y1 > ih or x1 <= x0 or y1 <= y0:
                    break
                # The glyph is thresholded to a hard mask before it is painted.
                # A grey antialiased edge would put a blend of two palette
                # entries on the screen, and the whole generator rests on a pixel
                # carrying one flat colour; sharpness here is a contract, not a
                # cosmetic choice.
                cell = Image.new("L", (iw, ih), 0)
                ImageDraw.Draw(cell).text((offset, 0), character, font=font, fill=255)
                mask = cell.point(lambda v: 255 if v >= 128 else 0)
                box = mask.getbbox()
                offset += int(round(font.getlength(character)))
                if box is None:
                    continue
                image.paste(INK, (ix, iy), mask)
                glyphs.append({"id": w["id"], "slot": slot, "code": ALPHABET.index(character),
                               "rect": [ix + box[0], iy + box[1],
                                        box[2] - box[0], box[3] - box[1]]})
    return image, owner, glyphs
