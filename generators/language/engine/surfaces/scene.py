"""A picture *of the scene*, rather than a picture of the sentence describing it.

Every other surface in this package is a transcode: the same string, carried
differently. This one is not. It reads the episode's **structure** — the scene a
generator actually built, before any language was chosen — and draws it. The
question stays in text, so what comes out is visual question answering, with
ground truth computed from the construction exactly as in the text version.

That makes it a different measurement, and worth keeping honest about. A
transcode cannot change what a lesson tests, because the evidence is
word-for-word the same. A native rendering can: a scene whose objects are drawn
at their coordinates makes a spatial question easier and a lexical one
meaningless, and the floor has to be re-measured rather than inherited. See
``INTENT.md``.

Only lessons that build a scene record can be drawn this way, and
:func:`supports` answers that by looking rather than by consulting a list.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .content import Asset, Content, Fidelity
from .font import HEIGHT, WIDTH, glyph
from .raster import Page, png

__all__ = ["RENDERER_VERSION", "COLOURS", "SHAPES", "supports", "objects",
           "draw", "transcode_structured"]

RENDERER_VERSION = "scene_v1"

#: The six colours the curriculum's scenes use, as RGB. Chosen far enough apart
#: that a greyscale conversion would still separate them, so the picture does not
#: quietly become a colour-vision test.
COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (206, 48, 48),
    "blue": (48, 96, 214),
    "green": (40, 150, 66),
    "yellow": (226, 190, 40),
    "purple": (140, 66, 186),
    "orange": (226, 124, 36),
}

#: The six shapes, drawn as themselves rather than as labelled boxes.
SHAPES = ("cube", "sphere", "cone", "prism", "disc", "rod")

_BG = (250, 250, 248)
_GRID = (226, 226, 220)
_INK = (24, 24, 28)


def objects(structured: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The scene's objects, read out of an episode's structure.

    ``structured`` is what :meth:`langcurriculum.lesson.Lesson.structured`
    returns — plain data, no library types.
    """
    obs = structured.get("observation") or {}
    if obs.get("t") != "record":
        return []
    scene = (obs.get("fields") or {}).get("scene")
    if not scene or scene.get("t") != "list":
        return []
    out = []
    for item in scene.get("items", []):
        if item.get("t") != "pred" or item.get("head") != "obj":
            continue
        args = [a.get("v") for a in item.get("args", [])]
        if len(args) < 5:
            continue
        oid, colour, shape, x, y = args[:5]
        out.append({"id": str(oid), "colour": str(colour), "shape": str(shape),
                    "x": int(x), "y": int(y)})
    return out


def supports(structured: Mapping[str, Any]) -> bool:
    """Whether this episode has a scene that can be drawn."""
    objs = objects(structured)
    return bool(objs) and all(o["colour"] in COLOURS and o["shape"] in SHAPES
                              for o in objs)


class _Canvas:
    """A small RGB drawing surface. Integer coordinates, no anti-aliasing.

    No smoothing anywhere, on purpose: an anti-aliased edge is a rounding
    decision, and rounding decisions are where two machines start disagreeing
    about the bytes.
    """

    def __init__(self, width: int, height: int, fill: tuple[int, int, int] = _BG):
        self.w, self.h = width, height
        self.buf = bytearray(bytes(fill) * (width * height))

    def px(self, x: int, y: int, rgb: tuple[int, int, int]) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.buf[i:i + 3] = bytes(rgb)

    def rect(self, x0: int, y0: int, x1: int, y1: int, rgb, *, fill: bool = True) -> None:
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if fill or y in (y0, y1) or x in (x0, x1):
                    self.px(x, y, rgb)

    def disc(self, cx: int, cy: int, r: int, rgb, *, ring: int = 0) -> None:
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                d = (x - cx) ** 2 + (y - cy) ** 2
                if d <= r * r and (not ring or d >= (r - ring) ** 2):
                    self.px(x, y, rgb)

    def triangle(self, cx: int, cy: int, r: int, rgb) -> None:
        for row in range(2 * r + 1):
            y = cy - r + row
            half = int(r * row / (2 * r)) if r else 0
            for x in range(cx - half, cx + half + 1):
                self.px(x, y, rgb)

    def diamond(self, cx: int, cy: int, r: int, rgb) -> None:
        for y in range(cy - r, cy + r + 1):
            half = r - abs(y - cy)
            for x in range(cx - half, cx + half + 1):
                self.px(x, y, rgb)

    def text(self, x: int, y: int, s: str, rgb, scale: int = 1) -> None:
        for i, ch in enumerate(s):
            bitmap = glyph(ch)
            ox = x + i * (WIDTH + 1) * scale
            for gy in range(HEIGHT):
                for gx in range(WIDTH):
                    if bitmap[gy][gx] != "#":
                        continue
                    for sy in range(scale):
                        for sx in range(scale):
                            self.px(ox + gx * scale + sx, y + gy * scale + sy, rgb)

    def page(self) -> Page:
        return Page(self.w, self.h, bytes(self.buf), channels=3)


def draw(objs: Sequence[Mapping[str, Any]], *, cell: int = 44, margin: int = 26,
         grid: int = 10, labels: bool = True, scale: int = 2) -> Page:
    """Draw a scene: one shape per object, at its own coordinates."""
    size = margin * 2 + cell * grid
    c = _Canvas(size, size)

    for i in range(grid + 1):
        p = margin + i * cell
        c.rect(margin, p, margin + cell * grid, p, _GRID)
        c.rect(p, margin, p, margin + cell * grid, _GRID)

    r = cell // 2 - 6
    for o in objs:
        rgb = COLOURS[o["colour"]]
        cx = margin + o["x"] * cell + cell // 2
        # y counts upward in the scene and downward on a screen
        cy = margin + (grid - 1 - o["y"]) * cell + cell // 2
        shape = o["shape"]
        if shape == "cube":
            c.rect(cx - r, cy - r, cx + r, cy + r, rgb)
        elif shape == "sphere":
            c.disc(cx, cy, r, rgb)
        elif shape == "disc":
            c.disc(cx, cy, r, rgb, ring=max(2, r // 3))
        elif shape == "cone":
            c.triangle(cx, cy, r, rgb)
        elif shape == "prism":
            c.diamond(cx, cy, r, rgb)
        elif shape == "rod":
            c.rect(cx - r, cy - max(1, r // 3), cx + r, cy + max(1, r // 3), rgb)
        if labels:
            c.text(cx - r, cy + r + 2, o["id"], _INK, scale=1)

    page = c.page()
    if scale > 1:
        page = _upscale(page, scale)
    return page


def _upscale(page: Page, factor: int) -> Page:
    """Nearest-neighbour, so every output pixel is a copy rather than a blend."""
    w, h = page.width * factor, page.height * factor
    buf = bytearray(w * h * 3)
    for y in range(page.height):
        row = page.pixels[y * page.width * 3:(y + 1) * page.width * 3]
        wide = bytearray()
        for x in range(page.width):
            wide += row[x * 3:x * 3 + 3] * factor
        for sy in range(factor):
            start = ((y * factor) + sy) * w * 3
            buf[start:start + w * 3] = wide
    return Page(w, h, bytes(buf), channels=3)


def transcode_structured(structured: Mapping[str, Any], question: str = "",
                         target: str = "", **options) -> Content:
    """Draw the scene, and hand back the question as text beside it.

    The prompt a caller assembles from this is a picture plus a sentence, which
    is what visual question answering is. The answer is still computed from the
    construction, so the ground truth is exactly as strong as the text version's.
    """
    objs = objects(structured)
    if not objs:
        raise ValueError("this episode has no scene to draw")
    unknown = sorted({o["colour"] for o in objs} - set(COLOURS)
                     | {o["shape"] for o in objs} - set(SHAPES))
    opts = {k: v for k, v in options.items()
            if k in ("cell", "margin", "grid", "labels", "scale")}
    page = draw(objs, **opts)
    notes = []
    if not opts.get("labels", True):
        notes.append("object ids are not drawn, so an episode whose answer is an "
                     "id cannot be answered from this picture")
    if unknown:
        notes.append(f"no way to draw {unknown}")
    return Content(
        surface="scene", text=question, target=target,
        assets=(Asset(mime="image/png", data=png(page), role="prompt"),),
        fidelity=Fidelity(lossless=not unknown and opts.get("labels", True),
                          dropped=tuple(unknown), notes=tuple(notes)),
        meta={"renderer": RENDERER_VERSION, "objects": len(objs),
              "width": page.width, "height": page.height,
              "native": True, "note": "a picture of the scene, not of the sentence"})
