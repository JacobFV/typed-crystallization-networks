"""Text as pixels: a deterministic rasterizer with no dependencies.

Vector internally, raster at the boundary — the same arrangement as text, which
is symbolic internally and characters at the boundary. A glyph is a small grid,
a page is a layout over glyphs, and the last step turns that into a PNG. Nothing
is sampled, nothing is anti-aliased, no font is loaded from the host, so the
bytes are a function of the text and the options and nothing else.

The reproducibility claim is therefore "same text, same options, same
``RENDERER_VERSION`` gives the same bytes", which is what
:data:`RENDERER_VERSION` is for. Bumping it is how you say a rendering changed;
leaving it alone while changing the output is how a published corpus silently
stops matching the code that made it.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Sequence

from .content import Asset, Content, Fidelity
from .font import HEIGHT, WIDTH, covers, decompose, glyph

__all__ = ["RENDERER_VERSION", "Page", "render", "png", "apng", "coverage", "layout"]

#: Bump when a change would alter the bytes of an already-published rendering.
#: ``v2`` draws composed diacritics, which ``v1`` rendered as a missing-glyph box.
RENDERER_VERSION = "raster_v2"

_SIG = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class Page:
    """A rasterized page: width, height, and one or three bytes per pixel.

    Greyscale for text, colour for a scene. Both go through the same PNG writer;
    the channel count picks the format's colour type and nothing else changes.
    """

    width: int
    height: int
    pixels: bytes
    channels: int = 1

    def ascii_art(self, on: str = "#", off: str = ".") -> str:
        """The page as text, for eyeballing a glyph without opening an image."""
        step = self.channels
        return "\n".join(
            "".join(on if self.pixels[(y * self.width + x) * step] < 128 else off
                    for x in range(self.width))
            for y in range(self.height))


def layout(text: str, *, columns: int = 88) -> list[str]:
    """Hard-wrap to a column count, preserving the paragraph structure.

    Wrapping is by character rather than by word measurement because the font is
    fixed-width: every glyph is exactly :data:`~langcurriculum.surfaces.font.WIDTH`
    wide, so a column count is a pixel count divided by a constant, and the two
    never disagree.
    """
    out: list[str] = []
    for raw in text.split("\n"):
        if not raw:
            out.append("")
            continue
        line = ""
        for word in raw.split(" "):
            while len(word) > columns:                 # a token longer than the page
                if line:
                    out.append(line)
                    line = ""
                out.append(word[:columns])
                word = word[columns:]
            if not line:
                line = word
            elif len(line) + 1 + len(word) <= columns:
                line = f"{line} {word}"
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


def render(text: str, *, columns: int = 88, scale: int = 2, padding: int = 8,
           line_gap: int = 3, letter_gap: int = 1, invert: bool = False) -> Page:
    """Lay text out and draw it, one glyph at a time."""
    from .font import GLYPHS as _direct
    lines = layout(text, columns=columns)
    cell_w = WIDTH + letter_gap
    cell_h = HEIGHT + line_gap
    width = padding * 2 + cell_w * columns * scale
    height = padding * 2 + max(1, len(lines)) * cell_h * scale
    bg, fg = (0, 255) if invert else (255, 0)
    buf = bytearray([bg]) * (width * height)

    def stamp(bitmap, ox: int, oy: int) -> None:
        """Blit one 5-row-or-fewer bitmap, clipped to the page."""
        for gy, srow in enumerate(bitmap):
            for gx in range(min(WIDTH, len(srow))):
                if srow[gx] != "#":
                    continue
                for sy in range(scale):
                    y = oy + gy * scale + sy
                    if not 0 <= y < height:
                        continue
                    base = y * width + ox + gx * scale
                    for sx in range(scale):
                        x = ox + gx * scale + sx
                        if 0 <= x < width:
                            buf[base + sx] = fg

    for row, line in enumerate(lines):
        for col, ch in enumerate(line[:columns]):
            ox = padding + col * cell_w * scale
            oy = padding + row * cell_h * scale
            if ch in _direct:
                stamp(glyph(ch), ox, oy)
                continue
            parts = decompose(ch)
            if parts is None:
                stamp(glyph(ch), ox, oy)               # the missing-glyph box
                continue
            # A mark sits in the leading above the letter, or just below it.
            # Both bands already exist as line spacing, so composing costs no
            # extra height and leaves every un-accented line where it was.
            base_ch, above, below = parts
            stamp(glyph(base_ch), ox, oy)
            for mark in above:
                stamp(mark, ox, oy - len(mark) * scale)
            for mark in below:
                stamp(mark, ox, oy + HEIGHT * scale)
    return Page(width, height, bytes(buf))


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def _scanlines(page: "Page") -> bytes:
    raw = bytearray()
    stride = page.width * page.channels
    for y in range(page.height):
        raw.append(0)                                  # filter type 0: none
        raw += page.pixels[y * stride:(y + 1) * stride]
    return bytes(raw)


def _colour_type(channels: int) -> int:
    if channels == 1:
        return 0                                       # greyscale
    if channels == 3:
        return 2                                       # truecolour
    raise ValueError(f"{channels} channels; PNG here is greyscale or RGB")


def apng(pages: Sequence["Page"], *, delay_ms: int = 700, loops: int = 0) -> bytes:
    """An animated PNG of a frame sequence.

    A container that is byte-exact, which no video codec is: APNG is PNG chunks
    all the way down, so the same frames give the same file on every machine and
    the reproducibility claim survives packaging. Every frame must be the same
    size, which the reveal renderer guarantees by keeping the page height fixed.

    Delays are a rational number of seconds in the format, so ``delay_ms`` is
    written as ``ms/1000`` exactly rather than converted to some other tick.
    """
    if not pages:
        raise ValueError("an animation needs at least one frame")
    w, h = pages[0].width, pages[0].height
    if any((p.width, p.height, p.channels) != (w, h, pages[0].channels) for p in pages):
        raise ValueError("every frame of an animation must be the same size")

    ct = _colour_type(pages[0].channels)
    out = [_SIG, _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ct, 0, 0, 0))]
    out.append(_chunk(b"acTL", struct.pack(">II", len(pages), loops)))
    seq = 0
    for i, page in enumerate(pages):
        out.append(_chunk(b"fcTL", struct.pack(
            ">IIIIIHHBB", seq, w, h, 0, 0, delay_ms, 1000, 0, 0)))
        seq += 1
        data = zlib.compress(_scanlines(page), 9)
        if i == 0:
            out.append(_chunk(b"IDAT", data))          # the first frame is the still
        else:
            out.append(_chunk(b"fdAT", struct.pack(">I", seq) + data))
            seq += 1
    out.append(_chunk(b"IEND", b""))
    return b"".join(out)


def png(page: Page) -> bytes:
    """An 8-bit greyscale PNG. No palette, no interlace, no ancillary chunks."""
    ihdr = struct.pack(">IIBBBBB", page.width, page.height, 8,
                       _colour_type(page.channels), 0, 0, 0)
    return (_SIG + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(_scanlines(page), 9))
            + _chunk(b"IEND", b""))


def coverage(text: str) -> tuple[str, ...]:
    """Characters of ``text`` the font cannot draw."""
    return covers(text)


def transcode(text: str, target: str = "", **options) -> Content:
    """Rasterize a prompt. The target stays text — replies are always text."""
    missing = coverage(text)
    opts = {k: v for k, v in options.items()
            if k in ("columns", "scale", "padding", "line_gap", "letter_gap", "invert")}
    page = render(text, **opts)
    fid = Fidelity(
        lossless=not missing,
        dropped=missing,
        notes=(f"{len(missing)} characters outside the bundled font were drawn as "
               f"a box; an episode whose answer depends on them is not readable "
               f"from this image",) if missing else ())
    return Content(surface="raster", text=text, target=target,
                   assets=(Asset(mime="image/png", data=png(page), role="prompt"),),
                   fidelity=fid,
                   meta={"renderer": RENDERER_VERSION, "width": page.width,
                         "height": page.height, **opts})
