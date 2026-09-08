"""Text as a sequence of frames, so that reading it takes time.

A page can be taken in at a glance; a video cannot. Revealing the same text a
line at a time forces integration across frames, which is a real perceptual
demand and not one the raster surface makes — while still carrying exactly the
same string, so the task underneath is untouched.

The artifact is the **frame sequence**, and the container is APNG. That pairing
is deliberate: frames are what the reproducibility claim is about, and APNG is
the one container that does not weaken it, being PNG chunks all the way down
rather than the output of a codec whose bytes drift between builds. A caller who
wants an mp4 can transcode the frames with whatever encoder they like; that step
is packaging, and its bytes are nobody's guarantee. See ``INTENT.md``.
"""

from __future__ import annotations

from typing import Sequence

from .content import Asset, Content, Fidelity
from .raster import apng, coverage, layout, png, render

__all__ = ["RENDERER_VERSION", "frames", "pages", "transcode"]

RENDERER_VERSION = "video_v1"

_PAGE_OPTS = ("scale", "padding", "line_gap", "letter_gap", "invert")


def pages(text: str, *, mode: str = "reveal", window: int = 12,
          columns: int = 88, **options):
    """The frames of a reading, as :class:`~langcurriculum.surfaces.raster.Page`.

    ``reveal`` adds one line per frame, so the page fills up and the reader
    accumulates. ``scroll`` keeps a fixed window and moves it down, so nothing is
    on screen for the whole clip and the reader has to hold it.
    """
    lines = layout(text, columns=columns)
    opts = {k: v for k, v in options.items() if k in _PAGE_OPTS}
    out = []
    if mode == "reveal":
        for i in range(1, len(lines) + 1):
            shown = lines[:i] + [""] * (len(lines) - i)          # keep the page size fixed
            page = render("\n".join(shown), columns=columns, **opts)
            # Revealing a blank line changes nothing, and a frame identical to
            # the one before it is dead time rather than a reading step.
            if out and page.pixels == out[-1].pixels:
                continue
            out.append(page)
    elif mode == "scroll":
        span = max(1, window)
        last = max(1, len(lines) - span + 1)
        for i in range(last):
            shown = lines[i:i + span] + [""] * max(0, span - len(lines[i:i + span]))
            out.append(render("\n".join(shown), columns=columns, **opts))
    else:
        raise ValueError(f"unknown mode {mode!r}; try reveal or scroll")
    return out


def frames(text: str, **options) -> list[bytes]:
    """The frames of a reading, as PNG bytes."""
    return [png(p) for p in pages(text, **options)]


def transcode(text: str, target: str = "", *, delay_ms: int = 700,
              container: bool = True, **options) -> Content:
    """Render a prompt as a frame sequence, packaged as an APNG."""
    missing = coverage(text)
    mode = options.get("mode", "reveal")
    drawn = pages(text, **options)
    assets = [Asset(mime="image/png", data=png(p), role="frame", index=i)
              for i, p in enumerate(drawn)]
    if container:
        assets.insert(0, Asset(mime="image/apng",
                               data=apng(drawn, delay_ms=delay_ms), role="prompt"))
    notes = []
    if missing:
        notes.append(f"{len(missing)} characters outside the bundled font were "
                     f"drawn as a box")
    if mode == "scroll":
        notes.append("a scrolling window shows no frame containing the whole "
                     "episode; this is a working-memory demand the text surface "
                     "does not make")
    return Content(
        surface="video", text=text, target=target, assets=tuple(assets),
        fidelity=Fidelity(lossless=not missing, dropped=missing, notes=tuple(notes)),
        meta={"renderer": RENDERER_VERSION, "mode": mode, "frames": len(drawn),
              "delay_ms": delay_ms,
              "container": "apng; byte-exact, unlike any codec"})
