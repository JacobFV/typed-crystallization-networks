"""Surfaces: the media an episode's text can be carried in.

Every surface here is a **transcode**. It takes the string the text surface
produced and re-presents it — as pixels, as words read aloud, as frames — without
changing the problem underneath. A picture is a picture *of the sentence*, not a
picture of the scene the sentence describes.

That restriction is the measurement, not a limitation. Because the surface is
provably incidental, a system that answers correctly through one and not another
has learned the surface. A native rendering would measure something else, and is
a later phase; renderers register against roles here so that adding one does not
disturb what exists. See ``INTENT.md``.

===========  ===========================================================
``text``     the string itself
``raster``   an 8-bit greyscale PNG, drawn with a bundled 5x7 font
``spoken``   the transcript a dictation would be read from
``video``    a sequence of PNG frames revealing the text over time, as APNG
``audio``    that transcript synthesized into a waveform, by rule
===========  ===========================================================

One surface is **not** a transcode and is kept apart for that reason:

===========  ===========================================================
``scene``    a picture of the scene the episode built, with the question
             beside it in text — visual question answering, and a
             different measurement rather than the same one re-presented
===========  ===========================================================

Each declares a ``RENDERER_VERSION``. It belongs in the address of any cached
rendering, because a renderer that changes without its version changing is how a
published corpus silently stops matching the code that made it.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .content import Asset, Content, Fidelity
from . import audio as _audio
from . import raster as _raster
from . import scene as _scene
from . import spoken as _spoken
from . import video as _video

__all__ = ["Asset", "Content", "Fidelity", "SURFACES", "RENDERER_VERSIONS",
           "NATIVE_SURFACES", "transcode", "transcode_example", "render_native",
           "renders_natively", "surface_names", "reproducibility"]

RENDERER_VERSION = "text_v1"


def _text_transcode(text: str, target: str = "", **_options) -> Content:
    """The identity surface. Kept explicit so ``text`` is not a special case."""
    return Content(surface="text", text=text, target=target,
                   meta={"renderer": RENDERER_VERSION})


#: surface name -> the function that renders text into it
SURFACES: dict[str, Callable[..., Content]] = {
    "text": _text_transcode,
    "raster": _raster.transcode,
    "spoken": _spoken.transcode,
    "video": _video.transcode,
    "audio": _audio.transcode,
}

#: surface name -> the version of the renderer behind it
RENDERER_VERSIONS: dict[str, str] = {
    "text": RENDERER_VERSION,
    "raster": _raster.RENDERER_VERSION,
    "spoken": _spoken.RENDERER_VERSION,
    "video": _video.RENDERER_VERSION,
    "audio": _audio.RENDERER_VERSION,
    "scene": _scene.RENDERER_VERSION,
}

#: What "the same bytes" actually means, per surface, stated at the granularity
#: it holds. Quoted by the CLI and the site rather than a blanket claim.
REPRODUCIBILITY: dict[str, str] = {
    "text": "exact, given the language database version",
    "raster": "exact, given the renderer version and the bundled font",
    "spoken": "exact; the transcript is rule-based and the audio step is pinned "
              "separately",
    "video": "exact for the frames, and for the APNG container, which is PNG "
             "chunks rather than a codec",
    "audio": "exact; rule-based formant synthesis, no model and no host voice",
    "scene": "exact; procedural drawing, nearest-neighbour scaling, no blending",
}


#: Surfaces that read the episode's *structure* rather than its text, and so
#: cannot be produced from a rendered string. They do not inherit a lesson's
#: floor the way a transcode does, and have to be verified in their own right.
NATIVE_SURFACES = frozenset({"scene"})


#: Surfaces that need the answer set to say whether the episode survived being
#: rendered. Only the spoken ones do: two options can sound the same.
_WANTS_CHOICES = frozenset({"spoken", "audio"})


def surface_names() -> list[str]:
    """Every surface, transcodes first and then the native ones."""
    return list(SURFACES) + sorted(NATIVE_SURFACES)


def reproducibility(surface: str) -> str:
    return REPRODUCIBILITY.get(surface, "unstated")


def transcode(text: str, surface: str = "text", *, target: str = "",
              choices: Iterable[str] = (), **options: Any) -> Content:
    """Render text into a surface.

    ``choices`` is passed through for the surfaces that need the answer set to
    say whether the episode survived the transcode — dictation is the one that
    does, because two options can sound the same.
    """
    if surface not in SURFACES:
        raise ValueError(f"unknown surface {surface!r}; try one of {surface_names()}")
    if surface in _WANTS_CHOICES:
        return SURFACES[surface](text, target, choices=tuple(choices), **options)
    options.pop("language", None)          # only the spoken surfaces read it
    return SURFACES[surface](text, target, **options)


def transcode_example(example, surface: str = "text", **options: Any) -> Content:
    """Render an :class:`~langcurriculum.lesson.Example` into a surface."""
    options.setdefault("language", example.language)
    return transcode(example.prompt, surface, target=example.target,
                     choices=example.choices, **options)


def renders_natively(lesson, seed: int = 0, *, difficulty: float | None = None) -> bool:
    """Whether a lesson builds something a native surface can draw.

    Asked by rendering, not by consulting a table -- the same discipline the
    vocabulary harvest uses. A lesson that stops building a scene stops being
    drawable, and nobody has to remember to update a list.
    """
    try:
        return _scene.supports(lesson.structured(seed, difficulty=difficulty))
    except Exception:
        return False


def render_native(lesson, seed: int = 0, *, language: str = "english",
                  difficulty: float | None = None, surface: str = "scene",
                  **options: Any) -> Content:
    """Draw an episode from its structure, with its question rendered as text.

    The text half still goes through the language packs, so a scene can be asked
    about in any language the engine speaks; only the picture is language-free,
    which is the point of drawing it.
    """
    if surface not in NATIVE_SURFACES:
        raise ValueError(f"{surface!r} is a transcode, not a native surface; "
                         f"use transcode() for it")
    structured = lesson.structured(seed, difficulty=difficulty)
    example = lesson.example(seed, language=language, difficulty=difficulty)
    return _scene.transcode_structured(structured, question=example.prompt,
                                       target=example.target, **options)
