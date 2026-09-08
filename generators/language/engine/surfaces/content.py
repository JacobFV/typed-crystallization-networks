"""What a rendered episode is, once it may not be a string.

Every surface in this package is a **transcode**: it carries the same underlying
text that the text surface does, in a different medium. That is a deliberate
restriction — see ``INTENT.md`` — and it makes the content model simple, because
the text is never lost. A :class:`Content` is always the string plus whatever
bytes the medium needed to express it.

Assets are content-addressed. The digest is over the bytes, so two renderings
that produced the same image are the same asset, and a cache can be keyed on it
without a registry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Asset", "Content", "Fidelity"]

_EXT = {"image/png": "png", "audio/wav": "wav", "text/plain": "txt",
        "application/json": "json"}


@dataclass(frozen=True)
class Asset:
    """Bytes with a type, addressed by their own digest."""

    mime: str
    data: bytes
    #: what this asset is within the content: ``prompt``, ``frame``, ``target``
    role: str = "prompt"
    #: frame index, for a sequence
    index: int = 0

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def ext(self) -> str:
        return _EXT.get(self.mime, "bin")

    @property
    def size(self) -> int:
        return len(self.data)

    def name(self, stem: str) -> str:
        """A stable filename for this asset under a given stem."""
        n = f"{stem}.{self.role}" if self.role != "prompt" else stem
        if self.index:
            n = f"{n}.{self.index:04d}"
        return f"{n}.{self.ext}"

    def to_dict(self) -> dict[str, Any]:
        return {"mime": self.mime, "role": self.role, "index": self.index,
                "sha256": self.sha256, "size": self.size}

    def __repr__(self) -> str:
        return f"<Asset {self.mime} {self.size}B {self.sha256[:8]}>"


@dataclass(frozen=True)
class Fidelity:
    """What a transcode lost, stated rather than assumed.

    The floor guarantee survives a transcode only if the answer is still
    recoverable from the surface. Rasterizing a glyph the font does not have, or
    dictating a bracket, loses information — and a lesson whose answer depends on
    what was lost is simply invalid in that surface. Saying so is the job of this
    record, and :func:`langcurriculum.verify.verify_surface` is what reads it.
    """

    #: True when the surface preserves everything the answer could depend on
    lossless: bool = True
    #: characters the surface could not represent
    dropped: tuple[str, ...] = ()
    #: a short human account of any transformation that was not reversible
    notes: tuple[str, ...] = ()

    def merged(self, other: "Fidelity") -> "Fidelity":
        return Fidelity(lossless=self.lossless and other.lossless,
                        dropped=tuple(dict.fromkeys([*self.dropped, *other.dropped])),
                        notes=tuple(dict.fromkeys([*self.notes, *other.notes])))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"lossless": self.lossless}
        if self.dropped:
            d["dropped"] = list(self.dropped)
        if self.notes:
            d["notes"] = list(self.notes)
        return d


@dataclass(frozen=True)
class Content:
    """One episode rendered into one surface.

    ``text`` is always the underlying string — for ``text`` it *is* the content,
    for ``raster`` it is what the pixels say, for ``spoken`` it is the words as
    they would be read aloud. Keeping it means no consumer of a transcoded corpus
    ever has to run OCR to know what the episode said.
    """

    surface: str
    text: str
    target: str = ""
    assets: tuple[Asset, ...] = ()
    fidelity: Fidelity = field(default_factory=Fidelity)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def bytes_total(self) -> int:
        return sum(a.size for a in self.assets)

    def to_dict(self, *, inline: bool = False) -> dict[str, Any]:
        import base64
        d: dict[str, Any] = {"surface": self.surface, "text": self.text,
                             "target": self.target,
                             "fidelity": self.fidelity.to_dict()}
        if self.assets:
            d["assets"] = [
                {**a.to_dict(), **({"base64": base64.b64encode(a.data).decode()}
                                   if inline else {})}
                for a in self.assets]
        if self.meta:
            d["meta"] = self.meta
        return d

    def __repr__(self) -> str:
        return (f"<Content {self.surface} {len(self.text)} chars, "
                f"{len(self.assets)} assets, {self.bytes_total}B>")
