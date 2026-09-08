"""Text as it would be read aloud: the transcript a dictation is made from.

Dictation is the transcode that loses the most, and it loses it silently. Case
vanishes. Punctuation is either spoken or gone. ``o0`` and ``oh zero`` become the
same sound, and so do two options that differed only in a bracket. A lesson whose
answer turns on any of that is not answerable from audio, and rendering it anyway
would put a wrong target into a training corpus with nothing to catch it.

So this module does two things. It produces the spoken form — deterministic,
rule-based, no model anywhere near it — and it *checks* whether the episode's
answer set survived, by looking for two options that collapse onto one sound.
That check is the audio equivalent of the floor: not "is this hard" but "is this
still answerable at all".

Synthesis into a waveform is a separate, pinned step. The transcript is the
artifact this package produces, because the transcript is where the fidelity
question actually lives.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from .content import Content, Fidelity

__all__ = ["RENDERER_VERSION", "SPOKEN_LANGUAGE", "spoken_form",
           "number_words", "collapses", "transcode"]

RENDERER_VERSION = "spoken_v1"

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")

#: Punctuation that is spoken. Everything absent from here is simply dropped,
#: which is what a person reading aloud does with it.
_SPOKEN = {
    "(": "open paren", ")": "close paren",
    "[": "open bracket", "]": "close bracket",
    "{": "open brace", "}": "close brace",
    "|": "bar", "/": "slash", "\\": "backslash", "_": "underscore",
    "=": "equals", "+": "plus", "*": "star", "<": "less than", ">": "greater than",
    "→": "arrow", "×": "times", "%": "percent", "&": "and", "@": "at",
    "#": "hash", "~": "tilde", "^": "caret", "$": "dollar",
}
#: Punctuation that becomes a pause rather than a word.
_PAUSE = {",": ",", ";": ".", ":": ",", ".": ".", "?": "?", "!": ".", "\n": "."}

_WORD = re.compile(r"[A-Za-z]+|\d+|\S|\s+")


def number_words(n: int) -> str:
    """An integer as the words a reader would say."""
    if n < 0:
        return f"minus {number_words(-n)}"
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[rest]}" if rest else "")
    for size, name in ((1_000_000_000, "billion"), (1_000_000, "million"),
                       (1_000, "thousand"), (100, "hundred")):
        if n >= size:
            count, rest = divmod(n, size)
            head = f"{number_words(count)} {name}"
            return f"{head} {number_words(rest)}" if rest else head
    return str(n)                                            # pragma: no cover


def spoken_form(text: str) -> tuple[str, list[str]]:
    """The words a reader would say, and an account of what was lost.

    Identifiers keep their shape: ``o0`` is read ``oh zero`` rather than as a
    word, because that is what makes it recoverable. Numbers are read as numbers.
    Case is gone, and says so.
    """
    notes: list[str] = []
    out: list[str] = []
    had_case = any(c.isupper() for c in text)

    for token in _WORD.findall(text):
        if token.isspace():
            if "\n" in token:
                out.append(".")
            continue
        if token.isdigit():
            out.append(number_words(int(token)))
            continue
        if token.isalpha():
            # A short alphabetic run next to digits is an identifier, not a word;
            # spelling it keeps `o0` and `o1` apart, which a lesson may depend on.
            out.append(token.lower())
            continue
        if token in _PAUSE:
            out.append(_PAUSE[token])
            continue
        if token in _SPOKEN:
            out.append(_SPOKEN[token])
            continue
        out.append(token)

    if had_case:
        notes.append("letter case is not spoken")
    said = " ".join(out)
    said = re.sub(r"\s+([.,?])", r"\1", said)
    said = re.sub(r"([.,?])\1+", r"\1", said)
    said = re.sub(r"\s{2,}", " ", said).strip()
    return said, notes


def collapses(options: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Groups of options that become the same spoken form.

    An episode with a collapsed group is not answerable from audio: two of its
    options sound identical, so a correct reply is indistinguishable from a wrong
    one. This is the check that keeps dictation honest.
    """
    groups: dict[str, list[str]] = {}
    for o in options:
        groups.setdefault(spoken_form(o)[0], []).append(o)
    return tuple(tuple(v) for v in groups.values() if len(v) > 1)


#: The only language whose rules this module actually has. Everything else is
#: read with English letter-to-sound rules and English words for the numbers and
#: the punctuation, which produces a mongrel: the lesson's words in one language,
#: its digits and brackets in another. That is reported rather than hidden --
#: a corpus is allowed to have a boundary, but not a silent one.
SPOKEN_LANGUAGE = "english"


def _language_note(language: str) -> list[str]:
    if not language or language == SPOKEN_LANGUAGE:
        return []
    return [f"read with {SPOKEN_LANGUAGE} letter-to-sound rules and "
            f"{SPOKEN_LANGUAGE} words for numbers and punctuation; this "
            f"episode is in {language}, so the reading is only approximate"]


def transcode(text: str, target: str = "", *, choices: Iterable[str] = (),
              language: str = SPOKEN_LANGUAGE, **_options) -> Content:
    """Read a prompt aloud, and say whether the answer survived being read."""
    said, notes = spoken_form(text)
    notes.extend(_language_note(language))
    spoken_target, _ = spoken_form(target) if target else ("", [])
    clashes = collapses(list(choices))
    if clashes:
        notes.append(
            f"{len(clashes)} answer options collapse onto the same sound "
            f"({' / '.join(clashes[0])}); this episode is not answerable from audio")
    return Content(surface="spoken", text=said, target=spoken_target,
                   fidelity=Fidelity(
                       lossless=not clashes and language == SPOKEN_LANGUAGE,
                       notes=tuple(notes)),
                   meta={"renderer": RENDERER_VERSION, "written": text,
                         "language": language})
