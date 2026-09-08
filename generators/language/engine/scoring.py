"""Turning a text agent's reply into a score.

Grading free text against an exact answer is where benchmarks usually leak. Two
failure directions matter and they pull against each other: too strict and you
punish a correct answer for saying "The answer is blue."; too loose and you
credit a reply that hedged across every option.

The rule here is: normalize, then look for **exactly one** of the episode's own
choices in the reply. If the reply names one choice, that is the answer. If it
names none, or more than one, it scores zero. Because the choice set is small,
closed, and generated with the episode, this is decidable rather than a
judgement call — and it never needs a second model to grade the first.

``strict=True`` disables the containment step and requires the normalized reply
to be the normalized answer, for callers who want the harshest reading.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

__all__ = ["normalize", "extract_choice", "score", "AMBIGUOUS", "NO_CHOICE"]

#: what :func:`extract_choice` returns when the reply names several choices
AMBIGUOUS = "<ambiguous>"
#: what :func:`extract_choice` returns when the reply names none of them
NO_CHOICE = "<none>"

_PUNCT = re.compile(r"[\s‘’“”\"'`.,;:!?()\[\]{}]+")


def normalize(text: str) -> str:
    """Casefold, strip accents and surrounding punctuation, collapse whitespace."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.casefold().strip()
    t = _PUNCT.sub(" ", t).strip()
    return re.sub(r"\s+", " ", t)


def _token_pattern(choice: str) -> re.Pattern[str]:
    """Match a choice as a standalone run, not as a fragment of a longer word."""
    esc = re.escape(choice)
    return re.compile(rf"(?<![0-9a-z_]){esc}(?![0-9a-z_])")


def extract_choice(reply: str, choices: Sequence[str]) -> str:
    """The single choice a reply names, or :data:`AMBIGUOUS` / :data:`NO_CHOICE`.

    Longer choices are tested first, so naming ``o10`` is not read as ``o1``.
    """
    norm_reply = normalize(reply)
    norm = {c: normalize(c) for c in choices}
    exact = [c for c, n in norm.items() if n and n == norm_reply]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:                      # two choices that normalize alike
        return AMBIGUOUS
    if not norm_reply:
        return NO_CHOICE
    hits: list[str] = []
    covered: list[tuple[int, int]] = []
    for c in sorted(choices, key=lambda x: -len(normalize(x))):
        n = norm[c]
        if not n:
            continue
        m = _token_pattern(n).search(norm_reply)
        if not m:
            continue
        if any(s <= m.start() < e or s < m.end() <= e for s, e in covered):
            continue                        # inside a longer choice already found
        covered.append(m.span())
        hits.append(c)
    if len(hits) == 1:
        return hits[0]
    return AMBIGUOUS if hits else NO_CHOICE


def score(reply: str, answer: str, choices: Sequence[str], *, strict: bool = False) -> float:
    """1.0 if the reply names the gold answer and nothing else, else 0.0."""
    if strict:
        return 1.0 if normalize(reply) == normalize(answer) else 0.0
    return 1.0 if extract_choice(reply, choices) == _canonical(answer, choices) else 0.0


def _canonical(answer: str, choices: Iterable[str]) -> str:
    """The choice string equal to the gold answer (they may differ in spacing)."""
    n = normalize(answer)
    for c in choices:
        if normalize(c) == n:
            return c
    return answer
