"""English spelling to phonemes, by rule.

A dictation has to know how the words sound, and there are exactly two ways to
get that: look it up in a pronouncing dictionary, or work it out from the
letters. A dictionary would not help here, because most of what this curriculum
says aloud is not in any dictionary — the lexicons are invented per episode, and
``blicket`` and ``o0`` have to be read as confidently as ``cube``.

So this is letter-to-sound rules, in the tradition of the rule-based synthesizers
that predate any statistical model. Rules are matched longest-first with left and
right context, which is what lets ``ch`` differ from ``c``, ``ce`` from ``ca``,
and a silent final ``e`` lengthen the vowel before it rather than sound.

A short exception list carries the words English refuses to spell sensibly and
that the curriculum says constantly — ``the``, ``one``, ``eight``, ``which``.
Everything else goes through the rules, which is the point: coverage is not a
function of how many words somebody remembered to list.

The inventory is ARPAbet without stress marks. See
:mod:`langcurriculum.surfaces.audio` for what turns these into sound.
"""

from __future__ import annotations

import re

__all__ = ["ARPABET", "VOWELS", "pronounce", "pronounce_word", "EXCEPTIONS"]

#: Every phone this module can emit.
VOWELS = ("IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "ER",
          "AY", "EY", "OW", "AW", "OY")
CONSONANTS = ("P", "B", "T", "D", "K", "G", "F", "V", "TH", "DH", "S", "Z",
              "SH", "ZH", "CH", "JH", "M", "N", "NG", "L", "R", "W", "Y", "HH")
ARPABET = VOWELS + CONSONANTS

#: Words the rules get wrong and the curriculum uses constantly. Kept short on
#: purpose: a long list here is a way of hiding that the rules do not work.
EXCEPTIONS: dict[str, tuple[str, ...]] = {
    "a": ("AH",), "the": ("DH", "AH"), "of": ("AH", "V"), "to": ("T", "UW"),
    "is": ("IH", "Z"), "are": ("AA", "R"), "was": ("W", "AH", "Z"),
    "one": ("W", "AH", "N"), "two": ("T", "UW"), "four": ("F", "AO", "R"),
    "eight": ("EY", "T"), "nine": ("N", "AY", "N"), "zero": ("Z", "IH", "R", "OW"),
    "which": ("W", "IH", "CH"), "what": ("W", "AH", "T"), "who": ("HH", "UW"),
    "where": ("W", "EH", "R"), "there": ("DH", "EH", "R"), "here": ("HH", "IY", "R"),
    "answer": ("AE", "N", "S", "ER"), "colour": ("K", "AH", "L", "ER"),
    "color": ("K", "AH", "L", "ER"), "scene": ("S", "IY", "N"),
    "object": ("AA", "B", "JH", "EH", "K", "T"), "one's": ("W", "AH", "N", "Z"),
    "you": ("Y", "UW"), "your": ("Y", "AO", "R"), "said": ("S", "EH", "D"),
    "says": ("S", "EH", "Z"), "does": ("D", "AH", "Z"), "done": ("D", "AH", "N"),
    "some": ("S", "AH", "M"), "come": ("K", "AH", "M"), "none": ("N", "AH", "N"),
    "only": ("OW", "N", "L", "IY"), "many": ("M", "EH", "N", "IY"),
    "any": ("EH", "N", "IY"), "again": ("AH", "G", "EH", "N"),
    "give": ("G", "IH", "V"), "live": ("L", "IH", "V"), "have": ("HH", "AE", "V"),
    "move": ("M", "UW", "V"), "prove": ("P", "R", "UW", "V"),
    "true": ("T", "R", "UW"), "false": ("F", "AO", "L", "S"),
    "yes": ("Y", "EH", "S"), "no": ("N", "OW"), "not": ("N", "AA", "T"),
    "or": ("AO", "R"), "and": ("AE", "N", "D"), "all": ("AO", "L"),
    "both": ("B", "OW", "TH"), "each": ("IY", "CH"), "every": ("EH", "V", "R", "IY"),
    "above": ("AH", "B", "AH", "V"), "below": ("B", "IH", "L", "OW"),
    "left": ("L", "EH", "F", "T"), "right": ("R", "AY", "T"),
    "eye": ("AY",), "one_": ("W", "AH", "N"),
}

# ---------------------------------------------------------------------------
# The rule table.
#
# Each entry is (left context regex, letters, right context regex, phones).
# Contexts are matched against the text either side of the letters; ``#`` in a
# context means a word boundary. Rules are tried in order, so the long and
# specific ones come first and a bare single letter is the last resort.
# ---------------------------------------------------------------------------
_C = "[bcdfghjklmnpqrstvwxyz]"          # a consonant letter
_V = "[aeiouy]"                          # a vowel letter

RULES: list[tuple[str, str, str, tuple[str, ...]]] = [
    # --- digraphs and trigraphs, longest first ----------------------------
    ("", "tch", "", ("CH",)),
    ("", "sch", "", ("S", "K")),
    ("", "ough", "", ("AH", "F")),
    ("", "augh", "", ("AE", "F")),
    ("", "eigh", "", ("EY",)),
    ("", "tion", "", ("SH", "AH", "N")),
    ("", "sion", "", ("ZH", "AH", "N")),
    ("", "cious", "", ("SH", "AH", "S")),
    ("", "ture", "", ("CH", "ER")),
    ("", "sure", "", ("ZH", "ER")),
    ("", "ck", "", ("K",)),
    ("", "ch", "", ("CH",)),
    ("", "sh", "", ("SH",)),
    ("", "ph", "", ("F",)),
    ("", "th", "", ("TH",)),
    ("", "wh", "", ("W",)),
    ("", "qu", "", ("K", "W")),
    ("", "ng", "#", ("NG",)),
    ("", "nk", "", ("NG", "K")),
    ("", "gh", "", ()),                  # silent: light, though
    ("", "kn", "", ("N",)),              # word-initial handled by order
    ("", "wr", "", ("R",)),
    ("", "mb", "#", ("M",)),
    ("", "dge", "", ("JH",)),
    ("", "ss", "", ("S",)),
    ("", "ll", "", ("L",)),
    ("", "tt", "", ("T",)),
    ("", "pp", "", ("P",)),
    ("", "bb", "", ("B",)),
    ("", "dd", "", ("D",)),
    ("", "gg", "", ("G",)),
    ("", "ff", "", ("F",)),
    ("", "mm", "", ("M",)),
    ("", "nn", "", ("N",)),
    ("", "rr", "", ("R",)),
    ("", "cc", "", ("K",)),
    ("", "zz", "", ("Z",)),

    # --- vowel digraphs ---------------------------------------------------
    ("", "ee", "", ("IY",)),
    ("", "ea", "r", ("IY",)),
    ("", "ea", "", ("IY",)),
    ("", "ie", "#", ("IY",)),
    ("", "ie", "", ("IY",)),
    ("", "ei", "", ("EY",)),
    ("", "ai", "", ("EY",)),
    ("", "ay", "", ("EY",)),
    ("", "oa", "", ("OW",)),
    ("", "oe", "", ("OW",)),
    ("", "oo", "", ("UW",)),
    ("", "ou", "", ("AW",)),
    ("", "ow", "#", ("OW",)),
    ("", "ow", "", ("AW",)),
    ("", "oi", "", ("OY",)),
    ("", "oy", "", ("OY",)),
    ("", "au", "", ("AO",)),
    ("", "aw", "", ("AO",)),
    ("", "eu", "", ("Y", "UW")),
    ("", "ew", "", ("UW",)),
    ("", "ue", "", ("UW",)),
    ("", "ui", "", ("UW",)),

    # --- r-coloured vowels ------------------------------------------------
    ("", "ar", "#", ("AA", "R")),
    ("", "ar", _C, ("AA", "R")),
    ("", "or", "#", ("AO", "R")),
    ("", "or", _C, ("AO", "R")),
    ("", "er", "#", ("ER",)),
    ("", "er", _C, ("ER",)),
    ("", "ir", "#", ("ER",)),
    ("", "ir", _C, ("ER",)),
    ("", "ur", "#", ("ER",)),
    ("", "ur", _C, ("ER",)),

    # --- single vowels, magic e first ------------------------------------
    ("", "a", _C + "e#", ("EY",)),
    ("", "e", _C + "e#", ("IY",)),
    ("", "i", _C + "e#", ("AY",)),
    ("", "o", _C + "e#", ("OW",)),
    ("", "u", _C + "e#", ("Y", "UW")),
    ("", "e", "#", ()),                  # silent final e
    ("", "es", "#", ("Z",)),
    ("", "a", "", ("AE",)),
    ("", "e", "", ("EH",)),
    ("", "i", "#", ("AY",)),
    ("", "i", "", ("IH",)),
    ("", "o", "", ("AA",)),
    ("", "u", "", ("AH",)),
    ("#" + _C, "y", "#", ("AY",)),
    ("", "y", "#", ("IY",)),
    ("#", "y", "", ("Y",)),
    ("", "y", "", ("IH",)),

    # --- single consonants ------------------------------------------------
    ("", "c", "[eiy]", ("S",)),
    ("", "c", "", ("K",)),
    ("", "g", "[eiy]", ("JH",)),
    ("", "g", "", ("G",)),
    ("", "s", "#", ("S",)),
    ("", "s", "", ("S",)),
    ("", "x", "", ("K", "S")),
    ("", "j", "", ("JH",)),
    ("", "z", "", ("Z",)),
    ("", "b", "", ("B",)),
    ("", "d", "", ("D",)),
    ("", "f", "", ("F",)),
    ("", "h", "", ("HH",)),
    ("", "k", "", ("K",)),
    ("", "l", "", ("L",)),
    ("", "m", "", ("M",)),
    ("", "n", "", ("N",)),
    ("", "p", "", ("P",)),
    ("", "q", "", ("K",)),
    ("", "r", "", ("R",)),
    ("", "t", "", ("T",)),
    ("", "v", "", ("V",)),
    ("", "w", "", ("W",)),
]

_COMPILED = [
    (re.compile(left + "$") if left else None,
     letters,
     re.compile("^" + right) if right else None,
     phones)
    for left, letters, right, phones in RULES
]


def pronounce_word(word: str) -> tuple[str, ...]:
    """One word as phones. Unknown spellings still get a pronunciation."""
    w = word.lower()
    if w in EXCEPTIONS:
        return EXCEPTIONS[w]
    padded = f"#{w}#"
    out: list[str] = []
    i = 1
    while i < len(padded) - 1:
        for left_re, letters, right_re, phones in _COMPILED:
            if not padded.startswith(letters, i):
                continue
            if left_re is not None and not left_re.search(padded[:i]):
                continue
            if right_re is not None and not right_re.match(padded[i + len(letters):]):
                continue
            out.extend(phones)
            i += len(letters)
            break
        else:
            i += 1                                    # a letter no rule covers
    if not out:                                       # never return silence
        out = ["AH"]
    return tuple(out)


_TOKEN = re.compile(r"[a-zA-Z']+|[.,?]|\s+")


def pronounce(text: str) -> list[tuple[str, ...] | str]:
    """A spoken line as a list of words-in-phones and punctuation marks.

    Punctuation survives as itself, because a synthesizer needs to know where to
    put a pause and how to move the pitch, and both of those are the difference
    between a sentence and a list of syllables.
    """
    out: list[tuple[str, ...] | str] = []
    for token in _TOKEN.findall(text):
        if token.isspace():
            continue
        if token in (".", ",", "?"):
            out.append(token)
            continue
        phones = pronounce_word(token.strip("'"))
        if phones:
            out.append(phones)
    return out
