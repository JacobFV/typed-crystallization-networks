"""Phonemes to a waveform: a formant synthesizer with no model and no dependency.

This is the last step of dictation, and it is built the way speech was
synthesized before anybody trained anything: a source-filter model. A buzz at the
pitch of the voice, or a hiss for the sounds that have no pitch, pushed through
three resonators tuned to where the mouth would put its formants. Nothing here is
learned, sampled from a corpus, or fetched; the bytes are a function of the
phones and the settings, so the same episode dictates to the same WAV on every
machine.

It sounds like a machine from 1980, which is the correct trade. What dictation
has to preserve is *which words were said* — whether ``o0`` and ``o1`` stay
distinguishable, whether two options collapse onto one sound — and that survives
a robotic voice perfectly well. Naturalness would cost a model, and a model would
make the corpus a function of somebody's weights. See ``INTENT.md``.

The formant values are the standard measured averages for American English
vowels; the consonant targets are their loci, which is what makes the transition
into the next vowel carry the place of articulation — the cue a listener actually
uses to tell ``b`` from ``d``.
"""

from __future__ import annotations

import io
import math
import struct
import wave
from typing import Iterable, Sequence

from .phonemes import pronounce

__all__ = ["RENDERER_VERSION", "SAMPLE_RATE", "synthesize", "wav", "say",
           "duration_of", "PHONES"]

RENDERER_VERSION = "audio_v1"

SAMPLE_RATE = 16_000
FRAME_MS = 5
_FRAME = SAMPLE_RATE * FRAME_MS // 1000

# name -> (F1, F2, F3, duration_ms, manner)
#
# manner drives the source: vowels and approximants are voiced and open, stops
# are a closure then a burst, fricatives are noise, nasals are voiced with a
# damped low first formant.
PHONES: dict[str, tuple[int, int, int, int, str]] = {
    # monophthongs
    "IY": (270, 2290, 3010, 130, "vowel"),
    "IH": (390, 1990, 2550, 100, "vowel"),
    "EH": (530, 1840, 2480, 110, "vowel"),
    "AE": (660, 1720, 2410, 140, "vowel"),
    "AA": (730, 1090, 2440, 140, "vowel"),
    "AO": (570, 840, 2410, 130, "vowel"),
    "UH": (440, 1020, 2240, 100, "vowel"),
    "UW": (300, 870, 2240, 130, "vowel"),
    "AH": (640, 1190, 2390, 100, "vowel"),
    "ER": (490, 1350, 1690, 130, "vowel"),
    # consonants: the numbers are loci, not steady states
    "P": (350, 900, 2200, 70, "stop_u"),
    "B": (300, 900, 2200, 60, "stop_v"),
    "T": (350, 1750, 2600, 70, "stop_u"),
    "D": (300, 1750, 2600, 60, "stop_v"),
    "K": (350, 1900, 2400, 75, "stop_u"),
    "G": (300, 1900, 2400, 60, "stop_v"),
    "F": (400, 1400, 2200, 90, "fric_u"),
    "V": (350, 1400, 2200, 70, "fric_v"),
    "TH": (400, 1600, 2400, 90, "fric_u"),
    "DH": (350, 1600, 2400, 60, "fric_v"),
    "S": (400, 1700, 2600, 100, "fric_u"),
    "Z": (350, 1700, 2600, 80, "fric_v"),
    "SH": (400, 1800, 2500, 110, "fric_u"),
    "ZH": (350, 1800, 2500, 80, "fric_v"),
    "HH": (500, 1500, 2400, 70, "fric_u"),
    "M": (300, 900, 2200, 80, "nasal"),
    "N": (300, 1600, 2600, 80, "nasal"),
    "NG": (300, 1900, 2400, 80, "nasal"),
    "L": (310, 1050, 2880, 80, "approx"),
    "R": (310, 1060, 1380, 80, "approx"),
    "W": (290, 610, 2150, 70, "approx"),
    "Y": (260, 2070, 3020, 70, "approx"),
}

#: A diphthong is two targets and a glide between them.
DIPHTHONGS: dict[str, tuple[str, str, int]] = {
    "AY": ("AA", "IY", 190),
    "EY": ("EH", "IY", 170),
    "OW": ("AO", "UW", 170),
    "AW": ("AA", "UH", 190),
    "OY": ("AO", "IY", 190),
}

#: Affricates are a stop released into a fricative.
AFFRICATES = {"CH": ("T", "SH"), "JH": ("D", "ZH")}

#: How long a pause each punctuation mark buys.
PAUSE_MS = {",": 140, ".": 300, "?": 300}

_GAIN = {"vowel": 1.0, "approx": 0.8, "nasal": 0.55,
         "fric_v": 0.35, "fric_u": 0.30, "stop_v": 0.5, "stop_u": 0.45}

#: Noise band per fricative: where the hiss actually lives.
_HISS = {"S": (6000, 1400), "Z": (6000, 1400), "SH": (3000, 1200),
         "ZH": (3000, 1200), "F": (5000, 3000), "V": (5000, 3000),
         "TH": (5500, 3000), "DH": (5500, 3000), "HH": (1500, 1500)}


class _Resonator:
    """A two-pole filter. Three of them in series make a vocal tract."""

    __slots__ = ("a", "b", "c", "y1", "y2")

    def __init__(self) -> None:
        self.a = 1.0
        self.b = self.c = self.y1 = self.y2 = 0.0

    def tune(self, freq: float, bandwidth: float) -> None:
        r = math.exp(-math.pi * bandwidth / SAMPLE_RATE)
        theta = 2.0 * math.pi * freq / SAMPLE_RATE
        self.b = 2.0 * r * math.cos(theta)
        self.c = -(r * r)
        self.a = 1.0 - self.b - self.c

    def step(self, x: float) -> float:
        y = self.a * x + self.b * self.y1 + self.c * self.y2
        self.y2 = self.y1
        self.y1 = y
        return y


class _Noise:
    """A deterministic hiss.

    Its own linear congruential generator rather than :mod:`random`, so that a
    caller who has seeded the global RNG for their own reasons cannot change what
    a word sounds like.
    """

    __slots__ = ("state",)

    def __init__(self, seed: int = 0x2545F491) -> None:
        self.state = seed & 0xFFFFFFFF

    def __call__(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state / 0x7FFFFFFF) - 1.0


def _expand(phones: Iterable[str]) -> list[tuple[str, int]]:
    """Diphthongs into their two targets, affricates into stop plus fricative."""
    out: list[tuple[str, int]] = []
    for p in phones:
        if p in DIPHTHONGS:
            a, b, dur = DIPHTHONGS[p]
            out.append((a, dur // 2))
            out.append((b, dur - dur // 2))
        elif p in AFFRICATES:
            a, b = AFFRICATES[p]
            out.append((a, PHONES[a][3]))
            out.append((b, PHONES[b][3]))
        elif p in PHONES:
            out.append((p, PHONES[p][3]))
    return out


def _frames(spoken: Sequence, rate: float) -> list[tuple]:
    """A parameter track: one tuple of synthesis settings per 5 ms."""
    track: list[tuple] = []
    for item in spoken:
        if isinstance(item, str):                       # punctuation: a pause
            for _ in range(int(PAUSE_MS.get(item, 120) / rate / FRAME_MS)):
                track.append((0, 0, 0, "silence", 0.0))
            continue
        for name, dur in _expand(item):
            f1, f2, f3, _d, manner = PHONES[name]
            n = max(1, int(dur / rate / FRAME_MS))
            if manner.startswith("stop"):
                # a closure, then the burst that actually carries the place
                for _ in range(max(1, n // 2)):
                    track.append((f1, f2, f3, "silence", 0.0))
                for _ in range(max(1, n - n // 2)):
                    track.append((f1, f2, f3, manner, _GAIN[manner]))
            else:
                for i in range(n):
                    track.append((f1, f2, f3, manner, _GAIN[manner]))
        track.append((0, 0, 0, "gap", 0.0))             # a beat between words
    return track


def _smooth(track: list[tuple]) -> list[tuple]:
    """Slide the formants into each other.

    Speech is transitions. Holding a target and jumping to the next one gives
    something that sounds like a slide projector; averaging the tracks over about
    twenty-five milliseconds gives the glide a listener reads as a consonant's
    place of articulation.
    """
    if not track:
        return track
    win = 5
    out = []
    for i, (f1, f2, f3, manner, gain) in enumerate(track):
        lo, hi = max(0, i - win // 2), min(len(track), i + win // 2 + 1)
        near = [t for t in track[lo:hi] if t[0]]
        if near and f1:
            f1 = sum(t[0] for t in near) / len(near)
            f2 = sum(t[1] for t in near) / len(near)
            f3 = sum(t[2] for t in near) / len(near)
        out.append((f1, f2, f3, manner, gain))
    return out


def synthesize(text: str, *, rate: float = 1.0, pitch: float = 110.0,
               question: bool | None = None) -> list[float]:
    """A spoken line as floating-point samples in [-1, 1].

    ``rate`` scales every duration, so 1.5 is half again as slow. ``pitch`` is
    the starting fundamental; it declines across the utterance the way a voice
    does, and rises at the end of a question instead.
    """
    spoken = pronounce(text)
    if question is None:
        question = text.rstrip().endswith("?")
    track = _smooth(_frames(spoken, rate))
    if not track:
        return []

    res = (_Resonator(), _Resonator(), _Resonator())
    hiss_res = _Resonator()
    noise = _Noise()
    out: list[float] = []
    phase = 0.0
    total = len(track)

    for i, (f1, f2, f3, manner, gain) in enumerate(track):
        # declination, and a terminal rise when the sentence is a question
        pos = i / max(1, total - 1)
        f0 = pitch * (1.0 - 0.25 * pos)
        if question and pos > 0.8:
            f0 = pitch * (0.8 + 1.2 * (pos - 0.8) / 0.2)

        if manner in ("silence", "gap"):
            out.extend([0.0] * _FRAME)
            continue

        res[0].tune(max(180.0, f1), 70.0)
        res[1].tune(max(500.0, f2), 110.0)
        res[2].tune(max(1200.0, f3), 180.0)
        voiced = manner in ("vowel", "approx", "nasal", "fric_v", "stop_v")
        fricative = manner.startswith("fric") or manner.startswith("stop")
        band = _HISS.get("S") if manner.startswith("stop") else None
        if band:
            hiss_res.tune(band[0], band[1])

        for _ in range(_FRAME):
            source = 0.0
            if voiced:
                phase += f0 / SAMPLE_RATE
                if phase >= 1.0:
                    phase -= 1.0
                # a glottal pulse, not a square: the shape is most of the timbre
                source = math.sin(math.pi * phase) ** 2 - 0.5
                if manner == "nasal":
                    source *= 0.6
            if fricative:
                h = noise()
                source += (hiss_res.step(h) if band else h) * (
                    0.7 if manner.startswith("fric") else 1.0)
            y = res[2].step(res[1].step(res[0].step(source)))
            out.append(y * gain)

    peak = max((abs(v) for v in out), default=0.0)
    if peak > 0:
        out = [v / peak * 0.85 for v in out]
    # a short fade at each end, so playback does not start with a click
    edge = min(len(out) // 2, SAMPLE_RATE // 200)
    for i in range(edge):
        k = i / edge
        out[i] *= k
        out[-1 - i] *= k
    return out


def wav(samples: Sequence[float], *, sample_rate: int = SAMPLE_RATE) -> bytes:
    """16-bit mono PCM in a RIFF container. Byte-exact for the same samples."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(b"".join(
            struct.pack("<h", max(-32768, min(32767, int(v * 32767))))
            for v in samples))
    return buf.getvalue()


def say(text: str, **options) -> bytes:
    """A spoken line straight to WAV bytes."""
    return wav(synthesize(text, **options))


def duration_of(data: bytes) -> float:
    """Seconds of audio in a WAV, read back from the container."""
    with wave.open(io.BytesIO(data), "rb") as fh:
        return fh.getnframes() / float(fh.getframerate())


#: Past this, an episode stops being a dictation and starts being a
#: working-memory test. Audio is serial and unskimmable: a listener cannot
#: glance back at the third premise the way a reader can, so a long episode is
#: measuring something the text surface never measured. Reported rather than
#: refused, because where the line sits is a judgement and the number should be
#: visible in the data.
LISTENABLE_SECONDS = 30.0


def transcode(text: str, target: str = "", *, choices=(), rate: float = 1.0,
              pitch: float = 110.0, language: str = "english", **_options):
    """Dictate a prompt: the transcript, and the waveform read from it."""
    from .content import Asset, Content, Fidelity
    from .spoken import SPOKEN_LANGUAGE, _language_note, collapses, spoken_form

    said, notes = spoken_form(text)
    notes.extend(_language_note(language))
    spoken_target, _ = spoken_form(target) if target else ("", [])
    data = wav(synthesize(said, rate=rate, pitch=pitch))
    seconds = duration_of(data)

    clashes = collapses(list(choices))
    if clashes:
        notes.append(
            f"{len(clashes)} answer options collapse onto the same sound "
            f"({' / '.join(clashes[0])}); this episode is not answerable from audio")
    if seconds > LISTENABLE_SECONDS:
        notes.append(
            f"{seconds:.0f} seconds of speech: past about {LISTENABLE_SECONDS:.0f} "
            f"this is a working-memory demand the text surface does not make")
    return Content(
        surface="audio", text=said, target=spoken_target,
        assets=(Asset(mime="audio/wav", data=data, role="prompt"),),
        fidelity=Fidelity(lossless=not clashes and language == SPOKEN_LANGUAGE,
                          notes=tuple(notes)),
        meta={"renderer": RENDERER_VERSION, "written": text, "language": language,
              "seconds": round(seconds, 2), "sample_rate": SAMPLE_RATE,
              "synthesis": "rule-based formant; no model"})
