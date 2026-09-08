"""How an episode is surfaced: the language, the answer format, the modality.

A lesson builds a problem. A :class:`Presentation` decides how that problem is
*shown* — which language it is read in, how its answer set is written into the
prompt, what the expected reply looks like, and which modality the whole thing is
transcoded into. None of that is the lesson's business, and a generator never
learns any of it.

The separation is the measurement. Because every surface carries the same
underlying string, a system that answers correctly through one and not another
has learned the surface rather than the problem. See ``INTENT.md``.

Three axes multiply here, and they are independent:

``language``
    Any of the codes :mod:`langcurriculum.languages` resolves — the seven
    hand-written packs or the several hundred the grammar database covers.

``answer_format``
    Where the answer set goes and what the target looks like. Lessons used to
    hand back ``choices`` as a separate field; now the options are written into
    the prompt body and the reply is open-form text, so the format decides
    whether a target reads ``B``, ``B) the red cube``, or ``the red cube``.

``surface``
    The modality. ``text`` is the string itself; the rest are transcodes of that
    same string — see :mod:`langcurriculum.surfaces`.

The answer set is *retained* even though it no longer appears as a field of its
own, because it is what makes the floor computable, and a floor is what makes an
accuracy number mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterator, Sequence

from .languages import DEFAULT_LANGUAGE

__all__ = ["Presentation", "AnswerFormat", "ANSWER_FORMATS", "OPEN_FORMAT",
           "DEFAULT_PRESENTATION", "presentations", "LABELS"]

#: The labels an option list is lettered with. Latin letters are used even in
#: languages that do not use the Latin script: the label is an index, not a
#: word, and a reply of "B" should mean the same thing everywhere.
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class AnswerFormat:
    """One way of writing the answer set into the prompt and the target out.

    ``options`` says how the choices are laid out: ``inline`` on one line,
    ``listed`` as a bulleted block, ``labelled`` as a lettered block, or
    ``hidden`` for the lessons that can be answered without being shown the
    vocabulary at all.

    ``target`` says what the expected reply is: the option's own text
    (``statement``), its letter (``label``), or both (``both``).
    """

    name: str
    options: str          # inline | listed | labelled | hidden
    target: str           # statement | label | both
    instruct: bool = True

    def render_options(self, lex, opts: Sequence[str]) -> str:
        """The part of the prompt that shows the answer set."""
        if self.options == "hidden":
            return ""
        if self.options == "inline":
            body = lex.instruction.format(choices=" | ".join(opts)) if self.instruct \
                else " | ".join(opts)
            return body
        if self.options == "listed":
            listed = "\n".join(f"{lex.bullet}{o}" for o in opts)
            head = f"{lex.options_heading}\n{listed}"
            return f"{head}\n{lex.instruction_many.format(n=len(opts))}" if self.instruct else head
        listed = "\n".join(f"{lex.bullet}{LABELS[i]}{lex.colon} {o}"
                           for i, o in enumerate(opts))
        head = f"{lex.options_heading}\n{listed}"
        return f"{head}\n{lex.instruction_many.format(n=len(opts))}" if self.instruct else head

    def render_target(self, lex, opts: Sequence[str], answer: str) -> str:
        """The expected reply, in this format."""
        if self.target == "statement" or self.options == "hidden":
            return answer
        try:
            i = list(opts).index(answer)
        except ValueError:                  # an answer outside its own option set
            return answer                   # is a lesson bug; verify catches it
        label = LABELS[i] if i < len(LABELS) else str(i + 1)
        return label if self.target == "label" else f"{label}{lex.colon} {answer}"

    def reply_vocabulary(self, lex, opts: Sequence[str]) -> tuple[str, ...]:
        """Every reply this format considers well-formed.

        Not the same as the answer set once options are lettered: the legal
        replies are then ``A``…``D`` rather than the statements, and scoring a
        reply against the statements would mark every correct answer wrong.
        """
        if self.target == "statement" or self.options == "hidden":
            return tuple(opts)
        labels = [LABELS[i] if i < len(LABELS) else str(i + 1) for i in range(len(opts))]
        if self.target == "label":
            return tuple(labels)
        return tuple(f"{l}{lex.colon} {o}" for l, o in zip(labels, opts))

    def labels_options(self) -> bool:
        return self.options == "labelled"


#: ``inline_bare`` is what the package emitted before answers went open-form, and
#: stays the default so that a caller who names nothing gets the familiar thing.
ANSWER_FORMATS: dict[str, AnswerFormat] = {
    f.name: f for f in (
        AnswerFormat("inline_bare", "inline", "statement"),
        AnswerFormat("listed_bare", "listed", "statement"),
        AnswerFormat("labelled_label", "labelled", "label"),
        AnswerFormat("labelled_both", "labelled", "both"),
        AnswerFormat("labelled_statement", "labelled", "statement"),
        AnswerFormat("inline_terse", "inline", "statement", instruct=False),
        AnswerFormat("labelled_terse", "labelled", "label", instruct=False),
        AnswerFormat("open", "hidden", "statement", instruct=False),
    )
}

#: The one format that shows no answer set. Only offered for lessons that
#: declare ``open_answerable``, because most lessons invent their vocabulary per
#: episode and are simply unanswerable without seeing it.
OPEN_FORMAT = "open"


@dataclass(frozen=True)
class Presentation:
    """Language, answer format and modality: everything about how a problem shows.

    Deliberately *not* carrying the difficulty, which changes the problem rather
    than its surface and therefore belongs with the seed. See
    :class:`langcurriculum.address.Address`.
    """

    language: str = DEFAULT_LANGUAGE
    answer_format: str = "inline_bare"
    surface: str = "text"
    #: options passed through to the surface renderer (width, font size, …)
    surface_options: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.answer_format not in ANSWER_FORMATS:
            raise ValueError(f"unknown answer format {self.answer_format!r}; "
                             f"try one of {sorted(ANSWER_FORMATS)}")

    @property
    def format(self) -> AnswerFormat:
        return ANSWER_FORMATS[self.answer_format]

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.surface_options)

    def with_(self, **kw: Any) -> "Presentation":
        if "surface_options" in kw and isinstance(kw["surface_options"], dict):
            kw["surface_options"] = tuple(sorted(kw["surface_options"].items()))
        return replace(self, **kw)

    def key(self) -> str:
        """A short stable string identifying this presentation.

        Part of an episode's address, so it must not change meaning between
        versions; adding a field means adding it here too.
        """
        opts = ",".join(f"{k}={v}" for k, v in sorted(self.surface_options))
        return f"{self.language}/{self.answer_format}/{self.surface}" + (f"/{opts}" if opts else "")

    def to_dict(self) -> dict[str, Any]:
        return {"language": self.language, "answer_format": self.answer_format,
                "surface": self.surface, **({"surface_options": self.options}
                                            if self.surface_options else {})}

    @classmethod
    def parse(cls, spec: "str | Presentation | None") -> "Presentation":
        """Accept a presentation, a bare language code, or a ``key()`` string.

        A bare code is by far the commonest thing a caller has, and reading it as
        a language keeps every existing ``language="spanish"`` call site working.
        """
        if isinstance(spec, Presentation):
            return spec
        if spec is None:
            return DEFAULT_PRESENTATION
        parts = str(spec).split("/")
        out = cls(language=parts[0] or DEFAULT_LANGUAGE)
        if len(parts) > 1:
            out = out.with_(answer_format=parts[1])
        if len(parts) > 2:
            out = out.with_(surface=parts[2])
        if len(parts) > 3 and parts[3]:
            out = out.with_(surface_options=tuple(
                sorted(tuple(kv.split("=", 1)) for kv in parts[3].split(","))))
        return out

    def __str__(self) -> str:
        return self.key()


DEFAULT_PRESENTATION = Presentation()


def presentations(languages: Sequence[str] = (DEFAULT_LANGUAGE,),
                  formats: Sequence[str] | None = None,
                  surfaces: Sequence[str] = ("text",)) -> Iterator[Presentation]:
    """The cross-product of the three axes, in a stable order.

    The product is what makes the corpus large — but see ``INTENT.md`` on
    invariance sets versus bulk: emitting every presentation of *one* seed gives
    near-duplicate records, and is worth doing deliberately rather than by
    default.
    """
    formats = list(formats or ["inline_bare"])
    for language in languages:
        for fmt in formats:
            for surface in surfaces:
                yield Presentation(language=language, answer_format=fmt, surface=surface)
