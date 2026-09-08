"""A formal language curriculum: 180 lessons, procedurally generated, exactly graded.

The premise is that "language ability" is not one number obtained by predicting
tokens. It is a profile over capabilities that build on each other — reference,
equivalence, discrimination, sequence memory, finite-state syntax, recursion,
parsing, variable binding, unification, quantification, compositional reference,
and upward from there through causality, ontology, scientific induction, proof,
argument, self-modeling and value inference.

Every lesson is a **generator**, not a dataset. An episode is a pure function of
a seed: the vocabulary is invented per episode, the grammar or ontology or proof
calculus is sampled fresh, and the answer is computed from the construction
rather than annotated by anyone. Two consequences follow, and they are the whole
reason the resource exists:

* **An evaluation set cannot have been trained on.** It did not exist until you
  asked for it. There is no contamination question to argue about.
* **Every score has a floor.** The answer set travels with the episode, so
  "what does knowing nothing get?" is computable per lesson, and an accuracy is
  never reported without it.

Lessons are flat and know nothing about ordering. How they fit together is the
business of a :class:`~langcurriculum.curricula.Curriculum`, of which there may
be many, overlapping and disagreeing. How an episode is *shown* — language,
answer format, modality — is the business of a
:class:`~langcurriculum.presentation.Presentation`. Neither is visible to a
generator, and that separation is what makes it measurable whether a system
learned the problem or the surface. See ``INTENT.md``.

Everything crossing this API is plain text or plain data — strings, dicts,
lists, numbers. Point it at anything that maps a string to a string.

    >>> import langcurriculum as lc
    >>> ex = lc.get("symbol_grounding").example(seed=0)
    >>> ex.prompt.splitlines()[-1]
    'Reply with the answer only.'
    >>> lc.curriculum("core170").linearize()[0].lesson
    'symbol_grounding'
    >>> report = lc.evaluate(lambda prompt: "o0", n=20)   # doctest: +SKIP
    >>> print(report.table())                             # doctest: +SKIP
"""

from __future__ import annotations

from .context import GenerationContext
from .curricula import Curriculum, Node, curricula, curriculum_ids
from .curricula import get as curriculum
from .dataset import export, iter_records, read_jsonl, splits, write_jsonl
from .evaluate import (LessonResult, Report, TextAgent, constant_agent, evaluate,
                       evaluate_lesson, random_agent)
from .lesson import (AXES, CORE_AXES, EXTENDED_AXES, Example, Lesson,
                     LessonNotImplemented, instance_id)
from .presentation import ANSWER_FORMATS, Presentation, presentations
from .registry import (REGISTRY, all_lessons, by_capability, by_tag, capabilities,
                       get, lesson_ids, resolve, tags)
from .scoring import extract_choice, normalize, score
from .languages import (DEFAULT_LANGUAGE, LANGUAGES, Language, Lexicon,
                        get_language, language_codes, languages,
                        register_language)
from .address import Address, Space, batch, draw
from .store import CachedRenderer, LocalStore, S3Store, store_from_env
from .surfaces import (NATIVE_SURFACES, SURFACES, Content, render_native,
                       renders_natively, transcode, transcode_example)
from .verify import verify_all, verify_lesson

__version__ = "0.2.0"

#: lessons in the registry
N_REGISTERED = 180
#: lessons in the ``core170`` curriculum
N_NUMBERED = 170

__all__ = [
    "__version__", "N_NUMBERED", "N_REGISTERED",
    # lessons
    "Lesson", "Example", "AXES", "CORE_AXES", "EXTENDED_AXES", "LessonNotImplemented",
    "instance_id", "GenerationContext",
    # curricula
    "Curriculum", "Node", "curriculum", "curricula", "curriculum_ids",
    # presentation
    "Presentation", "ANSWER_FORMATS", "presentations",
    "SURFACES", "NATIVE_SURFACES", "Content", "transcode", "transcode_example",
    "render_native", "renders_natively",
    # addressing and caching
    "Address", "Space", "batch", "draw",
    "CachedRenderer", "LocalStore", "S3Store", "store_from_env",
    # languages
    "DEFAULT_LANGUAGE", "LANGUAGES", "Language", "Lexicon",
    "get_language", "language_codes", "languages", "register_language",
    # registry
    "REGISTRY", "all_lessons", "get", "lesson_ids", "by_tag", "by_capability",
    "tags", "capabilities", "resolve",
    # evaluation
    "evaluate", "evaluate_lesson", "Report", "LessonResult", "TextAgent",
    "random_agent", "constant_agent", "score", "normalize", "extract_choice",
    # datasets
    "iter_records", "write_jsonl", "read_jsonl", "export", "splits",
    # verification
    "verify_lesson", "verify_all",
]
