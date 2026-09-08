"""The language database: millions of lexical entries, on disk, indexed.

The previous design held every vocabulary in memory as a dict loaded from JSON.
That is the right shape for 321 words and the wrong shape for what this package
now carries: **56.7 million inflected forms** — 13.0M from UniMorph across 170
languages and 43.7M from Wiktionary's inflection tables across 30 — plus three
million translations. Loading that eagerly would cost gigabytes
of resident memory to answer questions about one language.

So the lexicon lives in SQLite. Three properties made it the obvious choice and
none of them are about performance:

* it is in the **standard library**, so the package keeps its promise of zero
  runtime dependencies while shipping a dataset three orders of magnitude larger
  than before;
* it is **indexed**, so looking up one lemma in a ten-million-row table is a
  B-tree descent rather than a scan;
* it is **lazy**, so a process that only ever renders Turkish never reads a byte
  of Finnish.

Provenance is a column, not a footnote
--------------------------------------

Every row records where it came from. A benchmark whose vocabulary is partly
curated, partly scraped and partly inferred is only trustworthy if a reader can
ask which is which, per entry, and get an answer. :meth:`LanguageDB.provenance`
is that answer, and the coverage tests assert against it rather than against a
number someone typed into a docstring.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = ["LanguageDB", "DB_PATH", "SCHEMA", "open_db", "Entry"]

#: where the built database lives. Overridable for tests and for callers who
#: keep large data outside the package directory.
DB_PATH = Path(os.environ.get(
    "LANGCURRICULUM_DB",
    Path(__file__).resolve().parent / "data" / "languages.db"))


SCHEMA = """
PRAGMA journal_mode = WAL;

-- one row per language the package knows anything at all about
CREATE TABLE IF NOT EXISTS language (
    code        TEXT PRIMARY KEY,   -- ISO 639-3, the join key everywhere
    name        TEXT NOT NULL,
    autonym     TEXT,               -- what its speakers call it
    glottocode  TEXT,
    family      TEXT,
    macroarea   TEXT,
    script      TEXT,
    rtl         INTEGER DEFAULT 0,
    tier        INTEGER DEFAULT 4,  -- see LanguageDB.TIERS; 4 until
                                    -- something is actually known
    n_senses    INTEGER DEFAULT 0,
    n_forms     INTEGER DEFAULT 0,
    n_typology  INTEGER DEFAULT 0
);

-- raw typological coding, one row per (language, feature)
CREATE TABLE IF NOT EXISTS typology (
    code    TEXT NOT NULL,
    param   TEXT NOT NULL,
    value   TEXT NOT NULL,
    label   TEXT,
    source  TEXT NOT NULL,
    PRIMARY KEY (code, param, source)
) WITHOUT ROWID;

-- the derived engine parameters, as JSON, one row per language
CREATE TABLE IF NOT EXISTS profile (
    code    TEXT PRIMARY KEY,
    params  TEXT NOT NULL,
    derived_from TEXT NOT NULL
);

-- an English concept realized in a target language
CREATE TABLE IF NOT EXISTS sense (
    code    TEXT NOT NULL,          -- target language
    key     TEXT NOT NULL,          -- the English lemma the curriculum coins
    pos     TEXT,                   -- noun / adj / verb / other
    form    TEXT NOT NULL,          -- the target-language word
    gender  TEXT,                   -- where the source records one
    -- 0 is the primary sense. Ranked by how many languages translate the
    -- sense: Wiktionary lists "big = grown-up" before "big = large", and
    -- taking its order verbatim gives every language the wrong word.
    rank    INTEGER DEFAULT 0,
    source  TEXT NOT NULL
);

-- an inflected form: the paradigm cell, from UniMorph
CREATE TABLE IF NOT EXISTS wordform (
    code    TEXT NOT NULL,
    lemma   TEXT NOT NULL,
    surface TEXT NOT NULL,
    feats   TEXT NOT NULL,          -- raw UniMorph bundle, semicolon separated
    pos     TEXT,
    source  TEXT NOT NULL
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS sense_lookup    ON sense (code, key, rank);
CREATE INDEX IF NOT EXISTS sense_reverse   ON sense (code, form);
CREATE INDEX IF NOT EXISTS wordform_lemma  ON wordform (code, lemma);
CREATE INDEX IF NOT EXISTS wordform_surface ON wordform (code, surface);
CREATE INDEX IF NOT EXISTS typology_code   ON typology (code);
"""


@dataclass(frozen=True)
class Entry:
    """One lexical entry, with the provenance that makes it checkable."""

    form: str
    pos: str = ""
    gender: str = ""
    source: str = ""


def open_db(path: str | os.PathLike[str] | None = None, *,
            create: bool = False) -> sqlite3.Connection:
    """Open the database read-only unless a builder asks to write."""
    target = Path(path or DB_PATH)
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target)
        conn.executescript(SCHEMA)
        return conn
    if not target.exists():
        raise FileNotFoundError(
            f"no language database at {target}. Build it with "
            f"`python scripts/build_langdb.py`, which downloads UniMorph, "
            f"Wiktextract and WALS and takes a few minutes.")
    # immutable=1 lets several processes share one file with no locking at all
    conn = sqlite3.connect(f"file:{target}?immutable=1", uri=True)
    conn.execute("PRAGMA query_only = 1")
    return conn


class LanguageDB:
    """Read access to the language database, one connection per thread.

    SQLite connections are not shareable across threads, and the curriculum is
    routinely driven from a thread pool when generating data, so the connection
    is thread-local rather than a single shared handle.
    """

    #: what the tiers mean. Asserted by the test suite against the real counts,
    #: so a language cannot claim a tier its data does not support.
    TIERS = {
        1: "hand-written grammar, verified lexicon and morphology",
        2: "derived grammar, real imported lexicon and morphology",
        3: "derived grammar, real lexicon, no morphology",
        4: "derived grammar only — typology but no lexicon",
    }

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path or DB_PATH)
        self._local = threading.local()
        self._scripts: dict[str, str] = {}
        self._stress: dict[str, bool] = {}

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = open_db(self.path)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def exists(self) -> bool:
        return self.path.exists()

    # ---- languages ------------------------------------------------------
    def languages(self, *, min_tier: int = 4) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM language WHERE tier <= ? ORDER BY tier, n_senses DESC",
            (min_tier,)))

    def language(self, code: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM language WHERE code = ?", (code,)).fetchone()

    def codes(self, *, min_tier: int = 4) -> list[str]:
        return [r["code"] for r in self.languages(min_tier=min_tier)]

    # ---- typology -------------------------------------------------------
    def profile(self, code: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT params FROM profile WHERE code = ?", (code,)).fetchone()
        return json.loads(row["params"]) if row else {}

    def typology(self, code: str) -> dict[str, str]:
        return {r["param"]: r["value"] for r in self.conn.execute(
            "SELECT param, value FROM typology WHERE code = ?", (code,))}

    # ---- lexicon --------------------------------------------------------
    def script_of(self, code: str) -> str:
        """The script a language is written in, cached; "" if unrecorded."""
        if code not in self._scripts:
            row = self.language(code)
            self._scripts[code] = (row["script"] or "") if row is not None else ""
        return self._scripts[code]

    def marks_stress(self, code: str) -> bool:
        """Whether this language's paradigms write the stress mark its
        dictionary writes.

        Russian dictionaries print число́ for the reader's benefit and UniMorph
        keys on число, so every Russian paradigm lookup missed and every noun
        fell back to an analogical rule: the plural of число́ came out числои,
        which is not a word. 83% of Russian senses carry the mark and none of
        its 2.2 million paradigm rows do; Ukrainian, Belarusian, Bulgarian and
        Macedonian are the same.

        Asked of the data rather than answered from a list, because the mark
        is not always an aid to the reader. Navajo writes it for tone, it is
        part of the spelling, and 790 of its paradigm rows carry it -- strip
        it there and the word is a different word. Languages with no paradigm
        table say nothing either way and are left alone.
        """
        if code not in self._stress:
            rows = self.conn.execute(
                "SELECT COUNT(*), SUM(lemma LIKE '%'||char(769)||'%') "
                "FROM wordform WHERE code=?", (code,)).fetchone()
            total, marked = rows[0] or 0, rows[1] or 0
            self._stress[code] = not (total >= 1000 and marked == 0)
        return self._stress[code]

    def _unstressed(self, code: str, form: str) -> str:
        return form if self.marks_stress(code) else form.replace("\u0301", "")

    def _one(self, code: str, form: str) -> str:
        """One writing of a Han-script entry, not both at once.

        Wiktionary gives a Han headword as "Traditional /Simplified" --
        *詞典 /词典* for dictionary -- and 61,798 rows are like that across
        Mandarin, Cantonese, Hakka, Wu and Min. Printed whole it is a word in
        neither: Mandarin read "o0是黃 /黄立方體 /立方体".

        Split here rather than in one grammar. The derived grammars did it for
        themselves and the hand-written Chinese pack did not, so a word the
        pack happened to lack came back through the store unsplit -- 門檻 /门槛
        for a threshold, spaces and all, in a language written without them.
        """
        form = self._unstressed(code, form)
        if " /" not in form:
            return form
        script = self.script_of(code)
        if script not in ("Hans", "Hant"):
            # Hakka records both writings and is itself written in Latin.
            # Choosing one hands a romanised language a Han character, where
            # leaving the pair intact lets the usability screen reject it and
            # `gaps` say there is no single word for the copula -- which is
            # what it did before, and is true.
            return form
        left, _, right = form.partition(" /")
        return left if script == "Hant" else right

    def lookup(self, code: str, key: str, pos: str = "") -> Entry | None:
        """The best target word for an English concept.

        "Best" is the first row, and the builder writes rows in source-priority
        order, so a curated entry outranks a scraped one without the reader
        needing to sort at query time.
        """
        if pos:
            row = self.conn.execute(
                "SELECT * FROM sense WHERE code=? AND key=? AND pos=? "
                "ORDER BY rank LIMIT 1",
                (code, key, pos)).fetchone()
            if row is not None:
                return Entry(self._one(code, row["form"]), row["pos"] or "",
                             row["gender"] or "", row["source"])
        row = self.conn.execute(
            "SELECT * FROM sense WHERE code=? AND key=? ORDER BY rank LIMIT 1",
            (code, key)).fetchone()
        return None if row is None else Entry(
            self._one(code, row["form"]), row["pos"] or "", row["gender"] or "",
            row["source"])

    def lookup_all(self, code: str, key: str) -> list[Entry]:
        return [Entry(self._one(code, r["form"]), r["pos"] or "", r["gender"] or "",
                      r["source"])
                for r in self.conn.execute(
                    "SELECT * FROM sense WHERE code=? AND key=?", (code, key))]

    def bulk_lookup(self, code: str, keys: Sequence[str]) -> dict[str, Entry]:
        """Every key at once — one query instead of several hundred.

        A grammar warms its whole working vocabulary on first use, and doing that
        key by key turns startup into thousands of round trips for no reason.
        """
        if not keys:
            return {}
        out: dict[str, Entry] = {}
        chunk = 900                                   # under SQLite's variable cap
        for i in range(0, len(keys), chunk):
            part = list(keys[i:i + chunk])
            marks = ",".join("?" * len(part))
            for r in self.conn.execute(
                    f"SELECT key, form, pos, gender, source FROM sense "
                    f"WHERE code=? AND key IN ({marks}) ORDER BY rank", (code, *part)):
                out.setdefault(r["key"], Entry(r["form"], r["pos"] or "",
                                               r["gender"] or "", r["source"]))
        return out

    # ---- morphology -----------------------------------------------------
    def paradigm(self, code: str, lemma: str,
                 source: str | None = None) -> list[tuple[str, str]]:
        """Every attested (features, surface) cell for one lemma.

        ``source`` restricts to one annotation scheme. The two the database
        carries disagree about how much they say: UniMorph tags a Swedish
        comparative ``ADJ;CMPR;NEUT;SG;INDF`` and Wiktionary emits the same form
        as ``A;INDF;NEUT;SG`` with the degree dropped. Pooled, the untagged row
        answers a request for the plain adjective and *gult* comes back
        *gulare*. Read one scheme at a time and each is internally consistent.
        """
        # UniMorph leaves the lemma unmarked and marks the surface -- 1.65
        # million of Russian's 2.2 million rows carry the stress mark. Strip it
        # here as well, or a sentence reads "число" in the singular and
        # "чи́сла" in the plural, and neither ordinary Russian nor the
        # dictionary writes it both ways.
        rows = self.conn.execute(
            "SELECT feats, surface FROM wordform WHERE code=? AND lemma=?",
            (code, lemma)) if source is None else self.conn.execute(
            "SELECT feats, surface FROM wordform "
            "WHERE code=? AND lemma=? AND source=?", (code, lemma, source))
        return [(r["feats"], self._unstressed(code, r["surface"])) for r in rows]

    def has_source(self, code: str, source: str, pos: str) -> bool:
        """Whether a language has enough of one scheme to be worth preferring."""
        return bool(self.conn.execute(
            "SELECT 1 FROM wordform WHERE code=? AND source=? AND pos=? LIMIT 1",
            (code, source, pos)).fetchone())

    def surface_forms(self, code: str, lemma: str) -> set[str]:
        return {s for _, s in self.paradigm(code, lemma)} | {lemma}

    def sample_paradigms(self, code: str, *, pos: str = "N",
                         limit: int = 400) -> Iterator[tuple[str, list[tuple[str, str]]]]:
        """Whole paradigms, grouped, for the morphology inducer to generalize over."""
        rows = self.conn.execute(
            "SELECT lemma, feats, surface FROM wordform "
            "WHERE code=? AND pos=? ORDER BY lemma LIMIT ?",
            (code, pos, limit * 40))
        current, cells = None, []
        seen = 0
        for r in rows:
            if r["lemma"] != current:
                if current is not None and cells:
                    yield current, cells
                    seen += 1
                    if seen >= limit:
                        return
                current, cells = r["lemma"], []
            cells.append((r["feats"], r["surface"]))
        if current is not None and cells:
            yield current, cells

    # ---- provenance and coverage ----------------------------------------
    def provenance(self, code: str) -> dict[str, int]:
        """How many entries came from each source, for one language."""
        out: dict[str, int] = {}
        for table in ("sense", "wordform", "typology"):
            for r in self.conn.execute(
                    f"SELECT source, COUNT(*) n FROM {table} WHERE code=? "
                    f"GROUP BY source", (code,)):
                out[f"{table}:{r['source']}"] = r["n"]
        return out

    def totals(self) -> dict[str, int]:
        cur = self.conn.execute(
            "SELECT (SELECT COUNT(*) FROM language) languages,"
            " (SELECT COUNT(*) FROM sense) senses,"
            " (SELECT COUNT(*) FROM wordform) wordforms,"
            " (SELECT COUNT(*) FROM typology) typology,"
            " (SELECT COUNT(*) FROM profile) profiles")
        return dict(cur.fetchone())

    def __repr__(self) -> str:
        if not self.exists():
            return f"<LanguageDB (not built) {self.path}>"
        t = self.totals()
        return (f"<LanguageDB {t['languages']} languages, {t['senses']:,} senses, "
                f"{t['wordforms']:,} forms>")
