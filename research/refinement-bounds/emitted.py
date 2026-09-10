"""The emitted body of the hot module `_m1` in each arm, and the `_canon_fn` it
falls back to.  Written out so RESULTS.md quotes generated source rather than
paraphrasing it."""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parents[1]), str(HERE.parents[1] / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms as ARMS                                     # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def body(source, fn):
    lines = source.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith("def %s(" % fn):
            j = i + 1
            while j < len(lines) and (lines[j].startswith("    ") or not lines[j].strip()):
                j += 1
            return "\n".join(lines[i:j]).rstrip()
    return "<%s not found>" % fn


def main():
    chunks = []
    for arm in ("A0", "B", "B+"):
        real, ib = ("B", True) if arm == "B+" else (arm, None)
        _f, res, _m = ARMS.compiled(real, seeds=(200,), inline_bounded=ib)
        chunks.append("=" * 72)
        chunks.append("arm %s   (%d source bytes)" % (arm, res.stats["source_bytes"]))
        chunks.append("=" * 72)
        chunks.append(body(res.source, "_m1"))
        for fn in ("_c3", "_c0"):
            b = body(res.source, fn)
            if not b.startswith("<"):
                chunks.append("")
                chunks.append(b)
        chunks.append("")
    text = "\n".join(chunks)
    (OUT / "emitted_m1.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
