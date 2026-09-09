"""Print the recorded rows of a track output file as a table, for the writeup."""
from __future__ import annotations

import json
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parent / 'out'


def show(path, key='rows', cols=None):
    d = json.loads(pathlib.Path(path).read_text())
    rows = d.get(key) or []
    if not rows:
        print(path, 'has no', key)
        return d
    cols = cols or [c for c in rows[0] if not isinstance(rows[0][c], (dict, list))]
    print(' | '.join(cols))
    for r in rows:
        print(' | '.join(str(r.get(c, '')) for c in cols))
    return d


if __name__ == '__main__':
    p = sys.argv[1]
    if not pathlib.Path(p).exists():
        p = str(OUT / (p + '.json'))
    show(p, sys.argv[2] if len(sys.argv) > 2 else 'rows',
         sys.argv[3].split(',') if len(sys.argv) > 3 else None)
