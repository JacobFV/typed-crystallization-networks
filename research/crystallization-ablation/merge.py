"""Merge extra-arm result files into the main per-configuration JSONL files."""
import json
import sys
from pathlib import Path

def merge(main, extra, arm):
    main, extra = Path(main), Path(extra)
    rows = [json.loads(l) for l in main.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["arm"] != arm]
    rows += [r for r in (json.loads(l) for l in extra.read_text().splitlines() if l.strip())
             if r["arm"] == arm]
    rows.sort(key=lambda r: (r["seed"], r["arm"]))
    main.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)

if __name__ == "__main__":
    print(merge(sys.argv[1], sys.argv[2], sys.argv[3]))
