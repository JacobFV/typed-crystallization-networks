"""Substitute the TABLE_* placeholders in RESULTS.md with tables.py output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tables

HERE = Path(__file__).resolve().parent
MAP = {"TABLE_RUNG1_LADDER": tables.rung1_ladder, "TABLE_RUNG1_GAP": tables.rung1_gap,
       "TABLE_RUNG1_FUTURE": tables.rung1_future, "TABLE_RUNG2": tables.rung2,
       "TABLE_RUNG3_WIDTH": tables.rung3_width, "TABLE_RUNG3_COLOUR": tables.rung3_colour,
       "TABLE_RUNG3_DENSE": tables.rung3_dense,
       "TABLE_RUNG3_DENSE_FREE": tables.rung3_dense_free, "TABLE_RUNG4": tables.rung4}

src = HERE / "RESULTS.template.md"
if not src.exists():
    src.write_text((HERE / "RESULTS.md").read_text())
text = src.read_text()
missing = [k for k in MAP if k in text and "missing" in MAP[k]()]
for key in sorted(MAP, key=len, reverse=True):
    fn = MAP[key]
    text = text.replace(key, fn())
(HERE / "RESULTS.md").write_text(text)
print("rendered; unresolved:", missing or "none")
