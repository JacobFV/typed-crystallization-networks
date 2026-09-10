"""Recompute `ablation_summary` in the stage reports with the two-verdict rule.

`ablate.summarize` originally reported only the literal pre-registered reading
("exactly one construct's removal prevents the result").  Stage 2 falsified the
presumption behind that phrasing -- two different constructs each reach the win
-- so the summary was widened, and this rewrites the stored summaries with the
wider rule rather than re-running the searches.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablate


def main():
    for tag in ("stage1", "stage2", "stage3"):
        p = HERE / "out" / ("%s.json" % tag)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if "ablation_matrix" not in d:
            continue
        joint = d.get("ablation_no_find_no_scan")
        if joint is not None:
            joint = dict(joint)
            joint["blocked"] = not joint.get("solved")
        d["ablation_summary"] = ablate.summarize(d["ablation_matrix"], joint=joint)
        p.write_text(json.dumps(d, indent=2, default=str))
        print(tag, json.dumps(d["ablation_summary"], default=str))

if __name__ == "__main__":
    main()
