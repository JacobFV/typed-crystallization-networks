"""Two-sided Fisher exact test (no scipy dependency) for the arm success rates."""
from math import comb


def fisher_two_sided(a, b, c, d):
    """Table [[a, b], [c, d]] = [[succ1, fail1], [succ2, fail2]]."""
    n = a + b + c + d
    def p(x):
        return comb(a + b, x) * comb(c + d, a + c - x) / comb(n, a + c)
    lo, hi = max(0, a + c - (c + d)), min(a + b, a + c)
    p0 = p(a)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 + 1e-12)


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    HERE = Path(__file__).parent
    out = {}
    e1 = HERE / "e1_results.json"
    ctl = HERE / "control_results.json"
    e2 = HERE / "e2_results.json"

    def rates(path, arm=None, runs=None):
        d = json.loads(Path(path).read_text())
        rs = runs if runs is not None else [r for r in d["runs"] if r["arm"] == arm]
        s = sum(1 for r in rs if r["final_conformant"])
        return s, len(rs) - s

    if e2.exists():
        a = rates(e2, "A"); b = rates(e2, "B")
        out["e2_A_vs_B"] = dict(A=a, B=b, p=round(fisher_two_sided(*a, *b), 4))
    if e1.exists():
        a = rates(e1, "A"); b = rates(e1, "B")
        out["e1_A_vs_B"] = dict(A=a, B=b, p=round(fisher_two_sided(*a, *b), 4))
        if ctl.exists():
            c = rates(ctl, runs=json.loads(ctl.read_text())["runs"])
            out["e1_A_vs_C"] = dict(A=a, C=c, p=round(fisher_two_sided(*a, *c), 4))
            out["e1_B_vs_C"] = dict(B=b, C=c, p=round(fisher_two_sided(*b, *c), 4))
    print(json.dumps(out, indent=2))
    (HERE / "fisher.json").write_text(json.dumps(out, indent=2))
