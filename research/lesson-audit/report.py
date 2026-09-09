"""Rank lessons by exploitability and print the audit table."""
from __future__ import annotations
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

FAMILY = {
    "constant": "constant", "const_choice_index": "constant",
    "choice_by_length": "distractor", "choice_length_outlier": "distractor",
    "choice_global_prior": "distractor", "choice_ranker": "distractor",
    "choice_in_observation": "copy", "copy_salient_token": "copy",
    "copy_fixed_offset": "copy", "copy_near_keyword": "copy",
    "prompt_length": "surface-statistic", "byte_at_offset": "surface-statistic",
    "bag_of_chars": "surface-statistic", "char_count_tree": "surface-statistic",
    "nearest_neighbour": "memorisation", "train_lookup": "memorisation",
}


def rows(d):
    out = []
    for lid, r in d["lessons"].items():
        if "exploits" not in r:
            continue
        ex = {k: v for k, v in r["exploits"].items()
              if not k.endswith("_error") and v == v}
        st = r["stats"]
        base = max(st["majority_share"], st["random_floor"])
        out.append({
            "lesson": lid, "oracle": r["oracle"], "best": r["best_score"],
            "by": r["best_exploit"], "family": FAMILY.get(r["best_exploit"], "?"),
            "gap": round(r["oracle"] - r["best_score"], 4),
            "baseline": round(base, 4),
            "lift": round(r["best_score"] - base, 4),
            "n_answers": st["distinct_answers"], "H": st["answer_entropy_bits"],
            "k": st["mean_choices"], "maj": st["majority_share"],
            "copy_rate": st["answer_in_observation"],
            "repeat": st["repeat_observation_rate"],
            "exploits": ex,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=os.path.join(HERE, "audit_before.json"))
    p.add_argument("--md", default="")
    p.add_argument("--top", type=int, default=0)
    a = p.parse_args()
    d = json.load(open(a.json))
    rs = sorted(rows(d), key=lambda r: (-r["best"], r["gap"]))
    if a.top:
        rs = rs[:a.top]
    hdr = ("| # | lesson | oracle | best cheap exploit | score | gap | baseline | lift | "
           "family | #ans | H(bits) | copy | repeat |")
    lines = [hdr, "|" + "---|" * 13]
    for i, r in enumerate(rs, 1):
        lines.append(
            f"| {i} | `{r['lesson']}` | {r['oracle']:.3f} | `{r['by']}` | "
            f"**{r['best']:.3f}** | {r['gap']:.3f} | {r['baseline']:.3f} | "
            f"{r['lift']:+.3f} | {r['family']} | {r['n_answers']} | {r['H']:.2f} | "
            f"{r['copy_rate']:.2f} | {r['repeat']:.2f} |")
    txt = "\n".join(lines)
    if a.md:
        open(a.md, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
