"""Print the rows RESULTS.md quotes, straight from `out/*.json`.

Every table in RESULTS.md is transcribed from this script's output rather than
from a scrollback, so a reader can regenerate the document's numbers with one
command and diff them.

Run: `.venv/bin/python research/neural-baselines/summarise.py`
"""
from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "out"


def load(name):
    path = OUT / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def language():
    rule("LANGUAGE")
    t = load("tcn_language")
    if t:
        print(f"TCN budget: {t['budget']}")
        print(f"TCN size:   {t['size']}")
        print(f"TCN latency:{t['latency']}  rss {t['peak_rss_mb']:.1f} MB")
        for k, q in t["quality"].items():
            print(f"  TCN {k:22s} acc {q['accuracy']:.4f}  majority {q['majority_constant']:.4f} "
                  f" counting-oracle {q['oracle_counting']:.4f}  dyck-oracle {q['oracle_dyck']:.4f}"
                  f"  per_length {q['per_length']}")
    n = load("language")
    if not n:
        return
    print(f"\nsplits: " + json.dumps(n["splits"]))
    print(f"{'budget':16s} {'aux':4s} {'model':17s} {'params':>7s} "
          f"{'train':>6s} {'val':>6s} {'test':>6s} {'test_lo':>7s} {'test_hi':>7s} "
          f"{'current':>7s} {'sec':>6s}")
    for a in n["arms"]:
        print(f"{a['budget']:16s} {int(a['aux']):<4d} {a['model']:17s} {a['parameters']:7d} "
              f"{a['train_acc_median']:6.3f} {a['val_acc_median']:6.3f} "
              f"{a['test_acc_median']:6.3f} {a['test_acc_min']:7.3f} {a['test_acc_max']:7.3f} "
              f"{a.get('current_stream_acc_median', float('nan')):7.3f} "
              f"{a['train_seconds_median']:6.1f}")
    if "structural_hint_counting" in n:
        h = n["structural_hint_counting"]
        print(f"\nSTRUCTURAL HINT (not matched): counting bag-of-bytes, "
              f"{h['parameters']} params, test median {h['test_acc_median']:.4f}, "
              f"current stream {h.get('current_stream_acc_median')}")
    print("\nselection: " + json.dumps(n.get("selection"), indent=1))


def visual():
    rule("VISUAL")
    t = load("tcn_visual")
    if t:
        print(f"TCN budget: {t['budget']}")
        print(f"TCN size:   {t['size']}")
        print(f"TCN latency:{t['latency']}  rss {t['peak_rss_mb']:.1f} MB")
        print(f"TCN totals: {t['quality']['totals']}")
    b = load("visual_bounds")
    if b:
        for k, v in b["bounds"].items():
            print(f"  bound {k:12s} " + json.dumps(v))
        print(f"  plain python rule: {b['plain_python_rule_on_heldout']}")
    n = load("visual")
    if not n:
        return
    print(f"\ntrivial reference (test): {n['trivial_reference_test']}")
    print(f"{'budget':26s} {'model':14s} {'params':>7s} {'val_link':>8s} "
          f"{'rect':>6s} {'link':>6s} {'trees':>6s} {'cP':>5s} {'cR':>5s} {'ext':>5s} {'sec':>6s}")
    for a in n["arms"]:
        row = a["rows"][0]["test"]
        print(f"{a['budget']:26s} {a['model']:14s} {a['parameters']:7d} "
              f"{a['val_link_accuracy_median']:8.3f} "
              f"{a['test_rect_accuracy_median']:6.3f} {a['test_link_accuracy_median']:6.3f} "
              f"{a['test_trees_exact_median']:6.1f} "
              f"{row['corner_precision']:5.2f} {row['corner_recall']:5.2f} "
              f"{row['extent_accuracy']:5.2f} {a['train_seconds_median']:6.1f}")
    g = load("visual_augmented")
    if g:
        print("\nSTRUCTURAL HINT (not matched): colour-bijection augmentation")
        for a in g["arms"]:
            print(f"  {a['train_screens']:4d} screens {a['model']:14s} "
                  f"link {a['test_link_accuracy_median']:.3f} rect "
                  f"{a['test_rect_accuracy_median']:.3f} trees "
                  f"{a['test_trees_exact_median']:.1f} cP {a['corner_precision_median']:.2f} "
                  f"cR {a['corner_recall_median']:.2f} ext {a['extent_accuracy_median']:.2f}")


def computer():
    rule("COMPUTER")
    t = load("tcn_computer")
    if t:
        print(f"TCN budget: {t['budget']}")
        print(f"TCN size:   {t['size']}")
        print(f"TCN latency:{t['latency']}  rss {t['peak_rss_mb']:.1f} MB")
        print(f"TCN quality: solved {t['quality']['solved']}/{t['quality']['episodes']}, "
              f"mean return {t['quality']['mean_return']}/2")
    n = load("computer")
    if not n:
        return
    print(f"\nbudget: {n['budget']}")
    print(f"padding check: {n['padding_check']}")
    print(f"supervised trivial reference: {n['supervised_trivial_reference']}")
    for r in n["trivial_reference_closed_loop"]:
        print(f"  trivial {r['label']:38s} solved {r['solved']}/{r['episodes']} "
              f"mean return {r['mean_return']:.2f}/2")
    print(f"\n{'kind':16s} {'model':18s} {'byte':4s} {'params':>7s} {'tr_verb':>7s} "
          f"{'tr_byte':>7s} {'ho_byte':>7s} {'solved':>6s} {'ret':>5s} {'ms':>7s}")
    for a in n["arms"]:
        print(f"{a['kind']:16s} {a['model']:18s} {a['byte_head']:4s} {a['parameters']:7d} "
              f"{a['train_verb_accuracy_median']:7.3f} {a['train_byte_accuracy_median']:7.3f} "
              f"{a['heldout_byte_accuracy_median']:7.3f} "
              f"{a.get('solved_median', float('nan')):6} "
              f"{a.get('mean_return_median', float('nan')):5} "
              f"{a.get('policy_ms_median', float('nan')):7}")
    print("\nselection: " + json.dumps(n.get("selected_on_training_fit"), indent=1))


def latency():
    d = load("latency")
    if not d:
        return
    rule("LATENCY / SIZE (neural architectures, measured back to back)")
    print(f"{'arm':10s} {'model':18s} {'params':>7s} {'MACs':>10s} {'torch_save':>10s} "
          f"{'f32':>8s} {'gzip':>8s} {'cold_ms':>8s} {'warm_p50':>9s} {'warm_min':>9s} {'warm_p95':>9s}")
    for r in d["in_process"]:
        print(f"{r['arm']:10s} {r['model']:18s} {r['parameters']:7d} "
              f"{r.get('macs_per_inference', 0):10d} "
              f"{r['torch_save_bytes']:10d} {r['float32_bytes']:8d} {r['gzip_bytes']:8d} "
              f"{r['cold_first_call_ms']:8.3f} {r['warm_p50_ms']:9.4f} "
              f"{r.get('warm_min_ms', float('nan')):9.4f} {r['warm_p95_ms']:9.4f}")
    print("\ncold process (fresh interpreter, imports torch, one inference):")
    for r in d["cold_process"]:
        print("  " + json.dumps(r))


if __name__ == "__main__":
    language()
    visual()
    computer()
    latency()
