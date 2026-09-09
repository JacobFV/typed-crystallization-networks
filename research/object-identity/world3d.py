"""Is a DIFFERENT generator's segmentation probe determined?

`geometry` re-draws every object's colour each episode and labels by list index,
so `object_ids` is undetermined (bounds.py).  `world_3d` and `world_2d` use the
same renderer but build their object list differently: the agent body and its
arm carry FIXED colours written into the generator, and only the free bodies are
re-drawn.  So the same probe (`agent_0/visible_ids`) may be partly determined
where `geometry`'s is not, and this measures exactly which part.
"""
from __future__ import annotations
import collections, copy, json, time
import numpy as np
import common2 as C
from common2 import OUT
import common as DP
from tcn.generation import Host, Action, Value, SCALAR
from tcn.types import product
from generators.geometry.render import render

R = 12
BODIES = 5
VEC3 = product(SCALAR, SCALAR, SCALAR)
LOOK = (0., -.3, -1.)


def episode(name, seed, split="train", steps=1):
    """One declared `look` action from the generator's own action schema."""
    h = Host.create(name, seed=seed, split=split,
                    configuration={"resolution": R, "bodies": BODIES, "horizon": 8})
    rec = h.records[-1]
    for _ in range(steps):
        rec = h.step([Action("look", arguments=(("direction", Value.of(VEC3, LOOK)),))])
    view = rec.actor_view()
    px = view.observations["pixels"].decoded[3]
    ids = rec.probes["agent_0/visible_ids"].decoded
    return px, ids, h


def object_table(h):
    s = h.state
    objects = []
    for o in s["objects"]:
        v = dict(o) | s["poses"][o["id"]]
        objects.append(v)
        if o.get("kind") == "robot":
            objects.append({"id": "arm_" + o["id"], "mesh": "box", "size": [.3, .08, .08],
                            "color": [180, 190, 200]} | s["poses"]["arm_" + o["id"]])
    visible = [o for o in objects if o["id"] != "agent_0"]
    return visible


def permutation_certificate(name, seeds=range(4)):
    rows = []
    for seed in seeds:
        _, _, h = episode(name, seed)
        s = h.state
        visible = object_table(h)
        pos = np.array(s["poses"]["agent_0"]["position"]); eye = pos + np.array([0, .3, .1])
        cam = {"eye": eye.tolist(), "target": (eye + np.array(s["agents"]["agent_0"]["look"])).tolist()}
        rgb0, _, ids0, _ = render(visible, cam, R, R)
        perm = list(range(len(visible)))[::-1]
        rgb1, _, ids1, _ = render([copy.deepcopy(visible[i]) for i in perm], cam, R, R)
        rows.append({"seed": seed, "n_visible": len(visible),
                     "ids": [o["id"] for o in visible],
                     "rgb_identical": bool(np.array_equal(rgb0, rgb1)),
                     "ids_identical": bool(np.array_equal(ids0, ids1))})
    return rows


def ceiling(name, train_seeds, held_seeds):
    def rows(seeds, split):
        out = []
        for s in seeds:
            px, ids, h = episode(name, s, split)
            for i in range(R * R):
                key = tuple(int(px[3 * i + k]) for k in range(3))
                out.append((key, int(ids[i])))
        return out
    tr, he = rows(train_seeds, "train"), rows(held_seeds, "test")
    res = {"id_histogram_train": dict(sorted(collections.Counter(y for _, y in tr).items())),
           "id_histogram_held": dict(sorted(collections.Counter(y for _, y in he).items()))}
    for label, f, restrict in (("visible_ids", lambda y: y, None),
                               ("is_arm(id==0)", lambda y: int(y == 0), None),
                               ("foreground", lambda y: int(y >= 0), None),
                               ("visible_ids | foreground", lambda y: y, "fg")):
        trr = [(k, y) for k, y in tr if restrict != "fg" or y >= 0]
        her = [(k, y) for k, y in he if restrict != "fg" or y >= 0]
        counts = collections.defaultdict(collections.Counter)
        overall = collections.Counter()
        for k, y in trr:
            counts[k][f(y)] += 1; overall[f(y)] += 1
        table = {k: v.most_common(1)[0][0] for k, v in counts.items()}
        fb = overall.most_common(1)[0][0] if overall else 0
        hc = collections.Counter(f(y) for _, y in her)
        base = hc.most_common(1)[0][1] / max(1, len(her))
        acc = sum(table.get(k, fb) == f(y) for k, y in her) / max(1, len(her))
        res[label] = {"keys": len(table), "records": len(her),
                      "colliding_keys": sum(1 for v in counts.values() if len(v) > 1),
                      "train_acc": sum(table[k] == f(y) for k, y in trr) / max(1, len(trr)),
                      "held_acc": acc, "majority_baseline": base, "advantage": acc - base,
                      "held_key_recurrence": sum(1 for k, _ in her if k in table) / max(1, len(her)),
                      "classes": len(hc)}
    return res


def cross(a, b):
    r1, g1, b1 = a; r2, g2, b2 = b
    return (abs(r1 * g2 - r2 * g1), abs(g1 * b2 - g2 * b1), abs(r1 * b2 - r2 * b1))


def collinearity_transfer(name, train_seeds, held_seeds, T=144):
    """Does the geometry-searched same-object relation hold in this generator too?

    The module is a fixed program -- offset +1 pixel, three cross products, one
    threshold -- and nothing about it is specific to `geometry`; both generators
    call the same renderer.  This applies it unchanged.
    """
    def ps(seeds, split):
        out = []
        for s in seeds:
            px, ids, _ = episode(name, s, split)
            for i in range(R * R - R):
                a = tuple(int(px[3 * i + k]) for k in range(3))
                b = tuple(int(px[3 * (i + 1) + k]) for k in range(3))
                out.append((a, b, int(ids[i]) == int(ids[i + 1])))
        return out
    tr, he = ps(train_seeds, "train"), ps(held_seeds, "test")
    shapes = {"and_of_three": lambda d, t: all(x <= t for x in d),
              "searched": lambda d, t: (d[0] <= t or d[1] <= t) and d[2] <= t}
    def acc(rows, t, rule):
        return sum(rule(cross(a, b), t) == y for a, b, y in rows) / max(1, len(rows))
    base = collections.Counter(y for _, _, y in he).most_common(1)[0][1] / len(he)
    out = {"pairs_train": len(tr), "pairs_held": len(he), "majority_baseline": base,
           "transferred_T": T}
    for name, rule in shapes.items():
        out[name] = {"train_acc_at_T": acc(tr, T, rule), "held_acc_at_T": acc(he, T, rule),
                     "held_exact_intervals": [t for t in range(300)
                                              if acc(he, t, rule) == 1.0][:1]}
    out["held_acc_at_T"] = out["searched"]["held_acc_at_T"]
    out["train_acc_at_T"] = out["searched"]["train_acc_at_T"]
    out["best_T_on_train"] = T
    out["held_acc_at_best"] = out["searched"]["held_acc_at_T"]
    return out


def main():
    t0 = time.perf_counter()
    out = {"resolution": R}
    for name in ("world_3d", "world_2d"):
        perm = permutation_certificate(name)
        cel = ceiling(name, range(0, 8), range(100, 108))
        out[name] = {"permutation_certificate": perm, "ceiling": cel}
        DP.report(f"[{name}] permutation rgb identical / ids identical",
                  f"{sum(r['rgb_identical'] for r in perm)}/{len(perm)} / "
                  f"{sum(r['ids_identical'] for r in perm)}/{len(perm)}")
        DP.report(f"[{name}] visible list", perm[0]["ids"])
        tr = collinearity_transfer(name, range(0, 8), range(100, 108))
        out[name]["collinearity_transfer"] = tr
        DP.report(f"[{name}] same-object relation transferred at T=144: base / AND-of-three / searched shape",
                  f"{tr['majority_baseline']:.4f} / {tr['and_of_three']['held_acc_at_T']:.6f} / "
                  f"{tr['searched']['held_acc_at_T']:.6f}")
        for k, v in cel.items():
            if not (isinstance(v, dict) and "train_acc" in v):
                DP.report(f"[{name}] {k}", v); continue
            DP.report(f"[{name}] {k}",
                      f"train={v['train_acc']:.4f} held={v['held_acc']:.4f} "
                      f"base={v['majority_baseline']:.4f} adv={v['advantage']:+.4f} "
                      f"recur={v['held_key_recurrence']:.3f}")
    out["seconds"] = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "world3d.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote", OUT / "world3d.json")


if __name__ == "__main__":
    main()
