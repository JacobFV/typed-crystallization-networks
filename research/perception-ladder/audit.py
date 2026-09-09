"""Expressibility and information audits used throughout RESULTS.md.

Nothing here trains anything; it interrogates `tcn/operators.py`, `tcn/types.py`
and the generators to establish what a program *could* compute at all.
"""
from __future__ import annotations
import sys, json, collections, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tcn.generation import Host, Action, Value, BYTE, SCALAR
from tcn.operators import Registry
from tcn.types import Type, Encoding, product, integer, setof, BOOL
from tcn.graph import Program, Node, Candidate
from tcn.learning import relaxed
from common import dump

R = Registry()


def legality(name, args, out=None, params=None):
    try:
        op = R.resolve(name, args, out, params)
        return {"legal": True, "gradient": op.gradient, "output": op.output.kind,
                "output_role": op.output.role, "output_bits": op.output.bits}
    except Exception as exc:
        return {"legal": False, "error": f"{type(exc).__name__}: {exc}"}


def image_operator_audit(n=12):
    PIX = product(*(BYTE for _ in range(n)))
    U8, U24 = integer(8, signed=False), integer(24, signed=False)
    BYTEF = Type("int", 32, encoding=Encoding("float"), role="byte")
    rows = {
        "sum(image bytes)": legality("sum", (PIX,)),
        "mean(image bytes)": legality("mean", (PIX,)),
        "reduce_max(image bytes)": legality("reduce_max", (PIX,)),
        "count(image bytes)": legality("count", (PIX,)),
        "fft(image bytes)": legality("fft", (PIX,)),
        "add(byte,byte)": legality("add", (BYTE, BYTE)),
        "sub(byte,byte)": legality("sub", (BYTE, BYTE)),
        "lt(byte,byte)": legality("lt", (BYTE, BYTE)),
        "eq(byte,byte)": legality("eq", (BYTE, BYTE)),
        "project(image,i)->byte": legality("project", (PIX,), BYTE, {"index": 0}),
        "index(image,int)->byte": legality("index", (PIX, U8)),
        "decode(byte)->scalar": legality("decode", (BYTE,), SCALAR),
        "decode(byte)->float(role=byte)": legality("decode", (BYTE,), BYTEF),
        "pack((byte,))->uint8": legality("pack", (product(BYTE),), U8),
        "pack((b,b,b))->uint24": legality("pack", (product(BYTE, BYTE, BYTE),), U24),
        "decode(uint8)->scalar": legality("decode", (U8,), SCALAR),
        "sum(bool tuple)": legality("sum", (product(BOOL, BOOL, BOOL),)),
        "count(bool tuple)": legality("count", (product(BOOL, BOOL, BOOL),)),
    }
    rows["BYTE.numeric"] = BYTE.numeric
    rows["float32_with_role_byte.numeric"] = BYTEF.numeric
    return rows


def eq_surrogate_range():
    rows = []
    for d in (0, 1, 2, 3, 5, 8, 10, 11, 12, 16, 32, 64, 128, 255):
        a = torch.tensor([[0.]], requires_grad=True); b = torch.tensor([[float(d)]])
        y = torch.exp(-((a - b) ** 2).sum(-1, keepdim=True) / 1.); y.backward()
        rows.append({"delta": d, "eq_relaxed": float(y.detach()), "grad": float(a.grad)})
    for d in (0, 1, 5, 10, 20, 40, 60, 100, 200):
        a = torch.tensor([[0.]], requires_grad=True); b = torch.tensor([[float(d)]])
        y = torch.sigmoid((b - a) / 1.); y.backward()
        rows.append({"lt_delta": d, "lt_relaxed": float(y.detach()), "grad": float(a.grad)})
    return rows


def relaxation_bugs():
    out = {}
    op = R.resolve("tuple", (BOOL, BOOL))
    try:
        relaxed(R, op, [torch.zeros(5, 1), torch.zeros(1)]); out["tuple_broadcast"] = "no error"
    except Exception as exc:
        out["tuple_broadcast"] = f"{type(exc).__name__}: {exc}"
    I = integer(32, signed=True)
    out["mul_broadcast"] = list(relaxed(R, R.resolve("mul", (I, I)), [torch.zeros(5, 1), torch.zeros(1)]).shape)
    p = Program((("x", BOOL),), (Node("y", BOOL, (Candidate(R.resolve("identity", (BOOL,)), ("x",)),)),),
                (("o", "y"),), (("k", Value.of(BYTE, 24)),), trainable_constants=("k",))
    try:
        p.validate(R); out["trainable_byte_constant"] = "accepted"
    except Exception as exc:
        out["trainable_byte_constant"] = f"{type(exc).__name__}: {exc}"
    from tcn.learning import SoftProgram
    from examples.mixed import problem
    m1 = SoftProgram(problem()[0]); m2 = SoftProgram(problem()[0])
    out["softprogram_logits_zero_initialised"] = all(float(x.abs().sum()) == 0 for x in m1.choices)
    out["softprogram_two_instances_identical"] = all(torch.equal(x, y) for x, y in zip(m1.choices, m2.choices))
    return out


VEC3 = product(SCALAR, SCALAR, SCALAR)
APPROACH = Action("camera", arguments=(("delta", Value.of(VEC3, (-2.4, -2.2, -3.6))),))


def geometry_information(resolutions=(2, 3, 4, 6, 8, 16, 32), seeds=60):
    rows = {}
    for res in resolutions:
        for approach in (False, True):
            bg = []; seen = collections.defaultdict(set); seenid = collections.defaultdict(set)
            train_rgb = set(); test_hit = test_n = 0
            for seed in range(seeds):
                h = Host.create("geometry", seed=seed, configuration={"resolution": res, "objects": 3})
                if approach: h.step([APPROACH])
                rec = h.records[-1]
                _, _, _, data = rec.actor_view().observations["pixels"].decoded
                depth = rec.probes["depth"].decoded; ids = rec.probes["object_ids"].decoded
                bg.append(sum(1 for d in depth if d == 0.) / len(depth))
                for i in range(len(depth)):
                    rgb = tuple(data[3 * i:3 * i + 3])
                    seen[rgb].add(round(depth[i], 3)); seenid[rgb].add(ids[i])
                    if seed < seeds // 2: train_rgb.add(rgb)
                    else:
                        test_n += 1; test_hit += rgb in train_rgb
            rows[f"R{res}_{'approach' if approach else 'shipped_camera'}"] = {
                "background_fraction": statistics.mean(bg),
                "distinct_rgb": len(seen),
                "rgb_with_more_than_one_depth": sum(1 for v in seen.values() if len(v) > 1),
                "rgb_with_more_than_one_id": sum(1 for v in seenid.values() if len(v) > 1),
                "held_out_pixel_rgb_seen_in_training": test_hit / max(1, test_n)}
    return rows


def relations_coverage(entities=(3, 4, 5), seeds=400):
    rows = {}
    for n in entities:
        tot = pos = ok1 = ok2 = 0
        for seed in range(seeds):
            h = Host.create("relations", seed=seed, configuration={"entities": n})
            edges = set(map(tuple, h.state["edges"])); q = tuple(h.state["query"])
            target = h.records[-1].probes["target"].decoded
            tot += 1; pos += target
            ok1 += (q in edges) == target
            ok2 += (q in edges or any((q[0], m) in edges and (m, q[1]) in edges for m in range(n))) == target
        rows[f"entities{n}"] = {"P(target)": pos / tot, "one_step_accuracy": ok1 / tot,
                                "two_step_accuracy": ok2 / tot}
    return rows


def text_probe_cost(w=8, h=8, alphabet=256):
    from tcn.generation import text_value
    t = text_value("", 64).type
    return {"text_probe_type": t.to_dict()["kind"], "width": t.width,
            "byte_fields": 64, "byte_valued_sources": 3 * w * h + alphabet,
            "discrete_programs_for_a_free_byte_head": (3 * w * h + alphabet) ** 64,
            "aggregation_operators_available_over_a_byte_tuple": ["count (returns the static length)"]}


if __name__ == "__main__":
    result = {"image_operator_audit": image_operator_audit(),
              "eq_and_lt_surrogate_range": eq_surrogate_range(),
              "relaxation_bugs": relaxation_bugs(),
              "geometry_information": geometry_information(),
              "relations_coverage": relations_coverage(),
              "text_probe_cost": text_probe_cost()}
    dump("audit", result)
    print(json.dumps({k: v for k, v in result.items() if k != "eq_and_lt_surrogate_range"}, indent=1, default=str)[:6000])
