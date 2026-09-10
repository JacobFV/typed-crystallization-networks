"""V1-V3 of PREREGISTRATION §6.3 -- required before any arm is reported.

    python validate.py v2brute --gap 0        # counter == brute force, random sub-spaces
    python validate.py v2tcn   --gap 0        # counter == tcn.search.enumerate_prefix
    python validate.py v1 --shard i --of 4    # evaluator == Program.execute, 48 episodes
    python validate.py v3                     # live kernel rewards; enumerate_environment

Each writes `out/<name>.json`; `verify.py` reads them.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                     # noqa: E402
import cache                           # noqa: E402
import engine                          # noqa: E402
import env                             # noqa: E402
import family                          # noqa: E402
from family import KIND, SLOTS         # noqa: E402

OUT = HERE / "out"
PALETTE = family.PALETTE


def reference(pool, width=16, gap=0):
    """The hand-written program (for validation only; never an arm's answer)."""
    px, row = 3 * (1 + gap), 3 * width * (1 + gap)
    want = {"cpos": ("sub", "length", "k5"), "ra": ("add", "k13", "k0"),
            "lc1": ord("d"), "lc2": ord("n"), "lc3": ord("e"),
            "K1": PALETTE.index((240, 30, 30)), "K2": PALETTE.index((30, 240, 30)),
            "K3": PALETTE.index((30, 30, 240)), "K4": PALETTE.index((240, 240, 30)),
            "lr1": ord("r"), "lr2": ord("l"), "M": (1, 2, 8, 8),
            "X1": ("add", "hi", px), "X2": ("sub", "lo", px), "X3": ("sub", "lo", row)}
    out = {}
    for s in SLOTS:
        try:
            out[s] = pool[s].index(want[s])
        except ValueError:
            out[s] = None
    return out


# ------------------------------------------------------------- brute force
def brute_count(T, R, chunk=500_000):
    """Enumerate the restricted space program by program (LIT pools conditional)."""
    E = T.ep.E
    pools = T.pools

    def letters(slot_addr, i, lit_slots):
        A = T.addr[slot_addr]
        bv = A["byte"][i] if A["valid"][i] else [-2] * E
        out = []
        for s in lit_slots:
            allowed = []
            for j, v in enumerate(T.lit[s]):
                occurs = any(bv[e] == v for e in range(E) if R.occ_mask >> e & 1)
                if (R.occ[s][j] if occurs else R.abs[s][j]):
                    allowed.append(j)
            out.append(allowed)
        return out

    colour = []
    for i in np.flatnonzero(R.m["cpos"]):
        colour += [(i,) + c for c in itertools.product(*letters("cpos", i, engine.COLOUR_SLOTS))]
    rel = []
    for i in np.flatnonzero(R.m["ra"]):
        rel += [(i,) + c for c in itertools.product(*letters("ra", i, engine.REL_SLOTS))]
    rest = [list(np.flatnonzero(R.m[s])) for s in engine.K_SLOTS + ("M",) + engine.X_SLOTS]
    total = len(colour) * len(rel) * int(np.prod([len(r) for r in rest], dtype=object))
    found = 0
    order = {s: k for k, s in enumerate(SLOTS)}
    buf = []

    def flush():
        nonlocal found, buf
        if buf:
            c, _ = engine.evaluate_batch(T, np.array(buf, dtype=np.int64))
            found += int(c.sum())
            buf = []

    for c in colour:
        for r in rel:
            for tail in itertools.product(*rest):
                row = [0] * len(SLOTS)
                for s, v in zip(("cpos",) + engine.COLOUR_SLOTS, c):
                    row[order[s]] = v
                for s, v in zip(("ra",) + engine.REL_SLOTS, r):
                    row[order[s]] = v
                for s, v in zip(engine.K_SLOTS + ("M",) + engine.X_SLOTS, tail):
                    row[order[s]] = v
                buf.append(row)
                if len(buf) >= chunk:
                    flush()
    flush()
    return total, found


EQUIV = {  # reward-equivalent spellings at gap 0 (validation only): same pixel reached
    "X1": [("add", "hi", 4), ("add", "hi", 5), ("max", "hi", 3)],
    "X2": [("sub", "lo", 1), ("sub", "lo", 2)],
    "X3": [("sub", "lo", 46), ("sub", "lo", 47)],
    "ra": [("identity", "k13"), ("add", "k6", "k7"), ("max", "k13", "k0"), ("add", "k0", "k13")],
    "cpos": [("sub", "length", "k5")],
}


def random_restriction(pool, ref, rng, extras=(0, 3), conditional=False):
    masks, occ, ab = {}, {}, {}
    for s in SLOTS:
        n = len(pool[s])
        chosen = set(rng.sample(range(n), rng.randint(*extras)))
        if ref.get(s) is not None and rng.random() < .9:
            chosen.add(ref[s])
        for spec in EQUIV.get(s, ()):
            if spec in pool[s] and rng.random() < .5:
                chosen.add(pool[s].index(spec))
        if not chosen:
            chosen.add(rng.randrange(n))
        m = np.zeros(n, dtype=bool)
        m[list(chosen)] = True
        if KIND[s] == "LIT":
            occ[s] = m
            ab[s] = m.copy()
            if conditional:
                ab[s] = np.zeros(n, dtype=bool)
                ab[s][rng.randrange(n)] = True
        else:
            masks[s] = m
    return engine.Restriction(pool, masks, occ, ab)


def v2brute(gap, trials, seed):
    d = cache.load(gap)
    eps = engine.Episodes(d["episodes"][:d["n_train"]])
    rng = random.Random(seed)
    rows = []
    for arm in ("schema", "flat", "distractor"):
        pool = family.pools(arm)
        T = engine.Tables(eps, pool)
        ref = reference(pool, gap=gap)
        for t in range(trials):
            R = random_restriction(pool, ref, rng, conditional=(t % 3 == 2))
            S = engine.space_size(T, R)
            if S > 3_000_000:
                continue
            t0 = time.perf_counter()
            K = engine.count(T, R)
            tc = time.perf_counter() - t0
            t0 = time.perf_counter()
            S_b, K_b = brute_count(T, R)
            tb = time.perf_counter() - t0
            rows.append({"arm": arm, "trial": t, "S_counter": S, "S_brute": S_b, "K_counter": K,
                         "K_brute": K_b, "agree": S == S_b and K == K_b, "counter_s": tc,
                         "brute_s": tb, "conditional_lit": t % 3 == 2})
            print(rows[-1], flush=True)
    out = {"gap": gap, "seed": seed, "trials": rows, "n": len(rows),
           "n_agree": sum(r["agree"] for r in rows),
           "n_with_conformers": sum(r["K_brute"] > 0 for r in rows)}
    (OUT / f"v2_brute_gap{gap}.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------ tcn programs
def episode_inputs(record):
    from tcn.generation import image_value, text_value
    px = np.array(record["pixels"], dtype=np.uint8).reshape(record["height"], record["width"], 3)
    return {"text": text_value(record["text"], env.TEXT_CAPACITY), "pixels": image_value(px)}


def restricted_pool(pool, R):
    """The candidate lists a restriction keeps (unconditional LIT masks only)."""
    out = {}
    for s in SLOTS:
        m = R.occ[s] if KIND[s] == "LIT" else R.m[s]
        out[s] = [c for c, keep in zip(pool[s], m) if keep]
    return out


def v2tcn(gap):
    from tcn.search import enumerate_prefix
    from tcn.types import Value
    d = cache.load(gap)
    records = d["episodes"][:d["n_train"]]
    eps = engine.Episodes(records)
    pool = family.pools("schema")
    ref = reference(pool, gap=gap)
    designs = {
        "steps_and_relation": {"X1": [ref["X1"], pool["X1"].index(("add", "hi", 6)), 0, 5],
                               "X2": [ref["X2"], 1, 6, 11], "X3": [ref["X3"], 14, 17, 2],
                               "lr1": [ref["lr1"], ref["lr2"]], "lr2": [ref["lr2"], 0]},
        "grounding_and_steps": {"K1": [ref["K1"], 0], "K2": [ref["K2"], ref["K1"]],
                                "K3": [ref["K3"], 7], "K4": [ref["K4"], 3],
                                "X1": [ref["X1"], 3], "X3": [ref["X3"], 12]},
        "matcher_and_literals": {"M": [ref["M"], pool["M"].index((1, 2, 7, 2)), 0, 255],
                                 "lc1": [ref["lc1"], ref["lc2"]], "lc2": [ref["lc2"], ref["lc1"]],
                                 "lc3": [ref["lc3"], 0]},
    }
    rows = []
    for name, design in designs.items():
        masks, occ, ab = {}, {}, {}
        for s in SLOTS:
            n = len(pool[s])
            keep = design.get(s, [ref[s]])
            m = np.zeros(n, dtype=bool)
            m[keep] = True
            if KIND[s] == "LIT":
                occ[s] = ab[s] = m
            else:
                masks[s] = m
        R = engine.Restriction(pool, masks, occ, ab)
        T = engine.Tables(eps, pool)
        K = engine.count(T, R)
        S = engine.space_size(T, R)
        sub = restricted_pool(pool, R)
        program, registry = family.build(sub, head="key")
        examples = []
        for r in records:
            tgt = r["widgets"][r["target"]]["colour"]
            examples.append({"inputs": episode_inputs(r),
                             "targets": {"key": Value.of(family.KEY, family.pack_rgb(tgt))}})
        t0 = time.perf_counter()
        res = enumerate_prefix(program, examples, family.key_signal(), registry, tolerance=0.)
        rows.append({"design": name, "space_counter": S, "K_counter": K,
                     "tcn": res.to_dict(), "agree": res.space_size == S and res.conforming == K
                     and res.exhausted, "seconds": time.perf_counter() - t0})
        print({k: v for k, v in rows[-1].items() if k != "tcn"}, res.certificate, flush=True)
    out = {"gap": gap, "designs": rows, "n_agree": sum(r["agree"] for r in rows)}
    (OUT / f"v2_tcn_gap{gap}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def random_program(pool, T, rng, valid_bias=True):
    sel = {}
    for s in SLOTS:
        n = len(pool[s])
        sel[s] = rng.randrange(n)
    if valid_bias:
        for s in ("cpos", "ra"):
            ok = np.flatnonzero(T.addr[s]["valid"])
            if len(ok):
                sel[s] = int(rng.choice(ok))
        okm = np.flatnonzero(T.match_valid)
        sel["M"] = int(rng.choice(okm))
    return sel


def v1(shard, of, per_arm, seed):
    d = cache.load(0)
    records = d["episodes"]
    eps = engine.Episodes(records)
    inputs = [episode_inputs(r) for r in records]
    rng = random.Random(seed * 1000 + shard)
    rows = []
    for arm in ("schema", "flat", "distractor"):
        pool = family.pools(arm)
        T = engine.Tables(eps, pool)
        # Amendment A1: compare the `clk` node of the scaffold with no head. The
        # `key` and `agent` heads raise on an off-raster click by design (a miss),
        # which would hide the click value the evaluator must reproduce.
        program, registry = family.build(pool, head="none")
        for i in range(per_arm):
            if i % of != shard:
                continue
            sel = random_program(pool, T, rng, valid_bias=(i % 3 != 0))
            if i == 0 and arm == "schema":
                sel = reference(pool)
            want = engine.clicks(T, sel)
            got = []
            full = family.full_selection(program, sel)
            for inp in inputs:
                try:
                    out, _, trace = program.execute(inp, registry=registry, selections=full)
                    got.append(int(trace["clk"].decoded))
                except (ValueError, TypeError, OverflowError, ZeroDivisionError,
                        ArithmeticError, IndexError):
                    got.append(None)
            rows.append({"arm": arm, "i": i, "selection": sel, "engine": want, "tcn": got,
                         "agree": want == got})
            print(arm, i, rows[-1]["agree"], flush=True)
    out = {"shard": shard, "of": of, "rows": rows, "n": len(rows),
           "n_agree": sum(r["agree"] for r in rows), "episodes": len(records)}
    (OUT / f"v1_shard{shard}.json").write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------- live V3
def agent_program(pool, sel):
    prog, reg = family.build(pool, head="agent")
    return prog.harden(family.full_selection(prog, sel)), reg


def agent_config():
    from tcn.generation import Action
    from tcn.policy import ActionBinding
    from tcn.search import AgentConfig
    return AgentConfig(observations=("pixels", "text"), action_templates=(Action("click"),),
                       action_bindings=(ActionBinding(0, "pos", env.CLICK, "arg",
                                                      (-1., float(env.N_PIXELS))),), horizon=1)


def live_rewards(pool, sel, indices, split, gap, session):
    from tcn.agent import Agent
    frozen, reg = agent_program(pool, sel)
    out = []
    for index in indices:
        host = env.make_host(0, index, split, {"gap": gap, "session": session})
        try:
            Agent(frozen, reg, agent_config()).rollout(host, deterministic=True)
            out.append(float(host.records[-1].reward_components["goal"].decoded))
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, ArithmeticError,
                IndexError):
            out.append(None)
    return out


def v3(n_random, seed, extra=()):
    from tcn.search import EnvironmentTask, enumerate_environment
    from tcn.generation import Action
    d = cache.load(0)
    records = d["episodes"][:d["n_train"]]
    eps = engine.Episodes(records)
    rng = random.Random(seed)
    rows = []
    programs = []
    pool = family.pools("schema")
    T = engine.Tables(eps, pool)
    programs.append(("schema", reference(pool)))
    for name, arm_pool, sel in extra:
        programs.append((name, sel))
    for i in range(n_random):
        programs.append(("schema", random_program(pool, T, rng)))
    pools = {"schema": pool}
    tables = {"schema": T}
    for arm, sel in programs:
        p = pools.setdefault(arm, family.pools(arm if arm in ("flat", "distractor", "schema") else "schema"))
        Tt = tables.setdefault(arm, engine.Tables(eps, p))
        _, hits = engine.evaluate_batch(Tt, np.array([[sel[s] for s in SLOTS]]))
        live = live_rewards(p, sel, range(len(records)), "train", 0, "v3")
        want = [float(h) for h in hits[0]]
        rows.append({"arm": arm, "selection": sel, "engine_hits": want, "live": live,
                     "agree": [0. if x is None else x for x in live] == want})
        print(arm, rows[-1]["agree"], flush=True)
    # enumerate_environment through the integrated generator on one small sub-space
    ref = reference(pool)
    keep = {"X1": [ref["X1"], 0, 5], "lr2": [ref["lr2"], 0]}
    sub = {s: [pool[s][i] for i in keep.get(s, [ref[s]])] for s in SLOTS}
    prog, reg = family.build(sub, head="agent")
    cfg = agent_config()
    task = EnvironmentTask(generator="integrated", observations=cfg.observations,
                           action_templates=cfg.action_templates,
                           action_bindings=cfg.action_bindings,
                           generator_config={"gap": 0, "session": "v3env"},
                           indices=tuple(range(len(records))), horizon=1, split="train")
    t0 = time.perf_counter()
    res = enumerate_environment(prog, task, reg, threshold=1.0, ledger=env.Ledger())
    Ts = engine.Tables(eps, sub)
    K = engine.count(Ts, engine.Restriction(sub))
    envrow = {"space": res.space_size, "tcn": res.to_dict(), "K_counter": K,
              "agree": res.conforming == K and res.exhausted, "seconds": time.perf_counter() - t0}
    print({k: v for k, v in envrow.items() if k != "tcn"}, flush=True)
    out = {"programs": rows, "n": len(rows), "n_agree": sum(r["agree"] for r in rows),
           "enumerate_environment": envrow}
    (OUT / "v3_live.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=4)
    ap.add_argument("--per-arm", type=int, default=180)
    ap.add_argument("--random", type=int, default=50)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.what == "v2brute":
        v2brute(a.gap, a.trials, a.seed)
    elif a.what == "v2tcn":
        v2tcn(a.gap)
    elif a.what == "v1":
        v1(a.shard, a.of, a.per_arm, a.seed)
    elif a.what == "v3":
        v3(a.random, a.seed)
    else:
        raise SystemExit(a.what)
