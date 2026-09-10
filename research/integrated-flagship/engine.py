"""Exact evaluation and exact conformer counting for the program family.

Two instruments, both reproducing `Program.execute`'s semantics on the family of
`family.py` exactly (checked by V1/V2, `validate.py`):

* `Tables` + `evaluate_batch` -- a vectorized evaluator: per-episode click and
  hit for any batch of full selections. Used to run searches and to brute-force
  the counter on sub-spaces.
* `count` -- the exact factorized counter of PREREGISTRATION §6.2. It decides
  every program of a (possibly astronomically large) product space by grouping
  candidates by their per-episode behaviour:

    relation side  (ra, lr1, lr2)            -> routing masks (M0, M1), weights
    colour side    (cpos, lc1, lc2, lc3)     -> class masks (C0, C1, C2), weights
    matcher        M                         -> table tau[e][k] = (lo, hi) or ERR
    grounding      K1..K4                    -> per class, colours grouped by the
                                                (lo, hi) they give on that class
    steps          X1..X3                    -> for a fixed episode-wise (lo, hi)
                                                vector L, F_j(S) = #x in X_j that
                                                hits every episode of S, by a
                                                superset (zeta) transform

  and sums  sum_L w(L) * sum_routes w(r) * F_1(S_1) F_2(S_2) F_3(S_3)  with
  Python integers, so the count is exact.

Semantics reproduced (eager program, every node executes):
* an ADDR candidate must yield an address in [0, 63] on every episode (the text
  length's declared bounds are (0, 64); 64 is past the tuple), else the program
  is unusable;
* a matcher whose stride reads past the raster at the last position errors on
  every episode; an empty anchor set errors in that episode (`reduce_min`);
* raster arithmetic wraps at 16 bits; a click whose pixel index is >= 256 is a
  miss (the encoder's `log` fails / the scoring head's `index` fails);
* a hit is `owner[q] == target`.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np

from family import KIND, SLOTS, PALETTE

N_PIX = 256
RAW = 768
ERR = -1
COLOUR_SLOTS = ("lc1", "lc2", "lc3")
REL_SLOTS = ("lr1", "lr2")
K_SLOTS = ("K1", "K2", "K3", "K4")
X_SLOTS = ("X1", "X2", "X3")


# ------------------------------------------------------------------ episodes
class Episodes:
    def __init__(self, records):
        self.records = list(records)
        self.E = len(self.records)
        self.text = np.array([r["text_bytes"] for r in self.records], dtype=np.int64)
        self.length = np.array([r["text_length"] for r in self.records], dtype=np.int64)
        self.pixels = np.array([r["pixels"] for r in self.records], dtype=np.int64)
        self.owner = np.array([r["owner"] for r in self.records], dtype=np.int64)
        self.target = np.array([r["target"] for r in self.records], dtype=np.int64)

    def subset(self, indices):
        return Episodes([self.records[i] for i in indices])


# -------------------------------------------------------------- slot tables
def addr_values(spec, length):
    """Address per episode, or ERR where the program would raise."""
    op = spec[0]
    ports = []
    for s in spec[1:]:
        ports.append(length.copy() if s == "length" else np.full_like(length, int(s[1:])))
    out = np.full_like(length, ERR)
    if op == "identity":
        v = ports[0]
        ok = np.ones_like(v, dtype=bool)
    else:
        a, b = ports
        ok = np.ones_like(a, dtype=bool)
        if op == "add":
            v = a + b
        elif op == "sub":
            v = a - b
        elif op == "mul":
            v = a * b
        elif op == "min":
            v = np.minimum(a, b)
        elif op == "max":
            v = np.maximum(a, b)
        elif op in ("idiv", "mod"):
            ok = b != 0
            safe = np.where(ok, b, 1)
            v = a // safe if op == "idiv" else a % safe
        else:
            raise ValueError(op)
    ok &= (v >= 0) & (v <= 63)
    out[ok] = v[ok]
    return out


def truth(t, x, y):
    return ((t >> (2 * x.astype(np.int64) + y.astype(np.int64))) & 1).astype(bool)


class Tables:
    """Per-candidate behaviour on a set of episodes, for one arm's pools."""

    def __init__(self, episodes, pools):
        self.ep = episodes
        self.pools = pools
        E = episodes.E
        # ADDR: value per (candidate, episode); byte read; validity.
        self.addr = {}
        for slot in ("cpos", "ra"):
            vals = np.stack([addr_values(s, episodes.length) for s in pools[slot]])
            valid = (vals != ERR).all(axis=1)
            byte = np.where(vals != ERR,
                            np.take_along_axis(np.broadcast_to(episodes.text, (len(vals),) + episodes.text.shape),
                                               np.maximum(vals, 0)[..., None], axis=2)[..., 0], -1)
            self.addr[slot] = {"values": vals, "valid": valid, "byte": byte}
        self.lit = {s: np.array(pools[s], dtype=np.int64) for s in COLOUR_SLOTS + REL_SLOTS}
        self.ground = {s: np.array(pools[s], dtype=np.int64) for s in K_SLOTS}
        # MATCH: tau[m, e, k] -> index into self.lvals[e] (ERR for empty / invalid).
        self.lvals = [dict() for _ in range(E)]        # (lo, hi) -> index
        self.lkeys = [[] for _ in range(E)]
        self.match_valid = np.zeros(len(pools["M"]), dtype=bool)
        self.tau = np.full((len(pools["M"]), E, len(PALETTE)), ERR, dtype=np.int64)
        pix = episodes.pixels
        colours = np.array(PALETTE, dtype=np.int64)
        cache = {}
        for m, (sg, sb, rg, same) in enumerate(pools["M"]):
            if 3 * (N_PIX - 1) + max(sg, sb) >= RAW:
                continue                               # stride past the raster: always raises
            self.match_valid[m] = True
            key = (sg, sb)
            if key not in cache:
                base = np.arange(N_PIX) * 3
                r_ = pix[:, base]                       # (E, 256)
                g_ = pix[:, base + sg]
                b_ = pix[:, base + sb]
                er = r_[:, None, :] == colours[None, :, 0, None]   # (E, 8, 256)
                eg = g_[:, None, :] == colours[None, :, 1, None]
                eb = b_[:, None, :] == colours[None, :, 2, None]
                cache[key] = (er, eg, eb)
            er, eg, eb = cache[key]
            f = truth(same, truth(rg, er, eg), eb)      # (E, 8, 256)
            anyf = f.any(axis=2)
            lo = np.where(anyf, np.argmax(f, axis=2), -1) * 3
            hi = np.where(anyf, N_PIX - 1 - np.argmax(f[:, :, ::-1], axis=2), -1) * 3
            for e in range(E):
                for k in range(len(PALETTE)):
                    if not anyf[e, k]:
                        continue
                    lh = (int(lo[e, k]), int(hi[e, k]))
                    idx = self.lvals[e].get(lh)
                    if idx is None:
                        idx = self.lvals[e][lh] = len(self.lkeys[e])
                        self.lkeys[e].append(lh)
                    self.tau[m, e, k] = idx
        # STEP: hit[e] is (n_l(e), |X|) bool for the shared step pool of X1..X3.
        steps = pools["X1"]
        assert pools["X2"] == steps and pools["X3"] == steps
        self.steps = steps
        ops = np.array([("add", "sub", "mul", "min", "max").index(op) for op, _, _ in steps])
        bases = np.array([0 if b == "lo" else 1 for _, b, _ in steps])
        offs = np.array([o for _, _, o in steps], dtype=np.int64)
        self.hit = []
        self.click = []
        for e in range(E):
            if not self.lkeys[e]:
                self.hit.append(np.zeros((0, len(steps)), dtype=bool))
                self.click.append(np.zeros((0, len(steps)), dtype=np.int64))
                continue
            lh = np.array(self.lkeys[e], dtype=np.int64)           # (n_l, 2)
            base = lh[:, bases]                                     # (n_l, X)
            c = np.select([ops == 0, ops == 1, ops == 2, ops == 3, ops == 4],
                          [base + offs, base - offs, base * offs,
                           np.minimum(base, offs), np.maximum(base, offs)])
            c = np.mod(c, 1 << 16)
            q = c // 3
            inr = q < N_PIX
            h = np.zeros_like(inr)
            h[inr] = episodes.owner[e][q[inr]] == episodes.target[e]
            self.hit.append(h)
            self.click.append(c)


# ------------------------------------------------------------- evaluation
def evaluate_batch(T, sel):
    """Conformance of each row of `sel` (B x 15 slot indices, SLOTS order).

    Returns (conform[B], hits[B, E]). A program that raises in any episode does
    not conform and has hits all False from that episode on (as `evaluate`).
    """
    sel = np.asarray(sel, dtype=np.int64)
    B = sel.shape[0]
    col = {s: sel[:, SLOTS.index(s)] for s in SLOTS}
    E = T.ep.E
    ok = T.addr["cpos"]["valid"][col["cpos"]] & T.addr["ra"]["valid"][col["ra"]] \
        & T.match_valid[col["M"]]
    hits = np.zeros((B, E), dtype=bool)
    for e in range(E):
        cb = T.addr["cpos"]["byte"][col["cpos"], e]
        m1 = cb == T.lit["lc1"][col["lc1"]]
        m2 = cb == T.lit["lc2"][col["lc2"]]
        m3 = cb == T.lit["lc3"][col["lc3"]]
        cls = np.where(m1, 0, np.where(m2, 1, np.where(m3, 2, 3)))
        kcols = np.stack([T.ground[s][col[s]] for s in K_SLOTS], axis=1)
        k = kcols[np.arange(B), cls]
        lidx = T.tau[col["M"], e, k]
        good = lidx != ERR
        rb = T.addr["ra"]["byte"][col["ra"], e]
        r1 = rb == T.lit["lr1"][col["lr1"]]
        r2 = rb == T.lit["lr2"][col["lr2"]]
        xs = np.where(r1, col["X1"], np.where(r2, col["X2"], col["X3"]))
        h = np.zeros(B, dtype=bool)
        if T.hit[e].shape[0]:
            h[good] = T.hit[e][lidx[good], xs[good]]
        hits[:, e] = h & ok
    return hits.all(axis=1) & ok, hits


def clicks(T, chosen):
    """Per-episode click byte address (or None) of one full selection."""
    s = [chosen[k] for k in SLOTS]
    out = []
    for e in range(T.ep.E):
        a = T.addr["cpos"]["values"][s[0], e]
        r = T.addr["ra"]["values"][s[SLOTS.index("ra")], e]
        if a == ERR or r == ERR or not T.match_valid[chosen["M"]]:
            out.append(None)
            continue
        cb = T.ep.text[e, a]
        lits = [T.lit[x][chosen[x]] for x in COLOUR_SLOTS]
        cls = next((i for i, v in enumerate(lits) if cb == v), 3)
        k = T.ground[K_SLOTS[cls]][chosen[K_SLOTS[cls]]]
        li = T.tau[chosen["M"], e, k]
        if li == ERR:
            out.append(None)
            continue
        rb = T.ep.text[e, r]
        rl = [T.lit[x][chosen[x]] for x in REL_SLOTS]
        j = next((i for i, v in enumerate(rl) if rb == v), 2)
        out.append(int(T.click[e][li, chosen[X_SLOTS[j]]]))
    return out


# ------------------------------------------------------------------ counting
def zeta_superset(g, E):
    """F[S] = sum_{T superset of S} g[T]."""
    F = g.astype(np.int64).copy()
    for i in range(E):
        F = F.reshape(-1, 2, 1 << i)
        F[:, 0, :] += F[:, 1, :]
        F = F.reshape(-1)
    return F


class Restriction:
    """Which candidates of each slot are in the (sub)space being counted.

    ADDR / MATCH / GROUND / STEP: a boolean mask over the slot's pool.
    LIT: two masks over the letters -- allowed when the letter occurs at the
    slot's address in these episodes, and allowed when it does not -- because
    the learned prior's LIT feature is conditional on the address.
    """

    def __init__(self, pools, masks=None, lit_occurs=None, lit_absent=None):
        self.m = {}
        for s in SLOTS:
            if KIND[s] == "LIT":
                continue
            self.m[s] = np.ones(len(pools[s]), dtype=bool) if masks is None or s not in masks \
                else np.asarray(masks[s], dtype=bool)
        self.occ, self.abs = {}, {}
        for s in COLOUR_SLOTS + REL_SLOTS:
            n = len(pools[s])
            self.occ[s] = np.ones(n, dtype=bool) if lit_occurs is None or s not in lit_occurs \
                else np.asarray(lit_occurs[s], dtype=bool)
            self.abs[s] = np.ones(n, dtype=bool) if lit_absent is None or s not in lit_absent \
                else np.asarray(lit_absent[s], dtype=bool)


def _letter_groups(T, byte_vec, slot, R, E):
    """Allowed letters of a LIT slot grouped by their mask on this byte vector."""
    groups = defaultdict(int)
    for i, v in enumerate(T.lit[slot]):
        bits = 0
        for e in range(E):
            if byte_vec[e] == v:
                bits |= 1 << e
        allowed = R.occ[slot][i] if bits else R.abs[slot][i]
        if allowed:
            groups[bits] += 1
    return groups


def relation_side(T, R):
    """(M0, M1) routing masks -> weight, over valid allowed ra and allowed literals."""
    E = T.ep.E
    A = T.addr["ra"]
    by = defaultdict(int)
    for i in np.flatnonzero(R.m["ra"] & A["valid"]):
        by[tuple(A["byte"][i])] += 1
    routes = defaultdict(int)
    for bv, w in by.items():
        g1 = _letter_groups(T, bv, "lr1", R, E)
        g2 = _letter_groups(T, bv, "lr2", R, E)
        for m1, c1 in g1.items():
            for m2, c2 in g2.items():
                routes[(m1, m2 & ~m1)] += w * c1 * c2
    return routes


def colour_side(T, R):
    """(C0, C1, C2) class masks -> weight, over valid allowed cpos and literals."""
    E = T.ep.E
    A = T.addr["cpos"]
    by = defaultdict(int)
    for i in np.flatnonzero(R.m["cpos"] & A["valid"]):
        by[tuple(A["byte"][i])] += 1
    classes = defaultdict(int)
    for bv, w in by.items():
        gs = [_letter_groups(T, bv, s, R, E) for s in COLOUR_SLOTS]
        for m1, c1 in gs[0].items():
            for m2, c2 in gs[1].items():
                c12 = m2 & ~m1
                for m3, c3 in gs[2].items():
                    classes[(m1, c12, m3 & ~(m1 | m2))] += w * c1 * c2 * c3
    return classes


def space_size(T, R):
    """|S| of the restricted space, LIT pools conditional on each address."""
    E = T.ep.E

    def chain(slot_addr, lit_slots):
        A = T.addr[slot_addr]
        total = 0
        for i in np.flatnonzero(R.m[slot_addr]):
            bv = A["byte"][i] if A["valid"][i] else [-2] * E
            prod = 1
            for s in lit_slots:
                prod *= sum(_letter_groups(T, bv, s, R, E).values())
            total += prod
        return total

    s = chain("cpos", COLOUR_SLOTS) * chain("ra", REL_SLOTS)
    for slot in K_SLOTS + ("M",) + X_SLOTS:
        s *= int(R.m[slot].sum())
    return s


class Counter:
    """The exact counter, keeping enough structure to sample conformers uniformly.

    `total` is the exact number of conforming programs in the restricted space.
    `terms` are the (matcher table, class vector, grounding groups, L) cells with
    a non-zero contribution; `sample` draws conformers exactly uniformly from
    them, so "the program a uniform-order search finds first" can be drawn and
    scored on held-out episodes.
    """

    def __init__(self, T, R, collect=False):
        self.T, self.R = T, R
        E = self.E = T.ep.E
        full = self.full = (1 << E) - 1
        routes = relation_side(T, R)
        self.route_list = list(routes)
        self.r_w = [routes[k] for k in self.route_list]
        self.r_keys = np.array([[m0, m1, full & ~(m0 | m1)] for m0, m1 in self.route_list],
                               dtype=np.int64).reshape(-1, 3)
        classes = colour_side(T, R)
        self.class_list = list(classes)
        self.c_w = [classes[k] for k in self.class_list]
        self.xmask = {s: R.m[s] for s in X_SLOTS}
        self.kmask = {s: R.m[s] for s in K_SLOTS}
        allowed_m = np.flatnonzero(R.m["M"] & T.match_valid)
        tables = defaultdict(list)
        for m in allowed_m:
            tables[T.tau[m].tobytes()].append(int(m))
        self.tables = tables
        self.rcache = {}
        self.terms = [] if collect else None
        self.total = self._count() if self.route_list and self.class_list else 0

    # ------------------------------------------------------------ R(L)
    def _hm(self, L):
        hm = np.zeros(len(self.T.steps), dtype=np.int64)
        for e in range(self.E):
            hm |= self.T.hit[e][L[e]].astype(np.int64) << e
        return hm

    def route_products(self, L):
        hm = self._hm(L)
        if getattr(self, "direct", False) or self.E > 20:
            # Direct mode for many episodes (no 2^E transform): F_j(S) by testing
            # every allowed step candidate against every route's episode set.
            prods = np.ones(len(self.r_keys), dtype=np.int64)
            for j, s in enumerate(X_SLOTS):
                h = hm[self.xmask[s]]
                need = self.r_keys[:, j]
                F = np.zeros(len(need), dtype=np.int64)
                for x in h:
                    F += (x & need) == need
                prods *= F
            return prods, hm
        prods = np.ones(len(self.r_keys), dtype=np.int64)
        for j, s in enumerate(X_SLOTS):
            g = np.bincount(hm[self.xmask[s]], minlength=1 << self.E)
            F = zeta_superset(g, self.E)
            prods *= F[self.r_keys[:, j]]
        return prods, hm

    def R_of(self, L):
        r = self.rcache.get(L)
        if r is not None:
            return r
        prods, _ = self.route_products(L)
        nz = np.flatnonzero(prods)
        total = 0
        for start in range(0, len(nz), 2048):          # int64-safe partial sums
            idx = nz[start:start + 2048]
            w = np.array([self.r_w[i] for i in idx], dtype=object)
            total += int(np.dot(w, prods[idx].astype(object)))
        self.rcache[L] = total
        return total

    # ------------------------------------------------------------- count
    def _count(self):
        T, E, full = self.T, self.E, self.full
        cm = np.array([[c0, c1, c2, full & ~(c0 | c1 | c2)] for c0, c1, c2 in self.class_list],
                      dtype=np.int64)
        total = 0
        for tkey, members in self.tables.items():
            wt = len(members)
            tau = T.tau[members[0]]
            # vectorised prune: every non-empty class needs a colour valid on all its episodes
            alive = np.ones(len(cm), dtype=bool)
            for j, s in enumerate(K_SLOTS):
                cols = T.ground[s][np.flatnonzero(self.kmask[s])]
                if len(cols) == 0:
                    alive[:] = False
                    break
                validbits = np.array([sum(1 << e for e in range(E) if tau[e, k] != ERR)
                                      for k in cols], dtype=np.int64)
                ok = ((cm[:, j:j + 1] & ~validbits[None, :]) == 0).any(axis=1)
                alive &= ok
            for ci in np.flatnonzero(alive):
                cmask = cm[ci]
                per_class = []
                for j, s in enumerate(K_SLOTS):
                    eps = [e for e in range(E) if cmask[j] >> e & 1]
                    groups = defaultdict(list)
                    for ki in np.flatnonzero(self.kmask[s]):
                        k = T.ground[s][ki]
                        sig = tuple(int(tau[e, k]) for e in eps)
                        if ERR in sig:
                            continue
                        groups[sig].append(int(ki))
                    per_class.append((eps, list(groups.items())))
                for combo in itertools.product(*(g for _, g in per_class)):
                    L = [0] * E
                    mult = 1
                    for (eps, _), (sig, kis) in zip(per_class, combo):
                        mult *= len(kis)
                        for e, v in zip(eps, sig):
                            L[e] = v
                    L = tuple(L)
                    r = self.R_of(L)
                    if r:
                        add = wt * self.c_w[ci] * mult * r
                        total += add
                        if self.terms is not None:
                            self.terms.append((add, tkey, int(ci), combo, L))
        return total

    # ------------------------------------------------------------ sampling
    def _addr_members(self, slot_addr, lit_slots, target):
        """(address index, letter tuple) configurations producing `target` masks."""
        T, R, E = self.T, self.R, self.E
        A = T.addr[slot_addr]
        out = []
        for i in np.flatnonzero(R.m[slot_addr] & A["valid"]):
            bv = A["byte"][i]
            per = []
            for s in lit_slots:
                g = defaultdict(list)
                for j, v in enumerate(T.lit[s]):
                    bits = 0
                    for e in range(E):
                        if bv[e] == v:
                            bits |= 1 << e
                    if (R.occ[s][j] if bits else R.abs[s][j]):
                        g[bits].append(j)
                per.append(g)
            for combo in itertools.product(*(list(g.items()) for g in per)):
                masks = [m for m, _ in combo]
                if len(lit_slots) == 3:
                    got = (masks[0], masks[1] & ~masks[0], masks[2] & ~(masks[0] | masks[1]))
                else:
                    got = (masks[0], masks[1] & ~masks[0])
                if got == target:
                    out.append((int(i), [js for _, js in combo]))
        return out

    def sample(self, n, rng):
        if not self.terms:
            return []
        import bisect
        cum = []
        acc = 0
        for t in self.terms:
            acc += t[0]
            cum.append(acc)
        assert acc == self.total
        T = self.T
        out = []
        cache_c, cache_r, cache_l = {}, {}, {}
        for _ in range(n):
            u = rng.randrange(self.total)
            add, tkey, ci, combo, L = self.terms[bisect.bisect_right(cum, u)]
            sel = {"M": rng.choice(self.tables[tkey])}
            if ci not in cache_c:
                cache_c[ci] = self._addr_members("cpos", COLOUR_SLOTS, self.class_list[ci])
            members = cache_c[ci]
            weights = [math.prod(len(js) for js in lits) for _, lits in members]
            i, lits = rng.choices(members, weights=weights)[0]
            sel["cpos"] = i
            for s, js in zip(COLOUR_SLOTS, lits):
                sel[s] = rng.choice(js)
            for s, (sig, kis) in zip(K_SLOTS, combo):
                sel[s] = rng.choice(kis)
            if L not in cache_l:
                prods, hm = self.route_products(L)
                cache_l[L] = (hm, [self.r_w[k] * int(prods[k]) for k in range(len(prods))])
            hm, w = cache_l[L]
            k = rng.choices(range(len(w)), weights=w)[0]
            if k not in cache_r:
                cache_r[k] = self._addr_members("ra", REL_SLOTS, self.route_list[k])
            members = cache_r[k]
            weights = [math.prod(len(js) for js in lits) for _, lits in members]
            i, lits = rng.choices(members, weights=weights)[0]
            sel["ra"] = i
            for s, js in zip(REL_SLOTS, lits):
                sel[s] = rng.choice(js)
            for j, s in enumerate(X_SLOTS):
                need = int(self.r_keys[k, j])
                xs = np.flatnonzero(self.xmask[s] & ((hm & need) == need))
                sel[s] = int(rng.choice(xs))
            out.append(sel)
        return out


def count(T, R, return_detail=False):
    """Exact number of conforming programs in the restricted space."""
    c = Counter(T, R)
    if return_detail:
        return c.total, {"routes": len(c.route_list), "classes": len(c.class_list),
                         "tables": len(c.tables), "distinct_L": len(c.rcache)}
    return c.total


def expected_first(S, K):
    """Expected programs evaluated to the first conforming one, uniform random order
    without replacement: (S+1)/(K+1). Returned as an exact (numerator, denominator)."""
    return (S + 1, K + 1)


def ratio_float(num, den):
    return float(num) / float(den) if den else math.inf
