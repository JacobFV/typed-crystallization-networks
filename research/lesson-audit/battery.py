"""A battery of cheap exploits, fitted on one seed split and scored on another.

The question every exploit asks is the same: *how much of this lesson's answer
can be recovered without doing the thing the lesson is named after?*  Each
exploit is fitted on a train split of seeds and scored on a disjoint test split,
so a lookup table that memorises the training answers gets no credit unless it
actually transfers.

Nothing here imports ``tcn``.  It reads the language generator's catalogue
directly, because what is being audited is the episode distribution, not any
learner.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Callable, Sequence

import numpy as np

# ---------------------------------------------------------------- episodes

WORD = re.compile(r"[^\s]+")
ALNUM = re.compile(r"[0-9A-Za-z_]+")
PUNCT = re.compile(r"[\s‘’“”\"'`.,;:!?()\[\]{}]+")


def norm(t: str) -> str:
    """Cheap stand-in for ``engine.scoring.normalize`` (ASCII path only)."""
    t = (t or "").casefold().strip()
    return re.sub(r"\s+", " ", PUNCT.sub(" ", t).strip())


class Episode:
    __slots__ = ("obs", "prompt", "answer", "choices", "a", "cs", "raw", "alnum",
                 "obs_l")

    def __init__(self, obs: str, prompt: str, answer: str, choices: Sequence[str]):
        self.obs = obs
        self.prompt = prompt
        self.answer = answer
        self.choices = tuple(choices)
        self.a = norm(answer)
        self.cs = tuple(norm(c) for c in choices)
        self.raw = WORD.findall(obs)
        self.alnum = ALNUM.findall(obs.casefold())
        self.obs_l = obs.casefold()


# ------------------------------------------------------------- utilities

def acc(preds: Sequence[str], eps: Sequence[Episode]) -> float:
    if not eps:
        return 0.0
    return sum(1.0 for p, e in zip(preds, eps) if norm(p) == e.a) / len(eps)


def _majority(xs) -> str:
    c = Counter(xs)
    return c.most_common(1)[0][0] if c else ""


def _lookup_fit(keys, vals, default):
    """key -> majority value, with a global backoff."""
    d = defaultdict(Counter)
    for k, v in zip(keys, vals):
        d[k][v] += 1
    return {k: c.most_common(1)[0][0] for k, c in d.items()}, default


def _lookup_apply(table, default, keys):
    return [table.get(k, default) for k in keys]


# ------------------------------------------------- A. no-input baselines

def ex_constant(tr, te):
    m = _majority([e.a for e in tr])
    return [m] * len(te)


def ex_const_choice_index(tr, te):
    """Best fixed index into the (episode-local) choice list, from either end."""
    best, bp = -1.0, None
    kmax = max((len(e.cs) for e in tr), default=0)
    for frm_end in (False, True):
        for i in range(min(kmax, 24)):
            hit = 0
            for e in tr:
                if i < len(e.cs) and e.cs[-1 - i if frm_end else i] == e.a:
                    hit += 1
            s = hit / max(1, len(tr))
            if s > best:
                best, bp = s, (frm_end, i)
    if bp is None:
        return [""] * len(te)
    frm_end, i = bp
    return [e.choices[-1 - i if frm_end else i] if i < len(e.choices) else ""
            for e in te]


# --------------------------------------- B. choice-set surface exploits

def _choice_len_rank(tr, te, key):
    best, bp = -1.0, None
    kmax = max((len(e.cs) for e in tr), default=0)
    for desc in (False, True):
        for r in range(min(kmax, 24)):
            hit = 0
            for e in tr:
                o = sorted(range(len(e.cs)), key=lambda j: (key(e, j), j), reverse=desc)
                if r < len(o) and e.cs[o[r]] == e.a:
                    hit += 1
            s = hit / max(1, len(tr))
            if s > best:
                best, bp = s, (desc, r)
    if bp is None:
        return [""] * len(te)
    desc, r = bp
    out = []
    for e in te:
        o = sorted(range(len(e.cs)), key=lambda j: (key(e, j), j), reverse=desc)
        out.append(e.choices[o[r]] if r < len(o) else "")
    return out


def ex_choice_by_length(tr, te):
    return _choice_len_rank(tr, te, lambda e, j: len(e.cs[j]))


def ex_choice_length_outlier(tr, te):
    """Pick the choice whose length is furthest from the others' median.

    Catches distractors that were not drawn to match the answer's surface.
    """
    def key(e, j):
        ls = [len(c) for k, c in enumerate(e.cs) if k != j]
        med = sorted(ls)[len(ls) // 2] if ls else 0
        return abs(len(e.cs[j]) - med)
    return _choice_len_rank(tr, te, key)


def ex_choice_global_prior(tr, te):
    """Pick the choice string that was most often the answer in training."""
    pri = Counter(e.a for e in tr)
    out = []
    for e in te:
        if not e.choices:
            out.append("")
            continue
        j = max(range(len(e.cs)), key=lambda k: (pri.get(e.cs[k], 0), -k))
        out.append(e.choices[j])
    return out


# ----------------------------------- C. copying from the observation text

def _occ(e, c):
    return e.obs_l.count(c) if c else 0


def _first_pos(e, c):
    p = e.obs_l.find(c) if c else -1
    return p if p >= 0 else 10 ** 9


def ex_choice_in_obs(tr, te):
    """Family: choose by where/whether the choice string occurs in the text."""
    cands = {
        "earliest": lambda e, j: (_first_pos(e, e.cs[j]), j),
        "latest": lambda e, j: (-_first_pos(e, e.cs[j]) if _first_pos(e, e.cs[j]) < 10 ** 9 else 10 ** 9, j),
        "most": lambda e, j: (-_occ(e, e.cs[j]), j),
        "least": lambda e, j: (_occ(e, e.cs[j]), j),
        "absent": lambda e, j: (_occ(e, e.cs[j]) > 0, j),
    }
    best, bk = -1.0, None
    for k, f in cands.items():
        hit = 0
        for e in tr:
            if not e.cs:
                continue
            j = min(range(len(e.cs)), key=lambda x: f(e, x))
            hit += e.cs[j] == e.a
        s = hit / max(1, len(tr))
        if s > best:
            best, bk = s, k
    f = cands[bk]
    out = []
    for e in te:
        if not e.cs:
            out.append("")
            continue
        out.append(e.choices[min(range(len(e.cs)), key=lambda x: f(e, x))])
    return out


def _token_pick(e, kind):
    ts = e.raw
    if not ts:
        return ""
    if kind == "first":
        return ts[0]
    if kind == "last":
        return ts[-1]
    if kind == "longest":
        return max(ts, key=len)
    if kind == "shortest":
        return min(ts, key=len)
    c = Counter(e.alnum)
    if not c:
        return ""
    if kind == "most_common":
        return c.most_common(1)[0][0]
    return min(c.items(), key=lambda kv: (kv[1], kv[0]))[0]     # rarest


def ex_copy_salient_token(tr, te):
    kinds = ("first", "last", "longest", "shortest", "most_common", "rarest")
    best, bk = -1.0, kinds[0]
    for k in kinds:
        s = acc([_token_pick(e, k) for e in tr], tr)
        if s > best:
            best, bk = s, k
    return [_token_pick(e, bk) for e in te]


def ex_copy_fixed_offset(tr, te, span=48):
    """The token at a fixed index, from the left or the right of the text."""
    best, bp = -1.0, None
    for frm_end in (False, True):
        for i in range(span):
            hit = 0
            for e in tr:
                ts = e.raw
                if i < len(ts) and norm(ts[-1 - i if frm_end else i]) == e.a:
                    hit += 1
            s = hit / max(1, len(tr))
            if s > best:
                best, bp = s, (frm_end, i)
    if bp is None:
        return [""] * len(te)
    frm_end, i = bp
    out = []
    for e in te:
        ts = e.raw
        out.append(ts[-1 - i if frm_end else i] if i < len(ts) else "")
    return out


def ex_copy_near_keyword(tr, te, max_keys=80):
    """The token at a fixed offset from a fixed keyword.

    The keyword is drawn from tokens that appear in most training observations,
    so this is the "answer sits next to a template word" exploit.
    """
    df = Counter()
    for e in tr:
        df.update(set(e.alnum))
    n = max(1, len(tr))
    keys = [w for w, c in df.most_common(max_keys) if c >= 0.7 * n]
    best, bp = -1.0, None
    for kw in keys:
        for d in (-3, -2, -1, 1, 2, 3):
            hit = 0
            for e in tr:
                t = _near(e, kw, d)
                if t and norm(t) == e.a:
                    hit += 1
            s = hit / n
            if s > best:
                best, bp = s, (kw, d)
    if bp is None:
        return [""] * len(te)
    kw, d = bp
    return [_near(e, kw, d) for e in te]


def _near(e, kw, d):
    ts = e.alnum
    try:
        i = ts.index(kw)
    except ValueError:
        return ""
    j = i + d
    return ts[j] if 0 <= j < len(ts) else ""


# ------------------------------------------------ D. statistical features

def ex_prompt_length(tr, te):
    tbl, dflt = _lookup_fit([len(e.obs) for e in tr], [e.a for e in tr],
                            _majority([e.a for e in tr]))
    return _lookup_apply(tbl, dflt, [len(e.obs) for e in te])


def ex_byte_at_offset(tr, te, span=160):
    """A single byte at the best fixed offset -> majority answer (a control)."""
    dflt = _majority([e.a for e in tr])
    best, bp, btbl = -1.0, None, {}
    for frm_end in (False, True):
        for i in range(span):
            keys = [(e.obs[-1 - i] if i < len(e.obs) else "\x00") if frm_end
                    else (e.obs[i] if i < len(e.obs) else "\x00") for e in tr]
            tbl, _ = _lookup_fit(keys, [e.a for e in tr], dflt)
            hit = sum(1 for k, e in zip(keys, tr) if tbl.get(k, dflt) == e.a)
            s = hit / max(1, len(tr))
            if s > best:
                best, bp, btbl = s, (frm_end, i), tbl
    if bp is None:
        return [dflt] * len(te)
    frm_end, i = bp
    keys = [(e.obs[-1 - i] if i < len(e.obs) else "\x00") if frm_end
            else (e.obs[i] if i < len(e.obs) else "\x00") for e in te]
    return _lookup_apply(btbl, dflt, keys)


def _charmat(eps, vocab):
    m = np.zeros((len(eps), len(vocab) + 1), dtype=np.float32)
    idx = {c: i for i, c in enumerate(vocab)}
    for r, e in enumerate(eps):
        for ch, n in Counter(e.obs).items():
            j = idx.get(ch)
            if j is not None:
                m[r, j] = n
        m[r, -1] = len(e.obs)
    return m


def _vocab(tr):
    c = Counter()
    for e in tr:
        c.update(set(e.obs))
    return [ch for ch, _ in c.most_common(128)]


def _normrows(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def ex_bag_of_chars(tr, te):
    """Nearest class centroid over character counts.

    Predicts the answer string when the lesson has a small global answer
    vocabulary, and otherwise the *index* of the answer among the choices --
    the same feature set, applied to whichever target it can reach.
    """
    if not tr:
        return [""] * len(te)
    v = _vocab(tr)
    A, B = _normrows(_charmat(tr, v)), _normrows(_charmat(te, v))
    out = []
    labs = [e.a for e in tr]
    if len(set(labs)) <= 40:
        cls = sorted(set(labs))
        cen = _normrows(np.stack([A[[i for i, l in enumerate(labs) if l == c]].mean(0)
                                  for c in cls]))
        out.append([cls[i] for i in (B @ cen.T).argmax(1)])
    idxs = [e.cs.index(e.a) if e.a in e.cs else -1 for e in tr]
    ks = sorted({i for i in idxs if i >= 0})
    if ks:
        cen = _normrows(np.stack([A[[i for i, l in enumerate(idxs) if l == k]].mean(0)
                                  for k in ks]))
        pick = (B @ cen.T).argmax(1)
        out.append([e.choices[ks[p]] if ks[p] < len(e.choices) else ""
                    for e, p in zip(te, pick)])
    if not out:
        return [""] * len(te)
    return max(out, key=lambda p: acc(p, te))


def ex_nearest_neighbour(tr, te):
    """1-NN retrieval over training observations, by character-count cosine.

    Copies the neighbour's answer string, and separately the neighbour's answer
    *position*; the second is what memorising a small answer set looks like when
    the vocabulary is invented per episode.
    """
    if not tr:
        return [""] * len(te)
    v = _vocab(tr)
    A, B = _normrows(_charmat(tr, v)), _normrows(_charmat(te, v))
    nn = (B @ A.T).argmax(1)
    p1 = [tr[i].answer for i in nn]
    p2 = []
    for e, i in zip(te, nn):
        j = tr[i].cs.index(tr[i].a) if tr[i].a in tr[i].cs else 0
        p2.append(e.choices[j] if j < len(e.choices) else "")
    return max((p1, p2), key=lambda p: acc(p, te))


# --------------------------------------------- E. a fitted choice ranker

def _cfeat(e, j):
    c, cl = e.cs[j], e.choices[j]
    ls = [len(x) for x in e.cs] or [0]
    mean = sum(ls) / len(ls)
    occ = _occ(e, c)
    fp = _first_pos(e, c)
    return [
        1.0,
        len(c) / 16.0,
        (len(c) - mean) / 16.0,
        1.0 if occ else 0.0,
        min(occ, 8) / 8.0,
        (fp / max(1, len(e.obs))) if fp < 10 ** 9 else 1.5,
        j / max(1, len(e.cs) - 1) if len(e.cs) > 1 else 0.0,
        1.0 if cl[:1].isdigit() else 0.0,
        1.0 if any(ch.isdigit() for ch in cl) else 0.0,
        len(set(c) & set(e.obs_l)) / 16.0,
        1.0 if c and c in e.obs_l.split() else 0.0,
    ]


def ex_choice_ranker(tr, te, epochs=12):
    """An averaged perceptron over per-choice surface features.

    Not a model of the lesson -- a model of how the distractors were drawn.  If
    it wins, the answer is separable from its distractors on surface statistics
    alone.
    """
    d = len(_cfeat(Episode("x", "x", "a", ["a"]), 0))
    w = np.zeros(d, dtype=np.float64)
    acc_w = np.zeros(d, dtype=np.float64)
    n = 0
    F = [(np.array([_cfeat(e, j) for j in range(len(e.cs))]),
          e.cs.index(e.a) if e.a in e.cs else -1) for e in tr]
    for _ in range(epochs):
        for X, y in F:
            if y < 0 or len(X) < 2:
                continue
            p = int((X @ w).argmax())
            if p != y:
                w += X[y] - X[p]
            acc_w += w
            n += 1
    W = acc_w / n if n else w
    out = []
    for e in te:
        if not e.choices:
            out.append("")
            continue
        X = np.array([_cfeat(e, j) for j in range(len(e.cs))])
        out.append(e.choices[int((X @ W).argmax())])
    return out


def ex_train_lookup(tr, te):
    """Exact observation -> training majority answer.  Pure memorisation.

    Scores above the constant only when the lesson emits the *same* text for
    different seeds, which is the signature of an episode space too small to
    need solving.
    """
    dflt = _majority([e.a for e in tr])
    tbl, _ = _lookup_fit([e.obs for e in tr], [e.answer for e in tr], dflt)
    return _lookup_apply(tbl, dflt, [e.obs for e in te])


# ---------------------------------------------- a small decision tree

def _tree_fit(X, y, depth, min_leaf=4):
    """Greedy depth-limited tree on integer-ish features (Gini)."""
    lab, cnt = np.unique(y, return_counts=True)
    node = {"leaf": lab[cnt.argmax()]}
    if depth == 0 or len(lab) == 1 or len(y) < 2 * min_leaf:
        return node
    n = len(y)
    base = 1.0 - ((cnt / n) ** 2).sum()
    best = (0.0, None, None)
    for f in range(X.shape[1]):
        col = X[:, f]
        qs = np.unique(np.quantile(col, np.linspace(0.05, 0.95, 12)))
        for t in qs:
            m = col <= t
            a, b = int(m.sum()), n - int(m.sum())
            if a < min_leaf or b < min_leaf:
                continue
            g = 0.0
            for side, k in ((m, a), (~m, b)):
                _, c = np.unique(y[side], return_counts=True)
                g += (k / n) * (1.0 - ((c / k) ** 2).sum())
            if base - g > best[0] + 1e-12:
                best = (base - g, f, float(t))
    if best[1] is None:
        return node
    _, f, t = best
    m = X[:, f] <= t
    return {"f": f, "t": t,
            "l": _tree_fit(X[m], y[m], depth - 1, min_leaf),
            "r": _tree_fit(X[~m], y[~m], depth - 1, min_leaf)}


def _tree_apply(node, X):
    out = np.empty(len(X), dtype=object)
    for i, row in enumerate(X):
        nd = node
        while "f" in nd:
            nd = nd["l"] if row[nd["f"]] <= nd["t"] else nd["r"]
        out[i] = nd["leaf"]
    return out


def ex_char_count_tree(tr, te, depth=5):
    """A depth-4 decision tree over character counts.

    This is the family that defeats a bracket-counting lesson: it can express
    ``count('(') - count(')') == 0`` as two thresholds and nothing else.
    """
    if not tr:
        return [""] * len(te)
    v = _vocab(tr)
    A, B = _charmat(tr, v), _charmat(te, v)
    # Pairwise count differences over the most *variable* characters, so that
    # "#( equals #)" is one feature rather than an unreachable interaction.
    top = list(np.argsort(-A[:, :len(v)].var(0))[:12]) if len(v) > 1 else []
    pairs = [(i, j) for a, i in enumerate(top) for j in top[a + 1:]]
    if pairs:
        A = np.concatenate([A, np.stack([A[:, i] - A[:, j] for i, j in pairs], 1)], 1)
        B = np.concatenate([B, np.stack([B[:, i] - B[:, j] for i, j in pairs], 1)], 1)
    cands = []
    labs = np.array([e.a for e in tr], dtype=object)
    if len(set(labs)) <= 40:
        cands.append(list(_tree_apply(_tree_fit(A, labs, depth), B)))
    idxs = np.array([e.cs.index(e.a) if e.a in e.cs else -1 for e in tr])
    if (idxs >= 0).any():
        pick = _tree_apply(_tree_fit(A, idxs, depth), B)
        cands.append([e.choices[int(p)] if 0 <= int(p) < len(e.choices) else ""
                      for e, p in zip(te, pick)])
    if not cands:
        return [""] * len(te)
    return max(cands, key=lambda p: acc(p, te))


# ------------------------------------------------------------- registry

EXPLOITS: dict[str, Callable] = {
    "constant": ex_constant,
    "const_choice_index": ex_const_choice_index,
    "choice_by_length": ex_choice_by_length,
    "choice_length_outlier": ex_choice_length_outlier,
    "choice_global_prior": ex_choice_global_prior,
    "choice_in_observation": ex_choice_in_obs,
    "copy_salient_token": ex_copy_salient_token,
    "copy_fixed_offset": ex_copy_fixed_offset,
    "copy_near_keyword": ex_copy_near_keyword,
    "prompt_length": ex_prompt_length,
    "byte_at_offset": ex_byte_at_offset,
    "bag_of_chars": ex_bag_of_chars,
    "nearest_neighbour": ex_nearest_neighbour,
    "train_lookup": ex_train_lookup,
    "char_count_tree": ex_char_count_tree,
    "choice_ranker": ex_choice_ranker,
}


def answer_stats(eps: Sequence[Episode]) -> dict:
    c = Counter(e.a for e in eps)
    n = max(1, len(eps))
    ent = -sum((v / n) * math.log2(v / n) for v in c.values())
    k = [len(e.cs) for e in eps]
    obs = Counter(e.obs for e in eps)
    return {
        "n": len(eps),
        "distinct_answers": len(c),
        "answer_entropy_bits": round(ent, 3),
        "majority_share": round(c.most_common(1)[0][1] / n, 4) if c else 0.0,
        "mean_choices": round(sum(k) / n, 2),
        "min_choices": min(k) if k else 0,
        "random_floor": round(sum(1.0 / max(1, x) for x in k) / n, 4),
        "answer_in_choices": round(sum(1.0 for e in eps if e.a in e.cs) / n, 4),
        "answer_in_observation": round(
            sum(1.0 for e in eps if e.a and e.a in e.obs_l) / n, 4),
        "repeat_observation_rate": round(1.0 - len(obs) / n, 4),
    }


def run_battery(train: Sequence[Episode], test: Sequence[Episode]) -> dict:
    res = {}
    for name, fn in EXPLOITS.items():
        try:
            res[name] = round(acc(fn(train, test), test), 4)
        except Exception as exc:                      # pragma: no cover
            res[name] = float("nan")
            res[name + "_error"] = f"{type(exc).__name__}: {exc}"
    return res
