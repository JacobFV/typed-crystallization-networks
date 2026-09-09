"""A self-contained CDCL SAT solver.

Neither `python-sat` nor `z3` is installed in `.venv`, and the brief forbids
installing anything into the project environment, so this is a from-scratch
CDCL implementation: two-watched-literal propagation, 1-UIP clause learning,
non-chronological backjumping, VSIDS-style activity with decay, phase saving,
and Luby restarts.  It is entirely adequate at the sizes this track needs
(hundreds of variables, tens of thousands of clauses).  It is the honest
substitute for MiniSat/CaDiCaL/z3, not a claim to match them: every solver
wall-clock number in RESULTS.md is an UPPER bound on what a production solver
would take, typically by one to two orders of magnitude.

Literals are DIMACS integers: `v` for the positive literal of variable `v >= 1`
and `-v` for its negation.
"""
from __future__ import annotations


def luby(i, base=100):
    """Luby restart sequence, 1-indexed."""
    k = 1
    while (1 << (k + 1)) - 1 < i:
        k += 1
    while (1 << k) - 1 != i:
        i -= (1 << k) - 1
        k = 1
        while (1 << (k + 1)) - 1 < i:
            k += 1
    return base * (1 << (k - 1))


class Solver:
    def __init__(self, nvars=0):
        self.nvars = nvars
        self.clauses = []
        self.watches = {}
        self.value = {}
        self.level = {}
        self.reason = {}
        self.trail = []
        self.lim = []
        self.qhead = 0
        self.activity = {}
        self.phase = {}
        self.bump = 1.0
        self.decisions = 0
        self.propagations = 0
        self.conflicts = 0
        self.learned = 0
        self.unsat = False

    @staticmethod
    def _lit(x):
        return 2 * abs(x) + (1 if x < 0 else 0)

    def _sat(self, l):
        v = self.value.get(l >> 1)
        if v is None:
            return None
        return v == (0 if l & 1 else 1)

    def new_var(self):
        self.nvars += 1
        self.activity.setdefault(self.nvars, 0.0)
        return self.nvars

    def add_clause(self, lits):
        internal = []
        seen = set()
        for x in lits:
            l = self._lit(x)
            if l ^ 1 in seen:
                return
            if l not in seen:
                seen.add(l)
                internal.append(l)
        for l in internal:
            self.nvars = max(self.nvars, l >> 1)
            self.activity.setdefault(l >> 1, 0.0)
        if not internal:
            self.unsat = True
            return
        if len(internal) == 1:
            if not self._enqueue(internal[0], None):
                self.unsat = True
            return
        idx = len(self.clauses)
        self.clauses.append(internal)
        self.watches.setdefault(internal[0], []).append(idx)
        self.watches.setdefault(internal[1], []).append(idx)

    def _enqueue(self, l, reason):
        v = l >> 1
        val = 0 if l & 1 else 1
        if v in self.value:
            return self.value[v] == val
        self.value[v] = val
        self.level[v] = len(self.lim)
        self.reason[v] = reason
        self.trail.append(l)
        return True

    def _propagate(self):
        while self.qhead < len(self.trail):
            l = self.trail[self.qhead]
            self.qhead += 1
            self.propagations += 1
            false_lit = l ^ 1
            ws = self.watches.get(false_lit)
            if not ws:
                continue
            keep = []
            self.watches[false_lit] = keep
            i = 0
            while i < len(ws):
                ci = ws[i]
                i += 1
                c = self.clauses[ci]
                if c[0] == false_lit:
                    c[0], c[1] = c[1], c[0]
                if self._sat(c[0]) is True:
                    keep.append(ci)
                    continue
                moved = False
                for j in range(2, len(c)):
                    if self._sat(c[j]) is not False:
                        c[1], c[j] = c[j], c[1]
                        self.watches.setdefault(c[1], []).append(ci)
                        moved = True
                        break
                if moved:
                    continue
                keep.append(ci)
                if self._sat(c[0]) is False:
                    keep.extend(ws[i:])
                    return ci
                self._enqueue(c[0], ci)
        return None

    def _analyze(self, confl):
        seen = set()
        learnt = [0]
        counter = 0
        p = None
        idx = len(self.trail) - 1
        blevel = 0
        current = len(self.lim)
        while True:
            c = self.clauses[confl]
            for l in c:
                if l == p:
                    continue
                v = l >> 1
                lv = self.level.get(v, 0)
                if v not in seen and lv > 0:
                    seen.add(v)
                    self.activity[v] = self.activity.get(v, 0.0) + self.bump
                    if lv >= current:
                        counter += 1
                    else:
                        learnt.append(l)
                        blevel = max(blevel, lv)
            while True:
                p = self.trail[idx]
                idx -= 1
                if (p >> 1) in seen:
                    break
            seen.discard(p >> 1)
            counter -= 1
            if counter <= 0:
                break
            confl = self.reason[p >> 1]
        learnt[0] = p ^ 1
        if len(learnt) > 2:
            learnt[1:] = sorted(learnt[1:], key=lambda l: -self.level.get(l >> 1, 0))
        return learnt, blevel

    def _backtrack(self, level):
        while len(self.lim) > level:
            start = self.lim.pop()
            for l in self.trail[start:]:
                v = l >> 1
                self.phase[v] = self.value[v]
                del self.value[v]
                self.level.pop(v, None)
                self.reason.pop(v, None)
            del self.trail[start:]
        if self.qhead > len(self.trail):
            self.qhead = len(self.trail)

    def _decide(self):
        best, score = None, -1.0
        for v in range(1, self.nvars + 1):
            if v not in self.value:
                a = self.activity.get(v, 0.0)
                if a > score:
                    best, score = v, a
        if best is None:
            return None
        self.decisions += 1
        return 2 * best + (0 if self.phase.get(best, 0) == 1 else 1)

    def solve(self, max_conflicts=None, max_seconds=None):
        if self.unsat:
            return None
        import time as _time
        deadline = None if max_seconds is None else _time.perf_counter() + max_seconds
        self.qhead = 0
        restart_index = 1
        restart_at = luby(restart_index)
        since_restart = 0
        while True:
            confl = self._propagate()
            if confl is not None:
                self.conflicts += 1
                since_restart += 1
                if not self.lim:
                    return None
                learnt, blevel = self._analyze(confl)
                self._backtrack(blevel)
                if len(learnt) == 1:
                    self._backtrack(0)
                    if not self._enqueue(learnt[0], None):
                        return None
                else:
                    idx = len(self.clauses)
                    self.clauses.append(learnt)
                    self.learned += 1
                    self.watches.setdefault(learnt[0], []).append(idx)
                    self.watches.setdefault(learnt[1], []).append(idx)
                    self._enqueue(learnt[0], idx)
                self.bump *= 1.05
                if self.bump > 1e60:
                    for v in list(self.activity):
                        self.activity[v] /= 1e60
                    self.bump /= 1e60
                if max_conflicts is not None and self.conflicts > max_conflicts:
                    return "unknown"
                if deadline is not None and self.conflicts % 256 == 0 and _time.perf_counter() > deadline:
                    return "unknown"
                if since_restart >= restart_at:
                    since_restart = 0
                    restart_index += 1
                    restart_at = luby(restart_index)
                    self._backtrack(0)
                continue
            l = self._decide()
            if l is None:
                return {v: bool(self.value[v]) for v in range(1, self.nvars + 1) if v in self.value}
            self.lim.append(len(self.trail))
            self._enqueue(l, None)


def exactly_one(solver, lits):
    solver.add_clause(lits)
    for i in range(len(lits)):
        for j in range(i + 1, len(lits)):
            solver.add_clause([-lits[i], -lits[j]])
