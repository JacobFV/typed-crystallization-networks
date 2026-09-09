"""Choose the search backend from measured properties of the problem.

Six tracks measured where enumeration wins and where relaxation wins. The rule
is no longer speculative, so the caller should not have to guess it.

    enumeration wins  small, exactly checkable spaces (0.081 ms for 256
                      programs and 41 ms for 96, against 10-36 s of gradient
                      descent) and noisy partial credit, and only enumeration
                      returns a uniqueness certificate
    relaxation wins   large spaces (6/6 exact on 4.3e9 programs where brute
                      force projects to 107 days) and environment-coupled
                      search (5/5 against random search's 0/20 at a matched
                      704-step budget where a full sweep needs 16,384)
    hybrid wins       when the discrete structure is small but the program
                      carries continuous constants THIS supervision can move:
                      enumerate the structure, fit the constants inside it

    relaxation loses  behind a `gradient="none"` boundary, and wherever a
                      candidate's surrogate is dead at the data's operating
                      scale -- `eq` at tau=1 is exactly 0.0 past |a-b| >= 11

Every input to the rule is measured on the scaffold and the data at hand: the
space size, which trainable constants receive gradient from these signals, the
declared gradient boundaries, the surrogate's liveness under the actual values,
and the cost of one exact evaluation and of one constant fit. Two things are
worth stating because they were mistakes the first version of this rule made.
A constant declared trainable but off every supervised path is not a continuous
sub-problem, and treating it as one chose the hybrid over enumeration at a
thousand-fold cost on `examples/joint.py`. And the hybrid's price is not
enumeration's price: it pays a gradient fit per structure, so it is measured
separately or the rule justifies a choice on a budget that choice then blows
through.

The one property the selector cannot measure is whether the examples cost
environment rollouts, because that is a property of where the data came from and
not of the program; the caller declares it as `rollout_cost`.

Thresholds and their provenance are in `research/search-selection/RESULTS.md`
and repeated on each constant below. A threshold with no measurement behind it
is labelled a guess.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import itertools
import math
import random
import time
import torch
from .learning import SoftProgram, carrier_temperature, relaxed, tensor
from .operators import Registry
from .search import candidate_counts, space_size, evaluate, enumerate_fit

# --- thresholds -------------------------------------------------------------

DEAD_GRADIENT = 1e-12
"""Below this a surrogate is not differentiable in any usable sense.

Provenance, partly measured and partly derived. Measured: `eq`'s relaxation at
tau=1 is *exactly* 0.0 in float32 for |a-b| >= 11, so on the case that produced
the wrong "address relaxation is worse than chance" conclusion the reading is
0.0 and any threshold in [0, 1e-12] gives the same verdict. Derived: Adam's
update is `lr * g / (sqrt(v) + eps)` with eps = 1e-8, so a gradient of magnitude
g moves a logit by at most `lr * g / eps` per step; at the shipped lr = .05 and
a 1000-step budget, g = 1e-12 buys a total displacement of 5e-3, which cannot
reorder a softmax over candidates. It is a bound, not a fit."""

ENUMERATION_SECONDS = 30.
"""Wall-clock budget above which enumeration is not worth starting.

Provenance: measured, not chosen. FINDINGS section 7 records one gradient run on
the flagship scaffolds at 10-36 s. 30 s is inside that band, so enumeration is
selected only where it is projected to beat the method it replaces. The
projection itself is measured on the actual scaffold and data rather than
assumed, so this constant only fixes *how much* enumeration may cost, never how
big a space may be."""

ENUMERATION_PROBES = 12
"""Selections timed to estimate the per-program cost. A guess, and cheap: the
variance across candidate programs is what it measures, and 12 was enough for
the standard error to fall under 10% of the mean on every fixture in the repo
(reported in RESULTS.md). Nothing downstream is sensitive to it."""

ENVIRONMENT_BUDGET = 704
"""Environment steps the relaxation path was measured to need on the joint task.

Provenance: measured. FINDINGS section 7 -- the gradient path succeeds 5/5 at
its own budget of 704 environment steps where random search over the same 256
candidates succeeds 0/20, and a full enumerative sweep needs 16,384. So where a
sweep would exceed this, rollouts rather than CPU are the scarce resource and
the ordering flips."""


# --- surrogate liveness -----------------------------------------------------

@dataclass(frozen=True)
class CandidateLiveness:
    node: str
    index: int
    operator: str
    boundary: bool          # declares gradient="none"
    valid: bool             # the relaxation ran at all on this data
    sensitivity: float      # peak |d output / d inputs| over the batch
    live_rows: int = 0      # examples on which that derivative is representable
    rows: int = 0
    detail: str = ""
    @property
    def dead(self): return self.boundary or not self.valid or self.live_rows==0

@dataclass
class Liveness:
    candidates: tuple
    node_signal: dict       # node -> max |d probe_loss / d choice logits|
    dead_nodes: tuple       # free nodes with no live candidate, or no task signal
    blind_nodes: tuple      # free nodes where SOME candidate can never be chosen
    boundary_nodes: tuple
    reachable_fraction: float
    eq_spread: dict         # node -> observed max |a-b| per eq candidate
    carrier_scaled: bool
    constant_signal: dict = field(default_factory=dict)   # trainable constant -> |d loss / d constant|
    def to_dict(self):
        return {'carrier_scaled':self.carrier_scaled,'reachable_fraction':self.reachable_fraction,
                'dead_nodes':list(self.dead_nodes),'blind_nodes':list(self.blind_nodes),
                'boundary_nodes':list(self.boundary_nodes),
                'node_signal':{k:float(v) for k,v in self.node_signal.items()},
                'constant_signal':{k:float(v) for k,v in self.constant_signal.items()},
                'live_constants':list(self.live_constants),
                'eq_spread':{k:[float(x) for x in v] for k,v in self.eq_spread.items()},
                'candidates':[asdict(c)|{'dead':c.dead} for c in self.candidates]}
    @property
    def alive(self): return not self.dead_nodes
    @property
    def live_constants(self):
        """Trainable constants this supervision can actually move.

        A constant declared trainable but sitting off every supervised path --
        `examples/joint.py` declares five, all on the policy readout, none
        reachable from the probe signals -- is not a continuous sub-problem for
        this data. Fitting it is work that cannot change the answer, so the
        hybrid backend has nothing to add over plain enumeration.
        """
        return tuple(k for k,v in self.constant_signal.items() if v>DEAD_GRADIENT)

def _stacked(program, examples, signals):
    inputs={k:torch.stack([tensor(ex['inputs'][k]) for ex in examples]) for k,_ in program.inputs}
    targets={s.target:torch.stack([tensor(ex['targets'][s.target]) for ex in examples]) for s in signals}
    return inputs,targets

def liveness(program, examples, signals, registry=None, carrier_scaled=False, inits=(0.,.5), seed=0):
    """Liveness at more than one initialization, combined optimistically.

    `SoftProgram` zero-initializes every choice logit, and at that point a
    uniform mixture over the sixteen two-input truth tables is the constant
    0.5 on every row, so a balanced target makes the choice gradient cancel
    *exactly*. That is a property of the initialization, not of the relaxation --
    `examples/joint.py` carries a residual initialization for precisely this
    reason -- so reading it as a dead surrogate would be the same class of
    mistake this check exists to prevent. Every reading is therefore taken at
    the shipped zero initialization and at a perturbed one, and a candidate is
    dead only where it is dead at both.
    """
    parts=[_liveness_at(program,examples,signals,registry,carrier_scaled,n,seed) for n in inits]
    return parts[0] if len(parts)==1 else _combine(parts)

def _combine(parts):
    best={}
    for lv in parts:
        for c in lv.candidates:
            k=(c.node,c.index); prev=best.get(k)
            if prev is None or c.live_rows>prev.live_rows or (c.live_rows==prev.live_rows and c.sensitivity>prev.sensitivity):
                best[k]=c
    rows=tuple(best[k] for k in sorted(best))
    signal={k:max(lv.node_signal.get(k,0.) for lv in parts) for k in parts[0].node_signal}
    by={}
    for row in rows: by.setdefault(row.node,[]).append(row)
    dead=tuple(name for name in signal
               if all(row.dead for row in by.get(name,())) or signal[name]<=DEAD_GRADIENT)
    blind=tuple(name for name in signal if name not in dead and any(row.dead for row in by.get(name,())))
    reachable=1.
    for name in signal:
        rs=by.get(name,())
        if rs: reachable*=sum(not row.dead for row in rs)/len(rs)
    const_signal={k:max(lv.constant_signal.get(k,0.) for lv in parts) for k in parts[0].constant_signal}
    return Liveness(rows,signal,dead,blind,parts[0].boundary_nodes,reachable,parts[0].eq_spread,
                    parts[0].carrier_scaled,const_signal)

def _liveness_at(program, examples, signals, registry=None, carrier_scaled=False, noise=0., seed=0, model=None):
    """Is every candidate's relaxation non-degenerate at *this* data?

    Two measurements, because they fail for different reasons and only both
    together say "relaxation is viable here".

    **Local.** For each candidate of each free node, the relaxation is evaluated
    on the values that actually reach it in the soft forward pass, and its
    sensitivity to those values is read off by autograd **per example**. This is
    the `eq` check: `exp(-(a-b)^2/tau)` on bytes 25.6 apart returns exactly 0.0
    and its derivative with it, so the candidate is reported dead rather than
    differentiable. It is not `eq`-specific -- a saturated `lt` sigmoid, a
    `gradient="none"` module and an out-of-domain `log` are caught the same way.

    Per example, and not in aggregate, because that is what the shipped
    behaviour of the failing arm looks like: on the recorded rung-3 scaffold one
    of the two colour candidates is live on 5 of 12 examples and the other on
    **0 of 12**, so a peak taken over the batch reads the node as differentiable
    while the candidate the reference program needs can never be selected. A
    candidate live on no example is dead however large the batch peak is.

    **Global.** The gradient of the *unregularized* probe loss with respect to
    each free node's choice logits, at the uniform initialization. A node whose
    reading is zero has no learning signal for its choice at all, which is what
    a `gradient="none"` operator anywhere downstream produces.

    FINDINGS section 3 warns that treating an all-zero gradient as disconnected
    is wrong inside the crystallizer, because a legitimately concentrated choice
    has zero entropy-gradient. That objection does not apply here: this is
    measured at the uniform initialization, before any concentration, and
    against the task objective with no regularizer attached. A zero here really
    is the absence of signal.
    """
    r=registry or Registry()
    m=model if model is not None else SoftProgram(program,r)
    m.carrier_scaled=carrier_scaled
    if noise:
        g=torch.Generator().manual_seed(seed*7919+13)
        with torch.no_grad():
            for p in m.choices:
                if p.requires_grad: p.add_(torch.randn(p.shape,generator=g)*noise)
    inputs,targets=_stacked(program,examples,signals)
    _,_,trace=m(inputs,return_trace=True)
    batch=len(examples)
    rows=[]; spreads={}
    for n in program.nodes:
        if n.name in m.frozen: continue
        tau=m.temperatures[n.name]*m.surrogate_scale[n.name]
        # Reported for every node, including plumbing with one candidate: a dead
        # surrogate there carries no choice of its own but still cuts every
        # upstream choice off from the loss, which the global reading below sees.
        for j,c in enumerate(n.candidates):
            op=c.operator
            if op.gradient=="none":
                rows.append(CandidateLiveness(n.name,j,op.name,True,True,0.,0,batch,"declared gradient boundary")); continue
            xs=[trace[s].detach().clone().requires_grad_(True) for s in c.sources]
            try:
                y=relaxed(r,op,xs,tau,carrier_scaled)
            except (ValueError,RuntimeError,NotImplementedError) as e:
                rows.append(CandidateLiveness(n.name,j,op.name,False,False,0.,0,batch,str(e)[:80])); continue
            if op.name=="eq":
                d=(xs[0]-xs[1]).detach().abs().max()
                spreads.setdefault(n.name,[]).append(float(d))
            if not y.requires_grad:
                rows.append(CandidateLiveness(n.name,j,op.name,False,True,0.,0,batch,
                                              "relaxation is constant in its inputs")); continue
            gs=torch.autograd.grad(y.sum(),xs,allow_unused=True,retain_graph=False)
            per_row=None
            for g in gs:
                if g is None: continue
                if g.dim()>=1 and g.shape[0]==batch:
                    v=g.detach().abs().reshape(batch,-1).max(-1).values
                    per_row=v if per_row is None else torch.maximum(per_row,v)
            if per_row is None:
                peak=max((float(g.detach().abs().max()) for g in gs if g is not None),default=0.)
                live=batch if peak>DEAD_GRADIENT else 0
            else:
                peak=float(per_row.max()); live=int((per_row>DEAD_GRADIENT).sum())
            rows.append(CandidateLiveness(n.name,j,op.name,False,True,peak,live,batch))
    # global: does the task objective reach each free node's choice logits?
    _,_,trace=m(inputs,return_trace=True)
    loss=m.probe_loss(trace,targets,signals)
    # A one-candidate node has a constant softmax and therefore an exactly zero
    # choice gradient by construction. That is not an absence of learning
    # signal, it is the absence of a decision, so it is not a free node here.
    free=[(n.name,p) for n,p in zip(program.nodes,m.choices)
          if n.name not in m.frozen and p.requires_grad and len(n.candidates)>1]
    consts=[(k,p) for k,p in m.constants.items() if p.requires_grad]
    signal={}; const_signal={}
    if (free or consts) and loss.requires_grad:
        gs=torch.autograd.grad(loss,[p for _,p in free]+[p for _,p in consts],allow_unused=True)
        for (name,_),g in zip(free,gs[:len(free)]): signal[name]=0. if g is None else float(g.abs().max())
        for (name,_),g in zip(consts,gs[len(free):]): const_signal[name]=0. if g is None else float(g.abs().max())
    else:
        for name,_ in free: signal[name]=0.
        for name,_ in consts: const_signal[name]=0.
    by_node={}
    for row in rows: by_node.setdefault(row.node,[]).append(row)
    dead=tuple(name for name,_ in free
               if all(row.dead for row in by_node.get(name,())) or signal.get(name,0.)<=DEAD_GRADIENT)
    blind=tuple(name for name,_ in free
                if name not in dead and any(row.dead for row in by_node.get(name,())))
    # The share of the discrete space gradient descent could reach at all. A
    # candidate that is dead on every example contributes nothing to its node's
    # mixture and receives nothing back, so no descent can ever select it: the
    # relaxed search is over the product of the LIVE candidates, not the
    # declared ones. On the recorded rung-3 arm that is a quarter of the space,
    # and the reference program is outside it.
    reachable=1.
    for name,_ in free:
        rs=by_node.get(name,())
        if rs: reachable*=sum(not row.dead for row in rs)/len(rs)
    boundary=tuple(sorted({row.node for row in rows if row.boundary}))
    return Liveness(tuple(rows),signal,dead,blind,boundary,reachable,spreads,carrier_scaled,const_signal)


# --- enumeration cost -------------------------------------------------------

def enumeration_cost(program, examples, signals, registry=None, tolerance=.001,
                     probes=ENUMERATION_PROBES, seed=0):
    """Measured seconds per exactly-evaluated program, and the projected sweep.

    The projection is `space_size * median seconds per program`, over uniformly
    sampled selections evaluated exactly as `enumerate_fit` evaluates them --
    same `evaluate`, same tolerance, same early exit. It therefore prices the
    real sweep on the real data rather than assuming a rate. Selections whose
    candidates are unusable on this data (an illegal numeric domain, an address
    off the end of a tuple) are timed too, since a sweep pays for those as well.

    Median, and one discarded warm-up probe, both for the same measured reason.
    The first probe pays program validation and interpreter warm-up that the
    other 9,215 will not, and on a busy host a single descheduled probe drags the
    mean by a large factor. Taken as a mean over 12 probes including the warm-up,
    this projection read 55.7 s for a sweep that then ran in 8.8 s -- a 6.3x
    over-estimate, and enough to push the decision across a 30 s budget and pick
    the wrong backend. That is this selector's noisiest input; see
    `research/search-selection/RESULTS.md`.
    """
    r=registry or Registry()
    counts=candidate_counts(program); total=math.prod(counts)
    rng=random.Random(seed); times=[]; usable=0; warmup=None
    for i in range(min(probes,total)+1):
        sel={n.name:rng.randrange(c) for n,c in zip(program.nodes,counts)}
        t0=time.perf_counter()
        err=evaluate(program,sel,examples,signals,r,tolerance)
        dt=time.perf_counter()-t0
        if i==0 and total>1: warmup=dt; continue
        times.append(dt); usable+=err is not None
    mean=sum(times)/len(times) if times else 0.
    ordered=sorted(times); n=len(ordered)
    median=0. if not n else (ordered[n//2] if n%2 else .5*(ordered[n//2-1]+ordered[n//2]))
    sd=(sum((t-mean)**2 for t in times)/max(1,n-1))**.5 if n>1 else 0.
    return {'space_size':total,'probes':n,'seconds_per_program':median,'mean_seconds_per_program':mean,
            'warmup_seconds':warmup,'standard_error':sd/max(1,n)**.5,
            'usable_fraction':usable/max(1,n),'projected_seconds':total*median,
            'projected_seconds_from_mean':total*mean}

HYBRID_FULL_FITS = 8
"""Full constant fits the hybrid's projection assumes it will need after screening.

A guess, and stated as one. The screen orders structures and the search stops at
the first that conforms, so the true number is data-dependent and cannot be known
before searching. 8 is a round number chosen to make the projection an
over-estimate on every case measured here (the largest observed was 1), so the
error is on the side that prices the hybrid out rather than in."""

def hybrid_cost(program, examples, signals, registry=None, probes=3, screen_steps=25,
                constant_steps=200, lr=.05, seed=0):
    """Measured seconds per structure for the hybrid, and the projected search.

    Enumeration's price does not carry over: the hybrid pays a gradient fit per
    structure rather than one exact evaluation, and on the reconstructed
    continuous fixture that is two orders of magnitude more per structure. Left
    unpriced, the rule would justify choosing the hybrid on a budget the hybrid
    then blows through, so it is measured the same way -- on this scaffold, this
    data, and this host.
    """
    r=registry or Registry()
    counts=candidate_counts(program); total=math.prod(counts)
    inputs,targets=_stacked(program,examples,signals)
    rng=random.Random(seed); screen=[]; full=[]
    for _ in range(min(probes,total)+1):
        sel={n.name:rng.randrange(c) for n,c in zip(program.nodes,counts)}
        for steps,bucket in ((screen_steps,screen),(constant_steps,full)):
            t0=time.perf_counter()
            try:
                m,_=_fit_constants(program,sel,inputs,targets,signals,r,steps,lr)
                evaluate(m.export(),{},examples,signals,r,None)
            except (TypeError,ValueError,RuntimeError,OverflowError): pass
            bucket.append(time.perf_counter()-t0)
    med=lambda xs:(0. if not xs else sorted(xs)[len(xs)//2] if len(xs)%2
                   else .5*(sorted(xs)[len(xs)//2-1]+sorted(xs)[len(xs)//2]))
    s=med(screen[1:] or screen); f=med(full[1:] or full)      # first probe is warm-up
    return {'space_size':total,'probes':len(screen),'screen_seconds_per_structure':s,
            'full_seconds_per_structure':f,'assumed_full_fits':HYBRID_FULL_FITS,
            'projected_seconds':total*s+HYBRID_FULL_FITS*f}


# --- the rule ---------------------------------------------------------------

@dataclass
class Decision:
    mode: str                       # 'enumerate' | 'relax' | 'hybrid'
    reason: str
    feasible: bool = True
    space_size: int = 0
    projected_enumeration_seconds: float = 0.
    projected_hybrid_seconds: float = 0.
    seconds_per_program: float = 0.
    trainable_constants: tuple = ()
    live_constants: tuple = ()
    boundary_nodes: tuple = ()
    dead_nodes: tuple = ()
    blind_nodes: tuple = ()
    reachable_fraction: float = 1.
    scale_surrogates: bool = False
    environment_sweep: int = 0
    certified: bool = False
    notes: tuple = ()
    liveness: dict = field(default_factory=dict)
    def to_dict(self):
        d=asdict(self)
        for k in ('trainable_constants','live_constants','boundary_nodes','dead_nodes','blind_nodes','notes'):
            d[k]=list(d[k])
        return d

def select_backend(program, examples, signals, registry=None, tolerance=.001,
                   rollout_cost=0, environment_budget=ENVIRONMENT_BUDGET,
                   enumeration_seconds=ENUMERATION_SECONDS, probes=ENUMERATION_PROBES,
                   allow_surrogate_scaling=True):
    """Decide between enumeration, relaxation and the hybrid, from measurements.

    `rollout_cost` is the number of environment steps one program evaluation
    consumes; 0 means the examples are already in hand and only CPU is spent.
    It is the one declared input, because nothing in a `Program` records where
    its examples came from.

    The order of the tests is itself a claim, and it is the asymmetry of the
    costs. Being wrong toward enumeration costs at most
    `projected_enumeration_seconds`, which has already been measured, and never
    returns a program that was never checked. Being wrong toward relaxation
    costs a 10-36 s run that may return nothing, with no certificate either way.
    So the environment test comes first (it is the one place a sweep is
    genuinely unaffordable), the affordability test second, and relaxation is
    reached only when enumeration has been priced out.
    """
    r=registry or Registry()
    program.validate(r); program.validate_signals(signals)
    if not examples: raise ValueError('training examples required')
    notes=[]
    constants=tuple(program.trainable_constants)
    counts=candidate_counts(program); total=math.prod(counts)
    free=[n.name for n,c in zip(program.nodes,counts) if n.selected is None and c>1]

    live=liveness(program,examples,signals,r,carrier_scaled=False)
    scaled=None
    if (live.dead_nodes or live.reachable_fraction<1.) and allow_surrogate_scaling:
        scaled=liveness(program,examples,signals,r,carrier_scaled=True)
        if (scaled.reachable_fraction>live.reachable_fraction and not scaled.dead_nodes):
            notes.append(f'carrier scaling lifts the gradient-reachable share of the space from '
                         f'{live.reachable_fraction:.3g} to {scaled.reachable_fraction:.3g}')
        else: scaled=None
    best=scaled if scaled is not None else live
    relax_ok=not best.dead_nodes
    # A constant is a continuous sub-problem only where this supervision can move
    # it. `examples/joint.py` declares five trainable constants, all on the policy
    # readout and none reachable from the probe signals, so the hybrid would spend
    # a thousand-fold on a fit that cannot change the answer.
    live_constants=best.live_constants
    if constants and not live_constants:
        notes.append('trainable constants '+','.join(constants)+' receive no gradient from this '
                     'supervision, so there is no continuous sub-problem to fit')

    def decide(mode,reason,feasible=True,cost=None,certified=False):
        if mode=='relax' and best.reachable_fraction<1.:
            notes.append(f'relaxation searches only {best.reachable_fraction:.3g} of the declared space; '
                         'candidates dead on every example cannot be selected by descent')
        return Decision(mode,reason,feasible,total,
                        0. if cost is None else cost['projected_seconds'],0.,
                        0. if cost is None else cost['seconds_per_program'],
                        constants,live_constants,live.boundary_nodes,best.dead_nodes,best.blind_nodes,
                        best.reachable_fraction,best is scaled,total*rollout_cost,
                        certified,tuple(notes),best.to_dict())

    if not free:
        return decide('hybrid' if live_constants else 'enumerate',
                      'no free discrete choice; '+('the continuous part is all that is left'
                                                   if live_constants else 'the program is already determined'),
                      certified=not live_constants)

    # 1. environment coupling. A sweep spends rollouts, not CPU.
    if rollout_cost>0 and total*rollout_cost>environment_budget:
        if relax_ok:
            return decide('relax',f'a full sweep costs {total*rollout_cost} environment steps against a '
                                  f'{environment_budget}-step budget; relaxation was measured 5/5 at that budget '
                                  f'where random search over the same candidates was 0/20')
        return decide('enumerate','the sweep is unaffordable in rollouts and the relaxation is dead: '
                                  +','.join(best.dead_nodes),feasible=False)

    # 2. affordability, priced on this scaffold and this data.
    cost=enumeration_cost(program,examples,signals,r,tolerance,probes)
    if cost['projected_seconds']<=enumeration_seconds:
        if live_constants:
            # The hybrid pays a gradient fit per structure, not one exact
            # evaluation, so it is priced separately against the same budget.
            # No certificate either: uniqueness of a discrete structure is only
            # meaningful once its constants are fitted, and two structures can
            # each be made to conform by different constants.
            hcost=hybrid_cost(program,examples,signals,r)
            notes.append(f"hybrid priced at {hcost['projected_seconds']:.3g}s "
                         f"({hcost['screen_seconds_per_structure']:.3g}s per structure to screen, "
                         f"{hcost['full_seconds_per_structure']:.3g}s to fit)")
            if hcost['projected_seconds']<=enumeration_seconds:
                d=decide('hybrid','the discrete space is exhaustible in '
                                  f"{cost['projected_seconds']:.3g}s and {len(live_constants)} constant(s) "
                                  '('+','.join(live_constants)+') must still be fitted inside it, '
                                  f"which prices the hybrid at {hcost['projected_seconds']:.3g}s",cost=cost)
                d.projected_hybrid_seconds=hcost['projected_seconds']; return d
            notes.append('the hybrid is priced out, so the continuous part falls to `fit`\'s polish '
                         'phase, which holds the selected structure hard and refines the constants '
                         'alone -- the same split, without the exhaustive outer loop')
            d=decide('relax',f"the discrete space is exhaustible in {cost['projected_seconds']:.3g}s but "
                             f"fitting {len(live_constants)} constant(s) inside every structure projects to "
                             f"{hcost['projected_seconds']:.4g}s",cost=cost)
            d.projected_hybrid_seconds=hcost['projected_seconds']; return d
        return decide('enumerate',f"the whole space of {total} programs is exhaustible in "
                                  f"{cost['projected_seconds']:.3g}s"
                                  +(', which also certifies uniqueness' if total<=(1<<20) else
                                    f", but {total} exceeds `enumerate_fit`'s default cap so no "
                                    'certificate is available'),
                      cost=cost,certified=total<=(1<<20))

    # 3. too large to exhaust: relaxation, if it is alive.
    if relax_ok:
        why=f"enumeration projects to {cost['projected_seconds']:.4g}s over {total} programs"
        if best is scaled: why+='; the surrogate needs carrier scaling to be differentiable here'
        return decide('relax',why,cost=cost)
    return decide('enumerate',f"neither backend is sound: enumeration projects to {cost['projected_seconds']:.4g}s "
                              'and the relaxation is dead at '+','.join(best.dead_nodes),feasible=False,cost=cost)


# --- the hybrid backend -----------------------------------------------------

def _single(program, selections):
    """The program with one candidate per node and its constants still trainable.

    `harden` alone is not enough: it sets `Node.selected`, and `SoftProgram`
    treats a selected node as frozen, which routes it through `exact_tensor` and
    detaches it. A one-candidate node with `selected=None` has a constant
    softmax -- no choice to make -- but keeps the relaxation, so a constant
    upstream of it still receives a gradient. This is the same split `fit`'s
    `polish` phase makes with `trials`, done ahead of time so each structure
    costs one candidate per node instead of all of them.
    """
    from dataclasses import replace
    nodes=tuple(replace(n,candidates=(n.candidates[n.selected if n.selected is not None else selections[n.name]],),selected=None)
                for n in program.nodes)
    return replace(program,nodes=nodes,version=program.version+1)

def _fit_constants(program, selections, inputs, targets, signals, registry, steps, lr):
    """Fit one structure's constants and return (exact-scoring model, fitted values)."""
    m=SoftProgram(_single(program,selections),registry)
    opt=torch.optim.Adam(list(m.constants.values()),lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        _,_,trace=m(inputs,return_trace=True)
        loss=m.probe_loss(trace,targets,signals)
        if not loss.requires_grad: break
        loss.backward(); opt.step()
    return m,{k:v.detach().clone() for k,v in m.constants.items()}

def hybrid_fit(program, examples, signals, registry=None, tolerance=.001,
               constant_steps=200, lr=.05, max_programs=1<<20, stop_at_first=True, screen_steps=25):
    """Enumerate the discrete structure; fit the continuous constants inside each.

    `enumerate_fit` searches the discrete choice at the declared constants and
    says so; where a constant is wrong that search finds nothing. `fit` searches
    both at once and the constant's gradient is blurred by the candidate
    mixture, which is why it converges to 3.9e-2 rather than to tolerance. This
    does neither: one structure at a time, exact where the program is discrete
    and gradient where it is continuous.

    Two phases, because paying a full fit for every structure prices the hybrid
    out of the budget that justified choosing it. Phase one fits each structure
    for `screen_steps` and records its exact error; phase two re-fits for
    `constant_steps` in that order and stops at the first structure that
    conforms. The screen is only an **ordering**, with no threshold and nothing
    discarded, so the worst case is the same search and the typical case is an
    order of magnitude cheaper.

    Each structure is scored by the **exact** error of the exported program, not
    by the relaxed loss it was fitted against, so a structure is only ever
    accepted on the measurement that counts.
    """
    r=registry or Registry()
    program.validate(r); program.validate_signals(signals)
    if not program.trainable_constants:
        return enumerate_fit(program,examples,signals,r,tolerance,max_programs,stop_at_first)
    counts=candidate_counts(program); total=math.prod(counts)
    names=[n.name for n in program.nodes]
    inputs,targets=_stacked(program,examples,signals)
    started=time.perf_counter(); evaluated=0; found=[]; best=float('inf')
    screened=[]
    for combination in itertools.product(*(range(c) for c in counts)):
        if evaluated>=max_programs: break
        selections=dict(zip(names,combination)); evaluated+=1
        try:
            m,_=_fit_constants(program,selections,inputs,targets,signals,r,screen_steps,lr)
            err=evaluate(m.export(),{},examples,signals,r,None)
        except (TypeError,ValueError,RuntimeError,OverflowError): continue
        if err is None: continue
        screened.append((err,combination,selections))
    for err0,_,selections in sorted(screened,key=lambda x:x[0]):
        try:
            m,fitted=_fit_constants(program,selections,inputs,targets,signals,r,constant_steps,lr)
            err=evaluate(m.export(),{},examples,signals,r,None)
        except (TypeError,ValueError,RuntimeError,OverflowError): continue
        if err is None: continue
        best=min(best,err)
        if err<=tolerance:
            found.append((selections,fitted,err))
            if stop_at_first: break
    from .search import SearchResult, program_cost
    exhausted=evaluated>=total
    chosen=found[0] if found else None
    bits=cost=None
    if chosen: bits,cost=program_cost(program,chosen[0],r)
    result=SearchResult(bool(found),None if not chosen else chosen[0],
                        0. if not found else min(best,tolerance),evaluated,total,exhausted,
                        (len(found)==1) if exhausted and not stop_at_first else None,
                        time.perf_counter()-started,tuple(program.trainable_constants),
                        len(found),'order',bits,cost)
    return result,(None if not chosen else chosen[1])
