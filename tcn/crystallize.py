"""Outside-in hardening with residual optimization and transactional rollback."""
from dataclasses import dataclass
from typing import Callable
import copy
import torch

@dataclass
class FreezeEvent:
    node: str
    accepted: bool
    before: float
    after: float
    reason: str

@dataclass(frozen=True)
class Objective:
    """What the scheduler optimizes, and what it probes for learning signal.

    `total` is the regularized training objective: residual retraining descends
    it, and the degradation tolerance is measured against it. `task` is the same
    objective with the architecture regularizers removed -- the discreteness /
    entropy term and any description-size term, which attach to the choice
    logits directly instead of through the graph.

    The separation is load-bearing for the connectivity guard. A discreteness
    regularizer keeps every unfrozen logit reachable in the autograd graph, so
    probing `total` measures reachability and passes a severed interior. Probing
    `task` asks the question ARCHITECTURE.md section 5 asks: does any path from
    the task objective still reach the remaining trainable region?
    """
    total: Callable[[],torch.Tensor]
    task: Callable[[],torch.Tensor]
    @classmethod
    def of(cls,loss):
        # A bare callable is accepted so a caller with no regularizer need not
        # wrap it; that caller is then asserting the objective carries none, and
        # the connectivity guard is only as strong as the assertion.
        return loss if isinstance(loss,cls) else cls(loss,loss)

class Crystallizer:
    """Progressive crystallization: measure, harden, retrain, accept or roll back.

    Selection is perturbation-based. Each unfrozen node is scored by how much the
    objective degrades when a candidate is removed from its mixture; the frozen
    candidate is the one whose removal hurts most, and nodes are ordered by how
    decisive that measurement is. This is the DARTS-PT rule (Wang et al., ICLR
    2021, "Rethinking Architecture Selection in Differentiable NAS"), which
    showed that the magnitude of a learned architecture distribution does not
    indicate an operation's contribution. Softmax entropy plus argmax stability
    -- what this scheduler used to select on -- is exactly that falsified
    quantity. `selection="entropy"` restores the older rule for ablation only.

    Commitment timing is separately controlled. `eligibility="plateau"` makes a
    node eligible to freeze only once the unregularized task loss has stopped
    improving, spending every other round descending the objective instead, and
    accepts at most one freeze per opening so the next commitment waits for the
    residual graph to settle around the last one. `anneal="plateau"` puts the
    temperature/quantization schedule on the same signal instead of the fixed
    0.8x-per-round clock. Both default to the fixed clock. Per ARCHITECTURE.md
    section 5 the window, threshold and patience are experiment configuration.
    `anneal="never"` holds the schedule still for the whole run; it exists to
    separate "concentrate later" from "do not concentrate" and is an ablation.

    Residual retraining, the degradation tolerance, the connectivity guard, the
    conformance gate and transactional rollback are unchanged by the selection
    rule and apply to every trial.
    """
    def __init__(self,model,optimizer,tolerance=.01,stability_window=3,entropy_limit=.5,selection="perturbation",
                 eligibility="immediate",anneal="round",plateau_window=3,plateau_tolerance=1e-3,plateau_patience=12,
                 close_block=True):
        if selection not in ("perturbation","entropy"):raise ValueError("unknown selection rule "+selection)
        if eligibility not in ("immediate","plateau"):raise ValueError("unknown eligibility rule "+eligibility)
        if anneal not in ("round","plateau","never"):raise ValueError("unknown anneal rule "+anneal)
        if plateau_window<1:raise ValueError("plateau window must be positive")
        self.model=model; self.optimizer=optimizer; self.tolerance=tolerance
        self.stability_window=stability_window; self.entropy_limit=entropy_limit
        self.selection=selection; self.objective=None
        self.eligibility=eligibility; self.anneal=anneal
        self.plateau_window=plateau_window; self.plateau_tolerance=plateau_tolerance
        self.plateau_patience=plateau_patience
        self.close_block=close_block; self.block_events=[]
        self.history=[]; self.events=[]; self.selected={}; self.sweep_evaluations=0
        self.progress=[]; self.gate_evaluations=0; self.rounds_waited=0; self.gate_log=[]
    def sample_stability(self):
        self.history.append(self.model.selections())
        self.history=self.history[-self.stability_window:]
    def distances(self):
        """Hops to the nearest input or output boundary, for outside-in ordering."""
        m=self.model; nodes=m.program.nodes; outs={v for _,v in m.program.outputs}
        distances={n.name:0 if n.name in outs else 10**9 for n in nodes}
        input_names=set(dict(m.program.inputs))|{k for k,_,_ in m.program.state}
        for n in nodes:
            parents={s for c in n.candidates for s in c.sources}
            if parents & input_names: distances[n.name]=0
        for _ in nodes:
            for n in nodes:
                for c in n.candidates:
                    for s in c.sources:
                        if s in distances:
                            d=min(distances[n.name],distances[s])+1
                            distances[n.name]=min(distances[n.name],d); distances[s]=min(distances[s],d)
        return distances
    def entropy_candidates(self):
        """The superseded rule: readiness by softmax entropy and selection stability."""
        m=self.model; distances=self.distances(); ready=[]
        for n,p in zip(m.program.nodes,m.choices):
            if n.name in m.frozen: continue
            q=torch.softmax(p/m.temperatures[n.name],0); ent=float(-(q*q.clamp_min(1e-12).log()).sum().detach())
            stable=len(self.history)>=self.stability_window and len({x[n.name] for x in self.history}|{m.selections()[n.name]})==1
            if stable and ent<=self.entropy_limit: ready.append((distances[n.name],ent,n.name))
        return [name for _,_,name in sorted(ready)]
    def measure(self,loss):
        """One objective evaluation, counted; an invalid relaxation reads as infinite."""
        try:
            with torch.no_grad(): value=float(loss().detach())
        except (ValueError,OverflowError,RuntimeError): return float("inf")
        self.sweep_evaluations+=1
        return value if value==value else float("inf")
    def perturbation_scores(self,index,node,loss):
        """Objective under removal of each candidate from this node's mixture."""
        p=self.model.choices[index]; saved=p.detach().clone(); scores=[]
        for i in range(len(node.candidates)):
            with torch.no_grad(): p[i]=-1e9
            scores.append(self.measure(loss))
            with torch.no_grad(): p.copy_(saved)
        # A removal that leaves the relaxation invalid -- an empty numeric domain,
        # a nonfinite value -- says nothing about that candidate's contribution.
        # Score it as uninformative rather than as maximally important.
        return [s if s!=float("inf") else float("-inf") for s in scores]
    def score_node(self,index,node,loss):
        """(decisiveness, candidate) for one node, or None when nothing is measurable."""
        # A single-candidate node carries no architecture decision. It is ordered
        # last: hardening plumbing early detaches gradient paths before any real
        # choice has been measured.
        if len(node.candidates)==1: return (float("-inf"),0)
        scores=self.perturbation_scores(index,node,loss)
        order=sorted(range(len(scores)),key=lambda k:-scores[k])
        if scores[order[0]]==float("-inf"): return None
        return (scores[order[0]]-scores[order[1]],order[0])
    def probe_progress(self,task):
        """One unregularized task-loss reading for the plateau gate, counted apart."""
        try:
            with torch.no_grad(): value=float(task().detach())
        except (ValueError,OverflowError,RuntimeError): value=float("inf")
        self.gate_evaluations+=1
        self.progress.append(value if value==value else float("inf"))
        return self.progress[-1]
    def plateaued(self):
        """True when the task loss has stopped improving over the gate window.

        Relative improvement of the best of the last `plateau_window` readings
        against the reading that preceded them; below `plateau_tolerance` the
        objective counts as stopped. A worsening objective is also stopped: the
        gate asks whether more training is still buying anything, not whether the
        loss moved. The window is re-armed whenever the gate opens, so each
        commitment is measured against progress made since the previous one.
        """
        w=self.plateau_window; h=self.progress
        if len(h)<w+1: return False
        prior=h[-(w+1)]; recent=min(h[-w:])
        if prior!=prior or prior==float("inf"): return False
        return (prior-recent)/max(abs(prior),1e-12)<self.plateau_tolerance
    def train_residual(self,total,steps):
        """Descend the objective on a closed round, so the gate has progress to read."""
        for _ in range(steps):
            self.optimizer.zero_grad(); loss=total()
            if loss.requires_grad: loss.backward(); self.optimizer.step()
    def candidates(self,loss=None):
        """Nodes ready to freeze, most decisive first, recording the chosen candidate."""
        m=self.model; loss=loss if loss is not None else self.objective
        if self.selection=="entropy" or loss is None: return self.entropy_candidates()
        total=Objective.of(loss).total; scored=[]
        for index,n in enumerate(m.program.nodes):
            if n.name in m.frozen: continue
            score=self.score_node(index,n,total)
            if score is None: continue
            self.selected[n.name]=score[1]; scored.append((score[0],n.name))
        return [name for _,name in sorted(scored,key=lambda x:-x[0])]
    def try_freeze(self,name,loss_fn,retrain_steps=20,conformance=None,index=None):
        m=self.model; objective=Objective.of(loss_fn); loss_fn=objective.total
        weights=copy.deepcopy(m.state_dict()); opt=copy.deepcopy(self.optimizer.state_dict())
        frozen=dict(m.frozen); trials=dict(m.trials); pinned=dict(m.pinned); requires=[p.requires_grad for p in m.parameters()]
        before=float(loss_fn().detach())
        if index is None: index=m.selections()[name]
        accepted=False; reason="degradation"; after=float("inf")
        try:
            m.trials[name]=index
            for _ in range(retrain_steps):
                self.optimizer.zero_grad(); loss=loss_fn()
                if loss.requires_grad: loss.backward(); self.optimizer.step()
            m.freeze(name,index)
            after=float(loss_fn().detach())
            if not torch.isfinite(torch.tensor(after)): reason="nonfinite loss"
            elif after>before+self.tolerance: reason="degradation"
            else:
                active=[p for p in m.parameters() if p.requires_grad]
                # The viability probe uses the UNREGULARIZED task objective. A
                # discreteness or entropy term attaches every unfrozen logit to
                # the loss directly, so probing the regularized total tests
                # reachability in the autograd graph and lets a severed interior
                # pass. Against the task objective a severed logit has no path at
                # all and `allow_unused` returns None, while a merely concentrated
                # logit still has a path and returns a (possibly exactly zero)
                # tensor. That is why treating an all-zero gradient as
                # disconnected is the wrong strengthening: it cannot tell the two
                # apart, and was measured to block crystallization outright.
                task=objective.task()
                grads=torch.autograd.grad(task,active,allow_unused=True) if active and task.requires_grad else [None]*len(active)
                if any(g is None for g in grads): reason="disconnected remaining region"
                # Exported conformance is a property of the whole program. While any
                # node is still soft, export() argmaxes untrained nodes too, so a
                # mismatch reports their state rather than this freeze's validity.
                # Check it only once this freeze completes the program.
                elif conformance is not None and len(m.frozen)==len(m.program.nodes) and not conformance(m.export()): reason="runtime conformance"
                else: accepted=True; reason="validated"
        except (ValueError,OverflowError,RuntimeError) as e: reason=f"invalid trial: {e}"
        if not accepted:
            m.load_state_dict(weights); self.optimizer.load_state_dict(opt); m.frozen=frozen; m.trials=trials; m.pinned=pinned
            for p,flag in zip(m.parameters(),requires): p.requires_grad_(flag)
        event=FreezeEvent(name,accepted,before,after,reason); self.events.append(event); return event
    def try_freeze_block(self,names,loss_fn,retrain_steps=20,conformance=None):
        """One transactional trial that hardens a whole block of nodes at once.

        The per-node viability guard rejects a freeze whose remaining trainable
        region is severed from the task objective. That guard is correct per node
        and incomplete over sets: a residual set can reach a state in which
        freezing any single member disconnects the others, so every single-node
        trial is refused forever and the run cannot close. Measured on `joint` at
        40 episodes with the shipped crystallizer and no seasons machinery, seed 7
        of 8 ends 9/13 frozen with a residual `{goal_relation, prediction,
        relation, z}` refused 29 times for exactly that reason.

        Hardening the block together is not a weakening of the guard. When the
        block completes the program there is no remaining trainable region left to
        disconnect, which is the condition the guard tests for. Degradation,
        conformance and rollback are unchanged and apply to the block as a whole,
        so a block that degrades is rolled back exactly as a node would be.

        Section 5.1 of ARCHITECTURE already speaks of hardening "a candidate
        node/block"; this is the block case.
        """
        m=self.model; objective=Objective.of(loss_fn); loss_fn=objective.total
        names=[n for n in names if n not in m.frozen]
        if not names: return None
        weights=copy.deepcopy(m.state_dict()); opt=copy.deepcopy(self.optimizer.state_dict())
        frozen=dict(m.frozen); trials=dict(m.trials); pinned=dict(m.pinned); requires=[p.requires_grad for p in m.parameters()]
        before=float(loss_fn().detach())
        accepted=False; reason="degradation"; after=float("inf")
        try:
            selections=m.selections()
            for name in names: m.trials[name]=self.selected.get(name,selections[name])
            for _ in range(retrain_steps):
                self.optimizer.zero_grad(); loss=loss_fn()
                if loss.requires_grad: loss.backward(); self.optimizer.step()
            for name in names: m.freeze(name,self.selected.get(name,selections[name]))
            after=float(loss_fn().detach())
            if not torch.isfinite(torch.tensor(after)): reason="nonfinite loss"
            elif after>before+self.tolerance: reason="degradation"
            else:
                active=[p for p in m.parameters() if p.requires_grad]
                task=objective.task()
                grads=torch.autograd.grad(task,active,allow_unused=True) if active and task.requires_grad else [None]*len(active)
                if any(g is None for g in grads): reason="disconnected remaining region"
                elif conformance is not None and len(m.frozen)==len(m.program.nodes) and not conformance(m.export()): reason="runtime conformance"
                else: accepted=True; reason="validated"
        except (ValueError,OverflowError,RuntimeError) as e: reason=f"invalid trial: {e}"
        if not accepted:
            m.load_state_dict(weights); self.optimizer.load_state_dict(opt); m.frozen=frozen; m.trials=trials; m.pinned=pinned
            for p,flag in zip(m.parameters(),requires): p.requires_grad_(flag)
        event=FreezeEvent("+".join(names),accepted,before,after,"block "+reason)
        self.events.append(event); self.block_events.append(event); return event
    def run(self,loss_fn,rounds=12,retrain_steps=20,conformance=None):
        objective=Objective.of(loss_fn); self.objective=objective
        index_of={n.name:i for i,n in enumerate(self.model.program.nodes)}
        gated=self.eligibility=="plateau" or self.anneal=="plateau"
        last_round=len(self.events)
        for _ in range(rounds):
            last_round=len(self.events)
            self.sample_stability()
            # The gate reads the unregularized task loss once per round. Both the
            # eligibility rule and the anneal schedule can consume it; whichever
            # of them is left on the fixed clock keeps the shipped behaviour.
            open_gate=True
            if gated:
                self.probe_progress(objective.task)
                open_gate=self.plateaued() or self.rounds_waited>=self.plateau_patience
                self.gate_log.append({"loss":self.progress[-1],"open":open_gate,"waited":self.rounds_waited,
                                      "frozen":len(self.model.frozen)})
                if open_gate: self.rounds_waited=0; self.progress=self.progress[-1:]
                else: self.rounds_waited+=1
            if self.anneal=="round" or (self.anneal=="plateau" and open_gate):
                for n in self.model.program.nodes:
                    if n.name not in self.model.frozen:
                        # Choice concentration and surrogate sharpening are
                        # separate controls, but the relaxation reads
                        # `temperatures * surrogate_scale`, so this one schedule
                        # still sharpens both proportionally -- exactly the
                        # shipped behaviour. A caller that wants a wide surrogate
                        # under a concentrating choice raises
                        # `model.surrogate_scale` and leaves this alone.
                        self.model.temperatures[n.name]=max(.05,self.model.temperatures[n.name]*.8)
                        if n.output.kind=="int" and n.output.numeric:
                            self.model.quantization[n.name]=(n.output,min(1.,self.model.quantization.get(n.name,(n.output,0.))[1]+.1))
            if self.eligibility=="plateau" and not open_gate:
                # Still improving: spend the round descending the objective rather
                # than on an irreversible commitment measured mid-descent.
                self.train_residual(objective.total,retrain_steps); continue
            stale=False
            for name in self.candidates(objective):
                if self.selection=="entropy":
                    if name in self.candidates():self.try_freeze(name,objective,retrain_steps,conformance)
                    continue
                if name in self.model.frozen: continue
                # A committed freeze changes the graph the scores were measured on.
                # A rolled-back trial restores it exactly, so only re-measure after
                # an acceptance rather than sweeping every node again.
                if stale:
                    score=self.score_node(index_of[name],self.model.program.nodes[index_of[name]],objective.total)
                    if score is None: continue
                    self.selected[name]=score[1]
                accepted=self.try_freeze(name,objective,retrain_steps,conformance,self.selected.get(name)).accepted
                stale=accepted or stale
                # Under the plateau gate one commitment per opening: the next freeze
                # waits for the residual graph to stop improving around this one.
                if accepted and self.eligibility=="plateau": break
            if len(self.model.frozen)==len(self.model.program.nodes): break
        # The rounds are spent and nodes are still soft. Fire the block trial only
        # on evidence of the pathology it fixes: the final round refused a freeze
        # for "disconnected remaining region", which is the guard reporting a
        # residual set whose members sever each other. A run that simply ran out
        # of rounds, or one whose eligibility gate deliberately deferred every
        # commitment, produces no such refusal and gets no trial -- the fix must
        # not become a way to force a commitment the schedule declined to make.
        stranded=any(e.reason=="disconnected remaining region" for e in self.events[last_round:])
        if self.close_block and stranded and len(self.model.frozen)<len(self.model.program.nodes):
            self.try_freeze_block([n.name for n in self.model.program.nodes if n.name not in self.model.frozen],
                                  objective,retrain_steps,conformance)
        return self.events
