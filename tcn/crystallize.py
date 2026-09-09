"""Outside-in hardening with residual optimization and transactional rollback."""
from dataclasses import dataclass
import copy
import torch

@dataclass
class FreezeEvent:
    node: str
    accepted: bool
    before: float
    after: float
    reason: str

class Crystallizer:
    def __init__(self,model,optimizer,tolerance=.01,stability_window=3,entropy_limit=.5):
        self.model=model; self.optimizer=optimizer; self.tolerance=tolerance
        self.stability_window=stability_window; self.entropy_limit=entropy_limit
        self.history=[]; self.events=[]
    def sample_stability(self):
        self.history.append(self.model.selections())
        self.history=self.history[-self.stability_window:]
    def candidates(self):
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
        ready=[]
        for n,p in zip(nodes,m.choices):
            if n.name in m.frozen: continue
            q=torch.softmax(p/m.temperatures[n.name],0); ent=float(-(q*q.clamp_min(1e-12).log()).sum().detach())
            stable=len(self.history)>=self.stability_window and len({x[n.name] for x in self.history}|{m.selections()[n.name]})==1
            if stable and ent<=self.entropy_limit: ready.append((distances[n.name],ent,n.name))
        return [name for _,_,name in sorted(ready)]
    def try_freeze(self,name,loss_fn,retrain_steps=20,conformance=None):
        m=self.model
        weights=copy.deepcopy(m.state_dict()); opt=copy.deepcopy(self.optimizer.state_dict())
        frozen=dict(m.frozen); trials=dict(m.trials); requires=[p.requires_grad for p in m.parameters()]
        before=float(loss_fn().detach()); index=m.selections()[name]
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
                loss=loss_fn()
                grads=torch.autograd.grad(loss,active,allow_unused=True) if active and loss.requires_grad else [None]*len(active)
                # This tests reachability in the autograd graph, not the presence of
                # learning signal: a discreteness or entropy regularizer keeps every
                # logit connected, so a severed interior can still pass. Treating an
                # all-zero gradient as disconnected was tried and reverted -- it also
                # flags nodes whose choice has legitimately concentrated, which
                # blocked the joint fixture from crystallizing at all (286 deferrals
                # against 32). A correct guard needs the task objective separated
                # from its regularizers; see research/FINDINGS.md section 4.
                if any(g is None for g in grads): reason="disconnected remaining region"
                # Exported conformance is a property of the whole program. While any
                # node is still soft, export() argmaxes untrained nodes too, so a
                # mismatch reports their state rather than this freeze's validity.
                # Check it only once this freeze completes the program.
                elif conformance is not None and len(m.frozen)==len(m.program.nodes) and not conformance(m.export()): reason="runtime conformance"
                else: accepted=True; reason="validated"
        except (ValueError,OverflowError,RuntimeError) as e: reason=f"invalid trial: {e}"
        if not accepted:
            m.load_state_dict(weights); self.optimizer.load_state_dict(opt); m.frozen=frozen; m.trials=trials
            for p,flag in zip(m.parameters(),requires): p.requires_grad_(flag)
        event=FreezeEvent(name,accepted,before,after,reason); self.events.append(event); return event
    def run(self,loss_fn,rounds=12,retrain_steps=20,conformance=None):
        for _ in range(rounds):
            self.sample_stability()
            for n in self.model.program.nodes:
                if n.name not in self.model.frozen:
                    self.model.temperatures[n.name]=max(.05,self.model.temperatures[n.name]*.8)
                    if n.output.kind=="int" and n.output.numeric:
                        self.model.quantization[n.name]=(n.output,min(1.,self.model.quantization.get(n.name,(n.output,0.))[1]+.1))
            for name in self.candidates():
                if name in self.candidates():self.try_freeze(name,loss_fn,retrain_steps,conformance)
            if len(self.model.frozen)==len(self.model.program.nodes): break
        return self.events
