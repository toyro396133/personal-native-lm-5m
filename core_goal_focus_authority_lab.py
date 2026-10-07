from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_v17 import load_any
from model_self_variants import SelfVariantPersonalNativeLM
from train_hebrew import load_tokenizer
from core_goal_focus_sequence_lab import (
    TRAIN_CORES, TEST_CORES, TRAIN_GOALS, TEST_GOALS, TRAIN_FOCI, TEST_FOCI,
    TRAIN_TEMPLATES, TEST_TEMPLATES, build_cache,
)
from core_goal_focus_hierarchical_lab import make_hard_sequences, RelationState


def sha_model(model):
    h=hashlib.sha256()
    with torch.no_grad():
        for n,t in model.state_dict().items():
            x=t.detach().cpu().contiguous()
            h.update(n.encode()); h.update(str(tuple(x.shape)).encode()); h.update(bytes(x.untyped_storage()))
    return h.hexdigest()


class Projector(nn.Module):
    def __init__(self,d,h=96):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Linear(h,h),nn.GELU())
    def forward(self,x):
        return self.net(F.layer_norm(x,(x.shape[-1],)))


class AuthorityHead(nn.Module):
    def __init__(self, rel_dim=64, content_dim=96):
        super().__init__()
        # persistent relation + candidate relation + event query + pairwise interaction
        self.net=nn.Sequential(
            nn.Linear(rel_dim*3+content_dim,96),nn.GELU(),
            nn.Linear(96,48),nn.GELU(),nn.Linear(48,1)
        )
    def forward(self,current,candidate,event):
        return self.net(torch.cat((current,candidate,current*candidate,event),-1))


class ScopeAuthorityController(nn.Module):
    """
    Content says WHAT is being proposed.
    Persistent relations decide WHETHER that proposal has authority in this hierarchy.
    """
    def __init__(self,d_model,mode,content_dim=96,relation_dim=64):
        super().__init__()
        self.mode=mode
        self.content=Projector(d_model,content_dim)
        self.self_core=RelationState(d_model,content_dim,relation_dim)
        self.core_goal=RelationState(d_model,content_dim,relation_dim)
        self.goal_focus=RelationState(d_model,content_dim,relation_dim)

        # Proposal path: deliberately has no relation-state inputs.
        self.proposal=nn.Sequential(
            nn.Linear(content_dim*4,128),nn.GELU(),
            nn.Linear(128,64),nn.GELU(),nn.Linear(64,3)
        )

        # Authority path: cannot be bypassed in relational mode.
        self.auth_core=AuthorityHead(relation_dim,content_dim)
        self.auth_goal=AuthorityHead(relation_dim,content_dim)
        self.auth_focus=AuthorityHead(relation_dim,content_dim)

    def init_relations(self,root,core,goal,focus):
        return (
            self.self_core.initialize(root,core),
            self.core_goal.initialize(core,goal),
            self.goal_focus.initialize(goal,focus),
        )

    def relation_candidates(self,root,core,goal,cand_core,cand_goal,cand_focus):
        # Candidate relation objects are evaluated in the hierarchy they would occupy.
        cr_sc=self.self_core.initialize(root,cand_core)
        goal_parent=cand_core
        cr_cg=self.core_goal.initialize(goal_parent,cand_goal)
        focus_parent=cand_goal
        cr_gf=self.goal_focus.initialize(focus_parent,cand_focus)
        return cr_sc,cr_cg,cr_gf

    def forward(
        self,root,core,goal,focus,event,
        cand_core,cand_goal,cand_focus,
        relations, relation_override=None
    ):
        c=self.content(core); g=self.content(goal); f=self.content(focus); e=self.content(event)
        proposal_logits=self.proposal(torch.cat((c,g,f,e),-1))
        proposal=torch.sigmoid(proposal_logits)

        r_sc,r_cg,r_gf = relation_override if relation_override is not None else relations
        cr_sc,cr_cg,cr_gf=self.relation_candidates(
            root,core,goal,cand_core,cand_goal,cand_focus
        )
        authority_logits=torch.cat((
            self.auth_core(r_sc,cr_sc,e),
            self.auth_goal(r_cg,cr_cg,e),
            self.auth_focus(r_gf,cr_gf,e),
        ),-1)
        authority=torch.sigmoid(authority_logits)

        if self.mode=="direct":
            final=proposal
        else:
            # Structural requirement: content alone can never authorize a state mutation.
            final=proposal*authority

        # A higher level accepted change resets descendants.
        p_core=final[:,0:1]
        p_goal=1-(1-p_core)*(1-final[:,1:2])
        p_focus=1-(1-p_goal)*(1-final[:,2:3])
        final=torch.cat((p_core,p_goal,p_focus),-1).clamp(1e-5,1-1e-5)
        return proposal.clamp(1e-5,1-1e-5),authority.clamp(1e-5,1-1e-5),final

    def update_relations(self,relations,root,core,goal,focus,event,next_core,next_goal,next_focus,bits):
        r_sc,r_cg,r_gf=relations
        e=self.content(event)
        nc=self.content(next_core); ng=self.content(next_goal); nf=self.content(next_focus)
        prop_sc=self.self_core.propose_update(r_sc,self.content(root),nc,e)
        prop_cg=self.core_goal.propose_update(r_cg,nc,ng,e)
        prop_gf=self.goal_focus.propose_update(r_gf,ng,nf,e)
        bc,bg,bf=bits[:,0:1],bits[:,1:2],bits[:,2:3]
        return (
            r_sc*(1-bc)+prop_sc*bc,
            r_cg*(1-bg)+prop_cg*bg,
            r_gf*(1-bf)+prop_gf*bf,
        )


def proposal_target(kind,target,device):
    # Pressure events genuinely propose a high-level change but lack authority.
    if kind=="goal_pressure":
        x=[0.,1.,1.]
    elif kind=="core_pressure":
        x=[1.,1.,1.]
    else:
        x=list(target)
    return torch.tensor([x],dtype=torch.float32,device=device)


def authority_target(kind,target,device):
    # Only an authorized event may mutate hierarchy.
    return torch.tensor([target],dtype=torch.float32,device=device)


def vec(cache,text,device):
    return cache[text].unsqueeze(0).to(device)


def train_one(ctl,sequences,cache,root,device,epochs,lr,seed):
    opt=torch.optim.AdamW(ctl.parameters(),lr=lr,weight_decay=0.01)
    rng=random.Random(seed)
    for epoch in range(epochs):
        order=list(range(len(sequences))); rng.shuffle(order)
        ctl.train()
        for pos,idx in enumerate(order):
            seq=sequences[idx]
            core,goal,focus=seq["initial"]
            cv,gv,fv=vec(cache,core,device),vec(cache,goal,device),vec(cache,focus,device)
            rel=ctl.init_relations(root.unsqueeze(0),cv,gv,fv)
            losses=[]

            # A deterministic other hierarchy is used for counterfactual authority training.
            other=sequences[order[(pos+1)%len(order)]]
            oc,og,of=other["initial"]
            orel=ctl.init_relations(
                root.unsqueeze(0),vec(cache,oc,device),vec(cache,og,device),vec(cache,of,device)
            )

            for e in seq["steps"]:
                ev=vec(cache,e["event"],device)
                ccv=vec(cache,e["candidate_core"],device)
                cgv=vec(cache,e["candidate_goal"],device)
                cfv=vec(cache,e["candidate_focus"],device)

                proposal,auth,final=ctl(
                    root.unsqueeze(0),cv,gv,fv,ev,ccv,cgv,cfv,rel
                )
                y_final=torch.tensor([e["target"]],dtype=torch.float32,device=device)
                y_prop=proposal_target(e["kind"],e["target"],device)
                y_auth=authority_target(e["kind"],e["target"],device)

                # Direct arm is optimized for the real task. Relational arm receives
                # decomposition supervision plus a hard requirement on authority.
                if ctl.mode=="direct":
                    loss=F.binary_cross_entropy(final,y_final)
                else:
                    loss=(
                        0.8*F.binary_cross_entropy(proposal,y_prop)
                        +1.2*F.binary_cross_entropy(auth,y_auth)
                        +1.5*F.binary_cross_entropy(final,y_final)
                    )

                    # Same event + wrong hierarchy must not have authority to mutate it.
                    _,bad_auth,_=ctl(
                        root.unsqueeze(0),cv,gv,fv,ev,ccv,cgv,cfv,rel,
                        relation_override=orel
                    )
                    # Keep focus-level local work possible only when it was truly local;
                    # high-level authority is always denied under a mismatched hierarchy.
                    bad_target=torch.zeros_like(bad_auth)
                    loss=loss+0.6*F.binary_cross_entropy(bad_auth,bad_target)

                losses.append(loss)

                ncv=vec(cache,e["next_core"],device)
                ngv=vec(cache,e["next_goal"],device)
                nfv=vec(cache,e["next_focus"],device)
                rel=ctl.update_relations(
                    rel,root.unsqueeze(0),cv,gv,fv,ev,ncv,ngv,nfv,y_final
                )
                cv,gv,fv=ncv,ngv,nfv

            loss=torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(ctl.parameters(),1.0); opt.step()
    return ctl


@torch.no_grad()
def evaluate(ctl,sequences,cache,root,device,ablation="normal"):
    ctl.eval()
    total=state_exact=final_exact=whole_exact=0
    steps_first=[]
    false=[0,0,0]; false_den=[0,0,0]
    true=[0,0,0]; true_den=[0,0,0]
    slot=[0,0,0]
    pressure_goal_ok=pressure_goal_n=0
    pressure_core_ok=pressure_core_n=0
    after_focus_ok=after_focus_n=0

    for si,seq in enumerate(sequences):
        pc,pg,pf=seq["initial"]; tc,tg,tf=seq["initial"]
        cv,gv,fv=vec(cache,pc,device),vec(cache,pg,device),vec(cache,pf,device)
        rel=ctl.init_relations(root.unsqueeze(0),cv,gv,fv)

        if ablation=="zero_rel":
            rel=tuple(torch.zeros_like(x) for x in rel)
        elif ablation=="shuffle_rel":
            other=sequences[(si+37)%len(sequences)]
            oc,og,of=other["initial"]
            rel=ctl.init_relations(
                root.unsqueeze(0),vec(cache,oc,device),vec(cache,og,device),vec(cache,of,device)
            )
        elif ablation=="drop_sc":
            rel=(torch.zeros_like(rel[0]),rel[1],rel[2])
        elif ablation=="drop_cg":
            rel=(rel[0],torch.zeros_like(rel[1]),rel[2])
        elif ablation=="drop_gf":
            rel=(rel[0],rel[1],torch.zeros_like(rel[2]))

        seq_ok=True; first=None; focus_changes=0
        for t,e in enumerate(seq["steps"]):
            ev=vec(cache,e["event"],device)
            ccv=vec(cache,e["candidate_core"],device)
            cgv=vec(cache,e["candidate_goal"],device)
            cfv=vec(cache,e["candidate_focus"],device)
            _,_,probs=ctl(
                root.unsqueeze(0),cv,gv,fv,ev,ccv,cgv,cfv,rel
            )
            bits=(probs[0]>=0.5).float()
            truth=torch.tensor(e["target"],dtype=torch.float32,device=device)
            for s in range(3):
                if truth[s]>0.5:
                    true_den[s]+=1; true[s]+=int(bits[s].item())
                else:
                    false_den[s]+=1; false[s]+=int(bits[s].item())

            npc=e["candidate_core"] if bits[0] else pc
            npg=e["candidate_goal"] if bits[1] else pg
            npf=e["candidate_focus"] if bits[2] else pf
            ncv,ngv,nfv=vec(cache,npc,device),vec(cache,npg,device),vec(cache,npf,device)
            rel=ctl.update_relations(
                rel,root.unsqueeze(0),cv,gv,fv,ev,ncv,ngv,nfv,bits.unsqueeze(0)
            )
            pc,pg,pf=npc,npg,npf; cv,gv,fv=ncv,ngv,nfv
            tc,tg,tf=e["next_core"],e["next_goal"],e["next_focus"]

            total+=1
            ok=[pc==tc,pg==tg,pf==tf]
            state_exact+=int(all(ok))
            for s,x in enumerate(ok): slot[s]+=int(x)
            if not all(ok):
                seq_ok=False
                if first is None: first=t+1

            if e["kind"] in {"focus","tempting_focus"}: focus_changes+=1
            if e["kind"] in {"goal_change","goal_done","core_change"}: focus_changes=0
            if e["kind"]=="goal_pressure":
                pressure_goal_n+=1; pressure_goal_ok+=int(pg==tg)
            if e["kind"]=="core_pressure":
                pressure_core_n+=1; pressure_core_ok+=int(pc==tc)
            if focus_changes>=3:
                after_focus_n+=1; after_focus_ok+=int(pg==tg)

        final_exact+=int((pc,pg,pf)==(tc,tg,tf))
        whole_exact+=int(seq_ok)
        steps_first.append(first if first is not None else len(seq["steps"])+1)

    d=lambda a,b:a/b if b else None
    return {
        "state_exact_per_step":d(state_exact,total),
        "core_state_accuracy":d(slot[0],total),
        "goal_state_accuracy":d(slot[1],total),
        "focus_state_accuracy":d(slot[2],total),
        "final_state_exact":d(final_exact,len(sequences)),
        "whole_sequence_exact":d(whole_exact,len(sequences)),
        "mean_steps_until_first_error":statistics.mean(steps_first),
        "core_false_update_rate":d(false[0],false_den[0]),
        "goal_false_update_rate":d(false[1],false_den[1]),
        "focus_false_update_rate":d(false[2],false_den[2]),
        "core_change_recall":d(true[0],true_den[0]),
        "goal_change_recall":d(true[1],true_den[1]),
        "focus_change_recall":d(true[2],true_den[2]),
        "goal_pressure_retention":d(pressure_goal_ok,pressure_goal_n),
        "core_pressure_retention":d(pressure_core_ok,pressure_core_n),
        "goal_retention_after_3plus_focus_changes":d(after_focus_ok,after_focus_n),
    }


def aggregate(runs):
    keys=list(runs[0]["normal"].keys())
    out={}
    for k in keys:
        vals=[r["normal"][k] for r in runs if r["normal"][k] is not None]
        out[k]={"mean":statistics.mean(vals),"stdev":statistics.stdev(vals) if len(vals)>1 else 0.0}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--mode",choices=["direct","relational"],required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--train-sequences",type=int,default=320)
    ap.add_argument("--test-sequences",type=int,default=180)
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--lr",type=float,default=0.0012)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=ap.parse_args()

    ckpt,cfg,model=load_any(a.checkpoint,a.device)
    tok,_=load_tokenizer(a.tokenizer)
    before=sha_model(model)
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()

    train=make_hard_sequences(a.train_sequences,18,30,7101,TRAIN_CORES,TRAIN_GOALS,TRAIN_FOCI,TRAIN_TEMPLATES)
    test=make_hard_sequences(a.test_sequences,36,52,11103,TEST_CORES,TEST_GOALS,TEST_FOCI,TEST_TEMPLATES)
    cache=build_cache(model,tok,train+test,a.device)

    if isinstance(model,SelfVariantPersonalNativeLM):
        root=model.self_anchor.detach().clone().to(a.device)
    else:
        root=torch.zeros(cfg.d_model,device=a.device)
        if a.mode=="relational": raise SystemExit("relational mode requires learned SELF")

    runs=[]
    for seed in (151,302,453):
        print(f"[train] variant={ckpt.get('variant','baseline')} mode={a.mode} seed={seed}",flush=True)
        torch.manual_seed(seed)
        ctl=ScopeAuthorityController(cfg.d_model,a.mode).to(a.device)
        train_one(ctl,train,cache,root,a.device,a.epochs,a.lr,seed)
        r={"seed":seed,"normal":evaluate(ctl,test,cache,root,a.device)}
        if a.mode=="relational":
            for ab in ("zero_rel","shuffle_rel","drop_sc","drop_cg","drop_gf"):
                r[ab]=evaluate(ctl,test,cache,root,a.device,ablation=ab)
        runs.append(r)

    after=sha_model(model)
    if before!=after: raise SystemExit("frozen 15M model changed")

    result={
        "experiment":"v0.16f relation authority gate lab",
        "source_variant":ckpt.get("variant","baseline"),
        "source_tokens_seen":ckpt.get("tokens_seen"),
        "mode":a.mode,
        "frozen_model_unchanged":before==after,
        "design":{
            "content_role":"propose change scope",
            "relation_role":"authorize mutation in current hierarchy",
            "authority_is_multiplicative_and_required":a.mode=="relational",
            "counterfactual_wrong_hierarchy_training":a.mode=="relational",
            "persistent_relations":["SELF->CORE","CORE->GOAL","GOAL->FOCUS"],
            "closed_loop":True,
            "errors_persist":True,
        },
        "aggregate":aggregate(runs),
        "runs":runs,
    }
    if a.mode=="relational":
        base=result["aggregate"]["state_exact_per_step"]["mean"]
        causal={}
        for ab in ("zero_rel","shuffle_rel","drop_sc","drop_cg","drop_gf"):
            vals=[r[ab]["state_exact_per_step"] for r in runs]
            m=statistics.mean(vals)
            causal[ab]={"mean":m,"delta_vs_normal":m-base}
        result["causal"]=causal

    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
