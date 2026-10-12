"""V12 engineering invariants on a TINY FICTIONAL torch decoder.

It demonstrates LIVE internal neural conditioning and actual latent-only
gradient updates, not real SmolLM2 or verified human intent performance.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from train_queue import train_queue, validate_queue
from detachable_layer import (
    FrozenResidualBridge,PersonalProjectState,freeze_backbone,
    active_compass,pair_margins,personalized_update,inspect_live,
    check_source
)

SRC="git:"+("c"*40)+":docs/fake-test.md#L1-L4"
torch.set_num_threads(1)

class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(12,12,bias=False)
        self.act=nn.Tanh()
    def forward(self,hidden):
        return hidden + self.act(self.proj(hidden))*0.4

class TinyFrozenLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(18,12)
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([FakeDecoder(),FakeDecoder()])
        self.norm=nn.LayerNorm(12)
        self.lm_head=nn.Linear(12,18,bias=False)
    def forward(self,input_ids):
        h=self.embed(input_ids)
        for block in self.model.layers:
            h=block(h)
        return SimpleNamespace(logits=self.lm_head(self.norm(h)))

def params_snapshot(model):
    return {k:x.detach().clone() for k,x in model.state_dict().items()}

def same(a,b):
    return all(torch.equal(a[k],b[k]) for k in a)

def fake_feedback(n=1,*,verified=True,user=True):
    return {"event_id":f"reply.{n}","source":SRC,"speaker":"user" if user else "assistant",
            "confirmed_by_user":user,"source_verified":verified}

class V12Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model=TinyFrozenLM()
        self.bridge=FrozenResidualBridge(12,rank=4,gain=4.)
        self.bank=PersonalProjectState(rank=4)
        self.candidates=[torch.tensor([[1,2,3,4]],dtype=torch.long),
                         torch.tensor([[4,3,2,1]],dtype=torch.long)]
        self.model_pre=params_snapshot(self.model)
        self.bridge_pre=params_snapshot(self.bridge)

    def update(self,idx=1,project="project-alpha",user_id="user-alpha",**kwargs):
        return personalized_update(
            self.model,self.bridge,self.bank,user_id=user_id,project_id=project,
            scope="project",candidate_ids=self.candidates,chosen_index=idx,
            yes_token_id=7,no_token_id=8,feedback=fake_feedback(kwargs.get("n",1)),
            steps=12,lr=0.9,retention=0.001)

    def test_direct_hook_alters_language_model_logits(self):
        freeze_backbone(self.model)
        u,p=self.bank.latent("user-alpha","project-alpha")
        with torch.no_grad():
            raw=self.model(input_ids=self.candidates[0]).logits.detach()
            with active_compass(self.model,self.bridge,u,p):
                changed=self.model(input_ids=self.candidates[0]).logits.detach()
            after=self.model(input_ids=self.candidates[0]).logits.detach()
        self.assertFalse(torch.equal(raw,changed))
        self.assertTrue(torch.equal(raw,after))
        self.assertTrue(same(self.model_pre,params_snapshot(self.model)))

    def test_real_backprop_updates_only_project_latent(self):
        before=self.bank.latent("user-alpha","project-alpha")
        result=self.update()
        after=self.bank.latent("user-alpha","project-alpha")
        self.assertGreater(result["state_delta_l2"],1e-8)
        self.assertFalse(torch.equal(before[1],after[1]))
        self.assertTrue(torch.equal(before[0],after[0]))
        self.assertTrue(same(self.model_pre,params_snapshot(self.model)))
        self.assertTrue(same(self.bridge_pre,params_snapshot(self.bridge)))
        self.assertFalse(result["project_policy_approved"])
        self.assertFalse(result["bridge_updated"])
        self.assertFalse(result["backbone_updated"])

    def test_preference_margin_improves_under_supervised_feedback(self):
        u,p=self.bank.latent("user-alpha","project-alpha")
        with torch.no_grad():
            initial=pair_margins(self.model,self.bridge,u,p,self.candidates,7,8)
        result=self.update()
        u,p=self.bank.latent("user-alpha","project-alpha")
        with torch.no_grad():
            updated=pair_margins(self.model,self.bridge,u,p,self.candidates,7,8)
        margin=lambda v:float(v[1]-v[0])
        self.assertGreater(margin(updated),margin(initial))
        self.assertGreater(result["state_delta_l2"],0)

    def test_other_project_and_user_remain_unchanged(self):
        before_alpha=self._read("user-alpha","project-alpha")
        other_project_before=self._read("user-alpha","project-beta")
        other_user_before=self._read("user-beta","project-alpha")
        self.update()
        self.assertFalse(torch.allclose(before_alpha,self._read("user-alpha","project-alpha")))
        self.assertTrue(torch.equal(other_project_before,self._read("user-alpha","project-beta")))
        self.assertTrue(torch.equal(other_user_before,self._read("user-beta","project-alpha")))

    def _read(self,user,project):
        u,p=self.bank.latent(user,project)
        with torch.no_grad():
            return pair_margins(self.model,self.bridge,u,p,self.candidates,7,8).detach().clone()

    def test_global_user_feedback_separate_from_project_state(self):
        before=self._read("user-alpha","project-alpha")
        result=personalized_update(
            self.model,self.bridge,self.bank,user_id="user-alpha",project_id=None,
            scope="personal",candidate_ids=self.candidates,chosen_index=1,
            yes_token_id=7,no_token_id=8,feedback=fake_feedback(2),
            steps=12,lr=0.9)
        self.assertEqual(result["scope"],"personal")
        self.assertTrue(torch.equal(self.bank.latent("user-alpha","project-alpha")[1],torch.zeros(4)))
        self.assertFalse(torch.equal(before,self._read("user-alpha","project-alpha")))
        self.assertTrue(torch.equal(self._read("user-beta","project-alpha"),
                                    self._read("user-beta","project-beta")))

    def test_multiple_feedback_steps_persist(self):
        a=self.update(n=1)
        old=self.bank.latent("user-alpha","project-alpha")[1]
        b=self.update(n=2)
        new=self.bank.latent("user-alpha","project-alpha")[1]
        self.assertEqual(len(self.bank.events),2)
        self.assertFalse(torch.equal(old,new))
        self.assertNotEqual(a["source"],"")
        self.assertEqual(b["source"],SRC)

    def test_no_duplicate_feedback_replay(self):
        self.update(n=1)
        with self.assertRaisesRegex(ValueError,"already used"):
            self.update(n=1)

    def test_checkpoint_roundtrip_portable_without_lm(self):
        self.update(n=1)
        with tempfile.TemporaryDirectory() as folder:
            dest=Path(folder)/"personal.json"
            self.bank.save(dest)
            clone=PersonalProjectState.load(dest)
            self.assertEqual(clone.to_data(),self.bank.to_data())
            self.assertTrue(torch.equal(clone.latent("user-alpha","project-alpha")[1],
                                        self.bank.latent("user-alpha","project-alpha")[1]))
            self.assertNotIn("model_hidden_width",json.loads(dest.read_text()))

    def test_wrong_scoped_feedback_rejected(self):
        with self.assertRaises(ValueError):
            personalized_update(self.model,self.bridge,self.bank,
                user_id="user-alpha",project_id="project-alpha",scope="personal",
                candidate_ids=self.candidates,chosen_index=0,yes_token_id=7,
                no_token_id=8,feedback=fake_feedback(1))

    def test_non_user_or_unverified_feedback_rejected(self):
        for feedback in (fake_feedback(user=False),fake_feedback(verified=False),
                         {"event_id":"reply.1","source":SRC,"speaker":"user"}):
            with self.subTest(feedback=feedback),self.assertRaises(ValueError):
                personalized_update(self.model,self.bridge,self.bank,
                    user_id="user-alpha",project_id="project-alpha",scope="project",
                    candidate_ids=self.candidates,chosen_index=0,yes_token_id=7,
                    no_token_id=8,feedback=feedback)
        self.assertFalse(self.bank.events)

    def test_no_mismatched_bridge_shapes(self):
        with self.assertRaises(ValueError):
            with active_compass(self.model,self.bridge,torch.zeros(5),torch.zeros(4)):
                pass
        with self.assertRaises(ValueError):
            wrong=FrozenResidualBridge(13,rank=4)
            pair_margins(self.model,wrong,torch.zeros(4),torch.zeros(4),self.candidates,7,8)

    def test_hook_cleanup_after_exception(self):
        freeze_backbone(self.model)
        pre=self.model(input_ids=self.candidates[0]).logits.detach()
        with self.assertRaisesRegex(RuntimeError,"synthetic"):
            with active_compass(self.model,self.bridge,torch.zeros(4),torch.zeros(4)):
                raise RuntimeError("synthetic")
        post=self.model(input_ids=self.candidates[0]).logits.detach()
        self.assertTrue(torch.equal(pre,post))

    def test_state_scope_validation(self):
        with self.assertRaises(ValueError):
            self.bank.latent("bad/../name")
        for src in ("unscoped",f"git:{'f'*40}:docs/../leak.md#L1-L2",
                    f"git:{'f'*40}:docs/test.md#L9-L2"):
            with self.assertRaises(ValueError):
                check_source(src)

    def test_corrupted_weights_or_authority_rejected(self):
        original=self.bank.to_data()
        modified=copy.deepcopy(original)
        modified["users"]={"user-alpha":[float("nan")]*4}
        with self.assertRaises(ValueError):
            PersonalProjectState.from_data(modified)
        modified=copy.deepcopy(original)
        modified["no_auto_policy_approval"]=False
        with self.assertRaises(ValueError):
            PersonalProjectState.from_data(modified)

    def test_provenance_journal_no_promotion_to_policy(self):
        self.update()
        rows=self.bank.to_data()["events"]
        self.assertEqual(len(rows),1)
        self.assertFalse(rows[0]["approved_project_policy"])
        self.assertEqual(rows[0]["scope"],"project")
        self.assertNotIn("policy_approved",self.bank.to_data())

    def test_live_eval_reports_frozen_baseline_and_adapter(self):
        freeze_backbone(self.model)
        report=inspect_live(self.model,self.bridge,self.bank,
                            "user-alpha","project-alpha",self.candidates,7,8)
        self.assertEqual(len(report["frozen_no_compass"]),2)
        self.assertEqual(len(report["with_compass"]),2)
        self.assertTrue(report["margins_changed"])

    def test_frozen_bridge_can_reuse_v90_matrix_layout(self):
        self.assertEqual(tuple(self.bridge.down.weight.shape),(4,12))
        self.assertEqual(tuple(self.bridge.up.weight.shape),(12,4))
        self.assertEqual(sum(p.numel() for p in self.bridge.parameters()),96)
        self.assertEqual(sum(p.numel() for p in self.bridge.parameters() if p.requires_grad),0)

    def test_swapping_project_latents_changes_output_without_backbone_update(self):
        self.update(project="project-alpha",idx=1,n=1)
        self.update(project="project-beta",idx=0,n=2)
        self.assertFalse(torch.allclose(self._read("user-alpha","project-alpha"),
                                        self._read("user-alpha","project-beta")))
        self.assertTrue(same(self.model_pre,params_snapshot(self.model)))

    def fake_v11_queue(self, *, event="review.1", scope="project",
                       project="project-alpha", user="user-alpha"):
        row = {
            "schema":"COMPASS_V12_TRAINING_PAIR_NOT_EVALUATION",
            "user_id":user,"scope":scope,"project_id":project,
            "event_id":event,"source":SRC,"question":"Fictional old project knowledge",
            "candidates":["discard previous research","retain and improve research"],
            "chosen_index":1,"review_status":"USER_CONFIRMED",
            "prospective_holdout_eligible":False,
            "project_policy_approved":False,
            "feedback":{"event_id":event,"source":SRC,"speaker":"user",
                        "confirmed_by_user":True,"source_verified":True}}
        return {"schema":"COMPASS_V12_PRIVATE_TRAINING_QUEUE_V1",
                "user_id":user,"evaluation_pairs":0,"pairs":[row]}

    def fake_encode(self, question, candidate):
        return self.candidates[1] if candidate.startswith("retain") else self.candidates[0]

    def test_integrated_source_review_queue_changes_detached_project_state(self):
        q=self.fake_v11_queue()
        report=train_queue(self.model,self.bridge,self.bank,q,encode=self.fake_encode,
                           yes=7,no=8,steps=12,lr=0.9)
        self.assertEqual(report["updates"],1)
        self.assertFalse(report["authoritative_project_decisions_changed"])
        self.assertFalse(report["model_backbone_trained"])
        self.assertGreater(report["events"][0]["state_delta_l2"],0)
        self.assertTrue(torch.equal(self.bank.latent("user-alpha","project-beta")[1],torch.zeros(4)))

    def test_repeated_review_batches_resume_without_clobber(self):
        first=self.fake_v11_queue(event="review.1")
        second=self.fake_v11_queue(event="review.2")
        for q in (first,second):
            train_queue(self.model,self.bridge,self.bank,q,encode=self.fake_encode,
                        yes=7,no=8,steps=6,lr=0.6)
        self.assertEqual(len(self.bank.events),2)
        self.assertGreater(self.bank.latent("user-alpha","project-alpha")[1].norm().item(),0)

    def test_bad_future_label_or_missing_source_rejected_prior_to_any_update(self):
        q=self.fake_v11_queue()
        q["pairs"][0]["prospective_holdout_eligible"]=True
        with self.assertRaisesRegex(ValueError,"not prospective"):
            validate_queue(q,self.bank,self.bridge)
        self.assertEqual(len(self.bank.events),0)
        q=self.fake_v11_queue()
        q["pairs"][0]["feedback"]["source"]="not-a-real-source"
        with self.assertRaises(ValueError):
            validate_queue(q,self.bank,self.bridge)
        self.assertEqual(len(self.bank.events),0)

    def test_all_or_nothing_user_state_if_second_sample_encoding_fails(self):
        queue=self.fake_v11_queue()
        second=copy.deepcopy(queue["pairs"][0])
        second["event_id"]="review.2"
        second["feedback"]["event_id"]="review.2"
        queue["pairs"].append(second)
        calls=[0]
        def encode(question,candidate):
            calls[0]+=1
            if calls[0]>=3:
                raise ValueError("synthetic invalid second context")
            return self.fake_encode(question,candidate)
        before=self.bank.to_data()
        with self.assertRaisesRegex(ValueError,"synthetic invalid"):
            train_queue(self.model,self.bridge,self.bank,queue,encode=encode,
                        yes=7,no=8,steps=6,lr=0.6)
        self.assertEqual(before,self.bank.to_data())

if __name__=="__main__":
    unittest.main(verbosity=2)
