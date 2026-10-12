"""Gate 5: pre-trained L1 baseline, Q-to-write isolation and paired interventions."""
import sys
import tempfile
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gate5_q_write import QWriteLearner
from l1_support_learner import L1MemoryLearner


class Gate5Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(100)
        torch.set_num_threads(1)
        self.model = QWriteLearner(width=32)
        self.sx = torch.randint(0,4,(4,3,12,4))
        self.sy = torch.randint(0,4,(4,3,12,4))
        self.qx = torch.randint(0,4,(4,3,16,4))
        # Linked/control in each pair share B and C, never A.
        for a,b in ((0,1),(2,3)):
            for data in (self.sx,self.sy,self.qx):
                data[b,1:] = data[a,1:]

    def test_gate3_pretrained_state_loaded_exactly(self):
        base = L1MemoryLearner(32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'pretrained.pt'
            torch.save({'model':base.state_dict()},path)
            self.model.load_gate3(path)
        a = base(self.sx[:,0],self.sy[:,0],self.qx[:,0],12)
        zeros = torch.zeros(4,self.model.qdim)
        b, _ = self.model.stage(self.sx[:,0],self.sy[:,0],self.qx[:,0],zeros,12)
        self.assertTrue(torch.allclose(a,b,atol=1e-6))

    def test_static_q_reduces_to_stage_only_baseline(self):
        out = self.model(self.sx,self.sy,self.qx,mode='fixed')
        self.assertTrue(torch.allclose(out[0,1:],out[1,1:],atol=1e-6))
        self.assertTrue(torch.allclose(out[2,1:],out[3,1:],atol=1e-6))
        frozen = self.model(self.sx,self.sy,self.qx,mode='frozen')
        self.assertTrue(torch.equal(out,frozen))

    def test_no_a_cannot_influence_b_or_c_without_any_a_write(self):
        x=self.model(self.sx,self.sy,self.qx,mode='adaptive',intervention='no_A')
        self.assertTrue(torch.allclose(x[0,1:],x[1,1:],atol=1e-6))
        self.assertTrue(torch.allclose(x[2,1:],x[3,1:],atol=1e-6))

    def test_swapped_a_swaps_q_effect_on_matched_b_c(self):
        original=self.model(self.sx,self.sy,self.qx,intervention='normal')
        swapped=self.model(self.sx,self.sy,self.qx,intervention='swap_A')
        self.assertTrue(torch.allclose(original[0,1:],swapped[1,1:],atol=1e-6))
        self.assertTrue(torch.allclose(original[1,1:],swapped[0,1:],atol=1e-6))
        self.assertFalse(torch.allclose(original[0,1:],swapped[0,1:],atol=0,rtol=0))

    def test_q_cannot_predict_with_no_target_support(self):
        a=self.model(self.sx,self.sy,self.qx,support=(12,0,0))
        b=self.model(self.sx,self.sy,self.qx,support=(12,0,0),intervention='swap_A')
        self.assertTrue(torch.equal(a[:,1:],b[:,1:]))

    def test_q_only_modulates_memory_write_not_decoder(self):
        q0=torch.zeros(4,self.model.qdim)
        q1=torch.ones_like(q0)
        no0,_=self.model.stage(self.sx[:,1],self.sy[:,1],self.qx[:,1],q0,0)
        no1,_=self.model.stage(self.sx[:,1],self.sy[:,1],self.qx[:,1],q1,0)
        self.assertTrue(torch.equal(no0,no1))
        yes0,_=self.model.stage(self.sx[:,1],self.sy[:,1],self.qx[:,1],q0,12)
        yes1,_=self.model.stage(self.sx[:,1],self.sy[:,1],self.qx[:,1],q1,12)
        self.assertFalse(torch.allclose(yes0,yes1,atol=0,rtol=0))

    def test_loss_gradient_reaches_write_rule_and_q_controller(self):
        out=self.model(self.sx,self.sy,self.qx)
        gold=torch.randint(0,4,(4,3,16,4))
        loss=F.cross_entropy(out.reshape(-1,4),gold.reshape(-1))
        loss.backward()
        self.assertGreater(self.model.q_scale.weight.grad.abs().sum().item(),0)
        self.assertGreater(self.model.controller.weight_ih.grad.abs().sum().item(),0)

    def test_pair_reordering_preserves_inference(self):
        a=self.model(self.sx,self.sy,self.qx)
        b=self.model(self.sx.flip(0),self.sy.flip(0),self.qx.flip(0))
        self.assertTrue(torch.allclose(a,b.flip(0),atol=1e-6))

    def test_invalid_shapes_or_interventions_fail(self):
        with self.assertRaises(ValueError):
            self.model(self.sx,self.sy,self.qx,support=(12,13,12))
        with self.assertRaises(ValueError):
            self.model(self.sx,self.sy,self.qx,mode='unknown')
        with self.assertRaises(ValueError):
            self.model(self.sx[:3],self.sy[:3],self.qx[:3],intervention='swap_A')


if __name__ == '__main__':
    unittest.main()
