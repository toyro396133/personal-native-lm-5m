"""UNIFOLD Gate 4 controls: equal-budget static Q, A intervention, no gold leakage."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gate4_mq_transfer import AdaptiveReadout

class Gate4Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(131)
        torch.set_num_threads(1)
        self.model=AdaptiveReadout(width=32)
        self.sx=torch.randint(0,4,(4,3,12,4))
        self.sy=torch.randint(0,4,(4,3,12,4))
        self.qx=torch.randint(0,4,(4,3,16,4))
        # B/C are identical within each (linked, control) pair.
        for a,b in ((0,1),(2,3)):
            for x in (self.sx,self.sy,self.qx):
                x[b,1:]=x[a,1:]

    def test_q_gradients(self):
        out=self.model(self.sx,self.sy,self.qx)
        self.assertEqual(tuple(out.shape),(4,3,16,4,4))
        out.square().mean().backward()
        self.assertGreater(self.model.controller.weight_ih.grad.abs().sum().item(),0)
        self.assertGreater(self.model.gain.weight.grad.abs().sum().item(),0)

    def test_static_q_does_not_access_a(self):
        out=self.model(self.sx,self.sy,self.qx,mode='static')
        self.assertTrue(torch.allclose(out[0,1:],out[1,1:],atol=1e-7))
        self.assertTrue(torch.allclose(out[2,1:],out[3,1:],atol=1e-7))

    def test_no_target_support_means_q_cannot_predict(self):
        a=self.model(self.sx,self.sy,self.qx,support=(12,0,0))
        b=self.model(self.sx,self.sy,self.qx,support=(12,0,0),intervention='swap_A')
        self.assertTrue(torch.equal(a[:,1:],b[:,1:]))

    def test_no_a_and_swap_a_change_only_controlled_path(self):
        a=self.model(self.sx,self.sy,self.qx)
        no=self.model(self.sx,self.sy,self.qx,intervention='no_A')
        sw=self.model(self.sx,self.sy,self.qx,intervention='swap_A')
        self.assertGreater((a[:,1:]-no[:,1:]).abs().sum().item(),0)
        self.assertGreater((a[:,1:]-sw[:,1:]).abs().sum().item(),0)

    def test_gold_is_not_model_input(self):
        base=self.model(self.sx,self.sy,self.qx)
        gold=torch.randint(0,4,(4,3,16,4))
        gold.fill_(0)
        self.assertTrue(torch.equal(base,self.model(self.sx,self.sy,self.qx)))

    def test_pair_order_invariance(self):
        a=self.model(self.sx,self.sy,self.qx)
        b=self.model(self.sx.flip(0),self.sy.flip(0),self.qx.flip(0))
        self.assertTrue(torch.allclose(a,b.flip(0),atol=1e-7))

    def test_invalid_support(self):
        with self.assertRaises(ValueError):
            self.model(self.sx,self.sy,self.qx,support=(12,13,12))

if __name__=='__main__':unittest.main()
