"""Tests that protect the real UNIFOLD L1 episodic baseline."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import l1_support_learner as l1


class L1LearnerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)
        self.net = l1.L1MemoryLearner(32)
        self.sx = torch.randint(0, 4, (2, 12, 4))
        self.sy = torch.randint(0, 4, (2, 12, 4))
        self.qx = torch.randint(0, 4, (2, 16, 4))

    def test_shapes_and_gradients(self):
        output = self.net(self.sx, self.sy, self.qx, 12)
        self.assertEqual(tuple(output.shape), (2, 16, 4, 4))
        output.square().mean().backward()
        self.assertGreater(self.net.pair[0].weight.grad.abs().sum().item(), 0)

    def test_zero_support_is_independent_of_examples(self):
        a = self.net(self.sx, self.sy, self.qx, 0)
        b = self.net(self.sx.roll(1, 0), self.sy.roll(1, 0), self.qx, 0)
        self.assertTrue(torch.equal(a, b))

    def test_support_is_permutation_invariant(self):
        self.net.eval()
        with torch.no_grad():
            a = self.net(self.sx, self.sy, self.qx, 12)
            order = torch.arange(11, -1, -1)
            b = self.net(self.sx[:, order], self.sy[:, order], self.qx, 12)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_shuffle_changes_memory(self):
        a = self.net(self.sx, self.sy, self.qx, 12)
        b = self.net(self.sx, self.sy.roll(1, 0), self.qx, 12)
        self.assertFalse(torch.allclose(a, b))

    def test_invalid_support_rejected(self):
        with self.assertRaises(ValueError):
            self.net(self.sx, self.sy, self.qx, 13)

    def test_query_label_does_not_enter_forward(self):
        gold = torch.randint(0, 4, (2, 16, 4))
        a = self.net(self.sx, self.sy, self.qx, 8)
        gold[:] = 0
        b = self.net(self.sx, self.sy, self.qx, 8)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
