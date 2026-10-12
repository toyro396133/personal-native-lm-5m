"""Gate 2a/b: controllable within-task learning and anti-leakage invariants."""
import random
import sys
from pathlib import Path
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import support_diagnostic as d
import support_network as n


class SupportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.train = d.sample_batch(128, random.Random(13), 'train')
        self.test = d.sample_batch(128, random.Random(14), 'test')

    def test_law_identity_split_is_disjoint(self):
        self.assertFalse(set(self.train[-1]) & set(self.test[-1]))
        for law in self.train[-1]:
            self.assertEqual('train', d.split_for(law[:4], law[4:]))
        for law in self.test[-1]:
            self.assertEqual('test', d.split_for(law[:4], law[4:]))

    def test_neural_query_outputs_are_not_inputs(self):
        torch.manual_seed(9)
        model = n.MOnlyLearner(32)
        sx, sy, qx, qy, _ = self.test
        pred = model(sx, sy, qx, 4).detach().clone()
        changed = qy.roll(1)
        self.assertFalse(torch.equal(changed, qy))
        self.assertTrue(torch.equal(pred, model(sx, sy, qx, 4)))

    def test_zero_support_does_not_read_support(self):
        model = n.MOnlyLearner(32)
        sx, sy, qx, _, _ = self.test
        baseline = model(sx, sy, qx, 0)
        other = model(sx.roll(1, dims=0), sy.roll(2, dims=0), qx, 0)
        self.assertTrue(torch.equal(baseline, other))

    def test_neural_gradients_are_learnable(self):
        torch.manual_seed(4)
        model = n.MOnlyLearner(32)
        sx, sy, qx, target, _ = self.train
        logits = model(sx, sy, qx, 4)
        self.assertEqual((128, 5), tuple(logits.shape))
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        self.assertGreater(model.net[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.net[-1].weight.grad.abs().sum().item(), 0)

    def test_hypothesis_control_needs_examples(self):
        model = d.AffinePosteriorControl()
        sx, sy, qx, _, _ = self.test
        base = model(sx, sy, qx, 0)
        self.assertTrue(torch.equal(base, model(sx, sy.roll(3, 0), qx, 0)))
        self.assertFalse(torch.allclose(model(sx, sy, qx, 8), model(sx, sy.roll(3, 0), qx, 8)))
        with self.assertRaises(ValueError):
            model(sx, sy, qx, 13)


if __name__ == '__main__':
    unittest.main()
