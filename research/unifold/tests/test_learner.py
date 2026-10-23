"""Tests that prevent hidden oracle leakage and enforce the Q intervention."""
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import learner
import curriculum


class LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        directory = Path(cls.temp.name)/'curriculum'
        curriculum.generate(directory, pairs=15, seed=42)
        cls.pairs = learner.read_dataset(directory, 'train')[:2]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_state_control_intervention(self):
        model = learner.PlasticLearner(width=32)
        x, y = self.pairs[0][0][0][0, 0], self.pairs[0][0][1][0, 0]
        m, q = model.state(1, torch.device('cpu'))
        m1, q1 = model.observe(x[None], y[None], (m, q), 'adaptive')
        m2, q2 = model.observe(x[None], y[None], (m, q), 'frozen')
        self.assertGreater(q1.abs().sum().item(), 0)
        self.assertEqual(q2.abs().sum().item(), 0)
        self.assertGreater((m1 - m2).abs().sum().item(), 0)

    def test_query_gold_is_not_model_input(self):
        model = learner.PlasticLearner(width=32)
        sx, sy, qx, qy = learner.batch_tensors(self.pairs, torch.device('cpu'))
        torch.manual_seed(3)
        before = model(sx, sy, qx)
        qy += 1
        after = model(sx, sy, qx)
        self.assertTrue(torch.equal(before, after))

    def test_gradients_reach_q(self):
        model = learner.PlasticLearner(width=32)
        loss, _ = learner.run_one(model, self.pairs, 'adaptive', torch.device('cpu'))
        loss.backward()
        self.assertIsNotNone(model.control_cell.weight_ih.grad)
        self.assertGreater(model.control_cell.weight_ih.grad.abs().sum().item(), 0)

    def test_paired_arm_order_does_not_change_predicted_values(self):
        model = learner.PlasticLearner(width=32)
        p = self.pairs[0]
        sx, sy, qx, _ = learner.batch_tensors([p], torch.device('cpu'))
        a = model(sx, sy, qx)
        b = model(sx.flip(0), sy.flip(0), qx.flip(0))
        self.assertTrue(torch.allclose(a, b.flip(0), atol=1e-6))

    def test_invalid_support_size(self):
        model = learner.PlasticLearner(width=32)
        sx, sy, qx, _ = learner.batch_tensors(self.pairs, torch.device('cpu'))
        with self.assertRaises(ValueError):
            model(sx, sy, qx, support_sizes=(13, 12, 12))


if __name__ == '__main__':
    unittest.main()
