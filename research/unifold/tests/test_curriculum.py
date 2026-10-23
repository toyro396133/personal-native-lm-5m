"""UNIFOLD curriculum dataset tests: paired transfer, novel L2, anti-leakage."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import curriculum as c


class CurriculumTests(unittest.TestCase):
    def test_compositional_relation_pool(self):
        pool, relations = c.candidates(42)
        self.assertGreater(len(relations['train']), 10)
        self.assertGreater(len(relations['val']), 10)
        self.assertGreater(len(relations['test']), 10)
        self.assertTrue(all(len(program) == 3 for program, *_ in
                            [(case[4],) for case in relations['test'][:10]]))

    def test_small_dataset_and_counterfactual(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'curriculum'
            result = c.generate(path, pairs=20, seed=42)
            self.assertEqual(result['sequence_count'], 40)
            audit = c.validate(path)
            self.assertTrue(audit['matched_BC'])
            self.assertTrue(audit['C_compositional_witness'])
            rows = [json.loads(x) for x in (path/'sequences.jsonl').read_text().splitlines()]
            self.assertFalse(any('condition' in row for row in rows))
            self.assertFalse(any(row['split'] == 'train' and row['stages'][2]['level'] == 'L2' for row in rows))

    def test_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a', Path(temp)/'b'
            self.assertEqual(c.generate(a, pairs=12, seed=123), c.generate(b, pairs=12, seed=123))

    def test_gold_tampering_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)/'task'
            c.generate(d, pairs=6, seed=42)
            with (d/'answers.PRIVATE.jsonl').open('a') as f:
                f.write(' \n')
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                c.validate(d)

    def test_invalid_count(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                c.generate(Path(temp)/'x', pairs=2)


if __name__ == '__main__':
    unittest.main()
