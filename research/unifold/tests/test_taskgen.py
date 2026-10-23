"""Smoke tests for the independent UNIFOLD task corpus generator."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import taskgen


class GeneratorTest(unittest.TestCase):
    def test_operator_stays_in_domain(self):
        for name in taskgen.OPS:
            for value in taskgen.DOMAIN:
                result = taskgen.op(name, value)
                self.assertIn(result, taskgen.DOMAIN)

    def test_pool_semantics_are_disjoint(self):
        pools = taskgen.make_pool(42)
        all_sigs = []
        for key, rules in pools.items():
            self.assertTrue(rules, key)
            self.assertEqual(len(rules), len(set(sig for _, sig in rules)))
            for program, sig in rules:
                self.assertEqual(len(program), 2 if key.endswith("l1") else 3)
            all_sigs.append({s for _, s in rules})
        for i, first in enumerate(all_sigs):
            for second in all_sigs[i+1:]:
                self.assertFalse(first.intersection(second))

    def test_small_dataset_is_valid_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "a", Path(temp) / "b"
            a = taskgen.generate(first, 30, seed=73)
            b = taskgen.generate(second, 30, seed=73)
            self.assertEqual(a, b)
            self.assertTrue(taskgen.validate(first)["passed"])
            lines = [json.loads(line) for line in (first / "tasks.jsonl").read_text().splitlines()]
            self.assertEqual(len(lines), 30)
            self.assertTrue(all("query_outputs" not in x and "program" not in x for x in lines))
            self.assertFalse(any(x["split"] == "train" and x["level"] == "L2" for x in lines))

    def test_rejects_tampered_gold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            taskgen.generate(root, 15)
            gold = root / "answers.PRIVATE.jsonl"
            gold.write_text(gold.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Integrity mismatch"):
                taskgen.validate(root)

    def test_rejects_empty_target_or_too_many_examples(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                taskgen.generate(Path(temp)/"a", count=4)
            with self.assertRaises(ValueError):
                taskgen.generate(Path(temp)/"b", count=10, support_size=250, query_size=12)


if __name__ == "__main__":
    unittest.main()
