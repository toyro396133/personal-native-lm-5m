import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_dataset.py"
SPEC = importlib.util.spec_from_file_location("validate_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(example_id="a", scenario="s1", split="train"):
    return {
        "id": example_id,
        "scenario_id": scenario,
        "split": split,
        "domain": "soc",
        "task": "triage",
        "environment": "defensive",
        "difficulty": "intermediate",
        "orientation": "defensive",
        "source": {"name": "Synthetic", "url": "", "license": "Original", "checked_at": "2026-11-06", "usage": "Original synthetic"},
        "authorization": {"scope": "defensive", "status": "not_applicable"},
        "expected_skills": ["evidence_analysis"],
        "verification": {"method": "automated_structure_plus_pending_expert_review", "status": "pending"},
        "quality": {"score": None, "review_status": "pending", "notes": "Pending"},
        "provenance": {"content_origin": "original_synthetic", "source_family": "original", "source_usage": "original"},
        "messages": [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Review this alert."},
            {"role": "assistant", "content": "It needs investigation."},
        ],
    }


class DatasetValidationTests(unittest.TestCase):
    def test_valid_rows(self):
        self.assertEqual(MODULE.validate_rows([row()]), [])

    def test_scenario_cannot_cross_splits(self):
        errors = MODULE.validate_rows([row("a", "same", "train"), row("b", "same", "test")])
        self.assertTrue(any("crosses splits" in error for error in errors))

    def test_lab_requires_explicit_target(self):
        item = row()
        item["environment"] = "authorized_lab"
        item["orientation"] = "authorized_offensive"
        item["authorization"] = {"scope": "authorized_lab", "status": "explicit"}
        errors = MODULE.validate_rows([item])
        self.assertTrue(any("permitted_target" in error for error in errors))

    def test_duplicate_ids(self):
        errors = MODULE.validate_rows([row("same", "one"), row("same", "two")])
        self.assertTrue(any("duplicate id" in error for error in errors))

    def test_final_turn_must_be_assistant(self):
        item = row()
        item["messages"][-1]["role"] = "user"
        errors = MODULE.validate_rows([item])
        self.assertTrue(any("final turn must be assistant" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
