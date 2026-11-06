import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_skill_coverage.py"
SPEC = importlib.util.spec_from_file_location("audit_skill_coverage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

class SkillCoverageTests(unittest.TestCase):
    def test_counts_domain_scoped_expected_skills(self):
        rows = [
            {"domain": "soc", "expected_skills": ["telemetry_health", "evidence_analysis"]},
            {"domain": "soc", "expected_skills": ["telemetry_health"]},
            {"domain": "network_security", "expected_skills": ["telemetry_health"]},
        ]
        matrix = {"target_min_examples_per_skill": 2, "domains": {
            "soc": ["telemetry_health"], "network_security": ["telemetry_health"]
        }}
        report = MODULE.build_report(rows, matrix)
        self.assertEqual(report["domains"]["soc"]["skills"]["telemetry_health"]["examples"], 2)
        self.assertEqual(report["domains"]["network_security"]["skills"]["telemetry_health"]["examples"], 1)
        self.assertEqual(len(report["gaps"]), 1)

    def test_generic_skill_does_not_mask_domain_gap(self):
        rows = [{"domain": "soc", "expected_skills": ["evidence_analysis"]}]
        matrix = {"target_min_examples_per_skill": 1, "domains": {"network_security": ["evidence_analysis"]}}
        report = MODULE.build_report(rows, matrix)
        self.assertEqual(report["domains"]["network_security"]["skills"]["evidence_analysis"]["examples"], 0)

if __name__ == "__main__":
    unittest.main()
