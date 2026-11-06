import importlib.util,json,unittest
from pathlib import Path
SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"validate_dataset.py"
spec=importlib.util.spec_from_file_location("validator",SCRIPT); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
def sample():
 return {"id":"x","scenario_id":"s1","split":"train","domain":"soc","task":"triage","environment":"defensive","difficulty":"intermediate","messages":[{"role":"system","content":"Careful analyst"},{"role":"user","content":"Review a suspicious login pattern with adequate context."},{"role":"assistant","content":"The evidence is insufficient; collect correlated authentication and endpoint records before classifying."}],"expected_skills":["evidence_analysis"],"verification":{"method":"automated_structure_plus_pending_expert_review","status":"pending"},"source":{"name":"Synthetic","url":"","license":"Original","checked_at":"2026-11-06","usage":"Original synthetic"},"quality":{"score":None,"review_status":"pending","notes":"Pending"},"authorization":{"scope":"defensive","status":"not_applicable"},"provenance":{"content_origin":"original_synthetic","source_family":"original","source_usage":"original"}}
class ValidatorTests(unittest.TestCase):
 def test_valid(self): self.assertEqual(mod.validate_rows([sample()]),[])
 def test_duplicate_id(self):
  b=sample(); b["scenario_id"]="s2"; self.assertTrue(any("duplicate id" in e for e in mod.validate_rows([sample(),b])))
 def test_scenario_split_leak(self):
  b=sample(); b["id"]="y"; b["split"]="test"; self.assertTrue(any("crosses splits" in e for e in mod.validate_rows([sample(),b])))
 def test_lab_requires_target(self):
  a=sample(); a["environment"]="authorized_lab"; a["authorization"]={"scope":"authorized_lab","status":"explicit"}; self.assertTrue(any("permitted_target" in e for e in mod.validate_rows([a])))
 def test_missing_source(self):
  a=sample(); a["source"]={"name":"x"}; self.assertTrue(any("source metadata" in e for e in mod.validate_rows([a])))
 def test_review_status_consistency(self):
  a=sample(); a["quality"]["review_status"]="reviewed"; self.assertTrue(any("reviewed requires expert" in e for e in mod.validate_rows([a])))
if __name__=="__main__": unittest.main()
