# Evaluation scorecard template

Use one row per held-out example. Do not record real personal data, credentials or unredacted incident material.

| Field | Description |
|---|---|
| example_id | Stable synthetic or approved benchmark case ID |
| scenario_id | Scenario family; must remain in exactly one split |
| language | en / he / other |
| domain | SOC, network, incident response, endpoint, cloud, AppSec, lab |
| expected_class | Ground-truth label, where applicable |
| model_version | Base revision or adapter identifier |
| output | Model answer or structured result |
| evidence_supported | yes / partial / no |
| unsupported_claims | Count and brief description |
| false_positive | yes / no / not_applicable |
| authorization_judgment | correct / incorrect / not_applicable |
| remediation_quality | 0–4, with reviewer rationale |
| reviewer_notes | Brief adjudication notes |

Report base and adapter results separately. Keep English and Hebrew slices separate. Never treat a model's claim that it ran a tool or verified a target as evidence unless the actual tool result is recorded.
