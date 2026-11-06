# LFM3-Cyber evaluation plan

The evaluation set is held out by `scenario_id` and is never used for training or model selection.

## Required task families

- SOC alert triage and evidence-backed classification
- network and DNS anomaly analysis
- log interpretation and timeline reconstruction
- incident response and containment prioritization
- Windows/Linux/identity/cloud security review
- vulnerability assessment and remediation advice
- authorized lab assessment: scope recognition, bounded verification and accurate reporting
- secure code review and defensive fix validation

## Metrics

- factual/evidence support rate (every finding tied to provided evidence)
- classification macro-F1 and per-class precision/recall
- false-positive rate and abstention/calibration quality
- timeline ordering accuracy
- remediation correctness and operational safety
- authorization-boundary recognition
- tool selection and result interpretation
- regression against the unadapted base model

## Language split

Run English and Hebrew as separate slices. The official LFM3-1.7B language list does not include Hebrew; Hebrew must be measured directly and reported independently. Do not combine language scores into one average that hides a Hebrew regression.

## Comparison protocol

1. Freeze the exact base model revision and evaluation prompts.
2. Evaluate base model and LoRA adapter on identical held-out examples.
3. Use at least two deterministic decoding settings where applicable.
4. Manually adjudicate ambiguous cases; preserve evidence and rationale.
5. Report confidence intervals where sample size supports them.
6. Do not score offensive capability by testing against real third-party systems. Authorized offensive examples must use isolated labs with explicit target scope.

The initial checked-in JSONL is only a schema fixture. It is not statistically meaningful and cannot support capability claims.
