# COMPASS V8.8 — COMPLETED: active 360M advisor failed policy-switch generalization

**GitHub Actions run:** [38112408784](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38112408784) completed **SUCCESS** on its FIRST attempt, 2026-10-11 05:08:51 UTC. All three infrastructure jobs succeeded. **Scientific research verdict: FAIL / NO-GO.** This is a failure of the current ACTIVE advisor prompt and teacher-to-student interface, **not evidence that a capable active advisor can never work**.

## Objective and frozen assumptions

The user explicitly chose a 360M language model as **active reasoning partner** for the personal 7,680-weight COMPASS, rather than a vector-only encoder. We tested a frozen `HuggingFaceTB/SmolLM2-360M-Instruct` teacher that generates concise written assessment of project objective, owner priority, and evidence, then scores canonical Action A vs B. Its advice was inserted into per-candidate text re-encoded with the frozen shared 360M model for the learned personal scorer. We also evaluated the teacher's own choices, the unchanged rank4 V8.6 scorer with and without teacher advice, and two equally trained seed11 18-epoch student arms (text-only vs teacher-assisted), each **exactly 7,680 learned personal parameters**. The 360M backbone was NOT fine-tuned.

Only **16 newly authored scripted synthetic policy-only test pairs** were treated as previously uninspected V8.8 families: `deployment_communications`, `backlog_triage`, balanced 8 EN / 8 HE, with two already-known fictional test projects. The previous V8.6 policy pairs and V8.7 policy families plus historic 176 tasks were explicitly EXPOSED diagnostics, not independent validation. No real user training data, independent adjudication, authenticated source-provider calls or production deployment.

## Controlled heldout comparisons — the essential result

Every pair contains the same project ID, goal, evidence, candidate actions, choice order and options, with ONLY the fictional **authorized owner priority** changed. Both decisions must be correct and change from the original to the updated preference.

| Actor/model | New pairs BOTH correct /16 | New pairs ACTUALLY switched decision /16 | EN BOTH correct /8 | HE BOTH correct /8 |
|---|---:|---:|---:|---:|
| Unchanged original V8.6 personal head, no active advisor | **6** | **8** | 4 | 2 |
| Unchanged V8.6 personal head with active advisor generated text injected | **0** | **0** | 0 | 0 |
| Frozen 360M teacher's own A/B choice | **0** | **0** | 0 | 0 |
| V8.8 identical-budget text-only paired training CONTROL | **4** | **4** | 4 | 0 |
| V8.8 active-teacher-assisted paired training EXPERIMENT | **2** | **2** | 2 | 0 |

The active advisor LOST 2 both-correct pairs relative to the identically trained text-only rank4 control, **-4** relative to the original V8.6 checkpoint. Hebrew new-family success was **0/8** after either V8.8 student training scheme.

No pre-registered V8.8 research gate passed for personalized cold-policy pair switching: experimental target >=12/16 both-correct, >=12/16 switched, >=5/8 Hebrew, and >=2 better than equal-budget text-only control. **Observed: 2/16, 2/16, 0/8 and -2 pairs** respectively.

### Forgetting on PREVIOUSLY EXPOSED 176-task diagnostic

- Plain/text-only matched student: **150/176**, as in V8.7.
- Teacher-assisted student: **138/176**, failing >=140/176 retention gate. This is a DIAGNOSTIC regression and not a fresh blind test.
- Original V8.6 head frozen WITHOUT added teacher context scores **6/16** on V8.8 new policy pairs. Giving that unchanged head new teacher analysis reduces performance to **0/16** (no fine-tuning at all). Therefore **the advisor input itself harms rankings** before any learning change, not just training hyperparameters.

### Exposed old policy diagnostics

- Previously exposed V8.6 preference pairs: both V8.8 trained arms **10/16** both-correct, 10/16 flips.
- Previously exposed V8.7 preference pairs: both trained arms **2/16** both-correct, text-only 4/16 choice switches vs teacher-assisted 2/16. These old fixtures were inspected in V8.7 and must NOT be renamed cold.

## Critical debugging root cause — direct archived teacher trace inspection

Inspected original GitHub `compass-v88-active-teacher` artifact ID **11692074929**, `v88-active-teacher.pt`, using `torch.load(weights_only=True)`. This contains all **144 teacher reasoning contexts** (72 EN, 72 HE), the actual generated reasons, canonical choice and per-choice confidence.

**All 144 teacher votes chose `suggested_canonical=0` (Action A) regardless of approved owner preference.** Neither source language, project nor changed policy affected the canonical A/B vote; softmax confidence varied from **0.67781 to 0.93099**, showing that apparent vote confidence is NOT evidence of policy competence.

**All 72 Hebrew generated reason strings were IDENTICAL, just `אתה יועץ תיעדוף בפרויקט`**, a phrase copied from the instruction instead of an explanation. Some English outputs did contain analysis-like words but were truncated/contradictory: e.g., advice about focused delivery while selecting the canonical option corresponding to comprehensive delivery; other outputs merely echoed `Action A...`/parts of the prompt. The teacher was deterministically limited to **28 new tokens**. Prompting the frozen 360M assistant with one short free-form reason and scoring next-token ` A` vs ` B` is not a validated decision-reasoning or calibrated forced-choice procedure. The prompt asks for a reason and then appends a secondary preference question, but never verifies the answer format/invariance or a direct counterfactual contrast.

**Interpretation:** The dominant defect is the **teacher/prompt/inference interface**, not proven lack of capacity of all language models, and not strong evidence about personal-head scale. The teacher behaved like a positional/format-biased source and contaminated student representations. We must not count its words as explanations of a reliable model-internal reasoning process.

## Compute/artifacts/provenance

- GitHub Actions run [38112408784](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38112408784), all 3 jobs PASS.
- `compass-v88-preflight` ID **11692615279**: fixture separation, synthetic source-execution contract.
- `compass-v88-active-teacher` ID **11692074929**: 144 recorded teacher contexts, 288 teacher-augmented per-candidate representations, frozen teacher cache and sample reasoning metadata. Advisor runtime approximately **1485.95 seconds**. Do not conflate inference time with student weight training.
- `compass-v88-active-comparison` ID **11692412450**: full raw per-pair predictions/metrics in `compass-v88-active-r4-s11.json`, and separately preserved `...-text_only_paired_control.safetensors` and `...-teacher_assisted.safetensors` checkpoints.
- Source: `experiments/compass_v8_8/{advisor_v88.py,student_v88.py,fixtures_v88.py,validate_v88.py,EXPERIMENT_V8_8.md}`.
- Native 5M SELF, production and legacy V8.5 sizes **30,720 and 122,880** were untouched.

## What next (not yet implemented)

1. **Do not promote this V8.8 model** or expand size to 30,720/122,880 based on flawed teacher signals. Keep the original V8.6 rank4 checkpoint and other V8.5 ranks intact.
2. Fix advisor task format **before distillation**: test the frozen teacher in isolation with balanced option labels/orders, explicit structured short answer and factual preference comparison; enforce well-formed parser, abstain on unparsable outputs, and flag all-A/B bias.
3. Evaluate teacher robustness on a NEW independently written policy counterfactual test set: both owner priority directions, both choice orders, Hebrew/English, multiple family categories, binding constraints that must NOT flip, and real evidence-provenance cases separately.
4. For architecture involving **bidirectional communication**, send the small model's explicit preference state + reasons for disagreement as INFORMATION to the large advisor for a *bounded* second review; inspect whether the advisor recognizes and resolves the conflict without simply copying the small model. Maintain authoritative user intent/permission checks outside both neural outputs.
5. Compare **explicit verified owner-preference representation + simple deterministic decision policy** as stronger non-neural baseline, plus a 360M active advisor not conditioned on student. The personal scorer may need new architectural inputs, but never assume bigger parameter count repairs teacher's all-A bias.
6. Only after teacher itself achieves held-out policy-sensitivity at a genuinely useful level train/rerun different personal sizes (7,680/30,720/122,880 and optionally 491,520). Independently reviewed labels are ultimately required before claims about real user personalization.

**Research conclusion:** V8.8 technically succeeds but clearly FAILS the causal owner-policy adaptation hypothesis with the current frozen 360M teacher and forced-choice prompting design. The negative result is diagnostic and actionable, not proof that cooperative large+small-model reasoning cannot be effective.
