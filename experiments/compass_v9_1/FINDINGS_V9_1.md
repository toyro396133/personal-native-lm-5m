# COMPASS V9.1 — FINAL FINDINGS (2026-10-11)

**Run:** [GitHub Actions #38118071115](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38118071115) completed **SUCCESS**, attempt 1, 2026-10-11 07:03:31 UTC. Earlier start attempts #38117696748, #38117796678, and #38117934128 failed from dataset/runtime/preflight integration errors and yielded **no scientific results**; fixes were committed before the successful new run.

**Scientific verdict: FAIL / NO-GO**. Real internal neural personalization influences intermediate activations and logits, but has **0 out of 32** correct neutral preference-switch pairs, the same as the frozen model. This is not evidence of successful long-term personal learning and must not be deployed or promoted to SELF main.

## Actual V9.1 configuration

- Shared backbone: **361,821,120** frozen `HuggingFaceTB/SmolLM2-360M-Instruct` parameters. Exact non-change verified for two representative backbone tensors; all backbone `requires_grad_(False)`, no LM optimizer weights.
- Early neural user-state conditioner (rank16) inserted via actual live forward hook on the **third-to-last Transformer decoder block output**, with **two subsequent decoder blocks** still executing. Its **30,784 learned parameters** are **SHARED**, not a personalized neural checkpoint per user.
- 4-input state slots but only 2 used and a **single bit of true independent personal information**: `rapid=[1,0,0,0]`, `thorough=[0,1,0,0]`. Switching personal state uses **zero per-user gradient steps**. Same tokenized prompt, without owner policy words, used for the two states.
- Training: 8 CPU epochs, 36 owner-policy-contrastive synthetic paired cases per epoch (24 neutral + 12 mandatory binding); training explicitly rewards neutral preference reversals and binding stability. Development selection best epoch **3**, **6/16** paired both-correct. During training the objective changed minimally: epoch 1 loss 1.40007, epoch 8 loss 1.37498; dev correct never exceeded 6/16. One seed only.
- **NEW scripted cold set**: four project identities not used in train/development × four researcher-authored unseen task families × 2 binding statuses × mirrored candidate order = **64 paired owner-state interventions** (**32 neutral requiring different action**, **32 binding requiring same action**). Synthetic gold devised by experimenter, not reviewed independently.

## Outcomes from `compass-v91-early-personal.json` (artifact 11695360103)

| Arm | Neutral both correct /32 | Binding both correct /32 | All both correct /64 | Semantic action order invariance |
|---|---:|---:|---:|---:|
| Frozen owner-blind 360M | 0 | 32 | 32 | 62/64 |
| **V9.1 personal neural gate** | **0** | **32** | **32** | **64/64** |
| Frozen 360M + approved preference *as text* | 3 | 32 | 35 | 61/64 |
| Synthetic task-tuned lexical rule | 24 | 24 | 48 | 48/64 |

- Neural gate's **actual choice flips = 0/32** neutral, **0/32** binding; it always chose the deep/comprehensive action in both personal modes on all inspected new test categories. High **64/64** order consistency is a consequence of a consistent but unpersonalized choice and should **not** be presented as useful reasoning.
- The explicit-text control achieved only 3/32 neutral policy flips, illustrating that the frozen 360M model itself was weak at this contrived task. The lexical rule is specifically tuned to synthetic scope vocabulary; it is NOT a domain-general comparator and itself misses 8/32 binding cases.
- Actual live same-token neural intervention: downstream hidden difference **max 0.037725449**, action-candidate Yes/No margin difference **0.026943207** when state changes; frozen margin **−0.327864**, rapid **−0.233376**, thorough **−0.206432**. Changes did not reverse semantic action ranking. This establishes the neural path works, not that intent/values have been learned.
- All registered technical gates passed; registered effectiveness gates failed: neutral ≥24/32; strictly better than text and lexical rule. Binding ≥24/32 and order ≥56/64 passed.

## Interpretation: exact scope of what is, and is not, proven

**Proven:** a small, trainable shared neural component can alter hidden states through two additional frozen decoder attention/MLP blocks; an approved user-specific tensor state, not repeated contextual words, affects the large model's logits; no fine-tuning of backbone weights was required.

**Not proven:** effective preference reversals, understanding of user's intent, conceptual abstraction, transferable project vision, distinct permanently trained personalized neural models, long-term continual learning, safe memory update, or robust human generalization.

Likely causal bottlenecks (hypotheses, not proven):
1. Training targets were **single action labels** (and opposite paired margins) for one binary preference, rather than learning latent purposes and project principles; the representation did not encode higher-order goals.
2. Preference-conditioned shift may move both candidate scores in a similar direction, never overcoming the frozen model's semantic preference for complete actions; inspect full per-candidate margins or train explicit **paired relative decision-delta ablations**, not just per-candidate global Yes/No.
3. The shared gate and two one-hot modes are **not an individualized neural learner**. Four state channels are not four learned user attributes.
4. Small repetitive synthetic training data and selection on 16 development pairs; no independent human-generated or adjudicated evaluation.
5. Single early hook on last token in only one chosen layer; not a persistent multi-layer shared/user/project graph. Location change alone did not solve personal learning.

## Strategic direction after user's 2026-10-11 concept-learning clarification

**Change the research priority:** learn **project purposes / governing principles / justification** from sequences of statements, corrections and outcomes, not merely predict an action from a binary quick-vs-deep owner state. Keep the learned representation separate from source-truth/approval metadata. Distinguish user's durable global principles from project-specific purposes; separately test transfer and prevention of harmful leakage.

Next experimental line (proposed **V9.2 concept learning**; NOT run or shown as achieved yet):

1. Author **multiple project histories** with goals, rationales, changed implementations, corrections and counterexamples. Learning input contains earlier interactions, but test prompt must NOT repeat the principle; hold out **unseen tasks and different wording**. Do not accidentally leak gold via project IDs or template words.
2. Separately learn a **personal-general latent** and **project-specific latent**; two distinct update streams, explicit approvals/authority kept as a separate verified record. Test updates, reversals, preserving unchanged rules, conflict precedence and cross-project isolation.
3. Build direct probes of learned latent principles *before* integration, plus **causal neural control**: swap same observed task with different latent histories; ensure correct task changes vs stable mandatory constraints.
4. Use **six comparator arms**: frozen model only, frozen+carefully designed text retrieval, global personal only, project-specific only, both hierarchically connected, and approved explicit symbolic rule baseline. Report accuracy on unseen intent-consequence tests, transfer accuracy, project information leakage, cold concept retention and update/training cost.
5. Only connect the learned personal/project latents to an early frozen-LM module once their own out-of-distribution concept understanding is demonstrated. Compare to text and deterministic rule controls, and report independent reviewed/human-written test cases.
6. Distinguish engineering feasibility from actual capability. Avoid needless increasing parameter sizes or assuming that true intent understanding arises from scale alone.

No modification to native SELF 5M model, no production merge, no rank sweep and no new run in this final-results recording.

## Source artifacts

[Run #38118071115](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38118071115). [Full results artifact](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38118071115/artifacts/11695360103), ID **11695360103**: `compass-v91-early-personal.json` (full per-case predictions and all development history), `compass-v91-early-personal.safetensors` (30,784 shared adapter learned weights). Synthetic preflight artifact ID **11694571995**. Source lives in `experiments/compass_v9_1/`.
