# COMPASS V9.2 — Completed hierarchical concept-learning smoke study

**Official source of truth:** [GitHub Actions run 38121067104](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38121067104), 2026-10-11 07:21:47 UTC, **SUCCESS**. Preflight and real 600-step CPU training/evaluation completed. Initial run [38120988665](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38120988665) failed ONLY when saving weights because `numpy` was missing for safetensors. Workflow dependency fixed before successful rerun; **the experimental data, architecture, labels, and gates remained unchanged**.

**Scientific verdict: NO_GO_TO_LM_COUPLING.** The hierarchical architecture provides mechanically correct project isolation, but **the neural concept representation does NOT generalize reliably to rewritten input**, especially for user-global principles. A **simple textual keyword reader of the same conversation outperforms it substantially**. More layer injection or parameter enlargement is NOT justified.

## What was actually trained

- English-only, original researcher-generated short project/user correction histories. One synthetic episode has three independent streams of 3 statements: global user, project A, and project B. Randomly assigned binary latent concepts with late correction/reversal in ~45% of streams.
- Global user concept (two classes): **deliberate verification** vs **rapid small-scale experimentation**.
- Project concept (two classes): protect **underlying purpose/desired outcome** vs preserve **published compatibility contract**. This is an intentionally narrow laboratory approximation to project "vision"; it is NOT a genuine freely inferred vision of software projects.
- Trainable hashed 4,096 word vocabulary (24D embeddings), independent user/project GRUs (hidden 28), small supervised heads; **107,492 shared trainable parameters**. Each project gets its own transient recurrent state with **the same shared project encoder parameters**. There is **no separately trained project checkpoint** or persistent online user state.
- 600 optimizer steps × 32 synthetic training episodes/step; development set contains ONLY training-utterance templates, best checkpoint selected at **step 50** with **100% train-template development classification accuracy**. Such perfect internal development scores on a tiny shared-template world are NOT evidence of genuine abstraction.
- **600 held-out test episodes**, with non-identical cold rewriting templates that were not used in training **or** checkpoint selection, contrasting project modes and reversed user/project principles, plus newly written dilemma descriptions routed by known scope.
- **Very important honesty:** the new dilemma TEXT is **not consumed by the neural model**. The neural stage only predicts binary concepts from past correction sentences. A deterministic scope-aware decision component turns these concept classes into answers. Hence success here could prove limited history-based concept recovery, **not** semantic reasoning about unfamiliar dilemmas or an attached big LLM.
- No SmolLM2-360M calls, no new internal neural adapter, no native SELF 5M fine-tuning or merge, and no human-authored independent blind evaluation.

## Exact cold results

| Ablation | Global+two-project concept accuracy | Correct after recent correction | Opposing A/B projects separated |
|---|---:|---:|---:|
| No-memory constant prior | **50.56%** | **52.89%** | 0.00% |
| Personal-only (no project principles) | **46.22%** | **48.59%** | 0.00% |
| Project-only (no user principle) | **73.17%** | **74.29%** | 67.84% |
| **Combined hierarchical neural COMPASS** | **68.83%** | **69.99%** | **67.84%** |
| Textual full-history keyword parser | **95.06%** | **95.42%** | **100%** |
| Oracle known synthetic class labels | **100%** | **100%** | **100%** |

Neural breakdown:
- **User-global principle = 230/600 = 38.33%**. Late correction user case = **98/245 = 40.00%**. This is worse than a trivial balanced-prior guess; no support for user-global learning on held-out paraphrases.
- **Project A principle = 515/600 = 85.83%**, late corrections **217/252 = 86.11%**.
- **Project B principle = 494/600 = 82.33%**, late corrections **205/246 = 83.33%**.
- The combined average conceals the fact that personal-generalization is **failing badly**, while project-specific purpose/compatibility separation shows limited learning.

Neural cold accuracy is weaker than a **transparent human-written lexical reader** (which sees the final correction text directly, using keyword sets) by **26.22 percentage points** on overall class labels. It is **not** a frozen LM + high-quality long-context RAG baseline; claims about beating or losing to such a full system have NOT been tested.

### Causal isolation audit

128 counterfactual cases replaced only project B's history with an exact copy of A's, leaving the user and A unchanged:
- **128/128** user principle predictions unchanged.
- **128/128** project A predictions unchanged.
- **128/128** project B prediction agrees with project A after replacing history.

This is strong evidence that *program-level separation of these recurrent streams* works as implemented. It is not evidence that the decoded concepts are accurate or that training naturally discovered privacy constraints. Privacy/authentication/encryption were not exercised.

## Registered gates

| Preregistered gate | Result |
|---|---|
| Cold concept inference accuracy ≥85% | FAIL — 68.83% |
| Recovery after explicit late correction ≥80% | FAIL — 69.99% |
| Correctly separate opposite projects ≥80% | FAIL — 67.84% |
| Hierarchical better than user-only and project-only | FAIL — 68.83% vs project-only 73.17% |
| Neural beats a transparent text-history keyword reader | FAIL — 68.83% vs 95.06% |
| Swapping project B does not perturb user or A | PASS — 128/128 each |

**Final go/no-go = NO_GO_TO_LM_COUPLING**. User+project architecture was structurally sound; generalizable neural principle learning was not.

## Where it probably failed

1. **Semantic OOD gap:** The learner trains a tiny hashed-word embedding table from just 16 train history sentences. Held-out paraphrases shift vocabulary, so the personal branch (38.3%) makes systematically incorrect predictions. "Perfect" dev checkpoint selection on the same training phrasings is an invalid predictor of semantic generalization.
2. **Too narrow and manually programmed concept supervision:** One user bit and one project bit are not a rich continuously learned intention/purpose model.
3. **Deterministic scope decoder:** It sidesteps the hard problem of interpreting the *new task* with real language understanding. The "novel dilemma" benchmark is not an end-to-end semantic transfer test.
4. **No proper strong memory comparison:** The lexical baseline is a fair cheap text comparator, but not an optimized large-model with retrieved history. Any next version must include a strong full-text baseline and evaluate answer quality, not only concept class labels.
5. **Architecture guarantees independence but no learned cross-project transfer:** Global state is structurally separate, however poor generalization means little actual *transferable user understanding* was demonstrated.

All are testable hypotheses or explicit experiment boundaries. The failure does **not** disprove continuous user-specific neural models in general.

## Research redirection

Do **not** immediately inject this failed concept representation into the frozen 360M LM or try 491,520 weights. The next worthwhile step is a preregistered concept-semantic experiment using a **frozen pretrained semantic text encoder** (optionally the existing 360M embedding path) to make held-out paraphrase language meaningful, then a **tiny learned online global/project memory updater**. Compare to a strong frozen-LM full text/retrieval baseline and transparent rules. Keep train/development/cold wording separate; include **human-authored or externally sourced blind novel dilemmas** and explicit project-goal-vs-implementation conflict. Measure *end-to-end decision transfer* before talking about a permanent personal neural module.

One seed is a smoke test, not statistical evidence. Replicate seeds and paraphrase families, and test deletion/revocation/updated permission before deploying.

## Artifacts and reproducibility

- GitHub Actions [successful run 38121067104](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38121067104).
- [Results artifact ID 11695631053](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38121067104/artifacts/11695631053): separate `compass-v92-concept-transfer.json`, `compass-v92-concept-transfer.safetensors`, `compass-v92-preflight.log` (full per-state/per-arm accuracy, training curve, sample predictions, project swap audit).
- Reproducible source: `experiments/compass_v9_2/study_v92.py`, [original preregistration](EXPERIMENT_V9_2.md); experiment branch `experiment/compass-v9-2`.
