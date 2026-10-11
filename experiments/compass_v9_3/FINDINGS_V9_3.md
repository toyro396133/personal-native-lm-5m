# COMPASS V9.3 — completed scientific findings: semantic generalization improves, opposing project concepts still fail

**Official reproducible run:** [GitHub Actions #38121831473](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38121831473), completed **SUCCESS** on 2026-10-11 at ~07:36 UTC, both mechanical preflight and real frozen-encoder CPU study successful. Full artifact **11695438682** (`compass-v93-semantic-memory.json`, `compass-v93-semantic-memory.safetensors`). **Scientific conclusion: NO_GO_TO_BACKBONE_COUPLING** because the required heldout opposed-project concept separation failed.

## What was actually tested

- Frozen pretrained [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) **384-dimensional sentence embeddings**. No encoder gradients or retraining. A learned shared 384→64 projection, independent global-user and project GRUCell update streams (48D state each), independent classification heads. **57,796 trainable shared state-learner weights**; the encoder's pretrained weights are not in this count.
- Owner-global principle and project purpose/compatibility each remain **binary**, synthesized by experimenter. Three short correction sentences in each of three streams: user, project A, project B. The project A/B updater uses the same weights but **separate incremental state**. There is no per-user fine-tuning, independently learned user/project checkpoint, durable memory service, authority verification or active large-LM integration. It is **not** evidence that a personal model learned freely articulated abstract goals.
- Train 600 steps batch32 seed11, best checkpoint at **step 50**, internal training-template dev accuracy **576/576**. No near/cold/far sentences used for training or selecting step. The learned GRU state can be updated by one forward step per new utterance without backward gradients.
- Evaluated **600 previously exposed V9.2 "near" paraphrase episodes** and **600 newly researcher-authored V9.3 "far" paraphrase episodes**, each with global and two project principle histories (1,800 scored state labels per split). Cold far has 16 researcher-authored new sentences, no independent human blind review. All labels and task dimensions were authored by researchers and repeated across training/test semantic categories.
- Tested controls: no memory (constant), user-only stream, project-only stream, both, full-history rule-based latest-utterance keyword reader, frozen semantically embedded latest-utterance prototypes, frozen semantically embedded recency-weighted full-history prototypes, oracle class labels. There is **NO frozen 360M model used here and NO response to a novel dilemma text**; metric is fixed binary-concept recovery, not general reasoning over novel tasks or long-term project-vision understanding.

## Primary results (percent)

| Arm | NEAR 600 episodes: overall principle recovery | FAR 600 episodes: overall | FAR global user | FAR two projects combined | FAR late correction |
|---|---:|---:|---:|---:|---:|
| No memory | 52.44% | 50.50% | 51.83% | 49.83% | 50.37% |
| Personal only | 62.50% | 57.11% | 71.67% | 49.83% | 58.56% |
| Project only | 84.72% | 75.22% | 51.83% | 86.92% | 73.70% |
| **Hierarchical user + projects neural** | **94.78%** | **81.83%** | **71.67% (430/600)** | **86.92% (1,043/1,200)** | **81.89%** |
| Full-history simple keyword reader | 94.78% | 79.39% | 64.00% | 87.08% | 80.02% |
| Frozen semantic nearest-last-statement prototype | 94.78% | 79.11% | 63.50% | 86.92% | 77.42% |
| Frozen semantic recency-weighted full-history prototype | 88.67% | 78.83% | 64.00% | 86.25% | 76.43% |
| Known binary concept oracle | 100% | 100% | 100% | 100% | 100% |

**Important:** The **project-specific** accuracy of the neural updater (86.92%) and of frozen nearest-last-statement semantic prototypes (**86.92%**) are **numerically identical on the far set** (A = 522/600, B = 521/600 in neural arm). The gains over cheap alternatives are mainly from the **global user** stream rather than evidence of a superior project-level recurrent mechanism. Do not attribute all result gains to online learned state. The neural overall FAR advantage vs **strongest** listed non-neural baseline (keyword reader 79.39%) is just **2.44 percentage points** on research-authored synthetic cases, not an independent replicated significance result and not proof of an advantage over frozen large LM + retrieval.

- **Near diagnostic 94.78%** cannot be treated as a blind replication of V9.2: test templates have been exposed across iterations. Comparisons to V9.2 68.83% and V9.2 global 38.33% use **different random episode samples**, encoder/data generation regimes, and checkpoints; they are indicative, NOT controlled same-cases causal effects.
- **True project-concept opposition detection on far samples: 73.1343%**, below the locked **80%** minimum, despite 100% programmatic isolation. The model sometimes predicts the same concept for two actually different projects.
- Exact project B counterfactual history swap: **128/128** global state unchanged, **128/128** project A unchanged, **128/128** B equals A after copying A history. The causal test validates scope separation in code, not quality of learned concepts.
- Incremental GRU one-message-at-a-time state updates numerically match processing the same sequence at once.

## Preregistered gate results

| Gate | Result |
|---|---|
| Near concept accuracy ≥85% | PASS — 94.78% |
| Far concept accuracy ≥75% | PASS — 81.83% |
| Far global-user accuracy ≥70% | PASS — 71.67% |
| Far late correction recovery ≥75% | PASS — 81.89% |
| Far opposed project concept separation ≥80% | **FAIL — 73.13%** |
| Neural beats strongest rule/semantic comparison by ≥2 pp (far) | PASS — 81.83 vs 79.39 (+2.44 pp) |
| Perfect program-level isolation and incremental-state equivalence | PASS — 128/128 intervention cases |

**Verdict: NO_GO_TO_BACKBONE_COUPLING.** 6/7 registered gates passed, but this is an intentionally conjunctive research protocol; we do not waive the project opposition failure. There is also no high-quality frozen-LLM+RAG baseline, no independent human holdout, no multiple-seed replication or robust cross-domain task evaluation.

## What has been learned, and next controlled question

A pretrained semantic embedding front end is a credible response to V9.2's raw hashed-word semantic generalization failure on held-out *similar* paraphrases. In the new pilot, supervised small personal state produced measurable general-user concept recovery, but project-specific understanding remains nearly matched by one-statement exemplar similarity and conflicting project principles still confuse the model. This does **not** justify merely growing personal adapter weights or reverting to prompt-level score feedback.

**Next step should focus precisely on distinguishing contrasting project intents and genuine unseen decision reasoning:**
1. On held-out, human-authored or independently generated multiple project histories, introduce *matched contradictory goals*, revisions and realistic intent-vs-implementation dilemmas with irrelevant details and ambiguous phrases.
2. Contrastively train **project A-vs-B difference** and **global-vs-project priority** without conflating source authority with semantic prediction; audit per-concept confusion matrices.
3. Use a matched frozen-language-model full-text/retrieval baseline, semantic-prototype retrieval, and transparent rule control, and evaluate **answers to new problem statements**, not only binary principle classification.
4. Repeat seeds and independent domain families, report confidence intervals and compute/latency.
5. Only after real held-out concept transfer has passed these tests should a learned concept state be attached to a frozen 360M Transformer through a personal neural adapter. Ultimate user vision requires independently evolving user/project states and cross-project transfer under explicit authority and revocation safeguards.

## Provenance and files

- Branch: `experiment/compass-v9-3`, preregistration `EXPERIMENT_V9_3.md`, source `study_v93.py`.
- [GitHub Actions #38121831473](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38121831473) (both stages passed).
- [Run results archive](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38121831473/artifacts/11695438682), including per-state confusion counts, sample concept histories/predictions, 600-step train trace and learned memory weights. Not merged into native SELF 5M nor deployed.
