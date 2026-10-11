# COMPASS V9.3 — frozen semantic representations + incremental personal/project concept memory

**Pre-registered protocol, English-only.** Branch `experiment/compass-v9-3`. V9.2 revealed **68.83%** concept accuracy on test-history paraphrases (global user branch **38.33%**) versus a textual full-history keyword reader at **95.06%**, despite 100% train-template development performance. We test whether a pretrained language-semantic encoder supplies the missing paraphrase invariance, rather than increasing adapter rank or wiring a poorly understood latent state to SmolLM2.

## User's larger intent

A persistent user-personal neural module plus project-specific neural modules should encode WHY a decision was made, project purposes, transferable principles, evolving corrections, and distinguish intent from accidental code implementation. These latents ultimately need to alter a larger frozen model internally **without repeating the full personal context as text at every request**. V9.3 isolates *concept learning* as a prerequisite. It does not claim to implement the complete vision.

## Actual V9.3 model and memory updates

- Frozen pretrained semantic encoder: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), 384-dimensional normalized English sentence embeddings (Apache 2.0 listed on HF). Loaded via `transformers.AutoTokenizer` / `AutoModel`, mean pooling + masked attention and L2 normalization. **All pretrained encoder weights frozen; embeddings cached for unique research sentences.**
- **Trainable shared concept-memory updater**: linear 384→64 + LayerNorm/Tanh, distinct personal and project **GRUCell** updaters (48-dimensional hidden state each), and separate two-class heads. A future user/project can update its own hidden state with **one forward step for each new feedback statement, zero per-user gradient updates**. These transient 48D state vectors are functionally independent for one global user and project A/B within an episode; persistence across sessions is NOT yet built.
- The two supervised global classes remain: verify uncertainty before commitment vs try reversible pilots first. The two project classes remain: prioritize underlying purpose/outcome even when changing implementation vs preserve external compatibility contract. This is intentionally narrow **binary class inference**, not yet genuinely unlimited, deep project vision or self-guided lifelong learning.

## Splits and test construction

- Training histories: 3 researcher-scripted English feedback statements per stream, generated with `study_v92.make_episode(..., 'train')`, including 45% streams whose last event CORRECTS earlier statements. Development uses **only training statement templates** and distinct random episodes; no cold paraphrases in optimizer or checkpoint selection.
- Test near: **600** V9.2-style cold paraphrase episodes (the V9.2 language was previously examined in an earlier experiment; near is exposed cross-study diagnostic, not blind external evidence).
- Test far: **600 new V9.3** researcher-authored paraphrase histories, **16 new sentence templates** (4/class × 2 classes × 2 scopes), distinct from both training texts and V9.2 cold statements; includes same late-change structure. They are held out from optimization and development checkpoint selection, **but still researcher-authored**.
- Each episode has three scopes: user-general, project A, project B. No user ID, project ID, hidden class label or test gold in the neural text embeddings. The explicit scope (user vs project) selects the appropriate updater. Project A/B use shared project weights with **different hidden states**. Role/scope is a controlled external input in this pilot.
- This study tests **recovery of meaning from prior feedback**, not correct novel action choices: it does NOT ask the 360M frozen LM to evaluate unseen project tasks. Any story that it can solve genuinely unfamiliar project dilemmas would be misleading.

## Locked controls, metrics and gates

Same near and far sets, identical synthetic gold, compare:
1. No history (class 0 constant).
2. Learned user state only; learned project state only.
3. Learned **hierarchical** user + A + B memories.
4. Full-history **human-authored keyword reader** using V9.2 rules (strong simple text competitor on some templates, no pretrained semantic model).
5. **Frozen semantic exemplar retrieval**: classify using cosine similarity between latest message vector and averages of known training-example vectors. This has **NO learned updater**.
6. **Frozen semantic full-history retrieval**: cosine of recency-weighted three event vectors (weights .15/.15/.70) against the same training class centroids, **NO learned updater**.
7. Synthetic known-label oracle (diagnostic upper bound).

Primary outputs: overall cold per-stream concept accuracy, global-user accuracy, per-project accuracy, late correction recovery, correct separation when projects have opposing principles, and 128-case causal project B history swap (verify global+A unchanged, project B exactly follows replacement A history), plus incremental GRU-state equivalence to full sequence run.

**All pre-registered GO gates required**:
- Near overall hierarchical ≥85%.
- FAR overall hierarchical ≥75%.
- FAR global-user concept ≥70%.
- FAR late-correction ≥75%.
- FAR opposing projects correctly distinct ≥80%.
- FAR neural hierarchical exceeds **the strongest full-history text/semantic exemplar baseline by at least 2 percentage points**.
- Perfect counterfactual scope isolation and correct one-event incremental state update.

If ANY gates fail: **NO_GO_TO_BACKBONE_COUPLING**, do not connect new latent memory to LM, size-sweep it, deploy it or call it "learned user intent." Even if all pass, this is merely a synthetic *pretrained feature+memory* pilot: next require several seeds, independent human-written new histories/dilemmas, a frozen 360M LM+high-quality text retrieval baseline, and an end-to-end neural action-choice task.

## Files, compute and outputs

- `experiments/compass_v9_3/study_v93.py`: frozen encoder, sequence updater, separated baselines, fresh far template pool and tests.
- `experiments/compass_v9_3/EXPERIMENT_V9_3.md`: this locked contract.
- Run on a free GitHub Actions CPU runner with PyTorch, transformers, numpy and safetensors; train **600 steps** × **32 synthetic episodes**, dev checkpoints every 50; evaluate **600 near** + **600 far** episodes. Random seed 11 and fixed test seeds.
- Artifact `compass-v93-semantic-memory.json` includes scored comparisons, event examples, development trace and all gates; separate `compass-v93-semantic-memory.safetensors` contains **only the new shared GRU/compression/head weights**, NOT encoder weights. Large frozen MiniLM downloaded from HF on runner; no AI training on user's Windows laptop.
- Research isolation: no native SELF 5M model, SmolLM2 weights, production data, old experiment artifacts or deployed service changed.

No external source provenance, user authorization, durable store, revocation/delete, persistent user-trained weights, autonomous preference extraction, privacy control or independent evaluation is provided in this pilot.
