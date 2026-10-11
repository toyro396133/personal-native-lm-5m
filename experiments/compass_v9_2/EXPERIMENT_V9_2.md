# SELF / COMPASS V9.2 — Hierarchical Concept Learning & Cross-Project Transfer

**Preregistration / scope:** This deliberately tests **WHAT COMPASS learns** before investing more in **HOW to plug it into the frozen large LM**. Previous V9.0–V9.1 correctly changed hidden states inside SmolLM2-360M but returned **zero preference-driven decision reversals**. User vision is a persistent personal neural component plus separately learned per-project neural component, conveying *intentions and overarching principles*, not simply task history, through an eventual neural connection to a continuously running frozen LM.

## Exact limited question of this first V9.2 pilot

Can a **small trainable concept learner** extract (i) one global user working-style principle and (ii) one project-specific principle for **two independent projects** from short English *sequences of three feedback/correction statements*, including **late correction changing the principle**, and reliably recover the current concept under held-out paraphrase templates? Does a joint user+project representation improve over either stream alone and preserve strict project isolation?

This is not proof of rich goal abstraction. Each dimension is experimenter-authored **binary**, and the downstream decision interface uses an *explicit scope router*. The new task descriptions are not processed by the neural model: this phase validates concept-state inference, not free-form problem-solving on new dilemmas. **There is no LM call and no V9.2 neural backbone adapter.**

### Data construction and leakage protections

- Train users/projects and cold users/projects are randomly generated synthetic episodes with different random seeds, project identifiers never given to student. Every episode has 3 histories (user, project A, project B), each exactly **3 natural-language messages**. Training draws exclusively from author-written train utterances; 45% of streams contain an explicitly late correction reversing earlier stated views.
- Global latent: prefers checking uncertainties before committing OR running a small pilot and revising.
- Project latent: preserve the **underlying intended outcome** even if historical implementation differs OR preserve a **published compatibility contract** even when internals differ.
- Separate 4 train texts and 3 unseen paraphrase texts **for each concept class**. Unlike an initial local smoke test, **dev checkpoint selection uses only TRAIN paraphrases**, not the held-out cold paraphrase set.
- Cold scenarios additionally have 3 distinct user-type and 3 project-type dilemma wordings; each is evaluated in neutral and *binding* settings and both option orders. But because the decision layer is a **fixed trusted-scope decoder of the predicted binary concept**, these labels do not measure whether a neural network understands the new dilemma text, only whether the inferred latent guides a controlled symbolic decision. No gold labels/IDs/scope metadata are inserted into the user/project history text.

### Neural architecture and training

- Frozen hash mapping of English history words to a **4,096-token feature table**, trainable 24D word embeddings; separate 28-hidden GRUs for *user* and *project* histories; separate binary heads. **No pretrained language model**.
- One shared project GRU is applied to project A and project B **without sharing their transient hidden states**. Changes to B must not alter user or A predictions, measured by causal replacement of B history under identical other inputs. Dynamic per-episode history encoding, no persisted across-process memory or per-user separate neural weights.
- CPU-only, Python 3.11, PyTorch, one fixed seed 11, AdamW LR .003, batch 32, **600 optimizer steps** with internal dev checkpoint selection every 50 steps. **600 unseen cold episodes**, no cold labels seen during training or selection.
- Train one concept classifier/encoder jointly on user and project streams, using training oracle labels for the synthetic principles. This is supervised synthetic concept recovery, NOT self-guided extraction of user meaning from open-ended human interactions.

### Controlled baselines

1. **No-memory prior:** constant concept class 0 everywhere.
2. **Personal only:** learned user global state, no project context.
3. **Project only:** two project-specific learned states, no user global state.
4. **Hierarchical**: both learned neural streams, tested as a joint pair, with each project isolated.
5. **Textual full-history lexical keyword reader:** human-authored transparent semantic keyword rule on the **latest correction sentence**, receives all history as text. This is a relevant CHEAP full-text baseline but not a strong LLM+RAG comparison. Do not claim superiority over real retrieval.
6. **Known-concept oracle:** synthetic ground truth; diagnostic upper bound, not a deployed model.

### Prespecified requirements to pass this limited synthetic probe

- Cold overall inferred concept label accuracy ≥ **85%** across 1 user + 2 project streams.
- Correctly updated concept after late correction ≥ **80%**.
- Distinguish opposite project A/B concepts in ≥ **80%** of relevant episodes.
- Joint user+project concept accuracy strictly higher than user-only AND project-only.
- Joint neural concept accuracy strictly higher than full-history lexical keyword reader.
- In project-B history intervention, **global user and project-A predictions unchanged in 100%**, and project-B prediction equals A when its history is replaced with A's history.
- Report the new-dilemma controlled downstream decision table, task-scope oracle dependencies, training runtime, train/dev/cold separation, sample predictions, and ALL ablations.

**No-go:** If any substantive gate fails, do not claim conceptual learning is proven or immediately re-integrate latents into the frozen 360M LM. Diagnose semantic paraphrase generalization and correction retention first; add an independently authored human holdout and **frozen LLM + high-quality textual retrieval** baseline before any stronger claims.

## Artifacts and isolation

- Source branch: `experiment/compass-v9-2`.
- `experiments/compass_v9_2/study_v92.py` — train/evaluate script.
- `EXPERIMENT_LOG.md` — study registry and eventual findings.
- GitHub Actions (workflow stored on main for manual/automatic triggering): `.github/workflows/compass-v92-concept-learning.yml`.
- Outputs: `compass-v92-concept-transfer.json` per-arm comparisons and predictions; `compass-v92-concept-transfer.safetensors` learned **shared** concept encoder weights.

No changes to native SELF 5M, V9.1 trained adapter, production model or server. All tests in English.
