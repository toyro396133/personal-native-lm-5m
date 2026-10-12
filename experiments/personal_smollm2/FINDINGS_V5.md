# SELF V5 — completed three-profile versioned-memory audit

**Execution:** [GitHub Actions V5 #38092858179](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38092858179) — `completed / success`, all 3 profile jobs and aggregate job.
**Artifacts:** `self-v5-aggregate-results` (ID `11684099840`), and `self-v5-profile-{0,1,2}` (IDs `11684124748`, `11684159546`, `11684379123`). JSON and 3 per-stage adapter checkpoints per profile were inspected.
**Protocol/code:** `EXPERIMENT_V5.md`, `benchmark_v5.py`, `personal_v5_data.py`; frozen `HuggingFaceTB/SmolLM2-360M-Instruct`, per-profile residual adapter with 7,680 trainable weights.

## Final results: three synthetic profiles

Each profile has 12 current facts, including **two updates** (focus duration and dashboard accent), and 10 generic questions. Different profiles use different seeds (11,19,37), so seed and profile effects are confounded.

| Condition | 2-option personal QA | Open personal QA | Open updated-fact QA | Open generic QA |
|---|---:|---:|---:|---:|
| Frozen LM | **15/36 (41.7%)** | **0/36 (0%)** | **0/6 (0%)** | **30/30 (100%)** |
| Adapter only | **23/36 (63.9%)** | **2/36 (5.6%)** | **0/6 (0%)** | **25/30 (83.3%)** |
| Frozen LM + canonical memory retrieval | **36/36 (100%)** | **35/36 (97.2%)** | **6/6 (100%)** | **30/30 (100%)** |
| Always-on adapter + memory | **34/36 (94.4%)** | **28/36 (77.8%)** | **3/6 (50%)** | **25/30 (83.3%)** |
| Gated adapter + memory | **34/36 (94.4%)** | **28/36 (77.8%)** | **3/6 (50%)** | **30/30 (100%)** |
| Direct current-value lookup, no LM for personal slots | **36/36 (100%)** | **36/36 (100%)** | **6/6 (100%)** | **30/30 (100%)** |

These are **automatic benchmark scores**, not audited semantic correctness on real user conversations.

### Per-profile raw counts

| Profile, seed | Adapter-only open | Retrieval open | Hybrid open | Revised facts retrieved | Hybrid revisions | Generic adapter-only / hybrid | Generic gated |
|---|---|---|---|---|---|---|---|
| 0, 11 | 1/12 | 12/12 | 9/12 | 2/2 | 1/2 | 6/10 | 10/10 |
| 1, 19 | 0/12 | 11/12 | 10/12 | 2/2 | 1/2 | 9/10 | 10/10 |
| 2, 37 | 1/12 | 12/12 | 9/12 | 2/2 | 1/2 | 10/10 | 10/10 |

V5 reached **12/12 active-slot retrieval hit@1** in all three profiles, at every final stage; the canonical memory parser also verified exactly two revisions per profile, rejecting mismatched correction events. This is made much easier by a constrained assertion/correction grammar and matching slot names; no unrestricted natural-language extraction was tested.

## Direct inspection of generated replies

- **Latest focus duration:** values update to 25/30/45 minutes respectively. Frozen LM + current canonical memory correctly answers all 3; all three always-on/gated hybrid generations fail this updated focus question. Dashboard-accent updates succeed for the hybrid in all three profiles. `retrieval` and `direct_memory` handle all 6 updated facts correctly.
- **Retrieval-only miss:** profile 1 asks for its `planning tool` (`kanban board`), but the frozen LM answers `Figma` even after receiving the correct current fact. Direct-value lookup avoids this model-side confusion.
- **Adapter pathologies:** observed repetitive garbage, non sequitur pretrained-model strings, wrong basic arithmetic/week answers, and one multilingual fragment. For instance, the adapter can emit extended "3 3 3..." strings or a misleading `resnet50` model identifier rather than the requested personal fact.
- **Scoring false negative:** for profile 2, the expected value `a short table` was answered `short table` by the hybrid, semantically plausible but rejected by literal phrase grading. Therefore 28/36 is a scorer-defined rate that may understate correctness on some short variants; it is **not** grounds to dismiss the clear broader quality problems.
- More generally, answer strings can contain the expected phrase in irrelevant/contradictory text, which a heuristic evaluator may misgrade. Preserve raw generations for review.

## Interpretations

1. The intended primary V5 fix works **within the controlled format**: versioned memory retains authoritative active values and rejects inconsistent revisions. Passing the updated focus slot to a frozen model yields correct answers.
2. The selective gate protects generic queries: **all 30 generic questions bypassed the adapter**, restoring observed generic scores from 25/30 to 30/30. This is a routing result, *not* proof that training preserved the pretrained model's ability.
3. The small trained adapter offers **no advantage** over well-formed memory for factual questions here; adding it hurts scored open-answer quality (retrieval 35/36 vs hybrid 28/36). The adaptive model should *not* be made the default generator when strict factual recall is available.
4. Direct memory 36/36 is an instructive upper baseline **for explicitly parsed, fixed-slot queries**, not a demonstrated solution to general conversational memory. Both extraction and routing are assisted by constrained templates and fixed slots.
5. This test deliberately changes dataset/answer protocol from V4; do not compare percentages across V4 and V5 as if they were matched samples.
6. Only 3 distinct profiles with a single seed each were tested. The handful of generic questions and deterministic outputs do not prove generalization or safety of unseen queries.

## Research decision and next gates

**Keep the frozen backbone + canonical versioned external memory as the provisional preferred baseline** for exact personal facts. Turn the personal adapter *off* by default unless task-specific offline evidence demonstrates benefit. Do not merge the experimental adapters as validated production personalization.

Potential V6:
- **Unconstrained extraction:** natural conversational statement vs hypotheticals, quoted third-party text, uncertain claims, reversals; use provenance/confidence and explicit confirmation on conflicts. Test on independently authored unseen histories and avoid putting test labels in the extractor.
- **Routing with abstention:** distinguish factual lookup, preference-aware writing/planning, general knowledge, no-memory questions; allow direct slot answer, frozen+retrieval, or personal adapter only when a prevalidated task class warrants it. Prevent user-personal wording in otherwise generic question from unnecessarily routing to adapter.
- **Constrain or regularize adaptation:** train the adapter for *style and habits* rather than brittle factual labels; reduce training strength, measure KL drift on held-out general queries; target low general regression. Add ablation vs non-trained adapter and retrieval-only.
- **Evaluation robustness:** human-audited generated answers, number-word and article normalization, repetition/conflict detection, multi-seed per profile, multiple unseen profiles, longer histories and falsification controls.
- **Memory privacy lifecycle:** support update, explicit deletion, and expiry in canonical store and test post-deletion retrieval; previously trained adapters need a defined removal/retraining policy.
- A candidate must clear **open-answer correctness, updated-fact correctness, general non-regression, and zero unapproved stale fact exposure** before future production integration.

**Status:** V5 completed, audited and archived; V6 not started by this audit.
